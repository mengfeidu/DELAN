import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.distributed import is_default_gpu
from utils.misc import length2mask
from utils.logger import print_progress
from r2r.data_utils import get_inst_land_part, get_inst_land_part_cvdn,gen_sim_gt

from models.model_HAMT import VLNBertCMT, Critic, CrossEn, AllGather, ContrastiveLoss, ContrastiveLoss_obs

from .eval_utils import cal_dtw

from .agent_base import BaseAgent


allgather = AllGather.apply
class Seq2SeqCMTAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': (0,-1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0,-1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, rank=0):
        super().__init__(env)
        self.args = args

        self.default_gpu = is_default_gpu(self.args)
        self.rank = rank

        # Models
        self._build_model()

        if self.args.world_size > 1:
            self.vln_bert = DDP(self.vln_bert, device_ids=[self.rank], find_unused_parameters=True)
            self.critic = DDP(self.critic, device_ids=[self.rank], find_unused_parameters=True)
            if args.tcl_hist_weight > 0 or args.tcl_obs_weight > 0:
                self.contrastive_loss = DDP(self.contrastive_loss, device_ids=[self.rank], find_unused_parameters=True)

        self.models = (self.vln_bert, self.critic)
        self.device = torch.device('cuda:%d'%self.rank) #TODO 

        # Optimizers
        if self.args.optim == 'rms':
            optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            optimizer = torch.optim.Adam
        elif self.args.optim == 'adamW':
            optimizer = torch.optim.AdamW
        elif self.args.optim == 'sgd':
            optimizer = torch.optim.SGD
        else:
            assert False
        if self.default_gpu:
            print('Optimizer: %s' % self.args.optim)

        self.vln_bert_optimizer = optimizer(self.vln_bert.parameters(), lr=self.args.lr)
        self.critic_optimizer = optimizer(self.critic.parameters(), lr=self.args.lr)
        self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)
        if args.tcl_hist_weight > 0 or args.tcl_obs_weight > 0:
            self.contrastive_loss_optimizer = optimizer(self.contrastive_loss.parameters(), lr=self.args.lr)
            self.contrastive_loss_obs_optimizer = optimizer(self.contrastive_loss_obs.parameters(), lr=self.args.lr)
            self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer, self.contrastive_loss_optimizer, self.contrastive_loss_obs_optimizer)
        else:
            self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)
            
        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, size_average=False)

        # memory bank
        if args.use_mem_bank:
            self.queue_ptr_hist=torch.zeros(1,dtype=torch.long).cuda()
            self.queue_ptr_obs=torch.zeros(1,dtype=torch.long).cuda()
            self.queue_temp_ptr_obs=torch.zeros(1,dtype=torch.long).cuda()
            # hist-inst tcl
            self.queue_traj=torch.randn(args.capacity,args.hidden_dim).cuda()
            self.queue_traj=nn.functional.normalize(self.queue_traj, dim=-1).cuda()

            self.queue_frame=torch.randn(args.capacity,args.max_action_len,args.hidden_dim).cuda()
            self.queue_frame=nn.functional.normalize(self.queue_frame, dim=-1).cuda()

            self.queue_sent=torch.randn(args.capacity,args.hidden_dim).cuda()
            self.queue_sent=nn.functional.normalize(self.queue_sent, dim=-1).cuda()

            self.queue_word_inst=torch.randn(args.capacity,args.max_instr_len,args.hidden_dim).cuda()
            self.queue_word_inst=nn.functional.normalize(self.queue_word_inst, dim=-1).cuda()
            # land-obs tcl
            self.queue_patch=torch.randn(args.capacity,args.max_obs_num,args.hidden_dim).cuda()
            self.queue_patch=nn.functional.normalize(self.queue_patch, dim=-1).cuda()
            self.queue_word_land=torch.randn(args.capacity,args.max_instr_len,args.hidden_dim).cuda()
            self.queue_word_land=nn.functional.normalize(self.queue_word_land, dim=-1).cuda()
            # temporary queue for land-obs tcl
            self.temp_queue_patch=torch.randn(args.batch_size*args.max_action_len,args.max_obs_num,args.hidden_dim).cuda()
            self.temp_queue_patch=nn.functional.normalize(self.temp_queue_patch, dim=-1).cuda()
            self.temp_queue_word_land=torch.randn(args.batch_size*args.max_action_len,args.max_instr_len,args.hidden_dim).cuda()
            self.temp_queue_word_land=nn.functional.normalize(self.temp_queue_word_land, dim=-1).cuda()
      
        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)


    @torch.no_grad()
    def _dequeue_and_enqueue_inst_hist(self, traj, frame, sent, word_inst):
        batch_size = traj.shape[0]
        ptr = int(self.queue_ptr_hist)
        if ptr + batch_size > self.args.capacity:
            batch_size = self.args.capacity - ptr
        self.queue_traj[ptr:ptr + batch_size, :] = traj[:batch_size, :]
        self.queue_frame[ptr:ptr + batch_size, :, :] = frame[:batch_size, :, :]
        self.queue_sent[ptr:ptr + batch_size, :] = sent[:batch_size, :]
        self.queue_word_inst[ptr:ptr + batch_size, :] = word_inst[:batch_size, :, :]
        
        ptr = (ptr + batch_size) % self.args.capacity # move pointer
        self.queue_ptr_hist[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_temp_land_obs(self, patch, word_land):
        batch_size = patch.shape[0]
        ptr = int(self.queue_temp_ptr_obs)
        assert ptr + batch_size <= self.temp_queue_patch.shape[0]
        self.temp_queue_patch[ptr:ptr + batch_size, :, :] = patch[:batch_size, :, :]
        self.temp_queue_word_land[ptr:ptr + batch_size, :, :] = word_land[:batch_size, :, :]

        ptr = ptr + batch_size # move pointer
        self.queue_temp_ptr_obs[0] = ptr
    
    @torch.no_grad()
    def _dequeue_and_enqueue_land_obs(self):
        batch_len_size = int(self.queue_temp_ptr_obs)
        ptr = int(self.queue_ptr_obs)
        if ptr + batch_len_size > self.args.capacity:
            batch_len_size = self.args.capacity - ptr
        self.queue_patch[ptr:ptr + batch_len_size, :, :] = self.temp_queue_patch[:batch_len_size, :, :]
        self.queue_word_land[ptr:ptr + batch_len_size, :, :] = self.temp_queue_word_land[:batch_len_size, :, :]
        
        ptr = (ptr + batch_len_size) % self.args.capacity
        self.queue_ptr_obs[0] = ptr
        
        self.queue_temp_ptr_obs[0] = 0 # clear temporary queue
        
    def _build_model(self):
        self.vln_bert = VLNBertCMT(self.args).cuda()
        self.critic = Critic(self.args).cuda()
        if self.args.tcl_hist_weight > 0 or self.args.tcl_obs_weight > 0:
            self.contrastive_loss = ContrastiveLoss(self.args).cuda()
            self.contrastive_loss_obs = ContrastiveLoss_obs(self.args).cuda()

    def _language_variable(self, obs):
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]

        # set max length to max instr len
        if self.args.tcl_hist_weight > 0 or self.args.tcl_obs_weight > 0:
            seq_tensor = np.zeros((len(obs), self.args.max_instr_len), dtype=np.int64)
            mask = np.zeros((len(obs), self.args.max_instr_len), dtype=np.bool_)
        else:
            seq_tensor = np.zeros((len(obs), max(seq_lengths)), dtype=np.int64)
            mask = np.zeros((len(obs), max(seq_lengths)), dtype=np.bool_)
        for i, ob in enumerate(obs):
            seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
            mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor)
        mask = torch.from_numpy(mask)
        return seq_tensor.long().cuda(), mask.cuda(), seq_lengths

    def _cand_pano_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        ob_cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        ob_lens = []
        ob_img_fts, ob_ang_fts, ob_nav_types = [], [], []
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            cand_img_fts, cand_ang_fts, cand_nav_types = [], [], []
            cand_pointids = np.zeros((self.args.views,), dtype=np.bool_)
            for j, cc in enumerate(ob['candidate']):
                cand_img_fts.append(cc['feature'][:self.args.image_feat_size])
                cand_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                cand_pointids[cc['pointId']] = True
                cand_nav_types.append(1)

            # add [STOP] feature
            cand_img_fts.append(np.zeros((self.args.image_feat_size,), dtype=np.float32))
            cand_ang_fts.append(np.zeros((self.args.angle_feat_size,), dtype=np.float32))
            cand_img_fts = np.vstack(cand_img_fts)
            cand_ang_fts = np.vstack(cand_ang_fts)
            cand_nav_types.append(2)

            # add pano context
            pano_fts = ob['feature'][~cand_pointids]
            cand_pano_img_fts = np.concatenate([cand_img_fts, pano_fts[:, :self.args.image_feat_size]], 0)
            cand_pano_ang_fts = np.concatenate([cand_ang_fts, pano_fts[:, self.args.image_feat_size:]], 0)
            cand_nav_types.extend([0] * (self.args.views - np.sum(cand_pointids)))

            ob_lens.append(len(cand_nav_types))
            ob_img_fts.append(cand_pano_img_fts)
            ob_ang_fts.append(cand_pano_ang_fts)
            ob_nav_types.append(cand_nav_types)

        # pad features to max_len
        if self.args.tcl_hist_weight > 0 or self.args.tcl_obs_weight > 0:
            max_len = self.args.max_obs_num # max_degree+1+35=15+1+35=51
        else:
            max_len = max(ob_lens)
        
        for i in range(len(obs)):
            num_pads = max_len - ob_lens[i]
            ob_img_fts[i] = np.concatenate([ob_img_fts[i], \
                                            np.zeros((num_pads, ob_img_fts[i].shape[1]), dtype=np.float32)], 0)
            ob_ang_fts[i] = np.concatenate([ob_ang_fts[i], \
                                            np.zeros((num_pads, ob_ang_fts[i].shape[1]), dtype=np.float32)], 0)
            ob_nav_types[i] = np.array(ob_nav_types[i] + [0] * num_pads)

        ob_img_fts = torch.from_numpy(np.stack(ob_img_fts, 0)).cuda()
        ob_ang_fts = torch.from_numpy(np.stack(ob_ang_fts, 0)).cuda()
        ob_nav_types = torch.from_numpy(np.stack(ob_nav_types, 0)).cuda()

        return ob_img_fts, ob_ang_fts, ob_nav_types, ob_lens, ob_cand_lens

    def _candidate_variable(self, obs):
        cand_lens = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        max_len = max(cand_lens)
        cand_img_feats = np.zeros((len(obs), max_len, self.args.image_feat_size), dtype=np.float32)
        cand_ang_feats = np.zeros((len(obs), max_len, self.args.angle_feat_size), dtype=np.float32)
        cand_nav_types = np.zeros((len(obs), max_len), dtype=np.int64)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, cc in enumerate(ob['candidate']):
                cand_img_feats[i, j] = cc['feature'][:self.args.image_feat_size]
                cand_ang_feats[i, j] = cc['feature'][self.args.image_feat_size:]
                cand_nav_types[i, j] = 1
            cand_nav_types[i, cand_lens[i]-1] = 2

        cand_img_feats = torch.from_numpy(cand_img_feats).cuda()
        cand_ang_feats = torch.from_numpy(cand_ang_feats).cuda()
        cand_nav_types = torch.from_numpy(cand_nav_types).cuda()
        return cand_img_feats, cand_ang_feats, cand_nav_types, cand_lens

    def _history_variable(self, obs):
        hist_img_feats = np.zeros((len(obs), self.args.image_feat_size), np.float32)
        for i, ob in enumerate(obs):  
            hist_img_feats[i] = ob['feature'][ob['viewIndex'], :self.args.image_feat_size]
        hist_img_feats = torch.from_numpy(hist_img_feats).cuda()

        if self.args.hist_enc_pano:
            hist_pano_img_feats = np.zeros((len(obs), self.args.views, self.args.image_feat_size), np.float32)
            hist_pano_ang_feats = np.zeros((len(obs), self.args.views, self.args.angle_feat_size), np.float32)
            for i, ob in enumerate(obs):
                hist_pano_img_feats[i] = ob['feature'][:, :self.args.image_feat_size]
                hist_pano_ang_feats[i] = ob['feature'][:, self.args.image_feat_size:]
            hist_pano_img_feats = torch.from_numpy(hist_pano_img_feats).cuda()
            hist_pano_ang_feats = torch.from_numpy(hist_pano_ang_feats).cuda()
        else:
            hist_pano_img_feats, hist_pano_ang_feats = None, None

        return hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def _dagger_action(self, obs, is_IL, t, traj, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if is_IL:
                    if t == len(ob['gt_path']) - 1:
                        a[i] = len(ob['candidate'])
                    else:
                        assert ob['viewpoint'] == ob['gt_path'][t]
                        goal_vp = ob['gt_path'][t + 1]
                        for j, candidate in enumerate(ob['candidate']):
                            if candidate['viewpointId'] == goal_vp:   # Next view point
                                a[i] = j
                                break           
                else:
                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = len(ob['candidate'])
                    else:
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        for j, candidate in enumerate(ob['candidate']):
                            if self.args.expert_policy == 'ndtw':
                                dist = - cal_dtw(
                                        self.env.shortest_distances[scan], 
                                        sum(traj[i]['path'], []) + self.env.shortest_paths[scan][ob['viewpoint']][candidate['viewpointId']][1:], 
                                        ob['gt_path'], 
                                        threshold=3.0
                                    )['nDTW']
                            elif self.args.expert_policy == 'spl':
                                    # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                                    dist = self.env.shortest_distances[scan][candidate['viewpointId']][ob['gt_path'][-1]] \
                                            + self.env.shortest_distances[scan][cur_vp][candidate['viewpointId']]
                                            
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))         
                
        return torch.from_numpy(a).cuda()

    def _dagger_action_idx(self, obs, idx, is_IL, t, traj, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if i != idx:
                continue
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if is_IL:
                    if t == len(ob['gt_path']) - 1:
                        a[i] = len(ob['candidate'])
                    else:
                        assert ob['viewpoint'] == ob['gt_path'][t]
                        goal_vp = ob['gt_path'][t + 1]
                        for j, candidate in enumerate(ob['candidate']):
                            if candidate['viewpointId'] == goal_vp:   # Next view point
                                a[i] = j
                                break           
                else:
                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = len(ob['candidate'])
                    else:
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        for j, candidate in enumerate(ob['candidate']):
                            if self.args.expert_policy == 'ndtw':
                                dist = - cal_dtw(
                                        self.env.shortest_distances[scan], 
                                        sum(traj[i]['path'], []) + self.env.shortest_paths[scan][ob['viewpoint']][candidate['viewpointId']][1:], 
                                        ob['gt_path'], 
                                        threshold=3.0
                                    )['nDTW']
                            elif self.args.expert_policy == 'spl':
                                    # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                                    dist = self.env.shortest_distances[scan][candidate['viewpointId']][ob['gt_path'][-1]] \
                                            + self.env.shortest_distances[scan][cur_vp][candidate['viewpointId']]
                                            
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))         
                
        return torch.from_numpy(a).cuda()
    
    def make_equiv_action(self, a_t, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[i].makeAction([name], [0], [0])
            else:                       # Adjust
                self.env.env.sims[i].makeAction(*self.env_actions[name])

        for i, ob in enumerate(obs):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = ob['candidate'][action]
                src_point = ob['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12  # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, 'down')
                    src_level -= 1
                while self.env.env.sims[i].getState()[0].viewIndex != trg_point:    # Turn right until the target
                    take_action(i, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[i].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, select_candidate['idx'])

                state = self.env.env.sims[i].getState()[0]
                if traj is not None:
                    traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))

    def rollout(self, train_ml=None, train_rl=True, train_tcl_hist=None, train_tcl_obs=None, reset=True):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment

        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs(t=0)

        batch_size = len(obs)

        # Language input
        txt_ids, txt_masks, txt_lens = self._language_variable(obs)

        ''' Language BERT '''
        language_inputs = {
            'mode': 'language',
            'txt_ids': txt_ids,
            'txt_masks': txt_masks,
        }
        txt_embeds = self.vln_bert(**language_inputs)

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
        } for ob in obs]

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        last_ndtw = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']
            path_act = [vp[0] for vp in traj[i]['path']]
            last_ndtw[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        ended_flag = torch.tensor(0.,device='cuda')
        ended_flag_list=[torch.tensor(0.,device='cuda') for _ in range(self.args.world_size)]
        any_proc_ended=False
        
        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.

        # contrastive learning need
        inst_hist_loss = 0.
        inst_pano_loss = [0.] * batch_size
        sim_obs_loss = 0.
        if self.args.tcl_hist_weight or self.args.tcl_obs_weight:
            if self.args.dataset == 'landrxr':
                inst_part, land_part, inst_mask, land_mask = get_inst_land_part_cvdn(obs)
            else:
                inst_part, land_part, inst_mask, land_mask = get_inst_land_part(obs)
        GT_insts, GT_lands = [], []
        cl_loss_fct=CrossEn()
        tcl_length=0
        # for backtrack
        visited = [set() for _ in range(batch_size)]

        hist_embeds = [self.vln_bert('history').expand(batch_size, -1)]  # global embedding
        hist_lens = [1 for _ in range(batch_size)]

        for t in range(self.args.max_action_len):
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens = self._cand_pano_feature_variable(obs)
                if self.args.tcl_hist_weight > 0 or self.args.tcl_obs_weight > 0:
                    ob_masks = length2mask(ob_lens,size=self.args.max_obs_num).logical_not()
                else:
                    ob_masks = length2mask(ob_lens).logical_not() 
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()
            
            ''' Visual BERT '''
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,    # history before t step
                'hist_lens': hist_lens,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'return_states': True if self.feedback == 'sample' else False
            }
                            
            t_outputs = self.vln_bert(**visual_inputs)
            ## tcl
            logit = t_outputs[0]
            # hist_embeds = t_outputs[-2]
            # max_hist_len = hist_embeds.size(1)
            original_ob_embeds=t_outputs[-1]
            if self.feedback == 'sample':
                h_t = t_outputs[1]
                hidden_states.append(h_t)

            if train_ml is not None:
                target = self._teacher_action(obs, ended)
                ml_loss += self.criterion(logit, target)

            if train_tcl_obs and self.feedback=='teacher':
                if not any_proc_ended and self.args.use_aosm_obs_cl:
                    tcl_length+=1
                    # word-level textual feature
                    word_land_mask=torch.from_numpy(land_mask).cuda() # [bs,num_words]
                    if self.args.no_lang_ca:
                        word_land_features=txt_embeds[self.args.cl_lang_layer]/txt_embeds[self.args.cl_lang_layer].norm(dim=-1,keepdim=True)
                    else:
                        word_land_features=txt_embeds/txt_embeds.norm(dim=-1,keepdim=True) # [bs,num_patches,dim]
                    word_land_features=word_land_features*word_land_mask.unsqueeze(dim=-1)
                    
                    # patch-level observation feature
                    patch_features=original_ob_embeds/original_ob_embeds.norm(dim=-1,keepdim=True) # [bs,max_candi_num_dim]

                    if self.args.world_size>1:
                        word_land_features=allgather(word_land_features,self.args)
                        patch_features=allgather(patch_features,self.args)
                        torch.distributed.barrier()
                        
                    mem_neg_obs={
                        'patch':self.queue_patch.clone().detach(),
                        'word_land':self.queue_word_land.clone().detach(),
                    } if self.args.use_mem_bank else None
                                        
                    contrastive_obs_inputs={
                        'patch_features': patch_features,
                        'patch_mask': None,
                        'word_features': word_land_features,
                        'attention_mask': None,
                        'mem_neg': mem_neg_obs,
                    }
                    sim_obs_loss+=self.contrastive_loss_obs(**contrastive_obs_inputs)
                    
                    if self.args.use_mem_bank:
                        self._dequeue_and_enqueue_temp_land_obs(patch_features.detach(),word_land_features.detach())

            # mask logit where the agent backtracks in observation in evaluation
            if self.args.no_cand_backtrack:
                bt_masks = torch.zeros(ob_nav_types.size()).bool()
                for ob_id, ob in enumerate(obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            bt_masks[ob_id][c_id] = True
                bt_masks = bt_masks.cuda()
                logit.masked_fill_(bt_masks, -float('inf'))

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                 # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch


            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (ob_cand_lens[i]-1) or next_id == self.args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # get history input embeddings
            if train_rl or ((not np.logical_or(ended, (cpu_a_t == -1)).all()) and (t != self.args.max_action_len-1)):
                # DDP error: RuntimeError: Expected to mark a variable ready only once.
                # It seems that every output from DDP should be used in order to perform correctly
                hist_img_feats, hist_pano_img_feats, hist_pano_ang_feats = self._history_variable(obs)
                prev_act_angle = np.zeros((batch_size, self.args.angle_feat_size), np.float32)
                for i, next_id in enumerate(cpu_a_t):
                    if next_id != -1:
                        prev_act_angle[i] = obs[i]['candidate'][next_id]['feature'][-self.args.angle_feat_size:]
                prev_act_angle = torch.from_numpy(prev_act_angle).cuda()

                t_hist_inputs = {
                    'mode': 'history',
                    'hist_img_feats': hist_img_feats,
                    'hist_ang_feats': prev_act_angle,
                    'hist_pano_img_feats': hist_pano_img_feats,
                    'hist_pano_ang_feats': hist_pano_ang_feats,
                    'ob_step': t,
                }
                t_hist_embeds = self.vln_bert(**t_hist_inputs)
                hist_embeds.append(t_hist_embeds)

                for i, i_ended in enumerate(ended):
                    if not i_ended:
                        hist_lens[i] += 1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, traj)
            obs = self.env._get_obs(t=t+1)

            if train_rl:
                # Calculate the mask and reward
                dist = np.zeros(batch_size, np.float32)
                ndtw_score = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(obs):
                    dist[i] = ob['distance']
                    path_act = [vp[0] for vp in traj[i]['path']]
                    ndtw_score[i] = cal_dtw(self.env.shortest_distances[ob['scan']], path_act, ob['gt_path'])['nDTW']

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:                              # If the action now is end
                            if dist[i] < 3.0:                             # Correct
                                reward[i] = 2.0 + ndtw_score[i] * 2.0
                            else:                                         # Incorrect
                                reward[i] = -2.0
                        else:                                             # The action is not end
                            # Path fidelity rewards (distance & nDTW)
                            reward[i] = - (dist[i] - last_dist[i])  # this distance is not normalized
                            ndtw_reward = ndtw_score[i] - last_ndtw[i]
                            if reward[i] > 0.0:                           # Quantification
                                reward[i] = 1.0 + ndtw_reward
                            elif reward[i] < 0.0:
                                reward[i] = -1.0 + ndtw_reward
                            else:
                                raise NameError("The action doesn't change the move")
                            # Miss the target penalty
                            if (last_dist[i] <= 1.0) and (dist[i]-last_dist[i] > 0.0):
                                reward[i] -= (1.0 - last_dist[i]) * 2.0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist
                last_ndtw[:] = ndtw_score

            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            if self.args.ob_type == 'pano':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_lens, ob_cand_lens = self._cand_pano_feature_variable(obs)
                if self.args.tcl_hist_weight > 0 or self.args.tcl_obs_weight > 0:
                    ob_masks = length2mask(ob_lens,size=self.args.max_obs_num).logical_not()
                else:
                    ob_masks = length2mask(ob_lens).logical_not() 
            elif self.args.ob_type == 'cand':
                ob_img_feats, ob_ang_feats, ob_nav_types, ob_cand_lens = self._candidate_variable(obs)
                ob_masks = length2mask(ob_cand_lens).logical_not()

            ''' Visual BERT '''
            visual_inputs = {
                'mode': 'visual',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'hist_embeds': hist_embeds,
                'hist_lens': hist_lens,
                'ob_img_feats': ob_img_feats,
                'ob_ang_feats': ob_ang_feats,
                'ob_nav_types': ob_nav_types,
                'ob_masks': ob_masks,
                'return_states': True
            }
            _, last_h_,_,_ = self.vln_bert(**visual_inputs)
            
            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()        # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * self.args.gamma + rewards[t]  # If it ended, the reward will be 0
                mask_ = torch.from_numpy(masks[t]).cuda()
                clip_reward = discount_reward.copy()
                r_ = torch.from_numpy(clip_reward).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                t_policy_loss = (-policy_log_probs[t] * a_ * mask_).sum()
                t_critic_loss = (((r_ - v_) ** 2) * mask_).sum() * 0.5 # 1/2 L2 loss

                rl_loss += t_policy_loss + t_critic_loss
                if self.feedback == 'sample':
                    rl_loss += (- self.args.entropy_loss_weight * entropys[t] * mask_).sum()

                self.logs['critic_loss'].append(t_critic_loss.item())
                self.logs['policy_loss'].append(t_policy_loss.item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if self.args.normalize_loss == 'total':
                rl_loss /= total
            elif self.args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert self.args.normalize_loss == 'none'

            self.loss += rl_loss
            self.logs['RL_loss'].append(rl_loss.item()) # critic loss + policy loss + entropy loss

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size
            self.logs['IL_loss'].append((ml_loss * train_ml / batch_size).item())

        if train_tcl_hist and self.feedback == 'teacher':
            traj_embeds=hist_embeds+[torch.ones_like(hist_embeds[0]).cuda() for _ in range(self.args.max_action_len-len(hist_embeds))] # PAD to max_action_len
            traj_embeds=torch.stack(traj_embeds,dim=0).permute(1,0,2) # [bs,max_action,dim]
            traj_mask=1-length2mask(hist_lens,size=self.args.max_action_len).to(dtype=torch.float) # [bs,max_action]
            
            # frame-level visual features
            frame_features=traj_embeds/traj_embeds.norm(dim=-1,keepdim=True)
            frame_mask=(1-traj_mask)*-1000000.0

            # traj-level history feature
            traj_embeds=traj_embeds/traj_embeds.norm(dim=-1,keepdim=True)
            traj_mask_un=traj_mask.unsqueeze(dim=-1)
            traj_embeds=traj_embeds*traj_mask_un
            traj_mask_un_sum = torch.sum(traj_mask_un, dim=1, dtype=torch.float) # [bs,1]
            traj_output=torch.sum(traj_embeds,dim=1)/traj_mask_un_sum
            traj_output=traj_output/traj_output.norm(dim=-1,keepdim=True) # [bs,dim]

            # sentence-level textual feature
            if self.args.no_lang_ca:
                sentence_output=txt_embeds[self.args.cl_lang_layer][:,0,:].squeeze(1)
            else:
                sentence_output=txt_embeds[:,0,:].squeeze(1)
            sentence_output=sentence_output/sentence_output.norm(dim=-1,keepdim=True) # [bs,dim]
            
            # word-level textual feature
            word_inst_mask=torch.from_numpy(inst_mask).cuda() # [bs,num_words]
            if self.args.no_lang_ca:
                word_inst_features=txt_embeds[self.args.cl_lang_layer]/txt_embeds[self.args.cl_lang_layer].norm(dim=-1,keepdim=True) # [bs,num_words,dim]
            else:
                word_inst_features=txt_embeds/txt_embeds.norm(dim=-1,keepdim=True) # [bs,num_words,dim]
            word_inst_features=word_inst_features*word_inst_mask.unsqueeze(dim=-1)
            word_inst_mask=(1-word_inst_mask)*-1000000.0
            
            if self.args.world_size>1:
                traj_output=allgather(traj_output,self.args)
                frame_features=allgather(frame_features,self.args)
                frame_mask=allgather(frame_mask,self.args)
                sentence_output=allgather(sentence_output,self.args)
                word_inst_features=allgather(word_inst_features,self.args)
                word_inst_mask=allgather(word_inst_mask,self.args)
                torch.distributed.barrier()

            mem_neg_hist={
                'traj':self.queue_traj.clone().detach(),
                'frame':self.queue_frame.clone().detach(),
                'sent':self.queue_sent.clone().detach(),
                'word_inst':self.queue_word_inst.clone().detach(),
            } if self.args.use_mem_bank else None
            
            contrastive_inputs = {
                'traj_output': traj_output,
                'frame_features': frame_features,
                'frame_mask': None,
                'sentence_output': sentence_output,
                'word_features': word_inst_features,
                'attention_mask': None,
                'mem_neg': mem_neg_hist,                
            }
            sim_hist_loss=self.contrastive_loss(**contrastive_inputs)

            self.logs['CL_hist_loss'].append((sim_hist_loss).item())
            if self.args.norm_cl_weight and type(sim_hist_loss) is not float:
                sim_hist_loss = sim_hist_loss / (sim_hist_loss.item() + 1e-12)
            sim_hist_loss=sim_hist_loss*train_tcl_hist
            self.loss+=sim_hist_loss

            # enqueue and dequeue
            if self.args.use_mem_bank:
                self._dequeue_and_enqueue_inst_hist(traj_output.detach(),frame_features.detach(),sentence_output.detach(),word_inst_features.detach())

            
        if train_tcl_obs and self.feedback == 'teacher':
            sim_obs_loss=sim_obs_loss/tcl_length
            self.logs['CL_obs_loss'].append((sim_obs_loss).item())
            if self.args.norm_cl_weight and type(sim_obs_loss) is not float:
                sim_obs_loss = sim_obs_loss / (sim_obs_loss.item() + 1e-12)
            sim_obs_loss=sim_obs_loss*train_tcl_obs
            self.loss+=sim_obs_loss
            
            if self.args.use_mem_bank:
                self._dequeue_and_enqueue_land_obs()
            

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.args.max_action_len)  # This argument is useless.

        return traj

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        super().test(iters=iters)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=self.args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=self.args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

        self.vln_bert_optimizer.step()
        self.critic_optimizer.step()

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.vln_bert.train()
        self.critic.train()
        if self.args.tcl_hist_weight and self.args.tcl_obs_weight:
            self.contrastive_loss.train()
            self.contrastive_loss_obs.train()

        self.losses = []
        for iter in range(1, n_iters + 1):

            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            if self.args.tcl_hist_weight and self.args.tcl_obs_weight:
                self.contrastive_loss_optimizer.zero_grad()
                self.contrastive_loss_obs_optimizer.zero_grad()

            self.loss = 0

            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=self.args.teacher_weight, train_rl=False, train_tcl_hist=self.args.tcl_hist_weight, train_tcl_obs=self.args.tcl_obs_weight, **kwargs)
            elif feedback == 'sample':  # agents in IL and RL separately
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(train_ml=self.args.ml_weight, train_rl=False, train_tcl_hist=self.args.tcl_hist_weight, train_tcl_obs=self.args.tcl_obs_weight, **kwargs)
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, train_tcl_hist=self.args.tcl_hist_weight, train_tcl_obs=self.args.tcl_obs_weight, **kwargs)
            else:
                assert False

            #print(self.rank, iter, self.loss)
            self.loss.backward()

            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()
            if self.args.tcl_hist_weight and self.args.tcl_obs_weight:
                self.contrastive_loss_optimizer.step()
                self.contrastive_loss_obs_optimizer.step()

            if self.args.aug is None:
                print_progress(iter, n_iters+1, prefix='Progress:', suffix='Complete', bar_length=50)

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            state_dict = states[name]['state_dict']
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
                if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            state.update(state_dict)
            model.load_state_dict(state)
            if self.args.resume_optimizer:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1
