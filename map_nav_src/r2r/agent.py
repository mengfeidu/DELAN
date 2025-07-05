import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict
# import line_profiler

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.distributed import is_default_gpu
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from .agent_base import Seq2SeqAgent
from .eval_utils import cal_dtw

from models.graph_utils import GraphMap
from models.model import VLNBert, Critic
from models.ops import pad_tensors_wgrad
from models.model import CrossEn, AllGather, ContrastiveLoss, ContrastiveLoss_obs
from r2r.data_utils import get_inst_land_part,get_inst_land_part_cvdn
from utils.misc import length2mask
import random
import profile

allgather = AllGather.apply
class GMapNavAgent(Seq2SeqAgent):
    
    def _build_model(self):
        self.vln_bert = VLNBert(self.args).cuda()
        self.critic = Critic(self.args).cuda()
        # buffer
        self.scanvp_cands = {}
        if self.args.tcl_hist_weight or self.args.tcl_obs_weight:
            self.contrastive_loss = ContrastiveLoss(self.args, obs_flag=False).cuda()
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

        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
        mask = torch.from_numpy(mask).cuda()
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask
        }


    def _cand_caption_variable(self, obs, candvpids):
        captions = []
        for vpid in candvpids:
            captions.append(self.env.captions[vpid])

        seq_lengths = [len(caption) for caption in captions]
        seq_lengths = []
        for ob in obs:
            for caption in ob['cand_captions']:
                seq_lengths.append(len(caption))

        seq_tensor = np.zeros((len(seq_lengths), max(seq_lengths)), dtype=np.int64)
        mask = np.zeros((len(seq_lengths), max(seq_lengths)), dtype=np.bool_)
        i = 0
        for ob in obs:
            for caption in ob['cand_captions']:
                seq_tensor[i, :seq_lengths[i]] = caption
                mask[i, :seq_lengths[i]] = True

        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
        mask = torch.from_numpy(mask).cuda()
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask
        }

    def _panorama_feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_loc_fts, batch_nav_types = [], [], []
        batch_view_lens, batch_cand_vpids = [], []
        
        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'][:self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x \
                in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)
            
            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_loc_fts.append(torch.from_numpy(view_loc_fts))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))

        if self.args.tcl_hist_weight > 0 or self.args.tcl_obs_weight > 0:
            max_len = self.args.max_obs_num - 1# max_degree+1+35=15+1+35=51 1 for [stop]
            for i in range(len(obs)):
                num_pads = max_len - batch_view_lens[i]
                batch_view_img_fts[i] = np.concatenate([batch_view_img_fts[i], \
                                                np.zeros((num_pads, batch_view_img_fts[i].shape[1]), dtype=np.float32)], 0)
                batch_loc_fts[i] = np.concatenate([batch_loc_fts[i], \
                                                np.zeros((num_pads, batch_loc_fts[i].shape[1]), dtype=np.float32)], 0)
                batch_nav_types[i] = np.array(batch_nav_types[i].numpy().tolist() + [0] * num_pads)
            batch_view_img_fts = torch.from_numpy(np.stack(batch_view_img_fts, 0)).cuda()
            batch_loc_fts = torch.from_numpy(np.stack(batch_loc_fts, 0)).cuda()
            batch_nav_types = torch.from_numpy(np.stack(batch_nav_types, 0)).cuda()
            batch_view_lens = torch.LongTensor(batch_view_lens).cuda()
        else:
            # pad features to max_len
            batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
            batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
            batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
            batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts, 
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens, 
            'cand_vpids': batch_cand_vpids,
        }

    def _nav_gmap_variable(self, obs, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(obs)
        
        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []                
            for k in gmap.node_positions.keys():
                if self.args.act_visited_nodes:
                    if k == obs[i]['viewpoint']:
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
                else:
                    if gmap.graph.visited(k):
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            # print(gmap_img_embeds[0].shape)

            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )   # cuda

            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i+1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds, 
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks, 
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
        }

    def _nav_vp_variable(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, nav_types):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []
        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i], 
                obs[i]['heading'], obs[i]['elevation']
            )
            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp], 
                obs[i]['heading'], obs[i]['elevation']
            )                    
            # add [stop] token at beginning
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types == 1], 1)

        if self.args.tcl_hist_weight or self.args.tcl_obs_weight:
            return {
                'vp_img_embeds': vp_img_embeds,
                'vp_pos_fts': batch_vp_pos_fts,
                'vp_masks': length2mask((view_lens+1).cpu().numpy(),size=self.args.max_obs_num).logical_not(),
                'vp_nav_masks': vp_nav_masks,
                'vp_cand_vpids': [[None]+x for x in cand_vpids],
            }
        else:
            return {
                'vp_img_embeds': vp_img_embeds,
                'vp_pos_fts': batch_vp_pos_fts,
                'vp_masks': gen_seq_masks(view_lens+1),
                'vp_nav_masks': vp_nav_masks,
                'vp_cand_vpids': [[None]+x for x in cand_vpids],
            }

    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
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
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0    # Stop if arrived 
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                            dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                    + self.env.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % (scan))

        return torch.from_numpy(a).cuda()

    
    def _teacher_action_r4r(
        self, obs, vpids, ended, visited_masks=None, imitation_learning=False, t=None, traj=None
    ):
        """R4R is not the shortest path. The goal location can be visited nodes.
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if imitation_learning:
                    assert ob['viewpoint'] == ob['gt_path'][t]
                    if t == len(ob['gt_path']) - 1:
                        a[i] = 0    # stop
                    else:
                        goal_vp = ob['gt_path'][t + 1]
                        for j, vpid in enumerate(vpids[i]):
                            if goal_vp == vpid:
                                a[i] = j
                                break
                else:
                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = 0    # Stop if arrived 
                    else:
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        for j, vpid in enumerate(vpids[i]):
                            if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                                if self.args.expert_policy == 'ndtw':
                                    dist = - cal_dtw(
                                        self.env.shortest_distances[scan], 
                                        sum(traj[i]['path'], []) + self.env.shortest_paths[scan][ob['viewpoint']][vpid][1:], 
                                        ob['gt_path'], 
                                        threshold=3.0
                                    )['nDTW']
                                elif self.args.expert_policy == 'spl':
                                    # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                                    dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                            + self.env.shortest_distances[scan][cur_vp][vpid]
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = j
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))
        return torch.from_numpy(a).cuda()

    def _teacher_action_r4r_idx(
        self, obs, idx, vpids, ended, visited_masks=None, imitation_learning=False, t=None, traj=None
    ):
        """R4R is not the shortest path. The goal location can be visited nodes.
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if i != idx:
                continue
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if imitation_learning:
                    assert ob['viewpoint'] == ob['gt_path'][t]
                    if t == len(ob['gt_path']) - 1:
                        a[i] = 0    # stop
                    else:
                        goal_vp = ob['gt_path'][t + 1]
                        for j, vpid in enumerate(vpids[i]):
                            if goal_vp == vpid:
                                a[i] = j
                                break
                else:
                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = 0    # Stop if arrived 
                    else:
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        for j, vpid in enumerate(vpids[i]):
                            if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                                if self.args.expert_policy == 'ndtw':
                                    dist = - cal_dtw(
                                        self.env.shortest_distances[scan], 
                                        sum(traj[i]['path'], []) + self.env.shortest_paths[scan][ob['viewpoint']][vpid][1:], 
                                        ob['gt_path'], 
                                        threshold=3.0
                                    )['nDTW']
                                elif self.args.expert_policy == 'spl':
                                    # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                                    dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                            + self.env.shortest_distances[scan][cur_vp][vpid]
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = j
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))
        return torch.from_numpy(a).cuda()
    
    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:            # None is the <stop> action
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                if self.env.mattersim_version == 1:
                    self.env.env.sims[i].newEpisode(ob['scan'], action, heading, elevation)
                else:
                    self.env.env.sims[i].newEpisode([ob['scan']], [action], [heading], [elevation])

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    # @profile
    def rollout(self, train_ml=None, train_rl=False, train_tcl_hist=None, train_tcl_obs=None, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        self._update_scanvp_cands(obs)

        batch_size = len(obs)
        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
        } for ob in obs]
        
        # Init the logs
        masks = []
        entropys = []
        ml_loss = 0.   
        ## cl modified  
        tcl_length=0
        sim_obs_loss = 0.
        if self.args.tcl_hist_weight or self.args.tcl_obs_weight:
            if self.args.dataset == 'landrxr' or self.args.dataset == 'landcvdn':
                inst_part, land_part, inst_mask, land_mask = get_inst_land_part_cvdn(self.args, obs)
            elif self.args.dataset == 'landr2r' or self.args.dataset == 'landr4r':
                inst_part, land_part, inst_mask, land_mask = get_inst_land_part(self.args, obs)

        ##
        
        # Language input: txt_ids, txt_masks
        language_inputs = self._language_variable(obs)
        txt_embeds = self.vln_bert('language', language_inputs)
    
        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)


        for t in range(self.args.max_action_len):
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

            # graph representation
            pano_inputs = self._panorama_feature_variable(obs)
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                          torch.sum(pano_masks, 1, keepdim=True)
            pano_embeds_all = pano_embeds
            avg_pano_embeds_all = avg_pano_embeds

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds_all[i], rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds_all[i, j])

            # navigation policy
            nav_inputs = self._nav_gmap_variable(obs, gmaps)
            nav_inputs.update(
                    self._nav_vp_variable(
                        obs, gmaps, pano_embeds, pano_inputs['cand_vpids'],
                        pano_inputs['view_lens'], pano_inputs['nav_types'],
                    )
                )

            nav_inputs.update({
                'txt_embeds': txt_embeds,
                'txt_masks': language_inputs['txt_masks'],
            })
            ## cl modified
            nav_inputs.update({
                'tcl_hist_weight': self.args.tcl_hist_weight,
                'tcl_obs_weight': self.args.tcl_obs_weight,
            })
            ##
            nav_outs = self.vln_bert('navigation', nav_inputs)
            ## cl modified
            if train_tcl_obs and self.feedback=='teacher':
                original_ob_embeds = nav_outs['cl_vp_embeds']
                if self.args.use_aosm_obs_cl:
                    tcl_length+=1
                    # word-level textual feature
                    word_land_mask=torch.from_numpy(land_mask).cuda() # [bs,num_words]
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
            ##
            if self.args.fusion == 'local':
                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['vp_cand_vpids']
            elif self.args.fusion == 'global':
                nav_logits = nav_outs['global_logits']
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_logits = nav_outs['fused_logits']
                nav_vpids = nav_inputs['gmap_vpids']

            nav_probs = torch.softmax(nav_logits, 1)

            # update graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    gmap.node_stop_scores[i_vp] = {
                        'stop': nav_probs[i, 0].data.item(),
                    }
                                        
            if train_ml is not None:
                # Supervised training
                if self.args.dataset == 'r2r' or self.args.dataset == 'landr2r':
                    nav_targets = self._teacher_action_r4r(
                        obs, nav_vpids, ended, 
                        visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                        imitation_learning=(self.feedback=='teacher'), t=t, traj=traj
                    )
                elif self.args.dataset == 'r4r' or self.args.dataset == 'landr4r':
                    nav_targets = self._teacher_action_r4r(
                        obs, nav_vpids, ended, 
                        visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                        imitation_learning=(self.feedback=='teacher'), t=t, traj=traj
                    )
                elif self.args.dataset == 'cvdn' or self.args.dataset == 'landcvdn':
                    nav_targets = self._teacher_action_r4r(
                        obs, nav_vpids, ended,
                        visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                        imitation_learning=(self.feedback == 'teacher'), t=t, traj=traj
                    )
                elif self.args.dataset == 'rxr' or self.args.dataset == 'landrxr':
                    nav_targets = self._teacher_action_r4r(
                        obs, nav_vpids, ended,
                        visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                        imitation_learning=(self.feedback == 'teacher'), t=t, traj=traj
                    )
                # print(t, nav_logits, nav_targets)
                ml_loss += self.criterion(nav_logits, nav_targets)
                # print(t, 'ml_loss', ml_loss.item(), self.criterion(nav_logits, nav_targets).item())
                                                              
            # Determinate the next navigation viewpoint
            if self.feedback == 'teacher':
                a_t = nav_targets                 # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = nav_logits.max(1)        # student forcing - argmax
                a_t = a_t.detach() 
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach() 
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # hyper-param
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs['gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Determine stop actions
            if self.feedback == 'teacher' or self.feedback == 'sample': # in training
                # a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []  
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])   

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf')}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))
                    if self.args.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                            }

            # new observation and update graph
            obs = self.env._get_obs()
            self._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break
        
        ## cl modified    
        if train_tcl_hist and self.feedback == 'teacher':
            gmap_masks = nav_outs['gmap_masks']
            gmap_lens = gmap_masks.sum(dim=1).cpu().numpy()
            gmap_embeds = nav_outs['cl_gmap_embeds']
            tmp_embeds = torch.stack([torch.ones_like(gmap_embeds[:,0,:]).cuda() for _ in range(self.args.max_graph_len-gmap_embeds.size(1))],dim=1)

            traj_embeds = torch.cat([gmap_embeds, tmp_embeds],dim=1 ) # PAD to max_action_len
            #traj_embeds=torch.stack(traj_embeds,dim=0).permute(1,0,2) # [bs,max_action,dim]
            traj_mask=1-length2mask(gmap_lens,size=self.args.max_graph_len).to(dtype=torch.float) # [bs,max_action]
            
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
            sentence_output=txt_embeds[:,0,:].squeeze(1)
            sentence_output=sentence_output/sentence_output.norm(dim=-1,keepdim=True) # [bs,dim]
            
            word_inst_mask=torch.from_numpy(inst_mask).cuda() # [bs,num_words]
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
            
            self.loss+=sim_hist_loss*train_tcl_hist
        
            # enqueue and dequeue
            if self.args.use_mem_bank:
                self._dequeue_and_enqueue_inst_hist(traj_output.detach(),frame_features.detach(),sentence_output.detach(),word_inst_features.detach())

        if train_tcl_obs and self.feedback == 'teacher':
            sim_obs_loss=sim_obs_loss/tcl_length
            self.logs['CL_obs_loss'].append((sim_obs_loss).item())

            if self.args.norm_cl_weight and type(sim_obs_loss) is not float:
                sim_obs_loss = sim_obs_loss / (sim_obs_loss.item() + 1e-12)
            
            self.loss+=sim_obs_loss*train_tcl_obs
            if self.args.use_mem_bank:
                self._dequeue_and_enqueue_land_obs()
        ##
        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
            self.loss += ml_loss
            self.logs['IL_loss'].append(ml_loss.item())

        return traj