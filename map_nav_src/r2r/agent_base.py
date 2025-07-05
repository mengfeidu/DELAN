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
from utils.logger import print_progress


class BaseAgent(object):
    ''' Base class for an REVERIE agent to generate and save trajectories. '''

    def __init__(self, env):
        self.env = env
        self.results = {}

    def get_results(self, detailed_output=False):
        output = []
        for k, v in self.results.items():
            output.append({'instr_id': k, 'trajectory': v['path']})
            if detailed_output:
                output[-1]['details'] = v['details']
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0

        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj
                if looped:
                    break

    def test_viz(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj
        else:   # Do a full round
            while True:
                for traj in self.rollout_viz(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj
                if looped:
                    break

class Seq2SeqAgent(BaseAgent):
    env_actions = {
      'left': (0, -1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0, -1), # down
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
            if self.args.tcl_hist_weight or self.args.tcl_obs_weight:
                self.contrastive_loss = DDP(self.contrastive_loss, device_ids=[self.rank], find_unused_parameters=True)
                self.contrastive_loss_obs = DDP(self.contrastive_loss_obs, device_ids=[self.rank], find_unused_parameters=True)          

        self.models = (self.vln_bert, self.critic)
        self.device = torch.device('cuda:%d'%self.rank) 

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
        
        if self.args.tcl_hist_weight or self.args.tcl_obs_weight:
            self.contrastive_loss_optimizer = optimizer(self.contrastive_loss.parameters(), lr=self.args.lr)
            self.contrastive_loss_obs_optimizer = optimizer(self.contrastive_loss_obs.parameters(), lr=self.args.lr)
            self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer, self.contrastive_loss_optimizer, self.contrastive_loss_obs_optimizer)
        else:
            self.optimizers = (self.vln_bert_optimizer, self.critic_optimizer)
        # Evaluations
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.args.ignoreid, reduction='sum')

        # memory bank
        if args.use_mem_bank:
            self.queue_ptr_hist=torch.zeros(1,dtype=torch.long).cuda()
            self.queue_ptr_obs=torch.zeros(1,dtype=torch.long).cuda()
            self.queue_temp_ptr_obs=torch.zeros(1,dtype=torch.long).cuda()
            # hist-inst tcl
            self.queue_traj=torch.randn(args.capacity,args.hidden_dim).cuda()
            self.queue_traj=nn.functional.normalize(self.queue_traj, dim=-1).cuda()

            self.queue_frame=torch.randn(args.capacity,args.max_graph_len,args.hidden_dim).cuda()
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
            self.temp_queue_patch=torch.randn(args.batch_size*args.max_graph_len,args.max_obs_num,args.hidden_dim).cuda()
            self.temp_queue_patch=nn.functional.normalize(self.temp_queue_patch, dim=-1).cuda()
            self.temp_queue_word_land=torch.randn(args.batch_size*args.max_graph_len,args.max_instr_len,args.hidden_dim).cuda()
            self.temp_queue_word_land=nn.functional.normalize(self.temp_queue_word_land, dim=-1).cuda()
            

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)
        
    def _build_model(self):
        raise NotImplementedError('child class should implement _build_model: self.vln_bert & self.critic')

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

    def _dequeue_and_enqueue_land_obs_multi(self):
        batch_len_size = int(self.queue_temp_ptr_obs)
        ptr = int(self.queue_ptr_obs)
        if ptr + batch_len_size > self.args.capacity:
            batch_len_size = self.args.capacity - ptr
        self.queue_patch_mean[ptr:ptr + batch_len_size, :] = self.temp_queue_patch_mean[:batch_len_size, :]
        self.queue_patch[ptr:ptr + batch_len_size, :, :] = self.temp_queue_patch[:batch_len_size, :, :]
        self.queue_word_land_mean[ptr:ptr + batch_len_size, :] = self.temp_queue_word_land_mean[:batch_len_size, :]
        self.queue_word_land[ptr:ptr + batch_len_size, :, :] = self.temp_queue_word_land[:batch_len_size, :, :]
        
        ptr = (ptr + batch_len_size) % self.args.capacity
        self.queue_ptr_obs[0] = ptr
        
        self.queue_temp_ptr_obs[0] = 0 # clear temporary queue

    def _dequeue_and_enqueue_temp_land_obs_multi(self, traj, frame, sent, word_inst):
        batch_size = traj.shape[0]
        ptr = int(self.queue_ptr_obs)
        if ptr + batch_size > self.args.capacity:
            batch_size = self.args.capacity - ptr
        self.temp_queue_patch_mean[ptr:ptr + batch_size, :] = traj[:batch_size, :]
        self.temp_queue_patch[ptr:ptr + batch_size, :, :] = frame[:batch_size, :, :]
        self.temp_queue_word_land_mean[ptr:ptr + batch_size, :] = sent[:batch_size, :]
        self.temp_queue_word_land[ptr:ptr + batch_size, :] = word_inst[:batch_size, :, :]
        
        ptr = (ptr + batch_size) % self.args.capacity # move pointer
        self.queue_ptr_hist[0] = ptr
        
    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None, viz=False):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.vln_bert.train()
            self.critic.train()
        else:
            self.vln_bert.eval()
            self.critic.eval()
        if viz:
            super().test_viz(iters=iters)
        else:
            super().test(iters=iters)

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.vln_bert.train()
        self.critic.train()
        if self.args.tcl_hist_weight or self.args.tcl_obs_weight:
            self.contrastive_loss.train()
            self.contrastive_loss_obs.train()

        self.losses = []
        for iter in range(1, n_iters + 1):

            self.vln_bert_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            if self.args.tcl_hist_weight or self.args.tcl_obs_weight:
                self.contrastive_loss_optimizer.zero_grad()
                self.contrastive_loss_obs_optimizer.zero_grad()

            self.loss = 0

            if self.args.train_alg == 'imitation':
                self.feedback = 'teacher'
                self.rollout(
                    train_ml=1., train_rl=False, train_tcl_hist=self.args.tcl_hist_weight, train_tcl_obs=self.args.tcl_obs_weight, **kwargs
                )
            elif self.args.train_alg == 'dagger': 
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(
                        train_ml=self.args.ml_weight, train_rl=False, train_tcl_hist=self.args.tcl_hist_weight, train_tcl_obs=self.args.tcl_obs_weight, **kwargs
                    )
                self.feedback = 'expl_sample' if self.args.expl_sample else 'sample'
                self.rollout(train_ml=1, train_rl=False, train_tcl_hist=self.args.tcl_hist_weight, train_tcl_obs=self.args.tcl_obs_weight, **kwargs)
            else:
                if self.args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(
                        train_ml=self.args.ml_weight, train_rl=False, train_tcl_hist=self.args.tcl_hist_weight, train_tcl_obs=self.args.tcl_obs_weight, **kwargs
                    )
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, train_tcl_hist=self.args.tcl_hist_weight, train_tcl_obs=self.args.tcl_obs_weight, **kwargs)

            # print(self.rank, iter, self.loss)
            self.loss.backward()

            torch.nn.utils.clip_grad_norm_(self.vln_bert.parameters(), 40.0)

            self.vln_bert_optimizer.step()
            self.critic_optimizer.step()
            if self.args.tcl_hist_weight or self.args.tcl_obs_weight:
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
        states = torch.load(path, map_location=lambda storage, loc: storage)

        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            state_dict = states[name]['state_dict']
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
                if not list(model_keys)[0].startswith('module.') and list(load_keys)[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                if list(model_keys)[0].startswith('module.') and (not list(load_keys)[0].startswith('module.')):
                    state_dict = {'module.'+k: v for k, v in state_dict.items()}
                same_state_dict = {}
                extra_keys = []
                for k, v in state_dict.items():
                    if k in model_keys:
                        same_state_dict[k] = v
                    else:
                        extra_keys.append(k)
                state_dict = same_state_dict
                print('Extra keys in state_dict: %s' % (', '.join(extra_keys)))
            state.update(state_dict)
            model.load_state_dict(state)
            if self.args.resume_optimizer:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("vln_bert", self.vln_bert, self.vln_bert_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['vln_bert']['epoch'] - 1


