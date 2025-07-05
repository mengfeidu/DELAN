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

from utils.misc import length2mask

from r2r.agent_cmt import Seq2SeqCMTAgent
from models.model_HAMT import CrossEn, AllGather, ContrastiveLoss, ContrastiveLoss_obs
from r2r.data_utils import get_inst_land_part_cvdn

allgather = AllGather.apply
class NavCMTAgent(Seq2SeqCMTAgent):
    def rollout(self, train_ml=None, train_rl=True, train_tcl_hist=None, train_tcl_obs=None, reset=True):
        """ The rewards are different from R2R:
            1) do not require alignment with gt path (no ndtw reward)
            2) multiple end viewpoints
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
        for i, ob in enumerate(obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']

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
        GT_insts, GT_lands = [], []
        if self.args.tcl_hist_weight and self.args.tcl_obs_weight:
            try:
                inst_part, land_part, inst_mask, land_mask = get_inst_land_part_cvdn(obs)
            except:
                print('check')
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
            logit = t_outputs[0]
            original_ob_embeds=t_outputs[-1]
            if self.feedback == 'sample':
                h_t = t_outputs[1]
                hidden_states.append(h_t)

            if train_ml is not None:
                # Supervised training
                target = self._teacher_action(obs, ended)
                ml_loss += self.criterion(logit, target)

            if train_tcl_obs and self.feedback=='teacher':
                if not any_proc_ended and self.args.use_aosm_obs_cl:
                    tcl_length+=1
                    # word-level textual feature
                    word_land_mask=torch.from_numpy(land_mask).cuda() # [bs,num_words]
                    word_land_features=txt_embeds[self.args.cl_lang_layer]/txt_embeds[self.args.cl_lang_layer].norm(dim=-1,keepdim=True) # [bs,num_patches,dim]
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
                reward = np.zeros(batch_size, np.float32)
                mask = np.ones(batch_size, np.float32)
                for i, ob in enumerate(obs):
                    dist[i] = ob['distance']

                    if ended[i]:
                        reward[i] = 0.0
                        mask[i] = 0.0
                    else:
                        action_idx = cpu_a_t[i]
                        # Target reward
                        if action_idx == -1:                              # If the action now is end
                            if dist[i] == 0.:                             # Correct
                                reward[i] = 2.0
                            else:                                         # Incorrect
                                reward[i] = -2.0
                        else:                                             # The action is not end
                            # rewards (distance)
                            reward[i] = - (dist[i] - last_dist[i])  # this distance is not normalized
                            if reward[i] > 0.0:                           # Quantification
                                reward[i] = 1.0
                            elif reward[i] < 0.0:
                                reward[i] = -1.0
                            else:
                                reward[i] = 0
                rewards.append(reward)
                masks.append(mask)
                last_dist[:] = dist

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
            sentence_output=txt_embeds[self.args.cl_lang_layer][:,0,:].squeeze(1)
            sentence_output=sentence_output/sentence_output.norm(dim=-1,keepdim=True) # [bs,dim]
            
            # word-level textual feature
            word_inst_mask=torch.from_numpy(inst_mask).cuda() # [bs,num_words]
            word_inst_features=txt_embeds[self.args.cl_lang_layer]/txt_embeds[self.args.cl_lang_layer].norm(dim=-1,keepdim=True) # [bs,num_words,dim]
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
