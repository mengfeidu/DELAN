import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel

from .vlnbert_init import get_vlnbert_models

class VLNBert(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        
    def forward(self, mode, batch):
        batch = collections.defaultdict(lambda: None, batch)
        
        if mode == 'language':            
            txt_embeds = self.vln_bert(mode, batch)
            return txt_embeds
        
        elif mode == 'landmark':            
            land_embeds = self.vln_bert(mode, batch)
            return land_embeds
        
        elif mode == 'panorama':
            batch['view_img_fts'] = self.drop_env(batch['view_img_fts'])
            if 'obj_img_fts' in batch:
                batch['obj_img_fts'] = self.drop_env(batch['obj_img_fts'])
            # cl para
            batch['tcl_hist_weight'] = self.args.tcl_hist_weight
            batch['tcl_obs_weight'] = self.args.tcl_obs_weight
            batch['max_obs_num'] = self.args.max_obs_num
            pano_embeds, pano_masks = self.vln_bert(mode, batch)
            return pano_embeds, pano_masks

        elif mode == 'navigation':
            outs = self.vln_bert(mode, batch)
            return outs

        else:
            raise NotImplementedError('wrong mode: %s'%mode)


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()


class CrossEn(nn.Module):
    def __init__(self):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


class ContrastiveLoss(nn.Module):
    def __init__(self, args, obs_flag=False):
        super(ContrastiveLoss, self).__init__()
        self.args=args
        embed_dim=768
        num_words=args.max_instr_len
        if args.cl_duet:
            if obs_flag:
                num_frames=args.max_obs_num
            else:
                num_frames=args.max_graph_len
        else:
            num_frames=args.max_action_len
        # coarse-grained contrast weights
        self.global_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)
        # cross-grained contrast weights
        self.word_logit_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
        self.frame_logit_weight = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
        # fine-grained contrast weights
        self.local_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)
        self.frame_mat_weight = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
        self.word_mat_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
        self.frame_mat_weight2 = nn.parameter.Parameter(torch.eye(num_frames), requires_grad=True)
        self.word_mat_weight2 = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
        
        self.loss_fct = CrossEn()

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self,traj_output=None,frame_features=None,frame_mask=None,sentence_output=None,word_features=None,attention_mask=None,mem_neg=None):
        """
        traj_output: [bs,dim]
        frame_features: [bs,num_frames,dim]
        frame_mask: [bs,num_frames], value can be 0 or -1000000.0
        sentence_output: [bs,dim]
        word_features: [bs,num_words,dim]
        attention_mask: [bs,num_words], value can be 0 or -1000000.0
        mem_neg={
            'traj':self.queue_traj.clone().detach(),
            'frame':self.queue_frame.clone().detach(),
            'sent':self.queue_sent.clone().detach(),
            'word_inst':self.queue_word_inst.clone().detach(),
            } if self.args.use_mem_bank else None
        """
        traj_output=traj_output.contiguous()
        frame_features=frame_features.contiguous()
        sentence_output=sentence_output.contiguous()
        word_features=word_features.contiguous()

        if frame_mask is not None:
            frame_mask=frame_mask.unsqueeze(dim=1) # [bs,1,num_frames]
        if attention_mask is not None:
            attention_mask=attention_mask.unsqueeze(dim=-1) # [bs,num_words,1]
        
        # video-sentence score
        if self.args.traj_sent:
            traj_sent_sim=torch.matmul(torch.matmul(sentence_output,self.global_mat_weight),traj_output.t()) # [bs,bs]
        else:
            traj_sent_sim = 0.
        # video-word score
        if self.args.video_word:
            video_word_sim=torch.sum(torch.matmul(word_features, traj_output.t()) \
                * torch.matmul(torch.softmax(torch.matmul(word_features, traj_output.t())/1e-2, dim=1).permute(0,2,1), self.word_logit_weight).permute(0,2,1), dim=1) # [bs,bs]
        else:
            video_word_sim=0.
        # sentence-frame score
        if self.args.sentence_frame:
            sentence_frame_sim=torch.sum(torch.matmul(sentence_output, frame_features.permute(0, 2, 1))\
                * torch.matmul(torch.softmax(torch.matmul(sentence_output, frame_features.permute(0, 2, 1))/1e-2, dim=-1), self.frame_logit_weight), dim=-1).t() # [bs,bs]
        else:
            sentence_frame_sim = 0
        # frame-word score
        if self.args.frame_word:
            frame_word_sim=self._attention_over_fine_grained_sim_matrix(word_features,frame_features) # [bs,bs]
        else:
            frame_word_sim = 0
        # compute loss
        sim_matrix=(traj_sent_sim+video_word_sim+sentence_frame_sim+frame_word_sim)/(self.args.traj_sent+self.args.video_word+self.args.sentence_frame+self.args.frame_word) # [bs,bs]
        
        # consider memory bank
        if self.args.use_mem_bank:
            # x_bs * h_cap
            hist_neg_sim_matrix=self.multi_grained_sim(mem_neg['traj'],mem_neg['frame'],sentence_output,word_features) # [x_bs,h_cap]
            hist_sim_matrix_final=torch.cat((sim_matrix,hist_neg_sim_matrix),dim=1)
        else:
            hist_sim_matrix_final=sim_matrix
        sim_loss1=self.loss_fct(hist_sim_matrix_final)
        
        if self.args.use_mem_bank:
            # x_cap * h_bs
            inst_neg_sim_matrix=self.multi_grained_sim(traj_output,frame_features,mem_neg['sent'],mem_neg['word_inst']) # [x_cap,h_bs]
            inst_sim_matrix_final=torch.cat((sim_matrix,inst_neg_sim_matrix),dim=0) # examine the gradient!!!
        else:
            inst_sim_matrix_final=sim_matrix
        sim_loss2=self.loss_fct(inst_sim_matrix_final.T)
        sim_loss=(sim_loss1+sim_loss2)/2
        return sim_loss
    
    def multi_grained_sim(self,traj,frame,sent,word):
        # calculate various multi-grained sim
        # video-sentence score
        traj_sent_sim=torch.matmul(torch.matmul(sent,self.global_mat_weight),traj.t()) # [bs,bs]
        
        # video-word score
        video_word_sim=torch.sum(torch.matmul(word, traj.t()) \
            * torch.matmul(torch.softmax(torch.matmul(word, traj.t())/1e-2, dim=1).permute(0,2,1), self.word_logit_weight).permute(0,2,1), dim=1) # [bs,bs]
        
        # sentence-frame score
        sentence_frame_sim=torch.sum(torch.matmul(sent, frame.permute(0, 2, 1))\
            * torch.matmul(torch.softmax(torch.matmul(sent, frame.permute(0, 2, 1))/1e-2, dim=-1), self.frame_logit_weight), dim=-1).t() # [bs,bs]

        # frame-word score
        frame_word_sim=self._attention_over_fine_grained_sim_matrix(word,frame) # [bs,bs]

        sim_matrix=(traj_sent_sim+video_word_sim+sentence_frame_sim+frame_word_sim)/4 # [bs,bs]
        
        return sim_matrix

    def _attention_over_fine_grained_sim_matrix(self, word_features, frame_features):
        bs_video, num_frames, dim_video = frame_features.shape
        bs_text, num_words, dim_text = word_features.shape
        fine_grained_sim_scores = torch.matmul(torch.matmul(word_features.view(-1, dim_text), self.local_mat_weight), frame_features.view(-1, dim_video).t()).view(bs_text, num_words, bs_video, num_frames)  # [bs_text, num_words, bs_video, num_frames]

        word_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=1).permute(0,2,3,1), self.word_mat_weight).permute(0,3,1,2) * fine_grained_sim_scores, dim=1)               # [bs_text, bs_video, num_frames]
        frame_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=-1), self.frame_mat_weight) * fine_grained_sim_scores, dim=-1)                                             # [bs_text, num_words, bs_video]

        sent2frame_logits = torch.sum(torch.matmul(torch.softmax(word_level_logit/1e-2, dim=-1),self.frame_mat_weight2) * word_level_logit, dim=-1)                                # [bs_text, bs_video]
        video2word_logits = torch.sum(torch.matmul(torch.softmax(frame_level_logit/1e-2, dim=1).permute(0,2,1), self.word_mat_weight2).permute(0,2,1) * frame_level_logit, dim=1)  # [bs_text, bs_video]

        return (sent2frame_logits + video2word_logits) / 2

    def _attention_over_fine_grained_sim_matrix2(self, word_features, attention_mask, frame_features, frame_mask):
        """
        attention_mask : [bs,num_words]
        frame_mask : [bs,num_frames]
        """
        bs_video, num_frames, dim_video = frame_features.shape
        bs_text, num_words, dim_text = word_features.shape
        # [bs_text, num_words, bs_video, num_frames]
        fine_grained_sim_scores = torch.matmul(torch.matmul(word_features.view(-1, dim_text), self.local_mat_weight), frame_features.view(-1, dim_video).t()).view(bs_text, num_words, bs_video, num_frames)  # [bs_text, num_words, bs_video, num_frames]
        fine_grained_sim_scores_mask=fine_grained_sim_scores/1e-2 + attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1) + frame_mask.unsqueeze(dim=0).unsqueeze(dim=0)
        
        word_level_sim = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores_mask, dim=1).permute(0,2,3,1),self.word_mat_weight).permute(0,3,1,2) * fine_grained_sim_scores, dim=1) # [bs_text,bs_video,num_frames]
        frame_level_sim = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores_mask, dim=-1), self.frame_mat_weight) * fine_grained_sim_scores, dim=-1) # [bs_text,num_words,bs_video]

        word_level_sim_mask = word_level_sim/1e-2 + frame_mask.unsqueeze(dim=0)
        frame_level_sim_mask = frame_level_sim/1e-2 + attention_mask.unsqueeze(dim=-1)
        
        sent2frame_sims = torch.sum(torch.matmul(torch.softmax(word_level_sim_mask, dim=-1),self.frame_mat_weight2) * word_level_sim, dim=-1) # [bs_text,bs_video]
        video2word_sims = torch.sum(torch.matmul(torch.softmax(frame_level_sim_mask, dim=1).permute(0,2,1), self.word_mat_weight2).permute(0,2,1) * frame_level_sim, dim=1) # [bs_text,bs_video]

        return (sent2frame_sims + video2word_sims) / 2

class ContrastiveLoss_obs(nn.Module):
    def __init__(self, args):
        super(ContrastiveLoss_obs, self).__init__()
        self.args=args
        embed_dim=768
        num_words=args.max_instr_len
        num_patchs=args.max_obs_num
        # fine-grained contrast weights
        self.local_mat_weight = nn.parameter.Parameter(torch.eye(embed_dim), requires_grad=True)
        self.patch_mat_weight = nn.parameter.Parameter(torch.eye(num_patchs), requires_grad=True)
        self.word_mat_weight = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)
        self.patch_mat_weight2 = nn.parameter.Parameter(torch.eye(num_patchs), requires_grad=True)
        self.word_mat_weight2 = nn.parameter.Parameter(torch.eye(num_words), requires_grad=True)

        self.loss_fct = CrossEn()
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self,patch_features=None,patch_mask=None,word_features=None,attention_mask=None,mem_neg=None):
        """
        patch_features: [bs,num_frames,dim]
        patch_mask: [bs,num_frames], value can be 0 or -1000000.0
        word_features: [bs,num_words,dim]
        attention_mask: [bs,num_words], value can be 0 or -1000000.0
        mem_neg_obs={
            'patch':self.queue_patch.clone().detach(),
            'word_land':self.queue_word_land.clone().detach(),
        }
        """
        patch_features=patch_features.contiguous()
        word_features=word_features.contiguous()

        if patch_mask is not None:
            patch_mask=patch_mask.unsqueeze(dim=1) # [bs,1,num_frames]
        if attention_mask is not None:
            attention_mask=attention_mask.unsqueeze(dim=-1) # [bs,num_words,1]
        
        # frame-word score
        patch_word_sim=self._attention_over_fine_grained_sim_matrix(word_features,patch_features) # [x_bs,o_bs]

        # compute loss
        sim_matrix=patch_word_sim
        
        if self.args.use_mem_bank:
            # x_bs * o_cap
            obs_neg_sim_matrix=self._attention_over_fine_grained_sim_matrix(word_features,mem_neg['patch']) # [x_bs,o_cap]
            obs_sim_matrix_final=torch.cat((sim_matrix,obs_neg_sim_matrix),dim=1)
        else:
            obs_sim_matrix_final=sim_matrix
        sim_loss1=self.loss_fct(obs_sim_matrix_final)
        
        if self.args.use_mem_bank:
            # x_cap * o_bs
            land_neg_sim_matrix=self._attention_over_fine_grained_sim_matrix(mem_neg['word_land'],patch_features) # [x_cap,o_bs]
            land_sim_matrix_final=torch.cat((sim_matrix,land_neg_sim_matrix),dim=0)
        else:
            land_sim_matrix_final=sim_matrix
        sim_loss2=self.loss_fct(land_sim_matrix_final.T)
        
        sim_loss=(sim_loss1+sim_loss2)/2
        return sim_loss

    def aa():
        pass

    def _attention_over_fine_grained_sim_matrix(self, word_features, patch_features):
        bs_pano, num_patches, dim_pano = patch_features.shape
        bs_text, num_words, dim_text = word_features.shape
        fine_grained_sim_scores = torch.matmul(torch.matmul(word_features.view(-1, dim_text), self.local_mat_weight), patch_features.view(-1, dim_pano).t()).view(bs_text, num_words, bs_pano, num_patches)  # [bs_text, num_words, bs_pano, num_patches]

        word_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=1).permute(0,2,3,1), self.word_mat_weight).permute(0,3,1,2) * fine_grained_sim_scores, dim=1)               # [bs_text, bs_pano, num_patches]
        patch_level_logit = torch.sum(torch.matmul(torch.softmax(fine_grained_sim_scores/1e-2, dim=-1), self.patch_mat_weight) * fine_grained_sim_scores, dim=-1)                                             # [bs_text, num_words, bs_pano]

        sent2patch_logits = torch.sum(torch.matmul(torch.softmax(word_level_logit/1e-2, dim=-1),self.patch_mat_weight2) * word_level_logit, dim=-1)                                # [bs_text, bs_pano]
        pano2word_logits = torch.sum(torch.matmul(torch.softmax(patch_level_logit/1e-2, dim=1).permute(0,2,1), self.word_mat_weight2).permute(0,2,1) * patch_level_logit, dim=1)  # [bs_text, bs_pano]

        return (sent2patch_logits + pano2word_logits) / 2

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = args.local_rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )

