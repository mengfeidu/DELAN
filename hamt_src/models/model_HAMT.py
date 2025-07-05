import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import length2mask

from models.vlnbert_init import get_vlnbert_models

class VLNBertCMT(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        
    def forward(self, mode, txt_ids=None, txt_masks=None, txt_embeds=None, 
                hist_img_feats=None, hist_ang_feats=None, 
                hist_pano_img_feats=None, hist_pano_ang_feats=None,
                hist_embeds=None, hist_lens=None, ob_step=None,
                ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None, 
                ob_masks=None, return_states=False):

        if mode == 'language':
            encoded_sentence = self.vln_bert(mode, txt_ids=txt_ids, txt_masks=txt_masks)
            return encoded_sentence

        elif mode == 'history':
            # history inputs per step
            if hist_img_feats is not None:
                hist_img_feats = self.drop_env(hist_img_feats)
            if hist_pano_img_feats is not None:
                hist_pano_img_feats = self.drop_env(hist_pano_img_feats)
            if ob_step is not None:
                ob_step_ids = torch.LongTensor([ob_step]).cuda()
            else:
                ob_step_ids = None
            hist_embeds = self.vln_bert(mode, hist_img_feats=hist_img_feats, 
                hist_ang_feats=hist_ang_feats, ob_step_ids=ob_step_ids,
                hist_pano_img_feats=hist_pano_img_feats, 
                hist_pano_ang_feats=hist_pano_ang_feats)
            return hist_embeds

        elif mode == 'visual':
            hist_embeds = torch.stack(hist_embeds, 1)
            hist_masks = length2mask(hist_lens, size=hist_embeds.size(1)).logical_not()
            
            ob_img_feats = self.drop_env(ob_img_feats)
            
            act_logits, txt_embeds, hist_embeds, ob_embeds, original_ob_embeds = self.vln_bert(
                mode, txt_embeds=txt_embeds, txt_masks=txt_masks,
                hist_embeds=hist_embeds, hist_masks=hist_masks,
                ob_img_feats=ob_img_feats, ob_ang_feats=ob_ang_feats, 
                ob_nav_types=ob_nav_types, ob_masks=ob_masks)

            if return_states:
                if self.args.no_lang_ca:
                    states = hist_embeds[:, 0]
                else:
                    states = txt_embeds[:, 0] * hist_embeds[:, 0]   # [CLS]
                return act_logits, states, hist_embeds,original_ob_embeds
            return (act_logits, hist_embeds,original_ob_embeds)


class VLNBertCausalCMT(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        
    def forward(
        self, mode, txt_ids=None, txt_masks=None, txt_embeds=None,
        hist_img_feats=None, hist_ang_feats=None, 
        hist_pano_img_feats=None, hist_pano_ang_feats=None, ob_step=0,
        new_hist_embeds=None, new_hist_masks=None,
        prefix_hiddens=None, prefix_masks=None,
        ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None,
        ob_masks=None, return_states=False, batch_size=None,
    ):

        if mode == 'language':
            encoded_sentence = self.vln_bert(mode, txt_ids=txt_ids, txt_masks=txt_masks)
            return encoded_sentence

        elif mode == 'history':
            if ob_step == 0:
                hist_step_ids = torch.arange(1).long()
            else:
                hist_step_ids = torch.arange(2).long() + ob_step - 1
            hist_step_ids = hist_step_ids.unsqueeze(0)

            # history inputs per step
            if hist_img_feats is not None:
                hist_img_feats = self.drop_env(hist_img_feats)
            if hist_pano_img_feats is not None:
                hist_pano_img_feats = self.drop_env(hist_pano_img_feats)

            hist_embeds = self.vln_bert(
                mode, hist_img_feats=hist_img_feats, 
                hist_ang_feats=hist_ang_feats, 
                hist_pano_img_feats=hist_pano_img_feats, 
                hist_pano_ang_feats=hist_pano_ang_feats,
                hist_step_ids=hist_step_ids,
                batch_size=batch_size
            )
            return hist_embeds

        elif mode == 'visual':
            ob_img_feats = self.drop_env(ob_img_feats)
            
            act_logits, prefix_hiddens, states = self.vln_bert(
                mode, txt_embeds=txt_embeds, txt_masks=txt_masks,
                ob_img_feats=ob_img_feats, ob_ang_feats=ob_ang_feats, 
                ob_nav_types=ob_nav_types, ob_masks=ob_masks,
                new_hist_embeds=new_hist_embeds, new_hist_masks=new_hist_masks,
                prefix_hiddens=prefix_hiddens, prefix_masks=prefix_masks
            )

            if return_states:
                return act_logits, prefix_hiddens, states
            return (act_logits, prefix_hiddens)


class VLNBertMMT(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        
    def forward(
        self, mode, txt_ids=None, txt_masks=None, txt_embeds=None, 
        hist_img_feats=None, hist_ang_feats=None, 
        hist_pano_img_feats=None, hist_pano_ang_feats=None,
        hist_embeds=None, hist_masks=None, ob_step=None,
        ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None, 
        ob_masks=None, return_states=False, batch_size=None,
        prefix_embeds=None, prefix_masks=None
    ):

        if mode == 'language':
            encoded_sentence = self.vln_bert(mode, txt_ids=txt_ids, txt_masks=txt_masks)
            return encoded_sentence

        elif mode == 'history':
            if hist_img_feats is None:
                # only encode [sep] token
                hist_step_ids = torch.zeros((batch_size, 1), dtype=torch.long)
            else:
                # encode the new observation and [sep]
                hist_step_ids = torch.arange(2).long().expand(batch_size, -1) + ob_step - 1
            
            # history inputs per step
            if hist_img_feats is not None:
                hist_img_feats = self.drop_env(hist_img_feats)
            if hist_pano_img_feats is not None:
                hist_pano_img_feats = self.drop_env(hist_pano_img_feats)
            
            new_hist_embeds = self.vln_bert(
                mode, hist_img_feats=hist_img_feats, 
                hist_ang_feats=hist_ang_feats, 
                hist_step_ids=hist_step_ids,
                hist_pano_img_feats=hist_pano_img_feats, 
                hist_pano_ang_feats=hist_pano_ang_feats,
                batch_size=batch_size,
            )
            return new_hist_embeds

        elif mode == 'visual':
            ob_img_feats = self.drop_env(ob_img_feats)
            
            outs = self.vln_bert(
                mode, txt_embeds=txt_embeds, txt_masks=txt_masks,
                hist_embeds=hist_embeds, hist_masks=hist_masks,
                ob_img_feats=ob_img_feats, ob_ang_feats=ob_ang_feats, 
                ob_nav_types=ob_nav_types, ob_masks=ob_masks,
                prefix_embeds=prefix_embeds, prefix_masks=prefix_masks
            )

            act_logits, hist_state = outs[:2]

            if return_states:
                return (act_logits, hist_state) + outs[2:]

            return (act_logits, ) + outs[2:]


class VLNBertCMT3(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        
    def forward(
        self, mode, txt_ids=None, txt_masks=None,
        hist_img_feats=None, hist_ang_feats=None, 
        hist_pano_img_feats=None, hist_pano_ang_feats=None, ob_step=0,
        ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None,
        ob_masks=None, return_states=False, 
        txt_embeds=None, hist_in_embeds=None, hist_out_embeds=None, hist_masks=None
    ):

        if mode == 'language':
            encoded_sentence = self.vln_bert(mode, txt_ids=txt_ids, txt_masks=txt_masks)
            return encoded_sentence

        elif mode == 'history':
            if ob_step == 0:
                hist_step_ids = torch.arange(1).long()
            else:
                hist_step_ids = torch.arange(2).long() + ob_step - 1
            hist_step_ids = hist_step_ids.unsqueeze(0)

            # history inputs per step
            if hist_img_feats is not None:
                hist_img_feats = self.drop_env(hist_img_feats)
            if hist_pano_img_feats is not None:
                hist_pano_img_feats = self.drop_env(hist_pano_img_feats)

            hist_in_embeds, hist_out_embeds = self.vln_bert(
                mode, hist_img_feats=hist_img_feats, 
                hist_ang_feats=hist_ang_feats, 
                hist_pano_img_feats=hist_pano_img_feats, 
                hist_pano_ang_feats=hist_pano_ang_feats,
                hist_step_ids=hist_step_ids,
                hist_in_embeds=hist_in_embeds,
                hist_out_embeds=hist_out_embeds,
                hist_masks=hist_masks
            )
            return hist_in_embeds, hist_out_embeds

        elif mode == 'visual':
            ob_img_feats = self.drop_env(ob_img_feats)
            
            act_logits, states = self.vln_bert(
                mode, txt_embeds=txt_embeds, txt_masks=txt_masks,
                hist_out_embeds=hist_out_embeds, hist_masks=hist_masks,
                ob_img_feats=ob_img_feats, ob_ang_feats=ob_ang_feats, 
                ob_nav_types=ob_nav_types, ob_masks=ob_masks,
            )

            if return_states:
                return act_logits, states
            return (act_logits, )


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

############tcl
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
    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()
        self.args=args
        embed_dim=768
        num_words=args.max_instr_len
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
        traj_sent_sim=torch.matmul(torch.matmul(sentence_output,self.global_mat_weight),traj_output.t()) # [bs,bs]
        
        # video-word score
        video_word_sim=torch.sum(torch.matmul(word_features, traj_output.t()) \
            * torch.matmul(torch.softmax(torch.matmul(word_features, traj_output.t())/1e-2, dim=1).permute(0,2,1), self.word_logit_weight).permute(0,2,1), dim=1) # [bs,bs]
        
        # sentence-frame score
        sentence_frame_sim=torch.sum(torch.matmul(sentence_output, frame_features.permute(0, 2, 1))\
            * torch.matmul(torch.softmax(torch.matmul(sentence_output, frame_features.permute(0, 2, 1))/1e-2, dim=-1), self.frame_logit_weight), dim=-1).t() # [bs,bs]

        # frame-word score
        frame_word_sim=self._attention_over_fine_grained_sim_matrix(word_features,frame_features) # [bs,bs]

        # compute loss
        sim_matrix=(traj_sent_sim+video_word_sim+sentence_frame_sim+frame_word_sim)/4
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
        patch_word_sim=self._attention_over_fine_grained_sim_matrix(word_features,patch_features) # [bs,bs]

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
    