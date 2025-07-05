DATA_ROOT=../datasets

train_alg=dagger

features=clip-16
ft_dim=768

obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

outdir=${DATA_ROOT}/CVDN/trained_models/delan_duet_cvdn

flag="--root_dir ${DATA_ROOT}

      --dataset landcvdn
      --tcl_hist_weight 0.01
      --tcl_obs_weight 0.1
      --use_aosm_obs_cl
      --norm_cl_weight
      --use_mem_bank
      --capacity 128
      --cl_duet
      --max_graph_len 80

      --traj_sent
      --video_word
      --sentence_frame
      --frame_word
      
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert
      --use_player_path

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}

      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2

      --max_action_len 15
      --max_instr_len 200

      --batch_size 4
      --lr 1e-5
      --iters 200000
      --log_every 1000
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5

      --gamma 0."

# train
# CUDA_VISIBLE_DEVICES='0' python cvdn/main.py $flag  \
#       --tokenizer bert \
#       --bert_ckpt_file '../datasets/R2R/trained_models/clip-pretrain-duet/model_step_80000.pt' \


#test
CUDA_VISIBLE_DEVICES='0' python cvdn/main.py $flag  \
      --tokenizer bert \
      --resume_file ../datasets/CVDN/trained_models/delan_duet_cvdn/best_val_unseen\
      --submit \
      --test
