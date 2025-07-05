DATA_ROOT=../datasets
features=vitbase_r2rfte2e
ft_dim=768

ngpus=1
seed=0

outdir=../datasets/CVDN/trained_models/delan_hamt_cvdn111

flag="--root_dir ${DATA_ROOT}
      --output_dir ${outdir}

      --fast_mode
      --dataset landcvdn
      --use_player_path

      --tcl_hist_weight 0.01
      --tcl_obs_weight 0.01
      --use_aosm_obs_cl
      --norm_cl_weight
      --use_mem_bank
      --capacity 128

      --ob_type pano
      
      --world_size ${ngpus}
      --seed ${seed}
      
      --num_l_layers 9
      --num_x_layers 4
      
      --hist_enc_pano
      --hist_pano_num_layers 2
      
      --no_lang_ca
      --cl_lang_layer -1

      --features ${features}
      --feedback sample

      --max_action_len 25
      --max_instr_len 150

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 200000
      --log_every 1000
      --batch_size 4
      --optim adamW

      --ml_weight 0.2      

      --feat_dropout 0.4
      --dropout 0.5"


# train
# CUDA_VISIBLE_DEVICES='0' python cvdn/main.py $flag \
#       --bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt \


# inference
CUDA_VISIBLE_DEVICES='0' python cvdn/main.py $flag \
      --resume_file ../datasets/CVDN/trained_models/delan_hamt_cvdn/best_val_unseen \
      --test --submit