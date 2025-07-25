DATA_ROOT=../datasets
ob_type=pano
feedback=sample

features=vitbase_r2rfte2e  # or vitbase
ft_dim=768

ngpus=1
seed=0

outdir=../datasets/R2R/trained_models/delan_hamt_r2r_0710

flag="
      --fast_mode
      --root_dir ${DATA_ROOT}
      --output_dir ${outdir}

      --dataset landr2r
      --tcl_hist_weight 0.01
      --tcl_obs_weight 0.1
      --use_aosm_obs_cl
      --norm_cl_weight
      --use_mem_bank
      --capacity 480
      
      --vlnbert ${vlnbert}
      --ob_type ${ob_type}

      --world_size ${ngpus}
      --seed ${seed}

      --num_l_layers 9
      --num_x_layers 4

      --hist_enc_pano
      --hist_pano_num_layers 2

      --features ${features}
      --feedback ${feedback}

      --max_action_len 15
      --max_instr_len 100

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 5e-6
      --iters 600000
      --log_every 2000
      --batch_size 4
      --optim adamW

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5"

# train
# -m torch.distributed.launch --nproc_per_node=${ngpus}
CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag \
       --aug ../datasets/R2R/annotations/dual_level/LR2R_aug_train_enc.json \
       --bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt\
       --fast_mode



# inference
#CUDA_VISIBLE_DEVICES='0' python3 r2r/main.py $flag \
#      --resume_file /remote-home/share/mfdu_share/bs8_tcl_1/ckpts/best_val_unseen \
#      --test --submit
