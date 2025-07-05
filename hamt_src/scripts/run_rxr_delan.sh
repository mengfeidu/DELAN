DATA_ROOT=../datasets
features=vitbase_clip
ft_dim=512

ngpus=1
seed=0

outdir=../datasets/RxR/trained_models/delan_hamt_rxr

flag="--root_dir ${DATA_ROOT}
      --output_dir ${outdir}
      
      --dataset landrxr

      --tcl_hist_weight 0.01
      --tcl_obs_weight 0.1
      --use_aosm_obs_cl
      --norm_cl_weight
      --use_mem_bank
      --capacity 100

      --ob_type pano
      --no_lang_ca
      --cl_lang_layer -1

      --world_size ${ngpus}
      --seed ${seed}
      
      --num_l_layers 9
      --num_x_layers 4
      --hist_enc_pano
      --hist_pano_num_layers 2

      --features ${features}
      --feedback sample

      --max_action_len 20
      --max_instr_len 200
      --batch_size 4

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 250000
      --log_every 2000
      --optim adamW

      --ml_weight 0.2
      --featdropout 0.4
      --dropout 0.5"


      

# train
# CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag  \
#       --bert_ckpt_file ../datasets/RxR/trained_models/vitbase.clip-pretrain/model_step_200000.pt \
#       --fast_mode

# inference 
CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag  \
      --resume_file ../datasets/RxR/trained_models/delan_hamt_rxr/best_val_unseen \
      --test --submit
