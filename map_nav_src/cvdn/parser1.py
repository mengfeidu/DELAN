import argparse
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--root_dir', type=str, default='../datasets')
    parser.add_argument('--dataset', type=str, default='r2r', choices=['r2r', 'r4r', 'cvdn','landcvdn'])
    parser.add_argument('--output_dir', type=str, default='default', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--tokenizer', choices=['bert', 'xlm'], default='bert')

    parser.add_argument('--act_visited_nodes', action='store_true', default=False)
    parser.add_argument('--fusion', choices=['global', 'local', 'avg', 'dynamic'])
    parser.add_argument('--expl_sample', action='store_true', default=False)
    parser.add_argument('--expl_max_ratio', type=float, default=0.6)
    parser.add_argument('--expert_policy', default='spl', choices=['spl', 'ndtw'])

    # distributional training (single-node, multiple-gpus)
    parser.add_argument('--world_size', type=int, default=1, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--node_rank", type=int, default=0, help="Id of the node")

    # General
    parser.add_argument('--iters', type=int, default=100000, help='training iterations')
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--eval_first', action='store_true', default=False)

    # Data preparation
    parser.add_argument('--max_instr_len', type=int, default=80)
    parser.add_argument('--max_action_len', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ignoreid', type=int, default=-100, help='ignoreid for action')

    # Load the model from
    parser.add_argument("--resume_file", default=None, help='path of the trained model')
    parser.add_argument("--resume_optimizer", action="store_true", default=False)

    # Augmented Paths from
    parser.add_argument("--aug", default=None)
    parser.add_argument('--bert_ckpt_file', default=None, help='init vlnbert')

    # Listener Model Config
    parser.add_argument("--ml_weight", type=float, default=0.20)
    parser.add_argument('--entropy_loss_weight', type=float, default=0.01)

    parser.add_argument("--features", type=str, default='vitbase')

    parser.add_argument('--fix_lang_embedding', action='store_true', default=False)
    parser.add_argument('--fix_pano_embedding', action='store_true', default=False)
    parser.add_argument('--fix_local_branch', action='store_true', default=False)

    parser.add_argument('--num_l_layers', type=int, default=9)
    parser.add_argument('--num_pano_layers', type=int, default=2)
    parser.add_argument('--num_x_layers', type=int, default=4)

    parser.add_argument('--enc_full_graph', default=False, action='store_true')
    parser.add_argument('--graph_sprels', action='store_true', default=False)

    # Dropout Param
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--feat_dropout', type=float, default=0.3)

    # Submision configuration
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument("--submit", action='store_true', default=False)
    parser.add_argument('--no_backtrack', action='store_true', default=False)
    parser.add_argument('--detailed_output', action='store_true', default=False)

    # Training Configurations
    parser.add_argument(
        '--optim', type=str, default='rms',
        choices=['rms', 'adam', 'adamW', 'sgd']
    )  # rms, adam
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
    parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
    parser.add_argument(
        '--feedback', type=str, default='sample',
        help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``'
    )
    parser.add_argument('--epsilon', type=float, default=0.1, help='')

    # Model hyper params:
    parser.add_argument("--angle_feat_size", type=int, default=4)
    parser.add_argument('--image_feat_size', type=int, default=2048)
    parser.add_argument('--obj_feat_size', type=int, default=0)
    parser.add_argument('--views', type=int, default=36)
    parser.add_argument('--use_player_path', action='store_true', default=False)

    # # A2C
    parser.add_argument("--gamma", default=0.9, type=float, help='reward discount factor')
    parser.add_argument(
        "--normalize", dest="normalize_loss", default="total",
        type=str, help='batch or total'
    )
    parser.add_argument('--train_alg',
                        choices=['imitation', 'dagger'],
                        default='imitation'
                        )

    parser.add_argument("--augwsd", action='store_true', default=False)
    parser.add_argument("--sd_env", default="clip-vitb16-sd.tsv", nargs='*')
    parser.add_argument("--aug_prob", default=0.5, type=float)

    parser.add_argument("--future", action='store_true', default=False)
    parser.add_argument('--multi', action='store_true', default=False)
    parser.add_argument('--partial_aug', default=-1, type=int)


    # CL
    # fast mode
    parser.add_argument('--fast_mode', action='store_true', default=True)
    parser.add_argument("--tcl_hist_weight", type=float, default=0.0)
    parser.add_argument("--tcl_obs_weight", type=float, default=0.0)
    parser.add_argument('--norm_cl_weight', action='store_true', default=False)
    parser.add_argument("--use_aosm_obs_cl", action='store_true', default=False, help='use last layer embeddings to compute cl loss')
    parser.add_argument("--use_mem_bank", action='store_true', default=False)
    parser.add_argument("--capacity", type=int, default=192)
    parser.add_argument('--max_obs_num', type=int, default=51)
    parser.add_argument('--cl_duet', action='store_true', default=False)
    parser.add_argument('--max_graph_len', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=768)

    parser.add_argument('--traj_sent', action='store_true', default=False)
    parser.add_argument('--video_word', action='store_true', default=False)
    parser.add_argument('--sentence_frame', action='store_true', default=False)
    parser.add_argument('--frame_word', action='store_true', default=False)
    parser.add_argument('--cl_lang_layer', type=int, default=-1)

    parser.add_argument('--single_level', action='store_true', default=False)
    
    parser.add_argument('--aug_times', type=int, default=9)
    parser.add_argument("--env_aug", action='store_true', default=False)
    args, _ = parser.parse_known_args()

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir

    # Setup input paths
    ft_file_map = {
        'vitbase': 'pth_vit_base_patch16_224_imagenet.hdf5',
        "clip-16": 'CLIP-ViT-B-16-views.hdf5',
        'vitbase_clip':'pth_vit_base_patch32_224_clip.hdf5',
        'clip.h14': 'clip_vit-h14_mp3d_hm3d_gibson.hdf5',
        'clip.b16': 'clip_vit-b16_mp3d_hm3d_gibson.hdf5'
    }
    if args.features == 'clip-16':
        args.img_ft_file = ['/remote-home/mfdu/VLN/Mycode/datasets/R2R/features/hamt_features/CLIP-ViT-B-16-views.hdf5']
        args.mp3d_ft_files = args.val_ft_file = args.aug_ft_file = args.img_ft_file
    elif args.features == 'clip.h14':
        args.mp3d_ft_files = os.path.join('/remote-home/mfdu/VLN/Mycode/ScaleVLN', 'features', 'clip_vit-h14_mp3d_original.hdf5')
        args.img_ft_file = args.mp3d_ft_files
        args.val_ft_file = os.path.join('/remote-home/mfdu/VLN/Mycode/ScaleVLN', 'features', 'clip_vit-h14_mp3d_original.hdf5')
    elif args.features == 'clip.b16':
        args.mp3d_ft_files = [os.path.join('/remote-home/mfdu/VLN/Mycode/ScaleVLN', 'features', 'clip_vit-b16_mp3d_original.hdf5')]
        args.val_ft_file = os.path.join('/remote-home/mfdu/VLN/Mycode/ScaleVLN', 'features', 'clip_vit-b16_mp3d_original.hdf5')

    if args.env_aug: # only h14
        args.mp3d_ft_files = [
            os.path.join('/remote-home/mfdu/VLN/Mycode/ScaleVLN', 'features', 'clip_vit-h14_mp3d_img_image_synthesis.hdf5'), 
            os.path.join('/remote-home/mfdu/VLN/Mycode/ScaleVLN', 'features', 'clip_vit-h14_mp3d_img_mask_image_synthesis.hdf5'),
            os.path.join('/remote-home/mfdu/VLN/Mycode/ScaleVLN', 'features', 'clip_vit-h14_mp3d_img_style_transfer.hdf5'),
            os.path.join('/remote-home/mfdu/VLN/Mycode/ScaleVLN', 'features', 'clip_vit-h14_mp3d_original.hdf5'),
            ]
        args.aug_ft_file = os.path.join('/remote-home/mfdu/VLN/Mycode/ScaleVLN', 'features', ft_file_map[args.features])
    if  args.features == 'clip.b16':  
        args.aug_ft_file = os.path.join('/remote-home/mfdu/VLN/Mycode/ScaleVLN', 'features', ft_file_map[args.features])
    
    if args.aug:
        args.connectivity_dir = os.path.join('/remote-home/mfdu/VLN/Mycode/ScaleVLN/r2r_preprocess_data', 'connectivity')
    else:
        args.connectivity_dir = os.path.join(ROOTDIR, 'R2R', 'connectivity')
        
    args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')

    if args.dataset == 'cvdn':
        args.anno_dir = os.path.join(ROOTDIR, 'CVDN', 'annotations', 'ndh')
    else:
        args.anno_dir = os.path.join(ROOTDIR, 'CVDN', 'annotations', 'ndh', 'noun')

    if args.augwsd:
        args.img_ft_file_sd = []
        for file in args.sd_env:
            # print(type(args.sd_env))
            # print(file)
            args.img_ft_file_sd.append(os.path.join(ROOTDIR, 'R2R', 'features', file))
    else:
        args.img_ft_file_sd = None

    # Build paths
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    return args

