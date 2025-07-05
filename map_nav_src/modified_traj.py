import os
import json
import time
import numpy as np
from collections import defaultdict
import jsonlines
import torch
from tensorboardX import SummaryWriter
import sys

from models.vlnbert_init import get_tokenizer

from r2r.agent_cmt import Seq2SeqCMTAgent

from r2r.agent_r2rback import Seq2SeqBackAgent
from r2r.data_utils import ImageFeaturesDB, construct_instrs
from r2r.env import R2RBatch, R2RBackBatch
from cvdn.env import NDHNavBatch

def build_dataset_duet(args, rank=0):
    tok = get_tokenizer(args)

    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)

    dataset_class = R2RBatch
    
    val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, ['test'], tokenizer=tok, max_instr_len=args.max_instr_len
        )
    test_env = dataset_class(
            feat_db, val_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name='test'
        )
    return test_env

def construct_instrs_cvdn(anno_dir, dataset, splits, tokenizer=None, max_instr_len=512):
    data = []
    for split in splits:
        new_data = json.load(open(os.path.join(anno_dir, '%s_enc.json'%split)))
        for item in new_data:
            if 'landmarks_enc' in item.keys():
                item['instr_encoding'] = (item['instr_encoding'] + [100] + item['landmarks_enc'][0])[:max_instr_len] # include multi grained instruction
                del item['landmarks_enc']
            else:
                item['instr_encoding'] = item['instr_encoding'][-max_instr_len:]
            data.append(item)
    return data

def build_dataset_cvdn(args, rank=0):
    tok = get_tokenizer(args)

    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)

    dataset_class = NDHNavBatch


    val_instr_data = construct_instrs_cvdn(
        args.anno_dir, args.dataset, ['test'], tokenizer=tok, max_instr_len=args.max_instr_len
    )
    test_env = dataset_class(
        feat_db, val_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name='test',
    )
    return test_env

def modified_cvdn():
    from map_nav_src.cvdn.parser1 import parse_args
    args = parse_args()
    test_env = build_dataset_cvdn(args, rank=0)

    with open("/remote-home/mfdu/VLN/Mycode/TD-STP/datasets/CVDN/exprs_map/finetune/dagger-clip-16-seed.0-init.aug.45k/preds/submit_test.json",'r') as f:
        preds = json.load(f)

    new_preds = []
    type = 'repeat'
    for i in range(len(preds)):
        tmp = {}
        scan = test_env.data[i]['scan']
        instr_id = preds[i]['instr_id']
        traj = [x[0] for x in preds[i]['trajectory']]
        
        tmp['inst_idx'] = instr_id
        if type == 'shortest':
            new_path = test_env.shortest_paths[scan][traj[0]][traj[-1]]
            tmp['trajectory'] = [[_] for _ in new_path]
        elif type == 'repeat':
            new_path = [traj[0]]
            for vp_id in range(len(traj)-1):
                if traj[vp_id+1] in test_env.graphs[scan].neighbors(traj[vp_id]) or traj[vp_id] == traj[vp_id+1]:
                    new_path.append(traj[vp_id+1])
                    continue
                else:
                    extra_path = test_env.shortest_paths[scan][traj[vp_id]][traj[vp_id+1]]
                    new_path += extra_path[1:]
            tmp['trajectory'] = [[_] for _ in new_path]
        new_preds.append(tmp)
    with open("/remote-home/mfdu/VLN/Mycode/TD-STP/datasets/CVDN/exprs_map/finetune/dagger-clip-16-seed.0-init.aug.45k/preds/submit_test_modified.json",'w') as f:
        json.dump(new_preds,f)
    
                    
def modified_duet():
    from r2r.parser import parse_args
    args = parse_args()
    test_env = build_dataset_duet(args, rank=0)

    with open("/remote-home/mfdu/VLN/Mycode/TD-STP/datasets/R2R/exprs_map/finetune/scale_vln_w25/preds/submit_test.json",'r') as f:
        preds = json.load(f)

    new_preds = []
    type = 'repeat'
    for i in range(len(preds)):
        tmp = {}
        scan = test_env.data[i]['scan']
        instr_id = preds[i]['instr_id']
        traj = [x[0] for x in preds[i]['trajectory']]
        
        tmp['instr_id'] = instr_id
        if type == 'shortest':
            new_path = test_env.shortest_paths[scan][traj[0]][traj[-1]]
            tmp['trajectory'] = [[_] for _ in new_path]
        elif type == 'repeat':
            new_path = [traj[0]]
            for vp_id in range(len(traj)-1):
                if traj[vp_id+1] in test_env.graphs[scan].neighbors(traj[vp_id]) or traj[vp_id] == traj[vp_id+1]:
                    new_path.append(traj[vp_id+1])
                    continue
                else:
                    extra_path = test_env.shortest_paths[scan][traj[vp_id]][traj[vp_id+1]]
                    new_path += extra_path[1:]
            tmp['trajectory'] = [[_] for _ in new_path]
        new_preds.append(tmp)
    with open("/remote-home/mfdu/VLN/Mycode/TD-STP/datasets/R2R/exprs_map/finetune/scale_vln_w25/preds/submit_test_modified.json",'w') as f:
        json.dump(new_preds,f)
            

def modified_rxr():
    datasets = ['submit_test_challenge_public.json', 'submit_test_standard_public.json']
    for dataset in  datasets:
        with open("/remote-home/mfdu/VLN/Mycode/TD-STP/datasets/RxR/trained_models/rxr_tcl_w1/preds/" + dataset ,'r') as f:
            data = json.load(f)

        with jsonlines.open("/remote-home/mfdu/VLN/Mycode/TD-STP/datasets/RxR/trained_models/rxr_tcl_w1/preds/"+ dataset.split('.')[0] + '.jsonl','w') as w:
            for item in data:
                new_item = {}
                new_item["instruction_id"] = int(item['instr_id'])
                new_item["path"] = [_[0] for _ in item['trajectory']]
                w.write(new_item)

def modified_rxr_new(input_submit_file, out_put_submit_file):
    with open(input_submit_file,'r') as f:
        data = json.load(f)

    with jsonlines.open(output_submit_file,'w') as w:
        for item in data:
            new_item = {}
            new_item["instruction_id"] = int(item['instr_id'])
            new_item["path"] = [_[0] for _ in item['trajectory']]
            w.write(new_item)

# modified_duet()
# modified_rxr_new()   
