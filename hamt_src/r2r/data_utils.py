import os
import json
import jsonlines
import h5py
import networkx as nx
import math
import numpy as np

import torch
from r2r.parser import args

class ImageFeaturesDB(object):
    def __init__(self, img_ft_file, image_feat_size):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_file
        self._feature_store = {}

    def get_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                self._feature_store[key] = ft
        return ft


def load_instr_datasets(anno_dir, dataset, splits):
    data = []
    for split in splits:
        if "/" not in split:    # the official splits
            if dataset == 'r2r':
                with open(os.path.join(anno_dir, 'R2R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'landr2r': # noun
                with open(os.path.join(anno_dir, 'dual_level', 'LR2R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'r4r':
                with open(os.path.join(anno_dir, 'R4R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'landr4r':
                with open(os.path.join(anno_dir, 'dual_level', 'R4R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'rxr':
                new_data = []
                with jsonlines.open(os.path.join(anno_dir, 'rxr_%s_guide_enc_xlmr.jsonl'%split)) as f:
                    for item in f:
                        new_data.append(item)
            elif dataset == 'landrxr':
                with open(os.path.join(anno_dir, 'dual_level', 'rxr_%s_guide_enc_xlmr.json'%split)) as f:
                    new_data = json.load(f)
        else:   # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)

        # Join
        data += new_data
    return data

def construct_instrs(anno_dir, dataset, splits, tokenizer=None, max_instr_len=512):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits)):
        if dataset == 'rxr' or dataset == 'landrxr':
            # rxr annotations are already split
            new_item = dict(item)
            if 'path_id' in item:
                new_item['instr_id'] = '%d_%d'%(item['path_id'], item['instruction_id'])
            else: # test
                new_item['path_id'] = new_item['instr_id'] = str(item['instruction_id'])
            
            if 'landmarks_enc' in item.keys():
                new_item['instr_encoding'] = (new_item['instr_encoding'] + [100] +new_item['landmarks_enc'][0])[:max_instr_len] # include multi grained instruction
                del new_item['landmarks_enc']
            else:
                new_item['instr_encoding'] = new_item['instr_encoding'][-max_instr_len:]
            data.append(new_item)
        else:
            # Split multiple instructions into separate entries
            for j, instr in enumerate(item['instructions']):
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instruction'] = instr

                if 'land_encodings' in new_item.keys():
                    new_item['instr_encoding'] = (item['instr_encodings'][j] + item['land_encodings'][j])[:max_instr_len] # include multi grained instruction
                    del new_item['land_encodings']
                else:
                    new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
                    
                del new_item['instructions']
                del new_item['instr_encodings']

                # ''' BERT tokenizer '''
                # instr_tokens = ['[CLS]'] + tokenizer.tokenize(instr)[:max_instr_len-2] + ['[SEP]']
                # new_item['instr_encoding'] = tokenizer.convert_tokens_to_ids(instr_tokens)
                          
                data.append(new_item)
    return data


def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

 
def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading),math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)

def new_simulator(connectivity_dir, scan_data_dir=None):
    import MatterSim

    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.initialize()

    return sim

def get_point_angle_feature(sim, angle_feat_size, baseViewId=0, minus_elevation=False):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    if minus_elevation:
        base_elevation = (baseViewId // 12 - 1) * math.radians(30)
    else:
        base_elevation = 0
        
    for ix in range(36):
        if ix == 0:
            sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading
        elevation = state.elevation - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature

def get_all_point_angle_feature(sim, angle_feat_size, minus_elevation=False):
    return [get_point_angle_feature(
        sim, angle_feat_size, baseViewId, minus_elevation=minus_elevation
        ) for baseViewId in range(36)]

def length2mask(lens):
    """
    lens is [4,2,3] return 
    torch.tensor([[1.,1.,1.,1.],
                [1.,1.,0.,0.],
                [1.,1.,1.,0.]],device='cuda')
    """
    batch_size=len(lens)
    max_len=max(lens)
    mask=np.zeros([batch_size,max_len],dtype=np.float32)
    for i,len in enumerate(lens):
        mask[i,:len]=1.
    
    return torch.from_numpy(mask).cuda()


def get_inst_land_part(obs):
    # return positions of instruction and landmark part in complete instruction
    # example: [101,355,366,451,102,451,102] will return [1,2,3],[5]
    inst_part,land_part=[],[]
    sep_idx=102
    for ob in obs:
        complete_inst=ob['instr_encoding']
        if sep_idx in complete_inst:
            first_sep=complete_inst.index(sep_idx)
            inst=[i for i in range(1,first_sep)]
            if sep_idx in complete_inst[first_sep+1:]:
                second_sep=complete_inst.index(sep_idx,first_sep+1)
                assert second_sep==len(complete_inst)-1
                land=[i for i in range(first_sep+1,len(complete_inst)-1)]
            else:
                land=[i for i in range(first_sep+1,len(complete_inst))]
        else:
            inst=[i for i in range(1,len(complete_inst))]
            land=[]
        inst_part.append(inst)
        land_part.append(land)
    
    # return instruction mask and landmark mask according to inst part and land part
    # instruction mask : [bs,num_words], landmark mask : [bs,num_words]
    # example: [101,355,366,451,102,451,102] will return [0,1,1,1,0,0,0] and [0,0,0,0,0,1,0]
    bs=len(obs)
    num_words=args.max_instr_len
    # num_words=len(obs[0]['instr_encoding'])
    inst_mask=np.zeros([bs,num_words],dtype=np.float32)
    land_mask=np.zeros([bs,num_words],dtype=np.float32)
    assert len(inst_part)==len(land_part)
    for i,(inst,land) in enumerate(zip(inst_part,land_part)):
        inst_mask[i][inst]=1.
        land_mask[i][land]=1.
    return inst_part, land_part, inst_mask, land_mask

def get_inst_land_part_cvdn(obs):
    # return positions of instruction and landmark part in complete instruction
    # example: [101,355,366,451,102,451,102] will return [1,2,3],[5]
    # cvdn have serveral [SEP] token
    inst_part,land_part=[],[]
    sep_idx=100
    for ob in obs:
        complete_inst=ob['instr_encoding']

        if sep_idx in complete_inst:
            seg_point = complete_inst.index(sep_idx) # point
            inst=[i for i in range(1,seg_point)]
            land=[i for i in range(seg_point+1,len(complete_inst)-1)]
        else:
            inst=[i for i in range(1,len(complete_inst))]
            land=[]
        inst_part.append(inst)
        land_part.append(land)
    
    # return instruction mask and landmark mask according to inst part and land part
    # instruction mask : [bs,num_words], landmark mask : [bs,num_words]
    # example: [101,355,366,451,102,451,102] will return [0,1,1,1,0,0,0] and [0,0,0,0,0,1,0]
    bs=len(obs)
    num_words=args.max_instr_len
    # num_words=len(obs[0]['instr_encoding'])
    inst_mask=np.zeros([bs,num_words],dtype=np.float32)
    land_mask=np.zeros([bs,num_words],dtype=np.float32)
    assert len(inst_part)==len(land_part)
    for i,(inst,land) in enumerate(zip(inst_part,land_part)):
        inst_mask[i][inst]=1.
        land_mask[i][land]=1.
    return inst_part, land_part, inst_mask, land_mask


def gen_sim_gt(GT_insts,GT_lands,max_len_steps,max_num_words):
    """
    GT_insts is like [[[1,2,3],[0],[1,2],[1,2,3,4]],[[5,6],[1,2],[0],[0]]], (hist_len,batch_size)
    """
    bs=len(GT_insts[0])

    sim_inst_gt=np.zeros([bs,max_len_steps,max_num_words],dtype=np.float32)
    sim_land_gt=np.zeros([bs,max_len_steps,max_num_words],dtype=np.float32)
    for step,gt_inst in enumerate(GT_insts):
        if step==len(GT_insts)-1:
            break
        for batch,atten_inst in enumerate(gt_inst):
            if atten_inst!=[0]:
                sim_inst_gt[batch,step+1,atten_inst]=1.
    for step,gt_land in enumerate(GT_lands):
        if step==len(GT_lands)-1:
            break
        for batch,atten_land in enumerate(gt_land):
            if atten_land!=[0]:
                sim_land_gt[batch,step+1,atten_land]=1.
    return torch.from_numpy(sim_inst_gt).cuda(),torch.from_numpy(sim_land_gt).cuda()
  