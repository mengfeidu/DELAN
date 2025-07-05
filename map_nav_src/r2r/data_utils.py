import os
import json
import numpy as np
import jsonlines
def load_instr_datasets_duet(anno_dir, dataset, splits, tokenizer, is_test=True):
    data = []
    for split in splits:
        if "/" not in split:    # the official splits
            if tokenizer == 'bert':
                filepath = os.path.join(anno_dir, '%s_%s_enc.json' % (dataset.upper(), split))
            elif tokenizer == 'xlm':
                filepath = os.path.join(anno_dir, '%s_%s_enc_xlmr.json' % (dataset.upper(), split))
            else:
                raise NotImplementedError('unspported tokenizer %s' % tokenizer)

            with open(filepath) as f:
                new_data = json.load(f)

            if split == 'val_train_seen':
                new_data = new_data[:50]

            if not is_test:
                if dataset == 'r4r' and split == 'val_unseen':
                    ridxs = np.random.permutation(len(new_data))[:200]
                    new_data = [new_data[ridx] for ridx in ridxs]
        else:   # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)
        # Join
        data += new_data
    return data

def load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=True):
    data = []
    for split in splits:
        if "/" not in split:  # the official splits
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
                try:
                    new_data = []
                    with jsonlines.open(os.path.join(anno_dir, 'rxr_%s_guide_enc_xlmr.jsonl' % split)) as f:
                        for item in f:
                            new_data.append(item)
                except:
                    new_data = []
                    with jsonlines.open(os.path.join(anno_dir, 'rxr_%s_standard_public_guide_enc_xlmr.jsonl' % split)) as f:
                        for item in f:
                            new_data.append(item)
                # new_data = []
                # with jsonlines.open(os.path.join(anno_dir, 'rxr_%s_guide_enc_xlmr.jsonl' % split)) as f:
                #     for item in f:
                #         new_data.append(item)
            elif dataset == 'landrxr':
                try:
                    with open(os.path.join(anno_dir, 'dual_level', 'rxr_%s_guide_enc_xlmr.json'%split)) as f:
                        new_data = json.load(f)
                except:
                    with open(os.path.join(anno_dir, 'dual_level', 'rxr_%s_standard_public_guide_enc_xlmr.json'%split)) as f:
                        new_data = json.load(f)
            elif dataset == 'reverie':
                with open(os.path.join(anno_dir, 'REVERIE_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
        else:  # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)

        # Join
        data += new_data
    return data

def construct_instrs_duet(anno_dir, dataset, splits, tokenizer, max_instr_len=512, is_test=True):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=is_test)):
        # Split multiple instructions into separate entries
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
            new_item['instruction'] = instr
            new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
            del new_item['instructions']
            del new_item['instr_encodings']
            data.append(new_item)
    return data

def construct_instrs(anno_dir, dataset, splits, tokenizer=None, max_instr_len=512, is_test=True):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=is_test)):
        if dataset == 'rxr' or dataset == 'landrxr':
            # rxr annotations are already split
            new_item = dict(item)
            if 'path_id' in item:
                new_item['instr_id'] = '%d_%d' % (item['path_id'], item['instruction_id'])
            else:  # test
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
                if 'objId' in item:
                    new_item['instr_id'] = '%s_%s_%d' % (str(item['path_id']), str(item['objId']), j)
                else:
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instruction'] = instr
                
                if 'land_encodings' in new_item.keys():
                    new_item['instr_encoding'] = (item['instr_encodings'][j] + item['land_encodings'][j])[:max_instr_len] # include multi grained instruction
                    del new_item['land_encodings']
                else: 
                    # new_item['instr_encoding'] = (item['instr_encodings'][j] + [102])[:max_instr_len]
                    new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
                del new_item['instructions']
                del new_item['instr_encodings']

                # ''' BERT tokenizer '''
                # instr_tokens = ['[CLS]'] + tokenizer.tokenize(instr)[:max_instr_len-2] + ['[SEP]']
                # new_item['instr_encoding'] = tokenizer.convert_tokens_to_ids(instr_tokens)

                data.append(new_item)
    return data

def get_inst_land_part(args, obs):
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

def get_inst_land_part_cvdn(args, obs):
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