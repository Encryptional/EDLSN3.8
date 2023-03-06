import random
import pickle
import numpy as np
import os
import sys
import shutil
from tqdm import tqdm,trange
import torch
import prettytable as pt
import math
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc

##测试集验证集划分
def tv_split(train_list,seed):
    random.seed(seed)
    random.shuffle(train_list)
    valid_list = train_list[:int(len(train_list)*0.2)]
    train_list = train_list[int(len(train_list)*0.2):]
    return train_list,valid_list

##输出数据集表格
def StatisticsSampleNum(train_list,test_list,seqanno):
    def sub(seqlist,seqanno):
        pos_num_all = 0
        res_num_all = 0
        for seqid in seqlist:
            anno = list(map(int,list(seqanno[seqid]['anno'])))
            pos_num = sum(anno)
            res_num = len(anno)
            pos_num_all += pos_num
            res_num_all += res_num
        neg_num_all = res_num_all - pos_num_all
        pnratio = pos_num_all/float(neg_num_all)
        return len(seqlist), res_num_all, pos_num_all,neg_num_all,pnratio

    tb = pt.PrettyTable()
    tb.field_names = ['Dataset','NumSeq', 'NumRes', 'NumPos', 'NumNeg', 'PNratio']
    tb.float_format = '0.3'
    seq_num, res_num, pos_num, neg_num, pnratio = sub(train_list,seqanno)
    tb.add_row(['train',seq_num, res_num, pos_num, neg_num, pnratio])
    seq_num, res_num, pos_num, neg_num, pnratio = sub(test_list,seqanno)
    tb.add_row(['test',seq_num, res_num, pos_num, neg_num, pnratio])
    print(tb)
    return

##T5嵌入
def T5embedding(seq_list):
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_bfd", do_lower_case=False )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_bfd")
    gc.collect()
    #device = torch.device('cpu')
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    #添加空格分隔
    seq_split = []
    for i in seq_list:
        seq_split.append(str(' '.join([word for word in i])))
    sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in seq_split]
    features = []
    count = 0
    dataloader = torch.utils.data.DataLoader(sequences, batch_size=8, shuffle=False)
    for i in dataloader:
        ids = tokenizer.batch_encode_plus(i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            count+=1
            features.append(seq_emd)
    print('第一条蛋白质残基个数', len(seq_list[0]))
    print('序列条数',len(features))
    print('第一条蛋白质嵌入维度',features[0].shape)
    vec_train = []
    for i in range(len(features)):
        for j in range(features[i].shape[0]):
            vec_train.append(features[i][j][:])
    vec_train = np.array(vec_train)
    print('train嵌入结果',vec_train.shape)
    return vec_train

##读取训练集
trans_anno = False
seqanno = {}
train_list = []
test_list = []
with open('/home/zhangbin/DNA/Datasets/Datasets/DNA-573_Train.txt', 'r') as f:
    train_text = f.readlines()
if trans_anno:
    for i in range(0 ,len(train_text) , 4):
        pro_id = train_text[i].strip()[1:]
        if pro_id[-1].islower():
            pro_id += pro_id[-1]
        pro_seq = train_text[i +1].strip()
        pro_anno = train_text[i +2].strip()
        train_list.append(pro_id)
        seqanno[pro_id] = {'seq' :pro_seq ,'anno' :pro_anno}

else:
    for i in range(0 ,len(train_text) ,4):
        pro_id = train_text[i].strip()[1:]
        if pro_id[-1].islower():
            pro_id += pro_id[-1]
        pro_seq = train_text[i + 1].strip()
        pro_anno = train_text[i + 3].strip()
        train_list.append(pro_id)
        seqanno[pro_id] = {'seq' :pro_seq ,'anno' :pro_anno}


##读取测试集
with open('/home/zhangbin/DNA/Datasets/Datasets/DNA-129_Test.txt', 'r') as f:
    test_text = f.readlines()
for i in range(0, len(test_text), 3):
    pro_id = test_text[i].strip()[1:]
    if pro_id[-1].islower():
        pro_id += pro_id[-1]
    pro_seq = test_text[i + 1].strip()
    pro_ann = test_text[i + 2].strip()
    test_list.append(pro_id)
    seqanno[pro_id] = {'seq': pro_seq, 'anno': pro_ann}


##划分
train_seq = []
train_label = []

test_seq = []
test_label = []
for name in train_list:
    train_seq.append(seqanno[name]['seq'])
    train_label.append(seqanno[name]['anno'])

for name in test_list:
    test_seq.append(seqanno[name]['seq'])
    test_label.append(seqanno[name]['anno'])

#T5生成向量
train_vec = T5embedding(train_seq)
test_vec = T5embedding(test_seq)
np.save('/home/zhangbin/DNA/data_vec/t5-bfd-train_vec', train_vec)
np.save('/home/zhangbin/DNA/data_vec/t5-bfd-test_vec', test_vec) ##data_label

##train and test
StatisticsSampleNum(train_list,test_list,seqanno)

