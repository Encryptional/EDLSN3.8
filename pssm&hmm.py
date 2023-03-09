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

def cal_HMM(ligand,seq_list,hmm_dir,feature_dir):
    hmm_dict = {}
    for seqid in seq_list:
        file = seqid+'.hhm'
        with open(hmm_dir+'/'+file,'r') as fin:
            fin_data = fin.readlines()
            hhm_begin_line = 0
            hhm_end_line = 0
            for i in range(len(fin_data)):
                if '#' in fin_data[i]:
                    hhm_begin_line = i+5
                elif '//' in fin_data[i]:
                    hhm_end_line = i
            feature = np.zeros([int((hhm_end_line-hhm_begin_line)/3),30])
            axis_x = 0
            for i in range(hhm_begin_line,hhm_end_line,3):
                line1 = fin_data[i].split()[2:-1]
                line2 = fin_data[i+1].split()
                axis_y = 0
                for j in line1:
                    if j == '*':
                        feature[axis_x][axis_y]=9999/10000.0
                    else:
                        feature[axis_x][axis_y]=float(j)/10000.0
                    axis_y+=1
                for j in line2:
                    if j == '*':
                        feature[axis_x][axis_y]=9999/10000.0
                    else:
                        feature[axis_x][axis_y]=float(j)/10000.0
                    axis_y+=1
                axis_x+=1
            feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
            hmm_dict[file.split('.')[0]] = feature
    with open(feature_dir + '/{}_HMM.pkl'.format(ligand), 'wb') as f:
        pickle.dump(hmm_dict, f)
    return

def cal_PSSM(ligand,seq_list,pssm_dir,feature_dir):
    nor_pssm_dict = {}
    for seqid in seq_list:
        file = seqid+'.pssm'
        with open(pssm_dir+'/'+file,'r') as fin:
            fin_data = fin.readlines()
            pssm_begin_line = 3
            pssm_end_line = 0
            for i in range(1,len(fin_data)):
                if fin_data[i] == '\n':
                    pssm_end_line = i
                    break
            feature = np.zeros([(pssm_end_line-pssm_begin_line),20])
            axis_x = 0
            for i in range(pssm_begin_line,pssm_end_line):
                raw_pssm = fin_data[i].split()[2:22]
                axis_y = 0
                for j in raw_pssm:
                    feature[axis_x][axis_y]= (1 / (1 + math.exp(-float(j))))
                    axis_y+=1
                axis_x+=1
            nor_pssm_dict[file.split('.')[0]] = feature
    with open(feature_dir+'/{}_PSSM.pkl'.format(ligand),'wb') as f:
        pickle.dump(nor_pssm_dict,f)
    return

##读取训练集
seqdict = {}
train_list = []
test_list = []
with open('./Datasets/RNA-495_Train.txt', 'r') as f:
    train_text = f.readlines()
for i in range(0 ,len(train_text) , 4):
    id = train_text[i].strip()[1:]
    if id[-1].islower():
        id += id[-1]
    pro_seq = train_text[i +1].strip()
    pro_anno = train_text[i +2].strip()
    train_list.append(id)
    seqdict[id] = {'seq' :pro_seq ,'anno' :pro_anno}

##读取测试集
with open('./Datasets/RNA-117_Test.txt', 'r') as f:
    test_text = f.readlines()
for i in range(0, len(test_text), 3):
    id = test_text[i].strip()[1:]
    if id[-1].islower():
        id += id[-1]
    pro_seq = test_text[i + 1].strip()
    pro_ann = test_text[i + 2].strip()
    test_list.append(id)
    seqdict[id] = {'seq': pro_seq, 'anno': pro_ann}

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


##标签生成
label_train = []
label_test = []
for i in range(len(train_label)):
    for j in range(len(train_label[i])):
        label_train.append(train_label[i][j])
for i in range(len(test_label)):
    for j in range(len(test_label[i])):
        label_test.append(test_label[i][j])
print('训练集总标签条数', len(train_label))
print('测试集总标签条数',len(test_label))
label_train = np.array(label_train)
label_test = np.array(label_test)
print('训练集的标签格式', label_train.shape)
print('测试集的标签格式', label_test.shape)
np.save('./multi-feature/DNA/R_train_label', label_train)
np.save('./multi-feature/RNA/R_test_label', label_test)


##PSSM|HMM
seqlist = train_list + test_list
ligand = 'RNA'
PSSM_dir = 'pssm_path'
HMM_dir = 'hmm_path'
Dataset_dir = './multi-feature/RNA'
cal_PSSM(ligand, seqlist, PSSM_dir , Dataset_dir)
cal_HMM(ligand, seqlist, HMM_dir , Dataset_dir)

##存储bio_feature至local（）
residue_feature_list = ['PSSM', 'HMM']
for fea in residue_feature_list:
    with open('./multi-feature/RNA/RNA' + '_{}.pkl'.format(fea), 'rb') as f:
        locals()['RNA_' + fea] = pickle.load(f)

bio_PSSM = []
bio_HMM = []
for seq_id in train_list:
    pss = locals()['RNA_' + residue_feature_list[0]][seq_id]
    hmm = locals()['RNA_' + residue_feature_list[1]][seq_id]
    bio_PSSM.append(pss)
    bio_HMM.append(hmm)
t_bio = []
for i in range(len(bio_PSSM)):
    for j in range(bio_PSSM[i].shape[0]):
        t_bio.append(bio_PSSM[i][j][:])
train_pss = np.array(t_bio)
print('训练集bio_PSSM维度', train_pss.shape)
t_hmm = []
for i in range(len(bio_HMM)):
    for j in range(bio_HMM[i].shape[0]):
        t_hmm.append(bio_HMM[i][j][:])
train_hmm = np.array(t_hmm)
print('训练集bio_HMM维度', train_hmm.shape)
##test
bio_fea = []
bio_feast = []
for seq_id in test_list:
    pss_test = locals()['RNA_' + residue_feature_list[0]][seq_id]
    hmm_test = locals()['RNA_' + residue_feature_list[1]][seq_id]
    bio_fea.append(pss_test)
    bio_feast.append(hmm_test)
te_bio = []
for i in range(len(bio_fea)):
    for j in range(bio_fea[i].shape[0]):
        te_bio.append(bio_fea[i][j][:])
test_bio_pssm = np.array(te_bio)
print('测试集bio_PSSM维度', test_bio_pssm.shape)
te_bios = []
for i in range(len(bio_feast)):
    for j in range(bio_feast[i].shape[0]):
        te_bios.append(bio_feast[i][j][:])
test_bio_hmm = np.array(te_bios)
print('测试集bio_HMM维度', test_bio_hmm.shape)
#save
np.save('./multi-feature/RNA/bio/R_train_bio_PSSM_vec', train_pss)
np.save('./multi-feature/RNA/bio/R_train_bio__HMM_vec', train_hmm)
np.save('./multi-feature/RNA/bio/R_test_bio_PSSM_vec', test_bio_pssm)
np.save('./multi-feature/RNA/bio/R_test_bio_HMM_vec', test_bio_hmm)