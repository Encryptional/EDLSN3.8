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

# def cal_DSSP(ligand,seq_list,dssp_dir,feature_dir):
#
#     maxASA = {'G':188,'A':198,'V':220,'I':233,'L':304,'F':272,'P':203,'M':262,'W':317,'C':201,
#               'S':234,'T':215,'N':254,'Q':259,'Y':304,'H':258,'D':236,'E':262,'K':317,'R':319}
#     map_ss_8 = {' ':[1,0,0,0,0,0,0,0],'S':[0,1,0,0,0,0,0,0],'T':[0,0,1,0,0,0,0,0],'H':[0,0,0,1,0,0,0,0],
#                 'G':[0,0,0,0,1,0,0,0],'I':[0,0,0,0,0,1,0,0],'E':[0,0,0,0,0,0,1,0],'B':[0,0,0,0,0,0,0,1]}
#     dssp_dict = {}
#     for seqid in seq_list:
#         file = seqid+'.dssp'
#         with open(dssp_dir + '/' + file, 'r') as fin:
#             fin_data = fin.readlines()
#         seq_feature = {}
#         for i in range(25, len(fin_data)):
#             line = fin_data[i]
#             if line[13] not in maxASA.keys() or line[9]==' ':
#                 continue
#             res_id = float(line[5:10])
#             feature = np.zeros([14])
#             feature[:8] = map_ss_8[line[16]]
#             feature[8] = min(float(line[35:38]) / maxASA[line[13]], 1)
#             feature[9] = (float(line[85:91]) + 1) / 2
#             feature[10] = min(1, float(line[91:97]) / 180)
#             feature[11] = min(1, (float(line[97:103]) + 180) / 360)
#             feature[12] = min(1, (float(line[103:109]) + 180) / 360)
#             feature[13] = min(1, (float(line[109:115]) + 180) / 360)
#             seq_feature[res_id] = feature.reshape((1, -1))
#         dssp_dict[file.split('.')[0]] = seq_feature
#     with open(feature_dir + '/{}_SS.pkl'.format(ligand), 'wb') as f:
#         pickle.dump(dssp_dict, f)
#     return

##T5嵌入
def T5embedding(seq_list):
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
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
    # with open('/home/zhangbin/DNA/Datasets/compare/read_seq_name.txt', 'w') as f:
        # for i in range(len(train_list)):
            # f.write(train_list[i])
            # f.write('\r\n')
else:
    for i in range(0 ,len(train_text) ,4):
        pro_id = train_text[i].strip()[1:]
        if pro_id[-1].islower():
            pro_id += pro_id[-1]
        pro_seq = train_text[i + 1].strip()
        pro_anno = train_text[i + 3].strip()
        train_list.append(pro_id)
        seqanno[pro_id] = {'seq' :pro_seq ,'anno' :pro_anno}
        # with open('/home/zhangbin/DNA/Datasets/compare/read_seq_name_non.txt', 'w') as f:
            # for i in range(len(train_list)):
                # f.write(train_list[i])
                # f.write('\r\n')

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
    # with open('/home/zhangbin/DNA/Datasets/compare/test_name.txt', 'w') as f:
        # for l in range(len(test_list)):
            # f.write(test_list[l])
            # f.write('\r\n')

# train_list, valid_list = tv_split(train_list, 2022)

##划分
train_seq = []
train_label = []
# valid_seq = []
# valid_label = []
test_seq = []
test_label = []
for name in train_list:
    train_seq.append(seqanno[name]['seq'])
    train_label.append(seqanno[name]['anno'])
# for name in valid_list:
    # valid_seq.append(seqanno[name]['seq'])
    # valid_label.append(seqanno[name]['anno'])
for name in test_list:
    test_seq.append(seqanno[name]['seq'])
    test_label.append(seqanno[name]['anno'])

##T5生成向量
# train_vec = T5embedding(train_seq)
# test_vec = T5embedding(test_seq)
# np.save('/home/zhangbin/DNA/data_vec/train_vec', train_vec)
# np.save('/home/zhangbin/DNA/data_vec/test_vec', test_vec) ##data_label

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
np.save('/home/zhangbin/DNA/data_label/true-train_label', label_train)
np.save('/home/zhangbin/DNA/data_label/true-test_label', label_test)

##train and test
StatisticsSampleNum(train_list,test_list,seqanno)

# ##PSSM|HMM|SS
# concat = False
# seqlist = train_list + test_list
# ligand = 'PDNA'
# PSSM_dir = '/home/zhangbin/DNA/Datasets/PDNA/feature/PSSM'
# HMM_dir = '/home/zhangbin/DNA/Datasets/PDNA/feature/HMM'
# # SS_dir = '/home/zhangbin/DNA/Datasets/PDNA/feature/SS'
# Dataset_dir = '/home/zhangbin/DNA/Datasets/PDNA'
# cal_PSSM(ligand, seqlist, PSSM_dir , Dataset_dir)
# cal_HMM(ligand, seqlist, HMM_dir , Dataset_dir)
# # cal_DSSP(ligand, seqlist, SS_dir , Dataset_dir)
#
# ##存储bio_feature至local（）
# residue_feature_list = ['PSSM', 'HMM']
# for fea in residue_feature_list:
#     with open('/home/zhangbin/DNA/Datasets/PDNA/PDNA' + '_{}.pkl'.format(fea), 'rb') as f:
#         locals()['residue_fea_dict_' + fea] = pickle.load(f)
# ##train
# # with open('/home/zhangbin/DNA/Datasets/compare/bio_seq_train.txt',"w") as p:
# #     for seq_id in train_list:
# #         p.write(seq_id)
# #         p.write('\r\n')
#
# if concat:
#     bio_feature = []
#     for seq_id in train_list:
#         pss = locals()['residue_fea_dict_'+ residue_feature_list[0]][seq_id]
#         hmm = locals()['residue_fea_dict_'+ residue_feature_list[1]][seq_id]
#         # sss = locals()['residue_fea_dict_SS'][seq_id]
#         con = np.concatenate((pss, hmm), axis=1)
#         bio_feature.append(con)
#     t_bio = []
#     for i in range(len(bio_feature)):
#         for j in range(bio_feature[i].shape[0]):
#             t_bio.append(bio_feature[i][j][:])
#     train_bio = np.array(t_bio)
#     print('训练集bio维度',train_bio.shape)
#
#     ##test
#     # with open('/home/zhangbin/DNA/Datasets/compare/bio_seq_text.txt',"w") as p:
#     #     for seq_id in test_list:
#     #         p.write(seq_id)
#     #         p.write('\r\n')
#     bio_features = []
#     for seq_id in test_list:
#         pss_test = locals()['residue_fea_dict_'+ residue_feature_list[0]][seq_id]
#         hmm_test = locals()['residue_fea_dict_'+ residue_feature_list[1]][seq_id]
#         # sss = locals()['residue_fea_dict_SS'][seq_id]
#         con_test = np.concatenate((pss_test, hmm_test), axis=1)
#         bio_features.append(con_test)
#     te_bio = []
#     for i in range(len(bio_features)):
#         for j in range(bio_features[i].shape[0]):
#             te_bio.append(bio_features[i][j][:])
#     test_bio = np.array(te_bio)
#     print('测试集bio维度',test_bio.shape)
#     ##save
#     # np.save('/home/zhangbin/DNA/data_vec/train_bio_vec', train_bio)
#     # np.save('/home/zhangbin/DNA/data_vec/test_bio_vec', test_bio)
# else:
#     bio_PSSM = []
#     bio_HMM = []
#     for seq_id in train_list:
#         pss = locals()['residue_fea_dict_' + residue_feature_list[0]][seq_id]
#         hmm = locals()['residue_fea_dict_' + residue_feature_list[1]][seq_id]
#         # sss = locals()['residue_fea_dict_SS'][seq_id]
#         bio_PSSM.append(pss)
#         bio_HMM.append(hmm)
#     t_bio = []
#     for i in range(len(bio_PSSM)):
#         for j in range(bio_PSSM[i].shape[0]):
#             t_bio.append(bio_PSSM[i][j][:])
#     train_pss = np.array(t_bio)
#     print('训练集bio_PSSM维度', train_pss.shape)
#     t_hmm = []
#     for i in range(len(bio_HMM)):
#         for j in range(bio_HMM[i].shape[0]):
#             t_hmm.append(bio_HMM[i][j][:])
#     train_hmm = np.array(t_hmm)
#     print('训练集bio_HMM维度', train_hmm.shape)
#     ##test
#     bio_fea = []
#     bio_feast = []
#     for seq_id in test_list:
#         pss_test = locals()['residue_fea_dict_' + residue_feature_list[0]][seq_id]
#         hmm_test = locals()['residue_fea_dict_' + residue_feature_list[1]][seq_id]
#         # sss = locals()['residue_fea_dict_SS'][seq_id]
#         bio_fea.append(pss_test)
#         bio_feast.append(hmm_test)
#     te_bio = []
#     for i in range(len(bio_fea)):
#         for j in range(bio_fea[i].shape[0]):
#             te_bio.append(bio_fea[i][j][:])
#     test_bio_pssm = np.array(te_bio)
#     print('测试集bio_PSSM维度', test_bio_pssm.shape)
#     te_bios = []
#     for i in range(len(bio_feast)):
#         for j in range(bio_feast[i].shape[0]):
#             te_bios.append(bio_feast[i][j][:])
#     test_bio_hmm = np.array(te_bios)
#     print('测试集bio_HMM维度', test_bio_hmm.shape)
#     ##save
#     # np.save('/home/zhangbin/DNA/data_vec/train_bio_PSSM_vec', train_pss)
#     # np.save('/home/zhangbin/DNA/data_vec/train_bio__HMM_vec', train_hmm)
#     # np.save('/home/zhangbin/DNA/data_vec/test_bio_PSSM_vec', test_bio_pssm)
#     # np.save('/home/zhangbin/DNA/data_vec/test_bio_HMM_vec', test_bio_hmm)