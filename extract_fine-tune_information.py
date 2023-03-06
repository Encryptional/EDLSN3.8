import numpy as np
import torch
from transformers import BertModel, BertTokenizer
import re
import os
import requests
import pickle
from tqdm.auto import tqdm
import tensorflow as tf
import gc


def read_fasta():
    seqanno = {}
    train_list = []
    train_seq = []
    test_list = []
    test_seq = []
    with open('/home/zhangbin/RNA/Datasets/Datasets/RNA-495_Train.txt', 'r') as f:
        train_text = f.readlines()
    for i in range(0, len(train_text), 4):
        pro_id = train_text[i].strip()[1:]
        if pro_id[-1].islower():
            pro_id += pro_id[-1]
        pro_seq = train_text[i + 1].strip()
        pro_anno = train_text[i + 2].strip()
        train_list.append(pro_id)
        seqanno[pro_id] = {'seq': pro_seq, 'anno': pro_anno}
    for name in train_list:
        train_seq.append(seqanno[name]['seq'])

    with open('/home/zhangbin/RNA/Datasets/Datasets/RNA-117_Test.txt', 'r') as f:
        test_text = f.readlines()
    for i in range(0, len(test_text), 3):
        pro_id = test_text[i].strip()[1:]
        if pro_id[-1].islower():
            pro_id += pro_id[-1]
        pro_seq = test_text[i + 1].strip()
        pro_ann = test_text[i + 2].strip()
        test_list.append(pro_id)
        seqanno[pro_id] = {'seq': pro_seq, 'anno': pro_ann}
    for name in test_list:
        test_seq.append(seqanno[name]['seq'])
    return train_seq , test_seq

def GB_Bert(sequences):

    tokenizer = BertTokenizer.from_pretrained("/home/zhangbin/RNA/without_structure/prot_bfd_ft", do_lower_case=False)
    model = BertModel.from_pretrained("/home/zhangbin/RNA/without_structure/prot_bfd_ft")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    seq_split = []
    for i in sequences:
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
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len - 1]
            count += 1
            features.append(seq_emd)
    vec_train = []
    for i in range(len(features)):
        for j in range(features[i].shape[0]):
            vec_train.append(features[i][j][:])
    vec_train = np.array(vec_train)
    print('train嵌入结果', vec_train.shape)
    return vec_train


sequences1, sequences2 = read_fasta()
Features1 = GB_Bert(sequences1)
Features2 = GB_Bert(sequences2)
print(np.array(Features1).shape)
print(np.array(Features2).shape)
np.save('/home/zhangbin/RNA/data_vec/finetune/train_9266.npy',np.array(Features1))
np.save('/home/zhangbin/RNA/data_vec/finetune/test_9266.npy',np.array(Features2))

