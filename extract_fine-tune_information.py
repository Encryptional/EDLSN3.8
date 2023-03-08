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
    seqdict = {}
    train_list = []
    train_seq = []
    test_list = []
    test_seq = []
    with open('./Datasets/RNA-495_Train.txt', 'r') as f:
        train_text = f.readlines()
    for i in range(0, len(train_text), 4):
        id = train_text[i].strip()[1:]
        if id[-1].islower():
            id += id[-1]
        pro_seq = train_text[i + 1].strip()
        pro_anno = train_text[i + 2].strip()
        train_list.append(id)
        seqdict[id] = {'seq': pro_seq, 'anno': pro_anno}
    for name in train_list:
        train_seq.append(seqdict[name]['seq'])

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
    for name in test_list:
        test_seq.append(seqdict[name]['seq'])
    return train_seq , test_seq

def get_dynamic_embdedding(sequences):

    tokenizer = BertTokenizer.from_pretrained("./fine-tuned_model", do_lower_case=False)
    model = BertModel.from_pretrained("./fine-tuned_model")
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
    return vec_train


sequences1, sequences2 = read_fasta()
Features1 = get_dynamic_embdedding(sequences1)
Features2 = get_dynamic_embdedding(sequences2)

np.save('./data_vec/RNA/RNA_dyna_train.npy',np.array(Features1))
np.save('./data_vec/RNA/RNA_dyna_test.npy',np.array(Features2))

