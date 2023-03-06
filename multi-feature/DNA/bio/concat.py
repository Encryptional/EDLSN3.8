import numpy as np

train_raa = (np.load('/home/zhangbin/RNA/bio_vec/raa_train_RNA.npy').reshape(-1,1)).astype(np.float)
train_pca1 = (np.load('/home/zhangbin/RNA/bio_vec/pychar_train_RNA1.npy').reshape(-1,1)).astype(np.float)
train_pca2 = (np.load('/home/zhangbin/RNA/bio_vec/pychar_train_RNA2.npy').reshape(-1,1)).astype(np.float)
train_pca3 = (np.load('/home/zhangbin/RNA/bio_vec/pychar_train_RNA3.npy').reshape(-1,1)).astype(np.float)
train_pkx = (np.load('/home/zhangbin/RNA/bio_vec/pkx_train_RNA.npy').reshape(-1,1)).astype(np.float)
train_onehot = (np.load('/home/zhangbin/RNA/data_vec/train_one_hot.npy')).astype(np.float)
train_pssm = np.load('/home/zhangbin/RNA/data_vec/R_train_bio_PSSM_vec.npy')
train_hmm = np.load('/home/zhangbin/RNA/data_vec/R_train_bio__HMM_vec.npy')


test_raa = (np.load('/home/zhangbin/RNA/bio_vec/raa_test_RNA.npy').reshape(-1,1)).astype(np.float)
test_pca1 = (np.load('/home/zhangbin/RNA/bio_vec/pychar_test_RNA1.npy').reshape(-1,1)).astype(np.float)
test_pca2 = (np.load('/home/zhangbin/RNA/bio_vec/pychar_test_RNA2.npy').reshape(-1,1)).astype(np.float)
test_pca3 = (np.load('/home/zhangbin/RNA/bio_vec/pychar_test_RNA3.npy').reshape(-1,1)).astype(np.float)
test_pkx = (np.load('/home/zhangbin/RNA/bio_vec/pkx_test_RNA.npy').reshape(-1,1)).astype(np.float)
test_onehot = (np.load('/home/zhangbin/RNA/data_vec/test_one_hot.npy')).astype(np.float)
test_pssm = np.load('/home/zhangbin/RNA/data_vec/R_test_bio_PSSM_vec.npy')
test_hmm = np.load('/home/zhangbin/RNA/data_vec/R_test_bio_HMM_vec.npy')


train_96 = np.concatenate((train_onehot, train_pssm), axis=1)
test_96 = np.concatenate((test_onehot, test_pssm), axis=1)

np.save('/home/zhangbin/RNA/experiment/bio/bio_diff_com/train_o_pssm.npy', train_96)
np.save('/home/zhangbin/RNA/experiment/bio/bio_diff_com/test_o_pssm.npy', test_96)