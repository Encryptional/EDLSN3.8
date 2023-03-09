import numpy as np

train_raa = (np.load('./multi-feature/RNA/bio/raa_train_RNA.npy').reshape(-1,1)).astype(np.float)
train_pca1 = (np.load('./multi-feature/RNA/bio/pychar_train_RNA1.npy').reshape(-1,1)).astype(np.float)
train_pca2 = (np.load('./multi-feature/RNA/bio/pychar_train_RNA2.npy').reshape(-1,1)).astype(np.float)
train_pca3 = (np.load('./multi-feature/RNA/bio/pychar_train_RNA3.npy').reshape(-1,1)).astype(np.float)
train_pkx = (np.load('./multi-feature/RNA/bio/pkx_train_RNA.npy').reshape(-1,1)).astype(np.float)
train_onehot = (np.load('./multi-feature/RNA/bio/train_one_hot.npy')).astype(np.float)
train_pssm = np.load('./multi-feature/RNA/bio/R_train_bio_PSSM_vec.npy')
train_hmm = np.load('./multi-feature/RNA/bio/R_train_bio__HMM_vec.npy')


test_raa = (np.load('./multi-feature/RNA/bio/raa_test_RNA.npy').reshape(-1,1)).astype(np.float)
test_pca1 = (np.load('./multi-feature/RNA/bio/pychar_test_RNA1.npy').reshape(-1,1)).astype(np.float)
test_pca2 = (np.load('./multi-feature/RNA/bio/pychar_test_RNA2.npy').reshape(-1,1)).astype(np.float)
test_pca3 = (np.load('./multi-feature/RNA/bio/pychar_test_RNA3.npy').reshape(-1,1)).astype(np.float)
test_pkx = (np.load('./multi-feature/RNA/bio/pkx_test_RNA.npy').reshape(-1,1)).astype(np.float)
test_onehot = (np.load('./multi-feature/RNA/bio/test_one_hot.npy')).astype(np.float)
test_pssm = np.load('./multi-feature/RNA/bio/R_test_bio_PSSM_vec.npy')
test_hmm = np.load('./multi-feature/RNA/bio/R_test_bio_HMM_vec.npy')


train_bio = np.concatenate(((train_raa, train_pca1, train_pca2, train_pca3, train_pkx, train_pssm, train_hmm,train_onehot), axis=1)
test_bio = np.concatenate((test_raa, test_pca1, test_pca2, test_pca3, test_pkx, test_pssm, test_hmm, test_onehot), axis=1)

np.save('./data_vec/RNA/RNA_bio_train.npy', train_bio)
np.save('./data_vec/RNA/RNA_bio_test.npy', test_bio)