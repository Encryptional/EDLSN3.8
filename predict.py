import math
import numpy as np
from ensemble_DLsequence_net import ensemble_DLsequence_net
from utils import evaluate, label_sum, label_one_hot, softmax, split_data
from sklearn.model_selection import StratifiedShuffleSplit
from keras.callbacks import (EarlyStopping, LearningRateScheduler)
import os
import random
import warnings
random.seed(2022)
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_gen = np.load('./data_vec/RNA/RNA_gene_train.npy').reshape(-1,1,1024)
# train_bio_vec = np.load('/home/zhangbin/DNA/data_vec/train_bio_vec.npy').reshape(-1,1,50)
train_bio_vec = np.load('./data_vec/RNA/RNA_bio_train.npy').reshape(-1,1,75)
# train_bio_vec = np.load('/home/zhangbin/DNA/data_vec/train_bio_PSSM_vec.npy').reshape(-1,1,20)
train_dyna = np.load('./data_vec/RNA/RNA_dyna_train.npy').reshape(-1,1,1024)
train_label = np.array(label_one_hot(np.load('./multi-feature/RNA/label/train_label.npy')))

test_gen = np.load('./data_vec/RNA/RNA_gene_test.npy').reshape(-1,1,1024)
# tese_bio_vec = np.load('/home/zhangbin/DNA/data_vec/test_bio_vec.npy').reshape(-1,1,50)
tese_bio_vec = np.load('./data_vec/RNA/RNA_bio_test.npy').reshape(-1,1,75)
# tese_bio_vec = np.load('/home/zhangbin/DNA/data_vec/test_bio_PSSM_vec.npy').reshape(-1,1,20)
test_dyna = np.load('./data_vec/RNA/RNA_dyna_test.npy').reshape(-1,1,1024)
test_label = np.array(label_one_hot(np.load('./multi-feature/RNA/label/test_label.npy')))



def step_decay(epoch):
    initial_lrate = 0.0005
    drop = 0.5
    epochs_drop = 7.0
    lrate = initial_lrate * \
        math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    print(lrate)
    return lrate
callbacks = [
    EarlyStopping(monitor='val_loss', patience=6),
             LearningRateScheduler(step_decay)]

positive_list_gen, positive_list_bio, positive_list_dyna, sub_list_gen, sub_list_bio, sub_list_dyna = split_data(train_gen, train_bio_vec, train_dyna, 4)

predict_result = [[0,0]]*len(test_label)
for i in range(len(sub_list_gen)):
    print("**********************************" + 'emsemble_model:' + str((i)) + "*****************************************")
    train_con = np.array(np.concatenate((sub_list_gen[i], positive_list_gen), axis=0))
    train_bio_con = np.array(np.concatenate((sub_list_bio[i], positive_list_bio), axis=0))
    trian_dyna_con = np.array(np.concatenate((sub_list_dyna[i], positive_list_dyna), axis=0))
    label_con = np.concatenate((np.zeros(len(sub_list_gen[i]), dtype=int), np.ones(len(positive_list_gen), dtype=int)))
    label_now = [str(i) for i in label_con]
    train_label_now = np.array(label_one_hot(label_now))
    ##split
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2022)
    for train_index, val_index in split.split(train_con, train_label_now):
        train_X_vec = train_con[train_index]
        train_X_bio = train_bio_con[train_index]
        train_X_dyna = trian_dyna_con[train_index]
        val_X_xl = train_con[val_index]
        val_X_bio = train_bio_con[val_index]
        val_X_dyna = trian_dyna_con[val_index]
        train_y = train_label_now[train_index]
        val_y = train_label_now[val_index]
    ##model
    batchSize = 1024
    maxEpochs = 30
    model = ensemble_DLsequence_net()
    model.fit([train_X_vec, train_X_bio, train_X_dyna], y=train_y,
              epochs=maxEpochs,
              batch_size=batchSize,
              callbacks=callbacks,
              verbose=1,
              validation_data=([val_X_xl, val_X_bio,val_X_dyna], val_y),
              shuffle=True)
    # model.save('./train_model/DNA_'+str(i)+'.h5')

    predict_result1 = label_sum(predict_result, model.predict([test_gen, tese_bio_vec, test_dyna]))

print("%s\t%s\t%s\t%s\t%s" % ('Rec', 'Pre', 'F1', 'MCC', 'AUROC'))
evaluate(test_label, predict_result)