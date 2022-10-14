import json
import numpy as np
import argparse
import os
import traceback
import time
import tqdm 
import pickle
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten, Dense, GlobalAveragePooling1D
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import sys
sys.path.insert(0, './src')
from utils import MaskedMeanPool, reco_seq_construct1, generate_seq_trn_dat, clf_kernel_score
from data import generate_cl_trn_dat, ml_rating_preprocess
from metric import eval_reco, classification_metric, ml_item_label
from model import Item2Vec, AvgEmbforRec, DenseforRec, AttnforRec, GruforRec, MLP, mlp_clf, logistic_reg, kernel_svm


# setting up GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

log_dir = 'log/ml_seq_reco/'
dat_dir = 'data/ml-1m/'
reload = False
save_emb = False

CL_setting = [(3,3), (3,2), (3,1), (2,3), (2,2), (2,1)] # different emb pre-trn setting
embedding_dim = 32


def main(log_dir, dat_dir, emb_dim, CL_setting, reload, save_emb):
    # preprocessing data
    if not reload:
        user_hist, itemid2idx = ml_rating_preprocess(dat_dir + 'ratings.dat')
        order_hist_trn, order_hist_test, id_freq = reco_seq_construct1(list(user_hist.values()), itemid2idx)
        seqs_trn, targets_trn, labels_trn = generate_seq_trn_dat(order_hist_trn, 
                                                                 len(itemid2idx), 
                                                                 list(id_freq.values()), 
                                                                 for_trn=True, 
                                                                 num_ns=2, 
                                                                 SEED=23333, 
                                                                 min_len=4, 
                                                                 max_len=20, 
                                                                 distortion=0.)

        seqs_test, targets_test, labels_test = generate_seq_trn_dat(order_hist_test, 
                                                                 len(itemid2idx), 
                                                                 list(id_freq.values()), 
                                                                 for_trn=False, 
                                                                 num_ns=100, 
                                                                 SEED=23333, 
                                                                 min_len=4, 
                                                                 max_len=20, 
                                                                 distortion=0.)


        store_var = dict({"seqs_trn":seqs_trn, "targets_trn":targets_trn, "labels_trn":labels_trn,
                          "seqs_test":seqs_test, "targets_test":targets_test, "labels_test":labels_test,
                          "itemid2idx":itemid2idx, "id_freq":id_freq, "user_hist":user_hist})

        with open(log_dir + 'seq_dat.pkl', 'wb') as f:
            pickle.dump(store_var, f)

    else:
        with open(log_dir + 'seq_dat.pkl', 'rb') as f:
            store_var = pickle.load(f)

        seqs_trn, targets_trn, labels_trn = store_var['seqs_trn'], store_var['targets_trn'], store_var['labels_trn']
        seqs_test, targets_test, labels_test = store_var['seqs_test'], store_var['targets_test'], store_var['labels_test']
        itemid2idx = store_var['itemid2idx']
        id_freq = store_var['id_freq']
        user_hist = store_var['user_hist']

    # pre-training + downstream task
    dense_res, attn_res, gru_res, avgemb_res = [], [], [], []
    kernel_scores = []
    f1_micro_kernel, f1_macro_kernel, f1_micro_lr, f1_macro_lr, f1_micro_mlp, f1_macro_mlp = [],[],[],[],[],[]


    # generate seq reco task's tf dataset
    BATCH_SIZE = 1024
    BUFFER_SIZE = 10000
    seq_dataset = tf.data.Dataset.from_tensor_slices(((seqs_trn, targets_trn), labels_trn))
    seq_dataset = seq_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    for ws, ns in CL_setting:

        # pre-training item embedding
        targets, contexts, labels = generate_cl_trn_dat(list(user_hist.values()), ws, ns, len(itemid2idx))

        BATCH_SIZE = 1024
        BUFFER_SIZE = 10000
        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        item2vec = Item2Vec(len(itemid2idx), embedding_dim)
        item2vec.compile(optimizer='adam',
                         loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])

        item2vec.fit(dataset, epochs=20, verbose=0)

        item_emb = item2vec.get_layer('item_embedding').get_weights()[0]

        if save_emb:
            fname = "emb_{}_{}.pkl".format(ws,ns)
            with open(log_dir + fname, 'wb') as f:
                pickle.dump(item_emb, f)


        # training dense reco with pre-trained emb
        DenseRecPreTrn = DenseforRec(32, len(itemid2idx), hidden_units_l=[32,16], pre_emb = item_emb, l2_reg=0.)
        DenseRecPreTrn.compile(optimizer='adam',
                             loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                             metrics=['accuracy'])
        DenseRecPreTrn.fit(seq_dataset, epochs=30, verbose=0)

        dense_res.append(eval_reco(DenseRecPreTrn, seqs_test, targets_test, labels_test))

        # training GRU reco with pre-trained emb
        GruRecPreTrn = GruforRec(32, len(itemid2idx), hidden_units_l=[32,16], pre_emb = item_emb, l2_reg=0.)
        GruRecPreTrn.compile(optimizer='adam',
                             loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                             metrics=['accuracy'])
        GruRecPreTrn.fit(seq_dataset, epochs=30, verbose=0)

        gru_res.append(eval_reco(GruRecPreTrn, seqs_test, targets_test, labels_test))

        # training Attn reco with pre-trained emb
        AttnRecPreTrn = AttnforRec(32, len(itemid2idx), hidden_units=[32, 16], pre_emb = item_emb, l2_reg=0.)
        AttnRecPreTrn.compile(optimizer='adam',
                             loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                             metrics=['accuracy'])
        AttnRecPreTrn.fit(seq_dataset, epochs=30, verbose=0)

        attn_res.append(eval_reco(AttnRecPreTrn, seqs_test, targets_test, labels_test))

        # discounted avg emb reco with pre-trained emb
        avgemb_res.append(AvgEmbforRec(item_emb, id_freq, seqs_test, targets_test, labels_test, alpha=0.0001))

        # starting downstream item clf tasks
        emb_clf, label = ml_item_label(dat_dir + 'movies.dat', item_emb, itemid2idx)
        feat = emb_clf
        
        kernel_scores.append(clf_kernel_score(feat, label))

        emb_pred, emb_pred_score, test_label = kernel_svm(feat, label)
        kernel_clf_res = classification_metric(test_label, emb_pred, emb_pred_score, auc=False, return_res=True)
        f1_micro_kernel.append(kernel_clf_res['f1_micro'])
        f1_macro_kernel.append(kernel_clf_res['f1_macro'])

        emb_pred, emb_pred_score, test_label = logistic_reg(feat, label)
        lr_clf_res = classification_metric(test_label, emb_pred, emb_pred_score, auc=False, return_res=True)
        f1_micro_lr.append(lr_clf_res['f1_micro'])
        f1_macro_lr.append(lr_clf_res['f1_macro'])

        emb_pred, emb_pred_score, test_label = mlp_clf(feat, label, epochs=100, l2_reg=0.0001)
        mlp_clf_res = classification_metric(test_label, emb_pred, emb_pred_score, auc=False, return_res=True)
        f1_micro_mlp.append(mlp_clf_res['f1_micro'])
        f1_macro_mlp.append(mlp_clf_res['f1_macro'])


    # storing the results
    reco_res = {'dense':dense_res, 'gru':gru_res, 'attn':attn_res, 'avgemb':avgemb_res}
    with open(log_dir + 'emb_reco_res.pkl', 'wb') as f:
        pickle.dump(reco_res, f)

    clf_res = {'kernel_scores':kernel_scores, 'f1_micro_kernel':f1_micro_kernel, 'f1_macro_kernel':f1_macro_kernel,
               'f1_micro_lr':f1_micro_lr, 'f1_macro_lr':f1_macro_lr, 'f1_micro_mlp':f1_micro_mlp,
               'f1_macro_mlp':f1_macro_mlp}

    with open(log_dir+'clf_res.pkl', 'wb') as f:
        pickle.dump(clf_res, f)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="log/ml_seq_reco/")
    parser.add_argument("--dat_dir", type=str, default="data/ml-1m/")
    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--reload", type=int, default=0, choices=[0,1]) # whether to reuse logged data, or run preprocessing again
    parser.add_argument("--save_emb", type=int, default=0, choices=[0,1]) # whether to save trained embeddings
    parser.add_argument("--GPU", type=int, default=1)
    
    args = parser.parse_args()
    
    # setting up GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[args.GPU], True)
            tf.config.experimental.set_visible_devices(gpus[args.GPU], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)

    log_dir = args.log_dir
    dat_dir = args.dat_dir
    reload = args.reload
    save_emb = args.save_emb
    embedding_dim = args.emb_dim
    CL_setting = [(3,3), (3,2), (3,1), (2,3), (2,2), (2,1)] # different emb pre-trn setting
    
    main(log_dir, dat_dir, emb_dim, CL_setting, reload, save_emb)    
