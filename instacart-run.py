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
from utils import clf_kernel_score
from data import generate_cl_trn_dat, instacart_item_label, instacart_2_hist
from metric import classification_metric
from model import Item2Vec, MLP, mlp_clf, logistic_reg, kernel_svm



def main(log_dir, dat_dir, emb_dim, CL_setting, reload, save_emb):
    
    # preprocessing data
    if reload:
        with open(log_dir + "trn_dat.pkl", "rb") as f:
            store_var = pickle.load(f)

        itemidx2id = store_var['itemidx2id']
        itemid2idx = {v:k for k,v in itemidx2id.items()}
        order_hist_id = store_var['order_hist_id']

    else:
        order_hist_id, itemid2idx, itemidx2id = instacart_2_hist(dat_dir)

    # pre-training + downstream task
    kernel_scores = []
    f1_micro_kernel, f1_macro_kernel, f1_micro_lr, f1_macro_lr, f1_micro_mlp, f1_macro_mlp = [],[],[],[],[],[]


    for ws, ns in CL_setting:

        # pre-training item embedding
        targets, contexts, labels = generate_cl_trn_dat(order_hist_id, ws, ns, len(itemid2idx))

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

        label = instacart_item_label(dat_dir, itemid2idx)
        feat = item_emb
        
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


    clf_res = {'kernel_scores':kernel_scores, 'f1_micro_kernel':f1_micro_kernel, 'f1_macro_kernel':f1_macro_kernel,
               'f1_micro_lr':f1_micro_lr, 'f1_macro_lr':f1_macro_lr, 'f1_micro_mlp':f1_micro_mlp,
               'f1_macro_mlp':f1_macro_mlp}

    with open(log_dir+'clf_res.pkl', 'wb') as f:
        pickle.dump(clf_res, f)
    
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="log/instacart_emb_var_exp/")
    parser.add_argument("--dat_dir", type=str, default="data/instacart_2017_05_01/")
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