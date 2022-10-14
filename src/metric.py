import numpy as np
import io
import re
import string
import tqdm
import sys
import pickle
import tensorflow as tf


            
def classification_metric(label, pred, score, auc=False, compute_prob=False, return_res=False):
    """
    metrics for item classification
    inputs are under the same definition for sklearn clf metrics
    """
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score
    
    if len(np.unique(label)) > 2:
        acc = accuracy_score(label, pred)
        f1_micro = f1_score(label, pred, average='micro')
        f1_macro = f1_score(label, pred, average='macro')
        if auc:
            if compute_prob:
                score = np.exp(score) / np.expand_dims(np.sum(np.exp(score), axis=-1), axis=-1)
            auc = roc_auc_score(label, score, multi_class='ovr')
        else:
            auc = -1.0
        print("acc:{:.3f}, f1_micro:{:.3f}, f1_macro:{:.3f}, auc:{:.3f}".format(acc, f1_micro, f1_macro, auc))
        if return_res:
            res = {'acc':acc, 'f1_micro':f1_micro, 'f1_macro':f1_macro, 'auc':auc}
            return res
    else:
        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred)
        auc = roc_auc_score(label, score)
        print("acc:{:.3f}, f1_score:{:.3f}, auc:{:.3f}".format(acc, f1, auc))
        if return_res:
            res = {'acc':acc, 'f1_score':f1, 'auc':auc}
            return res
            
            

def eval_reco(model, test_seq, test_target, test_label, top_k=10, print_res=True):
    """
    metrics for sequential reco evaluation: top-K recall, ndcg and overall mean reciprocal rank
    input: model: TF model with model.predict attribute
           test_seq: testing sequences of [[i_1,...,i_{k-1}]]
           test_target: testing targets of [[pos_target + list_of_negative_targets]]
           test_label: testing labels for the sequence+target of [[1 + [0]*num_ns]]
    """
    recall, ndcg, mrr = 0., 0., 0.
    n = len(test_seq)
    test_dataset = tf.data.Dataset.from_tensor_slices(((test_seq, test_target), test_label))
    test_dataset = test_dataset.batch(len(test_seq))
    score = model.predict(test_dataset)
    for s in score:
        ranking = len(s) - s.argsort().argsort()
        
        mrr += ranking[0]
        if ranking[0] <= top_k:
            recall += 1
            ndcg += 1. / np.log(1+ranking[0])
    if print_res:        
        print("top-{} recall:{:3f}, ndcg:{:3f}, mrr:{:3f}".format(top_k, recall/n, ndcg/n, mrr/n ))
    
    return (recall/n, ndcg/n, mrr/n)