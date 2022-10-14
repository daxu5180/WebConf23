import numpy as np
import io
import re
import string
import tensorflow as tf
import tqdm
import sys
import pickle
from datetime import datetime
from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from collections import Counter


def generate_cl_trn_dat(seqs, ws, num_ns, vocab_size, save_last=True, SEED=23333):
    """
    generate tiplets for contrastive-learning of item emb, in a word2vec fashion
    input: seqs: [[item1,...,itemk]] # list of item sequences
           ws: window size that defines contextually similar items
           num_ns: number of negative samples
           save_last: whether the last item of the sequence will be used for contrastive-learning
                      # set to false for tuning
    output: targets: n*1 tensor of target items
            contexts: n*(1+num_ns)*1 tensor of context items (1 positive + num_ns negative)
            labels: n*(1+num_ns)*1 tensor of labels, each with [1, [0]*num_ns]
    """
    targets, contexts, labels = [], [], []
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
    
    for seq in seqs:
        if save_last:
            seq = seq[:-1]
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          seq,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=ws,
          negative_samples=0)
        
        for target_item, context_item in positive_skip_grams:
            context_class = tf.expand_dims(
              tf.constant([context_item], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
              true_classes=context_class,
              num_true=1,
              num_sampled=num_ns,
              unique=True,
              range_max=vocab_size,
              seed=SEED,
              name="negative_sampling")
            
            negative_sampling_candidates = tf.expand_dims(
                      negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype="int64")

            targets.append(target_item)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels




def instacart_2_hist(order_file:str, min_len=5, max_len=40, min_freq=5):
    """
    preprocess the instacart data, from order history to sequence
    
    """
    order_dict_raw = {}
    item_list_raw = []

    with open(order_file, "r") as f:
        for line in f:
            if not line[0].isdigit():
                continue
            line = line.strip("\n").split(",")
            try:
                order_dict_raw[line[0]].append(line[1])
            except KeyError:
                order_dict_raw[line[0]] = [line[1]]
            item_list_raw.append(line[1])


    item_freq_dict = Counter(item_list_raw)
    allowed_item_set = set([k for k,v in item_freq_dict.items() if v > min_freq])

    order_hist, item_list = [], []

    for key,val in order_dict_raw.items():

        if len(val) > min_len and len(val) <= max_len:
            order_hist.append([v for v in order_dict_raw[key] if v in allowed_item_set])
            item_list.extend(order_hist[-1])
        if len(val) > max_len:
            order_hist.append([v for v in order_dict_raw[key][:max_len] if v in allowed_item_set])
            item_list.extend(order_hist[-1])


    item_set = set(item_list)
    itemid2idx = dict(zip(list(item_set), range(len(item_set))))
    itemidx2id = {v:k for k,v in itemid2idx.items()}

    order_hist_id = [[itemid2idx[iid] for iid in hist] for hist in order_hist]
    
    return order_hist_id, itemid2idx, itemidx2id





def amzn_preprocess(fname:str):
    """
    preprocess amzn electronics dat
    input: amzn raw data file loc
    output: emb_seq_dat: [[item1,...,itemk]] # for fast word2vec-type item emb training
            user_hist: dict[user_id:[item1,...,item_k]] # for building seq reco model
            itemid2idx: dict[item_id:item_idx] # coverting id to consecutive index that starts from 0
    """
    item_set = []
    user_hist = {}
    with open('data/amazon/reviews_Electronics_5.json', 'r') as f:
        for line in f:
            line = eval(line)
            item_set.append(line['asin'])
            try:
                user_hist[line['reviewerID']].append([line['asin'], 
                                                      line['overall'], 
                                                      line['unixReviewTime'], 
                                                      datetime.strptime(line['reviewTime'], '%m %d, %Y')])
            except KeyError:
                user_hist[line['reviewerID']] = [[line['asin'], 
                                                      float(line['overall']), 
                                                      line['unixReviewTime'], 
                                                      datetime.strptime(line['reviewTime'], '%m %d, %Y')]]
    item_set = set(item_set)
    item_feat = {}
    with open('data/amazon/metadata.json', 'r') as f:
        for line in f:
            line = eval(line)
            if line['asin'] in item_set:
                try:
                    item_feat[line['asin']] = {'title':line['title'],
                                               'categories': line['categories']}
                except KeyError:
                    item_set.remove(line['asin'])
                    
    itemid2idx = dict(zip(list(item_set), range(len(item_set))))
    user_hist_processed = {}
    emb_seq_dat = []
    for k,v in user_hist.items():
        hist =  [vv for vv in v if vv[0] in item_set]
        hist = sorted(hist, key=lambda x:x[-2])
        user_hist_processed[k] = hist
        emb_seq_dat.append([itemid2idx[vv[0]] for vv in hist])
        
    return emb_seq_dat, user_hist, itemid2idx



def ml_rating_preprocess(fname:str):
    """
    preprocess movielens-1m dat
    input: ML-1M raw data file loc
    output: user_hist: dict[user_id:[item1,...,item_k]] # for building seq reco model
            itemid2idx: dict[item_id:item_idx] # coverting id to consecutive index that starts from 0
    """
    user_hist = {}
    itemfreq = {}
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip().split('::')
            line = list(map(int, line))
            try:
                itemfreq[line[0]] += 1
            except KeyError:
                itemfreq[line[0]] = 0
            try:
                user_hist[line[0]].append([line[1], line[-1]])
            except KeyError:
                user_hist[line[0]] = [[line[1], line[-1]]]
    
    
    itemfreq = {k: v for k, v in sorted(itemfreq.items(), key=lambda item: item[1])}
    itemid2idx = dict(zip(list(itemfreq.keys())[::-1], range(len(itemfreq))))
    itemfreq = {itemid2idx[k]:v for k,v in itemfreq.items()}
    
    for k,v in user_hist.items():
        try:
            hist = sorted(v, key=lambda x:x[1])
        except:
            print(v)
        user_hist[k] = [itemid2idx[v[0]] for v in hist]
        
    return user_hist, itemid2idx


def instacart_item_label(instacart_dir, itemid2idx):
    """
    load item classification information from instacart
    """
    with open(instacart_dir + "products.csv", 'r') as f:
        for line in f:
            line = line.strip("\n").split(",")
            if line[0] in itemid2idx:
                itemid2dept[line[0]] = int(line[-1])
            
    label = [itemid2dept[itemidx2id[i]] for i in range(len(itemid2idx))]
    return label



def ml_item_label(dat_loc, emb, itemid2idx):
    """
    generating movies' genre classficaition dataset
    """
    cat = {}
    with open(dat_loc, 'r', encoding = "ISO-8859-1") as f:
        for line in f:
            line = line.strip('\n')
            if "|" in line:
                item_cat = line.split("|")[-1]
            else:
                item_cat = line.split("::")[-1]
            itemid = int(line.split("::")[0])
            cat[itemid2idx[itemid]] = item_cat
            
    unique_cat = np.unique(list(cat.values()))
    cat2id = dict(zip(unique_cat, range(len(unique_cat))))
    emb_clf = []
    label = []
    for i in range(len(itemid2idx)):
        if i in cat:
            label.append(cat2id[cat[i]])
            emb_clf.append(emb[i])

    emb_clf = np.array(emb_clf)
    
    return emb_clf, label


def amazon_item_label(file='data/amazon/metadata.json', itemid2idx):
    """
    generate amazon items' classficiation data
    """
    cat = {}
    with open(file, 'r') as f:
        for line in f:
            line = eval(line)
            if line['asin'] in itemid2idx:
                cat[itemid2idx[line['asin']]] = line[categories]
    label = [cat[i] for i in range(len(itemid2idx))]
    
    label_final = [l[1] for l in label]
    unique_cat = np.unique(label_final)
    cat2id = dict(zip(unique_cat, range(len(unique_cat))))
    label_final = [cat2id[item_cat] for item_cat in label_final]
                
    return label


def instacart_item_label(instacart_dir, itemid2idx):
    """
    generate instacart items' classification data
    """
    with open(instacart_dir + "products.csv", 'r') as f:
    for line in f:
        line = line.strip("\n").split(",")
        if line[0] in itemid2idx:
            itemid2dept[line[0]] = int(line[-1])
            
    label = [itemid2dept[itemidx2id[i]] for i in range(len(itemid2idx))]
    return label