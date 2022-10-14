import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten, Dense
from tensorflow.keras import backend as K
import numpy as np
import tqdm


class CLCallback(tf.keras.callbacks.Callback):
    """
    tf callback util func
    """
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy= []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))
        
        
        
class MaskedMeanPool(tf.keras.layers.Layer):
    """
    off-the-shelf 1-D mean pooling that supports masking
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskedMeanPool, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        if mask is not None:
            # mask (batch, time)
            mask = K.cast(mask, K.floatx())
            # mask (batch, x_dim, time)
            mask = K.repeat(mask, x.shape[-1])
            # mask (batch, time, x_dim)
            mask = tf.transpose(mask, [0,2,1])
            x = x * mask
        return K.sum(x, axis=1) / K.sum(mask, axis=1)

    def get_output_shape_for(self, input_shape):
        # remove temporal dimension
        return input_shape[0], input_shape[2]
    
    

def reco_seq_construct1(hist_dat, itemid2idx, min_len=4, ds='ml-1m'):
    """
    step-wise construction of training data for requential reco
    for a sequence of item1,...,itemk, the step-wise construction gives:
    item1,item2
    item1,item2, item3,
    ...
    item1,item2, item3,...,itemk
    
    input: hist_dat: dict[user_id:[item1,...,itemk]] # outputs from the preprocessing functions of the "data" module
           itemid2idx: dict[item_id, item_idx]
           min_len: threshold on the minimum lenght of the resulting sequences
    """
    trn, test = [], []
    freq = dict(zip(range(len(itemid2idx)), [0]*len(itemid2idx)))
    for dat in hist_dat:
        if ds == 'amazon':
            dat = [itemid2idx[d[0]] for d in dat if d[0] in itemid2idx]
        if ds == 'ml-1m':
            pass
        
        if len(dat) < min_len:
            continue
        
        for k in dat:
            freq[k] += 1
        
        for i in range(min_len, len(dat)-1):
            trn.append(dat[:i])
        
        test.append(dat)
        
    return trn, test, freq
    
    


def generate_seq_trn_dat(hist, vocab_size, unigrams, for_trn=True, num_ns=10, min_len=4, max_len=20, distortion=1., SEED=23333):
    """
    generate the training/testing sequence+target+label, according to the outputs from "reco_seq_construct1"
    input: hist: [[item1,...,itemk]], the history interaction data for training/testing
           vocab_size: number of items
           unigrams: if the negative sampling is done wrt. item frequence, input the item_freq list
           for_trn: the last interaction will be left-out if the output data will be used for training
           num_ns: number of negative samples, e.g. 3 for training dat, 100 for testing dat
           distortion: the degree for which the negative sampling is conducted under the unigram distribution
                       (e.g. 1.0 for complete unigram-based sampling, 0.0 for complete uniform sampling)
    """
    
    seqs, targets, labels = [], [], []
    for record in hist:
        
        if len(record) < min_len:
            continue
        
        if for_trn:
            target, seq = record[-2], np.array(record[:-2])
        else:
            target, seq = record[-1], np.array(record[:-1]) 
            
        if len(seq) > max_len:
            seq = seq[-max_len:]
        target_class = tf.expand_dims(
              tf.constant([target], dtype="int64"), 1)

        negative_samples = np.random.choice(range(vocab_size), num_ns, replace=False)
        negative_sampling_candidates = tf.constant(negative_samples, dtype="int64")
            
        negative_sampling_candidates = tf.expand_dims(
                      negative_sampling_candidates, 1)

        target = tf.concat([target_class, negative_sampling_candidates], 0)
        label = tf.constant([1] + [0]*num_ns, dtype="int64")
        
        seqs.append(seq+1)
        targets.append(target+1)
        labels.append(label)
    
    return tf.keras.preprocessing.sequence.pad_sequences(
    seqs, maxlen=max_len,padding='pre'), targets, labels



def clf_kernel_score(emb, label):
    """
    compute the kernel-based eval score for OOD performance on entity tasks.
    input: emb: pre-trained embedding of n*d
           label: item class label of n*1
    """
    label = np.array(label)
    emb = emb / np.expand_dims(np.linalg.norm(emb, 2, axis=-1), axis=-1)
    emb_K = emb.dot(emb.T)
    label_K = np.array([[(label[i] == label).astype(np.float32)] for i in range(len(label))])
    label_K = np.squeeze(label_K, axis=1)
    val, cnt = 0., 0.
    
    for i in range(label_K.shape[0]):
        val += np.sum(label_K[i] * emb_K[i])
        cnt += np.sum(label_K[i])
        
    return val / cnt



def generate_trn_dat_by_label(itemid2dept, itemid2idx, n_pos, neg_pos_ratio=3):
    """
    use the item category information to generate positive/negative samples
    """
    item_set = set(list(itemid2idx.keys()))
    dept2itemid = {}
    for key,val in itemid2dept.items():
        try:
            dept2itemid[val].append(key)
        except KeyError:
            dept2itemid[val] = [key]

    targets, contexts, labels = [], [], []

    for iid in list(item_set):
        dept = itemid2dept[iid]
        neg_pool = [v for k,v in dept2itemid.items() if k!=dept]
        neg_pool = [v for vv in neg_pool for v in vv]

        pos_context = np.random.choice(dept2itemid[dept], n_pos, replace=True)
        neg_context = np.random.choice(neg_pool, n_pos*neg_pos_ratio, replace=True)

        for i in range(n_pos):
            pos = [itemid2idx[pos_context[i]]]

            neg = [itemid2idx[iid] for iid in neg_context[i*neg_pos_ratio:(i+1)*neg_pos_ratio]]

            pos = tf.expand_dims(tf.constant(pos), 1)
            neg = tf.expand_dims(tf.constant(neg), 1)

            context = tf.concat([pos, neg], 0)
            label = tf.constant([1] + [0]*neg_pos_ratio, dtype="int64")

            targets.append(itemid2idx[iid])
            contexts.append(context)
            labels.append(label)
            
    return targets, contexts, labels