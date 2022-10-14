import numpy as np
import io
import re
import string
import tensorflow as tf
import tqdm
import sys
import pickle

from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten, Dense, GlobalAveragePooling1D
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from utils import MaskedMeanPool



class Item2Vec(Model):
    """
    modifying the word2vec model for pre-training item emb
    """
    def __init__(self, vocab_size, embedding_dim):
        super(Item2Vec, self).__init__()
        self.item_embedding = Embedding(vocab_size,
                                          embedding_dim,
                                          input_length=1,
                                          embeddings_initializer='uniform',
                                          name="item_embedding")

        self.dots = Dot(axes=(3, 2))
        self.flatten = Flatten()

    def call(self, pair):
        target, context = pair
        word_emb = self.item_embedding(target)
        context_emb = self.item_embedding(context)
        dots = self.dots([context_emb, word_emb])
        return self.flatten(dots)
    
    

class MLP(Model):
    """
    the regular two-layer MLP for item clf based on pre-trained item emb
    """
    def __init__(self, hidden_units, n_class, activation='relu', l2_reg=0.05):
        super(MLP, self).__init__()
        self.dense1 = Dense(hidden_units, activation=activation, kernel_regularizer=tf.keras.regularizers.L2(0.01))
        self.dense2 = Dense(n_class, kernel_regularizer=tf.keras.regularizers.L2(0.01))
    
    def call(self, x):
        hidden = self.dense1(x)
        hidden = self.dense2(hidden)
        return hidden
    
    
    
def mlp_clf(emb, label, hidden_units=32, epochs=20, split_ratio=0.2, SEED=0, verbose=0):
    """
    a wrap-up function for setting up and evaluting MLP-based item clf
    """
    
    from sklearn.model_selection import train_test_split
    
    n_class = len(np.unique(label))
    
    emb_trn, emb_test, trn_label, test_label = train_test_split(
        emb, label, test_size=split_ratio, random_state=SEED)
    
    trn_label = tf.one_hot(np.array(trn_label)-1, depth=n_class).numpy()
    
    model = MLP(hidden_units, n_class)
    model.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
    
    BATCH_SIZE = 256
    BUFFER_SIZE = 10000
    dataset = tf.data.Dataset.from_tensor_slices((emb_trn, trn_label))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    
    model.fit(dataset, epochs=epochs, verbose=verbose)
    score = model.predict(emb_test)
    pred = np.argmax(score, axis=-1)+1
    
    return pred, score, test_label



def logistic_reg(emb, label, split_ratio=0.2, SEED=0):
    """
    a wrap-up function for setting up and evaluting LR-based item clf
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    emb_trn, emb_test, trn_label, test_label = train_test_split(
        emb, label, test_size=split_ratio, random_state=SEED)
    
    model = LogisticRegression(random_state=SEED, 
                               max_iter=300, fit_intercept=False).fit(emb_trn, trn_label)
    
    prediction = model.predict(emb_test)
    prob = model.predict_proba(emb_test)
    
    return prediction, prob, test_label



def kernel_svm(emb, label, split_ratio=0.2, SEED=0):
    """
    a wrap-up function for setting up and evaluting kernel-SVM-based item clf
    """
    
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    
    emb_trn, emb_test, trn_label, test_label = train_test_split(
        emb, label, test_size=split_ratio, random_state=SEED)
    
    trn_kernel = np.dot(emb_trn, emb_trn.T)
    model = SVC(gamma='auto', kernel='precomputed')
    model.fit(trn_kernel, trn_label)
    
    test_kernel = np.dot(emb_test, emb_trn.T)
    prediction = model.predict(test_kernel)
    score = model.decision_function(test_kernel)
    
    return prediction, score, test_label


def d2v(item_desc, dim, epochs=50):
    """
    modifying the doc2vec model for pre-training item emb
    """
    
    import gensim
    from gensim.parsing.preprocessing import remove_stopwords
    from gensim.parsing.preprocessing import preprocess_string
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    
    
    item_desc = [preprocess_string(v) for k,v in itemid2desc.items()]
    item_desc = [TaggedDocument(doc, [i]) for i, doc in enumerate(item_desc)]

    d2v_model = gensim.models.doc2vec.Doc2Vec(vector_size=dim, min_count=2, dm=0, dbow_words=1, epochs=epochs)
    d2v_model.build_vocab(item_desc)
    d2v_model.train(item_desc, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)
    
    d2v_emb = []
    for i in range(len(itemid2desc)):
        d2v_emb.append(d2v_model.docvecs[i])
        
    return d2v_emb


    

def AvgEmbforRec(pre_emb, itemid2freq, test_seq, test_target, test_label, alpha=0.1, top_k=10, print_res=True):
    """
    using the discounted sum of pre-trained item emb as the sequence emb, for sequential recommendation
    input: pre_emb: pre-trained item embeddings;
           itemid2freq: dict[item_id: frequency], used for discounting the item emb
           test_seq, test_target, test_label: the testing data 
           alpha: the damping factor that counters the item-frequency-based discount
           top_k: number of top reco used in evaluation
    """
    emb_dim, n = pre_emb.shape[1], len(test_seq)
    emb = np.concatenate((np.zeros((1,emb_dim)), pre_emb), axis=0)
    tol = np.sum(list(itemid2freq.values()))
    itemid2freq = {(k+1):(v*1.0/tol) for k,v in itemid2freq.items()}
    itemid2freq[0] = 0
    
    recall, ndcg, mrr = 0, 0, 0
    for seq, target, label in zip(test_seq, test_target, test_label):
        target = target.numpy().flatten()
        label = label.numpy().flatten()
        s = []
        seq_emb = np.zeros(emb_dim)
        for iid in seq:
            seq_emb += (alpha / (alpha + itemid2freq[iid])) * emb[iid]
        for iid in target:
            s.append(np.dot(seq_emb, emb[iid]))
        assert label[0] == 1
        s = np.array(s)
        ranking = len(s) - s.argsort().argsort()
        
        mrr += ranking[0]
        if ranking[0] <= top_k:
            recall += 1
            ndcg += 1. / np.log(1+ranking[0])
    
    if print_res:        
        print("top-{} recall:{:3f}, ndcg:{:3f}, mrr:{:3f}".format(top_k, recall/n, ndcg/n, mrr/n ))
    
    return (recall/n, ndcg/n, mrr/n)
    

"""
the following three models implements Dense / GRU / Attn for rec, with simplifed (yet still powerful) two-tower architectures:


                Binary cross-ent loss
                         |
                    [Inner prod]
               /                      \
        [Dense]                        | 
           |                           |
[aggregate by Dense/GRU/Attn ]      [Dense]
  /       |             \              |
item1   item2   ...   itemk         item(k+1)

Item emb can be pre-trained, or jointly optimized with the rest of the model.

"""

class DenseforRec(Model):
    """
    input: emb_dim: embedding dimension
           vocab_size: number of items 
           pre_emb: pre-trained embedding; set to None if use jointly training
           hidden_units: hidden dims for the dense layers
    """
    
    def __init__(self, emb_dim, vocab_size, 
                 pre_emb=None, 
                 hidden_units_l=[16, 8], 
                 activation='sigmoid', 
                 l2_reg=0.):
        
        super(DenseforRec, self).__init__()
        if pre_emb is None:
            self.emb = Embedding(vocab_size+1, emb_dim, mask_zero=True,
                                          input_length=1,
                                          embeddings_initializer='uniform',
                                          name="item_embedding")
        else:
            assert pre_emb.shape[0] == vocab_size and pre_emb.shape[1] == emb_dim
            embedding_matrix = np.concatenate((np.zeros((1,emb_dim)), pre_emb), axis=0)
            self.emb = Embedding(vocab_size+1, emb_dim, mask_zero=True,
                                          input_length=1,
                                          embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                          trainable=False,
                                          name="item_embedding")
        
        
        # we use layer_l, layer_r to denote whethe the layer is used for the left or right tower
        self.Dense1_l = Dense(hidden_units_l[0], activation=activation, 
                            kernel_regularizer=tf.keras.regularizers.L2(l2_reg), use_bias=True)
        
        self.Dense2_l = Dense(hidden_units_l[1], activation=activation, 
                            kernel_regularizer=tf.keras.regularizers.L2(l2_reg), use_bias=True)
        
        self.Dense_r = Dense(hidden_units_l[1], activation=activation, 
                            kernel_regularizer=tf.keras.regularizers.L2(l2_reg), use_bias=True)
        
        # use off-the-shelf masked mean pooling
        self.pooling = MaskedMeanPool()
        self.dots = Dot(axes=(3, 2))
        self.flatten = Flatten()
        
    def call(self, x):
        seq, target = x
        seq_hidden = self.emb(seq)
        seq_mask = seq_hidden._keras_mask
        seq_hidden = self.Dense1_l(seq_hidden)
        seq_hidden = self.pooling(seq_hidden, seq_mask)
        seq_hidden = tf.expand_dims(self.Dense2_l(seq_hidden), axis=-2)
        
        target_hidden = self.emb(target)
        target_hidden = self.Dense_r(target_hidden)

        dots = self.dots([target_hidden, seq_hidden])
        return self.flatten(dots)
    

    
class AttnforRec(Model):
    
    def __init__(self, emb_dim, vocab_size, 
                 pre_emb=None, 
                 hidden_units=[16, 16], 
                 activation='sigmoid', 
                 l2_reg=0.):
        
        super(AttnforRec, self).__init__()
        
        if pre_emb is None:
            self.emb = Embedding(vocab_size+1, emb_dim, mask_zero=True,
                                          input_length=1,
                                          embeddings_initializer='uniform',
                                          name="item_embedding")
        else:
            assert pre_emb.shape[0] == vocab_size and pre_emb.shape[1] == emb_dim
            embedding_matrix = np.concatenate((np.zeros((1,emb_dim)), pre_emb), axis=0)
            self.emb = Embedding(vocab_size+1, emb_dim, mask_zero=True,
                                          input_length=1,
                                          embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                          trainable=False,
                                          name="item_embedding")
            
        self.attn = tf.keras.layers.Attention()
        # use separate dense layers for thee key, value and query mapping
        self.dense_key = Dense(hidden_units[0], activation=activation, 
                            kernel_regularizer=tf.keras.regularizers.L2(l2_reg), use_bias=True)
        
        self.dense_val = Dense(hidden_units[0], activation=activation, 
                            kernel_regularizer=tf.keras.regularizers.L2(l2_reg), use_bias=True)
        
        self.dense_query = Dense(hidden_units[0], activation=activation, 
                            kernel_regularizer=tf.keras.regularizers.L2(l2_reg), use_bias=True)
        
        # the dense layers for the left and right towers
        self.dense_l = Dense(hidden_units[1], activation=activation, 
                            kernel_regularizer=tf.keras.regularizers.L2(l2_reg), use_bias=True)
        
        self.dense_r = Dense(hidden_units[1], activation=activation, 
                            kernel_regularizer=tf.keras.regularizers.L2(l2_reg), use_bias=True)
        
        # keras.layers.Attention already handles the masking, so just use regular mean pooling
        self.pooling = layers.GlobalAveragePooling1D()
        self.dots = Dot(axes=(1, 3))
        self.flatten = Flatten()
        
    def call(self, x):
        seq, target = x
        seq_hidden = self.emb(seq)
        seq_mask = seq_hidden._keras_mask
        
        
        seq_key = self.dense_key(seq_hidden)
        seq_val = self.dense_val(seq_hidden)
        seq_query = self.dense_query(seq_hidden)
        seq_attn_hidden = self.attn([seq_query, seq_key, seq_val], mask=[seq_mask,seq_mask])
        seq_attn_hidden = self.pooling(seq_attn_hidden, seq_mask)
        seq_attn_hidden = self.dense_l(seq_attn_hidden)
        
        target_hidden = self.emb(target)
        
        target_hidden = self.dense_r(target_hidden)
        dots = self.dots([seq_attn_hidden, target_hidden])
        
        return self.flatten(dots)
    
    
class GruforRec(Model):
    
    def __init__(self, emb_dim, vocab_size, 
                 pre_emb=None, 
                 hidden_units_l=[16, 8], 
                 activation='sigmoid', 
                 l2_reg=0.):
        
        super(GruforRec, self).__init__()
        
        if pre_emb is None:
            self.emb = Embedding(vocab_size+1, emb_dim, mask_zero=True,
                                          input_length=1,
                                          embeddings_initializer='uniform',
                                          name="item_embedding")
        else:
            assert pre_emb.shape[0] == vocab_size and pre_emb.shape[1] == emb_dim
            embedding_matrix = np.concatenate((np.zeros((1,emb_dim)), pre_emb), axis=0)
            self.emb = Embedding(vocab_size+1, emb_dim, mask_zero=True,
                                          input_length=1,
                                          embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                          trainable=False,
                                          name="item_embedding")
            
        self.gru = layers.GRU(units=hidden_units_l[0])
        self.dense_l = Dense(hidden_units_l[1], activation=activation, 
                            kernel_regularizer=tf.keras.regularizers.L2(l2_reg), use_bias=True)
        
        self.dense_r = Dense(hidden_units_l[1], activation=activation, 
                            kernel_regularizer=tf.keras.regularizers.L2(l2_reg), use_bias=True)
        
        self.dots = Dot(axes=(1, 3))
        self.flatten = Flatten()
        
    def call(self, x):
        
        seq, target = x
        seq_hidden = self.emb(seq)
        seq_mask = seq_hidden._keras_mask
        
        # make sure to use the mask when obtaining the GRU output
        seq_output = self.gru(seq_hidden, mask = seq_mask)
        seq_output = self.dense_l(seq_output)
        
        target_hidden = self.emb(target)
        target_hidden = self.dense_r(target_hidden)

        dots = self.dots([seq_output, target_hidden])
        
        return self.flatten(dots)