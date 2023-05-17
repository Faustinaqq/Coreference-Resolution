"""
辅助函数
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import bz2file
import pickle
import os
from conll import conll2012

def load_from_pretrained(pretrained_vector_path):
    if not os.path.exists("./save/word.txt") or not os.path.exists("./save/word_vector.npy"):
        word_vectors, i2w, w2i = load_pretrained_vector(pretrained_vector_path)
        dataset = conll2012()
        vocab, vocab_size = dataset.get_vocab()
        vectors = np.zeros((vocab_size+1, word_vectors.shape[-1]), dtype=np.float32)
        cnt = 0
        for i, word in enumerate(vocab):
            if word in i2w:
                vectors[i+1] = word_vectors[w2i[word]]
                cnt += 1
            else:
                vectors[i+1] = np.random.randn(300)
        print("cnt/sum: ", cnt, "/", vocab_size)
        if not os.path.exists("./save"):
            os.makedirs("./save")
        save_i2w("./save/word.txt", vocab)
        np.save("./save/word_vector.npy", vectors)
        w2i = from_i2w_get_w2i(vocab)
        return vocab, w2i, vectors
    else:
        i2w = read_i2w("./save/word.txt")
        vectors = np.load("./save/word_vector.npy")
        print("Loading Vector Done!")
        w2i = from_i2w_get_w2i(i2w)
        return i2w, w2i, vectors


def load_pretrained_vector(word_vec_file):
    '''
    load pretained word vector
    matrix: word vector matrix
    vocab: combination of word2idx and idx2word
    size: embedding_dim
    '''
    i2w = []
    w2i = {}
    idx = 0
    with bz2file.open(word_vec_file, 'r') as f:
        first_line = True
        for line in f:
            line = line.decode('utf-8')
            if first_line:
                first_line = False
                vocab_size = int(line.strip().split()[0])
                size = int(line.rstrip().split()[1])
                word_vectors = np.zeros(shape=(vocab_size, size), dtype=np.float32)
                continue
            vec = line.strip().split()
            if not w2i.__contains__(vec[0]):
                w2i[vec[0]] = idx
                word_vectors[idx, :] = np.array([float(x) for x in vec[1:]])
                idx += 1
    for w in w2i.keys():
        i2w.append(w)
        
    return word_vectors, i2w, w2i


# 保存字符串列表到文件
def save_i2w(filename, i2w):
    with open(filename, "wb") as f:
        pickle.dump(i2w, f)
    return i2w

# 加载文件中的字符串列表
def read_i2w(filename):
    with open(filename, "rb") as f:
        i2w = pickle.load(f)
    return i2w
    
    
def from_i2w_get_w2i(i2w):
    w2i = dict()
    for i, word in enumerate(i2w):
        w2i[word] = i + 1
    return w2i


class FocalLoss(nn.Module):
    def __init__(self, gamma=5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def extract_mentions_to_clusters_from_clusters(clusters):
    mention_to_cluster = {}
    for gc in clusters:
        for mention in gc:
            mention_to_cluster[tuple(mention)] = tuple(gc)
    return mention_to_cluster

def get_f1(precision, recall):
    if precision == 0 and recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)