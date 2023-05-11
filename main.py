from conll import conll2012

from train import Trainer

from model import CorefScore
import torch
import codecs
import numpy as np
import bz2file
import pickle
import os

def load_pretrained_vector(word_vec_file):
    '''
    load pretained word vector
    matrix: word vector matrix
    vocab: combination of word2idx and idx2word
    size: embedding_dim
    '''
    i2w = []
    w2i = {}
    idx = 1
    with bz2file.open(word_vec_file, 'r') as f:
        first_line = True
        for line in f:
            line = line.decode('utf-8')
            if first_line:
                first_line = False
                vocab_size = int(line.strip().split()[0])
                size = int(line.rstrip().split()[1])
                word_vectors = np.zeros(shape=(vocab_size+1, size), dtype=np.float32)
                continue
            vec = line.strip().split()
            if not w2i.__contains__(vec[0]):
                w2i[vec[0]] = idx
                word_vectors[idx, :] = np.array([float(x) for x in vec[1:]])
                idx += 1
    for w in w2i.keys():
        i2w.append(w)
        
    print(i2w[:100])
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
        w2i[word] = i
    return w2i

if __name__ == '__main__':
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if not os.path.exists("word.txt") or not os.path.exists("word_vector.npy"):
        word_vectors, i2w, w2i = load_pretrained_vector('/home/fqq/data/sgns.merge.word.bz2')
        dataset = conll2012()
        vocab, vocab_size = dataset.get_vocab()
        vectors = np.zeros((vocab_size, word_vectors.shape[-1]))
        cnt = 0
        for i, word in enumerate(vocab):
            if word in i2w:
                vectors[i] = word_vectors[w2i[word]]
                cnt += 1
            else:
                vectors[i] = np.random.randn(300)
        print("cnt/sum: ", cnt, "/", vocab_size)
        vectors = np.zeros((vocab_size, word_vectors.shape[-1]))
        save_i2w("word.txt", vocab)
        np.save("word_vector.npy", vectors)
        
    i2w = read_i2w("word.txt")
    word_vectors = np.load("word_vector.npy").astype(np.float32)
    print("Loading Vector Done!")
    w2i = from_i2w_get_w2i(i2w)
    dataset = conll2012(i2w, w2i)
    model = CorefScore(embedding_dim=word_vectors.shape[-1], vocab_size=len(word_vectors), weight=word_vectors).to(device)
    trainer = Trainer(model, dataset.train_dataset, dataset.valid_dataset, dataset.test_dataset, device)
    trainer.train(100, 1)