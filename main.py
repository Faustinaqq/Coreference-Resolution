from conll import conll2012
from train import Trainer
from model import CorefScore
import torch
import numpy as np
import os
from argparse import ArgumentParser
from utils import load_from_pretrained


def parse_args():
    
    parser = ArgumentParser(description='2 classification for coreference resolution')
    
    parser.add_argument('--sample_size', type=int, default=100, help='sample size')
    
    parser.add_argument('--lr', type=float, default=0.005, help='learning tate')
    
    parser.add_argument('--epochs', type=int, default=500, help='train epochs')
    
    parser.add_argument('--eval_epoch', type=int, default=20, help='every eval_epoch to evaluate on valdiation dataset')
    
    parser.add_argument("--log_path", type=str, default='./log/log.txt', help='log file path')
    
    parser.add_argument('--attention', type=int, default=1, help='use attention or not')
    
    parser.add_argument('--model', type=str, choices=['rnn', 'lstm'], default='lstm', help='use RNN or LSTM')
    
    parser.add_argument('--pretrained_vector_path', type=str, default=None, help='path to pretrained word vector file')

    return parser.parse_args()



if __name__ == '__main__':
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    if args.pretrained_vector_path is not None:
        i2w, w2i, word_vectors = load_from_pretrained(args.pretrained_vector_path)
        dataset = conll2012(i2w, w2i)
        model = CorefScore(embedding_dim=word_vectors.shape[-1], vocab_size=len(word_vectors), net=args.model, attention=args.attention, weight=word_vectors).to(device)
    else:
        dataset = conll2012()
        model = CorefScore(net=args.model, attention=args.attention).to(device)
        
    trainer = Trainer(model, dataset.train_dataset, dataset.valid_dataset, dataset.test_dataset, device, lr=args.lr, steps=args.sample_size, logfile=args.log_path)
    trainer.train(args.epochs, args.eval_epoch)