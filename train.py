from model import CorefScore
import torch
import torch.optim as optim
import random
from conll import Document
from tqdm import tqdm
import logging
import numpy as np
import networkx as nx
from utils import get_f1, FocalLoss, extract_mentions_to_clusters_from_clusters
from metrics import CorefEvaluator
import os

def get_logger(filename, verbosity=1, name=None):
    cur_dir = os.path.dirname(filename)
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class Trainer:
    
    def __init__(self, model: CorefScore, train_corpus, val_corpus, test_corpus, device="cuda:0", lr=1e-03, steps=100, logfile='./log.txt'):
        
        self.train_corpus = list(train_corpus)
        
        self.val_corpus = list(val_corpus)
        
        self.test_corpus = list(test_corpus)
        
        self.model = model.to(device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
         
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.001)
        
        self.logger = get_logger(logfile)
        
        self.device = device
        
        self.steps = steps
    
    def train(self, epochs, eval_epochs):
        for epoch in range(1, epochs + 1):
            self.train_epoch(epoch)
            max_f1 = 0
            if epoch % eval_epochs == 0:
                self.logger.info("Evaluate Validation...")
                muc_res, ceaf_res, b_cubed_res, conll_coref_f1_res = self.evaluate(self.val_corpus)
                self.logger.info("validation muc: {}, ceaf: {}, b_cubed: {}, coref_f1:{}".format(muc_res, ceaf_res, b_cubed_res, conll_coref_f1_res))
                if conll_coref_f1_res > max_f1:
                    self.logger.info("Epoch: {}, Save Model...".format(epoch))
                    max_f1 = conll_coref_f1_res
                    if not os.path.exists("./save"):
                        os.makedirs("./save")
                    self.save_model("./save/best_model.pth")
                
        self.logger.info("Evaluate Test...")
        muc_res, ceaf_res, b_cubed_res, conll_coref_f1_res = self.evaluate(self.test_corpus)
        self.logger.info("test muc: {}, ceaf: {}, b_cubed: {}, coref_f1:{}".format(muc_res, ceaf_res, b_cubed_res, conll_coref_f1_res))
        
        # load model and eval:
        self.logger.info("Load Model ad Evaluate...")
        self.load_model("best_model.pth")
        muc_res, ceaf_res, b_cubed_res, conll_coref_f1_res = self.evaluate(self.test_corpus)
        self.logger.info("test muc: {}, ceaf: {}, b_cubed: {}, coref_f1:{}".format(muc_res, ceaf_res, b_cubed_res, conll_coref_f1_res))
        
        
        


    def train_epoch(self, epoch):
        self.model.train()
        
        batch = random.sample(self.test_corpus, self.steps)
        
        epoch_loss, epoch_precision, epoch_recall, epoch_f1 = [], [], [], []
        
        for doc in tqdm(batch):
            loss, find_true_corefs_num, gold_corefs_num, predict_corefs_num, total_predict_num = self.train_doc(doc)
            precision = find_true_corefs_num / predict_corefs_num if predict_corefs_num > 0 else 0
            recall = find_true_corefs_num / gold_corefs_num if gold_corefs_num > 0 else 0
            f1 = get_f1(precision, recall)
            self.logger.info("Epoch: {}, Document: {} | Loss: {:.6f} | Total Predict: {} | Corefs Precision: {}/{}={:.6f} | Corefs Recall: {}/{}={:.6f} | F1 Score: {}".format(epoch, doc.id, loss, total_predict_num, find_true_corefs_num, predict_corefs_num, precision, find_true_corefs_num, gold_corefs_num, recall, f1))
            epoch_loss.append(loss)
            epoch_precision.append(precision)
            epoch_recall.append(recall)
            epoch_f1.append(f1)
            
        self.scheduler.step()
        self.logger.info("Epoch: {} | Loss: {:.6f} | Coref precision: {:.6f} | Coref recall: {:.6f} | Coref F1: {:.6f}".format(epoch, np.mean(epoch_loss), np.mean(epoch_precision), np.mean(epoch_recall), np.mean(epoch_f1)))
        
    
    def train_doc(self, doc: Document):
        
        gold_corefs, _ = doc.get_span_labels()
        
        gold_corefs_num = len(gold_corefs)
        
        self.optimizer.zero_grad()
        
        find_true_corefs_num, predict_corefs_num, total_predict_num = 0, 0, 0
        spans, scores = self.model(doc)
        gold_indexes = torch.zeros(scores.size()[:-1]).to(self.device)
        non_gold_indexes = torch.zeros(scores.size()[:-1]).to(self.device)
        for idx, span in enumerate(spans):
            predict_corefs_num += torch.sum(scores[idx, : len(span.candidate_antecedent_idx), 1] > scores[idx, : len(span.candidate_antecedent_idx), 0]).detach().item()
            total_predict_num += len(span.candidate_antecedent_idx)
            golds = [i for i, link in enumerate(span.candidate_antecedent_idx) if link in gold_corefs]
            non_golds = list(set(range(len(span.candidate_antecedent_idx))).difference(golds))
            if golds:
                gold_indexes[idx, golds] = 1
                find_true_corefs_num += torch.sum(scores[idx, golds, 1] > scores[idx, golds, 0]).detach().item()
                
            if non_golds:
                non_gold_indexes[idx, non_golds] = 1
        
        loss = 0
        if torch.sum(gold_indexes) > 0:
            loss += -torch.mean(torch.log(scores[gold_indexes.bool()].reshape(-1, 2)[:, 1]))
        if torch.sum(non_gold_indexes) > 0:
            loss += -torch.mean(torch.log(scores[non_gold_indexes.bool()].reshape(-1, 2)[:, 0]))
            
        loss.backward()
        param_num = 0
        param_norm = torch.tensor(0.0).to(scores.device)
        for _, param in self.model.named_parameters():
            if param.grad is not None:
                # self.logger.info("Parameter: {}, Gradient Norm: {}".format(name, torch.norm(param.grad)))
                param_num += 1
                param_norm += torch.norm(param.grad)
        self.logger.info("Average Gradient Norm: {}".format(torch.mean(param_norm).item()))
        self.optimizer.step()
        
        return loss.item(), find_true_corefs_num, gold_corefs_num, predict_corefs_num, total_predict_num
        
    
    def evaluate_doc(self, doc: Document):
        _, gold_clusters = doc.get_span_labels()
        
        self.model.eval()
        
        spans, scores = self.model(doc)
        graph = nx.Graph()
        for i, span in enumerate(spans):
            
            found_corefs = [link for idx, link in enumerate(span.candidate_antecedent_idx) if scores[i, idx, 1] > scores[i, idx, 0]]
            for link in found_corefs:
                graph.add_edge(link[1], link[0])
                
        clusters = list(nx.connected_components(graph))
        clusters = [sorted(list(cluster)) for cluster in clusters]
        gold_mentions_to_cluster = extract_mentions_to_clusters_from_clusters(gold_clusters)
        predict_mentions_to_cluster = extract_mentions_to_clusters_from_clusters(clusters)
        return clusters, gold_clusters, predict_mentions_to_cluster, gold_mentions_to_cluster
        
    def evaluate(self, corpus):
        coref_evaluater = CorefEvaluator()
        for doc in tqdm(corpus):
            clusters, gold_clusters, predict_mentions_to_cluster, gold_mentions_to_cluster = self.evaluate_doc(doc)
            coref_evaluater.update(clusters, gold_clusters, predict_mentions_to_cluster, gold_mentions_to_cluster)
        
        scores = coref_evaluater.get_scores()
        avg_f1 = (scores[0][-1] + scores[1][-1] + scores[2][-1]) / 3
            
        return scores[0], scores[1], scores[2], avg_f1
        
        
    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath)


    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)
