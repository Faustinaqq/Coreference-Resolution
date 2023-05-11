from model import CorefScore
import torch
import torch.nn as nn
import torch.optim as optim
import random
from conll import Document
from tqdm import tqdm
import logging
import numpy as np
import networkx as nx
from utils import muc, ceaf, b_cubed, conll_coref_f1, FocalLoss

def get_logger(filename, verbosity=1, name=None):
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
    
    def __init__(self, model: CorefScore, train_corpus, val_corpus, test_corpus, device="cuda:0", lr=1e-05, steps=100, logfile='./log.txt'):
        
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
        
            if epoch % eval_epochs == 0:
                self.logger.info("Evaluate Validation...")
                muc_res, ceaf_res, b_cubed_res, conll_coref_f1_res = self.evaluate(self.val_corpus)
                self.logger.info("validation muc: {}, ceaf: {}, b_cubed: {}, coref_f1:{}".format(muc_res, ceaf_res, b_cubed_res, conll_coref_f1_res))
                
        self.logger.info("Evaluate Test...")
        muc_res, ceaf_res, b_cubed_res, conll_coref_f1_res = self.evaluate(self.test_corpus)
        self.logger.info("test muc: {}, ceaf: {}, b_cubed: {}, coref_f1:{}".format(muc_res, ceaf_res, b_cubed_res, conll_coref_f1_res))
        


    def train_epoch(self, epoch):
        self.model.train()
        
        batch = random.sample(self.test_corpus, self.steps)
        
        # batch = self.train_corpus
        
        epoch_loss, epoch_corefs, epoch_identified = [], [], []
        
        for doc in tqdm(batch):
            loss, corefs_found, corefs_chosen, corefs_num, non_corefs_found, non_corefs_chosen, corefs_gen_num = self.train_doc(doc)
            self.logger.info("Epoch: {}, Document: {} | Loss: {:.6f} | Coref Found: {}/{} | Corefs precision: {}/{}/{} | Non Corefs precision: {}/{}/{}".format(epoch, doc.id, loss, corefs_found, corefs_num, corefs_chosen, corefs_found, corefs_gen_num, non_corefs_chosen, non_corefs_found, corefs_gen_num))
            epoch_loss.append(loss)
            epoch_corefs.append(corefs_found / corefs_num if corefs_num != 0 else 0)
            epoch_identified.append(corefs_chosen / corefs_num if corefs_num != 0 else 0)
            
        self.scheduler.step()
        self.logger.info("Epoch: {} | Loss: {:.6f} | Coref recall: {:.6f} | Coref precision: {:.6f}".format(epoch, np.mean(epoch_loss), np.mean(epoch_corefs), np.mean(epoch_identified)))
        
    
    def train_doc(self, doc: Document):
        
        gold_corefs, _ = doc.get_span_labels()
        
        corefs_num = len(gold_corefs)
        
        self.optimizer.zero_grad()
        
        corefs_found, corefs_chosen, non_corefs_found, non_corefs_chosen = 0, 0, 0, 0
        spans, scores = self.model(doc)
        # print("scores: ", scores.size(), scores)
        gold_indexes = torch.zeros(scores.size()[:-1]).to(self.device)
        non_gold_indexes = torch.zeros(scores.size()[:-1]).to(self.device)
        loss_fun = FocalLoss(gamma=0)
        # loss_fun = nn.CrossEntropyLoss()
        # print("spans: ", len(spans))
        # print("scores: ", scores.size())
        # idx = 0
        corefs_gen_num = 0
        for idx, span in enumerate(spans):
            corefs_gen_num += len(span.candidate_antecedent_idx)
            golds = [i for i, link in enumerate(span.candidate_antecedent_idx) if link in gold_corefs]
            non_golds = list(set(range(len(span.candidate_antecedent_idx))).difference(golds))
            if golds:
                corefs_found += len(golds)
                gold_indexes[idx, golds] = 1
                
                found_corefs = torch.sum(scores[idx, golds, 1] > scores[idx, golds, 0]).detach()
                
                corefs_chosen += found_corefs.item()
                
            if non_golds:
                non_corefs_found += len(non_golds)
                non_gold_indexes[idx, non_golds] = 1
                non_corefs_chosen += torch.sum(scores[idx, non_golds, 0] > scores[idx, non_golds, 1]).detach()
                    
        
        # predict_true = found_corefs + not_corefs_found
        # eps = 1e-8
        # loss = - torch.sum(torch.log(torch.sum(torch.mul(scores, gold_indexes), dim=1).clamp_(eps, 1-eps)), dim=0)
        ## cross entropy loss
        
        scores = torch.cat([scores[gold_indexes.bool()].reshape(-1, 2), scores[non_gold_indexes.bool()].reshape(-1, 2)], dim=0)
        labels = torch.cat([torch.ones(size=(torch.sum(gold_indexes).long().item(),), device=scores.device), torch.zeros(size=(torch.sum(non_gold_indexes).long().item(),), device=scores.device)], dim=0).long()
        # scores = torch.cat([scores[i, : len(span.candidate_antecedent_idx), :] for i, span in enumerate(spans)], dim=0)
        # labels = torch.cat([gold_indexes[i, : len(span.candidate_antecedent_idx)] for i, span in enumerate(spans)], dim=0).long()
        # corefs_gen_num = len(labels)
        # print("labeles size: ", labels.size(), " sum(labeles): ", torch.sum(labels))
        # loss = nn.BCELoss()(scores, labels.float())
        loss = loss_fun(scores, labels)
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
        
        return loss.item(), corefs_found, corefs_chosen, corefs_num, non_corefs_found, non_corefs_chosen, corefs_gen_num
        
    
    def evaluate_doc(self, doc):
        _, gold_clusters = doc.get_span_labels()
        
        self.model.eval()
        
        spans, scores = self.model(doc)
        graph = nx.Graph()
        for i, span in enumerate(spans):
            
            found_corefs = [link for idx, link in enumerate(span.candidate_antecedent_idx) if scores[i, idx, 1] > scores[i, idx, 0]]
            # if any(found_corefs):
            graph.add_node((span.start, span.end))
            for link in found_corefs:
                graph.add_edge(link[1], link[0])

        clusters = list(nx.connected_components(graph))
        clusters = [list(cluster) for cluster in clusters]
        
        muc_res = muc(clusters, gold_clusters)
        ceaf_res = ceaf(clusters, gold_clusters) 
        b_cubed_res = b_cubed(clusters, gold_clusters)
        conll_coref_f1_res = (muc_res[-1] + ceaf_res[-1] + b_cubed_res[-1]) / 3
        
        return muc_res, ceaf_res, b_cubed_res, conll_coref_f1_res
        
    
    def evaluate(self, corpus):
        muc_list = []
        ceaf_list = []
        b_cubed_list = []
        conll_coref_f1_list = []
        for doc in corpus:
            muc_res, ceaf_res, b_cubed_res, conll_coref_f1_res = self.evaluate_doc(doc)
            self.logger.info("Evaluate Document: {} | MUC: {} | CEAF: {} | B-CUBED: {} | Corefs F1: {}".format(doc.id, muc_res, ceaf_res, b_cubed_res, conll_coref_f1_res))
            muc_list.append(muc_res)
            ceaf_list.append(ceaf_res)
            b_cubed_list.append(b_cubed_res)
            conll_coref_f1_list.append(conll_coref_f1_res)
        muc_res = np.array(muc_list).mean(axis=0)
        ceaf_res = np.array(ceaf_list).mean(axis=0)
        b_cubed_res = np.array(b_cubed_list).mean(axis=0)
        conll_coref_f1_res = np.array(conll_coref_f1_list).mean()
        return muc_res, ceaf_res, b_cubed_res, conll_coref_f1_res
    
    
        
    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath + '.pth')


    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state).to(self.device)
