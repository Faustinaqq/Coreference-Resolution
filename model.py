import torch.nn as nn
import torch
from conll import Span, Document
# from boltons.iterutils import pairwise
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
import attr

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class DocumentEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, net='lstm', num_embeddings=5000, weight=None, num_layers=2):
        super(DocumentEncoder, self).__init__()
        
        if weight is not None:
            self.word_embed = nn.Embedding.from_pretrained(torch.from_numpy(weight))
            self.word_embed.weight.requires_grad = True
        else:
            self.word_embed = nn.Embedding(num_embeddings, embedding_dim)

        if net == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        # self.embed_dropout = nn.Dropout(0.5, inplace=True)
        # self.bilstm_dropout = nn.Dropout(0.3, inplace=True)
    
    def embed(self, sentence):
        return self.word_embed(torch.tensor(sentence).to(device))
    
    def forward(self, doc: Document):
        embeds = [self.embed(sen) for sen in doc.sentences]
        packed_embed = pack_sequence(embeds, enforce_sorted=False)
        
        outputs, _ = self.rnn(packed_embed)
        states, lens = pad_packed_sequence(outputs, batch_first=True)
        states = [states[i][:lens[i]] for i in range(len(lens))]
        
        return torch.cat(embeds, dim=0), torch.cat(states, dim=0)
    

class Speaker(nn.Module):
    
    def __init__(self, speaker_num=10, speaker_dim=20):
        super(Speaker, self).__init__()
        
        self.speaker_embed = nn.Sequential(
            nn.Embedding(speaker_num, speaker_dim, padding_idx=0),
            nn.Dropout(0.2)
        )
    
    def forward(self, speaker_labeles):
        return self.speaker_embed(speaker_labeles)
    

class Score(nn.Module):
    def __init__(self, embed_dim, hidden_dim=150):
        super(Score, self).__init__()
        
        self.score_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, x):
        return self.score_layer(x)
    
    
class Distance(nn.Module):
    
    def __init__(self, distance_dim=20):
        super(Distance, self).__init__()
        self.bins = torch.tensor([1, 2, 3, 4, 8, 16, 32, 64]).to(device)
        self.distance_layer = nn.Sequential(
            nn.Embedding(len(self.bins) + 1, distance_dim),
            nn.Dropout(0.2)
        )
        
    def forward(self, lengths):
        lengths = lengths.reshape(lengths.size(0), -1)
        distance_bins = torch.sum(lengths > self.bins, dim=-1).detach()
        return self.distance_layer(distance_bins)
    

def pad_and_stack(tensor_list):
    packed = pack_sequence(tensor_list, enforce_sorted=False)
    seq_unpacked, _ = pad_packed_sequence(packed, batch_first=True)
    return seq_unpacked
    

class SpanFeaure(nn.Module):
    def __init__(self, attn_dim, distance_dim, attention=True):
        super(SpanFeaure, self).__init__()
        
        # self.attention_layer = Score(attn_dim)
        self.attention = attention
        
        if attention:
            self.attention_layer = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=1, batch_first=False)
        
        self.span_length_layer = Distance(distance_dim)
        
    def forward(self, embeds, states, doc: Document, K=200):
        spans = doc.create_spans()
        
        if self.attention:
            attention_embeds, _ = self.attention_layer(states.unsqueeze(1), states.unsqueeze(1), embeds.unsqueeze(1)) # L * E
            attention_embeds = attention_embeds.squeeze(1)
        else:
            attention_embeds = states
            
        attention_embeds = torch.stack([torch.sum(attention_embeds[span.start: span.end + 1], dim=0) for span in spans])
        
        span_start_end_state = torch.stack([torch.cat([states[span.start], states[span.end]]) for span in spans])
        
        span_length_embed = self.span_length_layer(torch.tensor([len(span) for span in spans]).to(device))
        
        span_features = torch.cat([attention_embeds, span_start_end_state, span_length_embed], dim=1)
        
        spans = [attr.evolve(span, candidate_antecedent=spans[max(0, idx-K): idx]) for idx, span in enumerate(spans)]
        
        return spans, span_features
    

class PairwiseScore(nn.Module):
    def __init__(self, gij_dim, distance_dim, speaker_dim):
        super(PairwiseScore, self).__init__()
        self.distance_layer = Distance(distance_dim)
        
        self.speaker_layer = Speaker(3, speaker_dim)
        
        self.score = Score(gij_dim)
        
    def forward(self, spans, span_features):
        mention_ids, antecedent_ids, distances, speakers = zip(*[(span.idx, antecedent.idx, span.end - antecedent.start, self.get_pair_spearker(span, antecedent)) for span in spans for antecedent in span.candidate_antecedent])
                
        mention_ids = torch.tensor(mention_ids).long().to(device)
        
        antecedent_ids = torch.tensor(antecedent_ids).long().to(device)
        
        distances = torch.tensor(distances).to(device)
        
        speakers = torch.tensor(speakers).to(device)
        
        other_feature = torch.cat([self.distance_layer(distances), self.speaker_layer(speakers)], dim=1)
        
        gi = span_features[mention_ids]
        
        gj = span_features[antecedent_ids]
        
        pairs_feature = torch.cat([gi, gj, gi * gj, other_feature], dim=1)
        
        coref_scores = self.score(pairs_feature)
        
        spans = [attr.evolve(span, candidate_antecedent_idx=[((antecedent.start, antecedent.end), (span.start, span.end)) for antecedent in span.candidate_antecedent]) for span in spans]
        
        # probs = torch.sigmoid(coref_scores).squeeze(1)
        
        # print("probs: ", probs.size())
        
        antecedent_idx = [len(span.candidate_antecedent) for span in spans]
        
        split_scores = list(torch.split(coref_scores, antecedent_idx, dim=0))
        
        epsilon = torch.tensor([[0.0, 0.0]]).to(device).requires_grad_()
        
        split_scores = [torch.cat((prob, epsilon), dim=0) for prob in split_scores]
        
        probs = pad_and_stack(split_scores)
        
        return spans, probs
    
    def get_pair_spearker(self, span1: Span, span2: Span):
        if span1.speaker is None or span2.speaker is None:
            return torch.tensor(0).to(device)
        if span1.speaker == span2.speaker:
            return torch.tensor(1).to(device)
        return torch.tensor(2).to(device)


class CorefScore(nn.Module):
    
    def __init__(self, embedding_dim=300, hidden_dim=150, distance_dim=20, speaker_dim=20, vocab_size=100000, net='lstm', attention=True, weight=None):
        super(CorefScore, self).__init__()
        
        attn_dim = 2 * hidden_dim
        gi_dim = attn_dim * 2 + embedding_dim + distance_dim
        gij_dim = gi_dim * 3 + distance_dim + speaker_dim
        
        self.encoder = DocumentEncoder(embedding_dim, hidden_dim, net, vocab_size, weight)
        self.span_layer = SpanFeaure(attn_dim, distance_dim, attention)
        self.pairwise_score_layer = PairwiseScore(gij_dim, distance_dim, speaker_dim)
    
    def forward(self, doc: Document):
        embeds, states = self.encoder(doc)
        spans, span_features = self.span_layer(embeds, states, doc)
        spans, coref_probs = self.pairwise_score_layer(spans, span_features)
        return spans, coref_probs
    
        
        
        