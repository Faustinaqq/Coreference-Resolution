import datasets
from boltons.iterutils import windowed
import attr
from collections import defaultdict
from itertools import combinations

class conll2012(object):
    def __init__(self, vocab=None, vocab_map=None):
        super(conll2012, self).__init__()
        self.save_path = '/home/fqq/data/conll'
        self.dataset = datasets.load_dataset('conll2012_ontonotesv5', 'chinese_v4')
            
        if vocab is None:
            self.vocab, self.vocab_map = self.init_vocab()
        else:
            self.vocab = vocab
            self.vocab_map = vocab_map
        
        self.train_dataset = self.process_data(self.dataset['train'])
        self.valid_dataset = self.process_data(self.dataset['validation'])
        self.test_dataset = self.process_data(self.dataset['test'])
        
    def init_vocab(self):
        vocab = set()
        for doc in self.dataset['train']:
            for sentence in doc["sentences"]:
                for token in sentence['words']:
                    vocab.update(token)
        vocab = list(vocab)
        vocab_map = {}
        for i, token in enumerate(vocab):
            vocab_map[token] = i + 1
        return vocab, vocab_map
    
    def get_vocab(self):
        return self.vocab, len(self.vocab)
            
    def process_data(self, dataset):
        doc_list = []
        for doc in dataset:
            sentence_list = []
            span_list = []
            speaker_list = []
            part_id_list = []
            pre_token_num_list = []
            token_num = 0
            has_span = False
            for sentence in doc["sentences"]:
                sen = [self.vocab_map[word] if word in self.vocab else 0 for word in sentence['words']]
                sentence_list.append(sen)
                span_list.append(sentence['coref_spans'])
                speaker_list.append('speaker')
                part_id_list.append('part_id')
                pre_token_num_list.append(token_num)
                token_num += len(sen)
                if not has_span and len(sentence['coref_spans']) > 0:
                    has_span = True
            if has_span:
                doc_list.append(Document({"id": doc['document_id'], "sentences": sentence_list, "spans": span_list, "part_ids": part_id_list, "speakers": speaker_list, "pre_token_nums": pre_token_num_list}))
        return doc_list


class Document():
    def __init__(self, doc: dict):
        self.id = doc['id']
        self.sentences = doc['sentences']
        self.spans = doc['spans']
        self.part_ids = doc['part_ids']
        self.speakers = doc['speakers']
        self.pre_token_num = doc['pre_token_nums']
        self.token_len = self.pre_token_num[-1] + len(self.sentences[-1])
        
    def __getitem__(self, idx):
        return (self.sentences[idx], self.spans[idx], self.part_ids[idx], self.speakers[idx], self.pre_token_num[idx])
    
    def __repr__(self):
        return 'Document containing %d sentences' % len(self.sentences)
    
    def __len__(self):
        return self.token_len
    
    def get_span_labels(self):
        links = defaultdict(list)
        gold_corefs = []
        for i, sen_span in enumerate(self.spans):
            for span in sen_span:
                tmp_span = (span[1] + self.pre_token_num[i], span[2] + self.pre_token_num[i])
                links[span[0]].append(tmp_span)
        link_cluster = []
        for link in links.values():
            link = sorted(link)
            if len(link) > 1:
                gold_corefs.extend([coref for coref in combinations(link, 2)])
                link_cluster.append(link)
        gold_corefs = sorted(gold_corefs)
        return gold_corefs, link_cluster
    
    
    def create_spans(self):
        created_spans = []
        idx = 0
        for i, sen_spans in enumerate(self.spans):
            new_sent_spans = [Span(start=span[1] + self.pre_token_num[i], end=span[2] + self.pre_token_num[i], idx=idx + k, speaker=self.speakers[i], part_id=self.part_ids[i]) for k, span in enumerate(sen_spans)]
            idx += len(sen_spans)
            created_spans.extend(new_sent_spans)
        return created_spans


@attr.s(frozen=True, repr=False)
class Span:
    start = attr.ib()
    end = attr.ib()
    idx = attr.ib()
    speaker = attr.ib()
    part_id = attr.ib()
    
    candidate_antecedent = attr.ib(default=None)
    candidate_antecedent_idx = attr.ib(default=None)
    
    def __len__(self):
        return self.end - self.start + 1
    
    def __repr__(self) -> str:
        return "span information %d words" % (self.__len__())
        
        