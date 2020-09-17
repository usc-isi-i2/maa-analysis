import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarity(object):
    def __init__(self, embedding_file, labels_file):
        self.embedding_dict = self.create_embedding_dict(embedding_file)
        self.labels_dict = self.create_labels_dict(labels_file)

    def create_labels_dict(self, labels_file):
        all_labels_df = pd.read_csv(labels_file, sep='\t')
        labels_dict = {}
        for i, row in all_labels_df.iterrows():
            labels_dict[row['node1']] = row['label']
        return labels_dict

    def create_embedding_dict(self, embedding_file):
        f = open(embedding_file)
        _ = {}
        for line in f:
            vals = line.strip().split('\t')
            if vals[0].startswith('Q'):
                if vals[0] not in _:
                    _[vals[0]] = {}
                if vals[1] == 'text_embedding':
                    _[vals[0]]['t'] = [vals[2].split(',')]
                if vals[1] == 'embedding_sentence':
                    _[vals[0]]['s'] = vals[2]
        return _

    def compute_similarity(self, qnode1, qnode2):
        if qnode1 not in self.embedding_dict or qnode2 not in self.embedding_dict:
            return -1
        sim = cosine_similarity(self.embedding_dict[qnode1]['t'], self.embedding_dict[qnode2]['t'])[0][0]
        return {
            'sim': sim,
            'qnode1_label': self.labels_dict[qnode1],
            'qnode1_sentence': self.embedding_dict[qnode1]['s'],
            'qnode2_label': self.labels_dict[qnode2],
            'qnode2_sentence': self.embedding_dict[qnode2]['s']
        }
