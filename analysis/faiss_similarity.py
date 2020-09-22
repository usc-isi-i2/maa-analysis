import faiss
import numpy as np
import pandas as pd
import gzip


class FAISSIndex(object):
    def __init__(self, text_embedding_path, labels_file_path):
        self.text_embedding_path = text_embedding_path
        self.index = None
        self.qnode_to_id_dict = {}
        self.qnode_to_vector_dict = {}
        self.id_to_qnode_dict = {}
        self.qnode_to_label_dict = self.create_labels_dict(labels_file_path)
        self.qnode_to_sentence_dict = {}

    def create_labels_dict(self, labels_file):
        all_labels_df = pd.read_csv(labels_file, sep='\t')
        labels_dict = {}
        for i, row in all_labels_df.iterrows():
            labels_dict[row['node1']] = row['label']
        return labels_dict

    def build_index(self):
        if self.text_embedding_path.endswith(".gz"):
            f = gzip.open(self.text_embedding_path, mode='rt')
        else:
            f = open(self.text_embedding_path)
        id = 0
        ids = []
        vectors = []
        for line in f:
            vals = line.split('\t')
            if vals[0].startswith('Q'):
                if vals[1] == 'embedding_sentence':
                    self.qnode_to_sentence_dict[vals[0]] = vals[2]
                if vals[1] == 'text_embedding':
                    self.qnode_to_id_dict[vals[0]] = id
                    self.id_to_qnode_dict[id] = vals[0]

                    id += 1
                    x = vals[2].strip().split(',')
                    x = [np.float32(r) for r in x]
                    self.qnode_to_vector_dict[vals[0]] = np.array([x])
                    ids.append(self.qnode_to_id_dict[vals[0]])
                    vectors.append(x)
                    index = faiss.IndexFlatL2(len(x))
                    if self.index is None:
                        self.index = faiss.IndexIDMap(index)

        self.index.add_with_ids(np.array(vectors), np.array(ids))

    def nearest_neighbors(self, query_qnode, k=5):
        results = []
        d, i = self.index.search(self.qnode_to_vector_dict[query_qnode], int(k+1))
        for h, g in enumerate(i[0]):
            qnode = self.id_to_qnode_dict[g]
            if query_qnode != qnode:
                results.append({
                    'sim': float(d[0][h]),
                    'qnode1': query_qnode,
                    'qnode1_label': self.qnode_to_label_dict[query_qnode],
                    'qnode1_sentence': self.qnode_to_sentence_dict[query_qnode],
                    'qnode2': qnode,
                    'qnode2_label': self.qnode_to_label_dict[qnode],
                    'qnode2_sentence': self.qnode_to_sentence_dict[qnode]

                })

        return results
