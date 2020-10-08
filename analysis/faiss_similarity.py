import gzip
import faiss
import numpy as np
import pandas as pd
from compute_embedding_vectors import ComputeEmbeddings


class FAISSIndex(object):
    def __init__(self, text_embedding_path, labels_file_path):
        self.text_embedding_path = text_embedding_path
        self.index = None
        self.qnode_to_vector_dict = {}
        self.qnode_to_label_dict = self.create_labels_dict(labels_file_path)
        self.qnode_to_sentence_dict = {}
        self.ce = ComputeEmbeddings()

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

        ids = []
        vectors = []
        for line in f:
            vals = line.split('\t')
            if vals[0].startswith('Q'):
                qnode = vals[0]
                # use the number part of Qnodes as id
                id = int(qnode[1:])
                if vals[1] == 'embedding_sentence':
                    self.qnode_to_sentence_dict[qnode] = vals[2]
                if vals[1] == 'text_embedding':
                    x = vals[2].strip().split(',')
                    x = [np.float32(r) for r in x]
                    self.qnode_to_vector_dict[qnode] = np.array([x])
                    ids.append(id)
                    vectors.append(x)
                    index = faiss.IndexFlatL2(len(x))
                    if self.index is None:
                        self.index = faiss.IndexIDMap(index)

        self.index.add_with_ids(np.array(vectors), np.array(ids))

    def nearest_neighbors(self, query_qnode, k=5, debug=False):
        if query_qnode not in self.qnode_to_vector_dict:
            return None
        results = []
        d, i = self.index.search(self.qnode_to_vector_dict[query_qnode], k)
        for h, g in enumerate(i[0]):
            qnode = f'Q{g}'
            if query_qnode != qnode:
                _ = {
                    'sim': float(d[0][h]),
                    'qnode1': query_qnode,
                    'qnode1_label': self.qnode_to_label_dict[query_qnode],
                    'qnode2': qnode,
                    'qnode2_label': self.qnode_to_label_dict[qnode]
                }
                if debug:
                    _['qnode1_sentence'] = self.qnode_to_sentence_dict[query_qnode]
                    _['qnode2_sentence'] = self.qnode_to_sentence_dict[qnode]
                results.append(_)

        return results

    def nearest_neighbor_sentence(self, sentence, k=5, debug=False):
        sentence_vector = self.ce.get_vectors(sentence)
        results = []

        d, i = self.index.search(sentence_vector, k)
        for h, g in enumerate(i[0]):
            qnode = f'Q{g}'
            _ = {
                'sim': float(d[0][h]),
                'input_sentence': sentence,
                'qnode': qnode,
                'qnode_label': self.qnode_to_label_dict[qnode]
            }
            if debug:
                _['qnode_sentence'] = self.qnode_to_sentence_dict[qnode]

            results.append(_)
        return results

    def nearest_neighbors_batch(self, query_qnodes, k=5, debug=False):
        if not isinstance(query_qnodes, list):
            query_qnodes = [query_qnodes]

        query_vectors = [self.qnode_to_vector_dict[q][0] for q in query_qnodes]
        query_vectors = np.array(query_vectors)
        results = []
        distances, ids = self.index.search(query_vectors, k)
        for q_id_index, nns in enumerate(ids):
            qnode = query_qnodes[q_id_index]
            for i, nn in enumerate(nns):
                # qnode2 = self.id_to_qnode_dict[ids[q_id_index][i]]
                qnode2 = f'Q{ids[q_id_index][i]}'
                if qnode != qnode2:
                    _ = {
                        'sim': float(distances[q_id_index][i]),
                        'qnode1': qnode,
                        'qnode1_label': self.qnode_to_label_dict.get(qnode, ""),
                        'qnode2': qnode2,
                        'qnode2_label': self.qnode_to_label_dict.get(qnode2, "")
                    }
                    if debug:
                        _['qnode1_sentence'] = self.qnode_to_sentence_dict.get(qnode, "")
                        _['qnode2_sentence'] = self.qnode_to_sentence_dict.get(qnode2, "")
                    results.append(_)

        return results
