import gzip
import numpy as np
from annoy import AnnoyIndex
from utils import Utils


class AnnoySimilarity(object):

    def __init__(self, text_embedding_path, labels_file_path, annoy_index_path='annoy_index.ann', dimension=1024,
                 metric='angular', build_new_index=False):
        self.text_embedding_path = text_embedding_path
        self.qnode_to_label_dict = Utils.create_labels_dict(labels_file_path)
        self.qnode_to_sentence_dict = {}
        self.qnode_to_id_dict = {}  # cannot use qnode number as id, Annoy gets upset if the number is larger than a billion
        self.id_to_qnode_dict = {}
        self.annoy_index_path = annoy_index_path
        self.metric = metric
        self.index = None
        if build_new_index:
            self.build_index()
        else:
            try:
                self.index = AnnoyIndex(dimension, metric)
                self.index.load(annoy_index_path)
            except Exception as e:
                print(e)
                self.build_index()

    def build_index(self):
        if self.text_embedding_path.endswith(".gz"):
            f = gzip.open(self.text_embedding_path, mode='rt')
        else:
            f = open(self.text_embedding_path)
        id = 0
        for line in f:
            vals = line.split('\t')
            if vals[0].startswith('Q'):
                qnode = vals[0]
                # use the number part of Qnodes as id
                # id = int(qnode[1:])
                if vals[1] == 'embedding_sentence':
                    self.qnode_to_sentence_dict[qnode] = vals[2]
                if vals[1] == 'text_embedding':
                    x = vals[2].strip().split(',')
                    # x = [np.float32(r) for r in x]
                    x = [float(r) for r in x]
                    self.qnode_to_id_dict[qnode] = id
                    self.id_to_qnode_dict[id] = qnode

                    if self.index is None:
                        self.index = AnnoyIndex(len(x), self.metric)

                    self.index.add_item(id, x)
                    id += 1

        self.index.build(50)  # 50 trees, no idea how many trees should be built, more the better apparently
        self.index.save(self.annoy_index_path)

    def nearest_neighbors(self, query_qnode, k=5, debug=False):
        """([98, 99, 977, 979, 980, 981, 983, 986, 987, 989],
         [1.4142135381698608,
          1.4142135381698608,
          1.4142135381698608,
          1.4142135381698608,
          1.4142135381698608,
          1.4142135381698608,
          1.4142135381698608,
          1.4142135381698608,
          1.4142135381698608,
          1.4142135381698608])"""
        if query_qnode not in self.qnode_to_id_dict:
            print(f'{query_qnode} not found')
            return []

        results = []
        search_r = self.index.get_nns_by_item(self.qnode_to_id_dict[query_qnode], k, search_k=-1,
                                              include_distances=True)
        ids, distances = search_r[0], search_r[1]
        for i, r_id in enumerate(ids):
            qnode = self.id_to_qnode_dict[r_id]
            if query_qnode != qnode:
                _ = {
                    'sim': float(distances[i]),
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

#
# large_all = '/Users/amandeep/Github/maa-analysis/MAA_Datasets/v3.2.0/text_embeddings_all_large_v3.tsv.gz'
# large_2000 = '/Users/amandeep/Github/maa-analysis/MAA_Datasets/v3.2.0/embedding_files/2000.tsv'
# wiki_labels = '/Users/amandeep/Github/maa-analysis/MAA_Datasets/v3.2.0/qnodes-properties-labels-for-V3.2.0_KB.tsv'
#
# ai = AnnoySimilarity(large_all, wiki_labels, build_new_index=True, annoy_index_path='annot_index_all.ann')
