import faiss
import numpy as np
import pandas as pd


class FAISSIndex(object):
    def __init__(self, text_embedding_path, labels_file_path):
        self.text_embedding_path = text_embedding_path
        self.index = None
        # self.index = faiss.IndexIDMap()
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
                    # vectors = np.array([x])
                    self.qnode_to_vector_dict[vals[0]] = np.array([x])
                    ids.append(self.qnode_to_id_dict[vals[0]])
                    vectors.append(x)
                    index = faiss.IndexFlatL2(1024)
                    if self.index is None:
                        # self.index = faiss.IndexFlatL2(len(x))
                        self.index = faiss.IndexIDMap(index)

        self.index.add_with_ids(np.array(vectors), np.array(ids))


fi = FAISSIndex('/Users/amandeep/Github/maa-analysis/MAA_Datasets/v3.2.0/text_embeddings_all.tsv',
                '/Users/amandeep/Github/maa-analysis/MAA_Datasets/v3.2.0/qnodes-properties-labels-for-V3.2.0_KB.tsv')
fi.build_index()

results = []

queries = ['Q37828', 'Q48989064', 'Q271997', 'Q48989064', 'Q159683', 'Q127956']
for qn in queries:
    d, i = fi.index.search(fi.qnode_to_vector_dict[qn], 5)
    for h, g in enumerate(i[0]):
        qnode = fi.id_to_qnode_dict[g]
        if qn != qnode:
            results.append({
                'sim': d[0][h],
                'qnode1': qn,
                'qnode1_label': fi.qnode_to_label_dict[qn],
                'qnode1_sentence': fi.qnode_to_sentence_dict[qn],
                'qnode2': qnode,
                'qnode2_label': fi.qnode_to_label_dict[qnode],
                'qnode2_sentence': fi.qnode_to_sentence_dict[qnode]

            })
            print(d[0][h], qnode, fi.qnode_to_sentence_dict[qnode])

df = pd.DataFrame(results)
df.to_csv('/tmp/faiss_sim.tsv', index=False, sep='\t')
