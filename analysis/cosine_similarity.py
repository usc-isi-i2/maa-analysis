import json
from sklearn.metrics.pairwise import cosine_similarity

# f_path = '/Users/amandeep/Github/maa-analysis/MAA_Datasets/text_embeddings_1000.tsv'
f_path = '/Users/amandeep/Github/maa-analysis/MAA_Datasets/text_embeddings_all.tsv'
o_f = open('similarity_all.jl', 'w')
embeddings = {}
f = open(f_path)
for line in f:
    if not line.startswith('node'):
        node1, label, embedding = line.strip().split('\t')
        if label == 'text_embedding':
            embeddings[node1] = embedding.split(',')


def if_exists(k1, k2, d):
    if f'{k1}_{k2}' not in d and f'{k2}_{k1}' not in d:
        return False
    return True


# calculate cosine similarity
c_similarity = {}
for k1 in embeddings:
    print(k1)
    for k2 in embeddings:
        if k1 != k2:
            sim = cosine_similarity([embeddings[k1]], [embeddings[k2]])[0][0]
            # if sim >= 0.9:
            #     print(k1, k2, sim)

            if not if_exists(k1, k2, c_similarity):
                c_similarity[f'{k1}_{k2}'] = sim
                o_f.write(json.dumps({f'{k1}_{k2}': sim}))
                o_f.write('\n')

c_similarity = {k: v for k, v in sorted(c_similarity.items(), key=lambda item: item[1])}
open('similarity.json', 'w').write(json.dumps(c_similarity, indent=2))
