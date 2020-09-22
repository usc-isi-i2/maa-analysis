import pandas as pd
import gzip

f_path = '/Users/amandeep/Github/maa-analysis/MAA_Datasets/v3.2.0'

i_file = 'embedding_files/text_embeddings_all_large.tsv.gz'
o_file = 'qnode_to_vectors_with_labels_v2.0.tsv.gz'

all_labels_df = pd.read_csv(f'{f_path}/qnodes-properties-labels-for-V3.2.0_KB.tsv', sep='\t')
labels_dict = {}
for i, row in all_labels_df.iterrows():
    labels_dict[row['node1']] = row['label']

df = pd.read_csv(f'{f_path}/{i_file}', sep='\t')

df = df.loc[df['property'] == 'text_embedding']
reformat = []
missing_labels = 0
for i, row in df.iterrows():
    if row['node'] not in labels_dict:
        missing_labels += 1
    reformat.append(
        {
            'node': row['node'],
            'label': labels_dict.get(row['node'], ''),
            'text_embedding_vector': row['value']

        }
    )
new_df = pd.DataFrame(reformat)
print(missing_labels)
new_df.to_csv(f'{f_path}/{o_file}', sep='\t', index=False)
