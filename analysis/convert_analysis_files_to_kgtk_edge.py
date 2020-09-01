import pandas as pd
import json
import csv


# p_ath = '/Users/amandeep/Github/maa-analysis/MAA_Datasets/v3.2.0'
# edge_file_labels_descriptions = 'wikidata-20200803-all-edges-for-V3.2.0_KB-nodes-property-counts-with-labels-and-descriptions.tsv.gz'
# subgraph_sorted = 'wikidata_maa_subgraph_sorted_2.tsv'
# property_labels_id_file = 'property-labels-for-V3.2.0_KB_edge_id.tsv'
# qnodes_maa_labels_id_file = 'wikidata_maa_labels_edges_with_id.tsv'

class KGTKAnalysis(object):
    def __init__(self, p_ath):
        self.p_ath = p_ath

    def convert_node_labels_to_edge(self, edge_file_labels_descriptions):
        df = pd.read_csv(f'{self.p_ath}/{edge_file_labels_descriptions}', sep='\t')
        df = df.drop(columns=['label', 'node2']).fillna('')
        r = []
        for i, row in df.iterrows():
            r.append({
                'node1': row['node1'],
                'label': 'label',
                'node2': json.dumps(self.clean_string(row['node1;label'])),
                'id': f'{row["node1"]}-label-1'
            })
            r.append({
                'node1': row['node1'],
                'label': 'description',
                'node2': json.dumps(self.clean_string(row['node1;description'])),
                'id': f'{row["node1"]}-description-1'
            })
        df_r = pd.DataFrame(r)
        df_r.to_csv(f'{self.p_ath}/label-descriptions-for-V3.2.0_KB-nodes.tsv', sep='\t', index=False,
                    quoting=csv.QUOTE_NONE)

    @staticmethod
    def clean_string(i_str):
        if '@' in i_str:
            i = i_str.index('@')
            return i_str[1:i - 1]
        return i_str

    @staticmethod
    def clean_string_en(i_str):
        if '@' in i_str:
            vals = i_str.split('@')
            if vals[1] == 'en':
                return json.dumps(vals[0].replace("'", ""))
        return None

    def find_all_properties(self, subgraph_sorted):
        df = pd.read_csv(f'{self.p_ath}/{subgraph_sorted}', sep='\t').fillna('')
        properties = [x for x in list(df['label'].unique()) if x.startswith('P')]
        r = [{'node1': p} for p in properties]
        df_r = pd.DataFrame(r)
        df_r.to_csv(f'{self.p_ath}/properties-for-V3.2.0_KB-nodes.tsv', sep='\t', index=False)

    def property_english_labels_only(self, property_labels_id_file, qnodes_maa_labels_id_file):

        df_properties = pd.read_csv(f'{self.p_ath}/{property_labels_id_file}', sep='\t').fillna('')
        df_properties = df_properties.loc[df_properties['label'].isin(['label'])]
        df_properties['clean_node2'] = df_properties['node2'].map(lambda x: self.clean_string_en(x))
        df_properties = df_properties[df_properties.clean_node2.notnull()]
        df_properties.drop(columns=['node2', 'label', 'id'], inplace=True)
        df_properties.rename(columns={'clean_node2': 'label'}, inplace=True)

        df_qnodes = pd.read_csv(f'{self.p_ath}/{qnodes_maa_labels_id_file}', sep='\t').fillna('')
        df_qnodes = df_qnodes.loc[df_qnodes['label'].isin(['label'])]
        df_qnodes['clean_node2'] = df_qnodes['node2'].map(lambda x: self.clean_string_en(x))
        df_qnodes = df_qnodes[df_qnodes.clean_node2.notnull()]
        df_qnodes.drop(columns=['node2', 'label', 'id'], inplace=True)
        df_qnodes.rename(columns={'clean_node2': 'label'}, inplace=True)

        df = pd.concat([df_properties, df_qnodes])

        df.to_csv(f'{self.p_ath}/qnodes-properties-labels-for-V3.2.0_KB.tsv', sep='\t', index=False,
                  quoting=csv.QUOTE_NONE)
