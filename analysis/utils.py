import pandas as pd


class Utils(object):
    def __init__(self):
        pass

    @staticmethod
    def create_labels_dict(labels_file):
        all_labels_df = pd.read_csv(labels_file, sep='\t')
        labels_dict = {}
        for i, row in all_labels_df.iterrows():
            labels_dict[row['node1']] = row['label']
        return labels_dict
