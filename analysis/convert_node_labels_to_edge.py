from analysis.convert_analysis_files_to_kgtk_edge import KGTKAnalysis

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-f', '--folder', action='store', dest='folder_path',
                        help="folder where all files will be created")

    parser.add_argument('-e', '--edge-file-labels', action='store', dest='edge_file_labels_descriptions')

    args = parser.parse_args()

    folder_path = args.get('folder_path')
    edge_file = args.get('edge_file_labels_descriptions')
    ka = KGTKAnalysis(folder_path)
    ka.convert_node_labels_to_edge(edge_file)
