from analysis.convert_analysis_files_to_kgtk_edge import KGTKAnalysis

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-f', '--folder', action='store', dest='folder_path',
                        help="folder where all files will be created")

    parser.add_argument('-p', '--properties-labels', action='store', dest='properties_label')
    parser.add_argument('-q', '--qnodes-labels', action='store', dest='qnodes_label')

    args = parser.parse_args()

    folder_path = args.get('folder_path')
    properties_label = args.get('properties_label')
    qnodes_label = args.get('qnodes_label')
    ka = KGTKAnalysis(folder_path)
    ka.property_english_labels_only(properties_label, qnodes_label)
