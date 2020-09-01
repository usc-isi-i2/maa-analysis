from analysis.convert_analysis_files_to_kgtk_edge import KGTKAnalysis

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-f', '--folder', action='store', dest='folder_path',
                        help="folder where all files will be created")

    parser.add_argument('-s', '--subgraph-sorted', action='store', dest='subgraph_sorted')

    args = parser.parse_args()

    folder_path = args.get('folder_path')
    subgraph_sorted = args.get('subgraph_sorted')
    ka = KGTKAnalysis(folder_path)
    ka.find_all_properties(subgraph_sorted)
