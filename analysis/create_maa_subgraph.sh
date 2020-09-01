f_path='/Users/amandeep/Github/maa-analysis/MAA_Datasets'
version='v3.2.0'
v_path="$f_path/$version"

kgtk ifexists -i "$f_path/wikidata-20200803-all-edges.tsv.gz" --input-keys node1 \
  --filter-on "$v_path/V3.2.0_KB-nodes.tsv" --filter-keys id --mode NONE >$v_path/wikidata_maa_edges.tsv

kgtk ifexists -i $f_path/wikidata-20200803-all-nodes.tsv.gz --input-keys id --filter-on $v_path/V3.2.0_KB-nodes.tsv --filter-keys id --mode NONE \
  > $v_path/wikidata_maa_labels.tsv
kgtk normalize-nodes -i $v_path/wikidata_maa_labels.tsv -o $v_path/wikidata_maa_labels_edges.tsv
kgtk add-id -i $v_path/wikidata_maa_labels_edges.tsv -o $v_path/wikidata_maa_labels_edges_with_id.tsv --id-style node1-label-num

# Step 2
# Download wikidata-20200803-all-edges-for-V3.2.0_KB-nodes-property-counts-with-labels-and-descriptions.tsv.gz from analysis folder in google drive
# produced by Craig
# And run this function convert_node_labels_to_edge() in analysis/convert_analysis_files_to_kgtk_edge.py
python convert_node_labels_to_edge.py -f $v_path -e '$v_path/wikidata-20200803-all-edges-for-V3.2.0_KB-nodes-property-counts-with-labels-and-descriptions.tsv.gz'

# step 3 run this below command
kgtk cat $v_path/wikidata_maa_edges.tsv $v_path/label-descriptions-for-V3.2.0_KB-nodes.tsv >$v_path/wikidata_maa_subgraph_2.tsv

# step 4 sort the file
kgtk sort2 -i $v_path/wikidata_maa_subgraph_2.tsv -o $v_path/wikidata_maa_subgraph_sorted_2.tsv

# step 5
python find_all_properties.py -f $v_path -s $v_path/wikidata_maa_subgraph_sorted_2.tsv

# Step 6
kgtk ifexists -i $f_path/wikidata-20200803-all-nodes.tsv.gz --input-keys id \
  --filter-on $v_path/properties-for-V3.2.0_KB-nodes.tsv --filter-keys node1 --mode NONE >$v_path/property-labels-for-V3.2.0_KB.tsv

# step 7
kgtk normalize-nodes -i $v_path/property-labels-for-V3.2.0_KB.tsv -o $v_path/property-labels-for-V3.2.0_KB_edge.tsv

# step 8
kgtk add-id -i $v_path/property-labels-for-V3.2.0_KB_edge.tsv -o $v_path/property-labels-for-V3.2.0_KB_edge_id.tsv --id-style node1-label-num

# step 9
python create_labels_qnodes_properties.py -f $v_path -p $v_path/property-labels-for-V3.2.0_KB_edge_id.tsv -q $v_path/wikidata_maa_labels_edges_with_id.tsv

# step 10 run text embeddings
kgtk text-embedding $v_path/wikidata_maa_subgraph_sorted_2.tsv \
  --model roberta-large-nli-mean-tokens \
  --property-labels-file $v_path/qnodes-properties-labels-for-V3.2.0_KB.tsv \
  --parallel 1 --debug \
  --property-value all \
  --save-embedding-sentence > $v_path/text_embeddings_all.tsv
