kgtk text-embedding wikidata_maa_subgraph_sorted_2.tsv \
--model roberta-large-nli-mean-tokens \
--property-labels-file qnodes-properties-labels-for-V3.1.0_KB.tsv \
--parallel 1 --debug \
--property-value all \
--save-embedding-sentence > text_embeddings_all.tsv

kgtk text-embedding 1000.tsv \
--model roberta-large-nli-mean-tokens \
--property-labels-file qnodes-properties-labels-for-V3.1.0_KB.tsv \
--parallel 1 --debug \
--property-value all \
--save-embedding-sentence > text_embeddings_1000.tsv