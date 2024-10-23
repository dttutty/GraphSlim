# method : 'variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC','affinity_GS', 'kron', 'vng', 'clustering', 'averaging','gcond', 'doscond', 'gcondx', 'doscondx', 'sfgc', 'msgc', 'disco', 'sgdd', 'gcsntk', 'geom','cadm','cent_d', 'cent_p', 'kcenter', 'herding', 'random','random_edge'

# methods=('variation_neighborhoods' 'variation_edges' 'variation_cliques' 'heavy_edge' 'algebraic_JC' 'affinity_GS' 'kron' 'vng' 'clustering' 'averaging' 'gcond' 'doscond' 'gcondx' 'doscondx' 'sfgc' 'msgc' 'disco' 'sgdd' 'gcsntk' 'geom' 'cadm' 'cent_d' 'cent_p' 'kcenter' 'herding' 'random' 'random_edge')

# for method in "${methods[@]}"; do
#     python train_all.py --dataset citeseer --method "$method" > "logs/citeseer/log_$method.txt" 2>&1
# done


standard_names=('flickr' 'reddit' 'dblp' 'cora_ml' 'physics' 'cs' 'cora' 'citeseer' 'pubmed' 'photo' 'computers' 'ogbn-products' 'ogbn-proteins' 'ogbn-papers100m' 'ogbn-arxiv' 'yelp' 'amazon')


for standard_name in "${standard_names[@]}"; do
    python train_all.py --dataset "$standard_name" 
done
