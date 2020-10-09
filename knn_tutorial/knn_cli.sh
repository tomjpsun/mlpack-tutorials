#!/bin/bash
make
./gencsv
# wait until .csv are saved
wait
echo 'One dataset, 5 nearest neighbors'
mlpack_knn -r points_1000.csv -n neighbors_out_1.csv -d distances_out_1.csv -k 5 -v
echo '----------'
echo 'Query and reference dataset, 10 nearest neighbors'
mlpack_knn -q query_dataset.csv -r reference_dataset.csv -n neighbors_out_2.csv -d distances_out_2.csv -k 10 -v
echo '----------'
echo 'One dataset, 3 nearest neighbors, leaf size of 15 points'
mlpack_knn -r reference_dataset.csv -n neighbors_out_3.csv -d distances_out_3.csv -k 3 -l 15 -v
echo '----------'
echo 'c++ 5 nearest neighbors on a single dataset'
make k_neighbor_search
echo 'use dataset self as query set, this is also called naive search'
./k_neighbor_search
echo '----------'
echo 'The rest of this html introduces NeighborSearch, SortPolicy'
echo ', MetricType, TreeType and TraverseType classes, all are '
echo 'basic templates used in MLPack.'
echo ''
echo 'TraverseType is related with Dual Tree Traversal Algorithms, '
echo 'for more reference: see'
echo '[Tree-Independent Dual-Tree Algorithms] by Ryan R. Curtin et al.'
