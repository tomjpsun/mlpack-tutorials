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
