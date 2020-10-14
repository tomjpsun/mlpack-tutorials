#!/bin/bash
# generate csv
make
./gencsv
echo 'One dataset, points with distance <= 0.076'
mlpack_range_search -r query.csv -n neighbors_out.csv -d distances_out.csv -U 0.076 -v
echo '----------'
echo 'Query and reference dataset, range [1.0, 1.2]'
mlpack_range_search -q query.csv -r reference.csv -n neighbors_out_1.csv -d distances_out_1.csv -L 1.0 -U 1.2 -v
echo '----------'
echo 'One dataset, range [0.7 0.8], leaf size of 15 points'
mlpack_range_search -r query.csv -n neighbors_out_2.csv -d distances_out_2.csv -L 0.7 -U 0.8 -l 15 -v
echo '----------'
