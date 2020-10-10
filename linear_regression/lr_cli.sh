#!/bin/bash
# generate csv
make
./gencsv

echo 'One file, generating the function coefficients'
mlpack_linear_regression --training_file dataset.csv -v -M lr.xml
echo '----------'
echo 'Compute model and predict at the same time'
mlpack_linear_regression --training_file dataset.csv \
			 --test_file predict.csv \
			 --output_predictions_file predictions1.csv -v
echo '----------'
echo 'Prediction using a precomputed model'
mlpack_linear_regression --input_model_file lr.xml \
			 --test_file predict.csv \
			 --output_predictions_file predictions2.csv -v
