#include <mlpack/core.hpp>
#include <ensmallen.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/methods/ann/brnn.hpp>
#include <mlpack/core/data/binarize.hpp>
#include <mlpack/core/math/random.hpp>

void RNNModel();

using namespace mlpack;
using namespace mlpack::ann;
using namespace ens;
using namespace mlpack::math;

void GenerateNoisySines(arma::cube& data,
                        arma::mat& labels,
                        const size_t points,
                        const size_t sequences,
                        const double noise = 0.3)
{
	arma::colvec x =  arma::linspace<arma::colvec>(0, points - 1, points) /
		points * 20.0;
	arma::colvec y1 = arma::sin(x + arma::as_scalar(arma::randu(1)) * 3.0);
	arma::colvec y2 = arma::sin(x / 2.0 + arma::as_scalar(arma::randu(1)) * 3.0);

	data = arma::zeros(1 /* single dimension */, sequences * 2, points);
	labels = arma::zeros(2 /* 2 classes */, sequences * 2);

	for (size_t seq = 0; seq < sequences; seq++)
	{
		arma::vec sequence = arma::randu(points) * noise + y1 +
			arma::as_scalar(arma::randu(1) - 0.5) * noise;
		for (size_t i = 0; i < points; ++i)
			data(0, seq, i) = sequence[i];

		labels(0, seq) = 1;

		sequence = arma::randu(points) * noise + y2 +
			arma::as_scalar(arma::randu(1) - 0.5) * noise;
		for (size_t i = 0; i < points; ++i)
			data(0, sequences + seq, i) = sequence[i];

		labels(1, sequences + seq) = 1;
	}
}


void RNNModel()
{
	const size_t rho = 10;
	// Generate 12 (2 * 6) noisy sines. A single sine contains rho
	// points/features.

	// in mlpack/src/mlpack/tests/recurrent_network_test.cpp
	// the model train take 'cube' type
	arma::cube input;
	arma::mat labelsTemp;
	GenerateNoisySines(input, labelsTemp, rho, 6);
	//arma::mat labels = arma::zeros<arma::mat>(rho, labelsTemp.n_cols);
	arma::cube labels = arma::zeros<arma::cube>(1, labelsTemp.n_cols, rho);
	for (size_t i = 0; i < labelsTemp.n_cols; ++i)
	{
		const int value = arma::as_scalar(arma::find(
							  arma::max(labelsTemp.col(i)) == labelsTemp.col(i), 1)) + 1;
		labels.col(i).fill(value);
	}
	Add<> add(4);
	Linear<> lookup(1, 4);
	SigmoidLayer<> sigmoidLayer;
	Linear<> linear(4, 4);
	Recurrent<>* recurrent = new Recurrent<>(add, lookup, linear, sigmoidLayer, rho);
	RNN<> model(rho);
	model.Add<IdentityLayer<> >();
	model.Add(recurrent);
	model.Add<Linear<> >(4, 10);
	model.Add<LogSoftMax<> >();
	StandardSGD opt(0.1, 1, input.n_cols /* 1 epoch */, -100);
	model.Train(input, labels, opt);
}
