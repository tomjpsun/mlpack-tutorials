#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/rnn.hpp>

void RNNModel();

using namespace mlpack;
using namespace mlpack::ann;


void GenerateNoisySines(arma::mat& data,
                        arma::mat& labels,
                        const size_t points,
                        const size_t sequences,
                        const double noise = 0.3)
{
	arma::colvec x =  arma::linspace<arma::Col<double>>(0,
							    points - 1, points) / points * 20.0;
	arma::colvec y1 = arma::sin(x + arma::as_scalar(arma::randu(1)) * 3.0);
	arma::colvec y2 = arma::sin(x / 2.0 + arma::as_scalar(arma::randu(1)) * 3.0);
	data = arma::zeros(points, sequences * 2);
	labels = arma::zeros(2, sequences * 2);
	for (size_t seq = 0; seq < sequences; seq++)
	{
		data.col(seq) = arma::randu(points) * noise + y1 +
			arma::as_scalar(arma::randu(1) - 0.5) * noise;
		labels(0, seq) = 1;
		data.col(sequences + seq) = arma::randu(points) * noise + y2 +
			arma::as_scalar(arma::randu(1) - 0.5) * noise;
		labels(1, sequences + seq) = 1;
	}
}


void RNNModel()
{
	const size_t rho = 10;
	// Generate 12 (2 * 6) noisy sines. A single sine contains rho
	// points/features.
	arma::mat input, labelsTemp;
	GenerateNoisySines(input, labelsTemp, rho, 6);
	arma::mat labels = arma::zeros<arma::mat>(rho, labelsTemp.n_cols);
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
	Recurrent<> recurrent(add, lookup, linear, sigmoidLayer, rho);
	RNN<> model(rho);
	model.Add<IdentityLayer<> >();
	model.Add(recurrent);
	model.Add<Linear<> >(4, 10);
	model.Add<LogSoftMax<> >();
	StandardSGD opt(0.1, 1, input.n_cols /* 1 epoch */, -100);
	model.Train(input, labels, opt);
}
