/*
   Ridge regression with lambda
   based on the equation in the middle of the following post:
   https://towardsdatascience.com/ridge-regression-for-better-usage-2f19b3a202db
*/

#include <iostream>
#include <cerrno>
#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::regression;

// beta values to generate response data,
// the regressor can 'solve' these coefficients
// from given response data

arma::rowvec beta{
	0.2,
	2.3,
	7.8
};

int main()
{
	arma::mat X; // The dataset itself.
	// generate points on x-y plane
	X.randu(2, 100);
	// responses are the z-values
	arma::rowvec responses(100);
	for (int i=0; i<100; i++)
		responses[i] = beta[0] +
			beta[1] * X(0,i) +
			beta[2] * X(1,i);
        // Regress.
	LinearRegression lr(X, responses);
        // Get the parameters, or coefficients.
	arma::vec parameters = lr.Parameters();
	cout << "beta = " << parameters << endl;
}
