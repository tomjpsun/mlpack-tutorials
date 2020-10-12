#include <iostream>
#include <cerrno>
#include <mlpack/core.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::regression;


void debug_dump(std::string data_file)
{
	mat tempDataset;
	data::Load(data_file, tempDataset, true);
	cout << tempDataset << endl;
}

void sync_execute_shell(string cmd)
{
	FILE * fp;
	char buffer[80];
	fp = popen(cmd.c_str(), "r");
	fgets(buffer, sizeof(buffer), fp);
	pclose(fp);
}

int main(int argc, char** argv)
{

	const string data_filename("dataset.csv");
	const string pred_filename("predict.csv");
	sync_execute_shell("./gencsv");
	arma::mat data; // The dataset itself.
	data::Load(data_filename, data, true);
	arma::rowvec responses(100); // The responses, one row for each row in data.
	for (int i=0; i<100; i++) responses[i]=2*i;
        // Regress.
	LinearRegression lr(data, responses);
        // Get the parameters, or coefficients.
	arma::vec parameters = lr.Parameters();
	cout << "beta = " << parameters << endl;
}
