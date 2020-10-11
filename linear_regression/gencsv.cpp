#include <iostream>
#include <mlpack/core.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;

void debug_dump(std::string data_file)
{
	mat tempDataset;
	data::Load(data_file, tempDataset, true);
	cout << tempDataset << endl;
}

enum Distribution{
	UNIFORM,
	NORMAL
};

int main(int argc, char** argv)
{
	const int ROW=100;
	const int COL=2;
	bool uniform = false;
	const string data_filename("dataset.csv");
	const string pred_filename("predict.csv");

	std::mt19937 engine;  // Mersenne twister random number engine
	std::normal_distribution<double> distr(0, 0.1);

	//A.imbue( [&]() { return distr(engine); } );
	// add uniform distribution to column
	if (argc==2 && string(argv[0]).compare(string("--normal"))) {
		uniform = true;
	}

	mat A(ROW, COL);
	for (int idx = 0; idx < ROW; idx++) {
		A.row(idx) = rowvec{ double(idx),
				     (uniform)? idx + distr(engine): double(idx) };
	}
	colvec C = randu<colvec>( ROW );
	A.save( data_filename, csv_ascii );
	cout << "save " << data_filename
	     << "(" << A.n_rows << "," << A.n_cols << ")" << endl;

	// generate predict.csv
	mat P(3, 1);
	for (int idx = 0; idx < 3; idx++) {
		P.row(idx) = rowvec{ double(idx+2) };
	}
	P.save( pred_filename, csv_ascii );
	cout << "save " << pred_filename
	     << "(" << P.n_rows << "," << P.n_cols << ")" << endl;

        //debug_dump( data_file );
	return 0;
}
