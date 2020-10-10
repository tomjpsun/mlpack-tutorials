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

int main()
{
	const int ROW=5;
	const int COL=2;

	mat A(ROW, COL);
	for (int idx = 0; idx < ROW; idx++) {
		A.row(idx) = rowvec{ double(idx), double(idx) };
	}
	A.save( "dataset.csv", csv_ascii );

	// generate predict.csv
	mat P(3, 1);
	for (int idx = 0; idx < 3; idx++) {
		P.row(idx) = rowvec{ double(idx+2) };
	}
	P.save( "predict.csv", csv_ascii );
        //debug_dump( data_file );
}
