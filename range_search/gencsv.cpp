#include <iostream>
#include <mlpack/core.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;

void debug_dump(std::string fname)
{
	mat tempDataset;
	data::Load(fname, tempDataset, true);
	cout << "Dump " << fname << ":" << endl;
	cout << tempDataset << endl;
}

int main()
{
	const int ROW=1000;
	const int COL=3;

	const string query_filename("query.csv");
	const string ref_filename("reference.csv");

	mat A = randu<mat>(ROW, COL);

	A.save( query_filename, csv_ascii );
	cout << "save " << query_filename
	     << "(" << A.n_rows << "," << A.n_cols << ")" << endl;

	// generate reference.csv
	mat B = randu<mat>(50, 3);

	B.save( ref_filename, csv_ascii );
	cout << "save " << ref_filename
	     << "(" << B.n_rows << "," << B.n_cols << ")" << endl;

	return 0;
}
