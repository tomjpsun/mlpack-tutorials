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

void gen_csv(std::string file_name, int row, int col, Distribution dist)
{
	mat A;
	const std::string target_file(file_name);
	if (dist == NORMAL)
		A = randn<mat>(row, col);
	else
		A = randu<mat>(row, col);
	A.save( target_file, csv_ascii );
        //debug_dump( data_file );
}

int main()
{
	// generate for the 1st tutorial
        gen_csv("points_1000.csv", 1000, 3, UNIFORM);

	// generate for the 2nd tutorial
	// reference dataset is following normal distribution
	gen_csv("reference_dataset.csv", 1000, 3, NORMAL);
	gen_csv("query_dataset.csv", 1000, 3, UNIFORM);
}
