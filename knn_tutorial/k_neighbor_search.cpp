#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <string>

using namespace mlpack::neighbor;
using namespace std;
using namespace arma;
using namespace mlpack;

int main()
{
        // Our dataset matrix, which is column-major.
	string src("points_1000.csv");
	string neighbors("neighbors_out_4.csv");
	string distances("distances_out_4.csv");

        arma::mat data;
	data::Load(src, data, true);
	KNN a(data);
        // The matrices we will store output in.
	arma::Mat<size_t> resultingNeighbors;
	arma::mat resultingDistances;

	a.Search(5, resultingNeighbors, resultingDistances);

	std::cout << "save Neighbors to " << neighbors << std::endl;
	resultingNeighbors.save(neighbors, csv_ascii);

	std::cout << "save Distances to " << distances << std::endl;
	resultingDistances.save(distances, csv_ascii);
	return 0;
}
