#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <string>

using namespace mlpack::neighbor;
using namespace std;
using namespace arma;
using namespace mlpack;

arma::mat gen_lattice_points()
{
	arma::mat m(1000, 3);
	int idx = 0;
	for (int i=0; i< 10; i++)
		for (int j=0; j<10; j++)
			for (int k=0; k<10; k++) {
				m.row(idx++) =
					rowvec{ double(i),
						double(j),
						double(k) };
			}
	return m;
}

int main()
{
        // Our dataset matrix, which is column-major.

	string neighbors("neighbors_out_4.csv");
	string distances("distances_out_4.csv");

	// transpose data, the same as in mlpack::data::Load()
        arma::mat data = gen_lattice_points().t();
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
