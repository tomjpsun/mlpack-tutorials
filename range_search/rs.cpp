#include <mlpack/methods/range_search/range_search.hpp>
#include <vector>

using namespace mlpack::range;
// Our dataset matrix, which is column-major.
void demo_1()
{
	arma::mat data;
	RangeSearch<> a(data);

        // The vector-of-vector objects we will store output in.
	std::vector<std::vector<size_t> > resultingNeighbors;
	std::vector<std::vector<double> > resultingDistances;

        // The range we will use.
	math::Range r(0.0, 2.0); // [0.0, 2.0].
	a.Search(r, resultingNeighbors, resultingDistances);
