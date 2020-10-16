#include <vector>
#include <type_traits>
#include <iostream>
#include <mlpack/core/math/range.hpp>
#include <mlpack/methods/range_search/range_search.hpp>
#include <mlpack/core.hpp>
#include <mlpack/core/tree/binary_space_tree.hpp>
#include <mlpack/core/tree/cover_tree.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::range;

template <typename T>
void dump(T& obj);

template <typename T>
void dump_vec(T& obj)
{
	for (typename T::iterator it=obj.begin(); it!=obj.end(); ++it)
		dump(*it);
	cout << endl;
}

template <typename T>
void dump(T& obj)
{
	if constexpr ((std::is_same_v<T, double>) || (std::is_same_v<T, size_t>))  {
			cout << obj << ","; }
	else {
		dump_vec(obj);
	}
}

// given reference & query set, lower & upper bound, make range search on the sets,
// if query set is empty, then make range search on reference set (with itself)
void range_search(arma::mat& ref, arma::mat& query, double lower_bound, double upper_bound)
{
	RangeSearch<> a(ref);

        // The vector-of-vector objects we will store output in.
	vector<vector<size_t> > resultingNeighbors;
	vector<vector<double> > resultingDistances;

	mlpack::math::Range range(lower_bound, upper_bound);

	if (query.is_empty()) {
        // The range search with 3 params.
		a.Search(range, resultingNeighbors, resultingDistances);
	} else {
	// The range search with 4 params.
		a.Search(query, range, resultingNeighbors, resultingDistances);
	}

	cout << "resultingNeighbors:" << endl;
	dump<vector<vector<size_t>>>(resultingNeighbors);

	cout << "resultingDistances:" << endl;
	dump<vector<vector<double>>>(resultingDistances);

}

int main()
{
	// query set
	mat Q(0,0);
	// reference set
	mat B = randu<mat>(3, 50);

	// Query size is empty - use range search with 3 params
	cout << "Distance less than 2.0 on a single dataset" << endl;
	range_search(B, Q, 0.0, 2.0);
	cout << "----------" << endl;

	cout << "Range [1.0, 2.0] on a query and reference dataset" << endl;
	// Query size is not empty - use range with 4 params
	Q = randu<mat>(3, 10);
	range_search(B, Q, 1.0, 2.0);
	cout << "----------" << endl;

	cout << "Naive (exhaustive) search for distance greater than 5.0 on one dataset" << endl;
	cout << "skipped, since its the same usage with previous examples" << endl;
	cout << "----------" << endl;

        // Construct a RangeSearch object with ball bounds,
	// tree type is derived from binary_space_tree.hpp
	RangeSearch<
		metric::EuclideanDistance,
		arma::mat,
		tree::BallTree
		> rangeSearchBallTree();

        // Construct a RangeSearch object with cover tree,
	// tree type is derived from cover_tree.hpp
	RangeSearch<
		metric::EuclideanDistance,
		arma::mat,
		tree::StandardCoverTree
		> rangeSearchCoverTree();

		return 0;
}
