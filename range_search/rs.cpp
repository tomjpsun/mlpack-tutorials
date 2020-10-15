#include <vector>
#include <type_traits>
#include <iostream>
#include <mlpack/core/math/range.hpp>
#include <mlpack/methods/range_search/range_search.hpp>
#include <mlpack/core.hpp>
#include <type_traits>


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

// Our dataset matrix, which is column-major.
void demo_1(arma::mat& data)
{
	RangeSearch<> a(data);

        // The vector-of-vector objects we will store output in.
	vector<std::vector<size_t> > resultingNeighbors;
	vector<std::vector<double> > resultingDistances;

        // The range we will use.
	mlpack::math::Range r(0.0, 2.0); // [0.0, 2.0].
	a.Search(r, resultingNeighbors, resultingDistances);

	cout << "resultingNeighbors:" << endl;
	dump<vector<std::vector<size_t>>>(resultingNeighbors);

	cout << "resultingDistances:" << endl;
	dump<vector<std::vector<double>>>(resultingDistances);

}

int main()
{
	mat B = randu<mat>(3, 50);
	demo_1(B);
	return 0;
}
