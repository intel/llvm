// RUN: %clangxx -fsycl -fsyntax-only %s
#include <sycl/sycl.hpp>
#include <type_traits>
using namespace std;
int main() {
  static_assert(is_same_v<sycl::vec<int, 1>::value_type, int>);
  static_assert(is_same_v<sycl::vec<float, 2>::value_type, float>);
  static_assert(is_same_v<sycl::vec<double, 3>::value_type, double>);
  static_assert(is_same_v<sycl::vec<sycl::half, 4>::value_type, sycl::half>);
  return 0;
}
