// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} -D__SYCL_USE_LIBSYCL8_VEC_IMPL=1 -o %t2.out
// RUN: %{run} %t2.out

#include "vec_geometric.hpp"

// Alias is needed to const-qualify vec without template args.
template <typename T, int NumElems>
using ConstVec = const sycl::vec<T, NumElems>;

int main() {
  run_test<ConstVec>();
  return 0;
}
