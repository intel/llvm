// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "vec_geometric.hpp"

// Alias is needed to const-qualify vec without template args.
template <typename T, int NumElems>
using ConstVec = const sycl::vec<T, NumElems>;

int main() {
  run_test<ConstVec>();
  return 0;
}
