// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "vec_geometric.hpp"

int main() {
  run_test<sycl::vec>();
  return 0;
}
