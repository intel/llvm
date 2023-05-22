// REQUIRES: level_zero, gpu

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests the existence of the vendor test macro for graphs.

#include "graph_common.hpp"

int main() {
#ifndef SYCL_EXT_ONEAPI_GRAPH
  assert(false && "SYCL_EXT_ONEAPI_GRAPH vendor test macro not defined\n");
#else
  assert(SYCL_EXT_ONEAPI_GRAPH == 1);
#endif
}
