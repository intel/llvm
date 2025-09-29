// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "reduction_internal.hpp"

int main() {
  queue q;
  RedStorage Storage(q);

  testRange(Storage, nd_range<1>{range<1>{7}, range<1>{7}});
  testRange(Storage, nd_range<1>{range<1>{3 * 3}, range<1>{3}});

  // TODO: Strategies historically adopted from sycl::range implementation only
  // support 1-Dim case.
  //
  // testRange(Storage, nd_range<2>{range<2>{7, 3}, range<2> {7, 3}});
  // testRange(Storage, nd_range<2>{range<2>{14, 9}, range<2> {7, 3}});

  return 0;
}
