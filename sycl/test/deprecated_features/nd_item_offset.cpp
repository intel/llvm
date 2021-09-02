// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -D__SYCL_INTERNAL_API %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

using namespace std;
using cl::sycl::detail::Builder;

int main() {
  cl::sycl::nd_item<1> one_dim =
      Builder::createNDItem<1>(Builder::createItem<1, true>({4}, {2}, {0}),
                               Builder::createItem<1, false>({2}, {0}),
                               Builder::createGroup<1>({4}, {2}, {1}));
  assert((one_dim.get_offset() == cl::sycl::id<1>{0}));

  cl::sycl::nd_item<2> two_dim = Builder::createNDItem<2>(
      Builder::createItem<2, true>({4, 4}, {2, 2}, {0, 0}),
      Builder::createItem<2, false>({2, 2}, {0, 0}),
      Builder::createGroup<2>({4, 4}, {2, 2}, {1, 1}));
  assert((two_dim.get_offset() == cl::sycl::id<2>{0, 0}));

  cl::sycl::nd_item<3> three_dim = Builder::createNDItem<3>(
      Builder::createItem<3, true>({4, 4, 4}, {2, 2, 2}, {0, 0, 0}),
      Builder::createItem<3, false>({2, 2, 2}, {0, 0, 0}),
      Builder::createGroup<3>({4, 4, 4}, {2, 2, 2}, {1, 1, 1}));
  assert((three_dim.get_offset() == cl::sycl::id<3>{0, 0, 0}));

  return 0;
}
