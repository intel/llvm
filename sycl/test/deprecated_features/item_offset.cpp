// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -D__SYCL_INTERNAL_API %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

using cl::sycl::detail::Builder;

int main() {
  // one dimension item with offset
  cl::sycl::item<1, true> one_dim_with_offset =
      Builder::createItem<1, true>({4}, {2}, {1});
  assert(one_dim_with_offset.get_id() == cl::sycl::id<1>{2});
  assert(one_dim_with_offset.get_id(0) == 2);
  assert(one_dim_with_offset[0] == 2);
  assert(one_dim_with_offset.get_range() == cl::sycl::range<1>{4});
  assert(one_dim_with_offset.get_range(0) == 4);
  assert(one_dim_with_offset.get_offset() == cl::sycl::id<1>{1});
  assert(one_dim_with_offset.get_linear_id() == 1);

  // two dimension item with offset
  cl::sycl::item<2, true> two_dim_with_offset =
      Builder::createItem<2, true>({4, 8}, {2, 4}, {1, 1});
  assert((two_dim_with_offset.get_id() == cl::sycl::id<2>{2, 4}));
  assert(two_dim_with_offset.get_id(0) == 2);
  assert(two_dim_with_offset.get_id(1) == 4);
  assert(two_dim_with_offset[0] == 2);
  assert(two_dim_with_offset[1] == 4);
  assert((two_dim_with_offset.get_range() == cl::sycl::range<2>{4, 8}));
  assert((two_dim_with_offset.get_range(0) == 4));
  assert((two_dim_with_offset.get_range(1) == 8));
  assert((two_dim_with_offset.get_offset() == cl::sycl::id<2>{1, 1}));
  assert(two_dim_with_offset.get_linear_id() == 11);

  // three dimension item with offset
  cl::sycl::item<3, true> three_dim_with_offset =
      Builder::createItem<3, true>({4, 8, 16}, {2, 4, 8}, {1, 1, 1});
  assert((three_dim_with_offset.get_id() == cl::sycl::id<3>{2, 4, 8}));
  assert(three_dim_with_offset.get_id(0) == 2);
  assert(three_dim_with_offset.get_id(1) == 4);
  assert(three_dim_with_offset.get_id(2) == 8);
  assert(three_dim_with_offset[0] == 2);
  assert(three_dim_with_offset[1] == 4);
  assert(three_dim_with_offset[2] == 8);
  assert((three_dim_with_offset.get_range() == cl::sycl::range<3>{4, 8, 16}));
  assert((three_dim_with_offset.get_range(0) == 4));
  assert((three_dim_with_offset.get_range(1) == 8));
  assert((three_dim_with_offset.get_range(2) == 16));
  assert((three_dim_with_offset.get_offset() == cl::sycl::id<3>{1, 1, 1}));
  assert(three_dim_with_offset.get_linear_id() == 183);

  cl::sycl::item<1, false> one_dim = Builder::createItem<1, false>({4}, {2});
  cl::sycl::item<2, false> two_dim =
      Builder::createItem<2, false>({4, 8}, {2, 4});
  cl::sycl::item<3, false> three_dim =
      Builder::createItem<3, false>({4, 8, 16}, {2, 4, 8});
  // A conversion to item with offset
  cl::sycl::item<1, true> one_dim_transformed = one_dim;
  cl::sycl::item<1, true> one_dim_check =
      Builder::createItem<1, true>({4}, {2}, {0});
  assert(one_dim_transformed == one_dim_check);
  cl::sycl::item<2, true> two_dim_transformed = two_dim;
  cl::sycl::item<2, true> two_dim_check =
      Builder::createItem<2, true>({4, 8}, {2, 4}, {0, 0});
  assert(two_dim_transformed == two_dim_check);
  cl::sycl::item<3, true> three_dim_transformed = three_dim;
  cl::sycl::item<3, true> three_dim_check =
      Builder::createItem<3, true>({4, 8, 16}, {2, 4, 8}, {0, 0, 0});
  assert((three_dim_transformed == three_dim_check));

  return 0;
}
