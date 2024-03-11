// RUN: %clangxx -fsycl -fsyntax-only %s -o %t.out
#include <sycl/sycl.hpp>

template <int N> void check_dim_member() {
  using namespace sycl;
  static_assert(range<N>::dimensions == N);
  static_assert(nd_range<N>::dimensions == N);
  static_assert(id<N>::dimensions == N);
  static_assert(item<N>::dimensions == N);
  static_assert(nd_item<N>::dimensions == N);
  static_assert(h_item<N>::dimensions == N);
  static_assert(group<N>::dimensions == N);
}

int main() {
  check_dim_member<1>();
  check_dim_member<2>();
  check_dim_member<3>();
}
