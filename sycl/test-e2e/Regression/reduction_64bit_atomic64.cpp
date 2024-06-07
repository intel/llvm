// REQUIRES: aspect-atomic64
// RUN: %{build} -o %t.out
//
// RUN: %{run} %t.out
// Tests that a previously known case for reduction doesn't cause a requirement
// for atomic64.
// TODO: When aspect requirements are added to testing, this test could be set
//       to require that atomic64 is NOT supported, to limit how frequently the
//       test is run. However, it should work on devices that support atomic64
//       as well.

#include <sycl/detail/core.hpp>

#include <sycl/reduction.hpp>

#include <iostream>

using namespace sycl;

int main() {
  queue Q;

  long long *Out = malloc_device<long long>(1, Q);

  // Case 1: nd_range reduction with 64-bit integer and either sycl::plus,
  // sycl::minimum or sycl::maximum. group_reduce_and_atomic_cross_wg strategy
  // would normally be picked, but if the device does not support atomic64 that
  // strategy is invalid.
  Q.submit([&](handler &CGH) {
     auto Redu = reduction(Out, 0ll, sycl::plus<long long>{});
     CGH.parallel_for(nd_range<1>{range<1>{32}, range<1>{32}}, Redu,
                      [=](nd_item<1> It, auto &Sum) {
                        Sum.combine(It.get_global_linear_id());
                      });
   }).wait();

  // Case 2: nd_range reduction with 64-bit integer and either sycl::bit_or,
  // sycl::bit_xor, sycl::bit_and. local_mem_tree_and_atomic_cross_wg strategy
  // would normally be picked, but if the device does not support atomic64 that
  // strategy is invalid.
  Q.submit([&](handler &CGH) {
     auto Redu = reduction(Out, 0ll, sycl::bit_and<long long>{});
     CGH.parallel_for(nd_range<1>{range<1>{32}, range<1>{32}}, Redu,
                      [=](nd_item<1> It, auto &Sum) {
                        Sum.combine(It.get_global_linear_id());
                      });
   }).wait();

  // Case 3: range reduction with 64-bit integer and either sycl::bit_or,
  // sycl::bit_xor, sycl::bit_and. local_atomic_and_atomic_cross_wg strategy
  // would normally be picked, but if the device does not support atomic64 that
  // strategy is invalid.
  Q.submit([&](handler &CGH) {
     auto Redu = reduction(Out, 0ll, sycl::bit_and<long long>{});
     CGH.parallel_for(range<1>{32}, Redu,
                      [=](item<1> It, auto &Sum) { Sum.combine(It); });
   }).wait();
  sycl::free(Out, Q);
  return 0;
}
