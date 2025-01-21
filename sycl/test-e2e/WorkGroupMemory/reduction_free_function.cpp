// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: cuda
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/16004

#include "common_free_function.hpp"

// Basic usage reduction test using free function kernels.
// A global buffer is allocated using USM and it is passed to the kernel on the
// device. On the device, a work group memory buffer is allocated and each item
// copies the correspondng element of the global buffer to the corresponding
// element of the work group memory buffer using its lcoal index. The leader of
// every work-group, after waiting for every work-item to complete, then sums
// these values storing the result in another work group memory object. Finally,
// each work item then verifies that the sum of the work group memory elements
// equals the sum of the global buffer elements. This is repeated for several
// data types.

queue q;
context ctx = q.get_context();

constexpr size_t SIZE = 128;
constexpr size_t VEC_SIZE = 16;

template <typename T, typename... Ts> void test_marray() {
  if (!check_half_aspect<T>(q) || !check_double_aspect<T>(q))
    return;

  constexpr size_t WGSIZE = VEC_SIZE;
  T *buf = malloc_shared<T>(WGSIZE, q);
  assert(buf && "Shared USM allocation failed!");
  T expected = 0;
  for (int i = 0; i < WGSIZE; ++i) {
    buf[i] = T(i) / WGSIZE;
    expected = expected + buf[i];
  }
  nd_range ndr{{SIZE}, {WGSIZE}};
#ifndef __SYCL_DEVICE_ONLY__
  // Get the kernel object for the "mykernel" kernel.
  auto Bundle = get_kernel_bundle<sycl::bundle_state::executable>(ctx);
  kernel_id sum_id = ext::oneapi::experimental::get_kernel_id<sum_marray<T>>();
  kernel k_sum = Bundle.get_kernel(sum_id);
  q.submit([&](sycl::handler &cgh) {
     ext::oneapi::experimental::work_group_memory<marray<T, WGSIZE>> mem{cgh};
     ext::oneapi::experimental ::work_group_memory<T> result{cgh};
     cgh.set_args(mem, buf, result, expected);
     cgh.parallel_for(ndr, k_sum);
   }).wait();
#endif // __SYCL_DEVICE_ONLY
  free(buf, q);
  if constexpr (sizeof...(Ts))
    test_marray<Ts...>();
}

template <typename T, typename... Ts> void test_vec() {
  if (!check_half_aspect<T>(q) || !check_double_aspect<T>(q))
    return;

  constexpr size_t WGSIZE = VEC_SIZE;
  T *buf = malloc_shared<T>(WGSIZE, q);
  assert(buf && "Shared USM allocation failed!");
  T expected = 0;
  for (int i = 0; i < WGSIZE; ++i) {
    buf[i] = T(i) / WGSIZE;
    expected = expected + buf[i];
  }
  nd_range ndr{{SIZE}, {WGSIZE}};
#ifndef __SYCL_DEVICE_ONLY__
  // Get the kernel object for the "mykernel" kernel.
  auto Bundle = get_kernel_bundle<sycl::bundle_state::executable>(ctx);
  kernel_id sum_id = ext::oneapi::experimental::get_kernel_id<sum_vec<T>>();
  kernel k_sum = Bundle.get_kernel(sum_id);
  q.submit([&](sycl::handler &cgh) {
     ext::oneapi::experimental::work_group_memory<vec<T, WGSIZE>> mem{cgh};
     ext::oneapi::experimental ::work_group_memory<T> result{cgh};
     cgh.set_args(mem, buf, result, expected);
     cgh.parallel_for(ndr, k_sum);
   }).wait();
#endif // __SYCL_DEVICE_ONLY
  free(buf, q);
  if constexpr (sizeof...(Ts))
    test_vec<Ts...>();
}

template <typename T, typename... Ts>
void test(size_t SIZE, size_t WGSIZE, bool UseHelper) {
  if (!check_half_aspect<T>(q) || !check_double_aspect<T>(q))
    return;

  T *buf = malloc_shared<T>(WGSIZE, q);
  assert(buf && "Shared USM allocation failed!");
  T expected = 0;
  for (int i = 0; i < WGSIZE; ++i) {
    buf[i] = T(i);
    expected = expected + buf[i];
  }
  nd_range ndr{{SIZE}, {WGSIZE}};
  // The following ifndef is required due to a number of limitations of free
  // function kernels. See CMPLRLLVM-61498.
  // TODO: Remove it once these limitations are no longer there.
#ifndef __SYCL_DEVICE_ONLY__
  // Get the kernel object for the "mykernel" kernel.
  auto Bundle = get_kernel_bundle<sycl::bundle_state::executable>(ctx);
  kernel_id sum_id = ext::oneapi::experimental::get_kernel_id<sum<T>>();
  kernel k_sum = Bundle.get_kernel(sum_id);
  q.submit([&](sycl::handler &cgh) {
     ext::oneapi::experimental::work_group_memory<T[]> mem{WGSIZE, cgh};
     ext::oneapi::experimental ::work_group_memory<T> result{cgh};
     cgh.set_args(mem, buf, result, expected, WGSIZE, UseHelper);
     cgh.parallel_for(ndr, k_sum);
   }).wait();

#endif // __SYCL_DEVICE_ONLY
  free(buf, q);
  if constexpr (sizeof...(Ts))
    test<Ts...>(SIZE, WGSIZE, UseHelper);
}

int main() {
  test<int, uint16_t, half, double, float>(SIZE, SIZE, true /* UseHelper */);
  test<int, float, half>(SIZE, SIZE, false);
  test<int, double, char>(SIZE, SIZE / 2, false);
  test<int, bool, char>(SIZE, SIZE / 4, false);
  test_marray<float, double, half>();
  test_vec<float, double, half>();
  return 0;
}
