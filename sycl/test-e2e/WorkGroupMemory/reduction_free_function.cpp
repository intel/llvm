// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: cuda
// UNSUPPORTED-INTENDED: The name mangling for free function kernels currently
// does not work with PTX.

// Usage of work group memory parameters in free function kernels is not yet
// implemented.
// TODO: Remove the following directive once
// https://github.com/intel/llvm/pull/15861 is merged.
// XFAIL: *
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/15927

#include <cassert>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

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

template <typename T>
void sum_helper(sycl::ext::oneapi::experimental::work_group_memory<T[]> mem,
                sycl::ext::oneapi::experimental::work_group_memory<T> ret,
                size_t WGSIZE) {
  for (int i = 0; i < WGSIZE; ++i) {
    ret = ret + mem[i];
  }
}

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void sum(sycl::ext::oneapi::experimental::work_group_memory<T[]> mem, T *buf,
         sycl::ext::oneapi::experimental::work_group_memory<T> result,
         T expected, size_t WGSIZE, bool UseHelper) {
  const auto it = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  size_t local_id = it.get_local_id();
  mem[local_id] = buf[local_id];
  group_barrier(it.get_group());
  if (it.get_group().leader()) {
    result = 0;
    if (!UseHelper) {
      for (int i = 0; i < WGSIZE; ++i) {
        result = result + mem[i];
      }
    } else {
      sum_helper(mem, result, WGSIZE);
    }
    assert(result == expected);
  }
}

// Explicit instantiations for the relevant data types.
// These are needed because free function kernel support is not fully
// implemented yet.
// TODO: Remove these once free function kernel support is fully there.
#define SUM(T)                                                                 \
  template void sum<T>(                                                        \
      sycl::ext::oneapi::experimental::work_group_memory<T[]> mem, T * buf,    \
      sycl::ext::oneapi::experimental::work_group_memory<T> result,            \
      T expected, size_t WGSIZE, bool UseHelper);

SUM(int)
SUM(uint16_t)
SUM(half)
SUM(double)
SUM(float)
SUM(char)
SUM(bool)

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void sum_marray(
    sycl::ext::oneapi::experimental::work_group_memory<sycl::marray<T, 16>> mem,
    T *buf, sycl::ext::oneapi::experimental::work_group_memory<T> result,
    T expected) {
  const auto it = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  size_t local_id = it.get_local_id();
  constexpr T tolerance = 0.0001;
  sycl::marray<T, 16> &data = mem;
  data[local_id] = buf[local_id];
  group_barrier(it.get_group());
  if (it.get_group().leader()) {
    result = 0;
    for (int i = 0; i < 16; ++i) {
      result = result + data[i];
    }
    assert((result - expected) * (result - expected) <= tolerance);
  }
}

// Explicit instantiations for the relevant data types.
#define SUM_MARRAY(T)                                                          \
  template void sum_marray<T>(                                                 \
      sycl::ext::oneapi::experimental::work_group_memory<sycl::marray<T, 16>>  \
          mem,                                                                 \
      T * buf, sycl::ext::oneapi::experimental::work_group_memory<T> result,   \
      T expected);

SUM_MARRAY(float);
SUM_MARRAY(double);
SUM_MARRAY(half);

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void sum_vec(
    sycl::ext::oneapi::experimental::work_group_memory<sycl::vec<T, 16>> mem,
    T *buf, sycl::ext::oneapi::experimental::work_group_memory<T> result,
    T expected) {
  const auto it = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  size_t local_id = it.get_local_id();
  constexpr T tolerance = 0.0001;
  sycl::vec<T, 16> &data = mem;
  data[local_id] = buf[local_id];
  group_barrier(it.get_group());
  if (it.get_group().leader()) {
    result = 0;
    for (int i = 0; i < 16; ++i) {
      result = result + data[i];
    }
    assert((result - expected) * (result - expected) <= tolerance);
  }
}

// Explicit instantiations for the relevant data types.
#define SUM_VEC(T)                                                             \
  template void sum_vec<T>(                                                    \
      sycl::ext::oneapi::experimental::work_group_memory<sycl::vec<T, 16>>     \
          mem,                                                                 \
      T * buf, sycl::ext::oneapi::experimental::work_group_memory<T> result,   \
      T expected);

SUM_VEC(float);
SUM_VEC(double);
SUM_VEC(half);

template <typename T, typename... Ts> void test_marray() {
  if (std::is_same_v<sycl::half, T> && !q.get_device().has(sycl::aspect::fp16))
    return;
  if (std::is_same_v<T, double> && !q.get_device().has(aspect::fp64))
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
  if (std::is_same_v<sycl::half, T> && !q.get_device().has(sycl::aspect::fp16))
    return;
  if (std::is_same_v<T, double> && !q.get_device().has(aspect::fp64))
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
  if (std::is_same_v<sycl::half, T> && !q.get_device().has(sycl::aspect::fp16))
    return;
  if (std::is_same_v<T, double> && !q.get_device().has(aspect::fp64))
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
