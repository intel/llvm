// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// The name mangling for free function kernels currently does not work with PTX.
// UNSUPPORTED: cuda

#include <cassert>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

queue q;
context ctx = q.get_context();

void sum_helper(sycl::ext::oneapi::experimental::work_group_memory<int[]> mem,
                sycl::ext::oneapi::experimental::work_group_memory<int> ret,
                size_t WGSIZE) {
  for (int i = 0; i < WGSIZE; ++i) {
    ret += mem[i];
  }
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void sum(sycl::ext::oneapi::experimental::work_group_memory<int[]> mem,
         int *buf,
         sycl::ext::oneapi::experimental::work_group_memory<int> result,
         int expected, size_t WGSIZE, bool UseHelper) {
  const auto it = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  size_t local_id = it.get_local_id();
  mem[local_id] = buf[local_id];
  group_barrier(it.get_group());
  if (it.get_group().leader()) {
    result = 0;
    if (!UseHelper) {
      for (int i = 0; i < WGSIZE; ++i) {
        result += mem[i];
      }
    } else {
      sum_helper(mem, result, WGSIZE);
    }
    assert(result == expected);
  }
}

void test(size_t SIZE, size_t WGSIZE, bool UseHelper) {
  int *buf = malloc_shared<int>(WGSIZE, q);
  assert(buf && "Shared USM allocation failed!");
  int expected = 0;
  for (int i = 0; i < WGSIZE; ++i) {
    buf[i] = i;
    expected += buf[i];
  }

  // The following ifndef is required due to a number of limitations of free
  // function kernels. See CMPLRLLVM-61498.
  // TODO: Remove it once these limitations are no longer there.
#ifndef __SYCL_DEVICE_ONLY__

  // Get the kernel object for the "mykernel" kernel.
  auto Bundle = get_kernel_bundle<sycl::bundle_state::executable>(ctx);
  kernel_id sum_id = ext::oneapi::experimental::get_kernel_id<sum>();
  kernel k_sum = Bundle.get_kernel(sum_id);
  q.submit([&](sycl::handler &cgh) {
     ext::oneapi::experimental::work_group_memory<int[]> mem{WGSIZE, cgh};
     ext::oneapi::experimental ::work_group_memory<int> result{cgh};
     cgh.set_args(mem, buf, result, expected, WGSIZE, UseHelper);
     nd_range ndr{{SIZE}, {WGSIZE}};
     cgh.parallel_for(ndr, k_sum);
   }).wait();

#endif // __SYCL_DEVICE_ONLY
  free(buf, q);
}

int main() {
  constexpr size_t SIZE = 1024;
  test(SIZE, SIZE, true /* UseHelper */);
  test(SIZE, SIZE, false);
  // Test with more than one work group
  test(SIZE, SIZE / 2, false);
  test(SIZE, SIZE / 4, false);
  return 0;
}
