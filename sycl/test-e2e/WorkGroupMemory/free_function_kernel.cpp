#include <cassert>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/work_group_memory.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

constexpr size_t SIZE = 1024;
size_t WGSIZE = SIZE;
queue q;
context ctx = q.get_context();

int sum_helper(
    sycl::ext::oneapi::experimental::work_group_memory<int[WGSIZE]> mem) {
  int ret = 0;
  for (int i = 0; i < WGSIZE; ++i) {
    ret += mem[i];
  }
  return ret;
}

template <bool UseHelper>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void sum(sycl::ext::oneapi::experimental::work_group_memory<int[WGSIZE]> mem,
         int *buf, int *Result) {
  const auto it = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  size_t local_id = it.get_local_id();
  mem[local_id] = buf[local_id];
  group_barrier(it.get_group());
  if (it.get_group().leader()) {
    if constexpr (UseHelper) {
      for (int i = 0; i < WGSIZE; ++i) {
        *Result += mem[i];
      }
    } else {
      *Result = sum_helper(mem);
    }
  }
}

template <bool UseHelper> void test() {
  int *buf = malloc_shared<int>(WGSIZE, q);
  assert(buf && "Shared allocation failed!");
  int expected = 0;
  for (int i = 0; i < WGSIZE; ++i) {
    buf[i] = i + (int)UseHelper;
    expected += buf[i];
  }
  int *result = malloc_shared<int>(1, q);
  assert(result && "Shared allocation failed!");
#ifndef __SYCL_DEVICE_ONLY__
  // Get the kernel object for the "mykernel" kernel.
  auto Bundle = get_kernel_bundle<sycl::bundle_state::executable>(ctx);
  kernel_id sum_id = ext::oneapi::experimental::get_kernel_id<sum<UseHelper>>();
  kernel k_sum = Bundle.get_kernel(sum_id);
  q.submit([&](sycl::handler &cgh) {
     ext::oneapi::experimental::work_group_memory<int[WGSIZE]> mem{cgh};
     cgh.set_args(mem, buf, result);
     nd_range ndr{{SIZE}, {WGSIZE}};
     cgh.parallel_for(ndr, k_sum);
   }).wait();
#endif
  assert(expected == *result);
  free(buf, q);
  free(result, q);
}

int main() {
  test<true /* UseHelper */>();
  test<false /* UseHelper */>();
  // Test with more than one work group
  WGSIZE = SIZE / 2;
  test<false>();
  WGSIZE = SIZE / 4;
  test<false>();
  return 0;
}
