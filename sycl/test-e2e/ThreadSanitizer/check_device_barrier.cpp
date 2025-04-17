// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_tsan_flags -O2 -g -o %t1.out
// RUN: %{run} %t1.out 2>&1 | FileCheck %s
// UNSUPPORTED: cpu
// UNSUPPORTED-TRACKER: CMPLRLLVM-66827
#include "sycl/detail/core.hpp"
#include "sycl/ext/oneapi/experimental/root_group.hpp"
#include "sycl/group_barrier.hpp"
#include "sycl/usm.hpp"

const size_t N = 32;

struct TestKernel {
  int *m_array;
  TestKernel(int *array) : m_array(array) {}

  void operator()(sycl::nd_item<1> item) const {
    auto root = item.ext_oneapi_get_root_group();
    if (item.get_group_linear_id() == 0 && item.get_local_linear_id() == 0)
      m_array[0]++;

    sycl::group_barrier(root);

    if (item.get_group_linear_id() == 1 && item.get_local_linear_id() == 0)
      m_array[0]++;

    sycl::group_barrier(root);

    if (item.get_group_linear_id() == 2 && item.get_local_linear_id() == 0)
      m_array[0]++;

    sycl::group_barrier(root);

    if (item.get_group_linear_id() == 3 && item.get_local_linear_id() == 0)
      m_array[0]++;
  }

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::oneapi::experimental::use_root_sync};
  }
};

int main() {
  sycl::queue queue;
  int *array = sycl::malloc_shared<int>(1, queue);
  array[0] = 0;

  queue
      .submit([&](sycl::handler &h) {
        h.parallel_for<class Test>(sycl::nd_range<1>(N, N / 4),
                                   TestKernel(array));
      })
      .wait();
  // CHECK-NOT: WARNING: DeviceSanitizer: data race

  assert(array[0] == 4);

  return 0;
}
