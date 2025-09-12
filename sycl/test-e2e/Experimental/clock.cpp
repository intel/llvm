// UNSUPPORTED: cpu
// UNSUPPORTED-INTENDED: Bug in CPU RT. Waiting for the new version.

// REQUIRES: aspect-usm_shared_allocations
// REQUIRES: aspect-ext_oneapi_clock_sub_group || aspect-ext_oneapi_clock_work_group || aspect-ext_oneapi_clock_device
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/clock.hpp>
#include <sycl/usm.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

template <syclex::clock_scope scope, sycl::aspect aspect> void test() {
  sycl::queue q;
  if (!q.get_device().has(aspect))
    return;

  uint64_t *data = sycl::malloc_shared<uint64_t>(2, q);

  q.parallel_for(2, [=](sycl::id<1> idx) {
    if (idx == 0) {
      data[0] = syclex::clock<scope>();
      int count = 0;
      for (int i = 0; i < 1e6; ++i)
        count++;
      data[1] = syclex::clock<scope>();
    }
  });
  q.wait();

  assert(data[1] > data[0]);
  sycl::free(data, q);
}

int main() {
  test<syclex::clock_scope::sub_group,
       sycl::aspect::ext_oneapi_clock_sub_group>();
  test<syclex::clock_scope::work_group,
       sycl::aspect::ext_oneapi_clock_work_group>();
  test<syclex::clock_scope::device, sycl::aspect::ext_oneapi_clock_device>();

  return 0;
}
