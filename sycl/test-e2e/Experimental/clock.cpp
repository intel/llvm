// REQUIRES: ext_oneapi_clock

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/clock.hpp>
#include <sycl/usm.hpp>

int main() {
  sycl::queue q;
  uint64_t *data = sycl::malloc_shared<uint64_t>(3, q);

  q.single_task([=]() {
    uint64_t sg_clock_start = sycl::ext::oneapi::experimental::clock(
        sycl::ext::oneapi::experimental::clock_scope::sub_group);
    uint64_t wg_clock_start = sycl::ext::oneapi::experimental::clock(
        sycl::ext::oneapi::experimental::clock_scope::work_group);
    uint64_t dev_clock_start = sycl::ext::oneapi::experimental::clock(
        sycl::ext::oneapi::experimental::clock_scope::device);

    int count = 0;
    for (int i = 0; i < 1e6; ++i)
      count++;

    uint64_t sg_clock_end = sycl::ext::oneapi::experimental::clock(
        sycl::ext::oneapi::experimental::clock_scope::sub_group);
    uint64_t wg_clock_end = sycl::ext::oneapi::experimental::clock(
        sycl::ext::oneapi::experimental::clock_scope::work_group);
    uint64_t dev_clock_end = sycl::ext::oneapi::experimental::clock(
        sycl::ext::oneapi::experimental::clock_scope::device);
    data[0] = sg_clock_end - sg_clock_start;
    data[1] = wg_clock_end - wg_clock_start;
    data[2] = dev_clock_end - dev_clock_start;
  });
  q.wait();

  assert(data[0] > 0);
  assert(data[1] > 0);
  assert(data[2] > 0);

  return 0;
}
