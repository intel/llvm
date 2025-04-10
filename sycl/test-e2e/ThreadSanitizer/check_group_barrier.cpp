// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: %{build} %device_tsan_flags -O2 -g -o %t1.out
// RUN: %{run} %t1.out 2>&1 | FileCheck %s
#include "sycl/detail/core.hpp"
#include "sycl/sub_group.hpp"
#include "sycl/usm.hpp"
#include <algorithm>

int main() {
  sycl::queue queue;
  constexpr size_t reqd_sg_size = 32;
  constexpr size_t N = reqd_sg_size * 4;

  auto device = queue.get_device();
  auto supported_sg_sizes =
      device.get_info<sycl::info::device::sub_group_sizes>();
  if (std::none_of(supported_sg_sizes.begin(), supported_sg_sizes.end(),
                   [](size_t size) { return size == reqd_sg_size; }))
    return 0;

  int *array = sycl::malloc_shared<int>(1, queue);
  array[0] = 0;

  queue
      .submit([&](sycl::handler &h) {
        h.parallel_for<class Test>(
            sycl::nd_range<1>(N, N),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_sub_group_size(reqd_sg_size)]] {
                  auto sg = item.get_sub_group();
                  if (item.get_group_linear_id() == 0 &&
                      sg.get_group_linear_id() == 0 &&
                      sg.get_local_linear_id() == 0)
                    array[0]++;

                  item.barrier();

                  if (item.get_group_linear_id() == 0 &&
                      sg.get_group_linear_id() == 1 &&
                      sg.get_local_linear_id() == 0)
                    array[0]++;

                  item.barrier();

                  if (item.get_group_linear_id() == 0 &&
                      sg.get_group_linear_id() == 2 &&
                      sg.get_local_linear_id() == 0)
                    array[0]++;

                  item.barrier();

                  if (item.get_group_linear_id() == 0 &&
                      sg.get_group_linear_id() == 3 &&
                      sg.get_local_linear_id() == 0)
                    array[0]++;
                });
      })
      .wait();
  // CHECK-NOT: WARNING: DeviceSanitizer: data race

  assert(array[0] == 4);

  return 0;
}
