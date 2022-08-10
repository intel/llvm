// REQUIRES: gpu-intel-pvc, level_zero

// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env ZE_DEBUG=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Check that queues created on sub-sub-devices are going to specific compute
// engines:
// CHECK: [getZeQueue]: create queue ordinal = 0, index = 0 (round robin in [0, 0])
// CHECK: [getZeQueue]: create queue ordinal = 0, index = 1 (round robin in [1, 1])
// CHECK: [getZeQueue]: create queue ordinal = 0, index = 2 (round robin in [2, 2])
// CHECK: [getZeQueue]: create queue ordinal = 0, index = 3 (round robin in [3, 3])

#include "CL/sycl.hpp"
#include "CL/sycl/usm.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <math.h>
#include <unistd.h>

namespace sycl = cl::sycl;
using namespace std::chrono;

#define random_float() (rand() / double(RAND_MAX))
#define INTER_NUM (150)
#define KERNEL_NUM (2000)

void run(std::vector<sycl::queue> &queues) {
  auto N = 1024 * 16;
  size_t global_range = 1024;
  size_t local_range = 16;

  float *buffer_host0 = sycl::malloc_host<float>(N, queues[0]);
  float *buffer_device0 = sycl::malloc_device<float>(N, queues[0]);

  float *buffer_host1 = sycl::malloc_host<float>(N, queues[1]);
  float *buffer_device1 = sycl::malloc_device<float>(N, queues[1]);

  float *buffer_host2 = sycl::malloc_host<float>(N, queues[2]);
  float *buffer_device2 = sycl::malloc_device<float>(N, queues[2]);

  float *buffer_host3 = sycl::malloc_host<float>(N, queues[3]);
  float *buffer_device3 = sycl::malloc_device<float>(N, queues[3]);

  for (int i = 0; i < N; ++i) {
    buffer_host0[i] = static_cast<float>(random_float());
    buffer_host1[i] = static_cast<float>(random_float());
    buffer_host2[i] = static_cast<float>(random_float());
    buffer_host3[i] = static_cast<float>(random_float());
  }

  queues[0].memcpy(buffer_device0, buffer_host0, N * sizeof(float)).wait();
  queues[1].memcpy(buffer_device1, buffer_host1, N * sizeof(float)).wait();
  queues[2].memcpy(buffer_device2, buffer_host2, N * sizeof(float)).wait();
  queues[3].memcpy(buffer_device3, buffer_host3, N * sizeof(float)).wait();

  for (auto m = 0; m < INTER_NUM; ++m) {
    for (int k = 0; k < KERNEL_NUM; ++k) {
      auto event0 = queues[0].submit([&](sycl::handler &h) {
        h.parallel_for<class kernel0>(
            cl::sycl::nd_range<1>(cl::sycl::range<1>{global_range},
                                  cl::sycl::range<1>{local_range}),
            [=](cl::sycl::nd_item<1> item) {
              int i = item.get_global_linear_id();
              buffer_device0[i] = buffer_device0[i] + float(2.0);
            });
      });
      auto event1 = queues[1].submit([&](sycl::handler &h) {
        h.parallel_for<class kernel1>(
            cl::sycl::nd_range<1>(cl::sycl::range<1>{global_range},
                                  cl::sycl::range<1>{local_range}),
            [=](cl::sycl::nd_item<1> item) {
              int i = item.get_global_linear_id();
              buffer_device1[i] = buffer_device1[i] + float(2.0);
            });
      });
      auto event2 = queues[2].submit([&](sycl::handler &h) {
        h.parallel_for<class kernel2>(
            cl::sycl::nd_range<1>(cl::sycl::range<1>{global_range},
                                  cl::sycl::range<1>{local_range}),
            [=](cl::sycl::nd_item<1> item) {
              int i = item.get_global_linear_id();
              buffer_device2[i] = buffer_device2[i] + float(2.0);
            });
      });
      auto event3 = queues[3].submit([&](sycl::handler &h) {
        h.parallel_for<class kernel3>(
            cl::sycl::nd_range<1>(cl::sycl::range<1>{global_range},
                                  cl::sycl::range<1>{local_range}),
            [=](cl::sycl::nd_item<1> item) {
              int i = item.get_global_linear_id();
              buffer_device3[i] = buffer_device3[i] + float(2.0);
            });
      });
    }
    queues[0].wait();
    queues[1].wait();
    queues[2].wait();
    queues[3].wait();
  }

  free(buffer_host0, queues[0]);
  free(buffer_device0, queues[0]);

  free(buffer_host1, queues[1]);
  free(buffer_device1, queues[1]);

  free(buffer_host2, queues[2]);
  free(buffer_device2, queues[2]);

  free(buffer_host3, queues[3]);
  free(buffer_device3, queues[3]);

  std::cout << "[info] Finish all" << std::endl;
}

int main(void) {
  std::cout << "[info] this case is used to submit workloads to queues on "
               "subsub device"
            << std::endl;
  std::vector<sycl::device> subsub;

  auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  std::cout << "[info] device count = " << devices.size() << std::endl;

  // watch out device here
  auto subdevices =
      devices[1]
          .create_sub_devices<
              sycl::info::partition_property::partition_by_affinity_domain>(
              sycl::info::partition_affinity_domain::next_partitionable);
  std::cout << "[info] sub device size = " << subdevices.size() << std::endl;
  for (auto &subdev : subdevices) {
    auto subsubdevices = subdev.create_sub_devices<
        sycl::info::partition_property::partition_by_affinity_domain>(
        sycl::info::partition_affinity_domain::next_partitionable);
    std::cout << "[info] sub-sub device size = " << subsubdevices.size()
              << std::endl;
    for (auto &subsubdev : subsubdevices) {
      subsub.push_back(subsubdev);
    }
  }

  std::cout << "[info] all sub-sub devices count: " << subsub.size()
            << std::endl;
  std::cout << "[important] create 4 sycl queues on first 4 sub-sub devices"
            << std::endl;

  sycl::queue q0(subsub[0], {cl::sycl::property::queue::enable_profiling(),
                             cl::sycl::property::queue::in_order()});
  sycl::queue q1(subsub[1], {cl::sycl::property::queue::enable_profiling(),
                             cl::sycl::property::queue::in_order()});
  sycl::queue q2(subsub[2], {cl::sycl::property::queue::enable_profiling(),
                             cl::sycl::property::queue::in_order()});
  sycl::queue q3(subsub[4], {cl::sycl::property::queue::enable_profiling(),
                             cl::sycl::property::queue::in_order()});

  std::vector<sycl::queue> queues;

  queues.push_back(std::move(q0));
  queues.push_back(std::move(q1));
  queues.push_back(std::move(q2));
  queues.push_back(std::move(q3));

  run(queues);

  return 0;
}
