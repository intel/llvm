// REQUIRES:  arch-intel_gpu_pvc, level_zero

// UNSUPPORTED: gpu-intel-pvc-1T

// RUN: %{build} %level_zero_options -o %t.out

// TODO - at this time PVC 1T systems aren't correctly supporting affinity
// subdomain partitioning so this test is marked as UNSUPPORTED on those
// systems.

// TODO - at this time ZEX_NUMBER_OF_CCS is not working with FLAT hierachy,
// which is the new default on PVC.  Once it is supported, we'll test on both.
// In the interim, these are the environment vars that must be used in
// conjunction with ZEX_NUMBER_OF_CCS
// DEFINE: %{setup_env} = env ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE ZE_AFFINITY_MASK=0 ZEX_NUMBER_OF_CCS=0:4

// RUN: %{setup_env} env UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s
// RUN: %{setup_env} %{run} %t.out

// Check that queues created on sub-sub-devices are going to specific compute
// engines:
// CHECK: [getZeQueue]: create queue ordinal = 0, index = 0 (round robin in [0, 0])
// CHECK: [getZeQueue]: create queue ordinal = 0, index = 1 (round robin in [1, 1])
// CHECK: [getZeQueue]: create queue ordinal = 0, index = 2 (round robin in [2, 2])
// CHECK: [getZeQueue]: create queue ordinal = 0, index = 3 (round robin in [3, 3])

#include <chrono>
#include <cmath>
#include <iostream>
#include <math.h>
#include <sycl/detail/core.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>
#include <unistd.h>

using namespace sycl;
using namespace std::chrono;

#define random_float() (rand() / double(RAND_MAX))
#define INTER_NUM (150)
#define KERNEL_NUM (2000)

void make_queue_and_run_workload(std::vector<device> &subsubdevices) {
  std::cout << "[important] create " << subsubdevices.size()
            << " sycl queues, one for each sub-sub device" << std::endl;

  auto N = 1024 * 16;
  size_t global_range = 1024;
  size_t local_range = 16;

  std::vector<queue> queues;
  std::vector<float *> host_mem_ptrs;
  std::vector<float *> device_mem_ptrs;

  // Create queues for each subdevice.
  for (auto &ccs : subsubdevices) {
    queue q(ccs,
            {property::queue::enable_profiling(), property::queue::in_order()});
    auto *host_mem_ptr = malloc_host<float>(N, q);
    auto *device_mem_ptr = malloc_device<float>(N, q);

    for (int i = 0; i < N; ++i) {
      host_mem_ptr[i] = static_cast<float>(random_float());
    }

    q.memcpy(device_mem_ptr, host_mem_ptr, N * sizeof(float)).wait();

    host_mem_ptrs.push_back(host_mem_ptr);
    device_mem_ptrs.push_back(device_mem_ptr);
    queues.push_back(q);
  }

  // Run workload.
  for (auto m = 0; m < INTER_NUM; ++m) {
    for (int k = 0; k < KERNEL_NUM; ++k) {
      for (int j = 0; j < queues.size(); j++) {
        queue current_queue = queues[j];
        float *device_mem_ptr = device_mem_ptrs[j];

        auto event0 = current_queue.parallel_for<>(
            nd_range<1>(range<1>{global_range}, range<1>{local_range}),
            [=](nd_item<1> item) {
              int i = item.get_global_linear_id();
              device_mem_ptr[i] = device_mem_ptr[i] + float(2.0);
            });
      }
    }

    for (auto q : queues)
      q.wait();
  }

  for (int j = 0; j < queues.size(); j++) {
    sycl::free(device_mem_ptrs[j], queues[j]);
    sycl::free(host_mem_ptrs[j], queues[j]);
  }

  std::cout << "[info] Finish running workload" << std::endl;
}

int main(void) {
  std::cout << "[info] this case is used to submit workloads to queues on "
               "subsub device"
            << std::endl;
  std::vector<device> subsub;

  device d;

  // watch out device here
  auto subdevices = d.create_sub_devices<
      info::partition_property::partition_by_affinity_domain>(
      info::partition_affinity_domain::next_partitionable);
  std::cout << "[info] sub device size = " << subdevices.size() << std::endl;
  for (auto &subdev : subdevices) {
    auto subsubdevices = subdev.create_sub_devices<
        info::partition_property::ext_intel_partition_by_cslice>();

    std::cout << "[info] sub-sub device size = " << subsubdevices.size()
              << std::endl;
    for (auto &subsubdev : subsubdevices) {
      subsub.push_back(subsubdev);
    }
  }

  std::cout << "[info] all sub-sub devices count: " << subsub.size()
            << std::endl;

  make_queue_and_run_workload(subsub);

  return 0;
}
