// REQUIRES: gpu,level_zero
// RUN: %{build} -o %t.out
// RUN: env UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck %s
// UNSUPPORTED: ze_debug

// Temporarily disabled for DG2 due to failures.
// UNSUPPORTED: gpu-intel-dg2

#include <iostream>
#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  constexpr int Size = 100;
  queue Queue;
  auto D = Queue.get_device();
  auto NumOfDevices = Queue.get_context().get_devices().size();
  buffer<::cl_int, 1> Buffer(Size);
  Queue.submit([&](handler &cgh) {
    accessor Accessor{Buffer, cgh, read_write};
    if (D.has(aspect::usm_host_allocations))
      std::cerr << "Integrated GPU should use zeMemAllocHost\n";
    else
      std::cerr << "Discrete GPU should use zeMemAllocDevice\n";
    cgh.parallel_for<class CreateBuffer>(range<1>(Size),
                                         [=](id<1> ID) { Accessor[ID] = 0; });
  });
  Queue.wait();

  return 0;
}

// CHECK: {{Integrated|Discrete}} GPU should use [[API:zeMemAllocHost|zeMemAllocDevice]]
// CHECK: ZE ---> [[API]](
