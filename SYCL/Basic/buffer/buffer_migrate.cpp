// The test is flaky, disable until investigated/fixed.
// REQUIRES: TEMPORARILY_DISABLED
// REQUIRES: gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: env NEOReadDebugKeys=1 CreateMultipleRootDevices=2 %GPU_RUN_PLACEHOLDER %t.out
// RUN: env NEOReadDebugKeys=1 CreateMultipleRootDevices=3 %GPU_RUN_PLACEHOLDER %t.out
//
// Test for buffer use in a context with multiple devices (all found
// root-devices)
//

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {

  int Data = 0;
  int Result = 0;
  buffer<int, 1> Buffer(&Data, range<1>(1));

  const auto &Devices =
      platform(gpu_selector_v).get_devices(info::device_type::gpu);
  std::cout << Devices.size() << " devices found" << std::endl;
  context C(Devices);

  int Index = 0;
  for (auto D : Devices) {
    std::cout << "Using on device " << Index << ": "
              << D.get_info<info::device::name>() << std::endl;
    Result |= (1 << Index);

    queue Q(C, D);
    Q.submit([&](handler &cgh) {
      accessor Accessor{Buffer, cgh, read_write};
      cgh.parallel_for<class MigrateBuffer>(
          range<1>(1), [=](id<1> ID) { Accessor[ID] |= (1 << Index); });
    });
    Q.wait();
    ++Index;
  }

  auto HostAcc = Buffer.get_host_access();
  auto Passed = (HostAcc[0] == Result);
  std::cout << "Checking result on host: " << (Passed ? "passed" : "FAILED")
            << std::endl;
  std::cout << HostAcc[0] << " ?= " << Result << std::endl;
  return !Passed;
}
