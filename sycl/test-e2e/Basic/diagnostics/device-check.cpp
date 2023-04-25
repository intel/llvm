// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=cpu %BE_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_DEVICE_TYPE=gpu %BE_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_DEVICE_TYPE=acc %BE_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_DEVICE_TYPE=CPU %BE_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_DEVICE_TYPE=GPU %BE_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_DEVICE_TYPE=ACC %BE_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_DEVICE_TYPE=Cpu %BE_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_DEVICE_TYPE=Gpu %BE_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_DEVICE_TYPE=Acc %BE_RUN_PLACEHOLDER %t.out
// RUN: env SYCL_DEVICE_TYPE=XPU %BE_RUN_PLACEHOLDER %t.out

//==------------------- device-check.cpp --------------------------==//
// This is a diagnostic test which ensures that
// device types are case-insensitive.
// It also checks for SYCL_DEVICE being set incorrectly.
//==---------------------------------------------------------------==//

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  try {
    queue q = queue();
    auto device = q.get_device();
    auto deviceName = device.get_info<sycl::info::device::name>();
    std::cout << " Device Name: " << deviceName << std::endl;
  }

  catch (runtime_error &E) {
    if (std::string(E.what()).find("SYCL_DEVICE_TYPE is not recognized.  Must "
                                   "be GPU, CPU, ACC or HOST.") ==
            std::string::npos &&
        std::string(E.what()).find("No device of requested type") ==
            std::string::npos) {
      std::cout << "Test failed: received error is incorrect." << std::endl;
      return 1;
    } else {
      std::cout << "Test passed: caught the expected error." << std::endl;
      return 0;
    }
  }

  std::cout << "Test passed: results are correct." << std::endl;
  return 0;
}
