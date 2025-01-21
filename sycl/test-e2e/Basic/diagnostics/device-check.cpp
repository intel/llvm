// RUN: %{build} -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:cpu" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:fpga" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:CPU" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:GPU" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:Fpga" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:Cpu" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:Gpu" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:FPGA" %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:XPU" %{run-unfiltered-devices} %t.out

//==------------------- device-check.cpp --------------------------==//
// This is a diagnostic test which ensures that
// device types are case-insensitive.
// It also checks for exception when the device in ONEAPI_DEVICE_SELECTOR
// is set incorrectly.
//==---------------------------------------------------------------==//

#include <iostream>
#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  try {
    queue q = queue();
    auto device = q.get_device();
    auto deviceName = device.get_info<sycl::info::device::name>();
    std::cout << " Device Name: " << deviceName << std::endl;
  }

  catch (exception const &E) {

    // Exception when cpu/gpu/fpga is not available on the system.
    if (E.code() == errc::runtime) {
      if (std::string(E.what()).find("No device of requested type") ==
          std::string::npos) {
        std::cout << "Test failed: received error is incorrect." << std::endl;
        return 1;
      } else {
        std::cout << "Test passed: caught the expected error." << std::endl;
        return 0;
      }
    }
    // Exception while parsing an invalid device name.
    else if (E.code() == errc::invalid) {
      if (std::string(E.what()).find("error parsing device number: xpu") ==
          std::string::npos) {
        std::cout << "Test failed: received error is incorrect." << std::endl;
        return 1;
      } else {
        std::cout << "Test passed: caught the expected error." << std::endl;
        return 0;
      }
    } else {
      std::cout << "Test failed: unexpected exception." << std::endl;
      return 1;
    }
  }

  std::cout << "Test passed: results are correct." << std::endl;
  return 0;
}
