// RUN: %{build} -o %t.out

// Test discard filters in ONEAPI_DEVICE_SELECTOR.
// RUN: env ONEAPI_DEVICE_SELECTOR="!*:gpu" %{run-unfiltered-devices} %t.out | FileCheck %s --allow-empty --implicit-check-not="[{{.*}}{{gpu|GPU|Gpu}}{{.*}}]:[{{.*}}]"
// RUN: env ONEAPI_DEVICE_SELECTOR="!*:cpu" %{run-unfiltered-devices} %t.out | FileCheck %s --allow-empty --implicit-check-not="[{{.*}}{{cpu|CPU|cpu}}{{.*}}]:[{{.*}}]"
// RUN: env ONEAPI_DEVICE_SELECTOR="!*:cpu,gpu" %{run-unfiltered-devices} %t.out | FileCheck %s --allow-empty --implicit-check-not="[{{.*}}{{cpu|CPU|cpu|gpu|GPU|Gpu}}{{.*}}]:[{{.*}}]"

// RUN: env ONEAPI_DEVICE_SELECTOR="!opencl:*" %{run-unfiltered-devices} %t.out | FileCheck %s --allow-empty --implicit-check-not="[{{.*}}]:[{{.*}}{{OpenCL|opencl|Opencl}}{{.*}}]"
// RUN: env ONEAPI_DEVICE_SELECTOR="!level_zero:*" %{run-unfiltered-devices} %t.out | FileCheck %s --allow-empty --implicit-check-not="[{{.*}}]:[{{.*}}{{Level-Zero}}{{.*}}]"

// RUN: env ONEAPI_DEVICE_SELECTOR="!level_zero:cpu" %{run-unfiltered-devices} %t.out | FileCheck %s --allow-empty --implicit-check-not="[{{.*}}{{cpu|CPU|Cpu}}{{.*}}]:[{{.*}}{{Level-Zero}}{{.*}}]"

#include <iostream>
#include <sycl/detail/core.hpp>
using namespace sycl;

int main() {
  for (auto &d : device::get_devices()) {
    // Get device name and backend name.
    std::string device_name = d.get_info<info::device::name>();
    std::string be_name = d.get_platform().get_info<info::platform::name>();

    std::cout << "[" << device_name << "]:[" << be_name << "]" << std::endl;
  }
  return 0;
}
