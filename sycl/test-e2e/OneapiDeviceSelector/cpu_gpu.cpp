// REQUIRES: any-device-is-cpu, gpu, opencl

// RUN: %{build} -o %t.out

// RUN: %{run-unfiltered-devices} env ONEAPI_DEVICE_SELECTOR=opencl:gpu %t.out | FileCheck %s --check-prefixes=CHECK-GPU-ONLY
// RUN: %{run-unfiltered-devices} env ONEAPI_DEVICE_SELECTOR=opencl:cpu %t.out | FileCheck %s --check-prefixes=CHECK-CPU-ONLY

// RUN: %{run-unfiltered-devices} env ONEAPI_DEVICE_SELECTOR=opencl:cpu,gpu %t.out | FileCheck %s --check-prefixes=CHECK-GPU-CPU
// RUN: %{run-unfiltered-devices} env ONEAPI_DEVICE_SELECTOR="opencl:*" %t.out | FileCheck %s --check-prefixes=CHECK-GPU-CPU

// CHECK-GPU-ONLY-NOT: Device: cpu
// CHECK-GPU-ONLY: Device: gpu

// CHECK-CPU-ONLY-NOT: Device: gpu
// CHECK-CPU-ONLY: Device: cpu

// CHECK-GPU-CPU-DAG: Device: gpu
// CHECK-GPU-CPU-DAG: Device: cpu

#include <iostream>
#include <map>
#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {

  std::map<info::device_type, std::string> m = {
      {info::device_type::cpu, "cpu"},
      {info::device_type::gpu, "gpu"},
      {info::device_type::all, "all"}};

  for (auto &d : device::get_devices()) {
    std::cout << "Device: " << m[d.get_info<info::device::device_type>()]
              << std::endl;
  }

  return 0;
}
