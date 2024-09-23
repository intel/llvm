// REQUIRES: cpu, gpu, accelerator, opencl

// RUN: %{build} -o %t.out

// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:fpga %t.out | FileCheck %s --check-prefixes=CHECK-ACC-ONLY
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:gpu %t.out | FileCheck %s --check-prefixes=CHECK-GPU-ONLY
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:cpu %t.out | FileCheck %s --check-prefixes=CHECK-CPU-ONLY

// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:fpga,gpu %t.out | FileCheck %s --check-prefixes=CHECK-ACC-GPU
// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:fpga,cpu %t.out | FileCheck %s --check-prefixes=CHECK-ACC-CPU

// RUN: env ONEAPI_DEVICE_SELECTOR=opencl:cpu,fpga,gpu %t.out | FileCheck %s --check-prefixes=CHECK-ACC-GPU-CPU
// RUN: env ONEAPI_DEVICE_SELECTOR="opencl:*" %t.out | FileCheck %s --check-prefixes=CHECK-ACC-GPU-CPU
//
// CHECK-ACC-ONLY-NOT: Device: cpu
// CHECK-ACC-ONLY-NOT: Device: gpu
// CHECK-ACC-ONLY: Device: acc

//
// CHECK-GPU-ONLY-NOT: Device: acc
// CHECK-GPU-ONLY-NOT: Device: cpu
// CHECK-GPU-ONLY: Device: gpu

//
// CHECK-CPU-ONLY-NOT: Device: acc
// CHECK-CPU-ONLY-NOT: Device: gpu
// CHECK-CPU-ONLY: Device: cpu

//
// CHECK-ACC-GPU-NOT: Device: cpu
// CHECK-ACC-GPU-DAG: Device: acc
// CHECK-ACC-GPU-DAG: Device: gpu

//
// CHECK-ACC-CPU-NOT: Device: gpu
// CHECK-ACC-CPU-DAG: Device: acc
// CHECK-ACC-CPU-DAG: Device: cpu

//
// CHECK-ACC-GPU-CPU-DAG: Device: acc
// CHECK-ACC-GPU-CPU-DAG: Device: gpu
// CHECK-ACC-GPU-CPU-DAG: Device: cpu
//

#include <iostream>
#include <map>
#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {

  std::map<info::device_type, std::string> m = {
      {info::device_type::cpu, "cpu"},
      {info::device_type::gpu, "gpu"},
      {info::device_type::accelerator, "acc"},
      {info::device_type::all, "all"}};

  for (auto &d : device::get_devices()) {
    std::cout << "Device: " << m[d.get_info<info::device::device_type>()]
              << std::endl;
  }

  return 0;
}
