// RUN: %clangxx -fsycl -fsycl-device-only %if cuda %{ -fsycl-targets=nvptx64-nvidia-cuda %} %if hip-amd %{ -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a %} -S -emit-llvm %s -o - | FileCheck %s  %if cuda || hip-amd %{ --check-prefixes=CHECK-CONST %}

// Tests that const T device_global uses const address space for cuda/hip and
// global address space otherwise

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

device_global<const int> DeviceGlobalVar;

int main() {
  queue Q;
  Q.single_task([]() {
    // CHECK-CONST: (ptr addrspace(4) @DeviceGlobalVar
    // CHECK: (ptr addrspace(1) @DeviceGlobalVar
    volatile int ReadVal = DeviceGlobalVar;
  });
  return 0;
}
