// RUN: %clangxx -fsycl -c -fsycl-device-only -S -emit-llvm %s -o - | FileCheck %s

// Tests that the underlying pointer in a const-qualified shared device_global
// is not optimized out during access.

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

const device_global<int> DeviceGlobalVar;

int main() {
  queue Q;
  Q.single_task([]() {
    // CHECK: load {{.*}} @_ZL15DeviceGlobalVar
    volatile int ReadVal = DeviceGlobalVar;
  });
  return 0;
}
