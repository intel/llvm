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
    // CHECK: load i32 {{.*}}({{.*}}* @_ZL15DeviceGlobalVar, i64 0, i32 0)
    volatile int ReadVal = DeviceGlobalVar;
  });
  return 0;
}
