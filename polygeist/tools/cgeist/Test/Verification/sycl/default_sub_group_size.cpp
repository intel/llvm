// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -S -emit-mlir %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-NONE
// RUN: clang++  -fsycl -fsycl-device-only -fsycl-default-sub-group-size=8 -O0 -w -S -emit-mlir %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-8
// RUN: clang++  -fsycl -fsycl-device-only -fsycl-default-sub-group-size=automatic -O0 -w -S -emit-mlir %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-AUTO
// RUN: clang++  -fsycl -fsycl-device-only -fsycl-default-sub-group-size=primary -O0 -w -S -emit-mlir %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-PRIM

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue q;
  q.submit([&](handler &h) { h.single_task<class kernel_name>([]() {}); });
  return 0;
}

// CHECK: kernel_name
// CHECK-NONE-NOT:    intel_reqd_sub_group_size
// CHECK-8:           intel_reqd_sub_group_size = 8
// CHECK-AUTO:        intel_reqd_sub_group_size = "automatic"
// CHECK-PRIM:        intel_reqd_sub_group_size = "primary"
