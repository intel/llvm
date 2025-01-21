// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -internal-isystem %S/Inputs -o - %s | FileCheck %s
// RUN: %clang_cc1 -fsycl-is-host -emit-llvm -internal-isystem %S/Inputs -o - %s | FileCheck %s --check-prefixes=HOST

// This test checks if the sycl-device module flag is created for device
// compilations and not for host compilations.
#include "sycl.hpp"

void foo() {
  sycl::handler h;
  h.single_task([]() {});
}

// Check for the presence of sycl-device module flag in device
// compilations and its absence in host compilations.
// CHECK: !{{[0-9]*}} = !{i32 1, !"sycl-device", i32 1}
// HOST-NOT: !"sycl-device"
