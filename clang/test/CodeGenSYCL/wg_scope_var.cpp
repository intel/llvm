// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -fsycl-is-device -disable-llvm-passes -I %S/Inputs -emit-llvm %s -o - | FileCheck %s

// Checked that local variables declared by the user in PWFG scope are turned into globals in the local address space.
// CHECK: @{{.*myLocal.*}} = internal addrspace(3) global i32 0

#include "sycl.hpp"

using namespace cl::sycl;

int main() {
  queue myQueue;

  myQueue.submit([&](handler &cgh) {
    cgh.parallel_for_work_group<class kernel>(
        range<3>(2, 2, 2), range<3>(2, 2, 2), [=](group<3> myGroup) {
          int myLocal;
        });
  });

  return 0;
}
