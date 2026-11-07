// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -Wno-sycl-work-group-dtor -emit-llvm %s -o - | FileCheck %s

// This test checks that the kernel object of a parallel_for_work_group kernel,
// which is promoted to a work group local variable in addrspace(3), does not
// get its destructor registered with __cxa_atexit, even if a captured variable
// has a non-trivial destructor.

#include "Inputs/sycl.hpp"

using namespace sycl;

struct NonTrivial {
  int I;
  ~NonTrivial() {}
};

int main() {
  queue q;
  q.submit([&](handler &h) {
    NonTrivial Captured;
    h.parallel_for_work_group<class kernel>(
        range<1>{1}, range<1>{1}, [=](group<1> G) { (void)Captured; });
  });
  return 0;
}

// CHECK: @{{.*}}__SYCLKernel = internal addrspace(3) global %class.anon undef
// CHECK-NOT: __cxa_atexit
// CHECK-NOT: @__dso_handle
