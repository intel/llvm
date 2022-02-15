// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks that a local variable initialized within a
// parallel_for_work_group scope is initialized as an UndefValue in addrspace(3)
// in LLVM IR.

#include "Inputs/sycl.hpp"

using namespace sycl;

int main() {
  queue q;
  q.submit([&](handler &h) {
    h.parallel_for_work_group<class kernel>(
        range<1>{1}, range<1>{1}, [=](group<1> G) {
          int WG_VAR = 10;
        });
  });
  return 0;
}

// CHECK: @{{.*}}WG_VAR = internal addrspace(3) global {{.*}} undef, {{.*}}
