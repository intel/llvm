// RUN: %clang_cc1 -opaque-pointers -fsycl-is-device -fintelfpga -triple spir64_fpga -aux-triple x86_64-unknown-linux-gnu -IInputs -emit-llvm %s -o - | FileCheck %s

// This test checks that we generate appropriate code for division
// operations of _BitInts of size greater than 128 bits, since it
// is allowed when -fintelfpga is enabled.  The test uses a value of
// 4096 for bitint size, the maximum that is currently supported.

#include "Inputs/sycl.hpp"

// CHECK: define{{.*}} void @_Z3fooDB4096_S_(ptr addrspace(4) {{.*}} sret(i4096) align 8 %agg.result, ptr {{.*}} byval(i4096) align 8 %[[ARG1:[0-9]+]], ptr {{.*}} byval(i4096) align 8 %[[ARG2:[0-9]+]])
signed _BitInt(4096) foo(signed _BitInt(4096) a, signed _BitInt(4096) b) {
  // CHECK: %a.addr.ascast = addrspacecast ptr %a.addr to ptr addrspace(4)
  // CHECK: %b.addr.ascast = addrspacecast ptr %b.addr to ptr addrspace(4)
  // CHECK: %a = load i4096, ptr %[[ARG1]], align 8
  // CHECK: %b = load i4096, ptr %[[ARG2]], align 8
  // CHECK: store i4096 %a, ptr addrspace(4) %a.addr.ascast, align 8
  // CHECK: store i4096 %b, ptr addrspace(4) %b.addr.ascast, align 8
  // CHECK: %2 = load i4096, ptr addrspace(4) %a.addr.ascast, align 8
  // CHECK: %3 = load i4096, ptr addrspace(4) %b.addr.ascast, align 8
  // CHECK: %div = sdiv i4096 %2, %3
  return a / b;
}

int main() {
  sycl::handler h;
  auto lambda = []() {
    _BitInt(4096) a, b = 3, c = 4;
    a = foo(b, c);
  };
  h.single_task(lambda);
}
