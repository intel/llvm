// RUN: %clangxx -O2 -std=c++17 -I %sycl_include/sycl -I %sycl_include -S -emit-llvm %s -o - | FileCheck %s

// The test verifies that the accessor::operator[] implementation is
// good enough for compiler to optimize away calls to getOffset and
// getMemoryRange and vectorize the loop.

#include <sycl/sycl.hpp>

// CHECK: define {{.*}}foo{{.*}} {
// CHECK-NOT: call
// CHECK-NOT: invoke
// CHECK: vector.body:
// CHECK-NOT: call
// CHECK-NOT: invoke
// CHECK: load <4 x i32>
// CHECK-NOT: call
// CHECK-NOT: invoke
// CHECK: store <4 x i32>
// CHECK-NOT: call
// CHECK-NOT: invoke
void foo(sycl::accessor<int, 1, sycl::access::mode::read_write,
                        sycl::target::host_buffer> &Acc,
         int *Src) {
  for (size_t I = 0; I < 64; ++I)
    Acc[I] = Src[I];
}
