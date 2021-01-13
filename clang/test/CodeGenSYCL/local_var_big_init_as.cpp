// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice \
// RUN: -emit-llvm -o - %s | FileCheck %s

// This test checks that data for big constant initializer lists is placed
// into the global address space by the SYCL compiler.

struct Test {
  Test() : set{
               0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
               0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15} {};
  int set[32];
};
// CHECK-DAG: @constinit = private unnamed_addr addrspace(1) constant
// CHECK: [32 x i32]
// CHECK: [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
// CHECK:  i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
// CHECK:  i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
// CHECK:  i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15
// CHECK: ], align 4
// CHECK-NOT: @constinit = private unnamed_addr addrspace(0)
// CHECK-NOT: @constinit = private unnamed_addr addrspace(2)
// CHECK-NOT: @constinit = private unnamed_addr addrspace(3)
// CHECK-NOT: @constinit = private unnamed_addr addrspace(4)

__attribute__((sycl_device)) void bar(Test &x);

__attribute__((sycl_device)) void zoo() {
  Test mc;
  bar(mc);
}
