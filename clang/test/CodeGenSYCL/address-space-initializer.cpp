// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice \
// RUN: -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// This test checks that data for big constant initializer lists is placed
// into the global address space by the SYCL compiler.

struct Test {
  Test() : set{
               0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
               0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
               // CHECK: @constinit = {{.*}}addrspace(1) {{.*}}[32 x i32]
               // CHECK-SAME: [i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15,
               // CHECK-SAME:  i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15]
           } {}
  int set[32];
};

__attribute__((sycl_device)) void foo() {
  Test t;
  (void)t;
}
