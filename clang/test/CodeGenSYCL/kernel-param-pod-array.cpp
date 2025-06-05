// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks a kernel with an argument that is a POD array.

#include "Inputs/sycl.hpp"

using namespace sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(const Func &kernelFunc) {
  kernelFunc();
}

struct foo_inner {
  int foo_inner_x;
  int foo_inner_y;
};

struct foo {
  int foo_a;
  foo_inner foo_b[2];
  int foo_c;
};

int main() {

  int a[2];
  int array_2D[2][1];
  foo struct_array[2];

  a_kernel<class kernel_B>(
      [=]() {
        int local = a[1];
      });

  a_kernel<class kernel_C>(
      [=]() {
        foo local = struct_array[1];
      });

  a_kernel<class kernel_D>(
      [=]() {
        int local = array_2D[0][0];
      });
}

// Check kernel_B parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_B
// CHECK-SAME:(ptr noundef byval(%class.anon) align 4 %[[ARR_ARG:.*]])

// Check kernel_C parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_C
// CHECK-SAME:(ptr noundef byval(%class.anon.0) align 4 %[[ARR_ARG:.*]])

// Check kernel_D parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_D
// CHECK-SAME:(ptr noundef byval(%class.anon.1) align 4 %[[ARR_ARG:.*]])
