// RUN: %clang_cc1 -fsycl -fsycl-is-device -I %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
// XFAIL: *

// This test checks a kernel with an argument that is a POD array.

#include <sycl.hpp>

using namespace cl::sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {

  int a[100];

  a_kernel<class kernel_B>(
      [=]() {
        int local = a[3];
      });
}

// Check kernel_B parameters
// CHECK: define spir_kernel void @{{.*}}kernel_B
// CHECK-SAME: %struct.{{.*}}.wrapped_array* byval{{.*}}align 4 [[ARG_STRUCT:%[a-zA-Z0-9_]+]]

// Check local lambda object alloca
// CHECK: [[LOCAL_OBJECT:%0]] = alloca %"class.{{.*}}.anon", align 4

// Check init of local array
// CHECK: [[ARRAY1:%[a-zA-Z0-9_]+]] = getelementptr inbounds %"class.{{.*}}.anon", %"class.{{.*}}.anon"* [[LOCAL_OBJECT]], i32 0, i32 0

// CHECK: [[ARRAY2:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct.{{.*}}.wrapped_array, %struct.{{.*}}.wrapped_array* [[ARG_STRUCT]], i32 0, i32 0

// CHECK: %{{[a-zA-Z0-9._]+}} = getelementptr inbounds [100 x i32], [100 x i32]* [[ARRAY1]], i64 0, i64 0

// CHECK: %{{[a-zA-Z0-9_]+}} = getelementptr inbounds [100 x i32], [100 x i32]* [[ARRAY2]], i64 0, i64
