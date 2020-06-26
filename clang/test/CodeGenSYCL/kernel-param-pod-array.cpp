// RUN: %clang_cc1 -fsycl -fsycl-is-device -I %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks a kernel with an argument that is a POD array.

#include <sycl.hpp>

using namespace cl::sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {

  int a[2];

  a_kernel<class kernel_B>(
      [=]() {
        int local = a[1];
      });
}

// Check kernel_B parameters
// CHECK: define spir_kernel void @{{.*}}kernel_B
// CHECK-SAME: i32 [[ELEM_ARG0:%[a-zA-Z0-9_]+]],
// CHECK-SAME: i32 [[ELEM_ARG1:%[a-zA-Z_]+_[0-9]+]])

// Check local lambda object alloca
// CHECK: [[LOCAL_OBJECT:%[0-9]+]] = alloca %"class.{{.*}}.anon", align 4

// Check local variables created for parameters
// CHECK: store i32 [[ELEM_ARG0]], i32* [[ELEM_L0:%[a-zA-Z_]+.addr]], align 4
// CHECK: store i32 [[ELEM_ARG1]], i32* [[ELEM_L1:%[a-zA-Z_]+.addr[0-9]*]], align 4

// Check init of local array
// CHECK: [[ARRAY:%[0-9]*]] = getelementptr inbounds %"class.{{.*}}.anon", %"class.{{.*}}.anon"* [[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: [[ARRAY_BEGIN:%[a-zA-Z_.]+]] = getelementptr inbounds [2 x i32], [2 x i32]* [[ARRAY]], i64 0, i64 0
// CHECK: [[ARRAY0:%[0-9]*]] = load i32, i32* [[ELEM_L0]], align 4
// CHECK: store i32 [[ARRAY0]], i32* [[ARRAY_BEGIN]], align 4
// CHECK: [[ARRAY_ELEMENT:%[a-zA-Z_.]+]] = getelementptr inbounds i32, i32* %arrayinit.begin, i64 1
// CHECK: [[ARRAY1:%[0-9]*]] = load i32, i32* [[ELEM_L1]], align 4
// CHECK: store i32 [[ARRAY1]], i32* [[ARRAY_ELEMENT]], align 4