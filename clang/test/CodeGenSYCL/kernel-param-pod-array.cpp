// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -opaque-pointers -emit-llvm %s -o - | FileCheck %s

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
// CHECK-SAME:(ptr noundef byval(%struct{{.*}}.__wrapper_class) align 4 %[[ARR_ARG:.*]])

// Check local lambda object alloca
// CHECK: %[[LOCAL_OBJECTA:[a-zA-Z0-9_]+]] = alloca %union{{.*}}.__wrapper_union, align 4
// CHECK: %[[LOCAL_OBJECT:[a-zA-Z0-9_.]+]] = addrspacecast ptr %[[LOCAL_OBJECTA]] to ptr addrspace(4)

// Check for Array init loop
// CHECK: %[[LAMBDA_PTR:.+]] = getelementptr inbounds %class{{.*}}.anon, ptr addrspace(4) %[[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: %[[WRAPPER_PTR:.+]] = getelementptr inbounds %struct{{.*}}.__wrapper_class, ptr addrspace(4) %[[ARR_ARG]].ascast, i32 0, i32 0
// CHECK: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 4 %[[LAMBDA_PTR]], ptr addrspace(4) align 4 %[[WRAPPER_PTR]], i64 8, i1 false)

// Check kernel_C parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_C
// CHECK-SAME:(ptr noundef byval(%struct{{.*}}.__wrapper_class{{.*}}) align 4 %[[ARR_ARG:.*]])

// Check local lambda object alloca
// CHECK: %[[LOCAL_OBJECTA:[a-zA-Z0-9_]+]] = alloca %union{{.*}}.__wrapper_union{{.*}}, align 4
// CHECK: %[[LOCAL_OBJECT:[a-zA-Z0-9_.]+]] = addrspacecast ptr %[[LOCAL_OBJECTA]] to ptr addrspace(4)

// Check for Array init loop
// CHECK: %[[LAMBDA_PTR:.+]] = getelementptr inbounds %class{{.*}}.anon{{.*}}, ptr addrspace(4) %[[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: %[[WRAPPER_PTR:.+]] = getelementptr inbounds %struct{{.*}}.__wrapper_class{{.*}}, ptr addrspace(4) %[[ARR_ARG]].ascast, i32 0, i32 0
// CHECK: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 4 %[[LAMBDA_PTR]], ptr addrspace(4) align 4 %[[WRAPPER_PTR]], i64 48, i1 false)

// Check kernel_D parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_D
// CHECK-SAME:(ptr noundef byval(%struct{{.*}}.__wrapper_class{{.*}}) align 4 %[[ARR_ARG:.*]])

// Check local lambda object alloca
// CHECK: %[[LOCAL_OBJECTA:[a-zA-Z0-9_]+]] = alloca %union{{.*}}.__wrapper_union{{.*}}, align 4
// CHECK: %[[LOCAL_OBJECT:[a-zA-Z0-9_.]+]] = addrspacecast ptr %[[LOCAL_OBJECTA]] to ptr addrspace(4)

// Check for Array init loop
// CHECK: %[[LAMBDA_PTR:.+]] = getelementptr inbounds %class{{.*}}.anon{{.*}}, ptr addrspace(4) %[[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: %[[WRAPPER_PTR:.+]] = getelementptr inbounds %struct{{.*}}.__wrapper_class{{.*}}, ptr addrspace(4) %[[ARR_ARG]].ascast, i32 0, i32 0
// CHECK: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 4 %[[LAMBDA_PTR]], ptr addrspace(4) align 4 %[[WRAPPER_PTR]], i64 8, i1 false)
