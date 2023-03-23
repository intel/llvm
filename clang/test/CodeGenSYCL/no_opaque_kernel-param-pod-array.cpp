// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -no-opaque-pointers -emit-llvm %s -o - | FileCheck %s

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
// CHECK-SAME:(%struct{{.*}}.__wrapper_class* noundef byval(%struct{{.*}}.__wrapper_class) align 4 %[[ARR_ARG:.*]])

// Check local lambda object alloca
// CHECK: %[[LOCAL_OBJECTA:[a-zA-Z0-9_]+]] = alloca %union{{.*}}.__wrapper_union, align 4
// CHECK: %[[LOCAL_OBJECT:[a-zA-Z0-9_.]+]] = addrspacecast %union{{.*}}.__wrapper_union* %[[LOCAL_OBJECTA]] to %union{{.*}}.__wrapper_union addrspace(4)*

// Check for Array init
// CHECK: %[[KERNEL_BC:[a-zA-Z0-9_.]+]] = bitcast %union{{.*}}.__wrapper_union addrspace(4)* %[[LOCAL_OBJECT]] to %class{{.*}}.anon addrspace(4)*
// CHECK: %[[A_DST_PTR:[a-zA-Z0-9_.]+]] = getelementptr inbounds %class{{.*}}.anon, %class{{.*}}.anon addrspace(4)* %[[KERNEL_BC]], i32 0, i32 0
// CHECK: %[[A_DST:[a-zA-Z0-9_.]+]] = bitcast [2 x i32] addrspace(4)* %[[A_DST_PTR]] to i8 addrspace(4)*
// CHECK: %[[A_SRC_PTR:[a-zA-Z0-9_.]+]] = getelementptr inbounds %struct{{.*}}.__wrapper_class, %struct{{.*}}.__wrapper_class addrspace(4)* %[[ARR_ARG]].ascast, i32 0, i32 0
// CHECK: %[[A_SRC:[a-zA-Z0-9_.]+]] = bitcast [2 x i32] addrspace(4)* %[[A_SRC_PTR]] to i8 addrspace(4)*
// CHECK: call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 4 %[[A_DST]], i8 addrspace(4)* align 4 %[[A_SRC]], i64 8, i1 false)

// Check kernel_C parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_C
// CHECK-SAME:(%struct{{.*}}.__wrapper_class{{.*}}* noundef byval(%struct{{.*}}.__wrapper_class{{.*}}) align 4 %[[ARR_ARG:.*]])

// Check local lambda object alloca
// CHECK: %[[LOCAL_OBJECTA:[a-zA-Z0-9_]+]] = alloca %union{{.*}}.__wrapper_union{{.*}}, align 4
// CHECK: %[[LOCAL_OBJECT:[a-zA-Z0-9_.]+]] = addrspacecast %union{{.*}}.__wrapper_union{{.*}}* %[[LOCAL_OBJECTA]] to %union{{.*}}.__wrapper_union{{.*}} addrspace(4)*

// Check for Array init
// CHECK: %[[KERNEL_BC:[a-zA-Z0-9_.]+]] = bitcast %union{{.*}}.__wrapper_union{{.*}} addrspace(4)* %[[LOCAL_OBJECT]] to %class{{.*}}.anon{{.*}} addrspace(4)*
// CHECK: %[[DST_PTR:[a-zA-Z0-9_.]+]] = getelementptr inbounds %class{{.*}}.anon{{.*}}, %class{{.*}}.anon{{.*}} addrspace(4)* %[[KERNEL_BC]], i32 0, i32 0
// CHECK: %[[DST:[a-zA-Z0-9_.]+]] = bitcast [2 x %struct.foo] addrspace(4)* %[[DST_PTR]] to i8 addrspace(4)*
// CHECK: %[[SRC_PTR:[a-zA-Z0-9_.]+]] = getelementptr inbounds %struct{{.*}}.__wrapper_class{{.*}}, %struct{{.*}}.__wrapper_class{{.*}} addrspace(4)* %[[ARR_ARG]].ascast, i32 0, i32 0
// CHECK: %[[SRC:[a-zA-Z0-9_.]+]] = bitcast [2 x %struct.foo] addrspace(4)* %[[SRC_PTR]] to i8 addrspace(4)*
// CHECK: call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 4 %[[DST]], i8 addrspace(4)* align 4 %[[SRC]], i64 48, i1 false)

// Check kernel_D parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_D
// CHECK-SAME:(%struct{{.*}}.__wrapper_class{{.*}}* noundef byval(%struct{{.*}}.__wrapper_class{{.*}}) align 4 %[[ARR_ARG:.*]])

// Check local lambda object alloca
// CHECK: %[[LOCAL_OBJECTA:[a-zA-Z0-9_]+]] = alloca %union{{.*}}.__wrapper_union{{.*}}, align 4
// CHECK: %[[LOCAL_OBJECT:[a-zA-Z0-9_.]+]] = addrspacecast %union{{.*}}.__wrapper_union{{.*}}* %[[LOCAL_OBJECTA]] to %union{{.*}}.__wrapper_union{{.*}} addrspace(4)*

// Check for Array init
// CHECK: %[[KERNEL_BC:[a-zA-Z0-9_.]+]] = bitcast %union{{.*}}.__wrapper_union{{.*}} addrspace(4)* %[[LOCAL_OBJECT]] to %class{{.*}}.anon{{.*}} addrspace(4)*
// CHECK: %[[DST_PTR:[a-zA-Z0-9_.]+]] = getelementptr inbounds %class{{.*}}.anon{{.*}}, %class{{.*}}.anon{{.*}} addrspace(4)* %[[KERNEL_BC]], i32 0, i32 0
// CHECK: %[[DST:[a-zA-Z0-9_.]+]] = bitcast [2 x [1 x i32]] addrspace(4)* %[[DST_PTR]] to i8 addrspace(4)*
// CHECK: %[[SRC_PTR:[a-zA-Z0-9_.]+]] = getelementptr inbounds %struct{{.*}}.__wrapper_class{{.*}}, %struct{{.*}}.__wrapper_class{{.*}} addrspace(4)* %[[ARR_ARG]].ascast, i32 0, i32 0
// CHECK: %[[SRC:[a-zA-Z0-9_.]+]] = bitcast [2 x [1 x i32]] addrspace(4)* %[[SRC_PTR]] to i8 addrspace(4)*
// CHECK: call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 4 %[[DST]], i8 addrspace(4)* align 4 %[[SRC]], i64 8, i1 false)
