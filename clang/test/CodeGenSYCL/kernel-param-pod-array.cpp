// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks a kernel with an argument that is a POD array.

#include "Inputs/sycl.hpp"

using namespace cl::sycl;

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
// CHECK: %[[LOCAL_OBJECTA:[0-9]+]] = alloca %class{{.*}}.anon, align 4
// CHECK: %[[LOCAL_OBJECT:[0-9]+]] = addrspacecast %class{{.*}}.anon* %[[LOCAL_OBJECTA]] to %class{{.*}}.anon addrspace(4)*

// Check for Array init loop
// CHECK: %[[LAMBDA_PTR:.+]] = getelementptr inbounds %class{{.*}}.anon, %class{{.*}}.anon addrspace(4)* %[[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: %[[WRAPPER_PTR:.+]] = getelementptr inbounds %struct{{.*}}.__wrapper_class, %struct{{.*}}.__wrapper_class addrspace(4)* %[[ARR_ARG]].ascast, i32 0, i32 0
// CHECK: %[[ARRAY_BEGIN:.+]] = getelementptr inbounds [2 x i32], [2 x i32] addrspace(4)* %[[LAMBDA_PTR]], i64 0, i64 0
// CHECK: br label %[[ARRAYINITBODY:.+]]

// The loop body itself
// CHECK: [[ARRAYINITBODY]]:
// CHECK: %[[ARRAYINDEX:.+]] = phi i64 [ 0, %{{.*}} ], [ %[[NEXTINDEX:.+]], %[[ARRAYINITBODY]] ]
// CHECK: %[[TARG_ARRAY_ELEM:.+]] = getelementptr inbounds i32, i32 addrspace(4)* %[[ARRAY_BEGIN]], i64 %[[ARRAYINDEX]]
// CHECK: %[[SRC_ELEM:.+]] = getelementptr inbounds [2 x i32], [2 x i32] addrspace(4)* %[[WRAPPER_PTR]], i64 0, i64 %[[ARRAYINDEX]]
// CHECK: %[[SRC_VAL:.+]] = load i32, i32 addrspace(4)* %[[SRC_ELEM]]
// CHECK: store i32 %[[SRC_VAL]], i32 addrspace(4)* %[[TARG_ARRAY_ELEM]]
// CHECK: %[[NEXTINDEX]] = add nuw i64 %[[ARRAYINDEX]], 1
// CHECK: %[[ISDONE:.+]] = icmp eq i64 %[[NEXTINDEX]], 2
// CHECK: br i1 %[[ISDONE]], label %{{.*}}, label %[[ARRAYINITBODY]]

// Check kernel_C parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_C
// CHECK-SAME:(%struct{{.*}}.__wrapper_class{{.*}}* noundef byval(%struct{{.*}}.__wrapper_class{{.*}}) align 4 %[[ARR_ARG:.*]])

// Check local lambda object alloca
// CHECK: %[[LOCAL_OBJECTA:[0-9]+]] = alloca %class{{.*}}.anon{{.*}}, align 4
// CHECK: %[[LOCAL_OBJECT:[0-9]+]] = addrspacecast %class{{.*}}.anon{{.*}}* %[[LOCAL_OBJECTA]] to %class{{.*}}.anon{{.*}} addrspace(4)*

// Check for Array init loop
// CHECK: %[[LAMBDA_PTR:.+]] = getelementptr inbounds %class{{.*}}.anon{{.*}}, %class{{.*}}.anon{{.*}} addrspace(4)* %[[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: %[[WRAPPER_PTR:.+]] = getelementptr inbounds %struct{{.*}}.__wrapper_class{{.*}}, %struct{{.*}}.__wrapper_class{{.*}} addrspace(4)* %[[ARR_ARG]].ascast, i32 0, i32 0
// CHECK: %[[ARRAY_BEGIN:.+]] = getelementptr inbounds [2 x %struct{{.*}}.foo], [2 x %struct{{.*}}.foo] addrspace(4)* %[[LAMBDA_PTR]], i64 0, i64 0
// CHECK: br label %[[ARRAYINITBODY:.+]]

// The loop body itself
// CHECK: [[ARRAYINITBODY]]:
// CHECK: %[[ARRAYINDEX:.+]] = phi i64 [ 0, %{{.*}} ], [ %[[NEXTINDEX:.+]], %[[ARRAYINITBODY]] ]
// CHECK: %[[TARG_ARRAY_ELEM:.+]] = getelementptr inbounds %struct{{.*}}.foo, %struct{{.*}}.foo addrspace(4)* %[[ARRAY_BEGIN]], i64 %[[ARRAYINDEX]]
// CHECK: %[[SRC_ELEM:.+]] = getelementptr inbounds [2 x %struct{{.*}}.foo], [2 x %struct{{.*}}.foo] addrspace(4)* %[[WRAPPER_PTR]], i64 0, i64 %[[ARRAYINDEX]]
// CHECK: %[[TARG_PTR:.+]] = bitcast %struct{{.*}}.foo addrspace(4)* %[[TARG_ARRAY_ELEM]] to i8 addrspace(4)*
// CHECK: %[[SRC_PTR:.+]] = bitcast %struct{{.*}}.foo addrspace(4)* %[[SRC_ELEM]] to i8 addrspace(4)*
// call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 4 %[[TARG_PTR]], i8 addrspace(4)* align %[[SRC_PTR]], i64 24, i1 false)
// CHECK: %[[NEXTINDEX]] = add nuw i64 %[[ARRAYINDEX]], 1
// CHECK: %[[ISDONE:.+]] = icmp eq i64 %[[NEXTINDEX]], 2
// CHECK: br i1 %[[ISDONE]], label %{{.*}}, label %[[ARRAYINITBODY]]

// Check kernel_D parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_D
// CHECK-SAME:(%struct{{.*}}.__wrapper_class{{.*}}* noundef byval(%struct{{.*}}.__wrapper_class{{.*}}) align 4 %[[ARR_ARG:.*]])

// Check local lambda object alloca
// CHECK: %[[LOCAL_OBJECTA:[0-9]+]] = alloca %class{{.*}}.anon{{.*}}, align 4
// CHECK: %[[LOCAL_OBJECT:[0-9]+]] = addrspacecast %class{{.*}}.anon{{.*}}* %[[LOCAL_OBJECTA]] to %class{{.*}}.anon{{.*}} addrspace(4)*

// Check for Array init loop
// CHECK: %[[LAMBDA_PTR:.+]] = getelementptr inbounds %class{{.*}}.anon{{.*}}, %class{{.*}}.anon{{.*}} addrspace(4)* %[[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: %[[WRAPPER_PTR:.+]] = getelementptr inbounds %struct{{.*}}.__wrapper_class{{.*}}, %struct{{.*}}.__wrapper_class{{.*}} addrspace(4)* %[[ARR_ARG]].ascast, i32 0, i32 0
// CHECK: %[[ARRAY_BEGIN:.+]] = getelementptr inbounds [2 x [1 x i32]], [2 x [1 x i32]] addrspace(4)* %[[LAMBDA_PTR]], i64 0, i64 0
// CHECK: br label %[[ARRAYINITBODY:.+]]

// Check Outer loop.
// CHECK: [[ARRAYINITBODY]]:
// CHECK: %[[ARRAYINDEX:.+]] = phi i64 [ 0, %{{.*}} ], [ %[[NEXTINDEX:.+]], %[[ARRAYINITEND:.+]] ]
// CHECK: %[[TARG_OUTER_ELEM:.+]] = getelementptr inbounds [1 x i32], [1 x i32] addrspace(4)* %[[ARRAY_BEGIN]], i64 %[[ARRAYINDEX]]
// CHECK: %[[SRC_OUTER_ELEM:.+]] = getelementptr inbounds [2 x [1 x i32]], [2 x [1 x i32]] addrspace(4)* %[[WRAPPER_PTR]], i64 0, i64 %[[ARRAYINDEX]]
// CHECK: %[[ARRAY_BEGIN_INNER:.+]] = getelementptr inbounds [1 x i32], [1 x i32] addrspace(4)* %[[TARG_OUTER_ELEM]], i64 0, i64 0
// CHECK: br label %[[ARRAYINITBODY_INNER:.+]]

// Check Inner Loop
// CHECK: [[ARRAYINITBODY_INNER]]:
// CHECK: %[[ARRAYINDEX_INNER:.+]] = phi i64 [ 0, %{{.*}} ], [ %[[NEXTINDEX_INNER:.+]], %[[ARRAYINITBODY_INNER:.+]] ]
// CHECK: %[[TARG_INNER_ELEM:.+]] = getelementptr inbounds i32, i32 addrspace(4)* %[[ARRAY_BEGIN_INNER]], i64 %[[ARRAYINDEX_INNER]]
// CHECK: %[[SRC_INNER_ELEM:.+]] = getelementptr inbounds [1 x i32], [1 x i32] addrspace(4)* %[[SRC_OUTER_ELEM]], i64 0, i64 %[[ARRAYINDEX_INNER]]
// CHECK: %[[SRC_LOAD:.+]] = load i32, i32 addrspace(4)* %[[SRC_INNER_ELEM]]
// CHECK: store i32 %[[SRC_LOAD]], i32 addrspace(4)* %[[TARG_INNER_ELEM]]
// CHECK: %[[NEXTINDEX_INNER]] = add nuw i64 %[[ARRAYINDEX_INNER]], 1
// CHECK: %[[ISDONE_INNER:.+]] = icmp eq i64 %[[NEXTINDEX_INNER]], 1
// CHECK: br i1 %[[ISDONE_INNER]], label %[[ARRAYINITEND]], label %[[ARRAYINITBODY_INNER]]

// Check Inner loop 'end'
// CHECK: [[ARRAYINITEND]]:
// CHECK: %[[NEXTINDEX]] = add nuw i64 %[[ARRAYINDEX]], 1
// CHECK: %[[ISDONE:.+]] = icmp eq i64 %[[NEXTINDEX]], 2
// CHECK: br i1 %[[ISDONE]], label %{{.*}}, label %[[ARRAYINITBODY]]
