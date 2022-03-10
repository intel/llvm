// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks a kernel argument that is an Accessor array

#include "Inputs/sycl.hpp"

using namespace cl::sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {

  using Accessor =
      accessor<int, 1, access::mode::read_write, access::target::global_buffer>;
  Accessor acc[2];

  a_kernel<class kernel_A>(
      [=]() {
        acc[1].use();
      });
}

// Check kernel_A parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_A
// CHECK-SAME: i32 addrspace(1)* noundef align 4 [[MEM_ARG1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %"struct.cl::sycl::range"* noundef byval{{.*}}align 4 [[ACC_RANGE1:%[a-zA-Z0-9_]+_1]],
// CHECK-SAME: %"struct.cl::sycl::range"* noundef byval{{.*}}align 4 [[MEM_RANGE1:%[a-zA-Z0-9_]+_2]],
// CHECK-SAME: %"struct.cl::sycl::id"* noundef byval{{.*}}align 4 [[OFFSET1:%[a-zA-Z0-9_]+_3]],
// CHECK-SAME: i32 addrspace(1)* noundef align 4 [[MEM_ARG2:%[a-zA-Z0-9_]+_4]],
// CHECK-SAME: %"struct.cl::sycl::range"* noundef byval{{.*}}align 4 [[ACC_RANGE2:%[a-zA-Z0-9_]+_6]],
// CHECK-SAME: %"struct.cl::sycl::range"* noundef byval{{.*}}align 4 [[MEM_RANGE2:%[a-zA-Z0-9_]+_7]],
// CHECK-SAME: %"struct.cl::sycl::id"* noundef byval{{.*}}align 4 [[OFFSET2:%[a-zA-Z0-9_]+_8]])

// CHECK alloca for pointer arguments
// CHECK: [[MEM_ARG1:%[a-zA-Z0-9_.]+]] = alloca i32 addrspace(1)*, align 8
// CHECK: [[MEM_ARG2:%[a-zA-Z0-9_.]+]] = alloca i32 addrspace(1)*, align 8

// CHECK lambda object alloca
// CHECK: [[LOCAL_OBJECTA:%0]] = alloca %class.anon, align 4

// CHECK allocas for ranges
// CHECK: [[ACC_RANGE1A:%[a-zA-Z0-9_.]+]] = alloca %"struct.cl::sycl::range"
// CHECK: [[MEM_RANGE1A:%[a-zA-Z0-9_.]+]] = alloca %"struct.cl::sycl::range"
// CHECK: [[OFFSET1A:%[a-zA-Z0-9_.]+]] = alloca %"struct.cl::sycl::id"
// CHECK: [[ACC_RANGE2A:%[a-zA-Z0-9_.]+]] = alloca %"struct.cl::sycl::range"
// CHECK: [[MEM_RANGE2A:%[a-zA-Z0-9_.]+]] = alloca %"struct.cl::sycl::range"
// CHECK: [[OFFSET2A:%[a-zA-Z0-9_.]+]] = alloca %"struct.cl::sycl::id"

// CHECK lambda object addrspacecast
// CHECK: [[LOCAL_OBJECT:%.*]] = addrspacecast %class.anon* [[LOCAL_OBJECTA]] to %class.anon addrspace(4)*

// CHECK addrspacecasts for ranges
// CHECK: [[ACC_RANGE1AS:%.*]] = addrspacecast %"struct.cl::sycl::range"* [[ACC_RANGE1A]] to %"struct.cl::sycl::range" addrspace(4)*
// CHECK: [[MEM_RANGE1AS:%.*]] = addrspacecast %"struct.cl::sycl::range"* [[MEM_RANGE1A]] to %"struct.cl::sycl::range" addrspace(4)*
// CHECK: [[OFFSET1AS:%.*]] = addrspacecast %"struct.cl::sycl::id"* [[OFFSET1A]] to %"struct.cl::sycl::id" addrspace(4)*
// CHECK: [[ACC_RANGE2AS:%.*]] = addrspacecast %"struct.cl::sycl::range"* [[ACC_RANGE2A]] to %"struct.cl::sycl::range" addrspace(4)*
// CHECK: [[MEM_RANGE2AS:%.*]] = addrspacecast %"struct.cl::sycl::range"* [[MEM_RANGE2A]] to %"struct.cl::sycl::range" addrspace(4)*
// CHECK: [[OFFSET2AS:%.*]] = addrspacecast %"struct.cl::sycl::id"* [[OFFSET2A]] to %"struct.cl::sycl::id" addrspace(4)*
// CHECK accessor array default inits
// CHECK: [[ACCESSOR_ARRAY1:%[a-zA-Z0-9_]+]] = getelementptr inbounds %class.anon, %class.anon addrspace(4)* [[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: [[BEGIN:%[a-zA-Z0-9._]*]] = getelementptr inbounds [2 x [[ACCESSOR:.*]]], [2 x [[ACCESSOR]]] addrspace(4)* [[ACCESSOR_ARRAY1]], i64 0, i64 0
// Clang takes advantage of element 1 having the same address as the array, so it doesn't do a GEP.
// CTOR Call #1
// CHECK: call spir_func void @{{.+}}([[ACCESSOR]] addrspace(4)* {{[^,]*}} [[BEGIN]])
// CHECK: [[ELEM2_GEP:%[a-zA-Z0-9_.]+]] = getelementptr inbounds [[ACCESSOR]], [[ACCESSOR]] addrspace(4)* [[BEGIN]], i64 1
// CTOR Call #2
// CHECK: call spir_func void @{{.+}}([[ACCESSOR]] addrspace(4)* {{[^,]*}} [[ELEM2_GEP]])

// CHECK acc[0] __init method call
// CHECK: [[ACCESSOR_ARRAY1:%[a-zA-Z0-9_]+]] = getelementptr inbounds %class.anon, %class.anon addrspace(4)* [[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: [[INDEX1:%[a-zA-Z0-9._]*]] = getelementptr inbounds [2 x [[ACCESSOR]]], [2 x [[ACCESSOR]]] addrspace(4)* [[ACCESSOR_ARRAY1]], i64 0, i64 0
// CHECK load from kernel pointer argument alloca
// CHECK: [[MEM_LOAD1:%[a-zA-Z0-9_]+]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* [[MEM_ARG1]]
// CHECK: [[ACC_RANGE1:%.*]] = addrspacecast %"struct.cl::sycl::range" addrspace(4)* [[ACC_RANGE1AS]] to %"struct.cl::sycl::range"*
// CHECK: [[MEM_RANGE1:%.*]] = addrspacecast %"struct.cl::sycl::range" addrspace(4)* [[MEM_RANGE1AS]] to %"struct.cl::sycl::range"*
// CHECK: [[OFFSET1:%.*]] = addrspacecast %"struct.cl::sycl::id" addrspace(4)* [[OFFSET1AS]] to %"struct.cl::sycl::id"*
// CHECK: call spir_func void @{{.*}}__init{{.*}}(%"class.cl::sycl::accessor" addrspace(4)* {{[^,]*}} [[INDEX1]], i32 addrspace(1)* noundef [[MEM_LOAD1]], %"struct.cl::sycl::range"* noundef byval({{.*}}) align 4 [[ACC_RANGE1]], %"struct.cl::sycl::range"* noundef byval({{.*}}) align 4 [[MEM_RANGE1]], %"struct.cl::sycl::id"* noundef byval({{.*}}) align 4 [[OFFSET1]])

// CHECK acc[1] __init method call
// CHECK: [[ACCESSOR_ARRAY2:%[a-zA-Z0-9_]+]] = getelementptr inbounds %class.anon, %class.anon addrspace(4)* [[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: [[INDEX2:%[a-zA-Z0-9._]*]] = getelementptr inbounds [2 x [[ACCESSOR]]], [2 x [[ACCESSOR]]] addrspace(4)* [[ACCESSOR_ARRAY2]], i64 0, i64 1
// CHECK load from kernel pointer argument alloca
// CHECK: [[MEM_LOAD2:%[a-zA-Z0-9_]+]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* [[MEM_ARG2]]
// CHECK: [[ACC_RANGE2:%.*]] = addrspacecast %"struct.cl::sycl::range" addrspace(4)* [[ACC_RANGE2AS]] to %"struct.cl::sycl::range"*
// CHECK: [[MEM_RANGE2:%.*]] = addrspacecast %"struct.cl::sycl::range" addrspace(4)* [[MEM_RANGE2AS]] to %"struct.cl::sycl::range"*
// CHECK: [[OFFSET2:%.*]] = addrspacecast %"struct.cl::sycl::id" addrspace(4)* [[OFFSET2AS]] to %"struct.cl::sycl::id"*
// CHECK: call spir_func void @{{.*}}__init{{.*}}(%"class.cl::sycl::accessor" addrspace(4)* {{[^,]*}} [[INDEX2]], i32 addrspace(1)* noundef [[MEM_LOAD2]], %"struct.cl::sycl::range"* noundef byval({{.*}}) align 4 [[ACC_RANGE2]], %"struct.cl::sycl::range"* noundef byval({{.*}}) align 4 [[MEM_RANGE2]], %"struct.cl::sycl::id"* noundef byval({{.*}}) align 4 [[OFFSET2]])
