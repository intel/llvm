// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

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
// CHECK-SAME: i32 addrspace(1)* [[MEM_ARG1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %"struct.{{.*}}.cl::sycl::range"* byval{{.*}}align 4 [[ACC_RANGE1:%[a-zA-Z0-9_]+_1]],
// CHECK-SAME: %"struct.{{.*}}.cl::sycl::range"* byval{{.*}}align 4 [[MEM_RANGE1:%[a-zA-Z0-9_]+_2]],
// CHECK-SAME: %"struct.{{.*}}.cl::sycl::id"* byval{{.*}}align 4 [[OFFSET1:%[a-zA-Z0-9_]+_3]],
// CHECK-SAME: i32 addrspace(1)* [[MEM_ARG2:%[a-zA-Z0-9_]+_4]],
// CHECK-SAME: %"struct.{{.*}}.cl::sycl::range"* byval{{.*}}align 4 [[ACC_RANGE2:%[a-zA-Z0-9_]+_6]],
// CHECK-SAME: %"struct.{{.*}}.cl::sycl::range"* byval{{.*}}align 4 [[MEM_RANGE2:%[a-zA-Z0-9_]+_7]],
// CHECK-SAME: %"struct.{{.*}}.cl::sycl::id"* byval{{.*}}align 4 [[OFFSET2:%[a-zA-Z0-9_]+_8]])

// CHECK alloca for pointer arguments
// CHECK: [[MEM_ARG1:%[a-zA-Z0-9_.]+]] = alloca i32 addrspace(1)*, align 8
// CHECK: [[MEM_ARG2:%[a-zA-Z0-9_.]+]] = alloca i32 addrspace(1)*, align 8

// CHECK lambda object alloca
// CHECK: [[LOCAL_OBJECT:%0]] = alloca %"class.{{.*}}.anon", align 4

// CHECK allocas for ranges
// CHECK: [[ACC_RANGE1:%[a-zA-Z0-9_.]+]] = alloca %"struct.{{.*}}.cl::sycl::range"
// CHECK: [[MEM_RANGE1:%[a-zA-Z0-9_.]+]] = alloca %"struct.{{.*}}.cl::sycl::range"
// CHECK: [[OFFSET1:%[a-zA-Z0-9_.]+]] = alloca %"struct.{{.*}}.cl::sycl::id"
// CHECK: [[ACC_RANGE2:%[a-zA-Z0-9_.]+]] = alloca %"struct.{{.*}}.cl::sycl::range"
// CHECK: [[MEM_RANGE2:%[a-zA-Z0-9_.]+]] = alloca %"struct.{{.*}}.cl::sycl::range"
// CHECK: [[OFFSET2:%[a-zA-Z0-9_.]+]] = alloca %"struct.{{.*}}.cl::sycl::id"

// CHECK accessor array default inits
// CHECK: [[ACCESSOR_ARRAY1:%[a-zA-Z0-9_]+]] = getelementptr inbounds %"class.{{.*}}.anon", %"class.{{.*}}.anon"* [[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: [[BEGIN:%[a-zA-Z0-9._]*]] = getelementptr inbounds [2 x [[ACCESSOR:.*]]], [2 x [[ACCESSOR]]]* [[ACCESSOR_ARRAY1]], i64 0, i64 0
// Clang takes advantage of element 1 having the same address as the array, so it doesn't do a GEP.
// CHECK: [[ELEM1_ASCAST:%[a-zA-Z0-9_.]+]] = addrspacecast [[ACCESSOR]]* [[BEGIN]] to [[ACCESSOR]] addrspace(4)*
// CTOR Call #1
// CHECK: call spir_func void @{{.+}}([[ACCESSOR]] addrspace(4)* {{[^,]*}} [[ELEM1_ASCAST]])
// CHECK: [[ELEM2_GEP:%[a-zA-Z0-9_.]+]] = getelementptr inbounds [[ACCESSOR]], [[ACCESSOR]]* [[BEGIN]], i64 1
// CHECK: [[ELEM2_ASCAST:%[a-zA-Z0-9_.]+]] = addrspacecast [[ACCESSOR]]* [[ELEM2_GEP]] to [[ACCESSOR]] addrspace(4)*
// CTOR Call #2
// CHECK: call spir_func void @{{.+}}([[ACCESSOR]] addrspace(4)* {{[^,]*}} [[ELEM2_ASCAST]])

// CHECK acc[0] __init method call
// CHECK: [[ACCESSOR_ARRAY1:%[a-zA-Z0-9_]+]] = getelementptr inbounds %"class.{{.*}}.anon", %"class.{{.*}}.anon"* [[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: [[INDEX1:%[a-zA-Z0-9._]*]] = getelementptr inbounds [2 x [[ACCESSOR]]], [2 x [[ACCESSOR]]]* [[ACCESSOR_ARRAY1]], i64 0, i64 0
// CHECK load from kernel pointer argument alloca
// CHECK: [[MEM_LOAD1:%[a-zA-Z0-9_]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[MEM_ARG1]]
// CHECK: [[ACC_CAST1:%[0-9]+]] = addrspacecast [[ACCESSOR]]* [[INDEX1]] to [[ACCESSOR]] addrspace(4)*
// CHECK: call spir_func void @{{.*}}__init{{.*}}(%"class.{{.*}}.cl::sycl::accessor" addrspace(4)* {{[^,]*}} [[ACC_CAST1]], i32 addrspace(1)* [[MEM_LOAD1]], %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[ACC_RANGE1]], %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[MEM_RANGE1]], %"struct.{{.*}}.cl::sycl::id"* byval({{.*}}) align 4 [[OFFSET1]])

// CHECK acc[1] __init method call
// CHECK: [[ACCESSOR_ARRAY2:%[a-zA-Z0-9_]+]] = getelementptr inbounds %"class.{{.*}}.anon", %"class.{{.*}}.anon"* [[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: [[INDEX2:%[a-zA-Z0-9._]*]] = getelementptr inbounds [2 x [[ACCESSOR]]], [2 x [[ACCESSOR]]]* [[ACCESSOR_ARRAY2]], i64 0, i64 1
// CHECK load from kernel pointer argument alloca
// CHECK: [[MEM_LOAD2:%[a-zA-Z0-9_]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[MEM_ARG2]]
// CHECK: [[ACC_CAST2:%[0-9]+]] = addrspacecast [[ACCESSOR]]* [[INDEX2]] to [[ACCESSOR]] addrspace(4)*
// CHECK: call spir_func void @{{.*}}__init{{.*}}(%"class.{{.*}}.cl::sycl::accessor" addrspace(4)* {{[^,]*}} [[ACC_CAST2]], i32 addrspace(1)* [[MEM_LOAD2]], %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[ACC_RANGE2]], %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[MEM_RANGE2]], %"struct.{{.*}}.cl::sycl::id"* byval({{.*}}) align 4 [[OFFSET2]])
