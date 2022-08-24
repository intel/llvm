// RUN: %clang_cc1 -fsycl-is-device -fsycl-int-header=%t.h -triple spir64-unknown-unknown -disable-llvm-passes -opaque-pointers -emit-llvm %s -o - | FileCheck %s

// This test checks a kernel with struct parameter that contains an Accessor array.

#include "Inputs/sycl.hpp"

using namespace sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void a_kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {

  using Accessor =
      accessor<int, 1, access::mode::read_write, access::target::global_buffer>;

  struct struct_acc_t {
    Accessor member_acc[2];
  } struct_acc;

  a_kernel<class kernel_C>(
      [=]() {
        struct_acc.member_acc[1].use();
      });
}

// CHECK kernel_C parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_C
// CHECK-SAME: ptr addrspace(1) noundef align 4 [[MEM_ARG1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr noundef byval({{.*}}) align 4 [[ACC_RANGE1:%[a-zA-Z0-9_]+1]],
// CHECK-SAME: ptr noundef byval({{.*}}) align 4 [[MEM_RANGE1:%[a-zA-Z0-9_]+2]],
// CHECK-SAME: ptr noundef byval({{.*}}) align 4 [[OFFSET1:%[a-zA-Z0-9_]+3]],
// CHECK-SAME: ptr addrspace(1) noundef align 4 [[MEM_ARG2:%[a-zA-Z0-9_]+4]],
// CHECK-SAME: ptr noundef byval({{.*}}) align 4 [[ACC_RANGE2:%[a-zA-Z0-9_]+6]],
// CHECK-SAME: ptr noundef byval({{.*}}) align 4 [[MEM_RANGE2:%[a-zA-Z0-9_]+7]],
// CHECK-SAME: ptr noundef byval({{.*}}) align 4 [[OFFSET2:%[a-zA-Z0-9_]+8]])

// Check alloca for pointer arguments
// CHECK: [[MEM_ARG1]].addr{{[0-9]*}} = alloca ptr addrspace(1), align 8
// CHECK: [[MEM_ARG1]].addr{{[0-9]*}} = alloca ptr addrspace(1), align 8

// Check lambda object alloca
// CHECK: [[LOCAL_OBJECTA:%__SYCLKernel]] = alloca %class{{.*}}.anon, align 4

// Check allocas for ranges
// CHECK: [[ACC_RANGE1A:%[a-zA-Z0-9_.]+]] = alloca %"struct.sycl::_V1::range"
// CHECK: [[MEM_RANGE1A:%[a-zA-Z0-9_.]+]] = alloca %"struct.sycl::_V1::range"
// CHECK: [[OFFSET1A:%[a-zA-Z0-9_.]+]] = alloca %"struct.sycl::_V1::id"
// CHECK: [[ACC_RANGE2A:%[a-zA-Z0-9_.]+]] = alloca %"struct.sycl::_V1::range"
// CHECK: [[MEM_RANGE2A:%[a-zA-Z0-9_.]+]] = alloca %"struct.sycl::_V1::range"
// CHECK: [[OFFSET2A:%[a-zA-Z0-9_.]+]] = alloca %"struct.sycl::_V1::id"

// Check lambda object addrspacecast
// CHECK: [[LOCAL_OBJECT:%.*]] = addrspacecast ptr %__SYCLKernel to ptr addrspace(4)

// Check addrspacecast for ranges
// CHECK: [[ACC_RANGE1AS:%.*]] = addrspacecast ptr [[ACC_RANGE1A]] to ptr addrspace(4)
// CHECK: [[MEM_RANGE1AS:%.*]] = addrspacecast ptr [[MEM_RANGE1A]] to ptr addrspace(4)
// CHECK: [[OFFSET1AS:%.*]] = addrspacecast ptr [[OFFSET1A]] to ptr addrspace(4)
// CHECK: [[ACC_RANGE2AS:%.*]] = addrspacecast ptr [[ACC_RANGE2A]] to ptr addrspace(4)
// CHECK: [[MEM_RANGE2AS:%.*]] = addrspacecast ptr [[MEM_RANGE2A]] to ptr addrspace(4)
// CHECK: [[OFFSET2AS:%.*]] = addrspacecast ptr [[OFFSET2A]] to ptr addrspace(4)

// CHECK accessor array default inits
// CHECK: [[ACCESSOR_WRAPPER:%[a-zA-Z0-9_]+]] = getelementptr inbounds %class{{.*}}.anon, ptr addrspace(4) [[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: [[ACCESSOR_ARRAY1:%[a-zA-Z0-9_.]+]] = getelementptr inbounds %struct{{.*}}.struct_acc_t, ptr addrspace(4) [[ACCESSOR_WRAPPER]], i32 0, i32 0
// CHECK: [[BEGIN:%[a-zA-Z0-9._]*]] = getelementptr inbounds [2 x [[ACCESSOR:.*]]], ptr addrspace(4) [[ACCESSOR_ARRAY1]], i64 0, i64 0
// CTOR Call #1
// CHECK: call spir_func void @{{.+}}(ptr addrspace(4) {{[^,]*}} [[BEGIN]])
// CHECK: [[ELEM2_GEP:%[a-zA-Z0-9_.]+]] = getelementptr inbounds [[ACCESSOR]], ptr addrspace(4) [[BEGIN]], i64 1
// CTOR Call #2
// CHECK: call spir_func void @{{.+}}(ptr addrspace(4) {{[^,]*}} [[ELEM2_GEP]])

// Check acc[0] __init method call
// CHECK: [[GEP_LAMBDA1:%[a-zA-Z0-9_]+]] = getelementptr inbounds %class{{.*}}.anon, ptr addrspace(4) [[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: [[GEP_MEMBER_ACC1:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct{{.*}}.struct_acc_t, ptr addrspace(4) [[GEP_LAMBDA1]], i32 0, i32 0
// CHECK: [[ARRAY_IDX1:%[a-zA-Z0-9._]*]] = getelementptr inbounds [2 x [[ACCESSOR]]], ptr addrspace(4) [[GEP_MEMBER_ACC1]], i64 0, i64 0
// CHECK: [[MEM_LOAD1:%[a-zA-Z0-9_]+]] = load ptr addrspace(1), ptr addrspace(4) [[MEM_ARG1]].addr
// CHECK: [[ACC_RANGE1:%.*]] = addrspacecast ptr addrspace(4) [[ACC_RANGE1AS]] to ptr
// CHECK: [[MEM_RANGE1:%.*]] = addrspacecast ptr addrspace(4) [[MEM_RANGE1AS]] to ptr
// CHECK: [[OFFSET1:%.*]] = addrspacecast ptr addrspace(4) [[OFFSET1AS]] to ptr
// CHECK: call spir_func void @{{.*}}__init{{.*}}(ptr addrspace(4) {{[^,]*}} [[ARRAY_IDX1]], ptr addrspace(1) noundef [[MEM_LOAD1]], ptr noundef byval({{.*}}) align 4 [[ACC_RANGE1]], ptr noundef byval({{.*}}) align 4 [[MEM_RANGE1]], ptr noundef byval({{.*}}) align 4 [[OFFSET1]])

// Check acc[1] __init method call
// CHECK: [[GEP_LAMBDA2:%[a-zA-Z0-9_]+]] = getelementptr inbounds %class{{.*}}.anon, ptr addrspace(4) [[LOCAL_OBJECT]], i32 0, i32 0
// CHECK: [[GEP_MEMBER_ACC2:%[a-zA-Z0-9_]+]] = getelementptr inbounds %struct{{.*}}.struct_acc_t, ptr addrspace(4) [[GEP_LAMBDA2]], i32 0, i32 0
// CHECK: [[ARRAY_IDX2:%[a-zA-Z0-9_]*]] = getelementptr inbounds [2 x [[ACCESSOR]]], ptr addrspace(4) [[GEP_MEMBER_ACC2]], i64 0, i64 1
// CHECK: [[MEM_LOAD2:%[a-zA-Z0-9_]+]] = load ptr addrspace(1), ptr addrspace(4) [[MEM_ARG1]].addr
// CHECK: [[ACC_RANGE2:%.*]] = addrspacecast ptr addrspace(4) [[ACC_RANGE2AS]] to ptr
// CHECK: [[MEM_RANGE2:%.*]] = addrspacecast ptr addrspace(4) [[MEM_RANGE2AS]] to ptr
// CHECK: [[OFFSET2:%.*]] = addrspacecast ptr addrspace(4) [[OFFSET2AS]] to ptr
// CHECK: call spir_func void @{{.*}}__init{{.*}}(ptr addrspace(4) {{[^,]*}} [[ARRAY_IDX2]], ptr addrspace(1) noundef [[MEM_LOAD2]], ptr noundef byval({{.*}}) align 4 [[ACC_RANGE2]], ptr noundef byval({{.*}}) align 4 [[MEM_RANGE2]], ptr noundef byval({{.*}}) align 4 [[OFFSET2]])
