// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks that compiler generates correct kernel wrapper for basic
// case.

#include "Inputs/sycl.hpp"

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  sycl::accessor<int, 1, sycl::access::mode::read_write> accessorA;
    kernel<class kernel_function>(
      [=]() {
        accessorA.use();
      });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_function
// CHECK-SAME: ptr addrspace(1) noundef align 4 [[MEM_ARG:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[ACC_RANGE:%[a-zA-Z0-9_]+1]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[MEM_RANGE:%[a-zA-Z0-9_]+2]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[OFFSET:%[a-zA-Z0-9_]+]])
// Check alloca for pointer argument
// CHECK: [[MEM_ARG]].addr = alloca ptr addrspace(1)
// Check lambda object alloca
// CHECK: [[ANONALLOCA:%[a-zA-Z0-9_]+]] = alloca %class.anon
// Check allocas for ranges
// CHECK: [[ARANGEA:%agg.tmp.*]] = alloca %"struct.sycl::_V1::range"
// CHECK: [[MRANGEA:%agg.tmp.*]] = alloca %"struct.sycl::_V1::range"
// CHECK: [[OIDA:%agg.tmp.*]] = alloca %"struct.sycl::_V1::id"
// CHECK: [[ANON:%[a-zA-Z0-9_.]+]] = addrspacecast ptr [[ANONALLOCA]] to ptr addrspace(4)
//
// Check store of kernel pointer argument to alloca
// CHECK: store ptr addrspace(1) [[MEM_ARG]], ptr addrspace(4) [[MEM_ARG]].addr.ascast, align 8

// Check for default constructor of accessor
// CHECK: call spir_func {{.*}}accessor

// Check accessor GEP
// CHECK: [[ACCESSOR:%[a-zA-Z0-9_]+]] = getelementptr inbounds nuw %class.anon, ptr addrspace(4) [[ANON]], i32 0, i32 0

// Check load from kernel pointer argument alloca
// CHECK: [[MEM_LOAD:%[a-zA-Z0-9_]+]] = load ptr addrspace(1), ptr addrspace(4) [[MEM_ARG]].addr.ascast

// Check accessor __init method call
// CHECK: call spir_func void @{{.*}}__init{{.*}}(ptr addrspace(4) {{[^,]*}} [[ACCESSOR]],
// CHECK-SAME: ptr addrspace(1) noundef [[MEM_LOAD]],
// CHECK-SAME: ptr noundef byval({{.*}}) align 4 [[ARANGEA]],
// CHECK-SAME: ptr noundef byval({{.*}}) align 4 [[MRANGEA]],
// CHECK-SAME: ptr noundef byval({{.*}}) align 4 [[OIDA]])

// Check lambda "()" operator call
// CHECK: call spir_func void @{{.*}}(ptr addrspace(4) {{[^,]*}})
