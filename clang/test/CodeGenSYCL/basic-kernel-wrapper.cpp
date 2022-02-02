// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks that compiler generates correct kernel wrapper for basic
// case.

#include "Inputs/sycl.hpp"

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write> accessorA;
    kernel<class kernel_function>(
      [=]() {
        accessorA.use();
      });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_function
// CHECK-SAME: i32 addrspace(1)* noundef [[MEM_ARG:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %"struct.cl::sycl::range"* noundef byval{{.*}}align 4 [[ACC_RANGE:%[a-zA-Z0-9_]+_1]],
// CHECK-SAME: %"struct.cl::sycl::range"* noundef byval{{.*}}align 4 [[MEM_RANGE:%[a-zA-Z0-9_]+_2]],
// CHECK-SAME: %"struct.cl::sycl::id"* noundef byval{{.*}}align 4 [[OFFSET:%[a-zA-Z0-9_]+]])
// Check alloca for pointer argument
// CHECK: [[MEM_ARG]].addr = alloca i32 addrspace(1)*
// Check lambda object alloca
// CHECK: [[ANONALLOCA:%[0-9]+]] = alloca %class.anon
// Check allocas for ranges
// CHECK: [[ARANGEA:%agg.tmp.*]] = alloca %"struct.cl::sycl::range"
// CHECK: [[MRANGEA:%agg.tmp.*]] = alloca %"struct.cl::sycl::range"
// CHECK: [[OIDA:%agg.tmp.*]] = alloca %"struct.cl::sycl::id"
// CHECK: [[ANON:%[0-9]+]] = addrspacecast %class.anon* [[ANONALLOCA]] to %class.anon addrspace(4)*
// CHECK: [[ARANGET:%agg.tmp.*]] = addrspacecast %"struct.cl::sycl::range"* [[ARANGEA]] to %"struct.cl::sycl::range" addrspace(4)*
// CHECK: [[MRANGET:%agg.tmp.*]] = addrspacecast %"struct.cl::sycl::range"* [[MRANGEA]] to %"struct.cl::sycl::range" addrspace(4)*
// CHECK: [[OIDT:%agg.tmp.*]] = addrspacecast %"struct.cl::sycl::id"* [[OIDA]] to %"struct.cl::sycl::id" addrspace(4)*
//
// Check store of kernel pointer argument to alloca
// CHECK: store i32 addrspace(1)* [[MEM_ARG]], i32 addrspace(1)* addrspace(4)* [[MEM_ARG]].addr.ascast, align 8

// Check for default constructor of accessor
// CHECK: call spir_func {{.*}}accessor

// Check accessor GEP
// CHECK: [[ACCESSOR:%[a-zA-Z0-9_]+]] = getelementptr inbounds %class.anon, %class.anon addrspace(4)* [[ANON]], i32 0, i32 0

// Check load from kernel pointer argument alloca
// CHECK: [[MEM_LOAD:%[a-zA-Z0-9_]+]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* [[MEM_ARG]].addr.ascast

// Check accessor __init method call
// CHECK: [[ARANGE:%agg.tmp.*]] = addrspacecast %"struct.cl::sycl::range" addrspace(4)* [[ARANGET]] to %"struct.cl::sycl::range"*
// CHECK: [[MRANGE:%agg.tmp.*]] = addrspacecast %"struct.cl::sycl::range" addrspace(4)* [[MRANGET]] to %"struct.cl::sycl::range"*
// CHECK: [[OID:%agg.tmp.*]] = addrspacecast %"struct.cl::sycl::id" addrspace(4)* [[OIDT]] to %"struct.cl::sycl::id"*
// CHECK: call spir_func void @{{.*}}__init{{.*}}(%"class.cl::sycl::accessor" addrspace(4)* {{[^,]*}} [[ACCESSOR]],
// CHECK-SAME: i32 addrspace(1)* noundef [[MEM_LOAD]],
// CHECK-SAME: %"struct.cl::sycl::range"* noundef byval({{.*}}) align 4 [[ARANGE]],
// CHECK-SAME: %"struct.cl::sycl::range"* noundef byval({{.*}}) align 4 [[MRANGE]],
// CHECK-SAME: %"struct.cl::sycl::id"* noundef byval({{.*}}) align 4 [[OID]])

// Check lambda "()" operator call
// CHECK: call spir_func void @{{.*}}(%class.anon addrspace(4)* {{[^,]*}})
