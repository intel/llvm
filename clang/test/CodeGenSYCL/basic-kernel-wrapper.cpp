// RUN: DISABLE_INFER_AS=1 %clang_cc1 -I %S/Inputs -triple spir64-unknown-linux-sycldevice -std=c++11 -fsycl-is-device -disable-llvm-passes -S -emit-llvm %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-OLD
// RUN: %clang_cc1 -I %S/Inputs -triple spir64-unknown-linux-sycldevice -std=c++11 -fsycl-is-device -disable-llvm-passes -S -emit-llvm %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-NEW

// This test checks that compiler generates correct kernel wrapper for basic
// case.

#include "sycl.hpp"

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
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

// CHECK: define spir_kernel void @{{.*}}kernel_function
// CHECK-SAME: i32 addrspace(1)* [[MEM_ARG:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %"struct.{{.*}}.cl::sycl::range"* byval{{.*}}align 4 [[ACC_RANGE:%[a-zA-Z0-9_]+_1]],
// CHECK-SAME: %"struct.{{.*}}.cl::sycl::range"* byval{{.*}}align 4 [[MEM_RANGE:%[a-zA-Z0-9_]+_2]],
// CHECK-SAME: %"struct.{{.*}}.cl::sycl::id"* byval{{.*}}align 4 [[OFFSET:%[a-zA-Z0-9_]+]])
// Check alloca for pointer argument
// CHECK: [[MEM_ARG]].addr = alloca i32 addrspace(1)*
// Check lambda object alloca
// CHECK: [[ANON:%[0-9]+]] = alloca %"class.{{.*}}.anon"
// Check allocas for ranges
// CHECK: [[ARANGE:%agg.tmp.*]] = alloca %"struct.{{.*}}.cl::sycl::range"
// CHECK: [[MRANGE:%agg.tmp.*]] = alloca %"struct.{{.*}}.cl::sycl::range"
// CHECK: [[OID:%agg.tmp.*]] = alloca %"struct.{{.*}}.cl::sycl::id"
//
// Check store of kernel pointer argument to alloca
// CHECK: store i32 addrspace(1)* [[MEM_ARG]], i32 addrspace(1)** [[MEM_ARG]].addr, align 8

// Check for default constructor of accessor
// CHECK: call spir_func {{.*}}accessor

// Check accessor GEP
// CHECK: [[ACCESSOR:%[a-zA-Z0-9_]+]] = getelementptr inbounds %"class.{{.*}}.anon", %"class.{{.*}}.anon"* [[ANON]], i32 0, i32 0

// Check load from kernel pointer argument alloca
// CHECK: [[MEM_LOAD:%[a-zA-Z0-9_]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[MEM_ARG]].addr

// Check accessor __init method call
// CHECK-OLD: call spir_func void @{{.*}}__init{{.*}}(%"class.{{.*}}.cl::sycl::accessor"* [[ACCESSOR]], i32 addrspace(1)* [[MEM_LOAD]], %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[ARANGE]], %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[MRANGE]], %"struct.{{.*}}.cl::sycl::id"* byval({{.*}}) align 4 [[OID]])
// CHECK-NEW: [[ACCESSORCAST:%[0-9]+]] = addrspacecast %"class{{.*}}accessor"* [[ACCESSOR]] to %"class{{.*}}accessor" addrspace(4)*
// CHECK-NEW: call spir_func void @{{.*}}__init{{.*}}(%"class.{{.*}}.cl::sycl::accessor" addrspace(4)* [[ACCESSORCAST]], i32 addrspace(1)* [[MEM_LOAD]], %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[ARANGE]], %"struct.{{.*}}.cl::sycl::range"* byval({{.*}}) align 4 [[MRANGE]], %"struct.{{.*}}.cl::sycl::id"* byval({{.*}}) align 4 [[OID]])

// Check lambda "()" operator call
// CHECK-OLD: call spir_func void @{{.*}}(%"class.{{.*}}.anon"* [[ANON]])
// CHECK-NEW: [[ANONCAST:%[0-9]+]] = addrspacecast %"class{{.*}}anon"* {{.*}} to %"class{{.*}}anon" addrspace(4)*
// CHECK-NEW: call spir_func void @{{.*}}(%"class.{{.*}}.anon" addrspace(4)* [[ANONCAST]])

// TODO: SYCL specific fail - analyze and enable
// XFAIL: windows-msvc
