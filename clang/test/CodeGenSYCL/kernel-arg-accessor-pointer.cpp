// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks if the metadata "kernel-arg-runtime-aligned"
// is generated if the kernel captures an accessor

#include "sycl.hpp"

using namespace cl::sycl;

queue q;

int main() {

  using Accessor =
      accessor<int, 1, access::mode::read_write, access::target::global_buffer>;
  Accessor acc[2];

  // kernel_A parameters : int*, sycl::range<1>, sycl::range<1>, sycl::id<1>,
  // int*, sycl::range<1>, sycl::range<1>,sycl::id<1>
  q.submit([&](cl::sycl::handler &h) {
    h.single_task<class kernel_A>([=]() {
      acc[1].use();
    });
  });

  // kernel_B parameters : none
  q.submit([&](cl::sycl::handler &h) {
    h.single_task<class kernel_B>([=]() {
      int result = 5;
    });
  });

  int a = 10;

  // kernel_C parameters : int
  q.submit([&](cl::sycl::handler &h) {
    h.single_task<class kernel_C>([=]() {
      int x = a;
    });
  });
}

// Check kernel_A parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_A
// CHECK-SAME: i32 addrspace(1)* [[MEM_ARG1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: %"struct.cl::sycl::range"* byval{{.*}}align 4 [[ACC_RANGE1:%[a-zA-Z0-9_]+_1]],
// CHECK-SAME: %"struct.cl::sycl::range"* byval{{.*}}align 4 [[MEM_RANGE1:%[a-zA-Z0-9_]+_2]],
// CHECK-SAME: %"struct.cl::sycl::id"* byval{{.*}}align 4 [[OFFSET1:%[a-zA-Z0-9_]+_3]],
// CHECK-SAME: i32 addrspace(1)* [[MEM_ARG2:%[a-zA-Z0-9_]+_4]],
// CHECK-SAME: %"struct.cl::sycl::range"* byval{{.*}}align 4 [[ACC_RANGE2:%[a-zA-Z0-9_]+_6]],
// CHECK-SAME: %"struct.cl::sycl::range"* byval{{.*}}align 4 [[MEM_RANGE2:%[a-zA-Z0-9_]+_7]],
// CHECK-SAME: %"struct.cl::sycl::id"* byval{{.*}}align 4 [[OFFSET2:%[a-zA-Z0-9_]+_8]])
// CHECK-SAME: !kernel_arg_runtime_aligned !5

// Check kernel_B parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_B
// CHECK-SAME: !kernel_arg_runtime_aligned !13

// Check kernel_C parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_C
// CHECK-SAME: i32 [[MEM_ARG1:%[a-zA-Z0-9_]+]]
// CHECK-SAME: !kernel_arg_runtime_aligned !15

// Check kernel-arg-runtime-aligned metadata.
// The value of any metadata element is 1 for any kernel arguments
// that corresponds to the base pointer of an accessor and 0 otherwise.
// CHECK: !5 = !{i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false}
// CHECK: !13 = !{}
// CHECK: !15 = !{i1 false}
