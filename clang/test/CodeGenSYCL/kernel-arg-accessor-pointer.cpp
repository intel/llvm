// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -opaque-pointers -emit-llvm %s -o - | FileCheck %s

// This test checks if the metadata "kernel-arg-runtime-aligned" and "kernel_arg_exclusive_ptr"
// are generated if the kernel captures an accessor.

#include "sycl.hpp"

using namespace sycl;

queue q;

int main() {

  using Accessor =
      accessor<int, 1, access::mode::read_write, access::target::global_buffer>;
  Accessor acc[2];

  accessor<int, 1, access::mode::read, access::target::global_buffer> readOnlyAccessor;

  accessor<float, 2, access::mode::write,
           access::target::local,
           access::placeholder::true_t>
      acc3;

  local_accessor<float, 2> acc4;

  // kernel_A parameters : int*, sycl::range<1>, sycl::range<1>, sycl::id<1>,
  // int*, sycl::range<1>, sycl::range<1>,sycl::id<1>.
  q.submit([&](handler &h) {
    h.single_task<class kernel_A>([=]() {
      acc[1].use();
    });
  });

  // kernel_readOnlyAcc parameters : int*, sycl::range<1>, sycl::range<1>, sycl::id<1>.
  q.submit([&](handler &h) {
    h.single_task<class kernel_readOnlyAcc>([=]() {
      readOnlyAccessor.use();
    });
  });

  // kernel_B parameters : none.
  q.submit([&](handler &h) {
    h.single_task<class kernel_B>([=]() {
      int result = 5;
    });
  });

  int a = 10;

  // kernel_C parameters : int.
  q.submit([&](handler &h) {
    h.single_task<class kernel_C>([=]() {
      int x = a;
    });
  });

  // Using raw pointers to represent USM pointers.
  // kernel_arg_runtime_aligned is not generated for raw pointers.
  int *x;
  float *y;
  q.submit([&](handler &h) {
    h.single_task<class usm_ptr>([=]() {
      *x = 42;
      *y = 3.14;
    });
  });

  // Using local accessor as a kernel parameter.
  // kernel_arg_runtime_aligned is generated for pointers from local accessors.
  q.submit([&](handler &h) {
    h.single_task<class localAccessorDep>([=]() {
      acc3.use();
    });
  });

  // Using local_accessor as a kernel parameter.
  // kernel_arg_runtime_aligned is generated for pointers from local accessors.
  q.submit([&](handler &h) {
    h.single_task<class localAccessor>([=]() {
      acc4.use();
    });
  });

  // kernel_acc_raw_ptr parameters : int*, sycl::range<1>, sycl::range<1>, sycl::id<1>, int*.
  int *rawPtr;
  q.submit([&](handler &h) {
    h.single_task<class kernel_acc_raw_ptr>([=]() {
      readOnlyAccessor.use();
      *rawPtr = 10;
    });
  });

  // Check if kernel_arg_accessor_ptr metadata is generated for ESIMD kernels that capture
  // an accessor.
  q.submit([&](handler &h) {
    h.single_task<class esimd_kernel_with_acc>([=]() __attribute__((sycl_explicit_simd)) {
      readOnlyAccessor.use();
    });
  });
}

// Check kernel_A parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_A
// CHECK-SAME: ptr addrspace(1) noundef align 4 [[MEM_ARG1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[ACC_RANGE1:%[a-zA-Z0-9_]+1]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[MEM_RANGE1:%[a-zA-Z0-9_]+2]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[OFFSET1:%[a-zA-Z0-9_]+3]],
// CHECK-SAME: ptr addrspace(1) noundef align 4 [[MEM_ARG2:%[a-zA-Z0-9_]+4]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[ACC_RANGE2:%[a-zA-Z0-9_]+6]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[MEM_RANGE2:%[a-zA-Z0-9_]+7]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[OFFSET2:%[a-zA-Z0-9_]+8]])
// CHECK-SAME: !kernel_arg_runtime_aligned ![[#ACCESSORMD1:]]
// CHECK-SAME: !kernel_arg_exclusive_ptr ![[#ACCESSORMD1]]

// Check kernel_readOnlyAcc parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_readOnlyAcc
// CHECK-SAME: ptr addrspace(1) noundef readonly align 4 [[MEM_ARG1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[ACC_RANGE1:%[a-zA-Z0-9_]+1]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[MEM_RANGE1:%[a-zA-Z0-9_]+2]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[OFFSET1:%[a-zA-Z0-9_]+3]]
// CHECK-SAME: !kernel_arg_runtime_aligned ![[#ACCESSORMD2:]]
// CHECK-SAME: !kernel_arg_exclusive_ptr ![[#ACCESSORMD2]]

// Check kernel_B parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_B
// CHECK-NOT: kernel_arg_runtime_aligned
// CHECK-NOT: kernel_arg_exclusive_ptr

// Check kernel_C parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_C
// CHECK-SAME: i32 noundef [[MEM_ARG1:%[a-zA-Z0-9_]+]]
// CHECK-NOT: kernel_arg_runtime_aligned
// CHECK-NOT: kernel_arg_exclusive_ptr

// Check usm_ptr parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}usm_ptr
// CHECK-SAME: ptr addrspace(1) noundef align 4 [[MEM_ARG1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr addrspace(1) noundef align 4 [[MEM_ARG1:%[a-zA-Z0-9_]+]]
// CHECK-NOT: kernel_arg_runtime_aligned
// CHECK-NOT: kernel_arg_exclusive_ptr

// CHECK: define {{.*}}spir_kernel void @{{.*}}localAccessorDep
// CHECK-SAME: ptr addrspace(1) noundef align 4 [[MEM_ARG1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[ACC_RANGE1:%[a-zA-Z0-9_]+1]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[MEM_RANGE1:%[a-zA-Z0-9_]+2]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[OFFSET1:%[a-zA-Z0-9_]+3]]
// CHECK-SAME: !kernel_arg_runtime_aligned ![[#ACCESSORMD2]]
// CHECK-SAME: !kernel_arg_exclusive_ptr ![[#ACCESSORMD2]]

// CHECK: define {{.*}}spir_kernel void @{{.*}}localAccessor
// CHECK-SAME: ptr addrspace(3) noundef align 4 [[MEM_ARG1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[ACC_RANGE1:%[a-zA-Z0-9_]+1]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[MEM_RANGE1:%[a-zA-Z0-9_]+2]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[OFFSET1:%[a-zA-Z0-9_]+3]]
// CHECK-SAME: !kernel_arg_runtime_aligned ![[#ACCESSORMD2]]
// CHECK-SAME: !kernel_arg_exclusive_ptr ![[#ACCESSORMD2]]

// Check kernel_acc_raw_ptr parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_acc_raw_ptr
// CHECK-SAME: ptr addrspace(1) noundef readonly align 4 [[MEM_ARG1:%[a-zA-Z0-9_]+]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[ACC_RANGE1:%[a-zA-Z0-9_]+1]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[MEM_RANGE1:%[a-zA-Z0-9_]+2]],
// CHECK-SAME: ptr noundef byval{{.*}}align 4 [[OFFSET1:%[a-zA-Z0-9_]+3]]
// CHECK-SAME: ptr addrspace(1) noundef align 4 [[MEM_ARG1:%[a-zA-Z0-9_]+]]
// CHECK-SAME: !kernel_arg_runtime_aligned ![[#ACCESSORMD3:]]
// CHECK-SAME: !kernel_arg_exclusive_ptr ![[#ACCESSORMD3]]

// Check esimd_kernel_with_acc parameters
// CHECK: define {{.*}}spir_kernel void @{{.*}}esimd_kernel_with_acc
// CHECK-SAME: !kernel_arg_accessor_ptr

// Check kernel-arg-runtime-aligned metadata.
// The value of any metadata element is 1 for any kernel arguments
// that corresponds to the base pointer of an accessor and 0 otherwise.
// CHECK: ![[#ACCESSORMD1]] = !{i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false}
// CHECK: ![[#ACCESSORMD2]] = !{i1 true, i1 false, i1 false, i1 false}
// CHECK: ![[#ACCESSORMD3]] = !{i1 true, i1 false, i1 false, i1 false, i1 false}
