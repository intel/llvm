// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -fsycl-is-device -emit-llvm %s -o - | FileCheck %s

// Test to check that readonly attribute is applied to accessors with access mode read.

#include "Inputs/sycl.hpp"

// CHECK-NOT: spir_kernel{{.*}}f0_kernel{{.*}}readonly
void f0(cl::sycl::queue &myQueue, cl::sycl::buffer<int, 1> &in_buf, cl::sycl::buffer<int, 1> &out_buf) {
  myQueue.submit([&](cl::sycl::handler &cgh) {
    auto write_acc = out_buf.get_access<cl::sycl::access::mode::write>(cgh);
    cgh.single_task<class f0_kernel>([write_acc] {});
  });
}

// CHECK: spir_kernel{{.*}}f1_kernel
// CHECK-NOT: readonly
// CHECK-SAME: %_arg_{{.*}}%_arg_1{{.*}}%_arg_2{{.*}}%_arg_3
// CHECK-SAME:  readonly %_arg_4
void f1(cl::sycl::queue &myQueue, cl::sycl::buffer<int, 1> &in_buf, cl::sycl::buffer<int, 1> &out_buf) {
  myQueue.submit([&](cl::sycl::handler &cgh) {
    auto write_acc = out_buf.get_access<cl::sycl::access::mode::write>(cgh);
    auto read_acc = in_buf.get_access<cl::sycl::access::mode::read>(cgh);
    cgh.single_task<class f1_kernel>([write_acc, read_acc] {});
  });
}

// CHECK: spir_kernel{{.*}}f2_kernel
// CHECK-SAME: readonly %_arg_
// CHECK-NOT: readonly
// CHECK-SAME: %_arg_8
void f2(cl::sycl::queue &myQueue, cl::sycl::buffer<int, 1> &in_buf, cl::sycl::buffer<int, 1> &out_buf) {
  myQueue.submit([&](cl::sycl::handler &cgh) {
    auto read_acc = in_buf.get_access<cl::sycl::access::mode::read>(cgh);
    auto write_acc = out_buf.get_access<cl::sycl::access::mode::write>(cgh);
    cgh.single_task<class f2_kernel>([read_acc, write_acc] {});
  });
}
