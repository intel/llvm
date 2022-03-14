// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// This test validates the behaviour of noalias parameter attribute.

#include "Inputs/sycl.hpp"

int main() {
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t,
                     cl::sycl::ext::oneapi::accessor_property_list<
                         cl::sycl::ext::oneapi::property::no_alias::instance<true>>>
      accessorA;

  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t,
                     cl::sycl::ext::oneapi::accessor_property_list<
                         cl::sycl::ext::intel::property::buffer_location::instance<1>>>
      accessorB;

  // Check that noalias parameter attribute is emitted when no_alias accessor property is used
  // CHECK: define {{.*}}spir_kernel void @_ZTSZ4mainE16kernel_function1({{.*}} noalias {{.*}} %_arg_accessorA, {{.*}})
  cl::sycl::kernel_single_task<class kernel_function1>(
      [=]() {
        accessorA.use();
      });

  // Check that noalias parameter attribute is NOT emitted when it is not used
  // CHECK: define {{.*}}spir_kernel void @_ZTSZ4mainE16kernel_function2
  // CHECK-NOT: noalias
  // CHECK-SAME: {
  cl::sycl::kernel_single_task<class kernel_function2>(
      [=]() {
        accessorB.use();
      });
  return 0;
}
