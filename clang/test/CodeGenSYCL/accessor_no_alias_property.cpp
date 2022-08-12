// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// This test validates the behaviour of noalias parameter attribute.

#include "Inputs/sycl.hpp"

int main() {
  sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::global_buffer,
                     sycl::access::placeholder::false_t,
                     sycl::ext::oneapi::accessor_property_list<
                         sycl::ext::oneapi::property::no_alias::instance<true>>>
      accessorA;

  sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::global_buffer,
                     sycl::access::placeholder::false_t,
                     sycl::ext::oneapi::accessor_property_list<
                         sycl::ext::intel::property::buffer_location::instance<1>>>
      accessorB;

  // Check that noalias parameter attribute is emitted when no_alias accessor property is used
  // CHECK: define {{.*}}spir_kernel void @_ZTSZ4mainE16kernel_function1({{.*}} noalias {{.*}} %_arg_accessorA, {{.*}})
  sycl::kernel_single_task<class kernel_function1>(
      [=]() {
        accessorA.use();
      });

  // Check that noalias parameter attribute is NOT emitted when it is not used
  // CHECK: define {{.*}}spir_kernel void @_ZTSZ4mainE16kernel_function2
  // CHECK-NOT: noalias
  // CHECK-SAME: {
  sycl::kernel_single_task<class kernel_function2>(
      [=]() {
        accessorB.use();
      });
  return 0;
}
