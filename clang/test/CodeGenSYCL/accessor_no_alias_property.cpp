// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -emit-llvm %s -o - | FileCheck %s
// check that noalias parameter attribute is emitted when no_alias accessor property is used
// CHECK: define {{.*}}spir_kernel void @_ZTSZ4mainE16kernel_function1({{.*}} noalias {{.*}} %_arg_, {{.*}})

// check that noalias parameter attribute is NOT emitted when it is not used
// CHECK: define {{.*}}spir_kernel void @_ZTSZ4mainE16kernel_function2{{.*}} !kernel_arg_buffer_location
// CHECK-NOT: define {{.*}}spir_kernel void @_ZTSZ4mainE16kernel_function2({{.*}} noalias {{.*}}

#include "Inputs/sycl.hpp"

int main() {
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t,
                     cl::sycl::ONEAPI::accessor_property_list<
                         cl::sycl::ONEAPI::property::no_alias::instance<true>>>
      accessorA;

  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t,
                     cl::sycl::ONEAPI::accessor_property_list<
                         cl::sycl::INTEL::property::buffer_location::instance<1>>>
      accessorB;

  cl::sycl::kernel_single_task<class kernel_function1>(
      [=]() {
        accessorA.use();
      });

  cl::sycl::kernel_single_task<class kernel_function2>(
      [=]() {
        accessorB.use();
      });
  return 0;
}
