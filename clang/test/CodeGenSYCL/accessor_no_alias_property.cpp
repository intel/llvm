// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -emit-llvm %s -o - | FileCheck %s

// CHECK: define {{.*}}spir_kernel void @_ZTSZ4mainE15kernel_function{{.*}} !noalias

#include "Inputs/sycl.hpp"

int main() {
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::false_t,
                     cl::sycl::ONEAPI::accessor_property_list<
                         cl::sycl::ONEAPI::property::no_alias::instance<true>>>
      accessorA;
  cl::sycl::kernel_single_task<class kernel_function>(
      [=]() {
        accessorA.use();
      });
  return 0;
}
