// RUN: %clang_cc1 -I %S/Inputs -fsycl-is-device -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct OpenCL kernel arguments for
// different accessors targets.

#include <sycl.hpp>

using namespace cl::sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {

  accessor<int, 1, access::mode::read_write,
           access::target::local>
      local_acc;
  accessor<int, 1, access::mode::read_write,
           access::target::global_buffer>
      global_acc;
  accessor<int, 1, access::mode::read_write,
           access::target::constant_buffer>
      constant_acc;
  kernel<class use_local>(
      [=]() {
        local_acc.use();
      });
  kernel<class use_global>(
      [=]() {
        global_acc.use();
      });
  kernel<class use_constant>(
      [=]() {
        constant_acc.use();
      });
}
// CHECK: {{.*}}use_local 'void (__attribute__((address_space(3))) int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>)'
// CHECK: {{.*}}use_global 'void (__attribute__((address_space(1))) int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>)'
// CHECK: {{.*}}use_constant 'void (__attribute__((address_space(2))) int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>)'
