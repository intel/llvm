// RUN: %clang %s -fsyntax-only -fsycl-device-only -DCHECKDIAG -Xclang -verify
// RUN: %clang %s -fsyntax-only -I %S/Inputs -Xclang -ast-dump -fsycl-device-only | FileCheck %s

#ifndef CHECKDIAG
#include "sycl.hpp"
#endif // CHECKDIAG

#ifdef CHECKDIAG
struct FuncObj {
  [[intelfpga::kernel_arg_buffer_location]] // expected-warning{{'kernel_arg_buffer_location' attribute cannot be used explicitly}}
  void operator()() {}
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
  [[intelfpga::kernel_arg_buffer_location]] int invalid = 42; // expected-error{{'kernel_arg_buffer_location' attribute only applies to functions}}
}
#endif // CHECKDIAG

int main() {
#ifdef CHECKDIAG
  kernel<class test_kernel1>(
      FuncObj());
#else
  // CHECK-LABEL: FunctionDecl {{.*}} _ZTSZ4mainE15kernel_function
  // CHECK:       SYCLIntelBufferLocationAttr
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write> accessorA;
  cl::sycl::kernel_single_task<class kernel_function>(
      [=]() {
        accessorA.use();
      });
#endif // CHECKDIAG
}
