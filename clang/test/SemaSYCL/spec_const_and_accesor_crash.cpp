// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -ast-dump %s | FileCheck %s
// The test checks that Clang doesn't crash if a specialization constant gets
// into the kernel capture list before an accessor

#include "Inputs/sycl.hpp"

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  cl::sycl::ext::oneapi::experimental::spec_constant<char, class MyInt32Const> spec_const;
  cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write> accessor;
  // CHECK: FieldDecl {{.*}} implicit referenced spec_const 'cl::sycl::ext::oneapi::experimental::spec_constant<char, class MyInt32Const>'
  // CHECK: FieldDecl {{.*}} implicit referenced accessor 'cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write>'
  kernel<class MyKernel>([spec_const, accessor]() {});
  return 0;
}
