// RUN: %clang_cc1 -fsycl -triple spir64_gen -DGPU -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsycl -triple spir64 -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsycl -triple spir64_gen -Wno-sycl-strict -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsycl -triple spir64_gen -Werror=sycl-strict -DERROR -fsycl-is-device -fsyntax-only -verify %s

#include "Inputs/sycl.hpp"
class Foo;

template <typename Name, typename F>
__attribute__((sycl_kernel)) void kernel(F KernelFunc) {
  KernelFunc();
}

template <typename Name, typename F>
void parallel_for(F KernelFunc) {
#ifdef GPU
  // expected-warning@+6 {{size of kernel arguments (7994 bytes) exceeds supported maximum of 2048 bytes on GPU}}
#elif ERROR
  // expected-error@+4 {{size of kernel arguments (7994 bytes) exceeds supported maximum of 2048 bytes on GPU}}
#else
  // expected-no-diagnostics
#endif
  kernel<Name>(KernelFunc);
}

using Accessor =
    cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>;

void use() {
  struct S {
    int A;
    int B;
    Accessor AAcc;
    Accessor BAcc;
    int Array[1991];
  } Args;
  auto L = [=]() { (void)Args; };
#if defined(GPU) || defined(ERROR)
  // expected-note@+2 {{in instantiation of function template specialization 'parallel_for<Foo}}
#endif
  parallel_for<Foo>(L);
}
