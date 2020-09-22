// RUN: %clang_cc1 -fsycl -triple spir64 -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsycl -triple spir64 -Werror=sycl-strict -DERROR -fsycl-is-device -fsyntax-only -verify %s

#include "Inputs/sycl.hpp"
class Foo;

template <typename Name, typename F>
__attribute__((sycl_kernel)) void kernel(F KernelFunc) {
  KernelFunc();
}

template <typename Name, typename F>
void parallel_for(F KernelFunc) {
#ifdef ERROR
  // expected-error@+6 {{size of kernel arguments (7994 bytes) may exceed supported maximum on some devices}}
  // expected-note@+5 {{supported maximum on some devices is 2048 bytes}}
#else
  // expected-warning@+3 {{size of kernel arguments (7994 bytes) may exceed supported maximum on some devices}}
  // expected-note@+2 {{supported maximum on some devices is 2048 bytes}}
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
  // expected-note@+1 {{in instantiation of function template specialization 'parallel_for<Foo}}
  parallel_for<Foo>(L);
}
