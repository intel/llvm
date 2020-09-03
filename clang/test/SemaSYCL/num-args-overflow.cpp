// RUN: %clang_cc1 -I %S/Inputs -fsycl -triple spir64_gen -DGPU -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -I %S/Inputs -fsycl -triple spir64 -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -I %S/Inputs -fsycl -triple spir64_gen -Wno-sycl-strict -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -I %S/Inputs -fsycl -triple spir64_gen -Werror=sycl-strict -DERROR -fsycl-is-device -fsyntax-only -verify %s

#include <sycl.hpp>

template <typename Name, typename F>
__attribute__((sycl_kernel)) void kernel(F KernelFunc) {
  KernelFunc();
}

template <typename Name, typename F>
void parallel_for(F KernelFunc) {
#ifdef GPU
  // expected-warning@+8 {{kernel arguments size (7994 bytes) exceeds supported maximum of 2048 on GPU}}
  // expected-note@+7 {{array elements and fields of a class/struct may be counted separately}}
#elif ERROR
  // expected-error@+5 {{kernel arguments size (7994 bytes) exceeds supported maximum of 2048 on GPU}}
  // expected-note@+4 {{array elements and fields of a class/struct may be counted separately}}
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
  parallel_for<class Foo>(L);
}
