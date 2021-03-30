// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -internal-isystem %S/Inputs -fsyntax-only -Wsycl-strict -sycl-std=2020 -verify %s -DSPIR64
// RUN: %clang_cc1 -fsycl-is-device -triple spir -internal-isystem %S/Inputs -fsyntax-only -Wsycl-strict -sycl-std=2020 -verify %s -DSPIR32

#include "sycl.hpp"
class Foo;

using namespace cl::sycl;

queue q;

using Accessor =
    accessor<int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer>;
#ifdef SPIR64
// expected-warning@#KernelSingleTask {{size of kernel arguments (7994 bytes) may exceed the supported maximum of 2048 bytes on some devices}}
#elif SPIR32
// expected-warning@#KernelSingleTask {{size of kernel arguments (7986 bytes) may exceed the supported maximum of 2048 bytes on some devices}}
#endif

void use() {
  struct S {
    int A;
    int B;
    Accessor AAcc;
    Accessor BAcc;
    int Array[1991];
  } Args;
  auto L = [=]() { (void)Args; };
  q.submit([&](handler &h) {
    // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class Foo>(L);
  });
}
