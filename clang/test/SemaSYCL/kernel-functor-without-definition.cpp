// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -verify -fsyntax-only %s
// This test checks that an error is thrown when a functor without a call operator defined or 
// when a functor without a definition is used as a kernel.

#include "sycl.hpp"

using namespace sycl;
queue q;

struct FunctorWithoutCallOperator; // expected-note {{forward declaration of 'FunctorWithoutCallOperator'}}

struct StructDefined {
  int x;
};

class FunctorWithCallOpDefined {
  int x;
  public:
  void operator()() const {}
};

int main() {
  
  q.submit([&](sycl::handler &cgh) {
    // expected-error@#KernelSingleTask {{kernel parameter must be a lambda or function object}}
    // expected-error@+2 {{invalid use of incomplete type 'FunctorWithoutCallOperator'}}
    // expected-note@+1 {{in instantiation of function template specialization}}
    cgh.single_task(FunctorWithoutCallOperator{});
  });

  q.submit([&](sycl::handler &cgh) {
    // expected-error@#KernelSingleTaskKernelFuncCall {{type 'const StructDefined' does not provide a call operator}}
    // expected-note@#KernelSingleTask {{in instantiation of function template specialization 'sycl::kernel_single_task<StructDefined, StructDefined>' requested here}}
    // expected-note@+1 {{in instantiation of function template specialization 'sycl::handler::single_task<sycl::auto_name, StructDefined>' requested here}}
    cgh.single_task(StructDefined{});
  });
  
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task(FunctorWithCallOpDefined{});
  });

}
