// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -verify -fsyntax-only %s

#include "sycl.hpp"

using namespace sycl;
queue q;

struct FunctorWithoutCallOperator; // expected-note {{forward declaration of 'FunctorWithoutCallOperator'}}

int main() {
    // expected-error@#KernelSingleTask {{kernel parameter must be a lambda or function object}}
  q.submit([&](sycl::handler &cgh) {
    // expected-error@+2 {{invalid use of incomplete type 'FunctorWithoutCallOperator'}}
    // expected-note@+1 {{in instantiation of function template specialization}}
    cgh.single_task(FunctorWithoutCallOperator{});
  });

}
