// RUN: %clang_cc1 -fsycl -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -Wsycl-strict -sycl-std=2020 -verify %s

#include "Inputs/sycl.hpp"
class kernel;

using namespace cl::sycl;

// expected-warning@Inputs/sycl.hpp:220 {{size of kernel arguments (8068 bytes) may exceed the supported maximum of 2048 bytes on some devices}}

int main() {

  struct S {
    int A;
    int B;
    int Array[2015];
  } Args;

  queue myQueue;

  myQueue.submit([&](handler &cgh) {
    // expected-note@+1 {{in instantiation of function template specialization 'cl::sycl::handler::single_task}}
    cgh.single_task<class kernel>([=]() { (void)Args; });
  });
  return 0;
}
