// RUN: %clangxx -fsycl -fsycl-device-only -Xclang -verify -Wsycl-strict %s -fsyntax-only

#include <CL/sycl.hpp>
class Foo;

using namespace cl::sycl;

// expected-warning@../../include/sycl/CL/sycl/handler.hpp:919 {{size of kernel arguments (8068 bytes) may exceed the supported maximum of 2048 bytes on some devices}}

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
