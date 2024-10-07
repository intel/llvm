// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fno-sycl-unnamed-lambda -fsyntax-only -verify %s -Werror=sycl-strict -DERROR
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fno-sycl-unnamed-lambda -fsyntax-only -verify %s  -Wsycl-strict -DWARN
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -verify %s -Werror=sycl-strict -DERROR

// This test verifies that incorrect kernel names are diagnosed correctly.

#include "sycl.hpp"

using namespace sycl;

// user-defined function
void function() {
}

// user-defined struct
struct myWrapper {
};

// user-declared class
class myWrapper2;

int main() {
  queue q;

  class InvalidKernelName1 {};
  // expected-error@#KernelSingleTask {{'InvalidKernelName1' is invalid; kernel name should be forward declarable at namespace scope}}
  // expected-note@+2 {{in instantiation of function template specialization}}
  q.submit([&](handler &h) {
    h.single_task<InvalidKernelName1>([]() {});
  });

  q.submit([&](handler &h) {
    h.single_task<class myWrapper>([]() { function(); });
  });

  q.submit([&](handler &h) {
    h.single_task<class myWrapper2>([]() { function(); });
  });
  return 0;
}
