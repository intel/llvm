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

#if defined(WARN)
  // expected-warning@#KernelSingleTask {{SYCL 1.2.1 specification requires an explicit forward declaration for a kernel type name; your program may not be portable}}
#elif defined(ERROR)
  // expected-error@#KernelSingleTask {{SYCL 1.2.1 specification requires an explicit forward declaration for a kernel type name; your program may not be portable}}
#endif

  q.submit([&](handler &h) {
  // expected-note@+2 {{fake_kernel declared here}}
  // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class fake_kernel>([]() { function(); });
  });

#if defined(WARN)
  // expected-warning@#KernelSingleTask {{SYCL 1.2.1 specification requires an explicit forward declaration for a kernel type name; your program may not be portable}}
#elif defined(ERROR)
  // expected-error@#KernelSingleTask {{SYCL 1.2.1 specification requires an explicit forward declaration for a kernel type name; your program may not be portable}}
#endif

  q.submit([&](handler &h) {
  // expected-note@+2 {{fake_kernel2 declared here}}
  // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class fake_kernel2>([]() {
      auto l = [](auto f) { f(); };
    });
  });

  q.submit([&](handler &h) {
    h.single_task<class myWrapper>([]() { function(); });
  });

  q.submit([&](handler &h) {
    h.single_task<class myWrapper2>([]() { function(); });
  });
  return 0;
}
