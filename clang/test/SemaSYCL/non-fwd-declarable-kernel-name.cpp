// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fno-sycl-unnamed-lambda -fsyntax-only -sycl-std=2020 -fsycl-int-header=%t.h %s
// RUN: %clang_cc1 -fsycl-is-host -internal-isystem %S/Inputs -fno-sycl-unnamed-lambda -fsyntax-only -verify -include %t.h %s

// This test verifies that incorrect kernel names are diagnosed correctly.

#include "sycl.hpp"

using namespace cl::sycl;

// user-defined function
void function() {
}

// user-defined struct
struct myWrapper {
  class insideStruct;
};

template <typename KernelName> class RandomTemplate;

int main() {
  queue q;

  q.submit([&](handler &h) {
    h.single_task<class Ok>([]() { function(); });
  });
  q.submit([&](handler &h) {
    h.single_task<RandomTemplate<class Ok>>([]() { function(); });
  });

  class NotOk;
  // expected-error@#KernelSingleTask {{'NotOk' is invalid; kernel name should be forward declarable at namespace scope}}
  // expected-note@+2 {{in instantiation of function template specialization}}
  q.submit([&](handler &h) {
    h.single_task<class NotOk>([]() { function(); });
  });
  // expected-error@#KernelSingleTask {{'myWrapper::insideStruct' is invalid; kernel name should be forward declarable at namespace scope}}
  // expected-note@+2 {{in instantiation of function template specialization}}
  q.submit([&](handler &h) {
    h.single_task<class myWrapper::insideStruct>([]() { function(); });
  });
  // expected-error@#KernelSingleTask {{'RandomTemplate<NotOk>' is invalid; kernel name should be forward declarable at namespace scope}}
  // expected-note@+2 {{in instantiation of function template specialization}}
  q.submit([&](handler &h) {
    h.single_task<RandomTemplate<NotOk>>([]() { function(); });
  });
  return 0;
}
