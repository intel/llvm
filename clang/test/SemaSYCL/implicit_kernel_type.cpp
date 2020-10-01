// RUN: %clang_cc1 -fsycl -fsycl-is-device -internal-isystem %S/Inputs -fsycl-int-header=%t.h -fsyntax-only -verify %s -Werror=sycl-strict -DERROR
// RUN: %clang_cc1 -fsycl -fsycl-is-device -internal-isystem %S/Inputs -fsycl-int-header=%t.h -fsyntax-only -verify %s  -Wsycl-strict -DWARN
// RUN: %clang_cc1 -fsycl -fsycl-is-device -internal-isystem %S/Inputs -fsycl-int-header=%t.h -fsycl-unnamed-lambda -fsyntax-only -verify %s  -Werror=sycl-strict

#include "Inputs/sycl.hpp"

using namespace cl::sycl;

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

#if defined(WARN)
  // expected-warning@Inputs/sycl.hpp:211 {{Passing of kernel functions by reference is a SYCL 2020 extension}}
  // expected-error@Inputs/sycl.hpp:220 {{kernel needs to have a globally-visible name}}
  // expected-note@+8 {{InvalidKernelName1 declared here}}
#elif defined(ERROR)
  // expected-error@Inputs/sycl.hpp:211 {{Passing of kernel functions by reference is a SYCL 2020 extension}}
  // expected-error@Inputs/sycl.hpp:220 {{kernel needs to have a globally-visible name}}
  // expected-note@+4 {{InvalidKernelName1 declared here}}
#elif defined(__SYCL_UNNAMED_LAMBDA__)
  // expected-error@Inputs/sycl.hpp:211 {{Passing of kernel functions by reference is a SYCL 2020 extension}}
#endif
  class InvalidKernelName1 {};
  // expected-note@+2 {{in instantiation of function template specialization}}
  q.submit([&](handler &h) {
    h.single_task<InvalidKernelName1>([]() {});
  });

#if defined(WARN)
  // expected-warning@Inputs/sycl.hpp:220 {{SYCL 1.2.1 specification requires an explicit forward declaration for a kernel type name; your program may not be portable}}
  // expected-warning@Inputs/sycl.hpp:211 {{Passing of kernel functions by reference is a SYCL 2020 extension}}
#elif defined(ERROR)
  // expected-error@Inputs/sycl.hpp:220 {{SYCL 1.2.1 specification requires an explicit forward declaration for a kernel type name; your program may not be portable}}
  // expected-error@Inputs/sycl.hpp:211 {{Passing of kernel functions by reference is a SYCL 2020 extension}}
#elif defined(__SYCL_UNNAMED_LAMBDA__)
  // expected-error@Inputs/sycl.hpp:211 {{Passing of kernel functions by reference is a SYCL 2020 extension}}
#endif

  q.submit([&](handler &h) {
#ifndef __SYCL_UNNAMED_LAMBDA__
  // expected-note@+3 {{fake_kernel declared here}}
#endif
    // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class fake_kernel>([]() { function(); });
  });

#if defined(WARN)
  // expected-warning@Inputs/sycl.hpp:220 {{SYCL 1.2.1 specification requires an explicit forward declaration for a kernel type name; your program may not be portable}}
  // expected-warning@Inputs/sycl.hpp:211 3{{Passing of kernel functions by reference is a SYCL 2020 extension}}
#elif defined(ERROR)
  // expected-error@Inputs/sycl.hpp:220 {{SYCL 1.2.1 specification requires an explicit forward declaration for a kernel type name; your program may not be portable}}
  // expected-error@Inputs/sycl.hpp:211 3{{Passing of kernel functions by reference is a SYCL 2020 extension}}
#elif defined(__SYCL_UNNAMED_LAMBDA__)
  // expected-error@Inputs/sycl.hpp:211 3{{Passing of kernel functions by reference is a SYCL 2020 extension}}
#endif

  q.submit([&](handler &h) {
#ifndef __SYCL_UNNAMED_LAMBDA__
  // expected-note@+3 {{fake_kernel2 declared here}}
#endif
    // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class fake_kernel2>([]() {
      auto l = [](auto f) { f(); };
    });
  });

  q.submit([&](handler &h) {
    // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class myWrapper>([]() { function(); });
  });

  q.submit([&](handler &h) {
    // expected-note@+1 {{in instantiation of function template specialization}}
    h.single_task<class myWrapper2>([]() { function(); });
  });
  return 0;
}
