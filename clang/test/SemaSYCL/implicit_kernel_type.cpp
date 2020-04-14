// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -fsycl-int-header=%t.h -fsyntax-only -verify %s -Werror=sycl-strict -DERROR
// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -fsycl-int-header=%t.h -fsyntax-only -verify %s -Wsycl-strict -DWARN
// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -fsycl-int-header=%t.h -fsycl-unnamed-lambda -fsyntax-only -verify %s -Werror=sycl-strict
#include <sycl.hpp>

#ifdef __SYCL_UNNAMED_LAMBDA__
// expected-no-diagnostics
#endif

using namespace cl::sycl;

void function() {
}

// user-defined class
struct myWrapper {
};

// user-declared class
class myWrapper2;

int main() {
  cl::sycl::queue q;
#ifndef __SYCL_UNNAMED_LAMBDA__
  // expected-note@+1 {{InvalidKernelName1 declared here}}
  class InvalidKernelName1 {};
  q.submit([&](cl::sycl::handler &h) {
    // expected-error@+1 {{kernel needs to have a globally-visible name}}
    h.single_task<InvalidKernelName1>([]() {});
  });
#endif
#if defined(WARN)
  // expected-warning@+6 {{SYCL 1.2.1 specification requires an explicit forward declaration for a kernel type name; your program may not be portable}}
  // expected-note@+5 {{fake_kernel declared here}}
#elif defined(ERROR)
  // expected-error@+3 {{SYCL 1.2.1 specification requires an explicit forward declaration for a kernel type name; your program may not be portable}}
  // expected-note@+2 {{fake_kernel declared here}}
#endif
  cl::sycl::kernel_single_task<class fake_kernel>([]() { function(); });
#if defined(WARN)
  // expected-warning@+6 {{SYCL 1.2.1 specification requires an explicit forward declaration for a kernel type name; your program may not be portable}}
  // expected-note@+5 {{fake_kernel2 declared here}}
#elif defined(ERROR)
  // expected-error@+3 {{SYCL 1.2.1 specification requires an explicit forward declaration for a kernel type name; your program may not be portable}}
  // expected-note@+2 {{fake_kernel2 declared here}}
#endif
  cl::sycl::kernel_single_task<class fake_kernel2>([]() {
    auto l = [](auto f) { f(); };
  });
  cl::sycl::kernel_single_task<class myWrapper>([]() { function(); });
  cl::sycl::kernel_single_task<class myWrapper2>([]() { function(); });
  return 0;
}
