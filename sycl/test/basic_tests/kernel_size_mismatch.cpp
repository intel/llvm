// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note,warning %s

// Tests for static assertion failure when kernel lambda mismatches between host
// and device.

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  int A = 1;
  Q.single_task([=]() {
#ifdef __SYCL_DEVICE_ONLY__
     (void)A;
    // expected-no-diagnostics
#else
  // expected-error-re@sycl/handler.hpp:* {{static assertion failed due to requirement '{{.*}}': Unexpected kernel lambda size. This can be caused by an external host compiler producing a lambda with an unexpected layout. This is a limitation of the compiler.}}
#endif
   }).wait();
}
