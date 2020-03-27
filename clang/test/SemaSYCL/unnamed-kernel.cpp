// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -fsycl-int-header=%t.h -fsyntax-only -verify %s
// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -fsycl-int-header=%t.h -fsycl-unnamed-lambda -fsyntax-only -verify %s
#include <sycl.hpp>

#ifdef __SYCL_UNNAMED_LAMBDA__
// expected-no-diagnostics
#endif

namespace namespace1 {
template <typename T>
class KernelName;
}

struct MyWrapper {
private:
  class InvalidKernelName0 {};
  class InvalidKernelName3 {};
  class InvalidKernelName4 {};
  class InvalidKernelName5 {};

public:
  void test() {
    cl::sycl::queue q;
#ifndef __SYCL_UNNAMED_LAMBDA__
    // expected-error@+5 {{kernel needs to have a globally-visible name}}
    // expected-note@+2 {{InvalidKernelName1 declared here}}
#endif
    class InvalidKernelName1 {};
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<InvalidKernelName1>([] {});
    });

#ifndef __SYCL_UNNAMED_LAMBDA__
    // expected-error@+5 {{kernel needs to have a globally-visible name}}
    // expected-note@+2 {{InvalidKernelName2 declared here}}
#endif
    class InvalidKernelName2 {};
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<namespace1::KernelName<InvalidKernelName2>>([] {});
    });

#ifndef __SYCL_UNNAMED_LAMBDA__
    // expected-error@+4 {{kernel needs to have a globally-visible name}}
    // expected-note@16 {{InvalidKernelName0 declared here}}
#endif
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<InvalidKernelName0>([] {});
    });

#ifndef __SYCL_UNNAMED_LAMBDA__
    // expected-error@+4 {{kernel needs to have a globally-visible name}}
    // expected-note@17 {{InvalidKernelName3 declared here}}
#endif
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<namespace1::KernelName<InvalidKernelName3>>([] {});
    });

    using ValidAlias = MyWrapper;
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<ValidAlias>([] {});
    });

    using InvalidAlias = InvalidKernelName4;
#ifndef __SYCL_UNNAMED_LAMBDA__
    // expected-error@+4 {{kernel needs to have a globally-visible name}}
    // expected-note@18 {{InvalidKernelName4 declared here}}
#endif
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<InvalidAlias>([] {});
    });

    using InvalidAlias1 = InvalidKernelName5;
#ifndef __SYCL_UNNAMED_LAMBDA__
    // expected-error@+4 {{kernel needs to have a globally-visible name}}
    // expected-note@19 {{InvalidKernelName5 declared here}}
#endif
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<namespace1::KernelName<InvalidAlias1>>([] {});
    });
  }
};

int main() {
  cl::sycl::queue q;
#ifndef __SYCL_UNNAMED_LAMBDA__
  // expected-error@+2 {{kernel name is missing}}
#endif
  q.submit([&](cl::sycl::handler &h) { h.single_task([] {}); });

  return 0;
}
