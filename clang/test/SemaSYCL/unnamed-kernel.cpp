// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsycl-int-header=%t.h -fsyntax-only -Wno-sycl-2017-compat -verify %s
// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsycl-int-header=%t.h -fsycl-unnamed-lambda -fsyntax-only -Wno-sycl-2017-compat -verify %s
#include "Inputs/sycl.hpp"

#ifdef __SYCL_UNNAMED_LAMBDA__
// expected-no-diagnostics
#endif

namespace namespace1 {
template <typename T>
class KernelName;
}

namespace std {
typedef struct {
} max_align_t;
} // namespace std

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
    // expected-error@Inputs/sycl.hpp:220 {{kernel needs to have a globally-visible name}}
    // expected-note@+3 {{InvalidKernelName1 declared here}}
    // expected-note@+4{{in instantiation of function template specialization}}
#endif
    class InvalidKernelName1 {};
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<InvalidKernelName1>([] {});
    });

#ifndef __SYCL_UNNAMED_LAMBDA__
    // expected-error@Inputs/sycl.hpp:220 {{kernel needs to have a globally-visible name}}
    // expected-note@+3 {{InvalidKernelName2 declared here}}
    // expected-note@+4{{in instantiation of function template specialization}}
#endif
    class InvalidKernelName2 {};
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<namespace1::KernelName<InvalidKernelName2>>([] {});
    });

#ifndef __SYCL_UNNAMED_LAMBDA__
    // expected-error@Inputs/sycl.hpp:220 {{kernel needs to have a globally-visible name}}
    // expected-note@21 {{InvalidKernelName0 declared here}}
    // expected-note@+3{{in instantiation of function template specialization}}
#endif
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<InvalidKernelName0>([] {});
    });

#ifndef __SYCL_UNNAMED_LAMBDA__
    // expected-error@Inputs/sycl.hpp:220 {{kernel needs to have a globally-visible name}}
    // expected-note@22 {{InvalidKernelName3 declared here}}
    // expected-note@+3{{in instantiation of function template specialization}}
#endif
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<namespace1::KernelName<InvalidKernelName3>>([] {});
    });

    using ValidAlias = MyWrapper;
    q.submit([&](cl::sycl::handler &h) {
      
      h.single_task<ValidAlias>([] {});
    });

#ifndef __SYCL_UNNAMED_LAMBDA__
    // expected-error@+3 {{kernel name cannot be a type in the "std" namespace}}
#endif
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<std::max_align_t>([] {});
    });

    using InvalidAlias = InvalidKernelName4;
#ifndef __SYCL_UNNAMED_LAMBDA__
    // expected-error@Inputs/sycl.hpp:220 {{kernel needs to have a globally-visible name}}
    // expected-note@23 {{InvalidKernelName4 declared here}}
    // expected-note@+3{{in instantiation of function template specialization}}
#endif
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<InvalidAlias>([] {});
    });

    using InvalidAlias1 = InvalidKernelName5;
#ifndef __SYCL_UNNAMED_LAMBDA__
    // expected-error@Inputs/sycl.hpp:220 {{kernel needs to have a globally-visible name}}
    // expected-note@24 {{InvalidKernelName5 declared here}}
    // expected-note@+3{{in instantiation of function template specialization}}
#endif
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<namespace1::KernelName<InvalidAlias1>>([] {});
    });
  }
};

int main() {
  cl::sycl::queue q;
#ifndef __SYCL_UNNAMED_LAMBDA__
// expected-error@Inputs/sycl.hpp:220 {{kernel name is missing}}
// expected-note@+2{{in instantiation of function template specialization}}
#endif
  q.submit([&](cl::sycl::handler &h) { h.single_task([] {}); });

  return 0;
}
