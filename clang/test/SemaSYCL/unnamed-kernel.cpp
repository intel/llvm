// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -Wno-sycl-2017-compat -verify %s
// RUN: %clang_cc1 -fsycl-is-device -fsycl-unnamed-lambda -fsyntax-only -Wno-sycl-2017-compat -verify %s
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

template <typename T>
struct Templated_kernel_name;

template <typename T>
struct Templated_kernel_name2;

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
    // expected-error@#KernelSingleTask {{'InvalidKernelName1' is an invalid kernel name type}}
    // expected-note@#KernelSingleTask {{'InvalidKernelName1' should be globally-visible}}
    // expected-note@+4{{in instantiation of function template specialization}}
#endif
    class InvalidKernelName1 {};
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<InvalidKernelName1>([] {});
    });

#ifndef __SYCL_UNNAMED_LAMBDA__
    // expected-error@#KernelSingleTask {{'namespace1::KernelName<InvalidKernelName2>' is an invalid kernel name type}}
    // expected-note@#KernelSingleTask {{'InvalidKernelName2' should be globally-visible}}
    // expected-note@+4{{in instantiation of function template specialization}}
#endif
    class InvalidKernelName2 {};
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<namespace1::KernelName<InvalidKernelName2>>([] {});
    });

#ifndef __SYCL_UNNAMED_LAMBDA__
    // expected-error@#KernelSingleTask {{'MyWrapper::InvalidKernelName0' is an invalid kernel name type}}
    // expected-note@#KernelSingleTask {{'MyWrapper::InvalidKernelName0' should be globally-visible}}
    // expected-note@+3{{in instantiation of function template specialization}}
#endif
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<InvalidKernelName0>([] {});
    });

#ifndef __SYCL_UNNAMED_LAMBDA__
    // expected-error@#KernelSingleTask {{'namespace1::KernelName<MyWrapper::InvalidKernelName3>' is an invalid kernel name type}}
    // expected-note@#KernelSingleTask {{'MyWrapper::InvalidKernelName3' should be globally-visible}}
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
    // expected-error@#KernelSingleTask {{'std::max_align_t' is an invalid kernel name type}}
    // expected-note@#KernelSingleTask {{type 'std::max_align_t' cannot be in the "std" namespace}}
    // expected-note@+3{{in instantiation of function template specialization}}
#endif
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<std::max_align_t>([] {});
    });

    using InvalidAlias = InvalidKernelName4;
#ifndef __SYCL_UNNAMED_LAMBDA__
    // expected-error@#KernelSingleTask {{'MyWrapper::InvalidKernelName4' is an invalid kernel name type}}
    // expected-note@#KernelSingleTask {{'MyWrapper::InvalidKernelName4' should be globally-visible}}
    // expected-note@+3{{in instantiation of function template specialization}}
#endif
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<InvalidAlias>([] {});
    });

    using InvalidAlias1 = InvalidKernelName5;
#ifndef __SYCL_UNNAMED_LAMBDA__
    // expected-error@#KernelSingleTask {{'namespace1::KernelName<MyWrapper::InvalidKernelName5>' is an invalid kernel name type}}
    // expected-note@#KernelSingleTask {{'MyWrapper::InvalidKernelName5' should be globally-visible}}
    // expected-note@+3{{in instantiation of function template specialization}}
#endif
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<namespace1::KernelName<InvalidAlias1>>([] {});
    });
#ifndef __SYCL_UNNAMED_LAMBDA__
    // expected-error@#KernelSingleTask {{'Templated_kernel_name2<Templated_kernel_name<InvalidKernelName1>>' is an invalid kernel name type}}
    // expected-note@#KernelSingleTask {{'InvalidKernelName1' should be globally-visible}}
    // expected-note@+3{{in instantiation of function template specialization}}
#endif
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<Templated_kernel_name2<Templated_kernel_name<InvalidKernelName1>>>([] {});
    });
  }
};

int main() {
  cl::sycl::queue q;
#ifndef __SYCL_UNNAMED_LAMBDA__
  // expected-error-re@#KernelSingleTask {{'(lambda at {{.*}}unnamed-kernel.cpp{{.*}}' is an invalid kernel name type}}
  // expected-note@#KernelSingleTask {{unnamed type used in a SYCL kernel name}}
  // expected-note@+2{{in instantiation of function template specialization}}
#endif
  q.submit([&](cl::sycl::handler &h) { h.single_task([] {}); });

  return 0;
}
