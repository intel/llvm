// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fno-sycl-unnamed-lambda -fsyntax-only -sycl-std=2020 -verify %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -sycl-std=2020 -verify %s

// This test verifies that an error is thrown when kernel names are declared within function/class scope
// and if kernel names are empty.

#include "sycl.hpp"

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
    // expected-error@#KernelSingleTask {{'InvalidKernelName1' should be globally visible}}
    // expected-note@+3{{in instantiation of function template specialization}}
    class InvalidKernelName1 {};
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<InvalidKernelName1>([] {});
    });

    // expected-error@#KernelSingleTask {{'namespace1::KernelName<InvalidKernelName2>' should be globally visible}}
    // expected-note@+3{{in instantiation of function template specialization}}
    class InvalidKernelName2 {};
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<namespace1::KernelName<InvalidKernelName2>>([] {});
    });

    // expected-error@#KernelSingleTask {{'MyWrapper::InvalidKernelName0' should be globally visible}}
    // expected-note@+2{{in instantiation of function template specialization}}
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<InvalidKernelName0>([] {});
    });

    // expected-error@#KernelSingleTask {{'namespace1::KernelName<MyWrapper::InvalidKernelName3>' should be globally visible}}
    // expected-note@+2{{in instantiation of function template specialization}}
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<namespace1::KernelName<InvalidKernelName3>>([] {});
    });

    using ValidAlias = MyWrapper;
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<ValidAlias>([] {});
    });

    // expected-error@#KernelSingleTask {{'std::max_align_t' is an invalid kernel name, 'std::(anonymous)' is declared in the 'std' namespace}}
    // expected-note@+2{{in instantiation of function template specialization}}
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<std::max_align_t>([] {});
    });

    using InvalidAlias = InvalidKernelName4;
    // expected-error@#KernelSingleTask {{'MyWrapper::InvalidKernelName4' should be globally visible}}
    // expected-note@+2{{in instantiation of function template specialization}}
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<InvalidAlias>([] {});
    });

    using InvalidAlias1 = InvalidKernelName5;
    // expected-error@#KernelSingleTask {{'namespace1::KernelName<MyWrapper::InvalidKernelName5>' should be globally visible}}
    // expected-note@+2{{in instantiation of function template specialization}}
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<namespace1::KernelName<InvalidAlias1>>([] {});
    });
    // expected-error@#KernelSingleTask {{'Templated_kernel_name2<Templated_kernel_name<InvalidKernelName1>>' should be globally visible}}
    // expected-note@+2{{in instantiation of function template specialization}}
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<Templated_kernel_name2<Templated_kernel_name<InvalidKernelName1>>>([] {});
    });
  }

#ifdef __SYCL_UNNAMED_LAMBDA__
  // Test unnamed kernels the same way.  The above set should still be errors,
  // but this set is now fine.
  void test_unnamed() {
    cl::sycl::queue q;
    q.submit([&](cl::sycl::handler &h) {
      h.single_task([] {});
    });

    q.submit([&](cl::sycl::handler &h) {
      h.single_task([] {});
    });

    q.submit([&](cl::sycl::handler &h) {
      h.single_task([] {});
    });

    q.submit([&](cl::sycl::handler &h) {
      h.single_task([] {});
    });

    q.submit([&](cl::sycl::handler &h) {
      h.single_task([] {});
    });

    q.submit([&](cl::sycl::handler &h) {
      h.single_task([] {});
    });

    q.submit([&](cl::sycl::handler &h) {
      h.single_task([] {});
    });

    q.submit([&](cl::sycl::handler &h) {
      h.single_task([] {});
    });
    q.submit([&](cl::sycl::handler &h) {
      h.single_task([] {});
    });

    // This should have no problem, since the types match.
    q.submit([&](cl::sycl::handler &h) {
      auto SomeLambda = []() {};
      h.single_task<decltype(SomeLambda)>(SomeLambda);
    });

    // This errors because const decltype(SomeLambda) != decltype(SomeLambda),
    //  so this is not the unnamed lambda situation.
    // expected-error@#KernelSingleTask {{unnamed lambda 'const}}
    // expected-note@+3{{in instantiation of function template specialization}}
    q.submit([&](cl::sycl::handler &h) {
      auto SomeLambda = []() {};
      h.single_task<const decltype(SomeLambda)>(SomeLambda);
    });
  }
#endif
};

int main() {
  cl::sycl::queue q;
#ifndef __SYCL_UNNAMED_LAMBDA__
  // expected-error-re@#KernelSingleTask {{unnamed lambda '(lambda at {{.*}}unnamed-kernel.cpp{{.*}}' used}}
  // expected-note@+2{{in instantiation of function template specialization}}
#endif
  q.submit([&](cl::sycl::handler &h) { h.single_task([] {}); });

  return 0;
}
