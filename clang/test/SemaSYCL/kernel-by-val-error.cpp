
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fsycl-is-device -verify %s
// RUN: %clang_cc1 -fsycl-is-host -DHOST -fsyntax-only -verify %s

// Kernel function argument must be passed by reference

#ifndef HOST
template <typename Name, typename T>
__attribute__((sycl_kernel)) void foo(T P) {} // expected-error {{SYCL kernel function must be passed by reference}}
template <typename Name, typename T>
[[clang::sycl_kernel]] void bar(T P) {} // expected-error {{SYCL kernel function must be passed by reference}}
#else
// expected-no-diagnostics
template <typename Name, typename T>
__attribute__((sycl_kernel)) void foo(T P) {}
template <typename Name, typename T>
[[clang::sycl_kernel]] void bar(T P) {}
#endif

void F() {
    foo<class FooKernel>([](){});
    bar<class BarKernel>([](){});
}