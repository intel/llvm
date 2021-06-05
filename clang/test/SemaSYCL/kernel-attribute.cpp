// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fsycl-is-device -verify %s
// RUN: %clang_cc1 -fsycl-is-host -DHOST -fsyntax-only -verify %s

// Only function templates
[[clang::sycl_kernel]] int gv2 = 0; // expected-warning {{'sycl_kernel' attribute only applies to function templates}}
__attribute__((sycl_kernel)) int gv3 = 0; // expected-warning {{'sycl_kernel' attribute only applies to function templates}}

__attribute__((sycl_kernel)) void foo(); // expected-warning {{'sycl_kernel' attribute only applies to function templates}}
[[clang::sycl_kernel]] void foo1(); // expected-warning {{'sycl_kernel' attribute only applies to function templates}}

// Attribute takes no arguments
template <typename T, typename A>
__attribute__((sycl_kernel(1))) void foo(T P); // expected-error {{'sycl_kernel' attribute takes no arguments}}
template <typename T, typename A, int I>
[[clang::sycl_kernel(1)]] void foo1(T P);// expected-error {{'sycl_kernel' attribute takes no arguments}}

#ifndef HOST
// At least two template parameters
template <typename T>
__attribute__((sycl_kernel)) void foo(T P); // expected-warning {{'sycl_kernel' attribute only applies to a function template with at least two template parameters}}
template <typename T>
[[clang::sycl_kernel]] void foo1(T P); // expected-warning {{'sycl_kernel' attribute only applies to a function template with at least two template parameters}}
#endif

// First two template parameters cannot be non-type template parameters
template <typename T, int A>
__attribute__((sycl_kernel)) void foo(T P); // expected-warning {{template parameter of a function template with the 'sycl_kernel' attribute cannot be a non-type template parameter}}
template <int A, typename T>
[[clang::sycl_kernel]] void foo1(T P); // expected-warning {{template parameter of a function template with the 'sycl_kernel' attribute cannot be a non-type template parameter}}

// Must return void
template <typename T, typename A>
__attribute__((sycl_kernel)) int foo(T P); // expected-warning {{function template with 'sycl_kernel' attribute must have a 'void' return type}}
template <typename T, typename A>
[[clang::sycl_kernel]] int foo1(T P); // expected-warning {{function template with 'sycl_kernel' attribute must have a 'void' return type}}

// Must take at least one argument
template <typename T, typename A>
__attribute__((sycl_kernel)) void foo(); // expected-warning {{function template with 'sycl_kernel' attribute must have at least one parameter}}
template <typename T, typename A>
[[clang::sycl_kernel]] void foo1(T t, A a); // no diagnostics

// No diagnostics
template <typename T, typename A>
__attribute__((sycl_kernel)) void foo(T P);
template <typename T, typename A, int I>
[[clang::sycl_kernel]] void foo1(T P);

#ifdef HOST
// No diagnostics
template <typename Func>
void __attribute__((sycl_kernel))
KernelImpl4(const Func &f, int i, double d) {
  f(i, d);
}

template <typename Name, typename Func>
void __attribute__((sycl_kernel))
Kernel(const Func &f) {
  KernelImpl4(f, 1, 2.0);
}

void func() {
  auto Lambda = [](int i, double d) { d += i; };
  Kernel<class Foo>(Lambda);
}
#endif
