// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -verify -DNO_SYCL %s

#ifndef NO_SYCL

__attribute__((sycl_device)) // expected-warning {{'sycl_device' attribute only applies to functions}}
int N;

__attribute__((sycl_device(3))) // expected-error {{'sycl_device' attribute takes no arguments}}
void bar() {}

__attribute__((sycl_device)) // expected-error {{'sycl_device' attribute cannot be applied to a static function or function in an anonymous namespace}}
static void func1() {}

namespace {
  __attribute__((sycl_device)) // expected-error {{'sycl_device' attribute cannot be applied to a static function or function in an anonymous namespace}}
  void func2() {}
}

class A {
  __attribute__((sycl_device)) // expected-error {{'sycl_device' attribute cannot be applied to a class member function}}
  A() {}

  __attribute__((sycl_device)) // expected-error {{'sycl_device' attribute cannot be applied to a class member function}}
  int func3() {}
};

__attribute__((sycl_device)) // expected-error {{'sycl_device' attribute cannot be applied to a function with a raw pointer return type}}
int* func3() { return nullptr; }

__attribute__((sycl_device)) // expected-error {{'sycl_device' attribute cannot be applied to a function with a raw pointer parameter type}}
void func3(int *) {}

#else

__attribute__((sycl_device)) // expected-warning {{'sycl_device' attribute ignored}}
void baz() {}

#endif // NO_SYCL
