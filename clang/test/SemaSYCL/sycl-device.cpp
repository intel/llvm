// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -verify -DNO_SYCL %s

// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -DNOT_STRICT -Wno-error=sycl-strict -Wno-sycl-strict %s
// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -DWARN_STRICT -Wno-error=sycl-strict %s

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
  __attribute__((sycl_device))
  A() {}

  __attribute__((sycl_device))
  int func3() {}
};

class B {
  __attribute__((sycl_device)) // expected-error {{'sycl_device' attribute cannot be applied to a virtual or pure virtual class member function}}
  virtual int foo() {}

  __attribute__((sycl_device)) // expected-error {{'sycl_device' attribute cannot be applied to a virtual or pure virtual class member function}}
  virtual int bar() = 0;
};

#if defined(NOT_STRICT)
__attribute__((sycl_device))
int* func3() { return nullptr; }

__attribute__((sycl_device))
void func3(int *) {}
#elif defined(WARN_STRICT)
__attribute__((sycl_device)) // expected-warning {{SYCL 1.2.1 specification does not allow 'sycl_device' attribute applied to a function with a raw pointer return type}}
int* func3() { return nullptr; }

__attribute__((sycl_device)) // expected-warning {{SYCL 1.2.1 specification does not allow 'sycl_device' attribute applied to a function with a raw pointer parameter type}}
void func3(int *) {}
#else
__attribute__((sycl_device)) // expected-error {{SYCL 1.2.1 specification does not allow 'sycl_device' attribute applied to a function with a raw pointer return type}}
int* func3() { return nullptr; }

__attribute__((sycl_device)) // expected-error {{SYCL 1.2.1 specification does not allow 'sycl_device' attribute applied to a function with a raw pointer parameter type}}
void func3(int *) {}
#endif

#else // NO_SYCL
__attribute__((sycl_device)) // expected-warning {{'sycl_device' attribute ignored}}
void baz() {}

#endif // NO_SYCL
