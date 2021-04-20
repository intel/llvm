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
  __attribute__((sycl_device))
  A() {}

  __attribute__((sycl_device)) void func3() {}
};

class B {
public:
  __attribute__((sycl_device)) virtual void foo() {}

  __attribute__((sycl_device)) virtual void bar() = 0;
};

__attribute__((sycl_device)) int *func0() { return nullptr; }

__attribute__((sycl_device)) void func2(int *) {}

#else // NO_SYCL
__attribute__((sycl_device)) // expected-warning {{'sycl_device' attribute ignored}}
void baz() {}

#endif // NO_SYCL
