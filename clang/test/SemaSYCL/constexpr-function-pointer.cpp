// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -sycl-std=2020 -std=c++17 %s

// This test checks that the compiler doesn't emit an error when indirect call
// was made through a function pointer that is constant expression, and makes
// sure that the error is emitted when a function pointer is not a constant
// expression.

void t() {}

constexpr auto F = t;
const auto F1 = t;

typedef void (*SomeFunc)();

constexpr SomeFunc foo() { return t; }

const SomeFunc foo1() { return t; }

void bar1(const SomeFunc fptr) {
  fptr();
}

template <auto f> void fooNTTP() { f(); }

__attribute__((sycl_device)) void bar() {
  // OK
  constexpr auto f = t;
  f();
  const auto f1 = t;
  // expected-error@+1 {{SYCL kernel cannot call through a function pointer}}
  f1();
  auto f2 = t;
  // expected-error@+1 {{SYCL kernel cannot call through a function pointer}}
  f2();

  // OK
  F();
  // expected-error@+1 {{SYCL kernel cannot call through a function pointer}}
  F1();

  constexpr auto ff = foo();
  ff();
  const auto ff1 = foo();
  // expected-error@+1 {{SYCL kernel cannot call through a function pointer}}
  ff1();
  const auto fff = foo1();
  // expected-error@+1 {{SYCL kernel cannot call through a function pointer}}
  fff();

  fooNTTP<t>();
}
