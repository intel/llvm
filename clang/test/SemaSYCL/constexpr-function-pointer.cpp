// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -sycl-std=2020 -std=c++17 %s

[[intel::device_indirectly_callable]] void t() {}

const auto F = t;

typedef void (*SomeFunc)();

constexpr SomeFunc foo() { return t; }

const SomeFunc foo1() { return t; }

void recursiveFoo() { recursiveFoo(); }

void bar1(const SomeFunc fptr) {
  fptr();
}

__attribute__((sycl_device)) void bar() {
  // OK
  constexpr auto f = t;
  f();
  // OK
  const auto f1 = t;
  f1();
  auto f2 = t;
  // expected-error@+1 {{SYCL kernel cannot call through a function pointer}}
  f2();

  // OK
  F();

  // foo is a constexpr - OK
  const auto ff = foo();
  ff();
  const auto fff = foo1();
  // expected-error@+1 {{SYCL kernel cannot call through a function pointer}}
  fff();
}

