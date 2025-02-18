// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -sycl-std=2020 -std=c++17 %s
// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -verify=host -sycl-std=2020 -std=c++17 %s

// This test checks that the compiler doesn't emit an error when indirect call
// was made through a function pointer that is constant expression, and makes
// sure that the error is emitted when a function pointer is not a constant
// expression.

// host-no-diagnostics

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

template <typename FTy> void templated(FTy f) { f(); } // #call-templated

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

  templated(t);
  // expected-error@#call-templated {{SYCL kernel cannot call through a function pointer}}
  // expected-note@-2 {{called by 'bar'}}
}

 void from_device() {
  bar(); // #from_device
}

void from_host() {
  const auto f1 = t;
  f1();
  auto f2 = t;
  f2();

  fooNTTP<t>();

  templated(t);
}
