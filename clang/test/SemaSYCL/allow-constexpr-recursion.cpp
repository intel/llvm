// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fcxx-exceptions -Wno-return-type -sycl-std=2020 -verify -fsyntax-only -std=c++20 -Werror=vla %s

// This test verifies that a SYCL kernel executed on a device, cannot call a recursive function.

#include "sycl.hpp"

sycl::queue q;

constexpr int constexpr_recurse1(int n);

// expected-note@+1 3{{function implemented using recursion declared here}}
constexpr int constexpr_recurse(int n) {
  if (n)
    return constexpr_recurse1(n - 1);
  return 103;
}

constexpr int constexpr_recurse1(int n) {
  // expected-error@+1{{SYCL kernel cannot call a recursive function}}
  return constexpr_recurse(n) + 1;
}

template <int I>
void bar() {}

template <int... args>
void bar2() {}

enum class SomeE {
  Value = constexpr_recurse(5)
};

struct ConditionallyExplicitCtor {
  explicit(constexpr_recurse(5) == 103) ConditionallyExplicitCtor(int i) {}
};

void conditionally_noexcept() noexcept(constexpr_recurse(5)) {}

template <int I>
void ConstexprIf1() {
  if constexpr (I == 1)
    ConstexprIf1<I>();
}

// Same as the above, but split up so the diagnostic is more clear.
// expected-note@+2 2{{function implemented using recursion declared here}}
template <int I>
void ConstexprIf2() {
  if constexpr (I == 1)
    // expected-error@+1{{SYCL kernel cannot call a recursive function}}
    ConstexprIf2<I>();
}

// All of the uses of constexpr_recurse here are forced constant expressions, so
// they should not diagnose.
void constexpr_recurse_test() {
  constexpr int i = constexpr_recurse(1);
  bar<constexpr_recurse(2)>();
  bar2<1, 2, constexpr_recurse(2)>();
  static_assert(constexpr_recurse(2) == 105, "");

  int j;
  switch (105) {
  case constexpr_recurse(2):
    // expected-error@+1{{SYCL kernel cannot call a recursive function}}
    j = constexpr_recurse(5);
    break;
  }

  SomeE e = SomeE::Value;

  int ce_array[constexpr_recurse(5)];

  conditionally_noexcept();

  if constexpr ((bool)SomeE::Value) {
  }

  ConditionallyExplicitCtor c(1);

  ConstexprIf1<0>(); // Should not cause a diagnostic.
  // expected-error@+1{{SYCL kernel cannot call a recursive function}}
  ConstexprIf2<1>();
}

void constexpr_recurse_test_err() {
  // expected-error@+1{{SYCL kernel cannot call a recursive function}}
  int i = constexpr_recurse(1);
}

int main() {
  q.submit([&](sycl::handler &h) {
    h.single_task<class fake_kernel>([]() { constexpr_recurse_test(); });
  });

  q.submit([&](sycl::handler &h) {
    h.single_task<class fake_kernel>([]() { constexpr_recurse_test_err(); });
  });
}
