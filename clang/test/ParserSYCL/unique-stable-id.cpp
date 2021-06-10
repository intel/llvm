// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -Wno-unused %s

// Test various parsing related issues for __builtin_sycl_unique_stable_id.

int global;
void f(int var) {
  // expected-error@+1 {{expected '(' after '__builtin_sycl_unique_stable_id'}}
  __builtin_sycl_unique_stable_id global;
  // expected-error@+1 {{expected '(' after '__builtin_sycl_unique_stable_id'}}
  __builtin_sycl_unique_stable_id{global};
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  __builtin_sycl_unique_stable_id(global;

  // expected-error@+1 {{expected '(' for function-style cast or type construction}}
  __builtin_sycl_unique_stable_id(int);

  // expected-error@+1 {{expected expression}}
  __builtin_sycl_unique_stable_id(do{}while(true););
}
