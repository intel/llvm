// RUN: %clang_cc1 -fsycl-is-device -sycl-std=2020 -verify -fsyntax-only %s

// This test makes sure that SemaSYCL doesn't crash and emits correct error
// messages if operator new was used inside recursive function.

typedef __typeof__(sizeof(int)) size_t;

struct S {
  // expected-note@+1 {{'operator new' declared here}}
  void *operator new(size_t);
};

// expected-note@+2 {{function implemented using recursion declared here}}
// expected-note@+1 {{called by 'foo'}}
__attribute__((sycl_device)) bool foo() {
  // expected-error@+1 {{SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute}}
  S *P = new S();
  // expected-error@+1 {{SYCL kernel cannot allocate storage}}
  int *IP = new int;
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  foo();
  return true;
}
