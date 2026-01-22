// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -fsyntax-only -verify -pedantic %s

#include "sycl.hpp"

sycl::queue deviceQueue;

// Test for Intel FPGA loop attributes applied not to a loop
void foo() {
  // expected-error@+1 {{'intel::ivdep' attribute cannot be applied to a declaration}}
  [[intel::ivdep]] int a[10];
  // expected-error@+1 {{'intel::nofusion' attribute cannot be applied to a declaration}}
  [[intel::nofusion]] int k[10];
}

// Test for deprecated spelling of Intel FPGA loop attributes
void foo_deprecated() {
  int a[10];
  // expected-warning@+1 {{unknown attribute 'intelfpga::ivdep' ignored}}
  [[intelfpga::ivdep(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

// Test for incorrect number of arguments for Intel FPGA loop attributes
void boo() {
  int a[10];
  int b[10];
  // expected-error@+1 {{duplicate argument to 'ivdep'; attribute requires one or both of a safelen and array}}
  [[intel::ivdep(2, 2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-error@+1 {{duplicate argument to 'ivdep'; attribute requires one or both of a safelen and array}}
  [[intel::ivdep(2, 3)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{duplicate argument to 'ivdep'; attribute requires one or both of a safelen and array}}
  [[intel::ivdep(a, b)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{unknown argument to 'ivdep'; expected integer or array variable}}
  [[intel::ivdep(2, 3.0)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-error@+1 {{'intel::nofusion' attribute takes no arguments}}
  [[intel::nofusion(0)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

// Test for incorrect argument value for Intel FPGA loop attributes
void goo() {
  int a[10];
  // expected-warning@+1 {{'ivdep' attribute with value 0 has no effect; attribute ignored}}
  [[intel::ivdep(0)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{unknown argument to 'ivdep'; expected integer or array variable}}
  [[intel::ivdep("test123")]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{unknown argument to 'ivdep'; expected integer or array variable}}
  [[intel::ivdep("test123")]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // no diagnostics are expected
  [[intel::ivdep(a, 2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // no diagnostics are expected
  [[intel::ivdep(2, a)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  int *ptr;
  // no diagnostics are expected
  [[intel::ivdep(2, ptr)]] for (int i = 0; i != 10; ++i)
      ptr[i] = 0;

  struct S {
    int arr[10];
    int *ptr;
  } s;

  // no diagnostics are expected
  [[intel::ivdep(2, s.arr)]] for (int i = 0; i != 10; ++i)
      s.arr[i] = 0;
  // no diagnostics are expected
  [[intel::ivdep(2, s.ptr)]] for (int i = 0; i != 10; ++i)
      s.ptr[i] = 0;

  // no diagnostics are expected
  [[intel::nofusion]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

// Test for Intel FPGA loop attributes duplication
void zoo() {
  int a[10];
  // no diagnostics are expected
  [[intel::ivdep]]
  for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::ivdep]]
  // expected-warning@+2 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen INF}}
  // expected-note@-2 {{previous attribute is here}}
  [[intel::ivdep]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::ivdep]]
  // expected-warning@+2 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen 2}}
  // expected-note@-2 {{previous attribute is here}}
  [[intel::ivdep(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::ivdep(2)]]
  // expected-warning@-1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 4 >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intel::ivdep(4)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::ivdep]]
  // expected-warning@+2 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen INF}}
  // expected-note@-2 {{previous attribute is here}}
  [[intel::ivdep]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::ivdep(2)]]
  // expected-warning@-1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intel::ivdep]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::ivdep(a, 2)]]
  // expected-warning@-1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intel::ivdep(a)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::ivdep(2)]]
  // expected-warning@-1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 4 >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intel::ivdep(4)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // no diagnostics are expected
  [[intel::ivdep(a)]]
  [[intel::ivdep(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  [[intel::ivdep(a, 2)]]
  // expected-warning@-1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intel::ivdep]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // Ensure we only diagnose conflict with the 'worst', not all.
  // expected-warning@+1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 5 >= safelen 3}}
  [[intel::ivdep(3)]]
  // expected-warning@+1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 5 >= safelen 4}}
  [[intel::ivdep(4)]]
  // expected-note@+1 2 {{previous attribute is here}}
  [[intel::ivdep(5)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  [[intel::ivdep(a, 2)]]
  // expected-warning@-1 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 3 >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intel::ivdep(a, 3)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  [[intel::nofusion]]
  // expected-error@+1 {{duplicate Intel FPGA loop attribute 'intel::nofusion'}}
  [[intel::nofusion]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

template<int A, int B, int C>
void ivdep_dependent() {
  int a[10];
  // test this again to ensure we skip properly during instantiation.
  [[intel::ivdep(3)]]
  // expected-warning@-1 2{{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 5 >= safelen 3}}
  // expected-note@+1 2{{previous attribute is here}}
  [[intel::ivdep(5)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-warning@+1 {{'ivdep' attribute with value 1 has no effect; attribute ignored}}
  [[intel::ivdep(C)]]
  // expected-error@-1 {{'ivdep' attribute requires a non-negative integral compile time constant expression}}
  for (int i = 0; i != 10; ++i)
    a[i] = 0;

  // expected-warning@+3 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 4 >= safelen 2}}
  // expected-note@+1 {{previous attribute is here}}
  [[intel::ivdep(A)]]
  [[intel::ivdep(B)]]
  // expected-warning@-2 {{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen 4 >= safelen 2}}
  // expected-note@-2 {{previous attribute is here}}
  for (int i = 0; i != 10; ++i)
      a[i] = 0;

  (void)[]() {
  // expected-warning@+4 2{{ignoring redundant Intel FPGA loop attribute 'ivdep': safelen INF >= safelen INF}}
  // expected-note@-2 2{{while substituting into a lambda expression here}}
  // expected-note@+1 2{{previous attribute is here}}
  [[intel::ivdep]]
  [[intel::ivdep]] while (true);
  };
}

int main() {
  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_function>([]() {
      foo();
      foo_deprecated();
      boo();
      goo();
      zoo();
      ivdep_dependent<4, 2, 1>();
      //expected-note@-1 +{{in instantiation of function template specialization}}
      ivdep_dependent<2, 4, -1>();
      //expected-note@-1 +{{in instantiation of function template specialization}}
    });
  });

  return 0;
}
