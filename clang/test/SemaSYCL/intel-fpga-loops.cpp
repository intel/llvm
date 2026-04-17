// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -fsyntax-only -verify -pedantic %s

#include "sycl.hpp"

sycl::queue deviceQueue;

// Test for Intel FPGA loop attributes applied not to a loop
void foo() {
  // expected-error@+1 {{'intel::loop_coalesce' attribute cannot be applied to a declaration}}
  [[intel::loop_coalesce(2)]] int h[10];
  // expected-error@+1 {{'intel::max_interleaving' attribute cannot be applied to a declaration}}
  [[intel::max_interleaving(4)]] int i[10];
}

// Test for deprecated spelling of Intel FPGA loop attributes
void foo_deprecated() {
  int a[10];

  // expected-warning@+1 {{unknown attribute 'intelfpga::ii' ignored}}
  [[intelfpga::ii(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-warning@+1 {{unknown attribute 'intelfpga::max_interleaving' ignored}}
  [[intelfpga::max_interleaving(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-warning@+1 {{unknown attribute 'intelfpga::loop_coalesce' ignored}}
  [[intelfpga::loop_coalesce(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

}

// Test for incorrect number of arguments for Intel FPGA loop attributes
void boo() {
  int a[10];
  int b[10];

  // expected-error@+1 {{'intel::loop_coalesce' attribute takes no more than 1 argument}}
  [[intel::loop_coalesce(2, 3)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'intel::max_interleaving' attribute takes one argument}}
  [[intel::max_interleaving]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'intel::max_interleaving' attribute takes one argument}}
  [[intel::max_interleaving(2, 4)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

// Test for incorrect argument value for Intel FPGA loop attributes
void goo() {
  int a[10];
  // expected-error@+1 {{'intel::loop_coalesce' attribute requires a positive integral compile time constant expression}}
  [[intel::loop_coalesce(0)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'intel::max_interleaving' attribute requires integer constant value 0 or 1}}
  [[intel::max_interleaving(-1)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{'intel::max_interleaving' attribute requires integer constant value 0 or 1}}
  [[intel::max_interleaving(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{integral constant expression must have integral or unscoped enumeration type, not 'const char[8]'}}
  [[intel::loop_coalesce("test123")]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // expected-error@+1 {{integral constant expression must have integral or unscoped enumeration type, not 'const char[8]'}}
  [[intel::max_interleaving("test123")]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

// Test for Intel FPGA loop attributes duplication
void zoo() {
  int a[10];
  [[intel::loop_coalesce(2)]]
  // expected-error@+2 {{duplicate Intel FPGA loop attribute 'intel::loop_coalesce'}}
  [[intel::max_interleaving(1)]]
  [[intel::loop_coalesce]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::max_interleaving(1)]] // expected-note {{previous attribute is here}}
  // expected-error@+2 {{conflicting loop attribute 'intel::max_interleaving'}}
  [[intel::max_interleaving(1)]] // OK.
  [[intel::max_interleaving(0)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  [[intel::loop_coalesce]]
  for (int i = 0; i != 10; ++i)
      a[i] = 0;

  [[intel::max_interleaving(0)]] // expected-note 2{{previous attribute is here}}
  [[intel::max_interleaving(0)]] // OK
  [[intel::max_interleaving(1)]] // expected-error {{conflicting loop attribute 'intel::max_interleaving'}}
  [[intel::max_interleaving(1)]] // expected-error {{conflicting loop attribute 'intel::max_interleaving'}}
  for (int i = 0; i != 10; ++i) { a[i] = 0; }
}

template <int A, int B, int C, int D>
void max_interleaving_dependent() {
  int a[10];
  // expected-error@+1 {{'intel::max_interleaving' attribute requires integer constant value 0 or 1}}
  [[intel::max_interleaving(A)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-error@+1 {{'intel::max_interleaving' attribute requires integer constant value 0 or 1}}
  [[intel::max_interleaving(B)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // max_interleaving attribute accepts value 0.
  [[intel::max_interleaving(C)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-error@+2 {{conflicting loop attribute 'intel::max_interleaving'}}
  [[intel::max_interleaving(C)]] // expected-note {{previous attribute is here}}
  [[intel::max_interleaving(D)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  [[intel::max_interleaving(D)]]
  [[intel::max_interleaving(D)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  [[intel::max_interleaving(D)]] // expected-note 2{{previous attribute is here}}
  [[intel::max_interleaving(D)]] // OK
  [[intel::max_interleaving(C)]] // expected-error {{conflicting loop attribute 'intel::max_interleaving'}}
  [[intel::max_interleaving(C)]] // expected-error {{conflicting loop attribute 'intel::max_interleaving'}}
  for (int i = 0; i != 10; ++i) { a[i] = 0; }

}

template <int A, int B, int C>
void loop_coalesce_dependent() {
  int a[10];
  // expected-error@+1 {{'intel::loop_coalesce' attribute requires a positive integral compile time constant expression}}
  [[intel::loop_coalesce(A)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-error@+2 {{duplicate Intel FPGA loop attribute 'intel::loop_coalesce'}}
  [[intel::loop_coalesce]]
  [[intel::loop_coalesce(B)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-error@+1 {{'intel::loop_coalesce' attribute requires a positive integral compile time constant expression}}
  [[intel::loop_coalesce(C)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

void check_max_interleaving_expression() {
  int a[10];
  // Test that checks expression is not a constant expression.
  // expected-note@+1{{declared here}}
  int foo;
  // expected-error@+2{{expression is not an integral constant expression}}
  // expected-note@+1{{read of non-const variable 'foo' is not allowed in a constant expression}}
  [[intel::max_interleaving(foo + 1)]] for (int i = 0; i != 10; ++i)
       a[i] = 0;

  // Test that checks expression is a constant expression.
  constexpr int bar = 0;
  [[intel::max_interleaving(bar + 1)]] for (int i = 0; i != 10; ++i) // OK
      a[i] = 0;
}

void check_loop_coalesce_expression() {
  int a[10];
  // Test that checks expression is not a constant expression.
  // expected-note@+1{{declared here}}
  int foo;
  // expected-error@+2{{expression is not an integral constant expression}}
  // expected-note@+1{{read of non-const variable 'foo' is not allowed in a constant expression}}
  [[intel::loop_coalesce(foo + 1)]] for (int i = 0; i != 10; ++i)
       a[i] = 0;

  // Test that checks expression is a constant expression.
  constexpr int bar = 0;
  [[intel::loop_coalesce(bar + 2)]] for (int i = 0; i != 10; ++i) // OK
      a[i] = 0;
}

// Test that checks wrong template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.
struct S {};
template <typename Ty>
void check_loop_attr_template_instantiation() {
  int a[10];

  // expected-error@+2 {{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
  // expected-error@+1 {{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  [[intel::loop_coalesce(Ty{})]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // expected-error@+2 {{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
  // expected-error@+1 {{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  [[intel::max_interleaving(Ty{})]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

int main() {
  deviceQueue.submit([&](sycl::handler &h) {
    h.single_task<class kernel_function>([]() {
      foo();
      foo_deprecated();
      boo();
      goo();
      zoo();
      max_interleaving_dependent<-1, 4, 0, 1>(); // expected-note{{in instantiation of function template specialization 'max_interleaving_dependent<-1, 4, 0, 1>' requested here}}
      loop_coalesce_dependent<-1, 4,  0>(); // expected-note{{in instantiation of function template specialization 'loop_coalesce_dependent<-1, 4, 0>' requested here}}
      check_max_interleaving_expression();
      check_loop_coalesce_expression();
      check_loop_attr_template_instantiation<S>(); //expected-note{{in instantiation of function template specialization 'check_loop_attr_template_instantiation<S>' requested here}}
      check_loop_attr_template_instantiation<float>(); //expected-note{{in instantiation of function template specialization 'check_loop_attr_template_instantiation<float>' requested here}}
    });
  });

  return 0;
}
