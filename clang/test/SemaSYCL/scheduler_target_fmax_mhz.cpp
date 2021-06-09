// RUN: %clang_cc1 -fsycl-is-device -verify %s

// Test that checks scheduler_target_fmax_mhz attribute support on Function.

// Test for deprecated spelling of scheduler_target_fmax_mhz attribute.
// expected-warning@+2 {{attribute 'intelfpga::scheduler_target_fmax_mhz' is deprecated}}
// expected-note@+1 {{did you mean to use 'intel::scheduler_target_fmax_mhz' instead?}}
[[intelfpga::scheduler_target_fmax_mhz(2)]] void deprecate() {}

// Tests for incorrect argument values for Intel FPGA scheduler_target_fmax_mhz function attribute.
[[intel::scheduler_target_fmax_mhz(0)]] int Var = 0; // expected-error{{'scheduler_target_fmax_mhz' attribute only applies to functions}}

[[intel::scheduler_target_fmax_mhz(1048577)]] void correct() {} // OK

[[intel::scheduler_target_fmax_mhz("foo")]] void func() {} // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'const char [4]'}}

[[intel::scheduler_target_fmax_mhz(-1)]] void func1() {} // expected-error{{'scheduler_target_fmax_mhz' attribute requires a non-negative integral compile time constant expression}}

[[intel::scheduler_target_fmax_mhz(0, 1)]] void func2() {} // expected-error{{'scheduler_target_fmax_mhz' attribute takes one argument}}

// Tests for Intel FPGA scheduler_target_fmax_mhz function attribute duplication.
// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
[[intel::scheduler_target_fmax_mhz(2)]]
[[intel::scheduler_target_fmax_mhz(2)]] void
func3() {}

// No diagnostic is emitted because the arguments match.
[[intel::scheduler_target_fmax_mhz(4)]] void func4();
[[intel::scheduler_target_fmax_mhz(4)]] void func4(); // OK

// Diagnostic is emitted because the arguments mismatch.
[[intel::scheduler_target_fmax_mhz(2)]] // expected-note {{previous attribute is here}}
[[intel::scheduler_target_fmax_mhz(4)]] void
func5() {} // expected-warning@-1 {{attribute 'scheduler_target_fmax_mhz' is already applied with different arguments}}

[[intel::scheduler_target_fmax_mhz(1)]] void func6(); // expected-note {{previous attribute is here}}
[[intel::scheduler_target_fmax_mhz(3)]] void func6(); // expected-warning {{attribute 'scheduler_target_fmax_mhz' is already applied with different arguments}}

// Tests that check template parameter support for Intel FPGA scheduler_target_fmax_mhz function attributes.
template <int N>
[[intel::scheduler_target_fmax_mhz(N)]] void func7(); // expected-error {{'scheduler_target_fmax_mhz' attribute requires a non-negative integral compile time constant expression}}

template <int size>
[[intel::scheduler_target_fmax_mhz(10)]] void func8(); // expected-note {{previous attribute is here}}
template <int size>
[[intel::scheduler_target_fmax_mhz(size)]] void func8() {} // expected-warning {{attribute 'scheduler_target_fmax_mhz' is already applied with different arguments}}

void checkTemplates() {
  func7<4>();  // OK
  func7<-1>(); // expected-note {{in instantiation of function template specialization 'func7<-1>' requested here}}
  func7<0>();  // OK
  func8<20>(); // expected-note {{in instantiation of function template specialization 'func8<20>' requested here}}
}

// Test that checks expression is not a constant expression.
// expected-note@+1{{declared here}}
int baz();
// expected-error@+2{{expression is not an integral constant expression}}
// expected-note@+1{{non-constexpr function 'baz' cannot be used in a constant expression}}
[[intel::scheduler_target_fmax_mhz(baz() + 1)]] void func9();

// Test that checks expression is a constant expression.
constexpr int bar() { return 0; }
[[intel::scheduler_target_fmax_mhz(bar() + 2)]] void func10(); // OK

// Test that checks wrong function template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.
template <typename Ty>
// expected-error@+2 {{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
// expected-error@+1 {{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
[[intel::scheduler_target_fmax_mhz(Ty{})]] void func11() {}

struct S {};
void test() {
  //expected-note@+1{{in instantiation of function template specialization 'func11<S>' requested here}}
  func11<S>();
  //expected-note@+1{{in instantiation of function template specialization 'func11<float>' requested here}}
  func11<float>();
}
