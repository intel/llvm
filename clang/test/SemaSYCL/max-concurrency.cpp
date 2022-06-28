// RUN: %clang_cc1 -fsycl-is-device -verify %s

// Test that checks max_concurrency attribute support on function.

// Tests for incorrect argument values for Intel FPGA max_concurrency function attribute.
[[intel::max_concurrency]] void one() {} // expected-error {{'max_concurrency' attribute takes one argument}}

[[intel::max_concurrency(5)]] int a; // expected-error{{'max_concurrency' attribute only applies to 'for', 'while', 'do' statements, and functions}}

[[intel::max_concurrency("foo")]] void func() {} // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'const char[4]'}}

[[intel::max_concurrency(-1)]] void func1() {} // expected-error{{'max_concurrency' attribute requires a non-negative integral compile time constant expression}}

[[intel::max_concurrency(0, 1)]] void func2() {} // expected-error{{'max_concurrency' attribute takes one argument}}

// Tests for Intel FPGA max_concurrency function attribute duplication.
// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
[[intel::max_concurrency(2)]] [[intel::max_concurrency(2)]] void func3() {}

// No diagnostic is emitted because the arguments match.
[[intel::max_concurrency(4)]] void func4();
[[intel::max_concurrency(4)]] void func4(); // OK

// Diagnostic is emitted because the arguments mismatch.
[[intel::max_concurrency(2)]] // expected-note {{previous attribute is here}}
[[intel::max_concurrency(4)]] void
func5() {} // expected-warning@-1 {{attribute 'max_concurrency' is already applied with different arguments}}

[[intel::max_concurrency(1)]] void func6(); // expected-note {{previous attribute is here}}
[[intel::max_concurrency(3)]] void func6(); // expected-warning {{attribute 'max_concurrency' is already applied with different arguments}}

// Tests for Intel FPGA max_concurrency and disable_loop_pipelining function attributes compatibility.
// expected-error@+2 {{'max_concurrency' and 'disable_loop_pipelining' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::disable_loop_pipelining]] [[intel::max_concurrency(2)]] void func7();

// expected-error@+2 {{'disable_loop_pipelining' and 'max_concurrency' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::max_concurrency(4)]] [[intel::disable_loop_pipelining]] void func8();

// expected-error@+3 {{'disable_loop_pipelining' and 'max_concurrency' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::max_concurrency(4)]] void func9();
[[intel::disable_loop_pipelining]] void func9();

// Tests that check template parameter support for Intel FPGA initiation_interval function attributes
template <int N>
[[intel::max_concurrency(N)]] void func10(); // expected-error {{'max_concurrency' attribute requires a non-negative integral compile time constant expression}}

template <int size>
[[intel::max_concurrency(10)]] void func11(); // expected-note {{previous attribute is here}}
template <int size>
[[intel::max_concurrency(size)]] void func11() {} // expected-warning {{attribute 'max_concurrency' is already applied with different arguments}}

void checkTemplates() {
  func10<4>();  // OK
  func10<-1>(); // expected-note {{in instantiation of function template specialization 'func10<-1>' requested here}}
  func10<0>();  // OK
  func11<20>(); // expected-note {{in instantiation of function template specialization 'func11<20>' requested here}}
}

// Test that checks expression is not a constant expression.
// expected-note@+1{{declared here}}
int baz();
// expected-error@+2{{expression is not an integral constant expression}}
// expected-note@+1{{non-constexpr function 'baz' cannot be used in a constant expression}}
[[intel::max_concurrency(baz() + 1)]] void func12();

// Test that checks expression is a constant expression.
constexpr int bar() { return 0; }
[[intel::max_concurrency(bar() + 2)]] void func13(); // OK

// Test that checks wrong function template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.
template <typename Ty>
// expected-error@+2 {{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
// expected-error@+1 {{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
[[intel::max_concurrency(Ty{})]] void func14() {}

struct S {};
void test() {
  // expected-note@+1{{in instantiation of function template specialization 'func14<S>' requested here}}
  func14<S>();
  // expected-note@+1{{in instantiation of function template specialization 'func14<float>' requested here}}
  func14<float>();
}
