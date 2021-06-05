// RUN: %clang_cc1 -fsycl-is-device -verify %s

// Test that checks initiation_interval attribute support on function.

// Tests for incorrect argument values for Intel FPGA initiation_interval function attribute.
[[intel::initiation_interval]] void one() {} // expected-error {{'initiation_interval' attribute takes one argument}}

[[intel::initiation_interval(5)]] int a; // expected-error{{'initiation_interval' attribute only applies to 'for', 'while', 'do' statements, and functions}}

[[intel::initiation_interval("foo")]] void func() {} // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'const char [4]'}}

[[intel::initiation_interval(-1)]] void func1() {} // expected-error{{'initiation_interval' attribute requires a positive integral compile time constant expression}}

[[intel::initiation_interval(0, 1)]] void func2() {} // expected-error{{'initiation_interval' attribute takes one argument}}

// Tests for Intel FPGA initiation_interval function attribute duplication.
// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
[[intel::initiation_interval(2)]]
[[intel::initiation_interval(2)]] void func3() {}

// No diagnostic is emitted because the arguments match.
[[intel::initiation_interval(4)]] void func4();
[[intel::initiation_interval(4)]] void func4(); // OK

// Diagnostic is emitted because the arguments mismatch.
[[intel::initiation_interval(2)]]                  // expected-note {{previous attribute is here}}
[[intel::initiation_interval(4)]] void func5() {}  // expected-warning {{attribute 'initiation_interval' is already applied with different arguments}}

[[intel::initiation_interval(1)]] void func6(); // expected-note {{previous attribute is here}}
[[intel::initiation_interval(3)]] void func6(); // expected-warning {{attribute 'initiation_interval' is already applied with different arguments}}

// Tests for Intel FPGA initiation_interval and disable_loop_pipelining attributes compatibility checks.
// expected-error@+2 {{'initiation_interval' and 'disable_loop_pipelining' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::disable_loop_pipelining]] [[intel::initiation_interval(2)]] void func7();

// expected-error@+2 {{'disable_loop_pipelining' and 'initiation_interval' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::initiation_interval(4)]] [[intel::disable_loop_pipelining]] void func8();

// expected-error@+3 {{'disable_loop_pipelining' and 'initiation_interval' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::initiation_interval(4)]] void func9();
[[intel::disable_loop_pipelining]] void func9();

// Tests that check template parameter support for Intel FPGA initiation_interval function attributes
template <int N>
[[intel::initiation_interval(N)]] void func10(); // expected-error 2{{'initiation_interval' attribute requires a positive integral compile time constant expression}}

template <int size>
[[intel::initiation_interval(10)]] void func11();     // expected-note {{previous attribute is here}}
template <int size>
[[intel::initiation_interval(size)]] void func11() {} // expected-warning {{attribute 'initiation_interval' is already applied with different arguments}}

void checkTemplates() {
  func10<4>();  // OK
  func10<-1>(); // expected-note {{in instantiation of function template specialization 'func10<-1>' requested here}}
  func10<0>();  // expected-note {{in instantiation of function template specialization 'func10<0>' requested here}}
  func11<20>(); // expected-note {{in instantiation of function template specialization 'func11<20>' requested here}}
}

// Test that checks expression is not a constant expression.
// expected-note@+1{{declared here}}
int baz();
// expected-error@+2{{expression is not an integral constant expression}}
// expected-note@+1{{non-constexpr function 'baz' cannot be used in a constant expression}}
[[intel::initiation_interval(baz() + 1)]] void func12();

// Test that checks expression is a constant expression.
constexpr int bar() { return 0; }
[[intel::initiation_interval(bar() + 2)]] void func13(); // OK

// Test that checks wrong function template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.
template <typename Ty>
// expected-error@+2 {{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
// expected-error@+1 {{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
[[intel::initiation_interval(Ty{})]] void func14() {}

struct S {};
void test() {
  //expected-note@+1{{in instantiation of function template specialization 'func14<S>' requested here}}
  func14<S>();
  //expected-note@+1{{in instantiation of function template specialization 'func14<float>' requested here}}
  func14<float>();
}
