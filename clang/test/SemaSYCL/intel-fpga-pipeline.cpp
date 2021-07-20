// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -pedantic -verify %s

// Test that checks fpga_pipeline attribute support on function.

// Tests for incorrect argument values for Intel FPGA fpga_pipeline function attribute.
[[intel::fpga_pipeline]] void one() {} // OK

[[intel::fpga_pipeline(1)]] int a; // expected-error{{'fpga_pipeline' attribute only applies to 'for', 'while', 'do' statements, and functions}}

[[intel::fpga_pipeline("foo")]] void func() {} // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'const char [4]'}}

[[intel::fpga_pipeline(0, 1)]] void func1() {} // expected-error{{'fpga_pipeline' attribute takes no more than 1 argument}}

// Tests for Intel FPGA fpga_pipeline function attribute duplication.
// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
[[intel::fpga_pipeline]]
[[intel::fpga_pipeline]] void func2() {}

[[intel::fpga_pipeline]]
[[intel::fpga_pipeline(1)]] void func3() {}

// No diagnostic is emitted because the arguments match.
[[intel::fpga_pipeline(0)]] void func4();
[[intel::fpga_pipeline(0)]] void func4(); // OK

// Diagnostic is emitted because the arguments mismatch.
[[intel::fpga_pipeline]]                     // expected-note {{previous attribute is here}}
[[intel::fpga_pipeline(0)]] void func5() {}  // expected-warning {{attribute 'fpga_pipeline' is already applied with different arguments}}

[[intel::fpga_pipeline(0)]] void func6(); // expected-note {{previous attribute is here}}
[[intel::fpga_pipeline(1)]] void func6(); // expected-warning {{attribute 'fpga_pipeline' is already applied with different arguments}}

// Tests for Intel FPGA [[intel::initiation_interval]] and [[intel::fpga_pipeline]] attributes compatibility checks.
// expected-error@+2 {{'initiation_interval' and 'fpga_pipeline' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::fpga_pipeline(1)]] [[intel::initiation_interval(2)]] void func7();

// expected-error@+3 {{'fpga_pipeline' and 'initiation_interval' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::initiation_interval(4)]] void func8();
[[intel::fpga_pipeline(0)]] void func8();

// expected-error@+3 {{'initiation_interval' and 'fpga_pipeline' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::fpga_pipeline(1)]] void func9();
[[intel::initiation_interval(8)]] void func9();

// Tests for Intel FPGA [[intel::max_concurrency]] and [[intel::fpga_pipeline]] attributes compatibility checks.
// expected-error@+2 {{'max_concurrency' and 'fpga_pipeline' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::fpga_pipeline(1)]] [[intel::max_concurrency(2)]] void func10();

// expected-error@+3 {{'fpga_pipeline' and 'max_concurrency' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::max_concurrency(8)]] void func11();
[[intel::fpga_pipeline(0)]] void func11();

// expected-error@+3 {{'max_concurrency' and 'fpga_pipeline' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::fpga_pipeline(1)]] void func12();
[[intel::max_concurrency(8)]] void func12();

// Tests that check template parameter support for Intel FPGA fpga_pipeline function attribute.
template <int N>
[[intel::fpga_pipeline(N)]] void func13();

template <int size>
[[intel::fpga_pipeline(0)]] void func14();      // expected-note {{previous attribute is here}}
template <int size>
[[intel::fpga_pipeline(size)]] void func14() {} // expected-warning {{attribute 'fpga_pipeline' is already applied with different arguments}}

void checkTemplates() {
  func13<1>();
  func14<1>();  //expected-note {{in instantiation of function template specialization 'func14<1>' requested here}}
  func14<0>(); // OK
}

// Test that checks expression is not a constant expression.
// expected-note@+1{{declared here}}
int baz();
// expected-error@+2{{expression is not an integral constant expression}}
// expected-note@+1{{non-constexpr function 'baz' cannot be used in a constant expression}}
[[intel::fpga_pipeline(baz() + 1)]] void func15();

// Test that checks expression is a constant expression.
constexpr int bar() { return 0; }
[[intel::fpga_pipeline(bar() + 2)]] void func16(); // OK

// Test that checks wrong function template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.
template <typename Ty>
// expected-error@+1{{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
[[intel::fpga_pipeline(Ty{})]] void func17() {}

struct S {};
void var() {
  //expected-note@+1{{in instantiation of function template specialization 'func17<S>' requested here}}
  func17<S>();
}
