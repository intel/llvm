// RUN: %clang_cc1 -fsycl-is-device -verify %s

// Test that checks 'no_global_work_offset' attribute support on function.

// Tests for incorrect argument values for Intel FPGA 'no_global_work_offset' function attribute.
[[intel::no_global_work_offset(1)]] int a; // expected-error{{'no_global_work_offset' attribute only applies to functions}}

[[intel::no_global_work_offset("foo")]] void test() {} // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'const char[4]'}}

[[intel::no_global_work_offset(0, 1)]] void test1() {} // expected-error{{'no_global_work_offset' attribute takes no more than 1 argument}}

[[intelfpga::no_global_work_offset]] void RemovedSpell(); // expected-warning {{unknown attribute 'no_global_work_offset' ignored}}

// Test that checks wrong function template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.
template <typename Ty>
// expected-error@+1{{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
[[intel::no_global_work_offset(Ty{})]] void func() {}

struct S {};
void var() {
  // expected-note@+1{{in instantiation of function template specialization 'func<S>' requested here}}
  func<S>();
}

// Test that checks expression is not a constant expression.
// expected-note@+1{{declared here}}
int foo();
// expected-error@+2{{expression is not an integral constant expression}}
// expected-note@+1{{non-constexpr function 'foo' cannot be used in a constant expression}}
[[intel::no_global_work_offset(foo() + 12)]] void func1();

// Test that checks expression is a constant expression.
constexpr int bar() { return 0; }
[[intel::no_global_work_offset(bar() + 12)]] void func2(); // OK

// No diagnostic is thrown since arguments match. Silently ignore duplicate attribute.
[[intel::no_global_work_offset]] void func3();
[[intel::no_global_work_offset(1)]] void func3() {} // OK

[[intel::no_global_work_offset(0)]] void func4(); // expected-note {{previous attribute is here}}
[[intel::no_global_work_offset]] void func4();    // expected-warning{{attribute 'no_global_work_offset' is already applied with different arguments}}

// No diagnostic is emitted because the arguments match.
[[intel::no_global_work_offset(1)]] void func5();
[[intel::no_global_work_offset(1)]] void func5() {} // OK

// Diagnostic is emitted because the arguments mismatch.
[[intel::no_global_work_offset(0)]] void func6(); // expected-note {{previous attribute is here}}
[[intel::no_global_work_offset(1)]] void func6(); // expected-warning{{attribute 'no_global_work_offset' is already applied with different arguments}}

// Test that checks template parameter support on function.
template <int N>
[[intel::no_global_work_offset(0)]] void func7(); // expected-note {{previous attribute is here}}
template <int N>
[[intel::no_global_work_offset(N)]] void func7() {} // expected-warning {{attribute 'no_global_work_offset' is already applied with different arguments}}

int check() {
  func7<1>(); // expected-note {{in instantiation of function template specialization 'func7<1>' requested here}}
  return 0;
}
