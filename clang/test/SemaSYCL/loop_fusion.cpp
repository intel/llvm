// RUN: %clang_cc1 -fsycl-is-device -verify %s

// Tests for incorrect argument values for Intel FPGA loop fusion function attributes
[[intel::loop_fuse(5)]] int a; // expected-error{{'loop_fuse' attribute only applies to functions}}

[[intel::loop_fuse("foo")]] void func() {} // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'const char [4]'}}

[[intel::loop_fuse(1048577)]] void func1() {}        // OK
[[intel::loop_fuse_independent(-1)]] void func2() {} // expected-error{{'loop_fuse_independent' attribute requires a non-negative integral compile time constant expression}}

[[intel::loop_fuse(0, 1)]] void func3() {}             // expected-error{{'loop_fuse' attribute takes no more than 1 argument}}
[[intel::loop_fuse_independent(2, 3)]] void func4() {} // expected-error{{'loop_fuse_independent' attribute takes no more than 1 argument}}

// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
[[intel::loop_fuse]] [[intel::loop_fuse]] void func5() {}
[[intel::loop_fuse_independent(10)]] [[intel::loop_fuse_independent(10)]] void func6() {}

// Tests for merging of different argument values for Intel FPGA loop fusion function attributes
[[intel::loop_fuse]]                     // expected-note {{previous attribute is here}}
[[intel::loop_fuse(10)]] void func7() {} // expected-warning {{attribute 'loop_fuse' is already applied with different arguments}}

[[intel::loop_fuse_independent(5)]]                  // expected-note {{previous attribute is here}}
[[intel::loop_fuse_independent(10)]] void func8() {} // expected-warning {{attribute 'loop_fuse_independent' is already applied with different arguments}}

[[intel::loop_fuse]] void func9();
[[intel::loop_fuse]] void func9(); // OK

[[intel::loop_fuse_independent(10)]] void func10();
[[intel::loop_fuse_independent(10)]] void func10(); // OK

[[intel::loop_fuse(1)]] void func11(); // expected-note {{previous attribute is here}}
[[intel::loop_fuse(3)]] void func11(); // expected-warning {{attribute 'loop_fuse' is already applied with different arguments}}

[[intel::loop_fuse_independent(1)]] void func12(); // expected-note {{previous attribute is here}}
[[intel::loop_fuse_independent(3)]] void func12(); // expected-warning {{attribute 'loop_fuse_independent' is already applied with different arguments}}

[[intel::loop_fuse_independent]]
[[intel::loop_fuse_independent]] void func13() {} // OK

[[intel::loop_fuse_independent]]                      // expected-note {{previous attribute is here}}
[[intel::loop_fuse_independent(10)]] void func14() {} // expected-warning {{attribute 'loop_fuse_independent' is already applied with different arguments}}

// Tests for Intel FPGA loop fusion function attributes compatibility
// expected-error@+2 {{'loop_fuse_independent' and 'loop_fuse' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::loop_fuse]] [[intel::loop_fuse_independent]] void func15();

// expected-error@+2 {{'loop_fuse' and 'loop_fuse_independent' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::loop_fuse_independent]] [[intel::loop_fuse]] void func16();

// expected-error@+2 {{'loop_fuse' and 'loop_fuse_independent' attributes are not compatible}}
// expected-note@+2 {{conflicting attribute is here}}
[[intel::loop_fuse]] void func17();
[[intel::loop_fuse_independent]] void func17();

// expected-error@+2 {{'loop_fuse_independent' and 'loop_fuse' attributes are not compatible}}
// expected-note@+2 {{conflicting attribute is here}}
[[intel::loop_fuse_independent]] void func18();
[[intel::loop_fuse]] void func18();

// Tests that check template parameter support for Intel FPGA loop fusion function attributes
template <int N>
[[intel::loop_fuse(N)]] void func19(); // expected-error{{'loop_fuse' attribute requires a non-negative integral compile time constant expression}}

template <int size>
[[intel::loop_fuse(12)]] void func20();     // expected-note {{previous attribute is here}}
template <int size>
[[intel::loop_fuse(size)]] void func20() {} // expected-warning {{attribute 'loop_fuse' is already applied with different arguments}}

template <int size>
[[intel::loop_fuse_independent(5)]] void func21();      // expected-note {{previous attribute is here}}
template <int size>
[[intel::loop_fuse_independent(size)]] void func21() {} // expected-warning {{attribute 'loop_fuse_independent' is already applied with different arguments}}

void checkTemplates() {
  func19<-1>(); // expected-note{{in instantiation of}}
  func19<0>();  // OK
  func20<20>(); // expected-note {{in instantiation of function template specialization 'func20<20>' requested here}}
  func21<14>(); // expected-note {{in instantiation of function template specialization 'func21<14>' requested here}}
}

// Test that checks expression is not a constant expression.
// expected-note@+1{{declared here}}
int baz();
// expected-error@+2{{expression is not an integral constant expression}}
// expected-note@+1{{non-constexpr function 'baz' cannot be used in a constant expression}}
[[intel::loop_fuse(baz() + 1)]] void func22();

// Test that checks expression is a constant expression.
constexpr int bar() { return 0; }
[[intel::loop_fuse(bar() + 2)]] void func23(); // OK

// Test that checks wrong function template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.
template <typename Ty>
// expected-error@+2 {{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
// expected-error@+1 {{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
[[intel::loop_fuse(Ty{})]] void func24() {}

struct S {};
void test() {
  //expected-note@+1{{in instantiation of function template specialization 'func24<S>' requested here}}
  func24<S>();
  //expected-note@+1{{in instantiation of function template specialization 'func24<float>' requested here}}
  func24<float>();
}
