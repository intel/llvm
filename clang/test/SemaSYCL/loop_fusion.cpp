// RUN: %clang_cc1 -fsycl -fsycl-is-device -verify %s

[[intel::loop_fuse(5)]] int a; // expected-error{{'loop_fuse' attribute only applies to functions}}

[[intel::loop_fuse("foo")]] void func() {} // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'const char [4]'}}

[[intel::loop_fuse(1048577)]] void func1() {}        // OK
[[intel::loop_fuse_independent(-1)]] void func2() {} // expected-error{{'loop_fuse_independent' attribute requires a non-negative integral compile time constant expression}}

[[intel::loop_fuse(0, 1)]] void func3() {}             // expected-error{{'loop_fuse' attribute takes no more than 1 argument}}
[[intel::loop_fuse_independent(2, 3)]] void func4() {} // expected-error{{'loop_fuse_independent' attribute takes no more than 1 argument}}

// No diagnostic is thrown since arguments match. Duplicate attribute is silently ignored.
[[intel::loop_fuse]] [[intel::loop_fuse]] void func5() {}
[[intel::loop_fuse_independent(10)]] [[intel::loop_fuse_independent(10)]] void func6() {}

[[intel::loop_fuse]]                     // expected-note {{previous attribute is here}}
[[intel::loop_fuse(10)]] void func7() {} // expected-warning {{attribute 'loop_fuse' is already applied with different arguments}}

[[intel::loop_fuse_independent(5)]]                  // expected-note {{previous attribute is here}}
[[intel::loop_fuse_independent(10)]] void func8() {} // expected-warning {{attribute 'loop_fuse_independent' is already applied with different arguments}}

[[intel::loop_fuse]] void func9();
[[intel::loop_fuse]] void func9();

[[intel::loop_fuse_independent(10)]] void func10();
[[intel::loop_fuse_independent(10)]] void func10();

[[intel::loop_fuse(1)]] void func11(); // expected-note {{previous attribute is here}}
[[intel::loop_fuse(3)]] void func11(); // expected-warning {{attribute 'loop_fuse' is already applied with different arguments}}

[[intel::loop_fuse_independent(1)]] void func12(); // expected-note {{previous attribute is here}}
[[intel::loop_fuse_independent(3)]] void func12(); // expected-warning {{attribute 'loop_fuse_independent' is already applied with different arguments}}

// expected-error@+2 {{'loop_fuse_independent' and 'loop_fuse' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::loop_fuse]] [[intel::loop_fuse_independent]] void func13();

// expected-error@+2 {{'loop_fuse' and 'loop_fuse_independent' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::loop_fuse_independent]] [[intel::loop_fuse]] void func14();

// expected-error@+2 {{'loop_fuse' and 'loop_fuse_independent' attributes are not compatible}}
// expected-note@+2 {{conflicting attribute is here}}
[[intel::loop_fuse]] void func15();
[[intel::loop_fuse_independent]] void func15();

// expected-error@+2 {{'loop_fuse_independent' and 'loop_fuse' attributes are not compatible}}
// expected-note@+2 {{conflicting attribute is here}}
[[intel::loop_fuse_independent]] void func16();
[[intel::loop_fuse]] void func16();

template <int N>
[[intel::loop_fuse(N)]] void func17(); // expected-error{{'loop_fuse' attribute requires a non-negative integral compile time constant expression}}

template <typename Ty>
[[intel::loop_fuse(Ty{})]] void func18() {} // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}

template <int size>
[[intel::loop_fuse(12)]] void func19();     // expected-note {{previous attribute is here}}
template <int size>
[[intel::loop_fuse(size)]] void func19() {} // expected-warning {{attribute 'loop_fuse' is already applied with different arguments}}

template <int size>
[[intel::loop_fuse_independent(5)]] void func20();      // expected-note {{previous attribute is here}}
template <int size>
[[intel::loop_fuse_independent(size)]] void func20() {} // expected-warning {{attribute 'loop_fuse_independent' is already applied with different arguments}}

void checkTemplates() {
  func17<-1>();    // expected-note{{in instantiation of}}
  func17<0>();     // OK
  func18<float>(); // expected-note{{in instantiation of}}
  func19<20>(); //expected-note {{in instantiation of function template specialization 'func19<20>' requested here}}
  func20<14>(); //expected-note {{in instantiation of function template specialization 'func20<14>' requested here}}
}

// expected-note@+3 {{declared here}}
// expected-note@+3 {{non-constexpr function 'baz' cannot be used in a constant expression}}
// expected-error@+2 {{expression is not an integral constant expression}}
int baz();
[[intel::loop_fuse(baz())]] void func21();

[[intel::loop_fuse_independent]]
[[intel::loop_fuse_independent]] void func22() {} //OK

[[intel::loop_fuse_independent]]                      // expected-note {{previous attribute is here}}
[[intel::loop_fuse_independent(10)]] void func23() {} // expected-warning {{attribute 'loop_fuse_independent' is already applied with different arguments}}
