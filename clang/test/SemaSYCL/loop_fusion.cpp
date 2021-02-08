// RUN: %clang_cc1 -fsycl -fsycl-is-device -verify %s

[[intel::loop_fuse(5)]] int a; // expected-error{{'loop_fuse' attribute only applies to functions}}

[[intel::loop_fuse("foo")]] void func() {} // expected-error{{'loop_fuse' attribute requires an integer constant}}

[[intel::loop_fuse(1048577)]] void func1() {}        // expected-error{{'loop_fuse' attribute requires integer constant between 0 and 1048576 inclusive}}
[[intel::loop_fuse_independent(-1)]] void func2() {} // expected-error{{'loop_fuse_independent' attribute requires integer constant between 0 and 1048576 inclusive}}

[[intel::loop_fuse(0, 1)]] void func3() {}             // expected-error{{'loop_fuse' attribute takes no more than 1 argument}}
[[intel::loop_fuse_independent(2, 3)]] void func4() {} // expected-error{{'loop_fuse_independent' attribute takes no more than 1 argument}}

// No diagnostic is thrown since arguments match. Duplicate attribute is silently ignored.
[[intel::loop_fuse]] [[intel::loop_fuse]] void func5() {}
[[intel::loop_fuse_independent(10)]] [[intel::loop_fuse_independent(10)]] void func6() {}

[[intel::loop_fuse]] [[intel::loop_fuse(10)]] void func7() {}                            // expected-warning {{attribute 'loop_fuse' is already applied with different arguments}}
[[intel::loop_fuse_independent(5)]] [[intel::loop_fuse_independent(10)]] void func8() {} // expected-warning {{attribute 'loop_fuse_independent' is already applied with different arguments}}

[[intel::loop_fuse]] void func9();
[[intel::loop_fuse]] void func9();

[[intel::loop_fuse_independent(10)]] void func10();
[[intel::loop_fuse_independent(10)]] void func10();

[[intel::loop_fuse(1)]] void func11();
[[intel::loop_fuse(3)]] void func11(); // expected-warning {{attribute 'loop_fuse' is already applied with different arguments}}

[[intel::loop_fuse_independent(1)]] void func12();
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
[[intel::loop_fuse(N)]] void func17(); // expected-error{{'loop_fuse' attribute requires integer constant between 0 and 1048576 inclusive}}

template <typename Ty>
[[intel::loop_fuse(Ty{})]] void func18() {} // expected-error{{'loop_fuse' attribute requires an integer constant}}

void checkTemplates() {
  func17<-1>();    // expected-note{{in instantiation of}}
  func18<float>(); // expected-note{{in instantiation of}}
}

int baz();
[[intel::loop_fuse(baz())]] void func19(); // expected-error{{'loop_fuse' attribute requires an integer constant}}
