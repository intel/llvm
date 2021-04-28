// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

// This test check the semantics of sycl_esimd_vectorize attribute

// expected-error@+1{{'sycl_esimd_vectorize' attribute argument must be 8, 16, or 32}}
[[intel::sycl_esimd_vectorize(17)]] void foo1() {}
// expected-error@+1{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
[[intel::sycl_esimd_vectorize(3.f)]] void foo3() {}

[[intel::sycl_esimd_vectorize(8)]] void foo4() {}
[[intel::sycl_esimd_vectorize(16)]] void foo5() {}
[[intel::sycl_esimd_vectorize(32)]] void foo6() {}

// We explicitly do not support a GNU spelling for this attribute, which is why it is
// treated as an unknown attribute.
// expected-warning@+1{{unknown attribute 'sycl_esimd_vectorize' ignored}}
__attribute__((sycl_esimd_vectorize(8))) void foo7() {}

// expected-note@+2{{previous attribute is here}}
// expected-warning@+1{{attribute 'sycl_esimd_vectorize' is already applied with different arguments}}
[[intel::sycl_esimd_vectorize(8)]] [[intel::sycl_esimd_vectorize(16)]] void foo8() {}

// expected-note@+1{{previous attribute is here}}
[[intel::sycl_esimd_vectorize(8)]] void foo9();
// expected-warning@+1{{attribute 'sycl_esimd_vectorize' is already applied with different arguments}}
[[intel::sycl_esimd_vectorize(16)]] void foo9() {}

// No diagnostic is emitted because the arguments match.
[[intel::sycl_esimd_vectorize(16)]] void foo10();
[[intel::sycl_esimd_vectorize(16)]] void foo10() {}

// expected-error@+1{{'sycl_esimd_vectorize' attribute only applies to functions}}
[[intel::sycl_esimd_vectorize(8)]] int glob = 0;

class A {
  [[intel::sycl_esimd_vectorize(8)]] void func2() {}
};

struct Functor {
  [[intel::sycl_esimd_vectorize(8)]] void operator()(float) const {}
};

void test() {
  auto f2 = []() [[intel::sycl_esimd_vectorize(8)]]{};
}

template <int N>
[[intel::sycl_esimd_vectorize(N)]] void templateFunc();
