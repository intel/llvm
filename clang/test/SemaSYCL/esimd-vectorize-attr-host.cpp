// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -verify %s

// This test check the semantics of sycl_esimd_vectorize attribute

// expected-warning@+1{{'sycl_esimd_vectorize' attribute ignored}}
[[intel::sycl_esimd_vectorize(17)]] void foo1() {}

// We explicitly do not support a GNU spelling for this attribute, which is why it is
// treated as an unknown attribute.
// expected-warning@+1{{unknown attribute 'sycl_esimd_vectorize' ignored}}
__attribute__((sycl_esimd_vectorize(8))) void foo2() {}
