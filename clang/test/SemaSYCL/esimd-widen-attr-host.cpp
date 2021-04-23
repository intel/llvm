// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -verify %s

// This test check the semantics of sycl_esimd_widen attribute

// expected-warning@+1{{'sycl_esimd_widen' attribute ignored}}
[[intel::sycl_esimd_widen(17)]] void foo1() {}

// expected-warning@+1{{unknown attribute 'sycl_esimd_widen' ignored}}
__attribute__((sycl_esimd_widen(8))) void foo2() {}