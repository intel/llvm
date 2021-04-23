// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

// This test check the semantics of sycl_esimd_widen attribute

// expected-error@+1{{'sycl_esimd_widen' attribute argument must be 8, 16, or 32}}
__attribute__((sycl_esimd_widen(17))) void foo1() {}
// expected-warning@+1{{attribute 'sycl_esimd_widen' is already applied with different arguments}}
__attribute__((sycl_esimd_widen(8))) __attribute__((sycl_esimd_widen(16))) void foo2() {}

__attribute__((sycl_esimd_widen(8))) void foo3() {}
__attribute__((sycl_esimd_widen(16))) void foo4() {}
__attribute__((sycl_esimd_widen(32))) void foo5() {}
[[intel::sycl_esimd_widen(8)]] void foo6() {}

class A {
  __attribute__((sycl_esimd_widen(8))) void func1() {}
  [[intel::sycl_esimd_widen(8)]] void func2() {}
};

struct Functor {
  void operator()(int) const __attribute__((sycl_esimd_widen(8))) {}
  [[intel::sycl_esimd_widen(8)]] void operator()(float) const {}
};

void test() {
  auto f1 = []() __attribute__((sycl_esimd_widen(8))){};
  auto f2 = []() [[intel::sycl_esimd_widen(8)]]{};
}
