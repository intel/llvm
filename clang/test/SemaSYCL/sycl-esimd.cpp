// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsycl-explicit-simd -fsyntax-only -verify %s

// ----------- Negative tests

__attribute__((sycl_explicit_simd)) // expected-warning {{'sycl_explicit_simd' attribute only applies to functions}}
int N;

__attribute__((sycl_explicit_simd(3))) // expected-error {{'sycl_explicit_simd' attribute takes no arguments}}
void
bar() {}

// -- ESIMD kernel can't call functions with required subgroup size != 1

template <typename ID, typename F>
void kernel0(F f) __attribute__((sycl_kernel)) {
  f();
}

// expected-note@+1{{conflicting attribute is here}}
[[cl::intel_reqd_sub_group_size(2)]] void g0() {}

void test0() {
  // expected-error@+2{{conflicting attributes applied to a SYCL kernel}}
  // expected-note@+1{{conflicting attribute is here}}
  kernel0<class Kernel0>([=]() __attribute__((sycl_explicit_simd)) { g0(); });
}

// -- Usual kernel can't call ESIMD function
template <typename ID, typename F>
void kernel1(F f) __attribute__((sycl_kernel)) {
  f();
}

// expected-note@+1{{attribute is here}}
__attribute__((sycl_explicit_simd)) void g1() {}

void test1() {
  // expected-error@+1{{SYCL kernel without 'sycl_explicit_simd' attribute can't call a function with this attribute}}
  kernel1<class Kernel1>([=]() { g1(); });
}

// ----------- Positive tests

// -- Kernel-function call, both have the attribute, lambda kernel.
template <typename ID, typename F>
void kernel2(F f) __attribute__((sycl_kernel)) {
  f();
}

void g2() __attribute__((sycl_explicit_simd)) {}

void test2() {
  kernel2<class Kernel2>([=]() __attribute__((sycl_explicit_simd)) { g2(); });
}

// --  Class members
class A {
  __attribute__((sycl_explicit_simd))
  A() {}

  __attribute__((sycl_explicit_simd)) void func3() {}
};

// --  Functor object kernel.

template <typename F, typename ID = F>
void kernel3(F f) __attribute__((sycl_kernel)) {
  f();
}

struct Kernel3 {
  void operator()() __attribute__((sycl_explicit_simd)) {}
};

void bar3() {
  kernel3(Kernel3{});
}
