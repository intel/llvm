// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -sycl-std=2017 -Wno-sycl-2017-compat -verify %s

// This test checks specifics of semantic analysis of ESIMD kernels.

// ----------- Negative tests

void foo(
    __attribute__((sycl_explicit_simd)) // expected-warning {{'sycl_explicit_simd' attribute only applies to functions and global variables}}
    int N);

__attribute__((sycl_explicit_simd(3))) // expected-error {{'sycl_explicit_simd' attribute takes no arguments}}
void
bar() {}

// -- ESIMD kernel can't call functions with required subgroup size != 1

template <typename ID, typename F>
void kernel0(const F &f) __attribute__((sycl_kernel)) {
  f();
}

// expected-note@+1{{conflicting attribute is here}}
[[intel::reqd_sub_group_size(2)]] void g0() {}

void test0() {
  // expected-error@+2{{conflicting attributes applied to a SYCL kernel}}
  // expected-note@+1{{conflicting attribute is here}}
  kernel0<class Kernel0>([=]() __attribute__((sycl_explicit_simd)) { g0(); });
}

// -- Usual kernel can't call ESIMD function
template <typename ID, typename F>
void kernel1(const F &f) __attribute__((sycl_kernel)) {
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
void kernel2(const F &f) __attribute__((sycl_kernel)) {
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
void kernel3(const F &f) __attribute__((sycl_kernel)) {
  f();
}

struct Kernel3 {
  void operator()() const __attribute__((sycl_explicit_simd)) {}
};

void bar3() {
  kernel3(Kernel3{});
}

// -- Clang-style [[sycl_explicit_simd]] attribute for functor object kernel.

template <typename F, typename ID = F>
[[clang::sycl_kernel]] void kernel4(const F &f) {
  f();
}

struct Kernel4 {
  [[intel::sycl_explicit_simd]] void operator()() const {}
};

void bar4() {
  kernel4(Kernel4{});
}

// -- Clang-style [[sycl_explicit_simd]] attribute for lambda and free function.

template <typename ID, typename F>
[[clang::sycl_kernel]] void kernel5(const F &f) {
  f();
}

[[intel::sycl_explicit_simd]] void g5() {}

void test5() {
  kernel5<class Kernel5>([=]() [[intel::sycl_explicit_simd]] { g5(); });
}
