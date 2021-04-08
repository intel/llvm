// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

// This test checks that SYCL device functions cannot be called from ESIMD context.

__attribute__((sycl_device)) void sycl_func() {}
__attribute__((sycl_device)) void __reserved_func() {}

// Immediate diagnostic
__attribute__((sycl_device)) __attribute__((sycl_explicit_simd)) void esimd_func1() {
  // expected-error@+1{{SYCL device function cannot be called from ESIMD context}}
  sycl_func();
  // Reserved functions are allowed
  __reserved_func();
}

// Deffered diagnostic
void foo() {
  // expected-error@+1{{SYCL device function cannot be called from ESIMD context}}
  sycl_func();
  // Reserved functions are allowed
  __reserved_func();
}

__attribute__((sycl_device)) __attribute__((sycl_explicit_simd)) void esimd_func2() {
  // expected-note@+1{{called by}}
  foo();
}
