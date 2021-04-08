// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

// This test checks that SYCL device functions cannot be called from ESIMD context.

__attribute__((sycl_device)) void sycl_func() {}
__attribute__((sycl_device)) void __spirv_reserved_func() {}
__attribute__((sycl_device)) void __sycl_reserved_func() {}
__attribute__((sycl_device)) void __other_reserved_func() {}

// Immediate diagnostic
__attribute__((sycl_device)) __attribute__((sycl_explicit_simd)) void esimd_func1() {
  // expected-error@+1{{SYCL device function cannot be called from an ESIMD context}}
  sycl_func();
  // Reserved SPIRV and SYCL functions are allowed
  __spirv_reserved_func();
  __sycl_reserved_func();
  // expected-error@+1{{SYCL device function cannot be called from an ESIMD context}}
  __other_reserved_func();
}

// Deferred diagnostic
void foo() {
  // expected-error@+1{{SYCL device function cannot be called from an ESIMD context}}
  sycl_func();
}

__attribute__((sycl_device)) __attribute__((sycl_explicit_simd)) void esimd_func2() {
  // expected-note@+1{{called by}}
  foo();
}
