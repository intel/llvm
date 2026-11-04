// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fsyntax-only \
// RUN:   -verify %s

// This test checks the diagnostics issued for an ODR-use of a variadic function
// declared, but not defined, with the ignored sycl_external attribute.

// expected-warning@+1{{'clang::sycl_external' attribute ignored; a variadic function cannot be called from device code}}
[[clang::sycl_external]] void undefined_variadic_fn_test1(int, ...);
// expected-note@-1{{'undefined_variadic_fn_test1' declared here}}

[[clang::sycl_external]] void use_undefined_variadic_fn_test2() {
  // expected-note@-1{{called by 'use_undefined_variadic_fn_test2'}}
  // FIXME: Clang does not diagnose an ODR-use of a function that is neither
  // defined in this translation unit nor declared sycl_external.
  // expected error once it does.
  // sycl-web's SYCL device code restriction check already diagnoses it.
  (void)&undefined_variadic_fn_test1;
  // expected-error@-1{{SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute}}
}
