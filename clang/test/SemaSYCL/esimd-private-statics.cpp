// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -pedantic %s
// expected-no-diagnostics

// This test checks that non-const statics are allowed for ESIMD

static __attribute__((opencl_private)) __attribute__((sycl_explicit_simd)) int esimdPrivStatic;

__attribute__((sycl_device)) __attribute__((sycl_explicit_simd)) void usage() {
  // expected-note@+1{{called by}}
  esimdPrivStatic = 42;
}
