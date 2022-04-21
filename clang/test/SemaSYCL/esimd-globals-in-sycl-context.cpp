// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

// This test checks usage of an ESIMD global in ESIMD(positive) and SYCL(negative) contexts.

#define ESIMD_PRIVATE __attribute__((opencl_private)) __attribute__((sycl_explicit_simd))
ESIMD_PRIVATE int esimd_glob;

// -- Negative1: usage of ESIMD global reachable from SYCL code

// Deferred diagnostic
void foo_sycl(int x) {
  // expected-error@+1{{ESIMD globals cannot be used in a SYCL context}}
  esimd_glob = x;
}

// Immediate diagnostic
__attribute__((sycl_device)) void init_glob_sycl1(int x) {
  // expected-error@+1{{ESIMD globals cannot be used in a SYCL context}}
  esimd_glob = x;
  //expected-note@+1{{called by}}
  foo_sycl(x);
}

// -- Negative2: usage of ESIMD global reachable from both SYCL and ESIMD code

void foo_sycl_esimd(int x) {
  // expected-error@+1{{ESIMD globals cannot be used in a SYCL context}}
  esimd_glob = x;
}

__attribute__((sycl_device)) void init_glob_sycl2(int x) {
  //expected-note@+1{{called by}}
  foo_sycl_esimd(x);
}

__attribute__((sycl_device)) __attribute__((sycl_explicit_simd)) void init_glob_esimd1(int x) {
  foo_sycl_esimd(x);
}

// -- Positive: usage of ESIMD global in ESIMD code is allowed

void foo_esimd(int x) {
  esimd_glob = x;
}

__attribute__((sycl_device)) __attribute__((sycl_explicit_simd)) void init_glob_esimd2(int x) {
  esimd_glob = x;
  foo_esimd(x);
}
