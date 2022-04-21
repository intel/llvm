// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify -pedantic %s
// expected-no-diagnostics

// This test checks that non-const statics are allowed for ESIMD

static __attribute__((opencl_private)) __attribute__((sycl_explicit_simd)) int esimdPrivStatic;

struct S {
  static __attribute__((opencl_private)) __attribute__((sycl_explicit_simd)) int esimdPrivStaticMember;
};

__attribute__((sycl_device)) __attribute__((sycl_explicit_simd)) void usage() {
  esimdPrivStatic = 42;
  S::esimdPrivStaticMember = 42;
}
