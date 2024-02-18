// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-windows -aux-triple x86_64-pc-windows-msvc -fsyntax-only -verify %s
// expected-no-diagnostics

// The test validates ability to build ESIMD code with msvc math functions
// on windows platform.

extern __attribute__((sycl_device)) short _FDtest(float *px);

__attribute__((sycl_device)) __attribute__((sycl_explicit_simd)) void kern() {
  float a;
  _FDtest(&a);
}
