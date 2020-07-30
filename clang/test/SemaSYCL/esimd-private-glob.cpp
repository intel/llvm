// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsycl-explicit-simd -fsyntax-only -verify %s
// expected-no-diagnostics
int x = 0;
