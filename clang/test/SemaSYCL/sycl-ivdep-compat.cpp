// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -Wno-ivdep-compat -verify %s
// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -Wno-ignored-attributes -verify %s

// expected-no-diagnostics

// Test that the warning gets suppressed with a -Wno flag when the
// [[intel::ivdep]] attribute applies a safelen value of 0 or 1 to the loop.

void test_zero() {
  int a[10];
  [[intel::ivdep(0)]] for (int i = 0; i != 10; ++i)
    a[i] = 0;
}

void test_one() {
  int b[10];
  [[intel::ivdep(1)]] for (int i = 0; i != 10; ++i)
    b[i] = 0;
}
