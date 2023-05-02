// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsycl-is-host \
// RUN:   -internal-isystem %S/Inputs -sycl-std=2020 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsycl-is-device \
// RUN:   -internal-isystem %S/Inputs -sycl-std=2020 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsycl-is-host -fcuda-is-device \
// RUN:   -internal-isystem %S/Inputs -sycl-std=2020 -fsyntax-only -verify %s

// Test long double and __float128 support for host.

__float128 foo_0(__float128 P) { return P; }
long double foo_1(long double P) { return P;}

int main() {
  foo_0(1.0);
  foo_1(1.0);
  return 0;
}
// expected-no-diagnostics
