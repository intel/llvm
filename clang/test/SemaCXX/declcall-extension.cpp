// declcall (P2825) is not standardized. It is a native feature under -fsycl and
// a Clang extension otherwise.

// Extension warning by default in non-SYCL mode.
// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify=ext %s
// Promoted to an error under -pedantic-errors.
// RUN: %clang_cc1 -std=c++2c -pedantic-errors -fsyntax-only -verify=pedantic %s
// No diagnostic in SYCL mode, and available even before C++26.
// RUN: %clang_cc1 -fsycl-is-host -std=c++17 -fsyntax-only -verify=sycl %s
// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -fsyntax-only -verify=sycl %s

// sycl-no-diagnostics

int f(int);

auto p = declcall(f(0)); // ext-warning {{'declcall' is a Clang extension}} \
                         // pedantic-error {{'declcall' is a Clang extension}}
