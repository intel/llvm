// declcall (P2825) is not standardized. It is a keyword only in SYCL mode,
// in every C++ language mode; elsewhere it is an ordinary identifier.

// RUN: %clang_cc1 -fsycl-is-host -std=c++17 -fsyntax-only -verify=sycl %s
// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -fsyntax-only -verify=sycl %s
// RUN: %clang_cc1 -fsycl-is-host -std=c++2c -fsyntax-only -verify=sycl %s
// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify=nosycl -DNOSYCL %s
// RUN: %clang_cc1 -std=c++2c -pedantic-errors -fsyntax-only -verify=nosycl -DNOSYCL %s

// sycl-no-diagnostics
// nosycl-no-diagnostics

int f(int);

#ifdef NOSYCL
int declcall = 0;
#else
auto p = declcall(f(0));
#endif
