// RUN: %clang_cc1 -fsycl-is-device -x c++ -triple spir64-unknown-unknown -verify -fsyntax-only %s
// expected-no-diagnostics
int foo();
