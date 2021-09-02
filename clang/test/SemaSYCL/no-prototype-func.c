// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -verify -fsyntax-only %s
// expected-no-diagnostics
int foo();
