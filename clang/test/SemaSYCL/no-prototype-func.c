// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -emit-llvm -verify %s
// expected-no-diagnostics
int foo();