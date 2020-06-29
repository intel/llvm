// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -emit-llvm -verify -fms-extensions %s
// expected-no-diagnostics
int foo();