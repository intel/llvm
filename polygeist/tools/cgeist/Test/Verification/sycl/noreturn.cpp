// RUN: clang++ -O0 -S -fsycl -fsycl-device-only -w -emit-mlir %s -o - | FileCheck %s

#include <sycl/sycl.hpp>

// CHECK-NOT: "noreturn"

SYCL_EXTERNAL void noret0() __attribute__((__noreturn__));

SYCL_EXTERNAL void noret1() {
  noret0();
}

SYCL_EXTERNAL void foo() {
  noret0();
  noret1();
}
