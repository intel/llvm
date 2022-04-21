// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -flegacy-pass-manager -mllvm -sycl-opt %s -emit-llvm  -o - | FileCheck %s
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -fno-legacy-pass-manager -mllvm -sycl-opt %s -emit-llvm -o - | FileCheck %s
//
// This test checks that foo (which is @_Z3foov) is called twice after O3 optimizations.
//
// Usually clang with SimplifyCFG pass optimizes constructs like:
// if (i % 2 == 0)
//   func();
// else
//   func();
//
// into one simple func() invocation.
// This behaviour might be wrong in cases when func's behaviour depends on
// a place where it is written.
// There is a relevant discussion about introducing
// a reliable tool for such cases: https://reviews.llvm.org/D85603

// CHECK: tail call spir_func void @_Z3foov()
// CHECK: tail call spir_func void @_Z3foov()

SYCL_EXTERNAL void foo();

SYCL_EXTERNAL void bar(int i) {
  if (i % 2 == 0) {
    foo();
  } else {
    foo();
  }
}
