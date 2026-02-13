// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fsycl-is-device -internal-isystem %S/Inputs -disable-llvm-passes -triple spir64-unknown-unknown -ast-dump -o - %s | FileCheck %s --check-prefixes=NOINLINE,CHECK
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -disable-llvm-passes -triple spir64-unknown-unknown -ast-dump -o - %s | FileCheck %s --check-prefixes=INLINE,CHECK

// Tests that the appropriate inlining attributes are added to kernel lambda functions,
// with no inline attribute being added when -fno-sycl-force-inline-kernel-lambda is set
// and attribute not explicitly provided.

#include "sycl.hpp"

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &h) {
    // CHECK: LambdaExpr{{.*}}sycl-force-inline-kernel-lambda-ast.cpp:17
    // INLINE: AlwaysInlineAttr
    // NOINLINE-NOT: AlwaysInlineAttr
    h.parallel_for<class KernelName>([] {});
  });

  q.submit([&](sycl::handler &h) {
    // CHECK: LambdaExpr{{.*}}sycl-force-inline-kernel-lambda-ast.cpp:23
    // CHECK: AlwaysInlineAttr
    h.parallel_for<class KernelNameInline>([]() __attribute__((always_inline)) {});
  });

  q.submit([&](sycl::handler &h) {
    // CHECK: LambdaExpr{{.*}}sycl-force-inline-kernel-lambda-ast.cpp:30
    // CHECK: NoInlineAttr
    // CHECK-NOT: AlwaysInlineAttr
    h.parallel_for<class KernelNameNoInline>([]() __attribute__((noinline)) {});
  });

  /// The flag is ignored for ESIMD kernels
  q.submit([&](sycl::handler &h) {
    // CHECK: LambdaExpr{{.*}}sycl-force-inline-kernel-lambda-ast.cpp:39
    // CHECK: SYCLSimdAttr
    // CHECK-NOT: AlwaysInlineAttr
    // CHECK-NOT: NoInlineAttr
    h.parallel_for<class KernelNameESIMD>([]() __attribute__((sycl_explicit_simd)) {});
  });

  return 0;
}
