// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fsycl-is-device -internal-isystem %S/Inputs -disable-llvm-passes -triple spir64-unkown-unknown -ast-dump -o - %s | FileCheck %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -disable-llvm-passes -triple spir64-unkown-unknown -ast-dump -o - %s | FileCheck %s

#include "sycl.hpp"

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &h) {    
    // CHECK: LambdaExpr
    // CHECK-NOT: AlwaysInlineAttr
    // CHECK-NOT: NoInlineAttr
    h.parallel_for<class KernelName>([] {});
  });

  q.submit([&](sycl::handler &h) {    
    // CHECK: LambdaExpr
    // CHECK: AlwaysInlineAttr
    h.parallel_for<class KernelNameInline>([]() __attribute__((always_inline)) {});
  });
  
  q.submit([&](sycl::handler &h) {    
    // CHECK: LambdaExpr
    // CHECK: NoInlineAttr
    h.parallel_for<class KernelNameNoInline>([]() __attribute__((noinline)) {});
  });

  /// The flag is ignored for ESIMD kernels
  q.submit([&](sycl::handler &h) {
    // CHECK: LambdaExpr
    // CHECK: SYCLSimdAttr
    // CHECK-NOT: AlwaysInlineAttr
    // CHECK-NOT: NoInlineAttr
    h.parallel_for<class KernelNameESIMD>([]() __attribute__((sycl_explicit_simd)) {});
  });

  return 0;
}
