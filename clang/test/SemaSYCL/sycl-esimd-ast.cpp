// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump %s | FileCheck %s

// Tests for AST of sycl_explicit_simd function attribute in SYCL.

#include "sycl.hpp"

sycl::queue deviceQueue;

struct FuncObj {
  [[intel::sycl_explicit_simd]] void operator()() const {}
};

int main() {
  deviceQueue.submit([&](sycl::handler &h) {
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel1
    // CHECK:       SYCLSimdAttr {{.*}} Implicit
    // CHECK-NEXT:  SYCLKernelAttr {{.*}} Implicit
    // CHECK-NEXT:  AsmLabelAttr {{.*}} Implicit
    // CHECK-NEXT:  SYCLSimdAttr {{.*}}
    h.single_task<class test_kernel1>(
        FuncObj());

    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel2
    // CHECK:       SYCLSimdAttr {{.*}} Implicit
    // CHECK-NEXT:  SYCLKernelAttr {{.*}} Implicit
    // CHECK-NEXT:  AsmLabelAttr {{.*}} Implicit
    // CHECK-NEXT:  SYCLSimdAttr {{.*}}
    h.single_task<class test_kernel2>(
        []() [[intel::sycl_explicit_simd]]{});

  });
  return 0;
}
