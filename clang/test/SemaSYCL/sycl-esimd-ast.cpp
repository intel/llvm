// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2017 -ast-dump %s | FileCheck %s

// Tests for AST of sycl_explicit_simd function attribute in SYCL 1.2.1.

#include "sycl.hpp"

sycl::queue deviceQueue;

struct FuncObj {
  [[intel::sycl_explicit_simd]] void operator()() const {}
};

[[intel::sycl_explicit_simd]] void func() {}

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

    // Test attribute is propagated.
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel3
    // CHECK:       SYCLSimdAttr {{.*}} Implicit
    // CHECK-NEXT:  SYCLKernelAttr {{.*}} Implicit
    // CHECK-NEXT:  AsmLabelAttr {{.*}} Implicit
    // CHECK-NEXT:  SYCLSimdAttr {{.*}}
    // CHECK-NEXT:  SYCLSimdAttr {{.*}}
    h.single_task<class test_kernel3>(
        []() [[intel::sycl_explicit_simd]] { func(); });
  });
  return 0;
}
