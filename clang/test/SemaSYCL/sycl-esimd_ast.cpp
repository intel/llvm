// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -ast-dump -Wno-sycl-2017-compat -sycl-std=2017 -DSYCL2017 %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -ast-dump -Wno-sycl-2017-compat -sycl-std=2020 -DSYCL2020 %s

// Tests for AST of sycl_explicit_simd function attribute.

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
    // CHECK-NEXT:  SYCLSimdAttr {{.*}}
    h.single_task<class test_kernel1>(
        FuncObj());

    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel2
    // CHECK:       SYCLSimdAttr {{.*}} Implicit
    // CHECK-NEXT:  SYCLKernelAttr {{.*}} Implicit
    // CHECK-NEXT:  SYCLSimdAttr {{.*}}
    h.single_task<class test_kernel2>(
        []() [[intel::sycl_explicit_simd]]{});

#if defined(SYCL2017)
    // Test attribute is propagated.
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel3
    // CHECK:       SYCLSimdAttr {{.*}} Implicit
    // CHECK-NEXT:  SYCLKernelAttr {{.*}} Implicit
    // CHECK-NEXT:  SYCLSimdAttr {{.*}}
    // CHECK-NEXT:  SYCLSimdAttr {{.*}}
    h.single_task<class test_kernel3>(
        []() [[intel::sycl_explicit_simd]] { func(); });
#endif //SYCL2017

#if defined(SYCL2020)
    // Test attribute is not propagated.
    // CHECK-LABEL:    FunctionDecl {{.*}}test_kerne4
    // CHECK:          SYCLSimdAttr {{.*}} Implicit
    // CHECK-NEXT:     SYCLKernelAttr {{.*}} Implicit
    // CHECK-NEXT:     SYCLSimdAttr {{.*}}
    // CHECK-NEXT-NOT: SYCLSimdAttr {{.*}}
    h.single_task<class test_kernel4>(
        []() [[intel::sycl_explicit_simd]] { func(); });
#endif // SYCL2020
  });
  return 0;
}
