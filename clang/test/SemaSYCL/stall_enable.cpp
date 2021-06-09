// RUN: %clang_cc1 %s -fsyntax-only -internal-isystem %S/Inputs -fsycl-is-device -Wno-sycl-2017-compat -DTRIGGER_ERROR -verify
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -ast-dump -Wno-sycl-2017-compat %s | FileCheck %s

// [[intel::use_stall_enable_clusters]] is a function attribute, not a kernel one. The metadata to be attached
// to the FunctionDecl that the attribute is applied to, not collected and applied to callers/the kernel.

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

[[intel::use_stall_enable_clusters]] void test() {}
// CHECK: FunctionDecl{{.*}}test
// CHECK: SYCLIntelUseStallEnableClustersAttr

#ifdef TRIGGER_ERROR
[[intel::use_stall_enable_clusters(1)]] void test1() {} // expected-error{{'use_stall_enable_clusters' attribute takes no arguments}}
[[intel::use_stall_enable_clusters]] int test2;         // expected-error{{'use_stall_enable_clusters' attribute only applies to functions}}
#endif

struct FuncObj {
  [[intel::use_stall_enable_clusters]] void operator()() const {}
  // CHECK: CXXRecordDecl{{.*}}implicit struct FuncObj
  // CHECK-NEXT: CXXMethodDecl{{.*}}used operator() 'void () const'
  // CHECK-NEXT-NEXT:SYCLIntelUseStallEnableClustersAttr
};

void test3() {
  auto lambda = []() [[intel::use_stall_enable_clusters]]{};
  lambda();
  // CHECK: FunctionDecl{{.*}}test3
  // CHECK: LambdaExpr
  // CHECK: SYCLIntelUseStallEnableClustersAttr
}

int main() {
  q.submit([&](handler &h) {
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel1
    // CHECK-NOT:   SYCLIntelUseStallEnableClustersAttr {{.*}}
    h.single_task<class test_kernel1>(
        FuncObj());

    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel2
    // CHECK-NOT:   SYCLIntelUseStallEnableClustersAttr {{.*}}
    h.single_task<class test_kernel2>(
        []() [[intel::use_stall_enable_clusters]]{});

    // Test attribute is not collected and applied to callers/the kernel.
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel3
    // CHECK-NOT:   SYCLIntelUseStallEnableClustersAttr {{.*}}
    h.single_task<class test_kernel3>(
        []() { test(); });
  });
  return 0;
}
