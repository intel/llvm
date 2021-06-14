// RUN: %clang_cc1 %s -fsyntax-only -internal-isystem %S/Inputs -fsycl-is-device -Wno-sycl-2017-compat -DTRIGGER_ERROR -verify
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -ast-dump -Wno-sycl-2017-compat %s | FileCheck %s

// Test that checks [[intel::use_stall_enable_clusters]] attribute support on function.
// The metadata to be attached to the functionDecl that the attribute is applied to and
// the attributes do not get propagated to kernel metadata i.e. spir_kernel.

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

// Test attribute is presented on function definition.
[[intel::use_stall_enable_clusters]] void test() {}
// CHECK: FunctionDecl{{.*}}test
// CHECK: SYCLIntelUseStallEnableClustersAttr

// Tests for incorrect argument values for Intel FPGA use_stall_enable_clusters function attribute.
#ifdef TRIGGER_ERROR
[[intel::use_stall_enable_clusters(1)]] void test1() {} // expected-error{{'use_stall_enable_clusters' attribute takes no arguments}}
[[intel::use_stall_enable_clusters]] int test2;         // expected-error{{'use_stall_enable_clusters' attribute only applies to functions}}
#endif

// Test attribute is presented on function call operator (of a function object).
struct FuncObj {
  [[intel::use_stall_enable_clusters]] void operator()() const {}
  // CHECK: CXXRecordDecl{{.*}}implicit struct FuncObj
  // CHECK-NEXT: CXXMethodDecl{{.*}}used operator() 'void () const'
  // CHECK-NEXT-NEXT:SYCLIntelUseStallEnableClustersAttr
};

// Test attribute is presented on lambda function.
void test3() {
  auto lambda = []() [[intel::use_stall_enable_clusters]]{};
  lambda();
  // CHECK: FunctionDecl{{.*}}test3
  // CHECK: LambdaExpr
  // CHECK: SYCLIntelUseStallEnableClustersAttr
}

int main() {
  q.submit([&](handler &h) {
    // Test attribute is not propagated to the kernel.
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel1
    // CHECK-NOT:   SYCLIntelUseStallEnableClustersAttr {{.*}}
    h.single_task<class test_kernel1>(
        FuncObj());

    // Test attribute does not present on LambdaExpr called by kernel.
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel2
    // CHECK-NOT:   SYCLIntelUseStallEnableClustersAttr {{.*}}
    h.single_task<class test_kernel2>(
        []() [[intel::use_stall_enable_clusters]]{});

    // Test attribute is not propagated to the kernel.
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel3
    // CHECK-NOT:   SYCLIntelUseStallEnableClustersAttr {{.*}}
    h.single_task<class test_kernel3>(
        []() { test(); });
  });
  return 0;
}
