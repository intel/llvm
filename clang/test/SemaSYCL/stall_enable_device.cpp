// RUN: %clang_cc1 %s -fsyntax-only -internal-isystem %S/Inputs -fsycl-is-device -DTRIGGER_ERROR -verify
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -ast-dump %s | FileCheck %s

// Test that checks [[intel::use_stall_enable_clusters]] attribute support on function.

#include "sycl.hpp"

using namespace sycl;
queue q;

// Test attribute is presented on function definition.
[[intel::use_stall_enable_clusters]] void func() {}
// CHECK: FunctionDecl{{.*}}used func 'void ()'
// CHECK-NEXT: CompoundStmt{{.*}}
// CHECK-NEXT-NEXT: SYCLIntelUseStallEnableClustersAttr

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

// Test attribute is presented on lambda function(applied to a function type for the lambda's call operator).
void test3() {
  auto lambda = []() [[intel::use_stall_enable_clusters]]{};
  lambda();
  // CHECK: FunctionDecl{{.*}}test3
  // CHECK: LambdaExpr
  // CHECK: SYCLIntelUseStallEnableClustersAttr
}

// Test attribute is presented on functor.
// CHECK: CXXRecordDecl{{.*}}referenced class Functor definition
// CHECK: CXXRecordDecl{{.*}} implicit class Functor
// CHECK: AccessSpecDecl{{.*}} public
// CHECK-NEXT: CXXMethodDecl{{.*}}used operator() 'void () const'
// CHECK-NEXT-NEXT: SYCLIntelUseStallEnableClustersAttr
class Functor {
public:
  [[intel::use_stall_enable_clusters]] void operator()() const {
  }
};

int main() {
  q.submit([&](handler &h) {
    // Test attribute is propagated to the kernel.
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel1
    // CHECK:       SYCLIntelUseStallEnableClustersAttr
    h.single_task<class test_kernel1>(
        FuncObj());

    // Test attribute is presented on LambdaExpr called by kernel.
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel2
    // CHECK:       SYCLIntelUseStallEnableClustersAttr
    h.single_task<class test_kernel2>(
        []() [[intel::use_stall_enable_clusters]]{});

    // Test attribute is not propagated to the kernel.
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel3
    // CHECK-NOT:   SYCLIntelUseStallEnableClustersAttr
    h.single_task<class test_kernel3>(
        []() { func(); });

    // Test attribute is applied to kernel if directly applied through functor.
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel4
    // CHECK:       SYCLIntelUseStallEnableClustersAttr
    Functor f2;
    h.single_task<class test_kernel4>(f2);
  });
  return 0;
}
