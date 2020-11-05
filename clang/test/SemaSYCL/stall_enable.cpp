// RUN: %clang_cc1 %s -fsyntax-only -fsycl -internal-isystem %S/Inputs -fsycl-is-device -Wno-sycl-2017-compat -DTRIGGER_ERROR -verify
// RUN: %clang_cc1 -fsycl -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -ast-dump -Wno-sycl-2017-compat %s | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

[[intel::stall_enable]] void foo1() {}
// CHECK: FunctionDecl{{.*}}foo1
// CHECK: SYCLIntelStallEnableAttr

#ifdef TRIGGER_ERROR
[[intel::stall_enable(1)]]void bar1() {} // expected-error{{'stall_enable' attribute takes no arguments}}
#endif // TRIGGER_ERROR

void foo2() {
  auto lambda = []() [[intel::stall_enable]] {};
  lambda();
  // CHECK: FunctionDecl{{.*}}foo2
  // CHECK: LambdaExpr
  // CHECK: SYCLIntelStallEnableAttr
}

struct FuncObj {
  [[intel::stall_enable]] void operator()() const {}
};

[[intel::stall_enable]] void func_do_not_ignore() {}

int main() {
  q.submit([&](handler &h) {
  // CHECK-LABEL: FunctionDecl {{.*}}test_kernel1
  // CHECK:       SYCLIntelStallEnableAttr {{.*}}
  h.single_task<class test_kernel1>(
      FuncObj());

  // CHECK-LABEL: FunctionDecl {{.*}}test_kernel2
  // CHECK:       SYCLIntelStallEnableAttr {{.*}}
   h.single_task<class test_kernel2>(
      []() [[intel::stall_enable]]{});

  // CHECK-LABEL: FunctionDecl {{.*}}test_kernel3
  // CHECK:       SYCLIntelStallEnableAttr {{.*}}
  h.single_task<class test_kernel3>(
      []() { func_do_not_ignore(); });
 });
  return 0;
}
