// RUN: %clang_cc1 %s -fsyntax-only -fsycl -internal-isystem %S/Inputs -fsycl-is-device -Wno-sycl-2017-compat -DTRIGGER_ERROR -verify
// RUN: %clang_cc1 -fsycl -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -ast-dump -Wno-sycl-2017-compat %s | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

[[intel::stall_enable]] void test() {} //expected-warning{{'stall_enable' attribute ignored}}

#ifdef TRIGGER_ERROR
[[intel::stall_enable(1)]] void bar1() {} // expected-error{{'stall_enable' attribute takes no arguments}}
[[intel::stall_enable]] int N;            // expected-error{{'stall_enable' attribute only applies to functions}}
#endif

struct FuncObj {
  [[intel::stall_enable]] void operator()() const {}
};

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
    // CHECK-NOT:   SYCLIntelStallEnableAttr {{.*}}
    h.single_task<class test_kernel3>(
        []() { test(); });
  });
  return 0;
}
