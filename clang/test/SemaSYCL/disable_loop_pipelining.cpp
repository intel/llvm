// RUN: %clang_cc1 %s -fsyntax-only -internal-isystem %S/Inputs -fsycl-is-device -sycl-std=2020  -DTRIGGER_ERROR -verify
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -ast-dump -sycl-std=2020 %s | FileCheck %s

// Test that checks disable_loop_pipelining attribute support on Function.

#include "sycl.hpp"

sycl::queue deviceQueue;

#ifdef TRIGGER_ERROR
[[intel::disable_loop_pipelining(1)]] void bar1() {} // expected-error{{'disable_loop_pipelining' attribute takes no arguments}}
[[intel::disable_loop_pipelining]] int N;            // expected-error{{'disable_loop_pipelining' attribute only applies to 'for', 'while', 'do' statements, and functions}}
#endif

struct FuncObj {
  [[intel::disable_loop_pipelining]] void operator()() const {}
};

int main() {
  deviceQueue.submit([&](sycl::handler &h) {
    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel1
    // CHECK:       SYCLIntelFPGADisableLoopPipeliningAttr {{.*}}
    h.single_task<class test_kernel1>(
        FuncObj());

    // CHECK-LABEL: FunctionDecl {{.*}}test_kernel2
    // CHECK:       SYCLIntelFPGADisableLoopPipeliningAttr {{.*}}
    h.single_task<class test_kernel2>(
        []() [[intel::disable_loop_pipelining]]{});
  });
  return 0;
}
