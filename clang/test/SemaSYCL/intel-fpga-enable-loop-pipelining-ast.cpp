// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -Wno-sycl-2017-compat -ast-dump %s | FileCheck %s

// Add AST tests for Loop attribute: [[intel::enable_loop_pipelining]].

#include "sycl.hpp"

using namespace sycl;
queue q;

void fpga_enable_loop_pipelining() {
  int a1[10], a2[10];
  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelEnableLoopPipeliningAttr
  [[intel::enable_loop_pipelining]] for (int p = 0; p < 10; ++p) {
    a1[p] = a2[p] = 0;
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelEnableLoopPipeliningAttr
  int i = 0;
  [[intel::enable_loop_pipelining]] while (i < 10) {
    a1[i] += 3;
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelEnableLoopPipeliningAttr
  for (int i = 0; i < 10; ++i) {
    [[intel::enable_loop_pipelining]] for (int j = 0; j < 10; ++j) {
      a1[i] += a1[j];
    }
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelEnableLoopPipeliningAttr
  int b = 10;
  [[intel::enable_loop_pipelining]] do {
    b = b + 1;
  } while (b < 20);

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelEnableLoopPipeliningAttr
  int c[] = {0, 1, 2, 3, 4, 5};
  [[intel::enable_loop_pipelining]] for (int n : c) { n *= 2; }
}

void foo() {
  q.submit([&](handler &h) {
    h.single_task<class kernel_function>([]() { fpga_enable_loop_pipelining(); });
  });
}
