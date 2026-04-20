// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump %s | FileCheck %s

// Add AST tests for Loop attributes: [[intel::enable_loop_pipelining]],
// [[intel::max_interleaving()]], [[intel::loop_coalesce]],
// [[intel::max_concurrency()]], [[intel::initiation_interval()]],
// and [[intel::speculated_iterations()]].

#include "sycl.hpp"

using namespace sycl;
queue q;

void fpga_max_interleaving() {
  int a1[10], a2[10];
  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelMaxInterleavingAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 0
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
  [[intel::max_interleaving(0)]] for (int p = 0; p < 10; ++p) {
    a1[p] = a2[p] = 0;
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelMaxInterleavingAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 1
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
  int i = 0;
  [[intel::max_interleaving(1)]] while (i < 10) {
    a1[i] += 3;
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelMaxInterleavingAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 1
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
  for (int i = 0; i < 10; ++i) {
    [[intel::max_interleaving(1)]] for (int j = 0; j < 10; ++j) {
      a1[i] += a1[j];
    }
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelMaxInterleavingAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 0
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
  int b = 10;
  [[intel::max_interleaving(0)]] do {
    b = b + 1;
  } while (b < 20);

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelMaxInterleavingAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 1
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
  int c[] = {0, 1, 2, 3, 4, 5};
  [[intel::max_interleaving(1)]] for (int n : c) { n *= 2; }
}

void fpga_loop_coalesce() {
  int a1[10], a2[10];
  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelLoopCoalesceAttr
  [[intel::loop_coalesce]] for (int p = 0; p < 10; ++p) {
    a1[p] = a2[p] = 0;
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelLoopCoalesceAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 12
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 12
  int i = 0;
  [[intel::loop_coalesce(12)]] while (i < 10) {
    a1[i] += 3;
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelLoopCoalesceAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 16
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 16
  for (int i = 0; i < 10; ++i) {
    [[intel::loop_coalesce(16)]] for (int j = 0; j < 10; ++j) {
      a1[i] += a1[j];
    }
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelLoopCoalesceAttr
  int b = 10;
  [[intel::loop_coalesce]] do {
    b = b + 1;
  } while (b < 20);

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelLoopCoalesceAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 22
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 22
  int c[] = {0, 1, 2, 3, 4, 5};
  [[intel::loop_coalesce(22)]] for (int n : c) { n *= 2; }
}

void foo() {
  q.submit([&](handler &h) {
    h.single_task<class kernel_function>([]() {
      fpga_max_interleaving();
      fpga_loop_coalesce();
    });
  });
}
