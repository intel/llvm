// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -ast-dump %s | FileCheck %s

// Add AST tests for Loop attributes: [[intel::enable_loop_pipelining]],
// [[intel::max_interleaving()]], [[intel::loop_coalesce]],
// [[intel::max_concurrency()]], [[intel::initiation_interval()]],
// and [[intel::speculated_iterations()]].

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

void fpga_max_concurrency() {
  int a1[10], a2[10];
  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelMaxConcurrencyAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 0
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
  [[intel::max_concurrency(0)]] for (int p = 0; p < 10; ++p) {
    a1[p] = a2[p] = 0;
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelMaxConcurrencyAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 4
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 4
  int i = 0;
  [[intel::max_concurrency(4)]] while (i < 10) {
    a1[i] += 3;
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelMaxConcurrencyAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 8
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 8
  for (int i = 0; i < 10; ++i) {
    [[intel::max_concurrency(8)]] for (int j = 0; j < 10; ++j) {
      a1[i] += a1[j];
    }
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelMaxConcurrencyAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 12
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 12
  int b = 10;
  [[intel::max_concurrency(12)]] do {
    b = b + 1;
  } while (b < 20);

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelMaxConcurrencyAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 16
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 16
  int c[] = {0, 1, 2, 3, 4, 5};
  [[intel::max_concurrency(16)]] for (int n : c) { n *= 2; }
}

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

void fpga_speculated_iterations() {
  int a1[10], a2[10];
  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelSpeculatedIterationsAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 0
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
  [[intel::speculated_iterations(0)]] for (int p = 0; p < 10; ++p) {
    a1[p] = a2[p] = 0;
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelSpeculatedIterationsAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 12
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 12
  int i = 0;
  [[intel::speculated_iterations(12)]] while (i < 10) {
    a1[i] += 3;
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelSpeculatedIterationsAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 16
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 16
  for (int i = 0; i < 10; ++i) {
    [[intel::speculated_iterations(16)]] for (int j = 0; j < 10; ++j) {
      a1[i] += a1[j];
    }
  }
  
  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelSpeculatedIterationsAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 8
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 8
  int b = 10;
  [[intel::speculated_iterations(8)]] do {
    b = b + 1;
  } while (b < 20);

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelSpeculatedIterationsAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 22
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 22
  int c[] = {0, 1, 2, 3, 4, 5};
  [[intel::speculated_iterations(22)]] for (int n : c) { n *= 2; }
}

void fpga_initiation_interval() {
  int a1[10], a2[10];
  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelInitiationIntervalAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 10
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 10
  [[intel::initiation_interval(10)]] for (int p = 0; p < 10; ++p) {
    a1[p] = a2[p] = 0;
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelInitiationIntervalAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 12
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 12
  int i = 0;
  [[intel::initiation_interval(12)]] while (i < 10) {
    a1[i] += 3;
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelInitiationIntervalAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 16
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 16
  for (int i = 0; i < 10; ++i) {
    [[intel::initiation_interval(16)]] for (int j = 0; j < 10; ++j) {
      a1[i] += a1[j];
    }
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelInitiationIntervalAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 8
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 8
  int b = 10;
  [[intel::initiation_interval(8)]] do {
    b = b + 1;
  } while (b < 20);

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelInitiationIntervalAttr
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 22
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 22
  int c[] = {0, 1, 2, 3, 4, 5};
  [[intel::initiation_interval(22)]] for (int n : c) { n *= 2; }
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
      fpga_enable_loop_pipelining();
      fpga_max_concurrency();
      fpga_max_interleaving();
      fpga_speculated_iterations();
      fpga_initiation_interval();
      fpga_loop_coalesce();
    });
  });
}
