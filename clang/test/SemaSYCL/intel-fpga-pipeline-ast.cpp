// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -ast-dump -Wno-sycl-2017-compat -verify %s | FileCheck %s
// expected-no-diagnostics

// Add AST tests for Loop attribute: [[intel::fpga_pipeline()]].

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

template <int A>
void fpga_pipeline() {
  int a1[10], a2[10];
  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelFPGAPipelineAttr
  [[intel::fpga_pipeline(A)]] for (int p = 0; p < 10; ++p) {
    a1[p] = a2[p] = 0;
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelFPGAPipelineAttr
  // CHECK-NEXT:  ConstantExpr{{.*}}'int'
  // CHECK-NEXT:  value: Int 1
  // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
  int i = 0;
  [[intel::fpga_pipeline]] while (i < 10) {
    a1[i] += 3;
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelFPGAPipelineAttr
  // CHECK-NEXT:  ConstantExpr{{.*}}'int'
  // CHECK-NEXT:  value: Int 1
  // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
  for (int i = 0; i < 10; ++i) {
    [[intel::fpga_pipeline(1)]] for (int j = 0; j < 10; ++j) {
      a1[i] += a1[j];
    }
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelFPGAPipelineAttr
  // CHECK-NEXT:  ConstantExpr{{.*}}'int'
  // CHECK-NEXT:  value: Int 0
  // CHECK-NEXT:  IntegerLiteral{{.*}}0{{$}}
  [[intel::fpga_pipeline(0)]] for (int i = 0; i != 10; ++i)
    a1[i] = 0;

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelFPGAPipelineAttr
  // CHECK-NEXT:  ConstantExpr{{.*}}'int'
  // CHECK-NEXT:  value: Int 1
  // CHECK-NEXT:  IntegerLiteral{{.*}}1{{$}}
  int b = 10;
  [[intel::fpga_pipeline(1)]] do {
    b = b + 1;
  } while (b < 20);

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelFPGAPipelineAttr
  int c[] = {0, 1, 2, 3, 4, 5};
  [[intel::fpga_pipeline(A)]] for (int n : c) { n *= 2; }
}

int main() {
  q.submit([&](handler &h) {
    h.single_task<class kernel_function>([]() { fpga_pipeline<1>(); });
  });
  return 0;
}
