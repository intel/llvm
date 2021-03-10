// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -ast-dump -Wno-sycl-2017-compat -verify %s | FileCheck %s
// expected-no-diagnostics

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

void nofusion() {
  int a1[10], a2[10];

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelFPGANofusionAttr {{.*}}
  [[intel::nofusion]] for (int p = 0; p < 10; ++p) {
    a1[p] = a2[p] = 0;
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelFPGANofusionAttr {{.*}}
  int i = 0;
  [[intel::nofusion]] while (i < 10) {
    a1[i] += 3;
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelFPGANofusionAttr {{.*}}
  for (int i = 0; i < 10; ++i) {
    [[intel::nofusion]] for (int j = 0; j < 10; ++j) {
      a1[i] += a1[j];
    }
  }
}

int main() {
  q.submit([&](handler &h) {
    h.single_task<class kernel_function>([]() { nofusion(); });
  });
  return 0;
}
