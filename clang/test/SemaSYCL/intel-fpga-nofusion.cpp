// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsyntax-only -ast-dump -Wno-sycl-2017-compat -verify %s
// expected-no-diagnostics

void foo() {
  int a1[10], a2[10];

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelFPGANofusionAttr {{.*}}
  [[intel::nofusion]] for (int p = 0; p < 10; ++p) {
    a1[p] = a2[p] = 0;
  }

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelFPGANofusionAttr {{.*}}
  int i = 0; 
  [[intel::nofusion]] do {
    a1[i] += 4;
  } 
  while (i < 10);

  // CHECK: AttributedStmt
  // CHECK-NEXT: SYCLIntelFPGANofusionAttr {{.*}}
  [[intel::nofusion]] for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      a1[i] += a1[j];
    }
  }
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    foo();
  });
  return 0;
}
