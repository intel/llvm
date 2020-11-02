// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -fsycl -fsycl-is-device -emit-llvm %s -o - | FileCheck %s

// CHECK: br label %while.cond, !llvm.loop ![[MD_NF_1:[0-9]+]]
// CHECK: br label %for.cond3, !llvm.loop ![[MD_NF_2:[0-9]+]]
// CHECK: i1 %cmp18, label %do.body, label %do.end, !llvm.loop ![[MD_NF_3:[0-9]+]]
// CHECK: br label %for.cond20, !llvm.loop ![[MD_NF_4:[0-9]+]]
// CHECK: br label %for.cond41, !llvm.loop ![[MD_NF_5:[0-9]+]]
// CHECK: br label %for.cond50, !llvm.loop ![[MD_NF_6:[0-9]+]]

void nofusion() {
  int a[10];

  int i = 0;
  [[intel::nofusion]] while (i < 10) {
    a[i] += 7;
  }

  for (int i = 0; i < 10; ++i) {
    [[intel::nofusion]] for (int j = 0; j < 10; ++j) {
      a[i] += a[j];
    }
  }

  [[intel::nofusion]] do {
    a[i] += 4;
  }
  while (i < 10)
    ; 

  [[intel::nofusion]] for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      a[i] += a[j];
    }
  }

  int k = 0;
  [[intel::nofusion]] for (auto k : a) {
    k += 2;
  }

  [[intel::nofusion]] for (int i = 0; i < 10; ++i) {
    a[i] += 3;
  }
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    nofusion();
  });
  return 0;
}

// CHECK: ![[MD_NF_1]] = distinct !{![[MD_NF_1]], ![[MD_Nofusion:[0-9]+]]}
// CHECK: ![[MD_Nofusion]] = !{!"llvm.loop.fusion.disable"}
// CHECK: ![[MD_NF_2]] = distinct !{![[MD_NF_2]], ![[MD_Nofusion]]}
// CHECK: ![[MD_NF_3]] = distinct !{![[MD_NF_3]], ![[MD_Nofusion]]}
// CHECK: ![[MD_NF_4]] = distinct !{![[MD_NF_4]], ![[MD_Nofusion]]}
// CHECK: ![[MD_NF_5]] = distinct !{![[MD_NF_5]], ![[MD_Nofusion]]}
// CHECK: ![[MD_NF_6]] = distinct !{![[MD_NF_6]], ![[MD_Nofusion]]}
