// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -fsycl-is-device -internal-isystem %S/Inputs -emit-llvm %s -o - | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

void nofusion() {
  int a[10];

  int i = 0;
  [[intel::nofusion]] while (i < 10) {
    // CHECK: br label {{.*}}, !llvm.loop ![[MD_NF_1:.*]]
    a[i] += 2;
  }

  [[intel::nofusion]] do {
    // CHECK: br i1 %{{.*}}, !llvm.loop ![[MD_NF_2:.*]]
    a[i] += 3;
  }
  while (i < 10)
    ;

  int k;
  [[intel::nofusion]] for (auto k : a) {
    // CHECK: br label %{{.*}}, !llvm.loop ![[MD_NF_3:.*]]
    k += 4;
  }

  [[intel::nofusion]] for (int i = 0; i < 10; ++i) {
    // CHECK: br label %{{.*}}, !llvm.loop ![[MD_NF_4:.*]]
    a[i] += 5;
  }

  for (int i = 0; i < 10; ++i) {
    // CHECK-NOT: br label %{{.*}}, !llvm.loop !{{.*}}
    [[intel::nofusion]] for (int j = 0; j < 10; ++j) {
      // CHECK: br label %{{.*}}, !llvm.loop ![[MD_NF_6:.*]]
      a[i] += a[j];
    }
  }
}

int main() {
  q.submit([&](handler &h) {
    h.single_task<class kernel_function>([]() { nofusion(); });
  });
  return 0;
}

// CHECK: ![[MD_NF_1]] = distinct !{![[MD_NF_1]], ![[MP:[0-9]+]], ![[MD_Nofusion:[0-9]+]]}
// CHECK-NEXT: ![[MP]] = !{!"llvm.loop.mustprogress"}
// CHECK: ![[MD_Nofusion]] = !{!"llvm.loop.fusion.disable"}
// CHECK: ![[MD_NF_2]] = distinct !{![[MD_NF_2]], ![[MP]], ![[MD_Nofusion]]}
// CHECK: ![[MD_NF_3]] = distinct !{![[MD_NF_3]], ![[MD_Nofusion]]}
// CHECK: ![[MD_NF_4]] = distinct !{![[MD_NF_4]], ![[MP]], ![[MD_Nofusion]]}
// CHECK: ![[MD_NF_6]] = distinct !{![[MD_NF_6]], ![[MP]], ![[MD_Nofusion]]}
