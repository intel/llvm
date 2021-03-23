// RUN: %clang_cc1 -fsycl-is-host -triple spir64 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s
// CHECK: br label %{{.*}}, !llvm.loop ![[COUNT:[0-9]+]]
// CHECK: br label %{{.*}}, !llvm.loop ![[DISABLE:[0-9]+]]
// CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !llvm.loop ![[ENABLE:[0-9]+]]
// CHECK: br label %{{.*}}, !llvm.loop ![[COUNT_TEMPLATE:[0-9]+]]
// CHECK: br label %{{.*}}, !llvm.loop ![[DISABLE_TEMPLATE:[0-9]+]]

template <int A>
void unroll() {
  [[clang::loop_unroll(A)]]
  for (int i = 0; i < 1000; ++i);
}

int main() {
  // CHECK: ![[COUNT]] = distinct !{![[COUNT]], ![[MP:[0-9]+]], ![[COUNT_A:[0-9]+]]}
  // CHECK-NEXT: ![[MP]] = !{!"llvm.loop.mustprogress"}
  // CHECK-NEXT: ![[COUNT_A]] = !{!"llvm.loop.unroll.count", i32 4}
  [[clang::loop_unroll(4)]]
  for (int i = 0; i < 100; ++i);
  // CHECK: ![[DISABLE]] = distinct !{![[DISABLE]], ![[MP]], ![[DISABLE_A:[0-9]+]]}
  // CHECK-NEXT: ![[DISABLE_A]] = !{!"llvm.loop.unroll.disable"}
  int i = 1000;
  [[clang::loop_unroll(1)]]
  while (i--);
  // CHECK: ![[ENABLE]] = distinct !{![[ENABLE]], ![[MP]], ![[ENABLE_A:[0-9]+]]}
  // CHECK-NEXT: ![[ENABLE_A]] = !{!"llvm.loop.unroll.enable"}
  i = 1000;
  [[clang::loop_unroll]]
  do {} while (i--);

  // CHECK: ![[COUNT_TEMPLATE]] = distinct !{![[COUNT_TEMPLATE]], ![[MP]], ![[COUNT_TEMPLATE_A:[0-9]+]]}
  // CHECK-NEXT: ![[COUNT_TEMPLATE_A]] = !{!"llvm.loop.unroll.count", i32 8}
  unroll<8>();
  // CHECK: ![[DISABLE_TEMPLATE]] = distinct !{![[DISABLE_TEMPLATE]], ![[MP]], ![[DISABLE_A]]}
  unroll<1>();
  return 0;
}
