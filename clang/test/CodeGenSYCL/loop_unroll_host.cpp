// RUN: %clangxx -c -S -emit-llvm %s -o - | FileCheck %s
// CHECK: br label %{{.*}}, !llvm.loop ![[COUNT:[0-9]+]]
// CHECK: br label %{{.*}}, !llvm.loop ![[DISABLE:[0-9]+]]
// CHECK: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !llvm.loop ![[ENABLE:[0-9]+]]

int main() {
  // CHECK: ![[COUNT]] = distinct !{![[COUNT]], ![[COUNT_A:[0-9]+]]}
  // CHECK-NEXT: ![[COUNT_A]] = !{!"llvm.loop.unroll.count", i32 4}
  [[clang::loop_unroll(4)]]
  for (int i = 0; i < 100; ++i);
  // CHECK: ![[DISABLE]] = distinct !{![[DISABLE]], ![[DISABLE_A:[0-9]+]]}
  // CHECK-NEXT: ![[DISABLE_A]] = !{!"llvm.loop.unroll.disable"}
  int i = 1000;
  [[clang::loop_unroll(1)]]
  while (i--);
  // CHECK: ![[ENABLE]] = distinct !{![[ENABLE]], ![[ENABLE_A:[0-9]+]]}
  // CHECK-NEXT: ![[ENABLE_A]] = !{!"llvm.loop.unroll.enable"}
  i = 1000;
  [[clang::loop_unroll]]
  do {} while (i--);
  return 0;
}
