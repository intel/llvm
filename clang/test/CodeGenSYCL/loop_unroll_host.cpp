// RUN: %clangxx -c -S -emit-llvm %s -o - | FileCheck %s
// CHECK: br label %for.cond, !llvm.loop ![[COUNT:[0-9]+]]

int main() {
  // CHECK: ![[COUNT]] = distinct !{![[COUNT]], ![[COUNT_A:[0-9]+]]}
  // CHECK-NEXT: ![[COUNT_A]] = !{!"llvm.loop.unroll.count", i32 4}
  [[clang::unroll(4)]]
  for (int i = 0; i < 100; ++i);
  return 0;
}
