// RUN: %clang_cc1 -x c++ -triple spir64-unknown-linux-sycldevice -std=c++11 -disable-llvm-passes -fsycl-is-device -emit-llvm %s -o - | FileCheck %s

// CHECK: br label %for.cond, !llvm.loop ![[COUNT:[0-9]+]]
// CHECK: br label %while.cond, !llvm.loop ![[DISABLE:[0-9]+]]
// CHECK: br i1 %{{.*}}, label %do.body, label %do.end, !llvm.loop ![[ENABLE:[0-9]+]]

// CHECK: ![[COUNT]] = distinct !{![[COUNT]], ![[COUNT_A:[0-9]+]]}
// CHECK-NEXT: ![[COUNT_A]] = !{!"llvm.loop.unroll.count", i32 8}
void count() {
  [[clang::loop_unroll(8)]]
  for (int i = 0; i < 1000; ++i);
}

// CHECK: ![[DISABLE]] = distinct !{![[DISABLE]], ![[DISABLE_A:[0-9]+]]}
// CHECK-NEXT: ![[DISABLE_A]] = !{!"llvm.loop.unroll.disable"}
void disable() {
  int i = 1000;
  [[clang::loop_unroll(1)]]
  while (i--);
}

// CHECK: ![[ENABLE]] = distinct !{![[ENABLE]], ![[ENABLE_A:[0-9]+]]}
// CHECK-NEXT: ![[ENABLE_A]] = !{!"llvm.loop.unroll.enable"}
void enable() {
  int i = 1000;
  [[clang::loop_unroll]]
  do {} while (i--);
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    count();
    disable();
    enable();
  });
  return 0;
}
