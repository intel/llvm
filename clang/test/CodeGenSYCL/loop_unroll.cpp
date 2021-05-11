// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -fsycl-is-device -emit-llvm %s -o - | FileCheck %s

void enable() {
  // CHECK-LABEL: define dso_local spir_func void @_Z6enablev()
  int i = 1000;
  // CHECK: br i1 %{{.*}}, label %do.body, label %do.end, !llvm.loop ![[ENABLE:[0-9]+]]
  [[clang::loop_unroll]]
  do {} while (i--);
}

template <int A>
void count() {
  // CHECK-LABEL: define linkonce_odr spir_func void @_Z5countILi4EEvv()
  // CHECK: br label %for.cond, !llvm.loop ![[COUNT:[0-9]+]]
  [[clang::loop_unroll(8)]]
  for (int i = 0; i < 1000; ++i);
  // CHECK: br label %for.cond2, !llvm.loop ![[COUNT_TEMPLATE:[0-9]+]]
  [[clang::loop_unroll(A)]]
  for (int i = 0; i < 1000; ++i);
}

template <int A>
void disable() {
  // CHECK-LABEL: define linkonce_odr spir_func void @_Z7disableILi1EEvv()
  int i = 1000, j = 100;
  // CHECK: br label %while.cond, !llvm.loop ![[DISABLE:[0-9]+]]
  [[clang::loop_unroll(1)]]
  while (j--);
  // CHECK: br label %while.cond1, !llvm.loop ![[DISABLE_TEMPLATE:[0-9]+]]
  [[clang::loop_unroll(A)]]
  while (i--);
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    enable();
    count<4>();
    disable<1>();
  });
  return 0;
}

// CHECK: ![[ENABLE]] = distinct !{![[ENABLE]], ![[MP:[0-9]+]], ![[ENABLE_A:[0-9]+]]}
// CHECK-NEXT: ![[MP]] = !{!"llvm.loop.mustprogress"}
// CHECK-NEXT: ![[ENABLE_A]] = !{!"llvm.loop.unroll.enable"}
// CHECK: ![[COUNT]] = distinct !{![[COUNT]], ![[MP]], ![[COUNT_A:[0-9]+]]}
// CHECK-NEXT: ![[COUNT_A]] = !{!"llvm.loop.unroll.count", i32 8}
// CHECK: ![[COUNT_TEMPLATE]] = distinct !{![[COUNT_TEMPLATE]], ![[MP]], ![[COUNT_TEMPLATE_A:[0-9]+]]}
// CHECK-NEXT: ![[COUNT_TEMPLATE_A]] = !{!"llvm.loop.unroll.count", i32 4}
// CHECK: ![[DISABLE]] = distinct !{![[DISABLE]], ![[MP]], ![[DISABLE_A:[0-9]+]]}
// CHECK-NEXT: ![[DISABLE_A]] = !{!"llvm.loop.unroll.disable"}
// CHECK: ![[DISABLE_TEMPLATE]] = distinct !{![[DISABLE_TEMPLATE]], ![[MP]], ![[DISABLE_A]]}
