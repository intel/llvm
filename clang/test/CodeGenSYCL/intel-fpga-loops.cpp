// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -fsycl-is-device -emit-llvm %s -o - | FileCheck %s

// CHECK: br label %for.cond, !llvm.loop ![[MD_A:[0-9]+]]
// CHECK: br label %for.cond, !llvm.loop ![[MD_B:[0-9]+]]
// CHECK: br label %for.cond, !llvm.loop ![[MD_C:[0-9]+]]
// CHECK: br label %for.cond2, !llvm.loop ![[MD_D:[0-9]+]]
// CHECK: br label %for.cond,  !llvm.loop ![[MD_E:[0-9]+]]
// CHECK: br label %for.cond2, !llvm.loop ![[MD_F:[0-9]+]]

// CHECK: ![[MD_A]] = distinct !{![[MD_A]], ![[MD_ii:[0-9]+]]}
// CHECK-NEXT: ![[MD_ii]] = !{!"llvm.loop.ii.count", i32 2}
void goo() {
  int a[10];
  [[intelfpga::ii(2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
}

// CHECK: ![[MD_B]] = distinct !{![[MD_B]], ![[MD_max_concurrency:[0-9]+]]}
// CHECK-NEXT: ![[MD_max_concurrency]] = !{!"llvm.loop.max_concurrency.count", i32 2}
void zoo() {
  int a[10];
  [[intelfpga::max_concurrency(2)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
}

// CHECK: ![[MD_C]] = distinct !{![[MD_C]], ![[MD_ii_2:[0-9]+]]}
// CHECK-NEXT: ![[MD_ii_2]] = !{!"llvm.loop.ii.count", i32 4}
template <int A>
void boo() {
  int a[10];
  [[intelfpga::ii(A)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // CHECK: ![[MD_D]] = distinct !{![[MD_D]], ![[MD_ii_3:[0-9]+]]}
  // CHECK-NEXT: ![[MD_ii_3]] = !{!"llvm.loop.ii.count", i32 8}
  [[intelfpga::ii(8)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
}

// CHECK: ![[MD_E]] = distinct !{![[MD_E]], ![[MD_max_concurrency_2:[0-9]+]]}
// CHECK-NEXT: ![[MD_max_concurrency_2]] = !{!"llvm.loop.max_concurrency.count", i32 0}
template <int B>
void foo() {
  int a[10];
  [[intelfpga::max_concurrency(B)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // CHECK: ![[MD_F]] = distinct !{![[MD_F]], ![[MD_max_concurrency_3:[0-9]+]]}
  // CHECK-NEXT: ![[MD_max_concurrency_3]] = !{!"llvm.loop.max_concurrency.count", i32 4}
  [[intelfpga::max_concurrency(4)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    goo();
    zoo();
    boo<4>();
    foo<0>();
  });
  return 0;
}
