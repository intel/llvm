// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -fsycl-is-device -emit-llvm %s -o - | FileCheck %s

// CHECK: br label %for.cond,  !llvm.loop ![[MD_II:[0-9]+]]
// CHECK: br label %for.cond2, !llvm.loop ![[MD_II_2:[0-9]+]]
// CHECK: br label %for.cond,  !llvm.loop ![[MD_MC:[0-9]+]]
// CHECK: br label %for.cond2, !llvm.loop ![[MD_MC_2:[0-9]+]]

template <int A>
void ii() {
  int a[10];
  // CHECK: ![[MD_II]] = distinct !{![[MD_II]], ![[MD_ii:[0-9]+]]}
  // CHECK-NEXT: ![[MD_ii]] = !{!"llvm.loop.ii.count", i32 4}
  [[intelfpga::ii(A)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // CHECK: ![[MD_II_2]] = distinct !{![[MD_II_2]], ![[MD_ii_2:[0-9]+]]}
  // CHECK-NEXT: ![[MD_ii_2]] = !{!"llvm.loop.ii.count", i32 8}
  [[intelfpga::ii(8)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
}

template <int A>
void max_concurrency() {
  int a[10];
  // CHECK: ![[MD_MC]] = distinct !{![[MD_MC]], ![[MD_max_concurrency:[0-9]+]]}
  // CHECK-NEXT: ![[MD_max_concurrency]] = !{!"llvm.loop.max_concurrency.count", i32 0}
  [[intelfpga::max_concurrency(A)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // CHECK: ![[MD_MC_2]] = distinct !{![[MD_MC_2]], ![[MD_max_concurrency_2:[0-9]+]]}
  // CHECK-NEXT: ![[MD_max_concurrency_2]] = !{!"llvm.loop.max_concurrency.count", i32 4}
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
    ii<4>();
    max_concurrency<0>();
  });
  return 0;
}
