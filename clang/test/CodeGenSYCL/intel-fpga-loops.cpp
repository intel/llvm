// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -fsycl-is-device -emit-llvm %s -o - | FileCheck %s

// CHECK: br label %for.cond,   !llvm.loop ![[MD_DLP:[0-9]+]]
// CHECK: br label %for.cond,   !llvm.loop ![[MD_II:[0-9]+]]
// CHECK: br label %for.cond2,  !llvm.loop ![[MD_II_2:[0-9]+]]
// CHECK: br label %for.cond,   !llvm.loop ![[MD_MC:[0-9]+]]
// CHECK: br label %for.cond2,  !llvm.loop ![[MD_MC_2:[0-9]+]]
// CHECK: br label %for.cond,   !llvm.loop ![[MD_LC:[0-9]+]]
// CHECK: br label %for.cond2,  !llvm.loop ![[MD_LC_2:[0-9]+]]
// CHECK: br label %for.cond12, !llvm.loop ![[MD_LC_3:[0-9]+]]
// CHECK: br label %for.cond,   !llvm.loop ![[MD_MI:[0-9]+]]
// CHECK: br label %for.cond2,  !llvm.loop ![[MD_MI_2:[0-9]+]]
// CHECK: br label %for.cond,   !llvm.loop ![[MD_SI:[0-9]+]]
// CHECK: br label %for.cond2,  !llvm.loop ![[MD_SI_2:[0-9]+]]

void disable_loop_pipelining() {
  int a[10];
  // CHECK: ![[MD_DLP]] = distinct !{![[MD_DLP]], ![[MD_dlp:[0-9]+]]}
  // CHECK-NEXT: ![[MD_dlp]] = !{!"llvm.loop.intel.pipelining.enable", i32 0}
  [[intelfpga::disable_loop_pipelining]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

template <int A>
void ii() {
  int a[10];
  // CHECK: ![[MD_II]] = distinct !{![[MD_II]], ![[MD_ii_count:[0-9]+]]}
  // CHECK-NEXT: ![[MD_ii_count]] = !{!"llvm.loop.ii.count", i32 4}
  [[intelfpga::ii(A)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // CHECK: ![[MD_II_2]] = distinct !{![[MD_II_2]], ![[MD_ii_count_2:[0-9]+]]}
  // CHECK-NEXT: ![[MD_ii_count_2]] = !{!"llvm.loop.ii.count", i32 8}
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

template <int A>
void loop_coalesce() {
  int a[10];
  // CHECK: ![[MD_LC]] = distinct !{![[MD_LC]], ![[MD_loop_coalesce:[0-9]+]]}
  // CHECK-NEXT: ![[MD_loop_coalesce]] = !{!"llvm.loop.coalesce.count", i32 2}
  [[intelfpga::loop_coalesce(A)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // CHECK: ![[MD_LC_2]] = distinct !{![[MD_LC_2]], ![[MD_loop_coalesce_2:[0-9]+]]}
  // CHECK-NEXT: ![[MD_loop_coalesce_2]] = !{!"llvm.loop.coalesce.count", i32 4}
  [[intelfpga::loop_coalesce(4)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // CHECK: ![[MD_LC_3]] = distinct !{![[MD_LC_3]], ![[MD_loop_coalesce_3:[0-9]+]]}
  // CHECK-NEXT: ![[MD_loop_coalesce_3]] = !{!"llvm.loop.coalesce.enable"}
  [[intelfpga::loop_coalesce]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

template <int A>
void max_interleaving() {
  int a[10];
  // CHECK: ![[MD_MI]] = distinct !{![[MD_MI]], ![[MD_max_interleaving:[0-9]+]]}
  // CHECK-NEXT: ![[MD_max_interleaving]] = !{!"llvm.loop.max_interleaving.count", i32 3}
  [[intelfpga::max_interleaving(A)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // CHECK: ![[MD_MI_2]] = distinct !{![[MD_MI_2]], ![[MD_max_interleaving_2:[0-9]+]]}
  // CHECK-NEXT: ![[MD_max_interleaving_2]] = !{!"llvm.loop.max_interleaving.count", i32 2}
  [[intelfpga::max_interleaving(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

template <int A>
void speculated_iterations() {
  int a[10];
  // CHECK: ![[MD_SI]] = distinct !{![[MD_SI]], ![[MD_speculated_iterations:[0-9]+]]}
  // CHECK-NEXT: ![[MD_speculated_iterations]] = !{!"llvm.loop.intel.speculated.iterations.count", i32 4}
  [[intelfpga::speculated_iterations(A)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // CHECK: ![[MD_SI_2]] = distinct !{![[MD_SI_2]], ![[MD_speculated_iterations_2:[0-9]+]]}
  // CHECK-NEXT: ![[MD_speculated_iterations_2]] = !{!"llvm.loop.intel.speculated.iterations.count", i32 5}
  [[intelfpga::speculated_iterations(5)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    disable_loop_pipelining();
    ii<4>();
    max_concurrency<0>();
    loop_coalesce<2>();
    max_interleaving<3>();
    speculated_iterations<4>();
  });
  return 0;
}
