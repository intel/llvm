// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -fsycl-is-device -emit-llvm %s -o - | FileCheck %s

// CHECK: br label %for.cond,   !llvm.loop ![[MD_DLP:[0-9]+]]
// CHECK: br label %for.cond,   !llvm.loop ![[MD_II:[0-9]+]]
// CHECK: br label %for.cond2,  !llvm.loop ![[MD_II_2:[0-9]+]]
// CHECK: br label %for.cond,   !llvm.loop ![[MD_INITI:[0-9]+]]
// CHECK: br label %for.cond2,  !llvm.loop ![[MD_INITI_2:[0-9]+]]
// CHECK: br label %for.cond,   !llvm.loop ![[MD_MC:[0-9]+]]
// CHECK: br label %for.cond2,  !llvm.loop ![[MD_MC_2:[0-9]+]]
// CHECK: br label %for.cond,   !llvm.loop ![[MD_LC:[0-9]+]]
// CHECK: br label %for.cond2,  !llvm.loop ![[MD_LC_2:[0-9]+]]
// CHECK: br label %for.cond13, !llvm.loop ![[MD_LC_3:[0-9]+]]
// CHECK: br label %for.cond,   !llvm.loop ![[MD_MI:[0-9]+]]
// CHECK: br label %for.cond2,  !llvm.loop ![[MD_MI_2:[0-9]+]]
// CHECK: br label %for.cond13, !llvm.loop ![[MD_MI_3:[0-9]+]]
// CHECK: br label %for.cond,   !llvm.loop ![[MD_SI:[0-9]+]]
// CHECK: br label %for.cond2,  !llvm.loop ![[MD_SI_2:[0-9]+]]
// CHECK: br label %for.cond13, !llvm.loop ![[MD_SI_3:[0-9]+]]
// CHECK: br label %for.cond, !llvm.loop ![[MD_LCA:[0-9]+]]
// CHECK: br label %for.cond2, !llvm.loop ![[MD_LCA_1:[0-9]+]]
// CHECK: br label %for.cond13, !llvm.loop ![[MD_LCA_2:[0-9]+]]

void disable_loop_pipelining() {
  int a[10];
  // CHECK: ![[MD_DLP]] = distinct !{![[MD_DLP]], ![[MP:[0-9]+]], ![[MD_dlp:[0-9]+]]}
  // CHECK-NEXT: ![[MP]] = !{!"llvm.loop.mustprogress"}
  // CHECK-NEXT: ![[MD_dlp]] = !{!"llvm.loop.intel.pipelining.enable", i32 0}
  [[intel::disable_loop_pipelining]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

// Test templated and nontemplated use of [[intel::ii]] on a for loop.
template <int A>
void ii() {
  int a[10];
  // CHECK: ![[MD_II]] = distinct !{![[MD_II]], ![[MP]], ![[MD_ii_count:[0-9]+]]}
  // CHECK-NEXT: ![[MD_ii_count]] = !{!"llvm.loop.ii.count", i32 4}
  [[intel::ii(A)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // CHECK: ![[MD_II_2]] = distinct !{![[MD_II_2]], ![[MP]], ![[MD_ii_count_2:[0-9]+]]}
  // CHECK-NEXT: ![[MD_ii_count_2]] = !{!"llvm.loop.ii.count", i32 8}
  [[intel::ii(8)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

// Test templated and nontemplated use of [[intel::initiation_interval]] on a
// for loop. This shows that the behavior is the same as with [[intel::ii]].
template <int A>
void initiation_interval() {
  int a[10];
  // CHECK: ![[MD_INITI]] = distinct !{![[MD_INITI]], ![[MP]], ![[MD_initi_count:[0-9]+]]}
  // CHECK-NEXT: ![[MD_initi_count]] = !{!"llvm.loop.ii.count", i32 6}
  [[intel::initiation_interval(A)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // CHECK: ![[MD_INITI_2]] = distinct !{![[MD_INITI_2]], ![[MP]], ![[MD_initi_count_2:[0-9]+]]}
  // CHECK-NEXT: ![[MD_initi_count_2]] = !{!"llvm.loop.ii.count", i32 10}
  [[intel::initiation_interval(10)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

template <int A>
void max_concurrency() {
  int a[10];
  // CHECK: ![[MD_MC]] = distinct !{![[MD_MC]], ![[MP]], ![[MD_max_concurrency:[0-9]+]]}
  // CHECK-NEXT: ![[MD_max_concurrency]] = !{!"llvm.loop.max_concurrency.count", i32 0}
  [[intel::max_concurrency(A)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // CHECK: ![[MD_MC_2]] = distinct !{![[MD_MC_2]], ![[MP]], ![[MD_max_concurrency_2:[0-9]+]]}
  // CHECK-NEXT: ![[MD_max_concurrency_2]] = !{!"llvm.loop.max_concurrency.count", i32 4}
  [[intel::max_concurrency(4)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

template <int A>
void loop_coalesce() {
  int a[10];
  // CHECK: ![[MD_LC]] = distinct !{![[MD_LC]], ![[MP]], ![[MD_loop_coalesce:[0-9]+]]}
  // CHECK-NEXT: ![[MD_loop_coalesce]] = !{!"llvm.loop.coalesce.count", i32 2}
  [[intel::loop_coalesce(A)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // CHECK: ![[MD_LC_2]] = distinct !{![[MD_LC_2]], ![[MP]], ![[MD_loop_coalesce_2:[0-9]+]]}
  // CHECK-NEXT: ![[MD_loop_coalesce_2]] = !{!"llvm.loop.coalesce.count", i32 4}
  [[intel::loop_coalesce(4)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // CHECK: ![[MD_LC_3]] = distinct !{![[MD_LC_3]], ![[MP]], ![[MD_loop_coalesce_3:[0-9]+]]}
  // CHECK-NEXT: ![[MD_loop_coalesce_3]] = !{!"llvm.loop.coalesce.enable"}
  [[intel::loop_coalesce]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

template <int A, int B>
void max_interleaving() {
  int a[10];
  // CHECK: ![[MD_MI]] = distinct !{![[MD_MI]], ![[MP]], ![[MD_max_interleaving:[0-9]+]]}
  // CHECK-NEXT: ![[MD_max_interleaving]] = !{!"llvm.loop.max_interleaving.count", i32 3}
  [[intel::max_interleaving(A)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // CHECK: ![[MD_MI_2]] = distinct !{![[MD_MI_2]], ![[MP]], ![[MD_max_interleaving_2:[0-9]+]]}
  // CHECK-NEXT: ![[MD_max_interleaving_2]] = !{!"llvm.loop.max_interleaving.count", i32 2}
  [[intel::max_interleaving(2)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // CHECK: ![[MD_MI_3]] = distinct !{![[MD_MI_3]], ![[MP]], ![[MD_max_interleaving_3:[0-9]+]]}
  // CHECK-NEXT: ![[MD_max_interleaving_3]] = !{!"llvm.loop.max_interleaving.count", i32 0}
  [[intel::max_interleaving(B)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

}

template <int A, int B>
void speculated_iterations() {
  int a[10];
  // CHECK: ![[MD_SI]] = distinct !{![[MD_SI]], ![[MP]], ![[MD_speculated_iterations:[0-9]+]]}
  // CHECK-NEXT: ![[MD_speculated_iterations]] = !{!"llvm.loop.intel.speculated.iterations.count", i32 4}
  [[intel::speculated_iterations(A)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // CHECK: ![[MD_SI_2]] = distinct !{![[MD_SI_2]], ![[MP]], ![[MD_speculated_iterations_2:[0-9]+]]}
  // CHECK-NEXT: ![[MD_speculated_iterations_2]] = !{!"llvm.loop.intel.speculated.iterations.count", i32 5}
  [[intel::speculated_iterations(5)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;

  // CHECK: ![[MD_SI_3]] = distinct !{![[MD_SI_3]], ![[MP]], ![[MD_speculated_iterations_3:[0-9]+]]}
  // CHECK-NEXT: ![[MD_speculated_iterations_3]] = !{!"llvm.loop.intel.speculated.iterations.count", i32 0}
  [[intel::speculated_iterations(B)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

template <int A>
void loop_count_control() {
  int a[10];
  // CHECK: ![[MD_LCA]] = distinct !{![[MD_LCA]], ![[MP:[0-9]+]], ![[MD_loop_count_avg:[0-9]+]]}
  // CHECK-NEXT: ![[MD_loop_count_avg]] = !{!"llvm.loop.intel.loopcount_avg", i32 12}
  [[intel::loop_count_avg(A)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // CHECK: ![[MD_LCA_1]] = distinct !{![[MD_LCA_1]], ![[MP:[0-9]+]], ![[MD_loop_count_max:[0-9]+]]}
  // CHECK-NEXT: ![[MD_loop_count_max]] = !{!"llvm.loop.intel.loopcount_max", i32 4}
  [[intel::loop_count_max(4)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
  // CHECK: ![[MD_LCA_2]] = distinct !{![[MD_LCA_2]], ![[MP:[0-9]+]], ![[MD_loop_count_min:[0-9]+]], ![[MD_loop_count_max_1:[0-9]+]], ![[MD_loop_count_avg_1:[0-9]+]]}
  // CHECK: ![[MD_loop_count_min]] = !{!"llvm.loop.intel.loopcount_min", i32 4}
  // CHECK: ![[MD_loop_count_max_1]] = !{!"llvm.loop.intel.loopcount_max", i32 40}
  // CHECK-NEXT: ![[MD_loop_count_avg_1]] = !{!"llvm.loop.intel.loopcount_avg", i32 21}
  [[intel::loop_count_min(4)]] [[intel::loop_count_max(40)]] [[intel::loop_count_avg(21)]] for (int i = 0; i != 10; ++i)
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
    initiation_interval<6>();
    max_concurrency<0>();
    loop_coalesce<2>();
    max_interleaving<3, 0>();
    speculated_iterations<4, 0>();
    loop_count_control<12>();
  });
  return 0;
}
