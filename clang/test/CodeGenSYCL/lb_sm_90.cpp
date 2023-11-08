// RUN: %clang_cc1 -internal-isystem %S/Inputs -triple nvptx-unknown-unknown -target-cpu sm_90 -fsycl-is-device -Wno-c++23-extensions -S -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-IR

// Maximum work groups per multi-processor, mapped to maxclusterrank PTX
// directive, is an SM_90 feature, make sure that correct metadata is generated.

#include "sycl.hpp"

template <int N1, int N2, int N3> class Functor {
public:
  [[intel::max_work_group_size(1, 1, N1), intel::min_work_groups_per_cu(N2),
    intel::max_work_groups_per_mp(N3)]] void
  operator()() const {}
};

int main() {
  sycl::queue Q{};

  sycl::range<1> Gws(32);

  Q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class K1>(Gws, [=] [[intel::max_work_group_size(1, 1, 256),
                                          intel::min_work_groups_per_cu(2),
                                          intel::max_work_groups_per_mp(4)]] (
                                        sycl::id<1>) { volatile int A = 42; });
  });
  // CHECK-IR: !min_work_groups_per_cu [[MWGPCU:![0-9]+]]
  // CHECK-IR: !max_work_groups_per_mp [[MWGPMP:![0-9]+]]
  // CHECK-IR: !max_work_group_size [[MWGS:![0-9]+]]

  Q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class F>(Functor<512, 8, 16>{});
  });
  // CHECK-IR: !min_work_groups_per_cu [[MWGPCU_F:![0-9]+]]
  // CHECK-IR: !max_work_groups_per_mp [[MWGPMP_F:![0-9]+]]
  // CHECK-IR: !max_work_group_size [[MWGS_F:![0-9]+]]

  // CHECK-IR: [[MWGPCU]] = !{i32 2}
  // CHECK-IR: [[MWGPMP]] = !{i32 4}
  // CHECK-IR: [[MWGS]] = !{i32 256, i32 1, i32 1}

  // CHECK-IR: [[MWGPCU_F]] = !{i32 8}
  // CHECK-IR: [[MWGPMP_F]] = !{i32 16}
  // CHECK-IR: [[MWGS_F]] = !{i32 512, i32 1, i32 1}

  return 0;
}
