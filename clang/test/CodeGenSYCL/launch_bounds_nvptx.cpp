// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx-unknown-unknown -target-cpu sm_90 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

// Test correct handling of maximum work group size, minimum work groups per
// compute unit and maximum work groups per multi-processor attributes, that
// correspond to CUDA's launch bounds. Expect max_work_group_size,
// min_work_groups_per_cu and max_work_groups_per_mp that are mapped to
// maxntid[xyz], minctasm, and maxclusterrank NVVM annotations respectively.

#include "sycl.hpp"

using namespace sycl;
queue q;

class Foo {
public:
  [[intel::max_work_group_size(2, 4, 8), intel::min_work_groups_per_cu(2),
    intel::max_work_groups_per_mp(4)]] void
  operator()() const {}
};

template <int N> class Functor {
public:
  [[intel::max_work_group_size(N, 4, 8), intel::min_work_groups_per_cu(N),
    intel::max_work_groups_per_mp(N)]] void
  operator()() const {}
};

int main() {
  q.submit([&](handler &h) {
    // Test attribute argument size.
    Foo boo;
    h.single_task<class kernel_name1>(boo);

    // Test attribute is applied on lambda.
    h.single_task<class kernel_name2>(
        [] [[intel::max_work_group_size(2, 4, 8),
             intel::min_work_groups_per_cu(2),
             intel::max_work_groups_per_mp(4)]] () {});

    // Test class template argument.
    Functor<6> f;
    h.single_task<class kernel_name3>(f);
  });
  return 0;
}

// CHECK: define dso_local void @{{.*}}kernel_name1() #0 {{.*}} !min_work_groups_per_cu ![[MWGPC:[0-9]+]] !max_work_groups_per_mp ![[MWGPM:[0-9]+]] !max_work_group_size ![[MWGS:[0-9]+]]
// CHECK: define dso_local void @{{.*}}kernel_name2() #0 {{.*}} !min_work_groups_per_cu ![[MWGPC:[0-9]+]] !max_work_groups_per_mp ![[MWGPM:[0-9]+]] !max_work_group_size ![[MWGS:[0-9]+]]
// CHECK: define dso_local void @{{.*}}kernel_name3() #0 {{.*}} !min_work_groups_per_cu ![[MWGPC_MWGPM:[0-9]+]] !max_work_groups_per_mp ![[MWGPC_MWGPM]] !max_work_group_size ![[MWGS_2:[0-9]+]]

// CHECK: {{.*}}@{{.*}}kernel_name1, !"maxntidx", i32 8}
// CHECK: {{.*}}@{{.*}}kernel_name1, !"maxntidy", i32 4}
// CHECK: {{.*}}@{{.*}}kernel_name1, !"maxntidz", i32 2}
// CHECK: {{.*}}@{{.*}}kernel_name1, !"minctasm", i32 2}
// CHECK: {{.*}}@{{.*}}kernel_name1, !"maxclusterrank", i32 4}
// CHECK: {{.*}}@{{.*}}Foo{{.*}}, !"maxntidx", i32 8}
// CHECK: {{.*}}@{{.*}}Foo{{.*}}, !"maxntidy", i32 4}
// CHECK: {{.*}}@{{.*}}Foo{{.*}}, !"maxntidz", i32 2}
// CHECK: {{.*}}@{{.*}}Foo{{.*}}, !"minctasm", i32 2}
// CHECK: {{.*}}@{{.*}}Foo{{.*}}, !"maxclusterrank", i32 4}
// CHECK: {{.*}}@{{.*}}kernel_name2, !"maxntidx", i32 8}
// CHECK: {{.*}}@{{.*}}kernel_name2, !"maxntidy", i32 4}
// CHECK: {{.*}}@{{.*}}kernel_name2, !"maxntidz", i32 2}
// CHECK: {{.*}}@{{.*}}kernel_name2, !"minctasm", i32 2}
// CHECK: {{.*}}@{{.*}}kernel_name2, !"maxclusterrank", i32 4}
// CHECK: {{.*}}@{{.*}}main{{.*}}, !"maxntidx", i32 8}
// CHECK: {{.*}}@{{.*}}main{{.*}}, !"maxntidy", i32 4}
// CHECK: {{.*}}@{{.*}}main{{.*}}, !"maxntidz", i32 2}
// CHECK: {{.*}}@{{.*}}main{{.*}}, !"minctasm", i32 2}
// CHECK: {{.*}}@{{.*}}main{{.*}}, !"maxclusterrank", i32 4}
// CHECK: {{.*}}@{{.*}}kernel_name3, !"maxntidx", i32 8}
// CHECK: {{.*}}@{{.*}}kernel_name3, !"maxntidy", i32 4}
// CHECK: {{.*}}@{{.*}}kernel_name3, !"maxntidz", i32 6}
// CHECK: {{.*}}@{{.*}}kernel_name3, !"minctasm", i32 6}
// CHECK: {{.*}}@{{.*}}kernel_name3, !"maxclusterrank", i32 6}
// CHECK: {{.*}}@{{.*}}Functor{{.*}}, !"maxntidx", i32 8}
// CHECK: {{.*}}@{{.*}}Functor{{.*}}, !"maxntidy", i32 4}
// CHECK: {{.*}}@{{.*}}Functor{{.*}}, !"maxntidz", i32 6}
// CHECK: {{.*}}@{{.*}}Functor{{.*}}, !"minctasm", i32 6}
// CHECK: {{.*}}@{{.*}}Functor{{.*}}, !"maxclusterrank", i32 6}

// CHECK: ![[MWGPC]] = !{i32 2}
// CHECK: ![[MWGPM]] = !{i32 4}
// CHECK: ![[MWGS]] = !{i32 8, i32 4, i32 2}
// CHECK: ![[MWGPC_MWGPM]] = !{i32 6}
// CHECK: ![[MWGS_2]] = !{i32 8, i32 4, i32 6}
