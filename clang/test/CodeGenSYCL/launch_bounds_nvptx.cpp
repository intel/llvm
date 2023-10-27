// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2017 -triple nvptx-unknown-unknown -target-cpu sm_90 -disable-llvm-passes -S -emit-llvm -o - %s | FileCheck %s

// Test correct handling of maximum work group size, minimum work groups per
// compute unit and maximum work groups per multi-processor attributes, that
// correspond to CUDA's launch bounds. Expect max_work_group_size,
// min_work_groups_per_cu and max_work_groups_per_mp that are mapped to
// maxntidx, minnctapersm, maxclusterrank PTX directives respectively.

#include "sycl.hpp"

using namespace sycl;
queue q;

class Foo {
public:
  [[intel::max_work_group_size(8, 8, 8), intel::min_work_groups_per_cu(2),
    intel::max_work_groups_per_mp(4)]] void
  operator()() const {}
};

template <int N> class Functor {
public:
  [[intel::max_work_group_size(N, 8, 8), intel::min_work_groups_per_cu(N),
    intel::max_work_groups_per_mp(N)]] void
  operator()() const {}
};

template <int N>
[[intel::max_work_group_size(N, 8, 8), intel::min_work_groups_per_cu(N),
  intel::max_work_groups_per_mp(N)]] void
zoo() {}

[[intel::max_work_group_size(8, 8, 8), intel::min_work_groups_per_cu(2),
  intel::max_work_groups_per_mp(4)]] void
bar() {}

int main() {
  q.submit([&](handler &h) {
    // Test attribute argument size.
    Foo boo;
    h.single_task<class kernel_name1>(boo);

    // Test attribute is applied on lambda.
    h.single_task<class kernel_name2>(
        [] [[intel::max_work_group_size(8, 8, 8),
             intel::min_work_groups_per_cu(2),
             intel::max_work_groups_per_mp(4)]] () {});

    // Test class template argument.
    Functor<6> f;
    h.single_task<class kernel_name3>(f);

    // Test attribute is propagated.
    h.single_task<class kernel_name4>([]() { bar(); });

    // Test function template argument.
    h.single_task<class kernel_name5>([]() { zoo<16>(); });
  });
  return 0;
}

// CHECK: define dso_local void @{{.*}}kernel_name1() #0 {{.*}} !min_work_groups_per_cu ![[MWGPC:[0-9]+]] !max_work_groups_per_mp ![[MWGPM:[0-9]+]] !max_work_group_size ![[MWGS:[0-9]+]]
// CHECK: define dso_local void @{{.*}}kernel_name2() #0 {{.*}} !min_work_groups_per_cu ![[MWGPC:[0-9]+]] !max_work_groups_per_mp ![[MWGPM:[0-9]+]] !max_work_group_size ![[MWGS:[0-9]+]]
// CHECK: define dso_local void @{{.*}}kernel_name3() #0 {{.*}} !min_work_groups_per_cu ![[MWGPC_MWGPM:[0-9]+]] !max_work_groups_per_mp ![[MWGPC_MWGPM]] !max_work_group_size ![[MWGS_2:[0-9]+]]
// CHECK: define dso_local void @{{.*}}kernel_name4() #0 {{.*}} !min_work_groups_per_cu ![[MWGPC:[0-9]+]] !max_work_groups_per_mp ![[MWGPM:[0-9]+]] !max_work_group_size ![[MWGS:[0-9]+]]
// CHECK: define dso_local void @{{.*}}kernel_name5() #0 {{.*}} !min_work_groups_per_cu ![[MWGPC_MWGPM_2:[0-9]+]] !max_work_groups_per_mp ![[MWGPC_MWGPM_2]] !max_work_group_size ![[MWGS_3:[0-9]+]]

// CHECK: {{.*}}@{{.*}}kernel_name1, !"maxntidx", i32 512}
// CHECK: {{.*}}@{{.*}}kernel_name1, !"minnctapersm", i32 2}
// CHECK: {{.*}}@{{.*}}kernel_name1, !"maxclusterrank", i32 4}
// CHECK: {{.*}}@{{.*}}Foo{{.*}}, !"maxntidx", i32 512}
// CHECK: {{.*}}@{{.*}}Foo{{.*}}, !"minnctapersm", i32 2}
// CHECK: {{.*}}@{{.*}}Foo{{.*}}, !"maxclusterrank", i32 4}
// CHECK: {{.*}}@{{.*}}kernel_name2, !"maxntidx", i32 512}
// CHECK: {{.*}}@{{.*}}kernel_name2, !"minnctapersm", i32 2}
// CHECK: {{.*}}@{{.*}}kernel_name2, !"maxclusterrank", i32 4}
// CHECK: {{.*}}@{{.*}}main{{.*}}, !"maxntidx", i32 512}
// CHECK: {{.*}}@{{.*}}main{{.*}}, !"minnctapersm", i32 2}
// CHECK: {{.*}}@{{.*}}main{{.*}}, !"maxclusterrank", i32 4}
// CHECK: {{.*}}@{{.*}}kernel_name3, !"maxntidx", i32 384}
// CHECK: {{.*}}@{{.*}}kernel_name3, !"minnctapersm", i32 6}
// CHECK: {{.*}}@{{.*}}kernel_name3, !"maxclusterrank", i32 6}
// CHECK: {{.*}}@{{.*}}Functor{{.*}}, !"maxntidx", i32 384}
// CHECK: {{.*}}@{{.*}}Functor{{.*}}, !"minnctapersm", i32 6}
// CHECK: {{.*}}@{{.*}}Functor{{.*}}, !"maxclusterrank", i32 6}
// CHECK: {{.*}}@{{.*}}kernel_name4, !"maxntidx", i32 512}
// CHECK: {{.*}}@{{.*}}kernel_name4, !"minnctapersm", i32 2}
// CHECK: {{.*}}@{{.*}}kernel_name4, !"maxclusterrank", i32 4}
// CHECK: {{.*}}@{{.*}}bar{{.*}}, !"maxntidx", i32 512}
// CHECK: {{.*}}@{{.*}}bar{{.*}}, !"minnctapersm", i32 2}
// CHECK: {{.*}}@{{.*}}bar{{.*}}, !"maxclusterrank", i32 4}
// CHECK: {{.*}}@{{.*}}kernel_name5, !"maxntidx", i32 1024}
// CHECK: {{.*}}@{{.*}}kernel_name5, !"minnctapersm", i32 16}
// CHECK: {{.*}}@{{.*}}kernel_name5, !"maxclusterrank", i32 16}
// CHECK: {{.*}}@{{.*}}zoo{{.*}}, !"maxntidx", i32 1024}
// CHECK: {{.*}}@{{.*}}zoo{{.*}}, !"minnctapersm", i32 16}
// CHECK: {{.*}}@{{.*}}zoo{{.*}}, !"maxclusterrank", i32 16}

// CHECK: ![[MWGPC]] = !{i32 2}
// CHECK: ![[MWGPM]] = !{i32 4}
// CHECK: ![[MWGS]] = !{i32 8, i32 8, i32 8}
// CHECK: ![[MWGPC_MWGPM]] = !{i32 6}
// CHECK: ![[MWGS_2]] = !{i32 8, i32 8, i32 6}
// CHECK: ![[MWGPC_MWGPM_2]] = !{i32 16}
// CHECK: ![[MWGS_3]] = !{i32 8, i32 8, i32 16}
