// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-IR
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace ext::oneapi::experimental;

constexpr auto Props1 = properties{work_group_size_hint<1>};
constexpr auto Props2 = properties{work_group_size_hint<1, 2>};
constexpr auto Props3 = properties{work_group_size_hint<1, 2, 3>};

struct TestKernel_1 {
  void operator()() const {}
  auto get(properties_tag) const { return Props1; }
};

struct TestKernel_2 {
  void operator()() const {}
  auto get(properties_tag) const { return Props2; }
};

struct TestKernel_3 {
  void operator()() const {}
  auto get(properties_tag) const { return Props3; }
};

struct TestKernel_id1 {
  void operator()(id<1>) const {}
  auto get(properties_tag) const { return Props1; }
};

struct TestKernel_id2 {
  void operator()(id<2>) const {}
  auto get(properties_tag) const { return Props2; }
};

struct TestKernel_id3 {
  void operator()(id<3>) const {}
  auto get(properties_tag) const { return Props3; }
};

struct TestKernel_id1_1 {
  template <typename T1> void operator()(id<1>, T1 &) const {}
  auto get(properties_tag) const { return Props1; }
};

struct TestKernel_id2_1 {
  template <typename T1> void operator()(id<2>, T1 &) const {}
  auto get(properties_tag) const { return Props2; }
};

struct TestKernel_id3_1 {
  template <typename T1> void operator()(id<3>, T1 &) const {}
  auto get(properties_tag) const { return Props3; }
};

struct TestKernel_nd_item1 {
  void operator()(nd_item<1>) const {}
  auto get(properties_tag) const { return Props1; }
};

struct TestKernel_nd_item2 {
  void operator()(nd_item<2>) const {}
  auto get(properties_tag) const { return Props2; }
};

struct TestKernel_nd_item3 {
  void operator()(nd_item<3>) const {}
  auto get(properties_tag) const { return Props3; }
};

struct TestKernel_nd_item1_1 {
  template <typename T1> void operator()(nd_item<1>, T1 &) const {}
  auto get(properties_tag) const { return Props1; }
};

struct TestKernel_nd_item2_1 {
  template <typename T1> void operator()(nd_item<2>, T1 &) const {}
  auto get(properties_tag) const { return Props2; }
};

struct TestKernel_nd_item3_1 {
  template <typename T1> void operator()(nd_item<3>, T1 &) const {}
  auto get(properties_tag) const { return Props3; }
};

struct TestKernel_nd_item1_2 {
  template <typename T1, typename T2>
  void operator()(nd_item<1>, T1 &, T2 &) const {}
  auto get(properties_tag) const { return Props1; }
};

struct TestKernel_nd_item2_2 {
  template <typename T1, typename T2>
  void operator()(nd_item<2>, T1 &, T2 &) const {}
  auto get(properties_tag) const { return Props2; }
};

struct TestKernel_nd_item3_2 {
  template <typename T1, typename T2>
  void operator()(nd_item<3>, T1 &, T2 &) const {}
  auto get(properties_tag) const { return Props3; }
};

struct TestKernel_work_group1 {
  void operator()(group<1> G) const {
    G.parallel_for_work_item([&](h_item<1>) {});
  }
  auto get(properties_tag) const { return Props1; }
};

struct TestKernel_work_group2 {
  void operator()(group<2> G) const {
    G.parallel_for_work_item([&](h_item<2>) {});
  }
  auto get(properties_tag) const { return Props2; }
};

struct TestKernel_work_group3 {
  void operator()(group<3> G) const {
    G.parallel_for_work_item([&](h_item<3>) {});
  }
  auto get(properties_tag) const { return Props3; }
};

int main() {
  queue Q;
  event Ev;

  range<1> R1{1};
  range<2> R2{1, 2};
  range<3> R3{1, 2, 3};

  nd_range<1> NDR1{R1, R1};
  nd_range<2> NDR2{R2, R2};
  nd_range<3> NDR3{R3, R3};

  auto Redu1 = reduction<int>(nullptr, plus<int>());
  auto Redu2 = reduction<float>(nullptr, multiplies<float>());

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel0(){{.*}} #[[WGSizeHintAttr0:[0-9]+]]
  Q.single_task<class WGSizeHintKernel0>(TestKernel_1{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel1(){{.*}} #[[WGSizeHintAttr0]]
  Q.single_task<class WGSizeHintKernel1>(Ev, TestKernel_1{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel2(){{.*}} #[[WGSizeHintAttr0]]
  Q.single_task<class WGSizeHintKernel2>({Ev}, TestKernel_1{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel3(){{.*}} #[[WGSizeHintAttr2:[0-9]+]]
  Q.single_task<class WGSizeHintKernel3>(TestKernel_2{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel4(){{.*}} #[[WGSizeHintAttr2]]
  Q.single_task<class WGSizeHintKernel4>(Ev, TestKernel_2{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel5(){{.*}} #[[WGSizeHintAttr2]]
  Q.single_task<class WGSizeHintKernel5>({Ev}, TestKernel_2{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel6(){{.*}} #[[WGSizeHintAttr3:[0-9]+]]
  Q.single_task<class WGSizeHintKernel6>(TestKernel_3{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel7(){{.*}} #[[WGSizeHintAttr3]]
  Q.single_task<class WGSizeHintKernel7>(Ev, TestKernel_3{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel8(){{.*}} #[[WGSizeHintAttr3]]
  Q.single_task<class WGSizeHintKernel8>({Ev}, TestKernel_3{});

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel9(){{.*}} #[[WGSizeHintAttr4:[0-9]+]]
  Q.parallel_for<class WGSizeHintKernel9>(R1, TestKernel_id1{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel10(){{.*}} #[[WGSizeHintAttr4]]
  Q.parallel_for<class WGSizeHintKernel10>(R1, Ev, TestKernel_id1{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel11(){{.*}} #[[WGSizeHintAttr4]]
  Q.parallel_for<class WGSizeHintKernel11>(R1, {Ev}, TestKernel_id1{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel12(){{.*}} #[[WGSizeHintAttr7:[0-9]+]]
  Q.parallel_for<class WGSizeHintKernel12>(R2, TestKernel_id2{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel13(){{.*}} #[[WGSizeHintAttr7]]
  Q.parallel_for<class WGSizeHintKernel13>(R2, Ev, TestKernel_id2{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel14(){{.*}} #[[WGSizeHintAttr7]]
  Q.parallel_for<class WGSizeHintKernel14>(R2, {Ev}, TestKernel_id2{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel15(){{.*}} #[[WGSizeHintAttr8:[0-9]+]]
  Q.parallel_for<class WGSizeHintKernel15>(R3, TestKernel_id3{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel16(){{.*}} #[[WGSizeHintAttr8]]
  Q.parallel_for<class WGSizeHintKernel16>(R3, Ev, TestKernel_id3{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel17(){{.*}} #[[WGSizeHintAttr8]]
  Q.parallel_for<class WGSizeHintKernel17>(R3, {Ev}, TestKernel_id3{});

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel18{{.*}}{{.*}} #[[WGSizeHintAttr4]]
  parallel_for<class WGSizeHintKernel18>(Q, R1, TestKernel_id1_1{}, Redu1);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel19{{.*}}{{.*}} #[[WGSizeHintAttr4]]
  Q.submit([&](handler &CGH) {
    CGH.depends_on(Ev);
    parallel_for<class WGSizeHintKernel19>(Q, R1, TestKernel_id1_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel20{{.*}}{{.*}} #[[WGSizeHintAttr4]]
  Q.submit([&](handler &CGH) {
    CGH.depends_on({Ev});
    parallel_for<class WGSizeHintKernel20>(Q, R1, TestKernel_id1_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel21{{.*}}{{.*}} #[[WGSizeHintAttr7:[0-9]+]]
  parallel_for<class WGSizeHintKernel21>(Q, R2, TestKernel_id2_1{}, Redu1);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel22{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.submit([&](handler &CGH) {
    CGH.depends_on(Ev);
    parallel_for<class WGSizeHintKernel22>(Q, R2, TestKernel_id2_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel23{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.submit([&](handler &CGH) {
    CGH.depends_on({Ev});
    parallel_for<class WGSizeHintKernel23>(Q, R2, TestKernel_id2_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel24{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  parallel_for<class WGSizeHintKernel24>(Q, R3, TestKernel_id3_1{}, Redu1);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel25{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.submit([&](handler &CGH) {
    CGH.depends_on(Ev);
    parallel_for<class WGSizeHintKernel25>(Q, R3, TestKernel_id3_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel26{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.submit([&](handler &CGH) {
    CGH.depends_on({Ev});
    parallel_for<class WGSizeHintKernel26>(Q, R3, TestKernel_id3_1{}, Redu1);
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel27(){{.*}} #[[WGSizeHintAttr10:[0-9]+]]
  nd_launch<class WGSizeHintKernel27>(Q, NDR1, TestKernel_nd_item1{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel28(){{.*}} #[[WGSizeHintAttr10]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeHintKernel28>(CGH, NDR1, TestKernel_nd_item1{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel29(){{.*}} #[[WGSizeHintAttr10]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class WGSizeHintKernel29>(CGH, NDR1, TestKernel_nd_item1{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel30(){{.*}} #[[WGSizeHintAttr11:[0-9]+]]
  nd_launch<class WGSizeHintKernel30>(Q, NDR2, TestKernel_nd_item2{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel31(){{.*}} #[[WGSizeHintAttr11]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeHintKernel31>(CGH, NDR2, TestKernel_nd_item2{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel32(){{.*}} #[[WGSizeHintAttr11]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class WGSizeHintKernel32>(CGH, NDR2, TestKernel_nd_item2{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel33(){{.*}} #[[WGSizeHintAttr12:[0-9]+]]
  nd_launch<class WGSizeHintKernel33>(Q, NDR3, TestKernel_nd_item3{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel34(){{.*}} #[[WGSizeHintAttr12]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeHintKernel34>(CGH, NDR3, TestKernel_nd_item3{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel35(){{.*}} #[[WGSizeHintAttr12]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class WGSizeHintKernel35>(CGH, NDR3, TestKernel_nd_item3{});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel36{{.*}}{{.*}} #[[WGSizeHintAttr4]]
  nd_launch<class WGSizeHintKernel36>(Q, NDR1, TestKernel_nd_item1_1{}, Redu1);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel37{{.*}}{{.*}} #[[WGSizeHintAttr4]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeHintKernel37>(CGH, NDR1, TestKernel_nd_item1_1{},
                                        Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel38{{.*}}{{.*}} #[[WGSizeHintAttr4]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class WGSizeHintKernel38>(CGH, NDR1, TestKernel_nd_item1_1{},
                                        Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel39{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  nd_launch<class WGSizeHintKernel39>(Q, NDR2, TestKernel_nd_item2_1{}, Redu1);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel40{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeHintKernel40>(CGH, NDR2, TestKernel_nd_item2_1{},
                                        Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel41{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class WGSizeHintKernel41>(CGH, NDR2, TestKernel_nd_item2_1{},
                                        Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel42{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  nd_launch<class WGSizeHintKernel42>(Q, NDR3, TestKernel_nd_item3_1{}, Redu1);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel43{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeHintKernel43>(CGH, NDR3, TestKernel_nd_item3_1{},
                                        Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel44{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class WGSizeHintKernel44>(CGH, NDR3, TestKernel_nd_item3_1{},
                                        Redu1);
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel45{{.*}}{{.*}} #[[WGSizeHintAttr4]]
  nd_launch<class WGSizeHintKernel45>(Q, NDR1, TestKernel_nd_item1_2{}, Redu1,
                                      Redu2);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel46{{.*}}{{.*}} #[[WGSizeHintAttr4]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeHintKernel46>(CGH, NDR1, TestKernel_nd_item1_2{},
                                        Redu1, Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel47{{.*}}{{.*}} #[[WGSizeHintAttr4]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeHintKernel47>(CGH, NDR1, TestKernel_nd_item1_2{},
                                        Redu1, Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel48{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  nd_launch<class WGSizeHintKernel48>(Q, NDR2, TestKernel_nd_item2_2{}, Redu1,
                                      Redu2);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel49{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeHintKernel49>(CGH, NDR2, TestKernel_nd_item2_2{},
                                        Redu1, Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel50{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class WGSizeHintKernel50>(CGH, NDR2, TestKernel_nd_item2_2{},
                                        Redu1, Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel51{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  nd_launch<class WGSizeHintKernel51>(Q, NDR3, TestKernel_nd_item3_2{}, Redu1,
                                      Redu2);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel52{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeHintKernel52>(CGH, NDR3, TestKernel_nd_item3_2{},
                                        Redu1, Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel53{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class WGSizeHintKernel53>(CGH, NDR3, TestKernel_nd_item3_2{},
                                        Redu1, Redu2);
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel54(){{.*}} #[[WGSizeHintAttr0]]
  Q.submit([&](handler &CGH) {
    CGH.single_task<class WGSizeHintKernel54>(TestKernel_1{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel55(){{.*}} #[[WGSizeHintAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.single_task<class WGSizeHintKernel55>(TestKernel_2{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel56(){{.*}} #[[WGSizeHintAttr3]]
  Q.submit([&](handler &CGH) {
    CGH.single_task<class WGSizeHintKernel56>(TestKernel_3{});
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel57(){{.*}} #[[WGSizeHintAttr4]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class WGSizeHintKernel57>(R1, TestKernel_id1{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel58(){{.*}} #[[WGSizeHintAttr7]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class WGSizeHintKernel58>(R2, TestKernel_id2{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel59(){{.*}} #[[WGSizeHintAttr8]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class WGSizeHintKernel59>(R3, TestKernel_id3{});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel60{{.*}}{{.*}} #[[WGSizeHintAttr4]]
  Q.submit([&](handler &CGH) {
    parallel_for<class WGSizeHintKernel60>(CGH, R1, TestKernel_id1_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel61{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.submit([&](handler &CGH) {
    parallel_for<class WGSizeHintKernel61>(CGH, R2, TestKernel_id2_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel62{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.submit([&](handler &CGH) {
    parallel_for<class WGSizeHintKernel62>(CGH, R3, TestKernel_id3_1{}, Redu1);
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel63(){{.*}} #[[WGSizeHintAttr10]]
  Q.submit([&](handler &CGH) {
    nd_launch<class WGSizeHintKernel63>(CGH, NDR1, TestKernel_nd_item1{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel64(){{.*}} #[[WGSizeHintAttr11]]
  Q.submit([&](handler &CGH) {
    nd_launch<class WGSizeHintKernel64>(CGH, NDR2, TestKernel_nd_item2{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel65(){{.*}} #[[WGSizeHintAttr12]]
  Q.submit([&](handler &CGH) {
    nd_launch<class WGSizeHintKernel65>(CGH, NDR3, TestKernel_nd_item3{});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel66{{.*}}{{.*}} #[[WGSizeHintAttr4]]
  Q.submit([&](handler &CGH) {
    nd_launch<class WGSizeHintKernel66>(CGH, NDR1, TestKernel_nd_item1_1{},
                                        Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel67{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.submit([&](handler &CGH) {
    nd_launch<class WGSizeHintKernel67>(CGH, NDR2, TestKernel_nd_item2_1{},
                                        Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel68{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.submit([&](handler &CGH) {
    nd_launch<class WGSizeHintKernel68>(CGH, NDR3, TestKernel_nd_item3_1{},
                                        Redu1);
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel69{{.*}}{{.*}} #[[WGSizeHintAttr4]]
  Q.submit([&](handler &CGH) {
    nd_launch<class WGSizeHintKernel69>(CGH, NDR1, TestKernel_nd_item1_2{},
                                        Redu1, Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel70{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.submit([&](handler &CGH) {
    nd_launch<class WGSizeHintKernel70>(CGH, NDR2, TestKernel_nd_item2_2{},
                                        Redu1, Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel71{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.submit([&](handler &CGH) {
    nd_launch<class WGSizeHintKernel71>(CGH, NDR3, TestKernel_nd_item3_2{},
                                        Redu1, Redu2);
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel72(){{.*}} #[[WGSizeHintAttr4]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for_work_group<class WGSizeHintKernel72>(
        R1, TestKernel_work_group1{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel73(){{.*}} #[[WGSizeHintAttr7]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for_work_group<class WGSizeHintKernel73>(
        R2, TestKernel_work_group2{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel74(){{.*}} #[[WGSizeHintAttr8]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for_work_group<class WGSizeHintKernel74>(
        R3, TestKernel_work_group3{});
  });

  return 0;
}

// CHECK-IR: attributes #[[WGSizeHintAttr0]] = { {{.*}}"sycl-work-group-size-hint"="1"
// CHECK-IR: attributes #[[WGSizeHintAttr2]] = { {{.*}}"sycl-work-group-size-hint"="1,2"
// CHECK-IR: attributes #[[WGSizeHintAttr3]] = { {{.*}}"sycl-work-group-size-hint"="1,2,3"
// CHECK-IR: attributes #[[WGSizeHintAttr4]] = { {{.*}}"sycl-work-group-size-hint"="1"
// CHECK-IR: attributes #[[WGSizeHintAttr7]] = { {{.*}}"sycl-work-group-size-hint"="1,2"
// CHECK-IR: attributes #[[WGSizeHintAttr8]] = { {{.*}}"sycl-work-group-size-hint"="1,2,3"
// CHECK-IR: attributes #[[WGSizeHintAttr10]] = { {{.*}}"sycl-work-group-size-hint"="1"
// CHECK-IR: attributes #[[WGSizeHintAttr11]] = { {{.*}}"sycl-work-group-size-hint"="1,2"
// CHECK-IR: attributes #[[WGSizeHintAttr12]] = { {{.*}}"sycl-work-group-size-hint"="1,2,3"
