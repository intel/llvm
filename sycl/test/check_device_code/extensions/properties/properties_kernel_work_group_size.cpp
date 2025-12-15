// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-IR
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace ext::oneapi::experimental;

constexpr auto Props1 = properties{work_group_size<1>};
constexpr auto Props2 = properties{work_group_size<1, 2>};
constexpr auto Props3 = properties{work_group_size<1, 2, 3>};

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

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel0(){{.*}} #[[WGSizeAttr0:[0-9]+]]
  Q.single_task<class WGSizeKernel0>(TestKernel_1{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel1(){{.*}} #[[WGSizeAttr0]]
  Q.single_task<class WGSizeKernel1>(Ev, TestKernel_1{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel2(){{.*}} #[[WGSizeAttr0]]
  Q.single_task<class WGSizeKernel2>({Ev}, TestKernel_1{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel3(){{.*}} #[[WGSizeAttr2:[0-9]+]]
  Q.single_task<class WGSizeKernel3>(TestKernel_2{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel4(){{.*}} #[[WGSizeAttr2]]
  Q.single_task<class WGSizeKernel4>(Ev, TestKernel_2{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel5(){{.*}} #[[WGSizeAttr2]]
  Q.single_task<class WGSizeKernel5>({Ev}, TestKernel_2{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel6(){{.*}} #[[WGSizeAttr3:[0-9]+]]
  Q.single_task<class WGSizeKernel6>(TestKernel_3{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel7(){{.*}} #[[WGSizeAttr3]]
  Q.single_task<class WGSizeKernel7>(Ev, TestKernel_3{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel8(){{.*}} #[[WGSizeAttr3]]
  Q.single_task<class WGSizeKernel8>({Ev}, TestKernel_3{});

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel9(){{.*}} #[[WGSizeAttr4:[0-9]+]]
  Q.parallel_for<class WGSizeKernel9>(R1, TestKernel_id1{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel10(){{.*}} #[[WGSizeAttr4]]
  Q.parallel_for<class WGSizeKernel10>(R1, Ev, TestKernel_id1{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel11(){{.*}} #[[WGSizeAttr4]]
  Q.parallel_for<class WGSizeKernel11>(R1, {Ev}, TestKernel_id1{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel12(){{.*}} #[[WGSizeAttr7:[0-9]+]]
  Q.parallel_for<class WGSizeKernel12>(R2, TestKernel_id2{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel13(){{.*}} #[[WGSizeAttr7]]
  Q.parallel_for<class WGSizeKernel13>(R2, Ev, TestKernel_id2{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel14(){{.*}} #[[WGSizeAttr7]]
  Q.parallel_for<class WGSizeKernel14>(R2, {Ev}, TestKernel_id2{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel15(){{.*}} #[[WGSizeAttr8:[0-9]+]]
  Q.parallel_for<class WGSizeKernel15>(R3, TestKernel_id3{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel16(){{.*}} #[[WGSizeAttr8]]
  Q.parallel_for<class WGSizeKernel16>(R3, Ev, TestKernel_id3{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel17(){{.*}} #[[WGSizeAttr8]]
  Q.parallel_for<class WGSizeKernel17>(R3, {Ev}, TestKernel_id3{});

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel18{{.*}}{{.*}} #[[WGSizeAttr4]]
  parallel_for<class WGSizeKernel18>(Q, R1, TestKernel_id1_1{}, Redu1);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel19{{.*}}{{.*}} #[[WGSizeAttr4]]
  Q.submit([&](handler &CGH) {
    CGH.depends_on(Ev);
    parallel_for<class WGSizeKernel19>(Q, R1, TestKernel_id1_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel20{{.*}}{{.*}} #[[WGSizeAttr4]]
  Q.submit([&](handler &CGH) {
    CGH.depends_on({Ev});
    parallel_for<class WGSizeKernel20>(Q, R1, TestKernel_id1_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel21{{.*}}{{.*}} #[[WGSizeAttr7]]
  parallel_for<class WGSizeKernel21>(Q, R2, TestKernel_id2_1{}, Redu1);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel22{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.submit([&](handler &CGH) {
    CGH.depends_on(Ev);
    parallel_for<class WGSizeKernel22>(Q, R2, TestKernel_id2_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel23{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.submit([&](handler &CGH) {
    CGH.depends_on({Ev});
    parallel_for<class WGSizeKernel23>(Q, R2, TestKernel_id2_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel24{{.*}}{{.*}} #[[WGSizeAttr8]]
  parallel_for<class WGSizeKernel24>(Q, R3, TestKernel_id3_1{}, Redu1);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel25{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.submit([&](handler &CGH) {
    CGH.depends_on(Ev);
    parallel_for<class WGSizeKernel25>(Q, R3, TestKernel_id3_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel26{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.submit([&](handler &CGH) {
    CGH.depends_on({Ev});
    parallel_for<class WGSizeKernel26>(Q, R3, TestKernel_id3_1{}, Redu1);
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel27(){{.*}} #[[WGSizeAttr10:[0-9]+]]
  nd_launch<class WGSizeKernel27>(Q, NDR1, TestKernel_nd_item1{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel28(){{.*}} #[[WGSizeAttr10]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeKernel28>(CGH, NDR1, TestKernel_nd_item1{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel29(){{.*}} #[[WGSizeAttr10]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class WGSizeKernel29>(CGH, NDR1, TestKernel_nd_item1{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel30(){{.*}} #[[WGSizeAttr11:[0-9]+]]
  nd_launch<class WGSizeKernel30>(Q, NDR2, TestKernel_nd_item2{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel31(){{.*}} #[[WGSizeAttr11]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeKernel31>(CGH, NDR2, TestKernel_nd_item2{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel32(){{.*}} #[[WGSizeAttr11]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class WGSizeKernel32>(CGH, NDR2, TestKernel_nd_item2{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel33(){{.*}} #[[WGSizeAttr12:[0-9]+]]
  nd_launch<class WGSizeKernel33>(Q, NDR3, TestKernel_nd_item3{});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel34(){{.*}} #[[WGSizeAttr12]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeKernel34>(CGH, NDR3, TestKernel_nd_item3{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel35(){{.*}} #[[WGSizeAttr12]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class WGSizeKernel35>(CGH, NDR3, TestKernel_nd_item3{});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel36{{.*}}{{.*}} #[[WGSizeAttr4]]
  nd_launch<class WGSizeKernel36>(Q, NDR1, TestKernel_nd_item1_1{}, Redu1);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel37{{.*}}{{.*}} #[[WGSizeAttr4]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeKernel37>(CGH, NDR1, TestKernel_nd_item1_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel38{{.*}}{{.*}} #[[WGSizeAttr4]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class WGSizeKernel38>(CGH, NDR1, TestKernel_nd_item1_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel39{{.*}}{{.*}} #[[WGSizeAttr7]]
  nd_launch<class WGSizeKernel39>(Q, NDR2, TestKernel_nd_item2_1{}, Redu1);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel40{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeKernel40>(CGH, NDR2, TestKernel_nd_item2_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel41{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class WGSizeKernel41>(CGH, NDR2, TestKernel_nd_item2_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel42{{.*}}{{.*}} #[[WGSizeAttr8]]
  nd_launch<class WGSizeKernel42>(Q, NDR3, TestKernel_nd_item3_1{}, Redu1);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel43{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeKernel43>(CGH, NDR3, TestKernel_nd_item3_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel44{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class WGSizeKernel44>(CGH, NDR3, TestKernel_nd_item3_1{}, Redu1);
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel45{{.*}}{{.*}} #[[WGSizeAttr4]]
  nd_launch<class WGSizeKernel45>(Q, NDR1, TestKernel_nd_item1_2{}, Redu1,
                                  Redu2);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel46{{.*}}{{.*}} #[[WGSizeAttr4]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeKernel46>(CGH, NDR1, TestKernel_nd_item1_2{}, Redu1,
                                    Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel47{{.*}}{{.*}} #[[WGSizeAttr4]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeKernel47>(CGH, NDR1, TestKernel_nd_item1_2{}, Redu1,
                                    Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel48{{.*}}{{.*}} #[[WGSizeAttr7]]
  nd_launch<class WGSizeKernel48>(Q, NDR2, TestKernel_nd_item2_2{}, Redu1,
                                  Redu2);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel49{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeKernel49>(CGH, NDR2, TestKernel_nd_item2_2{}, Redu1,
                                    Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel50{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class WGSizeKernel50>(CGH, NDR2, TestKernel_nd_item2_2{}, Redu1,
                                    Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel51{{.*}}{{.*}} #[[WGSizeAttr8]]
  nd_launch<class WGSizeKernel51>(Q, NDR3, TestKernel_nd_item3_2{}, Redu1,
                                  Redu2);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel52{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class WGSizeKernel52>(CGH, NDR3, TestKernel_nd_item3_2{}, Redu1,
                                    Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel53{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class WGSizeKernel53>(CGH, NDR3, TestKernel_nd_item3_2{}, Redu1,
                                    Redu2);
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel54(){{.*}} #[[WGSizeAttr0]]
  Q.submit([&](handler &CGH) {
    CGH.single_task<class WGSizeKernel54>(TestKernel_1{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel55(){{.*}} #[[WGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.single_task<class WGSizeKernel55>(TestKernel_2{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel56(){{.*}} #[[WGSizeAttr3]]
  Q.submit([&](handler &CGH) {
    CGH.single_task<class WGSizeKernel56>(TestKernel_3{});
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel57(){{.*}} #[[WGSizeAttr4]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class WGSizeKernel57>(R1, TestKernel_id1{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel58(){{.*}} #[[WGSizeAttr7]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class WGSizeKernel58>(R2, TestKernel_id2{});
  }); // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel59(){{.*}} #[[WGSizeAttr8]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class WGSizeKernel59>(R3, TestKernel_id3{});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel60{{.*}}{{.*}} #[[WGSizeAttr4]]
  Q.submit([&](handler &CGH) {
    parallel_for<class WGSizeKernel60>(CGH, R1, TestKernel_id1_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel61{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.submit([&](handler &CGH) {
    parallel_for<class WGSizeKernel61>(CGH, R2, TestKernel_id2_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel62{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.submit([&](handler &CGH) {
    parallel_for<class WGSizeKernel62>(CGH, R3, TestKernel_id3_1{}, Redu1);
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel63(){{.*}} #[[WGSizeAttr10]]
  Q.submit([&](handler &CGH) {
    nd_launch<class WGSizeKernel63>(CGH, NDR1, TestKernel_nd_item1{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel64(){{.*}} #[[WGSizeAttr11]]
  Q.submit([&](handler &CGH) {
    nd_launch<class WGSizeKernel64>(CGH, NDR2, TestKernel_nd_item2{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel65(){{.*}} #[[WGSizeAttr12]]
  Q.submit([&](handler &CGH) {
    nd_launch<class WGSizeKernel65>(CGH, NDR3, TestKernel_nd_item3{});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel66{{.*}}{{.*}} #[[WGSizeAttr4]]
  Q.submit([&](handler &CGH) {
    nd_launch<class WGSizeKernel66>(CGH, NDR1, TestKernel_nd_item1_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel67{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.submit([&](handler &CGH) {
    nd_launch<class WGSizeKernel67>(CGH, NDR2, TestKernel_nd_item2_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel68{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.submit([&](handler &CGH) {
    nd_launch<class WGSizeKernel68>(CGH, NDR3, TestKernel_nd_item3_1{}, Redu1);
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel69{{.*}}{{.*}} #[[WGSizeAttr4]]
  Q.submit([&](handler &CGH) {
    nd_launch<class WGSizeKernel69>(CGH, NDR1, TestKernel_nd_item1_2{}, Redu1,
                                    Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel70{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.submit([&](handler &CGH) {
    nd_launch<class WGSizeKernel70>(CGH, NDR2, TestKernel_nd_item2_2{}, Redu1,
                                    Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel71{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.submit([&](handler &CGH) {
    nd_launch<class WGSizeKernel71>(CGH, NDR3, TestKernel_nd_item3_2{}, Redu1,
                                    Redu2);
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel72(){{.*}} #[[WGSizeAttr4]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for_work_group<class WGSizeKernel72>(R1,
                                                      TestKernel_work_group1{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel73(){{.*}} #[[WGSizeAttr7]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for_work_group<class WGSizeKernel73>(R2,
                                                      TestKernel_work_group2{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel74(){{.*}} #[[WGSizeAttr8]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for_work_group<class WGSizeKernel74>(R3,
                                                      TestKernel_work_group3{});
  });

  return 0;
}

// CHECK-IR: attributes #[[WGSizeAttr0]] = { {{.*}}"sycl-work-group-size"="1"
// CHECK-IR: attributes #[[WGSizeAttr2]] = { {{.*}}"sycl-work-group-size"="1,2"
// CHECK-IR: attributes #[[WGSizeAttr3]] = { {{.*}}"sycl-work-group-size"="1,2,3"
// CHECK-IR: attributes #[[WGSizeAttr4]] = { {{.*}}"sycl-work-group-size"="1"
// CHECK-IR: attributes #[[WGSizeAttr7]] = { {{.*}}"sycl-work-group-size"="1,2"
// CHECK-IR: attributes #[[WGSizeAttr8]] = { {{.*}}"sycl-work-group-size"="1,2,3"
// CHECK-IR: attributes #[[WGSizeAttr10]] = { {{.*}}"sycl-work-group-size"="1"
// CHECK-IR: attributes #[[WGSizeAttr11]] = { {{.*}}"sycl-work-group-size"="1,2"
// CHECK-IR: attributes #[[WGSizeAttr12]] = { {{.*}}"sycl-work-group-size"="1,2,3"
