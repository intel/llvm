// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-IR
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace ext::oneapi::experimental;

static constexpr auto sub_group_size_1 = properties{sub_group_size<1>};

struct TestKernel {
  void operator()() const {}
  auto get(properties_tag) const { return sub_group_size_1; }
};

struct TestKernel_id1 {
  void operator()(id<1>) const {}
  auto get(properties_tag) const { return sub_group_size_1; }
};

struct TestKernel_id2 {
  void operator()(id<2>) const {}
  auto get(properties_tag) const { return sub_group_size_1; }
};

struct TestKernel_id3 {
  void operator()(id<3>) const {}
  auto get(properties_tag) const { return sub_group_size_1; }
};

struct TestKernel_id1_1 {
  template <typename T1> void operator()(id<1>, T1 &) const {}
  auto get(properties_tag) const { return sub_group_size_1; }
};

struct TestKernel_id2_1 {
  template <typename T1> void operator()(id<2>, T1 &) const {}
  auto get(properties_tag) const { return sub_group_size_1; }
};

struct TestKernel_id3_1 {
  template <typename T1> void operator()(id<3>, T1 &) const {}
  auto get(properties_tag) const { return sub_group_size_1; }
};

struct TestKernel_nd_item1 {
  void operator()(nd_item<1>) const {}
  auto get(properties_tag) const { return sub_group_size_1; }
};

struct TestKernel_nd_item2 {
  void operator()(nd_item<2>) const {}
  auto get(properties_tag) const { return sub_group_size_1; }
};

struct TestKernel_nd_item3 {
  void operator()(nd_item<3>) const {}
  auto get(properties_tag) const { return sub_group_size_1; }
};

struct TestKernel_nd_item1_1 {
  template <typename T1> void operator()(nd_item<1>, T1 &) const {}
  auto get(properties_tag) const { return sub_group_size_1; }
};

struct TestKernel_nd_item2_1 {
  template <typename T1> void operator()(nd_item<2>, T1 &) const {}
  auto get(properties_tag) const { return sub_group_size_1; }
};

struct TestKernel_nd_item3_1 {
  template <typename T1> void operator()(nd_item<3>, T1 &) const {}
  auto get(properties_tag) const { return sub_group_size_1; }
};

struct TestKernel_nd_item1_2 {
  template <typename T1, typename T2>
  void operator()(nd_item<1>, T1 &, T2 &) const {}
  auto get(properties_tag) const { return sub_group_size_1; }
};

struct TestKernel_nd_item2_2 {
  template <typename T1, typename T2>
  void operator()(nd_item<2>, T1 &, T2 &) const {}
  auto get(properties_tag) const { return sub_group_size_1; }
};

struct TestKernel_nd_item3_2 {
  template <typename T1, typename T2>
  void operator()(nd_item<3>, T1 &, T2 &) const {}
  auto get(properties_tag) const { return sub_group_size_1; }
};

struct TestKernel_work_group1 {
  void operator()(group<1> G) const {
    G.parallel_for_work_item([&](h_item<1>) {});
  }
  auto get(properties_tag) const { return sub_group_size_1; }
};

struct TestKernel_work_group2 {
  void operator()(group<2> G) const {
    G.parallel_for_work_item([&](h_item<2>) {});
  }
  auto get(properties_tag) const { return sub_group_size_1; }
};

struct TestKernel_work_group3 {
  void operator()(group<3> G) const {
    G.parallel_for_work_item([&](h_item<3>) {});
  }
  auto get(properties_tag) const { return sub_group_size_1; }
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

  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel0(){{.*}} #[[SGSizeAttr0:[0-9]+]]
  Q.single_task<class SGSizeKernel0>(TestKernel{});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel1(){{.*}} #[[SGSizeAttr0]]
  Q.single_task<class SGSizeKernel1>(Ev, TestKernel{});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel2(){{.*}} #[[SGSizeAttr0]]
  Q.single_task<class SGSizeKernel2>({Ev}, TestKernel{});

  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel9(){{.*}} #[[SGSizeAttr2:[0-9]+]]
  Q.parallel_for<class SGSizeKernel9>(R1, TestKernel_id1{});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel10(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel10>(R1, Ev, TestKernel_id1{});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel11(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel11>(R1, {Ev}, TestKernel_id1{});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel12(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel12>(R2, TestKernel_id2{});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel13(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel13>(R2, Ev, TestKernel_id2{});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel14(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel14>(R2, {Ev}, TestKernel_id2{});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel15(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel15>(R3, TestKernel_id3{});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel16(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel16>(R3, Ev, TestKernel_id3{});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel17(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel17>(R3, {Ev}, TestKernel_id3{});

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel18{{.*}}{{.*}} #[[SGSizeAttr2:[0-9]+]]
  parallel_for<class SGSizeKernel18>(Q, R1, TestKernel_id1_1{}, Redu1);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel19{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.depends_on(Ev);
    parallel_for<class SGSizeKernel19>(Q, R1, TestKernel_id1_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel20{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.depends_on({Ev});
    parallel_for<class SGSizeKernel20>(Q, R1, TestKernel_id1_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel21{{.*}}{{.*}} #[[SGSizeAttr2]]
  parallel_for<class SGSizeKernel21>(Q, R2, TestKernel_id2_1{}, Redu1);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel22{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.depends_on(Ev);
    parallel_for<class SGSizeKernel22>(Q, R2, TestKernel_id2_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel23{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.depends_on({Ev});
    parallel_for<class SGSizeKernel23>(Q, R2, TestKernel_id2_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel24{{.*}}{{.*}} #[[SGSizeAttr2]]
  parallel_for<class SGSizeKernel24>(Q, R3, TestKernel_id3_1{}, Redu1);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel25{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.depends_on(Ev);
    parallel_for<class SGSizeKernel25>(Q, R3, TestKernel_id3_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel26{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.depends_on({Ev});
    parallel_for<class SGSizeKernel26>(Q, R3, TestKernel_id3_1{}, Redu1);
  });

  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel27(){{.*}} #[[SGSizeAttr6:[0-9]+]]
  nd_launch<class SGSizeKernel27>(Q, NDR1, TestKernel_nd_item1{});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel28(){{.*}} #[[SGSizeAttr6]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class SGSizeKernel28>(CGH, NDR1, TestKernel_nd_item1{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel29(){{.*}} #[[SGSizeAttr6]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class SGSizeKernel29>(CGH, NDR1, TestKernel_nd_item1{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel30(){{.*}} #[[SGSizeAttr6]]
  nd_launch<class SGSizeKernel30>(Q, NDR2, TestKernel_nd_item2{});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel31(){{.*}} #[[SGSizeAttr6]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class SGSizeKernel31>(CGH, NDR2, TestKernel_nd_item2{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel32(){{.*}} #[[SGSizeAttr6]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class SGSizeKernel32>(CGH, NDR2, TestKernel_nd_item2{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel33(){{.*}} #[[SGSizeAttr6]]
  nd_launch<class SGSizeKernel33>(Q, NDR3, TestKernel_nd_item3{});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel34(){{.*}} #[[SGSizeAttr6]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class SGSizeKernel34>(CGH, NDR3, TestKernel_nd_item3{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel35(){{.*}} #[[SGSizeAttr6]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class SGSizeKernel35>(CGH, NDR3, TestKernel_nd_item3{});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel36{{.*}}{{.*}} #[[SGSizeAttr2]]
  nd_launch<class SGSizeKernel36>(Q, NDR1, TestKernel_nd_item1_1{}, Redu1);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel37{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class SGSizeKernel37>(CGH, NDR1, TestKernel_nd_item1_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel38{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class SGSizeKernel38>(CGH, NDR1, TestKernel_nd_item1_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel39{{.*}}{{.*}} #[[SGSizeAttr2]]
  nd_launch<class SGSizeKernel39>(Q, NDR2, TestKernel_nd_item2_1{}, Redu1);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel40{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class SGSizeKernel40>(CGH, NDR2, TestKernel_nd_item2_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel41{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class SGSizeKernel41>(CGH, NDR2, TestKernel_nd_item2_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel42{{.*}}{{.*}} #[[SGSizeAttr2]]
  nd_launch<class SGSizeKernel42>(Q, NDR3, TestKernel_nd_item3_1{}, Redu1);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel43{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class SGSizeKernel43>(CGH, NDR3, TestKernel_nd_item3_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel44{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class SGSizeKernel44>(CGH, NDR3, TestKernel_nd_item3_1{}, Redu1);
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel45{{.*}}{{.*}} #[[SGSizeAttr2]]
  nd_launch<class SGSizeKernel45>(Q, NDR1, TestKernel_nd_item1_2{}, Redu1,
                                  Redu2);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel46{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class SGSizeKernel46>(CGH, NDR1, TestKernel_nd_item1_2{}, Redu1,
                                    Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel47{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class SGSizeKernel47>(CGH, NDR1, TestKernel_nd_item1_2{}, Redu1,
                                    Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel48{{.*}}{{.*}} #[[SGSizeAttr2]]
  nd_launch<class SGSizeKernel48>(Q, NDR2, TestKernel_nd_item2_2{}, Redu1,
                                  Redu2);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel49{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class SGSizeKernel49>(CGH, NDR2, TestKernel_nd_item2_2{}, Redu1,
                                    Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel50{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class SGSizeKernel50>(CGH, NDR2, TestKernel_nd_item2_2{}, Redu1,
                                    Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel51{{.*}}{{.*}} #[[SGSizeAttr2]]
  nd_launch<class SGSizeKernel51>(Q, NDR3, TestKernel_nd_item3_2{}, Redu1,
                                  Redu2);
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel52{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on(Ev);
    nd_launch<class SGSizeKernel52>(CGH, NDR3, TestKernel_nd_item3_2{}, Redu1,
                                    Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel53{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.depends_on({Ev});
    nd_launch<class SGSizeKernel53>(CGH, NDR3, TestKernel_nd_item3_2{}, Redu1,
                                    Redu2);
  });

  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel54(){{.*}} #[[SGSizeAttr0]]
  Q.submit([&](handler &CGH) {
    CGH.single_task<class SGSizeKernel54>(TestKernel{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel55(){{.*}} #[[SGSizeAttr0]]
  Q.submit([&](handler &CGH) {
    CGH.single_task<class SGSizeKernel55>(TestKernel{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel56(){{.*}} #[[SGSizeAttr0]]
  Q.submit([&](handler &CGH) {
    CGH.single_task<class SGSizeKernel56>(TestKernel{});
  });

  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel57(){{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class SGSizeKernel57>(R1, TestKernel_id1{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel58(){{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class SGSizeKernel58>(R2, TestKernel_id2{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel59(){{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<class SGSizeKernel59>(R3, TestKernel_id3{});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel60{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    parallel_for<class SGSizeKernel60>(CGH, R1, TestKernel_id1_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel61{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    parallel_for<class SGSizeKernel61>(CGH, R2, TestKernel_id2_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel62{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    parallel_for<class SGSizeKernel62>(CGH, R3, TestKernel_id3_1{}, Redu1);
  });

  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel63(){{.*}} #[[SGSizeAttr6]]
  Q.submit([&](handler &CGH) {
    nd_launch<class SGSizeKernel63>(CGH, NDR1, TestKernel_nd_item1{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel64(){{.*}} #[[SGSizeAttr6]]
  Q.submit([&](handler &CGH) {
    nd_launch<class SGSizeKernel64>(CGH, NDR2, TestKernel_nd_item2{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel65(){{.*}} #[[SGSizeAttr6]]
  Q.submit([&](handler &CGH) {
    nd_launch<class SGSizeKernel65>(CGH, NDR3, TestKernel_nd_item3{});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel66{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    nd_launch<class SGSizeKernel66>(CGH, NDR1, TestKernel_nd_item1_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel67{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    nd_launch<class SGSizeKernel67>(CGH, NDR2, TestKernel_nd_item2_1{}, Redu1);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel68{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    nd_launch<class SGSizeKernel68>(CGH, NDR3, TestKernel_nd_item3_1{}, Redu1);
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel69{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    nd_launch<class SGSizeKernel69>(CGH, NDR1, TestKernel_nd_item1_2{}, Redu1,
                                    Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel70{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    nd_launch<class SGSizeKernel70>(CGH, NDR2, TestKernel_nd_item2_2{}, Redu1,
                                    Redu2);
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel71{{.*}}{{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    nd_launch<class SGSizeKernel71>(CGH, NDR3, TestKernel_nd_item3_2{}, Redu1,
                                    Redu2);
  });

  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel72(){{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for_work_group<class SGSizeKernel72>(R1,
                                                      TestKernel_work_group1{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel73(){{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for_work_group<class SGSizeKernel73>(R2,
                                                      TestKernel_work_group2{});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel74(){{.*}} #[[SGSizeAttr2]]
  Q.submit([&](handler &CGH) {
    CGH.parallel_for_work_group<class SGSizeKernel74>(R3,
                                                      TestKernel_work_group3{});
  });

  return 0;
}

// CHECK-IR: attributes #[[SGSizeAttr0]] = { {{.*}}"sycl-sub-group-size"="1"
// CHECK-IR: attributes #[[SGSizeAttr2]] = { {{.*}}"sycl-sub-group-size"="1"
// CHECK-IR: attributes #[[SGSizeAttr6]] = { {{.*}}"sycl-sub-group-size"="1"
