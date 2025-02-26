// RUN: %clangxx -fsycl-device-only -S -Xclang -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK-IR
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  sycl::event Ev;

  sycl::range<1> R1{1};
  sycl::range<2> R2{1, 2};
  sycl::range<3> R3{1, 2, 3};

  sycl::nd_range<1> NDR1{R1, R1};
  sycl::nd_range<2> NDR2{R2, R2};
  sycl::nd_range<3> NDR3{R3, R3};

  constexpr auto Props = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::sub_group_size<1>};

  auto Redu1 = sycl::reduction<int>(nullptr, sycl::plus<int>());
  auto Redu2 = sycl::reduction<float>(nullptr, sycl::multiplies<float>());

  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel0(){{.*}} #[[SGSizeAttr1:[0-9]+]]
  Q.single_task<class SGSizeKernel0>(Props, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel1(){{.*}} #[[SGSizeAttr1]]
  Q.single_task<class SGSizeKernel1>(Ev, Props, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel2(){{.*}} #[[SGSizeAttr1]]
  Q.single_task<class SGSizeKernel2>({Ev}, Props, []() {});

  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel9(){{.*}} #[[SGSizeAttr2:[0-9]+]]
  Q.parallel_for<class SGSizeKernel9>(R1, Props, [](sycl::id<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel10(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel10>(R1, Ev, Props, [](sycl::id<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel11(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel11>(R1, {Ev}, Props, [](sycl::id<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel12(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel12>(R2, Props, [](sycl::id<2>) {});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel13(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel13>(R2, Ev, Props, [](sycl::id<2>) {});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel14(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel14>(R2, {Ev}, Props, [](sycl::id<2>) {});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel15(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel15>(R3, Props, [](sycl::id<3>) {});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel16(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel16>(R3, Ev, Props, [](sycl::id<3>) {});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel17(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel17>(R3, {Ev}, Props, [](sycl::id<3>) {});

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel18{{.*}}{{.*}} #[[SGSizeAttr3:[0-9]+]]
  Q.parallel_for<class SGSizeKernel18>(R1, Props, Redu1,
                                       [](sycl::id<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel19{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel19>(R1, Ev, Props, Redu1,
                                       [](sycl::id<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel20{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel20>(R1, {Ev}, Props, Redu1,
                                       [](sycl::id<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel21{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel21>(R2, Props, Redu1,
                                       [](sycl::id<2>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel22{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel22>(R2, Ev, Props, Redu1,
                                       [](sycl::id<2>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel23{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel23>(R2, {Ev}, Props, Redu1,
                                       [](sycl::id<2>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel24{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel24>(R3, Props, Redu1,
                                       [](sycl::id<3>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel25{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel25>(R3, Ev, Props, Redu1,
                                       [](sycl::id<3>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel26{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel26>(R3, {Ev}, Props, Redu1,
                                       [](sycl::id<3>, auto &) {});

  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel27(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel27>(NDR1, Props, [](sycl::nd_item<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel28(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel28>(NDR1, Ev, Props,
                                       [](sycl::nd_item<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel29(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel29>(NDR1, {Ev}, Props,
                                       [](sycl::nd_item<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel30(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel30>(NDR2, Props, [](sycl::nd_item<2>) {});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel31(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel31>(NDR2, Ev, Props,
                                       [](sycl::nd_item<2>) {});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel32(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel32>(NDR2, {Ev}, Props,
                                       [](sycl::nd_item<2>) {});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel33(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel33>(NDR3, Props, [](sycl::nd_item<3>) {});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel34(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel34>(NDR3, Ev, Props,
                                       [](sycl::nd_item<3>) {});
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel35(){{.*}} #[[SGSizeAttr2]]
  Q.parallel_for<class SGSizeKernel35>(NDR3, {Ev}, Props,
                                       [](sycl::nd_item<3>) {});

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel36{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel36>(NDR1, Props, Redu1,
                                       [](sycl::nd_item<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel37{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel37>(NDR1, Ev, Props, Redu1,
                                       [](sycl::nd_item<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel38{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel38>(NDR1, {Ev}, Props, Redu1,
                                       [](sycl::nd_item<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel39{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel39>(NDR2, Props, Redu1,
                                       [](sycl::nd_item<2>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel40{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel40>(NDR2, Ev, Props, Redu1,
                                       [](sycl::nd_item<2>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel41{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel41>(NDR2, {Ev}, Props, Redu1,
                                       [](sycl::nd_item<2>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel42{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel42>(NDR3, Props, Redu1,
                                       [](sycl::nd_item<3>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel43{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel43>(NDR3, Ev, Props, Redu1,
                                       [](sycl::nd_item<3>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel44{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel44>(NDR3, {Ev}, Props, Redu1,
                                       [](sycl::nd_item<3>, auto &) {});

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel45{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel45>(NDR1, Props, Redu1, Redu2,
                                       [](sycl::nd_item<1>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel46{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel46>(NDR1, Ev, Props, Redu1, Redu2,
                                       [](sycl::nd_item<1>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel47{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel47>(NDR1, {Ev}, Props, Redu1, Redu2,
                                       [](sycl::nd_item<1>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel48{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel48>(NDR2, Props, Redu1, Redu2,
                                       [](sycl::nd_item<2>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel49{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel49>(NDR2, Ev, Props, Redu1, Redu2,
                                       [](sycl::nd_item<2>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel50{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel50>(NDR2, {Ev}, Props, Redu1, Redu2,
                                       [](sycl::nd_item<2>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel51{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel51>(NDR3, Props, Redu1, Redu2,
                                       [](sycl::nd_item<3>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel52{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel52>(NDR3, Ev, Props, Redu1, Redu2,
                                       [](sycl::nd_item<3>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel53{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.parallel_for<class SGSizeKernel53>(NDR3, {Ev}, Props, Redu1, Redu2,
                                       [](sycl::nd_item<3>, auto &, auto &) {});

  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel54(){{.*}} #[[SGSizeAttr1]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task<class SGSizeKernel54>(Props, []() {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel55(){{.*}} #[[SGSizeAttr1]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task<class SGSizeKernel55>(Props, []() {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel56(){{.*}} #[[SGSizeAttr1]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task<class SGSizeKernel56>(Props, []() {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel57(){{.*}} #[[SGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class SGSizeKernel57>(R1, Props, [](sycl::id<1>) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel58(){{.*}} #[[SGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class SGSizeKernel58>(R2, Props, [](sycl::id<2>) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel59(){{.*}} #[[SGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class SGSizeKernel59>(R3, Props, [](sycl::id<3>) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel60{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class SGSizeKernel60>(R1, Props, Redu1,
                                           [](sycl::id<1>, auto &) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel61{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class SGSizeKernel61>(R2, Props, Redu1,
                                           [](sycl::id<2>, auto &) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel62{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class SGSizeKernel62>(R3, Props, Redu1,
                                           [](sycl::id<3>, auto &) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel63(){{.*}} #[[SGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class SGSizeKernel63>(NDR1, Props,
                                           [](sycl::nd_item<1>) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel64(){{.*}} #[[SGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class SGSizeKernel64>(NDR2, Props,
                                           [](sycl::nd_item<2>) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel65(){{.*}} #[[SGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class SGSizeKernel65>(NDR3, Props,
                                           [](sycl::nd_item<3>) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel66{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class SGSizeKernel66>(NDR1, Props, Redu1,
                                           [](sycl::nd_item<1>, auto &) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel67{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class SGSizeKernel67>(NDR2, Props, Redu1,
                                           [](sycl::nd_item<2>, auto &) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel68{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class SGSizeKernel68>(NDR3, Props, Redu1,
                                           [](sycl::nd_item<3>, auto &) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel69{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class SGSizeKernel69>(
        NDR1, Props, Redu1, Redu2, [](sycl::nd_item<1>, auto &, auto &) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel70{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class SGSizeKernel70>(
        NDR2, Props, Redu1, Redu2, [](sycl::nd_item<2>, auto &, auto &) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}SGSizeKernel71{{.*}}{{.*}} #[[SGSizeAttr3]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class SGSizeKernel71>(
        NDR3, Props, Redu1, Redu2, [](sycl::nd_item<3>, auto &, auto &) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel72(){{.*}} #[[SGSizeAttr3]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for_work_group<class SGSizeKernel72>(
        R1, Props, [](sycl::group<1> G) {
          G.parallel_for_work_item([&](sycl::h_item<1>) {});
        });
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel73(){{.*}} #[[SGSizeAttr3]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for_work_group<class SGSizeKernel73>(
        R2, Props, [](sycl::group<2> G) {
          G.parallel_for_work_item([&](sycl::h_item<2>) {});
        });
  });
  // CHECK-IR: spir_kernel void @{{.*}}SGSizeKernel74(){{.*}} #[[SGSizeAttr3]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for_work_group<class SGSizeKernel74>(
        R3, Props, [](sycl::group<3> G) {
          G.parallel_for_work_item([&](sycl::h_item<3>) {});
        });
  });

  return 0;
}

// CHECK-IR: attributes #[[SGSizeAttr1]] = { {{.*}}"sycl-sub-group-size"="1"
// CHECK-IR: attributes #[[SGSizeAttr2]] = { {{.*}}"sycl-sub-group-size"="1"
// CHECK-IR: attributes #[[SGSizeAttr3]] = { {{.*}}"sycl-sub-group-size"="1"
