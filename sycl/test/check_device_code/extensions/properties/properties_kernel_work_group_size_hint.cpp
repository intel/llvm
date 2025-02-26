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

  constexpr auto Props1 = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::work_group_size_hint<1>};
  constexpr auto Props2 = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::work_group_size_hint<1, 2>};
  constexpr auto Props3 = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::work_group_size_hint<1, 2, 3>};

  auto Redu1 = sycl::reduction<int>(nullptr, sycl::plus<int>());
  auto Redu2 = sycl::reduction<float>(nullptr, sycl::multiplies<float>());

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel0(){{.*}} #[[WGSizeHintAttr1:[0-9]+]]
  Q.single_task<class WGSizeHintKernel0>(Props1, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel1(){{.*}} #[[WGSizeHintAttr1]]
  Q.single_task<class WGSizeHintKernel1>(Ev, Props1, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel2(){{.*}} #[[WGSizeHintAttr1]]
  Q.single_task<class WGSizeHintKernel2>({Ev}, Props1, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel3(){{.*}} #[[WGSizeHintAttr2:[0-9]+]]
  Q.single_task<class WGSizeHintKernel3>(Props2, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel4(){{.*}} #[[WGSizeHintAttr2]]
  Q.single_task<class WGSizeHintKernel4>(Ev, Props2, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel5(){{.*}} #[[WGSizeHintAttr2]]
  Q.single_task<class WGSizeHintKernel5>({Ev}, Props2, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel6(){{.*}} #[[WGSizeHintAttr3:[0-9]+]]
  Q.single_task<class WGSizeHintKernel6>(Props3, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel7(){{.*}} #[[WGSizeHintAttr3]]
  Q.single_task<class WGSizeHintKernel7>(Ev, Props3, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel8(){{.*}} #[[WGSizeHintAttr3]]
  Q.single_task<class WGSizeHintKernel8>({Ev}, Props3, []() {});

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel9(){{.*}} #[[WGSizeHintAttr4:[0-9]+]]
  Q.parallel_for<class WGSizeHintKernel9>(R1, Props1, [](sycl::id<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel10(){{.*}} #[[WGSizeHintAttr4]]
  Q.parallel_for<class WGSizeHintKernel10>(R1, Ev, Props1, [](sycl::id<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel11(){{.*}} #[[WGSizeHintAttr4]]
  Q.parallel_for<class WGSizeHintKernel11>(R1, {Ev}, Props1,
                                           [](sycl::id<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel12(){{.*}} #[[WGSizeHintAttr5:[0-9]+]]
  Q.parallel_for<class WGSizeHintKernel12>(R2, Props2, [](sycl::id<2>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel13(){{.*}} #[[WGSizeHintAttr5]]
  Q.parallel_for<class WGSizeHintKernel13>(R2, Ev, Props2, [](sycl::id<2>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel14(){{.*}} #[[WGSizeHintAttr5]]
  Q.parallel_for<class WGSizeHintKernel14>(R2, {Ev}, Props2,
                                           [](sycl::id<2>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel15(){{.*}} #[[WGSizeHintAttr6:[0-9]+]]
  Q.parallel_for<class WGSizeHintKernel15>(R3, Props3, [](sycl::id<3>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel16(){{.*}} #[[WGSizeHintAttr6]]
  Q.parallel_for<class WGSizeHintKernel16>(R3, Ev, Props3, [](sycl::id<3>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel17(){{.*}} #[[WGSizeHintAttr6]]
  Q.parallel_for<class WGSizeHintKernel17>(R3, {Ev}, Props3,
                                           [](sycl::id<3>) {});

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel18{{.*}}{{.*}} #[[WGSizeHintAttr7:[0-9]+]]
  Q.parallel_for<class WGSizeHintKernel18>(R1, Props1, Redu1,
                                           [](sycl::id<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel19{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.parallel_for<class WGSizeHintKernel19>(R1, Ev, Props1, Redu1,
                                           [](sycl::id<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel20{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.parallel_for<class WGSizeHintKernel20>(R1, {Ev}, Props1, Redu1,
                                           [](sycl::id<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel21{{.*}}{{.*}} #[[WGSizeHintAttr8:[0-9]+]]
  Q.parallel_for<class WGSizeHintKernel21>(R2, Props2, Redu1,
                                           [](sycl::id<2>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel22{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.parallel_for<class WGSizeHintKernel22>(R2, Ev, Props2, Redu1,
                                           [](sycl::id<2>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel23{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.parallel_for<class WGSizeHintKernel23>(R2, {Ev}, Props2, Redu1,
                                           [](sycl::id<2>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel24{{.*}}{{.*}} #[[WGSizeHintAttr9:[0-9]+]]
  Q.parallel_for<class WGSizeHintKernel24>(R3, Props3, Redu1,
                                           [](sycl::id<3>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel25{{.*}}{{.*}} #[[WGSizeHintAttr9]]
  Q.parallel_for<class WGSizeHintKernel25>(R3, Ev, Props3, Redu1,
                                           [](sycl::id<3>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel26{{.*}}{{.*}} #[[WGSizeHintAttr9]]
  Q.parallel_for<class WGSizeHintKernel26>(R3, {Ev}, Props3, Redu1,
                                           [](sycl::id<3>, auto &) {});

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel27(){{.*}} #[[WGSizeHintAttr4]]
  Q.parallel_for<class WGSizeHintKernel27>(NDR1, Props1,
                                           [](sycl::nd_item<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel28(){{.*}} #[[WGSizeHintAttr4]]
  Q.parallel_for<class WGSizeHintKernel28>(NDR1, Ev, Props1,
                                           [](sycl::nd_item<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel29(){{.*}} #[[WGSizeHintAttr4]]
  Q.parallel_for<class WGSizeHintKernel29>(NDR1, {Ev}, Props1,
                                           [](sycl::nd_item<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel30(){{.*}} #[[WGSizeHintAttr5]]
  Q.parallel_for<class WGSizeHintKernel30>(NDR2, Props2,
                                           [](sycl::nd_item<2>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel31(){{.*}} #[[WGSizeHintAttr5]]
  Q.parallel_for<class WGSizeHintKernel31>(NDR2, Ev, Props2,
                                           [](sycl::nd_item<2>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel32(){{.*}} #[[WGSizeHintAttr5]]
  Q.parallel_for<class WGSizeHintKernel32>(NDR2, {Ev}, Props2,
                                           [](sycl::nd_item<2>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel33(){{.*}} #[[WGSizeHintAttr6]]
  Q.parallel_for<class WGSizeHintKernel33>(NDR3, Props3,
                                           [](sycl::nd_item<3>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel34(){{.*}} #[[WGSizeHintAttr6]]
  Q.parallel_for<class WGSizeHintKernel34>(NDR3, Ev, Props3,
                                           [](sycl::nd_item<3>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel35(){{.*}} #[[WGSizeHintAttr6]]
  Q.parallel_for<class WGSizeHintKernel35>(NDR3, {Ev}, Props3,
                                           [](sycl::nd_item<3>) {});

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel36{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.parallel_for<class WGSizeHintKernel36>(NDR1, Props1, Redu1,
                                           [](sycl::nd_item<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel37{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.parallel_for<class WGSizeHintKernel37>(NDR1, Ev, Props1, Redu1,
                                           [](sycl::nd_item<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel38{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.parallel_for<class WGSizeHintKernel38>(NDR1, {Ev}, Props1, Redu1,
                                           [](sycl::nd_item<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel39{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.parallel_for<class WGSizeHintKernel39>(NDR2, Props2, Redu1,
                                           [](sycl::nd_item<2>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel40{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.parallel_for<class WGSizeHintKernel40>(NDR2, Ev, Props2, Redu1,
                                           [](sycl::nd_item<2>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel41{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.parallel_for<class WGSizeHintKernel41>(NDR2, {Ev}, Props2, Redu1,
                                           [](sycl::nd_item<2>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel42{{.*}}{{.*}} #[[WGSizeHintAttr9]]
  Q.parallel_for<class WGSizeHintKernel42>(NDR3, Props3, Redu1,
                                           [](sycl::nd_item<3>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel43{{.*}}{{.*}} #[[WGSizeHintAttr9]]
  Q.parallel_for<class WGSizeHintKernel43>(NDR3, Ev, Props3, Redu1,
                                           [](sycl::nd_item<3>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel44{{.*}}{{.*}} #[[WGSizeHintAttr9]]
  Q.parallel_for<class WGSizeHintKernel44>(NDR3, {Ev}, Props3, Redu1,
                                           [](sycl::nd_item<3>, auto &) {});

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel45{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.parallel_for<class WGSizeHintKernel45>(
      NDR1, Props1, Redu1, Redu2, [](sycl::nd_item<1>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel46{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.parallel_for<class WGSizeHintKernel46>(
      NDR1, Ev, Props1, Redu1, Redu2, [](sycl::nd_item<1>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel47{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.parallel_for<class WGSizeHintKernel47>(
      NDR1, {Ev}, Props1, Redu1, Redu2,
      [](sycl::nd_item<1>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel48{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.parallel_for<class WGSizeHintKernel48>(
      NDR2, Props2, Redu1, Redu2, [](sycl::nd_item<2>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel49{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.parallel_for<class WGSizeHintKernel49>(
      NDR2, Ev, Props2, Redu1, Redu2, [](sycl::nd_item<2>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel50{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.parallel_for<class WGSizeHintKernel50>(
      NDR2, {Ev}, Props2, Redu1, Redu2,
      [](sycl::nd_item<2>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel51{{.*}}{{.*}} #[[WGSizeHintAttr9]]
  Q.parallel_for<class WGSizeHintKernel51>(
      NDR3, Props3, Redu1, Redu2, [](sycl::nd_item<3>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel52{{.*}}{{.*}} #[[WGSizeHintAttr9]]
  Q.parallel_for<class WGSizeHintKernel52>(
      NDR3, Ev, Props3, Redu1, Redu2, [](sycl::nd_item<3>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel53{{.*}}{{.*}} #[[WGSizeHintAttr9]]
  Q.parallel_for<class WGSizeHintKernel53>(
      NDR3, {Ev}, Props3, Redu1, Redu2,
      [](sycl::nd_item<3>, auto &, auto &) {});

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel54(){{.*}} #[[WGSizeHintAttr1]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task<class WGSizeHintKernel54>(Props1, []() {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel55(){{.*}} #[[WGSizeHintAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task<class WGSizeHintKernel55>(Props2, []() {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel56(){{.*}} #[[WGSizeHintAttr3]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task<class WGSizeHintKernel56>(Props3, []() {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel57(){{.*}} #[[WGSizeHintAttr4]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeHintKernel57>(R1, Props1, [](sycl::id<1>) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel58(){{.*}} #[[WGSizeHintAttr5]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeHintKernel58>(R2, Props2, [](sycl::id<2>) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel59(){{.*}} #[[WGSizeHintAttr6]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeHintKernel59>(R3, Props3, [](sycl::id<3>) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel60{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeHintKernel60>(R1, Props1, Redu1,
                                               [](sycl::id<1>, auto &) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel61{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeHintKernel61>(R2, Props2, Redu1,
                                               [](sycl::id<2>, auto &) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel62{{.*}}{{.*}} #[[WGSizeHintAttr9]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeHintKernel62>(R3, Props3, Redu1,
                                               [](sycl::id<3>, auto &) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel63(){{.*}} #[[WGSizeHintAttr4]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeHintKernel63>(NDR1, Props1,
                                               [](sycl::nd_item<1>) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel64(){{.*}} #[[WGSizeHintAttr5]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeHintKernel64>(NDR2, Props2,
                                               [](sycl::nd_item<2>) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel65(){{.*}} #[[WGSizeHintAttr6]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeHintKernel65>(NDR3, Props3,
                                               [](sycl::nd_item<3>) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel66{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeHintKernel66>(NDR1, Props1, Redu1,
                                               [](sycl::nd_item<1>, auto &) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel67{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeHintKernel67>(NDR2, Props2, Redu1,
                                               [](sycl::nd_item<2>, auto &) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel68{{.*}}{{.*}} #[[WGSizeHintAttr9]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeHintKernel68>(NDR3, Props3, Redu1,
                                               [](sycl::nd_item<3>, auto &) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel69{{.*}}{{.*}} #[[WGSizeHintAttr7]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeHintKernel69>(
        NDR1, Props1, Redu1, Redu2, [](sycl::nd_item<1>, auto &, auto &) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel70{{.*}}{{.*}} #[[WGSizeHintAttr8]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeHintKernel70>(
        NDR2, Props2, Redu1, Redu2, [](sycl::nd_item<2>, auto &, auto &) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeHintKernel71{{.*}}{{.*}} #[[WGSizeHintAttr9]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeHintKernel71>(
        NDR3, Props3, Redu1, Redu2, [](sycl::nd_item<3>, auto &, auto &) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel72(){{.*}} #[[WGSizeHintAttr7]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for_work_group<class WGSizeHintKernel72>(
        R1, Props1, [](sycl::group<1> G) {
          G.parallel_for_work_item([&](sycl::h_item<1>) {});
        });
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel73(){{.*}} #[[WGSizeHintAttr8]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for_work_group<class WGSizeHintKernel73>(
        R2, Props2, [](sycl::group<2> G) {
          G.parallel_for_work_item([&](sycl::h_item<2>) {});
        });
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeHintKernel74(){{.*}} #[[WGSizeHintAttr9]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for_work_group<class WGSizeHintKernel74>(
        R3, Props3, [](sycl::group<3> G) {
          G.parallel_for_work_item([&](sycl::h_item<3>) {});
        });
  });

  return 0;
}

// CHECK-IR: attributes #[[WGSizeHintAttr1]] = { {{.*}}"sycl-work-group-size-hint"="1"
// CHECK-IR: attributes #[[WGSizeHintAttr2]] = { {{.*}}"sycl-work-group-size-hint"="1,2"
// CHECK-IR: attributes #[[WGSizeHintAttr3]] = { {{.*}}"sycl-work-group-size-hint"="1,2,3"
// CHECK-IR: attributes #[[WGSizeHintAttr4]] = { {{.*}}"sycl-work-group-size-hint"="1"
// CHECK-IR: attributes #[[WGSizeHintAttr5]] = { {{.*}}"sycl-work-group-size-hint"="1,2"
// CHECK-IR: attributes #[[WGSizeHintAttr6]] = { {{.*}}"sycl-work-group-size-hint"="1,2,3"
// CHECK-IR: attributes #[[WGSizeHintAttr7]] = { {{.*}}"sycl-work-group-size-hint"="1"
// CHECK-IR: attributes #[[WGSizeHintAttr8]] = { {{.*}}"sycl-work-group-size-hint"="1,2"
// CHECK-IR: attributes #[[WGSizeHintAttr9]] = { {{.*}}"sycl-work-group-size-hint"="1,2,3"
