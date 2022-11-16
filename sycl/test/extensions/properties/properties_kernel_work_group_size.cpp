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
      sycl::ext::oneapi::experimental::work_group_size<1>};
  constexpr auto Props2 = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::work_group_size<1, 2>};
  constexpr auto Props3 = sycl::ext::oneapi::experimental::properties{
      sycl::ext::oneapi::experimental::work_group_size<1, 2, 3>};

  auto Redu1 = sycl::reduction<int>(nullptr, sycl::plus<int>());
  auto Redu2 = sycl::reduction<float>(nullptr, sycl::multiplies<float>());

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel0(){{.*}} #[[WGSizeAttr1:[0-9]+]]
  Q.single_task<class WGSizeKernel0>(Props1, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel1(){{.*}} #[[WGSizeAttr1]]
  Q.single_task<class WGSizeKernel1>(Ev, Props1, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel2(){{.*}} #[[WGSizeAttr1]]
  Q.single_task<class WGSizeKernel2>({Ev}, Props1, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel3(){{.*}} #[[WGSizeAttr2:[0-9]+]]
  Q.single_task<class WGSizeKernel3>(Props2, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel4(){{.*}} #[[WGSizeAttr2]]
  Q.single_task<class WGSizeKernel4>(Ev, Props2, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel5(){{.*}} #[[WGSizeAttr2]]
  Q.single_task<class WGSizeKernel5>({Ev}, Props2, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel6(){{.*}} #[[WGSizeAttr3:[0-9]+]]
  Q.single_task<class WGSizeKernel6>(Props3, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel7(){{.*}} #[[WGSizeAttr3]]
  Q.single_task<class WGSizeKernel7>(Ev, Props3, []() {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel8(){{.*}} #[[WGSizeAttr3]]
  Q.single_task<class WGSizeKernel8>({Ev}, Props3, []() {});

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel9(){{.*}} #[[WGSizeAttr4:[0-9]+]]
  Q.parallel_for<class WGSizeKernel9>(R1, Props1, [](sycl::id<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel10(){{.*}} #[[WGSizeAttr4]]
  Q.parallel_for<class WGSizeKernel10>(R1, Ev, Props1, [](sycl::id<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel11(){{.*}} #[[WGSizeAttr4]]
  Q.parallel_for<class WGSizeKernel11>(R1, {Ev}, Props1, [](sycl::id<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel12(){{.*}} #[[WGSizeAttr5:[0-9]+]]
  Q.parallel_for<class WGSizeKernel12>(R2, Props2, [](sycl::id<2>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel13(){{.*}} #[[WGSizeAttr5]]
  Q.parallel_for<class WGSizeKernel13>(R2, Ev, Props2, [](sycl::id<2>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel14(){{.*}} #[[WGSizeAttr5]]
  Q.parallel_for<class WGSizeKernel14>(R2, {Ev}, Props2, [](sycl::id<2>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel15(){{.*}} #[[WGSizeAttr6:[0-9]+]]
  Q.parallel_for<class WGSizeKernel15>(R3, Props3, [](sycl::id<3>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel16(){{.*}} #[[WGSizeAttr6]]
  Q.parallel_for<class WGSizeKernel16>(R3, Ev, Props3, [](sycl::id<3>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel17(){{.*}} #[[WGSizeAttr6]]
  Q.parallel_for<class WGSizeKernel17>(R3, {Ev}, Props3, [](sycl::id<3>) {});

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel18{{.*}}{{.*}} #[[WGSizeAttr7:[0-9]+]]
  Q.parallel_for<class WGSizeKernel18>(R1, Props1, Redu1,
                                       [](sycl::id<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel19{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.parallel_for<class WGSizeKernel19>(R1, Ev, Props1, Redu1,
                                       [](sycl::id<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel20{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.parallel_for<class WGSizeKernel20>(R1, {Ev}, Props1, Redu1,
                                       [](sycl::id<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel21{{.*}}{{.*}} #[[WGSizeAttr8:[0-9]+]]
  Q.parallel_for<class WGSizeKernel21>(R2, Props2, Redu1,
                                       [](sycl::id<2>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel22{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.parallel_for<class WGSizeKernel22>(R2, Ev, Props2, Redu1,
                                       [](sycl::id<2>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel23{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.parallel_for<class WGSizeKernel23>(R2, {Ev}, Props2, Redu1,
                                       [](sycl::id<2>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel24{{.*}}{{.*}} #[[WGSizeAttr9:[0-9]+]]
  Q.parallel_for<class WGSizeKernel24>(R3, Props3, Redu1,
                                       [](sycl::id<3>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel25{{.*}}{{.*}} #[[WGSizeAttr9]]
  Q.parallel_for<class WGSizeKernel25>(R3, Ev, Props3, Redu1,
                                       [](sycl::id<3>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel26{{.*}}{{.*}} #[[WGSizeAttr9]]
  Q.parallel_for<class WGSizeKernel26>(R3, {Ev}, Props3, Redu1,
                                       [](sycl::id<3>, auto &) {});

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel27(){{.*}} #[[WGSizeAttr4]]
  Q.parallel_for<class WGSizeKernel27>(NDR1, Props1, [](sycl::nd_item<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel28(){{.*}} #[[WGSizeAttr4]]
  Q.parallel_for<class WGSizeKernel28>(NDR1, Ev, Props1,
                                       [](sycl::nd_item<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel29(){{.*}} #[[WGSizeAttr4]]
  Q.parallel_for<class WGSizeKernel29>(NDR1, {Ev}, Props1,
                                       [](sycl::nd_item<1>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel30(){{.*}} #[[WGSizeAttr5]]
  Q.parallel_for<class WGSizeKernel30>(NDR2, Props2, [](sycl::nd_item<2>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel31(){{.*}} #[[WGSizeAttr5]]
  Q.parallel_for<class WGSizeKernel31>(NDR2, Ev, Props2,
                                       [](sycl::nd_item<2>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel32(){{.*}} #[[WGSizeAttr5]]
  Q.parallel_for<class WGSizeKernel32>(NDR2, {Ev}, Props2,
                                       [](sycl::nd_item<2>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel33(){{.*}} #[[WGSizeAttr6]]
  Q.parallel_for<class WGSizeKernel33>(NDR3, Props3, [](sycl::nd_item<3>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel34(){{.*}} #[[WGSizeAttr6]]
  Q.parallel_for<class WGSizeKernel34>(NDR3, Ev, Props3,
                                       [](sycl::nd_item<3>) {});
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel35(){{.*}} #[[WGSizeAttr6]]
  Q.parallel_for<class WGSizeKernel35>(NDR3, {Ev}, Props3,
                                       [](sycl::nd_item<3>) {});

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel36{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.parallel_for<class WGSizeKernel36>(NDR1, Props1, Redu1,
                                       [](sycl::nd_item<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel37{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.parallel_for<class WGSizeKernel37>(NDR1, Ev, Props1, Redu1,
                                       [](sycl::nd_item<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel38{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.parallel_for<class WGSizeKernel38>(NDR1, {Ev}, Props1, Redu1,
                                       [](sycl::nd_item<1>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel39{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.parallel_for<class WGSizeKernel39>(NDR2, Props2, Redu1,
                                       [](sycl::nd_item<2>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel40{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.parallel_for<class WGSizeKernel40>(NDR2, Ev, Props2, Redu1,
                                       [](sycl::nd_item<2>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel41{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.parallel_for<class WGSizeKernel41>(NDR2, {Ev}, Props2, Redu1,
                                       [](sycl::nd_item<2>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel42{{.*}}{{.*}} #[[WGSizeAttr9]]
  Q.parallel_for<class WGSizeKernel42>(NDR3, Props3, Redu1,
                                       [](sycl::nd_item<3>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel43{{.*}}{{.*}} #[[WGSizeAttr9]]
  Q.parallel_for<class WGSizeKernel43>(NDR3, Ev, Props3, Redu1,
                                       [](sycl::nd_item<3>, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel44{{.*}}{{.*}} #[[WGSizeAttr9]]
  Q.parallel_for<class WGSizeKernel44>(NDR3, {Ev}, Props3, Redu1,
                                       [](sycl::nd_item<3>, auto &) {});

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel45{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.parallel_for<class WGSizeKernel45>(NDR1, Props1, Redu1, Redu2,
                                       [](sycl::nd_item<1>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel46{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.parallel_for<class WGSizeKernel46>(NDR1, Ev, Props1, Redu1, Redu2,
                                       [](sycl::nd_item<1>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel47{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.parallel_for<class WGSizeKernel47>(NDR1, {Ev}, Props1, Redu1, Redu2,
                                       [](sycl::nd_item<1>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel48{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.parallel_for<class WGSizeKernel48>(NDR2, Props2, Redu1, Redu2,
                                       [](sycl::nd_item<2>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel49{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.parallel_for<class WGSizeKernel49>(NDR2, Ev, Props2, Redu1, Redu2,
                                       [](sycl::nd_item<2>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel50{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.parallel_for<class WGSizeKernel50>(NDR2, {Ev}, Props2, Redu1, Redu2,
                                       [](sycl::nd_item<2>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel51{{.*}}{{.*}} #[[WGSizeAttr9]]
  Q.parallel_for<class WGSizeKernel51>(NDR3, Props3, Redu1, Redu2,
                                       [](sycl::nd_item<3>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel52{{.*}}{{.*}} #[[WGSizeAttr9]]
  Q.parallel_for<class WGSizeKernel52>(NDR3, Ev, Props3, Redu1, Redu2,
                                       [](sycl::nd_item<3>, auto &, auto &) {});
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel53{{.*}}{{.*}} #[[WGSizeAttr9]]
  Q.parallel_for<class WGSizeKernel53>(NDR3, {Ev}, Props3, Redu1, Redu2,
                                       [](sycl::nd_item<3>, auto &, auto &) {});

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel54(){{.*}} #[[WGSizeAttr1]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task<class WGSizeKernel54>(Props1, []() {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel55(){{.*}} #[[WGSizeAttr2]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task<class WGSizeKernel55>(Props2, []() {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel56(){{.*}} #[[WGSizeAttr3]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.single_task<class WGSizeKernel56>(Props3, []() {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel57(){{.*}} #[[WGSizeAttr4]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeKernel57>(R1, Props1, [](sycl::id<1>) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel58(){{.*}} #[[WGSizeAttr5]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeKernel58>(R2, Props2, [](sycl::id<2>) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel59(){{.*}} #[[WGSizeAttr6]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeKernel59>(R3, Props3, [](sycl::id<3>) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel60{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeKernel60>(R1, Props1, Redu1,
                                           [](sycl::id<1>, auto &) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel61{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeKernel61>(R2, Props2, Redu1,
                                           [](sycl::id<2>, auto &) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel62{{.*}}{{.*}} #[[WGSizeAttr9]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeKernel62>(R3, Props3, Redu1,
                                           [](sycl::id<3>, auto &) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel63(){{.*}} #[[WGSizeAttr4]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeKernel63>(NDR1, Props1,
                                           [](sycl::nd_item<1>) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel64(){{.*}} #[[WGSizeAttr5]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeKernel64>(NDR2, Props2,
                                           [](sycl::nd_item<2>) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel65(){{.*}} #[[WGSizeAttr6]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeKernel65>(NDR3, Props3,
                                           [](sycl::nd_item<3>) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel66{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeKernel66>(NDR1, Props1, Redu1,
                                           [](sycl::nd_item<1>, auto &) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel67{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeKernel67>(NDR2, Props2, Redu1,
                                           [](sycl::nd_item<2>, auto &) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel68{{.*}}{{.*}} #[[WGSizeAttr9]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeKernel68>(NDR3, Props3, Redu1,
                                           [](sycl::nd_item<3>, auto &) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel69{{.*}}{{.*}} #[[WGSizeAttr7]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeKernel69>(
        NDR1, Props1, Redu1, Redu2, [](sycl::nd_item<1>, auto &, auto &) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel70{{.*}}{{.*}} #[[WGSizeAttr8]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeKernel70>(
        NDR2, Props2, Redu1, Redu2, [](sycl::nd_item<2>, auto &, auto &) {});
  });
  // CHECK-IR: spir_kernel void @{{.*}}MainKrn{{.*}}WGSizeKernel71{{.*}}{{.*}} #[[WGSizeAttr9]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class WGSizeKernel71>(
        NDR3, Props3, Redu1, Redu2, [](sycl::nd_item<3>, auto &, auto &) {});
  });

  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel72(){{.*}} #[[WGSizeAttr7]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for_work_group<class WGSizeKernel72>(
        R1, Props1, [](sycl::group<1> G) {
          G.parallel_for_work_item([&](sycl::h_item<1>) {});
        });
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel73(){{.*}} #[[WGSizeAttr8]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for_work_group<class WGSizeKernel73>(
        R2, Props2, [](sycl::group<2> G) {
          G.parallel_for_work_item([&](sycl::h_item<2>) {});
        });
  });
  // CHECK-IR: spir_kernel void @{{.*}}WGSizeKernel74(){{.*}} #[[WGSizeAttr9]]
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for_work_group<class WGSizeKernel74>(
        R3, Props3, [](sycl::group<3> G) {
          G.parallel_for_work_item([&](sycl::h_item<3>) {});
        });
  });

  return 0;
}

// CHECK-IR: attributes #[[WGSizeAttr1]] = { {{.*}}"sycl-work-group-size"="1"
// CHECK-IR: attributes #[[WGSizeAttr2]] = { {{.*}}"sycl-work-group-size"="1,2"
// CHECK-IR: attributes #[[WGSizeAttr3]] = { {{.*}}"sycl-work-group-size"="1,2,3"
// CHECK-IR: attributes #[[WGSizeAttr4]] = { {{.*}}"sycl-work-group-size"="1"
// CHECK-IR: attributes #[[WGSizeAttr5]] = { {{.*}}"sycl-work-group-size"="1,2"
// CHECK-IR: attributes #[[WGSizeAttr6]] = { {{.*}}"sycl-work-group-size"="1,2,3"
// CHECK-IR: attributes #[[WGSizeAttr7]] = { {{.*}}"sycl-work-group-size"="1"
// CHECK-IR: attributes #[[WGSizeAttr8]] = { {{.*}}"sycl-work-group-size"="1,2"
// CHECK-IR: attributes #[[WGSizeAttr9]] = { {{.*}}"sycl-work-group-size"="1,2,3"
