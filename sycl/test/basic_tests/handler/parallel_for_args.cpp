// This is a basic acceptance test which is intended to check that all valid
// combinations of arguments to parallel_for are accepted by the implementation
// Note: cases with 'auto' as an argument type are covered by
//   handler_generic_lambda_interface.cpp
//
// RUN: %clangxx -fsycl -fsyntax-only -ferror-limit=0 -Xclang -verify %s
// RUN: %if preview-breaking-changes-supported %{%clangxx -fsycl -fsyntax-only -Xclang -verify %s -I %sycl_include -fpreview-breaking-changes%}
//
// expected-no-diagnostics

#include <sycl/sycl.hpp>

template <int Dims> struct ConvertibleFromItem {
  ConvertibleFromItem(sycl::item<Dims>) {}
};

template <int Dims> struct ConvertibleFromNDItem {
  ConvertibleFromNDItem(sycl::nd_item<Dims>) {}
};

int main() {
  sycl::queue q;

  // 4.9.4.2.1. single_task invoke
  q.single_task([=]() {});
  q.single_task([=](sycl::kernel_handler kh) {});

  auto r1 = sycl::range<1>{4};
  auto r2 = sycl::range<2>{4, 4};
  auto r3 = sycl::range<3>{4, 4, 4};

  // 4.9.4.2.2. parallel_for invoke
  // parallel_for(range)
  // sycl::item
  q.parallel_for(r1, [=](sycl::item<1> it) {});
  q.parallel_for(r2, [=](sycl::item<2> it) {});
  q.parallel_for(r3, [=](sycl::item<3> it) {});

  q.parallel_for(r1, [=](sycl::item<1, false> it) {});
  q.parallel_for(r2, [=](sycl::item<2, false> it) {});
  q.parallel_for(r3, [=](sycl::item<3, false> it) {});

  // int, size_t -> sycl::item
  q.parallel_for(r1, [=](int it) {});
  q.parallel_for(r1, [=](size_t it) {});

  // sycl::item -> sycl::id
  q.parallel_for(r1, [=](sycl::id<1> it) {});
  q.parallel_for(r2, [=](sycl::id<2> it) {});
  q.parallel_for(r3, [=](sycl::id<3> it) {});

  // sycl::item -> user defined type convertible from item
  q.parallel_for(r1, [=](ConvertibleFromItem<1> it) {});
  q.parallel_for(r2, [=](ConvertibleFromItem<2> it) {});
  q.parallel_for(r3, [=](ConvertibleFromItem<3> it) {});

  // same as above, but with sycl::kernel_handler
  q.parallel_for(r1, [=](sycl::item<1> it, sycl::kernel_handler kh) {});
  q.parallel_for(r2, [=](sycl::item<2> it, sycl::kernel_handler kh) {});
  q.parallel_for(r3, [=](sycl::item<3> it, sycl::kernel_handler kh) {});

  q.parallel_for(r1, [=](int it, sycl::kernel_handler kh) {});
  q.parallel_for(r1, [=](size_t it, sycl::kernel_handler kh) {});

  q.parallel_for(r1, [=](sycl::item<1, false> it, sycl::kernel_handler kh) {});
  q.parallel_for(r2, [=](sycl::item<2, false> it, sycl::kernel_handler kh) {});
  q.parallel_for(r3, [=](sycl::item<3, false> it, sycl::kernel_handler kh) {});

  q.parallel_for(r1, [=](sycl::id<1> it, sycl::kernel_handler kh) {});
  q.parallel_for(r2, [=](sycl::id<2> it, sycl::kernel_handler kh) {});
  q.parallel_for(r3, [=](sycl::id<3> it, sycl::kernel_handler kh) {});

  q.parallel_for(r1,
                 [=](ConvertibleFromItem<1> it, sycl::kernel_handler kh) {});
  q.parallel_for(r2,
                 [=](ConvertibleFromItem<2> it, sycl::kernel_handler kh) {});
  q.parallel_for(r3,
                 [=](ConvertibleFromItem<3> it, sycl::kernel_handler kh) {});

  auto ndr1 = sycl::nd_range<1>(r1, r1);
  auto ndr2 = sycl::nd_range<2>(r2, r2);
  auto ndr3 = sycl::nd_range<3>(r3, r3);

  // parallel_for(nd_range)
  // sycl::nd_item
  q.parallel_for(ndr1, [=](sycl::nd_item<1> it) {});
  q.parallel_for(ndr2, [=](sycl::nd_item<2> it) {});
  q.parallel_for(ndr3, [=](sycl::nd_item<3> it) {});

  // sycl::nd_item -> user defined type convertible from nd_item
  q.parallel_for(ndr1, [=](ConvertibleFromNDItem<1> it) {});
  q.parallel_for(ndr2, [=](ConvertibleFromNDItem<2> it) {});
  q.parallel_for(ndr3, [=](ConvertibleFromNDItem<3> it) {});

  // same as above, but with sycl::kernel_handler
  q.parallel_for(ndr1, [=](sycl::nd_item<1> it, sycl::kernel_handler kh) {});
  q.parallel_for(ndr2, [=](sycl::nd_item<2> it, sycl::kernel_handler kh) {});
  q.parallel_for(ndr3, [=](sycl::nd_item<3> it, sycl::kernel_handler kh) {});

  q.parallel_for(ndr1,
                 [=](ConvertibleFromNDItem<1> it, sycl::kernel_handler kh) {});
  q.parallel_for(ndr2,
                 [=](ConvertibleFromNDItem<2> it, sycl::kernel_handler kh) {});
  q.parallel_for(ndr3,
                 [=](ConvertibleFromNDItem<3> it, sycl::kernel_handler kh) {});

  // TODO: consider adding test cases for hierarchical parallelism
}
