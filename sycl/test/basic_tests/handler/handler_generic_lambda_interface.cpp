// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s -I %sycl_include
// RUN: %if preview-breaking-changes-supported %{%clangxx -fsycl -fsyntax-only -Xclang -verify %s -I %sycl_include -fpreview-breaking-changes -DPREVIEW_BREAKING_CHANGES%}
// expected-no-diagnostics
//==--------------- handler_generic_lambda_interface.cpp -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/sycl.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

template <typename KernelName, typename ExpectedType, typename Range>
void test_parallel_for(Range r) {
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<KernelName>(r, [=](auto item) {
      static_assert(std::is_same<decltype(item), ExpectedType>::value,
                    "Argument type is unexpected");
    });
  });
}

template <typename KernelName, typename ExpectedType, typename Range>
void test_parallel_for_work_group(Range r) {
  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for_work_group<KernelName>(r, [=](auto item) {
      static_assert(std::is_same<decltype(item), ExpectedType>::value,
                    "Argument type is unexpected");
    });
  });
}

int main() {

  test_parallel_for<class Item1Name, sycl::item<1>>(sycl::range<1>{1});
  test_parallel_for<class Item2Name, sycl::item<2>>(sycl::range<2>{1, 1});
  test_parallel_for<class Item3Name, sycl::item<3>>(sycl::range<3>{1, 1, 1});

  test_parallel_for<class NdItem1Name, sycl::nd_item<1>>(sycl::nd_range<1>{
      sycl::range<1>{1}, sycl::range<1>{1}});
  test_parallel_for<class NdItem2Name, sycl::nd_item<2>>(sycl::nd_range<2>{
      sycl::range<2>{1, 1}, sycl::range<2>{1, 1}});
  test_parallel_for<class NdItem3Name, sycl::nd_item<3>>(sycl::nd_range<3>{
      sycl::range<3>{1, 1, 1}, sycl::range<3>{1, 1, 1}});

  test_parallel_for_work_group<class Group1Name, sycl::group<1>>(sycl::range<1>{1});
  test_parallel_for_work_group<class Group2Name, sycl::group<2>>(sycl::range<2>{1, 1});
  test_parallel_for_work_group<class Group3Name, sycl::group<3>>(sycl::range<3>{1, 1, 1});

  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range{1}, [=](auto &) {});
  });
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range{1, 1}, [=](auto &) {});
  });
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range{1, 1, 1}, [=](auto &) {});
  });

  return 0;
}
