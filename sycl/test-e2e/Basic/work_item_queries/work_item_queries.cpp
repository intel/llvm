// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//===- work_item_queries.cpp - KHR work item queries test -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define __DPCPP_ENABLE_UNFINISHED_KHR_EXTENSIONS

#include <cassert>
#include <sycl/detail/core.hpp>
#include <sycl/khr/work_item_queries.hpp>

template <size_t... Dims> static void check_this_nd_item_api() {
  // Define the kernel ranges.
  constexpr int Dimensions = sizeof...(Dims);
  const sycl::range<Dimensions> local_range{Dims...};
  const sycl::range<Dimensions> global_range = local_range;
  const sycl::nd_range<Dimensions> nd_range{global_range, local_range};
  // Launch an ND-range kernel.
  sycl::queue q;
  sycl::buffer<bool, Dimensions> results{global_range};
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor acc{results, cgh, sycl::write_only};
    cgh.parallel_for(nd_range, [=](sycl::nd_item<Dimensions> it) {
      // Compare it to this_nd_item<Dimensions>().
      acc[it.get_global_id()] = (it == sycl::khr::this_nd_item<Dimensions>());
    });
  });
  // Check the test results.
  sycl::host_accessor acc{results};
  for (const auto &result : acc)
    assert(result);
}

template <size_t... Dims> static void check_this_group_api() {
  // Define the kernel ranges.
  constexpr int Dimensions = sizeof...(Dims);
  const sycl::range<Dimensions> local_range{Dims...};
  const sycl::range<Dimensions> global_range = local_range;
  const sycl::nd_range<Dimensions> nd_range{global_range, local_range};
  // Launch an ND-range kernel.
  sycl::queue q;
  sycl::buffer<bool, Dimensions> results{global_range};
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor acc{results, cgh, sycl::write_only};
    cgh.parallel_for(nd_range, [=](sycl::nd_item<Dimensions> it) {
      // Compare it.get_group() to this_group<Dimensions>().
      acc[it.get_global_id()] =
          (it.get_group() == sycl::khr::this_group<Dimensions>());
    });
  });
  // Check the test results.
  sycl::host_accessor acc{results};
  for (const auto &result : acc)
    assert(result);
}

template <size_t... Dims> static void check_this_sub_group_api() {
  // Define the kernel ranges.
  constexpr int Dimensions = sizeof...(Dims);
  const sycl::range<Dimensions> local_range{Dims...};
  const sycl::range<Dimensions> global_range = local_range;
  const sycl::nd_range<Dimensions> nd_range{global_range, local_range};
  // Launch an ND-range kernel.
  sycl::queue q;
  sycl::buffer<bool, Dimensions> results{global_range};
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor acc{results, cgh, sycl::write_only};
    cgh.parallel_for(nd_range, [=](sycl::nd_item<Dimensions> it) {
      // Compare it.get_sub_group() to this_sub_group().
      acc[it.get_global_id()] =
          (it.get_sub_group() == sycl::khr::this_sub_group());
    });
  });
  // Check the test results.
  sycl::host_accessor acc{results};
  for (const auto &result : acc)
    assert(result);
}

int main() {
  // nd_item
  check_this_nd_item_api<2>();
  check_this_nd_item_api<2, 3>();
  check_this_nd_item_api<2, 3, 4>();
  // group
  check_this_group_api<2>();
  check_this_group_api<2, 3>();
  check_this_group_api<2, 3, 4>();
  // sub_group
  check_this_sub_group_api<2>();
  check_this_sub_group_api<2, 3>();
  check_this_sub_group_api<2, 3, 4>();
}
