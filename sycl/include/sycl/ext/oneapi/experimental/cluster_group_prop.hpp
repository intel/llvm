//==--- cluster_group.hpp --- SYCL extension for cluster group
//------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/range.hpp>

namespace sycl {
namespace _V1 {
namespace ext::oneapi::experimental::cuda {

template <int Dim> struct cluster_size {
  cluster_size(const range<Dim> &size) : size(size) {}
  std::size_t operator[](std::size_t i) { return size[i]; }
  range<Dim> size;
};

template <int Dim> using cluster_size_key = cluster_size<Dim>;
} // namespace ext::oneapi::experimental::cuda
} // namespace _V1
} // namespace sycl
