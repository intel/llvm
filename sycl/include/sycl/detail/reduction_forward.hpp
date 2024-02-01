//==---- reduction_forward.hpp - SYCL reduction forward decl ---*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <sycl/detail/helpers.hpp>   // for Builder
#include <sycl/detail/item_base.hpp> // for range
#include <sycl/id.hpp>               // for id
#include <sycl/item.hpp>             // for item
#include <sycl/nd_range.hpp>         // for nd_range
#include <sycl/range.hpp>            // for range

#include <stddef.h> // for size_t

// To be included in <sycl/handler.hpp>. Note that reductions implementation
// need complete sycl::handler type so we cannot include whole
// <sycl/reduction.hpp> there.

namespace sycl {
inline namespace _V1 {
namespace detail {
template <typename T, class BinaryOperation, int Dims, size_t Extent,
          bool ExplicitIdentity, typename RedOutVar>
class reduction_impl_algo;

namespace reduction {
enum class strategy : int {
  auto_select,

  // These three are only auto-selected for sycl::range entry point.
  group_reduce_and_last_wg_detection,
  local_atomic_and_atomic_cross_wg,
  range_basic,

  group_reduce_and_atomic_cross_wg,
  local_mem_tree_and_atomic_cross_wg,
  group_reduce_and_multiple_kernels,
  basic,

  multi,
};

// Reductions implementation need access to private members of handler. Those
// are limited to those below.
inline void finalizeHandler(handler &CGH);
template <class FunctorTy> void withAuxHandler(handler &CGH, FunctorTy Func);

template <int Dims>
item<Dims, false> getDelinearizedItem(range<Dims> Range, id<Dims> Id) {
  return Builder::createItem<Dims, false>(Range, Id);
}
} // namespace reduction

template <typename KernelName,
          reduction::strategy Strategy = reduction::strategy::auto_select,
          int Dims, typename PropertiesT, typename... RestT>
void reduction_parallel_for(handler &CGH, range<Dims> NDRange,
                            PropertiesT Properties, RestT... Rest);

template <typename KernelName,
          reduction::strategy Strategy = reduction::strategy::auto_select,
          int Dims, typename PropertiesT, typename... RestT>
void reduction_parallel_for(handler &CGH, nd_range<Dims> NDRange,
                            PropertiesT Properties, RestT... Rest);

template <typename T> struct IsReduction;
template <typename FirstT, typename... RestT> struct AreAllButLastReductions;

} // namespace detail
} // namespace _V1
} // namespace sycl
