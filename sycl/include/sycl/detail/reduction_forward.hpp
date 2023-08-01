//==---- reduction_forward.hpp - SYCL reduction forward decl ---*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <stddef.h>              // for size_t

#include "detail/item_base.hpp"  // for range
#include "id.hpp"                // for id
#include "item.hpp"              // for getDelinearizedItem, item
#include "nd_range.hpp"          // for nd_range
#include "range.hpp"             // for range
// To be included in <sycl/handler.hpp>. Note that reductions implementation
// need complete sycl::handler type so we cannot include whole
// <sycl/reduction.hpp> there.

namespace sycl {
inline namespace _V1 {
namespace detail {

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
  return {Range, Id};
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


} // namespace detail
} // namespace _V1
} // namespace sycl
