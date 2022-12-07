//==---- reduction_forward.hpp - SYCL reduction forward decl ---*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

// To be included in <sycl/handler.hpp>. Note that reductions implementation
// need complete sycl::handler type so we cannot include whole
// <sycl/reduction.hpp> there.

#include <sycl/detail/common.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {
template <typename T, class BinaryOperation, int Dims, size_t Extent,
          typename RedOutVar>
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

template <typename T> struct IsReduction;
template <typename FirstT, typename... RestT> struct AreAllButLastReductions;

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
