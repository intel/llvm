//==---------- nd_loop.hpp ---- ND iteration helpers ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __SYCL_ALWAYS_INLINE

#include <cstddef> // for size_t

namespace sycl {
inline namespace _V1 {
namespace detail {

// Produces N-dimensional object of type T whose all components are initialized
// to given integer value.
template <int N, template <int> class T> struct InitializedVal {
  template <int Val> static T<N> get();
};

// Specialization for a one-dimensional type.
template <template <int> class T> struct InitializedVal<1, T> {
  template <int Val> static T<1> get() { return T<1>{Val}; }
};

// Specialization for a two-dimensional type.
template <template <int> class T> struct InitializedVal<2, T> {
  template <int Val> static T<2> get() { return T<2>{Val, Val}; }
};

// Specialization for a three-dimensional type.
template <template <int> class T> struct InitializedVal<3, T> {
  template <int Val> static T<3> get() { return T<3>{Val, Val, Val}; }
};

/// Helper class for the \c NDLoop.
template <int NDims, int Dim, template <int> class LoopBoundTy, typename FuncTy,
          template <int> class LoopIndexTy>
struct NDLoopIterateImpl {
  NDLoopIterateImpl(const LoopIndexTy<NDims> &LowerBound,
                    const LoopBoundTy<NDims> &Stride,
                    const LoopBoundTy<NDims> &UpperBound, FuncTy f,
                    LoopIndexTy<NDims> &Index) {
    constexpr size_t AdjIdx = NDims - 1 - Dim;
    for (Index[AdjIdx] = LowerBound[AdjIdx]; Index[AdjIdx] < UpperBound[AdjIdx];
         Index[AdjIdx] += Stride[AdjIdx]) {

      NDLoopIterateImpl<NDims, Dim - 1, LoopBoundTy, FuncTy, LoopIndexTy>{
          LowerBound, Stride, UpperBound, f, Index};
    }
  }
};

// Specialization for Dim=0 to terminate recursion
template <int NDims, template <int> class LoopBoundTy, typename FuncTy,
          template <int> class LoopIndexTy>
struct NDLoopIterateImpl<NDims, 0, LoopBoundTy, FuncTy, LoopIndexTy> {
  NDLoopIterateImpl(const LoopIndexTy<NDims> &LowerBound,
                    const LoopBoundTy<NDims> &Stride,
                    const LoopBoundTy<NDims> &UpperBound, FuncTy f,
                    LoopIndexTy<NDims> &Index) {

    constexpr size_t AdjIdx = NDims - 1;
    for (Index[AdjIdx] = LowerBound[AdjIdx]; Index[AdjIdx] < UpperBound[AdjIdx];
         Index[AdjIdx] += Stride[AdjIdx]) {

      f(Index);
    }
  }
};

/// Generates an NDims-dimensional perfect loop nest. The purpose of this class
/// is to better support handling of situations where there must be a loop nest
/// over a multi-dimensional space - it allows to avoid generating unnecessary
/// outer loops like 'for (int z=0; z<1; z++)' in case of 1D and 2D iteration
/// spaces or writing specializations of the algorithms for 1D, 2D and 3D cases.
/// Loop is unrolled in a reverse directions, i.e. ID = 0 is the inner-most one.
template <int NDims> struct NDLoop {
  /// Generates ND loop nest with {0,..0} .. \c UpperBound bounds with unit
  /// stride. Applies \c f at each iteration, passing current index of
  /// \c LoopIndexTy<NDims> type as the parameter.
  template <template <int> class LoopBoundTy, typename FuncTy,
            template <int> class LoopIndexTy = LoopBoundTy>
  static __SYCL_ALWAYS_INLINE void iterate(const LoopBoundTy<NDims> &UpperBound,
                                           FuncTy f) {
    const LoopIndexTy<NDims> LowerBound =
        InitializedVal<NDims, LoopIndexTy>::template get<0>();
    const LoopBoundTy<NDims> Stride =
        InitializedVal<NDims, LoopBoundTy>::template get<1>();
    LoopIndexTy<NDims> Index =
        InitializedVal<NDims, LoopIndexTy>::template get<0>();

    NDLoopIterateImpl<NDims, NDims - 1, LoopBoundTy, FuncTy, LoopIndexTy>{
        LowerBound, Stride, UpperBound, f, Index};
  }

  /// Generates ND loop nest with \c LowerBound .. \c UpperBound bounds and
  /// stride \c Stride. Applies \c f at each iteration, passing current index of
  /// \c LoopIndexTy<NDims> type as the parameter.
  template <template <int> class LoopBoundTy, typename FuncTy,
            template <int> class LoopIndexTy = LoopBoundTy>
  static __SYCL_ALWAYS_INLINE void iterate(const LoopIndexTy<NDims> &LowerBound,
                                           const LoopBoundTy<NDims> &Stride,
                                           const LoopBoundTy<NDims> &UpperBound,
                                           FuncTy f) {
    LoopIndexTy<NDims> Index =
        InitializedVal<NDims, LoopIndexTy>::template get<0>();
    NDLoopIterateImpl<NDims, NDims - 1, LoopBoundTy, FuncTy, LoopIndexTy>{
        LowerBound, Stride, UpperBound, f, Index};
  }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
