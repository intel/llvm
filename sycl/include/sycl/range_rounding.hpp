//==----------- range_rounding.hpp --- SYCL range rounding utils -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/cg_types.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/helpers.hpp>
#include <sycl/detail/iostream_proxy.hpp>
#include <sycl/device.hpp>
#include <sycl/ext/oneapi/kernel_properties/properties.hpp>
#include <sycl/id.hpp>
#include <sycl/item.hpp>
#include <sycl/kernel_handler.hpp>
#include <sycl/range.hpp>

#include <tuple>
#include <type_traits>

#include <stddef.h>

namespace sycl {
inline namespace _V1 {

namespace detail {

template <int Dims> class RoundedRangeIDGenerator {
  id<Dims> Id;
  id<Dims> InitId;
  range<Dims> UserRange;
  range<Dims> RoundedRange;
  bool Done = false;

public:
  RoundedRangeIDGenerator(const id<Dims> &Id, const range<Dims> &UserRange,
                          const range<Dims> &RoundedRange)
      : Id(Id), InitId(Id), UserRange(UserRange), RoundedRange(RoundedRange) {
    for (int i = 0; i < Dims; ++i)
      if (Id[i] >= UserRange[i])
        Done = true;
  }

  explicit operator bool() { return !Done; }

  void updateId() {
    for (int i = 0; i < Dims; ++i) {
      Id[i] += RoundedRange[i];
      if (Id[i] < UserRange[i])
        return;
      Id[i] = InitId[i];
    }
    Done = true;
  }

  id<Dims> getId() { return Id; }

  template <typename KernelType> auto getItem() {
    if constexpr (std::is_invocable_v<KernelType, item<Dims> &> ||
                  std::is_invocable_v<KernelType, item<Dims> &, kernel_handler>)
      return detail::Builder::createItem<Dims, true>(UserRange, getId(), {});
    else {
      static_assert(std::is_invocable_v<KernelType, item<Dims, false> &> ||
                        std::is_invocable_v<KernelType, item<Dims, false> &,
                                            kernel_handler>,
                    "Kernel must be invocable with an item!");
      return detail::Builder::createItem<Dims, false>(UserRange, getId());
    }
  }
};

// TODO: The wrappers can be optimized further so that the body
// essentially looks like this:
//   for (auto z = it[2]; z < UserRange[2]; z += it.get_range(2))
//     for (auto y = it[1]; y < UserRange[1]; y += it.get_range(1))
//       for (auto x = it[0]; x < UserRange[0]; x += it.get_range(0))
//         KernelFunc({x,y,z});
template <typename TransformedArgType, int Dims, typename KernelType>
class RoundedRangeKernel {
public:
  range<Dims> UserRange;
  KernelType KernelFunc;
  void operator()(item<Dims> It) const {
    auto RoundedRange = It.get_range();
    for (RoundedRangeIDGenerator Gen(It.get_id(), UserRange, RoundedRange); Gen;
         Gen.updateId()) {
      auto item = Gen.template getItem<KernelType>();
      KernelFunc(item);
    }
  }

  // Copy the properties_tag getter from the original kernel to propagate
  // property(s)
  template <
      typename T = KernelType,
      typename = std::enable_if_t<ext::oneapi::experimental::detail::
                                      HasKernelPropertiesGetMethod<T>::value>>
  auto get(ext::oneapi::experimental::properties_tag) const {
    return KernelFunc.get(ext::oneapi::experimental::properties_tag{});
  }
};

template <typename TransformedArgType, int Dims, typename KernelType>
class RoundedRangeKernelWithKH {
public:
  range<Dims> UserRange;
  KernelType KernelFunc;
  void operator()(item<Dims> It, kernel_handler KH) const {
    auto RoundedRange = It.get_range();
    for (RoundedRangeIDGenerator Gen(It.get_id(), UserRange, RoundedRange); Gen;
         Gen.updateId()) {
      auto item = Gen.template getItem<KernelType>();
      KernelFunc(item, KH);
    }
  }

  // Copy the properties_tag getter from the original kernel to propagate
  // property(s)
  template <
      typename T = KernelType,
      typename = std::enable_if_t<ext::oneapi::experimental::detail::
                                      HasKernelPropertiesGetMethod<T>::value>>
  auto get(ext::oneapi::experimental::properties_tag) const {
    return KernelFunc.get(ext::oneapi::experimental::properties_tag{});
  }
};

template <typename WrapperT, typename TransformedArgType, int Dims,
          typename KernelType,
          std::enable_if_t<detail::KernelLambdaHasKernelHandlerArgT<
              KernelType, TransformedArgType>::value> * = nullptr>
auto getRangeRoundedKernelLambda(KernelType KernelFunc, range<Dims> UserRange) {
  return detail::RoundedRangeKernelWithKH<TransformedArgType, Dims, KernelType>{
      UserRange, KernelFunc};
}

template <typename WrapperT, typename TransformedArgType, int Dims,
          typename KernelType,
          std::enable_if_t<!detail::KernelLambdaHasKernelHandlerArgT<
              KernelType, TransformedArgType>::value> * = nullptr>
auto getRangeRoundedKernelLambda(KernelType KernelFunc, range<Dims> UserRange) {
  return detail::RoundedRangeKernel<TransformedArgType, Dims, KernelType>{
      UserRange, KernelFunc};
}

void __SYCL_EXPORT GetRangeRoundingSettings(size_t &MinFactor,
                                            size_t &GoodFactor,
                                            size_t &MinRange);

std::tuple<std::array<size_t, 3>, bool>
    __SYCL_EXPORT getMaxWorkGroups(const device &Device);

bool __SYCL_EXPORT DisableRangeRounding();

bool __SYCL_EXPORT RangeRoundingTrace();

template <int Dims>
std::tuple<range<Dims>, bool> getRoundedRange(range<Dims> UserRange,
                                              const device &Device) {
  range<Dims> RoundedRange = UserRange;
  // Disable the rounding-up optimizations under these conditions:
  // 1. The env var SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING is set.
  // 2. The kernel is provided via an interoperability method (this uses a
  // different code path).
  // 3. The range is already a multiple of the rounding factor.
  //
  // Cases 2 and 3 could be supported with extra effort.
  // As an optimization for the common case it is an
  // implementation choice to not support those scenarios.
  // Note that "this_item" is a free function, i.e. not tied to any
  // specific id or item. When concurrent parallel_fors are executing
  // on a device it is difficult to tell which parallel_for the call is
  // being made from. One could replicate portions of the
  // call-graph to make this_item calls kernel-specific but this is
  // not considered worthwhile.

  // Perform range rounding if rounding-up is enabled.
  if (DisableRangeRounding())
    return {range<Dims>{}, false};

  // Range should be a multiple of this for reasonable performance.
  size_t MinFactorX = 16;
  // Range should be a multiple of this for improved performance.
  size_t GoodFactor = 32;
  // Range should be at least this to make rounding worthwhile.
  size_t MinRangeX = 1024;

  // Check if rounding parameters have been set through environment:
  // SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS=MinRound:PreferredRound:MinRange
  GetRangeRoundingSettings(MinFactorX, GoodFactor, MinRangeX);

  // In SYCL, each dimension of a global range size is specified by
  // a size_t, which can be up to 64 bits.  All backends should be
  // able to accept a kernel launch with a 32-bit global range size
  // (i.e. do not throw an error).  The OpenCL CPU backend will
  // accept every 64-bit global range, but the GPU backends will not
  // generally accept every 64-bit global range.  So, when we get a
  // non-32-bit global range, we wrap the old kernel in a new kernel
  // that has each work item peform multiple invocations the old
  // kernel in a 32-bit global range.
  id<Dims> MaxNWGs = [&] {
    auto [MaxWGs, HasMaxWGs] = getMaxWorkGroups(Device);
    if (!HasMaxWGs) {
      id<Dims> Default;
      for (int i = 0; i < Dims; ++i)
        Default[i] = (std::numeric_limits<int32_t>::max)();
      return Default;
    }

    id<Dims> IdResult;
    size_t Limit = (std::numeric_limits<int>::max)();
    for (int i = 0; i < Dims; ++i)
      IdResult[i] = (std::min)(Limit, MaxWGs[Dims - i - 1]);
    return IdResult;
  }();
  auto M = (std::numeric_limits<uint32_t>::max)();
  range<Dims> MaxRange;
  for (int i = 0; i < Dims; ++i) {
    auto DesiredSize = MaxNWGs[i] * GoodFactor;
    MaxRange[i] =
        DesiredSize <= M ? DesiredSize : (M / GoodFactor) * GoodFactor;
  }

  bool DidAdjust = false;
  auto Adjust = [&](int Dim, size_t Value) {
    if (RangeRoundingTrace())
      std::cout << "parallel_for range adjusted at dim " << Dim << " from "
                << RoundedRange[Dim] << " to " << Value << std::endl;
    RoundedRange[Dim] = Value;
    DidAdjust = true;
  };

#ifdef __SYCL_EXP_PARALLEL_FOR_RANGE_ROUNDING__
  size_t GoodExpFactor = 1;
  switch (Dims) {
  case 1:
    GoodExpFactor = 32; // Make global range multiple of {32}
    break;
  case 2:
    GoodExpFactor = 16; // Make global range multiple of {16, 16}
    break;
  case 3:
    GoodExpFactor = 8; // Make global range multiple of {8, 8, 8}
    break;
  }

  // Check if rounding parameters have been set through environment:
  // SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS=MinRound:PreferredRound:MinRange
  GetRangeRoundingSettings(MinFactorX, GoodExpFactor, MinRangeX);

  for (auto i = 0; i < Dims; ++i)
    if (UserRange[i] % GoodExpFactor) {
      Adjust(i, ((UserRange[i] / GoodExpFactor) + 1) * GoodExpFactor);
    }
#else
  // Perform range rounding if there are sufficient work-items to
  // need rounding and the user-specified range is not a multiple of
  // a "good" value.
  if (RoundedRange[0] % MinFactorX != 0 && RoundedRange[0] >= MinRangeX) {
    // It is sufficient to round up just the first dimension.
    // Multiplying the rounded-up value of the first dimension
    // by the values of the remaining dimensions (if any)
    // will yield a rounded-up value for the total range.
    Adjust(0, ((RoundedRange[0] + GoodFactor - 1) / GoodFactor) * GoodFactor);
  }
#endif // __SYCL_EXP_PARALLEL_FOR_RANGE_ROUNDING__
#ifdef __SYCL_FORCE_PARALLEL_FOR_RANGE_ROUNDING__
  // If we are forcing range rounding kernels to be used, we always want the
  // rounded range kernel to be generated, even if rounding isn't needed
  DidAdjust = true;
#endif // __SYCL_FORCE_PARALLEL_FOR_RANGE_ROUNDING__

  for (int i = 0; i < Dims; ++i)
    if (RoundedRange[i] > MaxRange[i])
      Adjust(i, MaxRange[i]);

  if (!DidAdjust)
    return {range<Dims>{}, false};
  return {RoundedRange, true};
}

} // namespace detail
} // namespace _V1
} // namespace sycl
