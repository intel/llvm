//===-- value_conv.hpp - This file provides common functions generate values for
//      testing. ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions that let obtain converted reference data for
/// test according to current underlying types.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "type_traits.hpp"
#include "value.hpp"

namespace esimd_test::api::functional {

// Utility class to retrieve specific values for tests depending on the source
// and destination data types. May be used to retrieve converted reference data.
// All provided methods are safe to use and protected from UB when call
// static_cast<int>(unsigned int).
template <typename SrcT, typename DstT> struct value_conv {
  static inline SrcT denorm_min() {
    if constexpr (!type_traits::is_sycl_floating_point_v<SrcT>) {
      // Return zero for any integral type the same way std::denorm_min does
      return 0;
    } else if constexpr (sizeof(SrcT) > sizeof(DstT)) {
      // Use the biggest value so it would not degenerate to zero during
      // conversion from SrcT to DstT
      return static_cast<SrcT>(value<DstT>::denorm_min());
    } else {
      return value<SrcT>::denorm_min();
    }
  }
};

// Provides std::vector with the reference data according to the obtained data
// types and number of elements.
template <typename SrcT, typename DstT, int NumElems>
std::vector<SrcT> generate_ref_conv_data() {
  static_assert(std::is_integral_v<SrcT> ||
                    type_traits::is_sycl_floating_point_v<SrcT>,
                "Invalid source type.");
  static_assert(std::is_integral_v<DstT> ||
                    type_traits::is_sycl_floating_point_v<DstT>,
                "Invalid destination type.");

  // TODO: Implement functions for obtain lowest and max values without UB
  // cases.
  static const SrcT positive = static_cast<SrcT>(126.75);
  static const SrcT max = 10;
  // Use zero for unsigned types
  static const SrcT min = std::min<SrcT>(-max, 0);
  static const SrcT max_half = max / 2;
  static const SrcT min_half = min / 2;

  std::vector<SrcT> ref_data;

  if constexpr (type_traits::is_sycl_floating_point_v<SrcT> &&
                type_traits::is_sycl_floating_point_v<DstT>) {
    static const SrcT nan = value<SrcT>::nan();
    static const SrcT inf = value<SrcT>::inf();
    static const SrcT denorm = value_conv<SrcT, DstT>::denorm_min();

    ref_data = details::construct_ref_data<SrcT, NumElems>(
        {min, max, -0.0, +0.0, 0.1, denorm, nan, -inf});
  } else if constexpr (type_traits::is_sycl_floating_point_v<SrcT> &&
                       std::is_unsigned_v<DstT>) {
    ref_data = details::construct_ref_data<SrcT, NumElems>(
        {-0.0, max, max_half, -max_half});
  } else if constexpr (type_traits::is_sycl_floating_point_v<SrcT> &&
                       type_traits::is_sycl_signed_v<DstT>) {
    ref_data = details::construct_ref_data<SrcT, NumElems>(
        {-0.0, max, max_half, min, min_half});
  } else if constexpr (type_traits::is_sycl_signed_v<SrcT> &&
                       type_traits::is_sycl_signed_v<DstT>) {
    ref_data = details::construct_ref_data<SrcT, NumElems>(
        {min, min_half, 0, max_half, max});
  } else if constexpr (type_traits::is_sycl_signed_v<SrcT> &&
                       std::is_unsigned_v<DstT>) {
    static const SrcT src_min = value<SrcT>::lowest();
    static const SrcT src_min_half = src_min / 2;

    ref_data = details::construct_ref_data<SrcT, NumElems>(
        {src_min, src_min_half, 0, max_half, max});
  } else if constexpr (std::is_unsigned_v<SrcT>) {
    ref_data = details::construct_ref_data<SrcT, NumElems>({0, max_half, max});
  } else {
    static_assert(!std::is_same_v<SrcT, SrcT>, "Unexpected types combination");
  }

  return ref_data;
}

} // namespace esimd_test::api::functional
