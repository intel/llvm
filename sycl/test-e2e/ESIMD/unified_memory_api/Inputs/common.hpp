//==------- common.hpp - DPC++ ESIMD on-device test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../esimd_test_utils.hpp"

using namespace sycl;
using namespace sycl::ext::intel::esimd;

template <typename Key, typename PropertiesT>
constexpr cache_hint getCacheHint(PropertiesT) {
  if constexpr (PropertiesT::template has_property<Key>()) {
    constexpr auto ValueT = PropertiesT::template get_property<Key>();
    return ValueT.hint;
  } else {
    return cache_hint::none;
  }
}

template <typename PropertiesT>
constexpr size_t getAlignment(PropertiesT, size_t DefaultAlignment) {
  if constexpr (PropertiesT::template has_property<
                    sycl::ext::intel::esimd::alignment_key>()) {
    constexpr auto ValueT = PropertiesT::template get_property<
        sycl::ext::intel::esimd::alignment_key>();
    return ValueT.value;
  } else {
    return DefaultAlignment;
  }
}

template <typename T, uint16_t N, bool UseMask, typename PropertiesT>
constexpr size_t getAlignment(PropertiesT Props) {
  constexpr cache_hint L1Hint =
      getCacheHint<sycl::ext::intel::esimd::cache_hint_L1_key>(Props);
  constexpr cache_hint L2Hint =
      getCacheHint<sycl::ext::intel::esimd::cache_hint_L2_key>(Props);
  constexpr bool RequiresPVC =
      L1Hint != cache_hint::none || L2Hint != cache_hint::none || UseMask;

  constexpr bool IsMaxLoadSizePVC = RequiresPVC && (N * sizeof(T) > 256);
  constexpr size_t RequiredAlignment =
      IsMaxLoadSizePVC ? 8 : (RequiresPVC ? 4 : sizeof(T));
  constexpr size_t RequestedAlignment = getAlignment(Props, RequiredAlignment);
  static_assert(RequestedAlignment >= RequiredAlignment, "Too small alignment");
  return RequestedAlignment;
}

enum class TestFeatures { Generic, DG2, PVC };
