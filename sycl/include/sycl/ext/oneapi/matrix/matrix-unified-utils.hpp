//===------- matrix-unified.hpp - SYCL matrix extension ----*- C++ -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <optional>                     // std::optional
#include <string_view>                  // std::string_view
#include <sycl/__spirv/spirv_types.hpp> // __spv namespace
#include <sycl/ext/oneapi/bfloat16.hpp> // bfloat16
#include <utility>                      // std::pair

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {
namespace matrix {

enum class use { a, b, accumulator };

enum class layout {
  row_major = 0,
  col_major = 1,
  ext_intel_packed = 2,
  dynamic = 3
};

namespace precision {
class tf32 {
  tf32() = delete;
};
} // namespace precision

} // namespace matrix
} // namespace experimental
} // namespace oneapi
} // namespace ext

namespace detail {
using UseToUseStringPair =
    std::pair<ext::oneapi::experimental::matrix::use, const char *>;

constexpr UseToUseStringPair UseToUseStringMap[] = {
    {ext::oneapi::experimental::matrix::use::a, "use::a"},
    {ext::oneapi::experimental::matrix::use::b, "use::b"},
    {ext::oneapi::experimental::matrix::use::accumulator, "use::accumulator"},
};

constexpr const char *
convertMatrixUseEnumToString(ext::oneapi::experimental::matrix::use Use) {
  for (const auto &Item : UseToUseStringMap) {
    if (Item.first == Use)
      return Item.second;
  }
  return "";
}

constexpr std::optional<ext::oneapi::experimental::matrix::use>
convertMatrixUseStringToEnum(const char *UseString) {
  for (const auto &Item : UseToUseStringMap) {
    if (std::string_view(Item.second) == UseString)
      return Item.first;
  }
  return std::nullopt;
}

// propagateConstexprLayout uses the exact name of the function, so we use
// extern "C" here.
extern "C" constexpr __spv::MatrixLayout joint_matrix_layout_to_spv(
    sycl::ext::oneapi::experimental::matrix::layout Layout) {
  switch (Layout) {
  case sycl::ext::oneapi::experimental::matrix::layout::row_major:
    return __spv::MatrixLayout::RowMajor;
  case sycl::ext::oneapi::experimental::matrix::layout::col_major:
    return __spv::MatrixLayout::ColumnMajor;
  case sycl::ext::oneapi::experimental::matrix::layout::ext_intel_packed:
    return __spv::MatrixLayout::Packed;
  case sycl::ext::oneapi::experimental::matrix::layout::dynamic:
    return __spv::MatrixLayout::Dynamic;
  }
}

template <typename Ta, typename Tb, typename Tc, typename Td>
constexpr uint32_t CalculateMatrixOperand() {
  uint32_t returnValue = 0x00;
  if constexpr (std::is_same<Ta, sycl::ext::oneapi::bfloat16>::value &&
                std::is_same<Tb, sycl::ext::oneapi::bfloat16>::value)
    returnValue += static_cast<uint32_t>(
        __spv::MatrixOperands::MatrixAAndBBFloat16ComponentsINTEL);
  if constexpr (std::is_same<Tc, sycl::ext::oneapi::bfloat16>::value)
    returnValue += static_cast<uint32_t>(
        __spv::MatrixOperands::MatrixCBFloat16ComponentsINTEL);
  if constexpr (std::is_same<Td, sycl::ext::oneapi::bfloat16>::value)
    returnValue += static_cast<uint32_t>(
        __spv::MatrixOperands::MatrixResultBFloat16ComponentsINTEL);
  if constexpr (std::is_signed<Ta>::value)
    returnValue += static_cast<uint32_t>(
        __spv::MatrixOperands::MatrixASignedComponentsKHR);
  if constexpr (std::is_signed<Tb>::value)
    returnValue += static_cast<uint32_t>(
        __spv::MatrixOperands::MatrixBSignedComponentsKHR);
  return returnValue;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
