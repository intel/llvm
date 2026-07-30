//==---- integer_traits.hpp - Fixed-width integer selection ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides fixed_width_signed<Size> and fixed_width_unsigned<Size> — type
// aliases that select the exact-width integer type for a given byte size.
//
// Shared between generic_type_traits.hpp (general OpenCL type conversion) and
// async_work_group_copy_ptr.hpp (async-copy pointer rewriting) so both use
// the same mapping without duplication.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <stdint.h>    // for uint8_t, uint16_t, uint32_t, uint64_t
#include <type_traits> // for conditional_t

namespace sycl {
inline namespace _V1 {
namespace detail {

template <int Size>
using fixed_width_unsigned = std::conditional_t<
    Size == 1, uint8_t,
    std::conditional_t<Size == 2, uint16_t,
                       std::conditional_t<Size == 4, uint32_t, uint64_t>>>;

template <int Size>
using fixed_width_signed = std::conditional_t<
    Size == 1, int8_t,
    std::conditional_t<Size == 2, int16_t,
                       std::conditional_t<Size == 4, int32_t, int64_t>>>;

} // namespace detail
} // namespace _V1
} // namespace sycl
