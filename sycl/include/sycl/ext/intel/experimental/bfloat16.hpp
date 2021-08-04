//==--------- bfloat16.hpp ------- SYCL bfloat16 conversion ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/__spirv/spirv_ops.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {

class [[sycl_detail::uses_aspects(ext_intel_bf16_conversion)]]
bfloat16 {
  using storage_t = uint16_t;
  storage_t value;

public:
  bfloat16() = default;
  bfloat16(const bfloat16&) = default;
  ~bfloat16() = default;

  // Direct initialization
  bfloat16(const storage_t &a) : value(a) {}

  // convert from float to bfloat16
  bfloat16(const float &a) {
#if defined(__SYCL_DEVICE_ONLY__)
    value = __spirv_ConvertFToBF16INTEL(a);
#else
    throw runtime_error("Bfloat16 conversion is not supported on HOST device.",
                        PI_INVALID_DEVICE);
#endif
  }

  // convert from bfloat16 to float
  operator float() const {
#if defined(__SYCL_DEVICE_ONLY__)
    return __spirv_ConvertBF16ToFINTEL(value);
#else
    throw runtime_error("Bfloat16 conversion is not supported on HOST device.",
                        PI_INVALID_DEVICE);
#endif
  }

  // Get bfloat16 as uint16.
  operator storage_t() const { return value; }
};

} // namespace experimental
} // namespace intel
} // namespace ext

namespace __SYCL2020_DEPRECATED("use 'ext::intel' instead") INTEL {
  using namespace ext::intel;
}
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

