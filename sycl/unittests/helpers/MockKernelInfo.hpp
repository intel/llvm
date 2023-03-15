//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/kernel_desc.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace unittest {
struct MockKernelInfoBase {
  static constexpr unsigned getNumParams() { return 0; }
  static const detail::kernel_param_desc_t &getParamDesc(int) {
    static detail::kernel_param_desc_t Dummy;
    return Dummy;
  }
  static constexpr bool isESIMD() { return false; }
  static constexpr bool callsThisItem() { return false; }
  static constexpr bool callsAnyThisFreeFunction() { return false; }
  static constexpr int64_t getKernelSize() { return 1; }
};

} // namespace unittest
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
