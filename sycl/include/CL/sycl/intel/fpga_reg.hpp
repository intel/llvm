//==-------------- fpga_reg.hpp --- SYCL FPGA Reg Extensions ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace cl {
namespace sycl {
namespace intel {

template <typename T> T fpga_reg(const T &t) {
#if defined(__clang__) && __has_builtin(__builtin_intel_fpga_reg)
  return __builtin_intel_fpga_reg(t);
#else
  return t;
#endif
}

} // namespace intel
} // namespace sycl
} // namespace cl
