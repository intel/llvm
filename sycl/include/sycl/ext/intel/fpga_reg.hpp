//==-------------- fpga_reg.hpp --- SYCL FPGA Reg Extensions ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines.hpp>
#include <type_traits>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace intel {

// Returns a registered copy of the input
// This function is intended for FPGA users to instruct the compiler to insert
// at least one register stage between the input and the return value.
template <typename _T>
typename std::enable_if<std::is_trivially_copyable<_T>::value, _T>::type
fpga_reg(_T t) {
#if __has_builtin(__builtin_intel_fpga_reg)
  return __builtin_intel_fpga_reg(t);
#else
  return t;
#endif
}

template <typename _T>
[[deprecated(
    "ext::intel::fpga_reg will only support trivially_copyable types in a "
    "future release. The type used here will be disallowed.")]]
typename std::enable_if<std::is_trivially_copyable<_T>::value == false,
                        _T>::type
fpga_reg(_T t) {
#if __has_builtin(__builtin_intel_fpga_reg)
  return __builtin_intel_fpga_reg(t);
#else
  return t;
#endif
}

} // namespace intel
} // namespace ext

} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl

// Keep it consistent with FPGA attributes like intelfpga::memory()
// Currently clang does not support nested namespace for attributes
namespace intelfpga {
template <typename _T>
[[deprecated("intelfpga::fpga_reg will be removed in a future release.")]] _T
fpga_reg(const _T &t) {
  return sycl::ext::intel::fpga_reg(t);
}
} // namespace intelfpga
