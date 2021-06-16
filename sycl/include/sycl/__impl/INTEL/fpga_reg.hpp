//==-------------- fpga_reg.hpp --- SYCL FPGA Reg Extensions ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__impl/detail/defines.hpp>

namespace __sycl_internal {
inline namespace __v1 {
namespace INTEL {

template <typename _T> _T fpga_reg(const _T &t) {
#if __has_builtin(__builtin_intel_fpga_reg)
  return __builtin_intel_fpga_reg(t);
#else
  return t;
#endif
}

} // namespace INTEL
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

namespace sycl {
  using namespace __sycl_internal::__v1;
}

// Keep it consistent with FPGA attributes like intelfpga::memory()
// Currently clang does not support nested namespace for attributes
namespace intelfpga {
template <typename _T> _T fpga_reg(const _T &t) {
  return sycl::INTEL::fpga_reg(t);
}
} // namespace intelfpga
