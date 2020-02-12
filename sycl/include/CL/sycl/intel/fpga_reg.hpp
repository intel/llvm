//==-------------- fpga_reg.hpp --- SYCL FPGA Reg Extensions ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace intel {

template <typename T> T fpga_reg(const T &t) {
#if __has_builtin(__builtin_intel_fpga_reg)
  return __builtin_intel_fpga_reg(t);
#else
  return t;
#endif
}

} // namespace intel
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

// Keep it consistent with FPGA attributes like intelfpga::memory()
// Currently clang does not support nested namespace for attributes
namespace intelfpga {
template <typename T> T fpga_reg(const T &t) {
	return cl::sycl::intel::fpga_reg(t);
}	
}
