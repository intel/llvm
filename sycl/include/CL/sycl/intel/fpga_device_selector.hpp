//==- fpga_device_selector.hpp --- SYCL FPGA device selector shortcut  -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/INTEL/fpga_device_selector.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace intel {
using platform_selector = INTEL::platform_selector;
using fpga_selector = INTEL::fpga_selector;
using fpga_emulator_selector = INTEL::fpga_emulator_selector;
} // namespace intel
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
