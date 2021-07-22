//==- fpga_device_selector.hpp --- SYCL FPGA device selector shortcut  -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>

__SYCL_WARNING(
    "CL/sycl/INTEL/fpga_device_selector.hpp usage is deprecated, include "
    "sycl/ext/intel/fpga_device_selector.hpp instead")

#include <sycl/ext/intel/fpga_device_selector.hpp>
