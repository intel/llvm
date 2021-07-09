//==------------- fpga_utils.hpp --- SYCL FPGA Reg Extensions --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>

namespace {
[[__SYCL2020_DEPRECATED("include sycl/ext/intel/fpga_utils.hpp "
                        "instead")]] constexpr static bool HeaderDeprecated =
    true;
constexpr static bool TriggerHeaderDeprecationWarning = HeaderDeprecated;
} // namespace

#include <sycl/ext/intel/fpga_utils.hpp>
