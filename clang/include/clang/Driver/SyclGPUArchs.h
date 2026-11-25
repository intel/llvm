//===-- SyclGPUArchs.h - Shared SYCL GPU device arch data ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_SYCLGPUARCHS_H
#define LLVM_CLANG_DRIVER_SYCLGPUARCHS_H

#include "llvm/ADT/StringRef.h"

namespace clang::driver {

// Intel GPU device names with native bfloat16 hardware support.
inline constexpr llvm::StringLiteral SYCLGPUArchsWithNativeBF16[] = {
    "pvc",   "acm_g10", "acm_g11", "acm_g12", "bmg_g21", "lnl_m", "ptl_h",
    "ptl_u", "wcl",     "cri",     "nvl_s",   "nvl_u",   "nvl_p"};

} // namespace clang::driver

#endif // LLVM_CLANG_DRIVER_SYCLGPUARCHS_H
