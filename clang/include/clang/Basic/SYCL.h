//===--- SYCL.h - SYCL-specific offloading utilities ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_SYCL_H
#define LLVM_CLANG_BASIC_SYCL_H

#include "clang/Basic/OffloadArch.h"

namespace clang {

// Check if the given Arch value is a valid SYCL supported AMD GPU.
// GFX*_GENERIC AMD GPUs are not supported.
bool IsSYCLSupportedAMDGPUArch(OffloadArch Arch);

// Check if the given Arch value is a valid SYCL supported NVidia GPU.
bool IsSYCLSupportedNVidiaGPUArch(OffloadArch Arch);

} // namespace clang

#endif // LLVM_CLANG_BASIC_SYCL_H
