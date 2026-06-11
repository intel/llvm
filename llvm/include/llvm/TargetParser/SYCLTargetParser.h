//===-- SYCLTargetParser.h - SYCL target device resolution ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file implement a TargetParser for SYCL Offloading kind.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_SYCLTARGETPARSER_H
#define LLVM_TARGETPARSER_SYCLTARGETPARSER_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
namespace sycl_target {

/// Maps an intel_gpu_<device>, nvidia_gpu_<sm> and amd_gpu_<gfx>) target
/// name to its canonical ocloc / ptxas / hipcc device identifier.
/// Returns an empty StringRef if DeviceName is not recognized.
StringRef resolveGenDevice(StringRef DeviceName);

/// Maps a canonical device name (e.g. "pvc", "acm_g10", "sm_90") to the
/// corresponding SYCL preprocessor macro suffix.  The returned string is of
/// the form "__SYCL_TARGET_<SUFFIX>__", or empty if DeviceName is unknown.
SmallString<64> getGenDeviceMacro(StringRef DeviceName);

/// Maps a GRF mode string ("auto", "small", "large") to the corresponding
/// ocloc flag.  Returns an empty StringRef for unrecognised values.
StringRef getGenGRFFlag(StringRef GRFMode);

} // namespace sycl_target
} // namespace llvm

#endif // LLVM_TARGETPARSER_SYCLTARGETPARSER_H
