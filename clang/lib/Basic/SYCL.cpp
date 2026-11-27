//===--- SYCL.cpp - SYCL-specific offloading utilities --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SYCL.h"

#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/TargetParser/NVPTXTargetParser.h"

namespace clang {

// Generic AMD GPUs (gfx*-generic and the "generic"/"generic-hsa" pseudo
// targets) do not support SYCL offloading.
static bool IsAMDGenericGPUArch(OffloadArch Arch) {
  if (!Arch.isAMDGPU())
    return false;
  llvm::AMDGPU::GPUKind Kind = Arch.amdgpuKind();
  return llvm::AMDGPU::isPseudoTarget(Kind) ||
         llvm::AMDGPU::getArchNameAMDGCN(Kind).ends_with("-generic");
}

bool IsSYCLSupportedAMDGPUArch(OffloadArch Arch) {
  return Arch.isAMDGPU() && !IsAMDGenericGPUArch(Arch);
}

bool IsSYCLSupportedNVidiaGPUArch(OffloadArch Arch) {
  if (!Arch.isNVPTX())
    return false;
  unsigned SmVersion = llvm::NVPTX::getSmVersion(Arch.nvptxKind());
  return SmVersion >= 500 && SmVersion <= 900;
}

} // namespace clang
