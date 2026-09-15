//===--- NativeCPU.cpp - Implement NativeCPU target feature support -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements NativeCPU TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "NativeCPU.h"
#include <llvm/TargetParser/Host.h>

using namespace clang;
using namespace clang::targets;

static const LangASMap NativeCPUASMap = {
    {LangAS::Default, 0},
    {LangAS::opencl_global, 1},
    {LangAS::opencl_local, 3},
    {LangAS::opencl_constant, 4},
    {LangAS::opencl_private, 0},
    {LangAS::opencl_generic, 0},
    {LangAS::opencl_global_device, 1},
    {LangAS::opencl_global_host, 1},
    {LangAS::cuda_device, 1},
    {LangAS::cuda_constant, 4},
    {LangAS::cuda_shared, 3},
    {LangAS::sycl_global, 1},
    {LangAS::sycl_global_device, 1},
    {LangAS::sycl_global_host, 1},
    {LangAS::sycl_local, 3},
    {LangAS::sycl_private, 0},
    {LangAS::ptr32_sptr, 0},
    {LangAS::ptr32_uptr, 0},
    {LangAS::ptr64, 0},
    {LangAS::hlsl_groupshared, 0},
    {LangAS::hlsl_constant, 0},
    {LangAS::wasm_funcref, 20},
};

NativeCPUTargetInfo::NativeCPUTargetInfo(const llvm::Triple &Triple,
                                         const TargetOptions &Opts)
    : TargetInfo(Triple) {
  AddrSpaceMap = &NativeCPUASMap;
  UseAddrSpaceMapMangling = true;
  HasFastHalfType = true;
  HasFloat16 = true;

  llvm::Triple HostTriple([&] {
    // Take the default target triple if no other host triple is specified so
    // that system headers work.
    if (Opts.HostTriple.empty())
      return llvm::Triple(llvm::sys::getDefaultTargetTriple());

    return llvm::Triple(Opts.HostTriple);
  }());
  if (HostTriple.isNativeCPU()) {
    // This should never happen, just make sure we do not crash.
    resetDataLayout("e");
  } else {
    HostTarget = AllocateTarget(HostTriple, Opts);
  }
}

void NativeCPUTargetInfo::setAuxTarget(const TargetInfo *Aux) {
  assert(Aux && "Cannot invoke setAuxTarget without a valid auxiliary target!");
  copyAuxTarget(Aux);
  getTargetOpts() = Aux->getTargetOpts();
  resetDataLayout(Aux->getDataLayoutString());
}

// A target may initialise its DataLayoutString and potentially other features
// in `handleTargetFeatures` (as opposed to its constructor), so we can only
// copy the features and query DataLayoutString after that function was called.
bool NativeCPUTargetInfo::handleTargetFeatures(
    std::vector<std::string> &Features, DiagnosticsEngine &Diags) {
  if (HostTarget) {
    if (!HostTarget->handleTargetFeatures(Features, Diags))
      return false;
    copyAuxTarget(&*HostTarget);
    resetDataLayout(HostTarget->getDataLayoutString());
  }
  return true;
}
