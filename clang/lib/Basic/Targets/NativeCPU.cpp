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
    0,  // Default
    1,  // opencl_global
    3,  // opencl_local
    4,  // opencl_constant
    0,  // opencl_private
    0,  // opencl_generic
    1,  // opencl_global_device
    1,  // opencl_global_host
    1,  // cuda_device
    4,  // cuda_constant
    3,  // cuda_shared
    1,  // sycl_global
    1,  // sycl_global_device
    1,  // sycl_global_host
    3,  // sycl_local
    0,  // sycl_private
    0,  // ptr32_sptr
    0,  // ptr32_uptr
    0,  // ptr64
    0,  // hlsl_groupshared
    0,  // hlsl_constant
    20, // wasm_funcref
};

NativeCPUTargetInfo::NativeCPUTargetInfo(const llvm::Triple &,
                                         const TargetOptions &Opts)
    : TargetInfo(llvm::Triple()) {
  AddrSpaceMap = &NativeCPUASMap;
  UseAddrSpaceMapMangling = true;
  HasLegalHalfType = true;
  HasFloat16 = true;
  resetDataLayout("e");

  llvm::Triple HostTriple([&] {
    // Take the default target triple if no other host triple is specified so
    // that system headers work.
    if (Opts.HostTriple.empty())
      return llvm::sys::getDefaultTargetTriple();

    return Opts.HostTriple;
  }());
  if (HostTriple.getArch() != llvm::Triple::UnknownArch) {
    HostTarget = AllocateTarget(HostTriple, Opts);
    copyAuxTarget(&*HostTarget);
  }
}

void NativeCPUTargetInfo::setAuxTarget(const TargetInfo *Aux) {
  assert(Aux && "Cannot invoke setAuxTarget without a valid auxiliary target!");
  copyAuxTarget(Aux);
  getTargetOpts() = Aux->getTargetOpts();
}
