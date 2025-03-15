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

static const unsigned NativeCPUASMap[] = {
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

    // Copy properties from host target.
    BoolWidth = HostTarget->getBoolWidth();
    BoolAlign = HostTarget->getBoolAlign();
    IntWidth = HostTarget->getIntWidth();
    IntAlign = HostTarget->getIntAlign();
    HalfWidth = HostTarget->getHalfWidth();
    HalfAlign = HostTarget->getHalfAlign();
    FloatWidth = HostTarget->getFloatWidth();
    FloatAlign = HostTarget->getFloatAlign();
    DoubleWidth = HostTarget->getDoubleWidth();
    DoubleAlign = HostTarget->getDoubleAlign();
    LongWidth = HostTarget->getLongWidth();
    LongAlign = HostTarget->getLongAlign();
    LongLongWidth = HostTarget->getLongLongWidth();
    LongLongAlign = HostTarget->getLongLongAlign();
    PointerWidth = HostTarget->getPointerWidth(LangAS::Default);
    PointerAlign = HostTarget->getPointerAlign(LangAS::Default);
    MinGlobalAlign = HostTarget->getMinGlobalAlign(/* TypeSize = */ 0,
                                                   /* HasNonWeakDef = */ true);
    NewAlign = HostTarget->getNewAlign();
    DefaultAlignForAttributeAligned =
        HostTarget->getDefaultAlignForAttributeAligned();
    SizeType = HostTarget->getSizeType();
    PtrDiffType = HostTarget->getPtrDiffType(LangAS::Default);
    IntMaxType = HostTarget->getIntMaxType();
    WCharType = HostTarget->getWCharType();
    WIntType = HostTarget->getWIntType();
    Char16Type = HostTarget->getChar16Type();
    Char32Type = HostTarget->getChar32Type();
    Int64Type = HostTarget->getInt64Type();
    SigAtomicType = HostTarget->getSigAtomicType();
    ProcessIDType = HostTarget->getProcessIDType();

    UseBitFieldTypeAlignment = HostTarget->useBitFieldTypeAlignment();
    UseZeroLengthBitfieldAlignment =
        HostTarget->useZeroLengthBitfieldAlignment();
    UseExplicitBitFieldAlignment = HostTarget->useExplicitBitFieldAlignment();
    ZeroLengthBitfieldBoundary = HostTarget->getZeroLengthBitfieldBoundary();

    // This is a bit of a lie, but it controls __GCC_ATOMIC_XXX_LOCK_FREE, and
    // we need those macros to be identical on host and device, because (among
    // other things) they affect which standard library classes are defined,
    // and we need all classes to be defined on both the host and device.
    MaxAtomicInlineWidth = HostTarget->getMaxAtomicInlineWidth();
  }
}
