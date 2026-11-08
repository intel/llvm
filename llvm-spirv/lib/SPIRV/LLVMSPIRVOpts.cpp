//===- LLVMSPIRVOpts.cpp - Defines LLVM/SPIR-V options ----------*- C++ -*-===//
//
//                     The LLVM/SPIR-V Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2021 Intel Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides definitions for LLVM/SPIR-V Translator's CLI
/// functionality.
///
//===----------------------------------------------------------------------===//

#include "LLVMSPIRVOpts.h"

#include "SPIRVEnum.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/Support/AMDGPUAddrSpace.h>
#include <llvm/TargetParser/Triple.h>
#include <optional>

using namespace llvm;
using namespace SPIRV;

namespace {
// AddrSpaceMap is a positional std::array; Map is indexed by SPIRAS_*.
struct TargetAddrSpaceMapping {
  Triple::ArchType Arch;
  AddrSpaceMap Map;
  // Pinned so getFunctionProgramAddrSpace() does not fall back to the private
  // AS and emit a spurious -P<n> in the datalayout.
  uint32_t ProgramAS;
};

constexpr TargetAddrSpaceMapping BuiltinAddrSpaceMaps[] = {
    {Triple::amdgpu,
     {
         AMDGPUAS::PRIVATE_ADDRESS,  // SPIRAS_Private
         AMDGPUAS::GLOBAL_ADDRESS,   // SPIRAS_Global
         AMDGPUAS::CONSTANT_ADDRESS, // SPIRAS_Constant
         AMDGPUAS::LOCAL_ADDRESS,    // SPIRAS_Local
         AMDGPUAS::FLAT_ADDRESS,     // SPIRAS_Generic
         AMDGPUAS::GLOBAL_ADDRESS,   // SPIRAS_GlobalDevice
         AMDGPUAS::GLOBAL_ADDRESS,   // SPIRAS_GlobalHost
         // Keep identity (7, 8). For AMDGPU these are the buffer fat/resource
         // pointers. Globals in these address spaces are rejected by the
         // backend, so any StorageClassInput variable fails
         // codegen under an AMDGPU triple. Kept for parity with the ROCm map
         // (see
         // https://github.com/ROCm/SPIRV-LLVM-Translator/blob/f8f6c81b476a33cf06b2e73c493eb8c99f5306af/lib/SPIRV/OCLUtil.h#L508)
         AMDGPUAS::BUFFER_FAT_POINTER, // SPIRAS_Input
         AMDGPUAS::BUFFER_RESOURCE,    // SPIRAS_Output
         // SPIRAS_CodeSectionINTEL and BUFFER_STRIDED_POINTER share ID (9).
         // Leaving SPIRAS_CodeSectionINTEL unmapped would land function
         // pointers on a strided buffer pointer, which is incorrect. Land them
         // on FLAT instead.
         AMDGPUAS::FLAT_ADDRESS, // SPIRAS_CodeSectionINTEL
     },
     AMDGPUAS::FLAT_ADDRESS},
};
} // namespace

bool TranslatorOpts::deriveTargetAddrSpaces() {
  // An already-installed map wins: an explicit --spirv-addrspace-map, or a
  // prior derivation.
  if (getAddrSpaceMap())
    return true;
  Triple TT(Triple::normalize(getSPIRVTargetTriple()));
  for (const auto &Entry : BuiltinAddrSpaceMaps) {
    if (TT.getArch() != Entry.Arch)
      continue;
    setAddrSpaceMap(Entry.Map);
    // An explicit --spirv-function-program-addrspace wins.
    if (!FunctionProgramAS.has_value())
      setFunctionProgramAddrSpace(Entry.ProgramAS);
    return true;
  }
  // SPIR/SPIRV: default is correct. Any other valid target lacking a map: error
  // rather than emit SPIR numbering under it. UnknownArch is explicitly allowed
  // here because it will error out later with a better, more explicit error.
  return TT.isSPIR() || TT.isSPIRV() || TT.getArch() == Triple::UnknownArch;
}

void TranslatorOpts::enableAllExtensions() {
#define EXT(X) ExtStatusMap[ExtensionID::X] = true;
#include "LLVMSPIRVExtensions.inc"
#undef EXT
}

bool TranslatorOpts::isUnknownIntrinsicAllowed(
    IntrinsicInst *II) const noexcept {
  if (!SPIRVAllowUnknownIntrinsics.has_value())
    return false;
  const auto &IntrinsicPrefixList = SPIRVAllowUnknownIntrinsics.value();
  StringRef IntrinsicName = II->getCalledOperand()->getName();
  for (const auto &Prefix : IntrinsicPrefixList) {
    if (IntrinsicName.starts_with(Prefix)) // Also true if `Prefix` is empty
      return true;
  }
  return false;
}

bool TranslatorOpts::isSPIRVAllowUnknownIntrinsicsEnabled() const noexcept {
  return SPIRVAllowUnknownIntrinsics.has_value();
}

void TranslatorOpts::setSPIRVAllowUnknownIntrinsics(
    TranslatorOpts::ArgList IntrinsicPrefixList) noexcept {
  SPIRVAllowUnknownIntrinsics = IntrinsicPrefixList;
}

std::vector<std::string> TranslatorOpts::getAllowedSPIRVExtensionNames(
    std::function<bool(SPIRV::ExtensionID)> &Filter) const {
  std::vector<std::string> AllowExtNames;
  AllowExtNames.reserve(ExtStatusMap.size());
  for (const auto &It : ExtStatusMap) {
    if (!It.second || !Filter(It.first))
      continue;
    std::string ExtName;
    SPIRVMap<ExtensionID, std::string>::find(It.first, &ExtName);
    AllowExtNames.emplace_back(ExtName);
  }
  return AllowExtNames;
}

bool TranslatorOpts::validateFnVarOpts() const {
  if (getFnVarCategory() == std::nullopt &&
      (getFnVarFamily() != std::nullopt || getFnVarArch() != std::nullopt)) {
    errs() << "FnVar: Device category must be specified if the family or "
              "architecture are specified.";
    return false;
  }

  if (getFnVarFamily() == std::nullopt && getFnVarArch() != std::nullopt) {
    errs() << "FnVar: Device family must be specified if the architecture is "
              "specified.";
    return false;
  }

  if (getFnVarTarget() == std::nullopt && !getFnVarFeatures().empty()) {
    errs() << "Device target must be specified if the features are specified.";
    return false;
  }

  return true;
}
