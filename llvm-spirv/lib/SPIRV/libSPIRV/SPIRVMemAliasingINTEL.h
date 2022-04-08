//===- SPIRVMemAliasingINTEL.h -                               --*- C++ -*-===//
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
// Neither the names of Intel Corporation, nor the names of its contributors
// may be used to endorse or promote products derived from this Software without
// specific prior written permission.
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
/// This file defines the memory aliasing entries defined in SPIRV spec with op
/// codes.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVMEMALIASINGINTEL_H
#define SPIRV_LIBSPIRV_SPIRVMEMALIASINGINTEL_H

#include "SPIRVEntry.h"

namespace SPIRV {

template <Op TheOpCode, SPIRVWord TheFixedWordCount>
class SPIRVMemAliasingINTELGeneric : public SPIRVEntry {
public:
  SPIRVMemAliasingINTELGeneric(SPIRVModule *TheModule, SPIRVId TheId,
                               const std::vector<SPIRVId> &TheArgs)
      : SPIRVEntry(TheModule, TheArgs.size() + TheFixedWordCount, TheOpCode,
                   TheId), Args(TheArgs) {
    SPIRVMemAliasingINTELGeneric::validate();
    assert(TheModule && "Invalid module");
  }

  SPIRVMemAliasingINTELGeneric() : SPIRVEntry(TheOpCode) {}

  const std::vector<SPIRVId> &getArguments() const { return Args; }

  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
    Args.resize(TheWordCount - TheFixedWordCount);
  }

  void validate() const override { SPIRVEntry::validate(); }

  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilityMemoryAccessAliasingINTEL);
  }

  llvm::Optional<ExtensionID> getRequiredExtension() const override {
    return ExtensionID::SPV_INTEL_memory_access_aliasing;
  }

protected:
  static const SPIRVWord FixedWC = TheFixedWordCount;
  static const Op OC = TheOpCode;
  std::vector<SPIRVId> Args;
  _SPIRV_DEF_ENCDEC2(Id, Args)
};

#define _SPIRV_OP(x, ...)                                                      \
  typedef SPIRVMemAliasingINTELGeneric<Op##x, __VA_ARGS__> SPIRV##x;
// Intel Memory Alasing Instructions
_SPIRV_OP(AliasDomainDeclINTEL, 2)
_SPIRV_OP(AliasScopeDeclINTEL, 2)
_SPIRV_OP(AliasScopeListDeclINTEL, 2)
#undef _SPIRV_OP

} // SPIRV
#endif // SPIRV_LIBSPIRV_SPIRVMEMALIASINGINTEL_H
