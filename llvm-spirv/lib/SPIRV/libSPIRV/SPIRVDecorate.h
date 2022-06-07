//===- SPIRVDecorate.h - SPIR-V Decorations ---------------------*- C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
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
/// This file defines SPIR-V decorations.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVDECORATE_H
#define SPIRV_LIBSPIRV_SPIRVDECORATE_H

#include "SPIRVEntry.h"
#include "SPIRVStream.h"
#include "SPIRVUtil.h"
#include <string>
#include <utility>
#include <vector>

namespace SPIRV {
class SPIRVDecorationGroup;
class SPIRVDecorateGeneric : public SPIRVAnnotationGeneric {
public:
  // Complete constructor for decorations without literals
  SPIRVDecorateGeneric(Op OC, SPIRVWord WC, Decoration TheDec,
                       SPIRVEntry *TheTarget);
  // Complete constructor for decorations with one word literal
  SPIRVDecorateGeneric(Op OC, SPIRVWord WC, Decoration TheDec,
                       SPIRVEntry *TheTarget, SPIRVWord V);
  // Complete constructor for decorations with two word literals
  SPIRVDecorateGeneric(Op OC, SPIRVWord WC, Decoration TheDec,
                       SPIRVEntry *TheTarget, SPIRVWord V1, SPIRVWord V2);
  // Incomplete constructor
  SPIRVDecorateGeneric(Op OC);

  SPIRVWord getLiteral(size_t) const;
  std::vector<SPIRVWord> getVecLiteral() const;
  Decoration getDecorateKind() const;
  size_t getLiteralCount() const;
  SPIRVDecorationGroup *getOwner() const { return Owner; }

  void setOwner(SPIRVDecorationGroup *Owner) { this->Owner = Owner; }

  SPIRVCapVec getRequiredCapability() const override {
    switch (Dec) {
    case DecorationBuiltIn: {
      // Return the BuiltIn's capabilities.
      BuiltIn BI = static_cast<BuiltIn>(Literals.back());
      return getCapability(BI);
    }

    default:
      return getCapability(Dec);
    }
  }

  SPIRVWord getRequiredSPIRVVersion() const override {
    switch (Dec) {
    case DecorationSpecId:
      if (getModule()->hasCapability(CapabilityKernel))
        return static_cast<SPIRVWord>(VersionNumber::SPIRV_1_1);
      else
        return static_cast<SPIRVWord>(VersionNumber::SPIRV_1_0);

    case DecorationMaxByteOffset:
      return static_cast<SPIRVWord>(VersionNumber::SPIRV_1_1);

    default:
      return static_cast<SPIRVWord>(VersionNumber::SPIRV_1_0);
    }
  }

protected:
  Decoration Dec;
  std::vector<SPIRVWord> Literals;
  SPIRVDecorationGroup *Owner; // Owning decorate group
};

typedef std::vector<SPIRVDecorateGeneric *> SPIRVDecorateVec;

class SPIRVDecorate : public SPIRVDecorateGeneric {
public:
  static const Op OC = OpDecorate;
  static const SPIRVWord FixedWC = 3;
  // Complete constructor for decorations without literals
  SPIRVDecorate(Decoration TheDec, SPIRVEntry *TheTarget)
      : SPIRVDecorateGeneric(OC, 3, TheDec, TheTarget) {}
  // Complete constructor for decorations with one word literal
  SPIRVDecorate(Decoration TheDec, SPIRVEntry *TheTarget, SPIRVWord V)
      : SPIRVDecorateGeneric(OC, 4, TheDec, TheTarget, V) {}
  // Complete constructor for decorations with two word literals
  SPIRVDecorate(Decoration TheDec, SPIRVEntry *TheTarget, SPIRVWord V1,
                SPIRVWord V2)
      : SPIRVDecorateGeneric(OC, 5, TheDec, TheTarget, V1, V2) {}
  // Incomplete constructor
  SPIRVDecorate() : SPIRVDecorateGeneric(OC) {}

  llvm::Optional<ExtensionID> getRequiredExtension() const override {
    switch (static_cast<size_t>(Dec)) {
    case DecorationNoSignedWrap:
    case DecorationNoUnsignedWrap:
      return ExtensionID::SPV_KHR_no_integer_wrap_decoration;
    case DecorationRegisterINTEL:
    case DecorationMemoryINTEL:
    case DecorationNumbanksINTEL:
    case DecorationBankwidthINTEL:
    case DecorationMaxPrivateCopiesINTEL:
    case DecorationSinglepumpINTEL:
    case DecorationDoublepumpINTEL:
    case DecorationMaxReplicatesINTEL:
    case DecorationSimpleDualPortINTEL:
    case DecorationMergeINTEL:
    case DecorationBankBitsINTEL:
    case DecorationForcePow2DepthINTEL:
      return ExtensionID::SPV_INTEL_fpga_memory_attributes;
    case DecorationBurstCoalesceINTEL:
    case DecorationCacheSizeINTEL:
    case DecorationDontStaticallyCoalesceINTEL:
    case DecorationPrefetchINTEL:
      return ExtensionID::SPV_INTEL_fpga_memory_accesses;
    case DecorationReferencedIndirectlyINTEL:
    case internal::DecorationArgumentAttributeINTEL:
      return ExtensionID::SPV_INTEL_function_pointers;
    case DecorationIOPipeStorageINTEL:
      return ExtensionID::SPV_INTEL_io_pipes;
    case DecorationBufferLocationINTEL:
      return ExtensionID::SPV_INTEL_fpga_buffer_location;
    case DecorationFunctionFloatingPointModeINTEL:
    case DecorationFunctionRoundingModeINTEL:
    case DecorationFunctionDenormModeINTEL:
      return ExtensionID::SPV_INTEL_float_controls2;
    case DecorationStallEnableINTEL:
      return ExtensionID::SPV_INTEL_fpga_cluster_attributes;
    case DecorationFuseLoopsInFunctionINTEL:
      return ExtensionID::SPV_INTEL_loop_fuse;
    case internal::DecorationCallableFunctionINTEL:
      return ExtensionID::SPV_INTEL_fast_composite;
    case internal::DecorationMathOpDSPModeINTEL:
      return ExtensionID::SPV_INTEL_fpga_dsp_control;
    case internal::DecorationInitiationIntervalINTEL:
      return ExtensionID::SPV_INTEL_fpga_invocation_pipelining_attributes;
    case internal::DecorationMaxConcurrencyINTEL:
      return ExtensionID::SPV_INTEL_fpga_invocation_pipelining_attributes;
    case internal::DecorationPipelineEnableINTEL:
      return ExtensionID::SPV_INTEL_fpga_invocation_pipelining_attributes;
    case internal::DecorationRuntimeAlignedINTEL:
      return ExtensionID::SPV_INTEL_runtime_aligned;
    case internal::DecorationHostAccessINTEL:
    case internal::DecorationInitModeINTEL:
    case internal::DecorationImplementInCSRINTEL:
      return ExtensionID::SPV_INTEL_global_variable_decorations;
    default:
      return {};
    }
  }

  _SPIRV_DCL_ENCDEC
  void setWordCount(SPIRVWord) override;
  void validate() const override {
    SPIRVDecorateGeneric::validate();
    assert(WordCount == Literals.size() + FixedWC);
  }
};

class SPIRVDecorateId : public SPIRVDecorateGeneric {
public:
  static const Op OC = OpDecorateId;
  static const SPIRVWord FixedWC = 3;
  // Complete constructor for decorations with one id operand
  SPIRVDecorateId(Decoration TheDec, SPIRVEntry *TheTarget, SPIRVId V)
      : SPIRVDecorateGeneric(OC, 4, TheDec, TheTarget, V) {}
  // Incomplete constructor
  SPIRVDecorateId() : SPIRVDecorateGeneric(OC) {}

  llvm::Optional<ExtensionID> getRequiredExtension() const override {
    switch (static_cast<int>(Dec)) {
    case DecorationAliasScopeINTEL:
    case DecorationNoAliasINTEL:
      return ExtensionID::SPV_INTEL_memory_access_aliasing;
    default:
      return {};
    }
  }

  _SPIRV_DCL_ENCDEC
  void setWordCount(SPIRVWord) override;
  void validate() const override {
    SPIRVDecorateGeneric::validate();
    assert(WordCount == Literals.size() + FixedWC);
  }
};

class SPIRVDecorateLinkageAttr : public SPIRVDecorate {
public:
  // Complete constructor for LinkageAttributes decorations
  SPIRVDecorateLinkageAttr(SPIRVEntry *TheTarget, const std::string &Name,
                           SPIRVLinkageTypeKind Kind)
      : SPIRVDecorate(DecorationLinkageAttributes, TheTarget) {
    for (auto &I : getVec(Name))
      Literals.push_back(I);
    Literals.push_back(Kind);
    WordCount += Literals.size();
  }
  // Incomplete constructor
  SPIRVDecorateLinkageAttr() : SPIRVDecorate() {}

  std::string getLinkageName() const {
    return getString(Literals.cbegin(), Literals.cend() - 1);
  }
  SPIRVLinkageTypeKind getLinkageType() const {
    return (SPIRVLinkageTypeKind)Literals.back();
  }

  static void encodeLiterals(SPIRVEncoder &Encoder,
                             const std::vector<SPIRVWord> &Literals) {
#ifdef _SPIRV_SUPPORT_TEXT_FMT
    if (SPIRVUseTextFormat) {
      Encoder << getString(Literals.cbegin(), Literals.cend() - 1);
      Encoder.OS << " ";
      Encoder << (SPIRVLinkageTypeKind)Literals.back();
    } else
#endif
      Encoder << Literals;
  }

  static void decodeLiterals(SPIRVDecoder &Decoder,
                             std::vector<SPIRVWord> &Literals) {
#ifdef _SPIRV_SUPPORT_TEXT_FMT
    if (SPIRVUseTextFormat) {
      std::string Name;
      Decoder >> Name;
      SPIRVLinkageTypeKind Kind;
      Decoder >> Kind;
      std::copy_n(getVec(Name).begin(), Literals.size() - 1, Literals.begin());
      Literals.back() = Kind;
    } else
#endif
      Decoder >> Literals;
  }

  llvm::Optional<ExtensionID> getRequiredExtension() const override {
    if (getLinkageType() == SPIRVLinkageTypeKind::LinkageTypeLinkOnceODR)
      return ExtensionID::SPV_KHR_linkonce_odr;
    return {};
  }
};

class SPIRVMemberDecorate : public SPIRVDecorateGeneric {
public:
  static const Op OC = OpMemberDecorate;
  static const SPIRVWord FixedWC = 4;
  // Complete constructor for decorations without literals
  SPIRVMemberDecorate(Decoration TheDec, SPIRVWord Member,
                      SPIRVEntry *TheTarget)
      : SPIRVDecorateGeneric(OC, 4, TheDec, TheTarget), MemberNumber(Member) {}

  // Complete constructor for decorations with one word literal
  SPIRVMemberDecorate(Decoration TheDec, SPIRVWord Member,
                      SPIRVEntry *TheTarget, SPIRVWord V)
      : SPIRVDecorateGeneric(OC, 5, TheDec, TheTarget, V),
        MemberNumber(Member) {}

  // Incomplete constructor
  SPIRVMemberDecorate()
      : SPIRVDecorateGeneric(OC), MemberNumber(SPIRVWORD_MAX) {}

  llvm::Optional<ExtensionID> getRequiredExtension() const override {
    switch (static_cast<size_t>(Dec)) {
    case DecorationRegisterINTEL:
    case DecorationMemoryINTEL:
    case DecorationNumbanksINTEL:
    case DecorationBankwidthINTEL:
    case DecorationMaxPrivateCopiesINTEL:
    case DecorationSinglepumpINTEL:
    case DecorationDoublepumpINTEL:
    case DecorationMaxReplicatesINTEL:
    case DecorationSimpleDualPortINTEL:
    case DecorationMergeINTEL:
    case DecorationBankBitsINTEL:
    case DecorationForcePow2DepthINTEL:
      return ExtensionID::SPV_INTEL_fpga_memory_attributes;
    case DecorationBurstCoalesceINTEL:
    case DecorationCacheSizeINTEL:
    case DecorationDontStaticallyCoalesceINTEL:
    case DecorationPrefetchINTEL:
      return ExtensionID::SPV_INTEL_fpga_memory_accesses;
    case DecorationIOPipeStorageINTEL:
      return ExtensionID::SPV_INTEL_io_pipes;
    case DecorationBufferLocationINTEL:
      return ExtensionID::SPV_INTEL_fpga_buffer_location;
    case internal::DecorationRuntimeAlignedINTEL:
      return ExtensionID::SPV_INTEL_runtime_aligned;
    default:
      return {};
    }
  }

  SPIRVWord getMemberNumber() const { return MemberNumber; }
  std::pair<SPIRVWord, Decoration> getPair() const {
    return std::make_pair(MemberNumber, Dec);
  }

  _SPIRV_DCL_ENCDEC
  void setWordCount(SPIRVWord) override;

  void validate() const override {
    SPIRVDecorateGeneric::validate();
    assert(WordCount == Literals.size() + FixedWC);
  }

protected:
  SPIRVWord MemberNumber;
};

class SPIRVDecorationGroup : public SPIRVEntry {
public:
  static const Op OC = OpDecorationGroup;
  static const SPIRVWord WC = 2;
  // Complete constructor. Does not populate Decorations.
  SPIRVDecorationGroup(SPIRVModule *TheModule, SPIRVId TheId)
      : SPIRVEntry(TheModule, WC, OC, TheId) {
    validate();
  };
  // Incomplete constructor
  SPIRVDecorationGroup() : SPIRVEntry(OC) {}
  void encodeAll(spv_ostream &O) const override;
  _SPIRV_DCL_ENCDEC
  // Move the given decorates to the decoration group
  void takeDecorates(SPIRVDecorateVec &Decs) {
    Decorations = std::move(Decs);
    for (auto &I : Decorations)
      const_cast<SPIRVDecorateGeneric *>(I)->setOwner(this);
    Decs.clear();
  }

  SPIRVDecorateVec &getDecorations() { return Decorations; }

protected:
  SPIRVDecorateVec Decorations;
  void validate() const override {
    assert(OpCode == OC);
    assert(WordCount == WC);
  }
};

class SPIRVGroupDecorateGeneric : public SPIRVEntryNoIdGeneric {
public:
  static const SPIRVWord FixedWC = 2;
  // Complete constructor
  SPIRVGroupDecorateGeneric(Op OC, SPIRVDecorationGroup *TheGroup,
                            const std::vector<SPIRVId> &TheTargets)
      : SPIRVEntryNoIdGeneric(TheGroup->getModule(),
                              FixedWC + TheTargets.size(), OC),
        DecorationGroup(TheGroup), Targets(TheTargets) {}
  // Incomplete constructor
  SPIRVGroupDecorateGeneric(Op OC)
      : SPIRVEntryNoIdGeneric(OC), DecorationGroup(nullptr) {}

  void setWordCount(SPIRVWord WC) override {
    SPIRVEntryNoIdGeneric::setWordCount(WC);
    Targets.resize(WC - FixedWC);
  }
  virtual void decorateTargets() = 0;
  _SPIRV_DCL_ENCDEC
protected:
  SPIRVDecorationGroup *DecorationGroup;
  std::vector<SPIRVId> Targets;
};

class SPIRVGroupDecorate : public SPIRVGroupDecorateGeneric {
public:
  static const Op OC = OpGroupDecorate;
  // Complete constructor
  SPIRVGroupDecorate(SPIRVDecorationGroup *TheGroup,
                     const std::vector<SPIRVId> &TheTargets)
      : SPIRVGroupDecorateGeneric(OC, TheGroup, TheTargets) {}
  // Incomplete constructor
  SPIRVGroupDecorate() : SPIRVGroupDecorateGeneric(OC) {}

  void decorateTargets() override;
};

class SPIRVGroupMemberDecorate : public SPIRVGroupDecorateGeneric {
public:
  static const Op OC = OpGroupMemberDecorate;
  // Complete constructor
  SPIRVGroupMemberDecorate(SPIRVDecorationGroup *TheGroup,
                           const std::vector<SPIRVId> &TheTargets)
      : SPIRVGroupDecorateGeneric(OC, TheGroup, TheTargets) {}
  // Incomplete constructor
  SPIRVGroupMemberDecorate() : SPIRVGroupDecorateGeneric(OC) {}

  void decorateTargets() override;
};

template <Decoration D> class SPIRVDecorateStrAttrBase : public SPIRVDecorate {
public:
  // Complete constructor for decoration with string literal
  SPIRVDecorateStrAttrBase(SPIRVEntry *TheTarget, const std::string &Str)
      : SPIRVDecorate(D, TheTarget) {
    for (auto &I : getVec(Str))
      Literals.push_back(I);
    WordCount += Literals.size();
  }
  // Incomplete constructor
  SPIRVDecorateStrAttrBase() : SPIRVDecorate() {}

  static void encodeLiterals(SPIRVEncoder &Encoder,
                             const std::vector<SPIRVWord> &Literals) {
#ifdef _SPIRV_SUPPORT_TEXT_FMT
    if (SPIRVUseTextFormat) {
      Encoder << getString(Literals.cbegin(), Literals.cend());
    } else
#endif
      Encoder << Literals;
  }

  static void decodeLiterals(SPIRVDecoder &Decoder,
                             std::vector<SPIRVWord> &Literals) {
#ifdef _SPIRV_SUPPORT_TEXT_FMT
    if (SPIRVUseTextFormat) {
      std::string Str;
      Decoder >> Str;
      std::copy_n(getVec(Str).begin(), Literals.size(), Literals.begin());
    } else
#endif
      Decoder >> Literals;
  }
};

class SPIRVDecorateMemoryINTELAttr
    : public SPIRVDecorateStrAttrBase<DecorationMemoryINTEL> {
public:
  // Complete constructor for MemoryINTEL decoration
  SPIRVDecorateMemoryINTELAttr(SPIRVEntry *TheTarget,
                               const std::string &MemoryType)
      : SPIRVDecorateStrAttrBase(TheTarget, MemoryType) {}
};

class SPIRVDecorateUserSemanticAttr
    : public SPIRVDecorateStrAttrBase<DecorationUserSemantic> {
public:
  //  Complete constructor for UserSemantic decoration
  SPIRVDecorateUserSemanticAttr(SPIRVEntry *TheTarget,
                                const std::string &AnnotateString)
      : SPIRVDecorateStrAttrBase(TheTarget, AnnotateString) {}
};

class SPIRVDecorateFuncParamDescAttr
    : public SPIRVDecorateStrAttrBase<internal::DecorationFuncParamDescINTEL> {
public:
  //  Complete constructor for UserSemantic decoration
  SPIRVDecorateFuncParamDescAttr(SPIRVEntry *TheTarget,
                                 const std::string &AnnotateString)
      : SPIRVDecorateStrAttrBase(TheTarget, AnnotateString) {}
};

class SPIRVDecorateMergeINTELAttr : public SPIRVDecorate {
public:
  // Complete constructor for MergeINTEL decoration
  SPIRVDecorateMergeINTELAttr(SPIRVEntry *TheTarget, const std::string &Name,
                              const std::string &Direction)
      : SPIRVDecorate(DecorationMergeINTEL, TheTarget) {
    for (auto &I : getVec(Name))
      Literals.push_back(I);
    for (auto &I : getVec(Direction))
      Literals.push_back(I);
    WordCount += Literals.size();
  }

  static void encodeLiterals(SPIRVEncoder &Encoder,
                             const std::vector<SPIRVWord> &Literals) {
#ifdef _SPIRV_SUPPORT_TEXT_FMT
    if (SPIRVUseTextFormat) {
      std::string FirstString = getString(Literals.cbegin(), Literals.cend());
      Encoder << FirstString;
      Encoder.OS << " ";
      Encoder << getString(Literals.cbegin() + getVec(FirstString).size(),
                           Literals.cend());
    } else
#endif
      Encoder << Literals;
  }

  static void decodeLiterals(SPIRVDecoder &Decoder,
                             std::vector<SPIRVWord> &Literals) {
#ifdef _SPIRV_SUPPORT_TEXT_FMT
    if (SPIRVUseTextFormat) {
      std::string Name;
      Decoder >> Name;
      std::string Direction;
      Decoder >> Direction;
      std::string Buf = Name + ':' + Direction;
      std::copy_n(getVec(Buf).begin(), Literals.size(), Literals.begin());
    } else
#endif
      Decoder >> Literals;
  }
};

class SPIRVDecorateBankBitsINTELAttr : public SPIRVDecorate {
public:
  // Complete constructor for BankBitsINTEL decoration
  SPIRVDecorateBankBitsINTELAttr(SPIRVEntry *TheTarget,
                                 const std::vector<SPIRVWord> &TheBits)
      : SPIRVDecorate(DecorationBankBitsINTEL, TheTarget) {
    Literals = TheBits;
    WordCount += Literals.size();
  }
};

template <Decoration D>
class SPIRVMemberDecorateStrAttrBase : public SPIRVMemberDecorate {
public:
  // Complete constructor for decoration with string literal
  SPIRVMemberDecorateStrAttrBase(SPIRVEntry *TheTarget, SPIRVWord MemberNumber,
                                 const std::string &Str)
      : SPIRVMemberDecorate(D, MemberNumber, TheTarget) {
    for (auto &I : getVec(Str))
      Literals.push_back(I);
    WordCount += Literals.size();
  }
  // Incomplete constructor
  SPIRVMemberDecorateStrAttrBase() : SPIRVMemberDecorate() {}
};

class SPIRVMemberDecorateMemoryINTELAttr
    : public SPIRVMemberDecorateStrAttrBase<DecorationMemoryINTEL> {
public:
  // Complete constructor for MemoryINTEL decoration
  SPIRVMemberDecorateMemoryINTELAttr(SPIRVEntry *TheTarget,
                                     SPIRVWord MemberNumber,
                                     const std::string &MemoryType)
      : SPIRVMemberDecorateStrAttrBase(TheTarget, MemberNumber, MemoryType) {}
};

class SPIRVMemberDecorateUserSemanticAttr
    : public SPIRVMemberDecorateStrAttrBase<DecorationUserSemantic> {
public:
  // Complete constructor for UserSemantic decoration
  SPIRVMemberDecorateUserSemanticAttr(SPIRVEntry *TheTarget,
                                      SPIRVWord MemberNumber,
                                      const std::string &AnnotateString)
      : SPIRVMemberDecorateStrAttrBase(TheTarget, MemberNumber,
                                       AnnotateString) {}
};

class SPIRVMemberDecorateMergeINTELAttr : public SPIRVMemberDecorate {
public:
  // Complete constructor for MergeINTEL decoration
  SPIRVMemberDecorateMergeINTELAttr(SPIRVEntry *TheTarget,
                                    SPIRVWord MemberNumber,
                                    const std::string &Name,
                                    const std::string &Direction)
      : SPIRVMemberDecorate(DecorationMergeINTEL, MemberNumber, TheTarget) {
    for (auto &I : getVec(Name))
      Literals.push_back(I);
    for (auto &I : getVec(Direction))
      Literals.push_back(I);
    WordCount += Literals.size();
  }
};

class SPIRVMemberDecorateBankBitsINTELAttr : public SPIRVMemberDecorate {
public:
  // Complete constructor for BankBitsINTEL decoration
  SPIRVMemberDecorateBankBitsINTELAttr(SPIRVEntry *TheTarget,
                                       SPIRVWord MemberNumber,
                                       const std::vector<SPIRVWord> &TheBits)
      : SPIRVMemberDecorate(DecorationBankBitsINTEL, MemberNumber, TheTarget) {
    Literals = TheBits;
    WordCount += Literals.size();
  }
};

class SPIRVDecorateFunctionRoundingModeINTEL : public SPIRVDecorate {
public:
  // Complete constructor for SPIRVDecorateFunctionRoundingModeINTEL
  SPIRVDecorateFunctionRoundingModeINTEL(SPIRVEntry *TheTarget,
                                         SPIRVWord TargetWidth,
                                         spv::FPRoundingMode FloatControl)
      : SPIRVDecorate(spv::DecorationFunctionRoundingModeINTEL, TheTarget,
                      TargetWidth, static_cast<SPIRVWord>(FloatControl)){};

  SPIRVWord getTargetWidth() const { return Literals.at(0); };
  spv::FPRoundingMode getRoundingMode() const {
    return static_cast<spv::FPRoundingMode>(Literals.at(1));
  };
};

class SPIRVDecorateFunctionDenormModeINTEL : public SPIRVDecorate {
public:
  // Complete constructor for SPIRVDecorateFunctionDenormModeINTEL
  SPIRVDecorateFunctionDenormModeINTEL(SPIRVEntry *TheTarget,
                                       SPIRVWord TargetWidth,
                                       spv::FPDenormMode FloatControl)
      : SPIRVDecorate(spv::DecorationFunctionDenormModeINTEL, TheTarget,
                      TargetWidth, static_cast<SPIRVWord>(FloatControl)){};

  SPIRVWord getTargetWidth() const { return Literals.at(0); };
  spv::FPDenormMode getDenormMode() const {
    return static_cast<spv::FPDenormMode>(Literals.at(1));
  };
};

class SPIRVDecorateFunctionFloatingPointModeINTEL : public SPIRVDecorate {
public:
  // Complete constructor for SPIRVDecorateFunctionOperationModeINTEL
  SPIRVDecorateFunctionFloatingPointModeINTEL(SPIRVEntry *TheTarget,
                                              SPIRVWord TargetWidth,
                                              spv::FPOperationMode FloatControl)
      : SPIRVDecorate(spv::DecorationFunctionFloatingPointModeINTEL, TheTarget,
                      TargetWidth, static_cast<SPIRVWord>(FloatControl)){};

  SPIRVWord getTargetWidth() const { return Literals.at(0); };
  spv::FPOperationMode getOperationMode() const {
    return static_cast<spv::FPOperationMode>(Literals.at(1));
  };
};

class SPIRVDecorateStallEnableINTEL : public SPIRVDecorate {
public:
  // Complete constructor for SPIRVDecorateStallEnableINTEL
  SPIRVDecorateStallEnableINTEL(SPIRVEntry *TheTarget)
      : SPIRVDecorate(spv::DecorationStallEnableINTEL, TheTarget){};
};

class SPIRVDecorateFuseLoopsInFunctionINTEL : public SPIRVDecorate {
public:
  // Complete constructor for SPIRVDecorateFuseLoopsInFunctionINTEL
  SPIRVDecorateFuseLoopsInFunctionINTEL(SPIRVEntry *TheTarget, SPIRVWord Depth,
                                        SPIRVWord Independent)
      : SPIRVDecorate(spv::DecorationFuseLoopsInFunctionINTEL, TheTarget, Depth,
                      Independent){};
};

class SPIRVDecorateMathOpDSPModeINTEL : public SPIRVDecorate {
public:
  // Complete constructor for SPIRVDecorateMathOpDSPModeINTEL
  SPIRVDecorateMathOpDSPModeINTEL(SPIRVEntry *TheTarget, SPIRVWord Mode,
                                  SPIRVWord Propagate)
      : SPIRVDecorate(spv::internal::DecorationMathOpDSPModeINTEL, TheTarget,
                      Mode, Propagate){};
};

class SPIRVDecorateAliasScopeINTEL : public SPIRVDecorateId {
public:
  // Complete constructor for SPIRVDecorateAliasScopeINTEL
  SPIRVDecorateAliasScopeINTEL(SPIRVEntry *TheTarget, SPIRVId AliasList)
      : SPIRVDecorateId(spv::DecorationAliasScopeINTEL, TheTarget, AliasList){};
};

class SPIRVDecorateNoAliasINTEL : public SPIRVDecorateId {
public:
  // Complete constructor for SPIRVDecorateNoAliasINTEL
  SPIRVDecorateNoAliasINTEL(SPIRVEntry *TheTarget, SPIRVId AliasList)
      : SPIRVDecorateId(spv::DecorationNoAliasINTEL, TheTarget, AliasList){};
};

class SPIRVDecorateInitiationIntervalINTEL : public SPIRVDecorate {
public:
  // Complete constructor for SPIRVDecorateInitiationIntervalINTEL
  SPIRVDecorateInitiationIntervalINTEL(SPIRVEntry *TheTarget, SPIRVWord Cycles)
      : SPIRVDecorate(spv::internal::DecorationInitiationIntervalINTEL,
                      TheTarget, Cycles){};
};

class SPIRVDecorateMaxConcurrencyINTEL : public SPIRVDecorate {
public:
  // Complete constructor for SPIRVDecorateMaxConcurrencyINTEL
  SPIRVDecorateMaxConcurrencyINTEL(SPIRVEntry *TheTarget, SPIRVWord Invocations)
      : SPIRVDecorate(spv::internal::DecorationMaxConcurrencyINTEL, TheTarget,
                      Invocations){};
};

class SPIRVDecoratePipelineEnableINTEL : public SPIRVDecorate {
public:
  // Complete constructor for SPIRVDecoratePipelineEnableINTEL
  SPIRVDecoratePipelineEnableINTEL(SPIRVEntry *TheTarget, SPIRVWord Enable)
      : SPIRVDecorate(spv::internal::DecorationPipelineEnableINTEL, TheTarget,
                      Enable){};
};

class SPIRVDecorateHostAccessINTEL : public SPIRVDecorate {
public:
  // Complete constructor for SPIRVHostAccessINTEL
  SPIRVDecorateHostAccessINTEL(SPIRVEntry *TheTarget, SPIRVWord AccessMode,
                               const std::string &VarName)
      : SPIRVDecorate(spv::internal::DecorationHostAccessINTEL, TheTarget) {
    Literals.push_back(AccessMode);
    for (auto &I : getVec(VarName))
      Literals.push_back(I);
    WordCount += Literals.size();
  };

  SPIRVWord getAccessMode() const { return Literals.front(); }
  std::string getVarName() const {
    return getString(Literals.cbegin() + 1, Literals.cend());
  }

  static void encodeLiterals(SPIRVEncoder &Encoder,
                             const std::vector<SPIRVWord> &Literals) {
#ifdef _SPIRV_SUPPORT_TEXT_FMT
    if (SPIRVUseTextFormat) {
      Encoder << Literals.front();
      std::string Name = getString(Literals.cbegin() + 1, Literals.cend());
      Encoder << Name;
    } else
#endif
      Encoder << Literals;
  }

  static void decodeLiterals(SPIRVDecoder &Decoder,
                             std::vector<SPIRVWord> &Literals) {
#ifdef _SPIRV_SUPPORT_TEXT_FMT
    if (SPIRVUseTextFormat) {
      SPIRVWord Mode;
      Decoder >> Mode;
      std::string Name;
      Decoder >> Name;
      Literals.front() = Mode;
      std::copy_n(getVec(Name).begin(), Literals.size() - 1,
                  Literals.begin() + 1);

    } else
#endif
      Decoder >> Literals;
  }
};

class SPIRVDecorateInitModeINTEL : public SPIRVDecorate {
public:
  // Complete constructor for SPIRVInitModeINTEL
  SPIRVDecorateInitModeINTEL(SPIRVEntry *TheTarget, SPIRVWord Trigger)
      : SPIRVDecorate(spv::internal::DecorationInitModeINTEL, TheTarget,
                      Trigger){};
};

class SPIRVDecorateImplementInCSRINTEL : public SPIRVDecorate {
public:
  // Complete constructor for SPIRVImplementInCSRINTEL
  SPIRVDecorateImplementInCSRINTEL(SPIRVEntry *TheTarget, SPIRVWord Value)
      : SPIRVDecorate(spv::internal::DecorationImplementInCSRINTEL, TheTarget,
                      Value){};
};

} // namespace SPIRV

#endif // SPIRV_LIBSPIRV_SPIRVDECORATE_H
