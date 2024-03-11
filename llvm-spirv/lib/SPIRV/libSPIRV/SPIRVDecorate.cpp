//===- SPIRVDecorate.cpp -SPIR-V Decorations --------------------*- C++ -*-===//
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
/// This file implements SPIR-V decorations.
///
//===----------------------------------------------------------------------===//

#include "SPIRVDecorate.h"
#include "SPIRVModule.h"
#include "SPIRVStream.h"
#include "SPIRVValue.h"

namespace SPIRV {
template <class T>
spv_ostream &operator<<(spv_ostream &O, const std::vector<T *> &V) {
  for (auto &I : V)
    O << *I;
  return O;
}

SPIRVDecorateGeneric::SPIRVDecorateGeneric(Op OC, SPIRVWord WC,
                                           Decoration TheDec,
                                           SPIRVEntry *TheTarget)
    : SPIRVAnnotationGeneric(TheTarget->getModule(), WC, OC,
                             TheTarget->getId()),
      Dec(TheDec), Owner(nullptr) {
  validate();
  updateModuleVersion();
}

SPIRVDecorateGeneric::SPIRVDecorateGeneric(Op OC, SPIRVWord WC,
                                           Decoration TheDec,
                                           SPIRVEntry *TheTarget, SPIRVWord V)
    : SPIRVAnnotationGeneric(TheTarget->getModule(), WC, OC,
                             TheTarget->getId()),
      Dec(TheDec), Owner(nullptr) {
  Literals.push_back(V);
  validate();
  updateModuleVersion();
}

SPIRVDecorateGeneric::SPIRVDecorateGeneric(Op OC, SPIRVWord WC,
                                           Decoration TheDec,
                                           SPIRVEntry *TheTarget, SPIRVWord V1,
                                           SPIRVWord V2)
    : SPIRVDecorateGeneric(OC, WC, TheDec, TheTarget, V1) {
  Literals.push_back(V2);
  validate();
  updateModuleVersion();
}

SPIRVDecorateGeneric::SPIRVDecorateGeneric(Op OC, SPIRVWord WC,
                                           Decoration TheDec,
                                           SPIRVEntry *TheTarget, SPIRVWord V1,
                                           SPIRVWord V2, SPIRVWord V3)
    : SPIRVDecorateGeneric(OC, WC, TheDec, TheTarget, V1, V2) {
  Literals.push_back(V3);
  validate();
  updateModuleVersion();
}

SPIRVDecorateGeneric::SPIRVDecorateGeneric(Op OC)
    : SPIRVAnnotationGeneric(OC), Dec(DecorationRelaxedPrecision),
      Owner(nullptr) {}

Decoration SPIRVDecorateGeneric::getDecorateKind() const { return Dec; }

SPIRVWord SPIRVDecorateGeneric::getLiteral(size_t I) const {
  assert(I <= Literals.size() && "Out of bounds");
  return Literals[I];
}

std::vector<SPIRVWord> SPIRVDecorateGeneric::getVecLiteral() const {
  return Literals;
}

size_t SPIRVDecorateGeneric::getLiteralCount() const { return Literals.size(); }

void SPIRVDecorate::encode(spv_ostream &O) const {
  SPIRVEncoder Encoder = getEncoder(O);
  Encoder << Target << Dec;
  switch (static_cast<size_t>(Dec)) {
  case DecorationLinkageAttributes:
    SPIRVDecorateLinkageAttr::encodeLiterals(Encoder, Literals);
    break;
  case DecorationMemoryINTEL:
    SPIRVDecorateMemoryINTELAttr::encodeLiterals(Encoder, Literals);
    break;
  case DecorationMergeINTEL:
    SPIRVDecorateMergeINTELAttr::encodeLiterals(Encoder, Literals);
    break;
  case DecorationUserSemantic:
    SPIRVDecorateUserSemanticAttr::encodeLiterals(Encoder, Literals);
    break;
  case internal::DecorationFuncParamDescINTEL:
    SPIRVDecorateFuncParamDescAttr::encodeLiterals(Encoder, Literals);
  case internal::DecorationHostAccessINTEL:
    SPIRVDecorateHostAccessINTELLegacy::encodeLiterals(Encoder, Literals);
    break;
  case DecorationHostAccessINTEL:
    SPIRVDecorateHostAccessINTEL::encodeLiterals(Encoder, Literals);
    break;
  case DecorationInitModeINTEL:
    SPIRVDecorateInitModeINTEL::encodeLiterals(Encoder, Literals);
    break;
  default:
    Encoder << Literals;
  }
}

void SPIRVDecorate::setWordCount(SPIRVWord Count) {
  WordCount = Count;
  Literals.resize(WordCount - FixedWC);
}

void SPIRVDecorate::decode(std::istream &I) {
  SPIRVDecoder Decoder = getDecoder(I);
  Decoder >> Target >> Dec;
  switch (static_cast<size_t>(Dec)) {
  case DecorationLinkageAttributes:
    SPIRVDecorateLinkageAttr::decodeLiterals(Decoder, Literals);
    break;
  case DecorationMemoryINTEL:
    SPIRVDecorateMemoryINTELAttr::decodeLiterals(Decoder, Literals);
    break;
  case DecorationMergeINTEL:
    SPIRVDecorateMergeINTELAttr::decodeLiterals(Decoder, Literals);
    break;
  case DecorationUserSemantic:
    SPIRVDecorateUserSemanticAttr::decodeLiterals(Decoder, Literals);
    break;
  case internal::DecorationFuncParamDescINTEL:
    SPIRVDecorateFuncParamDescAttr::decodeLiterals(Decoder, Literals);
  case internal::DecorationHostAccessINTEL:
    SPIRVDecorateHostAccessINTELLegacy::decodeLiterals(Decoder, Literals);
    break;
  case DecorationHostAccessINTEL:
    SPIRVDecorateHostAccessINTEL::decodeLiterals(Decoder, Literals);
    break;
  default:
    Decoder >> Literals;
  }
  getOrCreateTarget()->addDecorate(this);
}

void SPIRVDecorateId::encode(spv_ostream &O) const {
  SPIRVEncoder Encoder = getEncoder(O);
  Encoder << Target << Dec << Literals;
}

void SPIRVDecorateId::setWordCount(SPIRVWord Count) {
  WordCount = Count;
  Literals.resize(WordCount - FixedWC);
}

void SPIRVDecorateId::decode(std::istream &I) {
  SPIRVDecoder Decoder = getDecoder(I);
  Decoder >> Target >> Dec >> Literals;
  getOrCreateTarget()->addDecorate(this);
}

void SPIRVMemberDecorate::encode(spv_ostream &O) const {
  SPIRVEncoder Encoder = getEncoder(O);
  Encoder << Target << MemberNumber << Dec;
  switch (static_cast<size_t>(Dec)) {
  case DecorationMemoryINTEL:
    SPIRVDecorateMemoryINTELAttr::encodeLiterals(Encoder, Literals);
    break;
  case DecorationMergeINTEL:
    SPIRVDecorateMergeINTELAttr::encodeLiterals(Encoder, Literals);
    break;
  case DecorationUserSemantic:
    SPIRVDecorateUserSemanticAttr::encodeLiterals(Encoder, Literals);
    break;
  case internal::DecorationFuncParamDescINTEL:
    SPIRVDecorateFuncParamDescAttr::encodeLiterals(Encoder, Literals);
    break;
  default:
    Encoder << Literals;
  }
}

void SPIRVMemberDecorate::setWordCount(SPIRVWord Count) {
  WordCount = Count;
  Literals.resize(WordCount - FixedWC);
}

void SPIRVMemberDecorate::decode(std::istream &I) {
  SPIRVDecoder Decoder = getDecoder(I);
  Decoder >> Target >> MemberNumber >> Dec;
  switch (static_cast<size_t>(Dec)) {
  case DecorationMemoryINTEL:
    SPIRVDecorateMemoryINTELAttr::decodeLiterals(Decoder, Literals);
    break;
  case DecorationMergeINTEL:
    SPIRVDecorateMergeINTELAttr::decodeLiterals(Decoder, Literals);
    break;
  case DecorationUserSemantic:
    SPIRVDecorateUserSemanticAttr::decodeLiterals(Decoder, Literals);
    break;
  case internal::DecorationFuncParamDescINTEL:
    SPIRVDecorateFuncParamDescAttr::decodeLiterals(Decoder, Literals);
    break;
  default:
    Decoder >> Literals;
  }
  getOrCreateTarget()->addMemberDecorate(this);
}

void SPIRVDecorationGroup::encode(spv_ostream &O) const { getEncoder(O) << Id; }

void SPIRVDecorationGroup::decode(std::istream &I) {
  getDecoder(I) >> Id;
  Module->addDecorationGroup(this);
}

void SPIRVDecorationGroup::encodeAll(spv_ostream &O) const {
  O << Decorations;
  SPIRVEntry::encodeAll(O);
}

void SPIRVGroupDecorateGeneric::encode(spv_ostream &O) const {
  getEncoder(O) << DecorationGroup << Targets;
}

void SPIRVGroupDecorateGeneric::decode(std::istream &I) {
  getDecoder(I) >> DecorationGroup >> Targets;
  Module->addGroupDecorateGeneric(this);
}

void SPIRVGroupDecorate::decorateTargets() {
  for (auto &I : Targets) {
    auto *Target = getOrCreate(I);
    for (auto &Dec : DecorationGroup->getDecorations()) {
      assert(Dec->isDecorate());
      Target->addDecorate(static_cast<SPIRVDecorate *>(Dec));
    }
  }
}

void SPIRVGroupMemberDecorate::decorateTargets() {
  for (auto &I : Targets) {
    auto *Target = getOrCreate(I);
    for (auto &Dec : DecorationGroup->getDecorations()) {
      assert(Dec->isMemberDecorate());
      Target->addMemberDecorate(static_cast<SPIRVMemberDecorate *>(Dec));
    }
  }
}
} // namespace SPIRV
