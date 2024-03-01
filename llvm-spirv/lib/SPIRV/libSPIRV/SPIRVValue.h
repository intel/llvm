//===- SPIRVValue.h - Class to represent a SPIR-V Value ---------*- C++ -*-===//
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
/// This file defines the values defined in SPIR-V spec with op codes.
///
/// The name of the SPIR-V values follow the op code name in the spec.
/// This is for readability and ease of using macro to handle types.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVVALUE_H
#define SPIRV_LIBSPIRV_SPIRVVALUE_H

#include "SPIRVDecorate.h"
#include "SPIRVEntry.h"
#include "SPIRVType.h"

namespace llvm {
class APInt;
} // namespace llvm

#include <iostream>

namespace SPIRV {

class SPIRVValue : public SPIRVEntry {
public:
  // Complete constructor for value with id and type
  SPIRVValue(SPIRVModule *M, unsigned TheWordCount, Op TheOpCode,
             SPIRVType *TheType, SPIRVId TheId)
      : SPIRVEntry(M, TheWordCount, TheOpCode, TheId), Type(TheType) {
    validate();
  }
  // Complete constructor for value with type but without id
  SPIRVValue(SPIRVModule *M, unsigned TheWordCount, Op TheOpCode,
             SPIRVType *TheType)
      : SPIRVEntry(M, TheWordCount, TheOpCode), Type(TheType) {
    setHasNoId();
    SPIRVValue::validate();
  }
  // Complete constructor for value with id but without type
  SPIRVValue(SPIRVModule *M, unsigned TheWordCount, Op TheOpCode, SPIRVId TheId)
      : SPIRVEntry(M, TheWordCount, TheOpCode, TheId), Type(NULL) {
    setHasNoType();
    SPIRVValue::validate();
  }
  // Complete constructor for value without id and type
  SPIRVValue(SPIRVModule *M, unsigned TheWordCount, Op TheOpCode)
      : SPIRVEntry(M, TheWordCount, TheOpCode), Type(NULL) {
    setHasNoId();
    setHasNoType();
    SPIRVValue::validate();
  }
  // Incomplete constructor
  SPIRVValue(Op TheOpCode) : SPIRVEntry(TheOpCode), Type(NULL) {}

  bool hasType() const { return !(Attrib & SPIRVEA_NOTYPE); }
  SPIRVType *getType() const {
    assert(hasType() && "value has no type");
    return Type;
  }
  bool isVolatile() const;
  bool hasAlignment(SPIRVWord *Result = 0) const;
  bool hasNoSignedWrap() const;
  bool hasNoUnsignedWrap() const;

  void setAlignment(SPIRVWord);
  void setVolatile(bool IsVolatile);

  template <spv::Decoration NoIntegerWrapDecoration>
  void setNoIntegerDecorationWrap(bool HasNoIntegerWrap);

  void setFPFastMathMode(SPIRVWord FPFastMathMode);

  void validate() const override {
    SPIRVEntry::validate();
    assert((!hasType() || Type) && "Invalid type");
  }

  void setType(SPIRVType *Ty) {
    Type = Ty;
    assert(!Ty || !Ty->isTypeVoid() || OpCode == OpFunction);
    if (Ty && (!Ty->isTypeVoid() || OpCode == OpFunction))
      setHasType();
    else
      setHasNoType();
  }

  SPIRVCapVec getRequiredCapability() const override {
    SPIRVCapVec CV;
    if (!hasType())
      return CV;
    return Type->getRequiredCapability();
  }

  std::optional<ExtensionID> getRequiredExtension() const override {
    std::optional<ExtensionID> EV;
    if (!hasType())
      return EV;
    EV = Type->getRequiredExtension();
    assert(Module &&
           (!EV.has_value() || Module->isAllowedToUseExtension(EV.value())));
    return EV;
  }

protected:
  void setHasNoType() { Attrib |= SPIRVEA_NOTYPE; }
  void setHasType() { Attrib &= ~SPIRVEA_NOTYPE; }

  SPIRVType *Type; // Value Type
};

template <spv::Op OC> class SPIRVConstantBase : public SPIRVValue {
public:
  // Complete constructor for integer constant
  SPIRVConstantBase(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId,
                    uint64_t TheValue)
      : SPIRVValue(M, 0, OC, TheType, TheId) {
    setWords(&TheValue);
  }
  // Incomplete constructor for AP integer constant
  SPIRVConstantBase(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId,
                    const llvm::APInt &TheValue);
  // Complete constructor for float constant
  SPIRVConstantBase(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId,
                    float TheValue)
      : SPIRVValue(M, 0, OC, TheType, TheId) {
    setWords(reinterpret_cast<uint64_t *>(&TheValue));
  }
  // Complete constructor for double constant
  SPIRVConstantBase(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId,
                    double TheValue)
      : SPIRVValue(M, 0, OC, TheType, TheId) {
    setWords(reinterpret_cast<uint64_t *>(&TheValue));
  }
  // Incomplete constructor
  SPIRVConstantBase() : SPIRVValue(OC), NumWords(0) {}
  uint64_t getZExtIntValue() const { return getValue<uint64_t>(); }
  float getFloatValue() const { return getValue<float>(); }
  double getDoubleValue() const { return getValue<double>(); }
  unsigned getNumWords() const { return NumWords; }
  const std::vector<SPIRVWord> &getSPIRVWords() { return Words; }

protected:
  constexpr static SPIRVWord FixedWC = 3;

  // Common method for getting values of size less or equal to 64 bits.
  template <typename T> T getValue() const {
    constexpr auto ValueSize = static_cast<unsigned>(sizeof(T));
    assert((ValueSize <= 8) && "Incorrect result type of requested value");
    T TheValue{};
    unsigned CopyBytes = std::min(ValueSize, NumWords * SpirvWordSize);
    std::memcpy(&TheValue, Words.data(), CopyBytes);
    return TheValue;
  }

  void setWords(const uint64_t *TheValue);
  void recalculateWordCount() {
    NumWords =
        (Type->getBitWidth() + SpirvWordBitWidth - 1) / SpirvWordBitWidth;
    WordCount = FixedWC + NumWords;
  }
  void validate() const override {
    SPIRVValue::validate();
    assert(NumWords >= 1 && "Invalid constant size");
  }
  void encode(spv_ostream &O) const override {
    getEncoder(O) << Type << Id;
    for (const auto &Word : Words)
      getEncoder(O) << Word;
  }
  void setWordCount(SPIRVWord WordCount) override {
    SPIRVValue::setWordCount(WordCount);
    NumWords = WordCount - FixedWC;
  }
  void decode(std::istream &I) override {
    getDecoder(I) >> Type >> Id;
    Words.resize(NumWords);
    for (auto &Word : Words)
      getDecoder(I) >> Word;
  }

  unsigned NumWords;

private:
  std::vector<SPIRVWord> Words;
};

using SPIRVConstant = SPIRVConstantBase<OpConstant>;
using SPIRVSpecConstant = SPIRVConstantBase<OpSpecConstant>;

template <Op OC> class SPIRVConstantEmpty : public SPIRVValue {
public:
  // Complete constructor
  SPIRVConstantEmpty(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId)
      : SPIRVValue(M, 3, OC, TheType, TheId) {
    validate();
  }
  // Incomplete constructor
  SPIRVConstantEmpty() : SPIRVValue(OC) {}

protected:
  void validate() const override { SPIRVValue::validate(); }
  _SPIRV_DEF_ENCDEC2(Type, Id)
};

template <Op OC> class SPIRVConstantBool : public SPIRVConstantEmpty<OC> {
public:
  // Complete constructor
  SPIRVConstantBool(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId)
      : SPIRVConstantEmpty<OC>(M, TheType, TheId) {}
  // Incomplete constructor
  SPIRVConstantBool() {}

protected:
  void validate() const override {
    SPIRVConstantEmpty<OC>::validate();
    assert(this->Type->isTypeBool() && "Invalid type");
  }
};

typedef SPIRVConstantBool<OpConstantTrue> SPIRVConstantTrue;
typedef SPIRVConstantBool<OpConstantFalse> SPIRVConstantFalse;
typedef SPIRVConstantBool<OpSpecConstantTrue> SPIRVSpecConstantTrue;
typedef SPIRVConstantBool<OpSpecConstantFalse> SPIRVSpecConstantFalse;

class SPIRVConstantNull : public SPIRVConstantEmpty<OpConstantNull> {
public:
  // Complete constructor
  SPIRVConstantNull(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId)
      : SPIRVConstantEmpty(M, TheType, TheId) {
    validate();
  }
  // Incomplete constructor
  SPIRVConstantNull() {}

protected:
  void validate() const override {
    SPIRVConstantEmpty::validate();
    assert((Type->isTypeBool() || Type->isTypeInt() || Type->isTypeFloat() ||
            Type->isTypeComposite() || Type->isTypeOpaque() ||
            Type->isTypeEvent() || Type->isTypePointer() ||
            Type->isTypeReserveId() || Type->isTypeDeviceEvent() ||
            (Type->isTypeSubgroupAvcINTEL() &&
             !Type->isTypeSubgroupAvcMceINTEL())) &&
           "Invalid type");
  }
};

class SPIRVUndef : public SPIRVConstantEmpty<OpUndef> {
public:
  // Complete constructor
  SPIRVUndef(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId)
      : SPIRVConstantEmpty(M, TheType, TheId) {
    validate();
  }
  // Incomplete constructor
  SPIRVUndef() {}

protected:
  void validate() const override { SPIRVConstantEmpty::validate(); }
};

template <spv::Op OC> class SPIRVConstantCompositeBase : public SPIRVValue {
public:
  // There are always 3 words in this instruction except constituents:
  // 1) WordCount + OpCode
  // 2) Result type
  // 3) Result Id
  constexpr static SPIRVWord FixedWC = 3;
  using ContinuedInstType = typename InstToContinued<OC>::Type;
  // Complete constructor for composite constant
  SPIRVConstantCompositeBase(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId,
                             const std::vector<SPIRVValue *> TheElements)
      : SPIRVValue(M, TheElements.size() + FixedWC, OC, TheType, TheId) {
    Elements = getIds(TheElements);
    validate();
  }
  // Incomplete constructor
  SPIRVConstantCompositeBase() : SPIRVValue(OC) {}
  std::vector<SPIRVValue *> getElements() const { return getValues(Elements); }

  // TODO: Should we attach operands of continued instructions as well?
  std::vector<SPIRVEntry *> getNonLiteralOperands() const override {
    std::vector<SPIRVValue *> Elements = getElements();
    return std::vector<SPIRVEntry *>(Elements.begin(), Elements.end());
  }

  std::vector<ContinuedInstType> getContinuedInstructions() {
    return ContinuedInstructions;
  }

  void addContinuedInstruction(ContinuedInstType Inst) {
    ContinuedInstructions.push_back(Inst);
  }

  void encodeChildren(spv_ostream &O) const override {
    O << SPIRVNL();
    for (auto &I : ContinuedInstructions)
      O << *I;
  }

protected:
  void validate() const override {
    SPIRVValue::validate();
    for (auto &I : Elements)
      getValue(I)->validate();
  }

  void setWordCount(SPIRVWord WordCount) override {
    SPIRVEntry::setWordCount(WordCount);
    Elements.resize(WordCount - FixedWC);
  }

  void encode(spv_ostream &O) const override {
    getEncoder(O) << Type << Id << Elements;
  }

  void decode(std::istream &I) override {
    SPIRVDecoder Decoder = getDecoder(I);
    Decoder >> Type >> Id >> Elements;

    for (SPIRVEntry *E : Decoder.getContinuedInstructions(ContinuedOpCode)) {
      addContinuedInstruction(static_cast<ContinuedInstType>(E));
    }
  }

  std::vector<SPIRVId> Elements;
  std::vector<ContinuedInstType> ContinuedInstructions;
  const spv::Op ContinuedOpCode = InstToContinued<OC>::OpCode;
};

using SPIRVConstantComposite = SPIRVConstantCompositeBase<OpConstantComposite>;
using SPIRVSpecConstantComposite =
    SPIRVConstantCompositeBase<OpSpecConstantComposite>;

class SPIRVConstantSampler : public SPIRVValue {
public:
  const static Op OC = OpConstantSampler;
  const static SPIRVWord WC = 6;
  // Complete constructor
  SPIRVConstantSampler(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId,
                       SPIRVWord TheAddrMode, SPIRVWord TheNormalized,
                       SPIRVWord TheFilterMode)
      : SPIRVValue(M, WC, OC, TheType, TheId), AddrMode(TheAddrMode),
        Normalized(TheNormalized), FilterMode(TheFilterMode) {
    validate();
  }
  // Incomplete constructor
  SPIRVConstantSampler()
      : SPIRVValue(OC), AddrMode(SPIRVSAM_Invalid), Normalized(SPIRVWORD_MAX),
        FilterMode(SPIRVSFM_Invalid) {}

  SPIRVWord getAddrMode() const { return AddrMode; }

  SPIRVWord getFilterMode() const { return FilterMode; }

  SPIRVWord getNormalized() const { return Normalized; }
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilityLiteralSampler);
  }

protected:
  SPIRVWord AddrMode;
  SPIRVWord Normalized;
  SPIRVWord FilterMode;
  void validate() const override {
    SPIRVValue::validate();
    assert(OpCode == OC);
    assert(WordCount == WC);
    assert(Type->isTypeSampler());
  }
  _SPIRV_DEF_ENCDEC5(Type, Id, AddrMode, Normalized, FilterMode)
};

class SPIRVConstantPipeStorage : public SPIRVValue {
public:
  const static Op OC = OpConstantPipeStorage;
  const static SPIRVWord WC = 6;
  // Complete constructor
  SPIRVConstantPipeStorage(SPIRVModule *M, SPIRVType *TheType, SPIRVId TheId,
                           SPIRVWord ThePacketSize, SPIRVWord ThePacketAlign,
                           SPIRVWord TheCapacity)
      : SPIRVValue(M, WC, OC, TheType, TheId), PacketSize(ThePacketSize),
        PacketAlign(ThePacketAlign), Capacity(TheCapacity) {
    validate();
  }
  // Incomplete constructor
  SPIRVConstantPipeStorage()
      : SPIRVValue(OC), PacketSize(0), PacketAlign(0), Capacity(0) {}

  SPIRVWord getPacketSize() const { return PacketSize; }

  SPIRVWord getPacketAlign() const { return PacketAlign; }

  SPIRVWord getCapacity() const { return Capacity; }
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilityPipes, CapabilityPipeStorage);
  }

protected:
  SPIRVWord PacketSize;
  SPIRVWord PacketAlign;
  SPIRVWord Capacity;
  void validate() const override {
    SPIRVValue::validate();
    assert(OpCode == OC);
    assert(WordCount == WC);
    assert(Type->isTypePipeStorage());
  }
  _SPIRV_DEF_ENCDEC5(Type, Id, PacketSize, PacketAlign, Capacity)
};

class SPIRVForward : public SPIRVValue, public SPIRVComponentExecutionModes {
public:
  const static Op OC = internal::OpForward;
  // Complete constructor
  SPIRVForward(SPIRVModule *TheModule, SPIRVType *TheTy, SPIRVId TheId)
      : SPIRVValue(TheModule, 0, OC, TheId) {
    if (TheTy)
      setType(TheTy);
  }
  SPIRVForward() : SPIRVValue(OC) { assert(0 && "should never be called"); }
  _SPIRV_DEF_ENCDEC1(Id)
  friend class SPIRVFunction;

protected:
  void validate() const override {}
};

} // namespace SPIRV

#endif // SPIRV_LIBSPIRV_SPIRVVALUE_H
