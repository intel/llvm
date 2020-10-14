//===- SPIRVAsm.h -                                           --*- C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines the inline assembler entries defined in SPIRV spec with op
/// codes.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVASM_H
#define SPIRV_LIBSPIRV_SPIRVASM_H

#include "SPIRVEntry.h"
#include "SPIRVInstruction.h"
#include "SPIRVValue.h"

namespace SPIRV {

class SPIRVAsmTargetINTEL : public SPIRVEntry {
public:
  static const SPIRVWord FixedWC = 2;
  static const Op OC = OpAsmTargetINTEL;
  // Complete constructor
  SPIRVAsmTargetINTEL(SPIRVModule *M, SPIRVId TheId,
                      const std::string &TheTarget)
      : SPIRVEntry(M, FixedWC + getSizeInWords(TheTarget), OC, TheId),
        Target(TheTarget) {
    validate();
  }
  // Incomplete constructor
  SPIRVAsmTargetINTEL() : SPIRVEntry(OC) {}
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilityAsmINTEL);
  }
  llvm::Optional<ExtensionID> getRequiredExtension() const override {
    return ExtensionID::SPV_INTEL_inline_assembly;
  }
  const std::string &getTarget() const { return Target; }

protected:
  void validate() const override {
    SPIRVEntry::validate();
    assert(WordCount > FixedWC);
    assert(OpCode == OC);
  }
  _SPIRV_DEF_ENCDEC2(Id, Target)
  std::string Target;
};

class SPIRVAsmINTEL : public SPIRVValue {
public:
  static const SPIRVWord FixedWC = 5;
  static const Op OC = OpAsmINTEL;
  // Complete constructor
  SPIRVAsmINTEL(SPIRVModule *M, SPIRVTypeFunction *TheFunctionType,
                SPIRVId TheId, SPIRVAsmTargetINTEL *TheTarget,
                const std::string &TheInstructions,
                const std::string &TheConstraints)
      : SPIRVValue(M,
                   FixedWC + getSizeInWords(TheInstructions) +
                       getSizeInWords(TheConstraints),
                   OC, TheFunctionType->getReturnType(), TheId),
        Target(TheTarget), FunctionType(TheFunctionType),
        Instructions(TheInstructions), Constraints(TheConstraints) {
    validate();
  }
  // Incomplete constructor
  SPIRVAsmINTEL() : SPIRVValue(OC) {}
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilityAsmINTEL);
  }
  llvm::Optional<ExtensionID> getRequiredExtension() const override {
    return ExtensionID::SPV_INTEL_inline_assembly;
  }
  const std::string &getInstructions() const { return Instructions; }
  const std::string &getConstraints() const { return Constraints; }
  SPIRVTypeFunction *getFunctionType() const { return FunctionType; }

protected:
  _SPIRV_DEF_ENCDEC6(Type, Id, FunctionType, Target, Instructions, Constraints)
  void validate() const override {
    SPIRVValue::validate();
    assert(WordCount > FixedWC);
    assert(OpCode == OC);
  }
  SPIRVAsmTargetINTEL *Target;
  SPIRVTypeFunction *FunctionType;
  std::string Instructions;
  std::string Constraints;
};

class SPIRVAsmCallINTEL : public SPIRVInstruction {
public:
  static const SPIRVWord FixedWC = 4;
  static const Op OC = OpAsmCallINTEL;
  // Complete constructor
  SPIRVAsmCallINTEL(SPIRVId TheId, SPIRVAsmINTEL *TheAsm,
                    const std::vector<SPIRVWord> &TheArgs,
                    SPIRVBasicBlock *TheBB)
      : SPIRVInstruction(FixedWC + TheArgs.size(), OC, TheAsm->getType(), TheId,
                         TheBB),
        Asm(TheAsm), Args(TheArgs) {
    validate();
  }
  // Incomplete constructor
  SPIRVAsmCallINTEL() : SPIRVInstruction(OC) {}
  SPIRVCapVec getRequiredCapability() const override {
    return getVec(CapabilityAsmINTEL);
  }
  llvm::Optional<ExtensionID> getRequiredExtension() const override {
    return ExtensionID::SPV_INTEL_inline_assembly;
  }
  bool isOperandLiteral(unsigned int Index) const override { return false; }
  void setWordCount(SPIRVWord TheWordCount) override {
    SPIRVEntry::setWordCount(TheWordCount);
    Args.resize(TheWordCount - FixedWC);
  }
  const std::vector<SPIRVWord> &getArguments() const { return Args; }

  SPIRVAsmINTEL *getAsm() const { return Asm; }

protected:
  _SPIRV_DEF_ENCDEC4(Type, Id, Asm, Args)
  void validate() const override {
    SPIRVInstruction::validate();
    assert(WordCount >= FixedWC);
    assert(OpCode == OC);
    assert(getBasicBlock() && "Invalid BB");
    assert(getBasicBlock()->getModule() == Asm->getModule());
  }
  SPIRVAsmINTEL *Asm;
  std::vector<SPIRVWord> Args;
};

} // namespace SPIRV
#endif // SPIRV_LIBSPIRV_SPIRVASM_H
