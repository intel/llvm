//===- SPIRVFnVar.cpp                                                     -===//
//
//                     The LLVM/SPIRV Translator
//
// Copyright (c) 2025 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements functions declared in its header file with the help of
/// additional helper functions.
///
//===----------------------------------------------------------------------===//

#include "SPIRVFnVar.h"

using namespace SPIRV;

namespace {

// Replace SPIRV value with OpConstantTrue/False based on a boolean value.
void replaceWithBoolConst(SPIRVModule *BM, SPIRVValue *&OldVal, bool Val) {
  auto *NewVal =
      Val ? static_cast<SPIRVValue *>(
                new SPIRVConstantTrue(BM, OldVal->getType(), OldVal->getId()))
          : static_cast<SPIRVValue *>(
                new SPIRVConstantFalse(BM, OldVal->getType(), OldVal->getId()));
  [[maybe_unused]] bool IsSuccess = BM->eraseValue(OldVal);
  assert(IsSuccess);
  OldVal = BM->addConstant(NewVal);
}

// Evaluate a constant pointed to by 'Id' and store the result in 'Res'.
// Evaluated instruction is replaced with OpConstantTrue/False depending on the
// result.
//
// To keep this simple, only boolean constants and a subset of OpSpecConstantOp
// operations are allowed. Support for more can be implemented as necessary.
bool evaluateConstant(SPIRVModule *BM, SPIRVId Id, bool &Res,
                      std::string &ErrMsg) {
  auto *BV = BM->getValue(Id);
  const Op OpCode = BV->getOpCode();

  assert(isConstantOpCode(OpCode));
  assert(BV->getType()->getOpCode() == spv::OpTypeBool);

  SPIRVWord SpecId = 0;
  if (BV->hasDecorate(DecorationSpecId, 0, &SpecId)) {
    if (OpCode != OpSpecConstantTrue && OpCode != OpSpecConstantFalse &&
        OpCode != OpSpecConstantArchitectureINTEL &&
        OpCode != OpSpecConstantTargetINTEL &&
        OpCode != OpSpecConstantCapabilitiesINTEL) {
      ErrMsg = "Setting only boolean spec constants is supported";
      return false;
    }

    bool IsTrue = OpCode == OpSpecConstantTrue;
    uint64_t ConstValue = 0;
    if (BM->getSpecializationConstant(SpecId, ConstValue)) {
      IsTrue = ConstValue;
    }
    Res = IsTrue;
    replaceWithBoolConst(BM, BV, Res);
    return true;
  }

  switch (OpCode) {
  case OpConstantTrue: {
    Res = true;
    break;
  }
  case OpConstantFalse: {
    Res = false;
    break;
  }
  case OpSpecConstantTrue: {
    Res = true;
    replaceWithBoolConst(BM, BV, Res);
    break;
  }
  case OpSpecConstantFalse: {
    Res = false;
    replaceWithBoolConst(BM, BV, Res);
    break;
  }
  case OpSpecConstantArchitectureINTEL: {
    Res =
        static_cast<SPIRVSpecConstantArchitectureINTEL *>(BV)->matchesDevice();
    replaceWithBoolConst(BM, BV, Res);
    break;
  }
  case OpSpecConstantTargetINTEL: {
    Res = static_cast<SPIRVSpecConstantTargetINTEL *>(BV)->matchesDevice();
    replaceWithBoolConst(BM, BV, Res);
    break;
  }
  case OpSpecConstantCapabilitiesINTEL: {
    Res =
        static_cast<SPIRVSpecConstantCapabilitiesINTEL *>(BV)->matchesDevice();
    replaceWithBoolConst(BM, BV, Res);
    break;
  }
  case OpSpecConstantOp: {
    auto OpWords = static_cast<SPIRVSpecConstantOp *>(BV)->getOpWords();
    auto OpOpCode = static_cast<Op>(OpWords[0]);
    if (OpOpCode != OpLogicalOr && OpOpCode != OpLogicalAnd &&
        OpOpCode != OpLogicalNot) {
      ErrMsg = "Unsupported operation: Only OpLogicalOr/And/Not are allowed.";
      return false;
    }

    bool Val1 = false;
    if (!evaluateConstant(BM, OpWords[1], Val1, ErrMsg)) {
      return false;
    }

    if (OpOpCode == OpLogicalNot) {
      assert(OpWords.size() == 2);
      if (OpOpCode == OpLogicalNot) {
        Res = !Val1;
      }
    } else {
      assert(OpWords.size() == 3);
      bool Val2 = false;
      if (!evaluateConstant(BM, OpWords[2], Val2, ErrMsg)) {
        return false;
      }

      if (OpOpCode == OpLogicalOr) {
        Res = Val1 || Val2;
      } else if (OpOpCode == OpLogicalAnd) {
        Res = Val1 && Val2;
      }
    }

    replaceWithBoolConst(BM, BV, Res);
    break;
  }
  default: {
    std::ostringstream S;
    S << "Evaluating unsupported instruction, opcode: " << OpCode;
    ErrMsg = S.str();
    return false;
  }
  }

  return true;
}

} // anonymous namespace

namespace SPIRV {

bool specializeFnVariants(SPIRVModule *BM, std::string &ErrMsg) {
  // Specialize conditional capabilities
  std::vector<std::pair<std::pair<SPIRVId, Capability>, bool>> CondCapabilities;
  for (const auto &CondCap : BM->getConditionalCapabilities()) {
    const SPIRVId Condition = CondCap.first.first;
    const Capability Cap = CondCap.first.second;
    const SPIRVConditionalCapabilityINTEL *Entry = CondCap.second.get();
    bool ShouldKeep = false;
    if (!evaluateConstant(BM, Entry->getCondition(), ShouldKeep, ErrMsg)) {
      return false;
    }
    CondCapabilities.emplace_back(
        std::make_pair(std::make_pair(Condition, Cap), ShouldKeep));
  }

  for (const auto &CondCap : CondCapabilities) {
    const SPIRVId Condition = CondCap.first.first;
    const Capability Cap = CondCap.first.second;
    const bool ShouldKeep = CondCap.second;
    if (ShouldKeep) {
      BM->addCapability(Cap);
    } else {
      // In case the capability was auto-added by other instruction
      BM->eraseCapability(Cap);
    }
    BM->eraseConditionalCapability(Condition, Cap);
  }

  // Specialize conditional extensions
  std::vector<std::pair<std::pair<uint32_t, std::string>, bool>> CondExtensions;
  for (const auto &CondExt : BM->getConditionalExtensions()) {
    const SPIRVId Cond = CondExt.first;
    const std::string Ext = CondExt.second;
    bool ShouldKeep = false;
    if (!evaluateConstant(BM, Cond, ShouldKeep, ErrMsg)) {
      return false;
    }
    CondExtensions.emplace_back(
        std::make_pair(std::make_pair(Cond, Ext), ShouldKeep));
  }

  for (const auto &CondExt : CondExtensions) {
    const auto &Ext = CondExt.first;
    const bool ShouldKeep = CondExt.second;
    if (ShouldKeep) {
      BM->getExtension().insert(Ext.second);
    } else {
      // In case the extension was auto-added by other instruction
      BM->getExtension().erase(Ext.second);
    }
    BM->getConditionalExtensions().erase(Ext);
  }

  // Specialize conditional entry points
  std::vector<std::pair<SPIRVId, bool>> CondEPs;
  for (const auto &CondEP : BM->getConditionalEntryPoints()) {
    const SPIRVId Cond = CondEP->getCondition();
    bool ShouldKeep = false;
    if (!evaluateConstant(BM, Cond, ShouldKeep, ErrMsg)) {
      return false;
    }
    CondEPs.emplace_back(std::make_pair(Cond, ShouldKeep));
  }

  for (const auto &CondEP : CondEPs) {
    const SPIRVId Cond = CondEP.first;
    const bool ShouldKeep = CondEP.second;
    BM->specializeConditionalEntryPoints(Cond, ShouldKeep);
  }

  // Specialize conditional copy object
  std::vector<std::pair<SPIRVInstruction *, SPIRVId>> ToReplace;
  for (unsigned IF = 0; IF < BM->getNumFunctions(); ++IF) {
    const auto *Fun = BM->getFunction(IF);
    for (unsigned IB = 0; IB < Fun->getNumBasicBlock(); ++IB) {
      const auto *BB = Fun->getBasicBlock(IB);
      for (unsigned II = 0; II < BB->getNumInst(); ++II) {
        auto *Inst = BB->getInst(II);
        if (Inst->getOpCode() == OpConditionalCopyObjectINTEL) {
          const auto OperandIds =
              static_cast<SPIRVConditionalCopyObjectINTEL *>(Inst)
                  ->getOperandIds();
          std::optional<unsigned> ITrue = std::nullopt;
          for (unsigned IO = 0; IO < OperandIds.size(); IO += 2) {
            const auto CondId = OperandIds[IO];
            bool Res;
            if (!evaluateConstant(BM, CondId, Res, ErrMsg)) {
              return false;
            }
            if (Res) {
              // Stop at the first condition operand that evaluates to true
              ITrue = IO;
              break;
            }
          }
          if (!ITrue.has_value()) {
            ErrMsg = "At least one conditional of OpConditionalCopyObjectINTEL "
                     "must be true. This could mean that all function variants "
                     "have been removed.";
            return false;
          }
          ToReplace.emplace_back(
              std::make_pair(Inst, OperandIds[ITrue.value() + 1]));
        }
      }
    }
  }

  for (auto &It : ToReplace) {
    auto *OldInst = It.first;
    auto *BB = OldInst->getBasicBlock();
    auto *NextInst = OldInst->getNext();
    auto *Operand = BM->getValue(It.second);
    auto *NewInst =
        new SPIRVCopyObject(OldInst->getType(), OldInst->getId(), Operand, BB);
    BM->eraseInstruction(OldInst, BB);
    BB->addInstruction(NewInst, NextInst);
  }

  // Specialize IDs annotated with  ConditionalINTEL decorations
  auto *Decors = BM->getDecorateVec();
  std::vector<SPIRVId> IdsToRemove;
  for (const auto &D : *Decors) {
    if (D->getDecorateKind() == DecorationConditionalINTEL) {
      const SPIRVId ConstId = static_cast<SPIRVWord>(D->getLiteral(0));
      bool ShouldKeep = false;
      if (!evaluateConstant(BM, ConstId, ShouldKeep, ErrMsg)) {
        return false;
      }
      if (!ShouldKeep) {
        IdsToRemove.push_back(D->getTargetId());
      }
    }
  }

  for (const auto &Id : IdsToRemove) {
    if (!BM->eraseReferencesOfInst(Id)) {
      ErrMsg = "Error removing references of instruction decorated with "
               "ConditionalINTEL";
      return false;
    }
    auto *Val = BM->getValue(Id);
    if (Val->getOpCode() == OpFunctionCall) {
      auto *Call = static_cast<SPIRVFunctionCall *>(Val);
      auto *BB = Call->getBasicBlock();
      BM->eraseInstruction(Call, BB);
    } else if (Val->getOpCode() == OpFunction) {
      const auto *Fun = static_cast<const SPIRVFunction *>(Val);
      for (unsigned I = 0; I < Fun->getNumArguments(); ++I) {
        const auto ArgId = Fun->getArgumentId(I);
        if (!BM->eraseReferencesOfInst(ArgId)) {
          ErrMsg = "Error erasing references of argument of a function "
                   "annotated with ConditionalINTEL";
          return false;
        }
      }
      for (const auto &VarId : Fun->getVariables()) {
        if (!BM->eraseReferencesOfInst(VarId)) {
          ErrMsg = "Error erasing references of variable within function "
                   "annotated with ConditionalINTEL";
          return false;
        }
      }
      for (unsigned IB = 0; IB < Fun->getNumBasicBlock(); ++IB) {
        const auto *const BB = Fun->getBasicBlock(IB);
        for (unsigned II = 0; II < BB->getNumInst(); ++II) {
          const auto *const Inst = BB->getInst(II);
          if (Inst->hasId()) {
            const auto InstId = Inst->getId();
            if (!BM->eraseReferencesOfInst(InstId)) {
              ErrMsg = "Error erasing references of instruction within "
                       "function annotated with ConditionalINTEL";
              return false;
            }
          }
        }
        if (!BM->eraseReferencesOfInst(BB->getId())) {
          ErrMsg = "Error erasing references of basic block label within "
                   "function annotated with ConditionalINTEL";
          return false;
        }
      }
      erase_if(*BM->getFuncVec(), [Id](auto F) { return F->getId() == Id; });
    } else if (Val->getOpCode() == OpVariable ||
               isTypeOpCode(Val->getOpCode()) ||
               Val->getOpCode() == OpExtInstImport ||
               isConstantOpCode(Val->getOpCode()) ||
               Val->getOpCode() == OpAsmINTEL ||
               Val->getOpCode() == OpAsmTargetINTEL) {
      if (!BM->eraseValue(Val)) {
        ErrMsg = "Error erasing value annotated with ConditionalINTEL";
        return false;
      }
    } else {
      ErrMsg = "Unsupported instruction annotated with ConditionalINTEL";
      return false;
    }
  }

  // Remove any leftover ConditionalINTEL decorations
  erase_if(*Decors, [](auto D) {
    return D->getDecorateKind() == DecorationConditionalINTEL;
  });

  // Remove capabilities/extensions of SPV_INTEL_function_variants
  BM->eraseCapability(CapabilityFunctionVariantsINTEL);
  BM->eraseCapability(CapabilitySpecConditionalINTEL);
  BM->getExtension().erase("SPV_INTEL_function_variants");

  return true;
}

} // namespace SPIRV
