//===- SPIRVLowerBool.cpp - Lower instructions with bool operands ---------===//
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
//
// This file implements lowering instructions with bool operands.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "spvbool"

#include "SPIRVInternal.h"
#include "libSPIRV/SPIRVDebug.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

using namespace llvm;
using namespace SPIRV;

namespace SPIRV {

class SPIRVLowerBoolBase : public InstVisitor<SPIRVLowerBoolBase> {
public:
  SPIRVLowerBoolBase() : Context(nullptr) {}
  virtual ~SPIRVLowerBoolBase() {}
  void replace(Instruction *I, Instruction *NewI) {
    NewI->takeName(I);
    I->replaceAllUsesWith(NewI);
    I->dropAllReferences();
    I->eraseFromParent();
  }
  bool isBoolType(Type *Ty) {
    if (Ty->isIntegerTy(1))
      return true;
    if (auto VT = dyn_cast<VectorType>(Ty))
      return isBoolType(VT->getElementType());
    return false;
  }
  virtual void visitTruncInst(TruncInst &I) {
    if (isBoolType(I.getType())) {
      auto Op = I.getOperand(0);
      auto And = BinaryOperator::CreateAnd(
          Op, getScalarOrVectorConstantInt(Op->getType(), 1, false), "", &I);
      auto Zero = getScalarOrVectorConstantInt(Op->getType(), 0, false);
      auto Cmp = new ICmpInst(&I, CmpInst::ICMP_NE, And, Zero);
      replace(&I, Cmp);
    }
  }
  void handleExtInstructions(Instruction &I) {
    auto Op = I.getOperand(0);
    if (isBoolType(Op->getType())) {
      auto Opcode = I.getOpcode();
      auto Ty = I.getType();
      auto Zero = getScalarOrVectorConstantInt(Ty, 0, false);
      auto One = getScalarOrVectorConstantInt(
          Ty, (Opcode == Instruction::SExt) ? ~0 : 1, false);
      assert(Zero && One && "Couldn't create constant int");
      auto Sel = SelectInst::Create(Op, One, Zero, "", &I);
      replace(&I, Sel);
    }
  }
  void handleCastInstructions(Instruction &I) {
    auto Op = I.getOperand(0);
    auto *OpTy = Op->getType();
    if (isBoolType(OpTy)) {
      Type *Ty = Type::getInt32Ty(*Context);
      if (auto VT = dyn_cast<FixedVectorType>(OpTy))
        Ty = llvm::FixedVectorType::get(Ty, VT->getNumElements());
      auto Zero = getScalarOrVectorConstantInt(Ty, 0, false);
      auto One = getScalarOrVectorConstantInt(Ty, 1, false);
      assert(Zero && One && "Couldn't create constant int");
      auto Sel = SelectInst::Create(Op, One, Zero, "", &I);
      I.setOperand(0, Sel);
    }
  }
  virtual void visitZExtInst(ZExtInst &I) { handleExtInstructions(I); }
  virtual void visitSExtInst(SExtInst &I) { handleExtInstructions(I); }
  virtual void visitUIToFPInst(UIToFPInst &I) { handleCastInstructions(I); }
  virtual void visitSIToFPInst(SIToFPInst &I) { handleCastInstructions(I); }
  bool runLowerBool(Module &M) {
    Context = &M.getContext();
    visit(M);

    verifyRegularizationPass(M, "SPIRVLowerBool");
    return true;
  }

private:
  LLVMContext *Context;
};

class SPIRVLowerBoolPass : public llvm::PassInfoMixin<SPIRVLowerBoolPass>,
                           public SPIRVLowerBoolBase {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM) {
    return runLowerBool(M) ? llvm::PreservedAnalyses::none()
                           : llvm::PreservedAnalyses::all();
  }
};

class SPIRVLowerBoolLegacy : public ModulePass, public SPIRVLowerBoolBase {
public:
  SPIRVLowerBoolLegacy() : ModulePass(ID) {
    initializeSPIRVLowerBoolLegacyPass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override { return runLowerBool(M); }

  static char ID;
};

char SPIRVLowerBoolLegacy::ID = 0;
} // namespace SPIRV

INITIALIZE_PASS(SPIRVLowerBoolLegacy, "spvbool",
                "Lower instructions with bool operands", false, false)

ModulePass *llvm::createSPIRVLowerBoolLegacy() {
  return new SPIRVLowerBoolLegacy();
}
