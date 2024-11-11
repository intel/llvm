// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "analysis/uniform_value_analysis.h"
#include "debugging.h"
#include "transform/passes.h"

#define DEBUG_TYPE "vecz"

using namespace llvm;
using namespace vecz;

/// @brief remove IntPtrs where possible.
PreservedAnalyses RemoveIntPtrPass::run(Function &F,
                                        FunctionAnalysisManager &) {
  static const StringRef name = "remove_intptr";

  SmallVector<PtrToIntInst *, 16> casts;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *int_ptr = dyn_cast<PtrToIntInst>(&I)) {
        casts.push_back(int_ptr);
      }
    }
  }

  if (casts.empty()) {
    return PreservedAnalyses::all();
  }

  while (!casts.empty()) {
    PtrToIntInst *int_ptr = casts.back();
    casts.pop_back();

    for (auto usei = int_ptr->use_begin(); usei != int_ptr->use_end();) {
      auto &use = *(usei++);
      auto *user = use.getUser();

      if (auto *ptr = dyn_cast<IntToPtrInst>(user)) {
        IRBuilder<> B(ptr);
        Value *new_cast = B.CreatePointerBitCastOrAddrSpaceCast(
            int_ptr->getOperand(0), ptr->getDestTy(), name);
        ptr->replaceAllUsesWith(new_cast);
        ptr->eraseFromParent();
      } else if (auto *phi = dyn_cast<PHINode>(user)) {
        // How we deal with PHI nodes is we create another PHI node with the
        // pointer type, moving the PtrToInt to the other side of it. We also
        // create IntToPtrs on the incoming side, where it does not consume
        // the PtrToInt that we are currently looking at. Any new casts will
        // hopefully be removed later.
        auto num_values = phi->getNumIncomingValues();
        PHINode *new_phi = PHINode::Create(int_ptr->getSrcTy(), num_values,
                                           phi->getName() + ".intptr");
        new_phi->insertBefore(phi->getIterator());

        Instruction *insert = phi;
        while (isa<PHINode>(insert)) {
          insert = insert->getNextNonDebugInstruction();
        }

        // Populate the replacement PHI node
        for (decltype(num_values) i = 0; i != num_values; ++i) {
          Value *incoming = phi->getIncomingValue(i);
          BasicBlock *inb = phi->getIncomingBlock(i);
          if (incoming == int_ptr) {
            incoming = int_ptr->getOperand(0);
          } else {
            IRBuilder<> B(inb->getTerminator());
            incoming = B.CreateIntToPtr(incoming, int_ptr->getSrcTy(), name);
          }
          new_phi->addIncoming(incoming, inb);
        }

        // Add the cast back to Int at the other side
        IRBuilder<> B(insert);
        Value *new_cast = B.CreatePtrToInt(new_phi, phi->getType(), name);
        phi->replaceAllUsesWith(new_cast);
        phi->eraseFromParent();
        casts.push_back(cast<PtrToIntInst>(new_cast));
      } else if (auto *bin_op = dyn_cast<BinaryOperator>(user)) {
        auto *i8_ty = IntegerType::getInt8Ty(F.getContext());

        IRBuilder<> B(bin_op);
        Value *index = nullptr;

        auto opcode = bin_op->getOpcode();
        if (opcode == Instruction::Add) {
          index = bin_op->getOperand(use.getOperandNo() == 0);
        } else if (opcode == Instruction::Sub && use.getOperandNo() == 0) {
          index = B.CreateNeg(bin_op->getOperand(1), name);
        }

        if (index) {
          Value *operand = int_ptr->getOperand(0);
          Value *new_gep = B.CreateGEP(i8_ty, operand, index, name);
          Value *new_cast = B.CreatePtrToInt(new_gep, bin_op->getType(), name);
          bin_op->replaceAllUsesWith(new_cast);
          bin_op->eraseFromParent();
          casts.push_back(cast<PtrToIntInst>(new_cast));
        }
      }
    }

    if (int_ptr->use_empty()) {
      int_ptr->eraseFromParent();
    }
  }

  auto Preserved = PreservedAnalyses::all();
  Preserved.abandon<UniformValueAnalysis>();
  return Preserved;
}
