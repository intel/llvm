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

#include "transform/instantiation_pass.h"

#include <compiler/utils/builtin_info.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include <memory>

#include "analysis/instantiation_analysis.h"
#include "analysis/uniform_value_analysis.h"
#include "debugging.h"
#include "llvm_helpers.h"
#include "memory_operations.h"
#include "transform/packetization_helpers.h"
#include "transform/packetizer.h"
#include "vectorization_context.h"
#include "vecz/vecz_choices.h"

#define DEBUG_TYPE "vecz-instantiation"

#undef VECZ_FAIL
#define VECZ_FAIL() return packetizer.getEmptyRange();

using namespace vecz;
using namespace llvm;

STATISTIC(VeczInstantiated, "Number of instructions instantiated [ID#I00]");
STATISTIC(VeczPacketizeFailInstantiate,
          "Packetize: instantiation failures [ID#P84]");

InstantiationPass::InstantiationPass(Packetizer &pp)
    : Ctx(pp.context()), packetizer(pp) {}

PacketRange InstantiationPass::instantiate(Value *V) {
  VECZ_FAIL_IF(packetizer.width().isScalable());
  if (auto info = packetizer.getPacketized(V)) {
    const unsigned SimdWidth = packetizer.width().getFixedValue();
    return info.getAsPacket(SimdWidth);
  }

  // Handle uniform values first, which instantiate to the same value for all
  // items.
  auto *Ins = dyn_cast<Instruction>(V);
  if (Ins && packetizer.uniform().isMaskVarying(V)) {
    const PacketRange P = simdBroadcast(Ins);
    if (!P) {
      emitVeczRemark(&packetizer.function(), V,
                     "Failed to broadcast Mask Varying instruction");
      VECZ_FAIL();
    }
    return assignInstance(P, V);
  }

  if (!packetizer.uniform().isVarying(V)) {
    return assignInstance(broadcast(V), V);
  }

  if (Ins) {
    return instantiateInstruction(Ins);
  }

  VECZ_STAT_FAIL_IF(true, VeczPacketizeFailInstantiate);
}

PacketRange InstantiationPass::instantiateInternal(Value *V) {
  if (packetizer.uniform().isVarying(V)) {
    // The packetizer will call back into the instantiator when it needs to
    VECZ_FAIL_IF(packetizer.width().isScalable());
    const unsigned SimdWidth = packetizer.width().getFixedValue();
    return packetizer.packetize(V).getAsPacket(SimdWidth);
  } else {
    return instantiate(V);
  }
}

PacketRange InstantiationPass::instantiateInstruction(Instruction *Ins) {
  // Figure out what kind of instruction it is and try to instantiate it.
  switch (Ins->getOpcode()) {
    default:
      // No special handling of this Instruction so just clone across lanes..
      break;

    case Instruction::Call:
      return assignInstance(instantiateCall(cast<CallInst>(Ins)), Ins);

    case Instruction::Alloca:
      return assignInstance(instantiateAlloca(cast<AllocaInst>(Ins)), Ins);
  }

  return assignInstance(instantiateByCloning(Ins), Ins);
}

PacketRange InstantiationPass::assignInstance(const PacketRange P, Value *V) {
  if (!P) {
    emitVeczRemarkMissed(&packetizer.function(), V, "Could not instantiate");
    VECZ_STAT_FAIL_IF(!P, VeczPacketizeFailInstantiate);
  } else {
    ++VeczInstantiated;
  }
  return P;
}

PacketRange InstantiationPass::broadcast(Value *V) {
  VECZ_FAIL_IF(packetizer.width().isScalable());
  const unsigned SimdWidth = packetizer.width().getFixedValue();
  PacketRange P = packetizer.createPacket(V, SimdWidth);
  for (unsigned i = 0; i < SimdWidth; i++) {
    P[i] = V;
  }
  return P;
}

PacketRange InstantiationPass::instantiateCall(CallInst *CI) {
  VECZ_FAIL_IF(packetizer.width().isScalable());
  const unsigned SimdWidth = packetizer.width().getFixedValue();
  // Handle special call instructions that return a lane ID.
  const compiler::utils::BuiltinInfo &BI = Ctx.builtins();
  const auto Builtin = BI.analyzeBuiltinCall(*CI, packetizer.dimension());
  if (Builtin.properties & compiler::utils::eBuiltinPropertyWorkItem) {
    const auto Uniformity = Builtin.uniformity;
    if (Uniformity == compiler::utils::eBuiltinUniformityNever) {
      // can't handle these (global/local linear ID probably)
      VECZ_FAIL();
    } else if (Uniformity & compiler::utils::eBuiltinUniformityInstanceID) {
      Type *RetTy = CI->getType();
      PacketRange P = packetizer.createPacket(CI, SimdWidth);
      VECZ_FAIL_IF(!P);
      IRBuilder<> B(CI);
      for (unsigned j = 0; j < SimdWidth; j++) {
        P[j] = B.CreateAdd(CI, ConstantInt::get(RetTy, j));
      }
      packetizer.deleteInstructionLater(CI);
      return P;
    }
  }

  // We can't instantiate noduplicate functions
  VECZ_FAIL_IF(CI->hasFnAttr(Attribute::NoDuplicate));

  packetizer.deleteInstructionLater(CI);
  // Check if the instruction has any uses or not, and also if we want to
  // instantiate call instructions with loops or not.
  if (CI->hasNUsesOrMore(1) ||
      !packetizer.choices().instantiateCallsInLoops()) {
    // Instantiate as always
    SmallVector<PacketRange, 4> OpPackets;
    for (unsigned i = 0; i < CI->arg_size(); i++) {
      Value *Op = CI->getArgOperand(i);
      const PacketRange OpPacket = instantiateInternal(Op);
      VECZ_FAIL_IF(!OpPacket);
      OpPackets.push_back(OpPacket);
    }
    PacketRange P = packetizer.createPacket(CI, SimdWidth);
    VECZ_FAIL_IF(!P);
    IRBuilder<> B(CI);
    for (unsigned j = 0; j < SimdWidth; j++) {
      SmallVector<Value *, 4> Ops;
      for (unsigned i = 0; i < CI->arg_size(); i++) {
        Ops.push_back(OpPackets[i][j]);
      }
      auto *NewCI = B.CreateCall(CI->getFunctionType(), CI->getCalledOperand(),
                                 Ops, CI->getName());
      NewCI->setCallingConv(CI->getCallingConv());
      NewCI->setAttributes(CI->getAttributes());
      P[j] = NewCI;
    }
    return P;
  } else {
    // Instantiate in a loop
    BasicBlock *BeforeCI = CI->getParent();
    BasicBlock *AfterCI = SplitBlock(BeforeCI, CI);
    BasicBlock *LoopHeader = BasicBlock::Create(
        CI->getContext(), "instloop.header", CI->getFunction(), AfterCI);
    BasicBlock *LoopBody = BasicBlock::Create(CI->getContext(), "instloop.body",
                                              CI->getFunction(), AfterCI);

    // Change the branch instruction from BeforeCI -> AfterCI to BeforeCI ->
    // LoopHeader
    BeforeCI->getTerminator()->setSuccessor(0, LoopHeader);

    IRBuilder<> B(LoopHeader);
    // Create the induction variable
    PHINode *Ind = B.CreatePHI(B.getInt32Ty(), 2, "instance");

    // Create the conditional jump based on the current iteration number
    Value *ICmp = B.CreateICmpULT(Ind, B.getInt32(SimdWidth));
    B.CreateCondBr(ICmp, LoopBody, AfterCI);

    B.SetInsertPoint(LoopBody);
    SmallVector<Value *, 4> Operands;
    for (auto &Arg : CI->args()) {
      // We call the packetizer explicitly, instead of calling the
      // instantiator, because we need a packetized value and not an
      // instantiateed one.
      Value *Packetized = packetizer.packetize(Arg).getAsValue();
      VECZ_FAIL_IF(!Packetized);
      VECZ_ERROR_IF(!Packetized->getType()->isVectorTy(),
                    "The packetized Value has to be of a vector type");
      Operands.push_back(Packetized);
    }
    // Each Op is an element extracted from a packetized instruction.
    SmallVector<Value *, 4> Ops;
    for (unsigned i = 0; i < Operands.size(); ++i) {
      Ops.push_back(B.CreateExtractElement(Operands[i], Ind));
    }
    // Create the function call
    auto CO = CI->getCalledOperand();
    FunctionType *FTy = CI->getFunctionType();
    CallInst *NewCI = B.CreateCall(FTy, CO, Ops);
    NewCI->setCallingConv(CI->getCallingConv());
    NewCI->setAttributes(CI->getAttributes());
    // Increment the induction variable and jump back to the loop header
    Value *IndInc = B.CreateAdd(Ind, B.getInt32(1), "");
    B.CreateBr(LoopHeader);

    // Set the operands to the Phi node in the loop header
    Ind->addIncoming(B.getInt32(0), BeforeCI);
    Ind->addIncoming(IndInc, LoopBody);

    // Set the Packet, even though we are not going to be using this value (we
    // have checked if the call has 0 users). We don't need to populate it.
    return packetizer.createPacket(CI, SimdWidth);
  }
}

PacketRange InstantiationPass::instantiateAlloca(AllocaInst *Alloca) {
  VECZ_FAIL_IF(packetizer.width().isScalable());
  const unsigned SimdWidth = packetizer.width().getFixedValue();
  PacketRange P = packetizer.createPacket(Alloca, SimdWidth);
  VECZ_FAIL_IF(!P);
  IRBuilder<> B(Alloca);
  for (unsigned i = 0; i < SimdWidth; i++) {
    Type *Ty = Alloca->getAllocatedType();
    AllocaInst *New = B.CreateAlloca(Ty, nullptr, Alloca->getName());
    New->setAlignment(Alloca->getAlign());

    P[i] = New;
  }
  packetizer.deleteInstructionLater(Alloca);
  return P;
}

PacketRange InstantiationPass::instantiateByCloning(Instruction *I) {
  VECZ_FAIL_IF(packetizer.width().isScalable());
  auto SimdWidth = packetizer.width().getFixedValue();
  PacketRange P = packetizer.createPacket(I, SimdWidth);
  if (!P || P.at(SimdWidth - 1)) {
    return P;
  }

  // Clone breadth first so that the packet is complete before fixing up the
  // operands, that way we get less stack-thrashing, especially when there
  // is a circular dependency.
  SmallVector<Instruction *, 16> Clones;
  for (decltype(SimdWidth) i = 0; i < SimdWidth; ++i) {
    if (P.at(i)) {
      Clones.push_back(nullptr);
      continue;
    }
    Instruction *Clone = I->clone();
    Clone->insertBefore(I);
    P[i] = Clone;
    Clones.push_back(Clone);
  }

  for (unsigned i = 0, n = I->getNumOperands(); i != n; ++i) {
    Value *V = I->getOperand(i);
    if (isa<BasicBlock>(V) || isa<Constant>(V)) {
      continue;
    }

    if (const auto OpP = instantiateInternal(V)) {
      for (decltype(SimdWidth) lane = 0; lane < SimdWidth; ++lane) {
        if (auto *Clone = Clones[lane]) {
          if (auto *At = OpP.at(lane)) {
            Clone->setOperand(i, At);
          }
        }
      }
    } else {
      VECZ_FAIL();
    }
  }

  packetizer.deleteInstructionLater(I);
  return P;
}

PacketRange InstantiationPass::simdBroadcast(Instruction *I) {
  VECZ_FAIL_IF(packetizer.width().isScalable());
  auto SimdWidth = packetizer.width().getFixedValue();
  PacketRange P = packetizer.createPacket(I, SimdWidth);
  if (!P || P.at(0)) {
    return P;
  }

  for (auto &i : P) {
    i = I;
  }

  auto Op = MemOp::get(I);
  if (!Op || !Op->getMaskOperand()) {
    return P;
  }

  if (auto *MaskInst = dyn_cast<Instruction>(Op->getMaskOperand())) {
    const auto MP = instantiateInternal(MaskInst);
    VECZ_FAIL_IF(!MP);

    auto W = SimdWidth;
    SmallVector<Value *, 16> Reduce;
    for (decltype(SimdWidth) i = 0; i < SimdWidth; i++) {
      Reduce.push_back(MP.at(i));
    }

    IRBuilder<> B(buildAfter(Reduce.back(), packetizer.function()));
    while ((W >>= 1)) {
      for (decltype(W) i = 0; i < W; ++i) {
        Reduce[i] = B.CreateOr(Reduce[i], Reduce[i + W], "any_of_mask");
      }
    }
    Op->setMaskOperand(Reduce.front());
  }

  return P;
}
