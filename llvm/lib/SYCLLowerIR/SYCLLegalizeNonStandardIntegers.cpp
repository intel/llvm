//===- SYCLLegalizeNonStandardIntegers.cpp - Legalize int types ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass legalizes non-standard integer types (i6, i24, i48, etc.) by
// widening them to standard widths (8, 16, 32, 64).
//
// Strategy: Pattern-based legalization that handles common cases:
// 1. trunc + op + zext: trunc i32 to i24, op i24, zext i24 to i32
//    -> Replace with op on widened type with mask
// 2. bitcast vector to non-std int: bitcast <3 x half> (totally i48)
//    -> Pad vector and bitcast to wider standard type
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLLegalizeNonStandardIntegers.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"

#define DEBUG_TYPE "sycl-legalize-nonstandard-integers"

using namespace llvm;

// Check if an integer type has non-standard bit width.
static bool isNonStdIntType(Type *Ty) {
  if (!Ty->isIntegerTy())
    return false;
  unsigned BitWidth = Ty->getIntegerBitWidth();
  return BitWidth > 64 || (BitWidth & (BitWidth - 1)) != 0;
}

// Widen bit width to next power of 2 (capped at 64).
static unsigned widenBitWidth(unsigned BitWidth) {
  if (BitWidth == 1)
    return 1;
  if (BitWidth > 64)
    report_fatal_error(
        Twine("BitWidth value cannot be greater than 64, found ") +
        Twine(BitWidth));
  return std::max(1u << Log2_32_Ceil(BitWidth), 8u);
}

// Convert a value to the target integer type, inserting trunc/zext as needed.
static Value *convertToType(IRBuilder<> &Builder, Value *V, Type *TargetTy) {
  Type *SrcTy = V->getType();
  if (SrcTy == TargetTy)
    return V;
  return SrcTy->getIntegerBitWidth() > TargetTy->getIntegerBitWidth()
             ? Builder.CreateTrunc(V, TargetTy)
             : Builder.CreateZExt(V, TargetTy);
}

// Mask a value to the given bit width.
static Value *maskToWidth(IRBuilder<> &Builder, Value *V, unsigned Width) {
  Type *Ty = V->getType();
  APInt Mask = APInt::getLowBitsSet(Ty->getIntegerBitWidth(), Width);
  return Builder.CreateAnd(V, ConstantInt::get(Ty, Mask));
}

// Replace a ZExt of a non-std int with the appropriately extended widened
// value. Returns true if the ZExt was replaced and erased.
static bool replaceZExt(IRBuilder<> &Builder, ZExtInst *ZExt, Value *WidenedVal,
                        Type *WidenedTy) {
  Builder.SetInsertPoint(ZExt);
  Type *DestTy = ZExt->getDestTy();
  Value *Replacement = (DestTy == WidenedTy)
                           ? WidenedVal
                           : Builder.CreateZExt(WidenedVal, DestTy);
  ZExt->replaceAllUsesWith(Replacement);
  ZExt->eraseFromParent();
  return true;
}

// Collect all uses of an instruction into a vector for safe iteration.
static SmallVector<Use *, 8> collectUses(Instruction *I) {
  SmallVector<Use *, 8> Uses;
  for (Use &U : I->uses())
    Uses.push_back(&U);
  return Uses;
}

// Erase dead instructions with non-standard integer types iteratively.
static void cleanupDeadNonStdIntInsts(Function &F) {
  bool Changed = true;
  while (Changed) {
    Changed = false;
    SmallVector<Instruction *, 8> ToErase;
    for (Instruction &I : instructions(F)) {
      if (isNonStdIntType(I.getType()) && I.use_empty() &&
          !I.mayHaveSideEffects())
        ToErase.push_back(&I);
    }
    for (Instruction *I : ToErase) {
      I->eraseFromParent();
      Changed = true;
    }
  }
}

// Legalize a binary operation with a non-standard integer operand.
static void legalizeBinaryOp(IRBuilder<> &Builder, BinaryOperator *BinOp,
                             unsigned OpIdx, Value *MaskedVal, Type *NewTy,
                             unsigned OldWidth, Type *NonStdTy) {
  Builder.SetInsertPoint(BinOp);
  Value *OtherOp = BinOp->getOperand(1 - OpIdx);
  Value *WidenedOther;

  // Widen the other operand based on its kind.
  if (auto *OtherTrunc = dyn_cast<TruncInst>(OtherOp);
      OtherTrunc && OtherTrunc->getDestTy() == NonStdTy) {
    Value *OtherSrc = OtherTrunc->getOperand(0);
    Value *OtherWide = convertToType(Builder, OtherSrc, NewTy);
    WidenedOther = maskToWidth(Builder, OtherWide, OldWidth);
  } else if (auto *CI = dyn_cast<ConstantInt>(OtherOp)) {
    APInt Mask = APInt::getLowBitsSet(NewTy->getIntegerBitWidth(), OldWidth);
    WidenedOther =
        ConstantInt::get(NewTy, CI->getZExtValue() & Mask.getZExtValue());
  } else {
    WidenedOther = Builder.CreateZExt(OtherOp, NewTy);
  }

  // Create the operation on the widened type.
  Value *LHS = (OpIdx == 0) ? MaskedVal : WidenedOther;
  Value *RHS = (OpIdx == 0) ? WidenedOther : MaskedVal;
  Value *NewOp = Builder.CreateBinOp(BinOp->getOpcode(), LHS, RHS);
  Value *MaskedResult = maskToWidth(Builder, NewOp, OldWidth);

  // Replace users of the original BinOp.
  for (Use *BU : collectUses(BinOp)) {
    Instruction *BinUser = cast<Instruction>(BU->getUser());
    if (auto *ZE = dyn_cast<ZExtInst>(BinUser))
      replaceZExt(Builder, ZE, MaskedResult, NewTy);
    else
      BU->set(MaskedResult);
  }
}

// Handle pattern: trunc -> ops on non-std int -> zext.
// Replaces operations on non-std types with masked operations on widened types.
static bool legalizeNonStdIntChain(Function &F) {
  SmallVector<TruncInst *, 16> NonStdTruncs;
  for (Instruction &I : instructions(F)) {
    if (auto *TI = dyn_cast<TruncInst>(&I))
      if (isNonStdIntType(TI->getDestTy()))
        NonStdTruncs.push_back(TI);
  }

  if (NonStdTruncs.empty())
    return false;

  IRBuilder<> Builder(F.getContext());
  for (TruncInst *TI : NonStdTruncs) {
    Type *NonStdTy = TI->getDestTy();
    unsigned OldWidth = NonStdTy->getIntegerBitWidth();
    unsigned NewWidth = widenBitWidth(OldWidth);
    Type *NewTy = IntegerType::get(F.getContext(), NewWidth);

    // Convert the trunc source to the widened type, then mask to original
    // width.
    Builder.SetInsertPoint(TI);
    Value *WidenedSrc = convertToType(Builder, TI->getOperand(0), NewTy);
    Value *MaskedVal = maskToWidth(Builder, WidenedSrc, OldWidth);

    // Process all users of the trunc result.
    for (Use *U : collectUses(TI)) {
      Instruction *User = cast<Instruction>(U->getUser());
      if (auto *ZExt = dyn_cast<ZExtInst>(User))
        replaceZExt(Builder, ZExt, MaskedVal, NewTy);
      else if (auto *BinOp = dyn_cast<BinaryOperator>(User))
        legalizeBinaryOp(Builder, BinOp, U->getOperandNo(), MaskedVal, NewTy,
                         OldWidth, NonStdTy);
      else
        U->set(MaskedVal);
    }
  }

  cleanupDeadNonStdIntInsts(F);
  return true;
}

// Handle pattern: bitcast <N x T> to iM where M is non-standard.
// Pads the vector to match the widened integer size and bitcasts to wider type.
static bool legalizeNonStdBitcasts(Function &F) {
  SmallVector<BitCastInst *, 8> NonStdBitcasts;
  for (Instruction &I : instructions(F)) {
    if (auto *BC = dyn_cast<BitCastInst>(&I))
      if (isNonStdIntType(BC->getDestTy()))
        NonStdBitcasts.push_back(BC);
  }

  if (NonStdBitcasts.empty())
    return false;

  IRBuilder<> Builder(F.getContext());
  for (BitCastInst *BC : NonStdBitcasts) {
    Type *NonStdTy = BC->getDestTy();
    unsigned NewWidth = widenBitWidth(NonStdTy->getIntegerBitWidth());
    Type *NewTy = IntegerType::get(F.getContext(), NewWidth);
    Value *Src = BC->getOperand(0);

    Builder.SetInsertPoint(BC);

    // If source is a vector smaller than the widened size, pad with poison.
    if (auto *VTy = dyn_cast<VectorType>(Src->getType())) {
      unsigned SrcBits = VTy->getPrimitiveSizeInBits().getFixedValue();
      if (SrcBits < NewWidth) {
        int NewNumElems = NewWidth / VTy->getScalarSizeInBits();
        int OldNumElems = VTy->getElementCount().getFixedValue();
        // Since only bitwidth = 64 is supported, vector size of 8 is
        // sufficient.
        SmallVector<int, 8> Mask;
        for (int I = 0; I < NewNumElems; ++I)
          Mask.push_back(I < OldNumElems ? I : -1);
        Src = Builder.CreateShuffleVector(Src, PoisonValue::get(VTy), Mask);
      }
    }

    Value *NewBC = Builder.CreateBitCast(Src, NewTy);

    // Handle uses of the original bitcast.
    for (Use *U : collectUses(BC)) {
      Instruction *User = cast<Instruction>(U->getUser());
      if (auto *ZExt = dyn_cast<ZExtInst>(User))
        replaceZExt(Builder, ZExt, NewBC, NewTy);
      else
        U->set(NewBC);
    }

    if (BC->use_empty())
      BC->eraseFromParent();
  }

  return true;
}

PreservedAnalyses
SYCLLegalizeNonStandardIntegersPass::run(Module &M,
                                         ModuleAnalysisManager &MAM) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    Changed |= legalizeNonStdBitcasts(F);
    Changed |= legalizeNonStdIntChain(F);
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
