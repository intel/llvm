//===============- SPIRVLowerBitCastToNonStandardType.cpp -================//
//
//                     The LLVM/SPIRV Translator
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
// Neither the names of Intel Corporation, nor the names of its
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
// This file implements lowering of BitCast to nonstandard types. LLVM
// transformations bitcast some vector types to scalar types, which are not
// universally supported across all targets. We need ensure that "optimized"
// LLVM IR doesn't have primitive types other than supported by the
// SPIR target (i.e. "scalar 8/16/32/64-bit integer and 16/32/64-bit floating
// point types, 2/3/4/8/16-element vector of scalar types").
//
//===----------------------------------------------------------------------===//

#include "SPIRVLowerBitCastToNonStandardType.h"
#include "SPIRVInternal.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/Transforms/Utils/Local.h"

#include <utility>

#define DEBUG_TYPE "spv-lower-bitcast-to-nonstandard-type"

using namespace llvm;

namespace SPIRV {

using NFIRBuilder = IRBuilder<NoFolder>;

static Value *removeBitCasts(Value *OldValue, Type *NewTy, NFIRBuilder &Builder,
                             std::vector<Instruction *> &InstsToErase) {
  IRBuilderBase::InsertPointGuard Guard(Builder);
  auto RauwBitcasts = [&](Instruction *OldValue, Value *NewValue) {
    // If there's only one use, don't create a bitcast for any uses, since it
    // will be immediately replaced anyways.
    if (OldValue->hasOneUse()) {
      OldValue->replaceAllUsesWith(UndefValue::get(OldValue->getType()));
    } else {
      OldValue->replaceAllUsesWith(
          Builder.CreateBitCast(NewValue, OldValue->getType()));
    }
    InstsToErase.push_back(OldValue);
    return NewValue;
  };

  if (auto *LI = dyn_cast<LoadInst>(OldValue)) {
    Builder.SetInsertPoint(LI);
    Value *Pointer = LI->getPointerOperand();
    if (!Pointer->getType()->isOpaquePointerTy()) {
      Type *NewPointerTy =
          PointerType::get(NewTy, LI->getPointerAddressSpace());
      Pointer = removeBitCasts(Pointer, NewPointerTy, Builder, InstsToErase);
    }
    LoadInst *NewLI = Builder.CreateAlignedLoad(NewTy, Pointer, LI->getAlign(),
                                                LI->isVolatile());
    NewLI->setOrdering(LI->getOrdering());
    NewLI->setSyncScopeID(LI->getSyncScopeID());
    return RauwBitcasts(LI, NewLI);
  }

  if (auto *ASCI = dyn_cast<AddrSpaceCastInst>(OldValue)) {
    Builder.SetInsertPoint(ASCI);
    Type *NewSrcTy = PointerType::getWithSamePointeeType(
        cast<PointerType>(NewTy), ASCI->getSrcAddressSpace());
    Value *Pointer = removeBitCasts(ASCI->getPointerOperand(), NewSrcTy,
                                    Builder, InstsToErase);
    return RauwBitcasts(ASCI, Builder.CreateAddrSpaceCast(Pointer, NewTy));
  }

  if (auto *BC = dyn_cast<BitCastInst>(OldValue)) {
    if (BC->getSrcTy() == NewTy) {
      if (BC->hasOneUse()) {
        BC->replaceAllUsesWith(UndefValue::get(BC->getType()));
        InstsToErase.push_back(BC);
      }
      return BC->getOperand(0);
    }
    Builder.SetInsertPoint(BC);
    return RauwBitcasts(BC, Builder.CreateBitCast(BC->getOperand(0), NewTy));
  }

  report_fatal_error("Cannot translate source of bitcast instruction.");
  return nullptr;
}

static bool isNonStdVecType(VectorType *VecTy) {
  uint64_t NumElems = VecTy->getElementCount().getFixedValue();
  return !isValidVectorSize(NumElems);
}

PreservedAnalyses
SPIRVLowerBitCastToNonStandardTypePass::run(Function &F,
                                            FunctionAnalysisManager &FAM) {
  // This pass doesn't cover all possible uses of non-standard types, only
  // known. We assume that bad type won't be passed to a function as
  // parameter, since it added by an optimization.
  bool Changed = false;

  // SPV_INTEL_vector_compute allows to use vectors with any number of
  // components. Since this method only lowers vectors with non-standard
  // in pure SPIR-V number of components, there is no need to do anything in
  // case SPV_INTEL_vector_compute is enabled.
  if (Opts.isAllowedToUseExtension(ExtensionID::SPV_INTEL_vector_compute))
    return PreservedAnalyses::all();

  // The basic pattern we're trying to fix is this InstCombine pattern:
  // trunc (extractelement) -> extractelement (bitcast)
  // (note that the bitcast itself can get propagated back to change the type
  // of load instructions, and even through those to pointer casts, if typed
  // pointers are enabled.
  std::vector<ExtractElementInst *> NonStdVecInsts;
  SmallVector<WeakTrackingVH, 4> MaybeDeletedInsts;
  for (auto &BB : F)
    for (auto &I : BB) {
      if (auto *EI = dyn_cast<ExtractElementInst>(&I)) {
        if (isNonStdVecType(EI->getVectorOperandType()))
          NonStdVecInsts.push_back(EI);
      } else if (auto *VT = dyn_cast<VectorType>(I.getType())) {
        if (isNonStdVecType(VT)) {
          MaybeDeletedInsts.push_back(&I);
        }
      }
    }

  std::vector<Instruction *> InstsToErase;
  NFIRBuilder Builder(F.getContext());
  for (auto &I : NonStdVecInsts) {
    VectorType *OldVecTy = I->getVectorOperandType();
    unsigned OldVecSize = OldVecTy->getElementCount().getFixedValue();

    // Compute the adjustment factor for the new vector size.
    unsigned VecFactor = 2;
    while (OldVecSize % VecFactor == 0 &&
           !isValidVectorSize(OldVecSize / VecFactor))
      VecFactor *= 2;
    if (OldVecSize % VecFactor != 0) {
      report_fatal_error(Twine("Invalid vector size for fixup: ") +
                         Twine(OldVecSize));
      return PreservedAnalyses::none();
    }
    unsigned NewElemSize = OldVecTy->getScalarSizeInBits() * VecFactor;
    VectorType *NewVecTy =
        VectorType::get(Type::getIntNTy(F.getContext(), NewElemSize),
                        OldVecSize / VecFactor, false);

    // Adjust the element index as appropriate.
    uint64_t OldElemIdx =
        cast<ConstantInt>(I->getIndexOperand())->getZExtValue();
    uint64_t NewElemIdx = OldElemIdx / VecFactor;
    uint64_t ShiftCount = OldElemIdx % VecFactor;
    Builder.SetInsertPoint(I);
    Value *NewVecOp =
        removeBitCasts(I->getVectorOperand(), NewVecTy, Builder, InstsToErase);
    Value *NewExtracted = Builder.CreateExtractElement(NewVecOp, NewElemIdx);

    // If the extract does higher-order bits of the value, shift as necessary.
    if (ShiftCount > 0)
      NewExtracted = Builder.CreateLShr(
          NewExtracted, ShiftCount * OldVecTy->getScalarSizeInBits());

    Value *NewValue = Builder.CreateTrunc(NewExtracted, I->getType());
    I->replaceAllUsesWith(NewValue);
    I->eraseFromParent();
    Changed = true;
  }

  for (auto *I : InstsToErase)
    RecursivelyDeleteTriviallyDeadInstructions(I);

  // Check if there are any residual unsupported vector types.
  for (auto &VH : MaybeDeletedInsts) {
    // Some vector-valued instructions were replaced with undef values, so if
    // that's what we got, it's still a dead instruction.
    if (VH.pointsToAliveValue() && !isa<UndefValue>(VH)) {
      auto *VT = dyn_cast<VectorType>(VH->getType());
      report_fatal_error(Twine("Unsupported vector type with ") +
                             Twine(VT->getElementCount().getFixedValue()) +
                             Twine(" elements"),
                         false);
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool SPIRVLowerBitCastToNonStandardTypeLegacy::runOnFunction(Function &F) {
  SPIRVLowerBitCastToNonStandardTypePass Impl(Opts);
  FunctionAnalysisManager FAM;
  auto PA = Impl.run(F, FAM);
  return !PA.areAllPreserved();
}

bool SPIRVLowerBitCastToNonStandardTypeLegacy::doFinalization(Module &M) {
  verifyRegularizationPass(M, "SPIRVLowerBitCastToNonStandardType");
  return false;
}

StringRef SPIRVLowerBitCastToNonStandardTypeLegacy::getPassName() const {
  return "Lower nonstandard type";
}

char SPIRVLowerBitCastToNonStandardTypeLegacy::ID = 0;

} // namespace SPIRV

INITIALIZE_PASS(SPIRVLowerBitCastToNonStandardTypeLegacy,
                "spv-lower-bitcast-to-nonstandard-type",
                "Remove bitcast to nonstandard types", false, false)

llvm::FunctionPass *llvm::createSPIRVLowerBitCastToNonStandardTypeLegacy(
    const SPIRV::TranslatorOpts &Opts) {
  return new SPIRVLowerBitCastToNonStandardTypeLegacy(Opts);
}
