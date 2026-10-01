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

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/Transforms/Utils/Local.h"

#include <numeric>
#include <utility>

#define DEBUG_TYPE "spv-lower-bitcast-to-nonstandard-type"

using namespace llvm;

namespace SPIRV {

using NFIRBuilder = IRBuilder<NoFolder>;

static FixedVectorType *getHalfTy(Type *Ty) {
  auto *VecTy = dyn_cast<FixedVectorType>(Ty);
  if (!VecTy)
    return nullptr;
  unsigned NumElems = VecTy->getNumElements();
  if (isValidVectorSize(NumElems) || NumElems % 2 != 0)
    return nullptr;
  auto *HalfTy = FixedVectorType::getHalfElementsVectorType(VecTy);
  return isValidVectorSize(HalfTy->getNumElements()) ? HalfTy : nullptr;
}

static Constant *getConstantLane(Constant *C, unsigned Idx) {
  Constant *Elem = C->getAggregateElement(Idx);
  return Elem ? Elem : PoisonValue::get(C->getType()->getScalarType());
}

namespace {

/// Splits values of an unsupported vector length into two legal halves. Only
/// the shapes SROA produces are handled; the rest is left to the diagnostic.
class NonStdVectorLegalizer {
public:
  explicit NonStdVectorLegalizer(NFIRBuilder &Builder) : Builder(Builder) {}

  bool run(Function &F);

private:
  using HalfPair = std::pair<Value *, Value *>;

  struct LaneRef {
    Value *Src;
    unsigned Idx;
  };

  bool canSplit(Value *V) const {
    return Halves.count(V) != 0 || isa<Constant>(V);
  }

  HalfPair getHalves(Value *V);
  LaneRef resolveLane(Value *V, unsigned Idx);
  Value *extractLane(Value *V, unsigned Idx);
  Value *buildShuffleLanes(ShuffleVectorInst &SVI, unsigned NumLanes);

  bool visit(Instruction &I);
  bool visitBitCast(BitCastInst &BCI);
  bool visitExtractElement(ExtractElementInst &EEI);
  bool visitShuffleVector(ShuffleVectorInst &SVI);

  NFIRBuilder &Builder;
  DenseMap<Value *, HalfPair> Halves;
};

} // namespace

bool NonStdVectorLegalizer::run(Function &F) {
  SmallVector<Instruction *, 32> Replaced;
  // Visit defining blocks before their users, regardless of block layout.
  for (BasicBlock *BB : ReversePostOrderTraversal<Function *>(&F)) {
    for (Instruction &I : *BB) {
      Builder.SetInsertPoint(&I);
      if (visit(I))
        Replaced.push_back(&I);
    }
  }

  // Erase only what became fully dead. An instruction with a use the visitors
  // declined to rewrite must survive for the diagnostic below to catch it.
  for (Instruction *I : reverse(Replaced))
    if (I->use_empty()) {
      // Keep debug users of a split bitcast referring to its original operand.
      salvageDebugInfo(*I);
      I->eraseFromParent();
    }
  return !Replaced.empty();
}

bool NonStdVectorLegalizer::visit(Instruction &I) {
  if (auto *BCI = dyn_cast<BitCastInst>(&I))
    return visitBitCast(*BCI);
  if (auto *EEI = dyn_cast<ExtractElementInst>(&I))
    return visitExtractElement(*EEI);
  if (auto *SVI = dyn_cast<ShuffleVectorInst>(&I))
    return visitShuffleVector(*SVI);
  return false;
}

NonStdVectorLegalizer::HalfPair NonStdVectorLegalizer::getHalves(Value *V) {
  auto It = Halves.find(V);
  if (It != Halves.end())
    return It->second;

  auto *C = cast<Constant>(V);
  unsigned HalfSize = cast<FixedVectorType>(C->getType())->getNumElements() / 2;
  SmallVector<Constant *, 16> LoElems, HiElems;
  for (unsigned Idx = 0; Idx != HalfSize; ++Idx) {
    LoElems.push_back(getConstantLane(C, Idx));
    HiElems.push_back(getConstantLane(C, Idx + HalfSize));
  }

  HalfPair Res = {ConstantVector::get(LoElems), ConstantVector::get(HiElems)};
  Halves[V] = Res;
  return Res;
}

NonStdVectorLegalizer::LaneRef
NonStdVectorLegalizer::resolveLane(Value *V, unsigned Idx) {
  auto *VecTy = cast<FixedVectorType>(V->getType());
  if (!getHalfTy(VecTy))
    return {V, Idx};

  auto [Lo, Hi] = getHalves(V);
  unsigned HalfSize = VecTy->getNumElements() / 2;
  return Idx < HalfSize ? LaneRef{Lo, Idx} : LaneRef{Hi, Idx - HalfSize};
}

Value *NonStdVectorLegalizer::extractLane(Value *V, unsigned Idx) {
  auto [Src, SrcIdx] = resolveLane(V, Idx);
  if (auto *C = dyn_cast<Constant>(Src))
    return getConstantLane(C, SrcIdx);
  return Builder.CreateExtractElement(Src, SrcIdx);
}

Value *NonStdVectorLegalizer::buildShuffleLanes(ShuffleVectorInst &SVI,
                                                unsigned NumLanes) {
  Type *ElemTy = cast<FixedVectorType>(SVI.getType())->getElementType();
  unsigned NumSrcElems =
      cast<FixedVectorType>(SVI.getOperand(0)->getType())->getNumElements();

  SmallVector<LaneRef, 16> Lanes(NumLanes, LaneRef{nullptr, 0});
  for (unsigned Lane = 0; Lane != NumLanes; ++Lane) {
    int MaskVal = SVI.getMaskValue(Lane);
    if (MaskVal < 0)
      continue;
    unsigned SrcIdx = MaskVal;
    Value *Src = SVI.getOperand(0);
    if (SrcIdx >= NumSrcElems) {
      SrcIdx -= NumSrcElems;
      Src = SVI.getOperand(1);
    }
    Lanes[Lane] = resolveLane(Src, SrcIdx);
  }

  SmallVector<Value *, 2> Srcs;
  bool OneShuffleIsEnough = true;
  for (const LaneRef &Lane : Lanes) {
    if (!Lane.Src || is_contained(Srcs, Lane.Src))
      continue;
    if (Srcs.size() == 2 ||
        (!Srcs.empty() && Srcs.front()->getType() != Lane.Src->getType())) {
      OneShuffleIsEnough = false;
      break;
    }
    Srcs.push_back(Lane.Src);
  }

  if (Srcs.empty())
    return PoisonValue::get(FixedVectorType::get(ElemTy, NumLanes));

  if (OneShuffleIsEnough) {
    unsigned SrcSize =
        cast<FixedVectorType>(Srcs.front()->getType())->getNumElements();
    SmallVector<int, 16> Mask(NumLanes, PoisonMaskElem);
    for (unsigned Lane = 0; Lane != NumLanes; ++Lane)
      if (Lanes[Lane].Src)
        Mask[Lane] =
            (Lanes[Lane].Src == Srcs.front() ? 0 : SrcSize) + Lanes[Lane].Idx;
    if (Srcs.size() == 1 && ShuffleVectorInst::isIdentityMask(Mask, SrcSize))
      return Srcs.front();

    Value *Src1 = Srcs.size() == 2 ? Srcs.back()
                                   : PoisonValue::get(Srcs.front()->getType());
    return Builder.CreateShuffleVector(Srcs.front(), Src1, Mask);
  }

  Value *Res = PoisonValue::get(FixedVectorType::get(ElemTy, NumLanes));
  for (unsigned Lane = 0; Lane != NumLanes; ++Lane) {
    auto [Src, Idx] = Lanes[Lane];
    if (!Src)
      continue;
    Value *Elem = isa<Constant>(Src) ? getConstantLane(cast<Constant>(Src), Idx)
                                     : Builder.CreateExtractElement(Src, Idx);
    Res = Builder.CreateInsertElement(Res, Elem, Lane);
  }
  return Res;
}

bool NonStdVectorLegalizer::visitBitCast(BitCastInst &BCI) {
  auto *DstHalfTy = getHalfTy(BCI.getType());
  Value *Src = BCI.getOperand(0);
  auto *SrcTy = dyn_cast<FixedVectorType>(Src->getType());
  // SPIR-V does not allow bitcasts to Boolean vectors, even with legal lengths.
  if (!DstHalfTy || DstHalfTy->getElementType()->isIntegerTy(1) || !SrcTy ||
      SrcTy->getNumElements() % 2 != 0)
    return false;
  unsigned HalfSrcSize = SrcTy->getNumElements() / 2;
  if (!isValidVectorSize(SrcTy->getNumElements()) ||
      !isValidVectorSize(HalfSrcSize))
    return false;

  SmallVector<int, 8> LoMask(HalfSrcSize), HiMask(HalfSrcSize);
  std::iota(LoMask.begin(), LoMask.end(), 0);
  std::iota(HiMask.begin(), HiMask.end(), HalfSrcSize);
  Halves[&BCI] = {Builder.CreateBitCast(
                      Builder.CreateShuffleVector(Src, LoMask), DstHalfTy),
                  Builder.CreateBitCast(
                      Builder.CreateShuffleVector(Src, HiMask), DstHalfTy)};
  return true;
}

bool NonStdVectorLegalizer::visitExtractElement(ExtractElementInst &EEI) {
  Value *Vec = EEI.getVectorOperand();
  auto *Idx = dyn_cast<ConstantInt>(EEI.getIndexOperand());
  if (!getHalfTy(Vec->getType()) || !Idx || !canSplit(Vec))
    return false;

  EEI.replaceAllUsesWith(extractLane(Vec, Idx->getZExtValue()));
  return true;
}

bool NonStdVectorLegalizer::visitShuffleVector(ShuffleVectorInst &SVI) {
  unsigned NumElems = cast<FixedVectorType>(SVI.getType())->getNumElements();
  if (!getHalfTy(SVI.getOperand(0)->getType()) || !isValidVectorSize(NumElems))
    return false;
  if (!canSplit(SVI.getOperand(0)) || !canSplit(SVI.getOperand(1)))
    return false;

  SVI.replaceAllUsesWith(buildShuffleLanes(SVI, NumElems));
  return true;
}

static Value *removeBitCasts(Value *OldValue, Type *NewTy, NFIRBuilder &Builder,
                             std::vector<Instruction *> &InstsToErase) {
  IRBuilderBase::InsertPointGuard Guard(Builder);
  auto RauwBitcasts = [&](Instruction *OldValue, Value *NewValue) {
    // If there's only one use, don't create a bitcast for any uses, since it
    // will be immediately replaced anyways.
    if (OldValue->hasOneUse()) {
      OldValue->replaceAllUsesWith(PoisonValue::get(OldValue->getType()));
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
    LoadInst *NewLI = Builder.CreateAlignedLoad(NewTy, Pointer, LI->getAlign(),
                                                LI->isVolatile());
    NewLI->setOrdering(LI->getOrdering());
    NewLI->setSyncScopeID(LI->getSyncScopeID());
    return RauwBitcasts(LI, NewLI);
  }

  if (auto *ASCI = dyn_cast<AddrSpaceCastInst>(OldValue)) {
    Builder.SetInsertPoint(ASCI);
    Type *NewSrcTy =
        PointerType::get(Builder.getContext(), ASCI->getSrcAddressSpace());
    Value *Pointer = removeBitCasts(ASCI->getPointerOperand(), NewSrcTy,
                                    Builder, InstsToErase);
    return RauwBitcasts(ASCI, Builder.CreateAddrSpaceCast(Pointer, NewTy));
  }

  if (auto *BC = dyn_cast<BitCastInst>(OldValue)) {
    if (BC->getSrcTy() == NewTy) {
      if (BC->hasOneUse()) {
        BC->replaceAllUsesWith(PoisonValue::get(BC->getType()));
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

  // SPV_EXT_long_vector and SPV_INTEL_vector_compute allow to use vectors with
  // any number of components. Since this method only lowers vectors with
  // non-standard in pure SPIR-V number of components, there is no need to do
  // anything in case any of them is enabled.
  if (Opts.isAllowedToUseExtension(ExtensionID::SPV_EXT_long_vector) ||
      Opts.isAllowedToUseExtension(ExtensionID::SPV_INTEL_vector_compute))
    return PreservedAnalyses::all();

  NFIRBuilder Builder(F.getContext());

  // Split vectors whose number of components is not supported by SPIR-V, but
  // whose half is, into a pair of halves. This covers the vectors that SROA
  // creates when it promotes a whole alloca to a single vector register.
  if (NonStdVectorLegalizer(Builder).run(F))
    Changed = true;

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
      auto *VT = cast<VectorType>(VH->getType());
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
