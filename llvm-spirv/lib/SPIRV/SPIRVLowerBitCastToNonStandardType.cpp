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
#define DEBUG_TYPE "spv-lower-bitcast-to-nonstandard-type"

#include "SPIRVInternal.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PassManager.h"

#include <utility>

using namespace llvm;

namespace SPIRV {

static VectorType *getVectorType(Type *Ty) {
  assert(Ty != nullptr && "Expected non-null type");
  if (auto *ElemTy = dyn_cast<PointerType>(Ty))
    Ty = ElemTy->getPointerElementType();
  return dyn_cast<VectorType>(Ty);
}

/// Since SPIR-V does not support non-standard vector types, instructions using
/// these types should be replaced in a special way to avoid using of
/// unsupported types.
/// lowerBitCastToNonStdVec function is designed to avoid using of bitcast to
/// unsupported vector types instructions and should be called if similar
/// instructions have been encountered in input LLVM IR.
bool lowerBitCastToNonStdVec(Instruction *OldInst, Value *NewInst,
                             const VectorType *OldVecTy,
                             std::vector<Instruction *> &InstsToErase,
                             IRBuilder<> &Builder,
                             unsigned RecursionDepth = 0) {
  static constexpr unsigned MaxRecursionDepth = 16;
  if (RecursionDepth++ > MaxRecursionDepth)
    report_fatal_error(
        llvm::Twine(
            "The depth of recursion exceeds the maximum possible depth"),
        false);

  bool Changed = false;
  VectorType *NewVecTy = getVectorType(NewInst->getType());
  if (NewVecTy) {
    Builder.SetInsertPoint(OldInst);
    for (auto *U : OldInst->users()) {
      // Handle addrspacecast instruction after bitcast if present
      if (auto *ASCastInst = dyn_cast<AddrSpaceCastInst>(U)) {
        unsigned DestAS = ASCastInst->getDestAddressSpace();
        auto *NewVecPtrTy = NewVecTy->getPointerTo(DestAS);
        // AddrSpaceCast is created explicitly instead of using method
        // IRBuilder<>.CreateAddrSpaceCast because IRBuilder doesn't create
        // separate instruction for constant values. Whereas SPIR-V translator
        // doesn't like several nested instructions in one.
        Value *LocalValue = new AddrSpaceCastInst(NewInst, NewVecPtrTy);
        Builder.Insert(LocalValue);
        Changed |=
            lowerBitCastToNonStdVec(ASCastInst, LocalValue, OldVecTy,
                                    InstsToErase, Builder, RecursionDepth);
      }
      // Handle load instruction which is following the bitcast in the pattern
      else if (auto *LI = dyn_cast<LoadInst>(U)) {
        Value *LocalValue = Builder.CreateLoad(NewVecTy, NewInst);
        Changed |= lowerBitCastToNonStdVec(
            LI, LocalValue, OldVecTy, InstsToErase, Builder, RecursionDepth);
      }
      // Handle extractelement instruction which is following the load
      else if (auto *EEI = dyn_cast<ExtractElementInst>(U)) {
        uint64_t NumElemsInOldVec = OldVecTy->getElementCount().getFixedValue();
        uint64_t NumElemsInNewVec = NewVecTy->getElementCount().getFixedValue();
        uint64_t OldElemIdx =
            cast<ConstantInt>(EEI->getIndexOperand())->getZExtValue();
        uint64_t NewElemIdx =
            OldElemIdx / (NumElemsInOldVec / NumElemsInNewVec);
        Value *LocalValue = Builder.CreateExtractElement(NewInst, NewElemIdx);
        // The trunc instruction truncates the high order bits in value, so it
        // may be necessary to shift right high order bits, if required bits are
        // not at the end of extracted value
        unsigned OldVecElemBitWidth =
            cast<IntegerType>(OldVecTy->getElementType())->getBitWidth();
        unsigned NewVecElemBitWidth =
            cast<IntegerType>(NewVecTy->getElementType())->getBitWidth();
        unsigned BitWidthRatio = NewVecElemBitWidth / OldVecElemBitWidth;
        if (auto RequiredBitsIdx =
                OldElemIdx % BitWidthRatio != BitWidthRatio - 1) {
          uint64_t Shift =
              OldVecElemBitWidth * (BitWidthRatio - RequiredBitsIdx);
          LocalValue = Builder.CreateLShr(LocalValue, Shift);
        }
        LocalValue =
            Builder.CreateTrunc(LocalValue, OldVecTy->getElementType());
        Changed |= lowerBitCastToNonStdVec(
            EEI, LocalValue, OldVecTy, InstsToErase, Builder, RecursionDepth);
      }
    }
  }
  InstsToErase.push_back(OldInst);
  if (!Changed)
    OldInst->replaceAllUsesWith(NewInst);
  return true;
}

class SPIRVLowerBitCastToNonStandardTypePass
    : public llvm::PassInfoMixin<SPIRVLowerBitCastToNonStandardTypePass> {
public:
  SPIRVLowerBitCastToNonStandardTypePass(const SPIRV::TranslatorOpts &Opts)
      : Opts(Opts) {}

  PreservedAnalyses
  runLowerBitCastToNonStandardType(Function &F, FunctionAnalysisManager &FAM) {
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

    std::vector<Instruction *> BCastsToNonStdVec;
    std::vector<Instruction *> InstsToErase;
    for (auto &BB : F)
      for (auto &I : BB) {
        auto *BC = dyn_cast<BitCastInst>(&I);
        if (!BC)
          continue;
        VectorType *SrcVecTy = getVectorType(BC->getSrcTy());
        if (SrcVecTy) {
          uint64_t NumElemsInSrcVec =
              SrcVecTy->getElementCount().getFixedValue();
          if (!isValidVectorSize(NumElemsInSrcVec))
            report_fatal_error(
                llvm::Twine("Unsupported vector type with the size of: " +
                            std::to_string(NumElemsInSrcVec)),
                false);
        }
        VectorType *DestVecTy = getVectorType(BC->getDestTy());
        if (DestVecTy) {
          uint64_t NumElemsInDestVec =
              DestVecTy->getElementCount().getFixedValue();
          if (!isValidVectorSize(NumElemsInDestVec))
            BCastsToNonStdVec.push_back(&I);
        }
      }
    IRBuilder<> Builder(F.getContext());
    for (auto &I : BCastsToNonStdVec) {
      Value *NewValue = I->getOperand(0);
      VectorType *OldVecTy = getVectorType(I->getType());
      Changed |=
          lowerBitCastToNonStdVec(I, NewValue, OldVecTy, InstsToErase, Builder);
    }

    for (auto *I : InstsToErase)
      I->eraseFromParent();

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }

private:
  SPIRV::TranslatorOpts Opts;
};

class SPIRVLowerBitCastToNonStandardTypeLegacy : public FunctionPass {
public:
  static char ID;
  SPIRVLowerBitCastToNonStandardTypeLegacy(const SPIRV::TranslatorOpts &Opts)
      : FunctionPass(ID), Opts(Opts) {}

  SPIRVLowerBitCastToNonStandardTypeLegacy() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    SPIRVLowerBitCastToNonStandardTypePass Impl(Opts);
    FunctionAnalysisManager FAM;
    auto PA = Impl.runLowerBitCastToNonStandardType(F, FAM);
    return !PA.areAllPreserved();
  }

  bool doFinalization(Module &M) override {
    verifyRegularizationPass(M, "SPIRVLowerBitCastToNonStandardType");
    return false;
  }

  StringRef getPassName() const override { return "Lower nonstandard type"; }

private:
  SPIRV::TranslatorOpts Opts;
};

char SPIRVLowerBitCastToNonStandardTypeLegacy::ID = 0;

} // namespace SPIRV

INITIALIZE_PASS(SPIRVLowerBitCastToNonStandardTypeLegacy,
                "spv-lower-bitcast-to-nonstandard-type",
                "Remove bitcast to nonstandard types", false, false)

llvm::FunctionPass *llvm::createSPIRVLowerBitCastToNonStandardTypeLegacy(
    const SPIRV::TranslatorOpts &Opts) {
  return new SPIRVLowerBitCastToNonStandardTypeLegacy(Opts);
}
