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

#include <llvm/IR/PassManager.h>
#include <llvm/Support/MathExtras.h>
#include <multi_llvm/vector_type_helper.h>

#include "analysis/stride_analysis.h"
#include "analysis/uniform_value_analysis.h"
#include "debugging.h"
#include "transform/packetization_helpers.h"
#include "transform/passes.h"

#define DEBUG_TYPE "vecz"

using namespace llvm;
using namespace vecz;

/// @brief replace loads of vectors of small vector loads and stores with scalar
/// loads and stores, where the entire vector fits into a legal integer.
///
/// The rationale here is that if we end up generating a scatter/gather, or
/// interleaved memop, it would be more efficient with the wider type than with
/// the vector of the narrower type. Although it's not trivial to know in
/// advance if we will get a scatter/gather or interleaved or contiguous load,
/// so we just do all of them and not worry too much about doing it when we
/// didn't really need to.
///
/// Be careful not to run Instruction Combine Pass between this pass and
/// packetization, because it is likely to undo it.
PreservedAnalyses SquashSmallVectorsPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {
  bool changed = false;

  const auto &UVR = AM.getResult<UniformValueAnalysis>(F);
  const auto &SAR = AM.getResult<StrideAnalysis>(F);
  auto &DL = F.getParent()->getDataLayout();
  auto &context = F.getContext();

  // Keep a cache of the bitcasts so we don't create multiple bitcasts for the
  // same value in each BasicBlock.
  DenseMap<const Value *, BitCastInst *> squashCasts;
  auto getSquashed = [&](Value *vector, Type *intTy,
                         IRBuilder<> &B) -> Value * {
    auto *&bitCast = squashCasts[vector];
    Value *element = bitCast;
    if (!element) {
      if (auto *const bcast = dyn_cast<BitCastInst>(vector)) {
        // "See through" existing bitcasts.
        element = bcast->getOperand(0);
      } else {
        element = vector;
      }

      if (element->getType() != intTy) {
        // Note we have to freeze the vector value first, because individual
        // elements can be `poison`, which would result in the entire value
        // becoming `poison`, which is not a valid transform (it is not valid to
        // increase the amount of `poison` in the IR).
        element = B.CreateBitCast(B.CreateFreeze(element), intTy,
                                  Twine(vector->getName(), ".squash"));
        bitCast = dyn_cast<BitCastInst>(element);
      }
    }
    return element;
  };

  SmallVector<Instruction *, 16> toErase;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *load = dyn_cast<LoadInst>(&I)) {
        if (!UVR.isVarying(load)) {
          continue;
        }

        auto *const ty = load->getType();
        auto *const scalarTy = ty->getScalarType();
        const unsigned numBits = ty->getPrimitiveSizeInBits();
        if ((numBits & (numBits - 1)) == 0 && scalarTy != ty &&
            DL.fitsInLegalInteger(numBits)) {
          const auto align = load->getAlign();
          auto *const intTy = IntegerType::get(context, numBits);
          if (DL.getABITypeAlign(intTy) > align) {
            // The alignment of this type is too strict to convert
            continue;
          }

          auto *const ptr = load->getPointerOperand();
          const auto *const info = SAR.getInfo(ptr);
          if (info && info->hasStride() &&
              info->getConstantMemoryStride(ty, &DL) == 1) {
            // No need to perform this transform on contiguous loads
            continue;
          }

          IRBuilder<> B(load);
          const auto name = load->getName();
          auto *const newPtrTy =
              PointerType::get(intTy, ptr->getType()->getPointerAddressSpace());
          auto *const ptrCast = B.CreatePointerCast(
              ptr, newPtrTy, Twine(ptr->getName(), ".squashptr"));
          auto *newLoad = cast<LoadInst>(
              B.CreateLoad(intTy, ptrCast, Twine(name, ".squashed")));
          newLoad->setAlignment(align);
          newLoad->copyMetadata(*load);

          auto *const newVec =
              B.CreateBitCast(newLoad, ty, Twine(name, ".unsquash"));

          load->replaceAllUsesWith(newVec);
          toErase.push_back(load);
          changed = true;
        }
      } else if (auto *store = dyn_cast<StoreInst>(&I)) {
        if (!UVR.isVarying(store)) {
          continue;
        }

        auto *const data = store->getValueOperand();
        auto *const ty = data->getType();
        auto *const scalarTy = ty->getScalarType();
        const unsigned numBits = ty->getPrimitiveSizeInBits();
        if ((numBits & (numBits - 1)) == 0 && scalarTy != ty &&
            DL.fitsInLegalInteger(numBits)) {
          const auto align = store->getAlign();
          auto *const intTy = IntegerType::get(context, numBits);
          if (DL.getABITypeAlign(intTy) > align) {
            // The alignment of this type is too strict to convert
            continue;
          }

          auto *const ptr = store->getPointerOperand();
          const auto *const info = SAR.getInfo(ptr);
          if (info && info->hasStride() &&
              info->getConstantMemoryStride(ty, &DL) == 1) {
            // No need to perform this transform on contiguous stores
            continue;
          }

          IRBuilder<> B(store);
          auto *const newPtrTy =
              PointerType::get(intTy, ptr->getType()->getPointerAddressSpace());
          auto *const newPtr = B.CreatePointerCast(
              ptr, newPtrTy, Twine(ptr->getName(), ".squashptr"));
          auto *const newData = getSquashed(data, intTy, B);
          auto *newStore = cast<StoreInst>(B.CreateStore(newData, newPtr));
          newStore->setAlignment(align);
          newStore->copyMetadata(*store);

          toErase.push_back(store);
          changed = true;
        }
      } else if (auto *zext = dyn_cast<ZExtInst>(&I)) {
        if (!UVR.isVarying(zext)) {
          continue;
        }
        // A zero-extend of an extract element can be squashed, if the source
        // vector size is the same as the extended integer size. That is (for
        // little-endian systems):
        //
        //   zext i32(extract <4 x i8> data, i32 3)
        //
        // becomes:
        //
        //   and(lshr(bitcast i32 data), i32 24), 0xFF)
        //
        // this avoids creating shufflevectors during packetization.
        //
        // We limit this optimization to vectors no larger than 64 bits in
        // size. This is primarily because this optimization focuses on 'small'
        // vectors but also, because LLVM's constants are limited to 64-bit
        // integers, the masking logic would need to be done with extra
        // instructions.
        auto *const srcOp = zext->getOperand(0);
        if (auto *const extract = dyn_cast<ExtractElementInst>(srcOp)) {
          auto *const vector = extract->getVectorOperand();
          auto *const indexOp = extract->getIndexOperand();
          auto *const intTy = zext->getType();
          auto *const vecTy = vector->getType();
          if (vecTy->getPrimitiveSizeInBits() ==
                  intTy->getPrimitiveSizeInBits() &&
              zext->getSrcTy()->getPrimitiveSizeInBits() <= 32 &&
              intTy->getScalarSizeInBits() <= 64 && isa<ConstantInt>(indexOp)) {
            IRBuilder<> B(zext);
            Value *element = getSquashed(vector, intTy, B);

            const auto bits = zext->getSrcTy()->getScalarSizeInBits();
            const auto scaled =
                cast<ConstantInt>(indexOp)->getZExtValue() * bits;

            // Note on Little Endian systems, element 0 occupies the least
            // significant bits of the vector. On Big Endian systems it occupies
            // the most significant bits. Thus, we shift by "maximum element
            // number minus current element number" times by "number of bits
            // per element".
            const auto shift =
                DL.isBigEndian()
                    ? intTy->getPrimitiveSizeInBits() - bits - scaled
                    : scaled;

            if (shift != 0) {
              element =
                  B.CreateLShr(element, ConstantInt::get(intTy, shift),
                               Twine(extract->getName(), ".squashExtract"));
            }
            element = B.CreateAnd(
                element,
                ConstantInt::get(intTy, maskTrailingOnes<uint64_t>(bits)),
                Twine(zext->getName(), ".squashZExt"));

            zext->replaceAllUsesWith(element);
            toErase.push_back(zext);
            changed = true;
          }
        }
      } else if (auto *sext = dyn_cast<SExtInst>(&I)) {
        if (!UVR.isVarying(sext)) {
          continue;
        }
        // We can squash sign extends in-place as well.
        // We do this by shifting the required element into most-significant
        // position, and then arithmetic-shifting it back down to the least-
        // significant position.
        auto *const srcOp = sext->getOperand(0);
        if (auto *const extract = dyn_cast<ExtractElementInst>(srcOp)) {
          auto *const vector = extract->getVectorOperand();
          auto *const indexOp = extract->getIndexOperand();
          auto *const intTy = sext->getType();
          auto *const vecTy = vector->getType();
          if (vecTy->getPrimitiveSizeInBits() ==
                  intTy->getPrimitiveSizeInBits() &&
              isa<ConstantInt>(indexOp)) {
            IRBuilder<> B(sext);
            Value *element = getSquashed(vector, intTy, B);

            const auto bits = sext->getSrcTy()->getScalarSizeInBits();
            const auto shiftr = intTy->getPrimitiveSizeInBits() - bits;
            const auto scaled =
                cast<ConstantInt>(indexOp)->getZExtValue() * bits;
            const auto shiftl = DL.isBigEndian() ? scaled : shiftr - scaled;

            if (shiftl != 0) {
              element =
                  B.CreateShl(element, ConstantInt::get(intTy, shiftl),
                              Twine(extract->getName(), ".squashExtract"));
            }
            element = B.CreateAShr(element, ConstantInt::get(intTy, shiftr),
                                   Twine(extract->getName(), ".squashSExt"));

            sext->replaceAllUsesWith(element);
            toErase.push_back(sext);
            changed = true;
          }
        }
      }
    }

    // only re-use casts within a basic block
    squashCasts.clear();
  }

  for (auto *I : toErase) {
    I->eraseFromParent();
  }

  auto preserved = PreservedAnalyses::all();
  if (changed) {
    preserved.abandon<UniformValueAnalysis>();
    preserved.abandon<StrideAnalysis>();
  }
  return preserved;
}
