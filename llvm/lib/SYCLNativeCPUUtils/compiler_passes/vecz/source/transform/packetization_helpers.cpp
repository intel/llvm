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
//
// This file contains all the code to perform, on demand, the plumbing between
// values that have been vectorized, vector-widened, instantiated, or
// semi-widened/instantiated (otherwise known as Vector Sub-Widening),
// including the broadcast of uniform values, scatters, gathers, vector splits
// and concatenations.

#include "transform/packetization_helpers.h"

#include <compiler/utils/group_collective_helpers.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Analysis/VectorUtils.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Transforms/Utils/LoopUtils.h>
#include <multi_llvm/multi_llvm.h>
#include <multi_llvm/vector_type_helper.h>

#include "debugging.h"
#include "transform/packetizer.h"
#include "vectorization_context.h"
#include "vectorization_unit.h"
#include "vecz/vecz_target_info.h"

#define DEBUG_TYPE "vecz-packetization"

using namespace llvm;
using namespace vecz;

namespace {
inline Type *getWideType(Type *ty, ElementCount factor) {
  if (!ty->isVectorTy()) {
    // The wide type of a struct literal is the wide type of each of its
    // elements.
    if (auto *structTy = dyn_cast<StructType>(ty);
        structTy && structTy->isLiteral()) {
      SmallVector<Type *, 4> wideElts(structTy->elements());
      for (unsigned i = 0, e = wideElts.size(); i != e; i++) {
        wideElts[i] = getWideType(wideElts[i], factor);
      }
      return StructType::get(ty->getContext(), wideElts);
    } else if (structTy) {
      VECZ_ERROR("Can't create wide type for structure type");
    }
    return VectorType::get(ty, factor);
  }
  const bool isScalable = isa<ScalableVectorType>(ty);
  assert((!factor.isScalable() || !isScalable) &&
         "Can't widen a scalable vector by a scalable amount");
  auto *vecTy = cast<llvm::VectorType>(ty);
  const unsigned elts = vecTy->getElementCount().getKnownMinValue();
  // If we're widening a scalable type then set the fixed factor to scalable
  // here.
  if (isScalable && !factor.isScalable()) {
    factor = ElementCount::getScalable(factor.getKnownMinValue());
  }
  ty = vecTy->getElementType();
  return VectorType::get(ty, factor * elts);
}

Value *scalableBroadcastHelper(Value *subvec, ElementCount factor,
                               const vecz::TargetInfo &TI, IRBuilder<> &B,
                               bool URem);

// Helper to broadcast a fixed vector thus:
// <A,B> -> vscale x 1 -> <A,B,A,B,A,B,...>
Value *createScalableBroadcastOfFixedVector(const vecz::TargetInfo &TI,
                                            IRBuilder<> &B, Value *subvec,
                                            ElementCount factor) {
  assert(factor.isScalable());
  return scalableBroadcastHelper(subvec, factor, TI, B, /*URem*/ true);
}

// Helper to broadcast a scalable vector thus:
// <A,B,C, ...> -> x 2 <A,A,B,B,C,C, ...>
Value *createFixedBroadcastOfScalableVector(const vecz::TargetInfo &TI,
                                            IRBuilder<> &B, Value *subvec,
                                            ElementCount factor) {
  assert(!factor.isScalable());
  return scalableBroadcastHelper(subvec, factor, TI, B, /*URem*/ false);
}
}  // namespace

namespace vecz {
Instruction *buildAfter(Value *V, Function &F, bool IsPhi) {
  if (auto *const I = dyn_cast<Instruction>(V)) {
    BasicBlock::iterator Next = I->getIterator();
    const BasicBlock::iterator End = Next->getParent()->end();
    do {
      ++Next;
    } while (!IsPhi && (Next != End) &&
             (isa<PHINode>(Next) || isa<AllocaInst>(Next)));
    return &*Next;
  }
  // Else find the first point in the function after any allocas.
  auto it = F.getEntryBlock().begin();
  while (isa<AllocaInst>(*it)) {
    ++it;
  }
  return &*it;
}

Constant *getShuffleMask(ShuffleVectorInst *shuffle) {
  // The mask value seems not to be a proper operand for LLVM 11.
  // NOTE this is marked as "temporary" in the docs!
  return shuffle->getShuffleMaskForBitcode();
}

Value *createOptimalShuffle(IRBuilder<> &B, Value *srcA, Value *srcB,
                            const SmallVectorImpl<int> &mask,
                            const Twine &name) {
  const auto &maskC = mask;
  auto *shuffleA = dyn_cast<ShuffleVectorInst>(srcA);
  // If we have a unary shuffle of a shuffle, we can just pre-shuffle the masks
  if (shuffleA && isa<UndefValue>(srcB)) {
    auto *const srcMask = getShuffleMask(shuffleA);
    auto *const newMask = ConstantExpr::getShuffleVector(
        srcMask, UndefValue::get(srcMask->getType()), maskC);

    return B.CreateShuffleVector(shuffleA->getOperand(0),
                                 shuffleA->getOperand(1), newMask, name);
  }

  auto *shuffleB = dyn_cast<ShuffleVectorInst>(srcB);

  if (shuffleA && shuffleB) {
    auto *const shuffleSrcA = shuffleA->getOperand(0);
    auto *const shuffleSrcB = shuffleA->getOperand(1);

    // If we have a shuffle of two shuffles with identical source operands,
    // we can just pre-shuffle their masks together.
    if (shuffleB->getOperand(0) == shuffleSrcA &&
        shuffleB->getOperand(1) == shuffleSrcB) {
      auto *const srcMaskA = getShuffleMask(shuffleA);
      auto *const srcMaskB = getShuffleMask(shuffleB);
      auto *const newMask =
          ConstantExpr::getShuffleVector(srcMaskA, srcMaskB, maskC);

      return B.CreateShuffleVector(shuffleSrcA, shuffleSrcB, newMask, name);
    }
  }

  // If either operand is a unary shuffle, we can pull a few more tricks..
  // For instance:
  //
  //    shuffle(shuffle(A, undef, maskA), shuffle(B, undef, maskB), maskC)
  // => shuffle(A, B, shuffle(maskA, adjust(maskB), maskC))
  // where "adjust" refers to adjusting the mask values to refer to the second
  // source vector by adding the width of the first operand to the indices.
  //
  // If either source operand is something other than a unary shuffle, we can
  // "pretend" it is a NOP shuffle of that operand (i.e. a mask of <0, 1, 2..>)
  // and proceed as before, absorbing the unary shuffle from the other operand.
  if (shuffleA && !isa<UndefValue>(shuffleA->getOperand(1))) {
    shuffleA = nullptr;
  }
  if (shuffleB && !isa<UndefValue>(shuffleB->getOperand(1))) {
    shuffleB = nullptr;
  }

  if (shuffleA || shuffleB) {
    // We can absorb one or two unary shuffles into the new shuffle..
    auto *const shuffleAsrc = shuffleA ? shuffleA->getOperand(0) : srcA;
    auto *const shuffleBsrc = shuffleB ? shuffleB->getOperand(0) : srcB;
    const auto srcASize =
        cast<FixedVectorType>(shuffleAsrc->getType())->getNumElements();
    const auto srcBSize =
        cast<FixedVectorType>(shuffleBsrc->getType())->getNumElements();
    if (srcASize == srcBSize) {
      Constant *srcMaskA = nullptr;
      Constant *srcMaskB = nullptr;

      if (shuffleA) {
        srcMaskA = getShuffleMask(shuffleA);
      } else {
        // if one operand is not a shuffle, we can make a pretend shuffle..
        SmallVector<Constant *, 16> newMaskA;
        for (unsigned i = 0; i < srcASize; ++i) {
          newMaskA.push_back(B.getInt32(i));
        }
        srcMaskA = ConstantVector::get(newMaskA);
      }

      if (shuffleB) {
        auto *const maskB = getShuffleMask(shuffleB);

        // adjust the second mask to refer to the second vector..
        srcMaskB = ConstantExpr::getAdd(
            maskB, ConstantVector::getSplat(
                       multi_llvm::getVectorElementCount(maskB->getType()),
                       B.getInt32(srcASize)));
      } else {
        // if one operand is not a shuffle, we can make a pretend shuffle..
        SmallVector<Constant *, 16> newMaskB;
        for (unsigned i = 0; i < srcBSize; ++i) {
          newMaskB.push_back(B.getInt32(i + srcASize));
        }
        srcMaskB = ConstantVector::get(newMaskB);
      }

      auto *const newMask =
          ConstantExpr::getShuffleVector(srcMaskA, srcMaskB, maskC);

      return B.CreateShuffleVector(shuffleAsrc, shuffleBsrc, newMask, name);
    }
  }

  // No more optimal alternative, just build a new one
  return B.CreateShuffleVector(srcA, srcB, maskC, name);
}

bool createSubSplats(const vecz::TargetInfo &TI, IRBuilder<> &B,
                     SmallVectorImpl<Value *> &srcs, unsigned subWidth) {
  // Scalable sub-splats must be handled specially.
  if (isa<ScalableVectorType>(srcs.front()->getType())) {
    if (srcs.size() != 1) {
      return false;
    }
    Value *&val = srcs.front();
    val = createFixedBroadcastOfScalableVector(
        TI, B, val, ElementCount::getFixed(subWidth));
    return val != nullptr;
  }

  auto *const vecTy = dyn_cast<FixedVectorType>(srcs.front()->getType());

  if (!vecTy) {
    return false;
  }

  const unsigned srcWidth = vecTy->getNumElements();

  // Build shuffle mask to widen the vector condition.
  SmallVector<int, 16> mask;
  for (unsigned i = 0; i < srcWidth; ++i) {
    for (unsigned j = 0; j < subWidth; ++j) {
      mask.push_back(i);
    }
  }

  auto *undef = UndefValue::get(srcs.front()->getType());
  for (auto &src : srcs) {
    src = createOptimalShuffle(B, src, undef, mask);
  }
  return true;
}

Value *createMaybeVPTargetReduction(IRBuilderBase &B,
                                    const TargetTransformInfo &TTI, Value *Val,
                                    RecurKind Kind, Value *VL) {
  assert(isa<VectorType>(Val->getType()) && "Must be vector type");
  // If VL is null, it's not a vector-predicated reduction.
  if (!VL) {
    return multi_llvm::createSimpleTargetReduction(B, &TTI, Val, Kind);
  }
  auto IntrinsicOp = Intrinsic::not_intrinsic;
  switch (Kind) {
    default:
      break;
    case RecurKind::None:
      return nullptr;
    case RecurKind::Add:
      IntrinsicOp = Intrinsic::vp_reduce_add;
      break;
    case RecurKind::Mul:
      IntrinsicOp = Intrinsic::vp_reduce_mul;
      break;
    case RecurKind::Or:
      IntrinsicOp = Intrinsic::vp_reduce_or;
      break;
    case RecurKind::And:
      IntrinsicOp = Intrinsic::vp_reduce_and;
      break;
    case RecurKind::Xor:
      IntrinsicOp = Intrinsic::vp_reduce_xor;
      break;
    case RecurKind::FAdd:
      IntrinsicOp = Intrinsic::vp_reduce_fadd;
      break;
    case RecurKind::FMul:
      IntrinsicOp = Intrinsic::vp_reduce_fmul;
      break;
    case RecurKind::SMin:
      IntrinsicOp = Intrinsic::vp_reduce_smin;
      break;
    case RecurKind::SMax:
      IntrinsicOp = Intrinsic::vp_reduce_smax;
      break;
    case RecurKind::UMin:
      IntrinsicOp = Intrinsic::vp_reduce_umin;
      break;
    case RecurKind::UMax:
      IntrinsicOp = Intrinsic::vp_reduce_umax;
      break;
    case RecurKind::FMin:
      IntrinsicOp = Intrinsic::vp_reduce_fmin;
      break;
    case RecurKind::FMax:
      IntrinsicOp = Intrinsic::vp_reduce_fmax;
      break;
  }

  auto *const F = Intrinsic::getDeclaration(B.GetInsertBlock()->getModule(),
                                            IntrinsicOp, Val->getType());
  assert(F && "Could not declare vector-predicated reduction intrinsic");

  auto *const VecTy = cast<VectorType>(Val->getType());
  auto *const NeutralVal =
      compiler::utils::getNeutralVal(Kind, VecTy->getElementType());
  auto *const Mask = createAllTrueMask(B, VecTy->getElementCount());
  return B.CreateCall(F, {NeutralVal, Val, Mask, VL});
}

Value *getGatherIndicesVector(IRBuilder<> &B, Value *Indices, Type *Ty,
                              unsigned FixedVecElts, const Twine &N) {
  auto *const Steps = B.CreateStepVector(Ty);

  const auto EltCount = multi_llvm::getVectorElementCount(Ty);
  auto *const ElTy = multi_llvm::getVectorElementType(Ty);

  auto *const FixedVecEltsSplat =
      B.CreateVectorSplat(EltCount, ConstantInt::get(ElTy, FixedVecElts));
  auto *const StepsMul = B.CreateMul(Steps, FixedVecEltsSplat);
  return B.CreateAdd(StepsMul, Indices, N);
}

Value *createAllTrueMask(IRBuilderBase &B, ElementCount EC) {
  return ConstantInt::getTrue(VectorType::get(B.getInt1Ty(), EC));
}

Value *createIndexSequence(IRBuilder<> &Builder, VectorType *VecTy,
                           const Twine &Name) {
  auto EC = VecTy->getElementCount();
  if (EC.isScalable()) {
    // FIXME: This intrinsic works on fixed-length types too: should we migrate
    // to using it starting from LLVM 13?
    return Builder.CreateStepVector(VecTy, Name);
  }

  SmallVector<Constant *, 16> Indices;
  auto *EltTy = VecTy->getElementType();
  for (unsigned i = 0, e = EC.getFixedValue(); i != e; i++) {
    Indices.push_back(ConstantInt::get(EltTy, i));
  }
  return ConstantVector::get(Indices);
}

}  // namespace vecz

PacketRange PacketInfo::getRange(std::vector<llvm::Value *> &d,
                                 unsigned width) const {
  auto found = packets.find(width);
  if (found != packets.end()) {
    return PacketRange(d, found->second, width);
  } else {
    return PacketRange(d);
  }
}

Value *Packetizer::Result::getAsValue() const {
  if (!scalar || !info) {
    return nullptr;
  }

  if (info->vector) {
    return info->vector;
  }

  const auto numInstances = info->numInstances;
  if (numInstances == 0) {
    return broadcast(1).info->vector;
  }

  const auto packet = getRange(numInstances);
  assert(packet && "Packet doesn't exist when it should");

  // If the instantiator broadcast the value, it will have set its own packet,
  // so we fix that here.
  bool splat = true;
  for (auto *v : packet) {
    if (v != scalar) {
      splat = false;
      break;
    }
  }

  if (splat) {
    info->numInstances = 0;
    return broadcast(1).info->vector;
  }

  Type *const eleTy = packet.front()->getType();
  assert(!eleTy->isVoidTy() && "Should not be getting a vector of voids");

  auto name = scalar->getName();

  if (FixedVectorType::isValidElementType(eleTy)) {
    Value *gather = UndefValue::get(FixedVectorType::get(eleTy, packet.size()));

    IRBuilder<> B(buildAfter(packet.back(), packetizer.F));
    for (unsigned i = 0; i < packet.size(); i++) {
      gather = B.CreateInsertElement(gather, packet.at(i), B.getInt32(i),
                                     Twine(name, ".gather"));
    }
    info->vector = gather;
  } else if (eleTy->isVectorTy()) {
    // Gathering an instantiated vector by concatenating all the lanes
    auto parts = narrow(2);
    auto *vecTy = cast<FixedVectorType>(parts.front()->getType());
    const unsigned fullWidth = vecTy->getNumElements() * 2;

    SmallVector<int, 16> mask;
    for (size_t j = 0; j < fullWidth; ++j) {
      mask.push_back(j);
    }

    IRBuilder<> B(buildAfter(parts[1], packetizer.F));
    info->vector = B.CreateShuffleVector(parts[0], parts[1], mask,
                                         Twine(name, ".concatenate"));
  } else {
    Value *gather = UndefValue::get(ArrayType::get(eleTy, packet.size()));

    IRBuilder<> B(buildAfter(packet.back(), packetizer.F));
    for (unsigned i = 0; i < packet.size(); i++) {
      gather =
          B.CreateInsertValue(gather, packet.at(i), i, Twine(name, ".gather"));
    }
    info->vector = gather;
  }
  return info->vector;
}

PacketRange Packetizer::Result::getAsPacket(unsigned width) const {
  if (!scalar || !info) {
    return PacketRange(packetizer.packetData);
  }

  if (const auto range = getRange(width)) {
    return range;
  }

  auto numInstances = info->numInstances;
  if (numInstances == 0) {
    return broadcast(width).getRange(width);
  }

  if (numInstances != 1) {
    if (numInstances < width) {
      return widen(width);
    } else if (numInstances > width) {
      return narrow(width);
    } else {
      assert(false && "Supposedly unreachable condition in Packetizer::Result");
    }
  }

  if (!info->vector) {
    return PacketRange(packetizer.packetData);
  }

  auto packet = createPacket(width);

  Value *vec = info->vector;
  if (auto *const vecTy = dyn_cast<FixedVectorType>(vec->getType())) {
    assert(isa<FixedVectorType>(vecTy) && "Must be a fixed vector type here!");
    const unsigned scalarWidth = vecTy->getNumElements() / width;
    if (scalarWidth > 1) {
      auto *const undef = UndefValue::get(vec->getType());

      // Build shuffle mask to perform the subvector extracts.
      IRBuilder<> B(buildAfter(vec, packetizer.F));
      for (size_t i = 0, k = 0; i < width; ++i) {
        SmallVector<int, 16> mask;
        for (size_t j = 0; j < scalarWidth; ++j, ++k) {
          mask.push_back(k);
        }
        packet[i] = createOptimalShuffle(B, vec, undef, mask,
                                         Twine(scalar->getName(), ".split"));
      }
    } else {
      IRBuilder<> B(buildAfter(vec, packetizer.F));
      for (unsigned i = 0; i < width; i++) {
        packet[i] = B.CreateExtractElement(vec, B.getInt32(i));
      }
    }
  } else {
    assert(isa<ArrayType>(vecTy) && "Must be an array here!");
    IRBuilder<> B(buildAfter(vec, packetizer.F));
    for (unsigned i = 0; i < width; i++) {
      packet[i] = B.CreateExtractValue(vec, i);
    }
  }
  return packet;
}

void Packetizer::Result::getPacketValues(SmallVectorImpl<Value *> &vals) const {
  assert(info && "No packet info for this packetization result");
  const auto width = info->numInstances;
  if (width != 0) {
    return getPacketValues(width, vals);
  }
}

void Packetizer::Result::getPacketValues(unsigned width,
                                         SmallVectorImpl<Value *> &vals) const {
  assert(width != 0 && "Can't get a zero width packet");
  if (width == 1) {
    if (auto *const val = getAsValue()) {
      vals.push_back(val);
    }
  } else {
    auto p = getAsPacket(width);
    vals.assign(p.begin(), p.end());
  }
}

PacketRange Packetizer::Result::createPacket(unsigned width) const {
  assert(info && "Can't create a packet on a fail state");
  assert(info->packets.count(width) == 0 &&
         "Shouldn't create the same packet twice");

  const auto start = packetizer.packetData.size();
  packetizer.packetData.resize(start + width, nullptr);
  info->packets[width] = start;
  return PacketRange(packetizer.packetData, start, width);
}

PacketRange Packetizer::Result::getRange(unsigned width) const {
  return info->getRange(packetizer.packetData, width);
}

// it makes a wider packet by splitting the sub-vectors
PacketRange Packetizer::Result::widen(unsigned width) const {
  const auto numInstances = info->numInstances;
  const auto parts = getRange(numInstances);
  auto *const vecTy = dyn_cast<FixedVectorType>(parts.front()->getType());
  assert(vecTy && "Expected a fixed vector type");

  auto packet = createPacket(width);
  const auto origWidth = vecTy->getNumElements();
  const auto newWidth = (origWidth * numInstances) / width;
  const auto name = scalar->getName();

  auto *it = parts.begin();
  IRBuilder<> B(buildAfter(parts.back(), packetizer.F));
  if (newWidth > 1) {
    auto *const undef = UndefValue::get(vecTy);

    // Build shuffle mask to perform the subvector extracts.
    for (size_t i = 0, origIdx = 0; i < width; ++i) {
      if (origIdx == origWidth) {
        origIdx = 0;
        ++it;
      }
      SmallVector<int, 16> mask;
      for (size_t j = 0; j < newWidth; ++j, ++origIdx) {
        mask.push_back(origIdx);
      }
      packet[i] =
          createOptimalShuffle(B, *it, undef, mask, Twine(name, ".split"));
    }
  } else {
    for (size_t i = 0, origIdx = 0; i < width; ++i, ++origIdx) {
      if (origIdx == origWidth) {
        origIdx = 0;
        ++it;
      }
      packet[i] = B.CreateExtractElement(*it, B.getInt32(origIdx),
                                         Twine(name, ".split"));
    }
  }
  return packet;
}

// it makes a narrower packet by concatenating the sub-vectors
PacketRange Packetizer::Result::narrow(unsigned width) const {
  if (const auto range = getRange(width)) {
    return range;
  }

  // Narrow recursively
  const auto parts = narrow(width * 2);
  assert(parts && "Error during packet narrowing");

  auto packet = createPacket(width);
  auto *const ty = parts.front()->getType();
  auto *const vecTy = dyn_cast<FixedVectorType>(ty);
  if (!vecTy) {
    // Build vectors out of pairs of scalar values
    const auto name = scalar->getName();
    IRBuilder<> B(buildAfter(parts.back(), packetizer.F));
    Value *undef = UndefValue::get(FixedVectorType::get(ty, 2));
    for (size_t i = 0, pairIdx = 0; i < width; ++i, pairIdx += 2) {
      Value *in = B.CreateInsertElement(undef, parts[pairIdx], B.getInt32(0),
                                        Twine(name, ".gather"));
      packet[i] = B.CreateInsertElement(in, parts[pairIdx + 1], B.getInt32(1),
                                        Twine(name, ".gather"));
    }
    return packet;
  }

  const unsigned fullWidth = vecTy->getNumElements() * 2;

  SmallVector<int, 16> mask;
  for (size_t j = 0; j < fullWidth; ++j) {
    mask.push_back(j);
  }

  // Build wider vectors by concatenating pairs of sub-vectors
  const auto name = scalar->getName();
  IRBuilder<> B(buildAfter(parts.back(), packetizer.F));
  for (size_t i = 0, pairIdx = 0; i < width; ++i, pairIdx += 2) {
    packet[i] = createOptimalShuffle(B, parts[pairIdx], parts[pairIdx + 1],
                                     mask, Twine(name, ".concatenate"));
  }
  return packet;
}

namespace {
// This method creates the following sequence to broadcast a fixed-length
// vector to a scalable one or broadcasting a scalable-vector by a fixed
// amount, barring any optimizations we can perform for broadcasting a splat
// vector.
// The general idea is first to store the subvector to a stack 'alloca', then
// use a gather operation with a vector of pointers created using a step vector
// modulo the fixed amount.
// Note that other sequences are possible, such as a series of blend
// operations. This could perhaps be a target choice.
Value *scalableBroadcastHelper(Value *subvec, ElementCount factor,
                               const vecz::TargetInfo &TI, IRBuilder<> &B,
                               bool URem) {
  auto *ty = subvec->getType();
  const auto subVecEltCount = multi_llvm::getVectorElementCount(ty);
  assert(subVecEltCount.isScalable() ^ factor.isScalable() &&
         "Must either broadcast fixed vector by scalable factor or scalable "
         "vector by fixed factor");
  auto *const wideTy = getWideType(ty, factor);
  auto wideEltCount = multi_llvm::getVectorElementCount(wideTy);

  // If this vector is a constant splat, just splat it to the wider scalable
  // type.
  if (auto *const cvec = dyn_cast<Constant>(subvec)) {
    if (auto *const splat = cvec->getSplatValue()) {
      return ConstantVector::getSplat(wideEltCount, splat);
    }
  }
  // Or if it's a splat value, re-splat it. Note we do Constants separately
  // above as it generates more canonical code, e.g., a splat of 0 becomes
  // zeroinitializer rather than a insertelement/shufflevector sequence.
  if (const auto *const splat = getSplatValue(subvec)) {
    return B.CreateVectorSplat(wideEltCount, const_cast<Value *>(splat));
  }

  // Compiler support for masked.gather on i1 vectors is lacking, so emit this
  // operation as the equivalent i8 vector instead.
  const bool upcast_i1_as_i8 = ty->getScalarType()->isIntegerTy(1);
  if (upcast_i1_as_i8) {
    auto *const int8Ty = Type::getInt8Ty(B.getContext());
    ty = llvm::VectorType::get(int8Ty, subVecEltCount);
    subvec = B.CreateSExt(subvec, ty);
  }

  Value *gather =
      URem ? TI.createOuterScalableBroadcast(B, subvec, /*VL*/ nullptr, factor)
           : TI.createInnerScalableBroadcast(B, subvec, /*VL*/ nullptr, factor);

  // If we've been performing this broadcast as i8, now's the time to truncate
  // back down to i1.
  if (upcast_i1_as_i8) {
    gather = B.CreateTrunc(gather, wideTy);
  }

  return gather;
}
}  // namespace

const Packetizer::Result &Packetizer::Result::broadcast(unsigned width) const {
  const auto factor = packetizer.width().divideCoefficientBy(width);
  auto *const ty = scalar->getType();
  assert(!ty->isVoidTy() && "Should not be broadcasting a void type");

  if (width != 1 && !factor.isScalable() && factor.getFixedValue() == 1) {
    // Pure instantiation broadcast..
    for (auto &v : createPacket(width)) {
      v = scalar;
    }
    return *this;
  }

  auto &F = packetizer.F;
  Value *result = nullptr;
  const auto &TI = packetizer.context().targetInfo();
  if (isa<PoisonValue>(scalar)) {
    result = PoisonValue::get(getWideType(ty, factor));
  } else if (isa<UndefValue>(scalar)) {
    result = UndefValue::get(getWideType(ty, factor));
  } else if (ty->isVectorTy() && factor.isScalable()) {
    IRBuilder<> B(buildAfter(scalar, F));
    result = createScalableBroadcastOfFixedVector(TI, B, scalar, factor);
  } else if (ty->isVectorTy()) {
    auto *const vecTy = cast<FixedVectorType>(ty);
    const unsigned scalarWidth = vecTy->getNumElements();

    const unsigned simdWidth = factor.getFixedValue();

    // Build shuffle mask to perform the splat.
    SmallVector<int, 16> mask;
    for (size_t i = 0; i < simdWidth; ++i) {
      for (size_t j = 0; j < scalarWidth; ++j) {
        mask.push_back(j);
      }
    }

    IRBuilder<> B(buildAfter(scalar, packetizer.F));
    result = createOptimalShuffle(B, scalar, UndefValue::get(ty), mask,
                                  Twine(scalar->getName(), ".broadcast"));
  } else if (auto *const C = dyn_cast<Constant>(scalar)) {
    result = ConstantVector::getSplat(factor, C);
  } else {
    IRBuilder<> B(buildAfter(scalar, packetizer.F));
    result = B.CreateVectorSplat(factor, scalar);
  }

  if (!result) {
    // Failed to broadcast this value, return the empty result
    return *this;
  }

  if (width == 1) {
    info->vector = result;
  } else {
    for (auto &v : createPacket(width)) {
      v = result;
    }
  }
  return *this;
}
