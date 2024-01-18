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

#include "offset_info.h"

#include <compiler/utils/builtin_info.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/KnownBits.h>

#include "analysis/instantiation_analysis.h"
#include "analysis/stride_analysis.h"
#include "analysis/uniform_value_analysis.h"
#include "debugging.h"
#include "memory_operations.h"
#include "vectorization_context.h"
#include "vectorization_unit.h"

using namespace vecz;
using namespace llvm;

namespace {
inline uint64_t SizeOrZero(TypeSize &&T) {
  return T.isScalable() ? 0 : T.getFixedValue();
}

uint8_t highbit(const uint32_t x) {
  assert((x & (x - 1)) == 0 && "Value must be a power of two");
  // This is a De Bruijn hash table, it returns the index of the highest
  // bit, which works when x is a power of 2. For details, see
  // https://en.wikipedia.org/wiki/De_Bruijn_sequence#Uses
  static const uint32_t deBruijn_magic = 0x06EB14F9U;
  static const uint8_t tab[32] = {
      0,  1,  16, 2,  29, 17, 3,  22, 30, 20, 18, 11, 13, 4, 7,  23,
      31, 15, 28, 21, 19, 10, 12, 6,  14, 27, 9,  5,  26, 8, 25, 24,
  };
  return tab[(uint32_t)(x * deBruijn_magic) >> 27];
};

// Returns a value extended or truncated to match the size type of the target.
// This will return the original value if it is already the correct size.
Value *matchSizeType(IRBuilder<> &B, Value *V, bool sext) {
  auto *const sizeTy = getSizeTy(B);

  if (sext) {
    return B.CreateSExtOrTrunc(V, sizeTy, "stride_conv");
  } else {
    return B.CreateZExtOrTrunc(V, sizeTy, "stride_conv");
  }
}

uint64_t getTypeMask(Type *Ty) {
  const auto bits = Ty->getIntegerBitWidth();
  return bits < 64 ? ((uint64_t(1) << bits) - 1) : ~uint64_t(0);
}

// The index size potentially depends on the address space of the pointer,
// but let's just use the pointer size for now.
uint64_t getSizeTypeMask(const DataLayout &DL) {
  const auto bits = DL.getPointerSizeInBits();
  return bits < 64 ? ((uint64_t(1) << bits) - 1) : ~uint64_t(0);
}

OffsetKind combineKinds(OffsetKind LHS, OffsetKind RHS) {
  assert(LHS != eOffsetLinear && RHS != eOffsetLinear &&
         "OffsetInfo analysis functions should handle all linear cases");

  if (LHS == RHS) {
    return LHS;
  }

  if (LHS == eOffsetMayDiverge || RHS == eOffsetMayDiverge) {
    return eOffsetMayDiverge;
  }

  // Uniform values are all that's left.
  return eOffsetUniformVariable;
}
}  // namespace

OffsetInfo::OffsetInfo(StrideAnalysisResult &SAR, Value *V)
    : Kind(eOffsetMayDiverge),
      ActualValue(V),
      StrideInt(0),
      ManifestStride(nullptr),
      BitMask(~uint64_t(0)) {
  auto *const ty = V->getType();
  if (ty->isIntegerTy()) {
    analyze(V, SAR);
  } else if (ty->isPointerTy()) {
    analyzePtr(V, SAR);
  } else {
    setMayDiverge();
  }
}

Value *OffsetInfo::getUniformValue() const {
  return isUniform() ? ActualValue : nullptr;
}

int64_t OffsetInfo::getValueAsConstantInt() const {
  ConstantInt *CInt = cast<ConstantInt>(ActualValue);
  return CInt->getSExtValue();
}

bool OffsetInfo::isStrideConstantInt() const {
  return (Kind == eOffsetLinear && StrideInt != 0);
}

int64_t OffsetInfo::getStrideAsConstantInt() const { return StrideInt; }

OffsetInfo &OffsetInfo::setMayDiverge() { return setKind(eOffsetMayDiverge); }

OffsetInfo &OffsetInfo::setStride(Value *Stride) {
  if (auto *const CInt = dyn_cast_or_null<ConstantInt>(Stride)) {
    StrideInt = CInt->getSExtValue();
  } else {
    StrideInt = 0;
  }
  ManifestStride = Stride;
  Kind = eOffsetLinear;
  return *this;
}

OffsetInfo &OffsetInfo::setStride(int64_t Stride) {
  if (Stride == 0) {
    Kind = eOffsetUniformVariable;
  } else {
    StrideInt = Stride;
    ManifestStride = nullptr;
    Kind = eOffsetLinear;
  }
  return *this;
}

OffsetInfo &OffsetInfo::setKind(OffsetKind K) {
  Kind = K;
  return *this;
}

OffsetInfo &OffsetInfo::analyze(Value *Offset, StrideAnalysisResult &SAR) {
  Type *OffsetTy = Offset->getType();
  if (!OffsetTy->isIntegerTy() || OffsetTy->isVectorTy()) {
    return setMayDiverge();
  }

  if (auto *const CInt = dyn_cast<ConstantInt>(Offset)) {
    BitMask = CInt->getZExtValue();
    return setKind(eOffsetConstant);
  }
  BitMask = getTypeMask(OffsetTy);

  if (isa<Argument>(Offset)) {
    return setKind(eOffsetUniformVariable);
  }

  Instruction *Ins = dyn_cast<Instruction>(Offset);
  if (!Ins) {
    return setMayDiverge();
  }

  // If we have a uniform value here we don't need to analyse any further.
  if (!SAR.UVR.isVarying(Ins)) {
    const auto &KB =
        computeKnownBits(Ins, SAR.F.getParent()->getDataLayout(), 0, &SAR.AC);
    const auto bitWidth = OffsetTy->getIntegerBitWidth();

    // We are interested in the bits that are not known to be zero.
    BitMask &= ~KB.Zero.extractBitsAsZExtValue(bitWidth, 0);
    return setKind(eOffsetUniformVariable);
  }

  // Analyse binary instructions.
  if (BinaryOperator *BOp = dyn_cast<BinaryOperator>(Offset)) {
    // Copy these values into local variables, because `SAR.analyze()` can
    // invalidate any previously obtained references.
    const auto LHS = SAR.analyze(BOp->getOperand(0));
    const auto RHS = SAR.analyze(BOp->getOperand(1));
    if (LHS.mayDiverge() || RHS.mayDiverge()) {
      return setMayDiverge();
    }

    if (isa<OverflowingBinaryOperator>(BOp) && !BOp->hasNoUnsignedWrap()) {
      // This operation can over/underflow, therefore all bets are off on
      // which bits are on. We set it to all ones so a ZExt will catch it.
      // SExt does not care since overflow is UB.
      BitMask = ~uint64_t(0);
    }

    switch (BOp->getOpcode()) {
      default:
        return setMayDiverge();
      case Instruction::Add:
        return combineAdd(LHS, RHS);
      case Instruction::Sub:
        return combineSub(LHS, RHS);
      case Instruction::And:
        return combineAnd(LHS, RHS);
      case Instruction::Or:
        return combineOr(LHS, RHS);
      case Instruction::Xor:
        return combineXor(LHS, RHS);
      case Instruction::Mul:
        return combineMul(LHS, RHS);
      case Instruction::Shl:
        return combineShl(LHS, RHS);
      case Instruction::AShr:
        return combineAShr(LHS, RHS);
    }
  }

  // Consider that integer casts cannot scale item IDs.
  if (CastInst *Cast = dyn_cast<CastInst>(Offset)) {
    const auto &Src = SAR.analyze(Cast->getOperand(0));
    if (Src.mayDiverge()) {
      return setMayDiverge();
    }

    // However, a Zero-extended offset can underflow.
    if (isa<ZExtInst>(Cast)) {
      // A zero-extended offset could underflow and result in an invalid base
      // address, rendering the entire strided MemOp invalid, even when masked
      // such that the read from the base address is not meant to execute.
      // Note that we don't care about overflowing the index type.
      const auto typeMask = getTypeMask(Cast->getSrcTy());
      const auto bitMaskSized =
          Src.BitMask & getSizeTypeMask(Cast->getModule()->getDataLayout());
      if ((bitMaskSized & typeMask) != bitMaskSized) {
        return setMayDiverge();
      }
      BitMask = Src.BitMask & typeMask;
    } else if (isa<SExtInst>(Cast)) {
      const uint64_t widthMask = getTypeMask(Cast->getSrcTy());
      const uint64_t signMask = (widthMask >> 1) + 1;
      if (Src.BitMask & signMask) {
        // If it's possible for the source value to be negative, all of the
        // bits in the extended value might be set.
        BitMask = Src.BitMask | ~widthMask;
      } else {
        BitMask = Src.BitMask & widthMask;
      }
    } else {
      // We don't truncate the bitmask here, since we don't know if it's going
      // to be sign extended or zero extended later, which affects whether we
      // can ignore overflow or not.
      BitMask = Src.BitMask;
    }
    return copyStrideFrom(Src);
  }

  if (auto *Select = dyn_cast<SelectInst>(Offset)) {
    if (SAR.UVR.isVarying(Select->getCondition())) {
      return setMayDiverge();
    }

    // If the condition isn't varying and both operands have the same
    // constant stride, the result will also have the same constant stride.
    const auto LHS = SAR.analyze(Select->getOperand(1));
    const auto RHS = SAR.analyze(Select->getOperand(2));
    if (LHS.hasStride() && RHS.hasStride() && LHS.StrideInt == RHS.StrideInt &&
        LHS.isStrideConstantInt()) {
      return copyStrideFrom(LHS);
    }
    return setMayDiverge();
  }

  if (auto *Phi = dyn_cast<PHINode>(Offset)) {
    if (auto *const CVal = Phi->hasConstantValue()) {
      return copyStrideFrom(SAR.analyze(CVal));
    }

    auto NumIncoming = Phi->getNumIncomingValues();
    if (NumIncoming == 1) {
      // LCSSA Phi, just go right through it..
      return copyStrideFrom(SAR.analyze(Phi->getIncomingValue(0)));
    } else if (NumIncoming == 2) {
      auto identifyIncrement = [&](Value *incoming) -> bool {
        if (auto *BOp = dyn_cast<BinaryOperator>(incoming)) {
          auto Opcode = BOp->getOpcode();
          // If it's a simple loop iterator, the stride can be analyzed from the
          // initial value.
          return ((Opcode == Instruction::Add || Opcode == Instruction::Sub) &&
                  BOp->getOperand(0) == Phi &&
                  !SAR.UVR.isVarying(BOp->getOperand(1))) ||
                 (Opcode == Instruction::Add && BOp->getOperand(1) == Phi &&
                  !SAR.UVR.isVarying(BOp->getOperand(0)));
        }
        return false;
      };

      // Try the PHI node's incoming values both ways round.
      if (identifyIncrement(Phi->getIncomingValue(1))) {
        return copyStrideFrom(SAR.analyze(Phi->getIncomingValue(0)));
      } else if (identifyIncrement(Phi->getIncomingValue(0))) {
        return copyStrideFrom(SAR.analyze(Phi->getIncomingValue(1)));
      }
    }
    return setMayDiverge();
  }

  // Analyse function calls.
  if (CallInst *CI = dyn_cast<CallInst>(Offset)) {
    const auto &BI = SAR.UVR.Ctx.builtins();
    const auto Builtin = BI.analyzeBuiltinCall(*CI, SAR.UVR.dimension);
    switch (Builtin.uniformity) {
      default:
      case compiler::utils::eBuiltinUniformityMaybeInstanceID:
      case compiler::utils::eBuiltinUniformityNever:
        return setMayDiverge();
      case compiler::utils::eBuiltinUniformityLikeInputs:
        break;
      case compiler::utils::eBuiltinUniformityAlways:
        return setKind(eOffsetUniformVariable);
      case compiler::utils::eBuiltinUniformityInstanceID:
        if (Builtin.properties & compiler::utils::eBuiltinPropertyLocalID) {
          // If the local size is unknown (represented by zero), the
          // resulting mask will be ~0ULL (all ones). Potentially, it is
          // possible to use the CL_​DEVICE_​MAX_​WORK_​ITEM_​SIZES
          // property as an upper bound in this case.
          uint64_t LocalBitMask = SAR.UVR.VU.getLocalSize() - 1;
          LocalBitMask |= LocalBitMask >> 32;
          LocalBitMask |= LocalBitMask >> 16;
          LocalBitMask |= LocalBitMask >> 8;
          LocalBitMask |= LocalBitMask >> 4;
          LocalBitMask |= LocalBitMask >> 2;
          LocalBitMask |= LocalBitMask >> 1;
          BitMask = LocalBitMask;
        }
        return setStride(1);
    }
  }

  return setMayDiverge();
}

OffsetInfo &OffsetInfo::analyzePtr(Value *Address, StrideAnalysisResult &SAR) {
  if (BitCastInst *BCast = dyn_cast<BitCastInst>(Address)) {
    return copyStrideFrom(SAR.analyze(BCast->getOperand(0)));
  } else if (auto *ASCast = dyn_cast<AddrSpaceCastInst>(Address)) {
    return copyStrideFrom(SAR.analyze(ASCast->getOperand(0)));
  } else if (auto *IntPtr = dyn_cast<IntToPtrInst>(Address)) {
    return copyStrideFrom(SAR.analyze(IntPtr->getOperand(0)));
  } else if (auto *Arg = dyn_cast<Argument>(Address)) {
    // 'Pointer return' arguments should be treated as having an implicit ItemID
    // offset. This allows memory operations to be packetized instead of
    // instantiated.
    if (Arg->getType()->isPointerTy()) {
      for (const VectorizerTargetArgument &VUArg : SAR.UVR.VU.arguments()) {
        if (((VUArg.OldArg == Arg) || (VUArg.NewArg == Arg)) &&
            VUArg.PointerRetPointeeTy) {
          Type *MemTy = VUArg.PointerRetPointeeTy;
          const uint64_t MemSize =
              SAR.UVR.Ctx.dataLayout()->getTypeAllocSize(MemTy);
          return setStride(MemSize);
        }
      }
    }
    return setKind(eOffsetUniformVariable);
  } else if (isa<GlobalVariable>(Address)) {
    return setKind(eOffsetUniformVariable);
  } else if (!SAR.UVR.isVarying(Address)) {
    // If it's uniform we can just return the uniform address.
    // Check this condition before bothering to descend into Phi nodes or GEPs,
    // since we know stride is zero anyway.
    return setKind(eOffsetUniformVariable);
  } else if (auto *const Alloca = dyn_cast<AllocaInst>(Address)) {
    if (needsInstantiation(SAR.UVR.Ctx, *Alloca)) {
      // Instantiated allocas result in scatter/gather
      return setMayDiverge();
    }

    Type *MemTy = Alloca->getAllocatedType();
    const uint64_t MemSize = SAR.UVR.Ctx.dataLayout()->getTypeAllocSize(MemTy);
    return setStride(MemSize);
  } else if (auto *const Phi = dyn_cast<PHINode>(Address)) {
    // If all the incoming values are the same, we can trace through it. In
    // the general case, it's not trivial to check that the stride is the same
    // from every incoming block, and since incoming values may not dominate
    // the IRBuilder insert point, we might not even be able to build the
    // offset expression instructions there.
    if (auto *const CVal = Phi->hasConstantValue()) {
      return copyStrideFrom(SAR.analyze(CVal));
    }

    // In the simple case of a loop-incremented pointer using a GEP, we can
    // handle it thus:
    auto NumIncoming = Phi->getNumIncomingValues();
    if (NumIncoming != 2) {
      // Perhaps we can handle more than one loop latch, but not yet.
      return setMayDiverge();
    }

    if (auto *const GEP =
            dyn_cast<GetElementPtrInst>(Phi->getIncomingValue(1))) {
      // If it's a simple loop iterator, the stride can be analyzed from the
      // initial value.
      if (GEP->getPointerOperand() == Phi) {
        for (const auto &index : GEP->indices()) {
          if (SAR.UVR.isVarying(index.get())) {
            return setMayDiverge();
          }
        }
        return copyStrideFrom(SAR.analyze(Phi->getIncomingValue(0)));
      }
    } else if (auto *const GEP =
                   dyn_cast<GetElementPtrInst>(Phi->getIncomingValue(0))) {
      // If it's a simple loop iterator, the stride can be analyzed from the
      // initial value.
      if (GEP->getPointerOperand() == Phi) {
        for (const auto &index : GEP->indices()) {
          if (SAR.UVR.isVarying(index.get())) {
            return setMayDiverge();
          }
        }
        return copyStrideFrom(SAR.analyze(Phi->getIncomingValue(1)));
      }
    }

    return setMayDiverge();
  } else if (auto *GEP = dyn_cast<GetElementPtrInst>(Address)) {
    {
      auto *const Ptr = GEP->getPointerOperand();
      const auto &PtrInfo = SAR.analyze(Ptr);
      if (PtrInfo.mayDiverge()) {
        if (isa<SelectInst>(Ptr)) {
          // For the benefit of the Ternary Transform Pass
          for (Value *idx : GEP->indices()) {
            SAR.analyze(idx);
          }
        }
        return setMayDiverge();
      }
      copyStrideFrom(PtrInfo);
    }

    PointerType *GEPPtrTy = dyn_cast<PointerType>(GEP->getPointerOperandType());
    if (!GEPPtrTy) {
      // A GEP base can be a vector of pointers, for instance. (Unexpected!)
      return setMayDiverge();
    }

    int64_t GEPStrideInt = StrideInt;
    bool StrideVariable = (hasStride() && StrideInt == 0);
    SmallVector<Value *, 4> Indices;
    for (unsigned i = 0; i < GEP->getNumIndices(); i++) {
      // Analyze each GEP offset.
      Value *GEPIndex = GEP->getOperand(1 + i);
      assert(GEPIndex && "Could not get operand from GEP");

      const auto &idxOffset = SAR.analyze(GEPIndex);
      if (idxOffset.mayDiverge()) {
        return setMayDiverge();
      }

      Indices.push_back(GEPIndex);
      if (!idxOffset.hasStride()) {
        continue;
      }

      Type *MemTy = GetElementPtrInst::getIndexedType(
          GEP->getSourceElementType(), Indices);
      if (!MemTy) {
        // A somewhat unlikely scenario...?
        return setMayDiverge();
      }

      if (idxOffset.isStrideConstantInt()) {
        // Add all the strides together,
        // since `Base + (A * X) + (B * X) == Base + (A + B) * X`
        const uint64_t MemSize = SizeOrZero(
            GEP->getModule()->getDataLayout().getTypeAllocSize(MemTy));
        GEPStrideInt += idxOffset.StrideInt * MemSize;
      } else {
        StrideVariable = true;
      }
    }

    if (StrideVariable) {
      // We don't know what the stride is yet,
      // but we know it's linear and variable.
      setStride(nullptr);
    } else {
      setStride(GEPStrideInt);
    }
    return *this;
  } else if (auto *Select = dyn_cast<SelectInst>(Address)) {
    const auto LHS = SAR.analyze(Select->getOperand(1));
    const auto RHS = SAR.analyze(Select->getOperand(2));
    if (SAR.UVR.isVarying(Select->getCondition())) {
      // Note that we analyze the operands before returning here, for the
      // benefit of the Ternary Transform Pass, which does its work ONLY
      // when the condition is varying.
      return setMayDiverge();
    }

    // If the condition isn't varying and both operands have the same
    // constant stride, the result will also have the same constant stride.
    if (LHS.hasStride() && RHS.hasStride() && LHS.StrideInt == RHS.StrideInt &&
        LHS.isStrideConstantInt()) {
      return copyStrideFrom(LHS);
    }
    return setMayDiverge();
  }

  // If it's varying we can't analyze it any further.
  return setMayDiverge();
}

OffsetInfo &OffsetInfo::manifest(IRBuilder<> &B, StrideAnalysisResult &SAR) {
  if (ManifestStride || Kind != eOffsetLinear) {
    // If we already manifested the stride, or if it's not a linear value,
    // there is nothing to do.
    return *this;
  }

  if (StrideInt != 0) {
    // It's an integer stride so we can just create a `ConstantInt`.
    ManifestStride = getSizeInt(B, StrideInt);
    return *this;
  }

  Instruction *Offset = cast<Instruction>(ActualValue);
  // Analyse binary instructions.
  if (BinaryOperator *BOp = dyn_cast<BinaryOperator>(Offset)) {
    const auto &LHS = SAR.manifest(B, BOp->getOperand(0));
    const auto &RHS = SAR.manifest(B, BOp->getOperand(1));

    // Build strides immediately before their instructions
    B.SetInsertPoint(BOp);
    switch (BOp->getOpcode()) {
      default:
        return *this;
      case Instruction::Add:
        return manifestAdd(B, LHS, RHS);
      case Instruction::Sub:
        return manifestSub(B, LHS, RHS);
      case Instruction::And:
        return manifestAnd(B, LHS, RHS);
      case Instruction::Or:
        return manifestOr(B, LHS, RHS);
      case Instruction::Xor:
        return manifestXor(B, LHS, RHS);
      case Instruction::Mul:
        return manifestMul(B, LHS, RHS);
      case Instruction::Shl:
        return manifestShl(B, LHS, RHS);
      case Instruction::AShr:
        return manifestAShr(B, LHS, RHS);
    }
  }

  // Consider that integer casts cannot scale item IDs.
  if (CastInst *Cast = dyn_cast<CastInst>(Offset)) {
    return copyStrideFrom(SAR.manifest(B, Cast->getOperand(0)));
  }

  if (auto *Phi = dyn_cast<PHINode>(Offset)) {
    auto NumIncoming = Phi->getNumIncomingValues();
    Value *SrcVal = nullptr;
    if (NumIncoming == 1) {
      // LCSSA Phi, just go right through it..
      SrcVal = Phi->getIncomingValue(0);
    } else if (auto *const CVal = Phi->hasConstantValue()) {
      SrcVal = CVal;
    } else if (NumIncoming == 2) {
      auto identifyIncrement = [&](Value *incoming) -> bool {
        if (auto *BOp = dyn_cast<BinaryOperator>(incoming)) {
          // If this consumes the Phi node, we have found the increment.
          return BOp->getOperand(0) == Phi || BOp->getOperand(1) == Phi;
        } else if (auto *GEP = dyn_cast<GetElementPtrInst>(incoming)) {
          return GEP->getPointerOperand() == Phi;
        }
        return false;
      };

      // Try the PHI node's incoming values both ways round.
      if (identifyIncrement(Phi->getIncomingValue(1))) {
        SrcVal = Phi->getIncomingValue(0);
      } else if (identifyIncrement(Phi->getIncomingValue(0))) {
        SrcVal = Phi->getIncomingValue(1);
      }
    }
    assert(SrcVal && "Unexpected Phi node during stride manifestation");
    return copyStrideFrom(SAR.manifest(B, SrcVal));
  }

  if (auto *GEP = dyn_cast<GetElementPtrInst>(Offset)) {
    const auto &Ptr = SAR.manifest(B, GEP->getPointerOperand());
    copyStrideFrom(Ptr);

    PointerType *GEPPtrTy = dyn_cast<PointerType>(GEP->getPointerOperandType());
    if (!GEPPtrTy) {
      // A GEP base can be a vector of pointers, for instance. (Unexpected!)
      return setMayDiverge();
    }

    Value *GEPStride = nullptr;
    SmallVector<Value *, 4> Indices;
    for (unsigned i = 0; i < GEP->getNumIndices(); i++) {
      // Analyze each GEP offset.
      Value *GEPIndex = GEP->getOperand(1 + i);
      assert(GEPIndex && "Could not get operand from GEP");

      const auto &idxOffset = SAR.manifest(B, GEPIndex);

      Indices.push_back(GEPIndex);
      if (!idxOffset.hasStride()) {
        continue;
      }

      Type *MemTy = GetElementPtrInst::getIndexedType(
          GEP->getSourceElementType(), Indices);

      // Build stride instructions immediately before the GEP. Note that the
      // process of manifesting the indices can change the insert point.
      B.SetInsertPoint(GEP);
      Value *idxStride = nullptr;
      const uint64_t MemSize =
          SizeOrZero(GEP->getModule()->getDataLayout().getTypeAllocSize(MemTy));
      if (MemSize == 1) {
        // Don't need to do anything if the size is 1
        idxStride = idxOffset.ManifestStride;
      } else {
        if ((MemSize & (MemSize - 1)) == 0) {
          // the size is a power of two, so shift to get the offset in bytes
          auto *const SizeVal = getSizeInt(B, highbit(MemSize));
          idxStride = B.CreateShl(idxOffset.ManifestStride, SizeVal);
        } else {
          // otherwise, multiply
          auto *const SizeVal = getSizeInt(B, MemSize);
          idxStride = B.CreateMul(idxOffset.ManifestStride, SizeVal);
        }
      }

      // Add all the strides together,
      // since `Base + (A * X) + (B * X) == Base + (A + B) * X`
      if (GEPStride) {
        GEPStride = B.CreateAdd(GEPStride, idxStride);
      } else {
        GEPStride = idxStride;
      }
    }

    if (GEPStride) {
      setStride(GEPStride);
    }
  }

  return *this;
}

uint64_t OffsetInfo::getConstantMemoryStride(Type *PtrEleTy,
                                             const DataLayout *DL) const {
  const uint64_t PtrEleSize = SizeOrZero(DL->getTypeAllocSize(PtrEleTy));
  VECZ_FAIL_IF(!PtrEleSize);

  // It's not a valid stride if it's not divisible by the element size.
  // Can't generate a valid interleaved MemOp from it!
  if (StrideInt != 0 && StrideInt % PtrEleSize != 0) {
    return 0;
  }
  return StrideInt / PtrEleSize;
}

Value *OffsetInfo::buildMemoryStride(IRBuilder<> &B, Type *PtrEleTy,
                                     const DataLayout *DL) const {
  if (!ManifestStride) {
    assert(Kind != eOffsetLinear &&
           "buildMemoryStride: linear stride not manifest");
    return nullptr;
  }

  const uint64_t PtrEleSize = SizeOrZero(DL->getTypeAllocSize(PtrEleTy));
  VECZ_FAIL_IF(!PtrEleSize);

  // It's not a valid stride if it's not divisible by the element size.
  // Can't generate a valid interleaved MemOp from it!
  if (StrideInt != 0 && StrideInt % PtrEleSize != 0) {
    return nullptr;
  }

  if ((PtrEleSize & (PtrEleSize - 1)) == 0) {
    auto ShiftVal = highbit(PtrEleSize);
    if (auto *BinOp = dyn_cast<BinaryOperator>(ManifestStride)) {
      if (BinOp->getOpcode() == Instruction::Shl) {
        if (auto *ConstSize = dyn_cast<ConstantInt>(BinOp->getOperand(1))) {
          if (ConstSize->getZExtValue() == ShiftVal) {
            return BinOp->getOperand(0);
          }
        }
      }
    }

    auto *const stride =
        B.CreateAShr(ManifestStride, ConstantInt::get(getSizeTy(B), ShiftVal));
    return stride;
  } else {
    if (auto *BinOp = dyn_cast<BinaryOperator>(ManifestStride)) {
      if (BinOp->getOpcode() == Instruction::Mul) {
        if (auto *ConstSize = dyn_cast<ConstantInt>(BinOp->getOperand(1))) {
          if (ConstSize->getZExtValue() == PtrEleSize) {
            return BinOp->getOperand(0);
          }
        }
      }
    }

    auto *const stride = B.CreateSDiv(
        ManifestStride, ConstantInt::get(getSizeTy(B), PtrEleSize));
    return stride;
  }
}

OffsetInfo &OffsetInfo::combineAdd(const OffsetInfo &LHS,
                                   const OffsetInfo &RHS) {
  BitMask &= LHS.BitMask | RHS.BitMask | (LHS.BitMask + RHS.BitMask);

  if (LHS.hasStride()) {
    if (RHS.hasStride()) {
      // Linear + Linear
      if (LHS.isStrideConstantInt() && RHS.isStrideConstantInt()) {
        return setStride(LHS.StrideInt + RHS.StrideInt);
      } else {
        return setStride(nullptr);
      }
    } else {
      // Linear + Uniform
      return copyStrideFrom(LHS);
    }
  } else if (RHS.hasStride()) {
    // Uniform + Linear
    return copyStrideFrom(RHS);
  }

  Kind = combineKinds(LHS.Kind, RHS.Kind);
  return *this;
}

OffsetInfo &OffsetInfo::manifestAdd(IRBuilder<> &B, const OffsetInfo &LHS,
                                    const OffsetInfo &RHS) {
  if (LHS.hasStride()) {
    if (RHS.hasStride()) {
      // Linear + Linear
      auto *const newAdd = B.CreateAdd(LHS.ManifestStride, RHS.ManifestStride);
      return setStride(newAdd);
    } else {
      // Linear + Uniform
      return copyStrideFrom(LHS);
    }
  } else if (RHS.hasStride()) {
    // Uniform + Linear
    return copyStrideFrom(RHS);
  }
  return *this;
}

OffsetInfo &OffsetInfo::combineSub(const OffsetInfo &LHS,
                                   const OffsetInfo &RHS) {
  if (LHS.hasStride()) {
    if (RHS.hasStride()) {
      // Linear - Linear
      if (LHS.isStrideConstantInt() && RHS.isStrideConstantInt()) {
        return setStride(LHS.StrideInt - RHS.StrideInt);
      } else {
        return setStride(nullptr);
      }
    } else {
      // Linear - Uniform
      return copyStrideFrom(LHS);
    }
  } else if (RHS.hasStride()) {
    // Uniform - Linear
    // Subtracting an item ID results in a negative stride.
    if (RHS.isStrideConstantInt()) {
      return setStride(-RHS.StrideInt);
    } else {
      return setStride(nullptr);
    }
  }
  Kind = combineKinds(LHS.Kind, RHS.Kind);
  return *this;
}

OffsetInfo &OffsetInfo::manifestSub(IRBuilder<> &B, const OffsetInfo &LHS,
                                    const OffsetInfo &RHS) {
  if (LHS.hasStride()) {
    if (RHS.hasStride()) {
      // Linear - Linear
      auto *const newSub = B.CreateSub(LHS.ManifestStride, RHS.ManifestStride);
      return setStride(newSub);
    } else {
      // Linear - Uniform
      return copyStrideFrom(LHS);
    }
  } else if (RHS.hasStride()) {
    // Uniform - Linear
    // Subtracting an item ID results in a negative stride.
    auto *const newNeg = B.CreateNeg(RHS.ManifestStride);
    return setStride(newNeg);
  }
  return *this;
}

OffsetInfo &OffsetInfo::combineAnd(const OffsetInfo &LHS,
                                   const OffsetInfo &RHS) {
  BitMask = LHS.BitMask & RHS.BitMask;
  if (LHS.hasStride()) {
    if (RHS.hasStride()) {
      // Linear & Linear -> can't analyze
      return setMayDiverge();
    } else {
      // Linear & Uniform
      // If we didn't lose any bits of the LHS, we can do it.
      if (BitMask == LHS.BitMask) {
        return copyStrideFrom(LHS);
      } else {
        return setMayDiverge();
      }
    }
  } else if (RHS.hasStride()) {
    // Uniform & Linear
    // If we didn't lose any bits of the RHS, we can do it.
    if (BitMask == RHS.BitMask) {
      return copyStrideFrom(RHS);
    } else {
      return setMayDiverge();
    }
  }

  Kind = combineKinds(LHS.Kind, RHS.Kind);
  return *this;
}

OffsetInfo &OffsetInfo::manifestAnd(IRBuilder<> &, const OffsetInfo &LHS,
                                    const OffsetInfo &RHS) {
  if (LHS.hasStride()) {
    return copyStrideFrom(LHS);
  } else if (RHS.hasStride()) {
    return copyStrideFrom(RHS);
  }
  return *this;
}

OffsetInfo &OffsetInfo::combineOr(const OffsetInfo &LHS,
                                  const OffsetInfo &RHS) {
  if ((LHS.BitMask & RHS.BitMask) == 0) {
    // An Or is equivalent to an Add if the operands have no bits in common.
    return combineAdd(LHS, RHS);
  }

  if (LHS.hasStride() || RHS.hasStride()) {
    return setMayDiverge();
  }

  BitMask = LHS.BitMask | RHS.BitMask;
  Kind = combineKinds(LHS.Kind, RHS.Kind);
  return *this;
}

OffsetInfo &OffsetInfo::manifestOr(IRBuilder<> &B, const OffsetInfo &LHS,
                                   const OffsetInfo &RHS) {
  if ((LHS.BitMask & RHS.BitMask) == 0) {
    // An Or is equivalent to an Add if the operands have no bits in common.
    return manifestAdd(B, LHS, RHS);
  }
  return *this;
}

OffsetInfo &OffsetInfo::combineXor(const OffsetInfo &LHS,
                                   const OffsetInfo &RHS) {
  if ((LHS.BitMask & RHS.BitMask) == 0) {
    // An Xor is equivalent to an Add if the operands have no bits in common.
    return combineAdd(LHS, RHS);
  }

  if (LHS.hasStride() || RHS.hasStride()) {
    return setMayDiverge();
  }

  BitMask = LHS.BitMask | RHS.BitMask;
  Kind = combineKinds(LHS.Kind, RHS.Kind);
  return *this;
}

OffsetInfo &OffsetInfo::manifestXor(IRBuilder<> &B, const OffsetInfo &LHS,
                                    const OffsetInfo &RHS) {
  if ((LHS.BitMask & RHS.BitMask) == 0) {
    // An Xor is equivalent to an Add if the operands have no bits in common.
    return manifestAdd(B, LHS, RHS);
  }
  return *this;
}

OffsetInfo &OffsetInfo::combineShl(const OffsetInfo &LHS,
                                   const OffsetInfo &RHS) {
  if (RHS.hasStride()) {
    return setMayDiverge();
  } else if (LHS.hasStride()) {
    auto *const Shift = RHS.getUniformValue();
    if (!Shift) {
      return setMayDiverge();
    }

    if (ConstantInt *CShift = dyn_cast<ConstantInt>(Shift)) {
      const auto CVal = CShift->getZExtValue();
      BitMask = LHS.BitMask << CVal;
      return setStride(LHS.StrideInt << CVal);
    }

    BitMask = ~uint64_t(0);
    return setStride(nullptr);
  }

  Kind = combineKinds(LHS.Kind, RHS.Kind);
  return *this;
}

OffsetInfo &OffsetInfo::manifestShl(IRBuilder<> &B, const OffsetInfo &LHS,
                                    const OffsetInfo &RHS) {
  auto *const Shift = RHS.getUniformValue();
  if (Shift && LHS.hasStride()) {
    auto *const sizeShift = matchSizeType(B, Shift, false);
    auto *const newShl = B.CreateShl(LHS.ManifestStride, sizeShift);
    return setStride(newShl);
  }
  return *this;
}

OffsetInfo &OffsetInfo::combineAShr(const OffsetInfo &LHS,
                                    const OffsetInfo &RHS) {
  if (RHS.hasStride()) {
    return setMayDiverge();
  } else if (LHS.hasStride()) {
    auto *const Shift = RHS.getUniformValue();
    if (!Shift) {
      return setMayDiverge();
    }

    // We have to be careful with right shifts, because some bits of the stride
    // could get shifted out of the right-hand-side, causing it not to be
    // uniform anymore.
    if (RHS.Kind == eOffsetConstant) {
      auto CShift = RHS.getValueAsConstantInt();
      if (CShift < 0 || CShift >= 64) {
        // Unlikely, but just in case..
        return setMayDiverge();
      }

      // Note that we shift the bitmask as a signed value.
      // Note also that the BitMask is been initialized to the width of the
      // integer type.
      const uint64_t signMask = (BitMask >> 1) + 1;
      if (LHS.BitMask & signMask) {
        // If it's possible for the source value to be negative, all of the
        // bits in the extended value might be set.
        BitMask &= (LHS.BitMask >> CShift) | ~(BitMask >> CShift);
      } else {
        BitMask &= LHS.BitMask >> CShift;
      }

      if (LHS.isStrideConstantInt()) {
        const auto lostBits = ((uint64_t(1) << CShift) - 1);
        if ((LHS.StrideInt & lostBits) == 0 || (LHS.BitMask & lostBits) == 0) {
          return setStride(LHS.StrideInt >> CShift);
        }
      } else if ((LHS.BitMask & ((uint64_t(1) << CShift) - 1)) == 0) {
        return setStride(nullptr);
      }
    }
    return setMayDiverge();
  }
  Kind = combineKinds(LHS.Kind, RHS.Kind);
  return *this;
}

OffsetInfo &OffsetInfo::manifestAShr(IRBuilder<> &B, const OffsetInfo &LHS,
                                     const OffsetInfo &RHS) {
  if (RHS.Kind == eOffsetConstant) {
    auto *const Shift = RHS.getUniformValue();
    const auto CShift = RHS.getValueAsConstantInt();

    if (!LHS.isStrideConstantInt() &&
        (LHS.BitMask & ((uint64_t(1) << CShift) - 1)) == 0) {
      auto *const sizeShift = matchSizeType(B, Shift, false);
      auto *const newAShr = B.CreateAShr(LHS.ManifestStride, sizeShift);
      return setStride(newAShr);
    }
  }
  return *this;
}

OffsetInfo &OffsetInfo::combineMul(const OffsetInfo &LHS,
                                   const OffsetInfo &RHS) {
  if (LHS.hasStride() && RHS.hasStride()) {
    // Linear * Linear = not Linear
    return setMayDiverge();
  }

  if (LHS.hasStride()) {
    // Linear * Uniform
    if (LHS.isStrideConstantInt() && RHS.Kind == eOffsetConstant) {
      return setStride(LHS.StrideInt * RHS.getValueAsConstantInt());
    } else {
      return setStride(nullptr);
    }
  } else if (RHS.hasStride()) {
    // Uniform * Linear
    if (RHS.isStrideConstantInt() && LHS.Kind == eOffsetConstant) {
      return setStride(RHS.StrideInt * LHS.getValueAsConstantInt());
    } else {
      return setStride(nullptr);
    }
  }

  Kind = combineKinds(LHS.Kind, RHS.Kind);
  return *this;
}

OffsetInfo &OffsetInfo::manifestMul(IRBuilder<> &B, const OffsetInfo &LHS,
                                    const OffsetInfo &RHS) {
  if (LHS.hasStride()) {
    // Linear * Uniform
    if (auto *const RHSUniform = RHS.getUniformValue()) {
      auto *const sizeMul = matchSizeType(B, RHSUniform, true);
      auto *const newMul = B.CreateMul(LHS.ManifestStride, sizeMul);
      return setStride(newMul);
    }
  } else if (RHS.hasStride()) {
    // Uniform * Linear
    if (auto *const LHSUniform = LHS.getUniformValue()) {
      auto *const sizeMul = matchSizeType(B, LHSUniform, true);
      auto *const newMul = B.CreateMul(RHS.ManifestStride, sizeMul);
      return setStride(newMul);
    }
  }
  return *this;
}

OffsetInfo &OffsetInfo::copyStrideFrom(const OffsetInfo &Other) {
  Kind = Other.Kind;
  StrideInt = Other.StrideInt;
  ManifestStride = Other.ManifestStride;
  return *this;
}
