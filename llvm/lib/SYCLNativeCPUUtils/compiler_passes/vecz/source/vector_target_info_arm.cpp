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

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicsAArch64.h>
#include <llvm/IR/IntrinsicsARM.h>
#include <multi_llvm/vector_type_helper.h>

#include "debugging.h"
#include "vecz/vecz_target_info.h"

using namespace vecz;
using namespace llvm;

namespace vecz {

class TargetInfoArm final : public TargetInfo {
 public:
  TargetInfoArm(TargetMachine *tm) : TargetInfo(tm) {}

  ~TargetInfoArm() = default;

  bool canOptimizeInterleavedGroup(const Instruction &val,
                                   InterleavedOperation kind, int stride,
                                   unsigned groupSize) const override;

  bool optimizeInterleavedGroup(IRBuilder<> &builder, InterleavedOperation kind,
                                ArrayRef<Value *> group,
                                ArrayRef<Value *> masks, Value *baseAddress,
                                int stride) const override;

 private:
  bool canOptimizeInterleavedGroupImpl(const Instruction &val,
                                       InterleavedOperation kind, int stride,
                                       unsigned groupSize,
                                       unsigned &intrinsicID) const;
};

class TargetInfoAArch64 final : public TargetInfo {
 public:
  TargetInfoAArch64(TargetMachine *tm) : TargetInfo(tm) {}

  ~TargetInfoAArch64() = default;

  bool canOptimizeInterleavedGroup(const Instruction &val,
                                   InterleavedOperation kind, int stride,
                                   unsigned groupSize) const override;

  bool optimizeInterleavedGroup(IRBuilder<> &builder, InterleavedOperation kind,
                                ArrayRef<Value *> group,
                                ArrayRef<Value *> masks, Value *baseAddress,
                                int stride) const override;

 private:
  bool canOptimizeInterleavedGroupImpl(const Instruction &val,
                                       InterleavedOperation kind, int stride,
                                       unsigned groupSize,
                                       unsigned &intrinsicID) const;
};

std::unique_ptr<TargetInfo> createTargetInfoArm(TargetMachine *tm) {
  return std::make_unique<TargetInfoArm>(tm);
}

std::unique_ptr<TargetInfo> createTargetInfoAArch64(TargetMachine *tm) {
  return std::make_unique<TargetInfoAArch64>(tm);
}

}  // namespace vecz

bool TargetInfoArm::canOptimizeInterleavedGroup(const Instruction &val,
                                                InterleavedOperation kind,
                                                int stride,
                                                unsigned groupSize) const {
  unsigned IntrID;
  return canOptimizeInterleavedGroupImpl(val, kind, stride, groupSize, IntrID);
}

bool TargetInfoArm::canOptimizeInterleavedGroupImpl(const Instruction &val,
                                                    InterleavedOperation kind,
                                                    int stride,
                                                    unsigned groupSize,
                                                    unsigned &IntrID) const {
  IntrID = Intrinsic::not_intrinsic;
  Type *dataType = nullptr;
  if (kind == eInterleavedStore) {
    switch (stride) {
      default:
        break;
      case 2:
        IntrID = Intrinsic::arm_neon_vst2;
        break;
      case 3:
        IntrID = Intrinsic::arm_neon_vst3;
        break;
      case 4:
        IntrID = Intrinsic::arm_neon_vst4;
        break;
    }
    dataType = val.getOperand(0)->getType();
  } else if (kind == eInterleavedLoad) {
    switch (stride) {
      default:
        break;
      case 2:
        IntrID = Intrinsic::arm_neon_vld2;
        break;
      case 3:
        IntrID = Intrinsic::arm_neon_vld3;
        break;
      case 4:
        IntrID = Intrinsic::arm_neon_vld4;
        break;
    }
    dataType = val.getType();
  } else {
    return false;
  }

  if (IntrID == Intrinsic::not_intrinsic || ((groupSize % stride) != 0)) {
    return false;
  }

  if (!dataType) {
    return false;
  }

  auto *VecTy = dyn_cast<FixedVectorType>(dataType);
  if (!VecTy) {
    return false;
  }

  const unsigned VecBits = VecTy->getPrimitiveSizeInBits();
  if ((VecBits != 128) && (VecBits != 64)) {
    return false;
  }

  // NEON interleave instructions only allow 8, 16, and 32 bit elements
  const unsigned ElementSize = VecTy->getScalarSizeInBits();
  if ((ElementSize != 32) && (ElementSize != 16) && (ElementSize != 8)) {
    return false;
  }

  return true;
}

bool TargetInfoArm::optimizeInterleavedGroup(IRBuilder<> &B,
                                             InterleavedOperation kind,
                                             ArrayRef<Value *> group,
                                             ArrayRef<Value *>, Value *address,
                                             int stride) const {
  const bool HasMask =
      (kind == eMaskedInterleavedLoad) || (kind == eMaskedInterleavedStore);
  // canOptimizeInterleavedGroup() should have returned false in this case.
  // ARM does not have masked vector load or store instructions.
  VECZ_FAIL_IF(HasMask);
  VECZ_FAIL_IF(stride < 0);

  // TODO CA-3100 fetch information on SubTargetInfo
  // load instructions seems to be easily split in the backend whereas stores
  // generate a backend error because of invalid data type on vector operands.
  // Vector operands are enabled in the backend only when SubTargetInfo ensures
  // NEON instrutions are supported.
  const bool subTargetHasNeon = false;
  if (!subTargetHasNeon && kind == eInterleavedStore) {
    return false;
  }

  // Validate the operations in the group.
  SmallVector<CallInst *, 4> Calls;
  for (unsigned i = 0; i < group.size(); i++) {
    CallInst *Op = dyn_cast<CallInst>(group[i]);
    if (!Op) {
      return false;
    }
    Calls.push_back(Op);
  }

  PointerType *PtrTy = dyn_cast<PointerType>(address->getType());
  if (!PtrTy) {
    return false;
  }

  CallInst *Op0 = Calls[0];
  // Determine the intrinsic to emit for this group.
  unsigned IntrID = Intrinsic::not_intrinsic;
  if (!canOptimizeInterleavedGroupImpl(*Op0, kind, stride, group.size(),
                                       IntrID)) {
    return false;
  }

  // canOptimizeInterleavedGroup() performs several checks, including valid
  // Kind and Op0 types. Thus, these casts are safe.
  FixedVectorType *VecTy = nullptr;
  if (kind == eInterleavedStore) {
    VecTy = cast<FixedVectorType>(Op0->getOperand(0)->getType());
  } else {  // eInterleavedLoad
    VecTy = cast<FixedVectorType>(Op0->getType());
  }

  Type *EleTy = VecTy->getElementType();
  const unsigned Alignment = (EleTy->getPrimitiveSizeInBits() / 8);

  // Declare the intrinsic if needed.
  SmallVector<Type *, 2> Tys;
  if (kind == eInterleavedStore) {
    Tys = {PtrTy, VecTy};
  } else if (kind == eInterleavedLoad) {
    Tys = {VecTy, PtrTy};
  }

  Function *IntrFn =
      Intrinsic::getDeclaration(Op0->getModule(), (Intrinsic::ID)IntrID, Tys);
  if (!IntrFn) {
    return false;
  }

  // Create a NEON load or store to replace the interleaved calls.
  SmallVector<Value *, 8> Ops;
  Ops.push_back(address);
  if (kind == eInterleavedStore) {
    for (unsigned i = 0; i < group.size(); i++) {
      CallInst *Op = Calls[i];
      Ops.push_back(Op->getOperand(0));
    }
  }
  Ops.push_back(B.getInt32(Alignment));
  CallInst *CI = B.CreateCall(IntrFn, Ops, Op0->getName());
  CI->setCallingConv(IntrFn->getCallingConv());
  if (kind == eInterleavedLoad) {
    for (unsigned i = 0; i < Calls.size(); i++) {
      CallInst *Op = Calls[i];
      const ArrayRef<unsigned> Indices(&i, 1);
      Value *Extract = B.CreateExtractValue(CI, Indices);
      Op->replaceAllUsesWith(Extract);
    }
  }
  return true;
}

bool TargetInfoAArch64::canOptimizeInterleavedGroup(const Instruction &val,
                                                    InterleavedOperation kind,
                                                    int stride,
                                                    unsigned groupSize) const {
  unsigned IntrID;
  return canOptimizeInterleavedGroupImpl(val, kind, stride, groupSize, IntrID);
}

bool TargetInfoAArch64::canOptimizeInterleavedGroupImpl(
    const Instruction &val, InterleavedOperation kind, int stride,
    unsigned groupSize, unsigned &IntrID) const {
  IntrID = Intrinsic::not_intrinsic;
  Type *dataType = nullptr;
  if (kind == eInterleavedStore) {
    switch (stride) {
      default:
        break;
      case 2:
        IntrID = Intrinsic::aarch64_neon_st2;
        break;
      case 3:
        IntrID = Intrinsic::aarch64_neon_st3;
        break;
      case 4:
        IntrID = Intrinsic::aarch64_neon_st4;
        break;
    }
    dataType = val.getOperand(0)->getType();
  } else if (kind == eInterleavedLoad) {
    switch (stride) {
      default:
        break;
      case 2:
        IntrID = Intrinsic::aarch64_neon_ld2;
        break;
      case 3:
        IntrID = Intrinsic::aarch64_neon_ld3;
        break;
      case 4:
        IntrID = Intrinsic::aarch64_neon_ld4;
        break;
    }
    dataType = val.getType();
  } else {
    return false;
  }

  if (IntrID == Intrinsic::not_intrinsic || ((groupSize % stride) != 0)) {
    return false;
  }

  if (!dataType) {
    return false;
  }

  auto *VecTy = dyn_cast<FixedVectorType>(dataType);
  if (!VecTy) {
    return false;
  }

  const unsigned VecBits = VecTy->getPrimitiveSizeInBits();
  if ((VecBits != 128) && (VecBits != 64)) {
    return false;
  }

  // NEON interleave instructions only allow 8, 16, and 32 bit elements
  const unsigned ElementSize = VecTy->getScalarSizeInBits();
  if ((ElementSize != 32) && (ElementSize != 16) && (ElementSize != 8)) {
    return false;
  }

  return true;
}

bool TargetInfoAArch64::optimizeInterleavedGroup(
    IRBuilder<> &B, InterleavedOperation kind, ArrayRef<Value *> group,
    ArrayRef<Value *>, Value *address, int stride) const {
  const bool HasMask =
      (kind == eMaskedInterleavedLoad) || (kind == eMaskedInterleavedStore);
  // canOptimizeInterleavedGroup() should have returned false in this case.
  // AArch64 does not have masked vector load or store instructions.
  VECZ_FAIL_IF(HasMask);
  VECZ_FAIL_IF(stride < 0);

  // TODO CA-3100 fetch information on SubTargetInfo
  // load instructions seems to be easily split in the backend whereas stores
  // generate a backend error because of invalid data type on vector operands.
  // Vector operands are enabled in the backend only when SubTargetInfo ensures
  // NEON instrutions are supported.
  const bool subTargetHasNeon = false;
  if (!subTargetHasNeon && kind == eInterleavedStore) {
    return false;
  }

  // Validate the operations in the group.
  SmallVector<CallInst *, 4> Calls;
  for (unsigned i = 0; i < group.size(); i++) {
    CallInst *Op = dyn_cast<CallInst>(group[i]);
    if (!Op) {
      return false;
    }
    Calls.push_back(Op);
  }

  PointerType *PtrTy = dyn_cast<PointerType>(address->getType());
  if (!PtrTy) {
    return false;
  }

  CallInst *Op0 = Calls[0];
  // Determine the intrinsic to emit for this group.
  unsigned IntrID = Intrinsic::not_intrinsic;
  if (!canOptimizeInterleavedGroupImpl(*Op0, kind, stride, group.size(),
                                       IntrID)) {
    return false;
  }

  // canOptimizeInterleavedGroup() performs several checks, including valid
  // Kind and Op0 types. Thus, these casts are safe.
  FixedVectorType *VecTy = nullptr;
  if (kind == eInterleavedStore) {
    VecTy = cast<FixedVectorType>(Op0->getOperand(0)->getType());
  } else {  // eInterleavedLoad
    VecTy = cast<FixedVectorType>(Op0->getType());
  }

  Function *IntrFn = Intrinsic::getDeclaration(
      Op0->getModule(), (Intrinsic::ID)IntrID, {VecTy, PtrTy});
  if (!IntrFn) {
    return false;
  }

  // Create a NEON load or store to replace the interleaved calls.
  SmallVector<Value *, 8> Ops;
  if (kind == eInterleavedStore) {
    for (unsigned i = 0; i < group.size(); i++) {
      CallInst *Op = Calls[i];
      Ops.push_back(Op->getOperand(0));
    }
  }
  Ops.push_back(address);
  CallInst *CI = B.CreateCall(IntrFn, Ops, Op0->getName());
  CI->setCallingConv(IntrFn->getCallingConv());
  if (kind == eInterleavedLoad) {
    for (unsigned i = 0; i < Calls.size(); i++) {
      CallInst *Op = Calls[i];
      const ArrayRef<unsigned> Indices(&i, 1);
      Value *Extract = B.CreateExtractValue(CI, Indices);
      Op->replaceAllUsesWith(Extract);
    }
  }
  return true;
}
