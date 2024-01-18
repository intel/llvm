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

#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include <multi_llvm/triple.h>
#include <multi_llvm/vector_type_helper.h>

#include "debugging.h"
#include "memory_operations.h"
#include "transform/packetization_helpers.h"
#include "vecz/vecz_target_info.h"

using namespace vecz;
using namespace llvm;

namespace {
/// @brief Applies @a EVL to @a Mask, clearing those bits in a position greater
/// than @a EVL.
Value *applyEVLToMask(IRBuilder<> &B, Value *EVL, Value *Mask) {
  if (EVL) {
    auto *const IndexVector = B.CreateStepVector(VectorType::get(
        EVL->getType(), multi_llvm::getVectorElementCount(Mask->getType())));
    auto *const Splat = B.CreateVectorSplat(
        multi_llvm::getVectorElementCount(Mask->getType()), EVL);
    auto *const M = B.CreateICmpULT(IndexVector, Splat);
    Mask = B.CreateLogicalAnd(Mask, M);
  }
  return Mask;
}

bool isLegalMaskedLoad(const TargetTransformInfo &TTI, Type *Ty,
                       unsigned Alignment) {
  return TTI.isLegalMaskedLoad(Ty, Align(Alignment));
}

bool isLegalMaskedStore(const TargetTransformInfo &TTI, Type *Ty,
                        unsigned Alignment) {
  return TTI.isLegalMaskedStore(Ty, Align(Alignment));
}

bool isLegalMaskedGather(const TargetTransformInfo &TTI, Type *Ty,
                         unsigned Alignment) {
  return TTI.isLegalMaskedGather(Ty, Align(Alignment));
}

bool isLegalMaskedScatter(const TargetTransformInfo &TTI, Type *Ty,
                          unsigned Alignment) {
  return TTI.isLegalMaskedScatter(Ty, Align(Alignment));
}
}  // namespace

// NOTE the TargetMachine is allowed to be null here; it isn't used in the
// implementation at present, but if it gets used in future it needs to be
// guarded.
TargetInfo::TargetInfo(TargetMachine *tm) : TM_(tm) {}

Value *TargetInfo::createLoad(IRBuilder<> &B, Type *Ty, Value *Ptr,
                              Value *Stride, unsigned Alignment,
                              Value *EVL) const {
  if (!Ptr || !Stride || !Ty->isVectorTy()) {
    return nullptr;
  }

  // Validate the pointer type.
  PointerType *PtrTy = dyn_cast<PointerType>(Ptr->getType());
  if (!PtrTy) {
    return nullptr;
  }
  Type *EleTy = Ty->getScalarType();

  // Trivial case: contiguous load.
  ConstantInt *CIntStride = dyn_cast<ConstantInt>(Stride);
  PointerType *VecPtrTy = Ty->getPointerTo(PtrTy->getAddressSpace());
  Value *VecPtr = B.CreateBitCast(Ptr, VecPtrTy);
  if (CIntStride && CIntStride->getSExtValue() == 1) {
    if (EVL) {
      const Function *F = B.GetInsertBlock()->getParent();
      const auto Legality = isVPLoadLegal(F, Ty, Alignment);
      if (!Legality.isVPLegal()) {
        emitVeczRemarkMissed(F,
                             "Could not create a VP load as the target "
                             "reported it would be illegal");
        VECZ_FAIL();
      }
      auto *Mask = createAllTrueMask(B, multi_llvm::getVectorElementCount(Ty));
      const SmallVector<llvm::Value *, 2> Args = {VecPtr, Mask, EVL};
      const SmallVector<llvm::Type *, 2> Tys = {Ty, VecPtr->getType()};
      return B.CreateIntrinsic(llvm::Intrinsic::vp_load, Tys, Args);
    }
    return B.CreateAlignedLoad(Ty, VecPtr, MaybeAlign(Alignment));
  }

  if (EVL) {
    emitVeczRemarkMissed(
        B.GetInsertBlock()->getParent(), Ptr,
        "Could not create vector-length-predicated interleaved load");
    return nullptr;
  }

  auto Elts = multi_llvm::getVectorElementCount(Ty);
  if (Elts.isScalable()) {
    emitVeczRemarkMissed(B.GetInsertBlock()->getParent(), Ptr,
                         "Could not create a scalable-vector interleaved load");
    VECZ_FAIL();
  }
  const unsigned SimdWidth = Elts.getFixedValue();
  // Load individual values.
  SmallVector<Value *, 8> Values;
  Value *Index = B.getInt64(0);
  for (unsigned i = 0; i < SimdWidth; i++) {
    Value *GEP = B.CreateGEP(EleTy, Ptr, Index);
    Values.push_back(B.CreateLoad(EleTy, GEP, false, "interleaved.load"));
    Index = B.CreateAdd(Index, Stride);
  }

  // Create a vector out of these values.
  Value *Result = UndefValue::get(Ty);
  for (unsigned i = 0; i < SimdWidth; i++) {
    Result = B.CreateInsertElement(Result, Values[i], B.getInt32(i));
  }
  return Result;
}

Value *TargetInfo::createStore(IRBuilder<> &B, Value *Data, Value *Ptr,
                               Value *Stride, unsigned Alignment,
                               Value *EVL) const {
  if (!Ptr || !Data || !Stride) {
    return nullptr;
  }

  // Validate the pointer type.
  PointerType *PtrTy = dyn_cast<PointerType>(Ptr->getType());
  if (!PtrTy) {
    return nullptr;
  }
  Type *VecTy = Data->getType();
  Type *EleTy = VecTy->getScalarType();

  // Trivial case: contiguous store.
  ConstantInt *CIntStride = dyn_cast<ConstantInt>(Stride);
  if (CIntStride && CIntStride->getSExtValue() == 1) {
    PointerType *VecPtrTy = VecTy->getPointerTo(PtrTy->getAddressSpace());
    Value *VecPtr = B.CreateBitCast(Ptr, VecPtrTy);
    if (EVL) {
      const Function *F = B.GetInsertBlock()->getParent();
      const auto Legality = isVPStoreLegal(F, VecTy, Alignment);
      if (!Legality.isVPLegal()) {
        emitVeczRemarkMissed(F,
                             "Could not create a VP store as the target "
                             "reported it would be illegal");
        VECZ_FAIL();
      }
      auto *Mask =
          createAllTrueMask(B, multi_llvm::getVectorElementCount(VecTy));
      const SmallVector<llvm::Value *, 3> Args = {Data, VecPtr, Mask, EVL};
      const SmallVector<llvm::Type *, 2> Tys = {Data->getType(),
                                                VecPtr->getType()};
      return B.CreateIntrinsic(llvm::Intrinsic::vp_store, Tys, Args);
    }
    return B.CreateAlignedStore(Data, VecPtr, MaybeAlign(Alignment));
  }

  if (EVL) {
    emitVeczRemarkMissed(
        B.GetInsertBlock()->getParent(), Ptr,
        "Could not create vector-length-predicated interleaved store");
    return nullptr;
  }

  auto Elts = multi_llvm::getVectorElementCount(VecTy);
  if (Elts.isScalable()) {
    emitVeczRemarkMissed(
        B.GetInsertBlock()->getParent(), Ptr,
        "Could not create a scalable-vector interleaved store");
    VECZ_FAIL();
  }
  const unsigned SimdWidth = Elts.getFixedValue();
  // Extract values from the vector.
  SmallVector<Value *, 8> Values;
  for (unsigned i = 0; i < SimdWidth; i++) {
    Values.push_back(B.CreateExtractElement(Data, B.getInt32(i)));
  }

  // Store individual values.
  Value *Ret = nullptr;
  Value *Index = B.getInt64(0);
  for (unsigned i = 0; i < SimdWidth; i++) {
    Value *GEP = B.CreateGEP(EleTy, Ptr, Index);
    Ret = B.CreateStore(Values[i], GEP);
    cast<StoreInst>(Ret)->setAlignment(MaybeAlign(Alignment).valueOrOne());

    Index = B.CreateAdd(Index, Stride);
  }
  return Ret;
}

Value *TargetInfo::createMaskedLoad(IRBuilder<> &B, Type *Ty, Value *Ptr,
                                    Value *Mask, Value *EVL,
                                    unsigned Alignment) const {
  VECZ_FAIL_IF(!Ptr || !Mask);
  PointerType *PtrTy = dyn_cast<PointerType>(Ptr->getType());
  VECZ_FAIL_IF(!PtrTy);
  Type *EleTy = Ty->getScalarType();

  // Validate the pointer and mask types.
  auto *DataVecTy = dyn_cast<VectorType>(Ty);
  auto *MaskVecTy = dyn_cast<VectorType>(Mask->getType());
  if (DataVecTy && MaskVecTy) {
    VECZ_ERROR_IF(multi_llvm::getVectorElementCount(DataVecTy) !=
                      multi_llvm::getVectorElementCount(MaskVecTy),
                  "The mask and the data need to have the same width");
  }

  // Use LLVM intrinsics for masked vector loads.
  if (Ty->isVectorTy()) {
    PtrTy = Ty->getPointerTo(PtrTy->getAddressSpace());
    Ptr = B.CreateBitCast(Ptr, PtrTy);
    const Function *F = B.GetInsertBlock()->getParent();
    const auto Legality = isVPLoadLegal(F, Ty, Alignment);
    if (EVL && Legality.isVPLegal()) {
      const SmallVector<llvm::Value *, 2> Args = {Ptr, Mask, EVL};
      const SmallVector<llvm::Type *, 2> Tys = {Ty, PtrTy};
      return B.CreateIntrinsic(llvm::Intrinsic::vp_load, Tys, Args);
    } else if (Legality.isMaskLegal()) {
      Mask = applyEVLToMask(B, EVL, Mask);
      VECZ_FAIL_IF(!Mask);
      return B.CreateMaskedLoad(Ty, Ptr, Align(Alignment), Mask);
    } else {
      emitVeczRemarkMissed(F,
                           "Could not create a masked load as the target "
                           "reported it would be illegal");
      VECZ_FAIL();
    }
  }

  const unsigned Width = 1;

  LLVMContext &Ctx = B.getContext();
  BasicBlock *Entry = B.GetInsertBlock();
  BasicBlock *Exit = nullptr;
  Function *F = Entry->getParent();
  VECZ_FAIL_IF(!F || !Ptr || !Mask || EVL);

  // Create all the required blocks.
  SmallVector<BasicBlock *, 4> TestBlocks;
  SmallVector<BasicBlock *, 4> LoadBlocks;
  TestBlocks.push_back(Entry);
  LoadBlocks.push_back(BasicBlock::Create(Ctx, "masked_load", F));
  for (unsigned i = 1; i < Width; i++) {
    TestBlocks.push_back(BasicBlock::Create(Ctx, "test_mask", F));
    LoadBlocks.push_back(BasicBlock::Create(Ctx, "masked_load", F));
  }
  Exit = BasicBlock::Create(Ctx, "masked_load_exit", F);

  Constant *const DefaultEleData = UndefValue::get(EleTy);
  SmallVector<Value *, 4> LoadedLanes;
  SmallVector<Value *, 4> LanePhis;
  for (unsigned i = 0; i < Width; i++) {
    BasicBlock *Next = ((i + 1) < Width) ? TestBlocks[i + 1] : Exit;

    // Extract the mask elements and branch.
    B.SetInsertPoint(TestBlocks[i]);
    if (i > 0) {
      PHINode *LanePhi = B.CreatePHI(EleTy, 2, "result_lane");
      LanePhi->addIncoming(LoadedLanes[i - 1], LoadBlocks[i - 1]);
      LanePhi->addIncoming(DefaultEleData, TestBlocks[i - 1]);
      LanePhis.push_back(LanePhi);
    }

    Value *MaskLane =
        (Width == 1) ? Mask
                     : B.CreateExtractElement(Mask, B.getInt32(i), "mask_lane");
    B.CreateCondBr(MaskLane, LoadBlocks[i], Next);

    // Load the element and branch.
    B.SetInsertPoint(LoadBlocks[i]);
    Value *LanePtr =
        i > 0 ? B.CreateGEP(EleTy, Ptr, B.getInt32(i), "lane_ptr") : Ptr;
    LoadInst *Load = B.CreateLoad(EleTy, LanePtr, false, "masked_load");
    Load->setAlignment(MaybeAlign(Alignment).valueOrOne());
    LoadedLanes.push_back(Load);
    B.CreateBr(Next);
  }

  // Aggregate the loaded lanes.
  B.SetInsertPoint(Exit);
  PHINode *LastLanePhi = B.CreatePHI(EleTy, 2, "result_lane");
  LastLanePhi->addIncoming(LoadedLanes[Width - 1], LoadBlocks[Width - 1]);
  LastLanePhi->addIncoming(DefaultEleData, TestBlocks[Width - 1]);
  LanePhis.push_back(LastLanePhi);

  Value *Result = nullptr;
  if (Width > 1) {
    Result = UndefValue::get(Ty);
    for (unsigned i = 0; i < Width; i++) {
      Result = B.CreateInsertElement(Result, LanePhis[i], B.getInt32(i));
    }
  } else {
    Result = LanePhis[Width - 1];
  }

  return Result;
}

Value *TargetInfo::createMaskedStore(IRBuilder<> &B, Value *Data, Value *Ptr,
                                     Value *Mask, Value *EVL,
                                     unsigned Alignment) const {
  PointerType *PtrTy = dyn_cast<PointerType>(Ptr->getType());
  VECZ_FAIL_IF(!PtrTy);
  Type *DataTy = Data->getType();
  Type *EleTy = DataTy->getScalarType();

  auto *DataVecTy = dyn_cast<VectorType>(DataTy);
  auto *MaskVecTy = dyn_cast<VectorType>(Mask->getType());
  if (DataVecTy && MaskVecTy) {
    VECZ_ERROR_IF(multi_llvm::getVectorElementCount(DataVecTy) !=
                      multi_llvm::getVectorElementCount(MaskVecTy),
                  "The mask and the data need to have the same width");
  }

  // Use LLVM intrinsics for masked vector Stores.
  if (DataTy->isVectorTy()) {
    PtrTy = DataTy->getPointerTo(PtrTy->getAddressSpace());
    Ptr = B.CreateBitCast(Ptr, PtrTy);
    const Function *F = B.GetInsertBlock()->getParent();
    const auto Legality = isVPStoreLegal(F, DataTy, Alignment);
    if (EVL && Legality.isVPLegal()) {
      const SmallVector<llvm::Value *, 3> Args = {Data, Ptr, Mask, EVL};
      const SmallVector<llvm::Type *, 2> Tys = {Data->getType(), PtrTy};
      return B.CreateIntrinsic(llvm::Intrinsic::vp_store, Tys, Args);
    } else if (Legality.isMaskLegal()) {
      Mask = applyEVLToMask(B, EVL, Mask);
      VECZ_FAIL_IF(!Mask);
      return B.CreateMaskedStore(Data, Ptr, Align(Alignment), Mask);
    } else {
      emitVeczRemarkMissed(F,
                           "Could not create a masked store as the target "
                           "reported it would be illegal");
      VECZ_FAIL();
    }
  }

  const unsigned Width = 1;

  LLVMContext &Ctx = B.getContext();
  BasicBlock *Entry = B.GetInsertBlock();
  BasicBlock *Exit = nullptr;
  StoreInst *FirstStore = nullptr;
  Function *F = Entry->getParent();
  VECZ_FAIL_IF(!F || EVL);

  // Create all the required blocks.
  SmallVector<BasicBlock *, 4> TestBlocks;
  SmallVector<BasicBlock *, 4> StoreBlocks;
  TestBlocks.push_back(Entry);
  StoreBlocks.push_back(BasicBlock::Create(Ctx, "masked_store", F));
  for (unsigned i = 1; i < Width; i++) {
    TestBlocks.push_back(BasicBlock::Create(Ctx, "test_mask", F));
    StoreBlocks.push_back(BasicBlock::Create(Ctx, "masked_store", F));
  }
  Exit = BasicBlock::Create(Ctx, "masked_store_exit", F);

  for (unsigned i = 0; i < Width; i++) {
    BasicBlock *Next = ((i + 1) < Width) ? TestBlocks[i + 1] : Exit;

    // Extract the mask elements and branch.
    B.SetInsertPoint(TestBlocks[i]);
    Value *MaskLane =
        (Width == 1) ? Mask
                     : B.CreateExtractElement(Mask, B.getInt32(i), "mask_lane");
    B.CreateCondBr(MaskLane, StoreBlocks[i], Next);

    // Extract the data elements and store.
    B.SetInsertPoint(StoreBlocks[i]);
    Value *DataLane =
        (Width == 1) ? Data
                     : B.CreateExtractElement(Data, B.getInt32(i), "data_lane");
    Value *LanePtr = Ptr;
    if (i > 0) {
      LanePtr = B.CreateGEP(EleTy, LanePtr, B.getInt32(i), "lane_ptr");
    }
    StoreInst *Store = B.CreateStore(DataLane, LanePtr);
    if (i == 0) {
      FirstStore = Store;
    }
    Store->setAlignment(MaybeAlign(Alignment).valueOrOne());
    B.CreateBr(Next);
  }

  B.SetInsertPoint(Exit);
  return FirstStore;
}

Value *TargetInfo::createInterleavedLoad(IRBuilder<> &B, Type *Ty, Value *Ptr,
                                         Value *Stride, Value *EVL,
                                         unsigned Alignment) const {
  auto EC = multi_llvm::getVectorElementCount(Ty);
  auto *const Mask = B.CreateVectorSplat(EC, B.getTrue());
  return createMaskedInterleavedLoad(B, Ty, Ptr, Mask, Stride, EVL, Alignment);
}

Value *TargetInfo::createInterleavedStore(IRBuilder<> &B, Value *Data,
                                          Value *Ptr, Value *Stride, Value *EVL,
                                          unsigned Alignment) const {
  auto EC = multi_llvm::getVectorElementCount(Data->getType());
  auto *const Mask = B.CreateVectorSplat(EC, B.getTrue());
  return createMaskedInterleavedStore(B, Data, Ptr, Mask, Stride, EVL,
                                      Alignment);
}

Value *TargetInfo::createMaskedInterleavedLoad(IRBuilder<> &B, Type *Ty,
                                               Value *Ptr, Value *Mask,
                                               Value *Stride, Value *EVL,
                                               unsigned Alignment) const {
  // We only support scalar pointer types
  assert(!Ptr->getType()->isVectorTy() && "Unsupported interleaved load");

  auto EC = multi_llvm::getVectorElementCount(Ty);
  Value *BroadcastAddr = B.CreateVectorSplat(EC, Ptr, "BroadcastAddr");
  Value *StrideSplat = B.CreateVectorSplat(EC, Stride);

  Value *IndicesVector =
      createIndexSequence(B, cast<VectorType>(StrideSplat->getType()));
  VECZ_FAIL_IF(!IndicesVector);
  IndicesVector = B.CreateMul(StrideSplat, IndicesVector);

  Value *Address =
      B.CreateGEP(Ty->getScalarType(), BroadcastAddr, IndicesVector);

  return createMaskedGatherLoad(B, Ty, Address, Mask, EVL, Alignment);
}

Value *TargetInfo::createMaskedInterleavedStore(IRBuilder<> &B, Value *Data,
                                                Value *Ptr, Value *Mask,
                                                Value *Stride, Value *EVL,
                                                unsigned Alignment) const {
  // We only support scalar pointer types
  assert(!Ptr->getType()->isVectorTy() && "Unsupported interleaved store");
  auto EC = multi_llvm::getVectorElementCount(Data->getType());
  Value *BroadcastAddr = B.CreateVectorSplat(EC, Ptr, "BroadcastAddr");
  Value *StrideSplat = B.CreateVectorSplat(EC, Stride);

  Value *IndicesVector =
      createIndexSequence(B, cast<VectorType>(StrideSplat->getType()));
  VECZ_FAIL_IF(!IndicesVector);
  IndicesVector = B.CreateMul(StrideSplat, IndicesVector);

  Value *Address = B.CreateGEP(Data->getType()->getScalarType(), BroadcastAddr,
                               IndicesVector);

  return createMaskedScatterStore(B, Data, Address, Mask, EVL, Alignment);
}

Value *TargetInfo::createGatherLoad(IRBuilder<> &B, Type *Ty, Value *Ptr,
                                    Value *EVL, unsigned Alignment) const {
  auto EC = multi_llvm::getVectorElementCount(Ty);
  auto *const Mask = B.CreateVectorSplat(EC, B.getTrue());
  return createMaskedGatherLoad(B, Ty, Ptr, Mask, EVL, Alignment);
}

Value *TargetInfo::createScatterStore(IRBuilder<> &B, Value *Data, Value *Ptr,
                                      Value *EVL, unsigned Alignment) const {
  auto EC = multi_llvm::getVectorElementCount(Data->getType());
  auto *const Mask = B.CreateVectorSplat(EC, B.getTrue());
  return createMaskedScatterStore(B, Data, Ptr, Mask, EVL, Alignment);
}

Value *TargetInfo::createMaskedGatherLoad(IRBuilder<> &B, Type *Ty, Value *Ptr,
                                          Value *Mask, Value *EVL,
                                          unsigned Alignment) const {
  LLVMContext &Ctx = B.getContext();
  BasicBlock *Entry = B.GetInsertBlock();
  BasicBlock *Exit = nullptr;
  Function *F = Entry->getParent();
  VECZ_FAIL_IF(!F || !Ptr || !Mask);

  auto *VecPtrTy = dyn_cast<VectorType>(Ptr->getType());
  VECZ_FAIL_IF(!VecPtrTy);
  PointerType *PtrTy = dyn_cast<PointerType>(VecPtrTy->getElementType());
  VECZ_FAIL_IF(!PtrTy);
  Type *EleTy = Ty->getScalarType();
  Constant *DefaultEleData = UndefValue::get(EleTy);

  if (Ty->isVectorTy()) {
    const auto Legality = isVPGatherLegal(F, Ty, Alignment);
    if (EVL && Legality.isVPLegal()) {
      const SmallVector<llvm::Value *, 2> Args = {Ptr, Mask, EVL};
      const SmallVector<llvm::Type *, 2> Tys = {Ty, VecPtrTy};
      return B.CreateIntrinsic(llvm::Intrinsic::vp_gather, Tys, Args);
    } else if (Legality.isMaskLegal()) {
      Function *MaskedGather = Intrinsic::getDeclaration(
          F->getParent(), Intrinsic::masked_gather, {Ty, VecPtrTy});

      if (MaskedGather) {
        Mask = applyEVLToMask(B, EVL, Mask);
        VECZ_FAIL_IF(!Mask);
        // Create the call to the function
        Value *Args[] = {Ptr, B.getInt32(Alignment), Mask, UndefValue::get(Ty)};
        CallInst *CI = B.CreateCall(MaskedGather, Args);
        if (CI) {
          CI->setCallingConv(MaskedGather->getCallingConv());
          CI->setAttributes(MaskedGather->getAttributes());
          return CI;
        }
      }
    } else {
      emitVeczRemarkMissed(F,
                           "Could not create a masked gather as the target "
                           "reported it would be illegal");
      VECZ_FAIL();
    }
  }

  VECZ_FAIL_IF(EVL);
  auto VecWidth = multi_llvm::getVectorElementCount(Ty);
  const unsigned Width = VecWidth.getFixedValue();

  // Fallback scalar function generator
  // Create all the required blocks.
  SmallVector<BasicBlock *, 4> TestBlocks;
  SmallVector<BasicBlock *, 4> LoadBlocks;
  TestBlocks.push_back(Entry);
  LoadBlocks.push_back(BasicBlock::Create(Ctx, "masked_load", F));
  for (unsigned i = 1; i < Width; i++) {
    TestBlocks.push_back(BasicBlock::Create(Ctx, "test_mask", F));
    LoadBlocks.push_back(BasicBlock::Create(Ctx, "masked_load", F));
  }
  Exit = BasicBlock::Create(Ctx, "masked_load_exit", F);

  SmallVector<Value *, 4> LoadedLanes;
  SmallVector<Value *, 4> LanePhis;
  for (unsigned i = 0; i < Width; i++) {
    BasicBlock *Next = ((i + 1) < Width) ? TestBlocks[i + 1] : Exit;

    // Extract the mask elements and branch.
    B.SetInsertPoint(TestBlocks[i]);
    if (i > 0) {
      PHINode *LanePhi = B.CreatePHI(EleTy, 2, "result_lane");
      LanePhi->addIncoming(LoadedLanes[i - 1], LoadBlocks[i - 1]);
      LanePhi->addIncoming(DefaultEleData, TestBlocks[i - 1]);
      LanePhis.push_back(LanePhi);
    }

    Value *MaskLane = B.CreateExtractElement(Mask, B.getInt32(i), "mask_lane");
    B.CreateCondBr(MaskLane, LoadBlocks[i], Next);

    // Load the element and branch.
    B.SetInsertPoint(LoadBlocks[i]);
    Value *PtrLane = B.CreateExtractElement(Ptr, B.getInt32(i), "ptr_lane");
    LoadInst *Load = B.CreateLoad(EleTy, PtrLane, false, "masked_load");
    Load->setAlignment(MaybeAlign(Alignment).valueOrOne());
    LoadedLanes.push_back(Load);
    B.CreateBr(Next);
  }

  // Aggregate the loaded lanes.
  B.SetInsertPoint(Exit);
  PHINode *LastLanePhi = B.CreatePHI(EleTy, 2, "result_lane");
  LastLanePhi->addIncoming(LoadedLanes[Width - 1], LoadBlocks[Width - 1]);
  LastLanePhi->addIncoming(DefaultEleData, TestBlocks[Width - 1]);
  LanePhis.push_back(LastLanePhi);
  Value *Result = UndefValue::get(Ty);
  for (unsigned i = 0; i < Width; i++) {
    Result = B.CreateInsertElement(Result, LanePhis[i], B.getInt32(i));
  }
  return Result;
}

Value *TargetInfo::createMaskedScatterStore(IRBuilder<> &B, Value *Data,
                                            Value *Ptr, Value *Mask, Value *EVL,
                                            unsigned Alignment) const {
  LLVMContext &Ctx = B.getContext();
  BasicBlock *Entry = B.GetInsertBlock();
  BasicBlock *Exit = nullptr;
  StoreInst *FirstStore = nullptr;
  Function *F = Entry->getParent();
  VECZ_FAIL_IF(!F || !Ptr || !Mask);
  auto *DataTy = Data->getType();

  if (DataTy->isVectorTy()) {
    auto *VecPtrTy = dyn_cast<VectorType>(Ptr->getType());
    VECZ_FAIL_IF(!VecPtrTy);
    const auto Legality = isVPScatterLegal(F, DataTy, Alignment);
    if (EVL && Legality.isVPLegal()) {
      const SmallVector<llvm::Value *, 3> Args = {Data, Ptr, Mask, EVL};
      const SmallVector<llvm::Type *, 2> Tys = {Data->getType(), VecPtrTy};
      return B.CreateIntrinsic(llvm::Intrinsic::vp_scatter, Tys, Args);
    } else if (Legality.isMaskLegal()) {
      Function *MaskedScatter = Intrinsic::getDeclaration(
          F->getParent(), Intrinsic::masked_scatter, {DataTy, VecPtrTy});

      if (MaskedScatter) {
        Mask = applyEVLToMask(B, EVL, Mask);
        VECZ_FAIL_IF(!Mask);
        // Create the call to the function
        Value *Args[] = {Data, Ptr, B.getInt32(Alignment), Mask};
        CallInst *CI = B.CreateCall(MaskedScatter, Args);
        if (CI) {
          CI->setCallingConv(MaskedScatter->getCallingConv());
          CI->setAttributes(MaskedScatter->getAttributes());
          return CI;
        }
      }
    } else {
      emitVeczRemarkMissed(F,
                           "Could not create a masked scatter as the target "
                           "reported it would be illegal");
      VECZ_FAIL();
    }
  }

  VECZ_FAIL_IF(EVL);
  auto VecWidth = multi_llvm::getVectorElementCount(DataTy);
  const unsigned Width = VecWidth.getFixedValue();

  // Fallback scalar function generator
  // Create all the required blocks.
  SmallVector<BasicBlock *, 4> TestBlocks;
  SmallVector<BasicBlock *, 4> StoreBlocks;
  TestBlocks.push_back(Entry);
  StoreBlocks.push_back(BasicBlock::Create(Ctx, "masked_store", F));
  for (unsigned i = 1; i < Width; i++) {
    TestBlocks.push_back(BasicBlock::Create(Ctx, "test_mask", F));
    StoreBlocks.push_back(BasicBlock::Create(Ctx, "masked_store", F));
  }
  Exit = BasicBlock::Create(Ctx, "masked_store_exit", F);

  for (unsigned i = 0; i < Width; i++) {
    BasicBlock *Next = ((i + 1) < Width) ? TestBlocks[i + 1] : Exit;

    // Extract the mask elements and branch.
    B.SetInsertPoint(TestBlocks[i]);
    Value *MaskLane = B.CreateExtractElement(Mask, B.getInt32(i), "mask_lane");
    B.CreateCondBr(MaskLane, StoreBlocks[i], Next);

    // Extract the data elements and store.
    B.SetInsertPoint(StoreBlocks[i]);
    Value *PtrLane = B.CreateExtractElement(Ptr, B.getInt32(i), "ptr_lane");
    Value *DataLane = B.CreateExtractElement(Data, B.getInt32(i), "data_lane");
    StoreInst *Store = B.CreateStore(DataLane, PtrLane);
    if (i == 0) {
      FirstStore = Store;
    }
    Store->setAlignment(MaybeAlign(Alignment).valueOrOne());
    B.CreateBr(Next);
  }

  B.SetInsertPoint(Exit);
  return FirstStore;
}

Value *TargetInfo::createScalableExtractElement(IRBuilder<> &B,
                                                VectorizationContext &Ctx,
                                                Instruction *extract,
                                                Type *narrowTy, Value *src,
                                                Value *index, Value *VL) const {
  (void)VL;
  const auto *origSrc = extract->getOperand(0);
  auto *eltTy = src->getType()->getScalarType();

  auto *wideTy = src->getType();

  auto it = B.GetInsertPoint();

  // Insert alloca at the beginning of the function.
  auto allocaIt =
      B.GetInsertBlock()->getParent()->getEntryBlock().getFirstInsertionPt();
  B.SetInsertPoint(&*allocaIt);
  auto *const alloc = B.CreateAlloca(wideTy, nullptr, "fixlen.alloc");

  // Reset the insertion point to wherever we must insert instructions
  B.SetInsertPoint(&*it);

  // Store the packetized vector to the allocation
  B.CreateStore(src, alloc);

  // Re-interpret the allocation as a pointer to the element type
  auto *const eltptrTy = eltTy->getPointerTo();
  auto *const bcastalloc =
      B.CreatePointerBitCastOrAddrSpaceCast(alloc, eltptrTy, "bcast.alloc");

  const unsigned fixedVecElts =
      multi_llvm::getVectorNumElements(origSrc->getType());

  Value *load = nullptr;
  if (!index->getType()->isVectorTy()) {
    // If the index remains a scalar (is uniform) then we can use a strided load
    // starting from the address '&alloc[index]', strided by the original vector
    // width: &alloc[index], &alloc[index+N], &alloc[index+2N], ...
    auto *const stride = getSizeInt(B, fixedVecElts);
    auto alignment = MaybeAlign(eltTy->getScalarSizeInBits() / 8).valueOrOne();
    // Index into the allocation, coming back with the starting offset from
    // which to begin our loads. This is either a scalar pointer, or a vector of
    // pointers.
    auto *const gep =
        B.CreateInBoundsGEP(eltTy, bcastalloc, index, "vec.alloc");

    load = ::createInterleavedLoad(Ctx, narrowTy, gep, stride, /*Mask*/ nullptr,
                                   /*EVL*/ nullptr, alignment.value(), "",
                                   &*B.GetInsertPoint());
  } else {
    // Else if we've got a varying, vector index, then we must use a gather.
    // Take our indices, and add them to a step multiplied by the original
    // vecor width. Use that to create a vector of pointers.
    auto alignment = MaybeAlign(eltTy->getScalarSizeInBits() / 8).valueOrOne();

    index = getGatherIndicesVector(
        B, index, index->getType(),
        multi_llvm::getVectorNumElements(origSrc->getType()), "idx");

    // Index into the allocation, coming back with the starting offset from
    // which to begin our striding load.
    auto *const gep =
        B.CreateInBoundsGEP(eltTy, bcastalloc, index, "vec.alloc");

    load = ::createGather(Ctx, narrowTy, gep, /*Mask*/ nullptr, /*EVL*/ nullptr,
                          alignment.value(), "", &*B.GetInsertPoint());
  }

  return load;
}

Value *TargetInfo::createOuterScalableBroadcast(IRBuilder<> &builder,
                                                Value *vector, Value *VL,
                                                ElementCount factor) const {
  return createScalableBroadcast(builder, vector, VL, factor,
                                 /* URem */ true);
}

Value *TargetInfo::createInnerScalableBroadcast(IRBuilder<> &builder,
                                                Value *vector, Value *VL,
                                                ElementCount factor) const {
  return createScalableBroadcast(builder, vector, VL, factor,
                                 /* URem */ false);
}

Value *TargetInfo::createScalableBroadcast(IRBuilder<> &B, Value *vector,
                                           Value *VL, ElementCount factor,
                                           bool URem) const {
  (void)VL;
  auto *const ty = vector->getType();
  auto *const wideTy = ScalableVectorType::get(
      multi_llvm::getVectorElementType(ty),
      factor.getKnownMinValue() *
          multi_llvm::getVectorElementCount(ty).getKnownMinValue());
  auto wideEltCount = multi_llvm::getVectorElementCount(wideTy);

  // The splats must be inserted after any Allocas
  auto it = B.GetInsertBlock()->getParent()->getEntryBlock().begin();
  while (isa<AllocaInst>(*it)) {
    ++it;
  }
  IRBuilder<> AllocaB(&*it);

  auto *const alloc = AllocaB.CreateAlloca(ty, nullptr, "fixlen.alloc");

  // Store the vector to the allocation.
  B.CreateStore(vector, alloc);

  auto *const eltTy = cast<llvm::VectorType>(ty)->getElementType();

  auto *const eltptrTy = eltTy->getPointerTo();
  auto *const bcastalloc =
      B.CreatePointerBitCastOrAddrSpaceCast(alloc, eltptrTy, "bcast.alloc");
  auto *const stepsRem = TargetInfo::createBroadcastIndexVector(
      B,
      ScalableVectorType::get(B.getInt32Ty(), cast<ScalableVectorType>(wideTy)),
      factor, URem, "idx1");
  auto *const gep =
      B.CreateInBoundsGEP(eltTy, bcastalloc, stepsRem, "vec.alloc");
  auto *const boolTrue = ConstantInt::getTrue(B.getContext());
  auto *const mask = B.CreateVectorSplat(wideEltCount, boolTrue, "truemask");
  // Set the alignment to that of vector element type.
  auto alignment = MaybeAlign(eltTy->getScalarSizeInBits() / 8).valueOrOne();
  return B.CreateMaskedGather(wideTy, gep, alignment, mask,
                              UndefValue::get(wideTy));
}

Value *TargetInfo::createBroadcastIndexVector(IRBuilder<> &B, Type *ty,
                                              ElementCount factor, bool URem,
                                              const llvm::Twine &N) {
  auto *const steps = B.CreateStepVector(ty, "idx0");
  const auto tyEC = multi_llvm::getVectorElementCount(ty);
  const unsigned factorMinVal = factor.getKnownMinValue();

  unsigned fixedAmt;
  Instruction::BinaryOps Opc;
  if (URem) {
    fixedAmt = tyEC.getKnownMinValue() / factorMinVal;
    Opc = BinaryOperator::URem;
  } else {
    fixedAmt = factorMinVal;
    Opc = BinaryOperator::UDiv;
  }
  auto *const vectorEltsSplat = B.CreateVectorSplat(
      tyEC, ConstantInt::get(multi_llvm::getVectorElementType(ty), fixedAmt));
  return B.CreateBinOp(Opc, steps, vectorEltsSplat, N);
}

Value *TargetInfo::createScalableInsertElement(IRBuilder<> &B,
                                               VectorizationContext &Ctx,
                                               Instruction *insert, Value *elt,
                                               Value *into, Value *index,
                                               Value *VL) const {
  (void)VL;
  auto *eltTy = elt->getType();
  auto *intoTy = into->getType();
  auto *scalarTy = elt->getType()->getScalarType();

  // The alloca must be inserted at the beginning of the function.
  auto allocaIt =
      B.GetInsertBlock()->getParent()->getEntryBlock().getFirstInsertionPt();
  auto it = B.GetInsertPoint();

  B.SetInsertPoint(&*allocaIt);
  auto *const alloc = B.CreateAlloca(intoTy, nullptr);

  // Reset the insertion point to wherever we must insert instructions
  B.SetInsertPoint(&*it);

  // Store the wide vector to the allocation
  B.CreateStore(into, alloc);

  // Re-interpret the allocation as a pointer to the element type
  auto *const eltptrTy = scalarTy->getPointerTo();
  auto *const bcastalloc =
      B.CreatePointerBitCastOrAddrSpaceCast(alloc, eltptrTy, "bcast.alloc");

  const unsigned fixedVecElts =
      multi_llvm::getVectorNumElements(insert->getOperand(0)->getType());

  // Construct the index, either by packetizing if (if varying) or by
  // splatting it and combining it with a step vector
  if (!index->getType()->isVectorTy()) {
    // If the index remains a scalar (is uniform) then we can use a strided
    // store starting from the address '&alloc[index]', strided by the original
    // vector width: &alloc[index], &alloc[index+N], &alloc[index+2N], ...
    auto *const stride = getSizeInt(B, fixedVecElts);
    auto alignment =
        MaybeAlign(scalarTy->getScalarSizeInBits() / 8).valueOrOne();
    // Index into the allocation, coming back with the starting offset from
    // which to begin our loads. This is either a scalar pointer, or a vector of
    // pointers.
    auto *const gep =
        B.CreateInBoundsGEP(scalarTy, bcastalloc, index, "vec.alloc");

    Value *store = ::createInterleavedStore(
        Ctx, elt, gep, stride, /*Mask*/ nullptr,
        /*EVL*/ nullptr, alignment.value(), "", &*B.GetInsertPoint());
    VECZ_FAIL_IF(!store);
  } else {
    // Else if we've got a varying, vector index, then we must use a scatter.
    // Take our indices, and add them to a step multiplied by the original
    // vecor width. Use that to create a vector of pointers.
    auto alignment =
        MaybeAlign(scalarTy->getScalarSizeInBits() / 8).valueOrOne();

    auto narrowEltCount = multi_llvm::getVectorElementCount(eltTy);

    auto *steps = B.CreateStepVector(index->getType(), "idx0");
    auto *const fixedVecEltsSplat = B.CreateVectorSplat(
        narrowEltCount,
        ConstantInt::get(index->getType()->getScalarType(), fixedVecElts));
    auto *const stepsMul = B.CreateMul(steps, fixedVecEltsSplat, "idx.scale");
    index = B.CreateAdd(stepsMul, index, "idx");

    // Index into the allocation, coming back with the starting offset from
    // which to begin our striding load.
    auto *const gep =
        B.CreateInBoundsGEP(scalarTy, bcastalloc, index, "vec.alloc");

    Value *store = ::createScatter(Ctx, elt, gep, /*Mask*/ nullptr,
                                   /*EVL*/ nullptr, alignment.value(), "",
                                   &*B.GetInsertPoint());
    VECZ_FAIL_IF(!store);
  }

  // Load the vector back from the stack
  return B.CreateLoad(intoTy, alloc);
}

bool TargetInfo::isVPVectorLegal(const Function &F, Type *Ty) const {
  return !TM_ ||
         TM_->getTargetTransformInfo(F).isElementTypeLegalForScalableVector(
             multi_llvm::getVectorElementType(Ty));
}

TargetInfo::VPMemOpLegality TargetInfo::checkMemOpLegality(
    const Function *F,
    function_ref<bool(const llvm::TargetTransformInfo &, Type *, unsigned)>
        Checker,
    Type *Ty, unsigned Alignment) const {
  assert(Ty->isVectorTy() && "Expected a vector type");
  const bool isMaskLegal =
      !(isa<ScalableVectorType>(Ty) && TM_) ||
      Checker(TM_->getTargetTransformInfo(*F), Ty, Alignment);
  // Assuming a pointer bit width of 64
  bool isVPLegal = isMaskLegal && isVPVectorLegal(*F, Ty);
  if (isVPLegal) {
    const unsigned PtrBitWidth =
        TM_ ? TM_->createDataLayout().getPointerTypeSizeInBits(
                  Ty->getPointerTo())
            : 64;
    auto &Ctx = Ty->getContext();
    auto *const IntTy = IntegerType::get(Ctx, PtrBitWidth);
    auto *const IntVecTy =
        VectorType::get(IntTy, multi_llvm::getVectorElementCount(Ty));
    isVPLegal = isVPVectorLegal(*F, IntVecTy);
  }
  return {isVPLegal, isMaskLegal};
}

TargetInfo::VPMemOpLegality TargetInfo::isVPLoadLegal(
    const Function *F, Type *Ty, unsigned Alignment) const {
  return checkMemOpLegality(F, isLegalMaskedLoad, Ty, Alignment);
}

TargetInfo::VPMemOpLegality TargetInfo::isVPStoreLegal(
    const Function *F, Type *Ty, unsigned Alignment) const {
  return checkMemOpLegality(F, isLegalMaskedStore, Ty, Alignment);
}

TargetInfo::VPMemOpLegality TargetInfo::isVPGatherLegal(
    const Function *F, Type *Ty, unsigned Alignment) const {
  return checkMemOpLegality(F, isLegalMaskedGather, Ty, Alignment);
}

TargetInfo::VPMemOpLegality TargetInfo::isVPScatterLegal(
    const Function *F, Type *Ty, unsigned Alignment) const {
  return checkMemOpLegality(F, isLegalMaskedScatter, Ty, Alignment);
}

bool TargetInfo::isLegalVPElementType(Type *) const { return true; }

llvm::Value *TargetInfo::createVectorShuffle(llvm::IRBuilder<> &B,
                                             llvm::Value *src,
                                             llvm::Value *mask,
                                             llvm::Value *evl) const {
  auto *const srcTy = dyn_cast<VectorType>(src->getType());
  auto *const maskTy = dyn_cast<VectorType>(mask->getType());
  assert(
      srcTy && maskTy &&
      "TargetInfo::createVectorShuffle: source and mask must have vector type");

  if (isa<Constant>(mask)) {
    // Special case if the mask happens to be a constant.
    return B.CreateShuffleVector(src, UndefValue::get(srcTy), mask);
  }

  // The alloca must be inserted at the beginning of the function.
  auto *const curBlock = B.GetInsertBlock();
  auto &entryBlock = curBlock->getParent()->getEntryBlock();
  const auto allocaIt = entryBlock.getFirstInsertionPt();
  const auto it = B.GetInsertPoint();

  B.SetInsertPoint(&entryBlock, allocaIt);
  auto *const alloc = B.CreateAlloca(srcTy, nullptr);

  // Reset the insertion point to wherever we must insert instructions
  B.SetInsertPoint(curBlock, it);

  // Store the wide vector to the allocation
  B.CreateStore(src, alloc);

  auto *const eltTy = srcTy->getElementType();

  // Re-interpret the allocation as a pointer to the element type
  auto *const eltptrTy = eltTy->getPointerTo();
  auto *const bcastalloc =
      B.CreatePointerBitCastOrAddrSpaceCast(alloc, eltptrTy, "bcast.alloc");

  // Index into the allocation.
  auto *const gep = B.CreateInBoundsGEP(eltTy, bcastalloc, mask, "vec.alloc");

  const auto eltCount = maskTy->getElementCount();
  auto *const dstTy = VectorType::get(eltTy, eltCount);
  const auto alignment =
      MaybeAlign(eltTy->getScalarSizeInBits() / 8).valueOrOne();

  Value *gatherMask = nullptr;
  if (evl) {
    const auto EC = srcTy->getElementCount();
    auto *const IndexTy = VectorType::get(evl->getType(), EC);
    auto *const step = B.CreateStepVector(IndexTy);
    gatherMask = B.CreateICmpULT(step, B.CreateVectorSplat(EC, evl));
  } else {
    gatherMask = B.CreateVectorSplat(eltCount, B.getTrue());
  }

  return B.CreateMaskedGather(dstTy, gep, alignment, gatherMask,
                              UndefValue::get(dstTy));
}

llvm::Value *TargetInfo::createVectorSlideUp(llvm::IRBuilder<> &B,
                                             llvm::Value *src,
                                             llvm::Value *insert,
                                             llvm::Value *) const {
  auto *const srcTy = dyn_cast<VectorType>(src->getType());
  assert(srcTy &&
         "TargetInfo::createVectorShuffle: source must have vector type");

  auto *const undef = UndefValue::get(srcTy);
  const auto EC = srcTy->getElementCount();
  if (!EC.isScalable()) {
    // Special case for fixed-width vectors
    const auto width = EC.getFixedValue();
    SmallVector<int, 16> mask(width);
    auto it = mask.begin();
    *it++ = 0;
    for (size_t i = 1; i < width; ++i) {
      *it++ = i - 1;
    }

    auto *const rotate =
        createOptimalShuffle(B, src, undef, mask, Twine("slide_up"));
    return B.CreateInsertElement(rotate, insert, B.getInt64(0), "slide_in");
  }

  auto *const rotate = B.CreateVectorSplice(undef, src, -1, "slide_up");
  return B.CreateInsertElement(rotate, insert, B.getInt64(0), "slide_in");
}

bool TargetInfo::canOptimizeInterleavedGroup(const Instruction &val,
                                             InterleavedOperation Kind,
                                             int Stride,
                                             unsigned GroupSize) const {
  if ((Stride == 2) || (Stride == 4)) {
    VECZ_FAIL_IF((int)GroupSize != abs(Stride));
    VECZ_FAIL_IF((Kind != eInterleavedLoad) && (Kind != eInterleavedStore) &&
                 (Kind != eMaskedInterleavedLoad) &&
                 (Kind != eMaskedInterleavedStore));
    Type *DataType = nullptr;
    if (Kind == eInterleavedStore || Kind == eMaskedInterleavedStore) {
      DataType = val.getOperand(0)->getType();
    } else {
      DataType = val.getType();
    }
    VECZ_FAIL_IF(!DataType);
    VECZ_FAIL_IF(!isa<FixedVectorType>(DataType));
    return true;
  }
  return false;
}

bool TargetInfo::optimizeInterleavedGroup(IRBuilder<> &B,
                                          InterleavedOperation Kind,
                                          ArrayRef<Value *> Group,
                                          ArrayRef<Value *> Masks,
                                          Value *Address, int Stride) const {
  VECZ_FAIL_IF(Stride < 0);

  // Validate the operations in the group.
  SmallVector<CallInst *, 4> Calls;
  for (unsigned i = 0; i < Group.size(); i++) {
    CallInst *Op = dyn_cast<CallInst>(Group[i]);
    VECZ_FAIL_IF(!Op);
    Calls.push_back(Op);
  }
  PointerType *PtrTy = dyn_cast<PointerType>(Address->getType());
  VECZ_FAIL_IF(!PtrTy);
  CallInst *Op0 = Calls[0];
  VECZ_FAIL_IF(!canOptimizeInterleavedGroup(*Op0, Kind, Stride, Group.size()));

  // canOptimizeInterleavedGroup() performs several checks, including valid
  // Kind and Op0 types. Thus, these casts are safe.
  FixedVectorType *VecTy = nullptr;
  if (Kind == eInterleavedStore || Kind == eMaskedInterleavedStore) {
    VecTy = cast<FixedVectorType>(Op0->getOperand(0)->getType());
  } else {  // eInterleavedLoad || eMaskedInterleavedLoad
    VecTy = cast<FixedVectorType>(Op0->getType());
  }

  auto VecWidth = multi_llvm::getVectorElementCount(VecTy);
  const unsigned SimdWidth = VecWidth.getFixedValue();

  Type *EleTy = VecTy->getElementType();
  const unsigned Align = EleTy->getScalarSizeInBits() / 8;

  const bool HasMask =
      (Kind == eMaskedInterleavedLoad) || (Kind == eMaskedInterleavedStore);
  SmallVector<Value *, 4> Vectors;
  SmallVector<Value *, 4> VecMasks(Masks.begin(), Masks.end());
  if (Kind == eInterleavedLoad || Kind == eMaskedInterleavedLoad) {
    // Create one regular vector load per interleaved load in the group.
    if (HasMask) {
      VECZ_FAIL_IF(!interleaveVectors(B, VecMasks, true));
    }

    for (unsigned i = 0; i < Group.size(); i++) {
      Value *AddressN = Address;
      if (i > 0) {
        const unsigned Offset = i * SimdWidth;
        AddressN = B.CreateGEP(EleTy, Address, B.getInt32(Offset));
      }
      Value *Load = nullptr;
      if (!HasMask) {
        Load = createLoad(B, VecTy, AddressN, getSizeInt(B, 1), Align);
      } else {
        Value *Mask = VecMasks[i];
        Load =
            createMaskedLoad(B, VecTy, AddressN, Mask, /*EVL*/ nullptr, Align);
      }
      VECZ_FAIL_IF(!Load);
      Vectors.push_back(Load);
    }
    // Transpose the loaded vectors and replace the original loads.
    VECZ_FAIL_IF(!interleaveVectors(B, Vectors, false));
    for (unsigned i = 0; i < Group.size(); i++) {
      Value *Vector = Vectors[i];
      Value *OrigLoad = Group[i];
      OrigLoad->replaceAllUsesWith(Vector);
    }
  } else if (Kind == eInterleavedStore || Kind == eMaskedInterleavedStore) {
    // Transpose the vectors to store with interleave.
    for (unsigned i = 0; i < Group.size(); i++) {
      CallInst *OrigStore = cast<CallInst>(Group[i]);
      Vectors.push_back(OrigStore->getOperand(0));
    }
    VECZ_FAIL_IF(!interleaveVectors(B, Vectors, true));
    if (HasMask) {
      VECZ_FAIL_IF(!interleaveVectors(B, VecMasks, true));
    }
    // Create one regular vector store per interleaved store in the group.
    for (unsigned i = 0; i < Group.size(); i++) {
      Value *Vector = Vectors[i];
      Value *AddressN = Address;
      if (i > 0) {
        const unsigned Offset = i * SimdWidth;
        AddressN = B.CreateGEP(EleTy, Address, B.getInt32(Offset));
      }
      Value *Store = nullptr;
      if (!HasMask) {
        Store = createStore(B, Vector, AddressN, getSizeInt(B, 1), Align);
      } else {
        Value *Mask = VecMasks[i];
        Store = createMaskedStore(B, Vector, AddressN, Mask, /*EVL*/ nullptr,
                                  Align);
      }
      VECZ_FAIL_IF(!Store);
    }
  }

  return true;
}

bool TargetInfo::interleaveVectors(IRBuilder<> &B,
                                   MutableArrayRef<Value *> Vectors,
                                   bool Forward) const {
  const unsigned Stride = Vectors.size();
  if (Stride == 0) {
    return true;
  }
  auto *VecTy = dyn_cast<FixedVectorType>(Vectors[0]->getType());
  VECZ_FAIL_IF(!VecTy);
  if (Stride == 1) {
    return true;
  }
  const unsigned Width = VecTy->getNumElements();
  VECZ_FAIL_IF(Width < Stride);
  VECZ_FAIL_IF((Width % Stride) != 0);
  for (unsigned i = 1; i < Stride; i++) {
    auto *VecTyN = dyn_cast<FixedVectorType>(Vectors[i]->getType());
    VECZ_FAIL_IF(!VecTyN || (VecTyN != VecTy));
  }

  // Prepare the masks.
  SmallVector<unsigned, 4> MaskLow2;
  SmallVector<unsigned, 4> MaskHigh2;

  StringRef Name;
  if (Forward) {
    Name = "interleave";
    const unsigned Width2 = Width >> 1;
    const unsigned Width3 = Width2 + Width;
    for (unsigned i = 0; i < Width2; ++i) {
      MaskLow2.push_back(i);
      MaskHigh2.push_back(i + Width2);
      MaskLow2.push_back(i + Width);
      MaskHigh2.push_back(i + Width3);
    }
  } else {
    Name = "deinterleave";
    const unsigned Width2 = Width << 1;
    for (unsigned i = 0; i < Width2; i += 2) {
      MaskLow2.push_back(i);
      MaskHigh2.push_back(i + 1);
    }
  }
  Constant *CMaskLow2 = ConstantDataVector::get(B.getContext(), MaskLow2);
  Constant *CMaskHigh2 = ConstantDataVector::get(B.getContext(), MaskHigh2);

  if (Stride == 2) {
    Value *Src0 = Vectors[0];
    Value *Src1 = Vectors[1];
    Vectors[0] = B.CreateShuffleVector(Src0, Src1, CMaskLow2, Name);
    Vectors[1] = B.CreateShuffleVector(Src0, Src1, CMaskHigh2, Name);

    return true;
  } else if (Stride == 4) {
    // For a 4-way interleave, we need two layers of shuffles.
    // Starting with vectors   a..A : b..B : c..C : d..D
    // first shuffle layer  -> ab.. : ..AB : cd.. : ..CD
    // second shuffle layer -> abcd : .... : .... : ABCD
    Value *Src0 = Vectors[0];
    Value *Src1 = Vectors[1];
    Value *Src2 = Vectors[2];
    Value *Src3 = Vectors[3];

    Constant *CMaskLow4 = nullptr;
    Constant *CMaskHigh4 = nullptr;
    if (Forward) {
      SmallVector<unsigned, 4> MaskLow4;
      SmallVector<unsigned, 4> MaskHigh4;
      const unsigned Width2 = Width >> 1;
      const unsigned Width3 = Width2 + Width;
      for (unsigned i = 0; i < Width2; i += 2) {
        MaskLow4.push_back(i);
        MaskLow4.push_back(i + 1);
        MaskLow4.push_back(i + Width);
        MaskLow4.push_back(i + 1 + Width);
        MaskHigh4.push_back(Width2 + i);
        MaskHigh4.push_back(Width2 + i + 1);
        MaskHigh4.push_back(Width3 + i);
        MaskHigh4.push_back(Width3 + i + 1);
      }
      CMaskLow4 = ConstantDataVector::get(B.getContext(), MaskLow4);
      CMaskHigh4 = ConstantDataVector::get(B.getContext(), MaskHigh4);
    } else {
      SmallVector<unsigned, 4> MaskLow4;
      SmallVector<unsigned, 4> MaskHigh4;
      const unsigned Width2 = Width << 1;
      for (unsigned i = 0; i < Width2; i += 4) {
        MaskLow4.push_back(i);
        MaskLow4.push_back(i + 1);
        MaskHigh4.push_back(i + 2);
        MaskHigh4.push_back(i + 3);
      }

      // to perform the de-interleave we reverse the functions of the masks.
      CMaskLow4 = CMaskLow2;
      CMaskHigh4 = CMaskHigh2;
      CMaskLow2 = ConstantDataVector::get(B.getContext(), MaskLow4);
      CMaskHigh2 = ConstantDataVector::get(B.getContext(), MaskHigh4);
    }

    Value *Tmp0 = B.CreateShuffleVector(Src0, Src1, CMaskLow2, Name);
    Value *Tmp1 = B.CreateShuffleVector(Src0, Src1, CMaskHigh2, Name);
    Value *Tmp2 = B.CreateShuffleVector(Src2, Src3, CMaskLow2, Name);
    Value *Tmp3 = B.CreateShuffleVector(Src2, Src3, CMaskHigh2, Name);
    Vectors[0] = B.CreateShuffleVector(Tmp0, Tmp2, CMaskLow4, Name);
    Vectors[1] = B.CreateShuffleVector(Tmp0, Tmp2, CMaskHigh4, Name);
    Vectors[2] = B.CreateShuffleVector(Tmp1, Tmp3, CMaskLow4, Name);
    Vectors[3] = B.CreateShuffleVector(Tmp1, Tmp3, CMaskHigh4, Name);

    return true;
  }
  return false;
}

unsigned TargetInfo::estimateSimdWidth(const TargetTransformInfo &TTI,
                                       const ArrayRef<const Value *> vals,
                                       unsigned width) const {
  const unsigned MaxVecRegBitWidth =
      TTI.getRegisterBitWidth(llvm::TargetTransformInfo::RGK_FixedWidthVector)
          .getFixedValue();

  const unsigned NumVecRegs =
      TTI.getNumberOfRegisters(TTI.getRegisterClassForType(true));

  unsigned VaryingUsage = 0;
  for (const auto *VI : vals) {
    const auto *Ty = VI->getType();
    VaryingUsage +=
        Ty->isPointerTy()
            ? TM_->getPointerSizeInBits(Ty->getPointerAddressSpace())
            : VI->getType()->getPrimitiveSizeInBits();
  }
  const unsigned MaxBits = MaxVecRegBitWidth * NumVecRegs;
  while (VaryingUsage * width > MaxBits) {
    width >>= 1;
  }

  return width;
}

unsigned TargetInfo::getVectorWidthForType(const llvm::TargetTransformInfo &TTI,
                                           const llvm::Type &Ty) const {
  const unsigned MaxVecRegBitWidth =
      TTI.getRegisterBitWidth(llvm::TargetTransformInfo::RGK_FixedWidthVector)
          .getFixedValue();

  if (MaxVecRegBitWidth == 0) {
    return 0;
  }

  unsigned BitWidth = 0;
  if (!Ty.isPtrOrPtrVectorTy()) {
    BitWidth = Ty.getScalarSizeInBits();
  } else if (TM_) {
    BitWidth = TM_->getPointerSizeInBits(Ty.getPointerAddressSpace());
  }

  if (BitWidth == 0) {
    // Couldn't work out the vector width..
    return 0;
  }

  // The floor of 8 prevents poor double precision performance.
  // Not sure why (CA-3461 related?)
  return std::max(MaxVecRegBitWidth / BitWidth, 8u);
}

bool TargetInfo::canPacketize(const llvm::Value *, ElementCount) const {
  return true;
}

namespace vecz {
std::unique_ptr<TargetInfo> createTargetInfoArm(TargetMachine *tm);

std::unique_ptr<TargetInfo> createTargetInfoAArch64(TargetMachine *tm);

std::unique_ptr<TargetInfo> createTargetInfoRISCV(TargetMachine *tm);
}  // namespace vecz

std::unique_ptr<TargetInfo> vecz::createTargetInfoFromTargetMachine(
    TargetMachine *tm) {
  // The TargetMachine is allowed to be null
  if (tm) {
    const Triple &TT(tm->getTargetTriple());
    switch (TT.getArch()) {
      case Triple::arm:
        return createTargetInfoArm(tm);
      case Triple::aarch64:
        return createTargetInfoAArch64(tm);
      case Triple::riscv32:
      case Triple::riscv64:
        return createTargetInfoRISCV(tm);
      default:
        // Just use the generic TargetInfo unless we know better
        break;
    }
  }
  return std::make_unique<TargetInfo>(tm);
}

AnalysisKey TargetInfoAnalysis::Key;

TargetInfoAnalysis::TargetInfoAnalysis()
    : TICallback([](const Module &) {
        return std::make_unique<TargetInfo>(/*TM*/ nullptr);
      }) {}

TargetInfoAnalysis::TargetInfoAnalysis(TargetMachine *TM)
    : TICallback([TM](const Module &) {
        return vecz::createTargetInfoFromTargetMachine(TM);
      }) {}
