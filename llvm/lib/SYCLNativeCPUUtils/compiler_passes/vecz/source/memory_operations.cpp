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

#include "memory_operations.h"

#include <compiler/utils/mangling.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <multi_llvm/vector_type_helper.h>

#include <string>

#include "analysis/instantiation_analysis.h"
#include "analysis/uniform_value_analysis.h"
#include "debugging.h"
#include "vectorization_context.h"
#include "vectorization_unit.h"

using namespace vecz;
using namespace llvm;

static std::string getMaskedMemOpName(Type *DataTy, PointerType *PtrTy,
                                      Type *MaskTy, unsigned Alignment,
                                      bool IsLoad, bool IsVP) {
  if (!DataTy) {
    return std::string();
  }
  compiler::utils::NameMangler Mangler(&DataTy->getContext());
  const char *BaseName = IsLoad ? "masked_load" : "masked_store";
  const compiler::utils::TypeQualifiers DataQuals(
      compiler::utils::eTypeQualNone);
  const compiler::utils::TypeQualifiers PtrQuals(
      compiler::utils::eTypeQualNone, compiler::utils::eTypeQualNone);
  const compiler::utils::TypeQualifiers MaskQuals(
      compiler::utils::eTypeQualNone);
  std::string Name;
  raw_string_ostream O(Name);
  O << VectorizationContext::InternalBuiltinPrefix << BaseName << Alignment
    << "_";
  if (IsVP) {
    O << "vp_";
  }
  if (!Mangler.mangleType(O, DataTy, DataQuals) ||
      !Mangler.mangleType(O, PtrTy, PtrQuals) ||
      !Mangler.mangleType(O, MaskTy, MaskQuals)) {
    return std::string();
  }
  if (IsVP) {
    const compiler::utils::TypeQualifiers VLQuals(
        compiler::utils::eTypeQualNone);
    if (!Mangler.mangleType(O, IntegerType::getInt32Ty(DataTy->getContext()),
                            VLQuals)) {
      return std::string();
    }
  }
  O.flush();
  return Name;
}

Function *vecz::getOrCreateMaskedMemOpFn(VectorizationContext &Ctx,
                                         Type *DataTy, PointerType *PtrTy,
                                         unsigned Alignment, bool IsLoad,
                                         bool IsVP) {
  const Module &M = Ctx.module();
  LLVMContext &LLVMCtx = M.getContext();
  Type *MaskTy = IntegerType::getInt1Ty(LLVMCtx);
  if (auto *VecTy = dyn_cast<VectorType>(DataTy)) {
    MaskTy = VectorType::get(MaskTy, multi_llvm::getVectorElementCount(VecTy));
  }

  // Try to retrieve the builtin if it already exists.
  const std::string Name =
      getMaskedMemOpName(DataTy, PtrTy, MaskTy, Alignment, IsLoad, IsVP);
  VECZ_FAIL_IF(Name.empty());
  Function *F = Ctx.getOrCreateInternalBuiltin(Name, nullptr);
  if (!F) {
    // Declare it if it doesn't exist.
    SmallVector<Type *, 4> Tys;
    if (!IsLoad) {
      Tys.push_back(DataTy);
    }
    Tys.push_back(PtrTy);
    Tys.push_back(MaskTy);
    if (IsVP) {
      Tys.push_back(IntegerType::getInt32Ty(LLVMCtx));
    }

    Type *RetTy = IsLoad ? DataTy : Type::getVoidTy(LLVMCtx);
    FunctionType *FT = FunctionType::get(RetTy, Tys, false);
    F = Ctx.getOrCreateInternalBuiltin(Name, FT);
  }
  return F;
}

static CallInst *createMaskedMemOp(VectorizationContext &Ctx, Value *Data,
                                   Type *DataTy, Value *Ptr, Value *Mask,
                                   Value *EVL, unsigned Alignment, Twine Name,
                                   Instruction *InsertBefore) {
  VECZ_FAIL_IF(!DataTy);
  VECZ_FAIL_IF(!Ptr || !Ptr->getType()->isPointerTy());
  VECZ_FAIL_IF(!Mask);
  assert(!Data || Data->getType() == DataTy);
  auto *PtrTy =
      PointerType::get(DataTy, Ptr->getType()->getPointerAddressSpace());
  if (Ptr->getType() != PtrTy) {
    Ptr = BitCastInst::CreatePointerCast(Ptr, PtrTy, "", InsertBefore);
  }
  Function *F =
      getOrCreateMaskedMemOpFn(Ctx, DataTy, PtrTy, Alignment,
                               /*IsLoad*/ Data == nullptr, EVL != nullptr);
  VECZ_FAIL_IF(!F);
  SmallVector<Value *, 4> Ops;
  if (Data) {
    Ops.push_back(Data);
  }
  Ops.push_back(Ptr);
  Ops.push_back(Mask);
  if (EVL) {
    Ops.push_back(EVL);
  }
  return CallInst::Create(F, Ops, Name, InsertBefore);
}

CallInst *vecz::createMaskedLoad(VectorizationContext &Ctx, Type *Ty,
                                 Value *Ptr, Value *Mask, Value *EVL,
                                 unsigned Alignment, Twine Name,
                                 Instruction *InsertBefore) {
  return createMaskedMemOp(Ctx, /*Data*/ nullptr, Ty, Ptr, Mask, EVL, Alignment,
                           Name, InsertBefore);
}

CallInst *vecz::createMaskedStore(VectorizationContext &Ctx, Value *Data,
                                  Value *Ptr, Value *Mask, Value *EVL,
                                  unsigned Alignment, Twine Name,
                                  Instruction *InsertBefore) {
  return createMaskedMemOp(Ctx, Data, Data->getType(), Ptr, Mask, EVL,
                           Alignment, Name, InsertBefore);
}

static std::string getInterleavedMemOpName(Type *DataTy, PointerType *PtrTy,
                                           Value *Stride, Type *MaskTy,
                                           unsigned Alignment, bool IsLoad,
                                           bool IsVP) {
  if (!DataTy) {
    return std::string();
  }
  compiler::utils::NameMangler Mangler(&DataTy->getContext());
  const char *BaseName = IsLoad ? "interleaved_load" : "interleaved_store";
  std::string Name;
  const compiler::utils::TypeQualifiers VecQuals(
      compiler::utils::eTypeQualNone, compiler::utils::eTypeQualNone);
  const compiler::utils::TypeQualifiers PtrQuals(
      compiler::utils::eTypeQualNone, compiler::utils::eTypeQualNone);
  raw_string_ostream O(Name);
  O << VectorizationContext::InternalBuiltinPrefix;
  if (MaskTy) {
    O << "masked_";
  }
  O << BaseName << Alignment << "_";
  if (IsVP) {
    O << "vp_";
  }
  if (auto *CVal = dyn_cast<ConstantInt>(Stride)) {
    O << CVal->getSExtValue();
  } else {
    O << "V";
  }
  O << "_";
  if (!Mangler.mangleType(O, DataTy, VecQuals) ||
      !Mangler.mangleType(O, PtrTy, PtrQuals)) {
    return std::string();
  }
  if (MaskTy) {
    const compiler::utils::TypeQualifiers MaskQuals(
        compiler::utils::eTypeQualNone);
    if (!Mangler.mangleType(O, MaskTy, MaskQuals)) {
      return std::string();
    }
  }
  if (IsVP) {
    const compiler::utils::TypeQualifiers VLQuals(
        compiler::utils::eTypeQualNone);
    if (!Mangler.mangleType(O, IntegerType::getInt32Ty(DataTy->getContext()),
                            VLQuals)) {
      return std::string();
    }
  }
  O.flush();
  return Name;
}

Function *vecz::getOrCreateInterleavedMemOpFn(VectorizationContext &Ctx,
                                              Type *DataTy, PointerType *PtrTy,
                                              Value *Stride, Type *MaskTy,
                                              unsigned Alignment, bool IsLoad,
                                              bool IsVP) {
  Module &M = Ctx.module();
  LLVMContext &LLVMCtx = M.getContext();

  // Try to retrieve the builtin if it already exists.
  const std::string Name = getInterleavedMemOpName(
      DataTy, PtrTy, Stride, MaskTy, Alignment, IsLoad, IsVP);
  VECZ_FAIL_IF(Name.empty());
  Function *F = Ctx.getOrCreateInternalBuiltin(Name, nullptr);
  if (!F) {
    // Declare it if it doesn't exist.
    SmallVector<Type *, 6> Tys;
    if (!IsLoad) {
      VECZ_FAIL_IF(!DataTy);
      Tys.push_back(DataTy);
    }
    VECZ_FAIL_IF(!PtrTy);
    Tys.push_back(PtrTy);
    if (MaskTy) {
      Tys.push_back(MaskTy);
    }
    if (IsVP) {
      Tys.push_back(IntegerType::getInt32Ty(LLVMCtx));
    }
    if (!isa<ConstantInt>(Stride)) {
      Tys.push_back(getSizeTy(M));
    }
    Type *RetTy = IsLoad ? DataTy : Type::getVoidTy(LLVMCtx);
    FunctionType *FT = FunctionType::get(RetTy, Tys, false);
    F = Ctx.getOrCreateInternalBuiltin(Name, FT);
  }
  return F;
}

static CallInst *createInterleavedMemOp(VectorizationContext &Ctx, Value *Data,
                                        Type *DataTy, Value *Ptr, Value *Stride,
                                        Value *Mask, Value *EVL,
                                        unsigned Alignment, llvm::Twine Name,
                                        llvm::Instruction *InsertBefore) {
  VECZ_FAIL_IF(!DataTy);
  VECZ_FAIL_IF(!Ptr || !Ptr->getType()->isPointerTy());
  assert(!Data || Data->getType() == DataTy);
  auto *PtrTy = PointerType::get(DataTy->getScalarType(),
                                 Ptr->getType()->getPointerAddressSpace());
  if (Ptr->getType() != PtrTy) {
    Ptr = BitCastInst::CreatePointerCast(Ptr, PtrTy, "", InsertBefore);
  }
  Type *MaskTy = Mask ? Mask->getType() : nullptr;
  Function *F = getOrCreateInterleavedMemOpFn(
      Ctx, DataTy, PtrTy, Stride, MaskTy, Alignment,
      /*IsLoad*/ Data == nullptr, EVL != nullptr);
  VECZ_FAIL_IF(!F);
  SmallVector<Value *, 4> Ops;
  if (Data) {
    Ops.push_back(Data);
  }
  Ops.push_back(Ptr);
  if (Mask) {
    Ops.push_back(Mask);
  }
  if (EVL) {
    Ops.push_back(EVL);
  }
  if (!isa<ConstantInt>(Stride)) {
    Ops.push_back(Stride);
  }
  return CallInst::Create(F, Ops, Name, InsertBefore);
}

CallInst *vecz::createInterleavedLoad(VectorizationContext &Ctx, Type *Ty,
                                      Value *Ptr, Value *Stride, Value *Mask,
                                      Value *EVL, unsigned Alignment,
                                      Twine Name, Instruction *InsertBefore) {
  return createInterleavedMemOp(Ctx, /*Data*/ nullptr, Ty, Ptr, Stride, Mask,
                                EVL, Alignment, Name, InsertBefore);
}

CallInst *vecz::createInterleavedStore(VectorizationContext &Ctx, Value *Data,
                                       Value *Ptr, Value *Stride, Value *Mask,
                                       Value *EVL, unsigned Alignment,
                                       Twine Name, Instruction *InsertBefore) {
  return createInterleavedMemOp(Ctx, Data, Data->getType(), Ptr, Stride, Mask,
                                EVL, Alignment, Name, InsertBefore);
}

static std::string getScatterGatherMemOpName(Type *DataTy, VectorType *VecPtrTy,
                                             Type *MaskTy, unsigned Alignment,
                                             bool IsGather, bool IsVP) {
  if (!DataTy) {
    return std::string();
  }
  compiler::utils::NameMangler Mangler(&DataTy->getContext());
  const char *BaseName = IsGather ? "gather_load" : "scatter_store";
  std::string Name;
  const compiler::utils::TypeQualifiers VecQuals(
      compiler::utils::eTypeQualNone, compiler::utils::eTypeQualNone);
  compiler::utils::TypeQualifiers PtrQuals(compiler::utils::eTypeQualNone,
                                           compiler::utils::eTypeQualNone);
  const compiler::utils::TypeQualifiers MaskQuals(
      compiler::utils::eTypeQualNone);
  PtrQuals.push_back(compiler::utils::eTypeQualNone);
  raw_string_ostream O(Name);
  O << VectorizationContext::InternalBuiltinPrefix;
  if (MaskTy) {
    O << "masked_";
  }
  O << BaseName << Alignment << "_";
  if (IsVP) {
    O << "vp_";
  }
  if (!Mangler.mangleType(O, DataTy, VecQuals) ||
      !Mangler.mangleType(O, VecPtrTy, PtrQuals)) {
    return std::string();
  }
  if (MaskTy && !Mangler.mangleType(O, MaskTy, MaskQuals)) {
    return std::string();
  }
  if (IsVP) {
    const compiler::utils::TypeQualifiers VLQuals(
        compiler::utils::eTypeQualNone);
    if (!Mangler.mangleType(O, IntegerType::getInt32Ty(DataTy->getContext()),
                            VLQuals)) {
      return std::string();
    }
  }
  O.flush();
  return Name;
}

Function *vecz::getOrCreateScatterGatherMemOpFn(vecz::VectorizationContext &Ctx,
                                                llvm::Type *DataTy,
                                                llvm::VectorType *VecPtrTy,
                                                llvm::Type *MaskTy,
                                                unsigned Alignment,
                                                bool IsGather, bool IsVP) {
  const Module &M = Ctx.module();
  LLVMContext &LLVMCtx = M.getContext();
  assert(VecPtrTy);
  assert(!MaskTy || multi_llvm::getVectorElementCount(MaskTy) ==
                        multi_llvm::getVectorElementCount(DataTy));

  // Try to retrieve the builtin if it already exists.
  const std::string Name = getScatterGatherMemOpName(DataTy, VecPtrTy, MaskTy,
                                                     Alignment, IsGather, IsVP);
  VECZ_FAIL_IF(Name.empty());
  Function *F = Ctx.getOrCreateInternalBuiltin(Name, nullptr);
  if (!F) {
    // Declare it if it doesn't exist.
    SmallVector<Type *, 4> Tys;
    if (!IsGather) {
      VECZ_FAIL_IF(!DataTy);
      Tys.push_back(DataTy);
    }
    Tys.push_back(VecPtrTy);
    if (MaskTy) {
      Tys.push_back(MaskTy);
    }
    if (IsVP) {
      Tys.push_back(IntegerType::getInt32Ty(LLVMCtx));
    }

    Type *RetTy = IsGather ? DataTy : Type::getVoidTy(LLVMCtx);
    FunctionType *FT = FunctionType::get(RetTy, Tys, false);
    F = Ctx.getOrCreateInternalBuiltin(Name, FT);
  }
  return F;
}

static CallInst *createScatterGatherMemOp(VectorizationContext &Ctx,
                                          Value *VecData, Type *DataTy,
                                          Value *VecPtr, Value *Mask,
                                          Value *EVL, unsigned Alignment,
                                          Twine Name,
                                          Instruction *InsertBefore) {
  VECZ_FAIL_IF(!DataTy);
  VECZ_FAIL_IF(!VecPtr || !VecPtr->getType()->isVectorTy() ||
               !VecPtr->getType()->getScalarType()->isPointerTy());
  Type *MaskTy = Mask ? Mask->getType() : nullptr;
  Function *F = getOrCreateScatterGatherMemOpFn(
      Ctx, DataTy, cast<VectorType>(VecPtr->getType()), MaskTy, Alignment,
      /*IsGather*/ VecData == nullptr, EVL != nullptr);
  VECZ_FAIL_IF(!F);
  SmallVector<Value *, 4> Ops;
  if (VecData) {
    Ops.push_back(VecData);
  }
  Ops.push_back(VecPtr);
  if (Mask) {
    Ops.push_back(Mask);
  }
  if (EVL) {
    Ops.push_back(EVL);
  }
  return CallInst::Create(F, Ops, Name, InsertBefore);
}

llvm::CallInst *vecz::createGather(VectorizationContext &Ctx, llvm::Type *Ty,
                                   llvm::Value *VecPtr, llvm::Value *Mask,
                                   llvm::Value *EVL, unsigned Alignment,
                                   llvm::Twine Name,
                                   llvm::Instruction *InsertBefore) {
  return createScatterGatherMemOp(Ctx, /*Data*/ nullptr, Ty, VecPtr, Mask, EVL,
                                  Alignment, Name, InsertBefore);
}

llvm::CallInst *vecz::createScatter(VectorizationContext &Ctx,
                                    llvm::Value *VecData, llvm::Value *VecPtr,
                                    llvm::Value *Mask, llvm::Value *EVL,
                                    unsigned Alignment, llvm::Twine Name,
                                    llvm::Instruction *InsertBefore) {
  return createScatterGatherMemOp(Ctx, VecData, VecData->getType(), VecPtr,
                                  Mask, EVL, Alignment, Name, InsertBefore);
}

MemOpDesc::MemOpDesc()
    : DataTy(nullptr),
      PtrTy(nullptr),
      MaskTy(nullptr),
      Kind(MemOpKind::Invalid),
      AccessKind(MemOpAccessKind::Native),
      IsVLOp(false),
      Alignment(1),
      Stride(nullptr),
      DataOpIdx(-1),
      PtrOpIdx(-1),
      MaskOpIdx(-1),
      VLOpIdx(-1) {}

bool MemOpDesc::isStrideConstantInt() const {
  return Stride && isa<ConstantInt>(Stride);
}

int64_t MemOpDesc::getStrideAsConstantInt() const {
  return cast<ConstantInt>(Stride)->getSExtValue();
}

Argument *MemOpDesc::getOperand(Function *F, int OpIdx) const {
  VECZ_FAIL_IF(!F || (OpIdx < 0) || ((size_t)OpIdx >= F->arg_size()));
  return F->getArg(OpIdx);
}

std::optional<MemOpDesc> MemOpDesc::analyzeMemOpFunction(Function &F) {
  if (auto Op = MemOpDesc::analyzeMaskedMemOp(F)) {
    return Op;
  }
  if (auto Op = MemOpDesc::analyzeInterleavedMemOp(F)) {
    return Op;
  }
  if (auto Op = MemOpDesc::analyzeMaskedInterleavedMemOp(F)) {
    return Op;
  }
  if (auto Op = MemOpDesc::analyzeScatterGatherMemOp(F)) {
    return Op;
  }
  if (auto Op = MemOpDesc::analyzeMaskedScatterGatherMemOp(F)) {
    return Op;
  }
  return std::nullopt;
}

std::optional<MemOpDesc> MemOpDesc::analyzeMaskedMemOp(Function &F) {
  const StringRef MangledName = F.getName();
  compiler::utils::Lexer L(MangledName);
  if (!L.Consume(VectorizationContext::InternalBuiltinPrefix)) {
    return std::nullopt;
  }

  MemOpDesc Desc;
  if (L.Consume("masked_store")) {
    if (!L.ConsumeInteger(Desc.Alignment)) {
      return std::nullopt;
    }
    if (!L.Consume("_")) {
      return std::nullopt;
    }
    Desc.IsVLOp = L.Consume("vp_");
    if (F.arg_size() != 3 + (unsigned)Desc.IsVLOp) {
      return std::nullopt;
    }

    Function::arg_iterator Arg = F.arg_begin();
    Desc.DataTy = Arg->getType();
    ++Arg;
    Desc.PtrTy = Arg->getType();
    Desc.Kind = MemOpKind::StoreCall;
    Desc.DataOpIdx = 0;
    Desc.PtrOpIdx = 1;
    Desc.MaskOpIdx = 2;
    Desc.MaskTy = F.getArg(Desc.MaskOpIdx)->getType();
    Desc.VLOpIdx = Desc.IsVLOp ? Desc.MaskOpIdx + 1 : -1;
    Desc.AccessKind = MemOpAccessKind::Masked;
    return Desc;
  }

  if (L.Consume("masked_load")) {
    if (!L.ConsumeInteger(Desc.Alignment)) {
      return std::nullopt;
    }
    if (!L.Consume("_")) {
      return std::nullopt;
    }
    Desc.IsVLOp = L.Consume("vp_");
    if (F.arg_size() != 2 + (unsigned)Desc.IsVLOp) {
      return std::nullopt;
    }

    Function::arg_iterator Arg = F.arg_begin();
    Desc.PtrTy = Arg->getType();
    Desc.DataTy = F.getReturnType();
    Desc.Kind = MemOpKind::LoadCall;
    Desc.DataOpIdx = -1;
    Desc.PtrOpIdx = 0;
    Desc.MaskOpIdx = 1;
    Desc.MaskTy = F.getArg(Desc.MaskOpIdx)->getType();
    Desc.VLOpIdx = Desc.IsVLOp ? Desc.MaskOpIdx + 1 : -1;
    Desc.AccessKind = MemOpAccessKind::Masked;
    return Desc;
  }
  return std::nullopt;
}

std::optional<MemOpDesc> MemOpDesc::analyzeInterleavedMemOp(Function &F) {
  const StringRef MangledName = F.getName();
  compiler::utils::Lexer L(MangledName);
  if (!L.Consume(VectorizationContext::InternalBuiltinPrefix)) {
    return std::nullopt;
  }
  MemOpDesc Desc;
  int ConstantStride{};
  if (L.Consume("interleaved_store")) {
    if (!L.ConsumeInteger(Desc.Alignment)) {
      return std::nullopt;
    }
    if (!L.Consume("_")) {
      return std::nullopt;
    }
    if (L.ConsumeSignedInteger(ConstantStride)) {
      VECZ_ERROR_IF(F.arg_size() != 2,
                    "Wrong argument list size for interleaved store");
      Desc.Stride = ConstantInt::get(getSizeTy(*F.getParent()), ConstantStride);
    } else if (L.Consume("V")) {
      VECZ_ERROR_IF(F.arg_size() != 3,
                    "Wrong argument list size for interleaved store");
      auto ArgIt = F.arg_begin();
      std::advance(ArgIt, 2);
      Desc.Stride = &*ArgIt;
    } else {
      return std::nullopt;
    }
    if (!L.Consume("_")) {
      return std::nullopt;
    }

    Function::arg_iterator Arg = F.arg_begin();
    Desc.DataTy = Arg->getType();
    ++Arg;
    Desc.PtrTy = Arg->getType();
    Desc.Kind = MemOpKind::StoreCall;
    Desc.DataOpIdx = 0;
    Desc.PtrOpIdx = 1;
    Desc.AccessKind = MemOpAccessKind::Interleaved;
    return Desc;
  }

  if (L.Consume("interleaved_load")) {
    if (!L.ConsumeInteger(Desc.Alignment)) {
      return std::nullopt;
    }
    if (!L.Consume("_")) {
      return std::nullopt;
    }
    if (L.ConsumeSignedInteger(ConstantStride)) {
      VECZ_ERROR_IF(F.arg_size() != 1,
                    "Wrong argument list size for interleaved load");
      Desc.Stride = ConstantInt::get(getSizeTy(*F.getParent()), ConstantStride);
    } else if (L.Consume("V")) {
      VECZ_ERROR_IF(F.arg_size() != 2,
                    "Wrong argument list size for interleaved load");
      auto ArgIt = F.arg_begin();
      std::advance(ArgIt, 1);
      Desc.Stride = &*ArgIt;
    } else {
      return std::nullopt;
    }
    if (!L.Consume("_")) {
      return std::nullopt;
    }

    Function::arg_iterator Arg = F.arg_begin();
    Desc.PtrTy = Arg->getType();
    Desc.DataTy = F.getReturnType();
    Desc.Kind = MemOpKind::LoadCall;
    Desc.DataOpIdx = -1;
    Desc.PtrOpIdx = 0;
    Desc.AccessKind = MemOpAccessKind::Interleaved;
    return Desc;
  }

  return std::nullopt;
}

std::optional<MemOpDesc> MemOpDesc::analyzeMaskedInterleavedMemOp(Function &F) {
  const StringRef MangledName = F.getName();
  compiler::utils::Lexer L(MangledName);
  if (!L.Consume(VectorizationContext::InternalBuiltinPrefix)) {
    return std::nullopt;
  }
  MemOpDesc Desc;
  if (L.Consume("masked_interleaved_store")) {
    if (!L.ConsumeInteger(Desc.Alignment)) {
      return std::nullopt;
    }
    if (!L.Consume("_")) {
      return std::nullopt;
    }
    Desc.IsVLOp = L.Consume("vp_");
    // KLOCWORK "UNINIT.STACK.MUST" possible false positive
    // Initialization of ConstantStride looks like an uninitialized access to
    // Klocwork
    int ConstantStride;
    if (L.ConsumeSignedInteger(ConstantStride)) {
      if (F.arg_size() != 3 + (unsigned)Desc.IsVLOp) {
        return std::nullopt;
      }
      Desc.Stride = ConstantInt::get(getSizeTy(*F.getParent()), ConstantStride);
    } else if (L.Consume("V")) {
      if (F.arg_size() != 4 + (unsigned)Desc.IsVLOp) {
        return std::nullopt;
      }
      auto ArgIt = F.arg_begin();
      std::advance(ArgIt, 3 + Desc.IsVLOp);
      Desc.Stride = &*ArgIt;
    } else {
      return std::nullopt;
    }
    if (!L.Consume("_")) {
      return std::nullopt;
    }

    Function::arg_iterator Arg = F.arg_begin();
    Desc.DataTy = Arg->getType();
    ++Arg;
    Desc.PtrTy = Arg->getType();
    Desc.Kind = MemOpKind::StoreCall;
    Desc.DataOpIdx = 0;
    Desc.PtrOpIdx = 1;
    Desc.MaskOpIdx = 2;
    Desc.MaskTy = F.getArg(Desc.MaskOpIdx)->getType();
    Desc.VLOpIdx = Desc.IsVLOp ? Desc.MaskOpIdx + 1 : -1;
    Desc.AccessKind = MemOpAccessKind::MaskedInterleaved;
    return Desc;
  }
  if (L.Consume("masked_interleaved_load")) {
    if (!L.ConsumeInteger(Desc.Alignment)) {
      return std::nullopt;
    }
    if (!L.Consume("_")) {
      return std::nullopt;
    }
    Desc.IsVLOp = L.Consume("vp_");
    // KLOCWORK "UNINIT.STACK.MUST" possible false positive
    // Initialization of ConstantStride looks like an uninitialized access to
    // Klocwork
    int ConstantStride;
    if (L.ConsumeSignedInteger(ConstantStride)) {
      if (F.arg_size() != 2 + (unsigned)Desc.IsVLOp) {
        return std::nullopt;
      }
      Desc.Stride = ConstantInt::get(getSizeTy(*F.getParent()), ConstantStride);
    } else if (L.Consume("V")) {
      if (F.arg_size() != 3 + (unsigned)Desc.IsVLOp) {
        return std::nullopt;
      }
      auto ArgIt = F.arg_begin();
      std::advance(ArgIt, 2 + Desc.IsVLOp);
      Desc.Stride = &*ArgIt;
    } else {
      return std::nullopt;
    }
    if (!L.Consume("_")) {
      return std::nullopt;
    }

    Function::arg_iterator Arg = F.arg_begin();
    Desc.PtrTy = Arg->getType();
    Desc.DataTy = F.getReturnType();
    Desc.Kind = MemOpKind::LoadCall;
    Desc.DataOpIdx = -1;
    Desc.PtrOpIdx = 0;
    Desc.MaskOpIdx = 1;
    Desc.MaskTy = F.getArg(Desc.MaskOpIdx)->getType();
    Desc.VLOpIdx = Desc.IsVLOp ? Desc.MaskOpIdx + 1 : -1;
    Desc.AccessKind = MemOpAccessKind::MaskedInterleaved;
    return Desc;
  }

  return std::nullopt;
}

std::optional<MemOpDesc> MemOpDesc::analyzeScatterGatherMemOp(Function &F) {
  const StringRef MangledName = F.getName();
  compiler::utils::Lexer L(MangledName);
  if (!L.Consume(VectorizationContext::InternalBuiltinPrefix)) {
    return std::nullopt;
  }
  MemOpDesc Desc;
  if (L.Consume("scatter_store")) {
    if (!L.ConsumeInteger(Desc.Alignment)) {
      return std::nullopt;
    }
    if (!L.Consume("_")) {
      return std::nullopt;
    }
    if (F.arg_size() != 2) {
      return std::nullopt;
    }

    Function::arg_iterator Arg = F.arg_begin();
    Desc.DataTy = Arg->getType();
    ++Arg;
    Desc.PtrTy = Arg->getType();
    Desc.Kind = MemOpKind::StoreCall;
    Desc.DataOpIdx = 0;
    Desc.PtrOpIdx = 1;
    Desc.AccessKind = MemOpAccessKind::ScatterGather;
    return Desc;
  }

  if (L.Consume("gather_load")) {
    if (!L.ConsumeInteger(Desc.Alignment)) {
      return std::nullopt;
    }
    if (!L.Consume("_")) {
      return std::nullopt;
    }
    if (F.arg_size() != 1) {
      return std::nullopt;
    }

    Function::arg_iterator Arg = F.arg_begin();
    Desc.PtrTy = Arg->getType();
    Desc.DataTy = F.getReturnType();
    Desc.Kind = MemOpKind::LoadCall;
    Desc.DataOpIdx = -1;
    Desc.PtrOpIdx = 0;
    Desc.AccessKind = MemOpAccessKind::ScatterGather;
    return Desc;
  }

  return std::nullopt;
}

std::optional<MemOpDesc> MemOpDesc::analyzeMaskedScatterGatherMemOp(
    Function &F) {
  const StringRef MangledName = F.getName();
  compiler::utils::Lexer L(MangledName);
  if (!L.Consume(VectorizationContext::InternalBuiltinPrefix)) {
    return std::nullopt;
  }

  MemOpDesc Desc;
  if (L.Consume("masked_scatter_store")) {
    if (!L.ConsumeInteger(Desc.Alignment)) {
      return std::nullopt;
    }
    if (!L.Consume("_")) {
      return std::nullopt;
    }
    Desc.IsVLOp = L.Consume("vp_");
    if (F.arg_size() != 3 + (unsigned)Desc.IsVLOp) {
      return std::nullopt;
    }

    Function::arg_iterator Arg = F.arg_begin();
    Desc.DataTy = Arg->getType();
    ++Arg;
    Desc.PtrTy = Arg->getType();
    Desc.Kind = MemOpKind::StoreCall;
    Desc.DataOpIdx = 0;
    Desc.PtrOpIdx = 1;
    Desc.MaskOpIdx = 2;
    Desc.MaskTy = F.getArg(Desc.MaskOpIdx)->getType();
    Desc.VLOpIdx = Desc.IsVLOp ? Desc.MaskOpIdx + 1 : -1;
    Desc.AccessKind = MemOpAccessKind::MaskedScatterGather;
    return Desc;
  }

  if (L.Consume("masked_gather_load")) {
    if (!L.ConsumeInteger(Desc.Alignment)) {
      return std::nullopt;
    }
    if (!L.Consume("_")) {
      return std::nullopt;
    }
    Desc.IsVLOp = L.Consume("vp_");
    if (F.arg_size() != 2 + (unsigned)Desc.IsVLOp) {
      return std::nullopt;
    }

    Function::arg_iterator Arg = F.arg_begin();
    Desc.PtrTy = Arg->getType();
    Desc.DataTy = F.getReturnType();
    Desc.Kind = MemOpKind::LoadCall;
    Desc.DataOpIdx = -1;
    Desc.PtrOpIdx = 0;
    Desc.MaskOpIdx = 1;
    Desc.MaskTy = F.getArg(Desc.MaskOpIdx)->getType();
    Desc.VLOpIdx = Desc.IsVLOp ? Desc.MaskOpIdx + 1 : -1;
    Desc.AccessKind = MemOpAccessKind::MaskedScatterGather;
    return Desc;
  }

  return std::nullopt;
}

////////////////////////////////////////////////////////////////////////////////

std::optional<MemOp> MemOp::get(llvm::Instruction *I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    MemOpDesc Desc;
    Desc.Kind = MemOpKind::LoadInstruction;
    Desc.Alignment = LI->getAlign().value();
    Desc.DataTy = LI->getType();
    auto *PO = LI->getPointerOperand();
    assert(PO && "Could not get pointer operand");
    Desc.PtrTy = PO->getType();
    return MemOp(I, Desc);
  }
  if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    MemOpDesc Desc;
    Desc.Kind = MemOpKind::StoreInstruction;
    Desc.Alignment = SI->getAlign().value();
    assert(SI->getValueOperand() && "Could not get value operand");
    Desc.DataTy = SI->getValueOperand()->getType();
    auto *PO = SI->getPointerOperand();
    assert(PO && "Could not get pointer operand");
    Desc.PtrTy = PO->getType();
    return MemOp(I, Desc);
  }
  if (CallInst *CI = dyn_cast<CallInst>(I)) {
    if (Function *Caller = CI->getCalledFunction()) {
      if (auto FnOp = MemOpDesc::analyzeMemOpFunction(*Caller)) {
        return MemOp(I, *FnOp);
      }
    }
  }
  return std::nullopt;
}

std::optional<MemOp> MemOp::get(llvm::CallInst *CI,
                                MemOpAccessKind AccessKind) {
  if (!CI->getCalledFunction()) {
    return std::nullopt;
  }
  std::optional<MemOpDesc> Desc;
  if (Function *Caller = CI->getCalledFunction()) {
    switch (AccessKind) {
      default:
        return std::nullopt;
      case MemOpAccessKind::Masked:
        Desc = MemOpDesc::analyzeMaskedMemOp(*Caller);
        break;
      case MemOpAccessKind::Interleaved:
        Desc = MemOpDesc::analyzeInterleavedMemOp(*Caller);
        break;
      case MemOpAccessKind::MaskedInterleaved:
        Desc = MemOpDesc::analyzeMaskedInterleavedMemOp(*Caller);
        break;
      case MemOpAccessKind::ScatterGather:
        Desc = MemOpDesc::analyzeScatterGatherMemOp(*Caller);
        break;
      case MemOpAccessKind::MaskedScatterGather:
        Desc = MemOpDesc::analyzeMaskedScatterGatherMemOp(*Caller);
        break;
    }
  }
  if (!Desc) {
    return std::nullopt;
  }
  return MemOp(CI, *Desc);
}

MemOp::MemOp(Instruction *I, const MemOpDesc &desc) {
  Ins = I;
  Desc = desc;
}

llvm::Value *MemOp::getCallOperand(int OpIdx) const {
  VECZ_FAIL_IF((Desc.getKind() != MemOpKind::LoadCall) &&
               (Desc.getKind() != MemOpKind::StoreCall));
  CallInst *CI = dyn_cast<CallInst>(Ins);
  VECZ_FAIL_IF(!CI || (OpIdx < 0) || ((unsigned)OpIdx >= CI->arg_size()));
  return CI->getArgOperand((unsigned)OpIdx);
}

bool MemOp::setCallOperand(int OpIdx, Value *V) {
  VECZ_FAIL_IF((Desc.getKind() != MemOpKind::LoadCall) &&
               (Desc.getKind() != MemOpKind::StoreCall));
  CallInst *CI = dyn_cast<CallInst>(Ins);
  VECZ_FAIL_IF(!CI || (OpIdx < 0) || ((unsigned)OpIdx >= CI->arg_size()));
  CI->setArgOperand((unsigned)OpIdx, V);
  return true;
}

llvm::Value *MemOp::getDataOperand() const {
  if (Desc.getKind() == MemOpKind::StoreInstruction) {
    return cast<StoreInst>(Ins)->getValueOperand();
  } else if (Desc.getKind() == MemOpKind::StoreCall) {
    return getCallOperand(Desc.getDataOperandIndex());
  } else {
    return nullptr;
  }
}

llvm::Value *MemOp::getPointerOperand() const {
  switch (Desc.getKind()) {
    default:
      return nullptr;
    case MemOpKind::LoadInstruction:
      return cast<LoadInst>(Ins)->getPointerOperand();
    case MemOpKind::StoreInstruction:
      return cast<StoreInst>(Ins)->getPointerOperand();
    case MemOpKind::LoadCall:
    case MemOpKind::StoreCall:
      return getCallOperand(Desc.getPointerOperandIndex());
  }
}

llvm::Value *MemOp::getMaskOperand() const {
  switch (Desc.getKind()) {
    default:
      return nullptr;
    case MemOpKind::LoadCall:
    case MemOpKind::StoreCall:
      return getCallOperand(Desc.getMaskOperandIndex());
  }
}

bool MemOp::setDataOperand(Value *V) {
  if (Desc.getKind() == MemOpKind::StoreInstruction) {
    cast<StoreInst>(Ins)->setOperand(0, V);
    return true;
  } else if (Desc.getKind() == MemOpKind::StoreCall) {
    return setCallOperand(Desc.getDataOperandIndex(), V);
  } else {
    return false;
  }
}

bool MemOp::setPointerOperand(Value *V) {
  switch (Desc.getKind()) {
    default:
      return false;
    case MemOpKind::LoadInstruction:
      cast<LoadInst>(Ins)->setOperand(0, V);
      return true;
    case MemOpKind::StoreInstruction:
      cast<StoreInst>(Ins)->setOperand(1, V);
      return true;
    case MemOpKind::LoadCall:
    case MemOpKind::StoreCall:
      return setCallOperand(Desc.getPointerOperandIndex(), V);
  }
}

bool MemOp::setMaskOperand(Value *V) {
  switch (Desc.getKind()) {
    default:
      return false;
    case MemOpKind::LoadCall:
    case MemOpKind::StoreCall:
      return setCallOperand(Desc.getMaskOperandIndex(), V);
  }
}

CallInst *MemOp::getCall() const {
  VECZ_FAIL_IF((Desc.getKind() != MemOpKind::LoadCall) &&
               (Desc.getKind() != MemOpKind::StoreCall));
  return dyn_cast<CallInst>(Ins);
}
