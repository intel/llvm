//===------------ ESIMDUtils.cpp - ESIMD utility functions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions for processing ESIMD code.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/ESIMD/ESIMDUtils.h"

#include "llvm/GenXIntrinsics/GenXIntrinsics.h"
#include "llvm/GenXIntrinsics/GenXMetadata.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Casting.h"

namespace llvm {
namespace esimd {

constexpr char SLM_ALLOC_PREFIX[] = "_Z17__esimd_slm_alloc";
constexpr char SLM_FREE_PREFIX[] = "_Z16__esimd_slm_free";
constexpr char SLM_INIT_PREFIX[] = "_Z16__esimd_slm_init";

constexpr char SLM_ALLOCATOR_CTOR_DTOR_PREFIX[] =
    "_ZN4sycl3_V13ext5intel5esimd13slm_allocatorILi";
constexpr char SLM_ALLOCATOR_CTOR_SUFFIX[] = "EEC2Ev";
constexpr char SLM_ALLOCATOR_DTOR_SUFFIX[] = "EED2Ev";

bool isSlmAllocatorConstructor(const Function &F) {
  auto Name = F.getName();
  return Name.starts_with(SLM_ALLOCATOR_CTOR_DTOR_PREFIX) &&
         Name.endswith(SLM_ALLOCATOR_CTOR_SUFFIX);
}

bool isSlmAllocatorDestructor(const Function &F) {
  auto Name = F.getName();
  return Name.starts_with(SLM_ALLOCATOR_CTOR_DTOR_PREFIX) &&
         Name.endswith(SLM_ALLOCATOR_DTOR_SUFFIX);
}

bool isSlmInit(const Function &F) {
  return F.getName().starts_with(SLM_INIT_PREFIX);
}

bool isSlmAlloc(const Function &F) {
  return F.getName().starts_with(SLM_ALLOC_PREFIX);
}

bool isSlmFree(const Function &F) {
  return F.getName().starts_with(SLM_FREE_PREFIX);
}

bool isAssertFail(const Function &F) {
  return F.getName().starts_with("__assert_fail") ||
         F.getName().starts_with("__devicelib_assert_fail");
}

bool isESIMD(const Function &F) {
  return F.getMetadata(ESIMD_MARKER_MD) != nullptr;
}

bool isKernel(const Function &F) {
  return (F.getCallingConv() == CallingConv::SPIR_KERNEL);
}

bool isESIMDKernel(const Function &F) { return isKernel(F) && isESIMD(F); }

Type *getVectorTyOrNull(StructType *STy) {
  Type *Res = nullptr;
  while (STy && (STy->getStructNumElements() == 1)) {
    Res = STy->getStructElementType(0);
    STy = dyn_cast<StructType>(Res);
  }
  if (!Res || !Res->isVectorTy())
    return nullptr;
  return Res;
}

UpdateUint64MetaDataToMaxValue::UpdateUint64MetaDataToMaxValue(
    Module &M, genx::KernelMDOp Key, uint64_t NewVal)
    : M(M), Key(Key), NewVal(NewVal) {
  // Pre-select nodes for update to do less work in the '()' operator.
  llvm::NamedMDNode *GenXKernelMD = M.getNamedMetadata(GENX_KERNEL_METADATA);
  llvm::esimd::assert_and_diag(GenXKernelMD, "invalid genx.kernels metadata");
  for (auto Node : GenXKernelMD->operands()) {
    if (Node->getNumOperands() <= (unsigned)Key) {
      continue;
    }
    llvm::Value *Old = getValue(Node->getOperand(Key));
    uint64_t OldVal = cast<llvm::ConstantInt>(Old)->getZExtValue();

    if (OldVal < NewVal) {
      CandidatesToUpdate.push_back(Node);
    }
  }
}

void UpdateUint64MetaDataToMaxValue::operator()(Function *F) const {
  // Update the meta data attribute for the current function.
  for (auto Node : CandidatesToUpdate) {
    assert(Node->getNumOperands() > (unsigned)Key);

    if (getValue(Node->getOperand(genx::KernelMDOp::FunctionRef)) != F) {
      continue;
    }
    llvm::Value *Old = getValue(Node->getOperand(Key));
#ifndef NDEBUG
    uint64_t OldVal = cast<llvm::ConstantInt>(Old)->getZExtValue();
    assert(OldVal < NewVal);
#endif // NDEBUG
    llvm::Value *New = llvm::ConstantInt::get(Old->getType(), NewVal);
    Node->replaceOperandWith(Key, getMetadata(New));
  }
}

} // namespace esimd
} // namespace llvm
