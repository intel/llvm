//===- SPIRVSanitizerCommonUtils.cpp- SPIRV Sanitizer commnon utils ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common infrastructure for SPIRV Sanitizer.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/SPIRVSanitizerCommonUtils.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Process.h"

using namespace llvm;

namespace llvm {
TargetExtType *getTargetExtType(Type *Ty) {
  if (auto *TargetTy = dyn_cast<TargetExtType>(Ty))
    return TargetTy;

  if (Ty->isVectorTy())
    return getTargetExtType(Ty->getScalarType());

  if (Ty->isArrayTy())
    return getTargetExtType(Ty->getArrayElementType());

  if (auto *STy = dyn_cast<StructType>(Ty)) {
    for (unsigned int i = 0; i < STy->getNumElements(); i++)
      if (auto *TargetTy = getTargetExtType(STy->getElementType(i)))
        return TargetTy;
    return nullptr;
  }

  return nullptr;
}

// Skip pointer operand that is sycl joint matrix access since it isn't from
// user code, e.g. %call:
// clang-format off
// %a = alloca %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix", align 8
// %0 = getelementptr inbounds %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix", ptr %a, i64 0, i32 0
// %call = call spir_func ptr
// @_Z19__spirv_AccessChainIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm8ELm8ELN5__spv9MatrixUseE0ELNS8_5Scope4FlagE3EEPT_PPNS8_28__spirv_CooperativeMatrixKHRIT0_XT4_EXT1_EXT2_EXT3_EEEm(ptr %0, i64 0)
// %1 = load float, ptr %call, align 4
// store float %1, ptr %call, align 4
// clang-format on
bool isJointMatrixAccess(Value *V) {
  auto *ActualV = V->stripInBoundsOffsets();
  if (auto *CI = dyn_cast<CallInst>(ActualV)) {
    for (Value *Op : CI->args()) {
      if (auto *AI = dyn_cast<AllocaInst>(Op->stripInBoundsOffsets()))
        if (auto *TargetTy = getTargetExtType(AI->getAllocatedType()))
          return TargetTy->getName().starts_with("spirv.") &&
                 TargetTy->getName().contains("Matrix");
    }
  }
  return false;
}

void getFunctionsOfUser(User *User, SmallVectorImpl<Function *> &Functions) {
  if (Instruction *Inst = dyn_cast<Instruction>(User)) {
    Functions.push_back(Inst->getFunction());
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(User)) {
    for (auto *U : CE->users())
      getFunctionsOfUser(U, Functions);
  }
}

// Must be random, not a content hash: identical device images (e.g. a
// header-only kernel built into two libraries) would hash alike and their
// external-linkage metadata globals would collide when linked into one
// program.
SmallString<40> computeMetadataUniqueId(Module &M) {
  // Return the cached hash if already computed for this module.
  constexpr StringRef CacheKey = "device.sanitizer.hash";
  if (NamedMDNode *NMD = M.getNamedMetadata(CacheKey))
    if (NMD->getNumOperands() > 0)
      if (auto *MS = dyn_cast<MDString>(NMD->getOperand(0)->getOperand(0)))
        return MS->getString();

  uint32_t Parts[4];
  for (uint32_t &P : Parts)
    P = sys::Process::GetRandomNumber();

  SmallString<40> UniqueId;
  raw_svector_ostream OS(UniqueId);
  for (uint32_t P : Parts)
    OS << format_hex_no_prefix(P, 8);

  // Cache the result in the module so subsequent calls are stable.
  NamedMDNode *NMD = M.getOrInsertNamedMetadata(CacheKey);
  NMD->addOperand(
      MDNode::get(M.getContext(), {MDString::get(M.getContext(), UniqueId)}));

  return UniqueId;
}

bool hasESIMDKernel(Module &M) {
  for (auto &F : M)
    if (F.hasMetadata("sycl_explicit_simd"))
      return true;
  return false;
}

} // namespace llvm
