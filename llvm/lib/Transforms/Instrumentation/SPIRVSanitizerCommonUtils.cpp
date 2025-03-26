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
} // namespace llvm
