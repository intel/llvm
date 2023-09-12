//===-- FPAccuracy.cpp ---- FP Accuracy Support ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file contains the implementations of functions that map standard
/// accuracy levels to required accuracy expressed in terms of ULPs.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/FPAccuracy.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"

namespace llvm {

static bool isFPBuiltinIntrinsic(Intrinsic::ID IID) {
  switch (IID) {
#define OPERATION(NAME, INTRINSIC) case Intrinsic::INTRINSIC:
#include "llvm/IR/FPBuiltinOps.def"
    return true;
  default:
    return false;
  }
}

static StringRef lookupSyclFloatAccuracy(Intrinsic::ID IID) {
  switch (IID) {
#define FP_ACCURACY(INTRINSIC, SYCL_FLOAT_ACCURACY, SDA, CFA, CDA)             \
  case Intrinsic::INTRINSIC:                                                   \
    return SYCL_FLOAT_ACCURACY;
#include "llvm/IR/FPAccuracy.def"
  default:
    return StringRef();
  }
}

static StringRef lookupSyclDoubleAccuracy(Intrinsic::ID IID) {
  switch (IID) {
#define FP_ACCURACY(INTRINSIC, SFA, SYCL_DOUBLE_ACCURACY, CFA, CDA)            \
  case Intrinsic::INTRINSIC:                                                   \
    return SYCL_DOUBLE_ACCURACY;
#include "llvm/IR/FPAccuracy.def"
  default:
    return StringRef();
  }
}

static StringRef lookupCudaFloatAccuracy(Intrinsic::ID IID) {
  switch (IID) {
#define FP_ACCURACY(INTRINSIC, SFA, SDA, CUDA_FLOAT_ACCURACY, CDA)             \
  case Intrinsic::INTRINSIC:                                                   \
    return CUDA_FLOAT_ACCURACY;
#include "llvm/IR/FPAccuracy.def"
  default:
    return StringRef();
  }
}

static StringRef lookupCudaDoubleAccuracy(Intrinsic::ID IID) {
  switch (IID) {
#define FP_ACCURACY(INTRINSIC, SFA, SDA, CFA, CUDA_DOUBLE_ACCURACY)            \
  case Intrinsic::INTRINSIC:                                                   \
    return CUDA_DOUBLE_ACCURACY;
#include "llvm/IR/FPAccuracy.def"
  default:
    return StringRef();
  }
}

StringRef fp::getAccuracyForFPBuiltin(Intrinsic::ID IID, const Type *Ty,
                                      fp::FPAccuracy AccuracyLevel) {
  assert(isFPBuiltinIntrinsic(IID) && "Invalid intrinsic ID for FPAccuracy");

  assert(Ty->isFPOrFPVectorTy() && "Invalid type for FPAccuracy");

  // Vector fpbuiltins have the same accuracy requirements as the corresponding
  // scalar operation.
  if (const auto *VecTy = dyn_cast<VectorType>(Ty))
    Ty = VecTy->getElementType();

  // This will probably change at some point.
  assert((Ty->isFloatTy() || Ty->isDoubleTy()) &&
         "Invalid type for FPAccuracy");

  // High and medium accuracy have the same requirement for all functions
  if (AccuracyLevel == fp::FPAccuracy::High)
    return "1.0";
  if (AccuracyLevel == fp::FPAccuracy::Medium)
    return "4.0";

  // Low accuracy is computed in terms of accurate bits, so it depends on the
  // type
  if (AccuracyLevel == fp::FPAccuracy::Low) {
    if (Ty->isFloatTy())
      return "8192.0";
    if (Ty->isDoubleTy())
      return "67108864.0"; // 2^(53-26-1) == 26-bits of accuracy

    // Other types are not supported
    llvm_unreachable("Unexpected type for FPAccuracy");
  }

  assert((AccuracyLevel == fp::FPAccuracy::SYCL ||
          AccuracyLevel == fp::FPAccuracy::CUDA) &&
         "Unexpected FPAccuracy level");

  if (Ty->isFloatTy()) {
    if (AccuracyLevel == fp::FPAccuracy::SYCL)
      return lookupSyclFloatAccuracy(IID);
    if (AccuracyLevel == fp::FPAccuracy::CUDA)
      return lookupCudaFloatAccuracy(IID);
    llvm_unreachable("Unexpected FPAccuracy level");
  } else if (Ty->isDoubleTy()) {
    if (AccuracyLevel == fp::FPAccuracy::SYCL)
      return lookupSyclDoubleAccuracy(IID);
    if (AccuracyLevel == fp::FPAccuracy::CUDA)
      return lookupCudaDoubleAccuracy(IID);
    llvm_unreachable("Unexpected FPAccuracy level");
  } else {
    // This is here for error detection if the logic above is changed.
    llvm_unreachable("Unexpected type for FPAccuracy");
  }
}

} // namespace llvm
