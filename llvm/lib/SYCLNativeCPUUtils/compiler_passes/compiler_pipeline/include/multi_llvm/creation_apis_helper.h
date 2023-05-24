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
#ifndef MULTI_LLVM_CREATION_APIS_HELPER_H_INCLUDED
#define MULTI_LLVM_CREATION_APIS_HELPER_H_INCLUDED

#include <llvm/ADT/None.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/TypeSize.h>
#include <multi_llvm/vector_type_helper.h>

namespace multi_llvm {

inline llvm::Value *createAllTrueMask(llvm::IRBuilder<> &B,
                                      llvm::ElementCount EC) {
  return llvm::ConstantInt::getTrue(llvm::VectorType::get(B.getInt1Ty(), EC));
}

inline llvm::Value *createIndexSequence(llvm::IRBuilder<> &Builder,
                                        llvm::Type *Ty, llvm::ElementCount EC,
                                        const llvm::Twine &Name = "") {
  (void)Builder;
  (void)Name;
  if (EC.isScalable()) {
    // FIXME: This intrinsic works on fixed-length types too: should we migrate
    // to using it starting from LLVM 13?
    return Builder.CreateStepVector(Ty, Name);
  }

  llvm::SmallVector<llvm::Constant *, 16> Indices;
  unsigned SimdWidth = EC.getFixedValue();
  for (unsigned i = 0; i < SimdWidth; i++) {
    Indices.push_back(llvm::ConstantInt::get(getVectorElementType(Ty), i));
  }
  return llvm::ConstantVector::get(Indices);
}

inline llvm::CallInst *createRISCVMaskedIntrinsic(
    llvm::IRBuilder<> &B, llvm::Intrinsic::ID ID,
    llvm::ArrayRef<llvm::Type *> Types, llvm::ArrayRef<llvm::Value *> Args,
    unsigned TailPolicy, llvm::Instruction *FMFSource = nullptr,
    const llvm::Twine &Name = "") {
  llvm::SmallVector<llvm::Value *> InArgs(Args.begin(), Args.end());
  InArgs.push_back(
      B.getIntN(Args.back()->getType()->getIntegerBitWidth(), TailPolicy));
  return B.CreateIntrinsic(ID, Types, InArgs, FMFSource, Name);
}

}  // namespace multi_llvm

#endif  // MULTI_LLVM_CREATION_APIS_HELPER_H_INCLUDED
