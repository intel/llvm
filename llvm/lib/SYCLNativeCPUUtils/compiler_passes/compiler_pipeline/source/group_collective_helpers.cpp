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

#include <compiler/utils/group_collective_helpers.h>
#include <compiler/utils/mangling.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

using namespace llvm;
static llvm::Constant *getNeutralIdentityHelper(RecurKind Kind, Type *Ty,
                                                bool UseNaN, bool UseFZero) {
  switch (Kind) {
    default:
      return nullptr;
    case RecurKind::And:
      return ConstantInt::getAllOnesValue(Ty);
    case RecurKind::Or:
    case RecurKind::Add:
    case RecurKind::Xor:
      return ConstantInt::getNullValue(Ty);
    case RecurKind::SMin:
      return ConstantInt::get(
          Ty, APInt::getSignedMaxValue(Ty->getScalarSizeInBits()));
    case RecurKind::SMax:
      return ConstantInt::get(
          Ty, APInt::getSignedMinValue(Ty->getScalarSizeInBits()));
    case RecurKind::UMin:
      return ConstantInt::get(Ty,
                              APInt::getMaxValue(Ty->getScalarSizeInBits()));
    case RecurKind::UMax:
      return ConstantInt::get(Ty,
                              APInt::getMinValue(Ty->getScalarSizeInBits()));
    case RecurKind::FAdd:
      // -0.0 + 0.0 = 0.0 meaning -0.0 (not 0.0) is the neutral value for floats
      // under addition.
      return UseFZero ? ConstantFP::get(Ty, 0.0) : ConstantFP::get(Ty, -0.0);
    case RecurKind::FMin:
      return UseNaN ? ConstantFP::getQNaN(Ty, /*Negative*/ false)
                    : ConstantFP::getInfinity(Ty, /*Negative*/ false);
    case RecurKind::FMax:
      return UseNaN ? ConstantFP::getQNaN(Ty, /*Negative*/ true)
                    : ConstantFP::getInfinity(Ty, /*Negative*/ true);
    case RecurKind::Mul:
      return ConstantInt::get(Ty, 1);
    case RecurKind::FMul:
      return ConstantFP::get(Ty, 1.0);
  }
}

llvm::Constant *compiler::utils::getNeutralVal(RecurKind Kind, Type *Ty) {
  return getNeutralIdentityHelper(Kind, Ty, /*UseNaN*/ true,
                                  /*UseFZero*/ false);
}

llvm::Constant *compiler::utils::getIdentityVal(RecurKind Kind, Type *Ty) {
  return getNeutralIdentityHelper(Kind, Ty, /*UseNaN*/ false, /*UseFZero*/
                                  true);
}
