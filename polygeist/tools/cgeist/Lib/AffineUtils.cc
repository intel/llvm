//===- AffineUtils.c --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AffineUtils.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "clang/AST/Decl.h"

#include <limits>

using namespace mlir;

namespace mlirclang {

struct AffineLoopDescriptorImpl {
  Value UpperBound = nullptr;
  Value LowerBound = nullptr;
  int64_t Step = std::numeric_limits<int64_t>::max();
  Type IndVarType = nullptr;
  clang::VarDecl *IndVar = nullptr;
  bool ForwardMode = true;
};

AffineLoopDescriptor::AffineLoopDescriptor()
    : Impl(std::make_unique<AffineLoopDescriptorImpl>()) {}
AffineLoopDescriptor::~AffineLoopDescriptor() = default;

Value AffineLoopDescriptor::getLowerBound() const { return Impl->LowerBound; }
void AffineLoopDescriptor::setLowerBound(Value Value) {
  Impl->LowerBound = Value;
}

Value AffineLoopDescriptor::getUpperBound() const { return Impl->UpperBound; }
void AffineLoopDescriptor::setUpperBound(Value Value) {
  Impl->UpperBound = Value;
}

int AffineLoopDescriptor::getStep() const { return Impl->Step; }
void AffineLoopDescriptor::setStep(int Value) { Impl->Step = Value; }

clang::VarDecl *AffineLoopDescriptor::getName() const { return Impl->IndVar; }
void AffineLoopDescriptor::setName(clang::VarDecl *Value) {
  Impl->IndVar = Value;
}

Type AffineLoopDescriptor::getType() const { return Impl->IndVarType; }
void AffineLoopDescriptor::setType(Type Type) { Impl->IndVarType = Type; }

bool AffineLoopDescriptor::getForwardMode() const { return Impl->ForwardMode; }
void AffineLoopDescriptor::setForwardMode(bool Value) {
  Impl->ForwardMode = Value;
}

} // namespace mlirclang
