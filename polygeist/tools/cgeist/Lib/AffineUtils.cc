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

namespace mlirclang {

struct AffineLoopDescriptorImpl {
  mlir::Value upperBound = nullptr;
  mlir::Value lowerBound = nullptr;
  int64_t step = std::numeric_limits<int64_t>::max();
  mlir::Type indVarType = nullptr;
  clang::VarDecl *indVar = nullptr;
  bool forwardMode = true;
};

AffineLoopDescriptor::AffineLoopDescriptor()
    : impl(std::make_unique<AffineLoopDescriptorImpl>()) {}
AffineLoopDescriptor::~AffineLoopDescriptor() = default;

mlir::Value AffineLoopDescriptor::getLowerBound() const {
  return impl->lowerBound;
}
void AffineLoopDescriptor::setLowerBound(mlir::Value value) {
  impl->lowerBound = value;
}

mlir::Value AffineLoopDescriptor::getUpperBound() const {
  return impl->upperBound;
}
void AffineLoopDescriptor::setUpperBound(mlir::Value value) {
  impl->upperBound = value;
}

int AffineLoopDescriptor::getStep() const { return impl->step; }
void AffineLoopDescriptor::setStep(int value) { impl->step = value; }

clang::VarDecl *AffineLoopDescriptor::getName() const { return impl->indVar; }
void AffineLoopDescriptor::setName(clang::VarDecl *value) {
  impl->indVar = value;
}

mlir::Type AffineLoopDescriptor::getType() const { return impl->indVarType; }
void AffineLoopDescriptor::setType(mlir::Type type) { impl->indVarType = type; }

bool AffineLoopDescriptor::getForwardMode() const { return impl->forwardMode; }
void AffineLoopDescriptor::setForwardMode(bool value) {
  impl->forwardMode = value;
}

} // namespace mlirclang
