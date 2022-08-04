//===- AffineUtils.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CLANG_AFFINE_UTILS_H
#define MLIR_CLANG_AFFINE_UTILS_H

#include <memory>

namespace clang {
class VarDecl;
} // end namespace clang

namespace mlir {
class Value;
class Type;
} // end namespace mlir

namespace mlirclang {

struct AffineLoopDescriptorImpl;

class AffineLoopDescriptor {
private:
  std::unique_ptr<AffineLoopDescriptorImpl> impl;

public:
  AffineLoopDescriptor();
  ~AffineLoopDescriptor();
  AffineLoopDescriptor(const AffineLoopDescriptor &) = delete;

  mlir::Value getLowerBound() const;
  void setLowerBound(mlir::Value value);

  mlir::Value getUpperBound() const;
  void setUpperBound(mlir::Value value);

  int getStep() const;
  void setStep(int value);

  clang::VarDecl *getName() const;
  void setName(clang::VarDecl *value);

  mlir::Type getType() const;
  void setType(mlir::Type type);

  bool getForwardMode() const;
  void setForwardMode(bool value);
};

} // end namespace mlirclang

#endif
