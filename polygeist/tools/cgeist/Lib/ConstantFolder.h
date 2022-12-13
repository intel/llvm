//===- ConstantFolder.h ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_MLIR_CONSTANT_FOLDER
#define CLANG_MLIR_CONSTANT_FOLDER

#include <type_traits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

namespace mlirclang {

// Builder that folds constant expressions.
class ConstantFolder {
public:
  explicit ConstantFolder(mlir::OpBuilder &Builder) : Builder{Builder} {}

  /// Folds a given operation if possible or returns an empty value.
  template <typename OpTy, typename... ArgTys>
  inline std::enable_if_t<OpTy::template hasTrait<mlir::OpTrait::OneResult>(),
                          mlir::Value>
  fold(mlir::Location Loc, ArgTys... Args) {
    // Return an empty value if folding is not implemented for a given
    // operation.
    return {};
  }

private:
  mlir::OpBuilder &Builder;

  /// Folds a FP cast operation not checking precision loss.
  mlir::Value foldFPCast(mlir::Location Loc, mlir::Type PromotionType,
                         mlir::arith::ConstantOp C);
};

} // namespace mlirclang

template <>
inline mlir::Value
mlirclang::ConstantFolder::fold<mlir::arith::TruncFOp, mlir::Type,
                                mlir::arith::ConstantOp>(
    mlir::Location Loc, mlir::Type PromotionType, mlir::arith::ConstantOp C) {
  return foldFPCast(Loc, PromotionType, C);
}

#endif /* CLANG_MLIR_CONSTANT_FOLDER */
