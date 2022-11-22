//===- utils.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRCLANG_UTILS_H
#define MLIR_TOOLS_MLIRCLANG_UTILS_H

#include "Lib/clang-mlir.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "clang/AST/DeclBase.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class Operation;
namespace func {
class FuncOp;
}
class Value;
class OpBuilder;
class AbstractOperation;
class Type;
} // namespace mlir

namespace llvm {
class StringRef;
} // namespace llvm

namespace clang {
class Expr;
}

class MLIRScanner;

namespace mlirclang {

/// Replace the given function by the operation with the given name, and use the
/// same argument list. For example, if the function is @foo(%a, %b) and opName
/// is "bar.baz", we will create an operator baz of the bar dialect, with
/// operands %a and %b. The new op will be inserted at where the insertion point
/// of the provided OpBuilder is.
mlir::Operation *
replaceFuncByOperation(mlir::func::FuncOp Func, llvm::StringRef OpName,
                       mlir::OpBuilder &B,
                       llvm::SmallVectorImpl<mlir::Value> &Input,
                       llvm::SmallVectorImpl<mlir::Value> &Output);

bool isNamespaceSYCL(const clang::DeclContext *DC);

/// Return the insertion context of the input builder.
FunctionContext getInputContext(const mlir::OpBuilder &Builder);

/// Return the device module in the input module.
mlir::gpu::GPUModuleOp getDeviceModule(mlir::ModuleOp Module);

/// Emit a warning if -w is not in effect.
llvm::raw_ostream &warning();

} // namespace mlirclang

#endif // MLIR_TOOLS_MLIRCLANG_UTILS_H
