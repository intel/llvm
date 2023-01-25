//===- utils.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRCLANG_UTILS_H
#define MLIR_TOOLS_MLIRCLANG_UTILS_H

#include "llvm/Support/CommandLine.h"

extern llvm::cl::opt<bool> SuppressWarnings;

#define CGEIST_WARNING(X)                                                      \
  if (!SuppressWarnings) {                                                     \
    X;                                                                         \
  }

namespace clang {
class DeclContext;
}

namespace mlir {
class OpBuilder;
class Operation;
class ModuleOp;
class Value;
class FunctionOpInterface;

namespace func {
class FuncOp;
}

namespace gpu {
class GPUModuleOp;
}
} // namespace mlir

namespace llvm {
class raw_ostream;
template <typename> class SmallVectorImpl;
class StringRef;
} // namespace llvm

enum class FunctionContext;

namespace mlirclang {

/// Whether the parent namespace is the sycl namespace, a namespace within it or
/// not.
enum class NamespaceKind {
  Other = 0,  /// namespace is not 'sycl'
  SYCL,       /// the 'sycl' namespace
  WithinSYCL, /// nested in the sycl namespace
};

/// Replace the given function by the operation with the given name, and use the
/// same argument list. For example, if the function is @foo(%a, %b) and opName
/// is "bar.baz", we will create an operator baz of the bar dialect, with
/// operands %a and %b. The new op will be inserted at where the insertion point
/// of the provided OpBuilder is.
mlir::Operation *
replaceFuncByOperation(mlir::func::FuncOp F, llvm::StringRef OpName,
                       mlir::OpBuilder &B,
                       llvm::SmallVectorImpl<mlir::Value> &Input,
                       llvm::SmallVectorImpl<mlir::Value> &Output);

NamespaceKind getNamespaceKind(const clang::DeclContext *DC);

/// Return the insertion context of the input builder.
FunctionContext getInputContext(const mlir::OpBuilder &Builder);

/// Return the device module in the input module.
mlir::gpu::GPUModuleOp getDeviceModule(mlir::ModuleOp Module);

/// Return the function context
FunctionContext getFuncContext(mlir::FunctionOpInterface Function);

/// Set the OpBuilder \p Builder insertion point depending on the given
/// FunctionContext \p FuncContext.
void setInsertionPoint(mlir::OpBuilder &Builder, FunctionContext FuncContext,
                       mlir::ModuleOp Module);

} // namespace mlirclang

#endif // MLIR_TOOLS_MLIRCLANG_UTILS_H
