// Copyright (C) Codeplay Software Limited

//===--- SYCLOpsDialect.h -------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SYCL_OPS_DIALECT_H_
#define MLIR_SYCL_OPS_DIALECT_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace sycl {
class MethodRegistry {
public:
  /// Register the methods in \p MethodNames for the SYCL type identified by
  /// \p TypeID, being \p OpName the name of the SYCL operation representing
  /// this method.
  ///
  /// Calls to `findMethod(TypeID, name)` (being name present in
  /// `MethodNames`), will return `OpName`.
  void registerMethod(mlir::TypeID typeID, llvm::StringRef methodName,
                      llvm::StringRef opName);

  /// Returns the name of the operation implementing the queried
  /// method, if present.
  ///
  /// For a method to be queried, it must have been registered
  /// first.
  llvm::Optional<llvm::StringRef> lookupMethod(::mlir::TypeID Type,
                                               llvm::StringRef Name) const;

private:
  llvm::DenseMap<std::pair<mlir::TypeID, llvm::StringRef>, llvm::StringRef>
      methods;
};

class SYCLContext {
public:
  /// Register the methods in \p MethodNames for the SYCL type identified by
  /// \p TypeID, being \p OpName the name of the SYCL operation representing
  /// this method.
  ///
  /// Calls to `findMethod(TypeID, name)` (being name present in
  /// `MethodNames`), will return `OpName`.
  void addMethod(mlir::TypeID typeID, llvm::StringRef methodName,
                 llvm::StringRef opName);

  /// Returns the name of the operation implementing the queried
  /// method, if present.
  ///
  /// For a method to be queried, it must have been registered
  /// first.
  llvm::Optional<llvm::StringRef> findMethod(::mlir::TypeID Type,
                                             llvm::StringRef Name) const;

private:
  MethodRegistry methodRegistry;
};
} // namespace sycl
} // namespace mlir

/// Include the auto-generated header file containing the declaration of the
/// sycl dialect.
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h.inc"

/// Include the header file containing the declaration of the sycl operation
/// interfaces.
#include "mlir/Dialect/SYCL/IR/SYCLOpInterfaces.h"

/// Include the auto-generated header file containing the declarations of the
/// sycl operations.
#define GET_OP_CLASSES
#include "mlir/Dialect/SYCL/IR/SYCLOps.h.inc"

#endif // MLIR_SYCL_OPS_DIALECT_H_
