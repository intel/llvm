//===--- SYCLOpsDialect.h -------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SYCL_OPS_DIALECT_H_
#define MLIR_SYCL_OPS_DIALECT_H_

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace sycl {
/// Table in which operations implementing SYCL methods are registered.
class MethodRegistry {
public:
  /// Register a SYCL method \p methodName for the SYCL type identified by
  /// \p typeID, being \p opName the name of the SYCL operation representing
  /// this method.
  ///
  /// Calls to `lookupMethod(typeID, methodName)` will return `opName`.
  ///
  /// \return Whether the insertion happened.
  bool registerMethod(mlir::TypeID typeID, llvm::StringRef methodName,
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
} // namespace sycl
} // namespace mlir

/// Include the auto-generated header file containing the declaration of the
/// sycl dialect.
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h.inc"

#endif // MLIR_SYCL_OPS_DIALECT_H_
