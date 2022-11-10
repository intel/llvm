//===--- SYCLOpsDialect.h -------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SYCL_OPS_DIALECT_H_
#define MLIR_SYCL_OPS_DIALECT_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

namespace mlir {
namespace sycl {
/// Table in which operations implementing SYCL methods are registered.
class MethodRegistry {
public:
  /// Initializes this registry.
  void init(mlir::MLIRContext &Ctx);

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

  /// Add a definition for the given method to be used when lowering
  /// SYCLMethodOpInterface instances.
  ///
  /// A call to this function will fail if \p MayOverride is false and we try to
  /// override an already present definition.
  void registerDefinition(llvm::StringRef methodName,
                          mlir::func::FuncOp Definition,
                          bool MayOverride = false);

  /// Retrieve a function definition previously registered with
  /// provideMethodDefinition().
  llvm::Optional<mlir::func::FuncOp>
  lookupDefinition(llvm::StringRef methodName,
                   mlir::FunctionType FunctionType) const;

private:
  static constexpr llvm::StringLiteral ModuleName{"SYCLDefs"};

  llvm::DenseMap<std::pair<mlir::TypeID, llvm::StringRef>, llvm::StringRef>
      methods;
  llvm::DenseMap<std::pair<llvm::SmallString<0>, mlir::FunctionType>,
                 mlir::func::FuncOp>
      definitions;
  mlir::ModuleOp Module;
};
} // namespace sycl
} // namespace mlir

/// Include the auto-generated header file containing the declaration of the
/// sycl dialect.
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h.inc"

#endif // MLIR_SYCL_OPS_DIALECT_H_
