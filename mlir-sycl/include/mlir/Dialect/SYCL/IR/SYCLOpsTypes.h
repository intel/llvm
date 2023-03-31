//===--- SYCLOpsTypes.h ---------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SYCL_OPS_TYPES_H_
#define MLIR_SYCL_OPS_TYPES_H_

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpAttributes.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace sycl {
template <typename Parameter> class SYCLInheritanceTypeTrait {
public:
  template <typename ConcreteType>
  class Trait : public mlir::TypeTrait::TraitBase<ConcreteType, Trait> {};
};

/// Return true if the given \p Ty is a SYCL type.
inline bool isSYCLType(Type Ty) { return isa<SYCLDialect>(Ty.getDialect()); }

/// Return the number of dimensions of type \p Ty.
unsigned getDimensions(Type Ty);

llvm::SmallVector<mlir::TypeID> getDerivedTypes(mlir::TypeID TypeID);
} // namespace sycl
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h.inc"

#endif // MLIR_SYCL_OPS_DIALECT_H_
