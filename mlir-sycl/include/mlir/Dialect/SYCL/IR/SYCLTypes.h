//===--- SYCLTypes.h ------------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SYCL_IR_SYCLTYPES_H
#define MLIR_DIALECT_SYCL_IR_SYCLTYPES_H

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SYCL/IR/SYCLAttributes.h"
#include "mlir/Dialect/SYCL/IR/SYCLDialect.h"
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

/// Return true if the given \p type is a SYCL type.
inline bool isSYCLType(Type type) {
  return isa<SYCLDialect>(type.getDialect());
}

/// Return the number of dimensions of type \p type.
unsigned getDimensions(Type type);

llvm::SmallVector<mlir::TypeID> getDerivedTypes(mlir::TypeID typeID);
} // namespace sycl
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h.inc"

namespace mlir {
namespace sycl {

/// Return true if type is 'memref<?xT>'.
template <typename T> bool isPtrOf(Type type);

/// Represent Value of type 'memref<?x!sycl.accessor>'.
class AccessorPtrValue : public Value {
public:
  AccessorPtrValue(Value val) : Value(val) {
    assert(classof(val) && "val should be an 'AccessorPtrValue");
  }

  AccessorType getAccessorType() {
    return mlir::cast<AccessorType>(
        mlir::cast<MemRefType>(getType()).getElementType());
  }

  bool operator<(const AccessorPtrValue &other) const {
    return impl < other.impl;
  }

  static bool classof(Value v);
};
using AccessorPtrPair = std::pair<AccessorPtrValue, AccessorPtrValue>;

/// Return whether \p baseType is a base type of \p derivedType
bool isBaseClass(Type baseType, Type derivedType);

} // namespace sycl
} // namespace mlir

#endif // MLIR_DIALECT_SYCL_IR_SYCLTYPES_H
