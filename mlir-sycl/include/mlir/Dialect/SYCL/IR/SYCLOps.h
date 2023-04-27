//===--- SYCLOps.h --------------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SYCL_IR_SYCLOPS_H
#define MLIR_DIALECT_SYCL_IR_SYCLOPS_H

#include "mlir/Dialect/SYCL/IR/SYCLTraits.h"
#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"

#include "mlir/IR/BuiltinOps.h"

#include "mlir/Dialect/SYCL/IR/SYCLMethodOpInterface.h"

namespace mlir {
namespace sycl {

// Return true if the operation \p op belongs to the SYCL MLIR dialect.
inline bool isSYCLOperation(Operation *op) {
  if (!op || !op->getDialect())
    return false;
  return isa<sycl::SYCLDialect>(op->getDialect());
}

/// Return true if type is 'memref<?x!sycl.accessor>'.
inline bool isAccessorPtrType(Type type) {
  auto mt = dyn_cast<MemRefType>(type);
  bool isMemRefWithExpectedShape =
      (mt && mt.hasRank() && (mt.getRank() == 1) &&
       ShapedType::isDynamic(mt.getShape()[0]) && mt.getLayout().isIdentity());
  if (!isMemRefWithExpectedShape)
    return false;

  return isa<sycl::AccessorType>(mt.getElementType());
}

/// Represent Value of type 'memref<?x!sycl.accessor>'.
class AccessorPtr : public Value {
public:
  AccessorPtr(Value accessorPtr) : Value(accessorPtr) {
    assert(classof(accessorPtr) &&
           "Expecting to construct with an AccessorPtr");
  }

  sycl::AccessorType getAccessorType() {
    return mlir::cast<sycl::AccessorType>(
        mlir::cast<MemRefType>(getType()).getElementType());
  }

  bool operator<(const AccessorPtr &other) const { return impl < other.impl; }

  static bool classof(Value v) { return isAccessorPtrType(v.getType()); }
};
using AccessorPtrPair = std::pair<AccessorPtr, AccessorPtr>;

} // namespace sycl
} // namespace mlir

/// Include the auto-generated header file containing the declarations of the
/// sycl operations.
#define GET_OP_CLASSES
#include "mlir/Dialect/SYCL/IR/SYCLOps.h.inc"

#endif // MLIR_DIALECT_SYCL_IR_SYCLOPS_H
