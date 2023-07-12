//===--- SYCLHostTraits.h -------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SYCL_IR_SYCLHOSTTRAITS_H
#define MLIR_DIALECT_SYCL_IR_SYCLHOSTTRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace sycl {
LogicalResult verifySYCLHandlerOpTrait(Operation *Op);

/// This interface marks operations that represent a SYCL math function.
template <typename ConcreteType>
class SYCLHostHandlerOp
    : public OpTrait::TraitBase<ConcreteType, SYCLHostHandlerOp> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return verifySYCLHandlerOpTrait(op);
  }
};
} // namespace sycl
} // namespace mlir

#endif // MLIR_DIALECT_SYCL_IR_SYCLTRAITS_H
