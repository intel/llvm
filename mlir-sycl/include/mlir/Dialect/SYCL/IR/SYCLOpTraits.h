//===--- SYCLOpTraits.h ---------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SYCL_OPS_TRAITS_H_
#define MLIR_SYCL_OPS_TRAITS_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace sycl {
LogicalResult verifySYCLGetIDTrait(Operation *Op);
LogicalResult verifySYCLGetComponentTrait(Operation *Op);
LogicalResult verifySYCLGetRangeTrait(Operation *Op);
LogicalResult verifySYCLGetGroupTrait(Operation *Op);
LogicalResult verifySYCLIndexSpaceGetIDTrait(Operation *Op);
LogicalResult verifySYCLIndexSpaceGetRangeTrait(Operation *Op);

///  This interface describes an SYCLMethodOpInterface that returns a range if
///  called with a single argument and a size_t if called with two arguments.
///
///  Special cases are:
///  * The `operator size_t` function, that can only receive a single argument
///  and return a size_t;
///  * The `operator[]` function, that must receive two arguments and return a
///  size_t.
template <typename ConcreteType>
class SYCLGetID : public OpTrait::TraitBase<ConcreteType, SYCLGetID> {
public:
  static LogicalResult verifyTrait(Operation *Op) {
    return verifySYCLGetIDTrait(Op);
  }
};

/// This interface describes an SYCLMethodOpInterface that returns a range if
/// called with a single argument and a size_t if called with two arguments.
template <typename ConcreteType>
class SYCLGetComponent
    : public OpTrait::TraitBase<ConcreteType, SYCLGetComponent> {
public:
  static LogicalResult verifyTrait(Operation *Op) {
    return verifySYCLGetComponentTrait(Op);
  }
};

/// This interface describes an SYCLMethodOpInterface that returns a range if
/// called with a single argument and a size_t if called with two arguments.
template <typename ConcreteType>
class SYCLGetRange : public OpTrait::TraitBase<ConcreteType, SYCLGetRange> {
public:
  static LogicalResult verifyTrait(Operation *Op) {
    return verifySYCLGetRangeTrait(Op);
  }
};

/// This interface describes an SYCLMethodOpInterface that returns a group if
/// called with a single argument and a size_t if called with two arguments.
template <typename ConcreteType>
class SYCLGetGroup : public OpTrait::TraitBase<ConcreteType, SYCLGetGroup> {
public:
  static LogicalResult verifyTrait(Operation *Op) {
    return verifySYCLGetGroupTrait(Op);
  }
};

///  This interface describes an operation returning either a SYCL ID type (for
///  cardinality 0) or an index type (for cardinality 1).
template <typename ConcreteType>
class SYCLIndexSpaceGetID
    : public OpTrait::TraitBase<ConcreteType, SYCLIndexSpaceGetID> {
public:
  static LogicalResult verifyTrait(Operation *Op) {
    return verifySYCLIndexSpaceGetIDTrait(Op);
  }
};

///  This interface describes an operation returning either a SYCL range type
///  (for cardinality 0) or an index type (for cardinality 1).
template <typename ConcreteType>
class SYCLIndexSpaceGetRange
    : public OpTrait::TraitBase<ConcreteType, SYCLIndexSpaceGetRange> {
public:
  static LogicalResult verifyTrait(Operation *Op) {
    return verifySYCLIndexSpaceGetRangeTrait(Op);
  }
};
} // namespace sycl
} // namespace mlir

#endif // MLIR_SYCL_OPS_TRAITS_H_
