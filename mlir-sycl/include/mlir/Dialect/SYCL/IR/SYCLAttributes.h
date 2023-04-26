//===--- SYCLAttributes.h -------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SYCL_IR_SYCLATTRIBUTES_H
#define MLIR_DIALECT_SYCL_IR_SYCLATTRIBUTES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/Dialect/SYCL/IR/SYCLEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/SYCL/IR/SYCLAttributes.h.inc"
#undef GET_ATTRDEF_CLASSES

namespace mlir {
namespace sycl {

// TODO: Replace with TargetAttr -> AccessAddrSpaceAttr mapping
inline unsigned targetToAddressSpace(Target target) {
  switch (target) {
  case Target::ConstantBuffer:
  case Target::GlobalBuffer:
    return 1;
  case Target::Local:
    return 3;
  default:
    llvm_unreachable("Invalid Target for an accessor");
  }
}

} // namespace sycl
} // namespace mlir

#endif // MLIR_DIALECT_SYCL_IR_SYCLATTRIBUTES_H
