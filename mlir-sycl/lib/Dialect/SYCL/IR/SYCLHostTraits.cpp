//===--- SYCLHostTraits.cpp -----------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/IR/SYCLHostTraits.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

using namespace mlir;
using namespace mlir::sycl;

LogicalResult mlir::sycl::verifySYCLHandlerOpTrait(Operation *op) {
  if (op->getNumOperands() == 0)
    return op->emitOpError("requires at least one operand");
  if (!isa<LLVM::LLVMPointerType>(op->getOperand(0).getType()))
    return op->emitOpError("requires the first argument to be a pointer");
  return success();
}
