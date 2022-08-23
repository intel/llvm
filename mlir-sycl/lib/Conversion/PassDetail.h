//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_CONVERSION_PASSDETAIL_H_
#define SYCL_CONVERSION_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionInterfaces.h"

namespace mlir {

namespace LLVM {
class LLVMDialect;
}

namespace sycl {
  class SYCLDialect;
}

#define GEN_PASS_CLASSES
#include "mlir/Conversion/SYCLPasses.h.inc"

} // namespace mlir

#endif // SYCL_CONVERSION_PASSDETAIL_H_
