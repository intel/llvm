//===- PolygeistToLLVMPass.h - Polygeist to LLVM Conversion Pass ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_POLYGEISTTOLLVM_POLYGEISTTOLLVM_H
#define MLIR_CONVERSION_POLYGEISTTOLLVM_POLYGEISTTOLLVM_H

#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/SYCL/IR/SYCLAttributes.h"

#include <memory>

namespace mlir {
class LowerToLLVMOptions;

#define GEN_PASS_DECL_CONVERTPOLYGEISTTOLLVM
#include "mlir/Conversion/PolygeistPasses.h.inc"
#undef GEN_PASS_DECL_CONVERTPOLYGEISTTOLLVM

} // namespace mlir

#endif // MLIR_CONVERSION_POLYGEISTTOLLVM_POLYGEISTTOLLVM_H
