//===- PolygeistPasses.h - Conversion Pass Construction and Registration --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_POLYGEISTPASSES_H
#define MLIR_CONVERSION_POLYGEISTPASSES_H

#include "mlir/Conversion/PolygeistToLLVM/PolygeistToLLVM.h"

namespace mlir {
namespace polygeist {
/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Conversion/PolygeistPasses.h.inc"
} // namespace polygeist
} // namespace mlir

#endif // MLIR_CONVERSION_POLYGEISTPASSES_H
