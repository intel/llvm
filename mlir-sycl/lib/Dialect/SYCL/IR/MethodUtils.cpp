//===--- MethodUtils.cpp --------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/MethodUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Polygeist/IR/Ops.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <iterator>

using namespace mlir;
using namespace mlir::sycl;

/// Returns the SYCL cast originating this value if such operation exists; None
/// otherwise.
///
/// This function relies on how arguments are casted to perform a function call.
/// Should be updated if this changes.
static Operation *trackCasts(Value Val) {
  auto *const DefiningOp = Val.getDefiningOp();
  if (!DefiningOp)
    return nullptr;

  return TypeSwitch<Operation *, Operation *>(DefiningOp)
      .Case<mlir::sycl::SYCLCastOp, memref::MemorySpaceCastOp>(
          [](Operation *Op) {
            if (auto *Res = trackCasts(Op->getOperand(0)))
              return Res;
            return Op;
          })
      .Default(static_cast<Operation *>(nullptr));
}

Value mlir::sycl::abstractCasts(Value Original) {
  Operation *Cast = trackCasts(Original);
  return Cast ? Cast->getOperand(0) : Original;
}
