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
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ValueRange.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <iterator>

using namespace mlir;
using namespace mlir::sycl;

namespace {
/// Returns the SYCL cast originating this value if such operation exists; None
/// otherwise.
///
/// This function relies on how arguments are casted to perform a function call.
/// Should be updated if this changes.
Operation *trackCasts(Value Val) {
  auto *const DefiningOp = Val.getDefiningOp();
  if (!DefiningOp)
    return nullptr;

  return TypeSwitch<Operation *, Operation *>(DefiningOp)
      .Case<mlir::sycl::SYCLCastOp, mlir::polygeist::Memref2PointerOp>(
          [](Operation *Op) {
            if (auto *Res = trackCasts(Op->getOperand(0)))
              return Res;
            return Op;
          })
      .Case<mlir::polygeist::Pointer2MemrefOp, mlir::LLVM::AddrSpaceCastOp>(
          [](Operation *Op) { return trackCasts(Op->getOperand(0)); })
      .Default(static_cast<Operation *>(nullptr));
}
} // namespace

SmallVector<Value> mlir::sycl::adaptSYCLMethodOpArguments(OpBuilder &Builder,
                                                          Location Loc,
                                                          ValueRange Original) {
  SmallVector<Value> Transformed;
  Transformed.reserve(Original.size());
  std::transform(
      Original.begin(), Original.end(), std::back_inserter(Transformed),
      [&](auto Val) {
        return TypeSwitch<Type, Value>(Val.getType())
            .Case<MemRefType>([&](auto MT) -> Value {
              return Builder.createOrFold<memref::LoadOp>(
                  Loc, Val,
                  Builder.createOrFold<arith::ConstantIndexOp>(Loc, 0));
            })
            .Default([Val](Type) { return Val; });
      });
  return Transformed;
}

Value mlir::sycl::abstractCasts(Value Original) {
  Operation *Cast = trackCasts(Original);
  return Cast ? Cast->getOperand(0) : Original;
}
