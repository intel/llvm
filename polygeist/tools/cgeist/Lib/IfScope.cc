//===---- IfScope.cc - Create an if statement to guard loop boundaries ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IfScope.h"
#include "clang-mlir.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

IfScope::IfScope(MLIRScanner &Scanner) : Scanner(Scanner), PrevBlock(nullptr) {
  if (Scanner.Loops.size() && Scanner.Loops.back().KeepRunning) {
    auto Lop = Scanner.Builder.create<memref::LoadOp>(
        Scanner.Loc, Scanner.Loops.back().KeepRunning);
    auto IfOp = Scanner.Builder.create<scf::IfOp>(Scanner.Loc, Lop,
                                                  /*hasElse*/ false);
    PrevBlock = Scanner.Builder.getInsertionBlock();
    PrevIterator = Scanner.Builder.getInsertionPoint();
    IfOp.getThenRegion().back().clear();
    Scanner.Builder.setInsertionPointToStart(&IfOp.getThenRegion().back());
    auto Er = Scanner.Builder.create<scf::ExecuteRegionOp>(
        Scanner.Loc, ArrayRef<mlir::Type>());
    Scanner.Builder.create<scf::YieldOp>(Scanner.Loc);
    Er.getRegion().push_back(new Block());
    Scanner.Builder.setInsertionPointToStart(&Er.getRegion().back());
  }
}

IfScope::~IfScope() {
  if (Scanner.Loops.size() && Scanner.Loops.back().KeepRunning) {
    Scanner.Builder.create<scf::YieldOp>(Scanner.Loc);
    Scanner.Builder.setInsertionPoint(PrevBlock, PrevIterator);
  }
}
