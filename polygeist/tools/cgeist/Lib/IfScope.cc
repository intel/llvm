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
  if (Scanner.loops.size() && Scanner.loops.back().keepRunning) {
    auto Lop = Scanner.builder.create<memref::LoadOp>(
        Scanner.loc, Scanner.loops.back().keepRunning);
    auto IfOp = Scanner.builder.create<scf::IfOp>(Scanner.loc, Lop,
                                                  /*hasElse*/ false);
    PrevBlock = Scanner.builder.getInsertionBlock();
    PrevIterator = Scanner.builder.getInsertionPoint();
    IfOp.getThenRegion().back().clear();
    Scanner.builder.setInsertionPointToStart(&IfOp.getThenRegion().back());
    auto Er = Scanner.builder.create<scf::ExecuteRegionOp>(
        Scanner.loc, ArrayRef<mlir::Type>());
    Scanner.builder.create<scf::YieldOp>(Scanner.loc);
    Er.getRegion().push_back(new Block());
    Scanner.builder.setInsertionPointToStart(&Er.getRegion().back());
  }
}

IfScope::~IfScope() {
  if (Scanner.loops.size() && Scanner.loops.back().keepRunning) {
    Scanner.builder.create<scf::YieldOp>(Scanner.loc);
    Scanner.builder.setInsertionPoint(PrevBlock, PrevIterator);
  }
}
