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

IfScope::IfScope(MLIRScanner &scanner) : scanner(scanner), prevBlock(nullptr) {
  mlir::OpBuilder builder = scanner.getBuilder();
  std::vector<LoopContext> &loops = scanner.getLoops();
  Location &loc = scanner.getLoc();

  if (loops.size() && loops.back().keepRunning) {
    auto lop = builder.create<memref::LoadOp>(loc, loops.back().keepRunning);
    auto ifOp = builder.create<scf::IfOp>(loc, lop,
                                          /*hasElse*/ false);
    prevBlock = builder.getInsertionBlock();
    prevIterator = builder.getInsertionPoint();
    ifOp.getThenRegion().back().clear();
    builder.setInsertionPointToStart(&ifOp.getThenRegion().back());
    auto er = builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
    builder.create<scf::YieldOp>(loc);
    er.getRegion().push_back(new Block());
    builder.setInsertionPointToStart(&er.getRegion().back());
  }
}

IfScope::~IfScope() {
  mlir::OpBuilder builder = scanner.getBuilder();
  std::vector<LoopContext> &loops = scanner.getLoops();
  Location &loc = scanner.getLoc();

  if (loops.size() && loops.back().keepRunning) {
    builder.create<scf::YieldOp>(loc);
    builder.setInsertionPoint(prevBlock, prevIterator);
  }
}
