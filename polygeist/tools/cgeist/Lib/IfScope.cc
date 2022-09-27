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
  if (scanner.loops.size() && scanner.loops.back().keepRunning) {
    auto lop = scanner.builder.create<memref::LoadOp>(
        scanner.loc, scanner.loops.back().keepRunning);
    auto ifOp = scanner.builder.create<scf::IfOp>(scanner.loc, lop,
                                                  /*hasElse*/ false);
    prevBlock = scanner.builder.getInsertionBlock();
    prevIterator = scanner.builder.getInsertionPoint();
    ifOp.getThenRegion().back().clear();
    scanner.builder.setInsertionPointToStart(&ifOp.getThenRegion().back());
    auto er = scanner.builder.create<scf::ExecuteRegionOp>(
        scanner.loc, ArrayRef<mlir::Type>());
    scanner.builder.create<scf::YieldOp>(scanner.loc);
    er.getRegion().push_back(new Block());
    scanner.builder.setInsertionPointToStart(&er.getRegion().back());
  }
}

IfScope::~IfScope() {
  if (scanner.loops.size() && scanner.loops.back().keepRunning) {
    scanner.builder.create<scf::YieldOp>(scanner.loc);
    scanner.builder.setInsertionPoint(prevBlock, prevIterator);
  }
}
