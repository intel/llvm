//===- MemoryAccessAnalysis.cpp - SYCL Memory Access Analysis -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Analysis/MemoryAccessAnalysis.h"

using namespace mlir;
using namespace mlir::sycl;

MemoryAccessMatrix::MemoryAccessMatrix(unsigned nRows, unsigned nColumns)
    : nRows(nRows), nColumns(nColumns), data(nRows * nColumns) {
  data.reserve(nRows * nColumns);
}
