//===- MemoryAccessAnalysis.cpp - SYCL Memory Access Analysis -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "mlir/Dialect/SYCL/Analysis/MemoryAccessAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "memory-access-analysis"

using namespace mlir;
using namespace mlir::sycl;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MemoryAccessMatrix
//===----------------------------------------------------------------------===//

MemoryAccessMatrix::MemoryAccessMatrix(unsigned nRows, unsigned nColumns)
    : nRows(nRows), nColumns(nColumns), data(nRows * nColumns) {
  data.reserve(nRows * nColumns);
}

void MemoryAccessMatrix::swapRows(unsigned row, unsigned otherRow) {
  assert((row < nRows && otherRow < nRows) && "out of bounds");
  if (row == otherRow)
    return;
  for (unsigned col = 0; col < nColumns; ++col)
    std::swap(at(row, col), at(otherRow, col));
}

void MemoryAccessMatrix::swapColumns(unsigned column, unsigned otherColumn) {
  assert((column < nColumns && otherColumn < nColumns) && "out of bounds");
  if (column == otherColumn)
    return;
  for (unsigned row = 0; row < nRows; ++row)
    std::swap(at(row, column), at(row, otherColumn));
}

SmallVector<Value> MemoryAccessMatrix::getRow(unsigned row) const {
  assert(row < nRows && "the matrix must contain the given row");
  SmallVector<Value> rowCopy;
  rowCopy.reserve(nColumns);
  for (unsigned col = 0; col < nColumns; ++col)
    rowCopy.emplace_back(at(row, col));
  return rowCopy;
}

void MemoryAccessMatrix::setRow(unsigned row, ArrayRef<Value> elems) {
  assert(row < nRows && "the matrix must contain the given row");
  assert(elems.size() == nColumns && "elems size must match row length!");
  for (unsigned col = 0; col < nColumns; ++col)
    at(row, col) = elems[col];
}

void MemoryAccessMatrix::fillRow(unsigned row, Value value) {
  assert(row < nRows && "the matrix must contain the given row");
  for (unsigned col = 0; col < nColumns; ++col)
    at(row, col) = value;
}

unsigned MemoryAccessMatrix::appendRow() {
  ++nRows;
  data.resize(nRows * nColumns);
  return nRows - 1;
}

unsigned MemoryAccessMatrix::appendRow(ArrayRef<Value> elems) {
  unsigned row = appendRow();
  setRow(row, elems);
  return row;
}

SmallVector<Value> MemoryAccessMatrix::getColumn(unsigned column) const {
  assert(column < nColumns && "the matrix must contain the given column");
  SmallVector<Value> columnCopy;
  columnCopy.reserve(nRows);
  for (unsigned row = 0; row < nRows; ++row)
    columnCopy.emplace_back(at(row, column));
  return columnCopy;
}

void MemoryAccessMatrix::setColumn(unsigned col, ArrayRef<Value> elems) {
  assert(col < nColumns && "the matrix must contain the given column");
  assert(elems.size() == nRows && "elems size must match column length!");
  for (unsigned row = 0; row < nRows; ++row)
    at(row, col) = elems[row];
}

MemoryAccessMatrix MemoryAccessMatrix::getRows(std::set<unsigned> rows) const {
  assert(llvm::all_of(rows, [this](unsigned row) { return row < nRows; }) &&
         "out of bounds");
  MemoryAccessMatrix subMatrix(rows.size(), nColumns);
  unsigned pos = 0;
  for (unsigned rowNum : rows)
    subMatrix.setRow(pos++, getRow(rowNum));
  return subMatrix;
}

MemoryAccessMatrix
MemoryAccessMatrix::getColumns(std::set<unsigned> columns) const {
  assert(llvm::all_of(columns,
                      [this](unsigned column) { return column < nColumns; }) &&
         "out of bounds");
  MemoryAccessMatrix subMatrix(nRows, columns.size());
  unsigned pos = 0;
  for (unsigned colNum : columns)
    subMatrix.setColumn(pos++, getRow(colNum));
  return subMatrix;
}

MemoryAccessMatrix
MemoryAccessMatrix::getSubMatrix(std::set<unsigned> rows,
                                 std::set<unsigned> columns) const {
  return getRows(std::move(rows)).getColumns(std::move(columns));
}

//===----------------------------------------------------------------------===//
// OffsetVector
//===----------------------------------------------------------------------===//

OffsetVector::OffsetVector(unsigned nRows) : nRows(nRows) {
  offsets.reserve(nRows);
}

void OffsetVector::swapRows(unsigned row, unsigned otherRow) {
  assert((row < nRows && otherRow < nRows) && "out of bounds");
  if (row == otherRow)
    return;
  std::swap(at(row), at(otherRow));
}

Value OffsetVector::getOffset(unsigned row) const { return at(row); }

void OffsetVector::setOffset(unsigned row, Value offset) { at(row) = offset; }

void OffsetVector::fill(Value value) {
  for (unsigned row = 0; row < nRows; ++row)
    at(row) = value;
}

unsigned OffsetVector::append(Value offset) {
  ++nRows;
  offsets.resize(nRows);
  unsigned lastRow = nRows - 1;
  setOffset(lastRow, offset);
  return lastRow;
}

//===----------------------------------------------------------------------===//
// MemoryAccess
//===----------------------------------------------------------------------===//

template <typename OpTy>
MemoryAccess<OpTy>::MemoryAccess(const OpTy &accessOp,
                                 MemoryAccessMatrix &&matrix,
                                 OffsetVector &&offsets)
    : accessOp(accessOp), matrix(std::move(matrix)),
      offsets(std::move(offsets)) {
  static_assert(std::is_same<AffineLoadOp, OpTy>::value ||
                    std::is_same<AffineStoreOp, OpTy>::value,
                "Expecting an affine load or store operation");
}

template <typename OpTy>
MemoryAccessPattern MemoryAccess<OpTy>::classify() const {
  // A linear access has an identity access matrix.

  return MemoryAccessPattern::Unkown;
}
