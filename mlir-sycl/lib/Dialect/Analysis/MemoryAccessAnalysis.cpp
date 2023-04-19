//===- MemoryAccessAnalysis.cpp - SYCL Memory Access Analysis -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Analysis/MemoryAccessAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <utility>

#define DEBUG_TYPE "memory-access-analysis"

using namespace mlir;
using namespace mlir::sycl;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Determine whether a value is an known integer value.
static Optional<APInt> getConstIntegerValue(Value val, DataFlowSolver &solver) {
  if (!isa<IntegerType>(val.getType()))
    return std::nullopt;

  auto *inferredRange =
      solver.lookupState<dataflow::IntegerValueRangeLattice>(val);
  if (!inferredRange || inferredRange->getValue().isUninitialized())
    return std::nullopt;

  const ConstantIntRanges &range = inferredRange->getValue().getValue();
  return range.getConstantValue();
}

/// Determine whether a value is equal to a given integer constant.
static bool isEqualTo(Value val, int constant, DataFlowSolver &solver) {
  Optional<APInt> constValue = getConstIntegerValue(val, solver);
  return (constValue.has_value() && *constValue == constant);
}

/// Determine whether a value is zero.
bool isZero(Value val, DataFlowSolver &solver) {
  return isEqualTo(val, 0, solver);
}

/// Determine whether a value is one.
bool isOne(Value val, DataFlowSolver &solver) {
  return isEqualTo(val, 1, solver);
}

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

bool MemoryAccessMatrix::isSquare() const { return (nRows == nColumns); }

bool MemoryAccessMatrix::isDiagonal(DataFlowSolver &solver) const {
  if (!isSquare())
    return false;

  for (unsigned row = 0; row < nRows; ++row)
    for (unsigned col = 0; col < nColumns; ++col) {
      Value val = at(row, col);
      bool isOnDiagonal = (row == col);

      // All values on the diagonal must be non-zero.
      if (isOnDiagonal && ::isZero(val, solver))
        return false;
      // All values not on the diagonal must be zero.
      if (!isOnDiagonal && !::isZero(val, solver))
        return false;
    }

  return true;
}

bool MemoryAccessMatrix::isLowerTriangular(DataFlowSolver &solver) const {
  if (!isSquare())
    return false;

  for (unsigned row = 0; row < nRows; ++row)
    for (unsigned col = 0; col < nColumns; ++col) {
      Value val = at(row, col);
      bool isAboveDiagonal = (col > row);

      // All values above the diagonal must be zero.
      if (isAboveDiagonal && !::isZero(val, solver))
        return false;

      // All values on or below the diagonal must be non-zero.
      if (!isAboveDiagonal && ::isZero(val, solver))
        return false;
    }

  return true;
}

bool MemoryAccessMatrix::isUpperTriangular(DataFlowSolver &solver) const {
  if (!isSquare())
    return false;

  for (unsigned row = 0; row < nRows; ++row)
    for (unsigned col = 0; col < nColumns; ++col) {
      Value val = at(row, col);
      bool isBelowDiagonal = (col < row);

      // All values below the diagonal must be zero.
      if (isBelowDiagonal && !::isZero(val, solver))
        return false;

      // All values on or above the diagonal must be non-zero.
      if (!isBelowDiagonal && ::isZero(val, solver))
        return false;
    }

  return true;
}

bool MemoryAccessMatrix::isIdentity(DataFlowSolver &solver) const {
  if (!isDiagonal(solver))
    return false;

  // Determine whether all values on the diagonal are equal to one.
  for (unsigned pos = 0; pos < nRows; ++pos) {
    Value val = at(pos, pos);
    if (!isOne(val, solver))
      return false;
  }
  return true;
}

bool MemoryAccessMatrix::isZero(DataFlowSolver &solver) const {
  return (llvm::all_of(data,
                       [&solver](Value val) { return ::isZero(val, solver); }));
}

Optional<APInt>
MemoryAccessMatrix::getConstIntegerValue(unsigned row, unsigned column,
                                         DataFlowSolver &solver) const {
  Value val = at(row, column);
  return ::getConstIntegerValue(val, solver);
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

bool OffsetVector::isZero(DataFlowSolver &solver) const {
  return llvm::all_of(offsets,
                      [&](Value offset) { return ::isZero(offset, solver); });
}

Optional<APInt>
OffsetVector::getConstIntegerValue(unsigned row, DataFlowSolver &solver) const {
  Value val = at(row);
  return ::getConstIntegerValue(val, solver);
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
