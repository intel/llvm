//===- MemoryAccessAnalysis.h - SYCL Memory Access Analysis -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a memory access analysis for the SYCL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SYCL_ANALYSIS_MEMORYACCESSANALYSIS_H
#define MLIR_DIALECT_SYCL_ANALYSIS_MEMORYACCESSANALYSIS_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

namespace mlir {

class AffineLoadOp;
class AffineStoreOp;
class DataFlowSolver;

namespace sycl {

/// Classify array access patterns.
enum MemoryAccessPattern {
  Unkown = 0,

  /// Array accessed contiguously, in increasing offset order:
  ///    |-------------------->|
  ///    |-------------------->|
  Linear = 1 << 0,
  Reverse = 1 << 1, /// Array accessed in decreasing offset order.

  /// Array accessed contiguously, in decreasing offset order:
  ///    |<-------------------|
  ///    |<-------------------|
  ReverseLinear = Reverse | Linear,
  Shifted = 1 << 2, /// Array accessed starting at an offset.

  /// Array accessed contiguously, increasing offset order, starting at an
  /// offset:
  ///    |  ----------------->|
  ///    |  ----------------->|
  LinearShifted = Linear | Shifted,

  /// Array accessed contiguously, decreasing offset order, starting at an
  /// offset:
  ///    |<-----------------  |
  ///    |<-----------------  |
  ReverseLinearShifted = ReverseLinear | Shifted,
  Overlapped = 1 << 3, // Array accessed starting at different row offsets.

  /// Array accessed contiguously, increasing offset order, starting at
  /// different row offsets:
  ///    |  ----------------->|
  ///    |    --------------->|
  LinearOverlapped = Linear | Overlapped,

  /// Array accessed contiguously, decreasing offset order, starting at
  /// different row offsets:
  ///    |<------------------ |
  ///    |<----------------   |
  ReverseLinearOverlapped = ReverseLinear | Overlapped,

  /// Array accessed in strided fashion, increasing offset order:
  ///    |x-->x-->x-->x-->x-->|
  ///    |x-->x-->x-->x-->x-->|
  Strided = 1 << 4,

  /// Array accessed in strided fashion, decreasing offset order:
  ///    |<--x<--x<--x<--x<--x|
  ///    |<--x<--x<--x<--x<--x|
  ReverseStrided = Reverse | Strided,

  /// Array accessed in strided fashion, increasing offset order, starting at
  /// an offset:
  ///    |    x-->x-->x-->x-->|
  ///    |x-->x-->x-->x-->x-->|
  StridedShifted = Strided | Shifted,

  /// Array accessed in strided fashion, decreasing offset order:
  ///    |<--x<--x<--x<--x<--x|
  ///    |<--x<--x<--x<--x<--x|
  ReverseStridedShifted = Reverse | StridedShifted,

  /// Array accessed in strided fashion, increasing offset order, starting at
  /// different row offsets:
  ///    |   x-->x-->x-->x-->x|
  ///    |       x-->x-->x-->x|
  StridedOverlapped = Strided | Overlapped,

  /// Array accessed in strided fashion, decreasing offset order, starting at
  /// different row offsets:
  ///    |<--x<--x<--x        |
  ///    |<--x<--x<--x<--x    |
  ReverseStridedOverlapped = Reverse | StridedOverlapped,

  /// Array accessed in random order.
  Random = 1 << 8
};

/// A matrix describing an array access pattern.
/// The matrix has size [nRows x nColumns] where nRows is equal to the number of
/// array dimensions and nColumns is equal to the number of loops enclosing the
/// array access.
/// Each column represents the memory access pattern of the corresponding loop
/// level (the outermost loop maps to the leftmost column in the matrix).
/// Each row represents the memory access pattern of the corresponding array
/// dimension (the leftmost dimension maps to first row in the matrix).
/// Note: The array access function is expected to be a linear combination of
/// the enclosing loop induction variables.
class MemoryAccessMatrix {
  friend raw_ostream &operator<<(raw_ostream &, const MemoryAccessMatrix &);

public:
  MemoryAccessMatrix() = delete;
  MemoryAccessMatrix(MemoryAccessMatrix &&) = default;

  /// Construct a matrix with the specified number of rows and columns.
  MemoryAccessMatrix(unsigned nRows, unsigned nColumns);

  /// Access the element at the specified \p row and \p column.
  Value &at(unsigned row, unsigned column) {
    assert(row < nRows && "Row outside of range");
    assert(column < nColumns && "Column outside of range");
    return data[row * nColumns + column];
  }

  Value at(unsigned row, unsigned column) const {
    assert(row < nRows && "Row outside of range");
    assert(column < nColumns && "Column outside of range");
    return data[row * nColumns + column];
  }

  Value &operator()(unsigned row, unsigned column) { return at(row, column); }
  Value operator()(unsigned row, unsigned column) const {
    return at(row, column);
  }

  /// Swap \p column with \p otherColumn.
  void swapColumns(unsigned column, unsigned otherColumn);

  /// Swap \p row with \p otherRow.
  void swapRows(unsigned row, unsigned otherRow);

  unsigned getNumRows() const { return nRows; }

  unsigned getNumColumns() const { return nColumns; }

  /// Get a copy of the specified \p row.
  SmallVector<Value> getRow(unsigned row) const;

  /// Set the specified \p row to \p elems.
  void setRow(unsigned row, ArrayRef<Value> elems);

  /// Fill \p row with the given \p value.
  void fillRow(unsigned row, Value value);

  /// Add an extra row at the bottom of the matrix.
  unsigned appendRow();

  /// Add an extra row at the bottom of the matrix and copy the given elements
  /// \p elems into the new row.
  unsigned appendRow(ArrayRef<Value> elems);

  /// Get a copy of the specified \p column.
  SmallVector<Value> getColumn(unsigned column) const;

  /// Set the specified \p column to \p elems.
  void setColumn(unsigned col, ArrayRef<Value> elems);

  /// Construct a new matrix containing the specified \p rows.
  MemoryAccessMatrix getRows(std::set<unsigned> rows) const;

  /// Construct a new matrix containing the specified \p columns.
  MemoryAccessMatrix getColumns(std::set<unsigned> columns) const;

  /// Construct a new matrix containing the sub-matrix specified by \p rows and
  /// \p columns.
  MemoryAccessMatrix getSubMatrix(std::set<unsigned> rows,
                                  std::set<unsigned> columns) const;

  //===----------------------------------------------------------------------===//
  // Queries
  //===----------------------------------------------------------------------===//

  /// Returns true if the matrix has equal number of rows and columns.
  bool isSquare() const;

  /// Returns true if the only non-zero entries are on the diagonal.
  bool isDiagonal(DataFlowSolver &solver) const;

  /// Returns true if all non-zero entries are below the diagonal.
  bool isLowerTriangular(DataFlowSolver &solver) const;

  /// Returns true if this is a lower triangular matrix with all non-zero
  /// entries having unit value.
  bool isLowerTriangularUnit(DataFlowSolver &solver) const;

  /// Returns true if all non-zero entries are above the diagonal.
  bool isUpperTriangular(DataFlowSolver &solver) const;

  /// Returns true if this is an upper triangular matrix with all non-zero
  /// entries having unit value.
  bool isUpperTriangularUnit(DataFlowSolver &solver) const;

  /// Returns true if the matrix is the unit matrix.
  bool isIdentity(DataFlowSolver &solver) const;

  /// Returns true if the matrix is diagonal with unit values up to the
  /// specified \p row.
  bool isIdentityUpTo(unsigned row, DataFlowSolver &solver) const;

  /// Returns true if the matrix is the filled with zero values.
  bool isZero(DataFlowSolver &solver) const;

private:
  /// Determine whether the value at \p row and \p column is a constant integer
  /// value.
  Optional<APInt> getConstIntegerValue(unsigned row, unsigned column,
                                       DataFlowSolver &solver) const;

private:
  unsigned nRows, nColumns;
  SmallVector<Value> data;
};

inline raw_ostream &operator<<(raw_ostream &os,
                               const MemoryAccessMatrix &matrix) {
  for (unsigned row = 0; row < matrix.getNumRows(); ++row) {
    llvm::interleave(
        matrix.getRow(row), os, [&os](Value elem) { os << elem; }, " ");
    os << '\n';
  }
  return os;
}

/// A column vector representing offsets used to access an array.
/// The size is equal to the number of array dimensions. The first vector
/// element corresponds to the leftmost array dimension.
class OffsetVector {
  friend raw_ostream &operator<<(raw_ostream &, const OffsetVector &);

public:
  OffsetVector() = delete;
  OffsetVector(OffsetVector &&) = default;

  /// Construct an offset vector with the specified number of rows.
  OffsetVector(unsigned nRows);

  /// Access the offset at the specified \p row.
  Value &at(unsigned row) {
    assert(row < nRows && "Row outside of range");
    return offsets[row];
  }

  Value at(unsigned row) const {
    assert(row < nRows && "Row outside of range");
    return offsets[row];
  }

  Value &operator()(unsigned row) { return at(row); }
  Value operator()(unsigned row) const { return at(row); }

  /// Swap \p row with \p otherRow.
  void swapRows(unsigned row, unsigned otherRow);

  unsigned getNumRows() const { return nRows; }

  ArrayRef<Value> getOffsets() const { return offsets; }

  /// Get the offset value at the specified \p row.
  Value getOffset(unsigned row) const;

  /// Set the specified \p row to the given \p offset value.
  void setOffset(unsigned row, Value offset);

  /// Fill the offset vector with the given \p value.
  void fill(Value value);

  /// Add an extra element at the bottom of the offset vector and set it to the
  /// given \p offset value.
  unsigned append(Value offset);

  //===----------------------------------------------------------------------===//
  // Queries
  //===----------------------------------------------------------------------===//

  /// Returns true if the vector contains all zeros.
  bool isZero(DataFlowSolver &solver) const;

  /// Returns true if the matrix is the filled with zero up to \p row.
  bool isZeroUpTo(unsigned row, DataFlowSolver &solver) const;

private:
  /// Determine whether the value at \p row is a constant integer value.
  Optional<APInt> getConstIntegerValue(unsigned row,
                                       DataFlowSolver &solver) const;

private:
  unsigned nRows;
  SmallVector<Value> offsets;
};

inline raw_ostream &operator<<(raw_ostream &os, const OffsetVector &vector) {
  llvm::interleave(
      vector.getOffsets(), os, [&os](Value elem) { os << elem; }, " ");
  os << "\n";
  return os;
}

/// Describes an array access via an access matrix and an offset vector.
/// For example consider the following access in a loop nest:
///   for (i)
///     for (j)
///       ...= A[c1*i+c2][c3*j+c4];
///
/// The access is described by:
///   |c1   0|   |i|   |c2|
///   |      | * | | + |  |
///   | 0  c3|   |j|   |c4|
///
template <typename OpTy> class MemoryAccess {
  friend raw_ostream &operator<<(raw_ostream &, const MemoryAccess &);

public:
  MemoryAccess() = delete;

  MemoryAccess(const OpTy &accessOp, MemoryAccessMatrix &&matrix,
               OffsetVector &&offsets);

  const OpTy &getAccessOp() const { return accessOp; }

  const MemoryAccessMatrix &getAccessMatrix() const { return matrix; }

  const OffsetVector &getOffsetVector() const { return offsets; }

  /// Analyze the array access and return its classification.
  MemoryAccessPattern classify() const;

private:
  const OpTy &accessOp;      /// The array load or store operation.
  MemoryAccessMatrix matrix; /// The memory access matrix.
  OffsetVector offsets;      /// The offset vector.
};

template <typename OpTy>
inline raw_ostream &operator<<(raw_ostream &os,
                               const MemoryAccess<OpTy> &access) {
  os << "--- MemoryAccess ---\n\n";
  os << "AccessMatrix:\n" << access.getAccessMatrix() << "\n";
  os << "OffsetVector:\n" << access.getOffsetVector() << "\n";
  os << "\n------------------\n";
  return os;
}

} // namespace sycl
} // namespace mlir

#endif // MLIR_DIALECT_SYCL_ANALYSIS_MEMORYACCESSANALYSIS_H