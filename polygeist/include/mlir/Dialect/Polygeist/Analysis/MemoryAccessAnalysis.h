//===- MemoryAccessAnalysis.h - Memory Access Analysis ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an analysis that attempts to classify affine memory
// accesses.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_POLYGEIST_ANALYSIS_MEMORYACCESSANALYSIS_H
#define MLIR_DIALECT_POLYGEIST_ANALYSIS_MEMORYACCESSANALYSIS_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

namespace mlir {

namespace affine {
struct MemRefAccess;
} // namespace affine

namespace sycl {
class SYCLAccessorSubscriptOp;
} // namespace sycl

class DataFlowSolver;
class FunctionOpInterface;

namespace polygeist {

class Definition;
class ReachingDefinition;

/// Classify array access patterns.
enum MemoryAccessPattern : uint32_t {
  Unknown = 0,

  /// Array accessed contiguously, in increasing offset order:
  ///    |-------------------->|
  ///    |-------------------->|
  Linear = 1 << 0,

  /// Array accessed contiguously, in decreasing offset order:
  ///    |<-------------------|
  ///    |<-------------------|
  Reverse = 1 << 1, /// Array accessed in decreasing offset order.
  ReverseLinear = Reverse | Linear,

  /// Array accessed contiguously, increasing offset order, starting at an
  /// offset:
  ///    | |----------------->|
  ///    | |----------------->|
  Shifted = 1 << 2, /// Array accessed starting at an offset.
  LinearShifted = Linear | Shifted,

  /// Array accessed contiguously, decreasing offset order, starting at an
  /// offset:
  ///    |<-----------------| |
  ///    |<-----------------| |
  ReverseLinearShifted = ReverseLinear | Shifted,

  /// Array accessed contiguously, increasing offset order, starting at
  /// different row offsets:
  ///    |------------------->|
  ///    | |----------------->|
  Overlapped = 1 << 3, // Array accessed starting at different row offsets.
  LinearOverlapped = Linear | Overlapped,

  /// Array accessed contiguously, decreasing offset order, starting at
  /// different row offsets:
  ///    |<-------------------|
  ///    |<-----------------| |
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
  ///    |  |x-->x-->x-->x-->x|
  ///    |  |x-->x-->x-->x-->x|
  StridedShifted = Strided | Shifted,

  /// Array accessed in strided fashion, decreasing offset order:
  ///    |<--x<--x<--x<--x|  |
  ///    |<--x<--x<--x<--x|  |
  ReverseStridedShifted = Reverse | StridedShifted,

  /// Array accessed in strided fashion, increasing offset order, starting at
  /// different row offsets:
  ///    |x-->x-->x-->x-->x  |
  ///    | |x-->x-->x-->x-->x|
  StridedOverlapped = Strided | Overlapped,

  /// Array accessed in strided fashion, decreasing offset order, starting at
  /// different row offsets:
  ///    |x<--x<--x<--x<--x| |
  ///    |  x<--x<--x<--x<--x|
  ReverseStridedOverlapped = Reverse | StridedOverlapped,
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
  MemoryAccessMatrix() = default;
  MemoryAccessMatrix(MemoryAccessMatrix &&) = default;
  MemoryAccessMatrix(const MemoryAccessMatrix &) = default;
  MemoryAccessMatrix &operator=(MemoryAccessMatrix &&) = default;
  MemoryAccessMatrix &operator=(const MemoryAccessMatrix &) = default;

  /// Construct a matrix with the specified number of rows and columns.
  MemoryAccessMatrix(size_t nRows, size_t nColumns);

  /// Construct a matrix from an initializer list.
  MemoryAccessMatrix(
      std::initializer_list<std::initializer_list<Value>> initList);

  /// Access the element at the specified \p row and \p column.
  Value &at(size_t row, size_t column) {
    assert(row < nRows && "Row outside of range");
    assert(column < nColumns && "Column outside of range");
    return data[row * nColumns + column];
  }

  Value at(size_t row, size_t column) const {
    assert(row < nRows && "Row outside of range");
    assert(column < nColumns && "Column outside of range");
    return data[row * nColumns + column];
  }

  Value &operator()(size_t row, size_t column) { return at(row, column); }
  Value operator()(size_t row, size_t column) const { return at(row, column); }

  /// Swap \p column with \p otherColumn.
  void swapColumns(size_t column, size_t otherColumn);

  /// Swap \p row with \p otherRow.
  void swapRows(size_t row, size_t otherRow);

  size_t getNumRows() const { return nRows; }

  size_t getNumColumns() const { return nColumns; }

  /// Get a copy of the specified \p row.
  SmallVector<Value> getRow(size_t row) const;

  /// Set the specified \p row to \p elems.
  void setRow(size_t row, ArrayRef<Value> elems);

  /// Fill \p row with the given value \p val.
  void fillRow(size_t row, Value val);

  /// Add an extra row at the bottom of the matrix and return the row index of
  /// the new row. The new row is uninitialized.
  size_t appendRow();

  /// Add an extra row at the bottom of the matrix and copy the given elements
  /// \p elems into the new row. Return the row index of the new row.
  size_t appendRow(ArrayRef<Value> elems);

  /// Get a copy of the specified \p column.
  SmallVector<Value> getColumn(size_t column) const;

  /// Set the specified \p column to \p elems.
  void setColumn(size_t col, ArrayRef<Value> elems);

  /// Fill \p col with the given value \p val.
  void fillColumn(size_t col, Value val);

  /// Fill the matrix with the given value \p val.
  void fill(Value val);

  /// Construct a new matrix containing the specified \p rows.
  /// Note: because \p rows is an ordered set, asking for rows {1,0} causes
  /// a new access matrix with rows zero and one (in that order) to be returned.
  MemoryAccessMatrix getRows(std::set<size_t> rows) const;

  /// Construct a new matrix containing the specified \p columns.
  /// Note: because \p columns is an ordered set, asking for columns {1,0}
  /// causes a new access matrix with columns zero and one (in that order) to be
  /// returned.
  MemoryAccessMatrix getColumns(std::set<size_t> columns) const;

  /// Construct a new matrix containing the sub-matrix specified by \p rows and
  /// \p columns.
  /// Note: because \p rows and \p columns are ordered sets, the sub matrix
  /// returned contains a sub-view or the original matrix (row, columns selected
  /// are in the same order as in the original matrix).
  MemoryAccessMatrix getSubMatrix(std::set<size_t> rows,
                                  std::set<size_t> columns) const;

  //===----------------------------------------------------------------------===//
  // Shape Queries
  //===----------------------------------------------------------------------===//

  /// Return true if the matrix has equal number of rows and columns.
  bool isSquare() const;

  /// Return true if the matrix is the filled with zero values.
  bool isZero(DataFlowSolver &solver) const;

  /// Return true if the only non-zero entries are on the diagonal.
  bool isDiagonal(DataFlowSolver &solver) const;

  /// Return true if the matrix is the unit matrix.
  bool isIdentity(DataFlowSolver &solver) const;

  /// Return true if all zero entries are above the diagonal.
  bool isLowerTriangular(DataFlowSolver &solver) const;

  /// Return true if all zero entries are below the diagonal.
  bool isUpperTriangular(DataFlowSolver &solver) const;

  //===----------------------------------------------------------------------===//
  // Access Pattern Queries
  //===----------------------------------------------------------------------===//

  /// Array accessed contiguously, increasing offset order. The matrix shape is:
  ///   |1  0|
  ///   |0  1|
  bool hasLinearAccessPattern(DataFlowSolver &solver) const;

  /// Array accessed contiguously, decreasing offset order. The matrix shape is:
  ///   |1  0|
  ///   |0 -1|
  bool hasReverseLinearAccessPattern(DataFlowSolver &solver) const;

  /// Array accessed contiguously, increasing offset order, starting at
  /// different row offsets. The matrix shape is:
  ///   |1  0|
  ///   |1  1|
  bool hasLinearOverlappedAccessPattern(DataFlowSolver &solver) const;

  /// Array accessed contiguously, decreasing  offset order, starting at
  /// different row offsets. The matrix shape is:
  ///   |1  0|
  ///   |1 -1|
  bool hasReverseLinearOverlappedAccessPattern(DataFlowSolver &solver) const;

  /// Array accessed in strided fashion, increasing offset order. The matrix
  /// shape is:
  ///   |1  0|
  ///   |0  C|  where C > 1
  bool hasStridedAccessPattern(DataFlowSolver &solver) const;

  /// Array accessed in strided fashion, decreasing offset order. The matrix
  /// shape is:
  ///   |1  0|
  ///   |0 -C|  where -C < -1
  bool hasReverseStridedAccessPattern(DataFlowSolver &solver) const;

  /// Array accessed in strided fashion, increasing offset order, starting at
  /// different row offsets. The matrix shape is:
  ///   |1  0|
  ///   |1  C|  where C > 1
  bool hasStridedOverlappedAccessPattern(DataFlowSolver &solver) const;

  /// Array accessed in strided fashion, decreasing offset order, starting at
  /// different row offsets. The matrix shape is:
  ///   |1  0|
  ///   |1 -C|  where -C < -1
  bool hasReverseStridedOverlappedAccessPattern(DataFlowSolver &solver) const;

private:
  /// Return the value at row \p row and column \p column if it is an integer
  /// constant and std::nullopt otherwise.
  std::optional<APInt> getConstIntegerValue(size_t row, size_t column,
                                            DataFlowSolver &solver) const;

private:
  size_t nRows, nColumns;
  SmallVector<Value> data;
};

/// A column vector representing offsets used to access an array.
/// The size is equal to the number of array dimensions. The first vector
/// element corresponds to the leftmost array dimension.
class OffsetVector {
  friend raw_ostream &operator<<(raw_ostream &, const OffsetVector &);

public:
  OffsetVector() = default;
  OffsetVector(OffsetVector &&) = default;
  OffsetVector(const OffsetVector &) = default;
  OffsetVector &operator=(OffsetVector &&) = default;
  OffsetVector &operator=(const OffsetVector &) = default;

  /// Construct an offset vector with the specified number of rows.
  explicit OffsetVector(size_t nRows);

  /// Construct an offset vector from an initialization list.
  OffsetVector(std::initializer_list<Value> initList);

  /// Access the offset at the specified \p row.
  Value &at(size_t row) {
    assert(row < nRows && "Row outside of range");
    return offsets[row];
  }

  Value at(size_t row) const {
    assert(row < nRows && "Row outside of range");
    return offsets[row];
  }

  Value &operator()(size_t row) { return at(row); }
  Value operator()(size_t row) const { return at(row); }

  /// Swap \p row with \p otherRow.
  void swapRows(size_t row, size_t otherRow);

  size_t getNumRows() const { return nRows; }

  ArrayRef<Value> getOffsets() const { return offsets; }

  /// Get the offset value at the specified \p row.
  Value getOffset(size_t row) const;

  /// Set the specified \p row to the given \p offset value.
  void setOffset(size_t row, Value offset);

  /// Fill the offset vector with the given value \p val.
  void fill(Value val);

  /// Add an extra element at the bottom of the offset vector and set it to the
  /// given \p offset value. Return the new element index.
  size_t append(Value offset);

  //===----------------------------------------------------------------------===//
  // Queries
  //===----------------------------------------------------------------------===//

  /// Return true if the vector contains all zeros.
  bool isZero(DataFlowSolver &solver) const;

  /// Return true if the vector contains all zeros and the last element has
  /// a strictly positive constant value.
  bool isZeroWithLastElementStrictlyPositive(DataFlowSolver &solver) const;

  /// Return true if the vector contains all zeros and the last element has
  /// value equal to the given constant \p k.
  bool isZeroWithLastElementEqualTo(int k, DataFlowSolver &solver) const;

private:
  /// Return the element at row \p row if it is a integer constant or
  /// std::nullopt otherwise.
  std::optional<APInt> getConstIntegerValue(size_t row,
                                            DataFlowSolver &solver) const;

private:
  size_t nRows;
  SmallVector<Value> offsets;
};

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
class MemoryAccess {
  friend raw_ostream &operator<<(raw_ostream &, const MemoryAccess &);

public:
  MemoryAccess() = default;
  MemoryAccess(MemoryAccessMatrix &&matrix, OffsetVector &&offsets)
      : matrix(std::move(matrix)), offsets(std::move(offsets)) {
    assert(matrix.getNumRows() == offsets.getNumRows() &&
           "Matrix and offset vector must have thes same number of rows");
  }

  const MemoryAccessMatrix &getAccessMatrix() const { return matrix; }

  const OffsetVector &getOffsetVector() const { return offsets; }

  /// Construct a new matrix containing columns [0...numThreads-1].
  MemoryAccessMatrix getInterThreadAccessMatrix(unsigned numThreads) const;

  /// Construct a new matrix containing columns [numThreads ... numColumns-1].
  MemoryAccessMatrix getIntraThreadAccessMatrix(unsigned numThreads) const;

  /// Analyze the memory access and classify its access pattern.
  MemoryAccessPattern classify(DataFlowSolver &solver) const;

  /// Analyze the given access matrix and offset vector and classify the access
  /// pattern.
  static MemoryAccessPattern classify(const MemoryAccessMatrix &matrix,
                                      const OffsetVector &offsets,
                                      DataFlowSolver &solver);

private:
  MemoryAccessMatrix matrix; /// The memory access matrix.
  OffsetVector offsets;      /// The offset vector.
};

class MemoryAccessAnalysis {
  friend raw_ostream &operator<<(raw_ostream &, const MemoryAccessMatrix &);

public:
  MemoryAccessAnalysis(Operation *op, AnalysisManager &am);

  MemoryAccessAnalysis &initialize(bool relaxedAliasing) {
    this->relaxedAliasing = relaxedAliasing;
    return *this;
  }

  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa);

  /// Return the operation this analysis was constructed from.
  Operation *getOperation() const { return operation; }

  /// Return the memory access for the given memref \p access.
  std::optional<MemoryAccess>
  getMemoryAccess(const affine::MemRefAccess &access) const;

  /// Return a vector containing the threads used in \p funcOp.
  SmallVector<Value> computeThreadVector(FunctionOpInterface funcOp,
                                         DataFlowSolver &solver) const;

private:
  /// Construct the access matrix and offset vector for the memory accesses
  /// contained in the operation associated with the analysis.
  void build();

  /// Attempt to create an entry in the accessMap for the given memory operation
  /// \p memoryOp.
  template <typename T> void build(T memoryOp, DataFlowSolver &solver);

  /// Construct the access matrix if possible.
  std::optional<MemoryAccessMatrix>
  buildAccessMatrix(sycl::SYCLAccessorSubscriptOp accessorSubscriptOp,
                    const SmallVectorImpl<Value> &loopAndThreadVars,
                    const SmallVectorImpl<Value> &underlyingVals,
                    DataFlowSolver &solver);

  /// Construct the offset vector if possible.
  std::optional<OffsetVector>
  buildOffsetVector(const MemoryAccessMatrix &matrix,
                    const SmallVectorImpl<Value> &loopAndThreadVars,
                    const SmallVectorImpl<Value> &underlyingVals,
                    DataFlowSolver &solver);

  /// Returns true if the memory access \p access has a single subscript that is
  /// zero, and false otherwise.
  bool hasZeroIndex(const affine::MemRefAccess &access) const;

  /// Returns the unique definition for the operand at index \p opIndex in
  /// operation \p op, or std::nullopt if it does not have a unique definition.
  std::optional<Definition> getUniqueDefinition(unsigned opIndex, Operation *op,
                                                DataFlowSolver &solver) const;

  /// Collect the underlying value(s) of the operand at index \p opIndex in
  /// operation \p op.
  /// For example given:
  ///
  ///   sycl.constructor @id(%id, %i, %j) : (memref<?x!sycl_id_2>, i64, i64)
  ///   %subscr = sycl.accessor.subscript %acc[%id] ...
  ///
  /// The underlying values for '%id' are {%i, %j}.
  SmallVector<Value> getUnderlyingValues(unsigned opIndex, Operation *op,
                                         DataFlowSolver &solver) const;

private:
  /// The operation associated with the analysis.
  Operation *operation;

  /// A map from memory accesses to their memory access matrix and offset
  /// vector.
  DenseMap<Operation *, MemoryAccess> accessMap;

  /// The analysis manager.
  AnalysisManager &am;

  /// Whether to assume the program abides to strict aliasing rules (i.e type
  /// based aliasing) or not.
  bool relaxedAliasing = false;
};

} // namespace polygeist
} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_ANALYSIS_MEMORYACCESSANALYSIS_H
