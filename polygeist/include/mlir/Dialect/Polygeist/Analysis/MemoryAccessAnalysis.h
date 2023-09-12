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

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/Polygeist/Utils/TransformUtils.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
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
  using IntegerValueRange = dataflow::IntegerValueRange;

  MemoryAccessMatrix() = default;
  MemoryAccessMatrix(MemoryAccessMatrix &&) = default;
  MemoryAccessMatrix(const MemoryAccessMatrix &) = default;
  MemoryAccessMatrix &operator=(MemoryAccessMatrix &&) = default;
  MemoryAccessMatrix &operator=(const MemoryAccessMatrix &) = default;

  /// Construct a matrix with the specified number of rows and columns.
  MemoryAccessMatrix(size_t nRows, size_t nColumns);

  /// Construct a matrix from an initializer list.
  MemoryAccessMatrix(
      std::initializer_list<std::initializer_list<IntegerValueRange>>);

  /// Access the element at the specified \p row and \p column.
  IntegerValueRange &at(size_t row, size_t column) {
    assert(row < nRows && "Row outside of range");
    assert(column < nColumns && "Column outside of range");
    return data[row * nColumns + column];
  }

  IntegerValueRange at(size_t row, size_t column) const {
    assert(row < nRows && "Row outside of range");
    assert(column < nColumns && "Column outside of range");
    return data[row * nColumns + column];
  }

  IntegerValueRange &operator()(size_t row, size_t column) {
    return at(row, column);
  }
  IntegerValueRange operator()(size_t row, size_t column) const {
    return at(row, column);
  }

  /// Swap \p column with \p otherColumn.
  void swapColumns(size_t column, size_t otherColumn);

  /// Swap \p row with \p otherRow.
  void swapRows(size_t row, size_t otherRow);

  size_t getNumRows() const { return nRows; }

  size_t getNumColumns() const { return nColumns; }

  /// Get a copy of the specified \p row.
  SmallVector<IntegerValueRange> getRow(size_t row) const;

  /// Set the specified \p row to \p elems.
  void setRow(size_t row, ArrayRef<IntegerValueRange> elems);

  /// Fill \p row with the given \p range.
  void fillRow(size_t row, IntegerValueRange range);

  /// Add an extra row at the bottom of the matrix and return the row index of
  /// the new row. The new row is uninitialized.
  size_t appendRow();

  /// Add an extra row at the bottom of the matrix and copy the given elements
  /// \p elems into the new row. Return the row index of the new row.
  size_t appendRow(ArrayRef<IntegerValueRange> elems);

  /// Get a copy of the specified \p column.
  SmallVector<IntegerValueRange> getColumn(size_t column) const;

  /// Set the specified \p column to \p elems.
  void setColumn(size_t col, ArrayRef<IntegerValueRange> elems);

  /// Fill \p col with the given range \p range.
  void fillColumn(size_t col, IntegerValueRange range);

  /// Fill the matrix with the given range \p range.
  void fill(IntegerValueRange range);

  /// Fill element at position \p row, \p col with zero.
  void setZero(size_t row, size_t col);

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
  bool isZero() const;

  /// Return true if the only non-zero entries are on the diagonal.
  bool isDiagonal() const;

  /// Return true if the matrix is the unit matrix.
  bool isIdentity() const;

  /// Return true if all zero entries are above the diagonal.
  bool isLowerTriangular() const;

  /// Return true if all zero entries are below the diagonal.
  bool isUpperTriangular() const;

  //===----------------------------------------------------------------------===//
  // Access Pattern Queries
  //===----------------------------------------------------------------------===//

  /// Array accessed contiguously, increasing offset order. The matrix shape is:
  ///   |1  0|
  ///   |0  1|
  bool hasLinearAccessPattern() const;

  /// Array accessed contiguously, decreasing offset order. The matrix shape is:
  ///   |1  0|
  ///   |0 -1|
  bool hasReverseLinearAccessPattern() const;

  /// Array accessed contiguously, increasing offset order, starting at
  /// different row offsets. The matrix shape is:
  ///   |1  0|
  ///   |1  1|
  bool hasLinearOverlappedAccessPattern() const;

  /// Array accessed contiguously, decreasing  offset order, starting at
  /// different row offsets. The matrix shape is:
  ///   |1  0|
  ///   |1 -1|
  bool hasReverseLinearOverlappedAccessPattern() const;

  /// Array accessed in strided fashion, increasing offset order. The matrix
  /// shape is:
  ///   |1  0|
  ///   |0  C|  where C > 1
  bool hasStridedAccessPattern() const;

  /// Array accessed in strided fashion, decreasing offset order. The matrix
  /// shape is:
  ///   |1  0|
  ///   |0 -C|  where -C < -1
  bool hasReverseStridedAccessPattern() const;

  /// Array accessed in strided fashion, increasing offset order, starting at
  /// different row offsets. The matrix shape is:
  ///   |1  0|
  ///   |1  C|  where C > 1
  bool hasStridedOverlappedAccessPattern() const;

  /// Array accessed in strided fashion, decreasing offset order, starting at
  /// different row offsets. The matrix shape is:
  ///   |1  0|
  ///   |1 -C|  where -C < -1
  bool hasReverseStridedOverlappedAccessPattern() const;

private:
  size_t nRows, nColumns;
  SmallVector<IntegerValueRange> data;
};

/// A column vector representing offsets used to access an array.
/// The size is equal to the number of array dimensions. The first vector
/// element corresponds to the leftmost array dimension.
class OffsetVector {
  friend raw_ostream &operator<<(raw_ostream &, const OffsetVector &);

public:
  using IntegerValueRange = dataflow::IntegerValueRange;

  OffsetVector() = default;
  OffsetVector(OffsetVector &&) = default;
  OffsetVector(const OffsetVector &) = default;
  OffsetVector &operator=(OffsetVector &&) = default;
  OffsetVector &operator=(const OffsetVector &) = default;

  /// Construct an offset vector with the specified number of rows.
  explicit OffsetVector(size_t nRows);

  /// Construct an offset vector from an initialization list.
  OffsetVector(std::initializer_list<IntegerValueRange> initList);

  /// Access the offset at the specified \p row.
  IntegerValueRange &at(size_t row) {
    assert(row < nRows && "Row outside of range");
    return offsets[row];
  }

  IntegerValueRange at(size_t row) const {
    assert(row < nRows && "Row outside of range");
    return offsets[row];
  }

  IntegerValueRange &operator()(size_t row) { return at(row); }
  IntegerValueRange operator()(size_t row) const { return at(row); }

  /// Swap \p row with \p otherRow.
  void swapRows(size_t row, size_t otherRow);

  size_t getNumRows() const { return nRows; }

  ArrayRef<IntegerValueRange> getOffsets() const { return offsets; }

  /// Get the offset at the specified \p row.
  IntegerValueRange getOffset(size_t row) const;

  /// Set the specified \p row to the given \p offset.
  void setOffset(size_t row, IntegerValueRange offset);

  /// Fill the offset vector with the given \p range.
  void fill(IntegerValueRange range);

  /// Add an extra element at the bottom of the offset vector and set it to the
  /// given \p offset. Return the new element index.
  size_t append(IntegerValueRange offset);

  //===----------------------------------------------------------------------===//
  // Queries
  //===----------------------------------------------------------------------===//

  /// Return true if the vector contains all zeros.
  bool isZero() const;

  /// Return true if the vector contains all zeros and the last element has
  /// a strictly positive constant value.
  bool isZeroWithLastElementStrictlyPositive() const;

  /// Return true if the vector contains all zeros and the last element has
  /// value equal to the given constant \p k.
  bool isZeroWithLastElementEqualTo(int k) const;

private:
  size_t nRows;
  SmallVector<IntegerValueRange> offsets;
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
           "Matrix and offset vector must have the same number of rows");
  }

  const MemoryAccessMatrix &getAccessMatrix() const { return matrix; }

  const OffsetVector &getOffsetVector() const { return offsets; }

  /// Construct a new matrix containing [0...numGridDimensions-1] columns.
  MemoryAccessMatrix
  getInterThreadAccessMatrix(unsigned numGridDimensions) const;

  /// Construct a new matrix containing [numGridDimensions...numColumns-1]
  /// columns.
  MemoryAccessMatrix
  getIntraThreadAccessMatrix(unsigned numGridDimensions) const;

  /// Analyze the memory access and classify its access pattern.
  MemoryAccessPattern classify() const;

  /// Analyze the given access matrix and offset vector and classify the access
  /// pattern.
  static MemoryAccessPattern classify(const MemoryAccessMatrix &matrix,
                                      const OffsetVector &offsets);

private:
  MemoryAccessMatrix matrix; /// The memory access matrix.
  OffsetVector offsets;      /// The offset vector.
};

class MemoryAccessAnalysis {
  friend raw_ostream &operator<<(raw_ostream &, const MemoryAccessMatrix &);

public:
  using IntegerValueRange = dataflow::IntegerValueRange;

  MemoryAccessAnalysis(Operation *op, AnalysisManager &am);

  /// Consumers of the analysis must call this member function immediately after
  /// construction.
  template <typename... Plugins>
  MemoryAccessAnalysis &initialize(bool relaxedAliasing) {
    AliasAnalysis &aliasAnalysis = am.getAnalysis<mlir::AliasAnalysis>();
    (aliasAnalysis.addAnalysisImplementation(Plugins(relaxedAliasing)), ...);

    // Run the dataflow analysis we depend on.
    DataFlowSolverWrapper solver(aliasAnalysis);
    solver.load<dataflow::IntegerRangeAnalysis>();
    solver.loadWithRequiredAnalysis<ReachingDefinitionAnalysis>(aliasAnalysis);

    if (failed(solver.initializeAndRun(operation))) {
      operation->emitError("Failed to run required dataflow analysis");
      return *this;
    }

    // Try to construct the memory access matrix and offset vector for affine
    // memory operation of interest.
    operation->walk<WalkOrder::PreOrder>([&](Operation *op) {
      TypeSwitch<Operation *>(op)
          .Case<affine::AffineStoreOp, affine::AffineLoadOp>(
              [&](auto memoryOp) { build(memoryOp, solver); });
    });

    isInitialized = true;
    return *this;
  }

  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa);

  /// Return the memory access for the given memref \p access.
  std::optional<MemoryAccess>
  getMemoryAccess(const affine::MemRefAccess &access) const;

private:
  /// Attempt to create an entry in the accessMap for the given memory operation
  /// \p memoryOp.
  template <typename T> void build(T memoryOp, DataFlowSolver &solver);

  /// Construct the access matrix if possible.
  std::optional<MemoryAccessMatrix>
  buildAccessMatrix(sycl::SYCLAccessorSubscriptOp accessorSubscriptOp,
                    ArrayRef<Value> threadVars, ArrayRef<Value> loopIVs,
                    ArrayRef<Value> underlyingVals, DataFlowSolver &solver);

  /// Construct the offset vector if possible.
  std::optional<OffsetVector>
  buildOffsetVector(const MemoryAccessMatrix &matrix,
                    ArrayRef<Value> threadVars, ArrayRef<Value> loopIVs,
                    ArrayRef<Value> underlyingVals, DataFlowSolver &solver);

  /// Returns true if the memory access \p access has a single subscript that is
  /// zero, and false otherwise.
  bool hasZeroIndex(const affine::MemRefAccess &access) const;

  /// Collect the underlying value(s) of the operand at index \p opIndex in
  /// operation \p op.
  /// For example given:
  ///
  ///   sycl.constructor @id(%id, %i, %j) : (memref<?x!sycl_id_2>, i64, i64)
  ///   %0 = sycl.accessor.subscript %acc[%id] ...
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

  /// Whether the analysis has been properly initialized before using it.
  bool isInitialized = false;
};

} // namespace polygeist
} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_ANALYSIS_MEMORYACCESSANALYSIS_H
