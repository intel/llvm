//===- MemoryAccessAnalysis.cpp - Memory Access Analysis ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Analysis/MemoryAccessAnalysis.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <utility>

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Polygeist/Utils/TransformUtils.h"

#define DEBUG_TYPE "memory-access-analysis"

using namespace mlir;
using namespace mlir::polygeist;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Determine whether a value is an known integer value.
static std::optional<APInt> getConstIntegerValue(Value val,
                                                 DataFlowSolver &solver) {
  if (!val.getType().isIntOrIndex())
    return std::nullopt;

  auto *inferredRange =
      solver.lookupState<dataflow::IntegerValueRangeLattice>(val);
  if (!inferredRange || inferredRange->getValue().isUninitialized()) {
    LLVM_DEBUG(llvm::dbgs() << "Solver not initialized correctly\n");
    return std::nullopt;
  }

  const ConstantIntRanges &range = inferredRange->getValue().getValue();
  LLVM_DEBUG(llvm::dbgs() << "range: " << range << "\n");

  return range.getConstantValue();
}

/// Determine whether a value is equal to a given integer constant.
static bool isEqualTo(Value val, int constant, DataFlowSolver &solver) {
  std::optional<APInt> optConstVal = getConstIntegerValue(val, solver);
  if (!optConstVal.has_value())
    return false;

  APInt constVal = *optConstVal;
  APInt c(constVal.getBitWidth(), constant, true /* signed */);
  return (constVal == c);
}

static bool isZero(Value val, DataFlowSolver &solver) {
  return isEqualTo(val, 0, solver);
}

static bool isOne(Value val, DataFlowSolver &solver) {
  return isEqualTo(val, 1, solver);
}

static bool isNegativeOne(Value val, DataFlowSolver &solver) {
  return isEqualTo(val, -1, solver);
}

static bool isStrictlyPositive(Value val, DataFlowSolver &solver) {
  std::optional<APInt> constValue = getConstIntegerValue(val, solver);
  return (constValue.has_value() && constValue->isStrictlyPositive());
}

static bool isGreaterThanOne(Value val, DataFlowSolver &solver) {
  std::optional<APInt> optConstVal = getConstIntegerValue(val, solver);
  return (optConstVal.has_value() && optConstVal->sgt(1));
}

static bool isSmallerThanNegativeOne(Value val, DataFlowSolver &solver) {
  std::optional<APInt> optConstVal = getConstIntegerValue(val, solver);
  return (optConstVal.has_value() && optConstVal->slt(-1));
}

//===----------------------------------------------------------------------===//
// MemoryAccessMatrix
//===----------------------------------------------------------------------===//

MemoryAccessMatrix::MemoryAccessMatrix(size_t nRows, size_t nColumns)
    : nRows(nRows), nColumns(nColumns), data(nRows * nColumns) {}

MemoryAccessMatrix::MemoryAccessMatrix(
    std::initializer_list<std::initializer_list<Value>> initList) {
  assert(initList.size() != 0 && initList.begin()->size() != 0 &&
         "Expecting a non-empty initializer list");
  nRows = initList.size();
  nColumns = initList.begin()->size();

  assert(llvm::all_of(initList,
                      [this](const std::initializer_list<Value> &initRow) {
                        return initRow.size() == nColumns;
                      }) &&
         "Rows should all have the same size");

  data.reserve(nRows * nColumns);
  for (const std::initializer_list<Value> &initRow : initList)
    data.append(initRow);
}

void MemoryAccessMatrix::swapRows(size_t row, size_t otherRow) {
  assert((row < nRows && otherRow < nRows) && "out of bounds");
  if (row == otherRow)
    return;
  for (size_t col = 0; col < nColumns; ++col)
    std::swap(at(row, col), at(otherRow, col));
}

void MemoryAccessMatrix::swapColumns(size_t column, size_t otherColumn) {
  assert((column < nColumns && otherColumn < nColumns) && "out of bounds");
  if (column == otherColumn)
    return;
  for (size_t row = 0; row < nRows; ++row)
    std::swap(at(row, column), at(row, otherColumn));
}

SmallVector<Value> MemoryAccessMatrix::getRow(size_t row) const {
  assert(row < nRows && "the matrix must contain the given row");
  SmallVector<Value> rowCopy;
  rowCopy.reserve(nColumns);
  for (size_t col = 0; col < nColumns; ++col)
    rowCopy.emplace_back(at(row, col));
  return rowCopy;
}

void MemoryAccessMatrix::setRow(size_t row, ArrayRef<Value> elems) {
  assert(row < nRows && "the matrix must contain the given row");
  assert(elems.size() == nColumns && "elems size must match row length!");
  for (size_t col = 0; col < nColumns; ++col)
    at(row, col) = elems[col];
}

void MemoryAccessMatrix::fillRow(size_t row, Value val) {
  assert(row < nRows && "the matrix must contain the given row");
  for (size_t col = 0; col < nColumns; ++col)
    at(row, col) = val;
}

size_t MemoryAccessMatrix::appendRow() {
  ++nRows;
  data.resize(nRows * nColumns);
  return nRows - 1;
}

size_t MemoryAccessMatrix::appendRow(ArrayRef<Value> elems) {
  size_t row = appendRow();
  setRow(row, elems);
  return row;
}

SmallVector<Value> MemoryAccessMatrix::getColumn(size_t column) const {
  assert(column < nColumns && "the matrix must contain the given column");
  SmallVector<Value> columnCopy;
  columnCopy.reserve(nRows);
  for (size_t row = 0; row < nRows; ++row)
    columnCopy.emplace_back(at(row, column));
  return columnCopy;
}

void MemoryAccessMatrix::setColumn(size_t col, ArrayRef<Value> elems) {
  assert(col < nColumns && "the matrix must contain the given column");
  assert(elems.size() == nRows && "elems size must match column length!");
  for (size_t row = 0; row < nRows; ++row)
    at(row, col) = elems[row];
}

void MemoryAccessMatrix::fillColumn(size_t col, Value val) {
  assert(col < nColumns && "the matrix must contain the given column");
  for (size_t row = 0; row < nRows; ++row)
    at(row, col) = val;
}

void MemoryAccessMatrix::fill(Value val) {
  for (size_t row = 0; row < nRows; ++row)
    for (size_t col = 0; col < nColumns; ++col)
      at(row, col) = val;
}

MemoryAccessMatrix MemoryAccessMatrix::getRows(std::set<size_t> rows) const {
  assert(llvm::all_of(rows, [this](size_t row) { return row < nRows; }) &&
         "out of bounds");
  MemoryAccessMatrix subMatrix(rows.size(), nColumns);
  size_t pos = 0;
  for (size_t rowNum : rows)
    subMatrix.setRow(pos++, getRow(rowNum));
  return subMatrix;
}

MemoryAccessMatrix
MemoryAccessMatrix::getColumns(std::set<size_t> columns) const {
  assert(llvm::all_of(columns,
                      [this](size_t column) { return column < nColumns; }) &&
         "out of bounds");
  MemoryAccessMatrix subMatrix(nRows, columns.size());
  size_t pos = 0;
  for (size_t colNum : columns)
    subMatrix.setColumn(pos++, getColumn(colNum));
  return subMatrix;
}

MemoryAccessMatrix
MemoryAccessMatrix::getSubMatrix(std::set<size_t> rows,
                                 std::set<size_t> columns) const {
  return getRows(std::move(rows)).getColumns(std::move(columns));
}

bool MemoryAccessMatrix::isSquare() const { return (nRows == nColumns); }

bool MemoryAccessMatrix::isZero(DataFlowSolver &solver) const {
  return (llvm::all_of(data,
                       [&solver](Value val) { return ::isZero(val, solver); }));
}

bool MemoryAccessMatrix::isDiagonal(DataFlowSolver &solver) const {
  if (!isSquare())
    return false;

  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      Value val = at(row, col);
      bool isOnDiagonal = (row == col);

      // All values on the diagonal must be non-zero.
      if (isOnDiagonal && ::isZero(val, solver))
        return false;
      // All other values must be zero.
      if (!isOnDiagonal && !::isZero(val, solver))
        return false;
    }
  }

  return true;
}

bool MemoryAccessMatrix::isIdentity(DataFlowSolver &solver) const {
  if (!isSquare())
    return false;

  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      Value val = at(row, col);
      bool isOnDiagonal = (row == col);

      // All values on the diagonal must be one.
      if (isOnDiagonal && !isOne(val, solver))
        return false;
      // All other values must be zero.
      if (!isOnDiagonal && !::isZero(val, solver))
        return false;
    }
  }

  return true;
}

bool MemoryAccessMatrix::isLowerTriangular(DataFlowSolver &solver) const {
  if (!isSquare())
    return false;

  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      Value val = at(row, col);
      bool isAboveDiagonal = (col > row);

      // All values above the diagonal must be zero.
      if (isAboveDiagonal && !::isZero(val, solver))
        return false;
      // All other values must be non-zero.
      if (!isAboveDiagonal && ::isZero(val, solver))
        return false;
    }
  }

  return true;
}

bool MemoryAccessMatrix::isUpperTriangular(DataFlowSolver &solver) const {
  if (!isSquare())
    return false;

  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      Value val = at(row, col);
      bool isBelowDiagonal = (col < row);

      // All values below the diagonal must be zero.
      if (isBelowDiagonal && !::isZero(val, solver))
        return false;
      // All other values must be non-zero.
      if (!isBelowDiagonal && ::isZero(val, solver))
        return false;
    }
  }

  return true;
}

bool MemoryAccessMatrix::hasLinearAccessPattern(DataFlowSolver &solver) const {
  return isIdentity(solver);
}

bool MemoryAccessMatrix::hasReverseLinearAccessPattern(
    DataFlowSolver &solver) const {
  if (!isSquare())
    return false;

  // Ensure the matrix is diagonal with all non-zero elements equal to one
  // except the last one which must be equal to negative one.
  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      Value val = at(row, col);
      bool isOnDiagonal = (col == row);

      if (!isOnDiagonal && !::isZero(val, solver))
        return false;

      if (isOnDiagonal) {
        bool isLastDiagonalElem = (row == nRows - 1 && col == nColumns - 1);
        if (!isLastDiagonalElem && !isOne(val, solver))
          return false;
        if (isLastDiagonalElem && !isNegativeOne(val, solver))
          return false;
      }
    }
  }

  return true;
}

bool MemoryAccessMatrix::hasLinearOverlappedAccessPattern(
    DataFlowSolver &solver) const {
  if (!isSquare())
    return false;

  // Ensure the matrix is lower triangular with all non-zero elements equal to
  // one.
  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      Value val = at(row, col);
      bool isAboveDiagonal = (col > row);

      if (isAboveDiagonal && !::isZero(val, solver))
        return false;
      if (!isAboveDiagonal && !isOne(val, solver))
        return false;
    }
  }

  return true;
}

bool MemoryAccessMatrix::hasReverseLinearOverlappedAccessPattern(
    DataFlowSolver &solver) const {
  if (!isSquare())
    return false;

  // Ensure the matrix is lower triangular with all non-zero elements equal to
  // one except the last one on the diagonal which must be equal to negative
  // one.
  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      Value val = at(row, col);
      bool isAboveDiagonal = (col > row);

      if (isAboveDiagonal && !::isZero(val, solver))
        return false;

      if (!isAboveDiagonal) {
        bool isLastDiagonalElem = (row == nRows - 1 && col == nColumns - 1);
        if (!isLastDiagonalElem && !isOne(val, solver))
          return false;
        if (isLastDiagonalElem && !isNegativeOne(val, solver))
          return false;
      }
    }
  }

  return true;
}

bool MemoryAccessMatrix::hasStridedAccessPattern(DataFlowSolver &solver) const {
  if (!isSquare())
    return false;

  // Ensure the matrix is diagonal with all elements equal to one except the
  // last one which must be greater than one.
  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      Value val = at(row, col);
      bool isOnDiagonal = (col == row);

      if (!isOnDiagonal && !::isZero(val, solver))
        return false;

      if (isOnDiagonal) {
        bool isLastDiagonalElem = (row == nRows - 1 && col == nColumns - 1);
        if (!isLastDiagonalElem && !isOne(val, solver))
          return false;
        if (isLastDiagonalElem && !isGreaterThanOne(val, solver))
          return false;
      }
    }
  }

  return true;
}

bool MemoryAccessMatrix::hasReverseStridedAccessPattern(
    DataFlowSolver &solver) const {
  if (!isSquare())
    return false;

  // Ensure the matrix is diagonal with all elements equal to one except the
  // last one which must be smaller than negative one.
  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      Value val = at(row, col);
      bool isOnDiagonal = (col == row);

      if (!isOnDiagonal && !::isZero(val, solver))
        return false;

      if (isOnDiagonal) {
        bool isLastDiagonalElem = (row == nRows - 1 && col == nColumns - 1);
        if (!isLastDiagonalElem && !isOne(val, solver))
          return false;
        if (isLastDiagonalElem && !isSmallerThanNegativeOne(val, solver))
          return false;
      }
    }
  }

  return true;
}

bool MemoryAccessMatrix::hasStridedOverlappedAccessPattern(
    DataFlowSolver &solver) const {
  if (!isSquare())
    return false;

  // Ensure the matrix is lower triangular with all non-zero elements equal to
  // one except the last one on the diagonal which must be strictly positive.
  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      Value val = at(row, col);
      bool isAboveDiagonal = (col > row);

      if (isAboveDiagonal && !::isZero(val, solver))
        return false;

      if (!isAboveDiagonal) {
        bool isLastDiagonalElem = (row == nRows - 1 && col == nColumns - 1);
        if (!isLastDiagonalElem && !isOne(val, solver))
          return false;
        if (isLastDiagonalElem && !isStrictlyPositive(val, solver))
          return false;
      }
    }
  }

  return true;
}

bool MemoryAccessMatrix::hasReverseStridedOverlappedAccessPattern(
    DataFlowSolver &solver) const {
  if (!isSquare())
    return false;

  // Ensure the matrix is lower triangular with all non-zero elements equal to
  // one except the last one on the diagonal which must be smaller than
  // negative one.
  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      Value val = at(row, col);
      bool isAboveDiagonal = (col > row);

      if (isAboveDiagonal && !::isZero(val, solver))
        return false;

      if (!isAboveDiagonal) {
        bool isLastDiagonalElem = (row == nRows - 1 && col == nColumns - 1);
        if (!isLastDiagonalElem && !isOne(val, solver))
          return false;
        if (isLastDiagonalElem && !isSmallerThanNegativeOne(val, solver))
          return false;
      }
    }
  }

  return true;
}

std::optional<APInt>
MemoryAccessMatrix::getConstIntegerValue(size_t row, size_t column,
                                         DataFlowSolver &solver) const {
  Value val = at(row, column);
  return ::getConstIntegerValue(val, solver);
}

//===----------------------------------------------------------------------===//
// OffsetVector
//===----------------------------------------------------------------------===//

OffsetVector::OffsetVector(size_t nRows) : nRows(nRows), offsets(nRows) {}

OffsetVector::OffsetVector(std::initializer_list<Value> initList) {
  assert(initList.size() != 0 && "Expecting a non-empty initializer list");
  nRows = initList.size();
  offsets.append(initList);
}

void OffsetVector::swapRows(size_t row, size_t otherRow) {
  assert((row < nRows && otherRow < nRows) && "out of bounds");
  if (row == otherRow)
    return;
  std::swap(at(row), at(otherRow));
}

Value OffsetVector::getOffset(size_t row) const { return at(row); }

void OffsetVector::setOffset(size_t row, Value offset) { at(row) = offset; }

void OffsetVector::fill(Value val) {
  for (size_t row = 0; row < nRows; ++row)
    at(row) = val;
}

size_t OffsetVector::append(Value offset) {
  ++nRows;
  offsets.resize(nRows);
  size_t lastRow = nRows - 1;
  setOffset(lastRow, offset);
  return lastRow;
}

bool OffsetVector::isZero(DataFlowSolver &solver) const {
  return llvm::all_of(offsets,
                      [&](Value offset) { return ::isZero(offset, solver); });
}

bool OffsetVector::isZeroWithLastElementStrictlyPositive(
    DataFlowSolver &solver) const {
  size_t lastIndex = nRows - 1;
  for (size_t pos = 0; pos < nRows; ++pos) {
    Value val = at(pos);
    bool isLastIndex = (pos == lastIndex);
    if (!isLastIndex && !::isZero(val, solver))
      return false;
    if (isLastIndex && !isStrictlyPositive(val, solver))
      return false;
  }
  return true;
}

bool OffsetVector::isZeroWithLastElementEqualTo(int k,
                                                DataFlowSolver &solver) const {
  LLVM_DEBUG(llvm::dbgs() << "In isZeroWithLastElementEqualTo\n");

  size_t lastIndex = nRows - 1;
  for (size_t pos = 0; pos < nRows; ++pos) {
    Value val = at(pos);
    bool isLastIndex = (pos == lastIndex);
    if (!isLastIndex && !::isZero(val, solver))
      return false;

    if (isLastIndex && !isEqualTo(val, k, solver))
      return false;
  }
  return true;
}

std::optional<APInt>
OffsetVector::getConstIntegerValue(size_t row, DataFlowSolver &solver) const {
  Value val = at(row);
  return ::getConstIntegerValue(val, solver);
}

//===----------------------------------------------------------------------===//
// MemoryAccess
//===----------------------------------------------------------------------===//

template <typename OpTy>
MemoryAccessPattern
MemoryAccess<OpTy>::classifyMemoryAccess(DataFlowSolver &solver) const {
  bool isZeroVector = offsets.isZero(solver);

  if (isZeroVector) {
    if (matrix.hasLinearAccessPattern(solver))
      return MemoryAccessPattern::Linear;

    if (matrix.hasLinearOverlappedAccessPattern(solver))
      return MemoryAccessPattern::LinearOverlapped;

    if (matrix.hasStridedAccessPattern(solver))
      return MemoryAccessPattern::Strided;

    if (matrix.hasStridedOverlappedAccessPattern(solver))
      return MemoryAccessPattern::StridedOverlapped;

    return MemoryAccessPattern::Unknown;
  }

  if (matrix.hasLinearAccessPattern(solver) &&
      offsets.isZeroWithLastElementStrictlyPositive(solver))
    return MemoryAccessPattern::LinearShifted;

  if (matrix.hasReverseLinearAccessPattern(solver) &&
      offsets.isZeroWithLastElementEqualTo(matrix.getNumColumns() - 1, solver))
    return MemoryAccessPattern::ReverseLinear;

  if (matrix.hasReverseLinearAccessPattern(solver) &&
      offsets.isZeroWithLastElementStrictlyPositive(solver))
    return MemoryAccessPattern::ReverseLinearShifted;

  if (matrix.hasReverseLinearOverlappedAccessPattern(solver) &&
      offsets.isZeroWithLastElementStrictlyPositive(solver))
    return MemoryAccessPattern::ReverseLinearOverlapped;

  if (matrix.hasReverseStridedAccessPattern(solver) &&
      offsets.isZeroWithLastElementEqualTo(matrix.getNumColumns() - 1, solver))
    return MemoryAccessPattern::ReverseStrided;

  if (matrix.hasStridedAccessPattern(solver) &&
      offsets.isZeroWithLastElementStrictlyPositive(solver))
    return MemoryAccessPattern::StridedShifted;

  if (matrix.hasReverseStridedAccessPattern(solver) &&
      offsets.isZeroWithLastElementStrictlyPositive(solver))
    return MemoryAccessPattern::ReverseStridedShifted;

  if (matrix.hasReverseStridedOverlappedAccessPattern(solver) &&
      offsets.isZeroWithLastElementStrictlyPositive(solver))
    return MemoryAccessPattern::ReverseStridedOverlapped;

  return MemoryAccessPattern::Unknown;
}

//===----------------------------------------------------------------------===//
// MemoryAccessAnalysis
//===----------------------------------------------------------------------===//

/// Implementation detail that walks up the parents and records the ones with
/// the specified type.
/// TODO: could also be implemented as a collect parents followed by a
/// filter and made available outside this file.
template <typename T>
static SetVector<Operation *> getParentsOfType(Block *block) {
  SetVector<Operation *> res;
  auto *current = block->getParentOp();
  while (current) {
    if (auto typedParent = dyn_cast<T>(current)) {
      assert(res.count(current) == 0 && "Already inserted");
      res.insert(current);
    }
    current = current->getParentOp();
  }
  return res;
}

MemoryAccessAnalysis::MemoryAccessAnalysis(Operation *op, AnalysisManager &am)
    : operation(op), am(am) {
  build();
}

bool MemoryAccessAnalysis::isInvalidated(
    const AnalysisManager::PreservedAnalyses &pa) {
  return !pa.isPreserved<AliasAnalysis>() ||
         !pa.isPreserved<dataflow::DeadCodeAnalysis>() ||
         !pa.isPreserved<dataflow::SparseConstantPropagation>() ||
         !pa.isPreserved<dataflow::IntegerRangeAnalysis>() ||
         !pa.isPreserved<polygeist::ReachingDefinitionAnalysis>();
}

std::optional<MemoryAccessMatrix>
MemoryAccessAnalysis::getMemoryAccessMatrix(const MemRefAccess &access) const {
  auto it = accessMap.find(access.opInst);
  if (it == accessMap.end())
    return std::nullopt;
  return it->second;
}

void MemoryAccessAnalysis::build() {
  AliasAnalysis &aliasAnalysis = am.getAnalysis<mlir::AliasAnalysis>();
  aliasAnalysis.addAnalysisImplementation(
      sycl::AliasAnalysis(false /* relaxedAliasing*/));

  // Run the dataflow analysis we depend on.
  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<dataflow::IntegerRangeAnalysis>();
  solver.load<polygeist::ReachingDefinitionAnalysis>(aliasAnalysis);

  Operation *op = getOperation();
  if (failed(solver.initializeAndRun(op))) {
    op->emitError("Failed to run required dataflow analysis");
    return;
  }

  // Try to construct the memory access matrix for each affine memory operation
  // contained within this operation.
  op->walk<WalkOrder::PreOrder>([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case([&](AffineStoreOp memoryOp) { build(memoryOp, solver); })
        .Case([&](AffineLoadOp memoryOp) { build(memoryOp, solver); });
  });
}

template <typename T>
bool MemoryAccessAnalysis::build(T memoryOp, DataFlowSolver &solver) {
  static_assert(llvm::is_one_of<T, AffineLoadOp, AffineStoreOp>::value);

  MemRefType memRefType = memoryOp.getMemRefType();
  if (!memRefType.getLayout().isIdentity())
    return false;

  // We are interested in memory accesses that have:
  //  - base operand equal to the result of a sycl accessor subscript, and
  //  - a single index with zero value.
  MemRefAccess access(memoryOp);
  if (!isa<sycl::SYCLAccessorSubscriptOp>(access.memref.getDefiningOp()) ||
      !hasZeroIndex(access))
    return false;

  LLVM_DEBUG(llvm::errs() << "Candidate op:" << memoryOp << "\n");

  // Try to determine the underlying value of the accessor subscript
  // offset operand.
  auto accessorSubscriptOp =
      cast<sycl::SYCLAccessorSubscriptOp>(access.memref.getDefiningOp());
  std::optional<Value> val = getUnderlyingValueOf(
      accessorSubscriptOp.getOffsetOperandIndex(), accessorSubscriptOp, solver);
  if (!val)
    return false;

  LLVM_DEBUG(llvm::dbgs() << "Underlying val: " << val << "\n");

  // Construct the memory access matrix.
  SetVector<Operation *> enclosingLoops =
      getParentsOfType<AffineForOp>(memoryOp->getBlock());
  MemoryAccessMatrix accessMatrix(access.getRank(), enclosingLoops.size());

  SmallVector<Value, 4> loopIVs;
  for (Operation *loop : enclosingLoops) {
    auto affineForOp = cast<AffineForOp>(loop);
    loopIVs.push_back(affineForOp.getInductionVar());
  }

  // Attempt to match an expression like (iv * other) or (other * iv).
  Value iv = loopIVs[0];
  Value other;
  if (matchPattern(*val, m_Op<arith::MulIOp>(matchers::m_Val(iv),
                                             matchers::m_Any(&other))) ||
      matchPattern(*val, m_Op<arith::MulIOp>(matchers::m_Any(&other),
                                             matchers::m_Val(iv)))) {
    llvm::errs() << "BINGO, other: " << other << "\n";
    accessMatrix(0, 0) = other;
  }

  LLVM_DEBUG(llvm::dbgs() << "accessMatrix constructed successfully\n");

  accessMap[access.opInst] = accessMatrix;
  return true;
}

bool MemoryAccessAnalysis::hasZeroIndex(const MemRefAccess &access) const {
  AffineValueMap accessValueMap;
  access.getAccessMap(&accessValueMap);
  if (accessValueMap.getNumDims() != 0)
    return false;

  auto index = accessValueMap.getResult(0).dyn_cast<AffineConstantExpr>();
  return (index && index.getValue() == 0);
}

std::shared_ptr<Definition>
MemoryAccessAnalysis::getUniqueDefinitionOrNull(unsigned opIndex, Operation *op,
                                                DataFlowSolver &solver) const {
  using ModifiersTy = ReachingDefinition::ModifiersTy;

  const polygeist::ReachingDefinition *reachingDef =
      solver.lookupState<polygeist::ReachingDefinition>(op);
  if (!reachingDef)
    return nullptr;

  Value operand = op->getOperand(opIndex);
  std::optional<ModifiersTy> mods = reachingDef->getModifiers(operand);
  std::optional<ModifiersTy> pMods =
      reachingDef->getPotentialModifiers(operand);

  // If there are potential modifiers then there is no unique modifier.
  if (pMods.has_value() && !pMods->empty())
    return nullptr;

  if (!mods.has_value() || mods->empty())
    return nullptr;

  return *mods->begin();
}

std::optional<Value>
MemoryAccessAnalysis::getUnderlyingValueOf(unsigned opIndex, Operation *op,
                                           DataFlowSolver &solver) const {
  Value operand = op->getOperand(opIndex);
  std::shared_ptr<Definition> def =
      getUniqueDefinitionOrNull(opIndex, op, solver);
  if (!def)
    return operand;

  LLVM_DEBUG({
    llvm::dbgs() << "operand: " << operand << "\n";
    llvm::dbgs() << "operand definition: " << *def << "\n";
  });

  return TypeSwitch<Operation *, std::optional<Value>>(def->getOperation())
      .Case([&](AffineStoreOp storeOp) -> std::optional<Value> {
        MemRefAccess storeAccess(storeOp);
        if (!hasZeroIndex(storeAccess))
          return std::nullopt;

        Value storedVal =
            storeOp.getOperand(storeOp.getStoredValOperandIndex());
        if (!isa<AffineLoadOp>(storedVal.getDefiningOp()))
          return getUnderlyingValueOf(storeOp.getStoredValOperandIndex(),
                                      storeOp, solver);

        AffineLoadOp loadOp = cast<AffineLoadOp>(storedVal.getDefiningOp());
        MemRefAccess loadAccess(loadOp);
        if (!hasZeroIndex(loadAccess))
          return std::nullopt;

        return getUnderlyingValueOf(loadOp.getMemRefOperandIndex(), loadOp,
                                    solver);
      })
      .Case([&](sycl::SYCLConstructorOp constructorOp) -> std::optional<Value> {
        if (constructorOp.getNumOperands() != 2)
          return std::nullopt;

        return getUnderlyingValueOf(1, constructorOp, solver);
      })
      .Default([&](auto *op) -> std::optional<Value> {
        if (op->getNumResults() != 1)
          return std::nullopt;

        return op->getResult(0);
      });
}

namespace mlir {
namespace polygeist {

template class MemoryAccess<AffineLoadOp>;
template class MemoryAccess<AffineStoreOp>;

} // namespace polygeist
} // namespace mlir
