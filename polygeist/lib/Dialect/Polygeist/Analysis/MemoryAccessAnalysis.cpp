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
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "memory-access-analysis"

using namespace mlir;
using namespace mlir::affine;
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

/// Walks up the parents and records the ones with the specified type.
template <typename T> static SetVector<T> getParentsOfType(Block *block) {
  SetVector<T> res;
  constexpr auto getInitialParent = [](Block *b) -> T {
    Operation *op = b->getParentOp();
    auto typedOp = dyn_cast<T>(op);
    return typedOp ? typedOp : op->getParentOfType<T>();
  };

  for (T parent = getInitialParent(block); parent;
       parent = parent->template getParentOfType<T>())
    res.insert(parent);

  return res;
}

static bool usesValue(Operation *op, Value val) {
  assert(op && "Expecting valid operation");
  if (Operation *valOp = val.getDefiningOp())
    if (valOp == op)
      return true;
  return llvm::any_of(op->getOperands(), [&](Value operand) {
    if (Operation *operandOp = operand.getDefiningOp())
      return usesValue(operandOp, val);
    return (operand == val);
  });
}

namespace {

/// Represents a multiplication factor.
class Multiplier {
  friend raw_ostream &operator<<(raw_ostream &, const Multiplier &);

public:
  Multiplier(Value val) : val(val) {}

  bool operator==(const Multiplier &other) { return val == other.val; }

  Value getValue() const { return val; }

  /// Create a multiplier with value one.
  static Multiplier one(MLIRContext *ctx) {
    OpBuilder b(ctx);
    return Multiplier(b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 1));
  }

private:
  Value val;
};

raw_ostream &operator<<(raw_ostream &os, const Multiplier &multiplier) {
  return os << multiplier.val;
}

/// Represents a multiplier or a value.
class ValueOrMultiplier {
  friend raw_ostream &operator<<(raw_ostream &, const ValueOrMultiplier &);

public:
  ValueOrMultiplier(Multiplier mul) : data(mul) {}
  ValueOrMultiplier(Value val) : data(val) {}

  bool isMultiplier() const { return std::holds_alternative<Multiplier>(data); }
  bool isValue() const { return std::holds_alternative<Value>(data); }

  Multiplier getMultiplier() const {
    assert(isMultiplier() && "Expecting multiplier");
    return std::get<Multiplier>(data);
  }

  Value getValue() const {
    assert(isValue() && "Expecting value");
    return std::get<Value>(data);
  }

  /// Determine whether this class holds the value one.
  bool isOne(DataFlowSolver &solver) const {
    Value val = isMultiplier() ? getMultiplier().getValue() : getValue();
    if (!val)
      return false;

    if (auto constVal = dyn_cast<arith::ConstantIndexOp>(val.getDefiningOp())) {
      if (constVal.value() == 1)
        return true;
    }
    std::optional<APInt> constVal = getConstIntegerValue(val, solver);
    return (constVal.has_value() && constVal->isOne());
  }

private:
  std::variant<Value, Multiplier> data;
};

[[maybe_unused]] raw_ostream &
operator<<(raw_ostream &os, const ValueOrMultiplier &valOrMultiplier) {
  if (valOrMultiplier.isMultiplier())
    return os << valOrMultiplier.getMultiplier();
  return os << valOrMultiplier.getValue();
}

} // namespace

// Visit a binary operation of type \tparam T. The LHS and RHS operand
// of the binary operation are processed by applying the function \p
// getMultiplier. The result of the visit is computed via the \p
// computeResult function.
template <typename T, typename ProcessOperandFuncT, typename ComputeResultFuncT,
          typename = std::enable_if_t<llvm::is_one_of<
              T, arith::AddIOp, arith::SubIOp, arith::MulIOp>::value>,
          typename = std::enable_if<
              std::is_invocable_r_v<ValueOrMultiplier, ProcessOperandFuncT, T,
                                    const Value, DataFlowSolver &>>,
          typename = std::enable_if<
              std::is_invocable_r_v<ValueOrMultiplier, ComputeResultFuncT,
                                    ValueOrMultiplier, ValueOrMultiplier>>>
static ValueOrMultiplier visitBinaryOp(T binOp, const Value factor,
                                       DataFlowSolver &solver,
                                       ProcessOperandFuncT getMultiplier,
                                       ComputeResultFuncT computeResult) {
  // Traverse my subtrees.
  auto lhsRes = getMultiplier(binOp.getLhs(), factor, solver);
  auto rhsRes = getMultiplier(binOp.getRhs(), factor, solver);

  // Compute the result to propagate up.
  auto res = computeResult(lhsRes, rhsRes);

  LLVM_DEBUG({
    llvm::dbgs() << "lhsRes: " << lhsRes << "\n";
    llvm::dbgs() << "rhsRes: " << rhsRes << "\n";
    llvm::dbgs() << "res = " << res << "\n";
  });

  return res;
}

/// Determine whether \p expr involves a subexpression which is a
/// multiplication of \p val, and return the multiplication factor if it does.
/// Example:
///   %c1_i32 = arith.constant 1 : i32
///   %c2_i32 = arith.constant 2 : i32
///   %index_cast = arith.index_cast %ii : index to i64
///   %c1_i64 = arith.extsi %c1_i32 : i32 to i64
///   %mul = arith.muli %index_cast, %c1_i64 : i64
///   %c2_i64 = arith.extsi %c2_i32 : i32 to i64
///   %add = arith.addi %mul, %c2_i64 : i64
///
/// Here getMultiplier(%add, %ii) should return '%c1_i32'.
static ValueOrMultiplier getMultiplier(const Value expr, const Value factor,
                                       DataFlowSolver &solver) {
  Operation *op = expr.getDefiningOp();
  if (!op) {
    // This is a block argument. If it is a match create a multiplier with value
    // one.
    if (expr == factor)
      return Multiplier::one(expr.getContext());
    return expr;
  }

  auto getOperandThatMatchesFactor = [&factor](Value lhs, Value rhs) {
    if (lhs == factor && lhs != rhs)
      return lhs;
    if (rhs == factor && lhs != rhs)
      return rhs;
    return Value();
  };

  return TypeSwitch<Operation *, ValueOrMultiplier>(op)
      .Case<arith::AddIOp, arith::SubIOp>([&](auto binOp) {
        auto computeResult = [&](ValueOrMultiplier lhs,
                                 ValueOrMultiplier rhs) -> ValueOrMultiplier {
          bool lhsIsMul = lhs.isMultiplier(), rhsIsMul = rhs.isMultiplier();

          // If the LHS (or RHS) subtree passed up a multiplier and the other
          // operand is not a match (of the factor), propagate the multiplier
          // up.
          if (lhsIsMul && !rhsIsMul && rhs.getValue() != factor)
            return lhs;
          if (rhsIsMul && !lhsIsMul && lhs.getValue() != factor)
            return rhs;

          // Otherwise, if neither the LHS nor the RHS subtrees passed up a
          // multiplier, and one of the operands is a match, the multiplier is
          // one.
          if (!lhsIsMul && !rhsIsMul &&
              getOperandThatMatchesFactor(lhs.getValue(), rhs.getValue()) !=
                  nullptr)
            return Multiplier::one(expr.getContext());

          return Value();
        };

        LLVM_DEBUG(llvm::dbgs() << "expr: " << expr << "\n");

        return visitBinaryOp(binOp, factor, solver, getMultiplier,
                             computeResult);
      })
      .Case<arith::MulIOp>([&](auto mulOp) {
        auto computeResult = [&](ValueOrMultiplier lhs,
                                 ValueOrMultiplier rhs) -> ValueOrMultiplier {
          bool lhsIsMul = lhs.isMultiplier(), rhsIsMul = rhs.isMultiplier();

          // If neither the LHS nor the RHS subtree passed up a multiplier,
          // attempt to find it in this multiplication expression.
          if (!lhsIsMul && !rhsIsMul) {
            Value lhsVal = lhs.getValue(), rhsVal = rhs.getValue();
            if (getOperandThatMatchesFactor(lhsVal, rhsVal) == lhsVal)
              return Multiplier(rhsVal);
            if (getOperandThatMatchesFactor(lhsVal, rhsVal) == rhsVal)
              return Multiplier(lhsVal);
            return Value();
          }

          // If the LHS (or RHS) subtrees passed up a multiplier. Return:
          //   - the multiplier if the other operand has value one
          //   - a new multiplier with value equal to the other operand if
          //     + the multiplier is one and
          //     + the other operand is not the factor.
          if (lhsIsMul && !rhsIsMul) {
            if (rhs.isOne(solver))
              return lhs;
            if (lhs.isOne(solver) && rhs.getValue() != factor)
              return Multiplier(rhs.getValue());
            return Value();
          }
          if (rhsIsMul && !lhsIsMul) {
            if (lhs.isOne(solver))
              return rhs;
            if (rhs.isOne(solver) && lhs.getValue() != factor)
              return Multiplier(lhs.getValue());
            return Value();
          }

          return Value();
        };

        LLVM_DEBUG(llvm::dbgs() << "expr: " << expr << "\n");

        return visitBinaryOp(mulOp, factor, solver, getMultiplier,
                             computeResult);
      })
      .Case<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp, arith::IndexCastOp,
            arith::IndexCastUIOp>([&](auto castOp) {
        return getMultiplier(castOp.getOperand(), factor, solver);
      })
      .Default([&](auto) { return expr; });
}

//===----------------------------------------------------------------------===//
// MemoryAccessMatrix
//===----------------------------------------------------------------------===//

raw_ostream &mlir::polygeist::operator<<(raw_ostream &os,
                                         const MemoryAccessMatrix &matrix) {
  auto getConstant = [](Value val) -> Optional<int64_t> {
    return TypeSwitch<Operation *, Optional<int64_t>>(val.getDefiningOp())
        .Case<arith::ConstantIndexOp, arith::ConstantIntOp>(
            [](auto op) { return op.value(); })
        .Default([](auto) { return std::nullopt; });
  };

  for (size_t row = 0; row < matrix.getNumRows(); ++row) {
    llvm::interleave(
        matrix.getRow(row), os,
        [&](Value elem) {
          std::optional<int64_t> constant = getConstant(elem);
          if (constant.has_value())
            os << constant;
          else
            os << elem;
        },
        " ");
    os << '\n';
  }
  return os;
}

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
         !pa.isPreserved<ReachingDefinitionAnalysis>();
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
  solver.load<ReachingDefinitionAnalysis>(aliasAnalysis);

  Operation *op = getOperation();
  if (failed(solver.initializeAndRun(op))) {
    op->emitError("Failed to run required dataflow analysis");
    return;
  }

  // Try to construct the memory access matrix for affine memory operation of
  // interest.
  op->walk<WalkOrder::PreOrder>([&](Operation *op) {
    TypeSwitch<Operation *>(op).Case<AffineStoreOp, AffineLoadOp>(
        [&](auto memoryOp) { build(memoryOp, solver); });
  });
}

template <typename T>
void MemoryAccessAnalysis::build(T memoryOp, DataFlowSolver &solver) {
  static_assert(llvm::is_one_of<T, AffineLoadOp, AffineStoreOp>::value);

  MemRefType memRefType = memoryOp.getMemRefType();
  if (!memRefType.getLayout().isIdentity())
    return;

  // Currently we are only interested in SYCL memory accesses that have:
  //  - base operand equal to the result of a sycl accessor subscript, and
  //  - a single index with zero value.
  MemRefAccess access(memoryOp);
  auto accessorSubscriptOp =
      dyn_cast<sycl::SYCLAccessorSubscriptOp>(access.memref.getDefiningOp());
  if (!accessorSubscriptOp || !hasZeroIndex(access))
    return;

  LLVM_DEBUG(llvm::errs() << "Candidate op:" << memoryOp << "\n");

  // Try to determine the underlying value(s) of the accessor subscript index
  // operand. The number of underlying values should be equal to the
  // dimensionality of the sycl id used as an index in the accessor subscript
  // operation. Example:
  //   sycl.constructor @id(%id, %i, %j) : (memref<?x!sycl_id_2>, i64, i64)
  //   %sub = sycl.accessor.subscript %acc[%id] ...
  // Here the underlying values for '%id' are {%i, %j}. The number of underlying
  // values matches the dimensionality of the sycl id.
  SmallVector<Value> underlyingVals = getUnderlyingValues(
      accessorSubscriptOp.getOffsetOperandIndex(), accessorSubscriptOp, solver);
  if (underlyingVals.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to resolve underlying value(s) for: "
                            << accessorSubscriptOp.getIndex() << "\n");
    return;
  }

  Value accSubIndex = accessorSubscriptOp.getIndex();
  assert(
      sycl::getDimensions(accSubIndex.getType()) == underlyingVals.size() &&
      "Number of underlying values should be equal to dimensionality of the id "
      "used to index the accessor");
  LLVM_DEBUG({
    for (Value val : underlyingVals)
      llvm::dbgs() << "Underlying val: " << val << "\n";
  });

  // Construct the memory access matrix. The number of rows is equal to the
  // dimensionality of the sycl.id used by the accessor subscript operation. The
  // number of columns is equal to the number of loops surrounding the memory
  // access plus (TODO) the dimensionality of the sycl item received by the
  // kernel.
  SetVector<AffineForOp> enclosingLoops =
      getParentsOfType<AffineForOp>(memoryOp->getBlock());
  MemoryAccessMatrix accessMatrix(sycl::getDimensions(accSubIndex.getType()),
                                  enclosingLoops.size());

  // Collect the loop induction variables.
  // TODO: collect also the global thread ids.
  SmallVector<Value, 4> IVs;
  for (AffineForOp loop : llvm::reverse(enclosingLoops))
    IVs.push_back(loop.getInductionVar());

  OpBuilder b(access.memref.getContext());
  Value zero = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 0);

  // Initialize the memory access matrix.
  for (size_t row = 0; row < accessMatrix.getNumRows(); ++row) {
    Value val = underlyingVals[row];
    for (size_t col = 0; col < accessMatrix.getNumColumns(); ++col) {
      if (!usesValue(val.getDefiningOp(), IVs[col])) {
        accessMatrix(row, col) = zero;
        continue;
      }

      ValueOrMultiplier valOrMul = getMultiplier(val, IVs[col], solver);
      if (!valOrMul.isMultiplier()) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to find multiplier\n");
        return;
      }

      accessMatrix(row, col) = valOrMul.getMultiplier().getValue();
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "accessMatrix:\n" << accessMatrix << "\n");

  accessMap[access.opInst] = accessMatrix;
}

bool MemoryAccessAnalysis::hasZeroIndex(const MemRefAccess &access) const {
  AffineValueMap accessValueMap;
  access.getAccessMap(&accessValueMap);
  if (accessValueMap.getNumDims() != 0)
    return false;

  auto index = accessValueMap.getResult(0).dyn_cast<AffineConstantExpr>();
  return (index && index.getValue() == 0);
}

std::optional<Definition>
MemoryAccessAnalysis::getUniqueDefinition(unsigned opIndex, Operation *op,
                                          DataFlowSolver &solver) const {
  using ModifiersTy = ReachingDefinition::ModifiersTy;

  const ReachingDefinition *reachingDef =
      solver.lookupState<ReachingDefinition>(op);
  if (!reachingDef)
    return std::nullopt;

  Value operand = op->getOperand(opIndex);
  std::optional<ModifiersTy> mods = reachingDef->getModifiers(operand);
  std::optional<ModifiersTy> pMods =
      reachingDef->getPotentialModifiers(operand);

  // If there are potential modifiers then there is no unique modifier.
  if (pMods.has_value() && !pMods->empty())
    return std::nullopt;

  if (!mods.has_value() || mods->size() != 1)
    return std::nullopt;

  return *mods->begin();
}

SmallVector<Value>
MemoryAccessAnalysis::getUnderlyingValues(unsigned opIndex, Operation *op,
                                          DataFlowSolver &solver) const {
  std::optional<Definition> def = getUniqueDefinition(opIndex, op, solver);
  if (!def.has_value())
    return {op->getOperand(opIndex)};

  LLVM_DEBUG({
    llvm::dbgs() << "operand: " << op->getOperand(opIndex) << "\n";
    llvm::dbgs() << "operand definition: " << *def << "\n";
  });

  return TypeSwitch<Operation *, SmallVector<Value>>(def->getOperation())
      .Case([&](AffineStoreOp storeOp) {
        // Memory accesses involving SYCL accessors should have zero index.
        MemRefAccess storeAccess(storeOp);
        assert(hasZeroIndex(storeAccess) && "Unexpected candidate operation");

        Value storedVal =
            storeOp.getOperand(storeOp.getStoredValOperandIndex());
        if (!isa<AffineLoadOp>(storedVal.getDefiningOp()))
          return getUnderlyingValues(storeOp.getStoredValOperandIndex(),
                                     storeOp, solver);

        // Try to determine the underlying value of the memory pointed to by the
        // memref operand of a load.
        AffineLoadOp loadOp = cast<AffineLoadOp>(storedVal.getDefiningOp());
        MemRefAccess loadAccess(loadOp);
        assert(hasZeroIndex(storeAccess) && "Unexpected candidate operation");

        return getUnderlyingValues(loadOp.getMemRefOperandIndex(), loadOp,
                                   solver);
      })
      .Case([&](sycl::SYCLConstructorOp constructorOp) {
        assert(
            sycl::isIDPtrType(
                constructorOp.getOperand(constructorOp.getOutputOperandIndex())
                    .getType()) &&
            "add support for other types of sycl constructors");

        // Collect the underlying values of the sycl.constructor inputs.
        SmallVector<Value> vec;
        for (unsigned i = 0; i < constructorOp.getNumOperands(); ++i) {
          if (i == constructorOp.getOutputOperandIndex())
            continue;

          auto vals = getUnderlyingValues(i, constructorOp, solver);
          assert(vals.size() == 1 && "Expecting single value");
          vec.push_back(vals.front());
        }
        return vec;
      })
      .Default([&](auto *op) {
        SmallVector<Value> vec;
        for (auto res : op->getResults())
          vec.push_back(res);
        return vec;
      });
}

namespace mlir {
namespace polygeist {

template class MemoryAccess<affine::AffineLoadOp>;
template class MemoryAccess<affine::AffineStoreOp>;

} // namespace polygeist
} // namespace mlir
