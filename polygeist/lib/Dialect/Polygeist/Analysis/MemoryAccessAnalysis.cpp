//===- MemoryAccessAnalysis.cpp - Memory Access Analysis ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Analysis/MemoryAccessAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "llvm/ADT/TypeSwitch.h"

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
  return range.getConstantValue();
}

/// Determine whether a value is equal to a given integer constant.
static bool isEqualTo(Value val, int64_t constant, DataFlowSolver &solver) {
  std::optional<APInt> optConstVal = getConstIntegerValue(val, solver);
  if (!optConstVal.has_value())
    return false;

  APInt constVal = *optConstVal;
  APInt c(constVal.getBitWidth(), constant, true /*signed*/);
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

/// Walk up the parents and records the ones with the specified type.
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

/// Retrieve the operations with the specified type in the given function.
template <typename T>
static SetVector<T> getOperationsOfType(FunctionOpInterface funcOp) {
  SetVector<T> res;
  funcOp->walk([&](T op) { res.insert(op); });
  return res;
}

/// Determine whether \p op uses \p val (directly or indirectly).
static bool usesValue(Operation *op, Value val) {
  if (!op)
    return false;

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

/// Represents a multiplication factor in an affine expression.
/// Example:
///   k1*i + k2
/// Here the multiplier is 'k1'.
class Multiplier : public Value {
  friend raw_ostream &operator<<(raw_ostream &, const Multiplier &);

public:
  Multiplier(Value val) : Value(val) {}

  static Multiplier one(MLIRContext *ctx) {
    OpBuilder b(ctx);
    return Multiplier(b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 1));
  }
};

[[maybe_unused]] raw_ostream &operator<<(raw_ostream &os,
                                         const Multiplier &mul) {
  return os << "Multiplier: " << static_cast<Value>(mul);
}

/// Represents an offset in an affine expression.
/// Example:
///   k1*i + k2
/// Here the offset is 'k2'.
class Offset : public Value {
  friend raw_ostream &operator<<(raw_ostream &, const Offset &);

public:
  Offset(Value val) : Value(val) {}

  static Offset zero(MLIRContext *ctx) {
    OpBuilder b(ctx);
    return Offset(b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 0));
  }
};

[[maybe_unused]] raw_ostream &operator<<(raw_ostream &os, const Offset &off) {
  return os << "Offset: " << static_cast<Value>(off);
}

/// Represent a generic value or a value of type \tparam T.
template <typename T,
          typename =
              std::enable_if_t<llvm::is_one_of<T, Multiplier, Offset>::value>>
class ValueOr {
  friend raw_ostream &operator<<(raw_ostream &os, const ValueOr<T> &valOr) {
    if (valOr.template is<T>())
      return os << valOr.template get<T>();
    return os << valOr.template get<Value>();
  }

public:
  ValueOr(T val) : data(val) {}
  ValueOr(Value val) : data(val) {}

  template <typename U,
            typename = std::enable_if_t<llvm::is_one_of<U, T, Value>::value>>
  bool is() const {
    return std::holds_alternative<U>(data);
  }

  template <typename U,
            typename = std::enable_if_t<llvm::is_one_of<U, T, Value>::value>>
  U get() const {
    assert(is<U>() && "Incorrect type");
    return std::get<U>(data);
  }

  /// Determine whether this class holds the value \p constant.
  bool isEqualTo(int64_t constant, DataFlowSolver &solver) const {
    Value val = is<T>() ? get<T>() : get<Value>();
    if (!val)
      return false;

    if (auto constVal = dyn_cast<arith::ConstantIndexOp>(val.getDefiningOp())) {
      if (constVal.value() == constant)
        return true;
    }

    return ::isEqualTo(val, constant, solver);
  }

private:
  std::variant<Value, T> data;
};

template class ValueOr<Multiplier>;
template class ValueOr<Offset>;
} // namespace

// Visit a binary operation of type \tparam T. The LHS and RHS operand of the
// binary operation are processed by applying the function \p getMultiplier. The
// result of the visit is computed via the \p computeResult function.
template <typename T, typename ProcessOperandFuncT, typename ComputeResultFuncT,
          typename = std::enable_if_t<llvm::is_one_of<
              T, arith::AddIOp, arith::SubIOp, arith::MulIOp>::value>,
          typename = std::enable_if<
              std::is_invocable_r_v<ValueOr<Multiplier>, ProcessOperandFuncT, T,
                                    const Value, DataFlowSolver &>>,
          typename = std::enable_if<
              std::is_invocable_r_v<ValueOr<Multiplier>, ComputeResultFuncT,
                                    ValueOr<Multiplier>, ValueOr<Multiplier>>>>
static ValueOr<Multiplier> visitBinaryOp(T binOp, const Value factor,
                                         DataFlowSolver &solver,
                                         ProcessOperandFuncT getMultiplier,
                                         ComputeResultFuncT computeResult) {
  // Traverse my subtrees.
  ValueOr<Multiplier> lhsRes = getMultiplier(binOp.getLhs(), factor, solver);
  ValueOr<Multiplier> rhsRes = getMultiplier(binOp.getRhs(), factor, solver);

  // Compute the result to propagate up.
  ValueOr<Multiplier> res = computeResult(lhsRes, rhsRes);

  LLVM_DEBUG({
    llvm::dbgs() << "lhsRes: " << lhsRes << "\n";
    llvm::dbgs() << "rhsRes: " << rhsRes << "\n";
    llvm::dbgs() << "res = " << res << "\n";
  });

  return res;
}

/// Determine whether \p expr involves a subexpression which is a multiplication
/// of \p val, and return the multiplication factor if it does.
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
static ValueOr<Multiplier> getMultiplier(const Value expr, const Value factor,
                                         DataFlowSolver &solver) {
  if (expr == factor)
    return Multiplier::one(expr.getContext());

  Operation *op = expr.getDefiningOp();
  if (!op)
    return expr;

  auto getOperandThatMatchesFactor = [&factor](Value lhs, Value rhs) {
    if (lhs == factor && lhs != rhs)
      return lhs;
    if (rhs == factor && lhs != rhs)
      return rhs;
    return Value();
  };

  return TypeSwitch<Operation *, ValueOr<Multiplier>>(op)
      .Case<arith::AddIOp, arith::SubIOp>([&](auto binOp) {
        auto computeResult =
            [&](ValueOr<Multiplier> lhs,
                ValueOr<Multiplier> rhs) -> ValueOr<Multiplier> {
          bool lhsIsMul = lhs.is<Multiplier>(), rhsIsMul = rhs.is<Multiplier>();

          // If the LHS (or RHS) subtree passed up a multiplier and the other
          // operand is not a match (of the factor), propagate the multiplier
          // up.
          if (lhsIsMul && !rhsIsMul && rhs.get<Value>() != factor)
            return lhs;
          if (rhsIsMul && !lhsIsMul && lhs.get<Value>() != factor)
            return rhs;

          // Otherwise, if neither the LHS nor the RHS subtrees passed up a
          // multiplier, and one of the operands is a match, the multiplier is
          // one.
          if (!lhsIsMul && !rhsIsMul &&
              getOperandThatMatchesFactor(lhs.get<Value>(), rhs.get<Value>()) !=
                  nullptr)
            return Multiplier::one(expr.getContext());
          return Value();
        };

        LLVM_DEBUG(llvm::dbgs() << "expr: " << expr << "\n");

        return visitBinaryOp(binOp, factor, solver, getMultiplier,
                             computeResult);
      })
      .Case<arith::MulIOp>([&](auto mulOp) {
        auto computeResult =
            [&](ValueOr<Multiplier> lhs,
                ValueOr<Multiplier> rhs) -> ValueOr<Multiplier> {
          bool lhsIsMul = lhs.is<Multiplier>(), rhsIsMul = rhs.is<Multiplier>();

          // If neither the LHS nor the RHS subtree passed up a multiplier,
          // attempt to find it in this multiplication expression.
          if (!lhsIsMul && !rhsIsMul) {
            Value lhsVal = lhs.get<Value>(), rhsVal = rhs.get<Value>();
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
            if (rhs.isEqualTo(1, solver))
              return lhs;
            if (lhs.isEqualTo(1, solver) && rhs.get<Value>() != factor)
              return Multiplier(rhs.get<Value>());
            return Value();
          }
          if (rhsIsMul && !lhsIsMul) {
            if (lhs.isEqualTo(1, solver))
              return rhs;
            if (rhs.isEqualTo(1, solver) && lhs.get<Value>() != factor)
              return Multiplier(lhs.get<Value>());
            return Value();
          }

          return Value();
        };

        LLVM_DEBUG(llvm::dbgs() << "expr: " << expr << "\n");

        return visitBinaryOp(mulOp, factor, solver, getMultiplier,
                             computeResult);
      })
      .Case<CastOpInterface>([&](auto) {
        return getMultiplier(op->getOperand(0), factor, solver);
      })
      .Default([&](auto) { return expr; });
}

// Visit a binary operation of type \tparam T. The LHS and RHS operand of the
// binary operation are processed by applying the function \p getOffset. The
// result of the visit is computed via the \p computeResult function.
template <
    typename T, typename ProcessOperandFuncT, typename ComputeResultFuncT,
    typename = std::enable_if_t<
        llvm::is_one_of<T, arith::AddIOp, arith::SubIOp, arith::MulIOp>::value>,
    typename = std::enable_if<std::is_invocable_r_v<
        ValueOr<Offset>, ProcessOperandFuncT, T, const SmallVectorImpl<Value> &,
        DataFlowSolver &>>,
    typename = std::enable_if<std::is_invocable_r_v<
        ValueOr<Offset>, ComputeResultFuncT, ValueOr<Offset>, ValueOr<Offset>>>>
static ValueOr<Offset>
visitBinaryOp(T binOp, const SmallVectorImpl<Value> &loopAndThreadVars,
              DataFlowSolver &solver, ProcessOperandFuncT getOffset,
              ComputeResultFuncT computeResult) {
  // Traverse my subtrees.
  ValueOr<Offset> lhsRes = getOffset(binOp.getLhs(), loopAndThreadVars, solver);
  ValueOr<Offset> rhsRes = getOffset(binOp.getRhs(), loopAndThreadVars, solver);

  // Compute the result to propagate up.
  ValueOr<Offset> res = computeResult(lhsRes, rhsRes);

  LLVM_DEBUG({
    llvm::dbgs() << "lhsRes: " << lhsRes << "\n";
    llvm::dbgs() << "rhsRes: " << rhsRes << "\n";
    llvm::dbgs() << "res = " << res << "\n";
  });

  return res;
}

static ValueOr<Offset>
getOffset(const Value expr, const SmallVectorImpl<Value> &loopAndThreadVars,
          DataFlowSolver &solver) {
  if (llvm::any_of(loopAndThreadVars, [&](Value var) { return expr == var; }))
    return Offset::zero(expr.getContext());

  Operation *op = expr.getDefiningOp();
  if (!op)
    return expr;

  // Determine whether 'expr' uses any of the values in 'values'.
  auto usesAny = [](Value expr, const SmallVectorImpl<Value> &values) {
    return llvm::any_of(values, [&](Value val) {
      if (!expr.getDefiningOp())
        return expr == val;
      return usesValue(expr.getDefiningOp(), val);
    });
  };

  return TypeSwitch<Operation *, ValueOr<Offset>>(op)
      .Case<arith::AddIOp>([&](auto binOp) -> ValueOr<Offset> {
        auto computeResult = [&](ValueOr<Offset> lhs,
                                 ValueOr<Offset> rhs) -> ValueOr<Offset> {
          bool lhsIsOff = lhs.is<Offset>(), rhsIsOff = rhs.is<Offset>();

          // If both the LHS and RHS subtrees passed up an offset, and either is
          // zero, return the other.
          if (lhsIsOff && rhsIsOff) {
            if (lhs.isEqualTo(0, solver))
              return rhs;
            if (rhs.isEqualTo(0, solver))
              return lhs;
          }

          // If the LHS (or RHS) subtree passed up an offset:
          //   - if the offset is zero and the other operand doesn't use a loop
          //     IV or thread ID, then the other operand is the offset
          //   - pass up the offset if the other operand uses a loop IV or
          //     thread ID or is zero.
          if (lhsIsOff && !rhsIsOff) {
            if (lhs.isEqualTo(0, solver) &&
                !usesAny(rhs.get<Value>(), loopAndThreadVars))
              return Offset(rhs.get<Value>());
            if (usesAny(rhs.get<Value>(), loopAndThreadVars) ||
                isZero(rhs.get<Value>(), solver))
              return lhs;
          }
          if (rhsIsOff && !lhsIsOff) {
            if (rhs.isEqualTo(0, solver) &&
                !usesAny(lhs.get<Value>(), loopAndThreadVars))
              return Offset(lhs.get<Value>());
            if (usesAny(lhs.get<Value>(), loopAndThreadVars) ||
                isZero(lhs.get<Value>(), solver))
              return rhs;
          }

          // Otherwise, if neither the LHS nor the RHS subtrees passed up an
          // offset, the offset is the operand that does not use any loop IV or
          // thread id.
          if (!lhsIsOff && !rhsIsOff) {
            if (usesAny(lhs.get<Value>(), loopAndThreadVars) &&
                !usesAny(rhs.get<Value>(), loopAndThreadVars))
              return Offset(rhs.get<Value>());
            if (usesAny(rhs.get<Value>(), loopAndThreadVars) &&
                !usesAny(lhs.get<Value>(), loopAndThreadVars))
              return Offset(lhs.get<Value>());
          }

          return Value();
        };

        LLVM_DEBUG(llvm::dbgs() << "expr: " << expr << "\n");

        return visitBinaryOp(binOp, loopAndThreadVars, solver, getOffset,
                             computeResult);
      })
      .Case<arith::MulIOp>([&](auto mulOp) -> ValueOr<Offset> {
        auto computeResult = [&](ValueOr<Offset> lhs,
                                 ValueOr<Offset> rhs) -> ValueOr<Offset> {
          bool lhsIsOff = lhs.is<Offset>(), rhsIsOff = rhs.is<Offset>();

          // If both the LHS and RHS subtrees passed up an offset, and either is
          // one, return the other.
          if (lhsIsOff && rhsIsOff) {
            if (lhs.isEqualTo(1, solver))
              return rhs;
            if (rhs.isEqualTo(1, solver))
              return lhs;
          }

          // If the LHS (or RHS) subtrees passed up an offset. Return:
          //   - the offset if the other operand has value one
          //   - a new offset with value equal to the other operand if the
          //     offset is one
          if (lhsIsOff && !rhsIsOff) {
            if (lhs.isEqualTo(0, solver) ||
                isEqualTo(rhs.get<Value>(), 1, solver))
              return lhs;
            if (lhs.isEqualTo(1, solver))
              return Offset(rhs.get<Value>());
          }
          if (rhsIsOff && !lhsIsOff) {
            if (rhs.isEqualTo(0, solver) ||
                isEqualTo(lhs.get<Value>(), 1, solver))
              return rhs;
            if (rhs.isEqualTo(1, solver))
              return Offset(lhs.get<Value>());
          }

          // Otherwise, if neither the LHS nor the RHS subtrees passed up an
          // offset, and the multiplication uses a loop IV or thread id, the
          // offset is zero.
          if (!lhsIsOff && !rhsIsOff && usesAny(mulOp, loopAndThreadVars))
            return Offset::zero(expr.getContext());

          return Value();
        };

        LLVM_DEBUG(llvm::dbgs() << "expr: " << expr << "\n");

        return visitBinaryOp(mulOp, loopAndThreadVars, solver, getOffset,
                             computeResult);
      })
      .Case<CastOpInterface>([&](auto castOp) {
        assert(castOp->getOperands().size() == 1 &&
               "Expecting a single operand");
        return getOffset(castOp->getOperand(0), loopAndThreadVars, solver);
      })
      .Default([&](auto) { return expr; });
}

//===----------------------------------------------------------------------===//
// MemoryAccessMatrix
//===----------------------------------------------------------------------===//

[[maybe_unused]] raw_ostream &
mlir::polygeist::operator<<(raw_ostream &os, const MemoryAccessMatrix &matrix) {
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
    if (row != (matrix.getNumRows() - 1))
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

[[maybe_unused]] raw_ostream &
mlir::polygeist::operator<<(raw_ostream &os, const OffsetVector &vector) {
  auto getConstant = [](Value val) -> Optional<int64_t> {
    return TypeSwitch<Operation *, Optional<int64_t>>(val.getDefiningOp())
        .Case<arith::ConstantIndexOp, arith::ConstantIntOp>(
            [](auto op) { return op.value(); })
        .Default([](auto) { return std::nullopt; });
  };

  llvm::interleave(
      vector.getOffsets(), os,
      [&](Value elem) {
        std::optional<int64_t> constant = getConstant(elem);
        if (constant.has_value())
          os << constant;
        else
          os << elem;
      },
      " ");

  return os << "\n";
}

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

MemoryAccessPattern
MemoryAccess::classifyMemoryAccess(DataFlowSolver &solver) const {
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

std::optional<MemoryAccess>
MemoryAccessAnalysis::getMemoryAccess(const MemRefAccess &access) const {
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

  // Try to construct the memory access matrix and offset vector for affine
  // memory operation of interest.
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

  LLVM_DEBUG(llvm::dbgs() << "Candidate op: " << memoryOp << "\n");

  // Try to determine the underlying value(s) of the accessor subscript index
  // operand. The number of underlying values should be equal to the
  // dimensionality of the sycl id used as an index in the accessor subscript
  // operation. Example:
  //   sycl.constructor @id(%id, %i, %j) : (memref<?x!sycl_id_2>, i64, i64)
  //   %sub = sycl.accessor.subscript %acc[%id] ...
  // Here the underlying values for '%id' are {%i, %j}. The number of
  // underlying values matches the dimensionality of the sycl id.
  SmallVector<Value> underlyingVals = getUnderlyingValues(
      accessorSubscriptOp.getOffsetOperandIndex(), accessorSubscriptOp, solver);
  if (underlyingVals.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to resolve underlying value(s) for: "
                            << accessorSubscriptOp.getIndex() << "\n");
    return;
  }

  Value accSubIndex = accessorSubscriptOp.getIndex();
  if (sycl::getDimensions(accSubIndex.getType()) != underlyingVals.size()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Number of underlying values should be equal to "
                  "dimensionality of the id used to index the accessor\n");
    return;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Underlying values:\n";
    for (Value val : underlyingVals)
      llvm::dbgs().indent(2) << val << "\n";
    llvm::dbgs() << "\n";
  });

  // Collect the "get_global_ids" operations (yielding the global thread ids).
  std::vector<sycl::SYCLNDItemGetGlobalIDOp> getGlobalIdOps =
      getOperationsOfType<sycl::SYCLNDItemGetGlobalIDOp>(
          memoryOp->template getParentOfType<FunctionOpInterface>())
          .takeVector();

  // Return the index value of a "get_global_ids" operation.
  auto getIndexValue =
      [&](sycl::SYCLNDItemGetGlobalIDOp &op) -> std::optional<APInt> {
    std::optional<TypedValue<IntegerType>> idx = op.getIndex();
    return idx.has_value() ? getConstIntegerValue(*idx, solver) : APInt();
  };

  // Ensure that all "get_global_ids" have index values known at compile time.
  if (llvm::any_of(getGlobalIdOps, [&](sycl::SYCLNDItemGetGlobalIDOp &op) {
        return !getIndexValue(op).has_value();
      })) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot order 'get_global_ids' operations in "
                               "increasing index value.\n");
    return;
  }

  // Order the "get_global_ids" operations in increasing index value.
  llvm::sort(getGlobalIdOps, [&](sycl::SYCLNDItemGetGlobalIDOp &op1,
                                 sycl::SYCLNDItemGetGlobalIDOp &op2) {
    return getIndexValue(op1)->slt(*getIndexValue(op2));
  });

  // Collect the loops enclosing the memory access operation.
  SetVector<AffineForOp> enclosingLoops =
      getParentsOfType<AffineForOp>(memoryOp->getBlock());

  // Create a vector containing the thread ids and loop induction variables.
  SmallVector<Value, 4> loopAndThreadVars;
  for (sycl::SYCLNDItemGetGlobalIDOp getGlobalIdOp : getGlobalIdOps)
    loopAndThreadVars.emplace_back(getGlobalIdOp.getResult());
  for (AffineForOp loop : llvm::reverse(enclosingLoops))
    loopAndThreadVars.emplace_back(loop.getInductionVar());

  std::optional<MemoryAccessMatrix> matrix = buildAccessMatrix(
      accessorSubscriptOp, loopAndThreadVars, underlyingVals, solver);
  if (!matrix.has_value())
    return;

  std::optional<OffsetVector> offsets =
      buildOffsetVector(*matrix, loopAndThreadVars, underlyingVals, solver);
  if (!offsets.has_value())
    return;

  accessMap[memoryOp] = {std::move(*matrix), std::move(*offsets)};
}

std::optional<MemoryAccessMatrix> MemoryAccessAnalysis::buildAccessMatrix(
    sycl::SYCLAccessorSubscriptOp accessorSubscriptOp,
    const SmallVectorImpl<Value> &loopAndThreadVars,
    const SmallVectorImpl<Value> &underlyingVals, DataFlowSolver &solver) {
  LLVM_DEBUG(llvm::dbgs() << "Computing access matrix\n");

  const Value accSubIndex = accessorSubscriptOp.getIndex();
  assert(sycl::getDimensions(accSubIndex.getType()) == underlyingVals.size() &&
         "Number of underlying values should be equal to dimensionality of the "
         "id used to index the accessor");

  // Construct the memory access matrix. The number of rows is equal to the
  // dimensionality of the sycl.id used by the accessor subscript operation.
  // The number of columns is equal to the number of loops surrounding the
  // memory access plus the number of threads used in the kernel.
  MemoryAccessMatrix accessMatrix(sycl::getDimensions(accSubIndex.getType()),
                                  loopAndThreadVars.size());

  OpBuilder b(accessorSubscriptOp.getContext());
  Value zero = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 0);

  for (size_t row = 0; row < accessMatrix.getNumRows(); ++row) {
    Value val = underlyingVals[row];
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "Analyzing underlying value: " << val << "\n");

    for (size_t col = 0; col < accessMatrix.getNumColumns(); ++col) {
      if (!usesValue(val.getDefiningOp(), loopAndThreadVars[col])) {
        LLVM_DEBUG(llvm::dbgs().indent(2)
                   << "Doesn't use " << loopAndThreadVars[col] << "\n");
        accessMatrix(row, col) = zero;
        continue;
      }

      ValueOr<Multiplier> valOrMultiplier =
          getMultiplier(val, loopAndThreadVars[col], solver);
      if (!valOrMultiplier.is<Multiplier>()) {
        LLVM_DEBUG(llvm::dbgs().indent(2) << "Failed to find multiplier\n");
        return std::nullopt;
      }

      accessMatrix(row, col) = valOrMultiplier.get<Multiplier>();
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "accessMatrix:\n" << accessMatrix << "\n");

  return accessMatrix;
}

std::optional<OffsetVector> MemoryAccessAnalysis::buildOffsetVector(
    const MemoryAccessMatrix &matrix,
    const SmallVectorImpl<Value> &loopAndThreadVars,
    const SmallVectorImpl<Value> &underlyingVals, DataFlowSolver &solver) {
  LLVM_DEBUG(llvm::dbgs() << "Computing offset vector\n");

  OffsetVector offsets(matrix.getNumRows());

  for (size_t row = 0; row < offsets.getNumRows(); ++row) {
    Value val = underlyingVals[row];
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "Analyzing underlying value: " << val << "\n");

    ValueOr<Offset> valOrOffset = getOffset(val, loopAndThreadVars, solver);
    if (!valOrOffset.is<Offset>()) {
      LLVM_DEBUG(llvm::dbgs().indent(2) << "Failed to find offset\n");
      return std::nullopt;
    }

    offsets(row) = valOrOffset.get<Offset>();
  }
  LLVM_DEBUG(llvm::dbgs() << "offset vector:\n" << offsets << "\n");

  return offsets;
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

        // Try to determine the underlying value of the memory pointed to by
        // the memref operand of a load.
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
