//===- MemoryAccessAnalysis.cpp - Memory Access Analysis ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Based on Research article:
// B.Jang, D.Schaa, P.Mistry, et al. 2011. Exploiting memory access patterns to
// improve memory performance in data-parallel architectures.
// IEEE Transactions on Parallel and Distributed Systems 22, 1(2011), 105–118
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Analysis/MemoryAccessAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Polygeist/Analysis/ReachingDefinitionAnalysis.h"
#include "mlir/Dialect/Polygeist/Utils/TransformUtils.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "llvm/ADT/TypeSwitch.h"
#include <numeric>

#define DEBUG_TYPE "memory-access-analysis"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::dataflow;
using namespace mlir::polygeist;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Determine whether an integer range \p range is equal to \p constant.
static bool isEqualTo(IntegerValueRange range, int64_t constant) {
  if (range.isUninitialized())
    return false;

  std::optional<APInt> constVal = range.getValue().getConstantValue();
  if (!constVal)
    return false;

  APInt c(constVal->getBitWidth(), constant, true /*signed*/);
  return (constVal == c);
}

static bool isZero(IntegerValueRange range) { return isEqualTo(range, 0); }

static bool isOne(IntegerValueRange range) { return isEqualTo(range, 1); }

static bool isNegativeOne(IntegerValueRange range) {
  return isEqualTo(range, -1);
}

static bool isStrictlyPositive(IntegerValueRange range) {
  if (range.isUninitialized())
    return false;

  std::optional<APInt> constVal = range.getValue().getConstantValue();
  return (constVal && constVal->isStrictlyPositive());
}

static bool isGreaterThanOne(IntegerValueRange range) {
  if (range.isUninitialized())
    return false;

  std::optional<APInt> constVal = range.getValue().getConstantValue();
  return (constVal && constVal->sgt(1));
}

static bool isSmallerThanNegativeOne(IntegerValueRange range) {
  if (range.isUninitialized())
    return false;

  std::optional<APInt> constVal = range.getValue().getConstantValue();
  return (constVal && constVal->slt(-1));
}

/// Determine whether \p op uses \p val (directly or indirectly).
static bool usesValue(Operation *op, Value val) {
  if (!op || !val)
    return false;

  if (Operation *valOp = val.getDefiningOp()) {
    if (valOp == op)
      return true;
  }

  return llvm::any_of(op->getOperands(), [&](Value operand) {
    if (Operation *operandOp = operand.getDefiningOp())
      return usesValue(operandOp, val);
    return (operand == val);
  });
}

/// Print the given integer value range \p range.
static raw_ostream &printRange(raw_ostream &os,
                               const IntegerValueRange &range) {
  if (range.isUninitialized())
    return os << "?";
  if (auto constVal = range.getValue().getConstantValue())
    return os << *constVal;
  range.print(os);
  return os;
}

namespace {

struct Helper {
  enum class ArithOp { Add, Sub, Mul };

  /// Return a new constant by applying the arithmetic operation \p op to \p lhs
  /// and \p rhs.
  template <typename T,
            typename = std::enable_if_t<llvm::is_one_of<
                T, arith::AddIOp, arith::SubIOp, arith::MulIOp>::value>>
  static IntegerValueRange getConst(const IntegerValueRange &lhs,
                                    const IntegerValueRange &rhs) {
    constexpr unsigned bitWidth = 64;
    const ConstantIntRanges &lhsConst = lhs.getValue();
    const ConstantIntRanges &rhsConst = rhs.getValue();
    const uint64_t lhsUMin = lhsConst.umin().getZExtValue(),
                   lhsUMax = lhsConst.umax().getZExtValue();
    const uint64_t rhsUMin = rhsConst.umin().getZExtValue(),
                   rhsUMax = rhsConst.umax().getZExtValue();
    const int64_t lhsSMin = lhsConst.smin().getSExtValue(),
                  lhsSMax = lhsConst.smax().getSExtValue();
    const int64_t rhsSMin = rhsConst.smin().getSExtValue(),
                  rhsSMax = rhsConst.smax().getSExtValue();

    uint64_t resUMin = 0, resUMax = 0;
    int64_t resSMin = 0, resSMax = 0;
    if constexpr (std::is_same_v<T, arith::AddIOp>) {
      resUMin = lhsUMin + rhsUMin;
      resUMax = lhsUMax + rhsUMax;
      resSMin = lhsSMin + rhsSMin;
      resSMax = lhsSMax + rhsSMax;
    } else if constexpr (std::is_same_v<T, arith::SubIOp>) {
      resUMin = std::max(0l, (int64_t)(lhsUMin - rhsUMax));
      resUMax = std::max(0l, (int64_t)(lhsUMax - rhsUMin));
      resSMin = lhsSMin - rhsSMax;
      resSMax = lhsSMax - rhsSMin;
    } else if constexpr (std::is_same_v<T, arith::MulIOp>) {
      resUMin = lhsUMin * rhsUMin;
      resUMax = lhsUMax * rhsUMax;
      resSMin = std::min(std::min(lhsSMin * rhsSMin, lhsSMin * rhsSMax),
                         rhsSMin * lhsSMax);
      resSMax = std::max(lhsSMin * rhsSMin, lhsSMax * rhsSMax);
    } else
      llvm_unreachable("Unexpected type for template argument 'T'");

    const APInt umin(bitWidth, resUMin), umax(bitWidth, resUMax);
    const APInt smin(bitWidth, resSMin, true), smax(bitWidth, resSMax, true);
    ConstantIntRanges constant(umin, umax, smin, smax);

    return IntegerValueRange(constant);
  }
};

/// Represents a multiplication factor in an affine expression.
/// Example:
///   k1*i + k2
/// Here the multiplier is 'k1'.
class Multiplier : public IntegerValueRange {
  friend raw_ostream &operator<<(raw_ostream &, const Multiplier &);

public:
  Multiplier(IntegerValueRange range) : IntegerValueRange(range) {}

  static Multiplier one() {
    auto one = ConstantIntRanges::constant(APInt(64, 1));
    return Multiplier(IntegerValueRange(one));
  }

  static Multiplier add(const IntegerValueRange &lhs,
                        const IntegerValueRange &rhs) {
    return create<arith::AddIOp>(lhs, rhs);
  }

  static Multiplier sub(const IntegerValueRange &lhs,
                        const IntegerValueRange &rhs) {
    return create<arith::SubIOp>(lhs, rhs);
  }

  static Multiplier mul(const IntegerValueRange &lhs,
                        const IntegerValueRange &rhs) {
    return create<arith::MulIOp>(lhs, rhs);
  }

private:
  template <typename OpTy,
            typename = std::enable_if_t<llvm::is_one_of<
                OpTy, arith::AddIOp, arith::SubIOp, arith::MulIOp>::value>>
  static Multiplier create(const IntegerValueRange &lhs,
                           const IntegerValueRange &rhs) {
    if (lhs.isUninitialized() || rhs.isUninitialized())
      return Multiplier(IntegerValueRange());
    return Multiplier(Helper::getConst<OpTy>(lhs, rhs));
  }
};

[[maybe_unused]] raw_ostream &operator<<(raw_ostream &os,
                                         const Multiplier &mul) {
  os << "Multiplier: ";
  return printRange(os, static_cast<IntegerValueRange>(mul));
}

/// Represents an offset in an affine expression.
/// Example:
///   k1*i + k2
/// Here the offset is 'k2'.
class Offset : public IntegerValueRange {
  friend raw_ostream &operator<<(raw_ostream &, const Offset &);

public:
  Offset(IntegerValueRange range) : IntegerValueRange(range) {}

  static Offset zero() {
    auto zero = ConstantIntRanges::constant(APInt(64, 0));
    return Offset(IntegerValueRange(zero));
  }

  static Offset create(Value val, DataFlowSolver &solver) {
    if (auto *range =
            solver.lookupState<dataflow::IntegerValueRangeLattice>(val))
      return Offset(range->getValue());
    return Offset(IntegerValueRange());
  }

  static Offset add(const IntegerValueRange &lhs,
                    const IntegerValueRange &rhs) {
    return create<arith::AddIOp>(lhs, rhs);
  }

  static Offset sub(const IntegerValueRange &lhs,
                    const IntegerValueRange &rhs) {
    return create<arith::SubIOp>(lhs, rhs);
  }

  static Offset mul(const IntegerValueRange &lhs,
                    const IntegerValueRange &rhs) {
    return create<arith::MulIOp>(lhs, rhs);
  }

private:
  template <typename OpTy,
            typename = std::enable_if_t<llvm::is_one_of<
                OpTy, arith::AddIOp, arith::SubIOp, arith::MulIOp>::value>>
  static Offset create(const IntegerValueRange &lhs,
                       const IntegerValueRange &rhs) {
    if (lhs.isUninitialized() || rhs.isUninitialized())
      return Offset(IntegerValueRange());
    return Offset(Helper::getConst<OpTy>(lhs, rhs));
  }
};

[[maybe_unused]] raw_ostream &operator<<(raw_ostream &os, const Offset &off) {
  os << "Offset: ";
  return printRange(os, static_cast<IntegerValueRange>(off));
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
  LLVM_DEBUG(llvm::dbgs() << "expr: " << binOp << "\n");

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
/// of \p factor, and return the (other) multiplication factor if it does.
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
    return Multiplier::one();

  Operation *op = expr.getDefiningOp();
  if (!op)
    return expr;

  return TypeSwitch<Operation *, ValueOr<Multiplier>>(op)
      .Case<arith::AddIOp, arith::SubIOp>([&](auto binOp) {
        auto computeResult =
            [&](ValueOr<Multiplier> lhs,
                ValueOr<Multiplier> rhs) -> ValueOr<Multiplier> {
          bool lhsIsMul = lhs.is<Multiplier>(), rhsIsMul = rhs.is<Multiplier>();
          constexpr bool isAdd = std::is_same_v<decltype(binOp), arith::AddIOp>;

          // If both the LHS and RHS subtrees passed up a multiplier,
          // add/subtract them.
          if (lhsIsMul && rhsIsMul)
            return isAdd ? Multiplier::add(lhs.get<Multiplier>(),
                                           rhs.get<Multiplier>())
                         : Multiplier::sub(lhs.get<Multiplier>(),
                                           rhs.get<Multiplier>());

          // If the LHS (or RHS) subtree passed up a multiplier propagate it up.
          if (lhsIsMul && !rhsIsMul)
            return lhs;
          if (rhsIsMul && !lhsIsMul)
            return rhs;

          return Value();
        };

        return visitBinaryOp(binOp, factor, solver, getMultiplier,
                             computeResult);
      })
      .Case<arith::MulIOp>([&](auto mulOp) {
        auto computeResult =
            [&](ValueOr<Multiplier> lhs,
                ValueOr<Multiplier> rhs) -> ValueOr<Multiplier> {
          bool lhsIsMul = lhs.is<Multiplier>(), rhsIsMul = rhs.is<Multiplier>();

          // If the LHS (or RHS) subtrees passed up a multiplier attempt to
          // multiply it by the RHS (or LHS) value if its integer range is
          // known.
          if (lhsIsMul && !rhsIsMul) {
            if (rhs.get<Value>() == nullptr)
              return lhs;
            if (auto *range =
                    solver.lookupState<dataflow::IntegerValueRangeLattice>(
                        rhs.get<Value>()))
              return Multiplier::mul(lhs.get<Multiplier>(), range->getValue());
          }
          if (rhsIsMul && !lhsIsMul) {
            if (lhs.get<Value>() == nullptr)
              return rhs;
            if (auto *range =
                    solver.lookupState<dataflow::IntegerValueRangeLattice>(
                        lhs.get<Value>()))
              return Multiplier::mul(range->getValue(), rhs.get<Multiplier>());
          }

          return Value();
        };

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
    return Offset::zero();

  Operation *op = expr.getDefiningOp();
  if (!op)
    return expr;

  if (llvm::none_of(loopAndThreadVars,
                    [&](Value var) { return usesValue(op, var); }))
    return Offset::create(expr, solver);

  return TypeSwitch<Operation *, ValueOr<Offset>>(op)
      .Case<arith::AddIOp, arith::SubIOp>([&](auto binOp) -> ValueOr<Offset> {
        auto computeResult = [&](ValueOr<Offset> lhs,
                                 ValueOr<Offset> rhs) -> ValueOr<Offset> {
          bool lhsIsOff = lhs.is<Offset>(), rhsIsOff = rhs.is<Offset>();
          constexpr bool isAdd = std::is_same_v<decltype(binOp), arith::AddIOp>;

          // If both the LHS and RHS subtrees passed up an offset,
          // add/subtract them.
          if (lhsIsOff && rhsIsOff)
            return isAdd ? Offset::add(lhs.get<Offset>(), rhs.get<Offset>())
                         : Offset::sub(lhs.get<Offset>(), rhs.get<Offset>());

          // If only the LHS (or RHS) subtree passed up an offset add it to
          // the other operand (if its range is known).
          if (lhsIsOff && !rhsIsOff) {
            if (auto *range =
                    solver.lookupState<dataflow::IntegerValueRangeLattice>(
                        rhs.get<Value>()))
              return isAdd ? Offset::add(lhs.get<Offset>(), range->getValue())
                           : Offset::sub(lhs.get<Offset>(), range->getValue());
          }
          if (rhsIsOff && !lhsIsOff) {
            if (auto *range =
                    solver.lookupState<dataflow::IntegerValueRangeLattice>(
                        lhs.get<Value>()))
              return isAdd ? Offset::add(range->getValue(), rhs.get<Offset>())
                           : Offset::sub(range->getValue(), rhs.get<Offset>());
          }

          llvm_unreachable("Should not happen");
        };

        return visitBinaryOp(binOp, loopAndThreadVars, solver, getOffset,
                             computeResult);
      })
      .Case<arith::MulIOp>([&](auto mulOp) -> ValueOr<Offset> {
        auto computeResult = [&](ValueOr<Offset> lhs,
                                 ValueOr<Offset> rhs) -> ValueOr<Offset> {
          bool lhsIsOff = lhs.is<Offset>(), rhsIsOff = rhs.is<Offset>();

          // If both the LHS and RHS subtrees passed up an offset, multiply
          // them.
          if (lhsIsOff && rhsIsOff)
            return Offset::mul(lhs.get<Offset>(), rhs.get<Offset>());

          // If only the LHS (or RHS) subtree passed up an offset multiply it
          // by the other operand (if its range is known).
          if (lhsIsOff && !rhsIsOff) {
            if (auto *range =
                    solver.lookupState<dataflow::IntegerValueRangeLattice>(
                        rhs.get<Value>()))
              return Offset::mul(lhs.get<Offset>(), range->getValue());
          }
          if (rhsIsOff && !lhsIsOff) {
            if (auto *range =
                    solver.lookupState<dataflow::IntegerValueRangeLattice>(
                        lhs.get<Value>()))
              return Offset::mul(range->getValue(), rhs.get<Offset>());
          }

          llvm_unreachable("Should not happen");
        };

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

namespace mlir {
namespace polygeist {

[[maybe_unused]] raw_ostream &operator<<(raw_ostream &os,
                                         const MemoryAccessMatrix &matrix) {
  for (size_t row = 0; row < matrix.getNumRows(); ++row) {
    llvm::interleave(
        matrix.getRow(row), os,
        [&os](const IntegerValueRange &elem) { printRange(os, elem); }, " ");
    if (row != (matrix.getNumRows() - 1))
      os << '\n';
  }
  return os;
}

} // namespace polygeist
} // namespace mlir

MemoryAccessMatrix::MemoryAccessMatrix(size_t nRows, size_t nColumns)
    : nRows(nRows), nColumns(nColumns), data(nRows * nColumns) {}

MemoryAccessMatrix::MemoryAccessMatrix(
    std::initializer_list<std::initializer_list<IntegerValueRange>> initList) {
  assert(initList.size() != 0 && initList.begin()->size() != 0 &&
         "Expecting a non-empty initializer list");
  nRows = initList.size();
  nColumns = initList.begin()->size();

  assert(llvm::all_of(
             initList,
             [this](const std::initializer_list<IntegerValueRange> &initRow) {
               return initRow.size() == nColumns;
             }) &&
         "Rows should all have the same size");

  data.reserve(nRows * nColumns);
  for (const std::initializer_list<IntegerValueRange> &initRow : initList)
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

SmallVector<IntegerValueRange> MemoryAccessMatrix::getRow(size_t row) const {
  assert(row < nRows && "the matrix must contain the given row");
  SmallVector<IntegerValueRange> rowCopy;
  rowCopy.reserve(nColumns);
  for (size_t col = 0; col < nColumns; ++col)
    rowCopy.emplace_back(at(row, col));
  return rowCopy;
}

void MemoryAccessMatrix::setRow(size_t row, ArrayRef<IntegerValueRange> elems) {
  assert(row < nRows && "the matrix must contain the given row");
  assert(elems.size() == nColumns && "elems size must match row length!");
  for (size_t col = 0; col < nColumns; ++col)
    at(row, col) = elems[col];
}

void MemoryAccessMatrix::fillRow(size_t row, IntegerValueRange val) {
  assert(row < nRows && "the matrix must contain the given row");
  for (size_t col = 0; col < nColumns; ++col)
    at(row, col) = val;
}

size_t MemoryAccessMatrix::appendRow() {
  ++nRows;
  data.resize(nRows * nColumns);
  return nRows - 1;
}

size_t MemoryAccessMatrix::appendRow(ArrayRef<IntegerValueRange> elems) {
  size_t row = appendRow();
  setRow(row, elems);
  return row;
}

SmallVector<IntegerValueRange>
MemoryAccessMatrix::getColumn(size_t column) const {
  assert(column < nColumns && "the matrix must contain the given column");
  SmallVector<IntegerValueRange> columnCopy;
  columnCopy.reserve(nRows);
  for (size_t row = 0; row < nRows; ++row)
    columnCopy.emplace_back(at(row, column));
  return columnCopy;
}

void MemoryAccessMatrix::setColumn(size_t col,
                                   ArrayRef<IntegerValueRange> elems) {
  assert(col < nColumns && "the matrix must contain the given column");
  assert(elems.size() == nRows && "elems size must match column length!");
  for (size_t row = 0; row < nRows; ++row)
    at(row, col) = elems[row];
}

void MemoryAccessMatrix::fillColumn(size_t col, IntegerValueRange range) {
  assert(col < nColumns && "the matrix must contain the given column");
  for (size_t row = 0; row < nRows; ++row)
    at(row, col) = range;
}

void MemoryAccessMatrix::fill(IntegerValueRange range) {
  for (size_t row = 0; row < nRows; ++row)
    for (size_t col = 0; col < nColumns; ++col)
      at(row, col) = range;
}

void MemoryAccessMatrix::setZero(size_t row, size_t col) {
  auto zero = ConstantIntRanges::constant(APInt());
  at(row, col) = IntegerValueRange(zero);
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

bool MemoryAccessMatrix::isZero() const {
  return (llvm::all_of(
      data, [](IntegerValueRange range) { return ::isZero(range); }));
}

bool MemoryAccessMatrix::isDiagonal() const {
  if (!isSquare())
    return false;

  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      IntegerValueRange range = at(row, col);
      bool isOnDiagonal = (row == col);

      // All values on the diagonal must be non-zero.
      if (isOnDiagonal && ::isZero(range))
        return false;
      // All other values must be zero.
      if (!isOnDiagonal && !::isZero(range))
        return false;
    }
  }

  return true;
}

bool MemoryAccessMatrix::isIdentity() const {
  if (!isSquare())
    return false;

  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      IntegerValueRange range = at(row, col);
      bool isOnDiagonal = (row == col);

      // All values on the diagonal must be one.
      if (isOnDiagonal && !isOne(range))
        return false;
      // All other values must be zero.
      if (!isOnDiagonal && !::isZero(range))
        return false;
    }
  }

  return true;
}

bool MemoryAccessMatrix::isLowerTriangular() const {
  if (!isSquare())
    return false;

  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      IntegerValueRange range = at(row, col);
      bool isAboveDiagonal = (col > row);

      // All values above the diagonal must be zero.
      if (isAboveDiagonal && !::isZero(range))
        return false;
      // All other values must be non-zero.
      if (!isAboveDiagonal && ::isZero(range))
        return false;
    }
  }

  return true;
}

bool MemoryAccessMatrix::isUpperTriangular() const {
  if (!isSquare())
    return false;

  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      IntegerValueRange range = at(row, col);
      bool isBelowDiagonal = (col < row);

      // All values below the diagonal must be zero.
      if (isBelowDiagonal && !::isZero(range))
        return false;
      // All other values must be non-zero.
      if (!isBelowDiagonal && ::isZero(range))
        return false;
    }
  }

  return true;
}

bool MemoryAccessMatrix::hasLinearAccessPattern() const { return isIdentity(); }

bool MemoryAccessMatrix::hasReverseLinearAccessPattern() const {
  if (!isSquare())
    return false;

  // Ensure the matrix is diagonal with all non-zero elements equal to one
  // except the last one which must be equal to negative one.
  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      IntegerValueRange range = at(row, col);
      bool isOnDiagonal = (col == row);

      if (!isOnDiagonal && !::isZero(range))
        return false;

      if (isOnDiagonal) {
        bool isLastDiagonalElem = (row == nRows - 1 && col == nColumns - 1);
        if (!isLastDiagonalElem && !isOne(range))
          return false;
        if (isLastDiagonalElem && !isNegativeOne(range))
          return false;
      }
    }
  }

  return true;
}

bool MemoryAccessMatrix::hasLinearOverlappedAccessPattern() const {
  if (!isSquare())
    return false;

  // Ensure the matrix is lower triangular with all non-zero elements equal to
  // one.
  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      IntegerValueRange range = at(row, col);
      bool isAboveDiagonal = (col > row);

      if (isAboveDiagonal && !::isZero(range))
        return false;
      if (!isAboveDiagonal && !isOne(range))
        return false;
    }
  }

  return true;
}

bool MemoryAccessMatrix::hasReverseLinearOverlappedAccessPattern() const {
  if (!isSquare())
    return false;

  // Ensure the matrix is lower triangular with all non-zero elements equal to
  // one except the last one on the diagonal which must be equal to negative
  // one.
  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      IntegerValueRange range = at(row, col);
      bool isAboveDiagonal = (col > row);

      if (isAboveDiagonal && !::isZero(range))
        return false;

      if (!isAboveDiagonal) {
        bool isLastDiagonalElem = (row == nRows - 1 && col == nColumns - 1);
        if (!isLastDiagonalElem && !isOne(range))
          return false;
        if (isLastDiagonalElem && !isNegativeOne(range))
          return false;
      }
    }
  }

  return true;
}

bool MemoryAccessMatrix::hasStridedAccessPattern() const {
  if (!isSquare())
    return false;

  // Ensure the matrix is diagonal with all elements equal to one except the
  // last one which must be greater than one.
  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      IntegerValueRange range = at(row, col);
      bool isOnDiagonal = (col == row);

      if (!isOnDiagonal && !::isZero(range))
        return false;

      if (isOnDiagonal) {
        bool isFirstDiagonalElem = (row == 0 && col == 0);
        if (isFirstDiagonalElem && !isOne(range))
          return false;
        bool isLastDiagonalElem = (row == nRows - 1 && col == nColumns - 1);
        if (!isLastDiagonalElem && !isOne(range) && !::isZero(range))
          return false;
        if (isLastDiagonalElem && !isGreaterThanOne(range) && !::isZero(range))
          return false;
      }
    }
  }

  return true;
}

bool MemoryAccessMatrix::hasReverseStridedAccessPattern() const {
  if (!isSquare())
    return false;

  // Ensure the matrix is diagonal with all elements equal to one except the
  // last one which must be smaller than negative one.
  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      IntegerValueRange range = at(row, col);
      bool isOnDiagonal = (col == row);

      if (!isOnDiagonal && !::isZero(range))
        return false;

      if (isOnDiagonal) {
        bool isLastDiagonalElem = (row == nRows - 1 && col == nColumns - 1);
        if (!isLastDiagonalElem && !isOne(range))
          return false;
        if (isLastDiagonalElem && !isSmallerThanNegativeOne(range))
          return false;
      }
    }
  }

  return true;
}

bool MemoryAccessMatrix::hasStridedOverlappedAccessPattern() const {
  if (!isSquare())
    return false;

  // Ensure the matrix is lower triangular with all non-zero elements equal to
  // one except the last one on the diagonal which must be strictly positive.
  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      IntegerValueRange range = at(row, col);
      bool isAboveDiagonal = (col > row);

      if (isAboveDiagonal && !::isZero(range))
        return false;

      if (!isAboveDiagonal) {
        bool isFirstDiagonalElem = (row == 0 && col == 0);
        if (isFirstDiagonalElem && !isOne(range))
          return false;
        bool isLastDiagonalElem = (row == nRows - 1 && col == nColumns - 1);
        if (!isLastDiagonalElem && !isOne(range) && !::isZero(range))
          return false;
        if (isLastDiagonalElem && !isStrictlyPositive(range) &&
            !::isZero(range))
          return false;
      }
    }
  }

  return true;
}

bool MemoryAccessMatrix::hasReverseStridedOverlappedAccessPattern() const {
  if (!isSquare())
    return false;

  // Ensure the matrix is lower triangular with all non-zero elements equal to
  // one except the last one on the diagonal which must be smaller than
  // negative one.
  for (size_t row = 0; row < nRows; ++row) {
    for (size_t col = 0; col < nColumns; ++col) {
      IntegerValueRange range = at(row, col);
      bool isAboveDiagonal = (col > row);

      if (isAboveDiagonal && !::isZero(range))
        return false;

      if (!isAboveDiagonal) {
        bool isLastDiagonalElem = (row == nRows - 1 && col == nColumns - 1);
        if (!isLastDiagonalElem && !isOne(range))
          return false;
        if (isLastDiagonalElem && !isSmallerThanNegativeOne(range))
          return false;
      }
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// OffsetVector
//===----------------------------------------------------------------------===//

namespace mlir {
namespace polygeist {

[[maybe_unused]] raw_ostream &operator<<(raw_ostream &os,
                                         const OffsetVector &vector) {
  llvm::interleave(
      vector.getOffsets(), os,
      [&os](const IntegerValueRange &elem) { printRange(os, elem); }, " ");
  return os << "\n";
}

} // namespace polygeist
} // namespace mlir

OffsetVector::OffsetVector(size_t nRows) : nRows(nRows), offsets(nRows) {}

OffsetVector::OffsetVector(std::initializer_list<IntegerValueRange> initList) {
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

IntegerValueRange OffsetVector::getOffset(size_t row) const { return at(row); }

void OffsetVector::setOffset(size_t row, IntegerValueRange offset) {
  at(row) = offset;
}

void OffsetVector::fill(IntegerValueRange range) {
  for (size_t row = 0; row < nRows; ++row)
    at(row) = range;
}

size_t OffsetVector::append(IntegerValueRange offset) {
  ++nRows;
  offsets.resize(nRows);
  size_t lastRow = nRows - 1;
  setOffset(lastRow, offset);
  return lastRow;
}

bool OffsetVector::isZero() const {
  return llvm::all_of(
      offsets, [&](IntegerValueRange offset) { return ::isZero(offset); });
}

bool OffsetVector::isZeroWithLastElementStrictlyPositive() const {
  size_t lastIndex = nRows - 1;
  for (size_t pos = 0; pos < nRows; ++pos) {
    IntegerValueRange range = at(pos);
    bool isLastIndex = (pos == lastIndex);
    if (!isLastIndex && !::isZero(range))
      return false;
    if (isLastIndex && !isStrictlyPositive(range))
      return false;
  }
  return true;
}

bool OffsetVector::isZeroWithLastElementEqualTo(int k) const {
  size_t lastIndex = nRows - 1;
  for (size_t pos = 0; pos < nRows; ++pos) {
    IntegerValueRange range = at(pos);
    bool isLastIndex = (pos == lastIndex);
    if (!isLastIndex && !::isZero(range))
      return false;

    if (isLastIndex && !isEqualTo(range, k))
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// MemoryAccess
//===----------------------------------------------------------------------===//

namespace mlir {
namespace polygeist {

[[maybe_unused]] raw_ostream &operator<<(raw_ostream &os,
                                         const MemoryAccess &access) {
  os << "--- MemoryAccess ---\n\n";
  os << "AccessMatrix:\n" << access.getAccessMatrix() << "\n";
  os << "OffsetVector:\n" << access.getOffsetVector() << "\n";
  os << "\n------------------\n";
  return os;
}

} // namespace polygeist
} // namespace mlir

MemoryAccessMatrix
MemoryAccess::getIntraThreadAccessMatrix(unsigned numGridDimensions) const {
  assert(numGridDimensions <= matrix.getNumColumns() &&
         "Expecting 'numGridDimensions' to be smaller than the number of "
         "columns in the memory access matrix");
  if (numGridDimensions == 0)
    return matrix;

  std::vector<size_t> v(matrix.getNumColumns() - numGridDimensions);
  std::iota(v.begin(), v.end(), numGridDimensions);
  std::set<size_t> columns(v.begin(), v.end());
  return matrix.getColumns(columns);
}

MemoryAccessMatrix
MemoryAccess::getInterThreadAccessMatrix(unsigned numGridDimensions) const {
  assert(numGridDimensions <= matrix.getNumColumns() &&
         "Expecting 'numGridDimensions' to be smaller than the number of "
         "columns in the memory access matrix");
  if (numGridDimensions == 0)
    return {};

  std::vector<size_t> v(numGridDimensions);
  std::iota(v.begin(), v.end(), 0);
  std::set<size_t> columns(v.begin(), v.end());
  return matrix.getColumns(columns);
}

MemoryAccessPattern MemoryAccess::classify() const {
  return MemoryAccess::classify(matrix, offsets);
}

MemoryAccessPattern MemoryAccess::classify(const MemoryAccessMatrix &matrix,
                                           const OffsetVector &offsets) {
  LLVM_DEBUG({
    llvm::dbgs() << "matrix:\n" << matrix << "\n";
    llvm::dbgs() << "offsets:\n" << offsets << "\n";
  });

  bool isZeroVector = offsets.isZero();

  if (isZeroVector) {
    if (matrix.hasLinearAccessPattern())
      return MemoryAccessPattern::Linear;

    if (matrix.hasLinearOverlappedAccessPattern())
      return MemoryAccessPattern::LinearOverlapped;

    if (matrix.hasStridedAccessPattern())
      return MemoryAccessPattern::Strided;

    if (matrix.hasStridedOverlappedAccessPattern())
      return MemoryAccessPattern::StridedOverlapped;

    return MemoryAccessPattern::Unknown;
  }

  if (matrix.hasLinearAccessPattern() &&
      offsets.isZeroWithLastElementStrictlyPositive())
    return MemoryAccessPattern::LinearShifted;

  if (matrix.hasReverseLinearAccessPattern() &&
      offsets.isZeroWithLastElementEqualTo(matrix.getNumColumns() - 1))
    return MemoryAccessPattern::ReverseLinear;

  if (matrix.hasReverseLinearAccessPattern() &&
      offsets.isZeroWithLastElementStrictlyPositive())
    return MemoryAccessPattern::ReverseLinearShifted;

  if (matrix.hasReverseLinearOverlappedAccessPattern() &&
      offsets.isZeroWithLastElementStrictlyPositive())
    return MemoryAccessPattern::ReverseLinearOverlapped;

  if (matrix.hasReverseStridedAccessPattern() &&
      offsets.isZeroWithLastElementEqualTo(matrix.getNumColumns() - 1))
    return MemoryAccessPattern::ReverseStrided;

  if (matrix.hasStridedAccessPattern() &&
      offsets.isZeroWithLastElementStrictlyPositive())
    return MemoryAccessPattern::StridedShifted;

  if (matrix.hasReverseStridedAccessPattern() &&
      offsets.isZeroWithLastElementStrictlyPositive())
    return MemoryAccessPattern::ReverseStridedShifted;

  if (matrix.hasReverseStridedOverlappedAccessPattern() &&
      offsets.isZeroWithLastElementStrictlyPositive())
    return MemoryAccessPattern::ReverseStridedOverlapped;

  return MemoryAccessPattern::Unknown;
}

//===----------------------------------------------------------------------===//
// MemoryAccessAnalysis
//===----------------------------------------------------------------------===//

MemoryAccessAnalysis::MemoryAccessAnalysis(Operation *op, AnalysisManager &am)
    : operation(op), am(am) {}

bool MemoryAccessAnalysis::isInvalidated(
    const AnalysisManager::PreservedAnalyses &pa) {
  assert(isInitialized && "Analysis not yet initialized");
  return !pa.isPreserved<AliasAnalysis>() ||
         !pa.isPreserved<dataflow::DeadCodeAnalysis>() ||
         !pa.isPreserved<dataflow::SparseConstantPropagation>() ||
         !pa.isPreserved<dataflow::IntegerRangeAnalysis>() ||
         !pa.isPreserved<ReachingDefinitionAnalysis>();
}

std::optional<MemoryAccess>
MemoryAccessAnalysis::getMemoryAccess(const MemRefAccess &access) const {
  assert(isInitialized && "Analysis not yet initialized");
  auto it = accessMap.find(access.opInst);
  if (it == accessMap.end())
    return std::nullopt;
  return it->second;
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
  auto accessorSubscriptOp = dyn_cast_or_null<sycl::SYCLAccessorSubscriptOp>(
      access.memref.getDefiningOp());
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
    for (const Value &val : underlyingVals)
      llvm::dbgs().indent(2) << val << "\n";
    llvm::dbgs() << "\n";
  });

  // Collect the global thread ids used in the function.
  auto funcOp = memoryOp->template getParentOfType<FunctionOpInterface>();
  SmallVector<Value> threadVars = getThreadVector(funcOp, solver);

  // Collect the loops enclosing the memory access operation.
  SetVector<LoopLikeOpInterface> enclosingLoops =
      getParentsOfType<LoopLikeOpInterface>(*memoryOp->getBlock());

  // Create a vector containing the loop induction variables.
  SmallVector<Value> loopIVs;
  for (LoopLikeOpInterface loop : llvm::reverse(enclosingLoops)) {
    std::optional<Value> iv = loop.getSingleInductionVar();
    if (!iv.has_value()) {
      LLVM_DEBUG(llvm::dbgs() << "Loop does not have a single IV\n");
      return;
    }
    loopIVs.emplace_back(*iv);
  }

  std::optional<MemoryAccessMatrix> matrix = buildAccessMatrix(
      accessorSubscriptOp, threadVars, loopIVs, underlyingVals, solver);
  if (!matrix.has_value()) {
    LLVM_DEBUG(llvm::dbgs() << "Unable to build the memory access matrix\n");
    return;
  }

  std::optional<OffsetVector> offsets =
      buildOffsetVector(*matrix, threadVars, loopIVs, underlyingVals, solver);
  if (!offsets.has_value()) {
    LLVM_DEBUG(llvm::dbgs() << "Unable to build the offset vector\n");
    return;
  }

  accessMap[memoryOp] = {std::move(*matrix), std::move(*offsets)};
}

template void MemoryAccessAnalysis::build<AffineLoadOp>(AffineLoadOp,
                                                        DataFlowSolver &);
template void MemoryAccessAnalysis::build<AffineStoreOp>(AffineStoreOp,
                                                         DataFlowSolver &);

std::optional<MemoryAccessMatrix> MemoryAccessAnalysis::buildAccessMatrix(
    sycl::SYCLAccessorSubscriptOp accessorSubscriptOp,
    ArrayRef<Value> threadVars, ArrayRef<Value> loopIVs,
    ArrayRef<Value> underlyingVals, DataFlowSolver &solver) {
  LLVM_DEBUG(llvm::dbgs() << "Computing access matrix\n");

  const Value accSubIndex = accessorSubscriptOp.getIndex();
  assert(sycl::getDimensions(accSubIndex.getType()) == underlyingVals.size() &&
         "Number of underlying values should be equal to dimensionality of the "
         "id used to index the accessor");

  // Construct the memory access matrix. The number of rows is equal to the
  // dimensionality of the sycl.id used by the accessor subscript operation.
  const unsigned numColumns = threadVars.size() + loopIVs.size();
  MemoryAccessMatrix accessMatrix(sycl::getDimensions(accSubIndex.getType()),
                                  numColumns);

  // Fill in the access matrix element at [row,col] with the multiplier in
  // 'expr' corresponding to the variable 'factor'.
  auto createMatrixEntry = [&](Value expr, Value factor, size_t row,
                               size_t col) -> bool {
    assert(row < accessMatrix.getNumRows() && "row out of bound");
    assert(col < accessMatrix.getNumColumns() && "col out of bound");

    if (!usesValue(expr.getDefiningOp(), factor)) {
      LLVM_DEBUG(llvm::dbgs().indent(2) << "Doesn't use " << factor << "\n");
      accessMatrix.setZero(row, col);
      return true;
    }

    if (ValueOr<Multiplier> valOrMul = getMultiplier(expr, factor, solver);
        valOrMul.is<Multiplier>()) {
      accessMatrix(row, col) = valOrMul.get<Multiplier>();
      return true;
    }

    LLVM_DEBUG(llvm::dbgs().indent(2) << "Failed to find multiplier\n");
    return false;
  };

  // Fill in the access matrix.
  for (size_t row = 0; row < accessMatrix.getNumRows(); ++row) {
    Value val = underlyingVals[row];
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "Analyzing underlying value: " << val << "\n");

    for (size_t col = 0; col < numColumns; ++col) {
      // The leftmost 'threadVars.size()' columns correspond to the thread
      // variables, create the corresponding matrix entries.
      if (col < threadVars.size()) {
        Value threadVar = threadVars[col];
        if (threadVar == nullptr) {
          accessMatrix.setZero(row, col);
          continue;
        }

        if (!createMatrixEntry(val, threadVar, row, col))
          return std::nullopt;

        continue;
      }

      // Create the remaining matrix entries corresponding to the loopIVs.
      Value loopIV = loopIVs[col - (numColumns - loopIVs.size())];
      if (!createMatrixEntry(val, loopIV, row, col))
        return std::nullopt;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "accessMatrix:\n" << accessMatrix << "\n");

  return accessMatrix;
}

std::optional<OffsetVector> MemoryAccessAnalysis::buildOffsetVector(
    const MemoryAccessMatrix &matrix, ArrayRef<Value> threadVars,
    ArrayRef<Value> loopIVs, ArrayRef<Value> underlyingVals,
    DataFlowSolver &solver) {
  LLVM_DEBUG(llvm::dbgs() << "Computing offset vector\n");

  SmallVector<Value> loopAndThreadVars(threadVars);
  for (Value loopIV : loopIVs)
    loopAndThreadVars.emplace_back(loopIV);

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

SmallVector<Value>
MemoryAccessAnalysis::getUnderlyingValues(unsigned opIndex, Operation *op,
                                          DataFlowSolver &solver) const {
  std::optional<Definition> def =
      ReachingDefinition::getUniqueDefinition(opIndex, op, solver);
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
            sycl::isPtrOf<sycl::IDType>(
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
