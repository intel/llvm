// Copyright (C) Codeplay Software Limited

//===- Mem2Reg.cpp - MemRef DataFlow Optimization pass ------ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to forward memref stores to loads, thereby
// potentially getting rid of intermediate memref's entirely.
// TODO: In the future, similar techniques could be used to eliminate
// dead memref store's and perform more complex forwarding when support for
// SSA scalars live out of 'affine.for'/'affine.if' statements is available.
//===----------------------------------------------------------------------===//
#include "PassDetails.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include <deque>
#include <iostream>
#include <set>

#include "polygeist/Ops.h"
#include "polygeist/Passes/Utils.h"

#define DEBUG_TYPE "mem2reg"

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;

enum class Match { Exact, Maybe, None };

bool operator<(Value lhs, Value rhs) {
  if (auto lhsBA = lhs.dyn_cast<BlockArgument>()) {
    if (auto rhsBA = rhs.dyn_cast<BlockArgument>()) {
      if (lhsBA.getOwner() != rhsBA.getOwner())
        return lhsBA.getOwner() < rhsBA.getOwner();
      else
        return lhsBA.getArgNumber() < rhsBA.getArgNumber();
    } else {
      return true;
    }
  }
  auto lhsOR = lhs.cast<OpResult>();
  if (auto rhsBA = rhs.dyn_cast<BlockArgument>()) {
    return false;
  } else {
    auto rhsOR = rhs.cast<OpResult>();
    if (lhsOR.getOwner() != rhsOR.getOwner())
      return lhsOR.getOwner() < rhsOR.getOwner();
    else
      return lhsOR.getResultNumber() < rhsOR.getResultNumber();
  }
}
class Offset {
public:
  enum class Type { Value, Index, Affine } type;
  mlir::Value val;
  size_t idx;
  AffineExpr aff;
  SmallVector<Value> dim;
  SmallVector<Value> sym;
  Offset(mlir::Value v) {
    if (auto op = v.getDefiningOp<ConstantIntOp>()) {
      idx = op.value();
      type = Type::Index;
      return;
    }
    if (auto op = v.getDefiningOp<ConstantIndexOp>()) {
      idx = op.value();
      type = Type::Index;
      return;
    }
    val = v;
    type = Type::Value;
  }
  Offset(AffineExpr op, unsigned numDims, unsigned numSymbols,
         mlir::OperandRange vals) {
    if (auto opc = op.dyn_cast<AffineConstantExpr>()) {
      idx = opc.getValue();
      type = Type::Index;
      return;
    }
    if (auto opd = op.dyn_cast<AffineDimExpr>()) {
      val = vals[opd.getPosition()];
      type = Type::Value;
      return;
    }
    if (auto ops = op.dyn_cast<AffineSymbolExpr>()) {
      val = vals[numDims + ops.getPosition()];
      type = Type::Value;
      return;
    }

    aff = op;
    for (unsigned i = 0; i < numDims; i++)
      dim.push_back(vals[i]);

    for (unsigned i = numDims; i < numSymbols; i++)
      sym.push_back(vals[i]);

    type = Type::Affine;
  }
  Match matches(const Offset o) const {
    if (type != o.type)
      return Match::Maybe;
    switch (type) {
    case Type::Affine:
      return (aff == o.aff && dim == o.dim && sym == o.sym) ? Match::Exact
                                                            : Match::Maybe;
    case Type::Value:
      return (val == o.val) ? Match::Exact : Match::Maybe;
    case Type::Index:
      return (idx == o.idx) ? Match::Exact : Match::None;
    }
  }
  bool operator<(const Offset o) const {
    if (type != o.type) {
      return type < o.type;
    } else {
      switch (type) {
      case Offset::Type::Affine:
        if (aff == o.aff) {
          for (auto pair : llvm::zip(dim, o.dim)) {
            if (std::get<0>(pair) != std::get<1>(pair))
              return std::get<0>(pair).getAsOpaquePointer() <
                     std::get<1>(pair).getAsOpaquePointer();
          }
          for (auto pair : llvm::zip(sym, o.sym)) {
            if (std::get<0>(pair) != std::get<1>(pair))
              return std::get<0>(pair).getAsOpaquePointer() <
                     std::get<1>(pair).getAsOpaquePointer();
          }
          return false;
        } else
          return hash_value(aff) < hash_value(o.aff);
      case Offset::Type::Value:
        return val.getAsOpaquePointer() < o.val.getAsOpaquePointer();
      case Offset::Type::Index:
        return idx < o.idx;
      }
    }
  }
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &o, const Offset off) {
  switch (off.type) {
  case Offset::Type::Affine:
    return o << off.aff;
  case Offset::Type::Value:
    return o << off.val;
  case Offset::Type::Index:
    return o << off.idx;
  }
}

namespace {
// The store to load forwarding relies on three conditions:
//
// 1) they need to have mathematically equivalent affine access functions
// (checked after full composition of load/store operands); this implies that
// they access the same single memref element for all iterations of the common
// surrounding loop,
//
// 2) the store op should dominate the load op,
//
// 3) among all op's that satisfy both (1) and (2), the one that postdominates
// all store op's that have a dependence into the load, is provably the last
// writer to the particular memref location being loaded at the load op, and its
// store value can be forwarded to the load. Note that the only dependences
// that are to be considered are those that are satisfied at the block* of the
// innermost common surrounding loop of the <store, load> being considered.
//
// (* A dependence being satisfied at a block: a dependence that is satisfied by
// virtue of the destination operation appearing textually / lexically after
// the source operation within the body of a 'affine.for' operation; thus, a
// dependence is always either satisfied by a loop or by a block).
//
// The above conditions are simple to check, sufficient, and powerful for most
// cases in practice - they are sufficient, but not necessary --- since they
// don't reason about loops that are guaranteed to execute at least once or
// multiple sources to forward from.
//
// TODO: more forwarding can be done when support for
// loop/conditional live-out SSA values is available.
// TODO: do general dead store elimination for memref's. This pass
// currently only eliminates the stores only if no other loads/uses (other
// than dealloc) remain.
//
struct Mem2Reg : public Mem2RegBase<Mem2Reg> {
  void runOnOperation() override;

  // return if changed
  bool forwardStoreToLoad(
      mlir::Value AI, std::vector<Offset> idx,
      SmallVectorImpl<Operation *> &loadOpsToErase,
      DenseMap<Operation *, SmallVector<Operation *>> &capturedAliasing);
};

} // end anonymous namespace

/// Creates a pass to perform optimizations relying on memref dataflow such as
/// store to load forwarding, elimination of dead stores, and dead allocs.
std::unique_ptr<Pass> mlir::polygeist::createMem2RegPass() {
  return std::make_unique<Mem2Reg>();
}

Match matchesIndices(mlir::OperandRange ops, const std::vector<Offset> &idx) {
  if (ops.size() != idx.size())
    return Match::None;
  for (size_t i = 0; i < idx.size(); i++) {
    switch (idx[i].matches(Offset(ops[i]))) {
    case Match::None:
      return Match::None;
    case Match::Maybe:
      return Match::Maybe;
    case Match::Exact:
      break;
    }
  }
  return Match::Exact;
}

Match matchesIndices(AffineMap map, mlir::OperandRange ops,
                     const std::vector<Offset> &idx) {
  auto idxs = map.getResults();
  if (idxs.size() != idx.size())
    return Match::None;
  for (size_t i = 0; i < idx.size(); i++) {
    switch (idx[i].matches(
        Offset(idxs[i], map.getNumDims(), map.getNumSymbols(), ops))) {
    case Match::None:
      return Match::None;
    case Match::Maybe:
      return Match::Maybe;
    case Match::Exact:
      break;
    }
  }
  return Match::Exact;
}

class ValueOrPlaceholder;

static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                            ValueOrPlaceholder &PH);

class ValueOrPlaceholder;

using ReplaceableUse = ValueOrPlaceholder *;

using BlockMap = std::map<Block *, ReplaceableUse>;

class ReplacementHandler {
public:
  Type elType;
  std::vector<std::unique_ptr<ValueOrPlaceholder>> allocs;
  DenseMap<Value, ValueOrPlaceholder *> valueMap;
  DenseMap<Operation *, ValueOrPlaceholder *> opOperands;

  BlockMap valueAtStartOfBlock;
  BlockMap valueAtEndOfBlock;
  SmallVector<std::pair<Value, ValueOrPlaceholder *>> replaceableValues;
  ReplacementHandler(Type elType) : elType(elType) {}

  ValueOrPlaceholder *get(std::nullptr_t);
  ValueOrPlaceholder *get(Value val);
  ValueOrPlaceholder *get(Block *val);
  ValueOrPlaceholder *get(scf::IfOp val, ValueOrPlaceholder *ifVal);
  ValueOrPlaceholder *get(AffineIfOp val, ValueOrPlaceholder *ifVal);
  ValueOrPlaceholder *get(scf::ExecuteRegionOp val);

  void replaceValue(Value orig, Value post);
  void replaceOpWithValue(Operation *orig, ValueOrPlaceholder *ph, Value post) {
    valueMap[post] = ph;
  }
};

class ValueOrPlaceholder {
  ReplacementHandler &metaMap;

public:
  bool overwritten;
  Value val;
  Block *valueAtStart;
  scf::ExecuteRegionOp exOp;
  Operation *ifOp;
  ValueOrPlaceholder(ValueOrPlaceholder &&) = delete;
  ValueOrPlaceholder(const ValueOrPlaceholder &) = delete;
  ValueOrPlaceholder(std::nullptr_t, ReplacementHandler &metaMap)
      : metaMap(metaMap), overwritten(true), val(nullptr),
        valueAtStart(nullptr), exOp(nullptr), ifOp(nullptr) {}
  ValueOrPlaceholder(Value val, ReplacementHandler &metaMap)
      : metaMap(metaMap), overwritten(false), val(val), valueAtStart(nullptr),
        exOp(nullptr), ifOp(nullptr) {
    assert(val);
  }
  ValueOrPlaceholder(Block *valueAtStart, ReplacementHandler &metaMap)
      : metaMap(metaMap), overwritten(false), val(nullptr),
        valueAtStart(valueAtStart), exOp(nullptr), ifOp(nullptr) {
    assert(valueAtStart);
  }
  ValueOrPlaceholder(scf::ExecuteRegionOp exOp, ReplacementHandler &metaMap)
      : metaMap(metaMap), overwritten(false), val(nullptr),
        valueAtStart(nullptr), exOp(exOp), ifOp(nullptr) {
    assert(exOp);
  }
  ValueOrPlaceholder(scf::IfOp ifOp, ReplaceableUse ifLastVal,
                     ReplacementHandler &metaMap)
      : metaMap(metaMap), overwritten(false), val(nullptr),
        valueAtStart(nullptr), exOp(nullptr), ifOp(ifOp) {
    assert(ifOp);
    if (ifLastVal)
      metaMap.opOperands[ifOp] = ifLastVal;
  }
  ValueOrPlaceholder(AffineIfOp ifOp, ReplaceableUse ifLastVal,
                     ReplacementHandler &metaMap)
      : metaMap(metaMap), overwritten(false), val(nullptr),
        valueAtStart(nullptr), exOp(nullptr), ifOp(ifOp) {
    assert(ifOp);
    if (ifLastVal)
      metaMap.opOperands[ifOp] = ifLastVal;
  }
  // Return true if this represents a full expression if all block argsare
  // defined at start Append the list of blocks requiring definition to block.
  bool definedWithArg(SmallPtrSetImpl<Block *> &block) {
    if (val)
      return true;
    if (overwritten)
      return false;
    if (valueAtStart) {
      auto found = metaMap.valueAtStartOfBlock.find(valueAtStart);
      if (found != metaMap.valueAtStartOfBlock.end()) {
        if (found->second->valueAtStart != valueAtStart)
          return found->second->definedWithArg(block);
      }
      block.insert(valueAtStart);
      return true;
    }
    if (ifOp) {
      if (auto sifOp = dyn_cast<scf::IfOp>(ifOp)) {
        auto thenFind = metaMap.valueAtEndOfBlock.find(getThenBlock(sifOp));
        assert(thenFind != metaMap.valueAtEndOfBlock.end());
        assert(thenFind->second);
        if (!thenFind->second->definedWithArg(block))
          return false;

        if (hasElse(sifOp)) {
          auto elseFind = metaMap.valueAtEndOfBlock.find(getElseBlock(sifOp));
          assert(elseFind != metaMap.valueAtEndOfBlock.end());
          assert(elseFind->second);
          if (!elseFind->second->definedWithArg(block))
            return false;
        } else {
          auto opFound = metaMap.opOperands.find(sifOp);
          assert(opFound != metaMap.opOperands.end());
          auto *ifLastValue = opFound->second;
          if (!ifLastValue->definedWithArg(block))
            return false;
        }
        return true;
      } else {
        auto aifOp = cast<AffineIfOp>(ifOp);
        auto thenFind = metaMap.valueAtEndOfBlock.find(getThenBlock(aifOp));
        assert(thenFind != metaMap.valueAtEndOfBlock.end());
        assert(thenFind->second);
        if (!thenFind->second->definedWithArg(block))
          return false;

        if (hasElse(aifOp)) {
          auto elseFind = metaMap.valueAtEndOfBlock.find(getElseBlock(aifOp));
          assert(elseFind != metaMap.valueAtEndOfBlock.end());
          assert(elseFind->second);
          if (!elseFind->second->definedWithArg(block))
            return false;
        } else {
          auto opFound = metaMap.opOperands.find(ifOp);
          assert(opFound != metaMap.opOperands.end());
          auto *ifLastValue = opFound->second;
          if (!ifLastValue->definedWithArg(block))
            return false;
        }
        return true;
      }
    }
    if (exOp) {
      for (auto &B : exOp.getRegion()) {
        if (auto yield = dyn_cast<scf::YieldOp>(B.getTerminator())) {
          auto found = metaMap.valueAtEndOfBlock.find(&B);
          assert(found != metaMap.valueAtEndOfBlock.end());
          assert(found->second);
          if (!found->second->definedWithArg(block))
            return false;
        }
      }
      return true;
    }
    assert(0 && "unhandled");
  }
  Value materialize(bool full = true) {
    if (overwritten)
      return nullptr;
    if (val)
      return val;
    if (valueAtStart) {
      auto found = metaMap.valueAtStartOfBlock.find(valueAtStart);
      if (found != metaMap.valueAtStartOfBlock.end()) {
        if (found->second->valueAtStart != valueAtStart)
          return found->second->materialize(full);
        // valueAtStart = nullptr;
        // return this->val = found->second;
      }
      if (!full)
        return nullptr;
      llvm::errs() << " could not get valueAtStart: " << valueAtStart << "; ";
      if (found != metaMap.valueAtStartOfBlock.end()) {
        llvm::errs() << " map vas: " << *found->second << "\n";
      } else {
        llvm::errs() << " no map\n";
      }
      Block *blk = valueAtStart;
      blk->dump();
      assert(0 && "no null");
    }
    if (exOp)
      return materializeEx(full);
    if (ifOp)
      return materializeIf(full);
    assert(0 && "");
  }

  Value materializeEx(bool full = true) {
    assert(exOp);

    SmallVector<scf::YieldOp> yields;
    SmallVector<Value> values;
    std::set<size_t> equivalent;
    for (size_t i = 0, num = exOp.getNumResults(); i < num; i++)
      equivalent.insert(i);

    // Force early materialization in case any materializations
    // overwrite subsequent ones.
    for (auto &B : exOp.getRegion()) {
      if (auto yield = dyn_cast<scf::YieldOp>(B.getTerminator())) {
        auto found = metaMap.valueAtEndOfBlock.find(&B);
        assert(found != metaMap.valueAtEndOfBlock.end());
        assert(found->second);
        Value post = found->second->materialize(full);
        if (found->second->overwritten) {
          this->overwritten = true;
          this->exOp = nullptr;
          return nullptr;
        }
        if (!post) {
          if (full) {
            this->overwritten = true;
            this->exOp = nullptr;
          }
          return nullptr;
        }
      }
    }
    for (auto &B : exOp.getRegion()) {
      if (auto yield = dyn_cast<scf::YieldOp>(B.getTerminator())) {
        auto found = metaMap.valueAtEndOfBlock.find(&B);
        assert(found != metaMap.valueAtEndOfBlock.end());
        assert(found->second);
        Value post = found->second->materialize(full);
        assert(post);
        yields.push_back(yield);
        values.push_back(post);
        for (auto pair : llvm::enumerate(yield.getOperands()))
          if (pair.value() != post)
            equivalent.erase(pair.index());
      }
    }

    // Must contain only region invariant results.
    bool allSame = true;
    for (auto v : values)
      allSame &= v == values[0];

    // If all all paths are the same, and the value is not defined within the
    // execute region, simply return that single value, rather than creating a
    // new return.
    if (allSame) {
      if (values[0].getDefiningOp() &&
          !exOp->isAncestor(values[0].getDefiningOp())) {
        return values[0];
      }
      if (auto ba = values[0].dyn_cast<BlockArgument>())
        if (!exOp->isAncestor(ba.getOwner()->getParentOp())) {
          return values[0];
        }
    }
    // If there's an equivalent return, don't create a new return and instead
    // use that result.
    if (equivalent.size() > 0) {
      return exOp.getResult(*equivalent.begin());
    }

    OpBuilder B(exOp.getContext());
    B.setInsertionPoint(exOp);
    SmallVector<mlir::Type, 4> tys(exOp.getResultTypes().begin(),
                                   exOp.getResultTypes().end());
    tys.push_back(metaMap.elType);
    auto nextEx = B.create<mlir::scf::ExecuteRegionOp>(exOp.getLoc(), tys);

    nextEx.getRegion().takeBody(exOp.getRegion());
    for (auto pair : llvm::zip(yields, values)) {
      SmallVector<Value, 4> vals = std::get<0>(pair).getOperands();
      vals.push_back(std::get<1>(pair));
      std::get<0>(pair)->setOperands(vals);
    }

    SmallVector<mlir::Value, 3> resvals = nextEx.getResults();
    this->val = resvals.back();
    resvals.pop_back();

    for (auto pair : llvm::zip(exOp.getResults(), resvals)) {
      metaMap.replaceValue(std::get<0>(pair), std::get<1>(pair));
    }
    metaMap.replaceOpWithValue(exOp, this, this->val);
    // StoringOperations.erase(exOp);
    // StoringOperations.insert(nextEx);
    exOp.erase();
    this->exOp = nullptr;
    return this->val;
  }

  Value materializeIf(bool full = true) {
    if (auto sop = dyn_cast<scf::IfOp>(ifOp))
      return materializeIf<scf::IfOp, scf::YieldOp>(sop, full);
    return materializeIf<AffineIfOp, AffineYieldOp>(cast<AffineIfOp>(ifOp),
                                                    full);
  }

  template <typename IfType, typename YieldType>
  Value materializeIf(IfType ifOp, bool full = true) {
    auto thenFind = metaMap.valueAtEndOfBlock.find(getThenBlock(ifOp));
    assert(thenFind != metaMap.valueAtEndOfBlock.end());
    assert(thenFind->second);
    Value thenVal = thenFind->second->materialize(full);
    if (thenFind->second->overwritten) {
      this->overwritten = true;
      metaMap.opOperands.erase(ifOp);
      this->ifOp = nullptr;
      return nullptr;
    }
    if (!thenVal) {
      if (full) {
        this->overwritten = true;
        metaMap.opOperands.erase(ifOp);
        this->ifOp = nullptr;
      }
      return nullptr;
    }
    Value elseVal;

    if (hasElse(ifOp)) {
      auto elseFind = metaMap.valueAtEndOfBlock.find(getElseBlock(ifOp));
      assert(elseFind != metaMap.valueAtEndOfBlock.end());
      assert(elseFind->second);
      elseVal = elseFind->second->materialize(full);
      if (elseFind->second->overwritten) {
        this->overwritten = true;
        metaMap.opOperands.erase(ifOp);
        this->ifOp = nullptr;
        return nullptr;
      }
    } else {
      auto opFound = metaMap.opOperands.find(ifOp);
      assert(opFound != metaMap.opOperands.end());
      auto ifLastValue = opFound->second;
      elseVal = ifLastValue->materialize(full);
      if (ifLastValue->overwritten) {
        this->overwritten = true;
        metaMap.opOperands.erase(ifOp);
        this->ifOp = nullptr;
        return nullptr;
      }
    }

    if (!elseVal) {
      if (full) {
        this->overwritten = true;
        metaMap.opOperands.erase(ifOp);
        this->ifOp = nullptr;
      }
      return nullptr;
    }

    // Rematerialize thenVal in case it was overwritten by the elseRegion
    //  materialization.
    thenVal = thenFind->second->materialize(full);

    if (thenVal == elseVal) {
      return thenVal;
    }

    if (hasElse(ifOp)) {
      for (auto tup : llvm::reverse(
               llvm::zip(ifOp.getResults(), getThenYield(ifOp).getOperands(),
                         getElseYield(ifOp).getOperands()))) {
        if (std::get<1>(tup) == thenVal && std::get<2>(tup) == elseVal) {
          return std::get<0>(tup);
        }
      }
    }

    OpBuilder B(ifOp.getContext());
    B.setInsertionPoint(ifOp);
    SmallVector<mlir::Type, 4> tys(ifOp.getResultTypes().begin(),
                                   ifOp.getResultTypes().end());
    tys.push_back(thenVal.getType());
    auto nextIf = cloneWithoutResults(ifOp, B, {}, tys);

    SmallVector<mlir::Value, 4> thenVals = getThenYield(ifOp).getOperands();
    thenVals.push_back(thenVal);
    getThenRegion(nextIf).takeBody(getThenRegion(ifOp));
    getThenYield(nextIf)->setOperands(thenVals);

    if (hasElse(ifOp)) {
      getElseRegion(nextIf).getBlocks().clear();
      SmallVector<mlir::Value, 4> elseVals = getElseYield(ifOp).getOperands();
      elseVals.push_back(elseVal);
      getElseRegion(nextIf).takeBody(getElseRegion(ifOp));
      getElseYield(nextIf)->setOperands(elseVals);
    } else {
      B.setInsertionPoint(&getElseRegion(nextIf).back(),
                          getElseRegion(nextIf).back().begin());
      SmallVector<mlir::Value, 4> elseVals = {elseVal};
      B.create<YieldType>(ifOp.getLoc(), elseVals);
    }

    SmallVector<mlir::Value, 3> resvals = nextIf.getResults();
    this->val = resvals.back();
    resvals.pop_back();

    for (auto pair : llvm::zip(ifOp.getResults(), resvals)) {
      metaMap.replaceValue(std::get<0>(pair), std::get<1>(pair));
    }
    metaMap.replaceOpWithValue(ifOp, this, this->val);

    // StoringOperations.erase(ifOp);
    // StoringOperations.insert(nextIf);
    ifOp.erase();
    metaMap.opOperands.erase(ifOp);
    this->ifOp = nullptr;
    return this->val;
  }
};

static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                            ValueOrPlaceholder &PH) {
  if (PH.overwritten)
    return os << "<overwritten>";
  if (PH.val)
    return os << "val:" << PH.val;
  if (PH.valueAtStart) {
    PH.valueAtStart->print(os);
    return os;
    ;
  }
  if (PH.ifOp)
    return os << "ifOp:" << *PH.ifOp;
  if (PH.exOp)
    return os << "exOp:" << PH.exOp;
  return os;
}
ValueOrPlaceholder *ReplacementHandler::get(std::nullptr_t val) {
  ValueOrPlaceholder *PH;
  allocs.emplace_back(PH = new ValueOrPlaceholder(val, *this));
  return PH;
}
ValueOrPlaceholder *ReplacementHandler::get(Value val) {
  auto found = valueMap.find(val);
  if (found != valueMap.end())
    return found->second;
  ValueOrPlaceholder *PH;
  allocs.emplace_back(PH = new ValueOrPlaceholder(val, *this));
  valueMap.insert(std::make_pair(val, PH));
  return PH;
}
ValueOrPlaceholder *ReplacementHandler::get(Block *val) {
  ValueOrPlaceholder *PH;
  allocs.emplace_back(PH = new ValueOrPlaceholder(val, *this));
  return PH;
}
ValueOrPlaceholder *ReplacementHandler::get(scf::IfOp val,
                                            ValueOrPlaceholder *ifVal) {
  ValueOrPlaceholder *PH;
  allocs.emplace_back(PH = new ValueOrPlaceholder(val, ifVal, *this));
  return PH;
}
ValueOrPlaceholder *ReplacementHandler::get(AffineIfOp val,
                                            ValueOrPlaceholder *ifVal) {
  ValueOrPlaceholder *PH;
  allocs.emplace_back(PH = new ValueOrPlaceholder(val, ifVal, *this));
  return PH;
}
ValueOrPlaceholder *ReplacementHandler::get(scf::ExecuteRegionOp val) {
  ValueOrPlaceholder *PH;
  allocs.emplace_back(PH = new ValueOrPlaceholder(val, *this));
  return PH;
}
void ReplacementHandler::replaceValue(Value orig, Value post) {
  orig.replaceAllUsesWith(post);
  auto found = valueMap.find(orig);
  if (found != valueMap.end()) {
    auto pfound = valueMap.find(post);
    if (pfound == valueMap.end()) {
      found->second->val = post;
    } else {
      for (auto &pair : valueAtStartOfBlock)
        if (pair.second == found->second)
          pair.second = pfound->second;
      for (auto &pair : valueAtEndOfBlock)
        if (pair.second == found->second)
          pair.second = pfound->second;
      for (auto &pair : replaceableValues)
        if (pair.second == found->second)
          pair.second = pfound->second;
      for (auto &pair : opOperands)
        if (pair.second == found->second)
          pair.second = pfound->second;
      valueMap.erase(found);
    }
  }
}

struct Analyzer {

  const std::set<Block *> &Good;
  const std::set<Block *> &Bad;
  const std::set<Block *> &Other;
  std::set<Block *> Legal;
  std::set<Block *> Illegal;
  size_t depth;
  Analyzer(const std::set<Block *> &Good, const std::set<Block *> &Bad,
           const std::set<Block *> &Other, std::set<Block *> Legal,
           std::set<Block *> Illegal, size_t depth = 0)
      : Good(Good), Bad(Bad), Other(Other), Legal(Legal), Illegal(Illegal),
        depth(depth) {}

  void analyze() {
    while (1) {
      std::deque<Block *> todo(Other.begin(), Other.end());
      todo.insert(todo.end(), Good.begin(), Good.end());
      while (todo.size()) {
        auto *block = todo.front();
        todo.pop_front();
        if (Legal.count(block) || Illegal.count(block))
          continue;
        bool currentlyLegal = !block->hasNoPredecessors();
        for (auto *pred : block->getPredecessors()) {
          if (Bad.count(pred)) {
            assert(!Legal.count(block));
            Illegal.insert(block);
            currentlyLegal = false;
            for (auto *succ : block->getSuccessors()) {
              todo.push_back(succ);
            }
            break;
          } else if (Good.count(pred) || Legal.count(pred)) {
            continue;
          } else if (Illegal.count(pred)) {
            Illegal.insert(block);
            currentlyLegal = false;
            for (auto *succ : block->getSuccessors()) {
              todo.push_back(succ);
            }
            break;
          } else {
            /*
            if (!Other.count(pred)) {
              pred->getParentOp()->dump();
              pred->dump();
              llvm::errs() << " - pred ptr: " << pred << "\n";
            }
            assert(Other.count(pred));
            */
            currentlyLegal = false;
            break;
          }
        }
        if (currentlyLegal) {
          Legal.insert(block);
          assert(!Illegal.count(block));
          for (auto *succ : block->getSuccessors()) {
            todo.push_back(succ);
          }
        }
      }
      bool changed = false;
      for (auto *O : Other) {
        if (Legal.count(O) || Illegal.count(O))
          continue;
        Analyzer AssumeLegal(Good, Bad, Other, Legal, Illegal, depth + 1);
        AssumeLegal.Legal.insert(O);
        AssumeLegal.analyze();
        bool currentlyLegal = true;
        for (auto *pred : O->getPredecessors()) {
          if (!AssumeLegal.Legal.count(pred) && !AssumeLegal.Good.count(pred)) {
            currentlyLegal = false;
            break;
          }
        }
        if (currentlyLegal) {
          Legal.insert(AssumeLegal.Legal.begin(), AssumeLegal.Legal.end());
          assert(!Illegal.count(O));
          changed = true;
          break;
        } else {
          Illegal.insert(O);
          assert(!Legal.count(O));
        }
      }
      if (!changed)
        break;
    }
  }
};

// Remove block arguments if possible
void removeRedundantBlockArgs(
    Value AI, Type elType,
    std::map<Block *, BlockArgument> &blocksWithAddedArgs) {
  std::deque<Block *> todo;
  for (auto &p : blocksWithAddedArgs)
    todo.push_back(p.first);

  while (todo.size()) {
    auto *block = todo.front();
    todo.pop_front();
    if (!blocksWithAddedArgs.count(block))
      continue;

    BlockArgument blockArg = blocksWithAddedArgs.find(block)->second;
    if (blockArg.getOwner() != block)
      continue;

    assert(blockArg.getOwner() == block);

    mlir::Value val = nullptr;
    bool legal = true;

    SetVector<Block *> prepred(block->getPredecessors().begin(),
                               block->getPredecessors().end());
    for (auto *pred : prepred) {
      mlir::Value pval = nullptr;

      if (auto op = dyn_cast<cf::BranchOp>(pred->getTerminator())) {
        pval = op.getOperands()[blockArg.getArgNumber()];
        if (pval.getType() != elType) {
          pval.getDefiningOp()->getParentRegion()->getParentOp()->dump();
          llvm::errs() << pval << " - " << AI << "\n";
        }
        assert(pval.getType() == elType);
        if (pval == blockArg)
          pval = nullptr;
      } else if (auto op = dyn_cast<cf::CondBranchOp>(pred->getTerminator())) {
        if (op.getTrueDest() == block) {
          if (blockArg.getArgNumber() >= op.getTrueOperands().size()) {
            block->dump();
            llvm::errs() << op << " ba: " << blockArg.getArgNumber() << "\n";
          }
          assert(blockArg.getArgNumber() < op.getTrueOperands().size());
          pval = op.getTrueOperands()[blockArg.getArgNumber()];
          assert(pval.getType() == elType);
          if (pval == blockArg)
            pval = nullptr;
        }
        if (op.getFalseDest() == block) {
          assert(blockArg.getArgNumber() < op.getFalseOperands().size());
          auto pval2 = op.getFalseOperands()[blockArg.getArgNumber()];
          assert(pval2.getType() == elType);
          if (pval2 != blockArg) {
            if (pval == nullptr) {
              pval = pval2;
            } else if (pval != pval2) {
              legal = false;
              break;
            }
          }
          if (pval == blockArg)
            pval = nullptr;
        }
      } else if (auto op = dyn_cast<cf::SwitchOp>(pred->getTerminator())) {
        mlir::OpBuilder subbuilder(op.getOperation());
        if (op.getDefaultDestination() == block) {
          pval = op.getDefaultOperands()[blockArg.getArgNumber()];
          if (pval == blockArg)
            pval = nullptr;
        }
        for (auto pair : llvm::enumerate(op.getCaseDestinations())) {
          if (pair.value() == block) {
            auto pval2 =
                op.getCaseOperands(pair.index())[blockArg.getArgNumber()];
            if (pval2 != blockArg) {
              if (pval == nullptr)
                pval = pval2;
              else if (pval != pval2) {
                legal = false;
                break;
              }
            }
          }
        }
        if (legal == false)
          break;
      } else {
        llvm::errs() << *pred->getParent()->getParentOp() << "\n";
        pred->dump();
        block->dump();
        assert(0 && "unknown branch");
      }

      assert(pval != blockArg);
      if (val == nullptr) {
        val = pval;
        if (pval)
          assert(val.getType() == elType);
      } else {
        if (pval != nullptr && val != pval) {
          legal = false;
          break;
        }
      }
    }
    if (legal)
      assert(val || block->hasNoPredecessors());

    bool used = false;
    for (auto *U : blockArg.getUsers()) {

      if (auto op = dyn_cast<cf::BranchOp>(U)) {
        size_t i = 0;
        for (auto V : op.getOperands()) {
          if (V == blockArg &&
              !(i == blockArg.getArgNumber() && op.getDest() == block)) {
            used = true;
            break;
          }
        }
        if (used)
          break;
      } else if (auto op = dyn_cast<cf::CondBranchOp>(U)) {
        size_t i = 0;
        for (auto V : op.getTrueOperands()) {
          if (V == blockArg &&
              !(i == blockArg.getArgNumber() && op.getTrueDest() == block)) {
            used = true;
            break;
          }
        }
        if (used)
          break;
        i = 0;
        for (auto V : op.getFalseOperands()) {
          if (V == blockArg &&
              !(i == blockArg.getArgNumber() && op.getFalseDest() == block)) {
            used = true;
            break;
          }
        }
      } else
        used = true;
    }
    if (!used) {
      legal = true;
    }

    if (legal) {
      for (auto *U : blockArg.getUsers()) {
        if (auto *block = U->getBlock()) {
          todo.push_back(block);
          for (auto *succ : block->getSuccessors())
            todo.push_back(succ);
        }
      }
      if (val != nullptr) {
        if (blockArg.getType() != val.getType()) {
          block->dump();
          llvm::errs() << " AI: " << AI << "\n";
          llvm::errs() << blockArg << " val " << val << "\n";
        }
        assert(blockArg.getType() == val.getType());
        blockArg.replaceAllUsesWith(val);
      } else {
      }

      SetVector<Block *> prepred(block->getPredecessors().begin(),
                                 block->getPredecessors().end());
      for (auto *pred : prepred) {
        if (auto op = dyn_cast<cf::BranchOp>(pred->getTerminator())) {
          mlir::OpBuilder subbuilder(op.getOperation());
          std::vector<Value> args(op.getOperands().begin(),
                                  op.getOperands().end());
          args.erase(args.begin() + blockArg.getArgNumber());
          assert(args.size() == op.getOperands().size() - 1);
          subbuilder.create<cf::BranchOp>(op.getLoc(), op.getDest(), args);
          op.erase();
        } else if (auto op =
                       dyn_cast<cf::CondBranchOp>(pred->getTerminator())) {

          mlir::OpBuilder subbuilder(op.getOperation());
          std::vector<Value> trueargs(op.getTrueOperands().begin(),
                                      op.getTrueOperands().end());
          std::vector<Value> falseargs(op.getFalseOperands().begin(),
                                       op.getFalseOperands().end());
          if (op.getTrueDest() == block) {
            trueargs.erase(trueargs.begin() + blockArg.getArgNumber());
          }
          if (op.getFalseDest() == block) {
            falseargs.erase(falseargs.begin() + blockArg.getArgNumber());
          }
          assert(trueargs.size() < op.getTrueOperands().size() ||
                 falseargs.size() < op.getFalseOperands().size());
          subbuilder.create<cf::CondBranchOp>(op.getLoc(), op.getCondition(),
                                              op.getTrueDest(), trueargs,
                                              op.getFalseDest(), falseargs);
          op.erase();
        } else if (auto op = dyn_cast<cf::SwitchOp>(pred->getTerminator())) {
          mlir::OpBuilder builder(op.getOperation());
          SmallVector<Value> defaultOps(op.getDefaultOperands().begin(),
                                        op.getDefaultOperands().end());
          if (op.getDefaultDestination() == block)
            defaultOps.erase(defaultOps.begin() + blockArg.getArgNumber());

          SmallVector<SmallVector<Value>> cases;
          SmallVector<ValueRange> vrange;
          for (auto pair : llvm::enumerate(op.getCaseDestinations())) {
            cases.emplace_back(op.getCaseOperands(pair.index()));
            if (pair.value() == block) {
              cases.back().erase(cases.back().begin() +
                                 blockArg.getArgNumber());
            }
          }
          for (auto &c : cases) {
            vrange.push_back(c);
          }
          builder.create<cf::SwitchOp>(
              op.getLoc(), op.getFlag(), op.getDefaultDestination(), defaultOps,
              op.getCaseValuesAttr(), op.getCaseDestinations(), vrange);
          op.erase();
        }
      }
      block->eraseArgument(blockArg.getArgNumber());
      blocksWithAddedArgs.erase(block);
    }
  }
}

std::set<std::string> NonCapturingFunctions = {
    "free",         "printf",        "fprintf", "scanf",     "fscanf",
    "gettimeofday", "clock_gettime", "getenv",  "strrchr",   "strlen",
    "sprintf",      "sscanf",        "mkdir",   "fwrite",    "fread",
    "memcpy",       "cudaMemcpy",    "memset",  "cudaMemset"};
// fopen, fclose
std::set<std::string> NoWriteFunctions = {"exit", "__errno_location"};
// This is a straightforward implementation not optimized for speed. Optimize
// if needed.
bool Mem2Reg::forwardStoreToLoad(
    mlir::Value AI, std::vector<Offset> idx,
    SmallVectorImpl<Operation *> &loadOpsToErase,
    DenseMap<Operation *, SmallVector<Operation *>> &capturedAliasing) {
  bool changed = false;
  std::set<mlir::Operation *> loadOps;
  mlir::Type subType = nullptr;
  mlir::Location loc = AI.getLoc();
  std::set<mlir::Operation *> allStoreOps;

  std::deque<std::pair<mlir::Value, /*indexed*/ bool>> list = {{AI, false}};

  SmallPtrSet<Operation *, 4> AliasingStoreOperations;

  LLVM_DEBUG(
      llvm::dbgs() << "Begin forwarding store of " << AI << " to load\n"
                   << *AI.getDefiningOp()->getParentOfType<func::FuncOp>()
                   << "\n");
  bool captured = AI.getDefiningOp<memref::GetGlobalOp>();
  while (list.size()) {
    auto pair = list.front();
    auto val = pair.first;
    auto modified = pair.second;
    list.pop_front();
    for (auto *user : val.getUsers()) {
      if (auto co = dyn_cast<mlir::memref::CastOp>(user)) {
        list.emplace_back((Value)co, modified);
        continue;
      }
      if (auto co = dyn_cast<polygeist::Memref2PointerOp>(user)) {
        list.emplace_back((Value)co, modified);
        continue;
      }
      if (auto co = dyn_cast<polygeist::Pointer2MemrefOp>(user)) {
        list.emplace_back((Value)co, modified);
        continue;
      }
      if (auto co = dyn_cast<polygeist::SubIndexOp>(user)) {
        list.emplace_back((Value)co, true);
        continue;
      }
      // If at the same index, the "hole" property applies
      // and we can go through.
      if (isa<polygeist::BarrierOp>(user)) {
        continue;
      }
      if (auto co = dyn_cast<mlir::LLVM::GEPOp>(user)) {
        list.emplace_back((Value)co, true);
        continue;
      }
      if (auto co = dyn_cast<mlir::LLVM::BitcastOp>(user)) {
        list.emplace_back((Value)co, modified);
        continue;
      }
      if (auto co = dyn_cast<mlir::LLVM::AddrSpaceCastOp>(user)) {
        list.emplace_back((Value)co, modified);
        continue;
      }
      if (auto loadOp = dyn_cast<mlir::memref::LoadOp>(user)) {
        if (!modified &&
            matchesIndices(loadOp.getIndices(), idx) == Match::Exact) {
          subType = loadOp.getType();
          loadOps.insert(loadOp);
          LLVM_DEBUG(llvm::dbgs() << "Matching Load: " << loadOp << "\n");
        }
        continue;
      }
      if (auto loadOp = dyn_cast<mlir::LLVM::LoadOp>(user)) {
        if (!modified) {
          subType = loadOp.getType();
          loadOps.insert(loadOp);
          LLVM_DEBUG(llvm::dbgs() << "Matching Load: " << loadOp << "\n");
        }
        continue;
      }
      if (auto loadOp = dyn_cast<AffineLoadOp>(user)) {
        if (!modified &&
            matchesIndices(loadOp.getAffineMapAttr().getValue(),
                           loadOp.getMapOperands(), idx) == Match::Exact) {
          subType = loadOp.getType();
          loadOps.insert(loadOp);
          LLVM_DEBUG(llvm::dbgs() << "Matching Load: " << loadOp << "\n");
        }
        continue;
      }
      if (auto storeOp = dyn_cast<mlir::memref::StoreOp>(user)) {
        if (storeOp.getValue() == val)
          captured = true;
        else if (!modified) {
          switch (matchesIndices(storeOp.getIndices(), idx)) {
          case Match::Exact:
            LLVM_DEBUG(llvm::dbgs() << "Matching Store: " << storeOp << "\n");
            allStoreOps.insert(storeOp);
            break;
          case Match::Maybe:
            LLVM_DEBUG(llvm::dbgs()
                       << "Mabye Aliasing Store: " << storeOp << "\n");
            AliasingStoreOperations.insert(storeOp);
            break;
          case Match::None:
            break;
          }
        } else
          AliasingStoreOperations.insert(storeOp);
        continue;
      }
      if (auto storeOp = dyn_cast<LLVM::StoreOp>(user)) {

        if (storeOp.getValue() == val) {
          captured = true;
        } else if (!modified) {
          LLVM_DEBUG(llvm::dbgs() << "Matching Store: " << storeOp << "\n");
          allStoreOps.insert(storeOp);
        } else
          AliasingStoreOperations.insert(storeOp);
        continue;
      }

      if (auto storeOp = dyn_cast<AffineStoreOp>(user)) {
        if (storeOp.getValue() == val) {
          captured = true;
        } else if (!modified) {
          switch (matchesIndices(storeOp.getAffineMapAttr().getValue(),
                                 storeOp.getMapOperands(), idx)) {
          case Match::Exact:
            LLVM_DEBUG(llvm::dbgs() << "Matching Store: " << storeOp << "\n");
            allStoreOps.insert(storeOp);
            break;
          case Match::Maybe:
            LLVM_DEBUG(llvm::dbgs()
                       << "Mabye Aliasing Store: " << storeOp << "\n");
            AliasingStoreOperations.insert(storeOp);
            break;
          case Match::None:
            break;
          }
        } else
          AliasingStoreOperations.insert(storeOp);
        continue;
      }
      if (auto callOp = dyn_cast<func::CallOp>(user)) {
        if (callOp.getCallee() != "free") {
          LLVM_DEBUG(llvm::dbgs() << "Aliasing Store: " << callOp << "\n");
          AliasingStoreOperations.insert(callOp);
          if (!NonCapturingFunctions.count(callOp.getCallee().str()))
            captured = true;
        }
        continue;
      }
      if (auto callOp = dyn_cast<mlir::LLVM::CallOp>(user)) {
        if (!callOp.getCallee() || *callOp.getCallee() != "free") {
          LLVM_DEBUG(llvm::dbgs() << "Aliasing Store: " << callOp << "\n");
          AliasingStoreOperations.insert(callOp);
          if (!callOp.getCallee() ||
              !NonCapturingFunctions.count(callOp.getCallee()->str()))
            captured = true;
        }
        continue;
      }
      if (auto op = dyn_cast<mlir::LLVM::MemsetOp>(user)) {
        if (op.getDst() == val) {
          LLVM_DEBUG(llvm::dbgs() << "Aliasing Store: " << op << "\n");
          AliasingStoreOperations.insert(op);
        }
        continue;
      }
      if (auto op = dyn_cast<mlir::LLVM::MemmoveOp>(user)) {
        if (op.getDst() == val) {
          LLVM_DEBUG(llvm::dbgs() << "Aliasing Store: " << op << "\n");
          AliasingStoreOperations.insert(op);
        }
        continue;
      }
      if (auto op = dyn_cast<mlir::LLVM::MemcpyOp>(user)) {
        if (op.getDst() == val) {
          LLVM_DEBUG(llvm::dbgs() << "Aliasing Store: " << op << "\n");
          AliasingStoreOperations.insert(op);
        }
        continue;
      }
      if (isa<mlir::memref::DeallocOp>(user)) {
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "Unknown, potential store: " << *user << "\n");
      AliasingStoreOperations.insert(user);
      captured = true;
    }
  }

  if (captured) {
    if (capturedAliasing.count(AI.getDefiningOp()) == 0) {
      SmallVector<Operation *> capEffects;
      AI.getDefiningOp()->getParentOp()->walk([&](Operation *op) {
        bool opMayHaveEffect = false;
        if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
          return;
        if (auto callOp = dyn_cast<mlir::LLVM::CallOp>(op)) {
          if (callOp.getCallee() && (*callOp.getCallee() == "printf" ||
                                     *callOp.getCallee() == "free" ||
                                     *callOp.getCallee() == "strlen")) {
            return;
          }
        }
        MemoryEffectOpInterface interface =
            dyn_cast<MemoryEffectOpInterface>(op);
        if (!interface)
          opMayHaveEffect = true;
        if (interface) {
          SmallVector<MemoryEffects::EffectInstance, 1> effects;
          interface.getEffects(effects);

          for (auto effect : effects) {
            // If op causes EffectType on a potentially aliasing location for
            // memOp, mark as having the effect.
            if (isa<MemoryEffects::Write>(effect.getEffect())) {
              if (Value val = effect.getValue()) {
                while (true) {
                  if (auto co = val.getDefiningOp<memref::CastOp>())
                    val = co.getSource();
                  else if (auto co = val.getDefiningOp<polygeist::SubIndexOp>())
                    val = co.getSource();
                  else if (auto co =
                               val.getDefiningOp<polygeist::Memref2PointerOp>())
                    val = co.getSource();
                  else if (auto co =
                               val.getDefiningOp<polygeist::Pointer2MemrefOp>())
                    val = co.getSource();
                  else if (auto co = val.getDefiningOp<LLVM::BitcastOp>())
                    val = co.getArg();
                  else if (auto co = val.getDefiningOp<LLVM::AddrSpaceCastOp>())
                    val = co.getArg();
                  else if (auto co = val.getDefiningOp<LLVM::GEPOp>())
                    val = co.getBase();
                  else
                    break;
                }
                if (val.getDefiningOp<memref::AllocaOp>() ||
                    val.getDefiningOp<memref::AllocOp>() ||
                    val.getDefiningOp<LLVM::AllocaOp>()) {
                  if (val != AI)
                    continue;
                }
                if (auto glob = val.getDefiningOp<memref::GetGlobalOp>()) {
                  if (auto Aglob = AI.getDefiningOp<memref::GetGlobalOp>()) {
                    if (glob.getName() != Aglob.getName())
                      continue;
                  } else
                    continue;
                }
              }
              opMayHaveEffect = true;
              break;
            }
          }
        }
        if (opMayHaveEffect) {
          capEffects.push_back(op);
        }
      });

      capturedAliasing[AI.getDefiningOp()] = capEffects;
    }

    for (auto op : capturedAliasing[AI.getDefiningOp()]) {
      if (allStoreOps.count(op))
        continue;
      LLVM_DEBUG(llvm::dbgs() << "Potential Op ith Effect: " << *op << "\n");
      AliasingStoreOperations.insert(op);
    }
  }

  if (loadOps.size() == 0) {
    return changed;
  }

  assert(AI.getDefiningOp());
  Region *parentAI = AI.getDefiningOp()->getParentRegion();
  assert(parentAI);

  // A list of all regions which contain loads to be replaced.
  SmallPtrSet<Region *, 4> ContainsLoadingOperation;
  {
    SmallVector<Region *> todo;
    for (auto *load : loadOps) {
      todo.push_back(load->getParentRegion());
    }
    while (todo.size()) {
      auto *op = todo.back();
      todo.pop_back();
      if (ContainsLoadingOperation.contains(op))
        continue;
      if (op == parentAI)
        continue;
      ContainsLoadingOperation.insert(op);
      auto *parent = op->getParentRegion();
      assert(parent);
      todo.push_back(parent);
    }
  }

  // List of operations which may store that are not storeops
  SmallPtrSet<Operation *, 4> StoringOperations;
  SmallPtrSet<Block *, 4> StoringBlocks;
  {
    std::deque<Block *> todo;
    for (const auto &pair : allStoreOps) {
      LLVM_DEBUG(llvm::dbgs() << " storing operation: " << *pair << "\n");
      todo.push_back(pair->getBlock());
    }
    for (auto *op : AliasingStoreOperations) {
      StoringOperations.insert(op);
      LLVM_DEBUG(llvm::dbgs()
                 << " aliasing storing operation: " << *op << "\n");
      todo.push_back(op->getBlock());
    }
    while (todo.size()) {
      auto *block = todo.front();
      assert(block);
      todo.pop_front();
      StoringBlocks.insert(block);
      LLVM_DEBUG(llvm::dbgs() << " initial storing block: " << block << "\n");
      if (auto *op = block->getParentOp()) {
        StoringOperations.insert(op);
        if (auto *next = op->getBlock()) {
          StoringBlocks.insert(next);
          LLVM_DEBUG(llvm::dbgs()
                     << " derived storing block: " << next << "\n");
          todo.push_back(next);
        }
      }
    }
  }

  Type elType;
  if (auto MT = AI.getType().dyn_cast<MemRefType>())
    elType = MT.getElementType();
  else
    elType = AI.getType().cast<LLVM::LLVMPointerType>().getElementType();

  ReplacementHandler metaMap(elType);

  // Last value stored in an individual block and the operation which stored it
  BlockMap &valueAtEndOfBlock = metaMap.valueAtEndOfBlock;

  // Last value stored in an individual block and the operation which stored it
  BlockMap &valueAtStartOfBlock = metaMap.valueAtStartOfBlock;

  SmallVector<std::pair<Value, ValueOrPlaceholder *>> &replaceableValues =
      metaMap.replaceableValues;

  std::map<Block *, BlockArgument> blocksWithAddedArgs;

  auto *emptyValue = metaMap.get(nullptr);

  auto replaceValue =
      [&](Value orig, ValueOrPlaceholder *replacement) -> ValueOrPlaceholder * {
    assert(replacement);
    replacement->materialize(/*full*/ false);
    assert(orig.getType() == elType);
    if (replacement->overwritten) {
      loadOps.erase(orig.getDefiningOp());
      return metaMap.get(orig);
    } else if (replacement->val) {
      changed = true;
      assert(orig != replacement->val);
      assert(replacement->val.getType() == elType);
      assert(orig.getType() == replacement->val.getType() &&
             "mismatched load type");
      LLVM_DEBUG(llvm::dbgs() << " replaced " << orig << " with "
                              << replacement->val << "\n");
      metaMap.replaceValue(orig, replacement->val);
      // Record this to erase later.
      loadOpsToErase.push_back(orig.getDefiningOp());
      loadOps.erase(orig.getDefiningOp());
      return replacement;
    } else {
      assert(replacement);
      SmallPtrSet<Block *, 1> requirements;
      if (replacement->definedWithArg(requirements)) {
        replaceableValues.push_back(std::make_pair(orig, replacement));
        assert(replaceableValues.back().second);
      }
      return metaMap.get(orig);
    }
  };

  // Start by setting valueAtEndOfBlock to the last store directly in that block
  // Note that this may miss a store within a region of an operation in that
  // block
  // endRequires denotes whether this value is needed at the end of the block
  // (yield)
  std::function<void(Block &, ValueOrPlaceholder *)> handleBlock =
      [&](Block &block, ValueOrPlaceholder *lastVal) {
        valueAtStartOfBlock.emplace(&block, lastVal);
        SmallVector<Operation *, 10> ops;
        for (auto &a : block) {
          ops.push_back(&a);
        }
        LLVM_DEBUG(llvm::dbgs()
                       << "\nstarting block: lastVal=" << *lastVal << "\n";
                   block.print(llvm::dbgs()); llvm::dbgs() << "\n";);
        for (auto *a : ops) {
          if (StoringOperations.count(a)) {
            // erase a, in case overwritten later in metamap replacement/lookup.
            StoringOperations.erase(a);
            if (auto exOp = dyn_cast<mlir::scf::ExecuteRegionOp>(a)) {
              for (auto &B : exOp.getRegion())
                handleBlock(B, (&B == &exOp.getRegion().front())
                                   ? lastVal
                                   : metaMap.get(&B));
              lastVal = metaMap.get(exOp);
              continue;
            } else if (auto ifOp = dyn_cast<mlir::scf::IfOp>(a)) {
              handleBlock(*ifOp.getThenRegion().begin(), lastVal);
              if (ifOp.getElseRegion().getBlocks().size()) {
                handleBlock(*ifOp.getElseRegion().begin(), lastVal);
                lastVal = metaMap.get(ifOp, emptyValue);
              } else {
                lastVal = metaMap.get(ifOp, lastVal);
              }
              continue;
            } else if (auto ifOp = dyn_cast<mlir::AffineIfOp>(a)) {
              handleBlock(*ifOp.getThenRegion().begin(), lastVal);
              if (ifOp.getElseRegion().getBlocks().size()) {
                handleBlock(*ifOp.getElseRegion().begin(), lastVal);
                lastVal = metaMap.get(ifOp, emptyValue);
              } else {
                lastVal = metaMap.get(ifOp, lastVal);
              }
              continue;
            }
            LLVM_DEBUG(llvm::dbgs() << "erased store due to: " << *a << "\n");

            for (auto &R : a->getRegions())
              for (auto &B : R)
                handleBlock(B,
                            (&B == &R.front()) ? emptyValue : metaMap.get(&B));
            lastVal = emptyValue;
          } else if (loadOps.count(a)) {
            Value loadOp = a->getResult(0);
            lastVal = replaceValue(loadOp, lastVal);
          } else if (auto storeOp = dyn_cast<memref::StoreOp>(a)) {
            if (allStoreOps.count(storeOp)) {
              lastVal = metaMap.get(storeOp.getValueToStore());
            }
          } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(a)) {
            if (allStoreOps.count(storeOp)) {
              lastVal = metaMap.get(storeOp.getValue());
            }
          } else if (auto storeOp = dyn_cast<AffineStoreOp>(a)) {
            if (allStoreOps.count(storeOp)) {
              lastVal = metaMap.get(storeOp.getValueToStore());
            }
          } else {
            // since not storing operation the value at the start of every block
            // is lastVal. However, if lastVal contains no information, use the
            // inductive block arg in case we can do load to load forwarding.
            for (auto &R : a->getRegions())
              for (auto &B : R)
                handleBlock(
                    B, lastVal->overwritten
                           ? ((&B == &R.front()) ? lastVal : metaMap.get(&B))
                           : lastVal);
          }
        }
        LLVM_DEBUG(llvm::dbgs() << " ending block: "; block.print(llvm::dbgs());
                   llvm::dbgs() << " with val:" << *lastVal << "\n";);
        assert(lastVal);
        valueAtEndOfBlock.emplace(&block, lastVal);
      };

  {
    assert(AI.getDefiningOp());
    SmallVector<Block *> todo = {AI.getDefiningOp()->getBlock()};
    SmallPtrSet<Block *, 2> done;
    while (todo.size()) {
      Block *cur = todo.back();
      todo.pop_back();
      if (done.contains(cur))
        continue;
      done.insert(cur);
      if (cur == AI.getDefiningOp()->getBlock())
        handleBlock(*cur, emptyValue);
      else
        handleBlock(*cur, metaMap.get(cur));
      for (auto *B : cur->getSuccessors())
        todo.push_back(B);
    }
  }

  if (replaceableValues.size() == 0)
    return changed;

  // Preserve only values which could inductively be replaced
  // by injecting relevant block arguments
  SmallPtrSet<Block *, 1> PotentiallyHelpfulArgs;
  for (auto &pair : replaceableValues) {
    SmallPtrSet<Block *, 1> requirements;
    bool _tmp = pair.second->definedWithArg(requirements);
    assert(_tmp);
    PotentiallyHelpfulArgs.insert(requirements.begin(), requirements.end());
  }

  enum class Legality {
    Unknown = 0,
    Illegal = 2,
    Required = 1,
  };
  // Map of block with potential arg added to the legality of adding that
  // argument
  std::map<Block *, Legality> PotentialArgs;
  // Map of block with potential arg added to the users which would require it
  // to compute their end value.
  std::map<Block *, std::set<Block *>> UserMap;

  // Map of block with potential arg added to the blocks it itself would
  // require.
  std::map<Block *, std::set<Block *>> RequirementMap;

  std::deque<Block *> todo(PotentiallyHelpfulArgs.begin(),
                           PotentiallyHelpfulArgs.end());
  while (todo.size()) {
    auto *block = todo.back();
    todo.pop_back();

    if (PotentialArgs.find(block) != PotentialArgs.end())
      continue;
    PotentialArgs[block] = Legality::Unknown;

    SetVector<Block *> prepred(block->getPredecessors().begin(),
                               block->getPredecessors().end());
    assert(prepred.size());

    for (auto *Pred : prepred) {

      auto endFind = valueAtEndOfBlock.find(Pred);
      assert(endFind != valueAtEndOfBlock.end());

      // Only handle known termination blocks
      if (!isa<cf::BranchOp, cf::CondBranchOp, cf::SwitchOp>(
              Pred->getTerminator())) {
        PotentialArgs[block] = Legality::Illegal;
        break;
      }

      SmallPtrSet<Block *, 1> requirements;
      if (endFind->second->definedWithArg(requirements)) {
        for (auto *r : requirements) {
          todo.push_back(r);
          UserMap[r].insert(block);
          RequirementMap[block].insert(r);
        }
      } else {
        PotentialArgs[block] = Legality::Illegal;
        break;
      }
    }
  }

  // Mark all blocks which may have an illegal predecessor
  for (auto &pair : PotentialArgs)
    if (pair.second == Legality::Illegal)
      for (auto *next : UserMap[pair.first])
        todo.push_back(next);

  while (todo.size()) {
    auto *block = todo.back();
    todo.pop_back();
    if (PotentialArgs[block] == Legality::Illegal)
      continue;
    PotentialArgs[block] = Legality::Illegal;
    for (auto *next : UserMap[block])
      todo.push_back(next);
  }

  // Go through all loading ops and confirm that all the requirements are met
  SmallVector<std::pair<Value, ValueOrPlaceholder *>> nextReplacements;
  for (auto pair : replaceableValues) {
    SmallPtrSet<Block *, 1> requirements;
    bool _tmp = pair.second->definedWithArg(requirements);
    assert(_tmp);
    bool illegal = false;
    for (auto *r : requirements) {
      if (PotentialArgs[r] == Legality::Illegal) {
        illegal = true;
        break;
      }
    }
    // If illegal, remove this from the list of things to update.
    if (illegal) {
      continue;
    }
    // Otherwise mark that block arg, and all of its dependencies as required.
    for (auto *r : requirements)
      todo.push_back(r);

    while (todo.size()) {
      auto *block = todo.back();
      todo.pop_back();
      if (PotentialArgs[block] == Legality::Required)
        continue;
      PotentialArgs[block] = Legality::Required;
      for (auto *prev : RequirementMap[block])
        todo.push_back(prev);
    }
    nextReplacements.push_back(pair);
  }
  replaceableValues = nextReplacements;

  if (replaceableValues.size() == 0)
    return changed;

  SmallVector<Block *, 1> Legal;
  for (auto &pair : PotentialArgs)
    if (pair.second == Legality::Required) {
      Legal.push_back(pair.first);
    }

  for (auto *block : Legal) {
    auto startFound = valueAtStartOfBlock.find(block);

    assert(startFound != valueAtStartOfBlock.end());
    assert(startFound->second->valueAtStart == block);
    auto arg = block->addArgument(subType, loc);
    auto *argVal = metaMap.get(arg);
    valueAtStartOfBlock[block] = argVal;
    blocksWithAddedArgs[block] = arg;
  }

  for (auto &pair : replaceableValues) {
    assert(pair.first);
    assert(pair.second);
    Value val = pair.second->materialize(true);
    assert(val);

    changed = true;
    assert(pair.first != val);
    assert(val.getType() == elType);
    assert(pair.first.getType() == val.getType() && "mismatched load type");
    LLVM_DEBUG(llvm::dbgs()
               << " replaced " << pair.first << " with " << val << "\n");
    metaMap.replaceValue(pair.first, val);

    // Record this to erase later.
    loadOpsToErase.push_back(pair.first.getDefiningOp());
    loadOps.erase(pair.first.getDefiningOp());
  }
  replaceableValues.clear();

  for (auto &pair : blocksWithAddedArgs) {
    Block *block = pair.first;
    assert(valueAtStartOfBlock.find(block) != valueAtStartOfBlock.end());

    Value maybeblockArg =
        valueAtStartOfBlock.find(block)->second->materialize(false);
    auto blockArg = maybeblockArg.dyn_cast<BlockArgument>();
    assert(blockArg && blockArg.getOwner() == block);

    SetVector<Block *> prepred(block->getPredecessors().begin(),
                               block->getPredecessors().end());
    for (auto *pred : prepred) {
      assert(pred && "Null predecessor");
      assert(valueAtEndOfBlock.find(pred) != valueAtEndOfBlock.end());
      assert(valueAtEndOfBlock.find(pred)->second);
      mlir::Value pval =
          valueAtEndOfBlock.find(pred)->second->materialize(true);
      if (!pval || pval.getType() != elType) {
        AI.getDefiningOp()->getParentOfType<func::FuncOp>().dump();
        pred->dump();
        llvm::errs() << "pval: " << *valueAtEndOfBlock.find(pred)->second
                     << " AI: " << AI << "\n";
        if (pval)
          llvm::errs() << " mat pval: " << pval << "\n";
      }
      assert(pval && "Null last stored");
      assert(pval.getType() == elType);
      assert(pred->getTerminator());

      assert(blockArg.getOwner() == block);
      if (auto op = dyn_cast<cf::BranchOp>(pred->getTerminator())) {
        mlir::OpBuilder subbuilder(op.getOperation());
        std::vector<Value> args(op.getOperands().begin(),
                                op.getOperands().end());
        args.push_back(pval);
        subbuilder.create<cf::BranchOp>(op.getLoc(), op.getDest(), args);
        op.erase();
      } else if (auto op = dyn_cast<cf::CondBranchOp>(pred->getTerminator())) {

        mlir::OpBuilder subbuilder(op.getOperation());
        std::vector<Value> trueargs(op.getTrueOperands().begin(),
                                    op.getTrueOperands().end());
        std::vector<Value> falseargs(op.getFalseOperands().begin(),
                                     op.getFalseOperands().end());
        if (op.getTrueDest() == block) {
          trueargs.push_back(pval);
        }
        if (op.getFalseDest() == block) {
          falseargs.push_back(pval);
        }
        subbuilder.create<cf::CondBranchOp>(op.getLoc(), op.getCondition(),
                                            op.getTrueDest(), trueargs,
                                            op.getFalseDest(), falseargs);
        op.erase();
      } else if (auto op = dyn_cast<cf::SwitchOp>(pred->getTerminator())) {
        mlir::OpBuilder builder(op.getOperation());
        SmallVector<Value> defaultOps(op.getDefaultOperands().begin(),
                                      op.getDefaultOperands().end());

        if (op.getDefaultDestination() == block)
          defaultOps.push_back(pval);

        SmallVector<SmallVector<Value>> cases;
        for (auto pair : llvm::enumerate(op.getCaseDestinations())) {
          cases.emplace_back(op.getCaseOperands(pair.index()).begin(),
                             op.getCaseOperands(pair.index()).end());
          if (pair.value() == block) {
            cases.back().push_back(pval);
          }
        }
        SmallVector<ValueRange> vrange;
        for (auto &c : cases)
          vrange.push_back(c);

        builder.create<cf::SwitchOp>(
            op.getLoc(), op.getFlag(), op.getDefaultDestination(), defaultOps,
            op.getCaseValuesAttr(), op.getCaseDestinations(), vrange);
        op.erase();
      } else {
        llvm_unreachable("unknown pred branch");
      }
    }
  }

  removeRedundantBlockArgs(AI, elType, blocksWithAddedArgs);

  for (auto *loadOp : llvm::make_early_inc_range(loadOps)) {
    assert(loadOp);
    if (loadOp->getResult(0).use_empty()) {
      loadOpsToErase.push_back(loadOp);
      loadOps.erase(loadOp);
    }
  }
  return changed;
}

bool isPromotable(mlir::Value AI) {
  std::deque<mlir::Value> list = {AI};

  while (list.size()) {
    auto val = list.front();
    list.pop_front();

    for (auto *U : val.getUsers()) {
      if (auto LO = dyn_cast<memref::LoadOp>(U)) {
        continue;
      } else if (auto LO = dyn_cast<LLVM::LoadOp>(U)) {
        continue;
      } else if (auto SO = dyn_cast<LLVM::StoreOp>(U)) {
        continue;
      } else if (auto LO = dyn_cast<AffineLoadOp>(U)) {
        continue;
      } else if (auto SO = dyn_cast<memref::StoreOp>(U)) {
        continue;
      } else if (auto SO = dyn_cast<AffineStoreOp>(U)) {
        continue;
      } else if (isa<memref::DeallocOp>(U)) {
        continue;
      } else if (isa<func::CallOp>(U) &&
                 cast<func::CallOp>(U).getCallee() == "free") {
        continue;
      } else if (isa<func::CallOp>(U)) {
        // TODO check "no capture", currently assume as a fallback always
        // nocapture
        continue;
      } else if (auto CO = dyn_cast<memref::CastOp>(U)) {
        list.push_back(CO);
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "non promotable " << AI << " due to " << *U << "\n");
        return false;
      }
    }
  }
  return true;
}

std::vector<std::vector<Offset>> getLastStored(mlir::Value AI) {
  std::map<std::vector<Offset>, unsigned> lastStored;

  std::deque<mlir::Value> list = {AI};

  while (list.size()) {
    auto val = list.front();
    list.pop_front();
    for (auto *U : val.getUsers()) {
      if (auto SO = dyn_cast<memref::StoreOp>(U)) {
        std::vector<Offset> vec;
        for (auto idx : SO.getIndices()) {
          vec.emplace_back(idx);
        }
        lastStored[vec]++;
      } else if (auto SO = dyn_cast<AffineLoadOp>(U)) {
        std::vector<Offset> vec;
        auto map = SO.getAffineMapAttr().getValue();
        for (auto idx : map.getResults()) {
          vec.emplace_back(idx, map.getNumDims(), map.getNumSymbols(),
                           SO.getMapOperands());
        }
        lastStored[vec]++;
      } else if (isa<LLVM::LoadOp>(U)) {
        std::vector<Offset> vec;
        lastStored[vec]++;
      } else if (isa<LLVM::StoreOp>(U)) {
        std::vector<Offset> vec;
        lastStored[vec]++;
      } else if (auto SO = dyn_cast<memref::LoadOp>(U)) {
        std::vector<Offset> vec;
        for (auto idx : SO.getIndices()) {
          vec.emplace_back(idx);
        }
        lastStored[vec]++;
      } else if (auto SO = dyn_cast<AffineStoreOp>(U)) {
        std::vector<Offset> vec;
        auto map = SO.getAffineMapAttr().getValue();
        for (auto idx : map.getResults()) {
          vec.emplace_back(idx, map.getNumDims(), map.getNumSymbols(),
                           SO.getMapOperands());
        }
        lastStored[vec]++;
      } else if (auto CO = dyn_cast<memref::CastOp>(U)) {
        list.push_back(CO);
      }
    }
  }

  std::vector<std::vector<Offset>> todo;
  for (auto &pair : lastStored) {
    if (pair.second > 1)
      todo.push_back(pair.first);
  }
  return todo;
}

void Mem2Reg::runOnOperation() {
  auto *f = getOperation();

  // Variable indicating that a memref has had a load removed
  // and or been deleted. Because there can be memrefs of
  // memrefs etc, we may need to do multiple passes (first
  // to eliminate the outermost one, then inner ones)
  bool changed;
  do {
    changed = false;

    // A list of memref's that are potentially dead / could be eliminated.
    SmallPtrSet<Value, 4> memrefsToErase;

    // Load op's whose results were replaced by those forwarded from stores.
    SmallVector<Operation *, 8> loadOpsToErase;

    // Walk all load's and perform store to load forwarding.
    SmallVector<mlir::Value, 4> toPromote;
    f->walk([&](mlir::memref::AllocaOp AI) {
      if (isPromotable(AI)) {
        toPromote.push_back(AI);
      }
    });
    f->walk([&](mlir::memref::AllocOp AI) {
      if (isPromotable(AI)) {
        toPromote.push_back(AI);
      }
    });
    f->walk([&](LLVM::AllocaOp AI) {
      if (isPromotable(AI)) {
        toPromote.push_back(AI);
      }
    });
    f->walk([&](memref::GetGlobalOp AI) {
      if (isPromotable(AI)) {
        toPromote.push_back(AI);
      }
    });
    DenseMap<Operation *, SmallVector<Operation *>> capturedAliasing;
    for (auto AI : toPromote) {
      LLVM_DEBUG(llvm::dbgs() << " attempting to promote " << AI << "\n");
      auto lastStored = getLastStored(AI);
      for (const auto &vec : lastStored) {
        LLVM_DEBUG(llvm::dbgs() << " + forwarding vec to promote {";
                   for (auto m
                        : vec) llvm::dbgs()
                   << m << ",";
                   llvm::dbgs() << "} of " << AI << "\n");
        // llvm::errs() << " PRE " << AI << "\n";
        // f.dump();
        changed |=
            forwardStoreToLoad(AI, vec, loadOpsToErase, capturedAliasing);
        // llvm::errs() << " POST " << AI << "\n";
        // f.dump();
      }
      if (!AI.getDefiningOp<memref::GetGlobalOp>())
        memrefsToErase.insert(AI);
    }

    // Erase all load op's whose results were replaced with store fwd'ed ones.
    for (auto *loadOp : loadOpsToErase) {
      changed = true;
      loadOp->erase();
    }

    // Check if the store fwd'ed memrefs are now left with only stores and can
    // thus be completely deleted. Note: the canonicalize pass should be able
    // to do this as well, but we'll do it here since we collected these anyway.
    for (auto memref : memrefsToErase) {

      // If the memref hasn't been alloc'ed in this function, skip.
      Operation *defOp = memref.getDefiningOp();
      if (!defOp ||
          !(isa<memref::AllocOp>(defOp) || isa<memref::AllocaOp>(defOp) ||
            isa<LLVM::AllocaOp>(defOp)))
        // TODO: if the memref was returned by a 'call' operation, we
        // could still erase it if the call had no side-effects.
        continue;

      std::deque<mlir::Value> list = {memref};
      std::vector<mlir::Operation *> toErase;
      bool error = false;
      while (list.size()) {
        auto val = list.front();
        list.pop_front();

        for (auto *U : val.getUsers()) {
          if (auto SO = dyn_cast<LLVM::StoreOp>(U)) {
            if (SO.getValue() == val) {
              error = true;
              break;
            }
            toErase.push_back(U);
          } else if (auto SO = dyn_cast<memref::StoreOp>(U)) {
            if (SO.getValue() == val) {
              error = true;
              break;
            }
            toErase.push_back(U);
          } else if (auto SO = dyn_cast<AffineStoreOp>(U)) {
            if (SO.getValue() == val) {
              error = true;
              break;
            }
            toErase.push_back(U);
          } else if (isa<memref::DeallocOp>(U)) {
            toErase.push_back(U);
          } else if (isa<func::CallOp>(U) &&
                     cast<func::CallOp>(U).getCallee() == "free") {
            toErase.push_back(U);
          } else if (auto CO = dyn_cast<memref::CastOp>(U)) {
            toErase.push_back(U);
            list.push_back(CO);
          } else if (auto CO = dyn_cast<polygeist::SubIndexOp>(U)) {
            toErase.push_back(U);
            list.push_back(CO);
          } else {
            error = true;
            break;
          }
        }
        if (error)
          break;
      }

      if (!error) {
        std::reverse(toErase.begin(), toErase.end());
        for (auto *user : toErase) {
          user->erase();
        }
        defOp->erase();
        changed = true;
      } else {
        // llvm::errs() << " failed to remove: " << memref << "\n";
      }
    }
  } while (changed);
}
