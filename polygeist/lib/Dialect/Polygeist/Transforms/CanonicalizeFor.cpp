//===- CanonicalizeFor.cpp - Loop Canonicalization --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Polygeist/IR/Ops.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <mlir/Dialect/Arith/IR/Arith.h>

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_SCFCANONICALIZEFOR
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

using namespace mlir;

namespace {
struct CanonicalizeFor
    : public polygeist::impl::SCFCanonicalizeForBase<CanonicalizeFor> {
  void runOnOperation() override;
};
} // namespace

struct PropagateInLoopBody : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    if (!forOp.hasIterOperands())
      return failure();

    Block &block = forOp.getRegion().front();
    auto yieldOp = cast<scf::YieldOp>(block.getTerminator());
    bool didSomething = false;
    for (auto it : llvm::zip(forOp.getIterOperands(), forOp.getRegionIterArgs(),
                             yieldOp.getOperands())) {
      Value iterOperand = std::get<0>(it);
      Value regionArg = std::get<1>(it);
      Value yieldOperand = std::get<2>(it);

      Operation *op = iterOperand.getDefiningOp();
      if (op && (op->getNumResults() == 1) && (iterOperand == yieldOperand)) {
        regionArg.replaceAllUsesWith(op->getResult(0));
        didSomething = true;
      }
    }
    return success(didSomething);
  }
};

struct ForOpInductionReplacement : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    // Defer until after the step is a constant, if possible
    if (auto indexCast = forOp.getStep().getDefiningOp<arith::IndexCastOp>())
      if (matchPattern(indexCast.getIn(), m_Constant()))
        return failure();

    bool didSomething = false;
    Block &block = forOp.getRegion().front();
    auto yieldOp = cast<scf::YieldOp>(block.getTerminator());

    for (auto it : llvm::zip(forOp.getIterOperands(),   // iter from outside
                             forOp.getRegionIterArgs(), // iter inside region
                             forOp.getResults(),        // op results
                             yieldOp.getOperands()      // iter yield
                             )) {
      auto addOp = std::get<3>(it).getDefiningOp<arith::AddIOp>();
      if (!addOp)
        continue;
      if (addOp.getOperand(0) != std::get<1>(it))
        continue;
      if (!addOp.getOperand(1).getParentRegion()->isAncestor(
              forOp->getParentRegion()))
        continue;

      bool sameValue = addOp.getOperand(1) == forOp.getStep();

      APInt rattr, sattr;
      if (matchPattern(addOp.getOperand(1), m_ConstantInt(&rattr)))
        if (matchPattern(forOp.getStep(), m_ConstantInt(&sattr))) {
          size_t maxWidth = (rattr.getBitWidth() > sattr.getBitWidth())
                                ? rattr.getBitWidth()
                                : sattr.getBitWidth();
          sameValue |= rattr.zext(maxWidth) == sattr.zext(maxWidth);
        }

      if (!std::get<1>(it).use_empty()) {
        Value init = std::get<0>(it);
        rewriter.setInsertionPointToStart(&forOp.getRegion().front());
        Value replacement = rewriter.create<arith::SubIOp>(
            forOp.getLoc(), forOp.getInductionVar(), forOp.getLowerBound());

        if (!sameValue) {
          replacement = rewriter.create<arith::DivUIOp>(
              forOp.getLoc(), replacement, forOp.getStep());

          Value step = addOp.getOperand(1);
          if (!isa<IndexType>(step.getType()))
            step = rewriter.create<arith::IndexCastOp>(
                forOp.getLoc(), replacement.getType(), step);

          replacement =
              rewriter.create<arith::MulIOp>(forOp.getLoc(), replacement, step);
        }

        if (!isa<IndexType>(init.getType()))
          init = rewriter.create<arith::IndexCastOp>(
              forOp.getLoc(), replacement.getType(), init);

        replacement =
            rewriter.create<arith::AddIOp>(forOp.getLoc(), init, replacement);

        if (!isa<IndexType>(std::get<1>(it).getType()))
          replacement = rewriter.create<arith::IndexCastOp>(
              forOp.getLoc(), std::get<1>(it).getType(), replacement);

        rewriter.updateRootInPlace(
            forOp, [&] { std::get<1>(it).replaceAllUsesWith(replacement); });
        didSomething = true;
      }

      if (!std::get<2>(it).use_empty()) {
        Value init = std::get<0>(it);
        rewriter.setInsertionPoint(forOp);
        Value replacement = rewriter.create<arith::SubIOp>(
            forOp.getLoc(), forOp.getUpperBound(), forOp.getLowerBound());

        if (!sameValue) {
          replacement = rewriter.create<arith::DivUIOp>(
              forOp.getLoc(), replacement, forOp.getStep());

          Value step = addOp.getOperand(1);
          if (!isa<IndexType>(step.getType()))
            step = rewriter.create<arith::IndexCastOp>(
                forOp.getLoc(), replacement.getType(), step);

          replacement =
              rewriter.create<arith::MulIOp>(forOp.getLoc(), replacement, step);
        }

        if (!isa<IndexType>(init.getType()))
          init = rewriter.create<arith::IndexCastOp>(
              forOp.getLoc(), replacement.getType(), init);

        replacement =
            rewriter.create<arith::AddIOp>(forOp.getLoc(), init, replacement);

        if (!isa<IndexType>(std::get<1>(it).getType()))
          replacement = rewriter.create<arith::IndexCastOp>(
              forOp.getLoc(), std::get<1>(it).getType(), replacement);

        rewriter.updateRootInPlace(
            forOp, [&] { std::get<2>(it).replaceAllUsesWith(replacement); });
        didSomething = true;
      }
    }

    return success(didSomething);
  }
};

/// Remove unused iterator operands.
// TODO: IRMapping for indVar.
struct RemoveUnusedArgs : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value> usedBlockArgs, usedOperands;
    SmallVector<OpResult> usedResults;

    unsigned i = 0;
    // if the block argument or the result at the
    // same index position have uses do not eliminate.
    for (auto blockArg : forOp.getRegionIterArgs()) {
      if (!blockArg.use_empty() || !forOp.getResult(i).use_empty()) {
        usedOperands.push_back(
            forOp.getOperand(forOp.getNumControlOperands() + i));
        usedResults.push_back(forOp->getOpResult(i));
        usedBlockArgs.push_back(blockArg);
      }
      i++;
    }

    // no work to do.
    if (usedOperands.size() == forOp.getIterOperands().size())
      return failure();

    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), usedOperands);

    if (!newForOp.getBody()->empty())
      rewriter.eraseOp(newForOp.getBody()->getTerminator());

    newForOp.getBody()->getOperations().splice(
        newForOp.getBody()->getOperations().begin(),
        forOp.getBody()->getOperations());

    rewriter.updateRootInPlace(forOp, [&] {
      forOp.getInductionVar().replaceAllUsesWith(newForOp.getInductionVar());
      for (auto pair : llvm::zip(usedBlockArgs, newForOp.getRegionIterArgs()))
        std::get<0>(pair).replaceAllUsesWith(std::get<1>(pair));
    });

    // adjust return.
    auto yieldOp = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
    SmallVector<Value> usedYieldOperands{};
    llvm::transform(usedResults, std::back_inserter(usedYieldOperands),
                    [&](OpResult result) {
                      return yieldOp.getOperand(result.getResultNumber());
                    });
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, usedYieldOperands);

    // Replace the operation's results with the new ones.
    SmallVector<Value> repResults(forOp.getNumResults());
    for (auto en : llvm::enumerate(usedResults))
      repResults[cast<OpResult>(en.value()).getResultNumber()] =
          newForOp.getResult(en.index());

    rewriter.replaceOp(forOp, repResults);
    return success();
  }
};

struct ReplaceRedundantArgs : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    bool didSomething = false;
    unsigned i = 0;
    for (auto blockArg : forOp.getRegionIterArgs()) {
      for (unsigned j = 0; j < i; j++) {
        if (forOp.getOperand(forOp.getNumControlOperands() + i) ==
                forOp.getOperand(forOp.getNumControlOperands() + j) &&
            yieldOp.getOperand(i) == yieldOp.getOperand(j)) {

          rewriter.updateRootInPlace(forOp, [&] {
            forOp.getResult(i).replaceAllUsesWith(forOp.getResult(j));
            blockArg.replaceAllUsesWith(forOp.getRegionIterArgs()[j]);
          });
          didSomething = true;
          goto skip;
        }
      }
    skip:
      i++;
    }

    return success(didSomething);
  }
};

#if 0
struct RemoveNotIf : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    // Replace the operation if only a subset of its results have uses.
    if (op.getNumResults() == 0)
      return failure();

    auto trueYield = cast<scf::YieldOp>(op.thenRegion().back().getTerminator());
    auto falseYield =
        cast<scf::YieldOp>(op.thenRegion().back().getTerminator());

    rewriter.setInsertionPoint(op->getBlock(),
                               op.getOperation()->getIterator());
    bool didSomething = false;
    for (auto tup :
         llvm::zip(trueYield.results(), falseYield.results(), op.results())) {
      if (!std::get<0>(tup).getType().isInteger(1))
        continue;
      if (auto top = std::get<0>(tup).getDefiningOp<ConstantOp>()) {
        if (auto fop = std::get<1>(tup).getDefiningOp<ConstantOp>()) {
          if (cast<IntegerAttr>(top.getValue()).getValue() == 0 &&
              cast<IntegerAttr>(fop.getValue()).getValue() == 1) {

            for (OpOperand &use :
                 llvm::make_early_inc_range(std::get<2>(tup).getUses())) {
              didSomething = true;
              rewriter.updateRootInPlace(use.getOwner(), [&]() {
                use.set(rewriter.create<XOrOp>(op.getLoc(), op.condition()));
              });
            }
          }
          if (cast<IntegerAttr>(top.getValue()).getValue() == 1 &&
              cast<IntegerAttr>(fop.getValue()).getValue() == 0) {
            for (OpOperand &use :
                 llvm::make_early_inc_range(std::get<2>(tup).getUses())) {
              didSomething = true;
              rewriter.updateRootInPlace(use.getOwner(),
                                         [&]() { use.set(op.condition()); });
            }
          }
        }
      }
    }
    return didSomething ? success() : failure();
  }
};

struct RemoveBoolean : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp op,
                                PatternRewriter &rewriter) const override {
    bool didSomething = false;

    if (llvm::all_of(op.results(), [](Value v) {
          return isa<IntegerType>(v.getType()) &&
                 cast<IntegerType>(v.getType()).getWidth() == 1;
        })) {
      if (op.thenRegion().getBlocks().size() == 1 &&
          op.elseRegion().getBlocks().size() == 1) {
        while (isa<CmpIOp>(op.thenRegion().front().front())) {
          op.thenRegion().front().front().moveBefore(op);
          didSomething = true;
        }
        while (isa<CmpIOp>(op.elseRegion().front().front())) {
          op.elseRegion().front().front().moveBefore(op);
          didSomething = true;
        }
        if (op.thenRegion().front().getOperations().size() == 1 &&
            op.elseRegion().front().getOperations().size() == 1) {
          auto yop1 =
              cast<scf::YieldOp>(op.thenRegion().front().getTerminator());
          auto yop2 =
              cast<scf::YieldOp>(op.elseRegion().front().getTerminator());
          size_t idx = 0;

          auto c1 = (Value)rewriter.create<ConstantOp>(
              op.getLoc(), op.condition().getType(),
              rewriter.getIntegerAttr(op.condition().getType(), 1));
          auto notCond = (Value)rewriter.create<XOrOp>(
              op.getLoc(), op.condition(), c1);

          std::vector<Value> replacements;
          for (auto res : op.results()) {
            auto rep = rewriter.create<OrOp>(
                op.getLoc(),
                rewriter.create<AndOp>(op.getLoc(), op.condition(),
                                       yop1.results()[idx]),
                rewriter.create<AndOp>(op.getLoc(), notCond,
                                       yop2.results()[idx]));
            replacements.push_back(rep);
            idx++;
          }
          rewriter.replaceOp(op, replacements);
          // op.erase();
          return success();
        }
      }
    }

    if (op.thenRegion().getBlocks().size() == 1 &&
        op.elseRegion().getBlocks().size() == 1 &&
        op.thenRegion().front().getOperations().size() == 1 &&
        op.elseRegion().front().getOperations().size() == 1) {
      auto yop1 = cast<scf::YieldOp>(op.thenRegion().front().getTerminator());
      auto yop2 = cast<scf::YieldOp>(op.elseRegion().front().getTerminator());
      size_t idx = 0;

      std::vector<Value> replacements;
      for (auto res : op.results()) {
        auto rep =
            rewriter.create<SelectOp>(op.getLoc(), op.condition(),
                                yop1.results()[idx], yop2.results()[idx]);
        replacements.push_back(rep);
        idx++;
      }
      rewriter.replaceOp(op, replacements);
      return success();
    }
    return didSomething ? success() : failure();
  }
};
#endif

static bool isTopLevelArgValue(Value value, Region *region) {
  if (auto arg = dyn_cast<BlockArgument>(value))
    return arg.getParentRegion() == region;
  return false;
}

static bool dominateWhile(Value value, scf::WhileOp loop) {
  Operation *op = value.getDefiningOp();
  assert(op && "expect non-null");
  DominanceInfo dom(loop);
  return dom.properlyDominates(op, loop);
}

static bool canMoveOpOutsideWhile(Operation *op, scf::WhileOp loop) {
  DominanceInfo dom(loop);
  for (auto operand : op->getOperands()) {
    if (!dom.properlyDominates(operand, loop))
      return false;
  }
  return true;
}

struct WhileToForHelper {
  scf::WhileOp loop;
  arith::CmpIOp cmpIOp;
  Value step;
  Value lb;
  bool lb_addOne;
  Value ub;
  bool ub_addOne;
  bool ub_cloneMove;
  bool negativeStep;
  arith::AddIOp addIOp;
  BlockArgument indVar;
  size_t afterArgIdx;

  bool computeLegality(bool sizeCheck, Value lookThrough = nullptr) {
    step = nullptr;
    lb = nullptr;
    lb_addOne = false;
    ub = nullptr;
    ub_addOne = false;
    ub_cloneMove = false;
    negativeStep = false;

    auto condOp = loop.getConditionOp();
    indVar = dyn_cast<BlockArgument>(cmpIOp.getLhs());
    Type extType = nullptr;
    // todo handle ext
    if (auto ext = cmpIOp.getLhs().getDefiningOp<arith::ExtSIOp>()) {
      indVar = dyn_cast<BlockArgument>(ext.getIn());
      extType = ext.getType();
    }

    // Condition is not the same as an induction variable
    if (!indVar || indVar.getOwner() != &loop.getBefore().front())
      return false;

    // Before region contains more than just the comparison
    {
      size_t size = loop.getBefore().front().getOperations().size();
      if (extType)
        size--;
      if (!sizeCheck)
        size--;
      if (size != 2)
        return false;
    }

    SmallVector<size_t> afterArgs;
    for (auto pair : llvm::enumerate(condOp.getArgs())) {
      if (pair.value() == indVar)
        afterArgs.push_back(pair.index());
    }

    auto endYield = cast<scf::YieldOp>(loop.getAfter().back().getTerminator());

    // Check that the block argument is actually an induction var:
    //   Namely, its next value adds to the previous with an invariant step.
    addIOp = endYield.getResults()[indVar.getArgNumber()]
                 .getDefiningOp<arith::AddIOp>();
    if (!addIOp) {
      if (auto ifOp = endYield.getResults()[indVar.getArgNumber()]
                          .getDefiningOp<scf::IfOp>()) {
        if (ifOp.getCondition() == lookThrough) {
          for (auto r : llvm::enumerate(ifOp.getResults())) {
            if (r.value() == endYield.getResults()[indVar.getArgNumber()]) {
              addIOp = ifOp.thenYield()
                           .getOperand(r.index())
                           .getDefiningOp<arith::AddIOp>();
              break;
            }
          }
        }
      } else if (auto selOp = endYield.getResults()[indVar.getArgNumber()]
                                  .getDefiningOp<arith::SelectOp>()) {
        if (selOp.getCondition() == lookThrough)
          addIOp = selOp.getTrueValue().getDefiningOp<arith::AddIOp>();
      }
    }
    if (!addIOp)
      return false;

    for (auto afterArg : afterArgs) {
      auto arg = loop.getAfter().getArgument(afterArg);
      if (addIOp.getOperand(0) == arg) {
        step = addIOp.getOperand(1);
        afterArgIdx = afterArg;
        break;
      }
      if (addIOp.getOperand(1) == arg) {
        step = addIOp.getOperand(0);
        afterArgIdx = afterArg;
        break;
      }
    }

    if (!step)
      return false;

    // Cannot transform for if step is not loop-invariant
    if (auto *op = step.getDefiningOp()) {
      if (loop->isAncestor(op))
        return false;
    }

    negativeStep = false;
    if (auto cop = step.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(cop.getValue()))
        negativeStep = intAttr.getValue().isNegative();
    }

    if (!negativeStep)
      lb = loop.getOperand(indVar.getArgNumber());
    else {
      ub = loop.getOperand(indVar.getArgNumber());
      ub_addOne = true;
    }

    if (isa<BlockArgument>(cmpIOp.getRhs()) ||
        dominateWhile(cmpIOp.getRhs(), loop)) {
      switch (cmpIOp.getPredicate()) {
      case arith::CmpIPredicate::ule:
      case arith::CmpIPredicate::sle:
        ub_addOne = true;
        LLVM_FALLTHROUGH;
      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult:
        ub = cmpIOp.getRhs();
        break;
      case arith::CmpIPredicate::ugt:
      case arith::CmpIPredicate::sgt:
        lb_addOne = true;
        LLVM_FALLTHROUGH;
      case arith::CmpIPredicate::uge:
      case arith::CmpIPredicate::sge:
        lb = cmpIOp.getRhs();
        break;
      case arith::CmpIPredicate::eq:
      case arith::CmpIPredicate::ne:
        return false;
      }
    } else {
      if (negativeStep)
        return false;
      auto *op = cmpIOp.getRhs().getDefiningOp();
      if (!op || !canMoveOpOutsideWhile(op, loop) || (op->getNumResults() != 1))
        return false;
      ub = cmpIOp.getRhs();
      ub_cloneMove = true;
    }

    return lb && ub;
  }

  void prepareFor(PatternRewriter &rewriter) {
    if (lb_addOne) {
      Value one =
          rewriter.create<arith::ConstantIntOp>(loop.getLoc(), 1, lb.getType());
      lb = rewriter.create<arith::AddIOp>(loop.getLoc(), lb, one);
    }
    if (ub_cloneMove) {
      auto *op = ub.getDefiningOp();
      assert(op);
      auto *newOp = rewriter.clone(*op);
      rewriter.replaceOp(op, newOp->getResults());
      ub = newOp->getResult(0);
    }
    if (ub_addOne) {
      Value one =
          rewriter.create<arith::ConstantIntOp>(loop.getLoc(), 1, ub.getType());
      ub = rewriter.create<arith::AddIOp>(loop.getLoc(), ub, one);
    }
    if (negativeStep) {
      if (auto cop = step.getDefiningOp<arith::ConstantOp>()) {
        auto intAttr = cast<IntegerAttr>(cop.getValue());
        step = rewriter.create<arith::ConstantIndexOp>(
            cop.getLoc(), -intAttr.getValue().getSExtValue());
      }
    }

    ub = rewriter.create<arith::IndexCastOp>(
        loop.getLoc(), IndexType::get(loop.getContext()), ub);
    lb = rewriter.create<arith::IndexCastOp>(
        loop.getLoc(), IndexType::get(loop.getContext()), lb);
    step = rewriter.create<arith::IndexCastOp>(
        loop.getLoc(), IndexType::get(loop.getContext()), step);
  }
};

struct MoveWhileToFor : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    scf::ConditionOp condOp = whileOp.getConditionOp();
    auto cmpIOp = condOp.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!cmpIOp)
      return failure();

    WhileToForHelper helper;
    helper.cmpIOp = cmpIOp;
    helper.loop = whileOp;
    if (!helper.computeLegality(/*sizeCheck*/ true))
      return failure();

    helper.prepareFor(rewriter);

    // input of the for goes the input of the scf::while plus the output taken
    // from the conditionOp.
    SmallVector<Value> forArgs;
    forArgs.append(whileOp.getInits().begin(), whileOp.getInits().end());

    for (Value arg : condOp.getArgs()) {
      Type cst = nullptr;
      if (auto idx = arg.getDefiningOp<arith::IndexCastOp>()) {
        cst = idx.getType();
        arg = idx.getIn();
      }

      Value res = arg;
      if (isTopLevelArgValue(arg, &whileOp.getBefore())) {
        auto blockArg = cast<BlockArgument>(arg);
        res = whileOp.getInits()[blockArg.getArgNumber()];
      }

      if (cst)
        res = rewriter.create<arith::IndexCastOp>(condOp.getLoc(), cst, res);
      forArgs.push_back(res);
    }

    auto forLoop = rewriter.create<scf::ForOp>(whileOp.getLoc(), helper.lb,
                                               helper.ub, helper.step, forArgs);

    if (!forLoop.getBody()->empty())
      rewriter.eraseOp(forLoop.getBody()->getTerminator());

    auto oldYield =
        cast<scf::YieldOp>(whileOp.getAfter().front().getTerminator());

    rewriter.updateRootInPlace(whileOp, [&] {
      for (auto pair :
           llvm::zip(whileOp.getAfter().getArguments(), condOp.getArgs()))
        std::get<0>(pair).replaceAllUsesWith(std::get<1>(pair));
    });

    Block &after = whileOp.getAfter().front();
    after.eraseArguments(0, after.getNumArguments());

    SmallVector<Value> yieldOperands;
    for (auto oldYieldArg : oldYield.getResults())
      yieldOperands.push_back(oldYieldArg);

    IRMapping outMap;
    outMap.map(whileOp.getBefore().getArguments(), yieldOperands);
    for (auto arg : condOp.getArgs())
      yieldOperands.push_back(outMap.lookupOrDefault(arg));

    rewriter.setInsertionPoint(oldYield);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(oldYield, yieldOperands);

    size_t pos = whileOp.getInits().size();

    rewriter.updateRootInPlace(whileOp, [&] {
      for (auto pair : llvm::zip(whileOp.getBefore().getArguments(),
                                 forLoop.getRegionIterArgs().drop_back(pos)))
        std::get<0>(pair).replaceAllUsesWith(std::get<1>(pair));
    });

    forLoop.getBody()->getOperations().splice(
        forLoop.getBody()->getOperations().begin(),
        whileOp.getAfter().front().getOperations());

    SmallVector<Value> replacements;
    replacements.append(forLoop.getResults().begin() + pos,
                        forLoop.getResults().end());

    rewriter.replaceOp(whileOp, replacements);
    return success();
  }
};

// If and and with something is preventing creating a for
// move the and into the after body guarded by an if
struct MoveWhileAndDown : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    scf::ConditionOp condOp = whileOp.getConditionOp();
    auto andIOp = condOp.getCondition().getDefiningOp<arith::AndIOp>();
    if (!andIOp)
      return failure();

    for (int i = 0; i < 2; i++) {
      auto cmpIOp = andIOp->getOperand(i).getDefiningOp<arith::CmpIOp>();
      if (!cmpIOp)
        continue;

      WhileToForHelper helper;
      helper.loop = whileOp;
      helper.cmpIOp = cmpIOp;
      auto oldYield =
          cast<scf::YieldOp>(whileOp.getAfter().front().getTerminator());

      Value extraCmp = andIOp->getOperand(1 - i);
      Value lookThrough = nullptr;
      if (auto BA = dyn_cast<BlockArgument>(extraCmp))
        lookThrough = oldYield.getOperand(BA.getArgNumber());
      if (!helper.computeLegality(/*sizeCheck*/ false, lookThrough))
        continue;

      SmallVector<BlockArgument> origBeforeArgs(
          whileOp.getBeforeArguments().begin(),
          whileOp.getBeforeArguments().end());

      SmallVector<BlockArgument> origAfterArgs(
          whileOp.getAfterArguments().begin(),
          whileOp.getAfterArguments().end());

      IRMapping preMap;
      for (auto tup : llvm::zip(origBeforeArgs, whileOp.getInits()))
        preMap.map(std::get<0>(tup), std::get<1>(tup));
      for (auto &op : whileOp.getBefore().front()) {
        if (&op == condOp)
          break;
        preMap.map(op.getResults(), rewriter.clone(op, preMap)->getResults());
      }
      auto unroll =
          rewriter.create<scf::IfOp>(whileOp.getLoc(), whileOp.getResultTypes(),
                                     preMap.lookup(condOp.getCondition()));

      if (unroll.getThenRegion().getBlocks().size())
        rewriter.eraseBlock(unroll.thenBlock());
      rewriter.createBlock(&unroll.getThenRegion());
      rewriter.createBlock(&unroll.getElseRegion());

      rewriter.setInsertionPointToEnd(unroll.elseBlock());
      SmallVector<Value> unrollYield;
      for (auto v : condOp.getArgs())
        unrollYield.push_back(preMap.lookup(v));
      rewriter.create<scf::YieldOp>(whileOp.getLoc(), unrollYield);
      rewriter.setInsertionPointToEnd(unroll.thenBlock());

      SmallVector<Value> nextInits(unrollYield.begin(), unrollYield.end());
      auto falseVal = rewriter.create<arith::ConstantIntOp>(whileOp.getLoc(), 0,
                                                            extraCmp.getType());
      auto trueVal = rewriter.create<arith::ConstantIntOp>(whileOp.getLoc(), 1,
                                                           extraCmp.getType());
      nextInits.push_back(trueVal);
      nextInits.push_back(whileOp.getInits()[helper.indVar.getArgNumber()]);

      SmallVector<Type> resTys;
      for (auto init : nextInits)
        resTys.push_back(init.getType());

      auto newWhileOp =
          rewriter.create<scf::WhileOp>(whileOp.getLoc(), resTys, nextInits);

      rewriter.createBlock(&newWhileOp.getBefore());

      SmallVector<Value> newBeforeYieldArgs;
      for (auto afterArg : origAfterArgs) {
        auto arg = newWhileOp.getBefore().addArgument(afterArg.getType(),
                                                      afterArg.getLoc());
        newBeforeYieldArgs.push_back(arg);
      }
      Value notExited = newWhileOp.getBefore().front().addArgument(
          extraCmp.getType(), whileOp.getLoc());
      newBeforeYieldArgs.push_back(notExited);

      Value trueInd = newWhileOp.getBefore().front().addArgument(
          helper.indVar.getType(), whileOp.getLoc());
      newBeforeYieldArgs.push_back(trueInd);

      {
        IRMapping postMap;
        postMap.map(helper.indVar, trueInd);
        auto newCmp =
            cast<arith::CmpIOp>(rewriter.clone(*helper.cmpIOp, postMap));
        rewriter.create<scf::ConditionOp>(condOp.getLoc(), newCmp,
                                          newBeforeYieldArgs);
      }

      rewriter.createBlock(&newWhileOp.getAfter());
      SmallVector<Value> postElseYields;
      for (auto afterArg : origAfterArgs) {
        auto arg = newWhileOp.getAfter().front().addArgument(afterArg.getType(),
                                                             afterArg.getLoc());
        postElseYields.push_back(arg);
        afterArg.replaceAllUsesWith(arg);
      }

      SmallVector<Type> resultTypes(whileOp.getResultTypes());
      resultTypes.push_back(notExited.getType());
      notExited = newWhileOp.getAfter().front().addArgument(notExited.getType(),
                                                            whileOp.getLoc());

      trueInd = newWhileOp.getAfter().front().addArgument(trueInd.getType(),
                                                          whileOp.getLoc());

      auto guard =
          rewriter.create<scf::IfOp>(whileOp.getLoc(), resultTypes, notExited);
      if (guard.getThenRegion().getBlocks().size())
        rewriter.eraseBlock(guard.thenBlock());
      Block *post = rewriter.splitBlock(&whileOp.getAfter().front(),
                                        whileOp.getAfter().front().begin());
      rewriter.createBlock(&guard.getThenRegion());
      rewriter.createBlock(&guard.getElseRegion());
      rewriter.mergeBlocks(post, guard.thenBlock());

      {
        IRMapping postMap;
        for (auto tup : llvm::zip(origBeforeArgs, oldYield.getOperands()))
          postMap.map(std::get<0>(tup), std::get<1>(tup));

        rewriter.setInsertionPoint(oldYield);
        for (auto &op : whileOp.getBefore().front()) {
          if (&op == condOp)
            break;
          postMap.map(op.getResults(),
                      rewriter.clone(op, postMap)->getResults());
        }

        SmallVector<Value> postIfYields;
        for (auto arg : condOp.getArgs())
          postIfYields.push_back(postMap.lookup(arg));

        postIfYields.push_back(postMap.lookup(extraCmp));
        oldYield->setOperands(postIfYields);
      }

      rewriter.setInsertionPointToEnd(guard.elseBlock());
      postElseYields.push_back(falseVal);
      rewriter.create<scf::YieldOp>(whileOp.getLoc(), postElseYields);

      rewriter.setInsertionPointToEnd(&newWhileOp.getAfter().front());
      SmallVector<Value> postAfter(guard.getResults());
      IRMapping postMap;
      postMap.map(helper.indVar, trueInd);
      postMap.map(postElseYields[helper.afterArgIdx], trueInd);
      assert(helper.addIOp.getLhs() == postElseYields[helper.afterArgIdx] ||
             helper.addIOp.getRhs() == postElseYields[helper.afterArgIdx]);
      postAfter.push_back(
          cast<arith::AddIOp>(rewriter.clone(*helper.addIOp, postMap)));
      rewriter.create<scf::YieldOp>(whileOp.getLoc(), postAfter);

      rewriter.setInsertionPointToEnd(unroll.thenBlock());
      rewriter.create<scf::YieldOp>(
          whileOp.getLoc(),
          newWhileOp.getResults().take_front(whileOp.getResults().size()));

      rewriter.replaceOp(whileOp, unroll.getResults());

      return success();
    }

    return failure();
  }
};

struct MoveWhileDown : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    auto term =
        cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
    if (auto ifOp = term.getCondition().getDefiningOp<scf::IfOp>()) {
      if (ifOp.getNumResults() != term.getArgs().size() + 1)
        return failure();
      if (ifOp.getResult(0) != term.getCondition())
        return failure();
      for (size_t i = 1; i < ifOp.getNumResults(); ++i) {
        if (ifOp.getResult(i) != term.getArgs()[i - 1])
          return failure();
      }

      auto isYieldOK = [](scf::YieldOp yieldOp,
                          std::function<bool(int)> isLegal) {
        auto cop = yieldOp.getOperand(0).getDefiningOp<arith::ConstantIntOp>();
        return (cop && isLegal(cop.value()));
      };

      auto thenYieldOp =
          cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
      auto elseYieldOp =
          cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());

      if (!isYieldOK(thenYieldOp, [](int val) { return val != 0; }))
        return failure();
      if (!isYieldOK(elseYieldOp, [](int val) { return val == 0; }))
        return failure();
      if (ifOp.getElseRegion().front().getOperations().size() != 1)
        return failure();

      whileOp.getAfter().front().getOperations().splice(
          whileOp.getAfter().front().begin(),
          ifOp.getThenRegion().front().getOperations());
      rewriter.updateRootInPlace(term, [&] {
        term.getConditionMutable().assign(ifOp.getCondition());
      });

      SmallVector<Value> args;
      for (size_t i = 1; i < elseYieldOp.getNumOperands(); ++i)
        args.push_back(elseYieldOp.getOperand(i));

      rewriter.updateRootInPlace(term,
                                 [&] { term.getArgsMutable().assign(args); });
      rewriter.eraseOp(elseYieldOp);
      rewriter.eraseOp(ifOp);

      for (size_t i = 0; i < whileOp.getAfter().front().getNumArguments(); ++i)
        whileOp.getAfter().front().getArgument(i).replaceAllUsesWith(
            thenYieldOp.getOperand(i + 1));

      rewriter.eraseOp(thenYieldOp);
      // TODO move operands from begin to after
      SmallVector<Value> todo(
          whileOp.getBefore().front().getArguments().begin(),
          whileOp.getBefore().front().getArguments().end());
      for (auto &op : whileOp.getBefore().front()) {
        for (auto res : op.getResults())
          todo.push_back(res);
      }

      rewriter.updateRootInPlace(whileOp, [&] {
        for (auto val : todo) {
          auto na = whileOp.getAfter().front().addArgument(val.getType(),
                                                           whileOp.getLoc());
          val.replaceUsesWithIf(na, [&](OpOperand &u) -> bool {
            return whileOp.getAfter().isAncestor(
                u.getOwner()->getParentRegion());
          });
          args.push_back(val);
        }
      });

      rewriter.updateRootInPlace(term,
                                 [&] { term.getArgsMutable().assign(args); });

      SmallVector<Type> tys;
      for (auto arg : args)
        tys.push_back(arg.getType());

      auto newWhileOp = rewriter.create<scf::WhileOp>(whileOp.getLoc(), tys,
                                                      whileOp.getInits());
      newWhileOp.getBefore().takeBody(whileOp.getBefore());
      newWhileOp.getAfter().takeBody(whileOp.getAfter());
      SmallVector<Value> replacements;
      for (auto res : newWhileOp.getResults()) {
        if (replacements.size() == whileOp.getResults().size())
          break;
        replacements.push_back(res);
      }
      rewriter.replaceOp(whileOp, replacements);
      return success();
    }
    return failure();
  }
};

// Given code with structure:
// scf.while()
//    ...
//    %z = if(%c) {
//       %i1 = ..
//       ..
//    } else {
//    }
//    condition(%c) %z#0 ..
//  } loop {
//    ...
//  }
// Move the body of the if into the lower loop

struct MoveWhileDown2 : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  /// Populates `crossing` with values (op results) that are defined in the same
  /// block as `op` and above it, and used by at least one op in the same block
  /// below `op`. Uses may be in nested regions.
  static void findValuesUsedBelow(scf::IfOp ifOp,
                                  llvm::SetVector<Value> &crossing) {
    for (Operation *it = ifOp->getPrevNode(); it != nullptr;
         it = it->getPrevNode()) {
      for (Value value : it->getResults()) {
        for (Operation *user : value.getUsers()) {
          // ignore use of condition
          if (user == ifOp)
            continue;

          if (ifOp->isAncestor(user)) {
            crossing.insert(value);
            break;
          }
        }
      }
    }

    for (Value value : ifOp->getBlock()->getArguments()) {
      for (Operation *user : value.getUsers()) {
        // ignore use of condition
        if (user == ifOp)
          continue;

        if (ifOp->isAncestor(user)) {
          crossing.insert(value);
          break;
        }
      }
    }
    // No need to process block arguments, they are assumed to be induction
    // variables and will be replicated.
  }

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    auto term =
        cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
    if (auto ifOp = dyn_cast_or_null<scf::IfOp>(term->getPrevNode())) {
      if (ifOp.getCondition() != term.getCondition())
        return failure();

      SmallVector<std::pair<BlockArgument, Value>> m;

      // The return results of the while which are used.
      SmallVector<Value> prevResults;
      // The corresponding value in the before which is to be returned.
      SmallVector<Value> condArgs;

      SmallVector<std::pair<size_t, Value>> afterYieldRewrites;
      auto afterYield = cast<scf::YieldOp>(whileOp.getAfter().front().back());
      for (auto pair : llvm::zip(whileOp.getResults(), term.getArgs(),
                                 whileOp.getAfterArguments())) {
        if (std::get<1>(pair).getDefiningOp() == ifOp) {
          Value thenYielded, elseYielded;
          for (auto p :
               llvm::zip(ifOp.thenYield().getResults(), ifOp.getResults(),
                         ifOp.elseYield().getResults())) {
            if (std::get<1>(pair) == std::get<1>(p)) {
              thenYielded = std::get<0>(p);
              elseYielded = std::get<2>(p);
              break;
            }
          }
          assert(thenYielded);
          assert(elseYielded);

          // If one of the if results is returned, only handle the case
          // where the value yielded is a block argument
          // %out-i:pair<0> = scf.while (... i:%blockArg=... ) {
          //   %z:j = scf.if (%c) {
          //      ...
          //   } else {
          //      yield ... j:%blockArg
          //   }
          //   condition %c ... i:pair<1>=%z:j
          // } loop ( ... i:) {
          //    yield   i:pair<2>
          // }
          if (!std::get<0>(pair).use_empty()) {
            if (auto blockArg = dyn_cast<BlockArgument>(elseYielded))
              if (blockArg.getOwner() == &whileOp.getBefore().front()) {

                if (afterYield.getResults()[blockArg.getArgNumber()] ==
                        std::get<2>(pair) &&
                    whileOp.getResults()[blockArg.getArgNumber()] ==
                        std::get<0>(pair)) {
                  prevResults.push_back(std::get<0>(pair));
                  condArgs.push_back(blockArg);
                  afterYieldRewrites.emplace_back(blockArg.getArgNumber(),
                                                  thenYielded);
                  continue;
                }
              }
            return failure();
          }

          // If the value yielded from then then is defined in the while before
          // but not being moved down with the if, don't change anything.
          if (!ifOp.getThenRegion().isAncestor(thenYielded.getParentRegion()) &&
              whileOp.getBefore().isAncestor(thenYielded.getParentRegion())) {
            prevResults.push_back(std::get<0>(pair));
            condArgs.push_back(thenYielded);
          } else {
            // Otherwise, mark the corresponding after argument to be replaced
            // with the value yielded in the if statement.
            m.emplace_back(std::get<2>(pair), thenYielded);
          }
        } else {
          assert(prevResults.size() == condArgs.size());
          prevResults.push_back(std::get<0>(pair));
          condArgs.push_back(std::get<1>(pair));
        }
      }

      SmallVector<Value> yieldArgs = afterYield.getResults();
      for (auto pair : afterYieldRewrites)
        yieldArgs[pair.first] = pair.second;

      rewriter.updateRootInPlace(afterYield, [&] {
        afterYield.getResultsMutable().assign(yieldArgs);
      });
      Block *afterB = &whileOp.getAfter().front();

      {
        llvm::SetVector<Value> sv;
        findValuesUsedBelow(ifOp, sv);

        for (auto v : sv) {
          condArgs.push_back(v);
          auto arg = afterB->addArgument(v.getType(), ifOp->getLoc());
          for (OpOperand &use : llvm::make_early_inc_range(v.getUses())) {
            if (ifOp->isAncestor(use.getOwner()) ||
                use.getOwner() == afterYield)
              rewriter.updateRootInPlace(use.getOwner(),
                                         [&]() { use.set(arg); });
          }
        }
      }

      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<scf::ConditionOp>(term, term.getCondition(),
                                                    condArgs);

      llvm::BitVector indices(afterB->getNumArguments());
      for (int i = m.size() - 1; i >= 0; i--) {
        assert(m[i].first.getType() == m[i].second.getType());
        m[i].first.replaceAllUsesWith(m[i].second);
        indices.set(m[i].first.getArgNumber());
      }
      afterB->eraseArguments(indices);

      rewriter.eraseOp(ifOp.thenYield());
      Block *thenB = ifOp.thenBlock();
      afterB->getOperations().splice(afterB->getOperations().begin(),
                                     thenB->getOperations());

      rewriter.eraseOp(ifOp);

      SmallVector<Type> resultTypes;
      for (auto v : condArgs)
        resultTypes.push_back(v.getType());

      rewriter.setInsertionPoint(whileOp);
      auto newWhileOp = rewriter.create<scf::WhileOp>(
          whileOp.getLoc(), resultTypes, whileOp.getInits());
      newWhileOp.getBefore().takeBody(whileOp.getBefore());
      newWhileOp.getAfter().takeBody(whileOp.getAfter());

      rewriter.updateRootInPlace(whileOp, [&] {
        for (auto pair : llvm::enumerate(prevResults))
          pair.value().replaceAllUsesWith(newWhileOp.getResult(pair.index()));
      });

      rewriter.eraseOp(whileOp);
      return success();
    }
    return failure();
  }
};

struct MoveWhileInvariantIfResult : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<BlockArgument> origAfterArgs(
        whileOp.getAfterArguments().begin(), whileOp.getAfterArguments().end());
    bool didSomething = false;
    scf::ConditionOp term =
        cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
    assert(origAfterArgs.size() == whileOp.getResults().size());
    assert(origAfterArgs.size() == term.getArgs().size());

    for (auto pair :
         llvm::zip(whileOp.getResults(), term.getArgs(), origAfterArgs)) {
      if (!std::get<0>(pair).use_empty()) {
        if (auto ifOp = std::get<1>(pair).getDefiningOp<scf::IfOp>()) {
          if (ifOp.getCondition() == term.getCondition()) {
            auto idx = cast<OpResult>(std::get<1>(pair)).getResultNumber();
            Value returnWith = ifOp.elseYield().getResults()[idx];
            if (!whileOp.getBefore().isAncestor(returnWith.getParentRegion())) {
              rewriter.updateRootInPlace(whileOp, [&] {
                std::get<0>(pair).replaceAllUsesWith(returnWith);
              });
              didSomething = true;
            }
          }
        } else if (auto selOp =
                       std::get<1>(pair).getDefiningOp<arith::SelectOp>()) {
          if (selOp.getCondition() == term.getCondition()) {
            Value returnWith = selOp.getFalseValue();
            if (!whileOp.getBefore().isAncestor(returnWith.getParentRegion())) {
              rewriter.updateRootInPlace(whileOp, [&] {
                std::get<0>(pair).replaceAllUsesWith(returnWith);
              });
              didSomething = true;
            }
          }
        }
      }
    }

    return success(didSomething);
  }
};

struct WhileLogicalNegation : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    bool didSomething = false;
    auto term =
        cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());

    SmallPtrSet<Value, 1> condOps;
    SmallVector<Value> todo = {term.getCondition()};
    while (todo.size()) {
      Value val = todo.back();
      todo.pop_back();
      condOps.insert(val);
      if (auto ao = val.getDefiningOp<arith::AndIOp>()) {
        todo.push_back(ao.getLhs());
        todo.push_back(ao.getRhs());
      }
    }

    for (auto pair : llvm::zip(whileOp.getResults(), term.getArgs(),
                               whileOp.getAfterArguments())) {
      auto termArg = std::get<1>(pair);
      bool afterValue;
      if (condOps.count(termArg))
        afterValue = true;
      else {
        bool found = false;
        if (auto termCmp = termArg.getDefiningOp<arith::CmpIOp>()) {
          for (auto cond : condOps) {
            if (auto condCmp = cond.getDefiningOp<arith::CmpIOp>()) {
              if (termCmp.getLhs() == condCmp.getLhs() &&
                  termCmp.getRhs() == condCmp.getRhs()) {
                // TODO generalize to logical negation of
                if (condCmp.getPredicate() == arith::CmpIPredicate::slt &&
                    termCmp.getPredicate() == arith::CmpIPredicate::sge) {
                  found = true;
                  afterValue = false;
                  break;
                }
              }
            }
          }
        }
        if (!found)
          continue;
      }

      if (!std::get<0>(pair).use_empty()) {
        rewriter.updateRootInPlace(whileOp, [&] {
          rewriter.setInsertionPoint(whileOp);
          auto trueVal = rewriter.create<arith::ConstantIntOp>(whileOp.getLoc(),
                                                               !afterValue, 1);
          std::get<0>(pair).replaceAllUsesWith(trueVal);
        });
        didSomething = true;
      }
      if (!std::get<2>(pair).use_empty()) {
        rewriter.updateRootInPlace(whileOp, [&] {
          rewriter.setInsertionPointToStart(&whileOp.getAfter().front());
          auto trueVal = rewriter.create<arith::ConstantIntOp>(whileOp.getLoc(),
                                                               afterValue, 1);
          std::get<2>(pair).replaceAllUsesWith(trueVal);
        });
        didSomething = true;
      }
    }

    return success(didSomething);
  }
};

/// TODO move the addi down and replace below with a subi
struct WhileCmpOffset : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<BlockArgument> origAfterArgs(
        whileOp.getAfterArguments().begin(), whileOp.getAfterArguments().end());
    auto term =
        cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
    assert(origAfterArgs.size() == whileOp.getResults().size());
    assert(origAfterArgs.size() == term.getArgs().size());

    if (auto condCmp = term.getCondition().getDefiningOp<arith::CmpIOp>()) {
      if (auto addI = condCmp.getLhs().getDefiningOp<arith::AddIOp>()) {
        if (addI.getOperand(1).getDefiningOp() &&
            !whileOp.getBefore().isAncestor(
                addI.getOperand(1).getDefiningOp()->getParentRegion()))
          if (auto blockArg = dyn_cast<BlockArgument>(addI.getOperand(0))) {
            if (blockArg.getOwner() == &whileOp.getBefore().front()) {
              auto rng = llvm::make_early_inc_range(blockArg.getUses());

              {
                rewriter.setInsertionPoint(whileOp);
                SmallVector<Value> oldInits = whileOp.getInits();
                oldInits[blockArg.getArgNumber()] =
                    rewriter.create<arith::AddIOp>(
                        addI.getLoc(), oldInits[blockArg.getArgNumber()],
                        addI.getOperand(1));
                whileOp.getInitsMutable().assign(oldInits);
                rewriter.updateRootInPlace(
                    addI, [&] { addI.replaceAllUsesWith(blockArg); });
              }

              auto afterYield =
                  cast<scf::YieldOp>(whileOp.getAfter().front().back());
              rewriter.setInsertionPoint(afterYield);
              SmallVector<Value> oldYields = afterYield.getResults();
              oldYields[blockArg.getArgNumber()] =
                  rewriter.create<arith::AddIOp>(
                      addI.getLoc(), oldYields[blockArg.getArgNumber()],
                      addI.getOperand(1));
              rewriter.updateRootInPlace(afterYield, [&] {
                afterYield.getResultsMutable().assign(oldYields);
              });

              rewriter.setInsertionPointToStart(&whileOp.getBefore().front());
              auto sub = rewriter.create<arith::SubIOp>(addI.getLoc(), blockArg,
                                                        addI.getOperand(1));
              for (OpOperand &use : rng)
                rewriter.updateRootInPlace(use.getOwner(),
                                           [&]() { use.set(sub); });

              rewriter.eraseOp(addI);
              return success();
            }
          }
      }
    }

    return failure();
  }
};

/// Given a while loop which yields a select whose condition is
/// the same as the condition, remove the select.
struct RemoveWhileSelect : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    scf::ConditionOp term =
        cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());

    SmallVector<BlockArgument> origAfterArgs(
        whileOp.getAfterArguments().begin(), whileOp.getAfterArguments().end());
    SmallVector<unsigned> newResults, newAfter;
    SmallVector<Value> newYields;
    bool didSomething = false;
    for (auto pair :
         llvm::zip(whileOp.getResults(), term.getArgs(), origAfterArgs)) {
      auto selOp = std::get<1>(pair).getDefiningOp<arith::SelectOp>();
      if (!selOp || selOp.getCondition() != term.getCondition()) {
        newResults.push_back(newYields.size());
        newAfter.push_back(newYields.size());
        newYields.push_back(std::get<1>(pair));
        continue;
      }
      newResults.push_back(newYields.size());
      newYields.push_back(selOp.getFalseValue());
      newAfter.push_back(newYields.size());
      newYields.push_back(selOp.getTrueValue());
      didSomething = true;
    }
    if (!didSomething)
      return failure();

    SmallVector<Type> resultTypes;
    for (auto v : newYields)
      resultTypes.push_back(v.getType());

    auto newWhileOp = rewriter.create<scf::WhileOp>(
        whileOp.getLoc(), resultTypes, whileOp.getInits());

    newWhileOp.getBefore().takeBody(whileOp.getBefore());

    auto *after = rewriter.createBlock(&newWhileOp.getAfter());
    for (auto y : newYields)
      after->addArgument(y.getType(), whileOp.getLoc());

    SmallVector<Value> replacedArgs;
    for (auto idx : newAfter)
      replacedArgs.push_back(after->getArgument(idx));
    rewriter.mergeBlocks(&whileOp.getAfter().front(), after, replacedArgs);

    SmallVector<Value> replacedReturns;
    for (auto idx : newResults)
      replacedReturns.push_back(newWhileOp.getResult(idx));
    rewriter.replaceOp(whileOp, replacedReturns);
    rewriter.setInsertionPoint(term);
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(term, term.getCondition(),
                                                  newYields);
    return success();
  }
};

struct MoveWhileDown3 : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    auto term =
        cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
    SmallVector<unsigned> toErase;
    SmallVector<Value> newOps, condOps;
    SmallVector<BlockArgument> origAfterArgs(
        whileOp.getAfterArguments().begin(), whileOp.getAfterArguments().end());
    SmallVector<Value> returns;
    assert(origAfterArgs.size() == whileOp.getResults().size());
    assert(origAfterArgs.size() == term.getArgs().size());
    for (auto pair :
         llvm::zip(whileOp.getResults(), term.getArgs(), origAfterArgs)) {
      if (std::get<0>(pair).use_empty()) {
        if (std::get<2>(pair).use_empty()) {
          toErase.push_back(std::get<2>(pair).getArgNumber());
          continue;
        }
        // TODO generalize to any non memory effecting op
        if (auto idx =
                std::get<1>(pair).getDefiningOp<MemoryEffectOpInterface>()) {
          if (idx.hasNoEffect() &&
              !llvm::is_contained(newOps, std::get<1>(pair))) {
            Operation *cloned = std::get<1>(pair).getDefiningOp();
            if (!std::get<1>(pair).hasOneUse()) {
              cloned = std::get<1>(pair).getDefiningOp()->clone();
              whileOp.getAfter().front().push_front(cloned);
            } else {
              cloned->moveBefore(&whileOp.getAfter().front().front());
            }
            rewriter.updateRootInPlace(std::get<1>(pair).getDefiningOp(), [&] {
              std::get<2>(pair).replaceAllUsesWith(cloned->getResult(0));
            });
            toErase.push_back(std::get<2>(pair).getArgNumber());
            for (auto &o :
                 llvm::make_early_inc_range(cloned->getOpOperands())) {
              {
                newOps.push_back(o.get());
                o.set(whileOp.getAfter().front().addArgument(o.get().getType(),
                                                             o.get().getLoc()));
              }
            }
            continue;
          }
        }
      }
      condOps.push_back(std::get<1>(pair));
      returns.push_back(std::get<0>(pair));
    }
    if (toErase.size() == 0)
      return failure();

    condOps.append(newOps.begin(), newOps.end());

    llvm::BitVector eraseIndices(whileOp.getAfter().front().getNumArguments());
    for (unsigned i : toErase)
      eraseIndices.set(i);
    rewriter.updateRootInPlace(
        term, [&] { whileOp.getAfter().front().eraseArguments(eraseIndices); });
    rewriter.setInsertionPoint(term);
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(term, term.getCondition(),
                                                  condOps);

    rewriter.setInsertionPoint(whileOp);
    SmallVector<Type> resultTypes;
    for (auto v : condOps)
      resultTypes.push_back(v.getType());

    auto newWhileOp = rewriter.create<scf::WhileOp>(
        whileOp.getLoc(), resultTypes, whileOp.getInits());

    newWhileOp.getBefore().takeBody(whileOp.getBefore());
    newWhileOp.getAfter().takeBody(whileOp.getAfter());

    rewriter.updateRootInPlace(whileOp, [&] {
      for (auto pair : llvm::enumerate(returns))
        pair.value().replaceAllUsesWith(newWhileOp.getResult(pair.index()));
    });

    assert(resultTypes.size() ==
           newWhileOp.getAfter().front().getNumArguments());
    assert(resultTypes.size() == condOps.size());

    rewriter.eraseOp(whileOp);
    return success();
  }
};

// Rewritten from LoopInvariantCodeMotion.cpp
struct WhileLICM : public OpRewritePattern<scf::WhileOp> {
  WhileLICM(AliasAnalysis &aliasAnalysis, MLIRContext *context,
            PatternBenefit benefit = 1)
      : OpRewritePattern<scf::WhileOp>(context, benefit),
        aliasAnalysis(aliasAnalysis) {}

  static bool canBeHoisted(Operation *op,
                           function_ref<bool(Value)> definedOutside,
                           bool isSpeculatable, scf::WhileOp whileOp,
                           AliasAnalysis &aliasAnalysis) {
    // TODO consider requirement of 'isSpeculatable'

    // Check that dependencies are defined outside of loop.
    if (!llvm::all_of(op->getOperands(), definedOutside))
      return false;

    // Check whether this op is side-effect free. If we already know that there
    // can be no side-effects because the surrounding op has claimed so, we can
    // (and have to) skip this step.
    if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      // If the operation doesn't have side effects and doesn't recursively have
      // side effects, it can be hoisted.
      if (memInterface.hasNoEffect() &&
          !op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        return true;

      if (!isReadOnly(op) || isSpeculatable)
        return false;

      SmallVector<MemoryEffects::EffectInstance> whileEffects, opEffects;
      collectEffects(whileOp, whileEffects, /*ignoreBarriers*/ false);
      collectEffects(op, opEffects, /*ignoreBarriers*/ false);

      for (MemoryEffects::EffectInstance &before : opEffects)
        for (MemoryEffects::EffectInstance &after : whileEffects) {
          if (after.getValue() == before.getValue())
            continue;

          AliasResult aliasRes =
              aliasAnalysis.alias(before.getValue(), after.getValue());
          if (aliasRes.isNo())
            continue;

          if (isa<MemoryEffects::Read>(before.getEffect()) &&
              isa<MemoryEffects::Read>(after.getEffect()))
            continue;

          return false;
        }
    } else if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      // If the operation doesn't provide the memory effect interface and it
      // doesn't the recursive side effects we treat it conservatively.
      return false;
    }

    // Recurse into the regions for this op and check whether the contained ops
    // can be hoisted.
    for (auto &region : op->getRegions()) {
      for (auto &block : region)
        for (auto &innerOp : block.without_terminator())
          if (!canBeHoisted(&innerOp, definedOutside, isSpeculatable, whileOp,
                            aliasAnalysis))
            return false;
    }

    return true;
  }

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    // We use two collections here as we need to preserve the order for
    // insertion and this is easiest.
    SmallPtrSet<Operation *, 8> willBeMovedSet;
    SmallVector<Operation *> opsToMove;

    // Helper to check whether an operation is loop invariant wrt. SSA
    // properties.
    auto isDefinedOutsideOfLoop = [&](Value value) {
      auto *definingOp = value.getDefiningOp();
      if (!definingOp) {
        if (auto ba = dyn_cast<BlockArgument>(value))
          definingOp = ba.getOwner()->getParentOp();
        assert(definingOp);
      }
      if (willBeMovedSet.count(definingOp))
        return true;
      return whileOp != definingOp && !whileOp->isAncestor(definingOp);
    };

    // Do not use walk here, as we do not want to go into nested regions and
    // hoist operations from there. These regions might have semantics unknown
    // to this rewriting. If the nested regions are loops, they will have been
    // processed.
    for (auto &block : whileOp.getBefore()) {
      for (auto &iop : block.without_terminator())
        if (canBeHoisted(&iop, isDefinedOutsideOfLoop, false, whileOp,
                         aliasAnalysis)) {
          opsToMove.push_back(&iop);
          willBeMovedSet.insert(&iop);
        }
    }

    for (auto &block : whileOp.getAfter()) {
      for (auto &iop : block.without_terminator())
        if (canBeHoisted(&iop, isDefinedOutsideOfLoop, true, whileOp,
                         aliasAnalysis)) {
          opsToMove.push_back(&iop);
          willBeMovedSet.insert(&iop);
        }
    }

    for (auto *moveOp : opsToMove)
      rewriter.updateRootInPlace(moveOp, [&] { moveOp->moveBefore(whileOp); });

    return success(opsToMove.size() > 0);
  }

private:
  AliasAnalysis &aliasAnalysis;
};

struct RemoveUnusedCondVar : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    auto term =
        cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
    SmallVector<Value> conditions;
    llvm::BitVector eraseArgs(whileOp.getAfter().front().getArguments().size());
    SmallVector<unsigned> keepArgs;
    SmallVector<Type> tys;
    unsigned i = 0;
    std::map<void *, unsigned> valueOffsets;
    std::map<unsigned, unsigned> resultOffsets;
    SmallVector<Value> resultArgs;
    for (auto pair :
         llvm::zip(term.getArgs(), whileOp.getAfter().front().getArguments(),
                   whileOp.getResults())) {
      auto arg = std::get<0>(pair);
      auto afterArg = std::get<1>(pair);
      auto res = std::get<2>(pair);
      if (afterArg.use_empty() && res.use_empty())
        eraseArgs.set((unsigned)i);
      else if (valueOffsets.find(arg.getAsOpaquePointer()) !=
               valueOffsets.end()) {
        resultOffsets[i] = valueOffsets[arg.getAsOpaquePointer()];
        afterArg.replaceAllUsesWith(
            resultArgs[valueOffsets[arg.getAsOpaquePointer()]]);
        eraseArgs.set((unsigned)i);
      } else {
        valueOffsets[arg.getAsOpaquePointer()] = keepArgs.size();
        resultOffsets[i] = keepArgs.size();
        resultArgs.push_back(afterArg);
        conditions.push_back(arg);
        keepArgs.push_back((unsigned)i);
        tys.push_back(arg.getType());
      }
      i++;
    }
    assert(i == whileOp.getAfter().front().getArguments().size());

    if (!eraseArgs.none()) {
      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<scf::ConditionOp>(term, term.getCondition(),
                                                    conditions);

      rewriter.setInsertionPoint(whileOp);
      auto newWhileOp = rewriter.create<scf::WhileOp>(whileOp.getLoc(), tys,
                                                      whileOp.getInits());

      newWhileOp.getBefore().takeBody(whileOp.getBefore());
      newWhileOp.getAfter().takeBody(whileOp.getAfter());
      for (auto pair : resultOffsets)
        whileOp.getResult(pair.first)
            .replaceAllUsesWith(newWhileOp.getResult(pair.second));
      rewriter.eraseOp(whileOp);
      newWhileOp.getAfter().front().eraseArguments(eraseArgs);
      return success();
    }
    return failure();
  }
};

struct MoveSideEffectFreeWhile : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp whileOp,
                                PatternRewriter &rewriter) const override {
    auto term =
        cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
    SmallVector<Value> conditions(term.getArgs().begin(), term.getArgs().end());
    bool didSomething = false;
    unsigned i = 0;
    for (auto arg : term.getArgs()) {
      if (auto IC = arg.getDefiningOp<arith::IndexCastOp>()) {
        if (arg.hasOneUse() && whileOp.getResult(i).use_empty()) {
          auto rep = whileOp.getAfter().front().addArgument(
              IC->getOperand(0).getType(), IC->getOperand(0).getLoc());
          IC->moveBefore(&whileOp.getAfter().front(),
                         whileOp.getAfter().front().begin());
          conditions.push_back(IC.getIn());
          IC.getInMutable().assign(rep);
          whileOp.getAfter().front().getArgument(i).replaceAllUsesWith(
              IC->getResult(0));
          didSomething = true;
        }
      }
      i++;
    }
    if (didSomething) {
      SmallVector<Type> tys;
      for (auto arg : conditions)
        tys.push_back(arg.getType());

      auto newWhileOp = rewriter.create<scf::WhileOp>(whileOp.getLoc(), tys,
                                                      whileOp.getInits());
      newWhileOp.getBefore().takeBody(whileOp.getBefore());
      newWhileOp.getAfter().takeBody(whileOp.getAfter());
      unsigned j = 0;
      for (auto res : whileOp.getResults()) {
        res.replaceAllUsesWith(newWhileOp.getResult(j));
        j++;
      }
      rewriter.eraseOp(whileOp);
      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<scf::ConditionOp>(term, term.getCondition(),
                                                    conditions);
      return success();
    }
    return failure();
  }
};

struct SubToAdd : public OpRewritePattern<arith::SubIOp> {
  using OpRewritePattern<arith::SubIOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::SubIOp op,
                                PatternRewriter &rewriter) const override {
    if (auto cop = op.getOperand(1).getDefiningOp<arith::ConstantIntOp>()) {
      rewriter.replaceOpWithNewOp<arith::AddIOp>(
          op, op.getOperand(0),
          rewriter.create<arith::ConstantIntOp>(cop.getLoc(), -cop.value(),
                                                cop.getType()));
      return success();
    }
    return failure();
  }
};

struct ReturnSq : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern<func::ReturnOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::ReturnOp subOp,
                                PatternRewriter &rewriter) const override {
    bool didSomething = false;
    SmallVector<Operation *> toErase;
    for (auto iter = subOp->getBlock()->rbegin();
         iter != subOp->getBlock()->rend() && &*iter != subOp; iter++) {
      didSomething = true;
      toErase.push_back(&*iter);
    }

    for (auto *op : toErase)
      rewriter.eraseOp(op);

    return success(didSomething);
  }
};

void CanonicalizeFor::runOnOperation() {
  MLIRContext &ctx = getContext();
  RewritePatternSet rpl(&ctx);
  AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
  aliasAnalysis.addAnalysisImplementation(sycl::AliasAnalysis(relaxedAliasing));

  rpl.add<WhileLICM>(aliasAnalysis, &ctx)
      .add<PropagateInLoopBody, ForOpInductionReplacement, RemoveUnusedArgs,
           MoveWhileToFor, RemoveWhileSelect, MoveWhileDown, MoveWhileDown2,
           ReplaceRedundantArgs, MoveWhileAndDown, MoveWhileDown3,
           MoveWhileInvariantIfResult, WhileLogicalNegation, SubToAdd,
           WhileCmpOffset, RemoveUnusedCondVar, ReturnSq,
           MoveSideEffectFreeWhile>(&ctx);

  GreedyRewriteConfig config;
  config.maxIterations = 247;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(rpl), config);
}

std::unique_ptr<Pass> polygeist::createCanonicalizeForPass() {
  return std::make_unique<CanonicalizeFor>();
}
