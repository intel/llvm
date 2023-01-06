#include "PassDetails.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include <mlir/Dialect/Arith/IR/Arith.h>

using namespace mlir;
using namespace mlir::scf;
using namespace mlir::func;
using namespace mlir::arith;
using namespace polygeist;

namespace {
struct CanonicalizeFor : public SCFCanonicalizeForBase<CanonicalizeFor> {
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
    bool matched = false;
    for (auto it : llvm::zip(forOp.getIterOperands(), forOp.getRegionIterArgs(),
                             yieldOp.getOperands())) {
      Value iterOperand = std::get<0>(it);
      Value regionArg = std::get<1>(it);
      Value yieldOperand = std::get<2>(it);

      Operation *op = iterOperand.getDefiningOp();
      if (op && (op->getNumResults() == 1) && (iterOperand == yieldOperand)) {
        regionArg.replaceAllUsesWith(op->getResult(0));
        matched = true;
      }
    }
    return success(matched);
  }
};

struct ForOpInductionReplacement : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    // Defer until after the step is a constant, if possible
    if (auto icast = forOp.getStep().getDefiningOp<IndexCastOp>())
      if (matchPattern(icast.getIn(), m_Constant()))
        return failure();
    bool canonicalize = false;
    Block &block = forOp.getRegion().front();
    auto yieldOp = cast<scf::YieldOp>(block.getTerminator());

    for (auto it : llvm::zip(forOp.getIterOperands(),   // iter from outside
                             forOp.getRegionIterArgs(), // iter inside region
                             forOp.getResults(),        // op results
                             yieldOp.getOperands()      // iter yield
                             )) {

      AddIOp addOp = std::get<3>(it).getDefiningOp<AddIOp>();
      if (!addOp)
        continue;

      if (addOp.getOperand(0) != std::get<1>(it))
        continue;

      if (!addOp.getOperand(1).getParentRegion()->isAncestor(
              forOp->getParentRegion()))
        continue;

      bool sameValue = addOp.getOperand(1) == forOp.getStep();

      APInt rattr;
      APInt sattr;
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
        Value replacement = rewriter.create<SubIOp>(
            forOp.getLoc(), forOp.getInductionVar(), forOp.getLowerBound());

        if (!sameValue)
          replacement = rewriter.create<DivUIOp>(forOp.getLoc(), replacement,
                                                 forOp.getStep());

        if (!sameValue) {
          Value step = addOp.getOperand(1);

          if (!step.getType().isa<IndexType>()) {
            step = rewriter.create<IndexCastOp>(forOp.getLoc(),
                                                replacement.getType(), step);
          }

          replacement =
              rewriter.create<MulIOp>(forOp.getLoc(), replacement, step);
        }

        if (!init.getType().isa<IndexType>()) {
          init = rewriter.create<IndexCastOp>(forOp.getLoc(),
                                              replacement.getType(), init);
        }

        replacement =
            rewriter.create<AddIOp>(forOp.getLoc(), init, replacement);

        if (!std::get<1>(it).getType().isa<IndexType>()) {
          replacement = rewriter.create<IndexCastOp>(
              forOp.getLoc(), std::get<1>(it).getType(), replacement);
        }

        rewriter.updateRootInPlace(
            forOp, [&] { std::get<1>(it).replaceAllUsesWith(replacement); });
        canonicalize = true;
      }

      if (!std::get<2>(it).use_empty()) {
        Value init = std::get<0>(it);
        rewriter.setInsertionPoint(forOp);
        Value replacement = rewriter.create<SubIOp>(
            forOp.getLoc(), forOp.getUpperBound(), forOp.getLowerBound());

        if (!sameValue)
          replacement = rewriter.create<DivUIOp>(forOp.getLoc(), replacement,
                                                 forOp.getStep());

        if (!sameValue) {
          Value step = addOp.getOperand(1);

          if (!step.getType().isa<IndexType>()) {
            step = rewriter.create<IndexCastOp>(forOp.getLoc(),
                                                replacement.getType(), step);
          }

          replacement =
              rewriter.create<MulIOp>(forOp.getLoc(), replacement, step);
        }

        if (!init.getType().isa<IndexType>()) {
          init = rewriter.create<IndexCastOp>(forOp.getLoc(),
                                              replacement.getType(), init);
        }

        replacement =
            rewriter.create<AddIOp>(forOp.getLoc(), init, replacement);

        if (!std::get<1>(it).getType().isa<IndexType>()) {
          replacement = rewriter.create<IndexCastOp>(
              forOp.getLoc(), std::get<1>(it).getType(), replacement);
        }

        rewriter.updateRootInPlace(
            forOp, [&] { std::get<2>(it).replaceAllUsesWith(replacement); });
        canonicalize = true;
      }
    }

    return success(canonicalize);
  }
};

/// Remove unused iterator operands.
// TODO: BlockAndValueMapping for indvar.
struct RemoveUnusedArgs : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 2> usedBlockArgs;
    SmallVector<OpResult, 2> usedResults;
    SmallVector<Value, 2> usedOperands;

    unsigned i = 0;
    // if the block argument or the result at the
    // same index position have uses do not eliminate.
    for (auto blockArg : op.getRegionIterArgs()) {
      if ((!blockArg.use_empty()) || (!op.getResult(i).use_empty())) {
        usedOperands.push_back(op.getOperand(op.getNumControlOperands() + i));
        usedResults.push_back(op->getOpResult(i));
        usedBlockArgs.push_back(blockArg);
      }
      i++;
    }

    // no work to do.
    if (usedOperands.size() == op.getIterOperands().size())
      return failure();

    auto newForOp =
        rewriter.create<ForOp>(op.getLoc(), op.getLowerBound(),
                               op.getUpperBound(), op.getStep(), usedOperands);

    if (!newForOp.getBody()->empty())
      rewriter.eraseOp(newForOp.getBody()->getTerminator());

    newForOp.getBody()->getOperations().splice(
        newForOp.getBody()->getOperations().begin(),
        op.getBody()->getOperations());

    rewriter.updateRootInPlace(op, [&] {
      op.getInductionVar().replaceAllUsesWith(newForOp.getInductionVar());
      for (auto pair : llvm::zip(usedBlockArgs, newForOp.getRegionIterArgs())) {
        std::get<0>(pair).replaceAllUsesWith(std::get<1>(pair));
      }
    });

    // adjust return.
    auto yieldOp = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
    SmallVector<Value, 2> usedYieldOperands{};
    llvm::transform(usedResults, std::back_inserter(usedYieldOperands),
                    [&](OpResult result) {
                      return yieldOp.getOperand(result.getResultNumber());
                    });
    rewriter.setInsertionPoint(yieldOp);
    rewriter.replaceOpWithNewOp<YieldOp>(yieldOp, usedYieldOperands);

    // Replace the operation's results with the new ones.
    SmallVector<Value, 4> repResults(op.getNumResults());
    for (auto en : llvm::enumerate(usedResults))
      repResults[en.value().cast<OpResult>().getResultNumber()] =
          newForOp.getResult(en.index());

    rewriter.replaceOp(op, repResults);
    return success();
  }
};

struct ReplaceRedundantArgs : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp op,
                                PatternRewriter &rewriter) const override {
    auto yieldOp = cast<scf::YieldOp>(op.getBody()->getTerminator());
    bool replaced = false;
    unsigned i = 0;
    for (auto blockArg : op.getRegionIterArgs()) {
      for (unsigned j = 0; j < i; j++) {
        if (op.getOperand(op.getNumControlOperands() + i) ==
                op.getOperand(op.getNumControlOperands() + j) &&
            yieldOp.getOperand(i) == yieldOp.getOperand(j)) {

          rewriter.updateRootInPlace(op, [&] {
            op.getResult(i).replaceAllUsesWith(op.getResult(j));
            blockArg.replaceAllUsesWith(op.getRegionIterArgs()[j]);
          });
          replaced = true;
          goto skip;
        }
      }
    skip:
      i++;
    }

    return success(replaced);
  }
};

/*
+struct RemoveNotIf : public OpRewritePattern<IfOp> {
+  using OpRewritePattern<IfOp>::OpRewritePattern;
+
+  LogicalResult matchAndRewrite(IfOp op,
+                                PatternRewriter &rewriter) const override {
+    // Replace the operation if only a subset of its results have uses.
+    if (op.getNumResults() == 0)
+      return failure();
+
+    auto trueYield =
cast<scf::YieldOp>(op.thenRegion().back().getTerminator());
+    auto falseYield =
+        cast<scf::YieldOp>(op.thenRegion().back().getTerminator());
+
+    rewriter.setInsertionPoint(op->getBlock(),
+                               op.getOperation()->getIterator());
+    bool changed = false;
+    for (auto tup :
+         llvm::zip(trueYield.results(), falseYield.results(), op.results())) {
+      if (!std::get<0>(tup).getType().isInteger(1))
+        continue;
+      if (auto top = std::get<0>(tup).getDefiningOp<ConstantOp>()) {
+        if (auto fop = std::get<1>(tup).getDefiningOp<ConstantOp>()) {
+          if (top.getValue().cast<IntegerAttr>().getValue() == 0 &&
+              fop.getValue().cast<IntegerAttr>().getValue() == 1) {
+
+            for (OpOperand &use :
+                 llvm::make_early_inc_range(std::get<2>(tup).getUses())) {
+              changed = true;
+              rewriter.updateRootInPlace(use.getOwner(), [&]() {
+                use.set(rewriter.create<XOrOp>(op.getLoc(), op.condition()));
+              });
+            }
+          }
+          if (top.getValue().cast<IntegerAttr>().getValue() == 1 &&
+              fop.getValue().cast<IntegerAttr>().getValue() == 0) {
+            for (OpOperand &use :
+                 llvm::make_early_inc_range(std::get<2>(tup).getUses())) {
+              changed = true;
+              rewriter.updateRootInPlace(use.getOwner(),
+                                         [&]() { use.set(op.condition()); });
+            }
+          }
+        }
+      }
+    }
+    return changed ? success() : failure();
+  }
+};
+struct RemoveBoolean : public OpRewritePattern<IfOp> {
+  using OpRewritePattern<IfOp>::OpRewritePattern;
+
+  LogicalResult matchAndRewrite(IfOp op,
+                                PatternRewriter &rewriter) const override {
+    bool changed = false;
+
+    if (llvm::all_of(op.results(), [](Value v) {
+          return v.getType().isa<IntegerType>() &&
+                 v.getType().cast<IntegerType>().getWidth() == 1;
+        })) {
+      if (op.thenRegion().getBlocks().size() == 1 &&
+          op.elseRegion().getBlocks().size() == 1) {
+        while (isa<CmpIOp>(op.thenRegion().front().front())) {
+          op.thenRegion().front().front().moveBefore(op);
+          changed = true;
+        }
+        while (isa<CmpIOp>(op.elseRegion().front().front())) {
+          op.elseRegion().front().front().moveBefore(op);
+          changed = true;
+        }
+        if (op.thenRegion().front().getOperations().size() == 1 &&
+            op.elseRegion().front().getOperations().size() == 1) {
+          auto yop1 =
+              cast<scf::YieldOp>(op.thenRegion().front().getTerminator());
+          auto yop2 =
+              cast<scf::YieldOp>(op.elseRegion().front().getTerminator());
+          size_t idx = 0;
+
+          auto c1 = (mlir::Value)rewriter.create<ConstantOp>(
+              op.getLoc(), op.condition().getType(),
+              rewriter.getIntegerAttr(op.condition().getType(), 1));
+          auto notcond = (mlir::Value)rewriter.create<mlir::XOrOp>(
+              op.getLoc(), op.condition(), c1);
+
+          std::vector<Value> replacements;
+          for (auto res : op.results()) {
+            auto rep = rewriter.create<OrOp>(
+                op.getLoc(),
+                rewriter.create<AndOp>(op.getLoc(), op.condition(),
+                                       yop1.results()[idx]),
+                rewriter.create<AndOp>(op.getLoc(), notcond,
+                                       yop2.results()[idx]));
+            replacements.push_back(rep);
+            idx++;
+          }
+          rewriter.replaceOp(op, replacements);
+          // op.erase();
+          return success();
+        }
+      }
+    }
+
+    if (op.thenRegion().getBlocks().size() == 1 &&
+        op.elseRegion().getBlocks().size() == 1 &&
+        op.thenRegion().front().getOperations().size() == 1 &&
+        op.elseRegion().front().getOperations().size() == 1) {
+      auto yop1 = cast<scf::YieldOp>(op.thenRegion().front().getTerminator());
+      auto yop2 = cast<scf::YieldOp>(op.elseRegion().front().getTerminator());
+      size_t idx = 0;
+
+      std::vector<Value> replacements;
+      for (auto res : op.results()) {
+        auto rep =
+            rewriter.create<SelectOp>(op.getLoc(), op.condition(),
+                                      yop1.results()[idx],
yop2.results()[idx]);
+        replacements.push_back(rep);
+        idx++;
+      }
+      rewriter.replaceOp(op, replacements);
+      return success();
+    }
+    return changed ? success() : failure();
+  }
+};
*/

bool isTopLevelArgValue(Value value, Region *region) {
  if (auto arg = value.dyn_cast<BlockArgument>())
    return arg.getParentRegion() == region;
  return false;
}

bool isBlockArg(Value value) {
  if (auto arg = value.dyn_cast<BlockArgument>())
    return true;
  return false;
}

bool dominateWhile(Value value, WhileOp loop) {
  Operation *op = value.getDefiningOp();
  assert(op && "expect non-null");
  DominanceInfo dom(loop);
  return dom.properlyDominates(op, loop);
}

bool canMoveOpOutsideWhile(Operation *op, WhileOp loop) {
  DominanceInfo dom(loop);
  for (auto operand : op->getOperands()) {
    if (!dom.properlyDominates(operand, loop))
      return false;
  }
  return true;
}

struct WhileToForHelper {
  WhileOp loop;
  CmpIOp cmpIOp;
  Value step;
  Value lb;
  bool lb_addOne;
  Value ub;
  bool ub_addOne;
  bool ub_cloneMove;
  bool negativeStep;
  AddIOp addIOp;
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
    indVar = cmpIOp.getLhs().dyn_cast<BlockArgument>();
    Type extType = nullptr;
    // todo handle ext
    if (auto ext = cmpIOp.getLhs().getDefiningOp<ExtSIOp>()) {
      indVar = ext.getIn().dyn_cast<BlockArgument>();
      extType = ext.getType();
    }
    // Condition is not the same as an induction variable
    {
      if (!indVar) {
        return false;
      }

      if (indVar.getOwner() != &loop.getBefore().front())
        return false;
    }

    // Before region contains more than just the comparison
    {
      size_t size = loop.getBefore().front().getOperations().size();
      if (extType)
        size--;
      if (!sizeCheck)
        size--;
      if (size != 2) {
        return false;
      }
    }

    SmallVector<size_t, 2> afterArgs;
    for (auto pair : llvm::enumerate(condOp.getArgs())) {
      if (pair.value() == indVar)
        afterArgs.push_back(pair.index());
    }

    auto endYield = cast<YieldOp>(loop.getAfter().back().getTerminator());

    // Check that the block argument is actually an induction var:
    //   Namely, its next value adds to the previous with an invariant step.
    addIOp =
        endYield.getResults()[indVar.getArgNumber()].getDefiningOp<AddIOp>();
    if (!addIOp) {
      if (auto ifOp = endYield.getResults()[indVar.getArgNumber()]
                          .getDefiningOp<IfOp>()) {
        if (ifOp.getCondition() == lookThrough) {
          for (auto r : llvm::enumerate(ifOp.getResults())) {
            if (r.value() == endYield.getResults()[indVar.getArgNumber()]) {
              addIOp = ifOp.thenYield()
                           .getOperand(r.index())
                           .getDefiningOp<AddIOp>();
              break;
            }
          }
        }
      } else if (auto selOp = endYield.getResults()[indVar.getArgNumber()]
                                  .getDefiningOp<SelectOp>()) {
        if (selOp.getCondition() == lookThrough)
          addIOp = selOp.getTrueValue().getDefiningOp<AddIOp>();
      }
    }
    if (!addIOp) {
      return false;
    }

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
      if (loop->isAncestor(op)) {
        return false;
      }
    }

    negativeStep = false;
    if (auto cop = step.getDefiningOp<ConstantIntOp>()) {
      if (cop.value() < 0) {
        negativeStep = true;
      }
    } else if (auto cop = step.getDefiningOp<ConstantIndexOp>()) {
      if (cop.value() < 0)
        negativeStep = true;
    }

    if (!negativeStep)
      lb = loop.getOperand(indVar.getArgNumber());
    else {
      ub = loop.getOperand(indVar.getArgNumber());
      ub_addOne = true;
    }

    if (isBlockArg(cmpIOp.getRhs()) || dominateWhile(cmpIOp.getRhs(), loop)) {
      switch (cmpIOp.getPredicate()) {
      case CmpIPredicate::slt:
      case CmpIPredicate::ult: {
        ub = cmpIOp.getRhs();
        break;
      }
      case CmpIPredicate::ule:
      case CmpIPredicate::sle: {
        ub = cmpIOp.getRhs();
        ub_addOne = true;
        break;
      }
      case CmpIPredicate::uge:
      case CmpIPredicate::sge: {
        lb = cmpIOp.getRhs();
        break;
      }

      case CmpIPredicate::ugt:
      case CmpIPredicate::sgt: {
        lb = cmpIOp.getRhs();
        lb_addOne = true;
        break;
      }
      case CmpIPredicate::eq:
      case CmpIPredicate::ne: {
        return false;
      }
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
          rewriter.create<ConstantIntOp>(loop.getLoc(), 1, lb.getType());
      lb = rewriter.create<AddIOp>(loop.getLoc(), lb, one);
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
          rewriter.create<ConstantIntOp>(loop.getLoc(), 1, ub.getType());
      ub = rewriter.create<AddIOp>(loop.getLoc(), ub, one);
    }

    if (negativeStep) {
      if (auto cop = step.getDefiningOp<ConstantIntOp>()) {
        step = rewriter.create<ConstantIndexOp>(cop.getLoc(), -cop.value());
      } else {
        auto cop2 = step.getDefiningOp<ConstantIndexOp>();
        step = rewriter.create<ConstantIndexOp>(cop2.getLoc(), -cop2.value());
      }
    }

    ub = rewriter.create<IndexCastOp>(loop.getLoc(),
                                      IndexType::get(loop.getContext()), ub);
    lb = rewriter.create<IndexCastOp>(loop.getLoc(),
                                      IndexType::get(loop.getContext()), lb);
    step = rewriter.create<IndexCastOp>(
        loop.getLoc(), IndexType::get(loop.getContext()), step);
  }
};

struct MoveWhileToFor : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp loop,
                                PatternRewriter &rewriter) const override {
    auto condOp = loop.getConditionOp();
    SmallVector<Value, 2> results = {condOp.getArgs()};
    WhileToForHelper helper;
    helper.loop = loop;
    helper.cmpIOp = condOp.getCondition().getDefiningOp<CmpIOp>();
    if (!helper.cmpIOp) {
      return failure();
    }
    if (!helper.computeLegality(/*sizeCheck*/ true))
      return failure();
    helper.prepareFor(rewriter);

    // input of the for goes the input of the scf::while plus the output taken
    // from the conditionOp.
    SmallVector<Value, 8> forArgs;
    forArgs.append(loop.getInits().begin(), loop.getInits().end());

    for (Value arg : condOp.getArgs()) {
      Type cst = nullptr;
      if (auto idx = arg.getDefiningOp<IndexCastOp>()) {
        cst = idx.getType();
        arg = idx.getIn();
      }
      Value res;
      if (isTopLevelArgValue(arg, &loop.getBefore())) {
        auto blockArg = arg.cast<BlockArgument>();
        auto pos = blockArg.getArgNumber();
        res = loop.getInits()[pos];
      } else
        res = arg;
      if (cst) {
        res = rewriter.create<IndexCastOp>(rewriter.getUnknownLoc(), cst, res);
      }
      forArgs.push_back(res);
    }

    auto forloop = rewriter.create<scf::ForOp>(loop.getLoc(), helper.lb,
                                               helper.ub, helper.step, forArgs);

    if (!forloop.getBody()->empty())
      rewriter.eraseOp(forloop.getBody()->getTerminator());

    auto oldYield = cast<scf::YieldOp>(loop.getAfter().front().getTerminator());

    rewriter.updateRootInPlace(loop, [&] {
      for (auto pair :
           llvm::zip(loop.getAfter().getArguments(), condOp.getArgs())) {
        std::get<0>(pair).replaceAllUsesWith(std::get<1>(pair));
      }
    });
    loop.getAfter().front().eraseArguments([](BlockArgument) { return true; });

    SmallVector<Value, 2> yieldOperands;
    for (auto oldYieldArg : oldYield.getResults())
      yieldOperands.push_back(oldYieldArg);

    BlockAndValueMapping outmap;
    outmap.map(loop.getBefore().getArguments(), yieldOperands);
    for (auto arg : condOp.getArgs())
      yieldOperands.push_back(outmap.lookupOrDefault(arg));

    rewriter.setInsertionPoint(oldYield);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(oldYield, yieldOperands);

    size_t pos = loop.getInits().size();

    rewriter.updateRootInPlace(loop, [&] {
      for (auto pair : llvm::zip(loop.getBefore().getArguments(),
                                 forloop.getRegionIterArgs().drop_back(pos))) {
        std::get<0>(pair).replaceAllUsesWith(std::get<1>(pair));
      }
    });

    forloop.getBody()->getOperations().splice(
        forloop.getBody()->getOperations().begin(),
        loop.getAfter().front().getOperations());

    SmallVector<Value, 2> replacements;
    replacements.append(forloop.getResults().begin() + pos,
                        forloop.getResults().end());

    rewriter.replaceOp(loop, replacements);
    return success();
  }
};

// If and and with something is preventing creating a for
// move the and into the after body guarded by an if
struct MoveWhileAndDown : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp loop,
                                PatternRewriter &rewriter) const override {
    auto condOp = loop.getConditionOp();
    auto andIOp = condOp.getCondition().getDefiningOp<AndIOp>();
    if (!andIOp)
      return failure();
    for (int i = 0; i < 2; i++) {
      WhileToForHelper helper;
      helper.loop = loop;
      helper.cmpIOp = andIOp->getOperand(i).getDefiningOp<CmpIOp>();
      if (!helper.cmpIOp)
        continue;

      YieldOp oldYield = cast<YieldOp>(loop.getAfter().front().getTerminator());

      Value extraCmp = andIOp->getOperand(1 - i);
      Value lookThrough = nullptr;
      if (auto BA = extraCmp.dyn_cast<BlockArgument>()) {
        lookThrough = oldYield.getOperand(BA.getArgNumber());
      }
      if (!helper.computeLegality(/*sizeCheck*/ false, lookThrough))
        continue;

      SmallVector<BlockArgument, 2> origBeforeArgs(
          loop.getBeforeArguments().begin(), loop.getBeforeArguments().end());

      SmallVector<BlockArgument, 2> origAfterArgs(
          loop.getAfterArguments().begin(), loop.getAfterArguments().end());

      BlockAndValueMapping preMap;
      for (auto tup : llvm::zip(origBeforeArgs, loop.getInits()))
        preMap.map(std::get<0>(tup), std::get<1>(tup));
      for (auto &op : loop.getBefore().front()) {
        if (&op == condOp)
          break;
        preMap.map(op.getResults(), rewriter.clone(op, preMap)->getResults());
      }
      IfOp unroll = rewriter.create<IfOp>(loop.getLoc(), loop.getResultTypes(),
                                          preMap.lookup(condOp.getCondition()));

      if (unroll.getThenRegion().getBlocks().size())
        rewriter.eraseBlock(unroll.thenBlock());
      rewriter.createBlock(&unroll.getThenRegion());
      rewriter.createBlock(&unroll.getElseRegion());

      rewriter.setInsertionPointToEnd(unroll.elseBlock());
      SmallVector<Value> unrollYield;
      for (auto v : condOp.getArgs())
        unrollYield.push_back(preMap.lookup(v));
      rewriter.create<YieldOp>(loop.getLoc(), unrollYield);
      rewriter.setInsertionPointToEnd(unroll.thenBlock());

      SmallVector<Value, 2> nextInits(unrollYield.begin(), unrollYield.end());
      Value falsev =
          rewriter.create<ConstantIntOp>(loop.getLoc(), 0, extraCmp.getType());
      Value truev =
          rewriter.create<ConstantIntOp>(loop.getLoc(), 1, extraCmp.getType());
      nextInits.push_back(truev);
      nextInits.push_back(loop.getInits()[helper.indVar.getArgNumber()]);

      SmallVector<Type> resTys;
      for (auto a : nextInits)
        resTys.push_back(a.getType());

      auto nop = rewriter.create<WhileOp>(loop.getLoc(), resTys, nextInits);
      rewriter.createBlock(&nop.getBefore());
      SmallVector<Value> newBeforeYieldArgs;
      for (auto a : origAfterArgs) {
        auto arg = nop.getBefore().addArgument(a.getType(), a.getLoc());
        newBeforeYieldArgs.push_back(arg);
      }
      Value notExited = nop.getBefore().front().addArgument(extraCmp.getType(),
                                                            loop.getLoc());
      newBeforeYieldArgs.push_back(notExited);

      Value trueInd = nop.getBefore().front().addArgument(
          helper.indVar.getType(), loop.getLoc());
      newBeforeYieldArgs.push_back(trueInd);

      {
        BlockAndValueMapping postMap;
        postMap.map(helper.indVar, trueInd);
        auto newCmp = cast<CmpIOp>(rewriter.clone(*helper.cmpIOp, postMap));
        rewriter.create<ConditionOp>(condOp.getLoc(), newCmp,
                                     newBeforeYieldArgs);
      }

      rewriter.createBlock(&nop.getAfter());
      SmallVector<Value> postElseYields;
      for (auto a : origAfterArgs) {
        auto arg = nop.getAfter().front().addArgument(a.getType(), a.getLoc());
        postElseYields.push_back(arg);
        a.replaceAllUsesWith(arg);
      }
      SmallVector<Type, 4> resultTypes(loop.getResultTypes());
      resultTypes.push_back(notExited.getType());
      notExited = nop.getAfter().front().addArgument(notExited.getType(),
                                                     loop.getLoc());

      trueInd =
          nop.getAfter().front().addArgument(trueInd.getType(), loop.getLoc());

      IfOp guard = rewriter.create<IfOp>(loop.getLoc(), resultTypes, notExited);
      if (guard.getThenRegion().getBlocks().size())
        rewriter.eraseBlock(guard.thenBlock());
      Block *post = rewriter.splitBlock(&loop.getAfter().front(),
                                        loop.getAfter().front().begin());
      rewriter.createBlock(&guard.getThenRegion());
      rewriter.createBlock(&guard.getElseRegion());
      rewriter.mergeBlocks(post, guard.thenBlock());

      {
        BlockAndValueMapping postMap;
        for (auto tup : llvm::zip(origBeforeArgs, oldYield.getOperands())) {
          postMap.map(std::get<0>(tup), std::get<1>(tup));
        }
        rewriter.setInsertionPoint(oldYield);
        for (auto &op : loop.getBefore().front()) {
          if (&op == condOp)
            break;
          postMap.map(op.getResults(),
                      rewriter.clone(op, postMap)->getResults());
        }
        SmallVector<Value> postIfYields;
        for (auto a : condOp.getArgs()) {
          postIfYields.push_back(postMap.lookup(a));
        }
        postIfYields.push_back(postMap.lookup(extraCmp));
        oldYield->setOperands(postIfYields);
      }

      rewriter.setInsertionPointToEnd(guard.elseBlock());
      postElseYields.push_back(falsev);
      rewriter.create<YieldOp>(loop.getLoc(), postElseYields);

      rewriter.setInsertionPointToEnd(&nop.getAfter().front());
      SmallVector<Value> postAfter(guard.getResults());
      BlockAndValueMapping postMap;
      postMap.map(helper.indVar, trueInd);
      postMap.map(postElseYields[helper.afterArgIdx], trueInd);
      assert(helper.addIOp.getLhs() == postElseYields[helper.afterArgIdx] ||
             helper.addIOp.getRhs() == postElseYields[helper.afterArgIdx]);
      postAfter.push_back(
          cast<AddIOp>(rewriter.clone(*helper.addIOp, postMap)));
      rewriter.create<YieldOp>(loop.getLoc(), postAfter);

      rewriter.setInsertionPointToEnd(unroll.thenBlock());
      rewriter.create<YieldOp>(
          loop.getLoc(), nop.getResults().take_front(loop.getResults().size()));

      rewriter.replaceOp(loop, unroll.getResults());

      return success();
    }

    return failure();
  }
};

struct MoveWhileDown : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto term = cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    if (auto ifOp = term.getCondition().getDefiningOp<scf::IfOp>()) {
      if (ifOp.getNumResults() != term.getArgs().size() + 1)
        return failure();
      if (ifOp.getResult(0) != term.getCondition())
        return failure();
      for (size_t i = 1; i < ifOp.getNumResults(); ++i) {
        if (ifOp.getResult(i) != term.getArgs()[i - 1])
          return failure();
      }
      auto yield1 =
          cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
      auto yield2 =
          cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
      if (auto cop = yield1.getOperand(0).getDefiningOp<ConstantIntOp>()) {
        if (cop.value() == 0)
          return failure();
      } else
        return failure();
      if (auto cop = yield2.getOperand(0).getDefiningOp<ConstantIntOp>()) {
        if (cop.value() != 0)
          return failure();
      } else
        return failure();
      if (ifOp.getElseRegion().front().getOperations().size() != 1)
        return failure();
      op.getAfter().front().getOperations().splice(
          op.getAfter().front().begin(),
          ifOp.getThenRegion().front().getOperations());
      rewriter.updateRootInPlace(term, [&] {
        term.getConditionMutable().assign(ifOp.getCondition());
      });
      SmallVector<Value, 2> args;
      for (size_t i = 1; i < yield2.getNumOperands(); ++i) {
        args.push_back(yield2.getOperand(i));
      }
      rewriter.updateRootInPlace(term,
                                 [&] { term.getArgsMutable().assign(args); });
      rewriter.eraseOp(yield2);
      rewriter.eraseOp(ifOp);

      for (size_t i = 0; i < op.getAfter().front().getNumArguments(); ++i) {
        op.getAfter().front().getArgument(i).replaceAllUsesWith(
            yield1.getOperand(i + 1));
      }
      rewriter.eraseOp(yield1);
      // TODO move operands from begin to after
      SmallVector<Value> todo(op.getBefore().front().getArguments().begin(),
                              op.getBefore().front().getArguments().end());
      for (auto &op : op.getBefore().front()) {
        for (auto res : op.getResults()) {
          todo.push_back(res);
        }
      }

      rewriter.updateRootInPlace(op, [&] {
        for (auto val : todo) {
          auto na =
              op.getAfter().front().addArgument(val.getType(), op->getLoc());
          val.replaceUsesWithIf(na, [&](OpOperand &u) -> bool {
            return op.getAfter().isAncestor(u.getOwner()->getParentRegion());
          });
          args.push_back(val);
        }
      });

      rewriter.updateRootInPlace(term,
                                 [&] { term.getArgsMutable().assign(args); });

      SmallVector<Type, 4> tys;
      for (auto a : args)
        tys.push_back(a.getType());

      auto op2 = rewriter.create<WhileOp>(op.getLoc(), tys, op.getInits());
      op2.getBefore().takeBody(op.getBefore());
      op2.getAfter().takeBody(op.getAfter());
      SmallVector<Value, 4> replacements;
      for (auto a : op2.getResults()) {
        if (replacements.size() == op.getResults().size())
          break;
        replacements.push_back(a);
      }
      rewriter.replaceOp(op, replacements);
      return success();
    }
    return failure();
  }
};

// Given code of the structure
// scf.while ()
//    ...
//    %z = if (%c) {
//       %i1 = ..
//       ..
//    } else {
//    }
//    condition (%c) %z#0 ..
//  } loop {
//    ...
//  }
// Move the body of the if into the lower loo

struct MoveWhileDown2 : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  /// Populates `crossing` with values (op results) that are defined in the same
  /// block as `op` and above it, and used by at least one op in the same block
  /// below `op`. Uses may be in nested regions.
  static void findValuesUsedBelow(IfOp op, llvm::SetVector<Value> &crossing) {
    for (Operation *it = op->getPrevNode(); it != nullptr;
         it = it->getPrevNode()) {
      for (Value value : it->getResults()) {
        for (Operation *user : value.getUsers()) {
          // ignore use of condition
          if (user == op)
            continue;

          if (op->isAncestor(user)) {
            crossing.insert(value);
            break;
          }
        }
      }
    }

    for (Value value : op->getBlock()->getArguments()) {
      for (Operation *user : value.getUsers()) {
        // ignore use of condition
        if (user == op)
          continue;

        if (op->isAncestor(user)) {
          crossing.insert(value);
          break;
        }
      }
    }
    // No need to process block arguments, they are assumed to be induction
    // variables and will be replicated.
  }

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto term = cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    if (auto ifOp = dyn_cast_or_null<scf::IfOp>(term->getPrevNode())) {
      if (ifOp.getCondition() != term.getCondition())
        return failure();

      SmallVector<std::pair<BlockArgument, Value>, 2> m;

      // The return results of the while which are used
      SmallVector<Value, 2> prevResults;
      // The corresponding value in the before which
      // is to be returned
      SmallVector<Value, 2> condArgs;

      SmallVector<std::pair<size_t, Value>, 2> afterYieldRewrites;
      auto afterYield = cast<YieldOp>(op.getAfter().front().back());
      for (auto pair :
           llvm::zip(op.getResults(), term.getArgs(), op.getAfterArguments())) {
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
            if (auto blockArg = elseYielded.dyn_cast<BlockArgument>())
              if (blockArg.getOwner() == &op.getBefore().front()) {
                if (afterYield.getResults()[blockArg.getArgNumber()] ==
                        std::get<2>(pair) &&
                    op.getResults()[blockArg.getArgNumber()] ==
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
              op.getBefore().isAncestor(thenYielded.getParentRegion())) {
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
      for (auto pair : afterYieldRewrites) {
        yieldArgs[pair.first] = pair.second;
      }

      rewriter.updateRootInPlace(afterYield, [&] {
        afterYield.getResultsMutable().assign(yieldArgs);
      });
      Block *afterB = &op.getAfter().front();

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
      rewriter.replaceOpWithNewOp<ConditionOp>(term, term.getCondition(),
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

      SmallVector<Type, 4> resultTypes;
      for (auto v : condArgs) {
        resultTypes.push_back(v.getType());
      }

      rewriter.setInsertionPoint(op);
      auto nop =
          rewriter.create<WhileOp>(op.getLoc(), resultTypes, op.getInits());
      nop.getBefore().takeBody(op.getBefore());
      nop.getAfter().takeBody(op.getAfter());

      rewriter.updateRootInPlace(op, [&] {
        for (auto pair : llvm::enumerate(prevResults)) {
          pair.value().replaceAllUsesWith(nop.getResult(pair.index()));
        }
      });

      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

struct MoveWhileInvariantIfResult : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<BlockArgument, 2> origAfterArgs(op.getAfterArguments().begin(),
                                                op.getAfterArguments().end());
    bool changed = false;
    scf::ConditionOp term =
        cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    assert(origAfterArgs.size() == op.getResults().size());
    assert(origAfterArgs.size() == term.getArgs().size());

    for (auto pair :
         llvm::zip(op.getResults(), term.getArgs(), origAfterArgs)) {
      if (!std::get<0>(pair).use_empty()) {
        if (auto ifOp = std::get<1>(pair).getDefiningOp<scf::IfOp>()) {
          if (ifOp.getCondition() == term.getCondition()) {
            auto idx = std::get<1>(pair).cast<OpResult>().getResultNumber();
            Value returnWith = ifOp.elseYield().getResults()[idx];
            if (!op.getBefore().isAncestor(returnWith.getParentRegion())) {
              rewriter.updateRootInPlace(op, [&] {
                std::get<0>(pair).replaceAllUsesWith(returnWith);
              });
              changed = true;
            }
          }
        } else if (auto selOp =
                       std::get<1>(pair).getDefiningOp<arith::SelectOp>()) {
          if (selOp.getCondition() == term.getCondition()) {
            Value returnWith = selOp.getFalseValue();
            if (!op.getBefore().isAncestor(returnWith.getParentRegion())) {
              rewriter.updateRootInPlace(op, [&] {
                std::get<0>(pair).replaceAllUsesWith(returnWith);
              });
              changed = true;
            }
          }
        }
      }
    }

    return success(changed);
  }
};

struct WhileLogicalNegation : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    scf::ConditionOp term =
        cast<scf::ConditionOp>(op.getBefore().front().getTerminator());

    SmallPtrSet<Value, 1> condOps;
    SmallVector<Value> todo = {term.getCondition()};
    while (todo.size()) {
      Value val = todo.back();
      todo.pop_back();
      condOps.insert(val);
      if (auto ao = val.getDefiningOp<AndIOp>()) {
        todo.push_back(ao.getLhs());
        todo.push_back(ao.getRhs());
      }
    }

    for (auto pair :
         llvm::zip(op.getResults(), term.getArgs(), op.getAfterArguments())) {
      auto termArg = std::get<1>(pair);
      bool afterValue;
      if (condOps.count(termArg)) {
        afterValue = true;
      } else {
        bool found = false;
        if (auto termCmp = termArg.getDefiningOp<arith::CmpIOp>()) {
          for (auto cond : condOps) {
            if (auto condCmp = cond.getDefiningOp<CmpIOp>()) {
              if (termCmp.getLhs() == condCmp.getLhs() &&
                  termCmp.getRhs() == condCmp.getRhs()) {
                // TODO generalize to logical negation of
                if (condCmp.getPredicate() == CmpIPredicate::slt &&
                    termCmp.getPredicate() == CmpIPredicate::sge) {
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
        rewriter.updateRootInPlace(op, [&] {
          rewriter.setInsertionPoint(op);
          auto truev =
              rewriter.create<ConstantIntOp>(op.getLoc(), !afterValue, 1);
          std::get<0>(pair).replaceAllUsesWith(truev);
        });
        changed = true;
      }
      if (!std::get<2>(pair).use_empty()) {
        rewriter.updateRootInPlace(op, [&] {
          rewriter.setInsertionPointToStart(&op.getAfter().front());
          auto truev =
              rewriter.create<ConstantIntOp>(op.getLoc(), afterValue, 1);
          std::get<2>(pair).replaceAllUsesWith(truev);
        });
        changed = true;
      }
    }

    return success(changed);
  }
};

/// TODO move the addi down and repalce below with a subi
struct WhileCmpOffset : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<BlockArgument, 2> origAfterArgs(op.getAfterArguments().begin(),
                                                op.getAfterArguments().end());
    scf::ConditionOp term =
        cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    assert(origAfterArgs.size() == op.getResults().size());
    assert(origAfterArgs.size() == term.getArgs().size());

    if (auto condCmp = term.getCondition().getDefiningOp<CmpIOp>()) {
      if (auto addI = condCmp.getLhs().getDefiningOp<AddIOp>()) {
        if (addI.getOperand(1).getDefiningOp() &&
            !op.getBefore().isAncestor(
                addI.getOperand(1).getDefiningOp()->getParentRegion()))
          if (auto blockArg = addI.getOperand(0).dyn_cast<BlockArgument>()) {
            if (blockArg.getOwner() == &op.getBefore().front()) {
              auto rng = llvm::make_early_inc_range(blockArg.getUses());

              {
                rewriter.setInsertionPoint(op);
                SmallVector<Value> oldInits = op.getInits();
                oldInits[blockArg.getArgNumber()] = rewriter.create<AddIOp>(
                    addI.getLoc(), oldInits[blockArg.getArgNumber()],
                    addI.getOperand(1));
                op.getInitsMutable().assign(oldInits);
                rewriter.updateRootInPlace(
                    addI, [&] { addI.replaceAllUsesWith(blockArg); });
              }

              YieldOp afterYield = cast<YieldOp>(op.getAfter().front().back());
              rewriter.setInsertionPoint(afterYield);
              SmallVector<Value> oldYields = afterYield.getResults();
              oldYields[blockArg.getArgNumber()] = rewriter.create<AddIOp>(
                  addI.getLoc(), oldYields[blockArg.getArgNumber()],
                  addI.getOperand(1));
              rewriter.updateRootInPlace(afterYield, [&] {
                afterYield.getResultsMutable().assign(oldYields);
              });

              rewriter.setInsertionPointToStart(&op.getBefore().front());
              auto sub = rewriter.create<SubIOp>(addI.getLoc(), blockArg,
                                                 addI.getOperand(1));
              for (OpOperand &use : rng) {
                rewriter.updateRootInPlace(use.getOwner(),
                                           [&]() { use.set(sub); });
              }
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
struct RemoveWhileSelect : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp loop,
                                PatternRewriter &rewriter) const override {
    scf::ConditionOp term =
        cast<scf::ConditionOp>(loop.getBefore().front().getTerminator());

    SmallVector<BlockArgument, 2> origAfterArgs(
        loop.getAfterArguments().begin(), loop.getAfterArguments().end());
    SmallVector<unsigned> newResults;
    SmallVector<unsigned> newAfter;
    SmallVector<Value> newYields;
    bool changed = false;
    for (auto pair :
         llvm::zip(loop.getResults(), term.getArgs(), origAfterArgs)) {
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
      changed = true;
    }
    if (!changed)
      return failure();

    SmallVector<Type, 4> resultTypes;
    for (auto v : newYields) {
      resultTypes.push_back(v.getType());
    }
    auto nop =
        rewriter.create<WhileOp>(loop.getLoc(), resultTypes, loop.getInits());

    nop.getBefore().takeBody(loop.getBefore());

    auto *after = rewriter.createBlock(&nop.getAfter());
    for (auto y : newYields)
      after->addArgument(y.getType(), loop.getLoc());

    SmallVector<Value> replacedArgs;
    for (auto idx : newAfter)
      replacedArgs.push_back(after->getArgument(idx));
    rewriter.mergeBlocks(&loop.getAfter().front(), after, replacedArgs);

    SmallVector<Value> replacedReturns;
    for (auto idx : newResults)
      replacedReturns.push_back(nop.getResult(idx));
    rewriter.replaceOp(loop, replacedReturns);
    rewriter.setInsertionPoint(term);
    rewriter.replaceOpWithNewOp<ConditionOp>(term, term.getCondition(),
                                             newYields);
    return success();
  }
};

struct MoveWhileDown3 : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    scf::ConditionOp term =
        cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    SmallVector<unsigned, 2> toErase;
    SmallVector<Value, 2> newOps;
    SmallVector<Value, 2> condOps;
    SmallVector<BlockArgument, 2> origAfterArgs(op.getAfterArguments().begin(),
                                                op.getAfterArguments().end());
    SmallVector<Value, 2> returns;
    assert(origAfterArgs.size() == op.getResults().size());
    assert(origAfterArgs.size() == term.getArgs().size());
    for (auto pair :
         llvm::zip(op.getResults(), term.getArgs(), origAfterArgs)) {
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
              op.getAfter().front().push_front(cloned);
            } else {
              cloned->moveBefore(&op.getAfter().front().front());
            }
            rewriter.updateRootInPlace(std::get<1>(pair).getDefiningOp(), [&] {
              std::get<2>(pair).replaceAllUsesWith(cloned->getResult(0));
            });
            toErase.push_back(std::get<2>(pair).getArgNumber());
            for (auto &o :
                 llvm::make_early_inc_range(cloned->getOpOperands())) {
              {
                newOps.push_back(o.get());
                o.set(op.getAfter().front().addArgument(o.get().getType(),
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

    llvm::BitVector eraseIndices(op.getAfter().front().getNumArguments());
    for (unsigned i : toErase)
      eraseIndices.set(i);
    rewriter.updateRootInPlace(
        term, [&] { op.getAfter().front().eraseArguments(eraseIndices); });
    rewriter.setInsertionPoint(term);
    rewriter.replaceOpWithNewOp<ConditionOp>(term, term.getCondition(),
                                             condOps);

    rewriter.setInsertionPoint(op);
    SmallVector<Type, 4> resultTypes;
    for (auto v : condOps) {
      resultTypes.push_back(v.getType());
    }
    auto nop =
        rewriter.create<WhileOp>(op.getLoc(), resultTypes, op.getInits());

    nop.getBefore().takeBody(op.getBefore());
    nop.getAfter().takeBody(op.getAfter());

    rewriter.updateRootInPlace(op, [&] {
      for (auto pair : llvm::enumerate(returns)) {
        pair.value().replaceAllUsesWith(nop.getResult(pair.index()));
      }
    });

    assert(resultTypes.size() == nop.getAfter().front().getNumArguments());
    assert(resultTypes.size() == condOps.size());

    rewriter.eraseOp(op);
    return success();
  }
};

// Rewritten from LoopInvariantCodeMotion.cpp
struct WhileLICM : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;
  static bool canBeHoisted(Operation *op,
                           function_ref<bool(Value)> definedOutside,
                           bool isSpeculatable, WhileOp whileOp) {
    // TODO consider requirement of isSpeculatable

    // Check that dependencies are defined outside of loop.
    if (!llvm::all_of(op->getOperands(), definedOutside))
      return false;
    // Check whether this op is side-effect free. If we already know that there
    // can be no side-effects because the surrounding op has claimed so, we can
    // (and have to) skip this step.
    if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      if (!memInterface.hasNoEffect()) {
        if (isReadOnly(op) && !isSpeculatable) {

          SmallVector<MemoryEffects::EffectInstance> whileEffects;
          collectEffects(whileOp, whileEffects, /*ignoreBarriers*/ false);

          SmallVector<MemoryEffects::EffectInstance> opEffects;
          collectEffects(op, opEffects, /*ignoreBarriers*/ false);

          bool conflict = false;
          for (auto before : opEffects)
            for (auto after : whileEffects) {
              if (mayAlias(before, after)) {
                // Read, read is okay
                if (isa<MemoryEffects::Read>(before.getEffect()) &&
                    isa<MemoryEffects::Read>(after.getEffect())) {
                  continue;
                }

                // Write, write is not okay because may be different offsets and
                // the later must subsume other conflicts are invalid.
                conflict = true;
                break;
              }
            }
          if (conflict)
            return false;
        } else
          return false;
      }
      // If the operation doesn't have side effects and it doesn't recursively
      // have side effects, it can always be hoisted.
      if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        return true;

      // Otherwise, if the operation doesn't provide the memory effect interface
      // and it doesn't have recursive side effects we treat it conservatively
      // as side-effecting.
    } else if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      return false;
    }

    // Recurse into the regions for this op and check whether the contained ops
    // can be hoisted.
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &innerOp : block.without_terminator())
          if (!canBeHoisted(&innerOp, definedOutside, isSpeculatable, whileOp))
            return false;
      }
    }
    return true;
  }

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    // We use two collections here as we need to preserve the order for
    // insertion and this is easiest.
    SmallPtrSet<Operation *, 8> willBeMovedSet;
    SmallVector<Operation *, 8> opsToMove;

    // Helper to check whether an operation is loop invariant wrt. SSA
    // properties.
    auto isDefinedOutsideOfBody = [&](Value value) {
      auto *definingOp = value.getDefiningOp();
      if (!definingOp) {
        if (auto ba = value.dyn_cast<BlockArgument>())
          definingOp = ba.getOwner()->getParentOp();
        assert(definingOp);
      }
      if (willBeMovedSet.count(definingOp))
        return true;
      return op != definingOp && !op->isAncestor(definingOp);
    };

    // Do not use walk here, as we do not want to go into nested regions and
    // hoist operations from there. These regions might have semantics unknown
    // to this rewriting. If the nested regions are loops, they will have been
    // processed.
    for (auto &block : op.getBefore()) {
      for (auto &iop : block.without_terminator()) {
        bool legal = canBeHoisted(&iop, isDefinedOutsideOfBody, false, op);
        if (legal) {
          opsToMove.push_back(&iop);
          willBeMovedSet.insert(&iop);
        }
      }
    }

    for (auto &block : op.getAfter()) {
      for (auto &iop : block.without_terminator()) {
        bool legal = canBeHoisted(&iop, isDefinedOutsideOfBody, true, op);
        if (legal) {
          opsToMove.push_back(&iop);
          willBeMovedSet.insert(&iop);
        }
      }
    }

    for (auto *moveOp : opsToMove)
      rewriter.updateRootInPlace(moveOp, [&] { moveOp->moveBefore(op); });

    return success(opsToMove.size() > 0);
  }
};

struct RemoveUnusedCondVar : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto term = cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    SmallVector<Value, 4> conds;
    llvm::BitVector eraseArgs(op.getAfter().front().getArguments().size());
    SmallVector<unsigned, 4> keepArgs;
    SmallVector<Type, 4> tys;
    unsigned i = 0;
    std::map<void *, unsigned> valueOffsets;
    std::map<unsigned, unsigned> resultOffsets;
    SmallVector<Value, 4> resultArgs;
    for (auto pair :
         llvm::zip(term.getArgs(), op.getAfter().front().getArguments(),
                   op.getResults())) {
      auto arg = std::get<0>(pair);
      auto afarg = std::get<1>(pair);
      auto res = std::get<2>(pair);
      if (afarg.use_empty() && res.use_empty()) {
        eraseArgs.set((unsigned)i);
      } else if (valueOffsets.find(arg.getAsOpaquePointer()) !=
                 valueOffsets.end()) {
        resultOffsets[i] = valueOffsets[arg.getAsOpaquePointer()];
        afarg.replaceAllUsesWith(
            resultArgs[valueOffsets[arg.getAsOpaquePointer()]]);
        eraseArgs.set((unsigned)i);
      } else {
        valueOffsets[arg.getAsOpaquePointer()] = keepArgs.size();
        resultOffsets[i] = keepArgs.size();
        resultArgs.push_back(afarg);
        conds.push_back(arg);
        keepArgs.push_back((unsigned)i);
        tys.push_back(arg.getType());
      }
      i++;
    }
    assert(i == op.getAfter().front().getArguments().size());

    if (!eraseArgs.none()) {

      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<scf::ConditionOp>(term, term.getCondition(),
                                                    conds);

      rewriter.setInsertionPoint(op);
      auto op2 = rewriter.create<WhileOp>(op.getLoc(), tys, op.getInits());

      op2.getBefore().takeBody(op.getBefore());
      op2.getAfter().takeBody(op.getAfter());
      for (auto pair : resultOffsets) {
        op.getResult(pair.first).replaceAllUsesWith(op2.getResult(pair.second));
      }
      rewriter.eraseOp(op);
      op2.getAfter().front().eraseArguments(eraseArgs);
      return success();
    }
    return failure();
  }
};

struct MoveSideEffectFreeWhile : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter &rewriter) const override {
    scf::ConditionOp term =
        cast<scf::ConditionOp>(op.getBefore().front().getTerminator());
    SmallVector<Value, 4> conds(term.getArgs().begin(), term.getArgs().end());
    bool changed = false;
    unsigned i = 0;
    for (auto arg : term.getArgs()) {
      if (auto IC = arg.getDefiningOp<IndexCastOp>()) {
        if (arg.hasOneUse() && op.getResult(i).use_empty()) {
          auto rep = op.getAfter().front().addArgument(
              IC->getOperand(0).getType(), IC->getOperand(0).getLoc());
          IC->moveBefore(&op.getAfter().front(), op.getAfter().front().begin());
          conds.push_back(IC.getIn());
          IC.getInMutable().assign(rep);
          op.getAfter().front().getArgument(i).replaceAllUsesWith(
              IC->getResult(0));
          changed = true;
        }
      }
      i++;
    }
    if (changed) {
      SmallVector<Type, 4> tys;
      for (auto arg : conds) {
        tys.push_back(arg.getType());
      }
      auto op2 = rewriter.create<WhileOp>(op.getLoc(), tys, op.getInits());
      op2.getBefore().takeBody(op.getBefore());
      op2.getAfter().takeBody(op.getAfter());
      unsigned j = 0;
      for (auto a : op.getResults()) {
        a.replaceAllUsesWith(op2.getResult(j));
        j++;
      }
      rewriter.eraseOp(op);
      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<scf::ConditionOp>(term, term.getCondition(),
                                                    conds);
      return success();
    }
    return failure();
  }
};

struct SubToAdd : public OpRewritePattern<SubIOp> {
  using OpRewritePattern<SubIOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SubIOp op,
                                PatternRewriter &rewriter) const override {
    if (auto cop = op.getOperand(1).getDefiningOp<ConstantIntOp>()) {
      rewriter.replaceOpWithNewOp<AddIOp>(
          op, op.getOperand(0),
          rewriter.create<ConstantIntOp>(cop.getLoc(), -cop.value(),
                                         cop.getType()));
      return success();
    }
    return failure();
  }
};

struct ReturnSq : public OpRewritePattern<ReturnOp> {
  using OpRewritePattern<ReturnOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReturnOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    SmallVector<Operation *> toErase;
    for (auto iter = op->getBlock()->rbegin();
         iter != op->getBlock()->rend() && &*iter != op; iter++) {
      changed = true;
      toErase.push_back(&*iter);
    }
    for (auto *op : toErase) {
      rewriter.eraseOp(op);
    }
    return success(changed);
  }
};

void CanonicalizeFor::runOnOperation() {
  mlir::RewritePatternSet rpl(getOperation()->getContext());
  rpl.add<PropagateInLoopBody, ForOpInductionReplacement, RemoveUnusedArgs,
          MoveWhileToFor, RemoveWhileSelect,

          MoveWhileDown, MoveWhileDown2,

          ReplaceRedundantArgs,

          MoveWhileAndDown, MoveWhileDown3, MoveWhileInvariantIfResult,
          WhileLogicalNegation, SubToAdd, WhileCmpOffset, WhileLICM,
          RemoveUnusedCondVar, ReturnSq, MoveSideEffectFreeWhile>(
      getOperation()->getContext());
  GreedyRewriteConfig config;
  config.maxIterations = 247;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(rpl), config);
}

std::unique_ptr<Pass> mlir::polygeist::createCanonicalizeForPass() {
  return std::make_unique<CanonicalizeFor>();
}
