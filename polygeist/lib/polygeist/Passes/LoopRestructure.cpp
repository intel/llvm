//===- LoopRestructure.cpp - Find natural Loops ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
#include "PassDetails.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopInfoImpl.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "LoopRestructure"

using namespace mlir;
using namespace polygeist;

struct Wrapper;

struct RWrapper {
  RWrapper(int x){};
  Wrapper &front();
};

struct Wrapper {
  mlir::Block blk;
  Wrapper() = delete;
  Wrapper(Wrapper &w) = delete;
  bool isLegalToHoistInto() const { return true; }
  void print(llvm::raw_ostream &OS) const {
    // B->print(OS);
  }
  void printAsOperand(llvm::raw_ostream &OS, bool b) const {
    // B->print(OS, b);
  }
  RWrapper *getParent() const {
    Region *R = ((Block *)(const_cast<Wrapper *>(this)))->getParent();
    return (RWrapper *)R;
  }
  mlir::Block &operator*() const {
    return *(Block *)(const_cast<Wrapper *>(this));
  }
  mlir::Block *operator->() const {
    return (Block *)(const_cast<Wrapper *>(this));
  }
};

Wrapper &RWrapper::front() { return *(Wrapper *)&((Region *)this)->front(); }

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, Wrapper &w) {
  return os << "<cannot print wrapper>";
}

template <typename T>
struct Iter
    : public std::iterator<std::input_iterator_tag, // iterator_category
                           Wrapper *, std::ptrdiff_t, Wrapper **, Wrapper *> {
  T it;
  Iter(T it) : it(it) {}
  Wrapper *operator*() const;
  bool operator!=(Iter I) const { return it != I.it; }
  bool operator==(Iter I) const { return it == I.it; }
  void operator++() { ++it; }
  Iter<T> operator--() { return --it; }
  Iter<T> operator++(int) {
    auto prev = *this;
    it++;
    return prev;
  }
};

template <> Wrapper *Iter<Region::iterator>::operator*() const {
  Block &B = *it;
  return (Wrapper *)&B;
}
template <> Wrapper *Iter<Region::reverse_iterator>::operator*() const {
  Block &B = *it;
  return (Wrapper *)&B;
}

template <typename T> Wrapper *Iter<T>::operator*() const {
  Block *B = *it;
  return (Wrapper *)B;
}

namespace llvm {
template <> struct GraphTraits<RWrapper *> {
  using nodes_iterator = Iter<Region::iterator>;
  static Wrapper *getEntryNode(RWrapper *bb) {
    return (Wrapper *)&((Region *)bb)->front();
  }
  static nodes_iterator nodes_begin(RWrapper *bb) {
    return ((Region *)bb)->begin();
  }
  static nodes_iterator nodes_end(RWrapper *bb) {
    return ((Region *)bb)->end();
  }
};
template <> struct GraphTraits<Inverse<RWrapper *>> {
  using nodes_iterator = Iter<Region::reverse_iterator>;
  static Wrapper *getEntryNode(RWrapper *bb) {
    return (Wrapper *)&((Region *)bb)->front();
  }
  static nodes_iterator nodes_begin(RWrapper *bb) {
    return ((Region *)bb)->rbegin();
  }
  static nodes_iterator nodes_end(RWrapper *bb) {
    return ((Region *)bb)->rend();
  }
};
template <> struct GraphTraits<const Wrapper *> {
  using ChildIteratorType = Iter<Block::succ_iterator>;
  using Node = const Wrapper;
  using NodeRef = Node *;

  static NodeRef getEntryNode(NodeRef bb) { return bb; }

  static ChildIteratorType child_begin(NodeRef node) {
    return (*node)->succ_begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return (*node)->succ_end();
  }
};
template <> struct GraphTraits<Wrapper *> {
  using ChildIteratorType = Iter<Block::succ_iterator>;
  using Node = Wrapper;
  using NodeRef = Node *;

  static NodeRef getEntryNode(NodeRef bb) { return bb; }

  static ChildIteratorType child_begin(NodeRef node) {
    return (*node)->succ_begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return (*node)->succ_end();
  }
};

template <> struct GraphTraits<Inverse<Wrapper *>> {
  using ChildIteratorType = Iter<Block::pred_iterator>;
  using Node = Wrapper;
  using NodeRef = Node *;

  static ChildIteratorType child_begin(NodeRef node) {
    return (*node)->pred_begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return (*node)->pred_end();
  }
};
template <> struct GraphTraits<Inverse<const Wrapper *>> {
  using ChildIteratorType = Iter<Block::pred_iterator>;
  using Node = const Wrapper;
  using NodeRef = Node *;

  static ChildIteratorType child_begin(NodeRef node) {
    return (*node)->pred_begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return (*node)->pred_end();
  }
};

template <>
struct GraphTraits<const DomTreeNodeBase<Wrapper> *>
    : public DomTreeGraphTraitsBase<const DomTreeNodeBase<Wrapper>,
                                    DomTreeNodeBase<Wrapper>::const_iterator> {
};

} // namespace llvm

namespace {

struct LoopRestructure : public LoopRestructureBase<LoopRestructure> {
  void runOnRegion(DominanceInfo &domInfo, Region &region);
  bool removeIfFromRegion(DominanceInfo &domInfo, Region &region,
                          Block *pseudoExit);
  void runOnOperation() override;
};

} // end anonymous namespace

// Instantiate a variant of LLVM LoopInfo that works on mlir::Block

template class llvm::DominatorTreeBase<Wrapper, false>;
template class llvm::DomTreeNodeBase<Wrapper>;
// template void
// llvm::DomTreeBuilder::ApplyUpdates<llvm::DominatorTreeBase<Wrapper, false>>;

namespace mlir {
class Loop : public llvm::LoopBase<Wrapper, mlir::Loop> {
private:
  Loop() = default;
  friend class llvm::LoopBase<Wrapper, Loop>;
  friend class llvm::LoopInfoBase<Wrapper, Loop>;
  explicit Loop(Wrapper *B) : llvm::LoopBase<Wrapper, Loop>(B) {}
  ~Loop() = default;
};
class LoopInfo : public llvm::LoopInfoBase<Wrapper, mlir::Loop> {
public:
  LoopInfo(const llvm::DominatorTreeBase<Wrapper, false> &DomTree) {
    analyze(DomTree);
  }
};
} // namespace mlir

template class llvm::LoopBase<Wrapper, ::mlir::Loop>;
template class llvm::LoopInfoBase<Wrapper, ::mlir::Loop>;

void LoopRestructure::runOnOperation() {
  // FuncOp f = getFunction();
  DominanceInfo &domInfo = getAnalysis<DominanceInfo>();
  for (auto &region : getOperation()->getRegions()) {
    runOnRegion(domInfo, region);
  }
}

bool attemptToFoldIntoPredecessor(Block *target) {
  SmallVector<Block *, 2> P(target->pred_begin(), target->pred_end());
  if (P.size() == 1) {
    if (auto op = dyn_cast<cf::BranchOp>(P[0]->getTerminator())) {
      assert(target->getNumArguments() == op.getNumOperands());
      for (size_t i = 0; i < target->getNumArguments(); ++i) {
        target->getArgument(i).replaceAllUsesWith(op.getOperand(i));
      }
      P[0]->getOperations().splice(P[0]->getOperations().end(),
                                   target->getOperations());
      op->erase();
      target->erase();
      return true;
    }
  } else if (P.size() == 2) {
    if (auto op = dyn_cast<cf::CondBranchOp>(P[0]->getTerminator())) {
      assert(target->getNumArguments() == op.getNumTrueOperands());
      assert(target->getNumArguments() == op.getNumFalseOperands());

      mlir::OpBuilder builder(op);
      SmallVector<mlir::Type> types;
      for (auto T : op.getTrueOperands()) {
        types.push_back(T.getType());
      }

      for (size_t i = 0; i < target->getNumArguments(); ++i) {
        auto sel = builder.create<mlir::arith::SelectOp>(
            op.getLoc(), op.getCondition(), op.getTrueOperand(i),
            op.getFalseOperand(i));
        target->getArgument(i).replaceAllUsesWith(sel);
      }
      P[0]->getOperations().splice(P[0]->getOperations().end(),
                                   target->getOperations());
      op->erase();
      target->erase();
      return true;
    }
  }
  return false;
}

bool LoopRestructure::removeIfFromRegion(DominanceInfo &domInfo, Region &region,
                                         Block *pseudoExit) {
  SmallVector<Block *, 4> Preds;
  for (auto *block : pseudoExit->getPredecessors()) {
    Preds.push_back(block);
  }
  SmallVector<Type, 4> emptyTys;
  SmallVector<Type, 4> condTys;
  for (auto a : pseudoExit->getArguments()) {
    condTys.push_back(a.getType());
  }
  if (Preds.size() == 2) {
    for (size_t i = 0; i < Preds.size(); ++i) {
      SmallVector<Block *, 4> Succs;
      for (auto *block : Preds[i]->getSuccessors()) {
        Succs.push_back(block);
      }
      if (Succs.size() == 2) {
        for (size_t j = 0; j < Succs.size(); ++j) {
          if (Succs[j] == pseudoExit && Succs[1 - j] == Preds[1 - i]) {
            OpBuilder builder(Preds[i]->getTerminator());
            auto condBr = cast<cf::CondBranchOp>(Preds[i]->getTerminator());
            auto ifOp = builder.create<scf::IfOp>(
                builder.getUnknownLoc(), condTys, condBr.getCondition(),
                /*hasElse*/ true);
            Succs[j] = new Block();
            if (j == 0) {
              ifOp.getElseRegion().getBlocks().splice(
                  ifOp.getElseRegion().getBlocks().end(), region.getBlocks(),
                  Succs[1 - j]);
              llvm::BitVector idx(Succs[1 - j]->getNumArguments());
              for (size_t i = 0; i < Succs[1 - j]->getNumArguments(); ++i) {
                Succs[1 - j]->getArgument(i).replaceAllUsesWith(
                    condBr.getFalseOperand(i));
                idx.set(i);
              }
              Succs[1 - j]->eraseArguments(idx);
              assert(!ifOp.getElseRegion().getBlocks().empty());
              assert(condTys.size() == condBr.getTrueOperands().size());
              OpBuilder tbuilder(&ifOp.getThenRegion().front(),
                                 ifOp.getThenRegion().front().begin());
              tbuilder.create<scf::YieldOp>(tbuilder.getUnknownLoc(), emptyTys,
                                            condBr.getTrueOperands());
            } else {
              if (!ifOp.getThenRegion().getBlocks().empty()) {
                ifOp.getThenRegion().front().erase();
              }
              ifOp.getThenRegion().getBlocks().splice(
                  ifOp.getThenRegion().getBlocks().end(), region.getBlocks(),
                  Succs[1 - j]);
              llvm::BitVector idx(Succs[1 - j]->getNumArguments());
              for (size_t i = 0; i < Succs[1 - j]->getNumArguments(); ++i) {
                Succs[1 - j]->getArgument(i).replaceAllUsesWith(
                    condBr.getTrueOperand(i));
                idx.set(i);
              }
              Succs[1 - j]->eraseArguments(idx);
              assert(!ifOp.getElseRegion().getBlocks().empty());
              OpBuilder tbuilder(&ifOp.getElseRegion().front(),
                                 ifOp.getElseRegion().front().begin());
              assert(condTys.size() == condBr.getFalseOperands().size());
              tbuilder.create<scf::YieldOp>(tbuilder.getUnknownLoc(), emptyTys,
                                            condBr.getFalseOperands());
            }
            auto *oldTerm = Succs[1 - j]->getTerminator();
            OpBuilder tbuilder(Succs[1 - j], Succs[1 - j]->end());
            tbuilder.create<scf::YieldOp>(tbuilder.getUnknownLoc(), emptyTys,
                                          oldTerm->getOperands());
            oldTerm->erase();

            builder.create<scf::YieldOp>(builder.getUnknownLoc(),
                                         ifOp->getResults());
            condBr->erase();

            pseudoExit->erase();
            return true;
          }
        }
      }
    }
  }
  return false;
}

void LoopRestructure::runOnRegion(DominanceInfo &domInfo, Region &region) {
  if (region.getBlocks().size() > 1) {
    const llvm::DominatorTreeBase<Block, false> *DT =
        &domInfo.getDomTree(&region);
    mlir::LoopInfo LI(*(const llvm::DominatorTreeBase<Wrapper, false> *)DT);
    for (auto *L : LI.getTopLevelLoops()) {
      Block *header = (Block *)L->getHeader();
      Block *target = (Block *)L->getUniqueExitBlock();
      if (!target) {
        // Only support one exit block
        llvm::errs()
            << " found mlir loop with more than one exit, skipping. \n";
        continue;
      }

      // Replace branch to exit block with a new block that calls
      // loop.natural.return In caller block, branch to correct exit block
      SmallVector<Wrapper *, 4> exitingBlocks;
      L->getExitingBlocks(exitingBlocks);

      // TODO: Support multiple exit blocks
      //  - Easy case all exit blocks have the same argument set

      // Create a caller block that will contain the loop op

      Block *wrapper = new Block();
      region.push_back(wrapper);
      mlir::OpBuilder builder(wrapper, wrapper->begin());

      // Copy the arguments across
      SmallVector<Type, 4> headerArgumentTypes(header->getArgumentTypes());
      SmallVector<Location> locs(headerArgumentTypes.size(),
                                 builder.getUnknownLoc());
      wrapper->addArguments(headerArgumentTypes, locs);

      SmallVector<Value> valsCallingLoop;
      for (auto a : wrapper->getArguments())
        valsCallingLoop.push_back(a);

      SmallVector<std::pair<Value, size_t>> preservedVals;
      for (auto *B : L->getBlocks()) {
        for (auto &O : *(Block *)B) {
          for (auto V : O.getResults()) {
            if (llvm::any_of(V.getUsers(), [&](Operation *user) {
                  Block *blk = user->getBlock();
                  while (blk->getParent() != &region)
                    blk = blk->getParentOp()->getBlock();
                  return !L->contains((Wrapper *)blk);
                })) {
              preservedVals.emplace_back(V, headerArgumentTypes.size());
              headerArgumentTypes.push_back(V.getType());
              valsCallingLoop.push_back(builder.create<mlir::LLVM::UndefOp>(
                  builder.getUnknownLoc(), V.getType()));
              header->addArgument(V.getType(), V.getLoc());
            }
          }
        }
      }

      SmallVector<Type, 4> combinedTypes = headerArgumentTypes;
      SmallVector<Type, 4> returns(target->getArgumentTypes());
      combinedTypes.append(returns);

      auto loop = builder.create<mlir::scf::WhileOp>(
          builder.getUnknownLoc(), combinedTypes, valsCallingLoop);
      {
        SmallVector<Value, 4> RetVals;
        for (size_t i = 0; i < returns.size(); ++i) {
          RetVals.push_back(loop.getResult(i + headerArgumentTypes.size()));
        }
        builder.create<cf::BranchOp>(builder.getUnknownLoc(), target, RetVals);
      }
      for (auto &pair : preservedVals) {
        pair.first.replaceUsesWithIf(loop.getResult(pair.second),
                                     [&](OpOperand &op) -> bool {
                                       Block *blk = op.getOwner()->getBlock();
                                       while (blk->getParent() != &region)
                                         blk = blk->getParentOp()->getBlock();
                                       return !L->contains((Wrapper *)blk);
                                     });
      }

      SmallVector<Block *, 4> Preds;

      for (auto *block : header->getPredecessors()) {
        if (!L->contains((Wrapper *)block))
          Preds.push_back(block);
      }

      Block *loopEntry = new Block();
      loop.getBefore().push_back(loopEntry);
      builder.setInsertionPointToEnd(loopEntry);
      SmallVector<Type, 4> tys = {builder.getI1Type()};
      for (auto t : combinedTypes)
        tys.push_back(t);
      auto exec =
          builder.create<scf::ExecuteRegionOp>(builder.getUnknownLoc(), tys);

      {
        SmallVector<Value> yields;
        for (auto a : exec.getResults())
          yields.push_back(a);
        yields.erase(yields.begin());
        builder.create<scf::ConditionOp>(builder.getUnknownLoc(),
                                         exec.getResult(0), yields);
      }

      Region &insertRegion = exec.getRegion();

      insertRegion.getBlocks().splice(insertRegion.getBlocks().begin(),
                                      region.getBlocks(), header);
      assert(header->getParent() == &insertRegion);
      for (auto *w : L->getBlocks()) {
        Block *b = &**w;
        if (b != header) {
          insertRegion.getBlocks().splice(insertRegion.getBlocks().end(),
                                          region.getBlocks(), b);
        }
      }

      Block *pseudoExit = new Block();
      {
        insertRegion.push_back(pseudoExit);
        SmallVector<Location> locs(tys.size(), builder.getUnknownLoc());
        pseudoExit->addArguments(tys, locs);
        OpBuilder builder(pseudoExit, pseudoExit->begin());
        tys.clear();
        builder.create<scf::YieldOp>(builder.getUnknownLoc(), tys,
                                     pseudoExit->getArguments());
      }

      for (auto *w : exitingBlocks) {
        Block *block = &**w;
        Operation *terminator = block->getTerminator();
        for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
          Block *successor = terminator->getSuccessor(i);
          if (successor == target) {

            OpBuilder builder(terminator);
            auto vfalse = builder.create<arith::ConstantIntOp>(
                builder.getUnknownLoc(), false, 1);

            SmallVector<Value> args = {vfalse};
            for (auto arg : header->getArguments())
              args.push_back(arg);
            for (auto v : preservedVals)
              args[v.second + 1] = v.first;

            if (auto op = dyn_cast<cf::BranchOp>(terminator)) {
              args.insert(args.end(), op.getOperands().begin(),
                          op.getOperands().end());
              builder.create<cf::BranchOp>(op.getLoc(), pseudoExit, args);
              op.erase();
            }
            if (auto op = dyn_cast<cf::CondBranchOp>(terminator)) {
              std::vector<Value> trueargs(op.getTrueOperands().begin(),
                                          op.getTrueOperands().end());
              std::vector<Value> falseargs(op.getFalseOperands().begin(),
                                           op.getFalseOperands().end());
              if (op.getTrueDest() == target) {
                trueargs.insert(trueargs.begin(), args.begin(), args.end());
              }
              if (op.getFalseDest() == target) {
                falseargs.insert(falseargs.begin(), args.begin(), args.end());
              }
              builder.create<cf::CondBranchOp>(
                  op.getLoc(), op.getCondition(),
                  op.getTrueDest() == target ? pseudoExit : op.getTrueDest(),
                  trueargs,
                  op.getFalseDest() == target ? pseudoExit : op.getFalseDest(),
                  falseargs);
              op.erase();
            }
            break;
          }
        }
      }

      // For each back edge create a new block and replace
      // the destination of that edge with said new block
      // in that new block call loop.natural.next
      SmallVector<Wrapper *, 4> loopLatches;
      L->getLoopLatches(loopLatches);
      for (auto *w : loopLatches) {
        Block *block = &**w;
        Operation *terminator = block->getTerminator();
        // Note: the terminator may be reassigned in the loop body so not
        // caching numSuccessors here.
        for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
          Block *successor = terminator->getSuccessor(i);
          if (successor == header) {

            OpBuilder builder(terminator);
            auto vtrue = builder.create<arith::ConstantIntOp>(
                builder.getUnknownLoc(), true, 1);

            if (auto op = dyn_cast<cf::BranchOp>(terminator)) {
              SmallVector<Value> args(op.getOperands());
              args.insert(args.begin(), vtrue);
              for (auto p : preservedVals)
                args.push_back(p.first);
              for (auto ty : returns) {
                args.push_back(builder.create<mlir::LLVM::UndefOp>(
                    builder.getUnknownLoc(), ty));
              }
              terminator =
                  builder.create<cf::BranchOp>(op.getLoc(), pseudoExit, args);
              op.erase();
            } else if (auto op = dyn_cast<cf::CondBranchOp>(terminator)) {
              std::vector<Value> trueargs(op.getTrueOperands().begin(),
                                          op.getTrueOperands().end());
              std::vector<Value> falseargs(op.getFalseOperands().begin(),
                                           op.getFalseOperands().end());
              if (op.getTrueDest() == header) {
                trueargs.insert(trueargs.begin(), vtrue);
                for (auto pair : preservedVals)
                  trueargs.push_back(pair.first);
                for (auto ty : returns) {
                  trueargs.push_back(builder.create<mlir::LLVM::UndefOp>(
                      builder.getUnknownLoc(), ty));
                }
              }
              if (op.getFalseDest() == header) {
                falseargs.insert(falseargs.begin(), vtrue);
                for (auto pair : preservedVals)
                  falseargs.push_back(pair.first);
                for (auto ty : returns) {
                  falseargs.push_back(builder.create<mlir::LLVM::UndefOp>(
                      builder.getUnknownLoc(), ty));
                }
              }
              // Recreate the terminator and store it so that its other
              // successor is visited on the next iteration of the loop.
              terminator = builder.create<cf::CondBranchOp>(
                  op.getLoc(), op.getCondition(),
                  op.getTrueDest() == header ? pseudoExit : op.getTrueDest(),
                  trueargs,
                  op.getFalseDest() == header ? pseudoExit : op.getFalseDest(),
                  falseargs);
              op.erase();
            }
          }
        }
      }

      Block *after = new Block();
      SmallVector<Location> locs2(combinedTypes.size(), region.getLoc());
      after->addArguments(combinedTypes, locs2);
      loop.getAfter().push_back(after);
      OpBuilder builder2(after, after->begin());
      SmallVector<Value, 4> yieldargs;
      for (auto a : after->getArguments()) {
        if (yieldargs.size() == headerArgumentTypes.size())
          break;
        yieldargs.push_back(a);
      }

      for (auto *block : Preds) {
        Operation *terminator = block->getTerminator();
        for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
          Block *successor = terminator->getSuccessor(i);
          if (successor == header) {
            terminator->setSuccessor(wrapper, i);
          }
        }
      }

      for (size_t i = 0; i < header->getNumArguments(); i++) {
        header->getArgument(i).replaceUsesWithIf(
            loop->getResult(i), [&](OpOperand &u) -> bool {
              return !loop.getOperation()->isProperAncestor(u.getOwner());
            });
      }

      SmallVector<Location> locs3(header->getArgumentTypes().size(),
                                  region.getLoc());
      for (auto pair : llvm::zip(
               header->getArguments(),
               loopEntry->addArguments(header->getArgumentTypes(), locs3))) {
        std::get<0>(pair).replaceAllUsesWith(std::get<1>(pair));
      }
      header->eraseArguments([](BlockArgument) { return true; });

      builder2.create<scf::YieldOp>(builder.getUnknownLoc(), yieldargs);
      domInfo.invalidate(&insertRegion);

      assert(header->getParent() == &insertRegion);

      runOnRegion(domInfo, insertRegion);

      if (!removeIfFromRegion(domInfo, insertRegion, pseudoExit)) {
        attemptToFoldIntoPredecessor(pseudoExit);
      }

      attemptToFoldIntoPredecessor(wrapper);
      attemptToFoldIntoPredecessor(target);

      if (llvm::hasSingleElement(insertRegion)) {
        Block *block = &insertRegion.front();
        IRRewriter B(exec->getContext());
        Operation *terminator = block->getTerminator();
        SmallVector<Value> results;
        llvm::append_range(results, terminator->getOperands());
        terminator->erase();
        B.mergeBlockBefore(block, exec);
        exec.replaceAllUsesWith(results);
        exec.erase();
      }

      assert(loop.getBefore().getBlocks().size() == 1);
      runOnRegion(domInfo, loop.getAfter());
      assert(loop.getAfter().getBlocks().size() == 1);
    }
  }

  for (auto &blk : region) {
    for (auto &op : blk) {
      for (auto &reg : op.getRegions()) {
        domInfo.invalidate(&reg);
        runOnRegion(domInfo, reg);
      }
    }
  }
}

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createLoopRestructurePass() {
  return std::make_unique<LoopRestructure>();
}
} // namespace polygeist
} // namespace mlir
