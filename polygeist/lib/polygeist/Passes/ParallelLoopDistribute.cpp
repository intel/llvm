//===- ParallelLoopDistrbute.cpp - Distribute loops around barriers -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "polygeist/BarrierUtils.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"
#include "polygeist/Passes/Utils.h"
#include <mlir/Dialect/Arith/IR/Arith.h>

#include <deque>

#define DEBUG_TYPE "cpuify"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE "] "

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;

static bool couldWrite(Operation *op) {
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> localEffects;
    iface.getEffects<MemoryEffects::Write>(localEffects);
    return localEffects.size() > 0;
  }
  return true;
}

struct Node {
  Operation *O;
  Value V;
  enum Type {
    NONE,
    VAL,
    OP,
  } type;
  Node(Operation *O) : O(O), type(OP){};
  Node(Value V) : V(V), type(VAL){};
  Node() : type(NONE){};
  bool operator<(const Node N) const {
    if (type != N.type)
      return type < N.type;
    else if (type == OP)
      return O < N.O;
    else if (type == VAL)
      return V.getAsOpaquePointer() < N.V.getAsOpaquePointer();
    else
      return true;
  }
  void dump() const {
    if (type == VAL)
      llvm::errs() << "[" << V << ", "
                   << "Value"
                   << "]\n";
    else if (type == OP)
      llvm::errs() << "[" << *O << ", "
                   << "Operation"
                   << "]\n";
    else
      llvm::errs() << "["
                   << "NULL"
                   << ", "
                   << "None"
                   << "]\n";
  }
};

typedef std::map<Node, std::set<Node>> Graph;

void dump(Graph &G) {
  for (auto &pair : G) {
    pair.first.dump();
    for (const auto &N : pair.second) {
      llvm::errs() << "\t";
      N.dump();
    }
  }
}

/* Returns true if there is a path from source 's' to sink 't' in
   residual graph. Also fills parent[] to store the path */
static inline void bfs(const Graph &G,
                       const llvm::SetVector<Operation *> &Sources,
                       std::map<Node, Node> &parent) {
  std::deque<Node> q;
  for (auto *O : Sources) {
    Node N(O);
    parent.emplace(N, Node(nullptr));
    q.push_back(N);
  }

  // Standard BFS Loop
  while (!q.empty()) {
    auto u = q.front();
    q.pop_front();
    auto found = G.find(u);
    if (found == G.end())
      continue;
    for (auto v : found->second) {
      if (parent.find(v) == parent.end()) {
        q.push_back(v);
        parent.emplace(v, u);
      }
    }
  }
}

// \p singleExecution denotes whether op is guaranteed to execute the body once
// and after the cloned values
static bool arePreceedingOpsFullyRecomputable(Operation *op,
                                              bool singleExecution) {
  SmallVector<MemoryEffects::EffectInstance> beforeEffects;
  getEffectsBefore(op, beforeEffects, /*stopAtBarrier*/ false);

  for (auto it : beforeEffects) {
    if (isa<MemoryEffects::Read>(it.getEffect())) {
      if (singleExecution)
        continue;
      if (Value v = it.getValue())
        if (!mayWriteTo(op, v, /*ignoreBarrier*/ true))
          continue;
    }
    return false;
  }

  return true;
}

static bool isRecomputableAfterDistribute(Operation *op,
                                          polygeist::BarrierOp barrier) {
  // The below logic should not disagree with the logic in interchange and wrap,
  // otherwise we might cache unneeded results or wrap* will ask us to
  // distribute again if it thinks the ops we decide here are recomputable here
  // are not, resulting into an infinite loop

  if (isa<polygeist::BarrierOp>(op))
    return false;

  SmallVector<MemoryEffects::EffectInstance> effects;
  collectEffects(op, effects, /*ignoreBarriers*/ false);

  if (effects.size() == 0)
    return true;
  for (auto it : effects)
    if (!isa<MemoryEffects::Read>(it.getEffect()))
      return false;

  // op now only has read effects

  // It is recomputable if nothing else may write to where it reads from from
  // after the previous barrier (begin) up to the next sync after the barrier we
  // are distributing around (end)

  Operation *begin = op;
  Operation *front = &barrier->getBlock()->front();
  while (true) {
    if (begin == front)
      break;
    Operation *prev = begin->getPrevNode();
    if (isa<polygeist::BarrierOp>(prev))
      break;
    begin = prev;
  }
  Operation *end = barrier;
  while (true) {
    end = end->getNextNode();
    if (end == nullptr)
      break;
    if (isa<polygeist::BarrierOp>(end))
      break;
  }

  for (auto it : effects) {
    assert(isa<MemoryEffects::Read>(it.getEffect()));
    if (Value v = it.getValue())
      for (Operation *op = begin; op != end; op = op->getNextNode())
        if (mayWriteTo(op, v, /*ignoreBarrier*/ true))
          return false;
  }
  return true;
}

static void minCutCache(polygeist::BarrierOp barrier,
                        llvm::SetVector<Value> &Required,
                        llvm::SetVector<Value> &Cache) {
  Graph G;
  llvm::SetVector<Operation *> NonRecomputable;

  for (Operation *op = &barrier->getBlock()->front(); op != barrier;
       op = op->getNextNode()) {

    if (!isRecomputableAfterDistribute(op, barrier))
      NonRecomputable.insert(op);

    for (Value value : op->getResults()) {
      G[Node(op)].insert(Node(value));
      for (Operation *user : value.getUsers()) {
        // If the user is nested in another op, find its ancestor op that lives
        // in the same block as the barrier.
        while (user->getBlock() != barrier->getBlock())
          user = user->getBlock()->getParentOp();

        G[Node(value)].insert(Node(user));
      }
    }
  }

  Graph Orig = G;

  // Augment the flow while there is a path from source to sink
  while (1) {
    std::map<Node, Node> parent;
    bfs(G, NonRecomputable, parent);
    Node end;
    for (auto req : Required) {
      if (parent.find(Node(req)) != parent.end()) {
        end = Node(req);
        break;
      }
    }
    if (end.type == Node::NONE)
      break;
    // update residual capacities of the edges and reverse edges
    // along the path
    Node v = end;
    while (1) {
      assert(parent.find(v) != parent.end());
      Node u = parent.find(v)->second;
      assert(u.type != Node::NONE);
      assert(G[u].count(v) == 1);
      assert(G[v].count(u) == 0);
      G[u].erase(v);
      G[v].insert(u);
      if (u.type == Node::OP && NonRecomputable.count(u.O))
        break;
      v = u;
    }
  }
  // Flow is maximum now, find vertices reachable from s

  std::map<Node, Node> parent;
  bfs(G, NonRecomputable, parent);

  // All edges that are from a reachable vertex to non-reachable vertex in the
  // original graph
  for (auto &pair : Orig) {
    if (parent.find(pair.first) != parent.end()) {
      for (auto N : pair.second) {
        if (parent.find(N) == parent.end()) {
          assert(pair.first.type == Node::OP && N.type == Node::VAL);
          assert(pair.first.O == N.V.dyn_cast<OpResult>().getOwner());
          Cache.insert(N.V);
        }
      }
    }
  }

  // When ambiguous, push to cache the last value in a computation chain
  // This should be considered in a cost for the max flow
  std::deque<Node> todo;
  for (auto V : Cache)
    todo.push_back(Node(V));

  while (todo.size()) {
    auto N = todo.front();
    todo.pop_front();
    auto found = Orig.find(N);
    (void)found;
    // TODO
    break;
  }
}

/// Populates `crossing` with values (op results) that are defined in the same
/// block as `op` and above it, and used by at least one op in the same block
/// below `op`. Uses may be in nested regions.
static void findValuesUsedBelow(polygeist::BarrierOp op,
                                llvm::SetVector<Value> &crossing,
                                llvm::SetVector<Operation *> &preserveAllocas) {
  llvm::SetVector<Operation *> descendantsUsed;

  // A set of pre-barrier operations which are potentially captured by a
  // subsequent pre-barrier operation.
  SmallVector<Operation *> Allocas;

  for (Operation *it = op->getPrevNode(); it != nullptr;
       it = it->getPrevNode()) {
    if (isa<memref::AllocaOp, LLVM::AllocaOp>(it))
      Allocas.push_back(it);
    for (Value value : it->getResults()) {
      for (Operation *user : value.getUsers()) {

        // If the user is nested in another op, find its ancestor op that lives
        // in the same block as the barrier.
        while (user->getBlock() != op->getBlock())
          user = user->getBlock()->getParentOp();

        if (op->isBeforeInBlock(user)) {
          crossing.insert(value);
        }
      }
    }
  }

  llvm::SmallVector<std::pair<Operation *, Operation *>> todo;
  for (auto *A : Allocas)
    todo.emplace_back(A, A);

  std::map<Operation *, SmallPtrSet<Operation *, 2>> descendants;
  while (todo.size()) {
    auto current = todo.back();
    todo.pop_back();
    if (descendants[current.first].count(current.second))
      continue;
    descendants[current.first].insert(current.second);
    for (Value value : current.first->getResults()) {
      for (Operation *user : value.getUsers()) {
        Operation *origUser = user;
        while (user->getBlock() != op->getBlock())
          user = user->getBlock()->getParentOp();

        if (!op->isBeforeInBlock(user)) {
          if (couldWrite(origUser) ||
              origUser->hasTrait<OpTrait::IsTerminator>()) {
            preserveAllocas.insert(current.second);
          }
          if (!isa<LLVM::LoadOp, memref::LoadOp, AffineLoadOp>(origUser)) {
            for (auto res : origUser->getResults()) {
              if (crossing.contains(res)) {
                preserveAllocas.insert(current.second);
              }
            }
            todo.emplace_back(user, current.second);
          }
        }
      }
    }
  }

  for (auto v : crossing) {
    if (isa<memref::AllocaOp, LLVM::AllocaOp>(v.getDefiningOp())) {
      preserveAllocas.insert(v.getDefiningOp());
    }
  }
}

/// Returns `true` if the given operation has a BarrierOp transitively nested in
/// one of its regions, but not within any nested ParallelOp.
static bool hasNestedBarrier(Operation *op, SmallVector<BlockArgument> &vals) {
  op->walk([&](polygeist::BarrierOp barrier) {
    // If there is a `parallel` op nested inside the given op (alternatively,
    // the `parallel` op is not an ancestor of `op` or `op` itself), the
    // barrier is considered nested in that `parallel` op and _not_ in `op`.
    for (auto arg : barrier->getOperands()) {
      if (auto ba = arg.dyn_cast<BlockArgument>()) {
        if (auto parallel =
                dyn_cast<scf::ParallelOp>(ba.getOwner()->getParentOp())) {
          if (parallel->isAncestor(op))
            vals.push_back(ba);
        } else if (auto parallel = dyn_cast<AffineParallelOp>(
                       ba.getOwner()->getParentOp())) {
          if (parallel->isAncestor(op))
            vals.push_back(ba);
        } else {
          assert(0 && "unknown barrier arg\n");
        }
      } else if (arg.getDefiningOp<ConstantIndexOp>())
        continue;
      else {
        assert(0 && "unknown barrier arg\n");
      }
    }
  });
  return vals.size();
}

namespace {

#if 0
/// Returns `true` if the loop has a form expected by interchange patterns.
static bool isNormalized(scf::ForOp op) {
  return isDefinedAbove(op.getLowerBound(), op) &&
         isDefinedAbove(op.getStep(), op);
}

/// Transforms a loop to the normal form expected by interchange patterns, i.e.
/// with zero lower bound and unit step.
struct NormalizeLoop : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (isNormalized(op) || !isa<scf::ParallelOp, AffineParallelOp>(op->getParentOp())) {
      LLVM_DEBUG(DBGS() << "[normalize-loop] loop already normalized\n");
      return failure();
    }
    if (op.getNumResults()) {
      LLVM_DEBUG(DBGS() << "[normalize-loop] not handling reduction loops\n");
      return failure();
    }

    OpBuilder::InsertPoint point = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(op->getParentOp());
    Value zero = rewriter.create<ConstantIndexOp>(op.getLoc(), 0);
    Value one = rewriter.create<ConstantIndexOp>(op.getLoc(), 1);
    rewriter.restoreInsertionPoint(point);

    Value difference = rewriter.create<SubIOp>(op.getLoc(), op.getUpperBound(),
                                               op.getLowerBound());
    Value tripCount = rewriter.create<AddIOp>(
        op.getLoc(),
        rewriter.create<DivUIOp>(
            op.getLoc(), rewriter.create<SubIOp>(op.getLoc(), difference, one),
            op.getStep()),
        one);
    // rewriter.create<CeilDivSIOp>(op.getLoc(), difference, op.getStep());
    auto newForOp =
        rewriter.create<scf::ForOp>(op.getLoc(), zero, tripCount, one);
    rewriter.setInsertionPointToStart(newForOp.getBody());
    Value scaled = rewriter.create<MulIOp>(
        op.getLoc(), newForOp.getInductionVar(), op.getStep());
    Value iv = rewriter.create<AddIOp>(op.getLoc(), op.getLowerBound(), scaled);
    rewriter.mergeBlockBefore(op.getBody(), &newForOp.getBody()->back(), {iv});
    rewriter.eraseOp(&newForOp.getBody()->back());
    rewriter.eraseOp(op);
    return success();
  }
};
#endif

/// Returns `true` if the loop has a form expected by interchange patterns.
static bool isNormalized(scf::ParallelOp op) {
  auto isZero = [](Value v) {
    APInt value;
    return matchPattern(v, m_ConstantInt(&value)) && value.isNullValue();
  };
  auto isOne = [](Value v) {
    APInt value;
    return matchPattern(v, m_ConstantInt(&value)) && value.isOneValue();
  };
  return llvm::all_of(op.getLowerBound(), isZero) &&
         llvm::all_of(op.getStep(), isOne);
}
static bool isNormalized(AffineParallelOp op) {
  auto isZero = [](AffineExpr v) {
    if (auto ce = v.dyn_cast<AffineConstantExpr>())
      return ce.getValue() == 0;
    return false;
  };
  return llvm::all_of(op.getLowerBoundsMap().getResults(), isZero) &&
         llvm::all_of(op.getSteps(), [](int64_t s) { return s == 1; });
}

/// Transforms a loop to the normal form expected by interchange patterns, i.e.
/// with zero lower bounds and unit steps.
struct NormalizeParallel : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (isNormalized(op)) {
      LLVM_DEBUG(DBGS() << "[normalize-parallel] loop already normalized\n");
      return failure();
    }
    if (op->getNumResults() != 0) {
      LLVM_DEBUG(
          DBGS() << "[normalize-parallel] not processing reduction loops\n");
      return failure();
    }
    SmallVector<BlockArgument> args;
    if (!hasNestedBarrier(op, args)) {
      LLVM_DEBUG(DBGS() << "[normalize-parallel] no nested barrier\n");
      return failure();
    }

    Value zero = rewriter.create<ConstantIndexOp>(op.getLoc(), 0);
    Value one = rewriter.create<ConstantIndexOp>(op.getLoc(), 1);
    SmallVector<Value> iterationCounts = emitIterationCounts(rewriter, op);
    auto newOp = rewriter.create<scf::ParallelOp>(
        op.getLoc(), SmallVector<Value>(iterationCounts.size(), zero),
        iterationCounts, SmallVector<Value>(iterationCounts.size(), one));

    SmallVector<Value> inductionVars;
    inductionVars.reserve(iterationCounts.size());
    rewriter.setInsertionPointToStart(newOp.getBody());
    for (unsigned i = 0, e = iterationCounts.size(); i < e; ++i) {
      Value scaled = rewriter.create<MulIOp>(
          op.getLoc(), newOp.getInductionVars()[i], op.getStep()[i]);
      Value shifted =
          rewriter.create<AddIOp>(op.getLoc(), op.getLowerBound()[i], scaled);
      inductionVars.push_back(shifted);
    }

    rewriter.mergeBlockBefore(op.getBody(), &newOp.getBody()->back(),
                              inductionVars);
    rewriter.eraseOp(&newOp.getBody()->back());
    rewriter.eraseOp(op);
    return success();
  }
};

LogicalResult splitSubLoop(scf::ParallelOp op, PatternRewriter &rewriter,
                           BarrierOp barrier, SmallVector<Value> &iterCounts,
                           scf::ParallelOp &preLoop, scf::ParallelOp &postLoop,
                           Block *&outerBlock, scf::ParallelOp &outerLoop,
                           memref::AllocaScopeOp &outerEx) {

  SmallVector<Value> outerLower;
  SmallVector<Value> outerUpper;
  SmallVector<Value> outerStep;
  SmallVector<Value> innerLower;
  SmallVector<Value> innerUpper;
  SmallVector<Value> innerStep;
  for (auto en : llvm::zip(op.getBody()->getArguments(), op.getLowerBound(),
                           op.getUpperBound(), op.getStep())) {
    bool found = false;
    for (auto v : barrier.getOperands())
      if (v == std::get<0>(en))
        found = true;
    if (found) {
      innerLower.push_back(std::get<1>(en));
      innerUpper.push_back(std::get<2>(en));
      innerStep.push_back(std::get<3>(en));
    } else {
      outerLower.push_back(std::get<1>(en));
      outerUpper.push_back(std::get<2>(en));
      outerStep.push_back(std::get<3>(en));
    }
  }
  if (!innerLower.size())
    return failure();
  if (outerLower.size()) {
    outerLoop = rewriter.create<scf::ParallelOp>(op.getLoc(), outerLower,
                                                 outerUpper, outerStep);
    rewriter.eraseOp(&outerLoop.getBody()->back());
    outerBlock = outerLoop.getBody();
  } else {
    outerEx = rewriter.create<memref::AllocaScopeOp>(op.getLoc(), TypeRange());
    outerBlock = new Block();
    outerEx.getRegion().push_back(outerBlock);
  }

  rewriter.setInsertionPointToEnd(outerBlock);
  for (auto tup : llvm::zip(innerLower, innerUpper, innerStep)) {
    iterCounts.push_back(rewriter.create<DivUIOp>(
        op.getLoc(),
        rewriter.create<SubIOp>(op.getLoc(), std::get<1>(tup),
                                std::get<0>(tup)),
        std::get<2>(tup)));
  }
  preLoop = rewriter.create<scf::ParallelOp>(op.getLoc(), innerLower,
                                             innerUpper, innerStep);
  rewriter.eraseOp(&preLoop.getBody()->back());
  postLoop = rewriter.create<scf::ParallelOp>(op.getLoc(), innerLower,
                                              innerUpper, innerStep);
  rewriter.eraseOp(&postLoop.getBody()->back());
  return success();
}

LogicalResult splitSubLoop(AffineParallelOp op, PatternRewriter &rewriter,
                           BarrierOp barrier, SmallVector<Value> &iterCounts,
                           AffineParallelOp &preLoop,
                           AffineParallelOp &postLoop, Block *&outerBlock,
                           AffineParallelOp &outerLoop,
                           memref::AllocaScopeOp &outerEx) {

  SmallVector<AffineMap> outerLower;
  SmallVector<AffineMap> outerUpper;
  SmallVector<int64_t> outerStep;
  SmallVector<AffineMap> innerLower;
  SmallVector<AffineMap> innerUpper;
  SmallVector<int64_t> innerStep;
  unsigned idx = 0;
  for (auto en : llvm::enumerate(
           llvm::zip(op.getBody()->getArguments(), op.getSteps()))) {
    bool found = false;
    for (auto v : barrier.getOperands())
      if (v == std::get<0>(en.value()))
        found = true;
    if (found) {
      innerLower.push_back(op.getLowerBoundsMap().getSliceMap(en.index(), 1));
      innerUpper.push_back(op.getUpperBoundsMap().getSliceMap(en.index(), 1));
      innerStep.push_back(std::get<1>(en.value()));
    } else {
      outerLower.push_back(op.getLowerBoundsMap().getSliceMap(en.index(), 1));
      outerUpper.push_back(op.getUpperBoundsMap().getSliceMap(en.index(), 1));
      outerStep.push_back(std::get<1>(en.value()));
    }
    idx++;
  }
  if (!innerLower.size())
    return failure();
  if (outerLower.size()) {
    outerLoop = rewriter.create<AffineParallelOp>(
        op.getLoc(), TypeRange(), ArrayRef<AtomicRMWKind>(), outerLower,
        op.getLowerBoundsOperands(), outerUpper, op.getUpperBoundsOperands(),
        outerStep);
    rewriter.eraseOp(&outerLoop.getBody()->back());
    outerBlock = outerLoop.getBody();
  } else {
    outerEx = rewriter.create<memref::AllocaScopeOp>(op.getLoc(), TypeRange());
    outerBlock = new Block();
    outerEx.getRegion().push_back(outerBlock);
  }

  rewriter.setInsertionPointToEnd(outerBlock);
  for (auto tup : llvm::zip(innerLower, innerUpper, innerStep)) {
    auto expr = (std::get<1>(tup).getResult(0) -
                 std::get<0>(tup)
                     .getResult(0)
                     .shiftDims(op.getLowerBoundsMap().getNumDims(),
                                op.getUpperBoundsMap().getNumDims())
                     .shiftSymbols(op.getLowerBoundsMap().getNumSymbols(),
                                   op.getUpperBoundsMap().getNumSymbols()))
                    .floorDiv(std::get<2>(tup));
    SmallVector<Value> symbols;
    SmallVector<Value> dims;
    size_t idx = 0;
    for (auto v : op.getUpperBoundsOperands()) {
      if (idx < op.getUpperBoundsMap().getNumDims())
        dims.push_back(v);
      else
        symbols.push_back(v);
      idx++;
    }
    idx = 0;
    for (auto v : op.getLowerBoundsOperands()) {
      if (idx < op.getLowerBoundsMap().getNumDims())
        dims.push_back(v);
      else
        symbols.push_back(v);
      idx++;
    }
    SmallVector<Value> ops = dims;
    ops.append(symbols);
    iterCounts.push_back(rewriter.create<AffineApplyOp>(
        op.getLoc(), AffineMap::get(dims.size(), symbols.size(), expr), ops));
  }
  preLoop = rewriter.create<AffineParallelOp>(
      op.getLoc(), TypeRange(), ArrayRef<AtomicRMWKind>(), innerLower,
      op.getLowerBoundsOperands(), innerUpper, op.getUpperBoundsOperands(),
      innerStep);
  rewriter.eraseOp(&preLoop.getBody()->back());
  postLoop = rewriter.create<AffineParallelOp>(
      op.getLoc(), TypeRange(), ArrayRef<AtomicRMWKind>(), innerLower,
      op.getLowerBoundsOperands(), innerUpper, op.getUpperBoundsOperands(),
      innerStep);
  rewriter.eraseOp(&postLoop.getBody()->back());
  return success();
}

template <typename T, bool UseMinCut>
static LogicalResult distributeAroundBarrier(T op, BarrierOp barrier,
                                             T &preLoop, T &postLoop,
                                             PatternRewriter &rewriter) {
  if (op.getNumResults() != 0) {
    LLVM_DEBUG(DBGS() << "[distribute] not matching reduction loops\n");
    return failure();
  }

  if (!isNormalized(op)) {
    LLVM_DEBUG(DBGS() << "[distribute] non-normalized loop\n");
    return failure();
  }

  llvm::SetVector<Value> usedBelow;
  llvm::SetVector<Operation *> preserveAllocas;
  findValuesUsedBelow(barrier, usedBelow, preserveAllocas);

  llvm::SetVector<Value> crossingCache;
  if (UseMinCut) {

    minCutCache(barrier, usedBelow, crossingCache);

    LLVM_DEBUG(DBGS() << "[distribute] min cut cache optimisation: "
                      << "preserveAllocas: " << preserveAllocas.size() << ", "
                      << "usedBelow: " << usedBelow.size() << ", "
                      << "crossingCache: " << crossingCache.size() << "\n");

    BlockAndValueMapping mapping;
    for (auto v : crossingCache)
      mapping.map(v, v);

    // Recalculate values used below the barrier up to available ones
    rewriter.setInsertionPointAfter(barrier);
    std::function<void(Operation *)> recalculateOp;
    recalculateOp = [&recalculateOp, &barrier, &mapping,
                     &rewriter](Operation *op) {
      Operation *pop = barrier->getParentOp();
      if (!pop->isProperAncestor(op))
        return;

      // We always have to recalculate operands of yields, otherwise check if we
      // don't already have the results
      if (!isa<scf::YieldOp, AffineYieldOp>(op) &&
          llvm::all_of(op->getResults(),
                       [&mapping](Value v) { return mapping.contains(v); }))
        return;

      for (Value operand : op->getOperands())
        if (auto operandOp = operand.getDefiningOp())
          recalculateOp(operandOp);
      for (Region &region : op->getRegions())
        for (auto &block : region)
          for (auto &nestedOp : block)
            recalculateOp(&nestedOp);

      if (op->getBlock() == barrier->getBlock())
        rewriter.clone(*op, mapping);
      else
        for (Value v : op->getResults())
          mapping.map(v, v);
    };

    for (auto v : usedBelow) {
      Operation *vOp = v.getDefiningOp();
      assert(vOp && "values used below barrier must be results of operations");
      recalculateOp(vOp);
      // Remap the uses of the recalculated val below the barrier
      for (auto &u : llvm::make_early_inc_range(v.getUses())) {
        auto *user = u.getOwner();
        while (user->getBlock() != barrier->getBlock())
          user = user->getBlock()->getParentOp();
        if (barrier->isBeforeInBlock(user)) {
          rewriter.startRootUpdate(user);
          u.set(mapping.lookup(v));
          rewriter.finalizeRootUpdate(user);
        }
      }
    }
  } else {
    crossingCache = usedBelow;
  }

  for (auto *alloca : preserveAllocas) {
    crossingCache.remove(alloca->getResult(0));
  }

  SmallVector<Value> iterCounts;

  Block *outerBlock;
  T outerLoop = nullptr;
  memref::AllocaScopeOp outerEx = nullptr;

  rewriter.setInsertionPoint(op);
  if (splitSubLoop(op, rewriter, barrier, iterCounts, preLoop, postLoop,
                   outerBlock, outerLoop, outerEx)
          .failed())
    return failure();

  assert(iterCounts.size() == preLoop.getBody()->getArguments().size());

  size_t outIdx = 0;
  size_t inIdx = 0;
  for (auto en : op.getBody()->getArguments()) {
    bool found = false;
    for (auto v : barrier.getOperands())
      if (v == en)
        found = true;
    if (found) {
      en.replaceAllUsesWith(preLoop.getBody()->getArguments()[inIdx]);
      inIdx++;
    } else {
      en.replaceAllUsesWith(outerLoop.getBody()->getArguments()[outIdx]);
      outIdx++;
    }
  }
  op.getBody()->eraseArguments([](BlockArgument) { return true; });
  rewriter.mergeBlocks(op.getBody(), preLoop.getBody());

  rewriter.setInsertionPoint(preLoop);
  // Allocate space for values crossing the barrier.
  SmallVector<Value> cacheAllocations;
  SmallVector<Value> allocaAllocations;
  cacheAllocations.reserve(crossingCache.size());
  allocaAllocations.reserve(preserveAllocas.size());
  auto mod = ((Operation *)op)->getParentOfType<ModuleOp>();
  assert(mod);
  DataLayout DLI(mod);
  auto addToAllocations = [&](Value v, SmallVector<Value> &allocations) {
    if (auto cl = v.getDefiningOp<polygeist::CacheLoad>()) {
      allocations.push_back(cl.getMemref());
    } else if (auto ao = v.getDefiningOp<LLVM::AllocaOp>()) {
      allocations.push_back(allocateTemporaryBuffer<LLVM::AllocaOp>(
          rewriter, v, iterCounts, true, &DLI));
    } else {
      allocations.push_back(
          allocateTemporaryBuffer<memref::AllocaOp>(rewriter, v, iterCounts));
    }
  };
  for (Value v : crossingCache)
    addToAllocations(v, cacheAllocations);
  for (Operation *o : preserveAllocas)
    addToAllocations(o->getResult(0), allocaAllocations);

  // Allocate alloca's we need to preserve outside the loop
  for (auto pair : llvm::zip(preserveAllocas, allocaAllocations)) {
    Operation *o = std::get<0>(pair);
    Value alloc = std::get<1>(pair);
    if (auto ao = dyn_cast<memref::AllocaOp>(o)) {
      for (auto &u : llvm::make_early_inc_range(ao.getResult().getUses())) {
        rewriter.setInsertionPoint(u.getOwner());
        auto buf = alloc;
        for (auto idx : preLoop.getBody()->getArguments()) {
          auto mt0 = buf.getType().cast<MemRefType>();
          std::vector<int64_t> shape(mt0.getShape());
          assert(shape.size() > 0);
          shape.erase(shape.begin());
          auto mt = MemRefType::get(shape, mt0.getElementType(),
                                    MemRefLayoutAttrInterface(),
                                    // mt0.getLayout(),
                                    mt0.getMemorySpace());
          auto subidx = rewriter.create<polygeist::SubIndexOp>(alloc.getLoc(),
                                                               mt, buf, idx);
          buf = subidx;
        }
        u.set(buf);
      }
      rewriter.eraseOp(ao);
    } else if (auto ao = dyn_cast<LLVM::AllocaOp>(o)) {
      Value sz = ao.getArraySize();
      rewriter.setInsertionPointAfter(alloc.getDefiningOp());
      alloc =
          rewriter.create<LLVM::BitcastOp>(ao.getLoc(), ao.getType(), alloc);
      for (auto &u : llvm::make_early_inc_range(ao.getResult().getUses())) {
        rewriter.setInsertionPoint(u.getOwner());
        Value idx = nullptr;
        // i0
        // i0 * s1 + i1
        // ( i0 * s1 + i1 ) * s2 + i2
        for (auto pair :
             llvm::zip(iterCounts, preLoop.getBody()->getArguments())) {
          if (idx) {
            idx = rewriter.create<arith::MulIOp>(ao.getLoc(), idx,
                                                 std::get<0>(pair));
            idx = rewriter.create<arith::AddIOp>(ao.getLoc(), idx,
                                                 std::get<1>(pair));
          } else
            idx = std::get<1>(pair);
        }
        idx = rewriter.create<MulIOp>(ao.getLoc(), sz,
                                      rewriter.create<arith::IndexCastOp>(
                                          ao.getLoc(), sz.getType(), idx));
        SmallVector<Value> vec = {idx};
        u.set(rewriter.create<LLVM::GEPOp>(ao.getLoc(), ao.getType(), alloc,
                                           idx));
      }
    } else {
      assert(false && "Wrong operation type in preserveAllocas");
    }
  }

  // Store values in the min cache immediately when ready and reload them
  // after the barrier
  for (auto pair : llvm::zip(crossingCache, cacheAllocations)) {
    Value v = std::get<0>(pair);
    Value alloc = std::get<1>(pair);

    // No need to store cache loads
    if (!isa<polygeist::CacheLoad>(v.getDefiningOp())) {
      // Store
      rewriter.setInsertionPointAfter(v.getDefiningOp());
      rewriter.create<memref::StoreOp>(v.getLoc(), v, alloc,
                                       preLoop.getBody()->getArguments());
    }
    // Reload
    rewriter.setInsertionPointAfter(barrier);
    Value reloaded = rewriter.create<polygeist::CacheLoad>(
        v.getLoc(), alloc, preLoop.getBody()->getArguments());
    for (auto &u : llvm::make_early_inc_range(v.getUses())) {
      auto *user = u.getOwner();
      while (user->getBlock() != barrier->getBlock())
        user = user->getBlock()->getParentOp();

      if (barrier->isBeforeInBlock(user)) {
        rewriter.startRootUpdate(user);
        u.set(reloaded);
        rewriter.finalizeRootUpdate(user);
      }
    }
  }

  // Insert the terminator for the new loop immediately before the barrier.
  rewriter.setInsertionPoint(barrier);
  rewriter.clone(preLoop.getBody()->back());
  Operation *postBarrier = barrier->getNextNode();
  rewriter.eraseOp(barrier);

  // Create the second loop.
  rewriter.setInsertionPointToEnd(outerBlock);
  if (outerLoop) {
    if (isa<scf::ParallelOp>(outerLoop))
      rewriter.create<scf::YieldOp>(op.getLoc());
    else {
      assert(isa<AffineParallelOp>(outerLoop));
      rewriter.create<AffineYieldOp>(op.getLoc());
    }
  } else {
    rewriter.create<memref::AllocaScopeReturnOp>(op.getLoc());
  }

  // Recreate the operations in the new loop with new values.
  rewriter.setInsertionPointToStart(postLoop.getBody());
  BlockAndValueMapping mapping;
  mapping.map(preLoop.getBody()->getArguments(),
              postLoop.getBody()->getArguments());
  SmallVector<Operation *> toDelete;
  for (Operation *o = postBarrier; o != nullptr; o = o->getNextNode()) {
    rewriter.clone(*o, mapping);
    toDelete.push_back(o);
  }

  // Erase original operations and the barrier.
  for (Operation *o : llvm::reverse(toDelete))
    rewriter.eraseOp(o);

  rewriter.eraseOp(op);

  LLVM_DEBUG(DBGS() << "[distribute] distributed around a barrier\n");
  return success();
}
template <typename T, bool UseMinCut>
static LogicalResult distributeAroundFirstBarrier(T op, T &preLoop, T &postLoop,
                                                  PatternRewriter &rewriter) {
  BarrierOp barrier = nullptr;
  {
    auto it =
        llvm::find_if(op.getBody()->getOperations(), [](Operation &nested) {
          return isa<polygeist::BarrierOp>(nested);
        });
    if (it == op.getBody()->end()) {
      LLVM_DEBUG(DBGS() << "[distribute] no barrier in the loop\n");
      return failure();
    }
    barrier = cast<BarrierOp>(&*it);
  }

  return distributeAroundBarrier<T, UseMinCut>(op, barrier, preLoop, postLoop,
                                               rewriter);
}
template <typename T, bool UseMinCut>
static LogicalResult distributeAroundFirstBarrier(T op,
                                                  PatternRewriter &rewriter) {
  T preLoop, postLoop;
  return distributeAroundFirstBarrier<T, UseMinCut>(op, preLoop, postLoop,
                                                    rewriter);
}

/// Splits a parallel loop around the first barrier it immediately contains.
/// Values defined before the barrier are stored in newly allocated buffers and
/// loaded back when needed.
template <typename T, bool UseMinCut>
struct DistributeAroundBarrier : public OpRewritePattern<T> {
  DistributeAroundBarrier(MLIRContext *ctx) : OpRewritePattern<T>(ctx) {}

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    return distributeAroundFirstBarrier<T, UseMinCut>(op, rewriter);
  }
};

/// Checks if `op` may need to be wrapped in a pair of barriers. This is a
/// necessary but insufficient condition.
static LogicalResult canWrapWithBarriers(Operation *op,
                                         SmallVector<BlockArgument> &vals) {
  if (!isa<scf::ParallelOp, AffineParallelOp>(op->getParentOp())) {
    LLVM_DEBUG(DBGS() << "[wrap] not nested in a pfor\n");
    return failure();
  }

  if (op->getNumResults() != 0) {
    LLVM_DEBUG(DBGS() << "[wrap] ignoring loop with reductions\n");
    return failure();
  }

  if (!hasNestedBarrier(op, vals)) {
    LLVM_DEBUG(DBGS() << "[wrap] no nested barrier\n");
    return failure();
  }

  return success();
}

bool isBarrierContainingAll(Operation *op, SmallVector<BlockArgument> &args) {
  auto bar = dyn_cast<polygeist::BarrierOp>(op);
  if (!bar)
    return false;
  SmallPtrSet<Value, 3> bargs(op->getOperands().begin(),
                              op->getOperands().end());
  for (auto a : args)
    if (!bargs.contains(a))
      return false;
  return true;
}

/// Puts a barrier before and/or after `op` if there isn't already one.
/// `extraPrevCheck` is called on the operation immediately preceding `op` and
/// can be used to look further upward if the immediately preceding operation is
/// not a barrier.
template <typename T>
static LogicalResult
wrapWithBarriers(T op, PatternRewriter &rewriter,
                 SmallVector<BlockArgument> &args, bool recomputable,
                 polygeist::BarrierOp &before, polygeist::BarrierOp &after) {
  Operation *prevOp = op->getPrevNode();
  Operation *nextOp = op->getNextNode();
  bool hasPrevBarrierLike =
      prevOp == nullptr || isBarrierContainingAll(prevOp, args) || recomputable;
  bool hasNextBarrierLike =
      nextOp == &op->getBlock()->back() || isBarrierContainingAll(nextOp, args);

  if (hasPrevBarrierLike && hasNextBarrierLike) {
    LLVM_DEBUG(DBGS() << "[wrap] already has sufficient barriers\n");
    return failure();
  }

  SmallVector<Value> vargs(args.begin(), args.end());

  if (!hasPrevBarrierLike)
    before = rewriter.create<polygeist::BarrierOp>(op->getLoc(), vargs);

  if (!hasNextBarrierLike) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);
    after = rewriter.create<polygeist::BarrierOp>(op->getLoc(), vargs);
  }

  // We don't actually change the op, but the pattern infra wants us to. Just
  // pretend we changed it in-place.
  rewriter.updateRootInPlace(op, [] {});
  LLVM_DEBUG(DBGS() << "[wrap] wrapped '" << op->getName().getStringRef()
                    << "' with barriers\n");
  return success();
}
template <typename T>
static LogicalResult wrapWithBarriers(T op, PatternRewriter &rewriter,
                                      SmallVector<BlockArgument> &args,
                                      bool recomputable) {
  polygeist::BarrierOp before, after;
  return wrapWithBarriers(op, rewriter, args, recomputable, before, after);
}

template <typename T, bool UseMinCut>
static LogicalResult distributeAfterWrap(Operation *pop, BarrierOp barrier,
                                         PatternRewriter &rewriter) {
  if (!barrier)
    return failure();
  T preLoop, postLoop;
  if (auto cast = dyn_cast<T>(pop)) {
    if (failed(distributeAroundBarrier<T, UseMinCut>(cast, barrier, preLoop,
                                                     postLoop, rewriter)))
      return failure();
    return success();
  } else {
    return failure();
  }
}

template <typename T, bool UseMinCut>
static LogicalResult wrapAndDistribute(T op, bool singleExecution,
                                       PatternRewriter &rewriter) {
  SmallVector<BlockArgument> vals;
  if (failed(canWrapWithBarriers(op, vals)))
    return failure();

  bool recomputable = arePreceedingOpsFullyRecomputable(op, singleExecution);
  if (recomputable && isa<scf::YieldOp, AffineYieldOp>(op->getNextNode())) {
    return failure();
  }

  polygeist::BarrierOp before, after;
  if (failed(
          wrapWithBarriers(op, rewriter, vals, recomputable, before, after))) {
    return failure();
  }

  // We have now introduced one or two barriers, distribute around the one
  // before the op immediately (if it exists), the one after can be handled by
  // the distribute pass, we need to do this now to prevent BarrierElim from
  // eliminating it in some cases when now two barriers appear before `op` and
  // only one of them is necessary
  auto pop = op->getParentOp();
  (void)distributeAfterWrap<scf::ParallelOp, UseMinCut>(pop, before, rewriter);
  (void)distributeAfterWrap<AffineParallelOp, UseMinCut>(pop, before, rewriter);

  return success();
}

/// Puts a barrier before and/or after an "if" operation if there isn't already
/// one, potentially with a single load that supplies the upper bound of a
/// (normalized) loop.
template <typename IfType, bool UseMinCut>
struct WrapIfWithBarrier : public OpRewritePattern<IfType> {
  WrapIfWithBarrier(MLIRContext *ctx) : OpRewritePattern<IfType>(ctx) {}
  LogicalResult matchAndRewrite(IfType op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 0)
      return failure();

    return wrapAndDistribute<IfType, UseMinCut>(op, /* singleExecution */ true,
                                                rewriter);
  }
};

/// Puts a barrier before and/or after a "for" operation if there isn't already
/// one, potentially with a single load that supplies the upper bound of a
/// (normalized) loop.
template <bool UseMinCut>
struct WrapForWithBarrier : public OpRewritePattern<scf::ForOp> {
  WrapForWithBarrier(MLIRContext *ctx) : OpRewritePattern<scf::ForOp>(ctx) {}

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    return wrapAndDistribute<scf::ForOp, UseMinCut>(
        op, /* singleExecution */ false, rewriter);
  }
};

template <bool UseMinCut>
struct WrapAffineForWithBarrier : public OpRewritePattern<AffineForOp> {
  WrapAffineForWithBarrier(MLIRContext *ctx)
      : OpRewritePattern<AffineForOp>(ctx) {}

  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    return wrapAndDistribute<AffineForOp, UseMinCut>(
        op, /* singleExecution */ false, rewriter);
  }
};

/// Puts a barrier before and/or after a "while" operation if there isn't
/// already one.
template <bool UseMinCut>
struct WrapWhileWithBarrier : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 0 || op.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[wrap-while] ignoring non-mem2reg'd loop ops: "
                        << op.getNumOperands() << " res: " << op.getNumResults()
                        << "\n";);
      return failure();
    }

    return wrapAndDistribute<scf::WhileOp, UseMinCut>(
        op, /* singleExecution */ false, rewriter);
  }
};

// Clone the recomputable ops from the old parallel to the new one up until the
// until op (we are excluding load ops that provide bounds conditions)
template <typename T, typename T2>
static void insertRecomputables(PatternRewriter &rewriter, T oldParallel,
                                T newParallel, T2 until) {
  rewriter.setInsertionPointToStart(newParallel.getBody());
  BlockAndValueMapping mapping;
  mapping.map(oldParallel.getBody()->getArguments(),
              newParallel.getBody()->getArguments());
  rewriter.setInsertionPointToStart(newParallel.getBody());
  for (auto it = oldParallel.getBody()->begin(); dyn_cast<T2>(*it) != until;
       ++it) {
    auto newOp = rewriter.clone(*it, mapping);
    rewriter.replaceOpWithinBlock(&*it, newOp->getResults(),
                                  newParallel.getBody());
  }
}

/// Moves the body from `ifOp` contained in `op` to a parallel op newly
/// created at the start of `newIf`.
template <typename T, typename IfType>
static void moveBodiesIf(PatternRewriter &rewriter, T op, IfType ifOp,
                         IfType newIf) {
  rewriter.startRootUpdate(op);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(getThenBlock(newIf));
    auto newParallel = rewriter.cloneWithoutRegions<T>(op);
    newParallel.getRegion().push_back(new Block());
    for (auto a : op.getBody()->getArguments())
      newParallel.getBody()->addArgument(a.getType(), op->getLoc());

    rewriter.setInsertionPointToEnd(newParallel.getBody());
    rewriter.clone(*op.getBody()->getTerminator());

    for (auto tup : llvm::zip(newParallel.getBody()->getArguments(),
                              op.getBody()->getArguments())) {
      // TODO do we not have to use the rewriter here?
      std::get<1>(tup).replaceUsesWithIf(
          std::get<0>(tup), [&](OpOperand &op) -> bool {
            return getThenBlock(ifOp)->getParent()->isAncestor(
                op.getOwner()->getParentRegion());
          });
    }

    rewriter.eraseOp(&getThenBlock(ifOp)->back());
    rewriter.mergeBlockBefore(getThenBlock(ifOp),
                              &newParallel.getBody()->back());

    insertRecomputables(rewriter, op, newParallel, ifOp);
  }

  if (hasElse(ifOp)) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(getElseBlock(newIf));
    auto newParallel = rewriter.cloneWithoutRegions<T>(op);
    newParallel.getRegion().push_back(new Block());
    for (auto a : op.getBody()->getArguments())
      newParallel.getBody()->addArgument(a.getType(), op->getLoc());

    rewriter.setInsertionPointToEnd(newParallel.getBody());
    rewriter.clone(*op.getBody()->getTerminator());

    for (auto tup : llvm::zip(newParallel.getBody()->getArguments(),
                              op.getBody()->getArguments())) {
      std::get<1>(tup).replaceUsesWithIf(
          std::get<0>(tup), [&](OpOperand &op) -> bool {
            return getElseBlock(ifOp)->getParent()->isAncestor(
                op.getOwner()->getParentRegion());
          });
    }
    rewriter.eraseOp(&getElseBlock(ifOp)->back());
    rewriter.mergeBlockBefore(getElseBlock(ifOp),
                              &newParallel.getBody()->back());

    insertRecomputables(rewriter, op, newParallel, ifOp);
  }

  rewriter.eraseOp(ifOp);
  rewriter.eraseOp(op);
  rewriter.finalizeRootUpdate(op);
}

mlir::OperandRange getLowerBounds(scf::ParallelOp op,
                                  PatternRewriter &rewriter) {
  return op.getLowerBound();
}
SmallVector<Value> getLowerBounds(AffineParallelOp op,
                                  PatternRewriter &rewriter) {
  SmallVector<Value> vals;
  for (AffineExpr expr : op.getLowerBoundsMap().getResults()) {
    vals.push_back(rewriter
                       .create<AffineApplyOp>(op.getLoc(), expr,
                                              op.getLowerBoundsOperands())
                       .getResult());
  }
  return vals;
}

/// Moves the body from `forLoop` contained in `op` to a parallel op newly
/// created at the start of `newForLoop`.
template <typename T, typename ForType>
static void moveBodiesFor(PatternRewriter &rewriter, T op, ForType forLoop,
                          ForType newForLoop) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(newForLoop.getBody());
  auto newParallel = rewriter.cloneWithoutRegions<T>(op);
  newParallel.getRegion().push_back(new Block());
  for (auto a : op.getBody()->getArguments())
    newParallel.getBody()->addArgument(a.getType(), op->getLoc());

  // Keep recomputable values in the parallel op (explicitly excluding loads
  // that provide for bounds as those are handles in the caller)
  BlockAndValueMapping mapping;
  mapping.map(op.getBody()->getArguments(),
              newParallel.getBody()->getArguments());
  rewriter.setInsertionPointToEnd(newParallel.getBody());
  for (auto it = op.getBody()->begin(); dyn_cast<ForType>(*it) != forLoop;
       ++it) {
    auto newOp = rewriter.clone(*it, mapping);
    rewriter.replaceOpWithinBlock(&*it, newOp->getResults(), forLoop.getBody());
  }
  rewriter.setInsertionPointToEnd(newParallel.getBody());
  rewriter.clone(*op.getBody()->getTerminator());

  // Merge in two stages so we can properly replace uses of two induction
  // varibales defined in different blocks.
  rewriter.mergeBlockBefore(op.getBody(), &newParallel.getBody()->back(),
                            newParallel.getBody()->getArguments());
  rewriter.eraseOp(&newParallel.getBody()->back());
  rewriter.eraseOp(&forLoop.getBody()->back());
  rewriter.mergeBlockBefore(forLoop.getBody(), &newParallel.getBody()->back(),
                            newForLoop.getBody()->getArguments());
  rewriter.eraseOp(op);
  rewriter.eraseOp(forLoop);
}

// TODO is this the best way to do this
template <typename ParallelOpType>
static void moveBodies(PatternRewriter &rewriter, ParallelOpType op,
                       scf::IfOp forIf, scf::IfOp newForIf) {
  moveBodiesIf(rewriter, op, forIf, newForIf);
}
template <typename ParallelOpType>
static void moveBodies(PatternRewriter &rewriter, ParallelOpType op,
                       AffineIfOp forIf, AffineIfOp newForIf) {
  moveBodiesIf(rewriter, op, forIf, newForIf);
}
template <typename ParallelOpType>
static void moveBodies(PatternRewriter &rewriter, ParallelOpType op,
                       scf::ForOp forIf, scf::ForOp newForIf) {
  moveBodiesFor(rewriter, op, forIf, newForIf);
}
template <typename ParallelOpType>
static void moveBodies(PatternRewriter &rewriter, ParallelOpType op,
                       AffineForOp forIf, AffineForOp newForIf) {
  moveBodiesFor(rewriter, op, forIf, newForIf);
}

/// Interchanges a parallel for loop with a for loop perfectly nested within it.

/// Interchanges a parallel for loop with an if perfectly nested within it.

/// Interchanges a parallel for loop with a normalized (zero lower bound and
/// unit step) for loop nested within it. The for loop must have a barrier
/// inside and is preceeded by a load operation that supplies its upper bound.
/// The barrier semantics implies that all threads must executed the same number
/// of times, which means that the inner loop must have the same trip count in
/// all iterations of the outer loop. Therefore, the load of the upper bound can
/// be hoisted and read any value, because all values are identical in a
/// semantically valid program.
template <typename ParallelOpType, typename ForIfType>
struct InterchangeForIfPFor : public OpRewritePattern<ParallelOpType> {
  InterchangeForIfPFor(MLIRContext *ctx)
      : OpRewritePattern<ParallelOpType>(ctx) {}

  LogicalResult matchAndRewrite(ParallelOpType op,
                                PatternRewriter &rewriter) const override {
    // Check if the block consists of recomputable operations (either ops with
    // no side effects or polygeist cache loads) and with the last operation of
    // type ForIfType which has a nested barrier
    if (std::next(op.getBody()->begin(), 1) == op.getBody()->end()) {
      LLVM_DEBUG(DBGS() << "[interchange] expected one or more nested ops\n");
      return failure();
    }

    // The actual last op is a yield, get the one before that
    auto lastOpIt = std::prev(op.getBody()->end(), 2);
    auto lastOp = dyn_cast<ForIfType>(*lastOpIt);
    if (!lastOp) {
      LLVM_DEBUG(DBGS() << "[interchange] unexpected last op type\n");
      return failure();
    }

    // We shouldn't have parallel reduction loops coming from GPU anyway, and
    // sequential reduction loops can be transformed by reg2mem.
    if (op.getNumResults() != 0 || lastOp.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[interchange] not matching reduction loops\n");
      return failure();
    }

    SmallVector<BlockArgument> args;
    if (!hasNestedBarrier(lastOp, args)) {
      LLVM_DEBUG(DBGS() << "[interchange] no nested barrier\n");
      return failure();
    }

    if (!arePreceedingOpsFullyRecomputable(
            lastOp, /* singleExecution */ isa<scf::IfOp, AffineIfOp>(
                (Operation *)lastOp))) {
      LLVM_DEBUG(DBGS() << "[interchange] found a nonrecomputable op\n");
      return failure();
    }

    // In the GPU model, the trip count of the inner sequential containing a
    // barrier must be the same for all threads. So read the value written by
    // the first thread outside of the loop to enable interchange.

    // Replicate the recomputable ops in case the condition or bound of lastOp
    // is getting "recomputed"
    BlockAndValueMapping mapping;
    rewriter.setInsertionPoint(op);
    mapping.map(op.getBody()->getArguments(), getLowerBounds(op, rewriter));
    rewriter.setInsertionPoint(op);
    for (auto it = op.getBody()->begin(); &*it != lastOp; ++it)
      rewriter.clone(*it, mapping);

    auto newOp = cloneWithoutResults(lastOp, rewriter, mapping);
    moveBodies(rewriter, op, lastOp, newOp);
    return success();
  }
};

/// Interchanges a parallel for loop with a while loop it contains. The while
/// loop is expected to have an empty "after" region.
template <typename T> struct InterchangeWhilePFor : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    if (std::next(op.getBody()->begin(), 1) == op.getBody()->end()) {
      LLVM_DEBUG(
          DBGS() << "[interchange-while] expected one or more nested ops\n");
      return failure();
    }

    auto whileOp = dyn_cast<scf::WhileOp>(op.getBody()->back().getPrevNode());
    if (!whileOp) {
      LLVM_DEBUG(DBGS() << "[interchange-while] not a while nest\n");
      return failure();
    }
    if (whileOp.getNumOperands() != 0 || whileOp.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[interchange-while] loop-carried values\n");
      return failure();
    }
    SmallVector<BlockArgument> args;
    if (!hasNestedBarrier(whileOp, args)) {
      LLVM_DEBUG(DBGS() << "[interchange-while] no nested barrier\n");
      return failure();
    }

    if (!arePreceedingOpsFullyRecomputable(whileOp,
                                           /* singleExecution */ false)) {
      LLVM_DEBUG(DBGS() << "[interchange-while] found a nonrecomputable op\n");
      return failure();
    }

    auto conditionOp =
        cast<scf::ConditionOp>(whileOp.getBefore().front().back());

    auto makeNewParallelOp = [&]() {
      rewriter.setInsertionPointAfter(op);
      auto newParallel = rewriter.cloneWithoutRegions<T>(op);
      newParallel.getRegion().push_back(new Block());
      for (auto a : op.getBody()->getArguments())
        newParallel.getBody()->addArgument(a.getType(), a.getLoc());
      rewriter.setInsertionPointToEnd(newParallel.getBody());
      rewriter.clone(*op.getBody()->getTerminator());
      return newParallel;
    };
    auto beforeParallelOp = makeNewParallelOp();
    auto afterParallelOp = makeNewParallelOp();

    rewriter.mergeBlockBefore(&whileOp.getBefore().front(),
                              beforeParallelOp.getBody()->getTerminator());
    whileOp.getBefore().push_back(new Block());
    conditionOp->moveBefore(&whileOp.getBefore().front(),
                            whileOp.getBefore().front().begin());
    beforeParallelOp->moveBefore(&whileOp.getBefore().front(),
                                 whileOp.getBefore().front().begin());

    auto yieldOp = cast<scf::YieldOp>(whileOp.getAfter().front().back());

    rewriter.mergeBlockBefore(&whileOp.getAfter().front(),
                              afterParallelOp.getBody()->getTerminator());
    whileOp.getAfter().push_back(new Block());
    yieldOp->moveBefore(&whileOp.getAfter().front(),
                        whileOp.getAfter().front().begin());
    afterParallelOp->moveBefore(&whileOp.getAfter().front(),
                                whileOp.getAfter().front().begin());

    insertRecomputables(rewriter, op, beforeParallelOp, whileOp);
    insertRecomputables(rewriter, op, afterParallelOp, whileOp);

    for (auto tup : llvm::zip(op.getBody()->getArguments(),
                              beforeParallelOp.getBody()->getArguments(),
                              afterParallelOp.getBody()->getArguments())) {
      std::get<0>(tup).replaceUsesWithIf(std::get<1>(tup), [&](OpOperand &op) {
        return beforeParallelOp.getRegion().isAncestor(
            op.getOwner()->getParentRegion());
      });
      std::get<0>(tup).replaceUsesWithIf(std::get<2>(tup), [&](OpOperand &op) {
        return afterParallelOp.getRegion().isAncestor(
            op.getOwner()->getParentRegion());
      });
    }

    whileOp->moveBefore(op);
    rewriter.eraseOp(op);

    Operation *conditionDefiningOp = conditionOp.getCondition().getDefiningOp();
    if (conditionDefiningOp &&
        !conditionOp.getCondition().getParentRegion()->isAncestor(
            whileOp->getParentRegion())) {
      rewriter.setInsertionPoint(beforeParallelOp);
      Value allocated = rewriter.create<memref::AllocaOp>(
          conditionDefiningOp->getLoc(),
          MemRefType::get({}, rewriter.getI1Type()));
      rewriter.setInsertionPointAfter(conditionDefiningOp);
      Value cond = rewriter.create<ConstantIntOp>(conditionDefiningOp->getLoc(),
                                                  true, 1);
      for (auto tup : llvm::zip(getLowerBounds(beforeParallelOp, rewriter),
                                beforeParallelOp.getBody()->getArguments())) {
        cond = rewriter.create<AndIOp>(
            conditionDefiningOp->getLoc(),
            rewriter.create<CmpIOp>(conditionDefiningOp->getLoc(),
                                    CmpIPredicate::eq, std::get<0>(tup),
                                    std::get<1>(tup)),
            cond);
      }
      auto ifOp =
          rewriter.create<scf::IfOp>(conditionDefiningOp->getLoc(), cond);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      rewriter.create<memref::StoreOp>(conditionDefiningOp->getLoc(),
                                       conditionOp.getCondition(), allocated);

      rewriter.setInsertionPoint(conditionOp);

      Value reloaded = rewriter.create<memref::LoadOp>(
          conditionDefiningOp->getLoc(), allocated);
      rewriter.replaceOpWithNewOp<scf::ConditionOp>(conditionOp, reloaded,
                                                    ValueRange());
    }
    return success();
  }
};

/// Moves the "after" region of a while loop into its "before" region using a
/// conditional, that is
///
/// scf.while {
///   @before()
///   scf.conditional(%cond)
/// } do {
///   @after()
///   scf.yield
/// }
///
/// is transformed into
///
/// scf.while {
///   @before()
///   scf.if (%cond) {
///     @after()
///   }
///   scf.conditional(%cond)
/// } do {
///   scf.yield
/// }
struct RotateWhile : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    if (llvm::hasSingleElement(op.getAfter().front())) {
      LLVM_DEBUG(DBGS() << "[rotate-while] the after region is empty");
      return failure();
    }
    SmallVector<BlockArgument> args;
    if (!hasNestedBarrier(op, args)) {
      LLVM_DEBUG(DBGS() << "[rotate-while] no nested barrier\n");
      return failure();
    }
    if (op.getNumOperands() != 0 || op.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[rotate-while] loop-carried values\n");
      return failure();
    }

    auto condition = cast<scf::ConditionOp>(op.getBefore().front().back());
    rewriter.setInsertionPoint(condition);
    auto conditional =
        rewriter.create<scf::IfOp>(op.getLoc(), condition.getCondition());
    rewriter.mergeBlockBefore(&op.getAfter().front(),
                              &conditional.getBody()->back());
    rewriter.eraseOp(&conditional.getBody()->back());

    rewriter.createBlock(&op.getAfter());
    rewriter.clone(conditional.getBody()->back());

    LLVM_DEBUG(DBGS() << "[rotate-while] done\n");
    return success();
  }
};

template <typename T = memref::LoadOp>
static void loadValues(Location loc, ArrayRef<Value> pointers,
                       PatternRewriter &rewriter,
                       SmallVectorImpl<Value> &loaded) {
  loaded.reserve(loaded.size() + pointers.size());
  for (Value alloc : pointers)
    loaded.push_back(rewriter.create<T>(loc, alloc, ValueRange()));
}

template <typename T, bool UseMinCut>
struct Reg2MemFor : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    SmallVector<BlockArgument> args;
    if (op.getNumResults() == 0 || !hasNestedBarrier(op, args))
      return failure();

    if (!isa<scf::ParallelOp, AffineParallelOp>(op->getParentOp())) {
      return failure();
    }

    SmallVector<Value> allocated;
    allocated.reserve(op.getNumIterOperands());
    for (Value operand : op.getIterOperands()) {
      Value alloc = rewriter.create<memref::AllocaOp>(
          op.getLoc(), MemRefType::get(ArrayRef<int64_t>(), operand.getType()),
          ValueRange());
      allocated.push_back(alloc);
      if (!operand.getDefiningOp<LLVM::UndefOp>())
        rewriter.create<memref::StoreOp>(op.getLoc(), operand, alloc,
                                         ValueRange());
    }

    auto newOp = cloneWithoutResults(op, rewriter);
    rewriter.setInsertionPointToStart(newOp.getBody());
    SmallVector<Value> newRegionArguments;
    newRegionArguments.push_back(newOp.getInductionVar());
    if (UseMinCut)
      loadValues<polygeist::CacheLoad>(op.getLoc(), allocated, rewriter,
                                       newRegionArguments);
    else
      loadValues<memref::LoadOp>(op.getLoc(), allocated, rewriter,
                                 newRegionArguments);

    auto oldTerminator = op.getBody()->getTerminator();
    rewriter.mergeBlockBefore(op.getBody(), newOp.getBody()->getTerminator(),
                              newRegionArguments);
    SmallVector<Value> oldOps;
    llvm::append_range(oldOps, oldTerminator->getOperands());
    rewriter.eraseOp(oldTerminator);

    Operation *IP = newOp.getBody()->getTerminator();
    while (IP != &IP->getBlock()->front()) {
      if (isa<BarrierOp>(IP->getPrevNode())) {
        IP = IP->getPrevNode();
      }
      break;
    }
    rewriter.setInsertionPoint(IP);
    for (auto en : llvm::enumerate(oldOps)) {
      if (!en.value().getDefiningOp<LLVM::UndefOp>())
        rewriter.create<memref::StoreOp>(op.getLoc(), en.value(),
                                         allocated[en.index()], ValueRange());
    }

    rewriter.setInsertionPointAfter(op);
    SmallVector<Value> loaded;
    for (Value alloc : allocated) {
      if (UseMinCut)
        loaded.push_back(
            rewriter
                .create<polygeist::CacheLoad>(op.getLoc(), alloc, ValueRange())
                ->getResult(0));
      else
        loaded.push_back(
            rewriter.create<memref::LoadOp>(op.getLoc(), alloc, ValueRange())
                ->getResult(0));
    }
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

bool isEquivalent(Value a, Value b) {
  if (a == b)
    return true;
  if (a.getType() != b.getType())
    return false;
  if (auto sa = a.getDefiningOp<polygeist::SubIndexOp>()) {
    if (auto sb = b.getDefiningOp<polygeist::SubIndexOp>()) {
      return isEquivalent(sa.getSource(), sb.getSource()) &&
             isEquivalent(sa.getIndex(), sb.getIndex());
    }
  }
  return false;
}

template <typename T, bool UseMinCut>
struct Reg2MemIf : public OpRewritePattern<T> {
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    SmallVector<BlockArgument> args;
    if (!op.getResults().size() || !hasNestedBarrier(op, args))
      return failure();

    if (!isa<scf::ParallelOp, AffineParallelOp>(op->getParentOp())) {
      return failure();
    }

    SmallPtrSet<Operation *, 1> equivThenStores;
    SmallPtrSet<Operation *, 1> equivElseStores;
    SmallVector<Value> allocated;
    allocated.reserve(op.getNumResults());
    Operation *thenYield = &getThenBlock(op)->back();
    Operation *elseYield = &getElseBlock(op)->back();
    for (auto tup : llvm::zip(op.getResults(), thenYield->getOperands(),
                              elseYield->getOperands())) {
      Value res = std::get<0>(tup);
      Type opType = res.getType();
      bool usesMustStore = false;
      for (auto user : res.getUsers()) {
        if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
          if (storeOp.getMemref() == res) {
            usesMustStore = true;
            break;
          }
          if (storeOp->getBlock() != op->getBlock()) {
            usesMustStore = true;
            break;
          }
          for (auto nex = op->getNextNode(); nex != storeOp;
               nex = nex->getNextNode()) {
            if (!mayReadFrom(nex, storeOp.getMemref()))
              continue;
            usesMustStore = true;
            break;
          }
          // TODO check that memref operands are recomputable at this location
          SmallVector<Value> todo = {storeOp.getMemref()};
          for (auto ind : storeOp.getIndices())
            todo.push_back(ind);
          while (!todo.empty()) {
            auto cur = todo.pop_back_val();

            if (auto BA = cur.dyn_cast<BlockArgument>())
              if (BA.getOwner() == op->getBlock())
                continue;

            if (cur.getParentRegion()->isProperAncestor(
                    op->getParentRegion())) {
              continue;
            }
            // If a value which
            if (auto op = cur.getDefiningOp()) {
              if (isReadNone(op)) {
                for (auto arg : op->getOperands()) {
                  todo.push_back(arg);
                }
                continue;
              }
            }

            usesMustStore = true;
            break;
          }
          if (!usesMustStore) {
            Value val = std::get<1>(tup);
            if (auto cl = val.getDefiningOp<polygeist::CacheLoad>()) {
              if (isEquivalent(cl.getMemref(), storeOp.getMemref()) &&
                  cl->getBlock() == storeOp->getBlock() &&
                  llvm::all_of(llvm::zip(cl.getIndices(), storeOp.getIndices()),
                               [](std::tuple<Value, Value> t) {
                                 return isEquivalent(std::get<0>(t),
                                                     std::get<1>(t));
                               })) {
                bool same = true;
                for (Operation *op = cl->getNextNode(); op != storeOp;
                     op = op->getNextNode()) {
                  if (mayWriteTo(op, cl.getMemref(), /*ignoreBarrier*/ true)) {
                    same = false;
                    break;
                  }
                }
                if (same)
                  equivThenStores.insert(storeOp);
              }
            }
            val = std::get<2>(tup);
            if (auto cl = val.getDefiningOp<polygeist::CacheLoad>()) {
              if (isEquivalent(cl.getMemref(), storeOp.getMemref()) &&
                  cl->getBlock() == storeOp->getBlock() &&
                  llvm::all_of(llvm::zip(cl.getIndices(), storeOp.getIndices()),
                               [](std::tuple<Value, Value> t) {
                                 return isEquivalent(std::get<0>(t),
                                                     std::get<1>(t));
                               })) {
                bool same = true;
                for (Operation *op = cl->getNextNode(); op != storeOp;
                     op = op->getNextNode()) {
                  if (mayWriteTo(op, cl.getMemref(), /*ignoreBarrier*/ true)) {
                    same = false;
                    break;
                  }
                }
                if (same)
                  equivElseStores.insert(storeOp);
              }
            }
          }
          continue;
        }
        usesMustStore = true;
      }
      Value alloc = nullptr;
      if (usesMustStore) {
        alloc = rewriter.create<memref::AllocaOp>(
            op.getLoc(), MemRefType::get(ArrayRef<int64_t>(), opType),
            ValueRange());
      }
      allocated.push_back(alloc);
    }

    auto newOp = cloneWithoutResults(op, rewriter);

    rewriter.setInsertionPoint(thenYield);
    for (auto pair :
         llvm::zip(thenYield->getOperands(), allocated, op.getResults())) {
      Value val = std::get<0>(pair);
      auto alloc = std::get<1>(pair);
      auto res = std::get<2>(pair);
      if (!alloc) {
        for (auto user : llvm::make_early_inc_range(res.getUsers())) {
          auto storeOp = dyn_cast<memref::StoreOp>(user);
          assert(storeOp);
          if (equivThenStores.count(storeOp))
            continue;
          BlockAndValueMapping map;
          SetVector<Operation *> seen;
          SmallVector<Value> todo = {storeOp.getMemref()};
          for (auto ind : storeOp.getIndices())
            todo.push_back(ind);
          while (!todo.empty()) {
            auto cur = todo.pop_back_val();

            if (auto BA = cur.dyn_cast<BlockArgument>())
              if (BA.getOwner() == op->getBlock())
                continue;
            if (cur.getParentRegion()->isProperAncestor(
                    op->getParentRegion())) {
              continue;
            }

            // If a value which
            auto op = cur.getDefiningOp();
            assert(op);
            seen.insert(op);
            for (auto arg : op->getOperands()) {
              todo.push_back(arg);
            }
          }
          for (auto op : llvm::reverse(seen)) {
            rewriter.clone(*op, map);
          }
          SmallVector<Value> inds;
          for (auto ind : storeOp.getIndices())
            inds.push_back(map.lookupOrDefault(ind));
          // Only erase during the else, since we need that there
          rewriter.create<memref::StoreOp>(
              storeOp.getLoc(), val, map.lookupOrDefault(storeOp.getMemref()),
              inds);
        }
      } else if (!val.getDefiningOp<LLVM::UndefOp>()) {
        rewriter.create<memref::StoreOp>(op.getLoc(), val, alloc, ValueRange());
      }
    }
    rewriter.setInsertionPoint(thenYield);
    if (isa<AffineIfOp>(op))
      rewriter.replaceOpWithNewOp<AffineYieldOp>(thenYield);
    else
      rewriter.replaceOpWithNewOp<scf::YieldOp>(thenYield);

    rewriter.setInsertionPoint(elseYield);
    for (auto pair :
         llvm::zip(elseYield->getOperands(), allocated, op.getResults())) {
      Value val = std::get<0>(pair);
      auto alloc = std::get<1>(pair);
      auto res = std::get<2>(pair);
      if (!alloc) {
        for (auto user : llvm::make_early_inc_range(res.getUsers())) {
          auto storeOp = dyn_cast<memref::StoreOp>(user);
          assert(storeOp);
          if (equivElseStores.count(storeOp)) {
            rewriter.eraseOp(storeOp);
            continue;
          }
          BlockAndValueMapping map;
          SetVector<Operation *> seen;
          SmallVector<Value> todo = {storeOp.getMemref()};
          for (auto ind : storeOp.getIndices())
            todo.push_back(ind);
          while (!todo.empty()) {
            auto cur = todo.pop_back_val();

            if (auto BA = cur.dyn_cast<BlockArgument>())
              if (BA.getOwner() == op->getBlock())
                continue;

            if (cur.getParentRegion()->isProperAncestor(
                    op->getParentRegion())) {
              continue;
            }

            // If a value which
            auto op = cur.getDefiningOp();
            assert(op);
            seen.insert(op);
            for (auto arg : op->getOperands()) {
              todo.push_back(arg);
            }
          }
          for (auto op : llvm::reverse(seen)) {
            rewriter.clone(*op, map);
          }
          SmallVector<Value> inds;
          for (auto ind : storeOp.getIndices())
            inds.push_back(map.lookupOrDefault(ind));

          rewriter.replaceOpWithNewOp<memref::StoreOp>(
              storeOp, val, map.lookupOrDefault(storeOp.getMemref()), inds);
        }
      } else if (!val.getDefiningOp<LLVM::UndefOp>()) {
        rewriter.create<memref::StoreOp>(op.getLoc(), val, alloc, ValueRange());
      }
    }
    rewriter.setInsertionPoint(elseYield);
    if (isa<AffineIfOp>(op))
      rewriter.replaceOpWithNewOp<AffineYieldOp>(elseYield);
    else
      rewriter.replaceOpWithNewOp<scf::YieldOp>(elseYield);

    rewriter.eraseOp(&getThenBlock(newOp)->back());
    rewriter.mergeBlocks(getThenBlock(op), getThenBlock(newOp));

    rewriter.eraseOp(&getElseBlock(newOp)->back());
    rewriter.mergeBlocks(getElseBlock(op), getElseBlock(newOp));

    rewriter.startRootUpdate(op);
    rewriter.setInsertionPoint(op);
    for (auto pair : llvm::zip(op->getResults(), allocated)) {
      auto alloc = std::get<1>(pair);
      if (alloc) {
        // TODO may want to move this far into the future.
        if (UseMinCut)
          std::get<0>(pair).replaceAllUsesWith(
              rewriter
                  .create<polygeist::CacheLoad>(op.getLoc(), alloc,
                                                ValueRange())
                  ->getResult(0));
        else
          std::get<0>(pair).replaceAllUsesWith(
              rewriter.create<memref::LoadOp>(op.getLoc(), alloc, ValueRange())
                  ->getResult(0));
      }
    }
    rewriter.finalizeRootUpdate(op);
    rewriter.eraseOp(op);
    return success();
  }
};

static void storeValues(Location loc, ValueRange values, ValueRange pointers,
                        PatternRewriter &rewriter) {
  for (auto pair : llvm::zip(values, pointers)) {
    if (!std::get<0>(pair).getDefiningOp<LLVM::UndefOp>())
      rewriter.create<memref::StoreOp>(loc, std::get<0>(pair),
                                       std::get<1>(pair), ValueRange());
  }
}

static void allocaValues(Location loc, ValueRange values,
                         PatternRewriter &rewriter,
                         SmallVector<Value> &allocated) {
  allocated.reserve(values.size());
  for (Value value : values) {
    Value alloc = rewriter.create<memref::AllocaOp>(
        loc, MemRefType::get(ArrayRef<int64_t>(), value.getType()),
        ValueRange());
    allocated.push_back(alloc);
  }
}

struct Reg2MemWhile : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern<scf::WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() == 0 && op.getNumResults() == 0)
      return failure();
    SmallVector<BlockArgument> args;
    if (!hasNestedBarrier(op, args)) {
      return failure();
    }

    // Value stackPtr = rewriter.create<LLVM::StackSaveOp>(
    //     op.getLoc(), LLVM::LLVMPointerType::get(rewriter.getIntegerType(8)));
    SmallVector<Value> beforeAllocated, afterAllocated;
    allocaValues(op.getLoc(), op.getOperands(), rewriter, beforeAllocated);
    storeValues(op.getLoc(), op.getOperands(), beforeAllocated, rewriter);
    allocaValues(op.getLoc(), op.getResults(), rewriter, afterAllocated);

    auto newOp =
        rewriter.create<scf::WhileOp>(op.getLoc(), TypeRange(), ValueRange());
    Block *newBefore =
        rewriter.createBlock(&newOp.getBefore(), newOp.getBefore().begin());
    SmallVector<Value> newBeforeArguments;
    loadValues(op.getLoc(), beforeAllocated, rewriter, newBeforeArguments);
    rewriter.mergeBlocks(&op.getBefore().front(), newBefore,
                         newBeforeArguments);

    auto beforeTerminator =
        cast<scf::ConditionOp>(newOp.getBefore().front().getTerminator());
    rewriter.setInsertionPoint(beforeTerminator);
    storeValues(op.getLoc(), beforeTerminator.getArgs(), afterAllocated,
                rewriter);

    rewriter.updateRootInPlace(
        beforeTerminator, [&] { beforeTerminator.getArgsMutable().clear(); });

    Block *newAfter =
        rewriter.createBlock(&newOp.getAfter(), newOp.getAfter().begin());
    SmallVector<Value> newAfterArguments;
    loadValues(op.getLoc(), afterAllocated, rewriter, newAfterArguments);
    rewriter.mergeBlocks(&op.getAfter().front(), newAfter, newAfterArguments);

    auto afterTerminator =
        cast<scf::YieldOp>(newOp.getAfter().front().getTerminator());
    rewriter.setInsertionPoint(afterTerminator);
    storeValues(op.getLoc(), afterTerminator.getResults(), beforeAllocated,
                rewriter);

    rewriter.updateRootInPlace(
        afterTerminator, [&] { afterTerminator.getResultsMutable().clear(); });

    rewriter.setInsertionPointAfter(op);
    SmallVector<Value> results;
    loadValues(op.getLoc(), afterAllocated, rewriter, results);
    // rewriter.create<LLVM::StackRestoreOp>(op.getLoc(), stackPtr);
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct LowerCacheLoad : public OpRewritePattern<polygeist::CacheLoad> {
  using OpRewritePattern<polygeist::CacheLoad>::OpRewritePattern;

  LogicalResult matchAndRewrite(polygeist::CacheLoad op,
                                PatternRewriter &rewriter) const override {
    auto memrefLoad = rewriter.create<memref::LoadOp>(
        op.getLoc(), op.getMemref(), op.getIndices());
    rewriter.replaceOp(op, memrefLoad.getResult());
    return success();
  }
};

struct CPUifyPass : public SCFCPUifyBase<CPUifyPass> {
  CPUifyPass() = default;
  CPUifyPass(StringRef method) { this->method.setValue(method.str()); }
  void runOnOperation() override {
    StringRef method(this->method);
    if (method.startswith("distribute")) {
      {
        RewritePatternSet patterns(&getContext());
        patterns.insert<BarrierElim</*TopLevelOnly*/ false>, Reg2MemWhile>(
            &getContext());

        if (method.contains("mincut")) {
          patterns.insert<
              Reg2MemFor<scf::ForOp, true>, Reg2MemFor<AffineForOp, true>,
              Reg2MemIf<scf::IfOp, true>, Reg2MemIf<AffineIfOp, true>,
              WrapForWithBarrier<true>, WrapAffineForWithBarrier<true>,
              WrapIfWithBarrier<scf::IfOp, true>,
              WrapIfWithBarrier<AffineIfOp, true>, WrapWhileWithBarrier<true>>(
              &getContext());
        } else {
          patterns.insert<
              Reg2MemFor<scf::ForOp, false>, Reg2MemFor<AffineForOp, false>,
              Reg2MemIf<scf::IfOp, false>, Reg2MemIf<AffineIfOp, false>,
              WrapForWithBarrier<false>, WrapAffineForWithBarrier<false>,
              WrapIfWithBarrier<scf::IfOp, false>,
              WrapIfWithBarrier<AffineIfOp, false>,
              WrapWhileWithBarrier<false>>(&getContext());
        }

        patterns.insert<InterchangeForIfPFor<scf::ParallelOp, scf::ForOp>,
                        InterchangeForIfPFor<AffineParallelOp, scf::ForOp>,
                        InterchangeForIfPFor<scf::ParallelOp, scf::IfOp>,
                        InterchangeForIfPFor<AffineParallelOp, scf::IfOp>,
                        InterchangeForIfPFor<scf::ParallelOp, AffineForOp>,
                        InterchangeForIfPFor<AffineParallelOp, AffineForOp>,
                        InterchangeForIfPFor<scf::ParallelOp, AffineIfOp>,
                        InterchangeForIfPFor<AffineParallelOp, AffineIfOp>,

                        InterchangeWhilePFor<scf::ParallelOp>,
                        InterchangeWhilePFor<AffineParallelOp>,
                        // NormalizeLoop,
                        NormalizeParallel
                        // RotateWhile,
                        >(&getContext());
        if (method.contains("mincut")) {
          patterns.insert<DistributeAroundBarrier<scf::ParallelOp, true>,
                          DistributeAroundBarrier<AffineParallelOp, true>>(
              &getContext());
        } else {
          patterns.insert<DistributeAroundBarrier<scf::ParallelOp, false>,
                          DistributeAroundBarrier<AffineParallelOp, false>>(
              &getContext());
        }
        GreedyRewriteConfig config;
        config.maxIterations = 142;
        if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(patterns), config))) {
          signalPassFailure();
          return;
        }
      }
      {
        RewritePatternSet patterns(&getContext());
        GreedyRewriteConfig config;
        patterns.insert<LowerCacheLoad>(&getContext());
        if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(patterns), config))) {
          signalPassFailure();
          return;
        }
      }
    } else if (method == "omp") {
      SmallVector<polygeist::BarrierOp> toReplace;
      getOperation()->walk(
          [&](polygeist::BarrierOp b) { toReplace.push_back(b); });
      for (auto b : toReplace) {
        OpBuilder Builder(b);
        Builder.create<omp::BarrierOp>(b.getLoc());
        b->erase();
      }
    } else {
      llvm::errs() << "unknown cpuify type: " << method << "\n";
      llvm_unreachable("unknown cpuify type");
    }
  }
};

} // end namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createCPUifyPass(StringRef str) {
  return std::make_unique<CPUifyPass>(str);
}
} // namespace polygeist
} // namespace mlir
