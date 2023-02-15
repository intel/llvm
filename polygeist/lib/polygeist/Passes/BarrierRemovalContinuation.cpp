//===- BarrierRemovalContinuation.cpp - Remove barriers in parallel loops -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the barrier removal using continuation-style approach
// following Karrenberg and Hack "Improving Performance of OpenCL on CPUs".
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "polygeist/BarrierUtils.h"
#include "polygeist/Passes/Passes.h"

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;

/// Returns true if the given parallel op has a nested barrier op that is not
/// nested in some other parallel op.
static bool hasImmediateBarriers(scf::ParallelOp op) {
  WalkResult result = op.walk([op](polygeist::BarrierOp barrier) {
    if (barrier->getParentOfType<scf::ParallelOp>() == op)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

/// Wrap the bodies of all parallel ops with immediate barriers, i.e. the
/// parallel ops that will persist after the partial loop-to-cfg conversion,
/// into an execute region op.
static void wrapPersistingLoopBodies(FunctionOpInterface function) {
  SmallVector<scf::ParallelOp> loops;
  function.walk([&](scf::ParallelOp op) {
    if (hasImmediateBarriers(op))
      loops.push_back(op);
  });

  for (scf::ParallelOp op : loops) {
    OpBuilder builder = OpBuilder::atBlockBegin(op.getBody());
    auto wrapper = builder.create<scf::ExecuteRegionOp>(
        op.getLoc(), op.getResults().getTypes());
    builder.createBlock(&wrapper.getRegion(), wrapper.getRegion().begin());
    wrapper.getRegion().front().getOperations().splice(
        wrapper.getRegion().front().begin(), op.getBody()->getOperations(),
        std::next(op.getBody()->begin()), op.getBody()->end());
    builder.setInsertionPointToEnd(op.getBody());
    builder.create<scf::YieldOp>(
        wrapper.getRegion().front().getTerminator()->getLoc(),
        wrapper.getResults());
  }
}

/// Convert SCF constructs except parallel ops with immediate barriers to a CFG.
static LogicalResult applyCFGConversion(FunctionOpInterface function) {
  RewritePatternSet patterns(function.getContext());
  populateSCFToControlFlowConversionPatterns(patterns);

  // Configure the target to preserve parallel ops with barriers, unless those
  // barriers are nested in deeper parallel ops.
  ConversionTarget target(*function.getContext());
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addIllegalOp<scf::ForOp, scf::IfOp, scf::WhileOp>();
  target.addLegalOp<scf::ExecuteRegionOp, func::FuncOp, ModuleOp>();
  target.addDynamicallyLegalOp<scf::ParallelOp>(
      [](scf::ParallelOp op) { return hasImmediateBarriers(op); });

  return applyPartialConversion(function, target, std::move(patterns));
}

/// Convert SCF constructs except parallel loops with immediate barriers to a
/// CFG after wrapping the bodies of such loops in an execute_region op so as to
/// comply with the single-block requirement of the body.
static LogicalResult convertToCFG(FunctionOpInterface function) {
  wrapPersistingLoopBodies(function);
  return applyCFGConversion(function);
}

/// Split any block in the given region into separate blocks immediately after
/// the barrier operation. As a result, any barrier operation always precedes an
/// unconditional branch operation.
static void splitBlocksWithBarrier(Region &region) {
  auto barriers = llvm::to_vector<4>(region.getOps<polygeist::BarrierOp>());
  for (polygeist::BarrierOp op : barriers) {
    Block *original = op->getBlock();
    Block *block = original->splitBlock(op->getNextNode());
    auto builder = OpBuilder::atBlockEnd(original);
    builder.create<cf::BranchOp>(builder.getUnknownLoc(), block);
  }
}

/// Split blocks with barriers into parts in the parallel ops of the given
/// function.
static LogicalResult splitBlocksWithBarrier(FunctionOpInterface function) {
  WalkResult result = function.walk([](scf::ParallelOp op) -> WalkResult {
    if (!hasImmediateBarriers(op))
      return success();

    if (op->walk([op](scf::ParallelOp nested) {
            return op == nested ? WalkResult::advance()
                                : WalkResult::interrupt();
          }).wasInterrupted()) {
      return op->emitError() << "nested parallel ops with barriers not "
                                "supported (consider outlining)";
    }

    splitBlocksWithBarrier(
        cast<scf::ExecuteRegionOp>(&op.getBody()->front()).getRegion());
    return success();
  });
  return success(!result.wasInterrupted());
}

/// Traverse the CFG starting from the given block collecting the traversed
/// blocks until a block with no successors or a block with a barrier operation
/// is reached.
static void traverseUntilBarrier(Block *start,
                                 llvm::SetVector<Block *> &traversed) {
  SmallVector<Block *, 16> stack;
  stack.push_back(start);

  // Perform a DFS with explicit stack.
  while (!stack.empty()) {
    Block *current = stack.pop_back_val();
    traversed.insert(current);
    if (isa_and_nonnull<polygeist::BarrierOp>(
            current->getTerminator()->getPrevNode()))
      continue;

    // Consider successors in reverse order so that we add the first successor
    // last to the stack and visit it first later.
    for (Block *block : llvm::reverse(current->getSuccessors())) {
      if (traversed.contains(block))
        continue;
      stack.push_back(block);
    }
  }
}

/// Emit the IR storing `id` into `storage` if values in `ivs` are equal to
/// those in `lowerBounds`, e.g. on the first iteration of some loop nest.
static void emitStoreContinuationID(Location loc, int id, ValueRange ivs,
                                    ValueRange lowerBounds, Value storage,
                                    OpBuilder &builder) {
  assert(!ivs.empty());
  assert(ivs.size() == lowerBounds.size());

  SmallVector<Value> comparisons;
  comparisons.resize(ivs.size());
  for (unsigned i = 0, e = ivs.size(); i < e; ++i) {
    comparisons[i] =
        builder.create<CmpIOp>(loc, CmpIPredicate::eq, ivs[i], lowerBounds[i]);
  }

  Value condition = comparisons[0];
  for (unsigned i = 1, e = ivs.size(); i < e; ++i) {
    condition = builder.create<AndIOp>(loc, condition.getType(), condition,
                                       comparisons[i]);
  }

  auto thenBuilder = [&](OpBuilder &nested, Location loc) {
    Value idValue = nested.create<ConstantIndexOp>(loc, id);
    nested.create<memref::StoreOp>(loc, idValue, storage);
    nested.create<scf::YieldOp>(loc);
  };

  builder.create<scf::IfOp>(loc, condition, thenBuilder);
}

/// Clone `blocks` into the given `region` as part of continuation-style barrier
/// implementation. The id of the next continuation will be stored in `storage`
/// when `ivs` are equal to `lowerBounds`. The list of entry blocks and the
/// mapping of the surrounding parallel loop induction variables must be
/// provided as arguments.
static void
replicateIntoRegion(Region &region, Value storage, ValueRange ivs,
                    ValueRange lowerBounds,
                    const llvm::SetVector<Block *> &blocks,
                    const llvm::SetVector<Block *> &subgraphEntryPoints,
                    const BlockAndValueMapping &ivMapping, OpBuilder &builder) {
  BlockAndValueMapping mapping(ivMapping);

  // Create a separate entry block because the subset of blocks might have
  // branches to its first block, which would not be possible for the region
  // entry block.
  Block *entryBlock = builder.createBlock(&region, region.end());

  // Clone blocks one by one and keep track of them and their arguments in the
  // mapping. Drop the arguments of the original entry block, if contained in
  // the set, these correspond to the loop induction variables and are already
  // remapped in the mapping.
  for (Block *block : blocks) {
    auto argLocations = llvm::to_vector<4>(llvm::map_range(
        block->getArguments(), [](BlockArgument arg) { return arg.getLoc(); }));
    Block *copy;
    if (block == blocks.front()) {
      assert(block->getNumArguments() == 0 || block->isEntryBlock());
      copy = builder.createBlock(&region, region.end());
    } else {
      copy = builder.createBlock(&region, region.end(),
                                 block->getArgumentTypes(), argLocations);
      mapping.map(block->getArguments(), copy->getArguments());
    }
    mapping.map(block, copy);
  }

  // Branch from the entry block to the first cloned block.
  builder.setInsertionPointToEnd(entryBlock);
  builder.create<cf::BranchOp>(builder.getUnknownLoc(),
                               mapping.lookup(blocks.front()));

  // Now that the block structure is created, clone the operations and introduce
  // the flow between continuations.
  for (Block *block : blocks) {
    builder.setInsertionPointToEnd(mapping.lookup(block));

    for (Operation &op : *block) {
      // Barriers can be omitted, the surrounding parallel operation has an
      // implicit brarrier at the end.
      if (isa<polygeist::BarrierOp>(op))
        continue;

      // There are two cases for branches. (1) Branches to block within the
      // currently processed subgraph can be kept as is. (2) Branches to other
      // blocks are assumed to branch to the entry block of another subgraph.
      // They are replaced with storing the correspnding continuation ID and a
      // yield.
      if (auto branch = dyn_cast<cf::BranchOp>(&op)) {
        // if (!blocks.contains(branch.dest())) {
        if (isa_and_nonnull<polygeist::BarrierOp>(branch->getPrevNode())) {
          auto it = llvm::find(subgraphEntryPoints, branch.getDest());
          assert(it != subgraphEntryPoints.end());
          emitStoreContinuationID(
              branch.getLoc(), std::distance(subgraphEntryPoints.begin(), it),
              ivs, lowerBounds, storage, builder);
          builder.create<scf::YieldOp>(branch.getLoc());
          continue;
        }
      }

      // Yields out of the original loop must store a continuation id that does
      // not match any block. Note the yield itself must still exist.
      if (isa<scf::YieldOp>(op)) {
        emitStoreContinuationID(op.getLoc(), -1, ivs, lowerBounds, storage,
                                builder);
      }

      builder.clone(op, mapping);
    }
  }
}

/// Emit the continuation containing the given set of blocks. Barriers are
/// replaced with writing the index of the next continuation to `storage` and
/// exiting. `parallel` indicates the loop that is being split into
/// continuations and `subgraphEntryPoints` contains the entry blocks of all
/// continuations. The index of a continuation correspond to the position of its
/// entry block in `subgraphEntryPoints`.
static void
emitContinuationCase(Value condition, Value storage, scf::ParallelOp parallel,
                     const llvm::SetVector<Block *> &subgraphEntryPoints,
                     const llvm::SetVector<Block *> &blocks,
                     OpBuilder &builder) {
  ImplicitLocOpBuilder b(parallel.getLoc(), builder);

  auto parallelBuilder = [&](OpBuilder &nested, Location loc, ValueRange ivs) {
    ImplicitLocOpBuilder bn(loc, nested);
    auto executeRegion =
        bn.create<scf::ExecuteRegionOp>(TypeRange(), ValueRange());
    BlockAndValueMapping mapping;
    mapping.map(parallel.getInductionVars(), ivs);
    replicateIntoRegion(executeRegion.getRegion(), storage, ivs,
                        parallel.getLowerBound(), blocks, subgraphEntryPoints,
                        mapping, builder);
  };

  auto thenBuilder = [&](OpBuilder &nested, Location loc) {
    ImplicitLocOpBuilder bn(loc, nested);
    bn.create<scf::ParallelOp>(parallel.getLowerBound(),
                               parallel.getUpperBound(), parallel.getStep(),
                               parallelBuilder);
    bn.create<scf::YieldOp>();
  };

  b.create<scf::IfOp>(condition, thenBuilder);
  builder.setInsertionPoint(b.getInsertionBlock(), b.getInsertionPoint());
}

/// Returns the insertion point (as block pointer and itertor in it) immediately
/// after the definition of `v`.
static std::pair<Block *, Block::iterator> getInsertionPointAfterDef(Value v) {
  if (Operation *op = v.getDefiningOp())
    return {op->getBlock(), std::next(Block::iterator(op))};

  BlockArgument blockArg = v.cast<BlockArgument>();
  return {blockArg.getParentBlock(), blockArg.getParentBlock()->begin()};
}

/// Returns the insertion point that post-dominates `first` and `second`.
static std::pair<Block *, Block::iterator>
findNearestPostDominatingInsertionPoint(
    const std::pair<Block *, Block::iterator> &first,
    const std::pair<Block *, Block::iterator> &second,
    const PostDominanceInfo &postDominanceInfo) {
  // Same block, take the last op.
  if (first.first == second.first)
    return first.second->isBeforeInBlock(&*second.second) ? second : first;

  // Same region, use "normal" dominance analysis.
  if (first.first->getParent() == second.first->getParent()) {
    Block *block =
        postDominanceInfo.findNearestCommonDominator(first.first, second.first);
    assert(block);
    if (block == first.first)
      return first;
    if (block == second.first)
      return second;
    return {block, block->begin()};
  }

  if (first.first->getParent()->isAncestor(second.first->getParent()))
    return second;

  assert(second.first->getParent()->isAncestor(first.first->getParent()) &&
         "expected values to be defined in nested regions");
  return first;
}

/// Returns the insertion point that post-dominates all `values`.
static std::pair<Block *, Block::iterator>
findNesrestPostDominatingInsertionPoint(
    ArrayRef<Value> values, const PostDominanceInfo &postDominanceInfo) {
  assert(!values.empty());
  std::pair<Block *, Block::iterator> insertPoint =
      getInsertionPointAfterDef(values[0]);
  for (unsigned i = 1, e = values.size(); i < e; ++i)
    insertPoint = findNearestPostDominatingInsertionPoint(
        insertPoint, getInsertionPointAfterDef(values[i]), postDominanceInfo);
  return insertPoint;
}

std::pair<Block *, Block::iterator>
findInsertionPointAfterLoopOperands(scf::ParallelOp op) {
  // Find the earliest insertion point where loop bounds are fully defined.
  PostDominanceInfo postDominanceInfo(op->getParentOfType<func::FuncOp>());
  SmallVector<Value> operands;
  llvm::append_range(operands, op.getLowerBound());
  llvm::append_range(operands, op.getUpperBound());
  llvm::append_range(operands, op.getStep());
  return findNesrestPostDominatingInsertionPoint(operands, postDominanceInfo);
}

/// Break SSA use-def pairs that would need to communicate between different
/// subgraphs by storing the value in a scratchpad storage when available and
/// loading it back before every use. Each scratchpad storage has as many
/// elements as the surrounding `parallel` loop has iterations, with each
/// iteration writing a different element.
static void reg2mem(ArrayRef<llvm::SetVector<Block *>> subgraphs,
                    scf::ParallelOp parallel, OpBuilder &allocaBuilder,
                    OpBuilder &freeBuilder) {
  // Check if a block exists in another subgraph than the given subgraph (there
  // may be duplicates).
  auto otherSubgraphContains = [&](const llvm::SetVector<Block *> &subgraph,
                                   Block *block) {
    for (const llvm::SetVector<Block *> &other : subgraphs) {
      if (&other == &subgraph)
        continue;
      if (other.contains(block))
        return true;
    }
    return false;
  };

  // Find all values that are defined in one subgraph and used in another
  // subgraph. These will have to be communicated through memory.
  SmallVector<Value> valuesToStore;
  auto checkValues = [&](ValueRange values,
                         const llvm::SetVector<Block *> &subgraph) {
    for (Value value : values) {
      for (Operation *user : value.getUsers()) {
        if (otherSubgraphContains(subgraph, user->getBlock())) {
          valuesToStore.push_back(value);
          return;
        }
      }
    }
  };

  for (const llvm::SetVector<Block *> &subgraph : subgraphs) {
    for (Block *block : subgraph) {
      checkValues(block->getArguments(), subgraph);
      for (Operation &op : *block) {
        checkValues(op.getResults(), subgraph);
      }
    }
  }

  // Insert allocations as early as possible, the stores immediately when the
  // value is available and the loads immediately before each use. Further
  // mem2reg is expected to clean up the cases where a value is stored and
  // loaded back in the same block or subsequent blocks because there is no
  // guarantee that the block was not copied in another subgraph.

  OpBuilder accessBuilder(parallel.getContext());
  SmallVector<Value> iterationCounts =
      emitIterationCounts(allocaBuilder, parallel);
  for (Value value : valuesToStore) {
    assert(!value.getDefiningOp<polygeist::SubIndexOp>());
    Value allocation = allocateTemporaryBuffer<mlir::memref::AllocOp>(
        allocaBuilder, value, iterationCounts, /*alloca*/ true);
    /*
    if
    (allocation.getType().cast<MemRefType>().getElementType().isa<MemRefType>())
    { llvm::errs() << " value: " << value << " alloc: " << allocation << "\n";
        llvm_unreachable("bad allocation\n");
    }
    */
    freeBuilder.create<memref::DeallocOp>(allocation.getLoc(), allocation);
    accessBuilder.setInsertionPointAfterValue(value);
    Operation *store = nullptr;
    if (!value.getDefiningOp<memref::AllocaOp>())
      store = accessBuilder.create<memref::StoreOp>(
          value.getLoc(), value, allocation, parallel.getInductionVars());
    llvm::SmallDenseMap<Operation *, Value, 4> reloadedCache;
    for (OpOperand &use : llvm::make_early_inc_range(value.getUses())) {
      if (use.getOwner() == store)
        continue;

      if (!value.getDefiningOp<memref::AllocaOp>()) {
        Value &reloaded = reloadedCache[use.getOwner()];
        if (!reloaded) {
          accessBuilder.setInsertionPoint(use.getOwner());
          reloaded = accessBuilder.create<memref::LoadOp>(
              value.getLoc(), allocation, parallel.getInductionVars());
        }
        use.set(reloaded);
      } else {
        accessBuilder.setInsertionPoint(use.getOwner());
        auto buf = allocation;
        for (auto idx : parallel.getInductionVars()) {
          auto mt0 = buf.getType().cast<MemRefType>();
          std::vector<int64_t> shape(mt0.getShape());
          shape.erase(shape.begin());
          auto mt = MemRefType::get(shape, mt0.getElementType(),
                                    mt0.getLayout(), mt0.getMemorySpace());
          auto subidx = accessBuilder.create<polygeist::SubIndexOp>(
              allocation.getLoc(), mt, buf, idx);
          buf = subidx;
        }
        use.set(buf);
      }
    }
  }
}

/// Implement the barriers in the given parallel loop using a continuation-based
/// approach. The body of the parallel loop is an execute_region containing a
/// CFG with barriers. Split the CFG into potentially overlapping subgraphs that
/// start at either the entry block or a block that immediately follows a block
/// with a barrier, splitting the blocks with a barrier if necessary. Each
/// subgraph becomes a separate continuation containing its own parallel loop,
/// which is used as a synchronization point instead of the explicit barrier.
/// Each continuation is identified by an integer, and the index of the next
/// continuation is stored in `storage` by the first operation of each parallel
/// loop (due to control flow convergence rules, all iterations must actually
/// have the same continuation). Continuations are executed in a loop until an
/// exit index (-1) is found. The overall IR resembles the following sketch:
///
///   %next = alloca index
///   %c0 = constant 0 : index
///   %c-1 = constant -1 : index
///   store %c0, %next[%c0]
///   scf.while {
///     %nextval = load %next[%c0]
///     %cond = cmpi ne, %nextval, %c-1
///     scf.conditional(%cond)
///   } do {
///     %nextval = load %next[%c0]
///     %case0 = cmpi eq, %nextval, %c0
///     %case1 = cmpi eq, %nextval, %c1
///     // ...
///     scf.if (%case0) {
///       scf.parallel %i = ... {
///         scf.execute_region {
///           // an extra block is introduced in case there are branches back to
///           // ^bb1
///           ^bb0:
///             br ^bb1
///           // only the original entry block will have block arguments that
///           // correspond to old loop IVs replace all uses of those with new
///           // IVs and drop them
///           ^bb1:
///             condbr .. ^bb2, ^bb3
///           ^bb2:
///             /// was scf.barrier followed by scf.yield
///             %should_store = cmpi eq, %i, lower-bound-of-%i
///             scf.if %should_store {
///               store id-of-next-block, %nextval[%c0]
///             }
///             scf.yield
///           ^bb3:
///             /// was just scf.yield (return)
///             should_store = cmpi eq, %i, lower-bound-of-%i
///             scf.if %should_store {
///               store -1, %nextval[%c0]
///             }
///             scf.yield
///         }
///         scf.yield
///       }
///     }
///     if (%case1) {
///     }
///     //...
///     scf.yield
///  }
static void createContinuations(scf::ParallelOp parallel, Value storage) {
  llvm::SetVector<Block *> startBlocks;
  auto outerExecuteRegion =
      cast<scf::ExecuteRegionOp>(&parallel.getBody()->front());
  startBlocks.insert(&outerExecuteRegion.getRegion().front());
  for (Block &block : outerExecuteRegion.getRegion()) {
    if (!isa_and_nonnull<polygeist::BarrierOp>(
            block.getTerminator()->getPrevNode()))
      continue;
    assert(block.getNumSuccessors() == 1 &&
           "expected one successor of a block with barrier after splitting");
    assert(llvm::hasSingleElement(block.getSuccessor(0)->getPredecessors()));
    startBlocks.insert(block.getSuccessor(0));
  }

  ImplicitLocOpBuilder builder(parallel.getLoc(), parallel);
  auto zero = builder.create<ConstantIndexOp>(0);
  auto negOne = builder.create<ConstantIndexOp>(-1);
  builder.create<memref::StoreOp>(zero, storage);
  auto loop = builder.create<scf::WhileOp>(TypeRange(), ValueRange());

  SmallVector<llvm::SetVector<Block *>> subgraphs;
  for (Block *block : startBlocks) {
    traverseUntilBarrier(block, subgraphs.emplace_back());
  }
  OpBuilder allocBuilder(loop);
  reg2mem(subgraphs, parallel, allocBuilder, builder);

  builder.createBlock(&loop.getBefore(), loop.getBefore().end());
  Value next = builder.create<memref::LoadOp>(storage);
  Value condition = builder.create<CmpIOp>(CmpIPredicate::ne, next, negOne);
  builder.create<scf::ConditionOp>(TypeRange(), condition, ValueRange());

  builder.createBlock(&loop.getAfter(), loop.getAfter().end());
  next = builder.create<memref::LoadOp>(storage);
  SmallVector<Value> caseConditions;
  caseConditions.resize(startBlocks.size());
  for (int i = 0, e = caseConditions.size(); i < e; ++i) {
    Value idValue = builder.create<ConstantIndexOp>(i);
    caseConditions[i] =
        builder.create<CmpIOp>(CmpIPredicate::eq, idValue, next);
  }

  for (auto en : llvm::enumerate(subgraphs)) {
    emitContinuationCase(caseConditions[en.index()], storage, parallel,
                         startBlocks, en.value(), builder);
  }
  builder.create<scf::YieldOp>();

  parallel.erase();
}

static void createContinuations(FunctionOpInterface func) {
  if (func->getNumRegions() == 0 || func.getFunctionBody().empty())
    return;

  OpBuilder allocaBuilder(&func.getFunctionBody().front(),
                          func.getFunctionBody().front().begin());
  func.walk([&](scf::ParallelOp parallel) {
    // Ignore parallel ops with no barriers.
    if (!hasImmediateBarriers(parallel))
      return;
    Value storage = allocaBuilder.create<memref::AllocaOp>(
        parallel.getLoc(), MemRefType::get({}, allocaBuilder.getIndexType()));
    createContinuations(parallel, storage);
  });
}

namespace {
struct BarrierRemoval
    : public SCFBarrierRemovalContinuationBase<BarrierRemoval> {
  void runOnOperation() override {
    auto f = getOperation();
    if (failed(convertToCFG(f)))
      return;
    if (failed(splitBlocksWithBarrier(f)))
      return;
    createContinuations(f);
  }
};
} // namespace

std::unique_ptr<Pass> polygeist::createBarrierRemovalContinuation() {
  return std::make_unique<BarrierRemoval>();
}
