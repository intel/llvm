//===- LoopInternalization.cpp - Promote memory access to local memory ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass tiles perfect loop nests to 'prefetch' memory accesses in shared
// local memory.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Polygeist/Analysis/MemoryAccessAnalysis.h"
#include "mlir/Dialect/Polygeist/Transforms/Passes.h"
#include "mlir/Dialect/Polygeist/Utils/TransformUtils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <numeric>

#define DEBUG_TYPE "loop-internalization"

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_LOOPINTERNALIZATION
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

using namespace mlir;
using namespace mlir::polygeist;

static llvm::cl::opt<bool>
    CollectReadOnlyAccesses(DEBUG_TYPE "-collect-read-only-accesses",
                            llvm::cl::init(true),
                            llvm::cl::desc("Promote only read-only accesses"));

static llvm::cl::list<unsigned> LoopInternalizationTileSizes(
    DEBUG_TYPE "-tile-sizes", llvm::cl::CommaSeparated,
    llvm::cl::desc("Tile sizes used in LoopInternalization"));

namespace {

//===----------------------------------------------------------------------===//
// Utilities functions
//===----------------------------------------------------------------------===//

/// A loop is a candidate when it is the outermost affine or scf for loop in a
/// perfect loop nest.
bool isCandidate(LoopLikeOpInterface loop) {
  if (!isa<affine::AffineForOp, scf::ForOp>(loop)) {
    LLVM_DEBUG(llvm::dbgs() << "not candidate: not affine or scf for loop\n");
    return false;
  }

  if (!LoopTools::isOutermostLoop(loop)) {
    LLVM_DEBUG(llvm::dbgs() << "not candidate: not outermost loop\n");
    return false;
  }

  if (!LoopTools::isPerfectLoopNest(loop)) {
    LLVM_DEBUG(llvm::dbgs() << "not candidate: not perfect loop nest\n");
    return false;
  }

  // TODO: check uniformity.

  return true;
}

template <typename T,
          typename = std::enable_if_t<llvm::is_one_of<
              T, affine::AffineForOp, scf::ForOp, LoopLikeOpInterface>::value>>
LogicalResult getTileSizes(const SmallVector<T> &nestedLoops,
                           SmallVectorImpl<Value> &tileSizes) {
  // TODO: calculate proper tile sizes.
  OpBuilder builder(nestedLoops.front());
  for (auto tileSize : LoopInternalizationTileSizes)
    tileSizes.push_back(builder.create<arith::ConstantIndexOp>(
        builder.getUnknownLoc(), tileSize));
  if (nestedLoops.size() != tileSizes.size()) {
    Value one =
        builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
    tileSizes.resize(nestedLoops.size(), one);
  }
  return success();
}

LogicalResult tile(MutableArrayRef<affine::AffineForOp> nestedLoops,
                   ArrayRef<Value> tileSizes,
                   SmallVectorImpl<affine::AffineForOp> &tiledNest) {
  SmallVector<affine::AffineForOp> newNestedLoops;
  unsigned numLoops = nestedLoops.size();
  LogicalResult res =
      tilePerfectlyNestedParametric(nestedLoops, tileSizes, &newNestedLoops);
  tiledNest = SmallVector<affine::AffineForOp>(
      newNestedLoops.begin() + numLoops, newNestedLoops.end());
  return res;
}
LogicalResult tile(ArrayRef<scf::ForOp> nestedLoops, ArrayRef<Value> tileSizes,
                   SmallVectorImpl<scf::ForOp> &tiledNest) {
  tiledNest = tile(nestedLoops, tileSizes, nestedLoops.back());
  return success();
}

void createLocalBarrier(OpBuilder builder) {
  // TODO: Use gpu.barrier, require GPUToSPIRV conversion in the pipeline.
  builder.create<spirv::ControlBarrierOp>(
      builder.getUnknownLoc(), spirv::Scope::Workgroup, spirv::Scope::Workgroup,
      spirv::MemorySemantics::SequentiallyConsistent |
          spirv::MemorySemantics::WorkgroupMemory);
}

//===----------------------------------------------------------------------===//
// MemorySelector
//===----------------------------------------------------------------------===//

/// Collect memory accesses in a loop and determine the memory space each access
/// should ideally use.
class MemorySelector {
public:
  MemorySelector(const MemoryAccessAnalysis &memAccessAnalysis,
                 DataFlowSolver &solver)
      : memAccessAnalysis(memAccessAnalysis), solver(solver) {}

  /// Enumerate memory spaces.
  enum class MemorySpace { Global, Shared, Constant, Texture };

  /// Returns the most suitable memory space the \p memref should use.
  std::optional<MemorySpace> selectMemorySpace(Value memref) const;

  /// Analyze the memory accesses in the given loop.
  void analyze(LoopLikeOpInterface loop);

private:
  /// Add the given \p access to the 'accesses' map.
  void addMemRefAccess(affine::MemRefAccess access);

  /// Retrieve all the memref accesses for \p access.
  //  const ArrayRef<affine::MemRefAccess *>
  //  getMemRefAccesses(affine::MemRefAccess *access) const;

  /// Return true iff no memref accesses in \p accesses are stores.
  bool areReadOnly(ArrayRef<affine::MemRefAccess> accesses) const;

  /// Return true iff all memref accesses in \p accesses are stores.
  bool areWriteOnly(ArrayRef<affine::MemRefAccess> accesses) const;

  /// Return true if memref accesses in \p accesses are a mix of loads and
  /// stores.
  bool areReadWrite(ArrayRef<affine::MemRefAccess> accesses) const;

  /// Determine whether the memref accesses in \p accesses exhibits temporal
  /// reuse.
  bool haveTemporalReuse(ArrayRef<affine::MemRefAccess> accesses,
                         const SmallVectorImpl<Value> &threadVars) const;

private:
  const MemoryAccessAnalysis &memAccessAnalysis;

  DataFlowSolver &solver;

  /// Collects all memory accesses for a given memref value.
  DenseMap<Value, SmallVector<affine::MemRefAccess>> accesses;

  /// The preferred memory space for each memref access;
  DenseMap<const affine::MemRefAccess *, MemorySpace> accessToMemSpace;
};

std::optional<MemorySelector::MemorySpace>
MemorySelector::selectMemorySpace(Value memref) const {
  assert(isa<MemRefType>(memref.getType()) && "Expecting a memref");

  auto it = accesses.find(memref);
  if (it == accesses.end())
    return std::nullopt;

  auto numShared = [this](ArrayRef<affine::MemRefAccess> accesses) {
    return llvm::count_if(accesses, [this](affine::MemRefAccess access) {
      auto it = accessToMemSpace.find(&access);
      if (it == accessToMemSpace.end())
        return false;
      return (it->second == MemorySpace::Shared);
    });
  };

  /// Recommend shared memory if at least half of the accesses for this memref
  /// should use shared memory.
  ArrayRef<affine::MemRefAccess> accesses = it->second;
  if (numShared(accesses) >= std::ceil((double)accesses.size() / 2))
    return MemorySpace::Shared;

  return MemorySpace::Global;
}

void MemorySelector::analyze(LoopLikeOpInterface loop) {
  assert(accesses.empty() && accessToMemSpace.empty() &&
         "Expecting empty maps");

  // Collect the global thread ids used in the function the loop is in.
  auto funcOp = loop->template getParentOfType<FunctionOpInterface>();
  SmallVector<Value> threadVars =
      memAccessAnalysis.computeThreadVector(funcOp, solver);

  // Collect candidate memref accesses in the loop.
  loop->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<affine::AffineLoadOp, affine::AffineStoreOp>(op))
      return;

    affine::MemRefAccess access(op);
    addMemRefAccess(access);
  });

  // Analyze the accesses collected and populate the 'accessToMemSpace' map.
  for (auto &entry : accesses) {
    ArrayRef<affine::MemRefAccess> accesses = entry.second;

    // Filter out memRef accesses that aren't read-only.
    if (CollectReadOnlyAccesses && !areReadOnly(accesses))
      continue;

    bool temporalReuse = haveTemporalReuse(accesses, threadVars);

    for (const affine::MemRefAccess &access : accesses) {
      LLVM_DEBUG(llvm::dbgs() << "Classify: " << *access.opInst << "\n");

      std::optional<MemoryAccess> memAccess =
          memAccessAnalysis.getMemoryAccess(access);
      if (!memAccess.has_value()) {
        LLVM_DEBUG(llvm::dbgs() << "Unable to analyze memory access\n");
        continue;
      }

      // Get the inter-thread access pattern and classify the memory access.
      MemoryAccessMatrix interThreadMatrix =
          memAccess->getInterThreadAccessMatrix(threadVars.size());
      MemoryAccessPattern interThreadAccessPattern = MemoryAccess::classify(
          interThreadMatrix, memAccess->getOffsetVector(), solver);

      switch (interThreadAccessPattern) {
      case Linear:
      case Reverse:
      case ReverseLinear:
        // These patterns lead to fully coalesced memory accesses.
        accessToMemSpace[&access] = MemorySpace::Global;
        break;
      case Shifted:
      case LinearShifted:
      case ReverseLinearShifted:
      case LinearOverlapped:
      case ReverseLinearOverlapped:
        // These patterns lead to partially coalesced memory accesses.
        accessToMemSpace[&access] = MemorySpace::Global;
        break;
      case Strided:
      case ReverseStrided:
      case StridedShifted:
      case ReverseStridedShifted:
      case Overlapped:
      case StridedOverlapped:
      case ReverseStridedOverlapped: {
        Value strideVal =
            interThreadMatrix(interThreadMatrix.getNumRows() - 1,
                              interThreadMatrix.getNumColumns() - 1);

        // Use shared memory only if the stride is known and it's greater than a
        // sufficiently large value (small stride values yield partially
        // coalesed memory accesses).
        if (auto stride = getConstIntegerValue(strideVal, solver)) {
          bool strideIsLargeEnough = stride->sgt(8) || stride->slt(-8);
          bool useSharedMemory =
              temporalReuse && (stride->isZero() || strideIsLargeEnough);
          accessToMemSpace[&access] =
              useSharedMemory ? MemorySpace::Shared : MemorySpace::Global;
        } else
          accessToMemSpace[&access] = MemorySpace::Global;
      } break;
      default:
        accessToMemSpace[&access] = MemorySpace::Global;
      }
    }
  }
}

void MemorySelector::addMemRefAccess(affine::MemRefAccess access) {
  auto it = accesses.find(access.memref);
  if (it == accesses.end())
    accesses[access.memref] = {access};
  else
    it->second.push_back(access);
}

bool MemorySelector::areReadOnly(
    ArrayRef<affine::MemRefAccess> accesses) const {
  return llvm::none_of(accesses, [](const affine::MemRefAccess &access) {
    return access.isStore();
  });
}

bool MemorySelector::areWriteOnly(
    ArrayRef<affine::MemRefAccess> accesses) const {
  return llvm::all_of(accesses, [](const affine::MemRefAccess &access) {
    return access.isStore();
  });
}

bool MemorySelector::areReadWrite(
    ArrayRef<affine::MemRefAccess> accesses) const {
  bool hasStores =
      llvm::any_of(accesses, [](const affine::MemRefAccess &access) {
        return access.isStore();
      });
  bool hasLoads =
      llvm::any_of(accesses, [](const affine::MemRefAccess &access) {
        return !access.isStore();
      });
  return hasLoads && hasStores;
}

bool MemorySelector::haveTemporalReuse(
    ArrayRef<affine::MemRefAccess> accesses,
    const SmallVectorImpl<Value> &threadVars) const {
  return llvm::any_of(accesses, [&](const affine::MemRefAccess &access) {
    std::optional<MemoryAccess> memAccess =
        memAccessAnalysis.getMemoryAccess(access);
    if (!memAccess)
      return false;

    // A non-zero intra-thread access matrix implies multiple threads access
    // the same array element (in a loop).
    MemoryAccessMatrix intraThreadMatrix =
        memAccess->getIntraThreadAccessMatrix(threadVars.size());
    return !intraThreadMatrix.isZero(solver);
  });
}

//===----------------------------------------------------------------------===//
// LoopInternalization
//===----------------------------------------------------------------------===//

struct LoopInternalization
    : public polygeist::impl::LoopInternalizationBase<LoopInternalization> {
  using LoopInternalizationBase<LoopInternalization>::LoopInternalizationBase;

  void runOnOperation() final {
    assert(CollectReadOnlyAccesses &&
           "Limitation: only able to handle read only accesses currently");

    Operation *module = getOperation();
    ModuleAnalysisManager mam(module, /*passInstrumentor=*/nullptr);
    AnalysisManager am = mam;
    auto &memAccessAnalysis =
        am.getAnalysis<MemoryAccessAnalysis>().initialize(relaxedAliasing);

    auto walkResult = module->walk([&](FunctionOpInterface func) {
      DataFlowSolver solver;
      solver.load<dataflow::DeadCodeAnalysis>();
      solver.load<dataflow::IntegerRangeAnalysis>();
      if (failed(solver.initializeAndRun(func)))
        return WalkResult::interrupt();

      LLVM_DEBUG(llvm::dbgs() << "LoopInternalization: Visiting Function "
                              << func.getName() << "\n");
      func->walk<WalkOrder::PreOrder>([&](LoopLikeOpInterface loop) {
        LLVM_DEBUG(llvm::dbgs() << "LoopInternalization: Visiting Loop:\n"
                                << loop << "\n");

        if (!isCandidate(loop))
          return;

        transform(loop, memAccessAnalysis, solver);
      });

      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted())
      signalPassFailure();
  }

private:
  void transform(LoopLikeOpInterface loop,
                 const MemoryAccessAnalysis &memAccessAnalysis,
                 DataFlowSolver &solver) const {
    TypeSwitch<Operation *>(loop).Case<affine::AffineForOp, scf::ForOp>(
        [&](auto loop) { transform(loop, memAccessAnalysis, solver); });
  }

  template <typename T, typename = std::enable_if_t<llvm::is_one_of<
                            T, affine::AffineForOp, scf::ForOp>::value>>
  void transform(T loop, const MemoryAccessAnalysis &memAccessAnalysis,
                 DataFlowSolver &solver) const {
    std::optional<LoopLikeOpInterface> innermostLoop =
        LoopTools::getInnermostLoop(loop);
    assert(innermostLoop.has_value() && "Failed to get the innermost loop");

    // Analyze affine memory accesses in the innermost loop.
    MemorySelector memorySelector(memAccessAnalysis, solver);
    memorySelector.analyze(*innermostLoop);

    // Determine which memory space should be used for each affine operation.
    DenseMap<Value, MemorySelector::MemorySpace> valueToMemorySpace;
    innermostLoop->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (!isa<affine::AffineLoadOp, affine::AffineStoreOp>(op))
        return;

      affine::MemRefAccess access(op);
      auto it = valueToMemorySpace.find(access.memref);
      if (it != valueToMemorySpace.end())
        return;

      std::optional<MemorySelector::MemorySpace> memSpace =
          memorySelector.selectMemorySpace(access.memref);
      if (memSpace)
        valueToMemorySpace[access.memref] = *memSpace;
    });

    // TODO: prioritize the array accesses that should use shared memory.

    SmallVector<T> nestedLoops;
    LoopTools::getPerfectlyNestedLoops(nestedLoops, loop);

    SmallVector<Value> tileSizes;
    if (getTileSizes(nestedLoops, tileSizes).failed())
      return;

    SmallVector<T> tiledNest;
    LogicalResult res = tile(nestedLoops, tileSizes, tiledNest);
    LLVM_DEBUG({
      if (res.succeeded())
        llvm::dbgs() << "Tiled loop:\n" << tiledNest.front() << "\n";
      else
        llvm::dbgs() << "Tile NOT performed\n";
    });

    // TODO: promote loop accesses to local memory.
    OpBuilder builder(loop);
    builder.setInsertionPointToStart(tiledNest.front()->getBlock());
    createLocalBarrier(builder);
    builder.setInsertionPointAfter(tiledNest.front());
    createLocalBarrier(builder);
  }
};

} // namespace

std::unique_ptr<Pass> polygeist::createLoopInternalizationPass() {
  return std::make_unique<LoopInternalization>();
}
