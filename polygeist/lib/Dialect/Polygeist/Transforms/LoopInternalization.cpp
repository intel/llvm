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
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "loop-internalization"

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_LOOPINTERNALIZATION
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

using namespace mlir;
using namespace mlir::polygeist;

namespace {

//===----------------------------------------------------------------------===//
// Utilities functions
//===----------------------------------------------------------------------===//

/// A function is a candidate iff is a kernel body functions with an nd_item
/// argument.
bool isCandidateFunction(FunctionOpInterface func) {
  if (!polygeist::isPotentialKernelBodyFunc(func))
    return false;

  // TODO: construct nd_item when not passed in.
  if (func.getNumArguments() == 0 ||
      !sycl::isPtrOf<sycl::NdItemType>(func.getArgumentTypes().back()))
    return false;

  return true;
}

/// A loop nest is a candidate iff is perfect and contains only affine or scf
/// for loops.
bool isCandidateLoopNest(LoopLikeOpInterface loop) {
  if (!LoopTools::isOutermostLoop(loop))
    return false;

  if (!LoopTools::isPerfectLoopNest(loop))
    return false;

  std::optional<LoopLikeOpInterface> innermostLoop =
      LoopTools::getInnermostLoop(loop);
  assert(innermostLoop.has_value() && "Failed to get the innermost loop");
  if (!isa<affine::AffineForOp, scf::ForOp>(*innermostLoop))
    return false;

  // TODO: check uniformity.

  return true;
}

/// Tile an affine for \p loop given the tile size \p tileSize.
LogicalResult tile(affine::AffineForOp loop, Value tileSize,
                   SmallVectorImpl<affine::AffineForOp> &tiledNest) {
  SmallVector<affine::AffineForOp> newNestedLoops;
  LogicalResult res =
      tilePerfectlyNestedParametric({loop}, tileSize, &newNestedLoops);
  tiledNest = SmallVector<affine::AffineForOp>(newNestedLoops.begin() + 1,
                                               newNestedLoops.end());
  return res;
}

/// Tile an SCF for \p loop given the tile size \p tileSize.
LogicalResult tile(scf::ForOp loop, Value tileSize,
                   SmallVectorImpl<scf::ForOp> &tiledNest) {
  tiledNest = tile({loop}, tileSize, loop);
  return success();
}

/// Create a group barrier.
void createLocalBarrier(OpBuilder &builder) {
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

  /// The kind of accesses to consider.
  enum class AccessKind { ReadOnly, WriteOnly, ReadWrite };

  /// Enumerate memory spaces.
  enum class MemorySpace { Global, Shared, Constant, Texture };

  /// Returns the most suitable memory space the \p op should use.
  std::optional<MemorySpace> getMemorySpace(Operation *op) const;

  /// Analyze the memory accesses in the given loop.
  void analyze(LoopLikeOpInterface loop, AccessKind accessKind);

private:
  /// Return true iff no memref accesses in \p accesses are stores.
  bool areReadOnly(ArrayRef<affine::MemRefAccess> accesses) const;

  /// Return true iff all memref accesses in \p accesses are stores.
  bool areWriteOnly(ArrayRef<affine::MemRefAccess> accesses) const;

  /// Return true if memref accesses in \p accesses are a mix of loads and
  /// stores.
  bool areReadWrite(ArrayRef<affine::MemRefAccess> accesses) const;

  /// Determine whether the memref \access exhibits temporal reuse.
  bool hasTemporalReuse(const affine::MemRefAccess &access,
                        const SmallVectorImpl<Value> &threadVars) const;

private:
  const MemoryAccessAnalysis &memAccessAnalysis;

  DataFlowSolver &solver;

  /// The preferred memory space for each memref access.
  DenseMap<Operation *, MemorySpace> accessToMemSpace;
};

std::optional<MemorySelector::MemorySpace>
MemorySelector::getMemorySpace(Operation *op) const {
  auto it = accessToMemSpace.find(op);
  if (it == accessToMemSpace.end())
    return std::nullopt;
  return it->second;
}

void MemorySelector::analyze(LoopLikeOpInterface loop, AccessKind accessKind) {
  // Collect the global thread ids used in the function the loop is in.
  auto funcOp = loop->template getParentOfType<FunctionOpInterface>();
  SmallVector<Value> threadVars =
      memAccessAnalysis.getThreadVector(funcOp, solver);

  // Collect candidate memref accesses in the loop.
  DenseMap<Value, SmallVector<affine::MemRefAccess>> accesses;
  loop->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<affine::AffineLoadOp, affine::AffineStoreOp>(op))
      return;

    affine::MemRefAccess access(op);
    accesses[access.memref] = {access};
  });

  // Return true iff the accesses are all of the requested access kind.
  auto areOfRequestedKind = [&](ArrayRef<affine::MemRefAccess> accesses) {
    switch (accessKind) {
    case AccessKind::ReadOnly:
      return areReadOnly(accesses);
    case AccessKind::WriteOnly:
      return areWriteOnly(accesses);
    case AccessKind::ReadWrite:
      return areReadWrite(accesses);
    }
  };

  // Analyze the accesses collected and populate the 'accessToMemSpace' map.
  for (auto &entry : accesses) {
    ArrayRef<affine::MemRefAccess> accesses = entry.second;

    // Skip accesses that aren't of the requested kind.
    // TODO: consider aliased accesses.
    if (!areOfRequestedKind(accesses))
      continue;

    // Note: all our candidate accesses have the same subscript and zero index
    // therefore we need to analyze the first one only.
    const affine::MemRefAccess &access = accesses.front();
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
      // These patterns imply fully coalesced memory accesses.
      accessToMemSpace[access.opInst] = MemorySpace::Global;
      break;
    case Shifted:
    case LinearShifted:
    case ReverseLinearShifted:
    case LinearOverlapped:
    case ReverseLinearOverlapped:
      // These patterns imply partially coalesced memory accesses.
      accessToMemSpace[access.opInst] = MemorySpace::Global;
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

      // Use shared memory iff:
      //   - the memory access exhibits temporal reuse, and
      //   - the stride is greater than a sufficiently large value (small
      //     stride values yield partially coalesed memory accesses).
      // Note that a zero stride is indicative of non-coalesed accesses.
      // Example (assume tx,ty are global thread ids):
      //     for(k)
      //       ... = A[{tx, k}] // increasing tx's values read across rows.
      // The inter-thread access matrix for A's load is:
      //   1 0
      //   0 C <- where C == 0 (C is the stride).
      bool useSharedMemory = false;
      if (auto stride = getConstIntegerValue(strideVal, solver)) {
        bool strideIsLargeEnough = stride->sgt(8) || stride->slt(-8);
        useSharedMemory = hasTemporalReuse(access, threadVars) &&
                          (stride->isZero() || strideIsLargeEnough);
        // FIXME: getConstIntegerValue doesn't return zero for
        // ConstantIndexOp.
      } else if (auto constVal = dyn_cast<arith::ConstantIndexOp>(
                     strideVal.getDefiningOp());
                 constVal.value() == 0)
        useSharedMemory = hasTemporalReuse(access, threadVars);

      accessToMemSpace[access.opInst] =
          useSharedMemory ? MemorySpace::Shared : MemorySpace::Global;
    } break;
    default:
      accessToMemSpace[access.opInst] = MemorySpace::Global;
    }

    LLVM_DEBUG({
      if (accessToMemSpace.at(access.opInst) == MemorySpace::Shared)
        llvm::dbgs().indent(2) << "shared memory space\n";
      else {
        assert(accessToMemSpace.at(access.opInst) == MemorySpace::Global);
        llvm::dbgs().indent(2) << "global memory space\n";
      }
    });
  }
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

bool MemorySelector::hasTemporalReuse(
    const affine::MemRefAccess &memRefAccess,
    const SmallVectorImpl<Value> &threadVars) const {
  std::optional<MemoryAccess> access =
      memAccessAnalysis.getMemoryAccess(memRefAccess);
  if (!access)
    return false;

  // A non-zero intra-thread access matrix implies that multiple threads access
  // the same array element (in a loop).
  return !access->getIntraThreadAccessMatrix(threadVars.size()).isZero(solver);
}

//===----------------------------------------------------------------------===//
// LoopInternalization
//===----------------------------------------------------------------------===//

struct LoopInternalization
    : public polygeist::impl::LoopInternalizationBase<LoopInternalization> {
  using LoopInternalizationBase<LoopInternalization>::LoopInternalizationBase;

  void runOnOperation() final;

private:
  /// Construct a map from memref accesses in \p loop to their ideal memory
  /// space.
  void selectMemorySpace(LoopLikeOpInterface loop,
                         const MemoryAccessAnalysis &memAccessAnalysis,
                         DataFlowSolver &solver);

  /// Determine the tile size for \p loop.
  Value getTileSize(LoopLikeOpInterface loop) const;

  /// Transform a candidate loop.
  template <typename T>
  void transform(T loop, const MemoryAccessAnalysis &memAccessAnalysis,
                 DataFlowSolver &solver) const;

private:
  /// A map from a candidate loop to memref values used in the loop.
  DenseMap<LoopLikeOpInterface, SmallVector<Value>> loopToMemref;

  /// Map from a candidate memref value to its ideal memory space.
  DenseMap<Value, MemorySelector::MemorySpace> memrefToMemorySpace;
};

void LoopInternalization::runOnOperation() {
  Operation *module = getOperation();
  ModuleAnalysisManager mam(module, /*passInstrumentor=*/nullptr);
  AnalysisManager am = mam;
  auto &memAccessAnalysis =
      am.getAnalysis<MemoryAccessAnalysis>().initialize(relaxedAliasing);

  // Walk each function in the module.
  module->walk([&](FunctionOpInterface func) {
    if (!isCandidateFunction(func))
      return;

    LLVM_DEBUG(llvm::dbgs()
               << "LoopInternalization: Visiting candidate function "
               << func.getName() << "\n");

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::IntegerRangeAnalysis>();
    if (failed(solver.initializeAndRun(func))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "LoopInternalization: Unable to run required dataflow "
                    "analysis on "
                 << func.getName() << "\n");
      return;
    }

    // Select the ideal memory space for memref accesses in candidate loops
    // contained by this function.
    func->walk<WalkOrder::PreOrder>([&](LoopLikeOpInterface loop) {
      if (!isCandidateLoopNest(loop))
        return;

      LLVM_DEBUG(
          llvm::dbgs()
          << "LoopInternalization: Visiting candidate loop nest rooted by:\n"
          << loop << "\n");

      std::optional<LoopLikeOpInterface> innermostLoop =
          LoopTools::getInnermostLoop(loop);
      assert(innermostLoop.has_value() && "Failed to get the innermost loop");

      // Determine the ideal memory space for memref accesses contained in the
      // innermost loop.
      selectMemorySpace(*innermostLoop, memAccessAnalysis, solver);

      // TODO: prioritize the array accesses that should use shared memory.
      // prioritize(memAccessAnalysis, solver);
    });

    DenseMap<LoopLikeOpInterface, SmallVector<Value>> loopToSharedMemref;
    for (auto &entry : loopToMemref) {
      copy_if(entry.second, std::back_inserter(loopToSharedMemref[entry.first]),
              [&](Value memref) {
                return (memrefToMemorySpace.at(memref) ==
                        MemorySelector::MemorySpace::Shared);
              });

      // No need to transform if no accesses need to be promoted to shared local
      // memory.
      if (loopToSharedMemref.at(entry.first).empty())
        loopToSharedMemref.erase(entry.first);
    }

    // Now that we have the ideal memory space for all analyzable memref
    // accesses in each loop nest's innermost loop, perform the transformation.
    for (auto &entry : loopToSharedMemref) {
      TypeSwitch<Operation *>(entry.first)
          .Case<affine::AffineForOp, scf::ForOp>(
              [&](auto loop) { transform(loop, memAccessAnalysis, solver); });
    }
  });
}

void LoopInternalization::selectMemorySpace(
    LoopLikeOpInterface loop, const MemoryAccessAnalysis &memAccessAnalysis,
    DataFlowSolver &solver) {
  assert(LoopTools::getInnermostLoop(loop) && "Expecting an innermost loop");
  assert(loopToMemref.find(loop) == loopToMemref.end() &&
         "The loop should not be already present in the map");

  // Use the memory selector to determine the ideal memory space for memref
  // accesses in the innermost loop.
  // TODO: allow memory selection on read-write accesses.
  MemorySelector memorySelector(memAccessAnalysis, solver);
  memorySelector.analyze(loop, MemorySelector::AccessKind::ReadOnly);

  loop->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<affine::AffineLoadOp, affine::AffineStoreOp>(op))
      return;

    affine::MemRefAccess memRefAccess(op);

    // Skip if the memref is already in the map.
    if (memrefToMemorySpace.find(memRefAccess.memref) !=
        memrefToMemorySpace.end())
      return;

    // Compute the ideal memory space if possible.
    std::optional<MemorySelector::MemorySpace> memSpace =
        memorySelector.getMemorySpace(op);
    if (!memSpace)
      return;

    // Record we have processed the memref in this loop...
    auto it = loopToMemref.find(loop);
    if (it == loopToMemref.end())
      loopToMemref[loop] = {memRefAccess.memref};
    else {
      SmallVector<Value> &memRefs = loopToMemref[loop];
      memRefs.push_back(memRefAccess.memref);
    }

    // ... and record the memref memory space.
    memrefToMemorySpace[memRefAccess.memref] = *memSpace;
  });
}

Value LoopInternalization::getTileSize(LoopLikeOpInterface loop) const {
  // TODO: calculate proper tile sizes.
  OpBuilder builder(loop);
  return builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(),
                                                tileSize);
}

template <typename T>
void LoopInternalization::transform(
    T loop, const MemoryAccessAnalysis &memAccessAnalysis,
    DataFlowSolver &solver) const {
  static_assert(llvm::is_one_of<T, affine::AffineForOp, scf::ForOp>::value);
  assert(LoopTools::isInnermostLoop(loop) && "Expecting an innermost loop");
  assert(loopToMemref.find(loop) != loopToMemref.end() &&
         "Loop should be in the map");

  SmallVector<T> tiledNest;
  LogicalResult res = tile(loop, getTileSize(loop), tiledNest);
  assert(res.succeeded() && "Expecting innermost loop to be tiled");

  LLVM_DEBUG(llvm::dbgs() << "Tiled loop: " << tiledNest.front() << "\n");

  // TODO: promote loop accesses to local memory.
  loop = tiledNest.front();
  OpBuilder builder(loop);
  builder.setInsertionPointToStart(loop->getBlock());
  createLocalBarrier(builder);
  builder.setInsertionPointAfter(loop);
  createLocalBarrier(builder);
}

} // namespace

std::unique_ptr<Pass> polygeist::createLoopInternalizationPass() {
  return std::make_unique<LoopInternalization>();
}
std::unique_ptr<Pass> polygeist::createLoopInternalizationPass(
    const LoopInternalizationOptions &options) {
  return std::make_unique<LoopInternalization>(options);
}
