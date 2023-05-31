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

// Only visit kernel body function with nd_item argument.
bool isCandidate(FunctionOpInterface func) {
  if (!polygeist::isPotentialKernelBodyFunc(func))
    return false;

  // TODO: construct nd_item when not passed in.
  if (func.getNumArguments() == 0 ||
      !sycl::isPtrOf<sycl::NdItemType>(func.getArgumentTypes().back()))
    return false;

  return true;
}

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
void getTileSizes(const SmallVector<T> &nestedLoops,
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
      memAccessAnalysis.getThreadVector(funcOp, solver);

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
        // These patterns imply fully coalesced memory accesses.
        accessToMemSpace[&access] = MemorySpace::Global;
        break;
      case Shifted:
      case LinearShifted:
      case ReverseLinearShifted:
      case LinearOverlapped:
      case ReverseLinearOverlapped:
        // These patterns imply partially coalesced memory accesses.
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
        }

        accessToMemSpace[&access] =
            useSharedMemory ? MemorySpace::Shared : MemorySpace::Global;
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

  /// Transform a candidate loop.
  template <typename T>
  void transform(T loop, const MemoryAccessAnalysis &memAccessAnalysis,
                 DataFlowSolver &solver) const;

private:
  /// A map from a candidate loop to memref values used in the loop.
  DenseMap<LoopLikeOpInterface, SmallVector<Value>> loopToMemref;

  /// Map from a candidate a memref value to its ideal memory space.
  DenseMap<Value, MemorySelector::MemorySpace> memrefToMemorySpace;
};

void LoopInternalization::runOnOperation() {
  assert(CollectReadOnlyAccesses &&
         "Limitation: only able to handle read only accesses currently");

  Operation *module = getOperation();
  ModuleAnalysisManager mam(module, /*passInstrumentor=*/nullptr);
  AnalysisManager am = mam;
  auto &memAccessAnalysis =
      am.getAnalysis<MemoryAccessAnalysis>().initialize(relaxedAliasing);

  // Walk each function in the module.
  module->walk([&](FunctionOpInterface func) {
    if (!isCandidate(func))
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
      if (!isCandidate(loop))
        return;

      LLVM_DEBUG(llvm::dbgs()
                 << "LoopInternalization: Visiting candidate loop:\n"
                 << loop << "\n");

      // Construct a map from memref accesses to their ideal memory space.
      selectMemorySpace(loop, memAccessAnalysis, solver);

      // TODO: prioritize the array accesses that should use shared memory.
      // prioritize(memAccessAnalysis, solver);
    });

    // Now that we have the ideal memory space for all analyzable memref
    // accesses in each loop, transform the loops.
    for (auto &entry : loopToMemref) {
      if (isa<affine::AffineForOp>(entry.first))
        transform(cast<affine::AffineForOp>(entry.first), memAccessAnalysis,
                  solver);
      else if (isa<scf::ForOp>(entry.first))
        transform(cast<scf::ForOp>(entry.first), memAccessAnalysis, solver);
      else
        llvm_unreachable("Unexpected loop kind");
    }
  });
}

void LoopInternalization::selectMemorySpace(
    LoopLikeOpInterface loop, const MemoryAccessAnalysis &memAccessAnalysis,
    DataFlowSolver &solver) {
  assert(loopToMemref.find(loop) == loopToMemref.end() &&
         "Loop should not be already present in the map");

  std::optional<LoopLikeOpInterface> innermostLoop =
      LoopTools::getInnermostLoop(loop);
  assert(innermostLoop.has_value() && "Failed to get the innermost loop");

  // Use the memory selector to determine the ideal memory space for memref
  // accesses in the innermost loop.
  MemorySelector memorySelector(memAccessAnalysis, solver);
  memorySelector.analyze(*innermostLoop);

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
        memorySelector.selectMemorySpace(memRefAccess.memref);
    if (!memSpace)
      return;

    // Record we have processes the memref in this loop...
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

template <typename T>
void LoopInternalization::transform(
    T loop, const MemoryAccessAnalysis &memAccessAnalysis,
    DataFlowSolver &solver) const {
  static_assert(llvm::is_one_of<T, affine::AffineForOp, scf::ForOp>::value);
  assert(loopToMemref.find(loop) != loopToMemref.end() &&
         "Loop should be in the map");

  SmallVector<T> nestedLoops;
  LoopTools::getPerfectlyNestedLoops(nestedLoops, loop);

  SmallVector<Value> tileSizes;
  getTileSizes(nestedLoops, tileSizes);

  SmallVector<T> tiledNest;
  LogicalResult res = tile(nestedLoops, tileSizes, tiledNest);
  LLVM_DEBUG({
    if (res.succeeded())
      llvm::dbgs() << "Tiled loop:\n" << tiledNest.front() << "\n";
    else
      llvm::dbgs() << "Tile NOT performed\n";
  });

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
