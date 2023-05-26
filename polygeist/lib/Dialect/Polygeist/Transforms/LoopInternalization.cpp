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

static llvm::cl::list<unsigned> LoopInternalizationTileSizes(
    DEBUG_TYPE "-tile-sizes", llvm::cl::CommaSeparated,
    llvm::cl::desc("Tile sizes used in LoopInternalization"));

namespace {

//===----------------------------------------------------------------------===//
// Utilities functions
//===----------------------------------------------------------------------===//

/// A loop is a candidate when it is the outermost affine or scf for loop.
bool isCandidate(LoopLikeOpInterface loop) {
  if (!LoopTools::isOutermostLoop(loop)) {
    LLVM_DEBUG(llvm::dbgs() << "not candidate: not outermost loop\n");
    return false;
  }

  if (!isa<affine::AffineForOp, scf::ForOp>(loop)) {
    LLVM_DEBUG(llvm::dbgs() << "not candidate: not affine or scf for loop\n");
    return false;
  }

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
  return tilePerfectlyNestedParametric(nestedLoops, tileSizes, &tiledNest);
}
LogicalResult tile(SmallVectorImpl<scf::ForOp> &nestedLoops,
                   ArrayRef<Value> tileSizes,
                   SmallVectorImpl<scf::ForOp> &tiledNest) {
  tile(nestedLoops, tileSizes, nestedLoops.back());
  tiledNest = nestedLoops;
  return success();
}

//===----------------------------------------------------------------------===//
// MemorySelector
//===----------------------------------------------------------------------===//

/// Collect memory accesses in a loop and determine the memory space each access
/// should ideally use.
class MemorySelector {
public:
  MemorySelector(LoopLikeOpInterface loop,
                 const MemoryAccessAnalysis &memAccessAnalysis,
                 DataFlowSolver &solver)
      : loop(loop), memAccessAnalysis(memAccessAnalysis), solver(solver) {
    assert(!loop->getParentOfType<LoopLikeOpInterface>() &&
           "Expecting an inner loop");
    populate();
  };

  /// Enumerate memory spaces.
  enum class MemorySpace { Global, Shared, Constant, Texture };

  /// Returns the memory space the given \p memRefAccess should use.
  MemorySpace selectMemorySpace(affine::MemRefAccess access) const {
    auto it = accessToMemSpace.find(&access);
    if (it == accessToMemSpace.end())
      return MemorySpace::Global;
    return accessToMemSpace.at(&access);
  }

private:
  void populate();

  /// Fill in \p threadsVars with the threads used in function containing the
  /// loop.
  void computeThreadVector(SmallVectorImpl<Value> &threadsVars) const;

  /// Add the given \p access to the 'accesses' map.
  void addMemRefAccess(affine::MemRefAccess access);

  /// Retrieve all the memref accesses for \p access.
  const SmallVector<affine::MemRefAccess *> *
  getMemRefAccesses(affine::MemRefAccess *access) const;

  /// Return true iff no memref accesses in \p accesses are stores.
  bool isReadOnly(ArrayRef<affine::MemRefAccess *> accesses) const;

  /// Return true iff all memref accesses in \p accesses are stores.
  bool isWriteOnly(ArrayRef<affine::MemRefAccess *> accesses) const;

  /// Return true if memref accesses in \p accesses are a mix of loads and
  /// stores.
  bool isReadWrite(ArrayRef<affine::MemRefAccess *> accesses) const;

  /// Return true if the access is coalesced.
  bool isCoalesced(affine::MemRefAccess *access) const;

private:
  LoopLikeOpInterface loop;

  const MemoryAccessAnalysis &memAccessAnalysis;

  DataFlowSolver &solver;

  /// Collects all memory accesses for a given memref value.
  DenseMap<Value, SmallVector<affine::MemRefAccess *>> accesses;

  /// The preferred memory space for each memref access;
  DenseMap<affine::MemRefAccess *, MemorySpace> accessToMemSpace;
};

void MemorySelector::populate() {
  assert(accesses.empty() && accessToMemSpace.empty() &&
         "Expecting empty maps");

  // Collect all memref accesses in the loop that have an entry in the memory
  // access analysis.
  loop->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<affine::AffineLoadOp, affine::AffineStoreOp>(op))
      return;

    affine::MemRefAccess access(op);
    const std::optional<MemoryAccess> memAccess =
        memAccessAnalysis.getMemoryAccess(access);
    if (!memAccess)
      return;

    addMemRefAccess(access);
  });

  // Analyze the memref accesses collected.
  for (auto entry : accesses) {
    const SmallVector<affine::MemRefAccess *> &accesses = entry.second;
    unsigned numAccesses = accesses.size();

    isReadOnly(entry.second);
    isReadWrite(entry.second);
    isWriteOnly(entry.second);

    for (const affine::MemRefAccess *access : entry.second) {
      std::optional<MemoryAccess> memAccess =
          memAccessAnalysis.getMemoryAccess(*access);
      assert(memAccess.has_value() &&
             "Expecting an entry in the memory access analysis");

      // Get the sub matrix describing the inter-thread access pattern.

      MemoryAccessPattern pattern = memAccess->classify(solver);

      assert(0 && "TODO");
    }
  }
}

void MemorySelector::computeThreadVector(
    SmallVectorImpl<Value> &threadsVars) const {
  assert(threadsVars.empty() && "Expecting an empty vector");

  assert(false && "TODO");
  //  for (sycl::SYCLNDItemGetGlobalIDOp getGlobalIdOp : getGlobalIdOps)
  //  loopAndThreadVars.emplace_back(getGlobalIdOp.getResult());
}

void MemorySelector::addMemRefAccess(affine::MemRefAccess access) {
  auto it = accesses.find(access.memref);
  if (it == accesses.end())
    accesses[access.memref] = {&access};
  else
    it->second.push_back(&access);
}

const SmallVector<affine::MemRefAccess *> *
MemorySelector::getMemRefAccesses(affine::MemRefAccess *access) const {
  assert(access && "Expecting a valid pointer");
  auto it = accesses.find(access->memref);
  if (it == accesses.end())
    return nullptr;
  return &it->second;
}

bool MemorySelector::isReadOnly(
    ArrayRef<affine::MemRefAccess *> accesses) const {
  return llvm::none_of(accesses, [](const affine::MemRefAccess *access) {
    return access->isStore();
  });
}

bool MemorySelector::isWriteOnly(
    ArrayRef<affine::MemRefAccess *> accesses) const {
  return llvm::all_of(accesses, [](const affine::MemRefAccess *access) {
    return access->isStore();
  });
}

bool MemorySelector::isReadWrite(
    ArrayRef<affine::MemRefAccess *> accesses) const {
  bool hasStores =
      llvm::any_of(accesses, [](const affine::MemRefAccess *access) {
        return access->isStore();
      });
  bool hasLoads =
      llvm::any_of(accesses, [](const affine::MemRefAccess *access) {
        return !access->isStore();
      });
  return hasLoads && hasStores;
}

//===----------------------------------------------------------------------===//
// LoopInternalization
//===----------------------------------------------------------------------===//

struct LoopInternalization
    : public polygeist::impl::LoopInternalizationBase<LoopInternalization> {
  using LoopInternalizationBase<LoopInternalization>::LoopInternalizationBase;

  void runOnOperation() final;

private:
  void transform(LoopLikeOpInterface loop,
                 const MemoryAccessAnalysis &memAccessAnalysis,
                 DataFlowSolver &solver) const;

  template <typename T, typename = std::enable_if_t<llvm::is_one_of<
                            T, affine::AffineForOp, scf::ForOp>::value>>
  void transform(T loop, const MemoryAccessAnalysis &memAccessAnalysis,
                 DataFlowSolver &solver) const {
    FunctionOpInterface func =
        loop->template getParentOfType<FunctionOpInterface>();

    // Retrieve the innermost loop (exits iff the loop nest is perfect).
    std::optional<LoopLikeOpInterface> innermostLoop =
        LoopTools::getInnermostLoop(loop);
    if (!innermostLoop) {
      llvm::dbgs() << "Not able to find innermost loop\n";
      return;
    }

    assert(LoopTools::isPerfectLoopNest(loop) &&
           "Expecting a perfect loop nest");

    // Analyze the memory accesses in the innermost loop.
    //    MemorySelector memorySelector(*innermostLoop, memAccessAnalysis,
    //    solver);

    SmallVector<T> nestedLoops;
    LoopTools::getPerfectlyNestedLoops(nestedLoops, loop);

    SmallVector<Value> tileSizes;
    if (getTileSizes(nestedLoops, tileSizes).failed())
      return;

    SmallVector<T> tiledNest;
    LogicalResult res = tile(nestedLoops, tileSizes, tiledNest);
    LLVM_DEBUG({
      if (res.succeeded())
        llvm::dbgs() << "Tiled loop: " << tiledNest.front() << "\n";
      else
        llvm::dbgs() << "Tile NOT performed\n";
    });
    // TODO: promote loop accesses to local memory.
  }
};

void LoopInternalization::runOnOperation() {
  Operation *op = getOperation();
  ModuleAnalysisManager mam(op, /*passInstrumentor=*/nullptr);
  AnalysisManager am = mam;
  auto &memAccessAnalysis =
      am.getAnalysis<MemoryAccessAnalysis>().initialize(relaxedAliasing);

  auto walkResult = op->walk([&](FunctionOpInterface func) {
    DataFlowSolver solver;
    //    solver.load<dataflow::DeadCodeAnalysis>();
    //  solver.load<dataflow::IntegerRangeAnalysis>();

    // if (failed(solver.initializeAndRun(func)))
    //  return WalkResult::interrupt();

    func->walk<WalkOrder::PreOrder>([&](LoopLikeOpInterface loop) {
      LLVM_DEBUG({
        llvm::dbgs() << "LoopInternalization: Visiting Function "
                     << func.getName() << "\n Loop: ";
        loop.dump();
      });

      if (!isCandidate(loop))
        return;

      transform(loop, memAccessAnalysis, solver);
    });

    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    signalPassFailure();
}

void LoopInternalization::transform(
    LoopLikeOpInterface loop, const MemoryAccessAnalysis &memAccessAnalysis,
    DataFlowSolver &solver) const {
  TypeSwitch<Operation *>(loop).Case<affine::AffineForOp, scf::ForOp>(
      [&](auto loop) { transform(loop, memAccessAnalysis, solver); });
}

} // namespace

std::unique_ptr<Pass> polygeist::createLoopInternalizationPass() {
  return std::make_unique<LoopInternalization>();
}
