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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Polygeist/Analysis/MemoryAccessAnalysis.h"
#include "mlir/Dialect/Polygeist/Transforms/Passes.h"
#include "mlir/Dialect/Polygeist/Utils/TransformUtils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/Utils/Utils.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <variant>

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

/// This class contains utilities to create or manipulate 'std::variant<Value,
/// unsigned>'.
class ValueOrUnsigned {
public:
  static std::variant<Value, unsigned> get(unsigned val, OpBuilder builder,
                                           bool createVal) {
    if (!createVal)
      return val;
    return builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), val);
  }

  static Value getValue(std::variant<Value, unsigned> val, OpBuilder builder) {
    if (std::holds_alternative<Value>(val))
      return std::get<Value>(val);
    return builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(),
                                                  std::get<unsigned>(val));
  }

  static std::variant<Value, unsigned>
  add(const std::variant<Value, unsigned> &lhs,
      const std::variant<Value, unsigned> &rhs, OpBuilder builder) {
    return binaryOperator<arith::AddIOp>(lhs, rhs, builder);
  }

  static std::variant<Value, unsigned>
  mul(const std::variant<Value, unsigned> &lhs,
      const std::variant<Value, unsigned> &rhs, OpBuilder builder) {
    return binaryOperator<arith::MulIOp>(lhs, rhs, builder);
  }

private:
  template <typename T, typename = std::enable_if_t<llvm::is_one_of<
                            T, arith::AddIOp, arith::MulIOp>::value>>
  static std::variant<Value, unsigned>
  binaryOperator(const std::variant<Value, unsigned> &lhs,
                 const std::variant<Value, unsigned> &rhs, OpBuilder builder) {
    if (std::holds_alternative<unsigned>(lhs)) {
      assert(std::holds_alternative<unsigned>(rhs));
      if (std::is_same_v<T, arith::AddIOp>)
        return std::get<unsigned>(lhs) + std::get<unsigned>(rhs);
      if (std::is_same_v<T, arith::MulIOp>)
        return std::get<unsigned>(lhs) * std::get<unsigned>(rhs);
    }

    assert(std::holds_alternative<Value>(lhs) &&
           std::holds_alternative<Value>(rhs));
    return builder.create<T>(builder.getUnknownLoc(), std::get<Value>(lhs),
                             std::get<Value>(rhs));
  }
};

//===----------------------------------------------------------------------===//
// WorkGroupSize
//===----------------------------------------------------------------------===//

/// Class to represent work group size of a kernel.
class WorkGroupSize {
public:
  using ElemTy = std::variant<Value, unsigned>;
  WorkGroupSize(Value ndItem, const sycl::ReqdWorkGroupSize &reqdWorkGroupSize,
                OpBuilder builder) {
    auto ndItemTy = cast<sycl::NdItemType>(
        cast<MemRefType>(ndItem.getType()).getElementType());
    assert((reqdWorkGroupSize.empty() ||
            reqdWorkGroupSize.size() == ndItemTy.getDimension()) &&
           "Expecting reqdWorkGroupSize to have same number of elements as "
           "nd_item dimension");
    for (unsigned dim = 0; dim < ndItemTy.getDimension(); ++dim) {
      /// Use constants provided by reqd_work_group_size if given (non-empty).
      if (!reqdWorkGroupSize.empty())
        wgSizes.push_back(reqdWorkGroupSize[dim]);
      else
        wgSizes.push_back(builder.create<arith::IndexCastOp>(
            builder.getUnknownLoc(), builder.getIndexType(),
            getLocalRange(ndItem, dim, builder)));
    }
  }

  /// Return true if the element holds the alternative T.
  template <typename T, typename = std::enable_if_t<
                            llvm::is_one_of<T, Value, unsigned>::value>>
  bool hasElemTy() const {
    return std::holds_alternative<T>(wgSizes.front());
  }

  ElemTy operator[](unsigned dim) const {
    assert(dim < wgSizes.size() && "Expecting valid dim");
    return wgSizes[dim];
  }

private:
  Value getLocalRange(Value ndItem, unsigned dim, OpBuilder builder) {
    return builder.create<sycl::SYCLNDItemGetLocalRangeOp>(
        builder.getUnknownLoc(), builder.getI64Type(), ndItem,
        builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), dim,
                                             builder.getI32Type()));
  }

  SmallVector<ElemTy> wgSizes;
};

//===----------------------------------------------------------------------===//
// Utilities functions
//===----------------------------------------------------------------------===//

bool isLocalAccessAddrSpace(Type ty) {
  if (auto memRefTy = dyn_cast<MemRefType>(ty)) {
    if (auto memSpace = dyn_cast_or_null<sycl::AccessAddrSpaceAttr>(
            memRefTy.getMemorySpace())) {
      if (memSpace.getValue() == sycl::AccessAddrSpace::LocalAccess)
        return true;
      return false;
    }
    return (memRefTy.getMemorySpaceAsInt() == 3);
  }
  if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(ty))
    return (ptrTy.getAddressSpace() == 3);
  return false;
}

/// A kernel is a candidate iff no dynamic sized local accessor is used.
bool isCandidateKernel(gpu::GPUFuncOp kernel) {
  assert(kernel.isKernel() && "Expecting kernel");
  // Available local memory of a kernel cannot be calculated when dynamic sized
  // local memory is used, as its size is not compile time known on device.
  return none_of(kernel.getArguments(), [](Value arg) {
    return isLocalAccessAddrSpace(arg.getType());
  });
}

/// A function is a candidate iff it has an nd_item argument.
bool isCandidateFunction(FunctionOpInterface func) {
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

/// Get the argument of \p func with nd_item type.
Value getNdItemArgument(FunctionOpInterface func) {
  assert(func.getNumArguments() > 0 && "Expecting at least one argument");
  Value lastArg = func.getArguments().back();
  assert(sycl::isPtrOf<sycl::NdItemType>(lastArg.getType()) &&
         "Expecting last argument to be nd_item");
  return lastArg;
}

/// Get the size of unused shared local memory arena in bytes.
unsigned getLocalMemoryRemain(gpu::GPUModuleOp &module,
                              unsigned localMemorySize) {
  assert(module.hasTrait<OpTrait::SymbolTable>() &&
         "Expecting module with SymbolTable trait");
  unsigned localMemoryRemain = localMemorySize;
  module.walk([&localMemoryRemain](memref::GlobalOp global) {
    MemRefType memRefTy = global.getType();
    if (!isLocalAccessAddrSpace(memRefTy))
      return WalkResult::advance();
    unsigned globalSize =
        memRefTy.getElementTypeBitWidth() * memRefTy.getNumElements() / 8;
    if (globalSize >= localMemoryRemain) {
      localMemoryRemain = 0;
      return WalkResult::interrupt();
    }
    localMemoryRemain -= globalSize;
    return WalkResult::advance();
  });
  return localMemoryRemain;
}

/// Get the required local memory for \p accTy in bytes.
std::variant<Value, unsigned>
getReqdLocalMemory(sycl::AccessorType accTy, const WorkGroupSize &workGroupSize,
                   OpBuilder builder) {
  unsigned elemSize = accTy.getType().getIntOrFloatBitWidth() / 8;
  std::variant<Value, unsigned> reqdLocalMemory =
      ValueOrUnsigned::get(elemSize, builder, workGroupSize.hasElemTy<Value>());
  for (unsigned dim = 0; dim < accTy.getDimension(); ++dim)
    reqdLocalMemory =
        ValueOrUnsigned::mul(reqdLocalMemory, workGroupSize[dim], builder);
  return reqdLocalMemory;
}

/// Get the required local memory for \p accTy in bytes.
unsigned getReqdLocalMemory(sycl::AccessorType accTy,
                            const sycl::ReqdWorkGroupSize &reqdWorkGroupSize) {
  assert(!reqdWorkGroupSize.empty() && "Expecting non-empty reqdWorkGroupSize");
  unsigned elemSize = accTy.getType().getIntOrFloatBitWidth() / 8;
  unsigned memrefReqdLocalMemory = elemSize;
  for (unsigned dim = 0; dim < accTy.getDimension(); ++dim) {
    memrefReqdLocalMemory *= reqdWorkGroupSize[dim];
  }
  return memrefReqdLocalMemory;
}

/// Get the require local memory for memrefs in \p loopToSharedMemref, i.e, for
/// each kernel. If there are multiple loops in the kernel that require local
/// memory, it returns the maximum amount required by any of them.
Optional<unsigned>
getReqdLocalMemory(const DenseMap<LoopLikeOpInterface, std::set<Operation *>>
                       &loopToSharedMemref,
                   const sycl::ReqdWorkGroupSize &reqdWorkGroupSize) {
  if (reqdWorkGroupSize.empty())
    return std::nullopt;

  unsigned reqdLocalMemory = 0;
  for (auto &entry : loopToSharedMemref) {
    unsigned loopReqdLocalMemory = 0;
    for (Operation *memref : entry.second) {
      sycl::AccessorType accTy =
          getAccessorType(cast<sycl::SYCLAccessorSubscriptOp>(memref));
      loopReqdLocalMemory += getReqdLocalMemory(accTy, reqdWorkGroupSize);
    }
    // Memref in one loop can reuse local memory allocated for another loop.
    reqdLocalMemory = std::max(reqdLocalMemory, loopReqdLocalMemory);
  }
  return reqdLocalMemory;
}

/// Create a GlobalOp for workgroup local memory.
memref::GlobalOp getWorkGroupLocalMemory(gpu::GPUModuleOp module,
                                         unsigned size) {
  assert(size != 0 && "Expecting non-zero size");
  std::string name("WGLocalMem");
  polygeist::getUniqueSymbolName(name, module);
  OpBuilder globalBuilder(module->getRegion(0));
  auto memRefTy = MemRefType::get(
      size, globalBuilder.getI8Type(), MemRefLayoutAttrInterface(),
      sycl::AccessAddrSpaceAttr::get(globalBuilder.getContext(),
                                     sycl::AccessAddrSpace::LocalAccess));
  return globalBuilder.create<memref::GlobalOp>(
      module->getLoc(), name,
      /*sym_visibility=*/globalBuilder.getStringAttr("private"), memRefTy,
      /*initial_value=*/Attribute(), /*constant=*/false,
      /*alignment=*/IntegerAttr());
}

/// Create ViewOp from source \p source, with offset \p offset, for AccessorType
/// \p accTy.
memref::ViewOp createViewOp(sycl::AccessorType accTy, Value offset,
                            memref::GetGlobalOp source,
                            const WorkGroupSize &workGroupSize,
                            OpBuilder builder, Location loc) {
  SmallVector<int64_t> shape;
  SmallVector<Value> sizes;
  for (int dim = accTy.getDimension() - 1; dim >= 0; --dim) {
    std::variant<Value, unsigned> size = workGroupSize[dim];
    if (workGroupSize.hasElemTy<unsigned>())
      shape.push_back(std::get<unsigned>(workGroupSize[dim]));
    else {
      shape.push_back(ShapedType::kDynamic);
      sizes.push_back(std::get<Value>(size));
    }
  }
  auto resTy = MemRefType::get(
      shape, accTy.getType(), MemRefLayoutAttrInterface(),
      sycl::AccessAddrSpaceAttr::get(builder.getContext(),
                                     sycl::AccessAddrSpace::LocalAccess));
  return builder.create<memref::ViewOp>(loc, resTy, source, offset, sizes);
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

/// Return true if \p op potentially writes the same memory as \p memRefAccess.
bool mayConflictWithWrite(affine::MemRefAccess memRefAccess, Operation *op,
                          AliasAnalysis &AA) {
  if (op == memRefAccess.opInst || isMemoryEffectFree(op))
    return false;

  // Conservatively assume operations with unknown memory effects may
  // conflict.
  if (!isa<MemoryEffectOpInterface>(op) &&
      !op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
    return true;

  if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance, 1> effects;
    memEffect.getEffects(effects);

    return any_of(effects, [&](const MemoryEffects::EffectInstance &EI) {
      if (isa<MemoryEffects::Read>(EI.getEffect()))
        return false;

      AliasResult aliasRes = AA.alias(EI.getValue(), memRefAccess.memref);
      return !aliasRes.isNo();
    });
  }

  return false;
}

/// Return true if any operation in \p loop potentially writes the same memory
/// as \p memRefAccess.
bool mayConflictWithWriteInLoop(affine::MemRefAccess memRefAccess,
                                LoopLikeOpInterface loop, AliasAnalysis &AA) {
  WalkResult walkResult = loop->walk([&](Operation *op) {
    if (mayConflictWithWrite(memRefAccess, op, AA)) {
      LLVM_DEBUG(llvm::dbgs() << "Found conflict between " << *op << " and "
                              << *memRefAccess.opInst << "\n");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return walkResult.wasInterrupted();
}

//===----------------------------------------------------------------------===//
// MemorySelector
//===----------------------------------------------------------------------===//

/// Collect memory accesses in a loop and determine the memory space each
/// access should ideally use.
class MemorySelector {
public:
  MemorySelector(MemoryAccessAnalysis &memAccessAnalysis,
                 AliasAnalysis &aliasAnalysis, DataFlowSolver &solver)
      : memAccessAnalysis(memAccessAnalysis), aliasAnalysis(aliasAnalysis),
        solver(solver) {}

  /// The kind of accesses to consider.
  enum class AccessKind { ReadOnly, WriteOnly, ReadWrite };

  /// Enumerate memory spaces.
  enum class MemorySpace { Global, Shared, Constant, Texture };

  /// Return the most suitable memory space the \p memref should use.
  std::optional<MemorySpace> getMemorySpace(Value memref) const;

  /// Analyze the memory accesses in the given loop.
  void analyze(LoopLikeOpInterface loop, AccessKind accessKind);

private:
  /// Return the ideal memory space for \p memAccess.
  MemorySpace selectMemorySpace(const MemoryAccess &memAccess,
                                ArrayRef<Value> threadVars,
                                AccessKind accessKind) const;

  /// Return true iff no memref accesses in \p accesses are stores.
  bool areReadOnly(ArrayRef<affine::MemRefAccess> memRefAccesses,
                   LoopLikeOpInterface loop);

private:
  MemoryAccessAnalysis &memAccessAnalysis;
  AliasAnalysis &aliasAnalysis;
  DataFlowSolver &solver;

  /// The preferred memory space for each memref access.
  DenseMap<Value, MemorySpace> memRefAccessToMemSpace;
};

std::optional<MemorySelector::MemorySpace>
MemorySelector::getMemorySpace(Value memref) const {
  assert(isa<MemRefType>(memref.getType()) && "Expecting a memref");

  auto it = memRefAccessToMemSpace.find(memref);
  if (it == memRefAccessToMemSpace.end())
    return std::nullopt;
  return it->second;
}

void MemorySelector::analyze(LoopLikeOpInterface loop, AccessKind accessKind) {
  // Collect the global thread ids used in the function the loop is in.
  auto funcOp = loop->template getParentOfType<FunctionOpInterface>();
  unsigned gridDim = memAccessAnalysis.getGridDimension(funcOp);
  SmallVector<Value> threadVars =
      memAccessAnalysis.getThreadVector(funcOp, solver);

  // Collect candidate memref accesses in the loop.
  DenseMap<Value, SmallVector<affine::MemRefAccess>> memRefToMemRefAccesses;
  loop->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<affine::AffineLoadOp, affine::AffineStoreOp>(op))
      return;

    affine::MemRefAccess memRefAccess(op);
    memRefToMemRefAccesses[memRefAccess.memref].push_back(memRefAccess);
  });

  // Analyze the memref accesses collected and populate the map.
  for (auto &entry : memRefToMemRefAccesses) {
    Value memRef = entry.first;
    ArrayRef<affine::MemRefAccess> memRefAccesses = entry.second;

    // If interested in read-only memref accesses, ensure none of them is a
    // store or aliases a write operation in the loop.
    if (accessKind == AccessKind::ReadOnly &&
        !areReadOnly(memRefAccesses, loop))
      continue;

    // Note: all our candidate memref accesses have the same subscript and
    // zero index therefore we need to analyze the first one only.
    const affine::MemRefAccess &memRefAccess = memRefAccesses.front();
    LLVM_DEBUG(llvm::dbgs() << "Classify: " << *memRefAccess.opInst << "\n");

    std::optional<MemoryAccess> memAccess =
        memAccessAnalysis.getMemoryAccess(memRefAccess);
    if (!memAccess.has_value()) {
      LLVM_DEBUG(llvm::dbgs() << "Unable to analyze memref access\n");
      continue;
    }

    memRefAccessToMemSpace[memRef] =
        selectMemorySpace(*memAccess, threadVars, accessKind);

    LLVM_DEBUG({
      if (memRefAccessToMemSpace.at(memRef) == MemorySpace::Shared)
        llvm::dbgs().indent(2) << "shared memory space\n";
      else {
        assert(memRefAccessToMemSpace.at(memRef) == MemorySpace::Global);
        llvm::dbgs().indent(2) << "global memory space\n";
      }
    });
  }
}

MemorySelector::MemorySpace
MemorySelector::selectMemorySpace(const MemoryAccess &memAccess,
                                  ArrayRef<Value> threadVars,
                                  AccessKind accessKind) const {
  const unsigned gridDimension = threadVars.size();

  // Whether the memory access is (partially) coalesced or not.
  auto isCoalesced = [&](const MemoryAccess &memAccess) {
    MemoryAccessMatrix interThreadMatrix =
        memAccess.getInterThreadAccessMatrix(gridDimension);
    MemoryAccessPattern interThreadAccessPattern =
        MemoryAccess::classify(interThreadMatrix, memAccess.getOffsetVector());

    switch (interThreadAccessPattern) {
    case Linear:
    case Reverse:
    case ReverseLinear:
      // These patterns imply fully coalesced memory accesses.
      return true;
    case Shifted:
    case LinearShifted:
    case ReverseLinearShifted:
    case LinearOverlapped:
    case ReverseLinearOverlapped:
      // These patterns imply partially coalesced memory accesses.
      return true;
    default:
      return false;
    }
  };

  // A non-zero intra-thread access matrix implies several threads access
  // the same array element (in a loop).
  bool hasTemporalReuse =
      !memAccess.getIntraThreadAccessMatrix(gridDimension).isZero();

  MemorySpace memSpace = MemorySpace::Global;
  switch (accessKind) {
  case AccessKind::ReadOnly:
    if (hasTemporalReuse)
      memSpace = MemorySpace::Shared;
    else if (!hasTemporalReuse && isCoalesced(memAccess))
      memSpace = MemorySpace::Global;
    else
      memSpace = MemorySpace::Texture;
    break;
  case AccessKind::ReadWrite:
  case AccessKind::WriteOnly:
    memSpace = hasTemporalReuse ? MemorySpace::Shared : MemorySpace::Global;
    break;
  }

  return memSpace;
}

bool MemorySelector::areReadOnly(ArrayRef<affine::MemRefAccess> memRefAccesses,
                                 LoopLikeOpInterface loop) {
  return llvm::none_of(
      memRefAccesses, [&](const affine::MemRefAccess &memRefAccess) {
        return memRefAccess.isStore() ||
               mayConflictWithWriteInLoop(memRefAccess, loop, aliasAnalysis);
      });
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
                         MemoryAccessAnalysis &memAccessAnalysis,
                         AliasAnalysis &aliasAnalysis, DataFlowSolver &solver);

  /// Determine the tile size for \p loop.
  Value getTileSize(LoopLikeOpInterface loop) const;

  /// Transform a candidate loop.
  template <typename T>
  void transform(T loop, memref::GlobalOp workGroupLocalMemory,
                 const WorkGroupSize &workGroupSize);

private:
  /// A map from a candidate loop to shared memref values used in the loop.
  DenseMap<LoopLikeOpInterface, std::set<Operation *>> loopToSharedMemref;
};

void LoopInternalization::runOnOperation() {
  Operation *module = getOperation();
  ModuleAnalysisManager mam(module, /*passInstrumentor=*/nullptr);
  AnalysisManager am = mam;
  auto &memAccessAnalysis =
      am.getAnalysis<MemoryAccessAnalysis>().initialize(relaxedAliasing);
  AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
  aliasAnalysis.addAnalysisImplementation(sycl::AliasAnalysis(relaxedAliasing));
  auto gpuModule = dyn_cast<gpu::GPUModuleOp>(
      module->getRegion(0).front().getOperations().front());
  if (!gpuModule)
    return;

  FunctionKernelInfo funcKernelInfo(gpuModule);

  // Collect kernel body functions of candidate kernels.
  std::set<FunctionOpInterface> kernelBodyFuncs;
  std::set<FunctionOpInterface> toRemove;
  gpuModule->walk([&](gpu::GPUFuncOp kernel) {
    if (!kernel.isKernel())
      return;
    FunctionOpInterface func = funcKernelInfo.getKernelBodyFunc(kernel);
    if (isCandidateKernel(kernel))
      kernelBodyFuncs.insert(func);
    else
      toRemove.insert(func);
  });
  // If func is a kernel body function of a non-candidate kernel, then remove
  // it from the set.
  for (FunctionOpInterface func : toRemove)
    kernelBodyFuncs.erase(func);

  // Walk each kernel body functions.
  for (FunctionOpInterface func : kernelBodyFuncs) {
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

    // Ensure there is local memory to be used.
    unsigned localMemoryRemain =
        getLocalMemoryRemain(gpuModule, localMemorySize);
    if (localMemoryRemain == 0)
      return;

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
      selectMemorySpace(*innermostLoop, memAccessAnalysis, aliasAnalysis,
                        solver);

      // TODO: prioritize the array accesses that should use shared memory.
      // prioritize(memAccessAnalysis, solver);
    });

    // Calculate the required local memory for all accesses in
    // loopToSharedMemref to be promoted.
    SmallVector<gpu::GPUFuncOp> kernels;
    funcKernelInfo.getKernelCallers(func, kernels);
    sycl::ReqdWorkGroupSize reqdWorkGroupSize(kernels);
    Optional<unsigned> reqdLocalMemory =
        getReqdLocalMemory(loopToSharedMemref, reqdWorkGroupSize);
    if (reqdLocalMemory.has_value() && *reqdLocalMemory > localMemoryRemain) {
      LLVM_DEBUG(llvm::dbgs() << "Not enough local memory\n");
      return;
    }

    memref::GlobalOp workGroupLocalMemory = getWorkGroupLocalMemory(
        gpuModule,
        reqdLocalMemory.has_value() ? *reqdLocalMemory : localMemoryRemain);

    // Get or create work group size.
    Value ndItem = getNdItemArgument(func);
    OpBuilder builder(func->getRegion(0));
    WorkGroupSize workGroupSize(ndItem, reqdWorkGroupSize, builder);

    // Now that we have a list of memref to promote to shared memory in each
    // loop nest's innermost loop, perform the transformation.
    for (auto &entry : loopToSharedMemref) {
      TypeSwitch<Operation *>(entry.first)
          .Case<affine::AffineForOp, scf::ForOp>([&](auto loop) {
            transform(loop, workGroupLocalMemory, workGroupSize);
          });
    }
    loopToSharedMemref.clear();
  }
}

void LoopInternalization::selectMemorySpace(
    LoopLikeOpInterface loop, MemoryAccessAnalysis &memAccessAnalysis,
    AliasAnalysis &aliasAnalysis, DataFlowSolver &solver) {
  assert(LoopTools::getInnermostLoop(loop) && "Expecting an innermost loop");
  assert(loopToSharedMemref.find(loop) == loopToSharedMemref.end() &&
         "The loop should not be already present in the map");

  // Use the memory selector to determine the ideal memory space for memref
  // accesses in the innermost loop.
  // TODO: allow memory selection on read-write accesses.
  MemorySelector memorySelector(memAccessAnalysis, aliasAnalysis, solver);
  memorySelector.analyze(loop, MemorySelector::AccessKind::ReadOnly);

  loop->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<affine::AffineLoadOp, affine::AffineStoreOp>(op))
      return;

    // Limitation: cannot calculate element size for non int or float type.
    if (!op->getResultTypes()[0].isIntOrFloat())
      return;

    affine::MemRefAccess memRefAccess(op);

    // Compute the ideal memory space if possible.
    std::optional<MemorySelector::MemorySpace> memSpace =
        memorySelector.getMemorySpace(memRefAccess.memref);
    if (!memSpace)
      return;

    if (*memSpace == MemorySelector::MemorySpace::Shared)
      loopToSharedMemref[loop].insert(memRefAccess.memref.getDefiningOp());
  });
}

Value LoopInternalization::getTileSize(LoopLikeOpInterface loop) const {
  // TODO: calculate proper tile sizes.
  OpBuilder builder(loop);
  return builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(),
                                                tileSize);
}

template <typename T>
void LoopInternalization::transform(T loop,
                                    memref::GlobalOp workGroupLocalMemory,
                                    const WorkGroupSize &workGroupSize) {
  static_assert(llvm::is_one_of<T, affine::AffineForOp, scf::ForOp>::value);
  assert(LoopTools::isInnermostLoop(loop) && "Expecting an innermost loop");
  assert(loopToSharedMemref.find(loop) != loopToSharedMemref.end() &&
         "Loop should be in the map");

  OpBuilder builder(loop);
  auto getGlobalOp = builder.create<memref::GetGlobalOp>(
      loop.getLoc(), workGroupLocalMemory.getType(),
      workGroupLocalMemory.getName());
  const std::set<Operation *> &memrefs = loopToSharedMemref.at(loop);

  SmallVector<T> tiledNest;
  LogicalResult res = tile(loop, getTileSize(loop), tiledNest);
  assert(res.succeeded() && "Expecting innermost loop to be tiled");
  ++numTiled;
  LLVM_DEBUG(llvm::dbgs() << "Tiled loop: " << tiledNest.front() << "\n");

  loop = tiledNest.front();
  builder.setInsertionPointToStart(loop->getBlock());
  // Get pointer to the local memory portion for each memref.
  std::variant<Value, unsigned> offset =
      ValueOrUnsigned::get(0, builder, workGroupSize.hasElemTy<Value>());
  for (Operation *memref : memrefs) {
    sycl::AccessorType accTy =
        getAccessorType(cast<sycl::SYCLAccessorSubscriptOp>(memref));
    createViewOp(accTy, ValueOrUnsigned::getValue(offset, builder), getGlobalOp,
                 workGroupSize, builder, memref->getLoc());

    // Only increment offset when the current memref is not the last one.
    if (memref == *memrefs.rbegin())
      continue;

    std::variant<Value, unsigned> reqdLocalMemory =
        getReqdLocalMemory(accTy, workGroupSize, builder);
    offset = ValueOrUnsigned::add(offset, reqdLocalMemory, builder);
  }

  // TODO: promote loop accesses to local memory.
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
