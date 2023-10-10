//===- LoopInternalization.cpp - Promote memory access to shared mem -----===//
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
// Before optimization:
//   size_t global_id = nditem.get_global_id(0)
//   for (size_t k = 0; k < N; ++k)
//     A[k];
//
// After optimization (pseudocode):
//   // Obtain shared local memory for all candidate memory accesses in a kernel
//   needed for this optimization.
//   memref.global "private" @WGLocalMem
//                             : memref<-xi8, #sycl.access.address_space<local>>
//   %shared_memory_ptr = memref.get_global @WGLocalMem
//                             : memref<-xi8, #sycl.access.address_space<local>>
//   size_t global_id = nditem.get_global_id(0);
//   %local_id = nditem.get_local_id(0)
//   %work_group_size = nditem.get_local_range(0)
//   // When there are more than one memory access promoted to shared local
//   memory, then %offset needs to incremented by the size of the previous
//   memory access.
//   %offset = 0
//   // Get pointer to the shared local memory portion for 'A'. Notice that
//   %view has %work_group_size number of entries.
//   %view = memref.view %shared_memory_ptr[%offset][%work_group_size]
//                             : memref<-xi8, #sycl.access.address_space<local>>
//                             to memref<?xf32,
//                             #sycl.access.address_space<local>>
//   // Loop is tiled with %work_group_size as the tile size.
//   for (size_t k = 0; k < N; k+= work_group_size) {
//     // Copy from global memory to shared local memory.
//     // A's index needs to be adjusted to outer loop induction variable plus
//     tiled loop lower bound, to get different portion of 'A'.
//     %view[local_id] = A[local_id+k]
//     // The optimization relies on each thread in a work group to initialize
//     %view, so we need a barrier here before reading from %view.
//     group_barrier(nditem.get_group());
//     for (size_t t = k; t < k + work_group_size; ++t)
//       // %view's index needs to be adjusted to tiled loop induction variable
//       minus tiled loop lower bound, as %view starts from zero.
//       %view[t-k]
//     // Before %view get overwritten in the next loop iteration, we need to
//     ensure it is done reading from, so we need another barrier here.
//     group_barrier(nditem.get_group());
//   }
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Polygeist/Analysis/MemoryAccessAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/UniformityAnalysis.h"
#include "mlir/Dialect/Polygeist/Transforms/Passes.h"
#include "mlir/Dialect/Polygeist/Utils/TransformUtils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SYCL/Utils/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <list>
#include <numeric>
#include <type_traits>
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

//===----------------------------------------------------------------------===//
// WorkGroupSize
//===----------------------------------------------------------------------===//

/// Class to represent work group size of a kernel.
class WorkGroupSize {
public:
  using ElemTy = std::variant<Value, unsigned>;

  /// Construct a SYCL work group size object given the rank \p numDims and the
  /// SYCL 'reqd_work_group_size' attribute \p reqdWorkGroupSize.
  WorkGroupSize(unsigned numDims,
                const sycl::ReqdWorkGroupSize &reqdWorkGroupSize,
                OpBuilder builder);

  /// Return true if the work group size is known at compile time and false
  /// otherwise.
  bool isKnown() const {
    return std::holds_alternative<unsigned>(wgSizes.front());
  }

  /// Return the element at position \p dim with type \tparam T.
  template <typename T, typename = std::enable_if_t<
                            llvm::is_one_of<T, Value, unsigned>::value>>
  T get(unsigned dim) const {
    assert(std::holds_alternative<T>(wgSizes.front()) && "Incorrect type");
    assert(dim < wgSizes.size() && "Expecting valid dim");
    return std::get<T>(wgSizes[dim]);
  }

  /// Return the element at position \p dim.
  ElemTy operator[](unsigned dim) const {
    assert(dim < wgSizes.size() && "Expecting valid dim");
    return wgSizes[dim];
  }

private:
  SmallVector<ElemTy> wgSizes;
};

//===----------------------------------------------------------------------===//
// ValueOrUnsigned
//===----------------------------------------------------------------------===//

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

  static std::variant<Value, unsigned>
  max(const std::variant<Value, unsigned> &lhs,
      const std::variant<Value, unsigned> &rhs, OpBuilder builder) {
    return binaryOperator<arith::MaxSIOp>(lhs, rhs, builder);
  }

private:
  template <typename T,
            typename = std::enable_if_t<llvm::is_one_of<
                T, arith::AddIOp, arith::MulIOp, arith::MaxSIOp>::value>>
  static std::variant<Value, unsigned>
  binaryOperator(const std::variant<Value, unsigned> &lhs,
                 const std::variant<Value, unsigned> &rhs, OpBuilder builder) {
    if (std::holds_alternative<unsigned>(lhs)) {
      assert(std::holds_alternative<unsigned>(rhs));
      if constexpr (std::is_same_v<T, arith::AddIOp>)
        return std::get<unsigned>(lhs) + std::get<unsigned>(rhs);
      if constexpr (std::is_same_v<T, arith::MulIOp>)
        return std::get<unsigned>(lhs) * std::get<unsigned>(rhs);
      if constexpr (std::is_same_v<T, arith::MaxSIOp>)
        return std::max(std::get<unsigned>(lhs), std::get<unsigned>(rhs));
    }

    assert(std::holds_alternative<Value>(lhs) &&
           std::holds_alternative<Value>(rhs));
    return builder.create<T>(builder.getUnknownLoc(), std::get<Value>(lhs),
                             std::get<Value>(rhs));
  }
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
  // Available shared memory of a kernel cannot be calculated when dynamic
  // sized shared memory is used, as its size is not compile time known.
  return none_of(kernel.getArguments(), [](Value arg) {
    return isLocalAccessAddrSpace(arg.getType());
  });
}

/// A function is a candidate iff we can get the grid dimension.
bool isCandidateFunction(FunctionOpInterface func) {
  unsigned numGridDim = getGridDimension(func);
  return (numGridDim > 0 && numGridDim <= 3);
}

/// A loop nest is a candidate iff is perfect and contains only affine or scf
/// for loops.
bool isCandidateLoopNest(LoopLikeOpInterface loop, DataFlowSolver &solver) {
  if (!LoopTools::isOutermostLoop(loop))
    return false;

  if (!LoopTools::isPerfectLoopNest(loop))
    return false;

  std::optional<LoopLikeOpInterface> innermostLoop =
      LoopTools::getInnermostLoop(loop);
  assert(innermostLoop.has_value() && "Failed to get the innermost loop");

  if (!isa<affine::AffineForOp, scf::ForOp>(*innermostLoop))
    return false;

  assert(
      innermostLoop->getSingleInductionVar() &&
      "Expecting single induction variable for affine for and scf for loops");

  // Cannot tile loop that yield values.
  Operation *innermostLoopOp = *innermostLoop;
  if (innermostLoopOp->getNumResults() > 0)
    return false;

  // Because the transformation inserts barriers, it requires the loop to be
  // non-divergent.
  return !isDivergent(innermostLoopOp, solver);
}

/// An access is a candidate iff it is AffineLoadOp or AffineStoreOp, with int
/// or float element type, and with accessor dimension the same as grid
/// dimension.
bool isCandidateAccess(Operation *op,
                       const MemoryAccessAnalysis &memAccessAnalysis) {
  assert(op && "Expecting valid op");

  if (!isa<affine::AffineLoadOp, affine::AffineStoreOp>(op))
    return false;

  affine::MemRefAccess memRefAccess(op);
  auto accSub = dyn_cast_or_null<sycl::SYCLAccessorSubscriptOp>(
      memRefAccess.memref.getDefiningOp());
  if (!accSub)
    return false;

  // Limitation: cannot calculate element size for non int or float type.
  if (!cast<MemRefType>(accSub.getType()).getElementType().isIntOrFloat())
    return false;

  // TODO: Add support for accessors with dimensions not equals to the grid
  // dimensions.
  if (auto func = op->getParentOfType<FunctionOpInterface>();
      getGridDimension(func) != getAccessorType(accSub).getDimension())
    return false;

  std::optional<MemoryAccess> memAccess =
      memAccessAnalysis.getMemoryAccess(memRefAccess);
  if (!memAccess.has_value())
    return false;

  // Limitation: Unable to transform memory access with indexes that use more
  // than one innermost loop induction variable or thread ids in the same
  // dimension.
  dataflow::IntegerValueRange zero(ConstantIntRanges::constant(APInt(1, 0)));
  MemoryAccessMatrix matrix = memAccess->getAccessMatrix();
  for (size_t i = 0; i < matrix.getNumRows(); ++i) {
    // Not a candidate when there exists more than one non-zero entry in a
    // row.
    if (count_if(matrix.getRow(i), [&](dataflow::IntegerValueRange range) {
          return !(range == zero);
        }) > 1)
      return false;
  }

  return true;
}

/// Get the size of unused shared memory arena in bytes.
unsigned getSharedMemoryRemaining(gpu::GPUModuleOp &module,
                                  const unsigned sharedMemorySize) {
  assert(module.hasTrait<OpTrait::SymbolTable>() &&
         "Expecting module with SymbolTable trait");

  unsigned sharedMemoryRemaining = sharedMemorySize;
  module.walk([&](memref::GlobalOp global) {
    MemRefType memRefTy = global.getType();
    if (!isLocalAccessAddrSpace(memRefTy))
      return WalkResult::advance();

    unsigned globalSize =
        memRefTy.getElementTypeBitWidth() * memRefTy.getNumElements() / 8;
    if (globalSize >= sharedMemoryRemaining) {
      sharedMemoryRemaining = 0;
      return WalkResult::interrupt();
    }

    sharedMemoryRemaining -= globalSize;
    return WalkResult::advance();
  });

  return sharedMemoryRemaining;
}

/// Get the amount of shared memory needed by \p accTy in bytes.
std::variant<Value, unsigned>
getReqdSharedMemory(sycl::AccessorType accTy,
                    const WorkGroupSize &workGroupSize, OpBuilder builder) {
  unsigned elemSize = accTy.getType().getIntOrFloatBitWidth() / 8;
  std::variant<Value, unsigned> reqdSharedMemory =
      ValueOrUnsigned::get(elemSize, builder, !workGroupSize.isKnown());

  const unsigned numDims = accTy.getDimension();
  for (unsigned dim = 0; dim < numDims; ++dim)
    reqdSharedMemory =
        ValueOrUnsigned::mul(reqdSharedMemory, workGroupSize[dim], builder);

  return reqdSharedMemory;
}

/// Get the required shared local memory for memRefs in \p loopToSharedMemref
/// (i.e, for each kernel). If there are multiple loops in the kernel that
/// require shared memory, return the maximum amount required by any of them.
std::variant<Value, unsigned>
getReqdSharedMemory(const DenseMap<LoopLikeOpInterface, SetVector<Operation *>>
                        &loopToSharedMemref,
                    const WorkGroupSize &workGroupSize, OpBuilder builder) {
  std::variant<Value, unsigned> reqdSharedMemory =
      ValueOrUnsigned::get(0, builder, !workGroupSize.isKnown());

  for (auto &entry : loopToSharedMemref) {
    std::variant<Value, unsigned> loopReqdSharedMemory =
        ValueOrUnsigned::get(0, builder, !workGroupSize.isKnown());
    for (Operation *memref : entry.second) {
      sycl::AccessorType accTy =
          getAccessorType(cast<sycl::SYCLAccessorSubscriptOp>(memref));
      loopReqdSharedMemory = ValueOrUnsigned::add(
          loopReqdSharedMemory,
          getReqdSharedMemory(accTy, workGroupSize, builder), builder);
    }
    // Memref in one loop can reuse shared memory allocated for another loop.
    reqdSharedMemory =
        ValueOrUnsigned::max(reqdSharedMemory, loopReqdSharedMemory, builder);
  }

  return reqdSharedMemory;
}

/// Create a GlobalOp for workgroup shared local memory.
memref::GlobalOp getWorkGroupSharedLocalMemory(gpu::GPUModuleOp module,
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
                            ArrayRef<unsigned> localIDOrdering,
                            const WorkGroupSize &workGroupSize,
                            OpBuilder builder, Location loc) {
  SmallVector<int64_t> shape;
  SmallVector<Value> sizes;
  for (unsigned dim = 0; dim < accTy.getDimension(); ++dim) {
    std::variant<Value, unsigned> size = workGroupSize[localIDOrdering[dim]];
    if (workGroupSize.isKnown())
      shape.push_back(std::get<unsigned>(size));
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
void tile(affine::AffineForOp loop, Value tileSize,
          SmallVectorImpl<affine::AffineForOp> &tiledNest) {
  SmallVector<affine::AffineForOp> newNestedLoops;
  [[maybe_unused]] LogicalResult res =
      tilePerfectlyNestedParametric({loop}, tileSize, &newNestedLoops);
  assert(res.succeeded() && "Expecting innermost loop to be tiled");
  tiledNest = SmallVector<affine::AffineForOp>(newNestedLoops.begin() + 1,
                                               newNestedLoops.end());
}

/// Tile an SCF for \p loop given the tile size \p tileSize.
void tile(scf::ForOp loop, Value tileSize,
          SmallVectorImpl<scf::ForOp> &tiledNest) {
  tiledNest = tile({loop}, tileSize, loop);
}

/// Create a work group barrier.
void createWorkGroupBarrier(OpBuilder &builder) {
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

/// Returns true if the memory access \p access has a single subscript that is
/// zero, and false otherwise.
bool hasZeroIndex(const affine::MemRefAccess &access) {
  affine::AffineValueMap accessValueMap;
  access.getAccessMap(&accessValueMap);
  if (accessValueMap.getNumDims() != 0)
    return false;

  auto index = accessValueMap.getResult(0).dyn_cast<AffineConstantExpr>();
  return (index && index.getValue() == 0);
}

/// Return the underlying sycl.constructor of the operand at index \p opIndex in
/// operation \p op.
sycl::SYCLConstructorOp getUnderlyingSYCLConstructor(unsigned opIndex,
                                                     Operation *op,
                                                     DataFlowSolver &solver) {
  std::optional<Definition> def =
      ReachingDefinition::getUniqueDefinition(opIndex, op, solver);
  if (!def.has_value())
    return nullptr;

  if (auto constructor = dyn_cast<sycl::SYCLConstructorOp>(def->getOperation()))
    return constructor;

  auto storeOp = dyn_cast<affine::AffineStoreOp>(def->getOperation());
  if (!storeOp)
    return nullptr;

  // Memory accesses involving SYCL accessors should have zero index.
  affine::MemRefAccess storeAccess(storeOp);
  assert(hasZeroIndex(storeAccess) && "Unexpected candidate operation");

  Value storedVal = storeOp.getOperand(storeOp.getStoredValOperandIndex());
  if (!isa<affine::AffineLoadOp>(storedVal.getDefiningOp()))
    return nullptr;

  // Try to determine the underlying value of the memory pointed to by
  // the memref operand of a load.
  auto loadOp = dyn_cast<affine::AffineLoadOp>(storedVal.getDefiningOp());
  if (!loadOp)
    return nullptr;

  affine::MemRefAccess loadAccess(loadOp);
  assert(hasZeroIndex(storeAccess) && "Unexpected candidate operation");

  return getUnderlyingSYCLConstructor(loadOp.getMemRefOperandIndex(), loadOp,
                                      solver);
}

/// Return the indexes used in \p accSub through sycl.constructor.
SmallVector<Value> getIndexes(sycl::SYCLAccessorSubscriptOp accSub,
                              DataFlowSolver &solver) {
  sycl::SYCLConstructorOp constructorOp = getUnderlyingSYCLConstructor(
      accSub.getOffsetOperandIndex(), accSub, solver);
  assert(constructorOp && "Expecting constructor definition");

  SmallVector<Value> indexes;
  for (unsigned i = 0; i < constructorOp->getNumOperands(); ++i) {
    if (i == constructorOp.getOutputOperandIndex())
      continue;
    indexes.push_back(constructorOp->getOperand(i));
  }

  return indexes;
}

/// Determine whether \p user uses \p use (directly or indirectly).
bool usesValue(Value user, Value use) {
  assert(user && use && "Expecting valid user and use");
  if (user == use)
    return true;

  Operation *op = user.getDefiningOp();
  if (!op)
    return false;

  return llvm::any_of(op->getOperands(),
                      [&](Value operand) { return usesValue(operand, use); });
}

/// Return the local id ordering for \p accSub to index the shared local memory
/// buffer created for \p accSub by this optimization. The algorithm is to use
/// the local id associated with the global id used by a dimension. For indexes
/// that do not reference global id, use unused local id in increasing order.
/// For example, assuming tx and ty are the global id of dimension 0 and
/// dimension 1,
///   A[ty][i], global id 1 is used as the first index,
///   so local id 1 is used as the first index, and remaining local id 0 is used
///   as the second index. The result is [1, 0].
///   A[tx][i], global id 0 is used as the first index,
///   so local id 0 is used as the first index, and remaining local id 1 is used
///   as the second index. The result is [0, 1].
///   A[i][tx], global id 0 is used as the second index,
///   so local id 0 is used as the second index, and remaining local id 1 is
///   used as the first index. The result is [1, 0]. A[i][j], global ids are not
///   used, so local id are used in increasing order. The result is [0, 1].
SmallVector<unsigned> getLocalIDOrdering(sycl::SYCLAccessorSubscriptOp accSub,
                                         DataFlowSolver &solver) {
  const SmallVector<Value> indexes = getIndexes(accSub, solver);
  auto func = accSub->getParentOfType<FunctionOpInterface>();
  const SmallVector<Value> globalIDs = polygeist::getThreadVector(func, solver);
  assert(globalIDs.size() == indexes.size() &&
         "Expecting number of global id the same as number of indexes");

  SmallVector<std::optional<unsigned>> localIDOrdering(globalIDs.size(),
                                                       std::nullopt);
  llvm::SmallSet<int, 3> unusedIndexes;
  for (unsigned globalDim = 0; globalDim < globalIDs.size(); ++globalDim) {
    std::list<unsigned> dimensions(indexes.size());
    std::iota(dimensions.begin(), dimensions.end(), 0);
    // Insert the dimension of unused local id in unusedIndexes.
    if (llvm::none_of(dimensions, [&](unsigned dim) {
          if (globalIDs[globalDim] &&
              usesValue(indexes[dim], globalIDs[globalDim])) {
            localIDOrdering[dim] = globalDim;
            return true;
          }
          return false;
        }))
      unusedIndexes.insert(globalDim);
  }

  SmallVector<unsigned> result;
  for (std::optional<unsigned> index : localIDOrdering) {
    if (index.has_value()) {
      result.push_back(*index);
      continue;
    }
    // Use unused local id for indexes that do not reference global id.
    int unusedIndex = *unusedIndexes.begin();
    result.push_back(unusedIndex);
    unusedIndexes.erase(unusedIndex);
  }
  return result;
}

/// Return Value of \p val with index type.
Value castToIndex(Value val) {
  if (val.getType().isIndex())
    return val;

  // To avoid generating unnecessary arith.index_cast:
  if (Operation *op = val.getDefiningOp()) {
    // When op is `%outVal = arith.index_cast %inVal : index to i64`, return
    // `%inVal`.
    if (auto cast = dyn_cast<arith::IndexCastOp>(op)) {
      Value inVal = cast.getIn();
      if (inVal.getType().isIndex())
        return inVal;
    }

    // When next operation of op is `%outVal = arith.index_cast %inVal : i64 to
    // index` and `%inVal` is val, return that operation.
    if (auto cast = dyn_cast<arith::IndexCastOp>(op->getNextNode())) {
      Value inVal = cast.getIn();
      Value outVal = cast.getOut();
      if (inVal == val && outVal.getType().isIndex())
        return outVal;
    }
  }

  OpBuilder builder(val.getContext());
  builder.setInsertionPointAfterValue(val);
  return builder.create<arith::IndexCastOp>(val.getLoc(),
                                            builder.getIndexType(), val);
}

/// Unroll the given \p loop by \p factor.
void unrollByFactor(LoopLikeOpInterface loop, unsigned factor) {
  if (factor == 1)
    return;

  // Lowering affine for loop to scf for loop, as not all affine loops can be
  // unrolled. Some affine loops (e.g., min expression upper bound) cannot be
  // unrolled, as the lower bound of the residual loop cannot be expressed as an
  // affine function.
  if (Operation *loopOp = loop;
      auto affineForOp = dyn_cast<affine::AffineForOp>(loopOp)) {
    OpBuilder builder(affineForOp);
    Location loc = affineForOp.getLoc();
    Value lowerBound = lowerAffineLowerBound(affineForOp, builder);
    Value upperBound = lowerAffineUpperBound(affineForOp, builder);
    Value step =
        builder.create<arith::ConstantIndexOp>(loc, affineForOp.getStep());
    auto scfForOp = builder.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, affineForOp.getInits());

    // Replace scfForOp body with affineForOp body.
    scfForOp.getBody()->getParent()->getBlocks().remove(scfForOp.getBody());
    scfForOp.getRegion().getBlocks().splice(
        scfForOp.getRegion().end(), affineForOp.getRegion().getBlocks());

    // Replace affine::AffineYieldOp with scf::YieldOp.
    Operation *yieldOp = scfForOp.getRegion().front().getTerminator();
    assert(isa<affine::AffineYieldOp>(yieldOp) &&
           "Expecting affine.yield operation");
    OpBuilder yieldBuilder(yieldOp);
    yieldBuilder.create<scf::YieldOp>(yieldOp->getLoc(),
                                      yieldOp->getOperands());
    yieldOp->erase();

    affineForOp->replaceAllUsesWith(scfForOp);
    affineForOp->erase();
    loop = scfForOp;
  }

  [[maybe_unused]] LogicalResult res =
      loopUnrollByFactor(cast<scf::ForOp>(loop), factor);
  assert(res.succeeded() && "Expecting tiled loop to be unrolled");
}

//===----------------------------------------------------------------------===//
// WorkGroupSize
//===----------------------------------------------------------------------===//

WorkGroupSize::WorkGroupSize(unsigned numDims,
                             const sycl::ReqdWorkGroupSize &reqdWorkGroupSize,
                             OpBuilder builder) {
  assert((reqdWorkGroupSize.empty() || reqdWorkGroupSize.size() == numDims) &&
         "Expecting 'reqdWorkGroupSize' to have same number of elements as "
         "'numDims'");

  // Use constants provided by reqd_work_group_size if given (non-empty).
  if (!reqdWorkGroupSize.empty()) {
    for (unsigned dim = 0; dim < numDims; ++dim)
      wgSizes.push_back(reqdWorkGroupSize[dim]);
    return;
  }

  SmallVector<Value> wgSizeVals;
  sycl::populateWorkGroupSize(wgSizeVals, numDims, builder,
                              builder.getUnknownLoc());
  for (Value wgSize : wgSizeVals)
    wgSizes.push_back(wgSize);
}

//===----------------------------------------------------------------------===//
// LoopInfo
//===----------------------------------------------------------------------===//

/// Class to compute and store loop information.
class LoopInfo {
public:
  LoopInfo(LoopLikeOpInterface loop) : loop(loop) {
    assert(loop.getSingleInductionVar().has_value() &&
           "Expecting single induction variable");
    inductionVar = *loop.getSingleInductionVar();
  }

  LoopLikeOpInterface getLoop() const { return loop; }

  Value getInductionVar() const { return inductionVar; }

  Value getLowerBound() {
    if (lowerBound)
      return lowerBound;
    OpBuilder builder(loop);
    lowerBound = TypeSwitch<Operation *, Value>(loop)
                     .Case<scf::ForOp>(
                         [&](auto scfLoop) { return scfLoop.getLowerBound(); })
                     .Case<affine::AffineForOp>([&](auto affineLoop) {
                       SmallVector<Value, 4> lbOperands(
                           affineLoop.getLowerBoundOperands());
                       return builder.create<affine::AffineApplyOp>(
                           affineLoop.getLoc(), affineLoop.getLowerBoundMap(),
                           lbOperands);
                     });
    return lowerBound;
  }

  Value getAdjustedIV() {
    if (adjustedIV)
      return adjustedIV;
    Value inductionVar = *loop.getSingleInductionVar();
    OpBuilder builder(loop.getContext());
    builder.setInsertionPointAfterValue(inductionVar);
    adjustedIV = builder.create<arith::SubIOp>(inductionVar.getLoc(),
                                               inductionVar, getLowerBound());
    return adjustedIV;
  }

private:
  LoopLikeOpInterface loop;
  Value inductionVar = nullptr;
  Value lowerBound = nullptr;
  Value adjustedIV = nullptr;
};

//===----------------------------------------------------------------------===//
// MemorySelector
//===----------------------------------------------------------------------===//

/// Collect memory accesses in a loop and determine the memory space each
/// access should ideally use.
class MemorySelector {

public:
  MemorySelector(MemoryAccessAnalysis &memAccessAnalysis,
                 AliasAnalysis &aliasAnalysis)
      : memAccessAnalysis(memAccessAnalysis), aliasAnalysis(aliasAnalysis) {}

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
                                const unsigned gridDimension,
                                AccessKind accessKind) const;

  /// Return true iff no memref accesses in \p accesses are stores.
  bool areReadOnly(ArrayRef<affine::MemRefAccess> memRefAccesses,
                   LoopLikeOpInterface loop);

private:
  MemoryAccessAnalysis &memAccessAnalysis;
  AliasAnalysis &aliasAnalysis;

  /// The preferred memory space for each memref access.
  DenseMap<Value, MemorySpace> memRefAccessToMemSpace;
};

[[maybe_unused]] raw_ostream &
operator<<(raw_ostream &os, const MemorySelector::AccessKind &accessKind) {
  switch (accessKind) {
  case MemorySelector::AccessKind::ReadOnly:
    return os << "read-only";
  case MemorySelector::AccessKind::WriteOnly:
    return os << "write-only";
  case MemorySelector::AccessKind::ReadWrite:
    return os << "read-write";
  }
  llvm_unreachable("Unexpected MemorySelector::AccessKind");
}

[[maybe_unused]] raw_ostream &
operator<<(raw_ostream &os, const MemorySelector::MemorySpace &memSpace) {
  switch (memSpace) {
  case MemorySelector::MemorySpace::Global:
    return os << "global";
  case MemorySelector::MemorySpace::Shared:
    return os << "shared";
  case MemorySelector::MemorySpace::Constant:
    return os << "constant";
  case MemorySelector::MemorySpace::Texture:
    return os << "texture";
  }
  llvm_unreachable("Unexpected MemorySelector::MemorySpace");
}

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
  const unsigned gridDimension = getGridDimension(funcOp);

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
      LLVM_DEBUG(llvm::dbgs()
                 << "Could not obtain a valid memory access analysis result");
      continue;
    }

    memRefAccessToMemSpace[memRef] =
        selectMemorySpace(*memAccess, gridDimension, accessKind);

    LLVM_DEBUG(llvm::dbgs().indent(2)
               << memRefAccessToMemSpace.at(memRef) << " memory space\n");
  }
}

MemorySelector::MemorySpace
MemorySelector::selectMemorySpace(const MemoryAccess &memAccess,
                                  const unsigned gridDimension,
                                  AccessKind accessKind) const {
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
    else if (isCoalesced(memAccess))
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
  void runOnGPUModule(gpu::GPUModuleOp gpuModule);

  /// Construct a map from memref accesses in \p loop to their ideal memory
  /// space.
  void selectMemorySpace(LoopLikeOpInterface loop,
                         MemoryAccessAnalysis &memAccessAnalysis,
                         AliasAnalysis &aliasAnalysis, DataFlowSolver &solver);

  /// Analyze the memory accesses that should use shared memory and aggregate
  /// which local ids for dimensions that use the loop induction variable.
  /// For example given (assume 'i' is the loop IV):
  ///   A[tx][i] => local id of dimension 1 is used for second index.
  ///   A[i][tx] => local id of dimension 1 is used for first index.
  ///   A[i][ty] => local id of dimension 0 is used for first index.
  ///   A[ty][i] => local id of dimension 0 is used for second index.
  /// The set collected for A[i][tx] and B[ty][i] would contain {0,1}.
  std::set<unsigned> getLocalIDDimForLoopIV(LoopLikeOpInterface loop,
                                            DataFlowSolver &solver) const;

  // If the memory accesses do not use the loop induction variable
  // 'consistently' (that is on the same dimension), create a versioning
  // condition:
  ///   (wgSize[dim1] == wgSize[dim2]) && (wgSize[dim1] == wgSize[dim3]) ...
  Value checkForConsistentUseOfLoopIV(LoopLikeOpInterface loop,
                                      const WorkGroupSize &workGroupSize,
                                      DataFlowSolver &solver) const;

  /// Generate the versioning condition if the loop should be versioned. Return
  /// nullptr if the loop doesn't need to be versioned.
  Value getVersionCondition(LoopLikeOpInterface loop,
                            const WorkGroupSize &workGroupSize,
                            std::variant<Value, unsigned> reqdSharedMemory,
                            const unsigned sharedMemoryRemaining,
                            DataFlowSolver &solver) const;

  /// Return false if it can be determined at compile time that \p loop cannot
  /// be transformed, and true otherwise.
  bool canBeTransformed(LoopLikeOpInterface loop,
                        const WorkGroupSize &workGroupSize,
                        DataFlowSolver &solver) const;

  /// Determine the tile size for \p loop given \p workGroupSize.
  /// Note: currently the tile size is equal to the work group size.
  Value getTileSize(LoopLikeOpInterface loop,
                    const WorkGroupSize &workGroupSize,
                    DataFlowSolver &solver) const;

  /// Transform a candidate kernel body function.
  void transform(FunctionOpInterface func,
                 const FunctionKernelInfo &funcKernelInfo,
                 const unsigned sharedMemoryRemaining, DataFlowSolver &solver);

  /// Transform a candidate loop.
  template <typename T>
  void transform(T loop, memref::GlobalOp wgSharedLocalMemory,
                 ArrayRef<Value> localIDs, const WorkGroupSize &workGroupSize,
                 std::variant<Value, unsigned> reqdSharedMemory,
                 const unsigned sharedMemoryRemaining, DataFlowSolver &solver);

  // Promote memory accesses identified by \p memref to shared memory, by
  // using a the 'view' operation \p viewOp.
  void promote(Operation *memref, memref::ViewOp viewOp, LoopInfo &loopInfo,
               ArrayRef<Value> localIDs, OpBuilder &builder,
               DataFlowSolver &solver) const;

private:
  /// A map from a candidate loop to shared memref values used in the loop.
  DenseMap<LoopLikeOpInterface, SetVector<Operation *>> loopToSharedMemref;
};

void LoopInternalization::runOnOperation() {
  getOperation()->walk(
      [&](gpu::GPUModuleOp gpuModule) { runOnGPUModule(gpuModule); });
}

void LoopInternalization::runOnGPUModule(gpu::GPUModuleOp gpuModule) {
  ModuleAnalysisManager mam(gpuModule, /*passInstrumentor=*/nullptr);
  AnalysisManager am = mam;
  auto &memAccessAnalysis =
      am.getAnalysis<MemoryAccessAnalysis>().initialize<sycl::AliasAnalysis>(
          relaxedAliasing);
  AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
  aliasAnalysis.addAnalysisImplementation(sycl::AliasAnalysis(relaxedAliasing));

  DataFlowSolverWrapper solver(aliasAnalysis);
  solver.load<dataflow::IntegerRangeAnalysis>();
  solver.loadWithRequiredAnalysis<UniformityAnalysis>();

  if (failed(solver.initializeAndRun(gpuModule))) {
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE ": Unable to run required dataflow analysis\n");
    return;
  }

  FunctionKernelInfo funcKernelInfo(gpuModule);

  // Collect kernel body functions of candidate kernels.
  std::set<FunctionOpInterface> kernelBodyFuncs;
  gpuModule->walk([&](gpu::GPUFuncOp kernel) {
    if (!kernel.isKernel() || !isCandidateKernel(kernel))
      return;

    llvm::SmallSet<FunctionOpInterface, 4> bodyFuncs =
        funcKernelInfo.getPotentialKernelBodyFunctions(kernel);
    kernelBodyFuncs.insert(bodyFuncs.begin(), bodyFuncs.end());
  });

  // Transform each kernel body function.
  for (FunctionOpInterface func : kernelBodyFuncs) {
    if (!isCandidateFunction(func))
      continue;

    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE ": Visiting candidate function "
                            << func.getName() << "\n");

    // Ensure there is shared memory available.
    unsigned sharedMemoryRemaining =
        getSharedMemoryRemaining(gpuModule, sharedMemorySize);
    if (sharedMemoryRemaining == 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << DEBUG_TYPE ": Not enough shared memory available\n");
      return;
    }

    // Select the ideal memory space for memref accesses in candidate loops
    // contained by this function.
    func->walk<WalkOrder::PreOrder>([&](LoopLikeOpInterface loop) {
      if (!isCandidateLoopNest(loop, solver))
        return;

      LLVM_DEBUG(llvm::dbgs()
                 << DEBUG_TYPE ": Visiting candidate loop nest rooted by:\n"
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

    if (loopToSharedMemref.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No candidate memory accesses found\n");
      continue;
    }

    transform(func, funcKernelInfo, sharedMemoryRemaining, solver);
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
  MemorySelector memorySelector(memAccessAnalysis, aliasAnalysis);
  memorySelector.analyze(loop, MemorySelector::AccessKind::ReadOnly);

  loop->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isCandidateAccess(op, memAccessAnalysis))
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

std::set<unsigned>
LoopInternalization::getLocalIDDimForLoopIV(LoopLikeOpInterface loop,
                                            DataFlowSolver &solver) const {
  const Value indVar = *loop.getSingleInductionVar();
  std::set<unsigned> dimsThatUseLoopIV;

  for (Operation *memref : loopToSharedMemref.at(loop)) {
    auto accSub = cast<sycl::SYCLAccessorSubscriptOp>(memref);
    const SmallVector<Value> indexes = getIndexes(accSub, solver);
    SmallVector<unsigned> localIDOrdering = getLocalIDOrdering(accSub, solver);

    for (unsigned dim = 0; dim < indexes.size(); ++dim) {
      if (usesValue(indexes[dim], indVar))
        dimsThatUseLoopIV.insert(localIDOrdering[dim]);
    }
  }

  return dimsThatUseLoopIV;
}

Value LoopInternalization::checkForConsistentUseOfLoopIV(
    LoopLikeOpInterface loop, const WorkGroupSize &workGroupSize,
    DataFlowSolver &solver) const {
  // Find out which dimensions use the loop IV in the memory accesses that
  // should use shared memory. For example given (assume 'i' is the loop IV):
  //   A[tx][ty][i], B[tx][i][ty]
  // The set collected would contain {0,1}.
  std::set<unsigned> dimsThatUseLoopIV = getLocalIDDimForLoopIV(loop, solver);
  if (dimsThatUseLoopIV.size() <= 1)
    return nullptr;

  // If the memory accesses do not use the loop induction variable
  // 'consistently' (that is on the same dimension) we need to version the
  // loop. Build the versioning condition as:
  //   (wgSize[dim1] == wgSize[dim2]) && wgSize[dim1] == wgSize[dim3] && ...)
  auto it = dimsThatUseLoopIV.begin();
  unsigned firstDim = *it++, secondDim = *it++;
  OpBuilder builder(loop);
  Value first = ValueOrUnsigned::getValue(workGroupSize[firstDim], builder);
  Value second = ValueOrUnsigned::getValue(workGroupSize[secondDim], builder);
  Value versionCond = builder.create<arith::CmpIOp>(
      loop.getLoc(), arith::CmpIPredicate::eq, first, second);

  for (; it != dimsThatUseLoopIV.end(); ++it) {
    Value wgSize = ValueOrUnsigned::getValue(workGroupSize[*it], builder);
    versionCond = builder.create<arith::AndIOp>(
        loop.getLoc(), versionCond,
        builder.create<arith::CmpIOp>(loop.getLoc(), arith::CmpIPredicate::eq,
                                      first, wgSize));
  }

  return versionCond;
}

Value LoopInternalization::getVersionCondition(
    LoopLikeOpInterface loop, const WorkGroupSize &workGroupSize,
    std::variant<Value, unsigned> reqdSharedMemory,
    const unsigned sharedMemoryRemaining, DataFlowSolver &solver) const {
  // Check that there is enough shared memory available.
  OpBuilder builder(loop);
  Value versionCond;
  if (std::holds_alternative<Value>(reqdSharedMemory)) {
    Value val = builder.create<arith::ConstantIndexOp>(loop.getLoc(),
                                                       sharedMemoryRemaining);
    versionCond =
        builder.create<arith::CmpIOp>(loop.getLoc(), arith::CmpIPredicate::ule,
                                      std::get<Value>(reqdSharedMemory), val);
  }

  // If promotable memory accesses do not reference the loop IV on the same
  // dimension, check that the work group sizes corresponding to the dimensions
  // that reference the loop IV are the same.
  if (Value cond = checkForConsistentUseOfLoopIV(loop, workGroupSize, solver))
    return versionCond
               ? builder.create<arith::AndIOp>(loop.getLoc(), versionCond, cond)
               : cond;

  return versionCond;
}

bool LoopInternalization::canBeTransformed(LoopLikeOpInterface loop,
                                           const WorkGroupSize &workGroupSize,
                                           DataFlowSolver &solver) const {
  if (workGroupSize.isKnown()) {
    std::set<unsigned> localIDDimForLoopIV =
        getLocalIDDimForLoopIV(loop, solver);
    const unsigned firstDim = *localIDDimForLoopIV.begin();

    if (llvm::any_of(localIDDimForLoopIV, [&](unsigned dim) {
          return (workGroupSize.get<unsigned>(dim) !=
                  workGroupSize.get<unsigned>(firstDim));
        })) {
      LLVM_DEBUG(llvm::dbgs() << "Loop cannot be transformed\n");
      return false;
    }
  }

  return true;
}

Value LoopInternalization::getTileSize(LoopLikeOpInterface loop,
                                       const WorkGroupSize &workGroupSize,
                                       DataFlowSolver &solver) const {
  std::set<unsigned> localIDDimForLoopIV = getLocalIDDimForLoopIV(loop, solver);
  assert(!localIDDimForLoopIV.empty() && "set should not be empty");

  OpBuilder builder(loop);
  return ValueOrUnsigned::getValue(workGroupSize[*localIDDimForLoopIV.begin()],
                                   builder);
}

void LoopInternalization::transform(FunctionOpInterface func,
                                    const FunctionKernelInfo &funcKernelInfo,
                                    const unsigned sharedMemoryRemaining,
                                    DataFlowSolver &solver) {
  // Calculate the total amount of shared memory needed to promote the memory
  // accesses that were identified by the analysis.
  SmallVector<gpu::GPUFuncOp> kernels;
  funcKernelInfo.getKernelCallers(func, kernels);

  const unsigned numDims = getGridDimension(func);
  sycl::ReqdWorkGroupSize reqdWorkGroupSize(kernels);
  OpBuilder builder(func->getRegion(0));
  WorkGroupSize workGroupSize(numDims, reqdWorkGroupSize, builder);

  std::variant<Value, unsigned> reqdSharedMemory =
      getReqdSharedMemory(loopToSharedMemref, workGroupSize, builder);

  if (std::holds_alternative<unsigned>(reqdSharedMemory) &&
      std::get<unsigned>(reqdSharedMemory) > sharedMemoryRemaining) {
    // This is a conservative check because 'reqdSharedMemory' is the max shared
    // memory required to transform any loop in the function, so there might be
    // a loop that require considerably less than the max.
    LLVM_DEBUG(llvm::dbgs() << "Not enough shared local memory\n");
    return;
  }

  // Create SYCL local ids corresponding to the grid dimensionality (per
  // kernel).
  SmallVector<Value> localIDs;
  sycl::populateLocalID(localIDs, numDims, builder, func.getLoc());

  // Reserve static shared local memory for this function.
  auto gpuModule = func->getParentOfType<gpu::GPUModuleOp>();
  assert(gpuModule && "Expecting valid GPUModuleOp");
  memref::GlobalOp wgSharedLocalMemory = getWorkGroupSharedLocalMemory(
      gpuModule, std::holds_alternative<unsigned>(reqdSharedMemory)
                     ? std::get<unsigned>(reqdSharedMemory)
                     : sharedMemoryRemaining);

  // Now that we have a list of memref to promote to shared memory in each
  // loop nest's innermost loop, perform the transformation.
  for (auto &entry : loopToSharedMemref) {
    LoopLikeOpInterface loop = entry.first;
    TypeSwitch<Operation *>(loop).Case<affine::AffineForOp, scf::ForOp>(
        [&](auto loop) {
          if (canBeTransformed(loop, workGroupSize, solver))
            transform(loop, wgSharedLocalMemory, localIDs, workGroupSize,
                      reqdSharedMemory, sharedMemoryRemaining, solver);
        });
  }

  loopToSharedMemref.clear();
  localIDs.clear();
}

template <typename T>
void LoopInternalization::transform(
    T loop, memref::GlobalOp wgSharedLocalMemory, ArrayRef<Value> localIDs,
    const WorkGroupSize &workGroupSize,
    std::variant<Value, unsigned> reqdSharedMemory,
    const unsigned sharedMemoryRemaining, DataFlowSolver &solver) {
  static_assert(llvm::is_one_of<T, affine::AffineForOp, scf::ForOp>::value);
  assert(LoopTools::isInnermostLoop(loop) && "Expecting an innermost loop");
  assert(localIDs.size() >= 1 && localIDs.size() <= 3 &&
         "Expecting valid localIDs");
  assert(loopToSharedMemref.find(loop) != loopToSharedMemref.end() &&
         "Loop should be in the map");

  // Version the loop if necessary.
  if (Value versionCond =
          getVersionCondition(loop, workGroupSize, reqdSharedMemory,
                              sharedMemoryRemaining, solver)) {
    LLVM_DEBUG(llvm::dbgs() << "Versioning loop: " << loop << "\n");
    LoopTools::versionLoop(loop, versionCond);
  }

  // Statically allocate shared memory.
  OpBuilder builder(loop);
  auto getGlobalOp = builder.create<memref::GetGlobalOp>(
      loop.getLoc(), wgSharedLocalMemory.getType(),
      wgSharedLocalMemory.getName());
  const SetVector<Operation *> &memRefs = loopToSharedMemref.at(loop);

  // Tile the loop.
  SmallVector<T> tiledNest;
  tile(loop, getTileSize(loop, workGroupSize, solver), tiledNest);
  LLVM_DEBUG(llvm::dbgs() << "Tiled loop: " << tiledNest.front() << "\n");

  // Rewrite selected memory accesses to use shared memory.
  loop = tiledNest.front();
  LoopInfo loopInfo(loop);
  builder.setInsertionPointToStart(loop->getBlock());
  std::variant<Value, unsigned> offset =
      ValueOrUnsigned::get(0, builder, !workGroupSize.isKnown());
  for (Operation *memref : memRefs) {
    Location loc = memref->getLoc();
    auto accSub = cast<sycl::SYCLAccessorSubscriptOp>(memref);
    sycl::AccessorType accTy =
        getAccessorType(cast<sycl::SYCLAccessorSubscriptOp>(memref));

    SmallVector<unsigned> localIDOrdering = getLocalIDOrdering(accSub, solver);
    // Get pointer to the shared memory portion for each memref.
    memref::ViewOp viewOp =
        createViewOp(accTy, ValueOrUnsigned::getValue(offset, builder),
                     getGlobalOp, localIDOrdering, workGroupSize, builder, loc);

    promote(memref, viewOp, loopInfo, localIDs, builder, solver);

    // Only increment offset when the current memref is not the last one.
    if (memref != *memRefs.rbegin()) {
      std::variant<Value, unsigned> reqdSharedMemory =
          getReqdSharedMemory(accTy, workGroupSize, builder);
      offset = ValueOrUnsigned::add(offset, reqdSharedMemory, builder);
    }
    ++numAccessInternalized;
  }
  LLVM_DEBUG(llvm::dbgs() << "Promoted loop: " << loop << "\n");

  builder.setInsertionPoint(loop);
  createWorkGroupBarrier(builder);
  builder.setInsertionPointAfter(loop);
  createWorkGroupBarrier(builder);

  ++numLoopInternalized;

  // When work group size is unknown at compile time, unroll the tiled loop to
  // expose more optimization opportunities.
  if (!workGroupSize.isKnown())
    unrollByFactor(loop, unrollFactor);
}

void LoopInternalization::promote(Operation *memref, memref::ViewOp viewOp,
                                  LoopInfo &loopInfo, ArrayRef<Value> localIDs,
                                  OpBuilder &builder,
                                  DataFlowSolver &solver) const {
  Location loc = memref->getLoc();
  LoopLikeOpInterface loop = loopInfo.getLoop();
  Value inductionVar = loopInfo.getInductionVar();
  auto accSub = cast<sycl::SYCLAccessorSubscriptOp>(memref);
  const SmallVector<Value> indexes = getIndexes(accSub, solver);

  SmallVector<Value> viewIndexes;
  for (unsigned index : getLocalIDOrdering(accSub, solver))
    viewIndexes.push_back(localIDs[index]);

  // Populate indexes needed for loading the accesses from global memory.
  SmallVector<Value> globalIndexes(indexes.size());
  for (unsigned dim = 0; dim < indexes.size(); ++dim) {
    Value index = indexes[dim];
    if (usesValue(index, inductionVar)) {
      Value lowerBound = loopInfo.getLowerBound();
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointAfterValue(lowerBound);
      Value adjustedGlobalIV = builder.create<arith::AddIOp>(
          viewIndexes[dim].getLoc(), viewIndexes[dim], lowerBound);
      IRMapping mapping;
      mapping.map(inductionVar, adjustedGlobalIV);
      globalIndexes[dim] = castToIndex(
          builder.clone(*index.getDefiningOp(), mapping)->getResult(0));
      continue;
    }
    if (!loop.isDefinedOutsideOfLoop(index))
      loop.moveOutOfLoop(index.getDefiningOp());
    globalIndexes[dim] = castToIndex(index);
  }

  // Load from global memory.
  builder.setInsertionPoint(loop);
  sycl::AccessorType accTy = getAccessorType(accSub);
  const auto idTy = cast<sycl::IDType>(
      cast<sycl::AccessorImplDeviceType>(accTy.getBody()[0]).getBody()[0]);
  TypedValue<MemRefType> id =
      sycl::createSYCLIDConstructorOp(idTy, globalIndexes, builder, loc);
  const Value zeroIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value globalAccSub =
      sycl::createSYCLAccessorSubscriptOp(accSub.getAcc(), id, builder, loc);
  auto load = builder.create<memref::LoadOp>(loc, globalAccSub, zeroIndex);

  // Store to shared memory.
  builder.create<memref::StoreOp>(loc, load, viewOp, viewIndexes);

  // Populate indexes will be used in loop with shared memory.
  SmallVector<Value> adjustedIndexes;
  for (unsigned dim = 0; dim < indexes.size(); ++dim) {
    Value index = indexes[dim];
    if (usesValue(index, inductionVar)) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointAfterValue(index);
      IRMapping mapping;
      Value adjustedIV = loopInfo.getAdjustedIV();
      mapping.map(inductionVar, adjustedIV);
      adjustedIndexes.push_back(castToIndex(
          builder.clone(*index.getDefiningOp(), mapping)->getResult(0)));
      continue;
    }
    adjustedIndexes.push_back(viewIndexes[dim]);
  }

  // Replace original accesses with accesses to shared memory.
  SmallVector<Operation *> users(memref->getUsers());
  for (Operation *user : users) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPoint(user);
    assert(isa<affine::AffineLoadOp>(user) && "Expecting affine load user");
    auto load =
        builder.create<memref::LoadOp>(user->getLoc(), viewOp, adjustedIndexes);
    user->replaceAllUsesWith(load);
    user->erase();
  }

  assert(memref->use_empty() && "Expecting all uses of memref to be replaced");
  memref->erase();
}

} // namespace

std::unique_ptr<Pass> polygeist::createLoopInternalizationPass() {
  return std::make_unique<LoopInternalization>();
}
std::unique_ptr<Pass> polygeist::createLoopInternalizationPass(
    const LoopInternalizationOptions &options) {
  return std::make_unique<LoopInternalization>(options);
}
