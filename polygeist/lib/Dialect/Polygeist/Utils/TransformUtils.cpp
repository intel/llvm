//===- TransformUtils.cpp - Polygeist Transform Utilities  ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Utils/TransformUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Polygeist/IR/PolygeistOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"
#include <optional>

using namespace mlir;
using namespace mlir::polygeist;

//===----------------------------------------------------------------------===//
// Utilities functions
//===----------------------------------------------------------------------===//

static constexpr StringLiteral linkageAttrName = "llvm.linkage";
bool polygeist::isLinkonceODR(FunctionOpInterface func) {
  if (!func->hasAttr(linkageAttrName))
    return false;
  auto attr = cast<LLVM::LinkageAttr>(func->getAttr(linkageAttrName));
  return attr.getLinkage() == LLVM::Linkage::LinkonceODR;
}

void polygeist::privatize(FunctionOpInterface func) {
  func->setAttr(
      linkageAttrName,
      LLVM::LinkageAttr::get(func->getContext(), LLVM::Linkage::Private));
  func.setPrivate();
}

bool polygeist::isTailCall(CallOpInterface call) {
  if (!call->getBlock()->hasNoSuccessors())
    return false;
  Operation *nextOp = call->getNextNode();
  return (nextOp->hasTrait<OpTrait::IsTerminator>() ||
          isRegionReturnLike(nextOp));
}

/// Populate \p funcMaxDepthMap with the maximum depth from a GPU kernel for \p
/// func and its callers.
static void getMaxDepthFromAnyGPUKernel(
    FunctionOpInterface func,
    DenseMap<FunctionOpInterface, Optional<unsigned>> &funcMaxDepthMap) {
  assert(!funcMaxDepthMap.contains(func) &&
         "Expecting maximum depth of func is not already calculated");

  // A function that does not reside in a GPU module cannot be called from a GPU
  // kernel.
  if (!func->getParentOfType<gpu::GPUModuleOp>()) {
    funcMaxDepthMap[func] = std::nullopt;
    return;
  }

  Operation *op = func;
  if (auto gpuFunc = dyn_cast<gpu::GPUFuncOp>(op))
    if (gpuFunc.isKernel()) {
      funcMaxDepthMap[func] = 0;
      return;
    }

  ModuleOp module = func->getParentOfType<ModuleOp>();
  SymbolTableCollection symTable;
  SymbolUserMap userMap(symTable, module);
  Optional<unsigned> maxDepth = std::nullopt;
  for (Operation *call : userMap.getUsers(func)) {
    auto caller = call->getParentOfType<FunctionOpInterface>();
    if (!funcMaxDepthMap.contains(caller))
      getMaxDepthFromAnyGPUKernel(caller, funcMaxDepthMap);
    Optional<unsigned> callerDepth = funcMaxDepthMap[caller];

    // Caller not called from a GPU kernel.
    if (!callerDepth.has_value())
      continue;

    unsigned depth = 1 + callerDepth.value();
    if (!maxDepth.has_value())
      maxDepth = depth;
    else if (depth > maxDepth.value())
      maxDepth = depth;
  }
  funcMaxDepthMap[func] = maxDepth;
}

Optional<unsigned>
polygeist::getMaxDepthFromAnyGPUKernel(FunctionOpInterface func) {
  DenseMap<FunctionOpInterface, Optional<unsigned>> funcMaxDepthMap;
  ::getMaxDepthFromAnyGPUKernel(func, funcMaxDepthMap);
  return funcMaxDepthMap[func];
}

bool polygeist::isPotentialKernelBodyFunc(FunctionOpInterface func) {
  // The function must be defined, and private or with linkonce_odr linkage.
  if (func.isExternal() || (!func.isPrivate() && !isLinkonceODR(func)))
    return false;

  ModuleOp module = func->getParentOfType<ModuleOp>();
  SymbolTableCollection symTable;
  SymbolUserMap userMap(symTable, module);

  if (!all_of(userMap.getUsers(func), [](Operation *op) {
        if (auto call = dyn_cast<CallOpInterface>(op))
          return isTailCall(call);
        return false;
      }))
    return false;

  Optional<unsigned> maxDepth = getMaxDepthFromAnyGPUKernel(func);
  // The function must to called from GPU kernel.
  if (!maxDepth.has_value())
    return false;
  // The function should be called directly by a GPU kernel, or called by a
  // function that directly called by a GPU kernel.
  return (maxDepth.value() == 1 || maxDepth.value() == 2);
}

Optional<Value> polygeist::getAccessorUsedByOperation(const Operation &op) {
  auto getMemrefOp = [](const Operation &op) {
    return TypeSwitch<const Operation &, Operation *>(op)
        .Case<affine::AffineLoadOp, affine::AffineStoreOp>(
            [](auto &affineOp) { return affineOp.getMemref().getDefiningOp(); })
        .Default([](auto &) { return nullptr; });
  };

  auto accSub =
      dyn_cast_or_null<sycl::SYCLAccessorSubscriptOp>(getMemrefOp(op));
  return accSub ? Optional<Value>(accSub.getAcc()) : std::nullopt;
}

static Block &getThenBlock(RegionBranchOpInterface ifOp) {
  return ifOp->getRegion(0).front();
}

static Block &getElseBlock(RegionBranchOpInterface ifOp) {
  return ifOp->getRegion(1).front();
}

// Replace uses of the operation \p op return value(s) with the value(s) yielded
// by the \p ifOp operation.
static void replaceUsesOfReturnValues(Operation *op,
                                      RegionBranchOpInterface ifOp) {
  assert(ifOp && "Expected valid ifOp");
  for (auto [opVal, ifVal] : llvm::zip(op->getResults(), ifOp->getResults()))
    opVal.replaceUsesWithIf(ifVal, [&](OpOperand &operand) {
      Block *useBlock = operand.getOwner()->getBlock();
      return useBlock != &getThenBlock(ifOp);
    });
}

static void createThenBody(Operation *op, scf::IfOp ifOp) {
  op->moveBefore(&getThenBlock(ifOp).front());
}

static void createThenBody(Operation *op, affine::AffineIfOp ifOp) {
  OpBuilder thenBodyBuilder = ifOp.getThenBodyBuilder();
  if (!op->getResults().empty())
    thenBodyBuilder.create<affine::AffineYieldOp>(op->getLoc(),
                                                  op->getResults());
  op->moveBefore(&getThenBlock(ifOp).front());
}

namespace {

struct SCFIfBuilder {
  static scf::IfOp createIfOp(Value condition, Operation::result_range results,
                              OpBuilder &builder, Location loc) {
    assert(condition && "Expecting a valid condition");
    return builder.create<scf::IfOp>(
        loc, condition,
        [&](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(loc, results);
        },
        [&](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(loc, results);
        });
  }
};

struct AffineIfBuilder {
  static affine::AffineIfOp createIfOp(IntegerSet ifCondSet,
                                       SmallVectorImpl<Value> &setOperands,
                                       Operation::result_range results,
                                       OpBuilder &builder, Location loc) {
    TypeRange types(results);
    return builder.create<affine::AffineIfOp>(loc, types, ifCondSet,
                                              setOperands, true);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// VersionBuilder
//===----------------------------------------------------------------------===//

void VersionBuilder::version(const VersionCondition &versionCond) const {
  OpBuilder builder(op);

  if (versionCond.hasSCFCondition()) {
    scf::IfOp ifOp = SCFIfBuilder::createIfOp(
        versionCond.getSCFCondition(), op->getResults(), builder, op->getLoc());
    createThenBody(op, ifOp);
    createElseBody(ifOp);
    replaceUsesOfReturnValues(op, ifOp);
  } else {
    assert(versionCond.hasAffineCondition() && "Expecting an affine condition");
    const auto &affineCond = versionCond.getAffineCondition();
    affine::AffineIfOp ifOp = AffineIfBuilder::createIfOp(
        affineCond.ifCondSet, affineCond.setOperands, op->getResults(), builder,
        op->getLoc());
    createThenBody(op, ifOp);
    createElseBody(ifOp);
    replaceUsesOfReturnValues(op, ifOp);
  }
}

void VersionBuilder::createElseBody(scf::IfOp ifOp) const {
  Operation &origYield = getElseBlock(ifOp).back();
  OpBuilder elseBodyBuilder = ifOp.getElseBodyBuilder();
  Operation *clonedLoop = elseBodyBuilder.clone(*op);
  elseBodyBuilder.create<scf::YieldOp>(op->getLoc(), clonedLoop->getResults());
  origYield.erase();
}

void VersionBuilder::createElseBody(affine::AffineIfOp ifOp) const {
  OpBuilder elseBodyBuilder = ifOp.getElseBodyBuilder();
  Operation *clonedLoop = elseBodyBuilder.clone(*op);
  if (!clonedLoop->getResults().empty())
    elseBodyBuilder.create<affine::AffineYieldOp>(op->getLoc(),
                                                  clonedLoop->getResults());
}

//===----------------------------------------------------------------------===//
// LoopGuardBuilder
//===----------------------------------------------------------------------===//

std::unique_ptr<LoopGuardBuilder>
LoopGuardBuilder::create(LoopLikeOpInterface loop) {
  return TypeSwitch<Operation *, std::unique_ptr<LoopGuardBuilder>>(loop)
      .Case<scf::ForOp>(
          [](auto loop) { return std::make_unique<SCFForGuardBuilder>(loop); })
      .Case<scf::ParallelOp>([](auto loop) {
        return std::make_unique<SCFParallelGuardBuilder>(loop);
      })
      .Case<affine::AffineForOp>([](auto loop) {
        return std::make_unique<AffineForGuardBuilder>(loop);
      })
      .Case<affine::AffineParallelOp>([](auto loop) {
        return std::make_unique<AffineParallelGuardBuilder>(loop);
      });
}

void LoopGuardBuilder::guardLoop(RegionBranchOpInterface ifOp) const {
  createThenBody(ifOp);
  createElseBody(ifOp);
  replaceUsesOfReturnValues(loop, ifOp);
}

//===----------------------------------------------------------------------===//
// SCFLoopGuardBuilder
//===----------------------------------------------------------------------===//

void SCFLoopGuardBuilder::guardLoop() const {
  OpBuilder builder(loop);
  Value condition = createCondition();
  RegionBranchOpInterface ifOp = SCFIfBuilder::createIfOp(
      condition, loop->getResults(), builder, loop.getLoc());
  LoopGuardBuilder::guardLoop(ifOp);
}

Value SCFLoopGuardBuilder::createCondition() const {
  OpBuilder builder(loop);
  Value cond;
  for (auto [lb, ub] : llvm::zip(getLowerBounds(), getUpperBounds())) {
    const Value val = builder.create<arith::CmpIOp>(
        loop.getLoc(), arith::CmpIPredicate::slt, lb, ub);
    cond = cond ? static_cast<Value>(
                      builder.create<arith::AndIOp>(loop.getLoc(), cond, val))
                : val;
  }
  return cond;
}

void SCFLoopGuardBuilder::createThenBody(RegionBranchOpInterface ifOp) const {
  ::createThenBody(loop, cast<scf::IfOp>(ifOp));
}

void SCFLoopGuardBuilder::createElseBody(RegionBranchOpInterface ifOp) const {
  Operation &origYield = getElseBlock(ifOp).back();
  bool yieldsResults = !loop->getResults().empty();
  OpBuilder elseBodyBuilder = cast<scf::IfOp>(ifOp).getElseBodyBuilder();
  if (yieldsResults) {
    elseBodyBuilder.create<scf::YieldOp>(loop.getLoc(), getInitVals());
    origYield.erase();
  } else
    getElseBlock(ifOp).erase();
}

//===----------------------------------------------------------------------===//
// AffineLoopGuardBuilder
//===----------------------------------------------------------------------===//

void AffineLoopGuardBuilder::guardLoop() const {
  OpBuilder builder(loop);
  SmallVector<Value> setOperands;
  IntegerSet ifCondSet = createCondition(setOperands);
  RegionBranchOpInterface ifOp = AffineIfBuilder::createIfOp(
      ifCondSet, setOperands, loop->getResults(), builder, loop.getLoc());
  LoopGuardBuilder::guardLoop(ifOp);
}

void AffineLoopGuardBuilder::createThenBody(
    RegionBranchOpInterface ifOp) const {
  ::createThenBody(loop, cast<affine::AffineIfOp>(ifOp));
}

void AffineLoopGuardBuilder::createElseBody(
    RegionBranchOpInterface ifOp) const {
  bool yieldsResults = !loop->getResults().empty();
  OpBuilder elseBodyBuilder =
      cast<affine::AffineIfOp>(ifOp).getElseBodyBuilder();
  if (yieldsResults)
    elseBodyBuilder.create<affine::AffineYieldOp>(loop.getLoc(), getInitVals());
  else
    getElseBlock(ifOp).erase();
}

IntegerSet AffineLoopGuardBuilder::createCondition(
    SmallVectorImpl<Value> &setOperands) const {
  OperandRange lb_ops = getLowerBoundsOperands(),
               ub_ops = getUpperBoundsOperands();
  const AffineMap lbMap = getLowerBoundsMap(), ubMap = getUpperBoundsMap();

  std::copy(lb_ops.begin(), lb_ops.begin() + lbMap.getNumDims(),
            std::back_inserter(setOperands));
  std::copy(ub_ops.begin(), ub_ops.begin() + ubMap.getNumDims(),
            std::back_inserter(setOperands));
  std::copy(lb_ops.begin() + lbMap.getNumDims(), lb_ops.end(),
            std::back_inserter(setOperands));
  std::copy(ub_ops.begin() + ubMap.getNumDims(), ub_ops.end(),
            std::back_inserter(setOperands));

  SmallVector<AffineExpr, 4> dims;
  for (unsigned idx = 0; idx < ubMap.getNumDims(); ++idx)
    dims.push_back(
        getAffineDimExpr(idx + lbMap.getNumDims(), loop.getContext()));

  SmallVector<AffineExpr, 4> symbols;
  for (unsigned idx = 0; idx < ubMap.getNumSymbols(); ++idx)
    symbols.push_back(
        getAffineSymbolExpr(idx + lbMap.getNumSymbols(), loop.getContext()));

  SmallVector<AffineExpr, 2> exprs;
  getConstraints(exprs, dims, symbols);
  SmallVector<bool, 2> eqFlags(exprs.size(), false);

  return IntegerSet::get(
      /*dim*/ lbMap.getNumDims() + ubMap.getNumDims(),
      /*symbols*/ lbMap.getNumSymbols() + ubMap.getNumSymbols(), exprs,
      eqFlags);
}

//===----------------------------------------------------------------------===//
// SCFForGuardBuilder
//===----------------------------------------------------------------------===//

SCFForGuardBuilder::SCFForGuardBuilder(scf::ForOp loop)
    : SCFLoopGuardBuilder(loop) {}

scf::ForOp SCFForGuardBuilder::getLoop() const {
  return cast<scf::ForOp>(loop);
}

OperandRange SCFForGuardBuilder::getInitVals() const {
  return getLoop().getInitArgs();
}

OperandRange SCFForGuardBuilder::getLowerBounds() const {
  return getLoop().getODSOperands(0);
}

OperandRange SCFForGuardBuilder::getUpperBounds() const {
  return getLoop().getODSOperands(1);
}

//===----------------------------------------------------------------------===//
// SCFParallelGuardBuilder
//===----------------------------------------------------------------------===//

SCFParallelGuardBuilder::SCFParallelGuardBuilder(scf::ParallelOp loop)
    : SCFLoopGuardBuilder(loop) {}

scf::ParallelOp SCFParallelGuardBuilder::getLoop() const {
  return cast<scf::ParallelOp>(loop);
}

OperandRange SCFParallelGuardBuilder::getInitVals() const {
  return getLoop().getInitVals();
}

OperandRange SCFParallelGuardBuilder::getLowerBounds() const {
  return getLoop().getLowerBound();
}

OperandRange SCFParallelGuardBuilder::getUpperBounds() const {
  return getLoop().getUpperBound();
}

//===----------------------------------------------------------------------===//
// AffineForGuardBuilder
//===----------------------------------------------------------------------===//

AffineForGuardBuilder::AffineForGuardBuilder(affine::AffineForOp loop)
    : AffineLoopGuardBuilder(loop) {}

affine::AffineForOp AffineForGuardBuilder::getLoop() const {
  return cast<affine::AffineForOp>(loop);
}

void AffineForGuardBuilder::getConstraints(SmallVectorImpl<AffineExpr> &exprs,
                                           ArrayRef<AffineExpr> dims,
                                           ArrayRef<AffineExpr> symbols) const {
  for (AffineExpr ub : getLoop().getUpperBoundMap().getResults()) {
    ub = ub.replaceDimsAndSymbols(dims, symbols);
    for (AffineExpr lb : getLoop().getLowerBoundMap().getResults()) {
      // Bound is whether this expr >= 0, which since we want ub > lb, we
      // rewrite as follows.
      exprs.push_back(ub - lb - 1);
    }
  }
}

OperandRange AffineForGuardBuilder::getInitVals() const {
  return getLoop().getIterOperands();
}

OperandRange AffineForGuardBuilder::getLowerBoundsOperands() const {
  return getLoop().getLowerBoundOperands();
}

OperandRange AffineForGuardBuilder::getUpperBoundsOperands() const {
  return getLoop().getUpperBoundOperands();
}

AffineMap AffineForGuardBuilder::getLowerBoundsMap() const {
  return getLoop().getLowerBoundMap();
}

AffineMap AffineForGuardBuilder::getUpperBoundsMap() const {
  return getLoop().getUpperBoundMap();
}

//===----------------------------------------------------------------------===//
// AffineParallelGuardBuilder
//===----------------------------------------------------------------------===//

AffineParallelGuardBuilder::AffineParallelGuardBuilder(
    affine::AffineParallelOp loop)
    : AffineLoopGuardBuilder(loop) {}

affine::AffineParallelOp AffineParallelGuardBuilder::getLoop() const {
  return cast<affine::AffineParallelOp>(loop);
}

void AffineParallelGuardBuilder::getConstraints(
    SmallVectorImpl<AffineExpr> &exprs, ArrayRef<AffineExpr> dims,
    ArrayRef<AffineExpr> symbols) const {
  for (auto step : llvm::enumerate(getLoop().getSteps()))
    for (AffineExpr ub :
         getLoop().getUpperBoundMap(step.index()).getResults()) {
      ub = ub.replaceDimsAndSymbols(dims, symbols);
      for (AffineExpr lb :
           getLoop().getLowerBoundMap(step.index()).getResults()) {
        // Bound is whether this expr >= 0, which since we want ub > lb, we
        // rewrite as follows.
        exprs.push_back(ub - lb - 1);
      }
    }
}

mlir::Operation::operand_range AffineParallelGuardBuilder::getInitVals() const {
  return getLoop().getMapOperands();
}

OperandRange AffineParallelGuardBuilder::getLowerBoundsOperands() const {
  return getLoop().getLowerBoundsOperands();
}

OperandRange AffineParallelGuardBuilder::getUpperBoundsOperands() const {
  return getLoop().getUpperBoundsOperands();
}

AffineMap AffineParallelGuardBuilder::getLowerBoundsMap() const {
  return getLoop().getLowerBoundsMap();
}

AffineMap AffineParallelGuardBuilder::getUpperBoundsMap() const {
  return getLoop().getUpperBoundsMap();
}

//===----------------------------------------------------------------------===//
// Loop Tools
//===----------------------------------------------------------------------===//

void LoopTools::guardLoop(LoopLikeOpInterface loop) {
  LoopGuardBuilder::create(loop)->guardLoop();
}

void LoopTools::versionLoop(LoopLikeOpInterface loop,
                            const VersionCondition &versionCond) {
  VersionBuilder(loop).version(versionCond);
}

bool LoopTools::isOutermostLoop(LoopLikeOpInterface loop) {
  return !loop->getParentOfType<LoopLikeOpInterface>();
}

bool LoopTools::isPerfectLoopNest(LoopLikeOpInterface root) {
  assert(root && "Expecting a valid pointer");

  LoopLikeOpInterface previousLoop = root;
  WalkResult walkResult =
      root->walk<WalkOrder::PreOrder>([&](LoopLikeOpInterface loop) {
        if (!arePerfectlyNested(previousLoop, loop))
          return WalkResult::interrupt();

        previousLoop = loop;
        return WalkResult::advance();
      });

  return !walkResult.wasInterrupted();
}

std::optional<LoopLikeOpInterface>
LoopTools::getInnermostLoop(LoopLikeOpInterface root) {
  assert(root && "Expecting a valid pointer");

  LoopLikeOpInterface previousLoop = root;
  WalkResult walkResult =
      root->walk<WalkOrder::PreOrder>([&](LoopLikeOpInterface loop) {
        if (!arePerfectlyNested(previousLoop, loop)) {
          llvm::errs() << "Not perfectly nested\n";
          return WalkResult::interrupt();
        }

        previousLoop = loop;
        return WalkResult::advance();
      });

  if (!walkResult.wasInterrupted())
    return previousLoop;
  return std::nullopt;
}

bool LoopTools::arePerfectlyNested(LoopLikeOpInterface outer,
                                   LoopLikeOpInterface inner) {
  assert(outer && inner && "Expecting valid pointers");
  if (outer == inner)
    return true;

  Block &outerLoopBody = outer.getLoopBody().front();
  if (outerLoopBody.begin() != std::prev(outerLoopBody.end(), 2))
    return false;

  return inner == dyn_cast<LoopLikeOpInterface>(&outerLoopBody.front());
}

//===----------------------------------------------------------------------===//
// VersionConditionBuilder
//===----------------------------------------------------------------------===//

static sycl::SYCLIDGetOp createSYCLIDGetOp(TypedValue<MemRefType> id,
                                           unsigned index, OpBuilder builder,
                                           Location loc) {
  const Value indexOp = builder.create<arith::ConstantIntOp>(loc, index, 32);
  const auto resTy = builder.getIndexType();
  return builder.create<sycl::SYCLIDGetOp>(
      loc, MemRefType::get(ShapedType::kDynamic, resTy), id, indexOp);
}

static sycl::SYCLRangeGetOp createSYCLRangeGetOp(TypedValue<MemRefType> range,
                                                 unsigned index,
                                                 OpBuilder builder,
                                                 Location loc) {
  const Value indexOp = builder.create<arith::ConstantIntOp>(loc, index, 32);
  const auto resTy = builder.getIndexType();
  return builder.create<sycl::SYCLRangeGetOp>(loc, resTy, range, indexOp);
}

static sycl::SYCLAccessorGetRangeOp
createSYCLAccessorGetRangeOp(sycl::AccessorPtrValue accessor, OpBuilder builder,
                             Location loc) {
  const sycl::AccessorType accTy = accessor.getAccessorType();
  const auto rangeTy = cast<sycl::RangeType>(
      cast<sycl::AccessorImplDeviceType>(accTy.getBody()[0]).getBody()[1]);
  return builder.create<sycl::SYCLAccessorGetRangeOp>(loc, rangeTy, accessor);
}

static sycl::SYCLAccessorSubscriptOp
createSYCLAccessorSubscriptOp(sycl::AccessorPtrValue accessor,
                              TypedValue<MemRefType> id, OpBuilder builder,
                              Location loc) {
  const sycl::AccessorType accTy = accessor.getAccessorType();
  assert(accTy.getDimension() != 0 && "Dimensions cannot be zero");
  const auto MT = MemRefType::get(
      ShapedType::kDynamic, accTy.getType(), MemRefLayoutAttrInterface(),
      builder.getI64IntegerAttr(targetToAddressSpace(accTy.getTargetMode())));
  return builder.create<sycl::SYCLAccessorSubscriptOp>(loc, MT, accessor, id);
}

static sycl::SYCLAccessorGetPointerOp
createSYCLAccessorGetPointerOp(sycl::AccessorPtrValue accessor,
                               OpBuilder builder, Location loc) {
  const sycl::AccessorType accTy = accessor.getAccessorType();
  const auto MT = MemRefType::get(
      ShapedType::kDynamic, accTy.getType(), MemRefLayoutAttrInterface(),
      builder.getI64IntegerAttr(targetToAddressSpace(accTy.getTargetMode())));
  return builder.create<sycl::SYCLAccessorGetPointerOp>(loc, MT, accessor);
}

static Value getSYCLAccessorBegin(sycl::AccessorPtrValue accessor,
                                  OpBuilder builder, Location loc) {
  const sycl::AccessorType accTy = accessor.getAccessorType();
  if (accTy.getDimension() == 0)
    return createSYCLAccessorGetPointerOp(accessor, builder, loc);

  const auto idTy = cast<sycl::IDType>(
      cast<sycl::AccessorImplDeviceType>(accTy.getBody()[0]).getBody()[0]);
  auto id = builder.create<memref::AllocaOp>(loc, MemRefType::get(1, idTy));
  const Value zeroIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
  for (unsigned i = 0; i < accTy.getDimension(); ++i) {
    Value idGetOp = createSYCLIDGetOp(id, i, builder, loc);
    builder.create<memref::StoreOp>(loc, zeroIndex, idGetOp, zeroIndex);
  }
  return createSYCLAccessorSubscriptOp(accessor, id, builder, loc);
}

static Value getSYCLAccessorEnd(sycl::AccessorPtrValue accessor,
                                OpBuilder builder, Location loc) {
  const sycl::AccessorType accTy = accessor.getAccessorType();
  if (accTy.getDimension() == 0) {
    Value getPointer = createSYCLAccessorGetPointerOp(accessor, builder, loc);
    const Value oneIndex = builder.create<arith::ConstantIndexOp>(loc, 1);
    return builder.create<polygeist::SubIndexOp>(loc, getPointer.getType(),
                                                 getPointer, oneIndex);
  }

  Value getRangeOp = createSYCLAccessorGetRangeOp(accessor, builder, loc);
  auto range = builder.create<memref::AllocaOp>(
      loc, MemRefType::get(1, getRangeOp.getType()));
  const Value zeroIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
  builder.create<memref::StoreOp>(loc, getRangeOp, range, zeroIndex);
  const auto idTy = cast<sycl::IDType>(
      cast<sycl::AccessorImplDeviceType>(accTy.getBody()[0]).getBody()[0]);
  auto id = builder.create<memref::AllocaOp>(loc, MemRefType::get(1, idTy));
  const Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  unsigned dim = accTy.getDimension();
  for (unsigned i = 0; i < dim; ++i) {
    Value idGetOp = createSYCLIDGetOp(id, i, builder, loc);
    Value rangeGetOp = createSYCLRangeGetOp(range, i, builder, loc);
    auto index = (i == dim - 1)
                     ? rangeGetOp
                     : builder.create<arith::SubIOp>(loc, rangeGetOp, one);
    builder.create<memref::StoreOp>(loc, index, idGetOp, zeroIndex);
  }
  return createSYCLAccessorSubscriptOp(accessor, id, builder, loc);
}

VersionConditionBuilder::VersionConditionBuilder(
    std::set<sycl::AccessorPtrPair> requireNoOverlapAccessorPairs,
    OpBuilder builder, Location loc)
    : accessorPairs(requireNoOverlapAccessorPairs), builder(builder), loc(loc) {
  assert(!accessorPairs.empty() &&
         "Expecting accessorPairs to have at least one pair");
}

VersionConditionBuilder::SCFCondition
VersionConditionBuilder::createSCFCondition(OpBuilder builder, Location loc,
                                            bool useOpaquePointers) const {
  auto GetMemref2PointerOp = [&](Value op) {
    auto MT = cast<MemRefType>(op.getType());
    auto PtrTy = (useOpaquePointers)
                     ? LLVM::LLVMPointerType::get(MT.getContext(),
                                                  MT.getMemorySpaceAsInt())
                     : LLVM::LLVMPointerType::get(MT.getElementType(),
                                                  MT.getMemorySpaceAsInt());
    return builder.create<polygeist::Memref2PointerOp>(loc, PtrTy, op);
  };

  Value condition;
  for (const sycl::AccessorPtrPair &accessorPair : accessorPairs) {
    Value begin1 = getSYCLAccessorBegin(accessorPair.first, builder, loc);
    Value end1 = getSYCLAccessorEnd(accessorPair.first, builder, loc);
    Value begin2 = getSYCLAccessorBegin(accessorPair.second, builder, loc);
    Value end2 = getSYCLAccessorEnd(accessorPair.second, builder, loc);
    auto beforeCond = builder.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::ule, GetMemref2PointerOp(end1),
        GetMemref2PointerOp(begin2));
    auto afterCond = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::uge,
                                                  GetMemref2PointerOp(begin1),
                                                  GetMemref2PointerOp(end2));
    Value orOp = builder.create<arith::OrIOp>(loc, beforeCond, afterCond);
    condition =
        condition ? builder.create<arith::AndIOp>(loc, condition, orOp) : orOp;
  }
  return condition;
}
