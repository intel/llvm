//===- LICM.cpp - Loop Invariant Code Motion ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Polygeist/IR/Ops.h"
#include "mlir/Dialect/Polygeist/IR/Polygeist.h"
#include "mlir/Dialect/Polygeist/Utils/TransformUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "licm"
#define REPORT_DEBUG_TYPE DEBUG_TYPE "-report"

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_LICM
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

using namespace mlir;

static llvm::cl::opt<bool> EnableLICMSYCLAccessorVersioning(
    "enable-licm-sycl-accessor-versioning", llvm::cl::init(false),
    llvm::cl::desc("Enable loop versioning for SYCL accessors in LICM"));

static llvm::cl::opt<unsigned> LICMSYCLAccessorPairsLimit(
    "licm-sycl-accessor-pairs-limit", llvm::cl::init(1),
    llvm::cl::desc(
        "Maximum number of versioning accessor pairs per operation in LICM"));

static llvm::cl::opt<unsigned> LICMVersionLimit(
    "licm-version-limit", llvm::cl::init(1),
    llvm::cl::desc("Maximum number of versioning allowed in LICM"));

namespace {

struct LICM : public mlir::polygeist::impl::LICMBase<LICM> {
  using LICMBase<LICM>::LICMBase;

  void runOnOperation() override;
};

/// Represents the side effects associated with an operation.
class OperationSideEffects {
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const OperationSideEffects &);

public:
  OperationSideEffects(const Operation &op, const AliasAnalysis &aliasAnalysis,
                       const DominanceInfo &domInfo)
      : op(op), aliasAnalysis(aliasAnalysis), domInfo(domInfo) {
    if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      memEffect.getEffects(effects);

      // Classify the side effects of the operation.
      for (MemoryEffects::EffectInstance EI : effects) {
        TypeSwitch<const MemoryEffects::Effect *>(EI.getEffect())
            .Case<MemoryEffects::Read>(
                [&](auto) { readResources.push_back(EI); })
            .Case<MemoryEffects::Write>(
                [&](auto) { writeResources.push_back(EI); })
            .Case<MemoryEffects::Free>(
                [&](auto) { freeResources.push_back(EI); })
            .Case<MemoryEffects::Allocate>(
                [&](auto) { allocateResources.push_back(EI); });
      }
    }
  }

  const Operation &getOperation() const { return op; };

  bool readsFromResource() const { return !readResources.empty(); }
  bool writesToResource() const { return !writeResources.empty(); }
  bool freesResource() const { return !freeResources.empty(); }
  bool allocatesResource() const { return !allocateResources.empty(); }

  /// Collects the side effects the symbol associated with this class has on the
  /// given resource \p res. The results is stored into \p effects.
  void getEffectsOnResource(
      SideEffects::Resource *res,
      SmallVectorImpl<MemoryEffects::EffectInstance> &effects) const {
    assert(res && "Expecting a valid resource");
    assert(effects.empty() && "Expecting an empty vector");
    cast<MemoryEffectOpInterface>(op).getEffectsOnResource(res, effects);
  }

  /// Returns true if the given operation \p other has side effects that
  /// conflict with the side effects summarized in this class, and false
  /// otherwise.
  bool conflictsWith(const Operation &other) const;

private:
  const Operation &op; /// Operation associated with the side effects.
  const AliasAnalysis &aliasAnalysis; /// Alias Analysis reference.
  const DominanceInfo &domInfo;       /// Dominance information reference.

  /// Side effects associated with reading resources.
  SmallVector<MemoryEffects::EffectInstance> readResources;
  /// Side effects associated with writing resources.
  SmallVector<MemoryEffects::EffectInstance> writeResources;
  /// Side effects associated with freeing resources.
  SmallVector<MemoryEffects::EffectInstance> freeResources;
  /// Side effects associated with allocating resources.
  SmallVector<MemoryEffects::EffectInstance> allocateResources;
};

[[maybe_unused]] inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &OS, const OperationSideEffects &ME) {
  auto printResources = [&OS](
                            StringRef title,
                            ArrayRef<MemoryEffects::EffectInstance> resources) {
    auto delimit = [](raw_ostream &OS, bool &isFirst) {
      if (!isFirst)
        OS << ", ";
      isFirst = false;
    };

    OS.indent(2) << title << ": ";
    bool isFirst = true;
    for (const MemoryEffects::EffectInstance &EI : resources) {
      delimit(OS, isFirst);
      OS << "{" << EI.getResource()->getName() << ", " << EI.getValue() << "}";
    }
    OS << "\n";
  };

  bool isSideEffectFree = !ME.readsFromResource() && !ME.writesToResource() &&
                          !ME.freesResource() && !ME.allocatesResource();

  OS << "Operation: " << ME.getOperation() << "\n";
  if (isSideEffectFree)
    OS.indent(2) << "=> is side effects free.\n";
  else {
    if (ME.readsFromResource())
      printResources("=> read resources", ME.readResources);
    if (ME.writesToResource())
      printResources("=> write resources", ME.writeResources);
    if (ME.freesResource())
      printResources("=> free resources", ME.freeResources);
    if (ME.allocatesResource())
      printResources("=> allocate resources", ME.allocateResources);
  }

  return OS;
}

} // namespace

//===----------------------------------------------------------------------===//
// OperationSideEffects
//===----------------------------------------------------------------------===//

bool OperationSideEffects::conflictsWith(const Operation &other) const {
  if (&op == &other || isMemoryEffectFree(const_cast<Operation *>(&other)))
    return false;

  // Conservatively assume operations with unknown side effects might write to
  // any memory.
  if (!isa<MemoryEffectOpInterface>(other) &&
      !const_cast<Operation &>(other)
           .hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "=> found conflict due to operation with unknown side effects:\n";
      llvm::dbgs().indent(2) << other << "\n";
    });
    return true;
  }

  // Check all the nested operations if 'other' has recursive side effects.
  bool hasRecursiveEffects =
      const_cast<Operation &>(other)
          .hasTrait<OpTrait::HasRecursiveMemoryEffects>();
  if (hasRecursiveEffects) {
    for (Region &region : const_cast<Operation &>(other).getRegions())
      for (Operation &innerOp : region.getOps())
        if (conflictsWith(innerOp))
          return true;
    return false;
  }

  // If the given operation has side effects, check whether they conflict with
  // the side effects summarized in this class.
  if (auto MEI = dyn_cast<MemoryEffectOpInterface>(other)) {
    OperationSideEffects sideEffects(other, aliasAnalysis, domInfo);

    // Checks for a conflicts on the given resource 'res' by applying the
    // supplied predicate function 'hasConflict'.
    auto checkForConflict =
        [&sideEffects](
            SideEffects::Resource *res,
            std::function<bool(const MemoryEffects::EffectInstance &EI)>
                hasConflict) {
          SmallVector<MemoryEffects::EffectInstance> effects;
          sideEffects.getEffectsOnResource(res, effects);
          return llvm::any_of(
              effects, [hasConflict](const MemoryEffects::EffectInstance &EI) {
                return hasConflict(EI);
              });
        };

    [[maybe_unused]] auto printConflictingSideEffects =
        [](const MemoryEffects::EffectInstance &EI, AliasResult aliasRes,
           const Operation &other) {
          llvm::dbgs().indent(2)
              << "=> found conflicting side effect with: " << other << "\n";
          llvm::dbgs().indent(2) << "=> aliasResult: " << aliasRes << "\n";
        };

    // Check whether the given operation 'other' allocates, writes, or frees a
    // resource that is read by the operation associated with this class.
    if (llvm::any_of(
            readResources, [&](const MemoryEffects::EffectInstance &readRes) {
              auto hasConflict = [&](const MemoryEffects::EffectInstance &EI) {
                if (isa<MemoryEffects::Read>(EI.getEffect()))
                  return false;

                AliasResult aliasRes =
                    const_cast<AliasAnalysis &>(aliasAnalysis)
                        .alias(EI.getValue(), readRes.getValue());
                if (aliasRes.isNo())
                  return false;

                LLVM_DEBUG(printConflictingSideEffects(EI, aliasRes, other));
                return true;
              };

              return checkForConflict(readRes.getResource(), hasConflict);
            })) {
      return true;
    }

    // Check whether the given operation 'other' allocates, reads, writes or
    // frees a resource that is written by the operation associated with this
    // class.
    if (llvm::any_of(
            writeResources, [&](const MemoryEffects::EffectInstance &writeRes) {
              auto hasConflict = [&](const MemoryEffects::EffectInstance &EI) {
                AliasResult aliasRes =
                    const_cast<AliasAnalysis &>(aliasAnalysis)
                        .alias(EI.getValue(), writeRes.getValue());
                if (aliasRes.isNo())
                  return false;

                // An aliased read operation doesn't prevent hoisting if it is
                // dominated by the write operation.
                if (isa<MemoryEffects::Read>(EI.getEffect()) &&
                    domInfo.dominates(const_cast<Operation *>(&op),
                                      const_cast<Operation *>(&other))) {
                  LLVM_DEBUG({
                    printConflictingSideEffects(EI, aliasRes, other);
                    llvm::dbgs().indent(2)
                        << "=> aliased write operation dominates the "
                           "read operation\n";
                  });
                  return false;
                }

                LLVM_DEBUG(printConflictingSideEffects(EI, aliasRes, other));
                return true;
              };

              return checkForConflict(writeRes.getResource(), hasConflict);
            })) {
      return true;
    }

    // Check whether the given operation 'other' allocates, reads, writes or
    // frees a resource that is freed by the operation associated with this
    // class.
    if (llvm::any_of(
            freeResources, [&](const MemoryEffects::EffectInstance &freeRes) {
              auto hasConflict = [&](const MemoryEffects::EffectInstance &EI) {
                AliasResult aliasRes =
                    const_cast<AliasAnalysis &>(aliasAnalysis)
                        .alias(EI.getValue(), freeRes.getValue());
                if (aliasRes.isNo())
                  return false;

                LLVM_DEBUG(printConflictingSideEffects(EI, aliasRes, other));
                return true;
              };

              return checkForConflict(freeRes.getResource(), hasConflict);
            })) {
      return true;
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// LICMCandidate
//===----------------------------------------------------------------------===//

/// This class represents an operation in a loop that is potentially invariant
/// (if a condition is satisfied). The class tracks a list of operations (in the
/// same loop) that need to be hoisted for this operation to be hoisted.
using AccessorType = TypedValue<MemRefType>;
using AccessorPairType = std::pair<AccessorType, AccessorType>;
class LICMCandidate {
public:
  LICMCandidate(Operation &op) : op(op) {}

  Operation &getOperation() const { return op; }

  const SmallVector<Operation *> &getPrerequisites() const {
    return prerequisites;
  }

  void addPrerequisite(Operation &op) { prerequisites.push_back(&op); }

  ArrayRef<AccessorPairType> getRequireNoOverlapAccessorPairs() const {
    return requireNoOverlapAccessorPairs;
  }

  void addRequireNoOverlapAccessorPairs(AccessorType acc1, AccessorType acc2) {
    requireNoOverlapAccessorPairs.push_back({acc1, acc2});
  }

private:
  Operation &op;
  /// Operations that need to be hoisted to allow this operation to be hoisted.
  SmallVector<Operation *> prerequisites;
  /// Pairs of accessors that are required to not overlap for this operation to
  /// be invariant.
  SmallVector<AccessorPairType> requireNoOverlapAccessorPairs;
};

//===----------------------------------------------------------------------===//
// VersionConditionBuilder
//===----------------------------------------------------------------------===//

class VersionConditionBuilder {
public:
  VersionConditionBuilder(
      LoopLikeOpInterface loop,
      ArrayRef<AccessorPairType> requireNoOverlapAccessorPairs)
      : loop(loop), accessorPairs(requireNoOverlapAccessorPairs) {}

  using SCFCondition = LoopVersionCondition::SCFCondition;
  using AffineCondition = LoopVersionCondition::AffineCondition;

  /// The kind of condition to create.
  enum class ConditionKind { SCF, Affine };

  std::unique_ptr<LoopVersionCondition>
  createCondition(ConditionKind condKind) const {
    OpBuilder builder(loop);
    Location loc = loop.getLoc();

    switch (condKind) {
    case ConditionKind::SCF: {
      SCFCondition scfCond = createConditionForSCFLoop(builder, loc);
      return std::make_unique<LoopVersionCondition>(scfCond);
    }
    case ConditionKind::Affine: {
      AffineCondition affineCond = createConditionForAffineLoop(builder, loc);
      return std::make_unique<LoopVersionCondition>(affineCond);
    }

      return nullptr;
    }
  }

private:
  /// Create a loop versioning condition suitable for versioning an SCF loop.
  SCFCondition createConditionForSCFLoop(OpBuilder builder,
                                         Location loc) const {
    auto GetMemref2PointerOp = [&](Value op) {
      auto MT = cast<MemRefType>(op.getType());
      return builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(MT.getElementType(),
                                     MT.getMemorySpaceAsInt()),
          op);
    };

    Value condition;
    for (const AccessorPairType &accessorPair : accessorPairs) {
      Value begin1 = getSYCLAccessorBegin(accessorPair.first, builder, loc);
      Value end1 = getSYCLAccessorEnd(accessorPair.first, builder, loc);
      Value begin2 = getSYCLAccessorBegin(accessorPair.second, builder, loc);
      Value end2 = getSYCLAccessorEnd(accessorPair.second, builder, loc);
      auto beforeCond = builder.create<LLVM::ICmpOp>(
          loc, LLVM::ICmpPredicate::ule, GetMemref2PointerOp(end1),
          GetMemref2PointerOp(begin2));
      auto afterCond = builder.create<LLVM::ICmpOp>(
          loc, LLVM::ICmpPredicate::uge, GetMemref2PointerOp(begin1),
          GetMemref2PointerOp(end2));
      Value orOp = builder.create<arith::OrIOp>(loc, beforeCond, afterCond);
      condition = condition
                      ? builder.create<arith::AndIOp>(loc, condition, orOp)
                      : orOp;
    }
    return condition;
  }

  /// Create a loop versioning condition suitable for versioning an affine loop.
  AffineCondition createConditionForAffineLoop(OpBuilder builder,
                                               Location loc) const {
    llvm_unreachable("TODO");
  }

  template <typename OpTy>
  static OpTy createMethodOp(OpBuilder builder, Location loc, Type resTy,
                             ValueRange arguments, StringRef functionName,
                             StringRef typeName) {
    NamedAttrList attrs;
    SmallVector<Type> argumentTypes;
    for (Value argument : arguments)
      argumentTypes.push_back(argument.getType());
    attrs.set(mlir::sycl::SYCLDialect::getArgumentTypesAttrName(),
              builder.getTypeArrayAttr(argumentTypes));
    attrs.set(mlir::sycl::SYCLDialect::getFunctionNameAttrName(),
              FlatSymbolRefAttr::get(builder.getStringAttr(functionName)));
    attrs.set(mlir::sycl::SYCLDialect::getTypeNameAttrName(),
              FlatSymbolRefAttr::get(builder.getStringAttr(typeName)));
    return builder.create<OpTy>(loc, resTy, ValueRange(arguments), attrs);
  }

  static sycl::SYCLIDGetOp createSYCLIDGetOp(TypedValue<MemRefType> id,
                                             unsigned index, OpBuilder builder,
                                             Location loc) {
    const Value indexOp = builder.create<arith::ConstantIntOp>(loc, index, 32);
    const auto resTy = builder.getIndexType();
    return createMethodOp<sycl::SYCLIDGetOp>(
        builder, loc, MemRefType::get(ShapedType::kDynamic, resTy),
        {id, indexOp}, "operator[]", "id");
  }

  static sycl::SYCLRangeGetOp createSYCLRangeGetOp(TypedValue<MemRefType> range,
                                                   unsigned index,
                                                   OpBuilder builder,
                                                   Location loc) {
    const Value indexOp = builder.create<arith::ConstantIntOp>(loc, index, 32);
    const auto resTy = builder.getIndexType();
    return createMethodOp<sycl::SYCLRangeGetOp>(
        builder, loc, resTy, {range, indexOp}, "get", "range");
  }

  static sycl::SYCLAccessorGetRangeOp
  createSYCLAccessorGetRangeOp(TypedValue<MemRefType> accessor,
                               OpBuilder builder, Location loc) {
    const auto accTy =
        cast<sycl::AccessorType>(accessor.getType().getElementType());
    const auto rangeTy = cast<sycl::RangeType>(
        cast<sycl::AccessorImplDeviceType>(accTy.getBody()[0]).getBody()[1]);
    return createMethodOp<sycl::SYCLAccessorGetRangeOp>(
        builder, loc, rangeTy, accessor, "get_range", "accessor");
  }

  static sycl::SYCLAccessorSubscriptOp
  createSYCLAccessorSubscriptOp(TypedValue<MemRefType> accessor,
                                TypedValue<MemRefType> id, OpBuilder builder,
                                Location loc) {
    const auto accTy =
        cast<sycl::AccessorType>(accessor.getType().getElementType());
    const auto MT = cast<MemRefType>(
        cast<LLVM::LLVMStructType>(accTy.getBody()[1]).getBody()[0]);
    return createMethodOp<sycl::SYCLAccessorSubscriptOp>(
        builder, loc, MT, {accessor, id}, "operator[]", "accessor");
  }

  static Value getSYCLAccessorBegin(TypedValue<MemRefType> accessor,
                                    OpBuilder builder, Location loc) {
    const auto accTy =
        cast<sycl::AccessorType>(accessor.getType().getElementType());
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

  static Value getSYCLAccessorEnd(TypedValue<MemRefType> accessor,
                                  OpBuilder builder, Location loc) {
    const auto accTy =
        cast<sycl::AccessorType>(accessor.getType().getElementType());
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

  mutable LoopLikeOpInterface loop;
  ArrayRef<AccessorPairType> accessorPairs;
};

/// Return the accessor used by \p op if found, and nullptr otherwise.
static Optional<AccessorType> getAccessorUsedByOperation(const Operation &op) {
  auto getMemrefOp = [](const Operation &op) {
    return TypeSwitch<const Operation &, Operation *>(op)
        .Case<AffineLoadOp, AffineStoreOp>(
            [](auto &affineOp) { return affineOp.getMemref().getDefiningOp(); })
        .Default([](auto &) { return nullptr; });
  };

  auto accSub =
      dyn_cast_or_null<sycl::SYCLAccessorSubscriptOp>(getMemrefOp(op));
  if (accSub)
    return accSub.getAcc();
  return std::nullopt;
}

/// Determine whether any operation in the \p loop has a conflict with the
/// given operation in LICMCandidate \p candidate that prevents hoisting the
/// operation out of the loop. Operations that are already known to have no
/// hoisting preventing conflicts in the loop are given in \p willBeMoved.
static bool hasConflictsInLoop(LICMCandidate &candidate,
                               LoopLikeOpInterface loop,
                               const SmallPtrSetImpl<Operation *> &willBeMoved,
                               const AliasAnalysis &aliasAnalysis,
                               const DominanceInfo &domInfo) {
  Operation &op = candidate.getOperation();
  const OperationSideEffects sideEffects(op, aliasAnalysis, domInfo);

  // For parallel loop, only check for conflicts with other previous operations
  // in the same block.
  Operation *point =
      (isa<scf::ParallelOp, AffineParallelOp>(loop)) ? &op : nullptr;
  for (Operation &other : *op.getBlock()) {
    if (point && !other.isBeforeInBlock(point))
      break;
    if (!sideEffects.conflictsWith(other))
      continue;

    if (willBeMoved.count(&other)) {
      candidate.addPrerequisite(other);
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "can be hoisted: conflicting operation will be hoisted\n");
      continue;
    }

    Optional<AccessorType> opAccessor = getAccessorUsedByOperation(op);
    Optional<AccessorType> otherAccessor = getAccessorUsedByOperation(other);
    if (opAccessor.has_value() && otherAccessor.has_value())
      if (*opAccessor != *otherAccessor &&
          loop.isDefinedOutsideOfLoop(*opAccessor) &&
          loop.isDefinedOutsideOfLoop(*otherAccessor)) {
        candidate.addRequireNoOverlapAccessorPairs(*opAccessor, *otherAccessor);
        LLVM_DEBUG(llvm::dbgs().indent(2)
                   << "can be hoisted: require loop versioning\n");
        continue;
      }

    return true;
  }

  // Check whether the parent operation has conflicts on the loop.
  if (op.getParentOp() == loop)
    return false;
  LICMCandidate parentCandidate(*op.getParentOp());
  if (hasConflictsInLoop(parentCandidate, loop, willBeMoved, aliasAnalysis,
                         domInfo))
    return true;

  // If the parent operation is not guaranteed to execute its
  // (single-block) region once, walk the block.
  bool conflict = false;
  if (!isa<scf::IfOp, AffineIfOp, memref::AllocaScopeOp>(op)) {
    op.walk([&](Operation *in) {
      if (willBeMoved.count(in))
        candidate.addPrerequisite(*in);
      else if (sideEffects.conflictsWith(*in)) {
        LLVM_DEBUG(llvm::dbgs().indent(2)
                   << "conflicting operation: " << *in << "\n");
        conflict = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }

  return conflict;
}

/// Returns true if the operation in LICMCandidate \p candidate can be hoisted
/// out of the given loop \p loop. The \p willBeMoved argument represents
/// operations that are known to be loop invariant (and therefore will be moved
/// outside of the loop).
static bool canBeHoisted(LICMCandidate &candidate, LoopLikeOpInterface loop,
                         const SmallPtrSetImpl<Operation *> &willBeMoved,
                         const AliasAnalysis &aliasAnalysis,
                         const DominanceInfo &domInfo) {
  // Returns true if the given value can be moved outside of the loop, and
  // false otherwise. A value cannot be moved outside of the loop if its
  // operands are not defined outside of the loop and cannot themselves be
  // moved.
  auto canBeMoved = [&](Value value) {
    if (loop.isDefinedOutsideOfLoop(value))
      return true;
    Operation *definingOp = value.getDefiningOp();
    if (auto BA = dyn_cast<BlockArgument>(value))
      definingOp = BA.getOwner()->getParentOp();
    if (definingOp && willBeMoved.count(definingOp)) {
      candidate.addPrerequisite(*definingOp);
      return true;
    }
    return false;
  };

  // Operations with unknown side effects cannot be hoisted.
  Operation &op = candidate.getOperation();
  if (!isa<MemoryEffectOpInterface>(op) &&
      !op.hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    LLVM_DEBUG({
      llvm::dbgs() << "Operation: " << op << "\n";
      llvm::dbgs().indent(2)
          << "**** cannot be hoisted: unknown side effects\n\n";
    });
    return false;
  }

  // Ensure operands can be hoisted.
  if (llvm::any_of(op.getOperands(),
                   [&](Value value) { return !canBeMoved(value); })) {
    LLVM_DEBUG({
      llvm::dbgs() << "Operation: " << op << "\n";
      llvm::dbgs().indent(2)
          << "**** cannot be hoisted: operand(s) can't be hoisted\n\n";
    });
    return false;
  }

  // If the operation has no side effects it can be hoisted.
  if (isMemoryEffectFree(&op)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Operation: " << op << "\n";
      llvm::dbgs().indent(2) << "**** can be hoisted: has no side effects\n\n";
    });
    return true;
  }

  // Do not hoist operations that allocate a resource.
  const OperationSideEffects sideEffects(op, aliasAnalysis, domInfo);
  if (sideEffects.allocatesResource()) {
    LLVM_DEBUG({
      llvm::dbgs() << "Operation: " << op << "\n";
      llvm::dbgs().indent(2)
          << "**** cannot be hoisted: operation allocates a resource\n\n";
    });
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << sideEffects);

  // If the operation has side effects, check whether other operations in the
  // loop prevent hosting it.
  if ((sideEffects.readsFromResource() || sideEffects.writesToResource() ||
       sideEffects.freesResource()) &&
      hasConflictsInLoop(candidate, loop, willBeMoved, aliasAnalysis,
                         domInfo)) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "**** cannot be hoisted: found conflicting operation\n\n");
    return false;
  }

  // Recurse into the regions for this op and check whether the contained ops
  // can be hoisted. We can inductively assume that this op will have its
  // block args available outside the loop.
  SmallPtrSet<Operation *, 2> willBeMoved2(willBeMoved.begin(),
                                           willBeMoved.end());
  willBeMoved2.insert(&op);

  for (Region &region : op.getRegions()) {
    for (Operation &innerOp : region.getOps()) {
      LICMCandidate innerCandidate(innerOp);
      if (!canBeHoisted(innerCandidate, loop, willBeMoved2, aliasAnalysis,
                        domInfo))
        return false;
      willBeMoved2.insert(&innerOp);
    }
  }

  LLVM_DEBUG(llvm::dbgs().indent(2)
             << "**** can be hoisted: no conflicts found\n\n");

  return true;
}

/// Populate \p LICMCandidates with operations that can be hoisted out of the
/// given loop \p loop.
static void
collectHoistableOperations(LoopLikeOpInterface loop,
                           const AliasAnalysis &aliasAnalysis,
                           const DominanceInfo &domInfo,
                           SmallVectorImpl<LICMCandidate> &LICMCandidates) {
  assert(LICMCandidates.empty() && "Expecting empty LICMCandidates");
  // Do not use walk here, as we do not want to go into nested regions and
  // hoist operations from there. These regions might have semantics unknown
  // to this rewriting. If the nested regions are loops, they will have been
  // processed.
  SmallVector<LICMCandidate> LICMPotentialCandidates;
  SmallPtrSet<Operation *, 8> willBeMoved;
  for (Block &block : loop.getLoopBody()) {
    for (Operation &op : block.without_terminator()) {
      LICMCandidate candidate(op);
      if (!canBeHoisted(candidate, loop, willBeMoved, aliasAnalysis, domInfo))
        continue;
      LICMPotentialCandidates.push_back(candidate);
      willBeMoved.insert(&op);
    }
  }

  // Some candidate operations require the loop to be versioned. If the loop is
  // not versioned because of heuristic considerations (e.g., exceeded the
  // versioning limit), then we have to filter out the operations that depend
  // upon them.
  size_t numVersion = 0;
  std::set<const Operation *> opsToHoist;
  for (const LICMCandidate &candidate : LICMPotentialCandidates) {
    // Cannot hoist if any of its prerequisites are not hoisted.
    if (any_of(candidate.getPrerequisites(),
               [&opsToHoist](Operation *prerequisite) {
                 return !opsToHoist.count(prerequisite);
               }))
      continue;

    ArrayRef<AccessorPairType> accessorPairs =
        candidate.getRequireNoOverlapAccessorPairs();
    bool requireVersioning = !accessorPairs.empty();
    // Currently only version for single accessor pair.
    bool willVersion = requireVersioning && EnableLICMSYCLAccessorVersioning &&
                       numVersion < LICMVersionLimit &&
                       accessorPairs.size() <= LICMSYCLAccessorPairsLimit;
    if (willVersion)
      ++numVersion;
    else if (requireVersioning)
      // Cannot hoist if not version.
      continue;

    opsToHoist.insert(&candidate.getOperation());
    LICMCandidates.push_back(candidate);
  }
}

static size_t moveLoopInvariantCode(LoopLikeOpInterface loop,
                                    const AliasAnalysis &aliasAnalysis,
                                    const DominanceInfo &domInfo) {
  Operation *loopOp = loop;
  if (!isa<scf::ForOp, scf::ParallelOp, AffineParallelOp, AffineForOp>(loopOp))
    return 0;

  SmallVector<LICMCandidate> LICMCandidates;
  collectHoistableOperations(loop, aliasAnalysis, domInfo, LICMCandidates);
  if (LICMCandidates.empty())
    return 0;

  LoopTools loopTools;
  loopTools.guardLoop(loop);

  size_t numOpsHoisted = 0;
  std::set<const Operation *> opsHoisted;
  for (const LICMCandidate &candidate : LICMCandidates) {
    ArrayRef<AccessorPairType> accessorPairs =
        candidate.getRequireNoOverlapAccessorPairs();
    if (!accessorPairs.empty()) {
      OpBuilder builder(loop);
      std::unique_ptr<LoopVersionCondition> condition =
          VersionConditionBuilder(loop, accessorPairs)
              .createCondition(VersionConditionBuilder::ConditionKind::SCF);
      loopTools.versionLoop(loop, *condition);
    }

    loop.moveOutOfLoop(&candidate.getOperation());
    ++numOpsHoisted;
  }

  return numOpsHoisted;
}

void LICM::runOnOperation() {
  DominanceInfo &domInfo = getAnalysis<DominanceInfo>();
  AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
  aliasAnalysis.addAnalysisImplementation(sycl::AliasAnalysis(relaxedAliasing));

  getOperation()->walk([&](LoopLikeOpInterface loop) {
    LLVM_DEBUG({
      llvm::dbgs() << "----------------\n";
      loop.print(llvm::dbgs() << "Original loop:\n");
      llvm::dbgs() << "\nIn:\n"
                   << *loop->getParentOfType<FunctionOpInterface>() << "\n";
    });

    // First use MLIR LICM to hoist simple operations.
    {
      size_t OpHoisted = moveLoopInvariantCode(loop);

      LLVM_DEBUG({
        llvm::dbgs() << "\nMLIR LICM hoisted " << OpHoisted
                     << " operation(s).\n";
        if (OpHoisted) {
          loop.print(llvm::dbgs() << "Loop after MLIR LICM:\n");
          llvm::dbgs() << "\nIn:\n"
                       << *loop->getParentOfType<FunctionOpInterface>() << "\n";
          assert(mlir::verify(loop->getParentOfType<FunctionOpInterface>())
                     .succeeded());
        }
        llvm::dbgs() << "----------------\n";
      });
    }

    // Now use this pass to hoist more complex operations.
    {
      size_t OpHoisted = moveLoopInvariantCode(loop, aliasAnalysis, domInfo);
      numOpHoisted += OpHoisted;

      LLVM_DEBUG({
        llvm::dbgs() << "\nLICM hoisted " << OpHoisted << " operation(s).\n";
        if (OpHoisted) {
          loop.print(llvm::dbgs() << "Loop after LICM:\n");
          llvm::dbgs() << "\nIn:\n"
                       << *loop->getParentOfType<FunctionOpInterface>() << "\n";
          assert(mlir::verify(loop->getParentOfType<FunctionOpInterface>())
                     .succeeded());
        }
        llvm::dbgs() << "----------------\n";
      });

      DEBUG_WITH_TYPE(REPORT_DEBUG_TYPE, {
        if (OpHoisted)
          llvm::dbgs() << "LICM: hoisted " << OpHoisted
                       << " operations(s) in : "
                       << loop->getParentOfType<FunctionOpInterface>().getName()
                       << "\n";
      });
    }
  });
}

std::unique_ptr<Pass> mlir::polygeist::createLICMPass() {
  return std::make_unique<LICM>();
}

std::unique_ptr<Pass>
mlir::polygeist::createLICMPass(const LICMOptions &options) {
  return std::make_unique<LICM>(options);
}
