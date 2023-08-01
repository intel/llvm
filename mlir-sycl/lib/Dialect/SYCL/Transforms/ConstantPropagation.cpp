//===- ConstantPropagation.cpp - Host-device constant propagation pass ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Polygeist/Analysis/SYCLIDAndRangeAnalysis.h"
#include "mlir/Dialect/Polygeist/Analysis/SYCLNDRangeAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>

namespace mlir {
namespace sycl {
#define GEN_PASS_DEF_CONSTANTPROPAGATIONPASS
#include "mlir/Dialect/SYCL/Transforms/Passes.h.inc"
} // namespace sycl
} // namespace mlir

#define DEBUG_TYPE "sycl-constant-propagation"

using namespace mlir;
using namespace mlir::sycl;

namespace {
class ConstantSYCLGridArgs;
/// Helper struct for std::visit
template <typename... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};
template <typename... Ts> overloaded(Ts...) -> overloaded<Ts...>;

class ConstantImplicitArgBase;
class ConstantPropagationPass
    : public mlir::sycl::impl::ConstantPropagationPassBase<
          ConstantPropagationPass> {
public:
  using ConstantPropagationPassBase<
      ConstantPropagationPass>::ConstantPropagationPassBase;

  void runOnOperation() final;

private:
  /// Return a range with all of the constant arguments of \p op.
  static auto getConstantArgs(SYCLHostScheduleKernel op);

  /// Return all of the constant implicit arguments of \p op.
  ///
  /// \p ndrAnalysis and \p idrAnalysis are used to perform analysis and track
  /// constant values of arguments.
  static SmallVector<std::unique_ptr<ConstantImplicitArgBase>>
  getConstantImplicitArgs(SYCLHostScheduleKernel op,
                          polygeist::SYCLNDRangeAnalysis &ndrAnalysis,
                          polygeist::SYCLIDAndRangeAnalysis &idrAnalysis);

  /// Propagate constants in \p constants to the function launched by \p launch.
  template <typename RangeTy>
  void propagateConstantArgs(RangeTy constants, gpu::GPUFuncOp op,
                             SYCLHostScheduleKernel launch);

  /// Propagate implicit constants in \p constants to function \p op.
  template <typename RangeTy>
  void propagateImplicitConstantArgs(RangeTy constants, FunctionOpInterface op);
};

/// Class representing a constant explicit argument, i.e., those kernel
/// functions receive as a parameter.
class ConstantExplicitArg {
public:
  enum class Kind { ConstantArithArg };

  unsigned getIndex() const { return index; }
  Kind getKind() const { return kind; }

protected:
  ConstantExplicitArg(Kind kind, unsigned index) : kind(kind), index(index) {}

private:
  Kind kind;
  unsigned index;
};

/// Class representing a constant arithmetic argument.
class ConstantArithArg : public ConstantExplicitArg {
public:
  ConstantArithArg(unsigned index, Operation *definingOp)
      : ConstantExplicitArg(ConstantExplicitArg::Kind::ConstantArithArg, index),
        definingOp(definingOp) {
    assert(definingOp && "Expecting valid operation");
    assert(definingOp->hasTrait<OpTrait::ConstantLike>() &&
           "Expecting constant");
  }

  /// Propagate the constant this argument represents.
  void propagate(OpBuilder builder, Region &region) const {
    Value toReplace = region.getArgument(getIndex());
    toReplace.replaceAllUsesWith(builder.clone(*definingOp)->getResult(0));
  }

  static bool classof(const ConstantExplicitArg *c) {
    return c->getKind() == ConstantExplicitArg::Kind::ConstantArithArg;
  }

private:
  Operation *definingOp;
};

/// Class representing a constant implicit argument.
class ConstantImplicitArgBase {
public:
  enum class Kind {
    ConstantSYCLGridArgs,
  };

  Kind getKind() const { return kind; }

  size_t getNumHits() const { return hits; }

  void propagate(FunctionOpInterface func);

protected:
  explicit ConstantImplicitArgBase(Kind kind) : kind(kind) {}

private:
  static FunctionOpInterface cloneFunction(FunctionOpInterface original,
                                           CallOpInterface call,
                                           Twine callPath);

  bool isRecursive() const;

  void propagate(FunctionOpInterface func, SymbolTableCollection &symbolTable,
                 bool recursive, Twine callPath = Twine(),
                 CallOpInterface call = nullptr);

  void printDebugMessage() const;

  void recordHit() { ++hits; }

  bool needsChanges(FunctionOpInterface func,
                    SymbolTableCollection &symbolTable, bool recursive) const;

  LogicalResult match(Operation *op) const;
  void rewrite(Operation *op, OpBuilder &builder) const;

  Kind kind;
  size_t hits = 0;
};

/// The default offset, i.e., `sycl::id<Dimension>()`
class DefaultOffset {};

/// Creates a constant id/range.
template <typename OpTy, typename RangeTy>
static Value createIDRange(OpBuilder &builder, Location loc, Type type,
                           RangeTy components) {
  SmallVector<Value, 3> indices;
  llvm::transform(components, std::back_inserter(indices), [&](size_t i) {
    return builder.create<arith::ConstantIndexOp>(loc, i);
  });
  return builder.create<OpTy>(loc, type, indices);
}

class OffsetInfo {
public:
  OffsetInfo(polygeist::IDRangeInformation &&info) : info(std::move(info)) {}
  OffsetInfo(DefaultOffset) {}

  bool isConstant() const { return !info || info->isConstant(); }

  Value getValue(OpBuilder &builder, Location loc, Type type) const {
    assert(isConstant() && "Expecting constant info");
    auto components = info ? ArrayRef<size_t>(info->getConstantValues())
                           : ArrayRef<size_t>(std::nullopt);
    return createIDRange<SYCLIDConstructorOp>(builder, loc, type, components);
  }

  friend raw_ostream &operator<<(raw_ostream &os, const OffsetInfo &info) {
    if (info.info)
      return os << *info.info;
    return os << "<Default>";
  }

private:
  /// Offset information.
  std::optional<polygeist::IDRangeInformation> info;
};

class RangeAndOffsetInfo {
public:
  /// Provide only range info
  RangeAndOffsetInfo(polygeist::IDRangeInformation &&rangeInfo)
      : rangeInfo(std::move(rangeInfo)) {}

  /// Provide range and offset info
  RangeAndOffsetInfo(polygeist::IDRangeInformation &&rangeInfo,
                     OffsetInfo &&offsetInfo)
      : rangeInfo(std::move(rangeInfo)), offsetInfo(std::move(offsetInfo)) {}

  /// Provide range and offset info
  RangeAndOffsetInfo(OffsetInfo &&offsetInfo)
      : offsetInfo(std::move(offsetInfo)) {}

  bool hasConstantRange() const { return rangeInfo && rangeInfo->isConstant(); }

  Value getRangeValue(OpBuilder &builder, Location loc, Type type) const {
    assert(hasConstantRange() && "Expecting constant range");
    return createIDRange<SYCLRangeConstructorOp>(
        builder, loc, type, rangeInfo->getConstantValues());
  }

  bool hasConstantOffset() const {
    return offsetInfo && offsetInfo->isConstant();
  }

  Value getOffsetValue(OpBuilder &builder, Location loc, Type type) const {
    assert(hasConstantOffset() && "Expecting constant offset");
    return offsetInfo->getValue(builder, loc, type);
  }

private:
  std::optional<polygeist::IDRangeInformation> rangeInfo;
  std::optional<OffsetInfo> offsetInfo;
};

class NDRInfo {
public:
  NDRInfo(polygeist::NDRangeInformation &&info) : info(std::move(info)) {}

  template <typename... Args>
  NDRInfo(Args &&...args)
      : info(std::in_place, std::in_place_type<RangeAndOffsetInfo>,
             std::forward<Args>(args)...) {}

  bool hasConstantGlobalSizeInfo() const {
    return info &&
           std::visit(overloaded{
                          [](const polygeist::NDRangeInformation &info) {
                            return info.getGlobalSizeInfo().isConstant();
                          },
                          [](const RangeAndOffsetInfo &info) {
                            return info.hasConstantRange();
                          },
                      },
                      *info);
  }

  Value getConstantGlobalSizeValue(OpBuilder &builder, Location loc,
                                   Type type) const {
    assert(hasConstantGlobalSizeInfo() &&
           "Constant global size information not present");
    return std::visit(
        overloaded{[&](const polygeist::NDRangeInformation &info) {
                     return createIDRange<SYCLRangeConstructorOp>(
                         builder, loc, type,
                         info.getGlobalSizeInfo().getConstantValues());
                   },
                   [&](const RangeAndOffsetInfo &info) {
                     return info.getRangeValue(builder, loc, type);
                   }},
        *info);
  }

  bool hasConstantLocalSizeInfo() const {
    return info &&
           std::visit(overloaded{
                          [](const polygeist::NDRangeInformation &info) {
                            return info.getLocalSizeInfo().isConstant();
                          },
                          [](const RangeAndOffsetInfo &) { return false; },
                      },
                      *info);
  }

  Value getConstantLocalSizeValue(OpBuilder &builder, Location loc,
                                  Type type) const {
    assert(hasConstantLocalSizeInfo() &&
           "Constant local size information not present");
    return std::visit(
        overloaded{[&](const polygeist::NDRangeInformation &info) {
                     return createIDRange<SYCLRangeConstructorOp>(
                         builder, loc, type,
                         info.getLocalSizeInfo().getConstantValues());
                   },
                   [](const RangeAndOffsetInfo &) -> Value {
                     llvm_unreachable("Local size not available");
                   }},
        *info);
  }

  bool hasConstantOffsetInfo() const {
    return info &&
           std::visit(overloaded{
                          [](const polygeist::NDRangeInformation &info) {
                            return info.getOffsetInfo().isConstant();
                          },
                          [](const RangeAndOffsetInfo &info) {
                            return info.hasConstantOffset();
                          },
                      },
                      *info);
  }

  Value getConstantOffsetValue(OpBuilder &builder, Location loc,
                               Type type) const {
    assert(hasConstantOffsetInfo() &&
           "Constant offset information not present");
    return std::visit(
        overloaded{[&](const polygeist::NDRangeInformation &info) {
                     return createIDRange<SYCLIDConstructorOp>(
                         builder, loc, type,
                         info.getOffsetInfo().getConstantValues());
                   },
                   [&](const RangeAndOffsetInfo &info) {
                     return info.getOffsetValue(builder, loc, type);
                   }},
        *info);
  }

  bool isNDRange() const {
    return info && std::holds_alternative<polygeist::NDRangeInformation>(*info);
  }

  const polygeist::NDRangeInformation &getNDRange() const {
    assert(isNDRange() && "Not an nd-range");
    return std::get<polygeist::NDRangeInformation>(*info);
  }

private:
  std::optional<std::variant<polygeist::NDRangeInformation, RangeAndOffsetInfo>>
      info;
};

class ConstantSYCLGridArgs : public ConstantImplicitArgBase {
public:
  static bool classof(const ConstantImplicitArgBase *c) {
    return c->getKind() == ConstantImplicitArgBase::Kind::ConstantSYCLGridArgs;
  }

  static std::optional<ConstantSYCLGridArgs> get(const NDRInfo &info) {
    SmallVector<std::unique_ptr<RewriterBase>, 4> rewriters =
        getRewriters(info);
    if (rewriters.empty())
      return std::nullopt;
    return ConstantSYCLGridArgs(info, std::move(rewriters));
  }

  constexpr bool isRecursive() const { return true; }

  LogicalResult match(Operation *op) const;
  void rewrite(Operation *op, OpBuilder &builder) const;

  void printDebugMessage() const;

private:
  class RewriterBase {
  public:
    enum class Kind {
      NumWorkItemsRewriter,
      NumWorkGroupsRewriter,
      WorkGroupSizeRewriter,
      GlobalOffsetRewriter,
    };

    Kind getKind() const { return kind; }

    LogicalResult match(Operation *op) const;
    void rewrite(Operation *op, const NDRInfo &info, OpBuilder &builder) const;

  protected:
    explicit RewriterBase(Kind kind) : kind(kind) {}

  private:
    Kind kind;
  };

  template <typename OpTy, RewriterBase::Kind K>
  class Rewriter : public RewriterBase {
  public:
    using operation_type = OpTy;

    static bool classof(const RewriterBase *w) { return w->getKind() == K; }

    Rewriter() : RewriterBase(K) {}

    LogicalResult match(Operation *op) const { return success(isa<OpTy>(op)); }
    void rewrite(OpTy op, const NDRInfo &info, OpBuilder &builder) const;

  private:
    Value getNewValue(OpBuilder &builder, Location loc, const NDRInfo &info,
                      Type type) const;
  };

  using NumWorkItemsRewriter =
      Rewriter<SYCLNumWorkItemsOp, RewriterBase::Kind::NumWorkItemsRewriter>;
  using NumWorkGroupsRewriter =
      Rewriter<SYCLNumWorkGroupsOp, RewriterBase::Kind::NumWorkGroupsRewriter>;
  using WorkGroupSizeRewriter =
      Rewriter<SYCLWorkGroupSizeOp, RewriterBase::Kind::WorkGroupSizeRewriter>;
  using GlobalOffsetRewriter =
      Rewriter<SYCLGlobalOffsetOp, RewriterBase::Kind::GlobalOffsetRewriter>;

  ConstantSYCLGridArgs(
      const NDRInfo &info,
      SmallVectorImpl<std::unique_ptr<RewriterBase>> &&rewriters)
      : ConstantImplicitArgBase(
            ConstantImplicitArgBase::Kind::ConstantSYCLGridArgs),
        info(info), rewriters(std::move(rewriters)) {}

  static SmallVector<std::unique_ptr<RewriterBase>, 4>
  getRewriters(const NDRInfo &info) {
    SmallVector<std::unique_ptr<RewriterBase>, 4> writers;

    bool hasLocalSizeInfo = info.hasConstantLocalSizeInfo();
    bool hasGlobalSizeInfo = info.hasConstantGlobalSizeInfo();
    if (hasGlobalSizeInfo) {
      writers.push_back(std::make_unique<NumWorkItemsRewriter>());
      if (hasLocalSizeInfo)
        writers.push_back(std::make_unique<NumWorkGroupsRewriter>());
    }
    if (hasLocalSizeInfo)
      writers.push_back(std::make_unique<WorkGroupSizeRewriter>());
    if (info.hasConstantOffsetInfo())
      writers.push_back(std::make_unique<GlobalOffsetRewriter>());

    return writers;
  }

  NDRInfo info;
  SmallVector<std::unique_ptr<RewriterBase>, 4> rewriters;
};
} // namespace

//===----------------------------------------------------------------------===//
// ConstantSYCLGridArgs::RewriterBase
//===----------------------------------------------------------------------===//

LogicalResult ConstantSYCLGridArgs::RewriterBase::match(Operation *op) const {
  return TypeSwitch<const RewriterBase *, LogicalResult>(this)
      .Case<NumWorkItemsRewriter, NumWorkGroupsRewriter, WorkGroupSizeRewriter,
            GlobalOffsetRewriter>(
          [=](const auto *rewriter) { return rewriter->match(op); });
}

void ConstantSYCLGridArgs::RewriterBase::rewrite(Operation *op,
                                                 const NDRInfo &info,
                                                 OpBuilder &builder) const {
  TypeSwitch<const RewriterBase *>(this)
      .Case<NumWorkItemsRewriter, NumWorkGroupsRewriter, WorkGroupSizeRewriter,
            GlobalOffsetRewriter>([&](const auto *rewriter) {
        return rewriter->rewrite(cast<typename std::remove_pointer_t<
                                     decltype(rewriter)>::operation_type>(op),
                                 info, builder);
      });
}

//===----------------------------------------------------------------------===//
// ConstantSYCLGridArgs::Rewriter
//===----------------------------------------------------------------------===//

template <typename OpTy, ConstantSYCLGridArgs::RewriterBase::Kind K>
void ConstantSYCLGridArgs::Rewriter<OpTy, K>::rewrite(
    OpTy op, const NDRInfo &info, OpBuilder &builder) const {
  Value oldValue = op;
  Location loc = op->getLoc();
  Type type = oldValue.getType();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value allocatedValue =
      getNewValue(builder, loc, info, MemRefType::get(1, type));
  Value value = builder.create<memref::LoadOp>(loc, type, allocatedValue, zero);
  oldValue.replaceAllUsesWith(value);
  LLVM_DEBUG({
    llvm::dbgs().indent(8) << "Replaced:\n";
    llvm::dbgs().indent(10) << oldValue << "\n";
    llvm::dbgs().indent(8) << "with:\n";
    llvm::dbgs().indent(10) << zero << "\n";
    for (Value operand : allocatedValue.getDefiningOp()->getOperands())
      llvm::dbgs().indent(10) << operand << "\n";
    llvm::dbgs().indent(10) << allocatedValue << "\n";
    llvm::dbgs().indent(10) << value << "\n";
  });
  op->erase();
}

template <>
Value ConstantSYCLGridArgs::NumWorkItemsRewriter::getNewValue(
    OpBuilder &builder, Location loc, const NDRInfo &info, Type type) const {
  return info.getConstantGlobalSizeValue(builder, loc, type);
}

template <>
Value ConstantSYCLGridArgs::NumWorkGroupsRewriter::getNewValue(
    OpBuilder &builder, Location loc, const NDRInfo &info, Type type) const {
  const polygeist::NDRangeInformation &ndrInfo = info.getNDRange();
  return createIDRange<SYCLRangeConstructorOp>(
      builder, loc, type,
      llvm::map_range(llvm::zip(ndrInfo.getGlobalSizeInfo().getConstantValues(),
                                ndrInfo.getLocalSizeInfo().getConstantValues()),
                      [&](auto iter) {
                        auto [globalSize, localSize] = iter;
                        return globalSize / localSize;
                      }));
}

template <>
Value ConstantSYCLGridArgs::WorkGroupSizeRewriter::getNewValue(
    OpBuilder &builder, Location loc, const NDRInfo &info, Type type) const {
  return info.getConstantLocalSizeValue(builder, loc, type);
}

template <>
Value ConstantSYCLGridArgs::GlobalOffsetRewriter::getNewValue(
    OpBuilder &builder, Location loc, const NDRInfo &info, Type type) const {
  return info.getConstantOffsetValue(builder, loc, type);
}

//===----------------------------------------------------------------------===//
// ConstantImplicitArgBase
//===----------------------------------------------------------------------===//

FunctionOpInterface
ConstantImplicitArgBase::cloneFunction(FunctionOpInterface original,
                                       CallOpInterface call, Twine callPath) {
  OpBuilder functionCloner(original);
  auto clone = cast<FunctionOpInterface>(functionCloner.clone(*original));
  clone->setAttr(
      SymbolTable::getSymbolAttrName(),
      functionCloner.getAttr<StringAttr>(
          Twine("__").concat(original.getName()).concat(callPath).str()));
  call.setCalleeFromCallable(
      FlatSymbolRefAttr::get(static_cast<SymbolOpInterface>(clone)));
  return clone;
}

LogicalResult ConstantImplicitArgBase::match(Operation *op) const {
  return TypeSwitch<const ConstantImplicitArgBase *, LogicalResult>(this)
      .Case<ConstantSYCLGridArgs>(
          [=](const auto *propagator) { return propagator->match(op); });
}

void ConstantImplicitArgBase::rewrite(Operation *op, OpBuilder &builder) const {
  TypeSwitch<const ConstantImplicitArgBase *>(this).Case<ConstantSYCLGridArgs>(
      [&](const auto *propagator) { return propagator->rewrite(op, builder); });
}

bool ConstantImplicitArgBase::isRecursive() const {
  return TypeSwitch<const ConstantImplicitArgBase *, bool>(this)
      .Case<ConstantSYCLGridArgs>(
          [](const auto *ptr) { return ptr->isRecursive(); });
}

void ConstantImplicitArgBase::printDebugMessage() const {
  TypeSwitch<const ConstantImplicitArgBase *>(this).Case<ConstantSYCLGridArgs>(
      [](const auto *ptr) { return ptr->printDebugMessage(); });
}

void ConstantImplicitArgBase::propagate(FunctionOpInterface func) {
  LLVM_DEBUG(printDebugMessage());
  SymbolTableCollection symbolTable;
  propagate(func, symbolTable, isRecursive());
}

bool ConstantImplicitArgBase::needsChanges(FunctionOpInterface func,
                                           SymbolTableCollection &symbolTable,
                                           bool recursive) const {
  // Declaration
  if (func.isExternal())
    return false;
  return func
      .walk([&](Operation *op) {
        if (succeeded(match(op)))
          return WalkResult::interrupt();
        if (!recursive)
          return WalkResult::advance();
        auto call = dyn_cast<CallOpInterface>(op);
        if (!call)
          return WalkResult::advance();
        SymbolRefAttr callable =
            call.getCallableForCallee().dyn_cast<SymbolRefAttr>();
        if (!callable)
          return WalkResult::advance();
        auto called = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
            call, callable);
        assert(called && "Function not found");
        return needsChanges(called, symbolTable, recursive)
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

void ConstantImplicitArgBase::propagate(FunctionOpInterface func,
                                        SymbolTableCollection &symbolTable,
                                        bool recursive, Twine callPath,
                                        CallOpInterface call) {
  LLVM_DEBUG(llvm::dbgs().indent(6)
             << "Entering function '"
             << static_cast<SymbolOpInterface>(func).getName() << "'\n");
  // Declaration
  if (func.isExternal())
    return;
  StringRef originalName = static_cast<SymbolOpInterface>(func).getName();
  // Need to clone called functions to specialize
  if (call) {
    // Avoid cloning if no change will be performed.
    // Only check whether func needs changes here to avoid iterating twice.
    if (!needsChanges(func, symbolTable, recursive))
      return;
    func = cloneFunction(func, call, callPath);
    LLVM_DEBUG(llvm::dbgs().indent(8)
               << "Cloned into '"
               << static_cast<SymbolOpInterface>(func).getName()
               << "' to specialize\n");
  }
  OpBuilder builder(func.getFunctionBody());
  func.walk([&](Operation *op) {
    if (succeeded(match(op))) {
      rewrite(op, builder);
      recordHit();
      return;
    }
    if (!recursive)
      return;
    auto call = dyn_cast<CallOpInterface>(op);
    if (!call)
      return;
    SymbolRefAttr callable =
        call.getCallableForCallee().dyn_cast<SymbolRefAttr>();
    if (!callable)
      return;
    auto called = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
        call, callable);
    assert(called && "Function not found");
    propagate(called, symbolTable, recursive,
              callPath.concat("_").concat(originalName), call);
  });
}

//===----------------------------------------------------------------------===//
// ConstantSYCLGridArgs
//===----------------------------------------------------------------------===//

void ConstantSYCLGridArgs::printDebugMessage() const {
  LLVM_DEBUG(llvm::dbgs().indent(4) << "Replacing SYCL grid arguments\n");
}

LogicalResult ConstantSYCLGridArgs::match(Operation *op) const {
  return success(llvm::any_of(
      rewriters, [=](const std::unique_ptr<RewriterBase> &rewriter) {
        return succeeded(rewriter->match(op));
      }));
}

void ConstantSYCLGridArgs::rewrite(Operation *op, OpBuilder &builder) const {
  const auto *rewriter = llvm::find_if(
      rewriters, [=](const std::unique_ptr<RewriterBase> &rewriter) {
        return succeeded(rewriter->match(op));
      });
  assert(rewriter != rewriters.end() && "Expecting a match");
  (*rewriter)->rewrite(op, info, builder);
}

//===----------------------------------------------------------------------===//
// ConstantPropagationPass
//===----------------------------------------------------------------------===//

static std::optional<NDRInfo>
getNDRangeInformation(polygeist::SYCLNDRangeAnalysis &ndrAnalysis,
                      polygeist::SYCLIDAndRangeAnalysis &idrAnalysis,
                      SYCLHostScheduleKernel op) {
  Value range = op.getRange();
  if (!range)
    return std::nullopt;

  if (op.getNdRange()) {
    // `parallel_for(nd_range, ...)` case
    std::optional<polygeist::NDRangeInformation> ndrInfo =
        ndrAnalysis.getNDRangeInformationFromConstruction(op, range);
    if (!ndrInfo)
      return std::nullopt;
    LLVM_DEBUG(llvm::dbgs().indent(6)
               << "ND-range information: " << *ndrInfo << "\n");
    return std::move(*ndrInfo);
  }

  // `parallel_for(range, [offset], ...)` case
  std::optional<polygeist::IDRangeInformation> rangeInfo =
      idrAnalysis.getIDRangeInformationFromConstruction<RangeType>(op, range);

  auto offsetInfo = [&]() -> std::optional<OffsetInfo> {
    Value offset = op.getOffset();
    if (!offset)
      return DefaultOffset();
    return idrAnalysis.getIDRangeInformationFromConstruction<IDType>(op,
                                                                     offset);
  }();

  if (rangeInfo) {
    LLVM_DEBUG(llvm::dbgs().indent(6)
               << "range information: " << *rangeInfo << "\n");
    if (offsetInfo) {
      LLVM_DEBUG(llvm::dbgs().indent(6)
                 << "offset information: " << *offsetInfo << "\n");
      return NDRInfo(std::move(*rangeInfo), std::move(*offsetInfo));
    }
    LLVM_DEBUG(llvm::dbgs().indent(6) << "no offset information\n");
    return std::move(*rangeInfo);
  }
  if (!offsetInfo)
    return std::nullopt;
  LLVM_DEBUG(llvm::dbgs().indent(6) << "no range information\n");
  LLVM_DEBUG(llvm::dbgs().indent(6)
             << "offset information: " << *offsetInfo << "\n");
  return std::move(*offsetInfo);
}

static std::optional<ConstantSYCLGridArgs>
getSYCLGridPropagator(polygeist::SYCLNDRangeAnalysis &ndrAnalysis,
                      polygeist::SYCLIDAndRangeAnalysis &idrAnalysis,
                      SYCLHostScheduleKernel op) {
  LLVM_DEBUG(llvm::dbgs().indent(4)
             << "Searching for constant SYCL grid arguments\n");
  std::optional<NDRInfo> ndrInfo =
      getNDRangeInformation(ndrAnalysis, idrAnalysis, op);
  if (!ndrInfo) {
    LLVM_DEBUG(llvm::dbgs().indent(6) << "No nd-range information available\n");
    return std::nullopt;
  }
  return ConstantSYCLGridArgs::get(*ndrInfo);
}

auto ConstantPropagationPass::getConstantArgs(SYCLHostScheduleKernel op) {
  LLVM_DEBUG(llvm::dbgs().indent(2) << "Searching for constant arguments\n");
  auto isConstant = m_Constant();
  // Map each argument to a ConstantExplicitArg and filter-out nullptr
  // (non-const) ones.
  return llvm::make_filter_range(
      llvm::map_range(llvm::enumerate(op.getArgs()),
                      [&](auto iter) -> std::unique_ptr<ConstantExplicitArg> {
                        Value value = iter.value();
                        if (matchPattern(value, isConstant)) {
                          LLVM_DEBUG(llvm::dbgs().indent(4)
                                     << "Arith constant: " << value
                                     << " at pos " << iter.index() << "\n");
                          return std::make_unique<ConstantArithArg>(
                              iter.index(), value.getDefiningOp());
                        }
                        return nullptr;
                      }),
      llvm::identity<std::unique_ptr<ConstantExplicitArg>>());
}

SmallVector<std::unique_ptr<ConstantImplicitArgBase>>
ConstantPropagationPass::getConstantImplicitArgs(
    SYCLHostScheduleKernel op, polygeist::SYCLNDRangeAnalysis &ndrAnalysis,
    polygeist::SYCLIDAndRangeAnalysis &idrAnalysis) {
  LLVM_DEBUG(llvm::dbgs().indent(2)
             << "Searching for constant implicit arguments\n");
  SmallVector<std::unique_ptr<ConstantImplicitArgBase>> args;
  if (std::optional<ConstantSYCLGridArgs> gridConstantPropagation =
          getSYCLGridPropagator(ndrAnalysis, idrAnalysis, op)) {
    args.push_back(std::make_unique<ConstantSYCLGridArgs>(
        std::move(*gridConstantPropagation)));
  } else {
    LLVM_DEBUG(llvm::dbgs().indent(6)
               << "No constant nd-range information available\n");
  }
  return args;
}

template <typename RangeTy>
void ConstantPropagationPass::propagateConstantArgs(
    RangeTy constants, gpu::GPUFuncOp op, SYCLHostScheduleKernel launch) {
  Region *region = op.getCallableRegion();
  OpBuilder builder(region);
  for (const std::unique_ptr<ConstantExplicitArg> &arg : constants) {
    TypeSwitch<ConstantExplicitArg *>(arg.get()).Case<ConstantArithArg>(
        [&](const auto *arith) { arith->propagate(builder, *region); });
    ++NumReplacedExplicitArguments;
  }
  NumPropagatedConstants += NumReplacedExplicitArguments;
}

template <typename RangeTy>
void ConstantPropagationPass::propagateImplicitConstantArgs(
    RangeTy constants, FunctionOpInterface op) {
  for (std::unique_ptr<ConstantImplicitArgBase> &arg : constants) {
    arg->propagate(op);
    size_t hits = arg->getNumHits();
    NumReplacedImplicitArguments += hits;
    NumPropagatedConstants += hits;
  }
}

void ConstantPropagationPass::runOnOperation() {
  auto &ndrAnalysis =
      getAnalysis<polygeist::SYCLNDRangeAnalysis>().initialize(relaxedAliasing);
  auto &idrAnalysis =
      getAnalysis<polygeist::SYCLIDAndRangeAnalysis>().initialize(
          relaxedAliasing);
  getOperation()->walk([&](gpu::GPUFuncOp op) {
    LLVM_DEBUG(llvm::dbgs()
               << "Performing constant propagation on function '"
               << static_cast<SymbolOpInterface>(op).getName() << "'\n");
    if (!op.isKernel() || op.isExternal()) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "Early exit: Function is not a kernel\n");
      return;
    }

    constexpr auto isSingleton = [](auto range) {
      return std::next(range.begin()) == range.end();
    };

    auto module = op->getParentOfType<ModuleOp>();
    std::optional<SymbolTable::UseRange> uses =
        SymbolTable::getSymbolUses(op, module);
    if (!(uses && isSingleton(*uses))) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "SYCL constant propagation pass expects a "
                    "single kernel launch point\n");
      return;
    }

    auto launchPoint =
        dyn_cast<SYCLHostScheduleKernel>(uses->begin()->getUser());
    if (!launchPoint) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "SYCL constant propagation pass expects launch "
                    "point to be 'sycl.host.schedule_kernel'\n");
    }

    propagateConstantArgs(getConstantArgs(launchPoint), op, launchPoint);
    propagateImplicitConstantArgs(
        getConstantImplicitArgs(launchPoint, ndrAnalysis, idrAnalysis), op);
  });
}

std::unique_ptr<Pass> sycl::createConstantPropagationPass(
    const ConstantPropagationPassOptions &options) {
  return std::make_unique<ConstantPropagationPass>(options);
}
