//===- ConstantPropagation.cpp - Host-device constant propagation pass ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SYCL/Analysis/SYCLAccessorAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/SYCLIDAndRangeAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/SYCLNDRangeAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>

#include <numeric>

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
// Forward declaration
class ConstantArg;

/// Helper struct for std::visit
template <typename... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};
template <typename... Ts> overloaded(Ts...) -> overloaded<Ts...>;

class ConstantPropagationPass
    : public mlir::sycl::impl::ConstantPropagationPassBase<
          ConstantPropagationPass> {
public:
  using ConstantPropagationPassBase<
      ConstantPropagationPass>::ConstantPropagationPassBase;

  void runOnOperation() final;

private:
  /// Inserts in \p constants all of the constant arguments of \p op.
  static SmallVectorImpl<std::unique_ptr<ConstantArg>> &getConstantExplicitArgs(
      SmallVectorImpl<std::unique_ptr<ConstantArg>> &constants,
      SYCLHostScheduleKernel op, ArrayRef<Type> argumentTypes,
      ArrayAttr argAttrs, SymbolTableCollection &symbolTable,
      SYCLAccessorAnalysis &accessorAnalysis);

  /// Inserts in \p constants all of the constant implicit arguments of \p op.
  ///
  /// \p ndrAnalysis and \p idrAnalysis are used to perform analysis and track
  /// constant values of arguments.
  static SmallVectorImpl<std::unique_ptr<ConstantArg>> &getConstantImplicitArgs(
      SmallVectorImpl<std::unique_ptr<ConstantArg>> &constants,
      SYCLHostScheduleKernel op, SYCLNDRangeAnalysis &ndrAnalysis,
      SYCLIDAndRangeAnalysis &idrAnalysis);

  /// Returns a list with all constant arguments (both implicit and explicit).
  static SmallVector<std::unique_ptr<ConstantArg>>
  getConstantArgs(SYCLHostScheduleKernel op, ArrayRef<Type> argumentTypes,
                  ArrayAttr argAttrs, SymbolTableCollection &symbolTable,
                  SYCLAccessorAnalysis &accessorAnalysis,
                  SYCLNDRangeAnalysis &ndrAnalysis,
                  SYCLIDAndRangeAnalysis &idrAnalysis);

  /// Propagate constants in \p constants to the function launched by \p launch.
  void propagateConstantArgs(
      SmallVectorImpl<std::unique_ptr<ConstantArg>> &&constants,
      gpu::GPUFuncOp op, SYCLHostScheduleKernel launch);
};

class ConstantArg {
public:
  enum class Kind {
    ConstantArithArg,
    ConstantArrayArg,
    ConstantAccessorArg,
    ConstantSYCLGridArgs,
    OpArgEnd = ConstantAccessorArg,
    ExplicitEnd = ConstantSYCLGridArgs,
    ImplicitBegin = ExplicitEnd
  };

  Kind getKind() const { return kind; }

  size_t getNumHits() { return hits; }

protected:
  explicit ConstantArg(Kind kind) : kind(kind) {}

  void recordHit() { ++hits; }

private:
  Kind kind;
  size_t hits = 0;
};

/// Class representing a constant explicit argument, i.e., those kernel
/// functions receive as a parameter.
class ConstantExplicitArgBase : public ConstantArg {
public:
  unsigned getIndex() const { return index; }

  static bool classof(const ConstantArg *arg) {
    return arg->getKind() < ConstantArg::Kind::ExplicitEnd;
  }

  void propagate(OpBuilder &builder, Region &region);

protected:
  ConstantExplicitArgBase(ConstantArg::Kind kind, unsigned index)
      : ConstantArg(kind), index(index) {
    assert(kind < ConstantArg::Kind::ExplicitEnd && "Invalid kind");
  }

private:
  unsigned index;
};

/// Class representing a constant argument backed by an operation in host code
class ConstantOpArg : public ConstantExplicitArgBase {
public:
  static bool classof(const ConstantArg *c) {
    return c->getKind() < ConstantArg::Kind::OpArgEnd;
  }

protected:
  ConstantOpArg(ConstantArg::Kind kind, unsigned index, Operation *definingOp)
      : ConstantExplicitArgBase(kind, index), definingOp(definingOp) {
    assert(definingOp && "Expecting valid operation");
  }

  Operation *getDefiningOp() const { return definingOp; }

  template <typename OpTy> OpTy getDefiningOp() const {
    return cast<OpTy>(definingOp);
  }

private:
  Operation *definingOp;
};

/// Class representing a constant arithmetic argument.
class ConstantArithArg : public ConstantOpArg {
public:
  ConstantArithArg(unsigned index, Operation *definingOp)
      : ConstantOpArg(ConstantArg::Kind::ConstantArithArg, index, definingOp) {
    assert(definingOp->hasTrait<OpTrait::ConstantLike>() &&
           "Expecting constant");
  }

  /// Propagate the constant this argument represents.
  void propagate(OpBuilder &builder, Region &region) {
    Value toReplace = region.getArgument(getIndex());
    toReplace.replaceAllUsesWith(builder.clone(*getDefiningOp())->getResult(0));
    recordHit();
  }

  static bool classof(const ConstantArg *c) {
    return c->getKind() == ConstantArg::Kind::ConstantArithArg;
  }
};

/// Class representing a constant array argument.
class ConstantArrayArg : public ConstantOpArg {
public:
  ConstantArrayArg(unsigned index, LLVM::GlobalOp definingOp)
      : ConstantOpArg(ConstantArg::Kind::ConstantArrayArg, index, definingOp) {}

  /// Propagate the constant this argument represents.
  void propagate(OpBuilder &builder, Region &region);

  static bool classof(const ConstantArg *c) {
    return c->getKind() == ConstantArg::Kind::ConstantArrayArg;
  }
};

/// Class representing a constant implicit argument.
class ConstantImplicitArgBase : public ConstantArg {
public:
  Kind getKind() const { return kind; }

  void propagate(FunctionOpInterface func);

  static bool classof(const ConstantArg *arg) {
    return arg->getKind() >= ConstantArg::Kind::ImplicitBegin;
  }

protected:
  explicit ConstantImplicitArgBase(ConstantArg::Kind kind) : ConstantArg(kind) {
    assert(kind >= ConstantArg::Kind::ImplicitBegin && "Invalid kind");
  }

private:
  static FunctionOpInterface cloneFunction(FunctionOpInterface original,
                                           CallOpInterface call,
                                           Twine callPath);

  bool isRecursive() const;

  void propagate(FunctionOpInterface func, SymbolTableCollection &symbolTable,
                 bool recursive, Twine callPath = Twine(),
                 CallOpInterface call = nullptr);

  void printDebugMessage() const;

  bool needsChanges(FunctionOpInterface func,
                    SymbolTableCollection &symbolTable, bool recursive) const;

  LogicalResult match(Operation *op) const;
  void rewrite(Operation *op, OpBuilder &builder) const;

  Kind kind;
};

/// The default offset, i.e., `sycl::id<Dimension>()`
class DefaultOffset {};

static Type get1DType(Type mt) {
  return MemRefType::get(1, cast<MemRefType>(mt).getElementType());
}

/// Creates a constant id/range.
template <typename OpTy, typename RangeTy>
static Value createIDRange(OpBuilder &builder, Location loc, Type type,
                           RangeTy components) {
  SmallVector<Value, 3> indices;
  llvm::transform(components, std::back_inserter(indices), [&](size_t i) {
    return builder.create<arith::ConstantIndexOp>(loc, i);
  });
  Value result = builder.create<OpTy>(loc, get1DType(type), indices);
  return result.getType() == type
             ? result
             : builder.create<memref::CastOp>(loc, type, result);
}

class OffsetInfo {
public:
  OffsetInfo(IDRangeInformation &&info) : info(std::move(info)) {}
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
  std::optional<IDRangeInformation> info;
};

class RangeAndOffsetInfo {
public:
  /// Provide only range info
  RangeAndOffsetInfo(IDRangeInformation &&rangeInfo)
      : rangeInfo(std::move(rangeInfo)) {}

  /// Provide range and offset info
  RangeAndOffsetInfo(IDRangeInformation &&rangeInfo, OffsetInfo &&offsetInfo)
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
  std::optional<IDRangeInformation> rangeInfo;
  std::optional<OffsetInfo> offsetInfo;
};

class NDRInfo {
public:
  NDRInfo(NDRangeInformation &&info) : info(std::move(info)) {}

  template <typename... Args>
  NDRInfo(Args &&...args)
      : info(std::in_place, std::in_place_type<RangeAndOffsetInfo>,
             std::forward<Args>(args)...) {}

  bool hasConstantGlobalSizeInfo() const {
    return info &&
           std::visit(overloaded{
                          [](const NDRangeInformation &info) {
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
        overloaded{[&](const NDRangeInformation &info) {
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
                          [](const NDRangeInformation &info) {
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
        overloaded{[&](const NDRangeInformation &info) {
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
    return info && std::visit(overloaded{
                                  [](const NDRangeInformation &info) {
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
        overloaded{[&](const NDRangeInformation &info) {
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
    return info && std::holds_alternative<NDRangeInformation>(*info);
  }

  const NDRangeInformation &getNDRange() const {
    assert(isNDRange() && "Not an nd-range");
    return std::get<NDRangeInformation>(*info);
  }

private:
  std::optional<std::variant<NDRangeInformation, RangeAndOffsetInfo>> info;
};

class ConstantSYCLGridArgs : public ConstantImplicitArgBase {
public:
  static bool classof(const ConstantArg *c) {
    return c->getKind() == ConstantArg::Kind::ConstantSYCLGridArgs;
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

/// Class representing an accessor with constant members.
class ConstantAccessorArg : public ConstantExplicitArgBase {
public:
  static std::unique_ptr<ConstantAccessorArg> get(unsigned index,
                                                  AccessorInformation &&info) {
    return std::unique_ptr<ConstantAccessorArg>(
        isValidInfo(info) ? new ConstantAccessorArg(index, std::move(info))
                          : nullptr);
  }

  /// Propagate the constant this argument represents.
  void propagate(OpBuilder &builder, Region &region);

  static bool classof(const ConstantArg *c) {
    return c->getKind() == ConstantArg::Kind::ConstantAccessorArg;
  }

private:
  /// Return whether \p can be used to replace arguments.
  static bool isValidInfo(const AccessorInformation &info);

  ConstantAccessorArg(unsigned index, AccessorInformation &&info)
      : ConstantExplicitArgBase(ConstantArg::Kind::ConstantAccessorArg, index),
        info(std::move(info)) {}

  AccessorInformation info;
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
        return rewriter->rewrite(cast<typename std::remove_pointer_t<decltype(
                                     rewriter)>::operation_type>(op),
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
  const NDRangeInformation &ndrInfo = info.getNDRange();
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
// ConstantExplicitArgBase
//===----------------------------------------------------------------------===//

void ConstantExplicitArgBase::propagate(OpBuilder &builder, Region &region) {
  TypeSwitch<ConstantExplicitArgBase *>(this)
      .Case<ConstantArithArg, ConstantArrayArg, ConstantAccessorArg>(
          [&](auto *arg) { arg->propagate(builder, region); });
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
// ConstantAccessorArg
//===----------------------------------------------------------------------===//

bool ConstantAccessorArg::isValidInfo(const AccessorInformation &info) {
  return !info.needsSubRange() || info.hasConstantSubRange() ||
         (info.hasBufferInformation() &&
          info.getBufferInfo().hasConstantSize()) ||
         !info.needsOffset() || info.hasConstantOffset();
}

/// Propagate the constant this argument represents.
void ConstantAccessorArg::propagate(OpBuilder &builder, Region &region) {
  // NOTE: Modifications to this function that extend this propagator
  // capabilities may need parallel modifications to
  // ConstantAccessorArg::isValidInfo.

  constexpr unsigned accessRangeMemberOffset = 1;
  constexpr unsigned memoryRangeMemberOffset = 2;
  constexpr unsigned offsetMemberOffset = 3;

  Location loc = region.getLoc();
  unsigned index = getIndex();

  const auto replace = [&](unsigned offset, auto createF) {
    Value toReplace = region.getArgument(index + offset);
    Value newValue = createF(toReplace.getType());
    toReplace.replaceAllUsesWith(newValue);
    recordHit();
  };

  LLVM_DEBUG({
    llvm::dbgs().indent(2) << "Propagating ";
    if (info.isLocalAccessor())
      llvm::dbgs() << "local_";
    llvm::dbgs() << "accessor at position #" << index << "\n";
  });

  if (info.needsSubRange()) {
    if (info.hasConstantSubRange()) {
      ArrayRef<size_t> accessRange = info.getConstantRange();
      LLVM_DEBUG({
        llvm::dbgs().indent(4)
            << "Constant access range provided on construction: <";
        llvm::interleaveComma(accessRange, llvm::dbgs());
        llvm::dbgs() << ">\n";
      });
      replace(accessRangeMemberOffset, [&](Type type) {
        return createIDRange<SYCLRangeConstructorOp>(builder, loc, type,
                                                     accessRange);
      });
    }
  } else {
    LLVM_DEBUG(
        llvm::dbgs().indent(4)
        << "No access range provided on construction: using memory range\n");
    replace(accessRangeMemberOffset, [&](Type) {
      return region.getArgument(index + memoryRangeMemberOffset);
    });
  }

  if (info.isLocalAccessor()) {
    replace(memoryRangeMemberOffset, [&](Type type) {
      // Memory range is initialized to 0.
      SmallVector<size_t> dimensions(getDimensions(type), 0);
      return createIDRange<SYCLRangeConstructorOp>(builder, loc, type,
                                                   dimensions);
    });
  } else if (info.hasBufferInformation()) {
    const BufferInformation &bufferInfo = info.getBufferInfo();
    if (bufferInfo.hasConstantSize()) {
      ArrayRef<size_t> memoryRange = bufferInfo.getConstantSize();
      LLVM_DEBUG({
        llvm::dbgs().indent(4)
            << "Constant memory range provided on construction: <";
        llvm::interleaveComma(memoryRange, llvm::dbgs());
        llvm::dbgs() << ">\n";
      });
      replace(memoryRangeMemberOffset, [&](Type type) {
        return createIDRange<SYCLRangeConstructorOp>(builder, loc, type,
                                                     memoryRange);
      });
    }
  }

  if (info.needsOffset()) {
    if (info.hasConstantOffset()) {
      ArrayRef<size_t> offset = info.getConstantOffset();
      LLVM_DEBUG({
        llvm::dbgs().indent(4) << "Constant offset provided on construction: <";
        llvm::interleaveComma(offset, llvm::dbgs());
        llvm::dbgs() << ">\n";
      });
      replace(offsetMemberOffset, [&](Type type) {
        return createIDRange<SYCLIDConstructorOp>(builder, loc, type, offset);
      });
    }
  } else {
    LLVM_DEBUG(llvm::dbgs().indent(4)
               << "No offset provided on construction: using default offset\n");
    replace(offsetMemberOffset, [&](Type type) {
      return createIDRange<SYCLIDConstructorOp>(builder, loc, type,
                                                ArrayRef<size_t>());
    });
  }
}

//===----------------------------------------------------------------------===//
// ConstantArrayArg
//===----------------------------------------------------------------------===//

void ConstantArrayArg::propagate(OpBuilder &builder, Region &region) {
  unsigned index = getIndex();
  Value toReplace = region.getArgument(index);
  OpBuilder globalCloner(region.getParentOfType<gpu::GPUFuncOp>());
  auto newGlobal =
      globalCloner.cloneWithoutRegions(getDefiningOp<LLVM::GlobalOp>());
  LLVM_DEBUG({
    llvm::dbgs().indent(2) << "Handling constant array at pos #" << index
                           << "\n";
    llvm::dbgs().indent(4) << "Replacing using new global " << newGlobal
                           << "\n";
  });
  Location loc = toReplace.getLoc();
  Value addressof = builder.create<LLVM::AddressOfOp>(
      loc, LLVM::LLVMPointerType::get(newGlobal.getContext()),
      newGlobal.getSymNameAttr());
  toReplace.replaceAllUsesWith(addressof);
  recordHit();
}

//===----------------------------------------------------------------------===//
// ConstantPropagationPass
//===----------------------------------------------------------------------===//

template <typename OpTy> static OpTy getUniqueUserOp(Value range) {
  auto allUsersOfType = llvm::make_filter_range(
      range.getUsers(), [](Operation *op) { return isa<OpTy>(op); });
  if (!llvm::hasSingleElement(allUsersOfType))
    return nullptr;
  return cast<OpTy>(*allUsersOfType.begin());
}

static std::optional<NDRInfo>
getNDRangeInformation(SYCLNDRangeAnalysis &ndrAnalysis,
                      SYCLIDAndRangeAnalysis &idrAnalysis,
                      SYCLHostScheduleKernel op) {
  Value range = op.getRange();
  if (!range)
    return std::nullopt;

  auto setNDRangeOp = getUniqueUserOp<SYCLHostHandlerSetNDRange>(range);
  if (!setNDRangeOp) {
    LLVM_DEBUG(llvm::dbgs().indent(6)
               << "Could not find unique 'sycl.host.nd_range.set_nd_range' "
                  "operation\n");
    return std::nullopt;
  }

  if (op.getNdRange()) {
    // `parallel_for(nd_range, ...)` case
    std::optional<NDRangeInformation> ndrInfo =
        ndrAnalysis.getNDRangeInformationFromConstruction(setNDRangeOp, range);
    if (!ndrInfo)
      return std::nullopt;
    LLVM_DEBUG(llvm::dbgs().indent(6)
               << "ND-range information: " << *ndrInfo << "\n");
    return std::move(*ndrInfo);
  }

  // `parallel_for(range, [offset], ...)` case
  std::optional<IDRangeInformation> rangeInfo =
      idrAnalysis.getIDRangeInformationFromConstruction<RangeType>(setNDRangeOp,
                                                                   range);

  auto offsetInfo = [&]() -> std::optional<OffsetInfo> {
    Value offset = op.getOffset();
    if (!offset)
      return DefaultOffset();
    return idrAnalysis.getIDRangeInformationFromConstruction<IDType>(
        setNDRangeOp, offset);
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
getSYCLGridPropagator(SYCLNDRangeAnalysis &ndrAnalysis,
                      SYCLIDAndRangeAnalysis &idrAnalysis,
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

static std::unique_ptr<ConstantAccessorArg>
getConstantAccessorArg(SYCLHostScheduleKernel op,
                       SYCLAccessorAnalysis &accessorAnalysis, unsigned index,
                       Value value) {
  LLVM_DEBUG(llvm::dbgs().indent(2)
             << "Handling accessor at operand #" << index << "\n");

  auto setCapturedOp = getUniqueUserOp<SYCLHostSetCaptured>(value);
  if (!setCapturedOp) {
    LLVM_DEBUG(llvm::dbgs().indent(6)
               << "Could not find unique 'sycl.host.set_captured' operation\n");
    return nullptr;
  }

  std::optional<AccessorInformation> accInfo =
      accessorAnalysis.getAccessorInformationFromConstruction(setCapturedOp,
                                                              value);
  if (!accInfo) {
    LLVM_DEBUG(llvm::dbgs().indent(4)
               << "Could not get accessor analysis information\n");
    return nullptr;
  }

  auto result = ConstantAccessorArg::get(index, std::move(*accInfo));

  LLVM_DEBUG({
    llvm::dbgs().indent(4) << "Accessor analysis information:\n" << *accInfo;
    if (!result)
      llvm::dbgs().indent(4)
          << "Not enough information; not performing accessor propagation\n";
  });

  return result;
}

template <typename RangeTy>
static unsigned getTrueIndex(const RangeTy &typeAttrs, unsigned originalIndex) {
  auto begin = typeAttrs.begin();
  return std::accumulate<decltype(begin), unsigned>(
      begin, std::next(begin, originalIndex), 0,
      [](unsigned index, TypeAttr attr) {
        Type type = attr.getValue();
        unsigned offset =
            TypeSwitch<Type, unsigned>(type)
                .Case<AccessorType, LocalAccessorType>([](auto) { return 4; })
                .template Case<NoneType>([](auto) { return 1; });
        return index + offset;
      });
}

static LLVM::GlobalOp
getConstantArrayDefinition(Value value, SymbolTableCollection &symbolTable) {
  auto addressOf = value.getDefiningOp<LLVM::AddressOfOp>();
  if (!addressOf)
    return nullptr;
  LLVM::GlobalOp global = addressOf.getGlobal(symbolTable);
  return global.getConstant() &&
                 isa<LLVM::LLVMArrayType>(global.getGlobalType())
             ? global
             : nullptr;
}

static LLVM::LLVMArrayType getArrayType(Type type) {
  auto st = llvm::dyn_cast_or_null<LLVM::LLVMStructType>(type);
  if (!st)
    return LLVM::LLVMArrayType();
  ArrayRef<Type> body = st.getBody();
  return body.size() == 1 ? dyn_cast<LLVM::LLVMArrayType>(body.front())
                          : LLVM::LLVMArrayType();
}

static bool isArrayArg(Type type, DictionaryAttr attrs) {
  constexpr auto isConstantArrayArgAttr = [](Attribute attr) {
    return attr &&
           TypeSwitch<Attribute, bool>(attr)
               .Case<TypeAttr>([](auto attr) -> bool {
                 return static_cast<bool>(getArrayType(attr.getValue()));
               })
               .Default(false);
  };
  return attrs && isa<LLVM::LLVMPointerType>(type) &&
         isConstantArrayArgAttr(
             attrs.get(LLVM::LLVMDialect::getByValAttrName()));
}

static bool canInitializeArgument(LLVM::GlobalOp arrayDefinitionOp,
                                  Type arrayType) {
  return arrayDefinitionOp.getGlobalType() == getArrayType(arrayType);
}

SmallVectorImpl<std::unique_ptr<ConstantArg>> &
ConstantPropagationPass::getConstantExplicitArgs(
    SmallVectorImpl<std::unique_ptr<ConstantArg>> &constants,
    SYCLHostScheduleKernel op, ArrayRef<Type> argumentTypes, ArrayAttr argAttrs,
    SymbolTableCollection &symbolTable,
    SYCLAccessorAnalysis &accessorAnalysis) {
  // Map each argument to a ConstantExplicitArgBase
  auto isConstant = m_Constant();
  auto types = op.getSyclTypes().getAsRange<TypeAttr>();
  for (auto [index, valueTypeIter] :
       llvm::enumerate(llvm::zip(op.getArgs(), types))) {
    auto [value, type] = valueTypeIter;
    unsigned trueIndex = getTrueIndex(types, index);
    if (matchPattern(value, isConstant)) {
      if (value.getType() == argumentTypes[trueIndex]) {
        LLVM_DEBUG(llvm::dbgs().indent(2) << "Arith constant: " << value
                                          << " at pos #" << trueIndex << "\n");
        constants.push_back(std::make_unique<ConstantArithArg>(
            trueIndex, value.getDefiningOp()));
      }
    }
    auto currArgAttrs =
        argAttrs ? cast<DictionaryAttr>(argAttrs[trueIndex]) : DictionaryAttr();
    if (currArgAttrs && isArrayArg(argumentTypes[trueIndex], currArgAttrs)) {
      if (LLVM::GlobalOp arrayDefinitionOp =
              getConstantArrayDefinition(value, symbolTable);
          arrayDefinitionOp &&
          canInitializeArgument(
              arrayDefinitionOp,
              cast<TypeAttr>(
                  currArgAttrs.get(LLVM::LLVMDialect::getByValAttrName()))
                  .getValue())) {
        LLVM_DEBUG(llvm::dbgs().indent(2)
                   << "Array constant: " << arrayDefinitionOp << " at pos #"
                   << trueIndex << "\n");
        constants.push_back(
            std::make_unique<ConstantArrayArg>(trueIndex, arrayDefinitionOp));
      }
    }
    TypeSwitch<Type>(type.getValue())
        .Case<AccessorType, LocalAccessorType>([&, value = value](auto) {
          std::unique_ptr<ConstantArg> accArg =
              getConstantAccessorArg(op, accessorAnalysis, trueIndex, value);
          if (accArg)
            constants.push_back(std::move(accArg));
        });
  }
  return constants;
}

SmallVectorImpl<std::unique_ptr<ConstantArg>> &
ConstantPropagationPass::getConstantImplicitArgs(
    SmallVectorImpl<std::unique_ptr<ConstantArg>> &constants,
    SYCLHostScheduleKernel op, SYCLNDRangeAnalysis &ndrAnalysis,
    SYCLIDAndRangeAnalysis &idrAnalysis) {
  LLVM_DEBUG(llvm::dbgs().indent(2)
             << "Searching for constant implicit arguments\n");
  if (std::optional<ConstantSYCLGridArgs> gridConstantPropagation =
          getSYCLGridPropagator(ndrAnalysis, idrAnalysis, op)) {
    constants.push_back(std::make_unique<ConstantSYCLGridArgs>(
        std::move(*gridConstantPropagation)));
  } else {
    LLVM_DEBUG(llvm::dbgs().indent(6)
               << "No constant nd-range information available\n");
  }
  return constants;
}

SmallVector<std::unique_ptr<ConstantArg>>
ConstantPropagationPass::getConstantArgs(
    SYCLHostScheduleKernel op, ArrayRef<Type> argumentTypes, ArrayAttr argAttrs,
    SymbolTableCollection &symbolTable, SYCLAccessorAnalysis &accessorAnalysis,
    SYCLNDRangeAnalysis &ndrAnalysis, SYCLIDAndRangeAnalysis &idrAnalysis) {
  SmallVector<std::unique_ptr<ConstantArg>> constants;
  getConstantExplicitArgs(constants, op, argumentTypes, argAttrs, symbolTable,
                          accessorAnalysis);
  getConstantImplicitArgs(constants, op, ndrAnalysis, idrAnalysis);
  return constants;
}

void ConstantPropagationPass::propagateConstantArgs(
    SmallVectorImpl<std::unique_ptr<ConstantArg>> &&constants,
    gpu::GPUFuncOp op, SYCLHostScheduleKernel launch) {
  Region *region = op.getCallableRegion();
  OpBuilder builder(region);
  for (std::unique_ptr<ConstantArg> &arg : constants) {
    Pass::Statistic *concreteStatPtr =
        TypeSwitch<ConstantArg *, Pass::Statistic *>(arg.get())
            .Case<ConstantExplicitArgBase>([&](auto *arg) {
              arg->propagate(builder, *region);
              return &NumReplacedExplicitArguments;
            })
            .Case<ConstantImplicitArgBase>([&](auto *arg) {
              arg->propagate(op);
              return &NumReplacedImplicitArguments;
            });
    size_t hits = arg->getNumHits();
    auto &concreteStat = *concreteStatPtr;
    concreteStat += hits;
    NumPropagatedConstants += hits;
  }
}

void ConstantPropagationPass::runOnOperation() {
  SymbolTableCollection symbolTable;
  auto &ndrAnalysis =
      getAnalysis<SYCLNDRangeAnalysis>().initialize(relaxedAliasing);
  auto &idrAnalysis =
      getAnalysis<SYCLIDAndRangeAnalysis>().initialize(relaxedAliasing);
  auto &accessorAnalysis =
      getAnalysis<SYCLAccessorAnalysis>().initialize(relaxedAliasing);
  getOperation()->walk([&](gpu::GPUFuncOp op) {
    LLVM_DEBUG(llvm::dbgs()
               << "Performing constant propagation on function '"
               << static_cast<SymbolOpInterface>(op).getName() << "'\n");
    if (!op.isKernel() || op.isExternal()) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "Early exit: Function is not a kernel\n");
      return;
    }

    auto module = op->getParentOfType<ModuleOp>();
    std::optional<SymbolTable::UseRange> optUses =
        SymbolTable::getSymbolUses(op, module);
    if (!optUses) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "SYCL constant propagation pass expects at least one "
                    "kernel launch point\n");
      return;
    }

    auto schedKernelOps = llvm::make_filter_range(
        llvm::map_range(*optUses,
                        [](const SymbolTable::SymbolUse &use) {
                          return dyn_cast<SYCLHostScheduleKernel>(
                              use.getUser());
                        }),
        llvm::identity<SYCLHostScheduleKernel>());

    if (!llvm::hasSingleElement(schedKernelOps)) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "SYCL constant propagation pass expects a "
                    "single kernel launch point\n");
      return;
    }

    SYCLHostScheduleKernel launchPoint = *schedKernelOps.begin();
    if (!launchPoint) {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "SYCL constant propagation pass expects launch "
                    "point to be 'sycl.host.schedule_kernel'\n");
      return;
    }

    propagateConstantArgs(getConstantArgs(launchPoint, op.getArgumentTypes(),
                                          op.getArgAttrsAttr(), symbolTable,
                                          accessorAnalysis, ndrAnalysis,
                                          idrAnalysis),
                          op, launchPoint);
  });
}

std::unique_ptr<Pass> sycl::createConstantPropagationPass(
    const ConstantPropagationPassOptions &options) {
  return std::make_unique<ConstantPropagationPass>(options);
}
