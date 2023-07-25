//===- CPP.cpp - Host-device constant propagation pass --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>

namespace mlir {
namespace sycl {
#define GEN_PASS_DEF_CONSTANTPROPAGATIONPASS
#include "mlir/Dialect/SYCL/Transforms/Passes.h.inc"
} // namespace sycl
} // namespace mlir

#define DEBUG_TYPE "sycl-cpp"

using namespace mlir;
using namespace mlir::sycl;

static raw_ostream &operator<<(raw_ostream &os, const llvm::BitVector &bv) {
  os << "{";
  SmallVector<unsigned> set;
  for (unsigned i = 0, size = static_cast<unsigned>(bv.size()); i < size; ++i) {
    if (bv.test(i))
      set.push_back(i);
  }
  llvm::interleaveComma(set, os);
  return os << "}";
}

namespace {
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

  /// Propagate constants in \p range to the function launched by \p launch.
  template <typename RangeTy>
  void propagateConstantArgs(RangeTy constants, SYCLHostScheduleKernel launch);

  /// Drop arguments \p constants from \p launch.
  template <typename RangeTy>
  void dropConstantArgs(RangeTy constants, Builder &builder,
                        SYCLHostScheduleKernel launch);
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
  void propagate(OpBuilder builder, Region &region) const;

  static bool classof(const ConstantExplicitArg *c) {
    return c->getKind() == ConstantExplicitArg::Kind::ConstantArithArg;
  }

private:
  Operation *definingOp;
};
} // namespace

//===----------------------------------------------------------------------===//
// ConstantPropagationPass
//===----------------------------------------------------------------------===//

void ConstantArithArg::propagate(OpBuilder builder, Region &region) const {
  Value toReplace = region.getArgument(getIndex());
  toReplace.replaceAllUsesWith(builder.clone(*definingOp)->getResult(0));
}

auto ConstantPropagationPass::getConstantArgs(SYCLHostScheduleKernel op) {
  auto isConstant = m_Constant();
  // Map each argument to a ConstantExplicitArg and filter-out nullptr
  // (non-const) ones.
  return llvm::make_filter_range(
      llvm::map_range(llvm::enumerate(op.getArgs()),
                      [&](auto iter) -> std::unique_ptr<ConstantExplicitArg> {
                        Value value = iter.value();
                        if (matchPattern(value, isConstant)) {
                          LLVM_DEBUG(llvm::dbgs()
                                     << "Arith constant: " << value
                                     << " at pos " << iter.index() << "\n");
                          return std::make_unique<ConstantArithArg>(
                              iter.index(), value.getDefiningOp());
                        }
                        return nullptr;
                      }),
      llvm::identity<std::unique_ptr<ConstantExplicitArg>>());
}

template <typename RangeTy>
void ConstantPropagationPass::dropConstantArgs(RangeTy constants,
                                               Builder &builder,
                                               SYCLHostScheduleKernel launch) {
  auto funcOp = static_cast<FunctionOpInterface>(getOperation());
  llvm::BitVector launchIndices(launch->getNumOperands());
  llvm::BitVector funcIndices(funcOp.getNumArguments());
  unsigned launchOffset = launchIndices.size() - funcIndices.size();
  for (const std::unique_ptr<ConstantExplicitArg> &arg : constants) {
    unsigned index = arg->getIndex();
    // Some constants may mark more than one argument to be erased.
    TypeSwitch<ConstantExplicitArg *>(arg.get()).Case<ConstantArithArg>(
        [&](const auto &) {
          funcIndices.set(index);
          launchIndices.set(index + launchOffset);
        });
    ++NumPropagatedConstants;
  }

  LLVM_DEBUG(llvm::dbgs() << "Dropping arguments: " << funcIndices << "\n");
  funcOp.eraseArguments(funcIndices);

  // We have to update the OperandSegmentSizes attribute as this operation may
  // receive more than one variadic argument.
  launch->eraseOperands(launchIndices);
  std::array<int32_t, 3> operandSegmentSizes{
      0, 0, static_cast<int32_t>(funcOp.getNumArguments())};
  launch->setAttr(launch.getOperandSegmentSizesAttrName(),
                  builder.getDenseI32ArrayAttr(operandSegmentSizes));
}

template <typename RangeTy>
void ConstantPropagationPass::propagateConstantArgs(
    RangeTy constants, SYCLHostScheduleKernel launch) {
  Region *region = getOperation().getCallableRegion();
  OpBuilder builder(region);
  for (const std::unique_ptr<ConstantExplicitArg> &arg : constants) {
    TypeSwitch<ConstantExplicitArg *>(arg.get()).Case<ConstantArithArg>(
        [&](const auto *arith) { arith->propagate(builder, *region); });
  }
  dropConstantArgs<RangeTy>(constants, builder, launch);
}

void ConstantPropagationPass::runOnOperation() {
  gpu::GPUFuncOp op = getOperation();
  if (!op.isKernel() || op.isExternal()) {
    LLVM_DEBUG(llvm::dbgs() << "Early exit: Function is not a kernel\n");
    return;
  }

  constexpr auto isSingleton = [](auto range) {
    return std::next(range.begin()) == range.end();
  };

  auto module = op->getParentOfType<ModuleOp>();
  std::optional<SymbolTable::UseRange> uses =
      SymbolTable::getSymbolUses(op, module);
  if (!(uses && isSingleton(*uses))) {
    LLVM_DEBUG(llvm::dbgs() << "SYCL constant propagation pass expects a "
                               "single kernel launch point\n");
    return;
  }

  auto launchPoint = dyn_cast<SYCLHostScheduleKernel>(uses->begin()->getUser());
  if (!launchPoint) {
    LLVM_DEBUG(llvm::dbgs() << "SYCL constant propagation pass expects launch "
                               "point to be 'sycl.host.schedule_kernel'\n");
  }

  propagateConstantArgs(getConstantArgs(launchPoint), launchPoint);
}

std::unique_ptr<Pass> sycl::createConstantPropagationPass() {
  return std::make_unique<ConstantPropagationPass>();
}
