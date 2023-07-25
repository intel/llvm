//===- ConstantPropagation.cpp - Host-device constant propagation pass ----===//
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

#define DEBUG_TYPE "sycl-constant-propagation"

using namespace mlir;
using namespace mlir::sycl;

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

  /// Propagate constants in \p constants to the function launched by \p launch.
  template <typename RangeTy>
  void propagateConstantArgs(RangeTy constants, SYCLHostScheduleKernel launch);
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

template <typename RangeTy>
void ConstantPropagationPass::propagateConstantArgs(
    RangeTy constants, SYCLHostScheduleKernel launch) {
  Region *region = getOperation().getCallableRegion();
  OpBuilder builder(region);
  for (const std::unique_ptr<ConstantExplicitArg> &arg : constants) {
    TypeSwitch<ConstantExplicitArg *>(arg.get()).Case<ConstantArithArg>(
        [&](const auto *arith) { arith->propagate(builder, *region); });
    ++NumPropagatedConstants;
  }
}

void ConstantPropagationPass::runOnOperation() {
  gpu::GPUFuncOp op = getOperation();
  LLVM_DEBUG(llvm::dbgs() << "Performing constant propagation on function '"
                          << static_cast<SymbolOpInterface>(op).getName()
                          << "'\n");
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

  auto launchPoint = dyn_cast<SYCLHostScheduleKernel>(uses->begin()->getUser());
  if (!launchPoint) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "SYCL constant propagation pass expects launch "
                  "point to be 'sycl.host.schedule_kernel'\n");
  }

  propagateConstantArgs(getConstantArgs(launchPoint), launchPoint);
}

std::unique_ptr<Pass> sycl::createConstantPropagationPass() {
  return std::make_unique<ConstantPropagationPass>();
}
