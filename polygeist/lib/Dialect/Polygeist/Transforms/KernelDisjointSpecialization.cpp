//===- KernelDisjointSpecialization.cpp - Specialize kernel body functions ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass attempts to version the callsite from SYCL kernel to the SYCL
// kernel body function, where the specialized version ensures SYCL accessors do
// not overlap.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Polygeist/IR/PolygeistOps.h"
#include "mlir/Dialect/Polygeist/Utils/TransformUtils.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "kernel-disjoint-specialization"

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_KERNELDISJOINTSPECIALIZATION
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

using namespace mlir;

static llvm::cl::opt<unsigned> KernelDisjointSpecializationAccessorLimit(
    DEBUG_TYPE "-accessor-limit", llvm::cl::init(5),
    llvm::cl::desc(
        "Maximum number of accessors allowed for function specialization"));

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Returns true if \p arg is a candidate argument. Currently, all arguments
/// with type 'memref<?x!sycl.accessor>' is a candidate argument.
static bool isCandidateArg(Value arg) {
  return sycl::isPtrOf<sycl::AccessorType>(arg.getType());
}

/// Populates \p candArgs with candidate arguments from \p call.
static void
collectCandidateArguments(CallOpInterface call,
                          SmallVectorImpl<sycl::AccessorPtrValue> &candArgs) {
  assert(candArgs.empty() && "Expecting empty candArgs");
  for (Value arg : call.getArgOperands())
    if (isCandidateArg(arg))
      candArgs.push_back(cast<sycl::AccessorPtrValue>(arg));
}

/// Returns the cloned version of \p func.
static FunctionOpInterface cloneFunction(FunctionOpInterface func) {
  std::string newFnName = (func.getName() + ".specialized").str();
  polygeist::getUniqueSymbolName(
      newFnName, func->getParentWithTrait<OpTrait::SymbolTable>());
  OpBuilder builder(func);
  FunctionOpInterface clonedFunc = func.clone();
  clonedFunc.setName(newFnName);
  builder.insert(clonedFunc);
  polygeist::privatize(clonedFunc);
  return clonedFunc;
}

/// Adds attribute 'sycl.inner.disjoint' to arguments of \p func that satisfy \p
/// predicate.
template <typename Pred,
          typename = std::enable_if_t<std::is_invocable_r_v<bool, Pred, Value>>>
static void setInnerDisjointAttribute(FunctionOpInterface func,
                                      Pred predicate) {
  for (unsigned i = 0; i < func.getNumArguments(); ++i)
    if (predicate(func.getArgument(i)))
      func.setArgAttr(i, sycl::SYCLDialect::getInnerDisjointAttrName(),
                      UnitAttr::get(func->getContext()));
}

namespace {

// Original code:
//   func.func private @callee(
//     %arg0 : memref<?x!sycl.accessor>,
//     %arg1 : memref<?x!sycl.accessor) {}
//   gpu.func @caller kernel {
//     func.call @callee(%acc1, %acc2)
//   }
// Optimized code:
//   func.func private @callee.specialized(
//     %arg0 : memref<?x!sycl.accessor> {sycl.inner.disjoint},
//     %arg1 : memref<?x!sycl.accessor> {sycl.inner.disjoint}) {}
//   func.func private @callee(
//     %arg0 : memref<?x!sycl.accessor>,
//     %arg1 : memref<?x!sycl.accessor) {}
//   gpu.func @caller kernel {
//     if (not_overlap(%acc1, %acc2))
//       func.call @callee.specialized(%acc1, %acc2)
//     else
//       func.call @callee(%acc1, %acc2)
//   }
class KernelDisjointSpecializationPass
    : public polygeist::impl::KernelDisjointSpecializationBase<
          KernelDisjointSpecializationPass> {
public:
  using KernelDisjointSpecializationBase<
      KernelDisjointSpecializationPass>::KernelDisjointSpecializationBase;

  void runOnOperation() override;

private:
  /// Returns true if \p func is a candidate.
  bool isCandidateFunction(
      FunctionOpInterface func,
      const polygeist::FunctionKernelInfo &funcKernelInfo) const;
  /// Returns true if \p acc1 and \p acc2 need to be checked for no overlap. For
  /// example, under strict aliasing rule, accessors with different element
  /// types are not alias, so return false.
  bool isCandidateAccessorPair(sycl::AccessorPtrValue acc1,
                               sycl::AccessorPtrValue acc2) const;
  /// Populate \p accessorPairs with accessor pairs that should be checked for
  /// no overlap for \p call.
  void collectMayOverlapAccessorPairs(
      CallOpInterface call,
      llvm::SmallSet<sycl::AccessorPtrPair, 4> &accessorPairs) const;
  /// Version \p call.
  void versionCall(CallOpInterface call) const;
};

} // namespace

//===----------------------------------------------------------------------===//
// KernelDisjointSpecializationPass
//===----------------------------------------------------------------------===//

void KernelDisjointSpecializationPass::runOnOperation() {
  auto gpuModule = dyn_cast<gpu::GPUModuleOp>(
      getOperation()->getRegion(0).front().getOperations().front());
  if (!gpuModule)
    return;
  polygeist::FunctionKernelInfo funcKernelInfo(gpuModule);

  SmallVector<FunctionOpInterface> candidates;
  gpuModule->walk([&](FunctionOpInterface func) {
    LLVM_DEBUG(llvm::dbgs()
               << "Processing function \"" << func.getName() << "\"\n");
    if (isCandidateFunction(func, funcKernelInfo))
      candidates.push_back(func);
  });

  ModuleOp module = getOperation();
  SymbolTableCollection symTable;
  SymbolUserMap userMap(symTable, module);

  for (FunctionOpInterface func : candidates) {
    FunctionOpInterface clonedFunc = cloneFunction(func);
    setInnerDisjointAttribute(clonedFunc, isCandidateArg);

    for (Operation *op : userMap.getUsers(func)) {
      auto call = cast<CallOpInterface>(op);
      versionCall(call);

      /// Update the callee of call to clonedFunc.
      call.setCalleeFromCallable(
          SymbolRefAttr::get(clonedFunc.getContext(), clonedFunc.getName()));
    }
    ++numFunctionSpecialized;
  }
}

bool KernelDisjointSpecializationPass::isCandidateFunction(
    FunctionOpInterface func,
    const polygeist::FunctionKernelInfo &funcKernelInfo) const {
  if (!funcKernelInfo.isPotentialKernelBodyFunction(func)) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "not a candidate: not a potential kernel body function\n");
    return false;
  }

  unsigned numCandidateArgs = count_if(func.getArguments(), isCandidateArg);
  if (numCandidateArgs < 2) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "not a candidate: not enough candidate accessors\n");
    return false;
  }
  if (numCandidateArgs > KernelDisjointSpecializationAccessorLimit) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "not a candidate: exceed accessor limit\n");
    return false;
  }

  LLVM_DEBUG(llvm::dbgs().indent(2)
             << "Found candidate: " << func.getName() << "\n");
  return true;
}

bool KernelDisjointSpecializationPass::isCandidateAccessorPair(
    sycl::AccessorPtrValue acc1, sycl::AccessorPtrValue acc2) const {
  assert(acc1 != acc2 && "Expecting the input accessors to be different");
  if (!relaxedAliasing) {
    sycl::AccessorType acc1Ty = acc1.getAccessorType();
    sycl::AccessorType acc2Ty = acc2.getAccessorType();
    if (acc1Ty.getType() != acc2Ty.getType())
      return false;
  }

  return true;
}

void KernelDisjointSpecializationPass::collectMayOverlapAccessorPairs(
    CallOpInterface call,
    llvm::SmallSet<sycl::AccessorPtrPair, 4> &accessorPairs) const {
  SmallVector<sycl::AccessorPtrValue> candArgs;
  collectCandidateArguments(call, candArgs);
  for (auto *i = candArgs.begin(); i != candArgs.end(); ++i)
    for (auto *j = i + 1; j != candArgs.end(); ++j)
      if (isCandidateAccessorPair(*i, *j))
        accessorPairs.insert({*i, *j});
}

void KernelDisjointSpecializationPass::versionCall(CallOpInterface call) const {
  llvm::SmallSet<sycl::AccessorPtrPair, 4> accessorPairs;
  collectMayOverlapAccessorPairs(call, accessorPairs);
  if (accessorPairs.empty())
    return;
  OpBuilder builder(call);
  std::unique_ptr<polygeist::IfCondition> condition =
      polygeist::VersionConditionBuilder(accessorPairs, builder, call->getLoc())
          .createCondition();
  polygeist::VersionBuilder(call).version(*condition);
}

std::unique_ptr<Pass>
mlir::polygeist::createKernelDisjointSpecializationPass() {
  return std::make_unique<KernelDisjointSpecializationPass>();
}
std::unique_ptr<Pass> mlir::polygeist::createKernelDisjointSpecializationPass(
    const KernelDisjointSpecializationOptions &options) {
  return std::make_unique<KernelDisjointSpecializationPass>(options);
}
