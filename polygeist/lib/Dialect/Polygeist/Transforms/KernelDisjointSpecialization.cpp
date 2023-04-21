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
#include "mlir/Dialect/Polygeist/IR/Ops.h"
#include "mlir/Dialect/Polygeist/IR/Polygeist.h"
#include "mlir/Dialect/Polygeist/Utils/TransformUtils.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
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
using AccessorPtrType = polygeist::VersionConditionBuilder::AccessorPtrType;
using AccessorPtrPairType =
    polygeist::VersionConditionBuilder::AccessorPtrPairType;

static llvm::cl::opt<unsigned> KernelDisjointSpecializationAccessorLimit(
    DEBUG_TYPE "-accessor-limit", llvm::cl::init(5),
    llvm::cl::desc(
        "Maximum number of accessors allowed for function specialization"));

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Returns true if \p type is 'memref<?x!sycl.accessor>', and false otherwise.
static bool isValidMemRefType(Type type) {
  auto mt = dyn_cast<MemRefType>(type);
  bool isMemRefWithExpectedShape =
      (mt && mt.hasRank() && (mt.getRank() == 1) &&
       ShapedType::isDynamic(mt.getShape()[0]) && mt.getLayout().isIdentity());
  if (!isMemRefWithExpectedShape)
    return false;

  return isa<sycl::AccessorType>(mt.getElementType());
}

/// Returns true if \p arg is a candidate argument. Currently, all arguments
/// with valid memref type determined by isValidMemRefType is a candidate
/// argument, i.e., 'memref<?x!sycl.accessor>`.
static bool isCandidateArg(Value arg) {
  if (isValidMemRefType(arg.getType())) {
    assert(isa<AccessorPtrType>(arg) &&
           "Expecting valid memref type to be AccessorPtrType");
    return true;
  }
  return false;
}

/// Populate \p candArgs with candidate arguments from \p call.
static void
collectCandidateArguments(CallOpInterface call,
                          SmallVectorImpl<AccessorPtrType> &candArgs) {
  assert(candArgs.empty() && "Expecting empty candArgs");
  for (Value arg : call.getArgOperands())
    if (isCandidateArg(arg))
      candArgs.push_back(cast<AccessorPtrType>(arg));
}

/// Returns the cloned version of \p func.
static FunctionOpInterface cloneFunction(FunctionOpInterface func) {
  std::string newFnName = (func.getName() + ".specialized").str();
#ifndef NDEBUG
  ModuleOp module = func->getParentOfType<ModuleOp>();
  module->walk([newFnName](FunctionOpInterface func) {
    assert(func.getName() != newFnName &&
           "Expecting new function name to be unique");
  });
#endif // NDEBUG
  OpBuilder builder(func);
  FunctionOpInterface clonedFunc = func.clone();
  clonedFunc.setName(newFnName);
  builder.insert(clonedFunc);
  polygeist::privatize(clonedFunc);
  return clonedFunc;
}

/// Add attribute 'sycl.inner.disjoint' to arguments of \p func that satisfy \p
/// predicate.
template <typename Pred,
          typename = std::enable_if_t<std::is_invocable_r_v<bool, Pred, Value>>>
static void setInnerDisjointAttribute(FunctionOpInterface func,
                                      Pred predicate) {
  constexpr StringRef innerDisjointAttrName = "sycl.inner.disjoint";
  for (unsigned i = 0; i < func.getNumArguments(); ++i)
    if (predicate(func.getArgument(i)))
      func.setArgAttr(i, innerDisjointAttrName,
                      UnitAttr::get(func->getContext()));
}

/// Update the callee of the call \p call to \p callee.
static void updateCallee(CallOpInterface call, FunctionOpInterface callee) {
  // TODO: find a generic way to update callee of a CallOpInterface.
  assert(call->hasAttr("callee") &&
         "Expecting the call to have callee attribute");
  call->setAttr("callee",
                SymbolRefAttr::get(callee.getContext(), callee.getName()));
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
  bool isCandidateFunction(FunctionOpInterface func) const;
  /// Returns true if \p acc1 and \p acc2 need to be checked for no overlap. For
  /// example, under strict aliasing rule, accessors with different element
  /// types are not alias, so return false.
  bool isCandidateAccessorPair(AccessorPtrType acc1, AccessorPtrType acc2,
                               const AliasAnalysis &aliasAnalysis) const;
  /// Populate \p accessorPairs with accessor pairs that should be checked for
  /// no overlap for \p call.
  void collectMayOverlapAccessorPairs(
      CallOpInterface call, SmallVectorImpl<AccessorPtrPairType> &accessorPairs,
      const AliasAnalysis &aliasAnalysis) const;
  /// Version \p call.
  void versionCall(CallOpInterface call,
                   const AliasAnalysis &aliasAnalysis) const;
};

} // namespace

//===----------------------------------------------------------------------===//
// KernelDisjointSpecializationPass
//===----------------------------------------------------------------------===//

void KernelDisjointSpecializationPass::runOnOperation() {
  AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
  aliasAnalysis.addAnalysisImplementation(sycl::AliasAnalysis(relaxedAliasing));

  SmallVector<FunctionOpInterface> candidates;
  getOperation()->walk([&](FunctionOpInterface func) {
    LLVM_DEBUG(llvm::dbgs()
               << "Processing function \"" << func.getName() << "\"\n");
    if (isCandidateFunction(func))
      candidates.push_back(func);
  });

  ModuleOp module = getOperation();
  SymbolTableCollection symTable;
  SymbolUserMap userMap(symTable, module);

  for (FunctionOpInterface func : candidates) {
    FunctionOpInterface clonedFunc = cloneFunction(func);
    setInnerDisjointAttribute(clonedFunc, isCandidateArg);

    for (Operation *op : userMap.getUsers(func)) {
      // Due to temporary condition to only allow function called directly by a
      // GPU kernel.
      assert(op->getParentOfType<gpu::GPUFuncOp>() &&
             "Expecting calls only in GPU kernel");

      auto call = cast<CallOpInterface>(op);
      versionCall(call, aliasAnalysis);
      updateCallee(call, clonedFunc);
    }
    ++numFunctionSpecialized;
  }
}

bool KernelDisjointSpecializationPass::isCandidateFunction(
    FunctionOpInterface func) const {
  if (!polygeist::isPotentialKernelBodyFunc(func)) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "not a candidate: not a potential kernel body function\n");
    return false;
  }

  // Temporary condition to only allow function called directly by a GPU kernel.
  // TODO: allow maximum depth of 2.
  Optional<unsigned> maxDepth = polygeist::getMaxDepthFromAnyGPUKernel(func);
  assert(maxDepth.has_value() && "Expecting valid maxDepth");
  if (maxDepth != 1)
    return false;

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
    AccessorPtrType acc1, AccessorPtrType acc2,
    const AliasAnalysis &aliasAnalysis) const {
  assert(acc1 != acc2 && "Expecting the input accessors to be different");
  if (const_cast<AliasAnalysis &>(aliasAnalysis).alias(acc1, acc2).isNo())
    return false;

  if (!relaxedAliasing) {
    auto acc1Ty = cast<sycl::AccessorType>(acc1.getType().getElementType());
    auto acc2Ty = cast<sycl::AccessorType>(acc2.getType().getElementType());
    if (acc1Ty.getType() != acc2Ty.getType())
      return false;
  }

  return true;
}

void KernelDisjointSpecializationPass::collectMayOverlapAccessorPairs(
    CallOpInterface call, SmallVectorImpl<AccessorPtrPairType> &accessorPairs,
    const AliasAnalysis &aliasAnalysis) const {
  SmallVector<AccessorPtrType> candArgs;
  collectCandidateArguments(call, candArgs);
  for (auto *i = candArgs.begin(); i != candArgs.end(); ++i)
    for (auto *j = i + 1; j != candArgs.end(); ++j)
      if (isCandidateAccessorPair(*i, *j, aliasAnalysis))
        accessorPairs.push_back({*i, *j});
}

void KernelDisjointSpecializationPass::versionCall(
    CallOpInterface call, const AliasAnalysis &aliasAnalysis) const {
  SmallVector<AccessorPtrPairType> accessorPairs;
  collectMayOverlapAccessorPairs(call, accessorPairs, aliasAnalysis);
  if (accessorPairs.empty())
    return;
  OpBuilder builder(call);
  std::unique_ptr<polygeist::VersionCondition> condition =
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
