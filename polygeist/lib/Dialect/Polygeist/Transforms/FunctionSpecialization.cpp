//===- FunctionSpecialization.cpp - Specialize functions ------------------===//
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
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "function-specialization"

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_FUNCTIONSPECIALIZATION
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

using namespace mlir;
using AccessorType = VersionConditionBuilder::AccessorType;
using AccessorPairType = VersionConditionBuilder::AccessorPairType;

static llvm::cl::opt<unsigned> functionSpecializationAccessorLimit(
    "function-specialization-accessor-limit", llvm::cl::init(5),
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

/// Returns true if \p arg is a candidate argument.
static bool isCandidateArgs(Value arg) {
  if (!isa<AccessorType>(arg))
    return false;
  return isValidMemRefType(arg.getType());
}

/// Populate \p candArgs with candidate arguments from \p call.
static void collectCandidateArguments(CallOpInterface call,
                                      SmallVectorImpl<AccessorType> &candArgs) {
  assert(candArgs.empty() && "Expecting empty candArgs");
  for (Value arg : call.getArgOperands())
    if (isCandidateArgs(arg))
      candArgs.push_back(cast<AccessorType>(arg));
}

/// Returns the cloned version of \p func.
static FunctionOpInterface cloneFunction(FunctionOpInterface func) {
  ModuleOp module = func->getParentOfType<ModuleOp>();
  FunctionOpInterface clonedFunc = func.clone();
  std::string newFnName = (func.getName() + ".specialized").str();
#ifndef NDEBUG
  module->walk([newFnName](FunctionOpInterface func) {
    assert(func.getName() != newFnName &&
           "Expecting new function name to be unique");
  });
#endif // NDEBUG
  clonedFunc.setName(newFnName);
  OpBuilder builder(func);
  builder.insert(clonedFunc);
  return clonedFunc;
}

/// Add attribute 'sycl.inner.disjoint' to all candidate arguments of \p func.
static void specializeFunction(FunctionOpInterface func) {
  for (unsigned i = 0; i < func.getNumArguments(); ++i)
    if (isCandidateArgs(func.getArgument(i)))
      func.setArgAttr(i, "sycl.inner.disjoint",
                      UnitAttr::get(func->getContext()));
}

namespace {

class FunctionSpecializationPass
    : public polygeist::impl::FunctionSpecializationBase<
          FunctionSpecializationPass> {
public:
  using FunctionSpecializationBase<
      FunctionSpecializationPass>::FunctionSpecializationBase;

  void runOnOperation() override;

private:
  /// Returns true if \p func is a candidate.
  bool isCandidateFunction(FunctionOpInterface func) const;
  /// Returns true if \p acc1 and \p acc2 need to be checked for no overlap.
  bool isCandidateAccessorPair(AccessorType acc1, AccessorType acc2) const;
  /// Populate \p accessorPairs with accessor pairs that should be checked for
  /// no overlap for \p call.
  void getRequireNoOverlapAccessorPairs(
      CallOpInterface call,
      SmallVectorImpl<AccessorPairType> &accessorPairs) const;
  /// Version \p call.
  void versionCall(CallOpInterface call) const;
};

} // namespace

//===----------------------------------------------------------------------===//
// FunctionSpecializationPass
//===----------------------------------------------------------------------===//

bool FunctionSpecializationPass::isCandidateAccessorPair(
    AccessorType acc1, AccessorType acc2) const {
  if (acc1 == acc2)
    return false;
  if (relaxedAliasing)
    return true;
  auto acc1Ty = cast<sycl::AccessorType>(acc1.getType().getElementType());
  auto acc2Ty = cast<sycl::AccessorType>(acc2.getType().getElementType());
  return (acc1Ty.getType() == acc2Ty.getType());
}

void FunctionSpecializationPass::getRequireNoOverlapAccessorPairs(
    CallOpInterface call,
    SmallVectorImpl<AccessorPairType> &accessorPairs) const {
  SmallVector<AccessorType> candArgs;
  collectCandidateArguments(call, candArgs);
  for (SmallVector<AccessorType>::iterator i = candArgs.begin();
       i != candArgs.end(); ++i)
    for (SmallVector<AccessorType>::iterator j = i + 1; j != candArgs.end();
         ++j)
      if (isCandidateAccessorPair(*i, *j))
        accessorPairs.push_back({*i, *j});
}

void FunctionSpecializationPass::versionCall(CallOpInterface call) const {
  SmallVector<AccessorPairType> accessorPairs;
  getRequireNoOverlapAccessorPairs(call, accessorPairs);
  if (accessorPairs.empty())
    return;
  OpBuilder builder(call);
  std::unique_ptr<VersionCondition> condition =
      VersionConditionBuilder(accessorPairs, builder, call->getLoc())
          .createCondition();
  CallVersionBuilder(call).version(*condition);
}

void FunctionSpecializationPass::runOnOperation() {
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
    specializeFunction(clonedFunc);

    for (Operation *op : userMap.getUsers(func)) {
      auto call = cast<CallOpInterface>(op);
      versionCall(call);
      call->setAttr("callee", SymbolRefAttr::get(func.getContext(),
                                                 clonedFunc.getName()));
    }
    ++numFunctionSpecialized;
  }
}

bool FunctionSpecializationPass::isCandidateFunction(
    FunctionOpInterface func) const {
  if (!isPotentialKernelBodyFunc(func)) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "not a candidate: not a potential kernel body function\n");
    return false;
  }

  // Temporary condition to only allow function called directly by a GPU kernel.
  // TODO: allow maximum depth of 2.
  Optional<unsigned> maxDepth = getMaxDepthFromAnyGPUKernel(func);
  if (maxDepth != 1)
    return false;

  // Limit the number of accessor arguments allowed as candidate.
  unsigned numCandidateArgs = count_if(func.getArguments(), [](Value arg) {
    return isValidMemRefType(arg.getType());
  });
  if (numCandidateArgs < 2 ||
      numCandidateArgs > functionSpecializationAccessorLimit) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "not a candidate: exceed accessor limit\n");
    return false;
  }

  LLVM_DEBUG(llvm::dbgs().indent(2)
             << "Found candidate: " << func.getName() << "\n");
  return true;
}

std::unique_ptr<Pass> mlir::polygeist::createFunctionSpecializationPass() {
  return std::make_unique<FunctionSpecializationPass>();
}
std::unique_ptr<Pass> mlir::polygeist::createFunctionSpecializationPass(
    const FunctionSpecializationOptions &options) {
  return std::make_unique<FunctionSpecializationPass>(options);
}
