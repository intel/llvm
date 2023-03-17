//===- ArgumentPromotion.cpp - Promote by-reference arguments -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass currently attempts to peel struct members from the argument pack
// passed to a SYCL kernel. For example given:
//
//  gpu.func @kernel_parallel_for(%arg0: i32) {
//    func.call @parallel_for(%0, ...) :
//        (memref<?x!llvm.struct<(i32, !sycl.accessor), ...)
//    gpu.return
//  }
//
// The pass modifies the call (and the callee):
//
//  gpu.func @kernel_parallel_for(%arg0: i32) {
//    %int_arg = <memref to first struct member>
//    %acc_arg = <memref to second struct member>
//    func.call @parallel_for(%int_arg, %acc_arg, ...) :
//        (memref<?xi32>, memref<?x!llvm.struct<(!sycl.accessor...)
//    gpu.return
//  }

//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Transforms/Passes.h"

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Polygeist/IR/Ops.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"
#include "mlir/Dialect/SYCL/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_ARGUMENTPROMOTION
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

#define DEBUG_TYPE "arg-promotion"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {
/// Represent a candidate for the argument promotion transformation.
class Candidate {
public:
  Candidate(CallOpInterface callOp) : callOp(callOp) {
    assert(isPrivateCallable(callOp) && "Callee must be a private definition");
    assert(!isRecursiveCall(callOp) && "Callee must not be recursive");
  }

  static bool isPrivateCallable(Operation *op);
  static bool isRecursiveCall(CallOpInterface callOp);
  static bool isValidMemRefType(Type type);
  static Region *getCallableRegion(CallOpInterface callOp);

  // Peel members and fixup the call site and the callee.
  void transform();

private:
  /// Identify call operands that should be peeled and peel them, returns the
  /// number of struct members peeled. Note: this function populates a map
  /// between the original operand and the peeled values.
  unsigned peelMembers();

  /// Modify the call by replacing the original operands with their
  /// corresponding peeled members.
  void modifyCall();

  /// Modify the called function by replacing the original operands with their
  /// corresponding peeled members.
  void modifyCallee();

  /// Replace the argument at position \p pos in the region \p callableRgn with
  /// \p newArgs. The reference parameter \p newArgAttrs is filled with the new
  /// argument attributes.
  void replaceArgumentWith(unsigned pos, Region &callableRgn,
                           const SmallVector<Value> &newArgs,
                           SmallVector<Attribute> &newArgAttrs) const;

  /// Replace uses of the argument \p origArg in the region \p callableRgn with
  /// the arguments starting at position \pos in the callable region.
  void replaceUsesOfArgument(Value origArg, unsigned pos,
                             Region &callableRgn) const;

  mutable CallOpInterface callOp;
  std::map<unsigned, SmallVector<Value>> operandToMembersPeeled;
};

class ArgumentPromotionPass
    : public polygeist::impl::ArgumentPromotionBase<ArgumentPromotionPass> {
public:
  void runOnOperation() override;

private:
  /// Populate \p candidates with call operations to transform.
  void collectCandidates(ModuleOp module,
                         SmallVectorImpl<Candidate> &candidates);
};
} // namespace

//===----------------------------------------------------------------------===//
// Candidate
//===----------------------------------------------------------------------===//

bool Candidate::isPrivateCallable(Operation *op) {
  // The callable region must be defined.
  if (auto callable = dyn_cast<CallableOpInterface>(op))
    if (!callable.getCallableRegion())
      return false;

  // The callable symbol must have private visibility.
  if (auto sym = dyn_cast<SymbolOpInterface>(op))
    if (!sym.isPrivate())
      return false;

  return true;
}

bool Candidate::isRecursiveCall(CallOpInterface callOp) {
  Region *callableRegion = getCallableRegion(callOp);
  return callableRegion &&
         callableRegion->isAncestor(callOp->getParentRegion());
}

bool Candidate::isValidMemRefType(Type type) {
  // We want a ranked memref type with shape 'memref<?x...>'.
  auto mt = dyn_cast<MemRefType>(type);
  if (!mt || !mt.hasRank() || mt.getShape().size() != 1 ||
      !ShapedType::isDynamic(mt.getShape()[0]) || !mt.getLayout().isIdentity())
    return false;

  // The element type must be a struct with at least 2 members.
  auto structType = dyn_cast<LLVM::LLVMStructType>(mt.getElementType());
  if (!structType || structType.getBody().size() < 2)
    return false;

  return true;
}

Region *Candidate::getCallableRegion(CallOpInterface callOp) {
  if (auto callable = dyn_cast<CallableOpInterface>(callOp.resolveCallable()))
    return callable.getCallableRegion();
  return nullptr;
}

void Candidate::transform() {
  // Identify arguments that can be peeled and and peel them.
  unsigned numMembersPeeled = peelMembers();
  if (!numMembersPeeled)
    return;

  modifyCall();
  modifyCallee();
}

unsigned Candidate::peelMembers() {
  OpBuilder builder(callOp);

  unsigned numMembersPeeled = 0;
  for (unsigned pos = 0; pos < callOp->getNumOperands(); ++pos) {
    Value op = callOp->getOperand(pos);
    if (!isValidMemRefType(op.getType()))
      continue;

    auto memRefType = cast<MemRefType>(op.getType());
    auto structType = cast<LLVM::LLVMStructType>(memRefType.getElementType());

    unsigned numMembers = structType.getBody().size();
    SmallVector<Value> membersPeeled;
    membersPeeled.reserve(numMembers);
    for (unsigned idx = 0; idx < numMembers; ++idx) {
      auto memberType = structType.getBody()[idx];
      auto mt =
          MemRefType::get(memRefType.getShape(), memberType,
                          memRefType.getLayout(), memRefType.getMemorySpace());
      auto subIndexOp = builder.create<polygeist::SubIndexOp>(
          op.getLoc(), mt, op,
          builder.create<arith::ConstantIndexOp>(op.getLoc(), idx));
      membersPeeled.push_back(subIndexOp);
      ++numMembersPeeled;
    }

    operandToMembersPeeled.insert({pos, membersPeeled});
  }

  return numMembersPeeled;
}

void Candidate::modifyCall() {
  SmallVector<Value, 8> newCallOperands;
  for (unsigned pos = 0; pos < callOp->getNumOperands(); ++pos) {
    if (operandToMembersPeeled.find(pos) == operandToMembersPeeled.end()) {
      newCallOperands.push_back(callOp->getOperand(pos));
      continue;
    }

    for (Value peeledMember : operandToMembersPeeled[pos])
      newCallOperands.push_back(peeledMember);
  }

  callOp->setOperands(newCallOperands);
  llvm::dbgs() << "callOp: " << callOp << "\n";
}

void Candidate::modifyCallee() {
  assert(callOp.resolveCallable() && "Could not find callee definition");
  auto callable = cast<CallableOpInterface>(callOp.resolveCallable());
  auto funcOp = cast<func::FuncOp>(callable.getCallableRegion()->getParentOp());
  Region &callableRgn = funcOp.getFunctionBody();
  ArrayAttr origArgAttrs = funcOp.getAllArgAttrs();
  SmallVector<Attribute> newArgAttrs;

  // Replace the old arguments with the new ones.
  unsigned pos = 0;
  const unsigned orinNumArgs = callableRgn.getNumArguments();
  for (unsigned origPos = 0; origPos < orinNumArgs; ++origPos) {
    if (operandToMembersPeeled.find(origPos) == operandToMembersPeeled.end()) {
      newArgAttrs.push_back(origArgAttrs[origPos]);
      continue;
    }

    // Replace the argument at 'pos' with the new arguments we peeled.
    const SmallVector<Value> &newArgs = operandToMembersPeeled[origPos];
    replaceArgumentWith(pos, callableRgn, newArgs, newArgAttrs);

    // Delete the original argument.
    llvm::dbgs() << "at line: " << __LINE__ << "\n";
    pos += newArgs.size();
    callableRgn.eraseArgument(pos);
  }

  auto newFuncType =
      FunctionType::get(funcOp.getContext(), callableRgn.getArgumentTypes(),
                        funcOp.getResultTypes());
  llvm::dbgs() << "at line: " << __LINE__ << "\n";

  funcOp.setFunctionType(newFuncType);
  funcOp.setAllArgAttrs(newArgAttrs);
}

void Candidate::replaceArgumentWith(unsigned pos, Region &callableRgn,
                                    const SmallVector<Value> &newArgs,
                                    SmallVector<Attribute> &newArgAttrs) const {
  assert(!newArgs.empty() && "Expecting a non-empty vector");

  Value origArg = callableRgn.getArgument(pos);

  for (unsigned offset = 0; offset < newArgs.size(); ++offset) {
    callableRgn.insertArgument(pos + offset, newArgs[offset].getType(),
                               callableRgn.getLoc());
    // TODO: add attribute NoAlias on the new argument.
    newArgAttrs.push_back(Attribute());
  }

  // Replace uses of the original argument with the new arguments injected.
  replaceUsesOfArgument(origArg, pos, callableRgn);
}

void Candidate::replaceUsesOfArgument(Value origArg, unsigned pos,
                                      Region &callableRgn) const {
  for (OpOperand &use : origArg.getUses()) {
    assert(isa<polygeist::SubIndexOp>(use.getOwner()) &&
           "Expecting a subindex operation");

    // The use must be a polygeist::SubIndexOp operation with a constant
    // index, remove it and rewrite its users.
    auto subIndexOp = cast<polygeist::SubIndexOp>(use.getOwner());
    TypedValue<IndexType> index = subIndexOp.getIndex();
    auto indexOp = index.getDefiningOp<arith::ConstantIndexOp>();
    assert(indexOp && "Must have a constant index");

    int64_t offset = indexOp.value();
    BlockArgument newArg = callableRgn.getArgument(pos + offset);
    subIndexOp.replaceAllUsesWith(newArg);
    subIndexOp.erase();
  }
}

//===----------------------------------------------------------------------===//
// ArgumentPromotionPass
//===----------------------------------------------------------------------===//

void ArgumentPromotionPass::runOnOperation() {
  ModuleOp module = getOperation();
  //  CallGraph &CG = getAnalysis<CallGraph>();

  llvm::dbgs() << "Entering ArgumentPromotionPass::runOnOperation()\n";

  SmallVector<Candidate> candidates;
  collectCandidates(module, candidates);
  if (candidates.empty())
    return;

  for (Candidate &cand : candidates)
    cand.transform();
}

void ArgumentPromotionPass::collectCandidates(
    ModuleOp module, SmallVectorImpl<Candidate> &candidateCalls) {
  module->walk([&](gpu::GPUFuncOp gpuFuncOp) {
    gpuFuncOp->walk([&](CallOpInterface callOp) {
      // We are only interested in non-recursive tail calls.
      if (!callOp->getBlock()->hasNoSuccessors() ||
          Candidate::isRecursiveCall(callOp))
        return WalkResult::advance();

      llvm::dbgs() << "at line: " << __LINE__ << "\n";

      // The first operand of the call must not be used by any other
      // operation.
      OperandRange callOperands = callOp.getArgOperands();
      if (callOperands.empty() || !callOperands.front().hasOneUse())
        return WalkResult::advance();
      llvm::dbgs() << "at line: " << __LINE__ << "\n";

      // The first operand must be a memref with a struct element type.
      if (!Candidate::isValidMemRefType(callOperands.front().getType()))
        return WalkResult::advance();
      llvm::dbgs() << "at line: " << __LINE__ << "\n";

      // The callee must be defined and private.
      if (!Candidate::isPrivateCallable(callOp.resolveCallable()))
        return WalkResult::advance();

      // TODO: ensure the call graph has one call edge to the callee (this
      // call).
      //      if (CallGraphNode *calleeCGN = CG.lookupNode(
      //            cast<CallableOpInterface>(callableOp).getCallableRegion()))

      llvm::dbgs() << "Candidate: " << callOp << "\n\n";
      candidateCalls.push_back(callOp);
      return WalkResult::advance();
    });
  });
}

std::unique_ptr<Pass> mlir::polygeist::createArgumentPromotionPass() {
  return std::make_unique<ArgumentPromotionPass>();
}
