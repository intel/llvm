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
//        (memref<?x!llvm.struct<(i32, !sycl.accessor...)
//    gpu.return
//  }
//
// The pass "peels off" the i32 struct member:
//
//  gpu.func @kernel_parallel_for(%arg0: i32) {
//    %int_arg = <first struct member>
//    %new_memref = <memref to struct with first member removed>
//    func.call @parallel_for(%int_arg, %new_memref, ...) :
//        (i32, memref<?x!llvm.struct<(!sycl.accessor...)
//    gpu.return
//  }

//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Transforms/Passes.h"

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Conversion/LLVMCommon/StructBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Polygeist/IR/Ops.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"
#include "mlir/Dialect/SYCL/Transforms/Passes.h"
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

  /// Identify call operands that can (and should) be peeled and and peel them.
  /// Note: this function populates a map between the original operand and the
  /// peeled values.
  void peelMembers();

  /// Modify the call by replacing the original operands with their
  /// corresponding peeled members.
  void modifyCall();

  /// Modify the called function by replacing the original operands with their
  /// corresponding peeled members.
  void modifyCallee();

private:
  static Region *getCallableRegion(CallOpInterface callOp);

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

void Candidate::peelMembers() {
  OpBuilder builder(callOp);
  for (unsigned pos = 0; pos < callOp->getNumOperands(); ++pos) {
    Value op = callOp->getOperand(pos);
    if (!isValidMemRefType(op.getType()))
      continue;

    llvm::dbgs() << "operandToPeel: " << op << "\n";
    Location loc = op.getLoc();
    auto opType = cast<MemRefType>(op.getType());
    auto structType = cast<LLVM::LLVMStructType>(opType.getElementType());

    SmallVector<Value> membersPeeled;
    for (unsigned idx = 0; idx < structType.getBody().size(); ++idx) {
      auto memberType = structType.getBody()[idx];
      auto mt0 = MemRefType::get(opType.getShape(), memberType,
                                 opType.getLayout(), opType.getMemorySpace());
      auto subIndexOp = builder.create<polygeist::SubIndexOp>(
          loc, mt0, op, builder.create<arith::ConstantIndexOp>(loc, idx));
      llvm::dbgs() << "subIndexOp: " << subIndexOp << "\n";
      membersPeeled.push_back(subIndexOp);
    }

    operandToMembersPeeled.insert({pos, membersPeeled});
  }
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
}

void Candidate::modifyCallee() {
  Region *callableRgn = getCallableRegion(callOp);
  auto callableOp = cast<CallableOpInterface>(callOp.resolveCallable());
  llvm::dbgs() << "callableOp: " << callableOp << "\n";
  OperandRange funcOperands = callableOp->getLoc()

                                  for (auto funcOp : funcOperands) llvm::dbgs()
                              << "funcOp: " << funcOp << "\n";

  SmallVector<Value, 8> newCalleeOperands;
  for (unsigned pos = 0; pos < callableOp->getNumOperands(); ++pos) {
    if (operandToMembersPeeled.find(pos) == operandToMembersPeeled.end()) {
      newCalleeOperands.push_back(callableOp->getOperand(pos));
      continue;
    }

    for (Value peeledMember : operandToMembersPeeled[pos]) {
      // OpOperand newOperand();
      // newCalleeOperands.push_back(newOperand);
    }
  }

  // callableRgn->eraseOperand(0);
}

Region *Candidate::getCallableRegion(CallOpInterface callOp) {
  if (auto callable = dyn_cast<CallableOpInterface>(callOp.resolveCallable()))
    return callable.getCallableRegion();
  return nullptr;
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
    return signalPassFailure();

  for (Candidate &cand : candidates) {
    cand.peelMembers();
    cand.modifyCall();
    cand.modifyCallee();
  }
}

void ArgumentPromotionPass::collectCandidates(
    ModuleOp module, SmallVectorImpl<Candidate> &candidateCalls) {
  module->walk([&](gpu::GPUFuncOp gpuFuncOp) {
    gpuFuncOp->walk([&](CallOpInterface callOp) {
      // We are only interested in non-recursive tail calls.
      if (!callOp->getBlock()->hasNoSuccessors() ||
          Candidate::isRecursiveCall(callOp))
        return WalkResult::advance();

      // The first operand of the call must not be used by any other
      // operation.
      OperandRange callOperands = callOp.getArgOperands();
      if (callOperands.empty() || !callOperands.front().hasOneUse())
        return WalkResult::advance();

      // The first operand must be a memref with a struct element type.
      if (!Candidate::isValidMemRefType(callOperands.front().getType()))
        return WalkResult::advance();

      // The callee must be defined and private.
      if (!Candidate::isPrivateCallable(callOp.resolveCallable()))
        return WalkResult::advance();

      // TODO: ensure the call graph has one call edge to the callee (this
      // call).
      //      if (CallGraphNode *calleeCGN = CG.lookupNode(
      //            cast<CallableOpInterface>(callableOp).getCallableRegion()))

      llvm::dbgs() << "Candidate: " << callOp << "\n";
      candidateCalls.push_back(callOp);
      return WalkResult::advance();
    });
  });
}

std::unique_ptr<Pass> mlir::polygeist::createArgumentPromotionPass() {
  return std::make_unique<ArgumentPromotionPass>();
}
