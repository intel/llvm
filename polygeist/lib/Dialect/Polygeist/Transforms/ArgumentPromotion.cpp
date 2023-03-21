//===- ArgumentPromotion.cpp - Promote by-reference arguments -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass attempts to peel struct members from the argument pack passed to
// the function called from a SYCL kernel. For example given:
//
//  gpu.func @kernel_parallel_for(%arg0: i32) {
//    func.call @parallel_for(%0, ...) :
//        (memref<?x!llvm.struct<(i32, !sycl.accessor), ...)
//    gpu.return
//  }
//
// The pass modifies the call site (and the callee):
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Polygeist/IR/Ops.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arg-promotion"
#define REPORT_DEBUG_TYPE DEBUG_TYPE "-report"

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_ARGUMENTPROMOTION
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

namespace {

/// Represent a candidate for the argument promotion transformation.
class Candidate {
public:
  Candidate(CallOpInterface callOp) : callOp(callOp) {
    assert(!isRecursiveCall(callOp) && "Callee must not be recursive");
  }

  /// Returns true if the callable operation \p callableOp has a callable region
  /// and private visibility, and false otherwise.
  static bool isPrivateCallable(CallableOpInterface callableOp);

  /// Returns true if the call \p callOp is recursive, and false otherwise.
  static bool isRecursiveCall(CallOpInterface callOp);

  /// Returns true if \p type is a pointer to a struct (memref<? x struct<>>),
  /// and false otherwise.
  static bool isValidMemRefType(Type type);

  /// Perform the transformation.
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
  /// the arguments starting at position \p pos in the callable region.
  void replaceUsesOfArgument(Value origArg, unsigned pos,
                             Region &callableRgn) const;

  /// The candidate call site.
  CallOpInterface callOp;

  /// Map between the position of the original call operand and its
  /// corresponding peeled arguments.
  std::map<unsigned, SmallVector<Value>> operandToMembersPeeled;
};

class ArgumentPromotionPass
    : public polygeist::impl::ArgumentPromotionBase<ArgumentPromotionPass> {
public:
  using ArgumentPromotionBase<ArgumentPromotionPass>::ArgumentPromotionBase;

  void runOnOperation() override;

private:
  /// Populate \p candidates with call operations to transform.
  void collectCandidates(SmallVectorImpl<Candidate> &candidates);

  /// Return true if the call operation \p callOp is a candidate and false
  /// otherwise.
  bool isCandidate(CallOpInterface callOp);

  /// Return true is the callee is a candidate and false otherwise.
  bool isCandidate(CallableOpInterface callableOp);
};

} // namespace

//===----------------------------------------------------------------------===//
// Candidate
//===----------------------------------------------------------------------===//

bool Candidate::isPrivateCallable(CallableOpInterface callableOp) {
  auto sym = dyn_cast<SymbolOpInterface>(callableOp.getOperation());
  return (sym && sym.isPrivate() && callableOp.getCallableRegion());
}

bool Candidate::isRecursiveCall(CallOpInterface callOp) {
  if (auto callable = dyn_cast<CallableOpInterface>(callOp.resolveCallable())) {
    Region *callableRegion = callable.getCallableRegion();
    return callableRegion &&
           callableRegion->isAncestor(callOp->getParentRegion());
  }
  return false;
}

bool Candidate::isValidMemRefType(Type type) {
  // We want a ranked memref type with shape 'memref<?x...>'.
  auto mt = dyn_cast<MemRefType>(type);
  if (!mt || !mt.hasRank() || mt.getShape().size() != 1 ||
      !ShapedType::isDynamic(mt.getShape()[0]) || !mt.getLayout().isIdentity())
    return false;

  // The element type must be a struct.
  if (!dyn_cast<LLVM::LLVMStructType>(mt.getElementType()))
    return false;

  return true;
}

void Candidate::transform() {
  LLVM_DEBUG(llvm::dbgs() << "\nProcessing candidate:\n" << callOp << "\n");

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

    // Generate code to peel the struct members.
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

      LLVM_DEBUG(
          { llvm::dbgs().indent(2) << "- generated: " << subIndexOp << "\n"; });
    }

    operandToMembersPeeled.insert({pos, membersPeeled});
  }

  LLVM_DEBUG(llvm::dbgs() << "Peeled " << numMembersPeeled << " members.\n\n");
  return numMembersPeeled;
}

void Candidate::modifyCall() {
  const unsigned numCallOperands = callOp->getNumOperands();
  const unsigned numKeys = operandToMembersPeeled.size();
  unsigned numMembersPeeled = 0;
  for (const auto &entry : operandToMembersPeeled)
    numMembersPeeled += entry.second.size();
  assert(numCallOperands - numKeys + numMembersPeeled > 0);

  SmallVector<Value> newCallOperands;
  newCallOperands.reserve(numCallOperands - numKeys + numMembersPeeled);

  for (unsigned pos = 0; pos < numCallOperands; ++pos) {
    if (operandToMembersPeeled.find(pos) == operandToMembersPeeled.end()) {
      newCallOperands.push_back(callOp->getOperand(pos));
      continue;
    }

    for (Value peeledMember : operandToMembersPeeled[pos])
      newCallOperands.push_back(peeledMember);
  }
  assert(newCallOperands.size() >= numCallOperands);

  callOp->setOperands(newCallOperands);
  LLVM_DEBUG(llvm::dbgs() << "New call:\n" << callOp << "\n\n");
}

void Candidate::modifyCallee() {
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
      newArgAttrs.push_back(origArgAttrs ? origArgAttrs[origPos] : Attribute());
      continue;
    }

    // Replace the argument at 'pos' with the new arguments we peeled.
    const SmallVector<Value> &newArgs = operandToMembersPeeled[origPos];
    replaceArgumentWith(pos, callableRgn, newArgs, newArgAttrs);

    // Delete the original argument.
    pos += newArgs.size();
    callableRgn.eraseArgument(pos);
  }

  auto newFuncType =
      FunctionType::get(funcOp.getContext(), callableRgn.getArgumentTypes(),
                        funcOp.getResultTypes());

  funcOp.setFunctionType(newFuncType);
  funcOp.setAllArgAttrs(newArgAttrs);

  LLVM_DEBUG(llvm::dbgs() << "New Callee:\n" << funcOp << "\n\n");
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
  SmallVector<Candidate> candidates;
  collectCandidates(candidates);

  for (Candidate &cand : candidates)
    cand.transform();
}

void ArgumentPromotionPass::collectCandidates(
    SmallVectorImpl<Candidate> &candidateCalls) {
  ModuleOp module = getOperation();
  SymbolTableCollection symTable;
  SymbolUserMap userMap(symTable, module);

  // Collect candidate call operations in GPU kernels.
  module->walk([&](gpu::GPUFuncOp gpuFuncOp) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "Analyzing: "
          << cast<SymbolOpInterface>(gpuFuncOp.getOperation()).getNameAttr()
          << "\n";
    });

    gpuFuncOp->walk([&](CallOpInterface callOp) {
      if (!isCandidate(callOp) ||
          !isCandidate(cast<CallableOpInterface>(callOp.resolveCallable())))
        return;

      LLVM_DEBUG(llvm::dbgs() << "Found candidate: " << callOp << "\n\n");
      candidateCalls.push_back(callOp);
    });
  });
}

bool ArgumentPromotionPass::isCandidate(CallOpInterface callOp) {
  // Only tail calls are candidates.
  if (!callOp->getBlock()->hasNoSuccessors()) {
    LLVM_DEBUG({
      llvm::dbgs() << "Not a candidate: " << callOp << "\n";
      llvm::dbgs().indent(2) << "not a tail call\n";
    });
    return false;
  }

  // Calls with no arguments are not interesting.
  OperandRange callOperands = callOp.getArgOperands();
  if (callOperands.empty()) {
    LLVM_DEBUG({
      llvm::dbgs() << "Not a candidate: " << callOp << "\n";
      llvm::dbgs().indent(2) << "no arguments\n";
    });
    return false;
  }

  // Calls to recursive functions are not candidates.
  if (Candidate::isRecursiveCall(callOp)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Not a candidate: " << callOp << "\n";
      llvm::dbgs().indent(2) << "call is recursive\n";
    });
    return false;
  }

  // At least one operand must peelable (i.e. a memref with struct element).
  if (llvm::all_of(callOperands, [](Value op) {
        return !Candidate::isValidMemRefType(op.getType());
      })) {
    LLVM_DEBUG({
      llvm::dbgs() << "Not a candidate: " << callOp << "\n";
      llvm::dbgs().indent(2) << "no peelable arguments\n";
    });
    return false;
  }

  // Returns the iterator to \p op in its containing block.
  auto getIter = [](Operation *op) {
    Block *block = op->getBlock();
    auto it = block->begin();
    for (; it != block->end(); ++it) {
      Operation *currOp = &*it;
      if (currOp == op)
        return it;
    }
    return block->end();
  };

  // Returns true if the block of \p startOp contains an instruction that has
  // a side effects after \p startOp.
  auto existsSideEffectsAfter = [&](Operation *startOp) {
    Block *block = startOp->getBlock();
    auto it = getIter(startOp);
    assert(it != block->end());
    WalkResult walk = block->walk(++it, block->end(), [](Operation *op) {
      if (!isMemoryEffectFree(op))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    return walk.wasInterrupted();
  };

  // Peelable operands should not be used by an instruction (after the call)
  // that has a side effect.
  for (Value callOperand : callOperands) {
    if (!Candidate::isValidMemRefType(callOperand.getType()))
      continue;

    if (existsSideEffectsAfter(callOp)) {
      LLVM_DEBUG({
        llvm::dbgs() << "Not a candidate: " << callOp << "\n";
        llvm::dbgs().indent(2)
            << "found operation(s) with side effects after call\n";
      });
      return false;
    }
  }

  return true;
}

bool ArgumentPromotionPass::isCandidate(CallableOpInterface callableOp) {
  ModuleOp module = getOperation();
  SymbolTableCollection symTable;
  SymbolUserMap userMap(symTable, module);
  StringRef calleeName =
      cast<SymbolOpInterface>(callableOp.getOperation()).getName();

  // The callee must be defined and private.
  if (!Candidate::isPrivateCallable(callableOp)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Not a candidate: " << calleeName << "\n";
      llvm::dbgs().indent(2) << "callee not privately defined\n";
    });
    return false;
  }

  // The callee must have a single call site.
  unsigned numUsers = userMap.getUsers(callableOp).size();
  if (numUsers != 1) {
    LLVM_DEBUG({
      llvm::dbgs() << "Not a candidate: " << calleeName << "\n";
      llvm::dbgs().indent(2) << "has " << numUsers << " call sites\n";
    });
    return false;
  }

  // In the callee, the members of the struct must be accessed via
  // polygeist subIndex operations.
  Value firstArg = callableOp.getCallableRegion()->getArgument(0);
  if (llvm::any_of(firstArg.getUses(), [](OpOperand &use) {
        return !isa<polygeist::SubIndexOp>(use.getOwner());
      })) {
    LLVM_DEBUG({
      llvm::dbgs() << "Not a candidate: " << calleeName << "\n";
      llvm::dbgs().indent(2) << "missing polygeist subindex operations\n";
    });
    return false;
  }

  return true;
}

std::unique_ptr<Pass> mlir::polygeist::createArgumentPromotionPass() {
  return std::make_unique<ArgumentPromotionPass>();
}
