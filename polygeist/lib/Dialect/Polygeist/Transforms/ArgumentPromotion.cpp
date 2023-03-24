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

/// Returns true if the callable operation \p callableOp has a callable region
/// and private visibility, and false otherwise.
static bool isPrivateDefinition(CallableOpInterface callableOp) {
  auto sym = dyn_cast<SymbolOpInterface>(callableOp.getOperation());
  return (sym && sym.isPrivate() && callableOp.getCallableRegion());
}

/// Returns true if \p type is 'memref<?xstruct<>>', and false otherwise.
static bool isValidMemRefType(Type type) {
  auto mt = dyn_cast<MemRefType>(type);
  bool isMemRefWithExpectedShape =
      (mt && mt.hasRank() && (mt.getRank() == 1) &&
       ShapedType::isDynamic(mt.getShape()[0]) && mt.getLayout().isIdentity());
  if (!isMemRefWithExpectedShape)
    return false;

  auto structType = dyn_cast<LLVM::LLVMStructType>(mt.getElementType());
  if (!structType || llvm::any_of(structType.getBody(), [](Type memType) {
        return isa<LLVM::LLVMStructType>(memType);
      }))
    return false;
  return true;
}

// Returns true if the block of \p startOp contains an instruction (after \p
// startOp) that has a side effects on \p val.
static auto existsSideEffectAfter(Value val, Operation *startOp) {
  assert(startOp && "Expecting a valid pointer");
  Block *block = startOp->getBlock();

  for (Operation &op : *block) {
    if (op.isBeforeInBlock(startOp) || isMemoryEffectFree(&op))
      continue;

    // Currently bail out if any subsequent operation has side effects.
    // TODO: check whether the operation alias with 'val'.
    return true;
  }
  return false;
}

namespace {

/// Represents an operand that can be peeled.
class CandidateOperand {
public:
  CandidateOperand(Value v, unsigned pos) : val(v), pos(pos) {
    assert(isValidMemRefType(val.getType()) &&
           "Candidate operand does not have the expected type");
  }

  Value value() const { return val; };
  unsigned position() const { return pos; };

  /// Peel the operand and populate \p membersPeeled with the members peeled.
  void peel(CallOpInterface callOp,
            SmallVectorImpl<Value> &membersPeeled) const;

private:
  Value val;    /// Operand value.
  unsigned pos; /// Operand position in the call operator.
};

/// Represents a candidate for the argument promotion transformation.
class Candidate {
public:
  using CandidateOperands = SmallVector<CandidateOperand>;

  Candidate(CallOpInterface callOp, CandidateOperands &&candidates)
      : callOp(callOp), candidateOps(std::move(candidates)) {
    assert(!candidateOps.empty() && "Expecting candidateOps to not be empty");
  }

  void transform();

  StringRef getCalleeName() {
    CallInterfaceCallable callableOp = callOp.getCallableForCallee();
    return callableOp.get<SymbolRefAttr>().getLeafReference();
  }

private:
  /// Peel the call operands.
  /// Note: this function populates a map between the original operands and the
  /// peeled members.
  void peelOperands();

  /// Modify the call by replacing the original operands with their
  /// corresponding peeled members.
  void modifyCall();

  /// Modify the called function by replacing the original operands with their
  /// corresponding peeled members.q
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

private:
  /// The call site to be peeled.
  CallOpInterface callOp;

  /// The operands to be peeled.
  CandidateOperands candidateOps;

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

  /// Return true if the call operation \p callOp is a candidate, and false
  /// otherwise.
  bool isCandidateCall(CallOpInterface callOp) const;

  /// Return true is the callee is a candidate, and false otherwise.
  bool isCandidateCallable(CallableOpInterface callableOp);

  /// Return true if the call \p callOp operand at position \p pos is a
  /// candidate for peeling, and false otherwise.
  bool isCandidateOperand(unsigned pos, CallOpInterface callOp) const;
};

} // namespace

//===----------------------------------------------------------------------===//
// CandidateOperand
//===----------------------------------------------------------------------===//

void CandidateOperand::peel(CallOpInterface callOp,
                            SmallVectorImpl<Value> &membersPeeled) const {
  OpBuilder builder(callOp);
  auto memRefType = cast<MemRefType>(val.getType());
  auto structType = cast<LLVM::LLVMStructType>(memRefType.getElementType());
  unsigned numMembers = structType.getBody().size();

  // Generate code to peel the struct members.
  for (unsigned idx = 0; idx < numMembers; ++idx) {
    auto memberType = structType.getBody()[idx];
    auto mt =
        MemRefType::get(memRefType.getShape(), memberType,
                        memRefType.getLayout(), memRefType.getMemorySpace());
    auto subIndexOp = builder.create<polygeist::SubIndexOp>(
        val.getLoc(), mt, val,
        builder.create<arith::ConstantIndexOp>(val.getLoc(), idx));
    membersPeeled.push_back(subIndexOp);

    LLVM_DEBUG(llvm::dbgs().indent(4) << "generated: " << subIndexOp << "\n");
  }
}

//===----------------------------------------------------------------------===//
// Candidate
//===----------------------------------------------------------------------===//

void Candidate::transform() {
  LLVM_DEBUG({
    StringRef funcName =
        callOp->getParentOfType<FunctionOpInterface>().getName();
    llvm::dbgs() << "\nProcessing candidates in function \"" << funcName
                 << "\":\n";
    llvm::dbgs().indent(2) << callOp << "\n";
    llvm::dbgs().indent(2) << "candidate has " << candidateOps.size()
                           << " peelable operand(s)\n";
  });

  peelOperands();
  modifyCall();
  modifyCallee();
}

void Candidate::peelOperands() {
  for (CandidateOperand candOp : candidateOps) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "peeling operand " << candOp.position() << "\n");
    SmallVector<Value> peeledMembers;
    candOp.peel(callOp, peeledMembers);
    operandToMembersPeeled.insert({candOp.position(), peeledMembers});
  }
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

  LLVM_DEBUG({
    llvm::dbgs().indent(2) << "New call:\n";
    llvm::dbgs().indent(4) << callOp << "\n";
  });
}

void Candidate::modifyCallee() {
  CallableOpInterface callableOp = callOp.resolveCallable();
  auto funcOp =
      cast<func::FuncOp>(callableOp.getCallableRegion()->getParentOp());
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

  LLVM_DEBUG({
    llvm::dbgs().indent(2) << "New Callee:\n";
    llvm::dbgs() << funcOp << "\n";
  });
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
    LLVM_DEBUG(llvm::dbgs()
               << "\nAnalyzing function " << gpuFuncOp.getNameAttr() << ":\n");

    gpuFuncOp->walk([&](CallOpInterface callOp) {
      // Perform some basic checks to filter out call operations that aren't
      // candidates.
      if (!isCandidateCall(callOp) ||
          !isCandidateCallable(callOp.resolveCallable()))
        return;

      // Determine which operands can be peeled and collect them.
      LLVM_DEBUG(llvm::dbgs() << "Analyzing operand(s) of: " << callOp << "\n");
      Candidate::CandidateOperands candidateOps;
      for (unsigned pos = 0; pos < callOp->getNumOperands(); ++pos) {
        if (isCandidateOperand(pos, callOp))
          candidateOps.push_back({callOp->getOperand(pos), pos});
      }

      if (candidateOps.empty()) {
        LLVM_DEBUG(llvm::dbgs().indent(2) << "Not a candidate\n");
        return;
      }

      LLVM_DEBUG(llvm::dbgs().indent(2) << "Found candidate\n");
      candidateCalls.push_back({callOp, std::move(candidateOps)});
    });
  });
}

bool ArgumentPromotionPass::isCandidateCall(CallOpInterface callOp) const {
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

  return true;
}

bool ArgumentPromotionPass::isCandidateOperand(unsigned pos,
                                               CallOpInterface callOp) const {
  assert(pos < callOp->getNumOperands() &&
         "pos must be smaller than the number of call number of operands");

  // The operand must have the expected type(memref<?xstruct<...>>).
  Value operand = callOp->getOperand(pos);
  if (!isValidMemRefType(operand.getType())) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "Operand " << pos << " doesn't have expected type\n");
    return false;
  }

  // The operand must not be used by any instruction (after the call
  // operation) which may read or write it.
  if (existsSideEffectAfter(operand, callOp->getNextNode())) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "Operand " << pos
               << " used after call by operation with side effects\n");
    return false;
  }

  // The corresponding callee operand must only be used by polygeist
  // subIndex operations.
  CallableOpInterface callableOp = callOp.resolveCallable();
  BlockArgument arg = callableOp.getCallableRegion()->getArgument(pos);
  if (llvm::any_of(arg.getUses(), [](OpOperand &use) {
        return !isa<polygeist::SubIndexOp>(use.getOwner());
      })) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "Operand " << pos
               << " used by illegal operation in callee\n");
    return false;
  }

  LLVM_DEBUG(llvm::dbgs().indent(2)
             << "Operand " << pos << " is a candidate\n");

  return true;
}

bool ArgumentPromotionPass::isCandidateCallable(
    CallableOpInterface callableOp) {
  ModuleOp module = getOperation();
  SymbolTableCollection symTable;
  SymbolUserMap userMap(symTable, module);
  [[maybe_unused]] StringRef calleeName =
      cast<SymbolOpInterface>(callableOp.getOperation()).getName();

  // The callee must be defined and private.
  if (!isPrivateDefinition(callableOp)) {
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

  return true;
}

std::unique_ptr<Pass> mlir::polygeist::createArgumentPromotionPass() {
  return std::make_unique<ArgumentPromotionPass>();
}
