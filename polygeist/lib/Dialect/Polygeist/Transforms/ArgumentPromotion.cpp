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

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Polygeist/IR/PolygeistOps.h"
#include "mlir/Dialect/Polygeist/Utils/TransformUtils.h"
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

// Returns true if the block of \p startOp contains an operation (after startOp)
// that has a side effects on \p val, and false otherwise.
static bool existsSideEffectAfter(Value val, Operation *startOp,
                                  AliasAnalysis &aliasAnalysis) {
  assert(startOp && "Expecting a valid pointer");

  for (Operation &op : *startOp->getBlock()) {
    if (op.isBeforeInBlock(startOp) || isMemoryEffectFree(&op))
      continue;

    // An operation with unknown side effects is conservatively assumed to have
    // a side effect on `val`.
    auto MEI = dyn_cast<MemoryEffectOpInterface>(op);
    if (!MEI)
      return true;

    // Conservatively assume an operation with a nested region has side effects
    // on 'val'.
    if (op.hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      return true;

    SmallVector<MemoryEffects::EffectInstance, 1> effects;
    MEI.getEffects(effects);
    for (MemoryEffects::EffectInstance &effect : effects) {
      Value effectVal = effect.getValue();
      if (effectVal == val)
        return true;

      // Check whether the operation has an effect on a value that is aliased
      // with 'val'.
      AliasResult aliasRes = aliasAnalysis.alias(effectVal, val);
      if (!aliasRes.isNo())
        return true;
    }
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

  CallOpInterface getCallOp() { return callOp; }
  CallableOpInterface getCallableOp() {
    return cast<CallableOpInterface>(callOp.resolveCallable());
  }
  CandidateOperands &getCandidateOperands() { return candidateOps; }

  StringRef getParentFunctionName() {
    return callOp->getParentOfType<FunctionOpInterface>().getName();
  }

  StringRef getCalleeName() {
    CallInterfaceCallable callableOp = callOp.getCallableForCallee();
    return callableOp.get<SymbolRefAttr>().getLeafReference();
  }

  /// Modify the call by replacing the original operands with their
  /// corresponding peeled members.
  void modifyCall();

  /// Modify the called function by replacing the original operands with their
  /// corresponding peeled members.
  void modifyCallee();

private:
  /// Peel the call operands.
  /// Note: this function populates a map between the original operands and the
  /// peeled members.
  void peelOperands();

  /// Replace the argument at position \p pos in the region \p callableRgn with
  /// \p newArgs. The reference parameter \p newArgAttrs is filled with the new
  /// argument attributes. The boolean \p useNoAliasAttrForNewArgs indicate
  /// whether to add the 'noalias' attribute on the new arguments.
  void replaceArgumentWith(unsigned pos, Region &callableRgn,
                           const SmallVector<Value> &newArgs,
                           SmallVector<Attribute> &newArgAttrs,
                           bool useNoAliasAttrForNewArgs) const;

  /// Replace uses of the argument \p origArg in the region \p callableRgn
  /// with the arguments starting at position \p pos in the callable region.
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
  using Candidates = SmallVector<Candidate>;

  void runOnOperation() override;

private:
  /// Populate \p candidates with call operations to transform.
  void
  collectCandidates(std::map<CallableOpInterface, Candidates> &callableToCalls,
                    AliasAnalysis &aliasAnalysis);

  /// Return true if the call operation \p callOp is a candidate, and false
  /// otherwise.
  bool isCandidateCall(CallOpInterface callOp) const;

  /// Return true is the callee is a candidate, and false otherwise.
  bool isCandidateCallable(CallableOpInterface callableOp,
                           const polygeist::FunctionKernelInfo &funcKernelInfo);

  /// Return true if the call \p callOp operand at position \p pos is a
  /// candidate for peeling, and false otherwise.
  bool isCandidateOperand(unsigned pos, CallOpInterface callOp,
                          AliasAnalysis &aliasAnalysis) const;

  // Return true if all candidate operands in \p candidateOperandMap have the
  // same position and false otherwise.
  bool haveSameCandidateOperands(
      const std::map<CallOpInterface, Candidate::CandidateOperands>
          &candidateOperandMap) const;
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

void Candidate::modifyCall() {
  peelOperands();

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
  CallableOpInterface callableOp =
      cast<CallableOpInterface>(callOp.resolveCallable());
  Region &callableRgn = *callableOp.getCallableRegion();
  auto funcOp = callableRgn.getParentOfType<FunctionOpInterface>();
  const ArrayAttr origArgAttrs = funcOp.getAllArgAttrs();
  SmallVector<Attribute> newArgAttrs;

  // Replace the old arguments with the new ones.
  unsigned pos = 0;
  const unsigned orinNumArgs = callableRgn.getNumArguments();
  for (unsigned origPos = 0; origPos < orinNumArgs; ++origPos) {
    if (operandToMembersPeeled.find(origPos) == operandToMembersPeeled.end()) {
      newArgAttrs.push_back(origArgAttrs ? origArgAttrs[origPos] : Attribute());
      ++pos;
      continue;
    }

    // Replace the argument at 'pos' with the new arguments we peeled.
    const SmallVector<Value> &newArgs = operandToMembersPeeled[origPos];
    bool useNoAliasAttrForNewArgs = (operandToMembersPeeled.size() == 1);
    replaceArgumentWith(pos, callableRgn, newArgs, newArgAttrs,
                        useNoAliasAttrForNewArgs);

    // Delete the original argument.
    pos += newArgs.size();
    callableRgn.eraseArgument(pos);
  }

  auto newFuncType =
      FunctionType::get(funcOp.getContext(), callableRgn.getArgumentTypes(),
                        funcOp.getResultTypes());

  funcOp.setType(newFuncType);
  funcOp.setAllArgAttrs(newArgAttrs);

  // Change the linkage from linkonce_odr to private.
  // There can be globals of the same name (with linkonce_odr linkage) in
  // another translation unit. As they have different arguments, we need to
  // change the linkage of the modified function to private.
  if (polygeist::isLinkonceODR(funcOp))
    polygeist::privatize(funcOp);

  LLVM_DEBUG(llvm::dbgs() << "\nNew Callee:\n" << funcOp << "\n";);
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

void Candidate::replaceArgumentWith(unsigned pos, Region &callableRgn,
                                    const SmallVector<Value> &newArgs,
                                    SmallVector<Attribute> &newArgAttrs,
                                    bool useNoAliasAttrForNewArgs) const {
  assert(!newArgs.empty() && "Expecting a non-empty vector");

  Value origArg = callableRgn.getArgument(pos);
  MLIRContext *ctx = callableRgn.getContext();

  for (unsigned offset = 0; offset < newArgs.size(); ++offset) {
    callableRgn.insertArgument(pos + offset, newArgs[offset].getType(),
                               callableRgn.getLoc());
    if (useNoAliasAttrForNewArgs) {
      NamedAttribute noAliasAttr(
          StringAttr::get(ctx, LLVM::LLVMDialect::getDialectNamespace() + "." +
                                   llvm::Attribute::getNameFromAttrKind(
                                       llvm::Attribute::NoAlias)),
          UnitAttr::get(ctx));
      newArgAttrs.push_back(DictionaryAttr::get(ctx, {noAliasAttr}));
    } else
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
  AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
  std::map<CallableOpInterface, Candidates> callableToCalls;
  collectCandidates(callableToCalls, aliasAnalysis);

  for (auto &entry : callableToCalls) {
    CallableOpInterface callableOp = entry.first;
    Candidates &candidates = entry.second;
    assert(llvm::all_of(candidates,
                        [&](Candidate &cand) {
                          return cand.getCallableOp() == callableOp;
                        }) &&
           "Expecting candidates to have the same callable");

    for (auto *it = candidates.begin(); it != candidates.end(); ++it) {
      Candidate &cand = *it;
      LLVM_DEBUG({
        llvm::dbgs() << "\nProcessing candidate in function \""
                     << cand.getParentFunctionName() << "\":\n";
        llvm::dbgs().indent(2) << cand.getCallOp() << "\n";
        llvm::dbgs().indent(2)
            << "candidate has " << cand.getCandidateOperands().size()
            << " peelable operand(s)\n";
      });

      cand.modifyCall();
      if (it == candidates.begin())
        candidates.front().modifyCallee();
    }
  }
}

void ArgumentPromotionPass::collectCandidates(
    std::map<CallableOpInterface, Candidates> &callableToCalls,
    AliasAnalysis &aliasAnalysis) {
  ModuleOp module = getOperation();
  SymbolTableCollection symTable;
  SymbolUserMap userMap(symTable, module);
  auto gpuModule = dyn_cast<gpu::GPUModuleOp>(
      module->getRegion(0).front().getOperations().front());
  if (!gpuModule)
    return;
  polygeist::FunctionKernelInfo funcKernelInfo(gpuModule);

  // Search for candidate call operations in GPU kernels.
  gpuModule->walk([&](gpu::GPUFuncOp gpuFuncOp) {
    assert(gpuFuncOp.isKernel() && "Expecting a kernel");
    LLVM_DEBUG(llvm::dbgs() << "\nAnalyzing GPU kernel "
                            << gpuFuncOp.getNameAttr() << ":\n");

    gpuFuncOp->walk([&](CallOpInterface callOp) {
      Operation *op = callOp.resolveCallable();
      auto callableOp = dyn_cast_or_null<CallableOpInterface>(op);
      if (!callableOp) {
        LLVM_DEBUG(llvm::dbgs().indent(4)
                   << callOp << " does not resolve to a callable\n");
        return;
      }
      [[maybe_unused]] StringRef callableName =
          cast<SymbolOpInterface>(callableOp.getOperation()).getName();
      LLVM_DEBUG({
        llvm::dbgs() << "Callable: " << callableName << "\n";
        llvm::dbgs().indent(2) << "has " << userMap.getUsers(callableOp).size()
                               << " call site(s)\n";
      });

      // Perform basic checks to ensure the callable is OK.
      if (!isCandidateCallable(callableOp, funcKernelInfo))
        return;

      // Perform basic checks to ensure all calls to the callable are OK.
      if (llvm::any_of(userMap.getUsers(callableOp), [this](Operation *op) {
            if (CallOpInterface callOp = dyn_cast<CallOpInterface>(op))
              return !isCandidateCall(callOp);
            return true;
          }))
        return;

      // Collect candidate operands for each call site.
      std::map<CallOpInterface, Candidate::CandidateOperands>
          candidateOperandMap;
      for (Operation *op : userMap.getUsers(callableOp)) {
        auto callOp = dyn_cast<CallOpInterface>(op);
        if (!callOp)
          continue;
        LLVM_DEBUG(llvm::dbgs().indent(2)
                   << "analyzing operand(s) of: " << *callOp << "\n");
        Candidate::CandidateOperands candidateOps;
        for (unsigned pos = 0; pos < callOp->getNumOperands(); ++pos) {
          if (isCandidateOperand(pos, callOp, aliasAnalysis))
            candidateOps.push_back({callOp->getOperand(pos), pos});
        }
        if (candidateOps.empty())
          return;
        candidateOperandMap.insert({callOp, std::move(candidateOps)});
      }

      // Ensure all call sites have the same candidate operands.
      if (!candidateOperandMap.empty() &&
          !haveSameCandidateOperands(candidateOperandMap)) {
        LLVM_DEBUG(llvm::dbgs().indent(2) << "not a candidate: call sites have "
                                             "different candidate operands\n");
        return;
      }

      // Create the candidate.
      for (const auto &entry : candidateOperandMap) {
        CallOpInterface callOp = cast<CallOpInterface>(entry.first);
        Candidate::CandidateOperands candidateOps = entry.second;
        LLVM_DEBUG(llvm::dbgs().indent(2)
                   << "Found candidate: " << callOp << "\n");
        callableToCalls[callableOp].push_back(
            {callOp, std::move(candidateOps)});
      }
    });
  });
}

bool ArgumentPromotionPass::isCandidateCall(CallOpInterface callOp) const {
  // Only calls in function exiting blocks are candidates.
  if (!callOp->getBlock()->hasNoSuccessors()) {
    LLVM_DEBUG({
      llvm::dbgs() << "Not a candidate: " << callOp << "\n";
      llvm::dbgs().indent(2) << "not in an existing block\n";
    });
    return false;
  }

  return true;
}

bool ArgumentPromotionPass::isCandidateOperand(
    unsigned pos, CallOpInterface callOp, AliasAnalysis &aliasAnalysis) const {
  assert(pos < callOp->getNumOperands() &&
         "pos must be smaller than the number of call number of operands");

  // The operand must have the expected type(memref<?xstruct<...>>).
  Value operand = callOp->getOperand(pos);
  if (!isValidMemRefType(operand.getType())) {
    LLVM_DEBUG(llvm::dbgs().indent(4)
               << "operand " << pos << " doesn't have expected type\n");
    return false;
  }

  // The operand must not be used by any instruction (after the call
  // operation) which may read or write it.
  if (existsSideEffectAfter(operand, callOp->getNextNode(), aliasAnalysis)) {
    LLVM_DEBUG(llvm::dbgs().indent(4)
               << "operand " << pos
               << " used after call by operation with side effects\n");
    return false;
  }

  // The corresponding callee operand must only be used by polygeist subIndex
  // operations.
  CallableOpInterface callableOp =
      cast<CallableOpInterface>(callOp.resolveCallable());
  BlockArgument arg = callableOp.getCallableRegion()->getArgument(pos);
  if (llvm::any_of(arg.getUses(), [](OpOperand &use) {
        return !isa<polygeist::SubIndexOp>(use.getOwner());
      })) {
    LLVM_DEBUG(llvm::dbgs().indent(4)
               << "operand " << pos
               << " used by illegal operation in callee\n");
    return false;
  }

  LLVM_DEBUG(llvm::dbgs().indent(4)
             << "operand " << pos << " is a candidate\n");

  return true;
}

bool ArgumentPromotionPass::isCandidateCallable(
    CallableOpInterface callableOp,
    const polygeist::FunctionKernelInfo &funcKernelInfo) {
  Operation *op = callableOp;
  auto funcOp = cast<FunctionOpInterface>(op);
  // The function must be defined, and private or with linkonce_odr linkage.
  if (funcOp.isExternal() ||
      (!funcOp.isPrivate() && !polygeist::isLinkonceODR(funcOp))) {
    LLVM_DEBUG(
        llvm::dbgs().indent(2)
        << "not a candidate: not defined or private or linkonce_odr linkage\n");
    return false;
  }

  // Functions with no arguments are not interesting.
  if (callableOp.getCallableRegion()->args_empty()) {
    LLVM_DEBUG(llvm::dbgs().indent(2) << "not a candidate: no arguments\n");
    return false;
  }

  // Ensure all the call sites for this function are either in a GPU kernel or
  // in a function that is called directly by a GPU kernel.
  // TODO: Could generalize by checking that the call chain from the GPU kernel
  // are all candidates.
  Optional<unsigned> maxDepth =
      funcKernelInfo.getMaxDepthFromAnyGPUKernel(funcOp);
  assert(maxDepth.has_value() &&
         "Expecting func to be called from a GPU kernel");
  assert(maxDepth.value() != 0 && "Expecting func is not itself a GPU kernel");
  if (maxDepth.value() > 2) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "not a candidate: found call site that is called by a GPU "
                  "kernel with depth more than 2.\n");
    return false;
  }

  return true;
}

// Ensure all call sites have the same candidate operands.
// Note: we could improve this condition and find the intersection of peelable
// operands for all call sites to increase the potential opportunity for
// peeling.
bool ArgumentPromotionPass::haveSameCandidateOperands(
    const std::map<CallOpInterface, Candidate::CandidateOperands>
        &candidateOperandMap) const {
  assert(!candidateOperandMap.empty() && "Expecting a nonempty map");

  auto it = candidateOperandMap.cbegin();
  const Candidate::CandidateOperands &firstCandOps = it->second;

  ++it;
  for (; it != candidateOperandMap.cend(); ++it) {
    const Candidate::CandidateOperands candOps = it->second;
    if (firstCandOps.size() != candOps.size())
      return false;
    for (unsigned opNum = 0; opNum < firstCandOps.size(); ++opNum) {
      if (firstCandOps[opNum].position() != candOps[opNum].position())
        return false;
    }
  }

  return true;
}

std::unique_ptr<Pass> mlir::polygeist::createArgumentPromotionPass() {
  return std::make_unique<ArgumentPromotionPass>();
}
