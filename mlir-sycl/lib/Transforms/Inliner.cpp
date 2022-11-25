//===- Inliner.cpp - Inline calls -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"
#include "mlir/Dialect/SYCL/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_INLINER
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "inlining"

using namespace mlir;

namespace {

/// This struct represents a resolved call to a given call graph node. Given
/// that the call does not actually contain a direct reference to the
/// Region(CallGraphNode) that it is dispatching to, we need to resolve them
/// explicitly.
class ResolvedCall {
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const ResolvedCall &);

public:
  ResolvedCall(CallOpInterface Call, const CallGraphNode *SrcNode,
               const CallGraphNode *TgtNode)
      : Call(Call), SrcNode(SrcNode), TgtNode(TgtNode) {
    assert(SrcNode && "Expecting valid source node");
    assert(TgtNode && "Expecting valid target node");
  }

  /// Return the func::FuncOp called by `callOp`.
  static FunctionOpInterface getCalledFunction(const CallOpInterface &Call) {
    if (SymbolRefAttr SymAttr = const_cast<CallOpInterface &>(Call)
                                    .getCallableForCallee()
                                    .dyn_cast<SymbolRefAttr>())
      return SymbolTable::lookupNearestSymbolFrom(Call, SymAttr);

    return nullptr;
  }

  CallOpInterface Call;
  const CallGraphNode *SrcNode, *TgtNode;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const ResolvedCall &RC) {
  auto PrintCallable = [&](Operation *Op) {
    llvm::TypeSwitch<Operation *>(Op)
        .Case<func::FuncOp>([&](func::FuncOp Op) { OS << Op.getSymName(); })
        .Case<gpu::GPUFuncOp>([&](gpu::GPUFuncOp Op) { Op.print(OS); })
        .Default(
            [&](Operation *) { assert(false && "Unhandled operation kind"); });
  };

  if (!RC.SrcNode->isExternal()) {
    OS << "SrcNode: ";
    PrintCallable(RC.SrcNode->getCallableRegion()->getParentOp());
    OS << "\n";
  }

  if (!RC.TgtNode->isExternal()) {
    OS << "TgtNode: ";
    PrintCallable(RC.TgtNode->getCallableRegion()->getParentOp());
    OS << "\n";
  }

  OS << "Call: ";
  PrintCallable(ResolvedCall::getCalledFunction(RC.Call));
  OS << "\n";

  return OS;
}

/// This class represents a specific call graph SCC.
class CallGraphSCC {
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const CallGraphSCC &);

public:
  CallGraphSCC(llvm::scc_iterator<const CallGraph *> &ParentIterator)
      : ParentIterator(ParentIterator) {}

  /// Return a range over the nodes within this SCC.
  std::vector<CallGraphNode *>::iterator begin() { return Nodes.begin(); }
  std::vector<CallGraphNode *>::iterator end() { return Nodes.end(); }

  /// Reset the nodes of this SCC with those provided.
  void reset(const std::vector<CallGraphNode *> &NewNodes) { Nodes = NewNodes; }

  /// Remove the given node from this SCC.
  void remove(CallGraphNode *Node) {
    auto It = llvm::find(Nodes, Node);
    if (It != Nodes.end()) {
      Nodes.erase(It);
      ParentIterator.ReplaceNode(Node, nullptr);
    }
  }

private:
  std::vector<CallGraphNode *> Nodes;
  llvm::scc_iterator<const CallGraph *> &ParentIterator;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const CallGraphSCC &SCC) {
  auto PrintCallable = [&](Operation *Op) {
    llvm::TypeSwitch<Operation *>(Op)
        .Case<func::FuncOp>([&](func::FuncOp Op) { OS << Op.getSymName(); })
        .Case<gpu::GPUFuncOp>([&](gpu::GPUFuncOp Op) { Op.print(OS); })
        .Default(
            [&](Operation *) { assert(false && "Unhandled operation kind"); });
  };

  OS << "{ ";
  llvm::interleaveComma(SCC.Nodes, OS, [&](const CallGraphNode *CGN) {
    if (!CGN->isExternal())
      if (Region *Callable = CGN->getCallableRegion())
        PrintCallable(Callable->getParentOp());
  });
  OS << " }";
  return OS;
}

/// Generic SCC Inliner.
class Inliner : public InlinerInterface {
public:
  Inliner(MLIRContext *Ctx, CallGraph &CG, SymbolTableCollection &SymTable)
      : InlinerInterface(Ctx), CG(CG), SymbolTable(SymTable) {}

  ResolvedCall &getCall(unsigned Index) {
    assert(Index < Calls.size() && "Out of bound index");
    return Calls[Index];
  }
  SmallVectorImpl<ResolvedCall> &getCalls() { return Calls; }
  void clear() { Calls.clear(); }

  CallGraph &getCG() const { return CG; }

  SymbolTableCollection &getSymbolTable() const { return SymbolTable; }

  /// Attempt to inline calls within the given scc, and run simplifications,
  /// until a fixed point is reached. This allows for the inlining of newly
  /// devirtualized calls. Returns failure if there was a fatal error during
  /// inlining.
  static LogicalResult inlineSCC(Inliner &Inliner, CallGraphSCC &SCC,
                                 MLIRContext *Ctx);

protected:
  /// Attempt to inline calls within the given SCC.
  static void inlineCallsInSCC(Inliner &Inliner, CallGraphSCC &SCC);

  /// Collect all of the callable operations within the given \p SrcNode.
  static void collectCallOps(const CallGraphNode &SrcNode, const CallGraph &CG,
                             SymbolTableCollection &SymTable,
                             SmallVectorImpl<ResolvedCall> &Calls);

  /// Returns true if the given call should be inlined.
  virtual bool shouldInline(ResolvedCall &ResolvedCall) const = 0;

private:
  /// The current set of call instructions to consider for inlining.
  SmallVector<ResolvedCall, 8> Calls;

  /// The call graph being operated on.
  CallGraph &CG;

  /// A symbol table to use when resolving call lookups.
  SymbolTableCollection &SymbolTable;
};

/// Inlines sycl.call operations if the callee has the 'alwaysinline' attribute.
class AlwaysInliner : public Inliner {
public:
  AlwaysInliner(MLIRContext *Ctx, CallGraph &CG,
                SymbolTableCollection &SymTable)
      : Inliner(Ctx, CG, SymTable) {}

protected:
  /// Returns true if the given call should be inlined.
  bool shouldInline(ResolvedCall &ResolvedCall) const final;
};

/// A pass that inlines sycl.call operations if the callee has the
/// 'alwaysinline' attribute.
class AlwaysInlinerPass : public impl::InlinerBase<AlwaysInlinerPass> {
public:
  AlwaysInlinerPass() = default;
  AlwaysInlinerPass(const AlwaysInlinerPass &) = default;

  /// Returns the command-line argument attached to this pass.
  llvm::StringRef getArgument() const final { return "always-inline"; }

  llvm::StringRef getDescription() const final {
    return "Inline function calls with the 'alwaysinline' attribute";
  }

  llvm::StringRef getName() const final { return "AlwaysInliner"; }

  static constexpr llvm::StringLiteral getPassName() {
    return llvm::StringLiteral("AlwaysInliner");
  }

  void runOnOperation() final;
};

} // namespace

/// Run a given transformation over the SCCs of the call graph in a bottom up
/// traversal.
static LogicalResult
runTransformOnSCCs(const CallGraph &CG,
                   function_ref<LogicalResult(CallGraphSCC &)> SCCTransformer) {
  llvm::scc_iterator<const CallGraph *> CGI = llvm::scc_begin(&CG);
  CallGraphSCC SCC(CGI);
  while (!CGI.isAtEnd()) {
    // Copy the current SCC and increment so that the transformer can modify
    // the SCC without invalidating our iterator.
    SCC.reset(*CGI);
    ++CGI;
    if (failed(SCCTransformer(SCC)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Inliner
//===----------------------------------------------------------------------===//

LogicalResult Inliner::inlineSCC(Inliner &Inliner, CallGraphSCC &SCC,
                                 MLIRContext *Ctx) {
  Inliner::inlineCallsInSCC(Inliner, SCC);
  return success();
}

/// Attempt to inline calls within the given scc. This function returns
/// success if any calls were inlined, failure otherwise.
void Inliner::inlineCallsInSCC(Inliner &Inliner, CallGraphSCC &SCC) {
  for (CallGraphNode *SrcNode : SCC) {
    if (SrcNode->isExternal())
      continue;

    Inliner::collectCallOps(*SrcNode, Inliner.getCG(), Inliner.getSymbolTable(),
                            Inliner.getCalls());
  }

  if (Inliner.getCalls().empty())
    return;

  LLVM_DEBUG({
    llvm::dbgs() << "* Inliner: SCC: " << SCC << "\n";
    llvm::dbgs() << "* Inliner: Initial calls in SCC are: {\n";
    for (unsigned i = 0, e = Inliner.getCalls().size(); i < e; ++i)
      llvm::dbgs() << "  " << i << ". " << Inliner.getCall(i).Call << ",\n";
    llvm::dbgs() << "}\n";
  });

  // Try to inline each of the call operations. Don't cache the end iterator
  // here as more calls may be added during inlining.
  for (unsigned I = 0; I < Inliner.getCalls().size(); ++I) {
    ResolvedCall It = Inliner.getCall(I);
    CallOpInterface Call = It.Call;

    bool DoInline = Inliner.shouldInline(It);
    LLVM_DEBUG(llvm::dbgs() << ((DoInline) ? "* Inlining call: "
                                           : "* Not inlining call: ")
                            << I << ". " << Call << "\n";);
    if (!DoInline)
      continue;

    Region *TgtRegion = It.TgtNode->getCallableRegion();
    LogicalResult InlineRes = inlineCall(
        Inliner, Call, cast<CallableOpInterface>(TgtRegion->getParentOp()),
        TgtRegion, /*shouldCloneInlinedRegion=*/true);

    if (failed(InlineRes)) {
      LLVM_DEBUG(llvm::dbgs() << "** Failed to inline\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "** Inline succeeded\n");

    // then erase the call.
    Call.erase();
  }

  Inliner.clear();
}

void Inliner::collectCallOps(const CallGraphNode &SrcNode, const CallGraph &CG,
                             SymbolTableCollection &SymTable,
                             SmallVectorImpl<ResolvedCall> &Calls) {
  SrcNode.getCallableRegion()->walk([&](Operation *Op) {
    if (auto Call = dyn_cast<CallOpInterface>(Op)) {
      CallInterfaceCallable Callable = Call.getCallableForCallee();
      if (SymbolRefAttr SymRef = dyn_cast<SymbolRefAttr>(Callable)) {
        if (!SymRef.isa<FlatSymbolRefAttr>())
          return WalkResult::advance();
      }

      CallGraphNode *TgtNode = CG.resolveCallable(Call, SymTable);
      if (!TgtNode->isExternal())
        Calls.emplace_back(Call, &SrcNode, TgtNode);
    }
    return WalkResult::advance();
  });
}

//===----------------------------------------------------------------------===//
// AlwaysInliner
//===----------------------------------------------------------------------===//

/// Returns true if the given call should be inlined.
bool AlwaysInliner::shouldInline(ResolvedCall &ResolvedCall) const {
  // Don't allow inlining if the target is an ancestor of the call. This
  // prevents inlining recursively.
  if (ResolvedCall.TgtNode->getCallableRegion()->isAncestor(
          ResolvedCall.Call->getParentRegion()))
    return false;

  FunctionOpInterface Callee =
      ResolvedCall::getCalledFunction(ResolvedCall.Call);

  NamedAttrList FnAttrs(Callee->getAttrDictionary());

  Optional<NamedAttribute> PassThroughAttr = FnAttrs.getNamed("passthrough");
  if (!PassThroughAttr)
    return false;

  return llvm::any_of(PassThroughAttr->getValue().cast<ArrayAttr>(),
                      [](Attribute Attr) {
                        return Attr.isa<StringAttr>() &&
                               (Attr.cast<StringAttr>() == "alwaysinline");
                      });
}

//===----------------------------------------------------------------------===//
// AlwaysInlinerPass
//===----------------------------------------------------------------------===//

void AlwaysInlinerPass::runOnOperation() {
  CallGraph &CG = getAnalysis<CallGraph>();
  auto *Ctx = &getContext();

  // The inliner should only be run on operations that define a symbol table,
  // as the call graph will need to resolve references.
  Operation *Op = getOperation();
  if (!Op->hasTrait<OpTrait::SymbolTable>()) {
    Op->emitOpError() << " was scheduled to run under the inliner, but does "
                         "not define a symbol table";
    return signalPassFailure();
  }

  SymbolTableCollection SymTable;
  AlwaysInliner Inliner(Ctx, CG, SymTable);
  LogicalResult Res = runTransformOnSCCs(CG, [&](CallGraphSCC &SCC) {
    return Inliner::inlineSCC(Inliner, SCC, Ctx);
  });

  if (failed(Res))
    return signalPassFailure();
}

std::unique_ptr<Pass> sycl::createAlwaysInlinePass() {
  return std::make_unique<AlwaysInlinerPass>();
}
