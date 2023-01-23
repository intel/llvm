//===- Inliner.cpp - Inline calls -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"
#include "mlir/Dialect/SYCL/Transforms/Passes.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace sycl {
#define GEN_PASS_DEF_INLINEPASS
#include "mlir/Dialect/SYCL/Transforms/Passes.h.inc"
} // namespace sycl
} // namespace mlir

#define DEBUG_TYPE "inlining"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Return the function corresponding to \p CGN.
static FunctionOpInterface getFunction(const CallGraphNode &CGN) {
  if (CGN.isExternal())
    return nullptr;
  return CGN.getCallableRegion()->getParentOp();
}

/// Return the function called by \p Call.
static FunctionOpInterface getCalledFunction(const CallOpInterface &Call) {
  if (SymbolRefAttr SymAttr = const_cast<CallOpInterface &>(Call)
                                  .getCallableForCallee()
                                  .dyn_cast<SymbolRefAttr>())
    return SymbolTable::lookupNearestSymbolFrom(Call, SymAttr);
  return nullptr;
}

namespace {

/// This struct represents a resolved call to a given call graph node. Given
/// that the call does not actually contain a direct reference to the
/// Region(CallGraphNode) that it is dispatching to, we need to resolve them
/// explicitly.
class ResolvedCall {
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const ResolvedCall &);

public:
  ResolvedCall(CallOpInterface Call, CallGraphNode *SrcNode,
               CallGraphNode *TgtNode)
      : Call(Call), SrcNode(SrcNode), TgtNode(TgtNode) {
    assert(SrcNode && "Expecting valid source node");
    assert(TgtNode && "Expecting valid target node");
  }

  CallOpInterface Call;
  CallGraphNode *SrcNode, *TgtNode;
};

[[maybe_unused]] inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                                      const ResolvedCall &RC) {
  if (!RC.SrcNode->isExternal())
    OS << "SrcNode: " << getFunction(*RC.SrcNode) << "\n";
  if (!RC.TgtNode->isExternal())
    OS << "TgtNode: " << getFunction(*RC.TgtNode) << "\n";
  OS << "Call: " << getCalledFunction(RC.Call).getName() << "\n";
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
  std::vector<CallGraphNode *>::const_iterator begin() const {
    return Nodes.begin();
  }
  std::vector<CallGraphNode *>::const_iterator end() const {
    return Nodes.end();
  }

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

[[maybe_unused]] inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                                      const CallGraphSCC &SCC) {
  OS << "{ ";
  llvm::interleaveComma(SCC.Nodes, OS, [&](const CallGraphNode *CGN) {
    if (!CGN->isExternal())
      OS << getFunction(*CGN).getName();
  });
  OS << " }";
  return OS;
}

/// Tracks the uses of callgraph nodes that can be dropped when no longer
/// referenced.
class CGUseList {
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, const CGUseList &);

public:
  /// This struct tracks the uses of callgraph nodes within a specific
  /// operation.
  struct CGUser {
    /// Any nodes referenced in the top-level attribute list of this user. We
    /// use a set here because the number of references does not matter.
    SmallPtrSet<CallGraphNode *, 16> TopLevelUses;

    /// Uses of nodes referenced by nested operations.
    DenseMap<CallGraphNode *, int> InnerUses;
  };

  CGUseList(Operation *Op, CallGraph &CG, SymbolTableCollection &SymbolTable);

  /// Drop uses of nodes referred to by the given \p Call operation that resides
  /// within \p CGN.
  void dropCallUses(CallGraphNode *CGN, Operation *Call, CallGraph &CG);

  /// Remove the given node from the use list.
  void eraseNode(CallGraphNode *CGN);

  /// Returns true if the given callgraph node has no uses and can be pruned.
  bool isDead(CallGraphNode *CGN) const;

  /// Returns true if the given callgraph node has a single use and can be
  /// discarded.
  bool hasOneUseAndDiscardable(CallGraphNode *CGN) const;

  /// Recompute the uses held by the given callgraph node.
  void recomputeUses(CallGraphNode *CGN, CallGraph &CG);

  /// Merge the uses of 'LHS' with the uses of the 'RHS' after inlining a copy
  /// of 'LHS' into 'RHS'.
  void mergeUsesAfterInlining(CallGraphNode *LHS, CallGraphNode *RHS);

private:
  /// Decrement the uses of discardable nodes referenced by the given user.
  void decrementDiscardableUses(CGUser &Uses);

  /// Walk all of the used symbol callgraph nodes referenced with the given op.
  void walkReferencedSymbolNodes(
      Operation *Op, CallGraph &CG, SymbolTableCollection &SymbolTable,
      DenseMap<Attribute, CallGraphNode *> &ResolvedRefs,
      function_ref<void(CallGraphNode *, Operation *)> Callback);

  /// A mapping between a discardable callgraph node (that is a symbol) and the
  /// number of uses for this node.
  DenseMap<CallGraphNode *, int> DiscardableSymNodeUses;

  /// A mapping between a callgraph node and the symbol callgraph nodes that it
  /// uses.
  DenseMap<CallGraphNode *, CGUser> NodeUses;

  /// A symbol table to use when resolving call lookups.
  SymbolTableCollection &SymbolTable;
};

[[maybe_unused]] inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &OS, const CGUseList &UseList) {
  for (auto &Use : UseList.NodeUses) {
    CallGraphNode *CGN = Use.first;
    CGUseList::CGUser Uses = Use.second;

    OS << "Func: " << getFunction(*CGN).getName() << "\n";

    OS.indent(2) << "TopLevelUses:\n";
    for (CallGraphNode *CGN : Uses.TopLevelUses)
      OS.indent(4) << "Op: " << getFunction(*CGN).getName() << "\n";

    OS.indent(2) << "InnerUses:\n";
    for (auto &It : Uses.InnerUses) {
      CallGraphNode *CGN = It.first;
      OS.indent(4) << "Op: " << getFunction(*CGN).getName() << "\n";
      OS.indent(4) << "It.second: " << It.second << "\n";
    }
  }

  return OS;
}

/// Inlining heuristics to use.
class InlineHeuristic {
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const InlineHeuristic &);

public:
  /// Inlining mode (alwaysinline, simple, aggressive, ludicrous).
  const sycl::InlineMode InlineMode;

  /// Maximum size (defined as number of operations) a callee may have to be
  /// considered for inlining.
  enum MaxCalleeSize {
    Large = 41,
    Medium = 19,
    Small = 7,
  };

  InlineHeuristic(sycl::InlineMode InlineMode) : InlineMode(InlineMode) {}

  /// Returns true if the given call should be inlined and false otherwise.
  bool shouldInline(ResolvedCall &ResolvedCall, const CGUseList &Uses) const;

private:
  /// Returns true if the target is an ancestor of the call and false otherwise.
  static bool isRecursiveCall(const ResolvedCall &ResolvedCall);

  /// Returns the number of operations in the target of the supplied call.
  static int64_t computeCalleeSize(const ResolvedCall &ResolvedCall);
};

[[maybe_unused]] inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &OS, const InlineHeuristic &Heuristic) {
  OS << "{ InlineMode = ";
  switch (Heuristic.InlineMode) {
  case sycl::InlineMode::Ludicrous:
    OS << "Ludicrous";
    break;
  case sycl::InlineMode::Aggressive:
    OS << "Aggressive";
    break;
  case sycl::InlineMode::Simple:
    OS << "Simple";
    break;
  case sycl::InlineMode::AlwaysInline:
    OS << "AlwaysInline";
    break;
  }
  OS << " }";

  return OS;
}

// Inliner functionality.
class Inliner : public InlinerInterface {
public:
  Inliner(MLIRContext *Ctx, CallGraph &CG, SymbolTableCollection &SymTable,
          const InlineHeuristic &Heuristic)
      : InlinerInterface(Ctx), CG(CG), SymbolTable(SymTable),
        Heuristic(Heuristic) {}

  ResolvedCall &getCall(unsigned Index) {
    assert(Index < Calls.size() && "Out of bound index");
    return Calls[Index];
  }
  SmallVectorImpl<ResolvedCall> &getCalls() { return Calls; }
  void clearCalls() { Calls.clear(); }

  CallGraph &getCG() const { return CG; }

  SymbolTableCollection &getSymbolTable() const { return SymbolTable; }

  /// Run a given transformation over the SCCs of the callgraph in a bottom up
  /// traversal.
  static LogicalResult runTransformOnSCCs(
      const CallGraph &CG,
      function_ref<LogicalResult(CallGraphSCC &)> SCCTransformer);

  /// Attempt to inline calls within the given scc, and run simplifications,
  /// until a fixed point is reached. This allows for the inlining of newly
  /// devirtualized calls. Returns failure if there was a fatal error during
  /// inlining.
  static LogicalResult inlineSCC(Inliner &Inliner, CGUseList &UseList,
                                 CallGraphSCC &SCC, unsigned MaxIterationCount,
                                 Pass::Statistic &NumInlinedCalls);

  /// This method properly disposes of callables that became dead during
  /// inlining. This should not be called while iterating over the SCCs.
  void eraseDeadCallables() const {
    for (CallGraphNode *CGN : DeadNodes) {
      getFunction(*CGN)->erase();
      LLVM_DEBUG(llvm::dbgs() << "Removed dead callable: "
                              << getFunction(*CGN).getName() << "\n");
    }
  }

protected:
  /// Attempt to inline calls within the given SCC. Returns true if at least a
  /// call was inlined in the SCC and false otherwise.
  static bool inlineCallsInSCC(Inliner &Inliner, CGUseList &UseList,
                               CallGraphSCC &SCC,
                               Pass::Statistic &NumInlinedCalls);

  /// Collect all of the callable operations within the given \p SrcNode.
  void collectCallOps(CallGraphNode &SrcNode, CallGraph &CG,
                      SymbolTableCollection &SymTable,
                      SmallVectorImpl<ResolvedCall> &Calls) const;

  /// Mark the given callgraph node for deletion.
  void markForDeletion(CallGraphNode *CGN) { DeadNodes.insert(CGN); }

private:
  /// The set of callables known to be dead.
  SmallPtrSet<CallGraphNode *, 8> DeadNodes;

  /// The current set of call instructions to consider for inlining.
  SmallVector<ResolvedCall, 8> Calls;

  /// The call graph being operated on.
  CallGraph &CG;

  /// A symbol table to use when resolving call lookups.
  SymbolTableCollection &SymbolTable;

  /// The inline heuristic controlling when to inline a call edge.
  const InlineHeuristic &Heuristic;
};

class InlinePass : public sycl::impl::InlinePassBase<InlinePass> {
public:
  InlinePass(const sycl::InlinePassOptions &Options)
      : sycl::impl::InlinePassBase<InlinePass>(Options) {}
  InlinePass(const InlinePass &) = default;

  void runOnOperation() final;

private:
  /// Inline function calls in the given callgraph \p CG (on each SCC in
  /// bottom up order).
  LogicalResult runOnCG(Inliner &Inliner, CGUseList &UseList, CallGraph &CG,
                        Pass::Statistic &NumInlinedCalls);

  /// Ensures that the inliner is run on operations that define a symbol
  /// table.
  bool checkForSymbolTable(Operation &Op);

  /// Return the number of times an SCC should be revisited while inlining.
  unsigned getMaxIterationCount() const;

private:
  const std::string PassName, FlagName, PassDescription;
};

} // namespace

//===----------------------------------------------------------------------===//
// CGUseList
//===----------------------------------------------------------------------===//

CGUseList::CGUseList(Operation *Op, CallGraph &CG,
                     SymbolTableCollection &SymbolTable)
    : SymbolTable(SymbolTable) {
  /// A set of callgraph nodes that are always known to be live during inlining.
  DenseMap<Attribute, CallGraphNode *> AlwaysLiveNodes;

  // Walk each of the symbol tables looking for discardable callgraph nodes.
  auto WalkFn = [&](Operation *SymbolTableOp, bool AllUsesVisible) {
    for (Operation &Op : SymbolTableOp->getRegion(0).getOps()) {
      // If this is a callgraph operation, check to see if it is discardable.
      if (auto Callable = dyn_cast<CallableOpInterface>(&Op)) {
        if (auto *Node = CG.lookupNode(Callable.getCallableRegion())) {
          SymbolOpInterface Symbol = dyn_cast<SymbolOpInterface>(&Op);
          if (Symbol && (AllUsesVisible || Symbol.isPrivate()) &&
              Symbol.canDiscardOnUseEmpty())
            DiscardableSymNodeUses.try_emplace(Node, 0);
          continue;
        }
      }
      // Otherwise, check for any referenced nodes. These will be always-live.
      walkReferencedSymbolNodes(&Op, CG, SymbolTable, AlwaysLiveNodes,
                                [](CallGraphNode *, Operation *) {});
    }
  };

  SymbolTable::walkSymbolTables(Op, /*allSymUsesVisible=*/!Op->getBlock(),
                                WalkFn);

  // Drop the use information for any discardable nodes that are always live.
  for (auto &It : AlwaysLiveNodes)
    DiscardableSymNodeUses.erase(It.second);

  // Compute the uses for each of the callable nodes in the graph.
  for (CallGraphNode *CGN : CG)
    recomputeUses(CGN, CG);
}

void CGUseList::walkReferencedSymbolNodes(
    Operation *Op, CallGraph &CG, SymbolTableCollection &SymbolTable,
    DenseMap<Attribute, CallGraphNode *> &ResolvedRefs,
    function_ref<void(CallGraphNode *, Operation *)> Callback) {
  auto SymbolUses = SymbolTable::getSymbolUses(Op);
  assert(SymbolUses && "expected uses to be valid");

  Operation *SymbolTableOp = Op->getParentOp();
  for (const SymbolTable::SymbolUse &Use : *SymbolUses) {
    auto RefIt = ResolvedRefs.insert({Use.getSymbolRef(), nullptr});
    CallGraphNode *&CGN = RefIt.first->second;

    // If this is the first instance of this reference, try to resolve a
    // callgraph node for it.
    if (RefIt.second) {
      auto *SymbolOp = SymbolTable.lookupNearestSymbolFrom(SymbolTableOp,
                                                           Use.getSymbolRef());
      auto CallableOp = dyn_cast_or_null<CallableOpInterface>(SymbolOp);
      if (!CallableOp)
        continue;
      CGN = CG.lookupNode(CallableOp.getCallableRegion());
    }
    if (CGN)
      Callback(CGN, Use.getUser());
  }
}

bool CGUseList::isDead(CallGraphNode *CGN) const {
  // If the parent operation isn't a symbol, simply check normal SSA deadness.
  Operation *Op = CGN->getCallableRegion()->getParentOp();
  if (!isa<SymbolOpInterface>(Op))
    return isMemoryEffectFree(Op) && Op->use_empty();

  // Otherwise, check the number of symbol uses.
  auto SymbolIt = DiscardableSymNodeUses.find(CGN);
  return SymbolIt != DiscardableSymNodeUses.end() && SymbolIt->second == 0;
}

bool CGUseList::hasOneUseAndDiscardable(CallGraphNode *CGN) const {
  // If this isn't a symbol node, check for side-effects and SSA use count.
  Operation *Op = CGN->getCallableRegion()->getParentOp();
  if (!isa<SymbolOpInterface>(Op))
    return isMemoryEffectFree(Op) && Op->hasOneUse();

  // Otherwise, check the number of symbol uses.
  auto SymbolIt = DiscardableSymNodeUses.find(CGN);
  return SymbolIt != DiscardableSymNodeUses.end() && SymbolIt->second == 1;
}

void CGUseList::dropCallUses(CallGraphNode *UserNode, Operation *CallOp,
                             CallGraph &CG) {
  auto &UserRefs = NodeUses[UserNode].InnerUses;
  auto WalkFn = [&](CallGraphNode *Node, Operation *User) {
    auto ParentIt = UserRefs.find(Node);
    if (ParentIt == UserRefs.end())
      return;
    --ParentIt->second;
    --DiscardableSymNodeUses[Node];
  };
  DenseMap<Attribute, CallGraphNode *> ResolvedRefs;
  walkReferencedSymbolNodes(CallOp, CG, SymbolTable, ResolvedRefs, WalkFn);
}

void CGUseList::eraseNode(CallGraphNode *CGN) {
  // Drop all child nodes.
  for (const CallGraphNode::Edge &Edge : *CGN)
    if (Edge.isChild())
      eraseNode(Edge.getTarget());

  // Drop the uses held by this node and erase it.
  auto UseIt = NodeUses.find(CGN);
  assert(UseIt != NodeUses.end() && "expected node to be valid");
  decrementDiscardableUses(UseIt->getSecond());
  NodeUses.erase(UseIt);
  DiscardableSymNodeUses.erase(CGN);
}

void CGUseList::recomputeUses(CallGraphNode *CGN, CallGraph &CG) {
  Operation *ParentOp = CGN->getCallableRegion()->getParentOp();
  CGUser &Uses = NodeUses[CGN];
  decrementDiscardableUses(Uses);

  // Collect the new discardable uses within this node.
  Uses = CGUser();
  DenseMap<Attribute, CallGraphNode *> ResolvedRefs;
  auto WalkFn = [&](CallGraphNode *RefNode, Operation *User) {
    auto DiscardSymIt = DiscardableSymNodeUses.find(RefNode);
    if (DiscardSymIt == DiscardableSymNodeUses.end())
      return;

    if (User != ParentOp)
      ++Uses.InnerUses[RefNode];
    else if (!Uses.TopLevelUses.insert(RefNode).second)
      return;
    ++DiscardSymIt->second;
  };
  walkReferencedSymbolNodes(ParentOp, CG, SymbolTable, ResolvedRefs, WalkFn);
}

void CGUseList::mergeUsesAfterInlining(CallGraphNode *LHS, CallGraphNode *RHS) {
  auto &LHSUses = NodeUses[LHS], &RHSUses = NodeUses[RHS];
  for (auto &UseIt : LHSUses.InnerUses) {
    RHSUses.InnerUses[UseIt.first] += UseIt.second;
    DiscardableSymNodeUses[UseIt.first] += UseIt.second;
  }
}

void CGUseList::decrementDiscardableUses(CGUser &Uses) {
  for (CallGraphNode *CGN : Uses.TopLevelUses)
    --DiscardableSymNodeUses[CGN];
  for (auto &It : Uses.InnerUses)
    DiscardableSymNodeUses[It.first] -= It.second;

  for (CallGraphNode *CGN : Uses.TopLevelUses) {
    Operation *ParentOp = CGN->getCallableRegion()->getParentOp();
    LLVM_DEBUG(llvm::dbgs() << ParentOp->getName() << " has "
                            << DiscardableSymNodeUses[CGN] << " uses\n");
  }
}

//===----------------------------------------------------------------------===//
// InlineHeuristic
//===----------------------------------------------------------------------===//

bool InlineHeuristic::shouldInline(ResolvedCall &ResolvedCall,
                                   const CGUseList &Uses) const {
  if (isRecursiveCall(ResolvedCall)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Call is recursive, not considered for inlining\n");
    return false;
  }

  FunctionOpInterface Callee = getCalledFunction(ResolvedCall.Call);
  NamedAttrList FnAttrs(Callee->getAttrDictionary());
  Optional<NamedAttribute> PassThroughAttr = FnAttrs.getNamed("passthrough");

  // Decide whether to inline a callee based on simple heuristics.
  bool ShouldInline = false;
  switch (InlineMode) {
  case sycl::InlineMode::Ludicrous:
    [[fallthrough]];
  case sycl::InlineMode::Aggressive:
    [[fallthrough]];
  case sycl::InlineMode::Simple:
    // Inline a function if it has an attribute suggesting that inlining is
    // desirable.
    if (PassThroughAttr)
      ShouldInline = llvm::any_of(
          PassThroughAttr->getValue().cast<ArrayAttr>(), [](Attribute Attr) {
            return Attr.isa<StringAttr>() &&
                   Attr.cast<StringAttr>() == "inlinehint";
          });

    // Inline a function if inlining makes it dead.
    ShouldInline |= Uses.hasOneUseAndDiscardable(ResolvedCall.TgtNode);
    [[fallthrough]];
  case sycl::InlineMode::AlwaysInline:
    // Inline a function iff it has the 'alwaysinline' attribute.
    if (PassThroughAttr)
      ShouldInline |= llvm::any_of(
          PassThroughAttr->getValue().cast<ArrayAttr>(), [](Attribute Attr) {
            return Attr.isa<StringAttr>() &&
                   Attr.cast<StringAttr>() == "alwaysinline";
          });
    break;
  }

  if (ShouldInline)
    return true;

  // Decide whether to inline a callee based on its size.
  unsigned MaxSize = 0;
  switch (InlineMode) {
  case sycl::InlineMode::Ludicrous:
    MaxSize = MaxCalleeSize::Large;
    break;
  case sycl::InlineMode::Aggressive:
    MaxSize = MaxCalleeSize::Medium;
    break;
  case sycl::InlineMode::Simple:
    MaxSize = MaxCalleeSize::Small;
    break;
  case sycl::InlineMode::AlwaysInline:
    return false;
  }

  unsigned Size = computeCalleeSize(ResolvedCall);
  LLVM_DEBUG({
    if (Size > MaxSize)
      llvm::dbgs() << "Callee has size = " << Size
                   << ", which is greater than the max (" << MaxSize << ")";
  });

  return (Size <= MaxSize);
}

bool InlineHeuristic::isRecursiveCall(const ResolvedCall &ResolvedCall) {
  return ResolvedCall.TgtNode->getCallableRegion()->isAncestor(
      ResolvedCall.Call->getParentRegion());
}

int64_t InlineHeuristic::computeCalleeSize(const ResolvedCall &ResolvedCall) {
  int64_t Count = 0;
  ResolvedCall.TgtNode->getCallableRegion()->walk(
      [&](Operation *) { ++Count; });

  return Count;
}

//===----------------------------------------------------------------------===//
// Inliner
//===----------------------------------------------------------------------===//

LogicalResult Inliner::runTransformOnSCCs(
    const CallGraph &CG,
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

LogicalResult Inliner::inlineSCC(Inliner &Inliner, CGUseList &UseList,
                                 CallGraphSCC &SCC, unsigned MaxIterationCount,
                                 Pass::Statistic &NumInlinedCalls) {
  unsigned IterationCount = 0;
  bool DidSomething = false;
  do {
    DidSomething = inlineCallsInSCC(Inliner, UseList, SCC, NumInlinedCalls);
  } while (DidSomething && ++IterationCount < MaxIterationCount);
  return success();
}

bool Inliner::inlineCallsInSCC(Inliner &Inliner, CGUseList &UseList,
                               CallGraphSCC &SCC,
                               Pass::Statistic &NumInlinedCalls) {
  llvm::SmallPtrSet<CallGraphNode *, 1> DeadNodes;

  for (CallGraphNode *SrcNode : SCC) {
    if (SrcNode->isExternal())
      continue;

    if (UseList.isDead(SrcNode)) {
      DeadNodes.insert(SrcNode);
      continue;
    }

    Inliner.collectCallOps(*SrcNode, Inliner.getCG(), Inliner.getSymbolTable(),
                           Inliner.getCalls());
  }

  if (Inliner.getCalls().empty())
    return false;

  LLVM_DEBUG({
    llvm::dbgs() << "* Inliner: SCC: " << SCC << "\n";
    llvm::dbgs() << "* Inliner: Initial calls in SCC are: {\n";
    for (unsigned i = 0, e = Inliner.getCalls().size(); i < e; ++i)
      llvm::dbgs() << "  " << i << ". " << Inliner.getCall(i).Call << ",\n";
    llvm::dbgs() << "}\n";
  });

  // Try to inline each of the call operations. Don't cache the end iterator
  // here as more calls may be added during inlining.
  bool DidSomething = false;
  for (unsigned I = 0; I < Inliner.getCalls().size(); ++I) {
    ResolvedCall ResolvedCall = Inliner.getCall(I);
    bool DoInline = Inliner.Heuristic.shouldInline(ResolvedCall, UseList);
    if (!DoInline)
      continue;

    LLVM_DEBUG(llvm::dbgs() << "* Inlining call: " << I << ". "
                            << ResolvedCall.Call << "\n");

    Region *TgtRegion = ResolvedCall.TgtNode->getCallableRegion();
    LogicalResult InlineRes =
        inlineCall(Inliner, ResolvedCall.Call,
                   cast<CallableOpInterface>(TgtRegion->getParentOp()),
                   TgtRegion, /*shouldCloneInlinedRegion=*/true);

    if (failed(InlineRes)) {
      LLVM_DEBUG(llvm::dbgs() << "** Failed to inline\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << "** Inline succeeded\n");

    DidSomething = true;
    ++NumInlinedCalls;

    // Merge the new uses into the source node.
    UseList.dropCallUses(ResolvedCall.SrcNode, ResolvedCall.Call.getOperation(),
                         Inliner.getCG());
    UseList.mergeUsesAfterInlining(ResolvedCall.TgtNode, ResolvedCall.SrcNode);

    // Erase the call.
    ResolvedCall.Call.erase();

    // If the last call to the target node was inlined, mark the callee for
    // deletion.
    if (UseList.isDead(ResolvedCall.TgtNode)) {
      UseList.eraseNode(ResolvedCall.TgtNode);
      DeadNodes.insert(ResolvedCall.TgtNode);
    }
  }

  for (CallGraphNode *CGN : DeadNodes) {
    LLVM_DEBUG(llvm::dbgs()
               << "** Marking " << getFunction(*CGN).getName() << " dead\n");
    SCC.remove(CGN);
    Inliner.markForDeletion(CGN);
  }

  Inliner.clearCalls();

  return DidSomething;
}

void Inliner::collectCallOps(CallGraphNode &SrcNode, CallGraph &CG,
                             SymbolTableCollection &SymTable,
                             SmallVectorImpl<ResolvedCall> &Calls) const {
  auto Collect = [this](const CallGraphNode *CGN, const Operation *Call) {
    if (CGN->isExternal())
      return false;

    // Select which call operations to collect based on heuristics.
    switch (Heuristic.InlineMode) {
    case sycl::InlineMode::Ludicrous:
      return true;
    case sycl::InlineMode::Aggressive:
      return isa<sycl::SYCLCallOp, sycl::SYCLConstructorOp,
                 sycl::SYCLMethodOpInterface, func::CallOp>(Call);
    case sycl::InlineMode::Simple:
      return isa<sycl::SYCLCallOp, sycl::SYCLConstructorOp, func::CallOp>(Call);
    case sycl::InlineMode::AlwaysInline:
      return isa<sycl::SYCLCallOp, func::CallOp>(Call);
    }
  };

  SrcNode.getCallableRegion()->walk([&](Operation *Op) {
    if (auto Call = dyn_cast<CallOpInterface>(Op)) {
      CallInterfaceCallable Callable = Call.getCallableForCallee();
      if (SymbolRefAttr SymRef = dyn_cast<SymbolRefAttr>(Callable)) {
        if (!SymRef.isa<FlatSymbolRefAttr>())
          return WalkResult::advance();
      }

      CallGraphNode *TgtNode = CG.resolveCallable(Call, SymTable);
      if (Collect(TgtNode, Op))
        Calls.emplace_back(Call, &SrcNode, TgtNode);
    }
    return WalkResult::advance();
  });
}

//===----------------------------------------------------------------------===//
// InlinePass
//===----------------------------------------------------------------------===//

void InlinePass::runOnOperation() {
  if (!checkForSymbolTable(*getOperation()))
    return signalPassFailure();

  MLIRContext *Ctx = &getContext();
  CallGraph &CG = getAnalysis<CallGraph>();
  SymbolTableCollection SymTable;
  CGUseList UseList(getOperation(), CG, SymTable);
  InlineHeuristic Heuristic(InlineMode);
  Inliner Inliner(Ctx, CG, SymTable, Heuristic);

  LLVM_DEBUG(llvm::dbgs() << "Inline Heuristic: " << Heuristic << "\n");

  if (failed(runOnCG(Inliner, UseList, CG, NumInlinedCalls)))
    return signalPassFailure();
}

unsigned InlinePass::getMaxIterationCount() const {
  // Use the supplied option value if present.
  if (MaxIterationCount.getNumOccurrences())
    return MaxIterationCount.getValue();

  // Set the max iteration count based on the inline mode.
  switch (InlineMode) {
  case sycl::InlineMode::Ludicrous:
    return 7;
  case sycl::InlineMode::Aggressive:
    return 5;
  case sycl::InlineMode::Simple:
    return 3;
  case sycl::InlineMode::AlwaysInline:
    return 2;
  }
}

LogicalResult InlinePass::runOnCG(Inliner &Inliner, CGUseList &UseList,
                                  CallGraph &CG,
                                  Pass::Statistic &NumInlinedCalls) {
  LogicalResult Res = Inliner::runTransformOnSCCs(CG, [&](CallGraphSCC &SCC) {
    return Inliner::inlineSCC(Inliner, UseList, SCC, getMaxIterationCount(),
                              NumInlinedCalls);
  });

  // After inlining, make sure to erase any callables proven to be dead.
  if (succeeded(Res) && RemoveDeadCallees)
    Inliner.eraseDeadCallables();

  return Res;
}

bool InlinePass::checkForSymbolTable(Operation &Op) {
  // The inliner should only be run on operations that define a symbol table,
  // as the callgraph will need to resolve references.
  if (!Op.hasTrait<OpTrait::SymbolTable>()) {
    Op.emitOpError() << " was scheduled to run under the inliner, but does "
                        "not define a symbol table";
    return false;
  }
  return true;
}

std::unique_ptr<Pass> sycl::createInlinePass() {
  const sycl::InlinePassOptions &Options = {InlineMode::Simple,
                                            /* RemoveDeadCallees */ false};
  return std::make_unique<InlinePass>(Options);
}

std::unique_ptr<Pass> sycl::createInlinePass(enum InlineMode InlineMode,
                                             bool RemoveDeadCallees) {
  const sycl::InlinePassOptions &Options = {InlineMode, RemoveDeadCallees};
  return std::make_unique<InlinePass>(Options);
}
