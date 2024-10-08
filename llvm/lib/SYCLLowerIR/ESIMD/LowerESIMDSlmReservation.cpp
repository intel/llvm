//===----- LowerESIMDSlmReservation.cpp - lower __esimd_slm_* intrinsics --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This implements the 'size_t lowerSLMAllocationCalls(Module& M)' invoked as a
// part of ESIMD lowering from LowerESIMD.cpp. This function lowers the
// following intrinsic calls:
// - __esimd_slm_init
//   This is allowed only in kernels, and tells the compiler that this kernel
//   will at least use the amount of SLM defined by the call's argument.
// - __esimd_slm_alloc/__esimd_slm_free
//   These result from defining a 'esimd::experimental::slm_allocator' object.
//   (See high-level interface in the user API documentation.)
//   Its constructor generates the 'alloc' call, destructor - the 'free' call.
//   Thus they always come in pairs forming scopes. __esimd_slm_free is assumed
//   to always have a result of __esimd_slm_alloc as its only argument, which
//   allows the SLM allocation analysis to find corresponding pairs of calls
//   delimiting a scope.
//
// This pass first builds a "scoped" call graph. Its nodes can be of two types:
// - function nodes, representing a specific function
// - scope nodes, representing an alloc/free pair
// Edges represent "called by/from" relation ("reverse callgraph"), for example
// if there is an 'func_node_a -> scope_node_b' edge, this reflects the
// following IR pattern:
//   %slm_offset = call @__esimd_slm_alloc(i32 N)
//   ...
//   call @foo(...)
//   ...
//   call @__esimd_slm_free(%slm_offset)
// where 'func_node_a' corresponds to function 'foo', and 'scope_node_b'
// corresponds to the scope uniquely identified the '%slm_offset' value.
// The hierarchy of scope nodes within a function is built based on depth-first
// preorder traversal of basic blocks, when a basic block is processed before
// any of its successors. This order provides correct nesting of scopes and
// ignores back-branches.
//
// Then the pass uses the call graph to
// 1) For each <scope, kernel> pair, determine a maximum possible amount of SLM
// allocated by __esimd_slm_alloc calls along any path from the scope to the
// kernel (its function node). Then the '%slm_offset' value is replaced by a
// constant equal to that amount, and both scope marker calls are erased from
// the IR.
// 2) For each kernel, find a path to a scope with a maximum possible amount of
// SLM allocated by __esimd_slm_alloc calls along that path. This amount is
// then recorded in a special kernel's metadata entry to be used later by the BE
// to generate proper "patch token" reflecting the SLM size.
//
// TODO __esimd_slm_alloc currently requires costant arguments only, which means
// specialization constants cannot be passed as arguments (unlike
// '__esimd_slm_init').
// To support specialization constants, this analysis/transformation should
// basically be copied into VC BE, where spec constants appear as normal
// constants bacause the spec constant SPIRV transformation which turns them
// into normal constants is done as a first JIT step.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/ESIMD/ESIMDUtils.h"
#include "llvm/SYCLLowerIR/ESIMD/LowerESIMD.h"
#include "llvm/SYCLLowerIR/SYCLUtils.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#include <set>
#include <unordered_map>

#define DEBUG_TYPE "LowerESIMDSlmAllocPass"

namespace llvm {

namespace {

#ifndef NDEBUG
constexpr int DebugLevel = 0;
#endif

bool isGenXSLMInit(const Function &F) {
  constexpr char SLM_INIT_PREFIX[] = "llvm.genx.slm.init";
  return F.getName().starts_with(SLM_INIT_PREFIX);
}

bool isSlmInitCall(const CallInst *CI) {
  if (!CI)
    return false;
  Function *F = CI->getCalledFunction();
  if (!F)
    return false;
  assert(!esimd::isSlmInit(*F) && "Should have been translated already");
  return isGenXSLMInit(*F);
}

bool isSlmAllocCall(const CallInst *CI) {
  if (!CI)
    return false;
  Function *F = CI->getCalledFunction();
  return F && esimd::isSlmAlloc(*F);
}

// Checks if given call is a call to '__esimd_slm_free' function, and if yes,
// finds the corresponding '__esimd_slm_alloc' call and returns it.
CallInst *isSlmFreeCall(const CallInst *CI) {
  if (!CI)
    return nullptr;

  Function *F = CI->getCalledFunction();
  if (!F || !esimd::isSlmFree(*F))
    return nullptr;
  Value *Arg = CI->getArgOperand(0);

  if (auto *CI1 = dyn_cast<CallInst>(Arg)) {
    assert(isSlmAllocCall(CI1));
    return CI1;
  }
  auto *LI = cast<LoadInst>(sycl::utils::stripCastsAndZeroGEPs(Arg));
  SmallPtrSet<Value *, 1> Vals;
  sycl::utils::collectPossibleStoredVals(LI->getPointerOperand(), Vals);
  esimd::assert_and_diag(Vals.size() == 1,
                         "unexpected data flow in __esimd_slm_free, function ",
                         CI->getFunction()->getName());
  auto *SLMAlloc = cast<CallInst>(*Vals.begin());
  esimd::assert_and_diag(isSlmAllocCall(SLMAlloc),
                         "bad __esimd_slm_free argument, function ",
                         CI->getFunction()->getName());
  return SLMAlloc;
}

// Determines the amount of SLM allocated by given SLM reservation call, which
// can be either __esimd_slm_init or __esimd_slm_alloc.
// If the amount can't be determined (when it is not a compile-time constant),
// -1 is returned.
int getSLMUsage(const CallInst *SLMReserveCall) {
  assert(isSlmInitCall(SLMReserveCall) || isSlmAllocCall(SLMReserveCall));
  auto *ArgV = SLMReserveCall->getArgOperand(0);

  if (!isa<ConstantInt>(ArgV)) {
    esimd::assert_and_diag(
        isSlmInitCall(SLMReserveCall),
        "__esimd_slm_alloc with non-constant argument, function ",
        SLMReserveCall->getFunction()->getName());
    return -1;
  }
  int64_t Res = cast<llvm::ConstantInt>(ArgV)->getZExtValue();
  assert(Res < std::numeric_limits<int>::max());
  return static_cast<int>(Res);
}

bool isNonConstSLMInit(const CallInst *SLMReserveCall) {
  return getSLMUsage(SLMReserveCall) < 0;
}

} // namespace

// Represents the scoped call graph. No explicit root node(s). Instead, contains
// a set of kernels (Function object pointers), and a map which maps them to
// function nodes (which are the roots). Also contains the set of "scopes" which
// are used to determine SLM usage in a particular scope and kernels.
class ScopedCallGraph {
public:
  class Node;
  using NodeSPtr = std::shared_ptr<Node>;

  // Generic node.
  // The graph is asyclic, so shared_ptr should not lead to pointer cycles and
  // therefore memory leaks.
  class Node {
#ifndef NDEBUG
    static unsigned InstanceN;
#endif
    SmallVector<NodeSPtr, 2> Preds;

  public:
    enum class Kind { func, scope };

  protected:
    Kind K;
#ifndef NDEBUG
    unsigned ID;
#endif

    Node(Kind K)
        : K(K)
#ifndef NDEBUG
          ,
          ID(InstanceN++)
#endif
    {
    }

  public:
    Kind getKind() const { return K; }

    void addPred(const NodeSPtr &N) { Preds.push_back(N); }

    const SmallVector<NodeSPtr, 2> &preds() const { return Preds; }
#ifndef NDEBUG
    virtual void dump() const = 0;
    void dumpPreds() const {
      llvm::errs() << "Preds: { ";
      for (const auto &Pred : preds()) {
        llvm::errs() << Pred.get()->ID << " ";
      }
      llvm::errs() << "}\n";
    }
#endif
    virtual ~Node() {}
  };

  // A function node of the scoped callgraph.
  class FuncNode : public Node {
    friend class ScopedCallGraph;
    Function *F;

  public:
    FuncNode(Function *F) : Node(Kind::func), F(F) {}

    // To enable usage of llvm dynamic type casting/checking infra.
    static bool classof(const Node *N) { return N->getKind() == Kind::func; }

    Function *getFunction() const { return F; }

#ifndef NDEBUG
    virtual void dump() const override {
      llvm::errs() << "(ID=" << ID << ") FuncNode {\n"
                   << "  function: " << F->getName() << "\n"
                   << "  ";
      dumpPreds();
      llvm::errs() << "}\n";
    }
#endif
  };

  // A scope node of the callgraph. Defined by a pair of calls:
  // %off = call @__esimd_slm_alloc(i32 N)
  // ...
  // call @__esimd_slm_free(%off)
  class ScopeNode : public Node {
    CallInst *ScopeStart;
    CallInst *ScopeEnd;

  public:
    ScopeNode(CallInst *Start)
        : Node(Kind::scope), ScopeStart(Start), ScopeEnd(nullptr) {}

    void setEnd(CallInst *End) {
      assert(!ScopeEnd && "Scope end already set");
      ScopeEnd = End;
    }

    // To enable usage of llvm dynamic type casting/checking infra.
    static bool classof(const Node *N) { return N->getKind() == Kind::scope; }

    CallInst *getStart() const { return ScopeStart; }

    CallInst *getEnd() const { return ScopeEnd; }

#ifndef NDEBUG
    virtual void dump() const override {
      llvm::errs() << "(ID=" << ID << ") ScopeNode {\n";
      if (getStart()) {
        llvm::errs() << "  start: ";
        getStart()->dump();
        llvm::errs() << "  SLM Usage: " << getSLMUsage(getStart()) << "\n";
      }
      if (getEnd()) {
        llvm::errs() << "  end: ";
        getEnd()->dump();
      }
      llvm::errs() << "  ";
      dumpPreds();
      llvm::errs() << "}\n";
    }
#endif
  };

  using FuncNodeSPtr = std::shared_ptr<FuncNode>;
  using ScopeNodeSPtr = std::shared_ptr<ScopeNode>;

private:
  struct ScopeNodeSPtrLess {
    bool operator()(const ScopeNodeSPtr &P1, const ScopeNodeSPtr &P2) const {
      return P1.get() < P2.get();
    }
  };

  std::unordered_map<const Function *, FuncNodeSPtr> Func2Node;
  SmallPtrSet<const Function *, 4> Kernels;
  std::set<ScopeNodeSPtr, ScopeNodeSPtrLess> Scopes;

public:
  ScopedCallGraph(Module &M) {
    auto IsScopeStart = [](Instruction *I) -> CallInst * {
      auto *CI = dyn_cast<CallInst>(I);
      return CI && isSlmAllocCall(CI) ? CI : nullptr;
    };
    auto IsScopeEnd = [](const Instruction *I) -> CallInst * {
      auto *CI = dyn_cast<CallInst>(I);
      CallInst *ScopeStart = CI ? isSlmFreeCall(CI) : nullptr;
      return ScopeStart;
    };
    // Function containing a call via a pointer (used for diagnostics).
    const Function *IndirectCallMet = nullptr;
    // A number of calls to __esimd_slm_alloc (used for diagnostics).
    int SLMAllocCnt = 0;

    for (auto &F : M) {
      if (F.isDeclaration()) {
        continue;
      }
      if (esimd::isESIMDKernel(F)) {
        Kernels.insert(&F);
      }
      SmallVector<NodeSPtr, 32> CurScopePath;
      // Add a FuncNode for the current function - it will dominate all other
      // nodes created for scopes in this function - emplace a mapping to dummy
      // null pointer to avoid double search.
      auto E = Func2Node.emplace(std::make_pair(&F, FuncNodeSPtr{nullptr}));

      if (E.second) {
        // Insertion took place, which means there was no FuncNode for F -
        // create it and replace the dummy nullptr node.
        E.first->second = std::make_shared<FuncNode>(&F);
      }
      CurScopePath.push_back(E.first->second);
      SmallPtrSet<BasicBlock *, 32> Visited;
      SmallVector<BasicBlock *, 32> Wl;
      Wl.push_back(&F.getEntryBlock());
      ScopeNodeSPtr SlmInitCall = nullptr;
      bool ScopeMet = false; // to diagnose slm_init use after SLM allocator.
      bool NonConstSlmInitMet = false;

      // Do preorder traversal so that successors are visited after the parent.
      while (Wl.size() > 0) {
        BasicBlock *BB = Wl.pop_back_val();

        // If we have already visited this BB but it
        // made it into the worklist, that means this BB
        // has multiple predecessors. We already processed the first one
        // so just skip it.
        if (Visited.contains(BB))
          continue;

        Visited.insert(BB);

        for (Instruction &I : *BB) {
          if (CallInst *ScopeStartCI = IsScopeStart(&I)) {
            ScopeMet = true;
            SLMAllocCnt++;
            auto N = std::make_shared<ScopeNode>(ScopeStartCI);
            N->addPred(CurScopePath.back());
            Scopes.insert(N);
            CurScopePath.emplace_back(std::move(N));
            continue;
          }
          if (CallInst *ScopeStartCI = IsScopeEnd(&I)) {
            (void)ScopeStartCI;
            ScopeMet = true;
            // Scope end marker encountered - verify all enclosed scopes have
            // ended and truncate current scope path to the enclosing node.
            auto *CurScope = cast<ScopeNode>(CurScopePath.pop_back_val().get());
            assert(ScopeStartCI == CurScope->getStart());
            CurScope->setEnd(cast<CallInst>(&I));
            continue;
          }
          if (auto *CB = dyn_cast<CallBase>(&I)) {
            if (isSlmInitCall(dyn_cast<CallInst>(CB))) {
              auto *CI = dyn_cast<CallInst>(CB);
              esimd::assert_and_diag(!SlmInitCall,
                                     "multiple slm_init calls in function ",
                                     F.getName());
              // TODO: this diagnostics incorrectly fires on functor's
              // operator() marked as SYCL_ESIMD_KERNEL, because becomes neither
              // spir_kernel nor SYCL_EXERNAL function in IR. It rather becomes
              // a function called from spir_kernel.
              // esimd::assert_and_diag(
              //    esimd::isESIMDKernel(F) ||
              //        sycl::utils::isSYCLExternalFunction(&F),
              //    "slm_init call met in non-kernel non-external function ",
              //    F.getName());
              esimd::assert_and_diag(
                  !ScopeMet,
                  "slm_init must precede any SLMAllocator object in function ",
                  F.getName());
              NonConstSlmInitMet |= isNonConstSLMInit(CI);
              SlmInitCall = std::make_shared<ScopeNode>(CI);
              // slm_init is special scope - does not have explicit end, it is
              // rather implicit at function's end
              SlmInitCall->addPred(CurScopePath.back());
              CurScopePath.push_back(SlmInitCall);
              Scopes.insert(SlmInitCall);
              continue;
            }
            Function *F1 = CB->getCalledFunction();

            if (!F1) {
              IndirectCallMet = CB->getFunction();
              // If __esimd_slm_alloc will also be met - this will be diagnosed
              // as a user error. Use of __esimd_slm_init does not require
              // accurate call graph and is OK to use with indirect calls. This
              // is because kernels can't be called indirectly, and
              // __esimd_slm_init can only be used in kernels (or in a '()'
              // operator called immediately from a kernel). Hence it is OK to
              // just 'continue' here.
              continue;
            }
            if (F1->isDeclaration()) {
              continue;
            }
            // A call encountered - add a node for the callee and a reverse edge
            // from it to the current scope. Emplace a mapping to dummy null
            // pointer to avoid double search.
            auto E1 =
                Func2Node.emplace(std::make_pair(F1, FuncNodeSPtr{nullptr}));

            if (E1.second) {
              // Insertion took place, which means there was no FuncNode for F1
              // - create it and replace the dummy nullptr node.
              E1.first->second = std::make_shared<FuncNode>(F1);
            }
            E1.first->second->addPred(CurScopePath.back());
          }
        }
        // Add unvisited successors to the work list.
        std::copy_if(succ_begin(BB), succ_end(BB), std::back_inserter(Wl),
                     [&](const BasicBlock *BB1) {
                       return Visited.find(BB1) == Visited.end();
                     });
      }
      // CurScopePath must've been (mostly) exhausted.
      assert((CurScopePath.size() > 0) &&
             (cast<FuncNode>(CurScopePath[0].get())->getFunction() == &F));
      assert((CurScopePath.size() < 2) ||
             isSlmInitCall(cast<ScopeNode>(CurScopePath[1].get())->getStart()));
      llvm::esimd::assert_and_diag(
          !NonConstSlmInitMet || !ScopeMet,
          "non-constant version of slm_init can't be used together with "
          "slm_allocator, function ",
          F.getName());
    }
    if (IndirectCallMet) {
      esimd::assert_and_diag(
          SLMAllocCnt == 0,
          "slm_allocator used together with indirect call in "
          "the same program in ",
          IndirectCallMet->getName());
    }
  }

  const SmallPtrSetImpl<const Function *> &getKernels() const {
    return Kernels;
  }

  const std::set<ScopeNodeSPtr, ScopeNodeSPtrLess> &getScopes() const {
    return Scopes;
  }

  const FuncNodeSPtr getNode(const Function *F) const {
    auto I = Func2Node.find(F);
    assert(I != Func2Node.end());
    return I->second;
  }

  size_t getNumSLMScopes() const { return Scopes.size(); }

#ifndef NDEBUG
  void dump() {
    llvm::errs() << "=== Kernels:\n";
    for (auto *F : Kernels) {
      llvm::errs() << F->getName() << "\n";
    }
    llvm::errs() << "\n=== Function nodes:\n";
    for (auto &E : Func2Node) {
      E.second->dump();
    }
    llvm::errs() << "\n=== Scope nodes:\n";
    for (auto &Scope : getScopes()) {
      Scope->dump();
    }
  }
#endif
};

#ifndef NDEBUG
unsigned ScopedCallGraph::Node::InstanceN = 0;
#endif

// Represents a set of kernel functions.
// TODO With large number of kernels this can be replaced with a bitset, 1 bit
// per kernel, for faster copying.
using KernelSet = SmallPtrSet<const Function *, 4>;
// Represents a result of the analysis for a single scope:
// - maximum dynamically possible SLM frame size at the point of scope start
// - a set of kernels reachable from this scope via predecessors
using TraversalResult = std::pair<int, KernelSet>;
// Records results of the analysis for each call graph node.
using Node2TraversalResultMap =
    std::unordered_map<const ScopedCallGraph::Node *, TraversalResult>;

// Employs dynamic programming technique to find maximum SLM usage along all
// paths in a scoped callgraph from given scope 'Cur' to all reachable kernels.
// 'Results' is the dynamic programming result cache, which later is also used
// to replace __esimd_slm_alloc calls with constant SLM offset.
TraversalResult findMaxSLMUsageAlongAllPaths(const ScopedCallGraph::Node *Cur,
                                             Node2TraversalResultMap &Results) {

  auto ResI = Results.find(Cur);

  if (ResI != Results.end()) {
    // This node has already been analyzed - return cached result.
    return ResI->second;
  }
  constexpr int KernelNotReached = -1;
  int MaxSLMUsage = KernelNotReached;
  KernelSet ReachedKernels{};

  // Solve sub-problems - find maximum SLM usage for each predecessor.
  for (const auto &Pred : Cur->preds()) {
    auto TResult = findMaxSLMUsageAlongAllPaths(Pred.get(), Results);
    KernelSet &PredReachedKernels = TResult.second;
    MaxSLMUsage = std::max(MaxSLMUsage, TResult.first);
    std::copy(PredReachedKernels.begin(), PredReachedKernels.end(),
              std::inserter(ReachedKernels, ReachedKernels.begin()));
  }
  int SLMUsageByCur = 0;

  if (const auto *Scope = dyn_cast<ScopedCallGraph::ScopeNode>(Cur)) {
    SLMUsageByCur = getSLMUsage(Scope->getStart());
  }
  bool CurIsKernel = false;

  if (const auto *FNode = dyn_cast<ScopedCallGraph::FuncNode>(Cur)) {
    const Function *F = FNode->getFunction();
    CurIsKernel = esimd::isESIMDKernel(*F);

    if (CurIsKernel) {
      ReachedKernels.insert(F);
    }
  }
  int ResSLM = CurIsKernel ? 0
                           : (MaxSLMUsage < 0 ? MaxSLMUsage
                                              : SLMUsageByCur + MaxSLMUsage);
  // Construct result for current node from sub-problem solution results and
  // cache it.
  TraversalResult Res = std::make_pair(ResSLM, std::move(ReachedKernels));
  Results[Cur] = Res;
  return Res;
}

PreservedAnalyses
ESIMDLowerSLMReservationCalls::run(Module &M, ModuleAnalysisManager &MAM) {
  // Create a detailed "scoped" call graph. Scope start/end is marked with
  // x = __esimd_slm_alloc / __esimd_slm_free(x)
  //
  // The alternative version may appears when inlining is off:
  //   %slm_obj = ...
  //   call spir_func void slm_allocator(%slm_obj)
  //   ...
  //   call spir_func void ~slm_allocator(%slm_obj)
  // This second variant though is automatically converted to the first one
  // by enforcing always-inliner pass started before this SLM reservation.
  // TODO: enforcing the inlining helps to simplify the alloc/free pattern
  // recognition, but even with inlining the use-def chains may be too complex
  // especially with -O0. So, some extra work is needed for -O0 to enable
  // usage of slm_allocator().

  ScopedCallGraph SCG(M);
#ifndef NDEBUG
  if (DebugLevel > 0) {
    SCG.dump();
  }
#endif
  if (SCG.getNumSLMScopes() == 0) {
    // Early bail out if nothing to analyze.
    return PreservedAnalyses::none();
  }
  // Use the detailed call graph nodes to calculate maximum possible SLM usage
  // at any "scope start" node, and record this info in the result map.
  Node2TraversalResultMap Node2TraversalResult;

  for (const auto &ScopeNodeSPtr : SCG.getScopes()) {
    (void)findMaxSLMUsageAlongAllPaths(ScopeNodeSPtr.get(),
                                       Node2TraversalResult);
  }
  int SLMAllocCallCnt = 0;
  // Maps a kernel to maximum possible SLM usage along any call graph's path.
  DenseMap<const Function *, int> Kernel2MaxSLM;

  // Perform actual lowering of the SLM management calls and calculate maximum
  // SLM usage per kernel.
  for (auto &E : Node2TraversalResult) {
    const auto *Scope = dyn_cast<ScopedCallGraph::ScopeNode>(E.first);

    if (!Scope) {
      // Non-scope nodes do not allocate SLM - skip.
      continue;
    }
    int MaxSLM = E.second.first;
    CallInst *ScopeStartCI = Scope->getStart();
    CallInst *ScopeEndCI = Scope->getEnd();

    if (isSlmAllocCall(ScopeStartCI)) {
      // '__esimd_slm_init' calls always allocate SLM starting from 0 offset, so
      // no IR replacement is necessary for them. '__esimd_slm_alloc' allocates
      // at the end of the maximum dynamically possible SLM frame at the call
      // site. Replace the call with the calculated frame size.
      int SLMOff = MaxSLM - getSLMUsage(ScopeStartCI);
      Type *Int32T = Type::getInt32Ty(ScopeStartCI->getContext());
      auto *SLMOffC = cast<ConstantInt>(ConstantInt::get(Int32T, SLMOff));
      ScopeStartCI->replaceAllUsesWith(SLMOffC);
#ifndef NDEBUG
      if (DebugLevel > 1) {
        llvm::errs() << ">> Replaced\n";
        Scope->dump();
        llvm::errs() << "->\n";
        SLMOffC->dump();
      }
#endif
    }
    if (!isNonConstSLMInit(ScopeStartCI)) {
      // slm_init(non_constant) calls are lowered by the BE to support
      // specialization constants.
      ScopeStartCI->eraseFromParent();
    }
    // slm_init does not have scope end (it is implicit)
    if (ScopeEndCI) {
      ScopeEndCI->eraseFromParent();
    }
    SLMAllocCallCnt++;

    // Now update max SLM usage for all kernels reachable via predecessors from
    // the 'Scope'.
    for (const Function *Kernel : E.second.second) {
      auto I = Kernel2MaxSLM.insert(std::make_pair(Kernel, MaxSLM));

      // if insertion did not happen, update max if needed:
      if (!I.second && (MaxSLM > I.first->second)) {
        I.first->second = MaxSLM;
      }
    }
  }
  // Update (assign SLM size metadata) all kernels' maximum SLM usage with
  // MAX_SLM (if it is greater).
  // - parse genx.kernels metadata and map kernel to its MDNode
  llvm::NamedMDNode *GenXKernelMD =
      M.getNamedMetadata(esimd::GENX_KERNEL_METADATA);
  llvm::esimd::assert_and_diag(GenXKernelMD, "invalid genx.kernels metadata");
  DenseMap<const Function *, MDNode *> Kernel2SlmMD;

  for (MDNode *Node : GenXKernelMD->operands()) {
    Function *Kernel = dyn_cast<Function>(
        esimd::getValue(Node->getOperand(genx::KernelMDOp::FunctionRef)));
    Kernel2SlmMD[Kernel] = Node;
  }
  // - now set each kernel's SLMSize metadata to the pre-calculated value
  for (auto &E : Kernel2MaxSLM) {
    int MaxSLM = E.second;
    // Clamp negative values to 0. MaxSLM could have been not estimated, e.g.
    // due to having __esimd_slm_init with non-const operand (specialization
    // constant case). VC backend will use size provided in __esimd_slm_init
    // if it is greater than value provided in metadata.
    if (MaxSLM < 0)
      MaxSLM = 0;
    llvm::Value *MaxSLMv =
        llvm::ConstantInt::get(Type::getInt32Ty(M.getContext()), MaxSLM);
    const Function *Kernel = E.first;
    Kernel2SlmMD[Kernel]->replaceOperandWith(genx::KernelMDOp::SLMSize,
                                             esimd::getMetadata(MaxSLMv));
#ifndef NDEBUG
    if (DebugLevel > 0) {
      llvm::errs() << ">> SLM usage for " << Kernel->getName() << ": " << MaxSLM
                   << "\n";
    }
#endif
  }
  // Return the number of API calls transformed.
  return SLMAllocCallCnt == 0 ? PreservedAnalyses::none()
                              : PreservedAnalyses::all();
}

} // namespace llvm
