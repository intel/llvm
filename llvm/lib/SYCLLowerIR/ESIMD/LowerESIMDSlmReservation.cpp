//===----- LowerESIMDSlmReservation - lower __esimd_slm_* -----------------===//
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

#define DEBUG_TYPE "LowerESIMDSlmAllocPass"

namespace llvm {

namespace {

#ifndef NDEBUG
constexpr int DebugLevel = 0;
#endif

constexpr char SLM_ALLOC_PREFIX[] = "_Z17__esimd_slm_alloc";
constexpr char SLM_FREE_PREFIX[] = "_Z16__esimd_slm_free";
constexpr char SLM_INIT_PREFIX[] = "_Z16__esimd_slm_init";

bool isSlmInitCall(const CallInst *CI) {
  return CI->getCalledFunction()->getName().startswith(SLM_INIT_PREFIX);
}

bool isSlmAllocCall(const CallInst *CI) {
  return CI->getCalledFunction()->getName().startswith(SLM_ALLOC_PREFIX);
}

// Checks if given call is a call to '__esimd_slm_free' function, and if yes,
// finds the corresponding '__esimd_slm_alloc' call and returns it.
CallInst *isSlmFreeCall(const CallInst *CI) {
  if (!CI->getCalledFunction()->getName().startswith(SLM_FREE_PREFIX))
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
  StringRef Name = SLMReserveCall->getCalledFunction()->getName();
  auto *ArgV = SLMReserveCall->getArgOperand(0);

  if (!isa<ConstantInt>(ArgV)) {
    esimd::assert_and_diag(
        isSlmInitCall(SLMReserveCall),
        "__esimd_slm_alloc with non-constant argument, function ",
        SLMReserveCall->getFunction()->getName());
    return -1;
  }
  size_t Res = cast<llvm::ConstantInt>(ArgV)->getZExtValue();
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
        Visited.insert(BB);

        for (Instruction &I : *BB) {
          if (CallInst *ScopeStartCI = IsScopeStart(&I)) {
            ScopeMet = true;
            auto N = std::make_shared<ScopeNode>(ScopeStartCI);
            N->addPred(CurScopePath.back());
            Scopes.insert(N);
            CurScopePath.emplace_back(std::move(N));
            continue;
          }
          if (CallInst *ScopeStartCI = IsScopeEnd(&I)) {
            ScopeMet = true;
            // Scope end marker encountered - verify all enclosed scopes have
            // ended and truncate current scope path to the enclosing node.
            auto *CurScope = cast<ScopeNode>(CurScopePath.pop_back_val().get());
            assert(ScopeStartCI == CurScope->getStart());
            CurScope->setEnd(cast<CallInst>(&I));
            continue;
          }
          if (auto *CI = dyn_cast<CallInst>(&I)) {
            if (isSlmInitCall(CI)) {
              esimd::assert_and_diag(!SlmInitCall,
                                     "multiple slm_init calls in function ",
                                     F.getName());
              esimd::assert_and_diag(
                  esimd::isESIMDKernel(F) ||
                      sycl::utils::isSYCLExternalFunction(&F),
                  "slm_init call met in non-kernel non-external function ",
                  F.getName());
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
            if (CI->getCalledFunction()->isDeclaration()) {
              continue;
            }
            // A call encountered - add a node for the callee and a reverse edge
            // from it to the current scope.
            Function *F1 = CI->getCalledFunction();
            // Emplace a mapping to dummy null pointer to avoid double search.
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

using Func2IntMap = DenseMap<const Function *, int>;
using SLMKernelUsageMap = DenseMap<
    std::pair<const ScopedCallGraph::Node *, const ScopedCallGraph::FuncNode *>,
    int>;
using Node2IntMap = DenseMap<const ScopedCallGraph::ScopeNode *, int>;

// Employs dynamic programming technique to find maximum SLM usage along all
// paths in a scoped callgraph from given scope 'Cur' to given kernel 'Kernel'.
// 'Kernel2MaxSLM' accumulates the maximum SLM usage per kernel (along any path
// from the kernel to any scope), 'Results' is the dynamic programming result
// cache, which later is also used to replace __esimd_slm_alloc calls with
// constant SLM offset.
int findMaxSLMUsageAlongAllPaths(const ScopedCallGraph::Node *Cur,
                                 const ScopedCallGraph::FuncNode *Kernel,
                                 Func2IntMap &Kernel2MaxSLM,
                                 SLMKernelUsageMap &Results) {

  // No protection from endless recursion in the algorithm, as recursion
  // (cycles) in the call graph is prohibited.
  auto ResI = Results.find({Cur, Kernel});

  if (ResI != Results.end()) {
    return ResI->second;
  }
  constexpr int KernelUnreachable = -1;
  int MaxSLMUse = KernelUnreachable;
  int SLMUseF = 0;

  if (const auto *Scope = dyn_cast<ScopedCallGraph::ScopeNode>(Cur)) {
    SLMUseF = getSLMUsage(Scope->getStart());
  }
  if (Cur == Kernel) {
    MaxSLMUse = 0;
  } else {
    for (const auto &Pred : Cur->preds()) {
      int SLMUse = findMaxSLMUsageAlongAllPaths(Pred.get(), Kernel,
                                                Kernel2MaxSLM, Results);
      MaxSLMUse = std::max(MaxSLMUse, SLMUse);
    }
  }
  // If Kernel can not be reached via any of the predecessors, then discard
  // SLMUseF as it does not affect total SLM size needed by the Kernel, and
  // return -1.
  int Res = MaxSLMUse < 0 ? MaxSLMUse : SLMUseF + MaxSLMUse;
  Results[{Cur, Kernel}] = Res;
  auto I = Kernel2MaxSLM.find(Kernel->getFunction());
  assert(I != Kernel2MaxSLM.end());

  // Update per-kernel maximum SLM usage.
  if (I->second < Res) {
    I->second = Res;
  }
  return Res;
}

size_t lowerSLMReservationCalls(Module &M) {
  // Create a detailed "scoped" call graph. Scope start/end is marked with
  // x = __esimd_slm_alloc / __esimd_slm_free(x)
  ScopedCallGraph SCG(M);
#ifndef NDEBUG
  if (DebugLevel > 0) {
    SCG.dump();
  }
#endif

  // This maps a kernel to all reachable __esimd_slm_alloc calls. Each call is
  // mapped to maximum prior SLM usage on any CG (reverse) path leading to the
  // kernel.
  SLMKernelUsageMap Results;
  // Maps a kernel to the maximum SLM size it can potentially use.
  Func2IntMap Kernel2MaxSLM;
  // Maps a scope node to maximum prior SLM usage on any (reverse) path leading
  // to any kernel.
  Node2IntMap Scope2MaxSLM;

  // Now, for each <ScopeNode, kernel FuncNode> pair:
  // find all possible (reverse) paths in the graph from the scope node to the
  // function node and select the one with maximal value of SLM allocated along
  // the path - MAX_SLM.
  for (const Function *Kernel : SCG.getKernels()) {
    Kernel2MaxSLM[Kernel] = 0;
    const ScopedCallGraph::FuncNode *KernelNode = SCG.getNode(Kernel).get();

    for (const auto &ScopeNodeSPtr : SCG.getScopes()) {
      int MaxSLM = findMaxSLMUsageAlongAllPaths(ScopeNodeSPtr.get(), KernelNode,
                                                Kernel2MaxSLM, Results);
      // Now update the global (among all kernels) maximum SLM usage at this
      // scope.
      auto E = Scope2MaxSLM.insert(std::make_pair(ScopeNodeSPtr.get(), 0));
      if (E.second) {
        // insertion happened, initialize
        E.first->second = MaxSLM;
      }
      E.first->second = std::max(E.first->second, MaxSLM);
    }
  }
  int SLMAllocCallCnt = 0;

  // Replace allocation calls with SLM offsets taken from the Scope2MaxSLM map:
  // 'off = __esimd_slm_alloc(N)' with 'MAX_SLM - N' constant.
  // Also, remove the scope end marker '__esimd_slm_free(off)'.
  for (const auto &E : Scope2MaxSLM) {
    const auto *Scope = dyn_cast<ScopedCallGraph::ScopeNode>(E.first);

    if (!Scope) {
      continue;
    }
    CallInst *ScopeStartCI = Scope->getStart();
    CallInst *ScopeEndCI = Scope->getEnd();

    if (isSlmAllocCall(ScopeStartCI)) {
      int SLMUse = E.second;
      int SLMOff = SLMUse - getSLMUsage(ScopeStartCI);
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
    int MaxSlm = E.second;
    llvm::Value *MaxSlmV =
        llvm::ConstantInt::get(Type::getInt32Ty(M.getContext()), MaxSlm);
    const Function *Kernel = E.first;
    Kernel2SlmMD[Kernel]->replaceOperandWith(genx::KernelMDOp::SLMSize,
                                             esimd::getMetadata(MaxSlmV));
#ifndef NDEBUG
    if (DebugLevel > 0) {
      llvm::errs() << ">> SLM usage for " << Kernel->getName() << ": " << MaxSlm
                   << "\n";
    }
#endif
  }
  // Return the number of API calls transformed.
  return SLMAllocCallCnt;
}

} // namespace llvm
