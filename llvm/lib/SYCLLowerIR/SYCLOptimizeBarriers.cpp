//==== SYCLOptimizeBarriers.cpp - SYCL barrier optimization pass ====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass optimizes __spirv_ControlBarrier and __spirv_MemoryBarrier calls.
//
// SYCL Barrier-Optimization Pass Overview
//
// 1) **Collect Phase**
//    * Walk through the function and record every barrier call into a list of
//    BarrierDesc:
//      – CI         : the call instruction
//      – ExecScope  : the execution-scope operand
//      – MemScope   : the memory-scope operand
//      – Semantic   : the fence-semantics bits
//    * At the same time, build a per-BB summary of memory accesses:
//      – None     : only private/constant or no accesses
//      – Local    : at least one addrspace(3) access
//      – Global   : at least one addrspace(1/5/6) access (with an exception of
//      loads from __spirv_BuiltIn GVs)
//      – Unknown  : any other mayReadOrWriteMemory() (intrinsics, calls,
//      generic addrspace)
//      - Unknown: any other mayReadOrWriteMemory() instruction
//
// 2) **At Entry and At Exit Elimination**
//    - **Entry**: For each barrier B, if on *every* path from function entry to
//    B there are no accesses to memory region greater than or equal to
//    B.MemScope, then remove B.
//    - **Exit** : For each barrier B, if on *every* path from B to any function
//    return there are no accesses to memory region greater than or equal to
//    B.MemScope, then remove B.
//
// 3) **Back-to-Back Elimination (per-BB)**
//    a) *Pure-Sync Collapse*
//       If BB summary == None (no local/global/unknown accesses):
//         – Find the single barrier with the *widest* (ExecScope, MemScope)
//         (ignore Unknown).
//         – Erase all other barriers (they synchronize nothing).
//    b) *General Redundancy Check*
//       Otherwise we walk the barriers in source order and compare each new
//       barrier to the most recent one that is still alive:
//       - If they fence the same execution + memory scope and there are no
//         accesses that need fencing between them, the later barrier is
//         redundant and removed.
//       - If the earlier barrier fences a superset of what the later one would
//         fence and there are no accesses that only the later barrier would
//         need to order, the later barrier is removed.
//       - Symmetrically, if the later barrier fences a superset and the
//       intervening code contains nothing that only the earlier barrier needed,
//       the earlier barrier is removed.
//    Any barrier whose execution or memory scope is Unknown is kept
//    conservatively. After a single pass every basic block contains only the
//    minimal set of barriers required to enforce ordering for the memory
//    operations it actually performs.
//
// 3) **CFG-Wide Optimization (Dominator/Post-Dominator)**
//    Perform barrier analysis across the entire CFG using dominance
//    and post-dominance to remove or narrow memory scope and semantic of
//    barrier calls:
//
//    a) *Dominator-Based Elimination* — For any two barriers A and B where
//       A's ExecScope and MemScope cover B's (i.e., A subsumes B in both
//       execution and memory ordering semantics) and A's fence semantics
//       include B's, if A dominates B and B post-dominates A, and there are no
//       memory accesses at or above the fenced scope on any path between A and
//       B, then B is fully redundant and can be removed.
//
//    b) *Global-to-Local Downgrade* — For barriers that fence global memory
//       (Device/CrossDevice or CrossWorkgroupMemory semantics), if another
//       global barrier A dominates or post-dominates barrier B with no
//       intervening global or unknown accesses, B's MemScope is lowered to
//       Workgroup. Their fence semantics are merged so that no ordering
//       guarantees are weakened.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLOptimizeBarriers.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "sycl-opt-barriers"

namespace {

// Hard-coded special names used in the pass.
static constexpr char CONTROL_BARRIER[] = "_Z22__spirv_ControlBarrieriii";
static constexpr char MEMORY_BARRIER[] = "_Z21__spirv_MemoryBarrierii";
static constexpr char ITT_BARRIER[] = "__itt_offload_wg_barrier_wrapper";
static constexpr char ITT_RESUME[] = "__itt_offload_wi_resume_wrapper";
static constexpr char SPIRV_BUILTIN_PREFIX[] = "__spirv_BuiltIn";

// Simple enum to capture whether a block has local/global/unknown accesses.
enum class RegionMemScope {
  None = 0,
  Local = 1,
  Global = 2,
  Generic = 3,
  Unknown = 4
};

// Known address spaces for SPIR target.
enum class SPIRAddrSpaces {
  Private = 0,
  Global = 1,
  Constant = 2,
  Local = 3,
  Generic = 4,
  GlobalDevice = 5,
  GlobalHost = 6
};

// Map SPIR-V address spaces to our little RegionMemScope domain.
static const std::unordered_map<uint32_t, RegionMemScope> AddrSpaceMap = {
    {static_cast<uint32_t>(SPIRAddrSpaces::Private), RegionMemScope::None},
    {static_cast<uint32_t>(SPIRAddrSpaces::Constant), RegionMemScope::None},

    {static_cast<uint32_t>(SPIRAddrSpaces::Global), RegionMemScope::Global},
    {static_cast<uint32_t>(SPIRAddrSpaces::GlobalDevice),
     RegionMemScope::Global},
    {static_cast<uint32_t>(SPIRAddrSpaces::GlobalHost), RegionMemScope::Global},

    {static_cast<uint32_t>(SPIRAddrSpaces::Local), RegionMemScope::Local},

    {static_cast<uint32_t>(SPIRAddrSpaces::Generic), RegionMemScope::Generic},
    // any future AS default to Unknown
};

// Scopes in SPIR-V.
enum class Scope {
  CrossDevice = 0,
  Device = 1,
  Workgroup = 2,
  Subgroup = 3,
  Invocation = 4,
  Unknown = 10
};

// This enum, map and compare function are added to compare widths of the
// barrier scopes and make pass forward compatible in case if new scopes
// appearing in SPIR-V and/or SYCL.
enum class CompareRes { BIGGER = 0, SMALLER = 1, EQUAL = 2, UNKNOWN = 3 };

const std::unordered_map<Scope, uint32_t> ScopeWeights = {
    {Scope::CrossDevice, 1000},
    {Scope::Device, 800},
    {Scope::Workgroup, 600},
    {Scope::Subgroup, 400},
    {Scope::Invocation, 10}};

static inline CompareRes compareScopesWithWeights(Scope LHS, Scope RHS) {
  auto LHSIt = ScopeWeights.find(LHS);
  auto RHSIt = ScopeWeights.find(RHS);

  if (LHSIt == ScopeWeights.end() || RHSIt == ScopeWeights.end())
    return CompareRes::UNKNOWN;

  const uint64_t LHSWeight = LHSIt->second;
  const uint64_t RHSWeight = RHSIt->second;

  if (LHSWeight > RHSWeight)
    return CompareRes::BIGGER;
  if (LHSWeight < RHSWeight)
    return CompareRes::SMALLER;
  return CompareRes::EQUAL;
}

enum class MemorySemantics {
  SubgroupMemory = 0x80,
  WorkgroupMemory = 0x100,
  CrossWorkgroupMemory = 0x200
};

enum class Ordering {
  Acquire = 0x2,
  Release = 0x4,
  AcquireRelease = 0x8,
  SequentiallyConsistent = 0x10
};

static constexpr uint32_t MemorySemanticMask = ~0x3fu;

// Normalize a raw 'memory semantics' bitmask to a canonical form.
static inline uint32_t canonicalizeSemantic(uint32_t Sem) {
  bool HasAcq = Sem & static_cast<uint32_t>(Ordering::Acquire);
  bool HasRel = Sem & static_cast<uint32_t>(Ordering::Release);
  bool HasAcqRel = Sem & static_cast<uint32_t>(Ordering::AcquireRelease);
  bool HasSeq = Sem & static_cast<uint32_t>(Ordering::SequentiallyConsistent);

  if (HasSeq)
    Sem &= MemorySemanticMask |
           static_cast<uint32_t>(Ordering::SequentiallyConsistent);
  else {
    if (HasAcq && HasRel)
      HasAcqRel = true;
    if (HasAcqRel) {
      Sem &= ~(static_cast<uint32_t>(Ordering::Acquire) |
               static_cast<uint32_t>(Ordering::Release));
      Sem |= static_cast<uint32_t>(Ordering::AcquireRelease);
    }
  }
  return Sem;
}

// Merge two semantics bitmasks into a single canonical form.
static inline uint32_t mergeSemantics(uint32_t A, uint32_t B) {
  return canonicalizeSemantic(A | B);
}

// Return the ordering class of a semantic bitmask.
static inline int orderingClass(uint32_t Sem) {
  Sem = canonicalizeSemantic(Sem);
  if (Sem & static_cast<uint32_t>(Ordering::SequentiallyConsistent))
    return 4;
  if (Sem & static_cast<uint32_t>(Ordering::AcquireRelease))
    return 3;
  if (Sem & static_cast<uint32_t>(Ordering::Release))
    return 2;
  if (Sem & static_cast<uint32_t>(Ordering::Acquire))
    return 1;
  return 0;
}

// Check if A is a superset of B in terms of semantics and ordering.
static inline bool semanticsSuperset(uint32_t A, uint32_t B) {
  A = canonicalizeSemantic(A);
  B = canonicalizeSemantic(B);
  uint32_t AMem = A & MemorySemanticMask;
  uint32_t BMem = B & MemorySemanticMask;
  if ((BMem & ~AMem) != 0)
    return false;

  int AOrd = orderingClass(A);
  int BOrd = orderingClass(B);

  if (AOrd == 4)
    return true;
  if (AOrd == 3)
    return BOrd <= 3;
  if (AOrd == 2)
    return BOrd == 2 || BOrd == 0;
  if (AOrd == 1)
    return BOrd == 1 || BOrd == 0;
  return BOrd == 0;
}

// Holds everything we know about one barrier invocation.
struct BarrierDesc {
  CallInst *CI;
  Scope ExecScope;
  Scope MemScope;
  uint32_t Semantic;
};

// Per-BB summary of what kinds of accesses appear.
using BBMemInfoMap = DenseMap<BasicBlock *, RegionMemScope>;

// Per-BB summary of Barriers.
using BarriersMap = MapVector<BasicBlock *, SmallVector<BarrierDesc, 2>>;

// Map SPIR-V Barrier Scope to the RegionMemScope that a barrier of that kind
// actually fences.
template <RegionMemScope SearchFor = RegionMemScope::Local>
static inline RegionMemScope getBarrierFencedScopeImpl(const BarrierDesc &BD) {
  uint32_t Sem = canonicalizeSemantic(BD.Semantic);
  constexpr uint32_t LocalMask =
      static_cast<uint32_t>(MemorySemantics::WorkgroupMemory) |
      static_cast<uint32_t>(MemorySemantics::SubgroupMemory);
  constexpr uint32_t GlobalMask =
      static_cast<uint32_t>(MemorySemantics::CrossWorkgroupMemory);

  if constexpr (SearchFor == RegionMemScope::Local) {
    if (Sem & LocalMask)
      return RegionMemScope::Local;
    if (Sem & GlobalMask)
      return RegionMemScope::Global;
  } else {
    if (Sem & GlobalMask)
      return RegionMemScope::Global;
    if (Sem & LocalMask)
      return RegionMemScope::Local;
  }

  return RegionMemScope::None;
}

static inline RegionMemScope getBarrierFencedScope(const BarrierDesc &BD) {
  return getBarrierFencedScopeImpl<RegionMemScope::Local>(BD);
}
static inline RegionMemScope getBarrierMaxFencedScope(const BarrierDesc &BD) {
  return getBarrierFencedScopeImpl<RegionMemScope::Global>(BD);
}

static bool isSPIRVBuiltinFunction(const StringRef FName) {
  return FName == "_Z22__spirv_BuiltInWorkDimv" ||
         FName == "_Z28__spirv_BuiltInWorkgroupSizei" ||
         FName == "_Z32__spirv_BuiltInLocalInvocationIdi" ||
         FName == "_Z33__spirv_BuiltInGlobalInvocationIdi" ||
         FName == "_Z26__spirv_BuiltInWorkgroupIdi" ||
         FName == "_Z27__spirv_BuiltInGlobalOffseti" ||
         FName == "_Z25__spirv_BuiltInGlobalSizei" ||
         FName == "_Z28__spirv_BuiltInNumWorkgroupsi" ||
         FName == "_Z30__spirv_BuiltInSubgroupMaxSizev" ||
         FName == "_Z25__spirv_BuiltInSubgroupIdv" ||
         FName == "_Z40__spirv_BuiltInSubgroupLocalInvocationIdv" ||
         FName == "_Z27__spirv_BuiltInSubgroupSizev" ||
         FName == "_Z27__spirv_BuiltInNumSubgroupsv" ||
         FName == "_Z35__spirv_BuiltInLocalInvocationIndexv" ||
         FName == "_Z29__spirv_BuiltInGlobalLinearIdv" ||
         FName == "_Z36__spirv_BuiltInEnqueuedWorkgroupSizei" ||
         FName == "_Z35__spirv_BuiltInNumEnqueuedSubgroupsv" ||
         FName == "_Z40__spirv_BuiltInSubgroupLocalInvocationIdv" ||
         FName == "_Z29__spirv_BuiltInSubgroupEqMaskv" ||
         FName == "_Z32__spirv_BuiltInSubgroupEqMaskKHRv" ||
         FName == "_Z29__spirv_BuiltInSubgroupGeMaskv" ||
         FName == "_Z32__spirv_BuiltInSubgroupGeMaskKHRv" ||
         FName == "_Z29__spirv_BuiltInSubgroupGtMaskv" ||
         FName == "_Z32__spirv_BuiltInSubgroupGtMaskKHRv" ||
         FName == "_Z29__spirv_BuiltInSubgroupLeMaskv" ||
         FName == "_Z32__spirv_BuiltInSubgroupLeMaskKHRv" ||
         FName == "_Z29__spirv_BuiltInSubgroupLtMaskv" ||
         FName == "_Z32__spirv_BuiltInSubgroupLtMaskKHRv";
}
// Classify a single instruction's memory scope. Used to set/update memory
// scope of a basic block.
static RegionMemScope classifyMemScope(Instruction *I) {
  if (CallInst *CI = dyn_cast<CallInst>(I)) {
    if (Function *F = CI->getCalledFunction()) {
      const StringRef FName = F->getName();
      if (FName == CONTROL_BARRIER || FName == MEMORY_BARRIER ||
          FName == ITT_BARRIER || FName == ITT_RESUME)
        return RegionMemScope::None;
      if (isSPIRVBuiltinFunction(FName))
        return RegionMemScope::None;
      if (FName.contains("__spirv_Atomic")) {
        // SPIR-V atomics all have the same signature:
        // arg0 = ptr, arg1 = SPIR-V Scope, arg2 = Semantics
        auto *ScopeC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
        auto *SemC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
        if (!ScopeC || !SemC)
          return RegionMemScope::Unknown;
        // If the semantics mention CrossWorkgroupMemory, treat as global.
        uint32_t SemVal = canonicalizeSemantic(SemC->getZExtValue());
        if (SemVal & (uint32_t)MemorySemantics::CrossWorkgroupMemory)
          return RegionMemScope::Global;
        if (SemVal & ((uint32_t)MemorySemantics::WorkgroupMemory |
                      (uint32_t)MemorySemantics::SubgroupMemory))
          return RegionMemScope::Local;
        switch (ScopeC->getZExtValue()) {
        case static_cast<uint32_t>(Scope::CrossDevice):
        case static_cast<uint32_t>(Scope::Device):
          return RegionMemScope::Global;
        case static_cast<uint32_t>(Scope::Workgroup):
        case static_cast<uint32_t>(Scope::Subgroup):
          return RegionMemScope::Local;
        case static_cast<uint32_t>(Scope::Invocation):
          return RegionMemScope::None;
        default:
          return RegionMemScope::Unknown;
        }
      }
      // TODO: handle other SPIR-V friendly function calls.
    }
  }

  // If it doesn't read or write, it doesn't affect the region memory scope.
  if (!I->mayReadOrWriteMemory())
    return RegionMemScope::None;

  auto resolveGeneric = [&](Value *Pointer) -> RegionMemScope {
    // If generic pointer originates from an alloca instruction within a
    // function - it's safe to assume, that it's a private allocation.
    // FIXME: use more comprehensive analysis.
    Value *Orig = Pointer->stripInBoundsConstantOffsets();
    if (isa<AllocaInst>(Orig))
      return RegionMemScope::None;
    uint32_t AS = cast<PointerType>(Orig->getType())->getAddressSpace();
    auto Pos = AddrSpaceMap.find(AS);
    if (Pos == AddrSpaceMap.end())
      return RegionMemScope::Unknown;
    return Pos->second == RegionMemScope::Generic ? RegionMemScope::Unknown
                                                  : Pos->second;
  };

  auto getScopeForPtr = [&](Value *Ptr, uint32_t AS) -> RegionMemScope {
    // Loads from __spirv_BuiltIn GVs are not fenced by barriers.
    if (auto *GV = dyn_cast<GlobalVariable>(Ptr))
      if (GV->getName().starts_with(SPIRV_BUILTIN_PREFIX))
        return RegionMemScope::None;
    auto Pos = AddrSpaceMap.find(AS);
    if (Pos == AddrSpaceMap.end())
      return RegionMemScope::Unknown;
    return Pos->second == RegionMemScope::Generic ? resolveGeneric(Ptr)
                                                  : Pos->second;
  };

  // Check for memory instructions.
  // TODO: check for other intrinsics
  if (auto *LD = dyn_cast<LoadInst>(I))
    return getScopeForPtr(LD->getPointerOperand(),
                          LD->getPointerAddressSpace());
  if (auto *ST = dyn_cast<StoreInst>(I))
    return getScopeForPtr(ST->getPointerOperand(),
                          ST->getPointerAddressSpace());
  if (auto *MI = dyn_cast<MemIntrinsic>(I)) {
    RegionMemScope Scope =
        getScopeForPtr(MI->getDest(), MI->getDestAddressSpace());

    if (auto *MT = dyn_cast<MemTransferInst>(MI)) {
      RegionMemScope SrcScope =
          getScopeForPtr(MT->getSource(), MT->getSourceAddressSpace());
      Scope = std::max(Scope, SrcScope);
    }
    return Scope;
  }
  if (isa<FenceInst>(I))
    return RegionMemScope::Global;

  if (auto *RMW = dyn_cast<AtomicRMWInst>(I))
    return getScopeForPtr(RMW->getPointerOperand(),
                          RMW->getPointerAddressSpace());
  if (auto *CompEx = dyn_cast<AtomicCmpXchgInst>(I))
    return getScopeForPtr(CompEx->getPointerOperand(),
                          CompEx->getPointerAddressSpace());

  return RegionMemScope::Unknown;
}

// Scan the function and build:
// - list of all BarrierDesc‘s
// - per-BB memory-scope summary
static void collectBarriersAndMemInfo(Function &F,
                                      SmallVectorImpl<BarrierDesc> &Barriers,
                                      BBMemInfoMap &BBMemInfo) {
  for (auto &BB : F) {
    RegionMemScope BlockScope = RegionMemScope::None;

    for (auto &I : BB) {
      // Update memory info.
      RegionMemScope InstScope = classifyMemScope(&I);
      BlockScope = std::max(BlockScope, InstScope);

      // Collect barriers.
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        Function *Callee = CI->getCalledFunction();
        if (!Callee) {
          BlockScope = RegionMemScope::Unknown;
          continue;
        }

        // Check if this is a control/memory barrier call and store it.
        StringRef Name = Callee->getName();
        auto getConst = [&](uint32_t idx) -> uint32_t {
          if (auto *C = dyn_cast<ConstantInt>(CI->getArgOperand(idx)))
            return C->getZExtValue();
          return static_cast<uint32_t>(Scope::Unknown);
        };
        if (Name == CONTROL_BARRIER) {
          LLVM_DEBUG(dbgs() << "Collected ControlBarrier: " << *CI << "\n");
          BarrierDesc BD = {CI, static_cast<Scope>(getConst(0)),
                            static_cast<Scope>(getConst(1)), getConst(2)};
          BD.Semantic = canonicalizeSemantic(BD.Semantic);
          Barriers.emplace_back(BD);
        } else if (Name == MEMORY_BARRIER) {
          LLVM_DEBUG(dbgs() << "Collected MemoryBarrier: " << *CI << "\n");
          BarrierDesc BD = {CI, Scope::Invocation,
                            static_cast<Scope>(getConst(0)), getConst(1)};
          BD.Semantic = canonicalizeSemantic(BD.Semantic);
          Barriers.emplace_back(BD);
        }
      }
    }
    BBMemInfo[&BB] = BlockScope;
  }
}

// Check if an instruction is an ITT wrapper call.
static bool isITT(Instruction *Inst) {
  if (CallInst *CI = dyn_cast<CallInst>(Inst)) {
    if (Function *Callee = CI->getCalledFunction()) {
      StringRef Name = Callee->getName();
      if (Name == ITT_RESUME || Name == ITT_BARRIER)
        return true;
    }
  }
  return false;
}

// Remove a single barrier CallInst and drop its surrounding ITT calls.
static bool eraseBarrierWithITT(BarrierDesc &BD) {
  if (BD.CI == nullptr)
    return false;
  SmallPtrSet<Instruction *, 3> ToErase;
  CallInst *CI = BD.CI;
  LLVM_DEBUG(dbgs() << "Erase barrier: " << *CI << "\n");
  // Look up/down for ITT markers.
  if (auto *Prev = CI->getPrevNode())
    if (isITT(Prev))
      ToErase.insert(Prev);
  if (auto *Next = CI->getNextNode())
    if (isITT(Next))
      ToErase.insert(Next);
  ToErase.insert(CI);
  BD.CI = nullptr;

  for (auto *I : ToErase) {
    I->dropAllReferences();
    I->eraseFromParent();
  }
  return !ToErase.empty();
}

// Helper to check if a whole block contains accesses fenced by
// 'Required'.
static bool hasFencedAccesses(BasicBlock *BB, RegionMemScope Required,
                              const BBMemInfoMap &BBMemInfo) {
  LLVM_DEBUG(dbgs() << "Checking for fenced accesses in basic block\n");
  RegionMemScope S = BBMemInfo.lookup(BB);
  if (S == RegionMemScope::Unknown)
    return true;
  return S >= Required;
}

// True if no fenced accesses of MemScope appear in [A->next, B).
static bool noFencedMemAccessesBetween(CallInst *A, CallInst *B,
                                       RegionMemScope Required,
                                       const BBMemInfoMap &BBMemInfo) {
  LLVM_DEBUG(dbgs() << "Checking for fenced accesses between: " << *A << " and "
                    << *B << "\n");
  RegionMemScope BBMemScope = BBMemInfo.lookup(A->getParent());
  if (Required == RegionMemScope::Unknown) {
    LLVM_DEBUG(dbgs() << "noFencedMemAccessesBetween(" << *A << ", " << *B
                      << ") returned " << false << "\n");
    return false;
  }

  // Early exit in case if the whole block has no accesses wider or equal to
  // required.
  if (BBMemScope < Required) {
    LLVM_DEBUG(dbgs() << "noFencedMemAccessesBetween(" << *A << ", " << *B
                      << ") returned " << true << "\n");
    return true;
  }

  for (auto It = ++BasicBlock::iterator(A), End = BasicBlock::iterator(B);
       It != End; ++It) {
    auto InstScope = classifyMemScope(&*It);
    if (InstScope == RegionMemScope::Unknown || InstScope >= Required) {
      LLVM_DEBUG(dbgs() << "noFencedMemAccessesBetween(" << *A << ", " << *B
                        << ") returned " << false << "\n");
      return false;
    }
  }
  LLVM_DEBUG(dbgs() << "noFencedMemAccessesBetween(" << *A << ", " << *B
                    << ") returned " << true << "\n");
  return true;
}

/// Return true if no accesses of >= Required scope occur on *every* path
/// from A to B through the CFG.  If A==nullptr, start at EntryBlock; if
/// B==nullptr, end at all exit blocks.
static bool noFencedAccessesCFG(CallInst *A, CallInst *B,
                                RegionMemScope Required,
                                const BBMemInfoMap &BBMemInfo) {
  LLVM_DEBUG(dbgs() << "Checking for fenced accesses between: " << *A << " and "
                    << *B << " in CFG" << "\n");
  if (Required == RegionMemScope::Unknown)
    return false;

  // Shortcut: same block and both non-null.
  if (A && B && A->getParent() == B->getParent())
    return noFencedMemAccessesBetween(A, B, Required, BBMemInfo);

  // Build the set of blocks that can reach B.
  SmallPtrSet<BasicBlock *, 32> ReachB;
  if (B) {
    SmallVector<BasicBlock *, 16> Stack{B->getParent()};
    ReachB.insert(B->getParent());
    while (!Stack.empty()) {
      BasicBlock *Cur = Stack.pop_back_val();
      for (BasicBlock *Pred : predecessors(Cur))
        if (ReachB.insert(Pred).second)
          Stack.push_back(Pred);
    }
  }

  Function *F = (A ? A->getFunction() : B->getFunction());
  BasicBlock *Entry = &F->getEntryBlock();

  // Worklist entries.
  SmallVector<BasicBlock *, 16> Worklist;
  SmallPtrSet<BasicBlock *, 16> Visited;

  auto enqueue = [&](BasicBlock *BB) {
    if (Visited.insert(BB).second)
      Worklist.push_back(BB);
  };

  // Initialize the worklist from CI or ...
  if (A)
    enqueue(A->getParent());
  else
    // ... from kernel's entry.
    enqueue(Entry);

  // Simple BFS-like traversal of the CFG to find all paths from A to B.
  while (!Worklist.empty()) {
    BasicBlock *BB = Worklist.pop_back_val();
    // Check if BB is reachable from B.
    if (B && !ReachB.contains(BB))
      continue;

    // If the BB may contain a violating access - exit.
    if (hasFencedAccesses(BB, Required, BBMemInfo))
      return false;

    // Do not traverse beyond sink block if B is specified.
    if (B && BB == B->getParent())
      continue;

    // Enqueue successors.
    for (BasicBlock *Succ : successors(BB))
      enqueue(Succ);
  }
  // If we never saw a disallowed memory access on any path, it's safe.
  LLVM_DEBUG(dbgs() << "noFencedAccessesCFG(" << *A << ", " << *B
                    << ") returned " << true << "\n");
  return true;
}

// The back-to-back elimination on one BB.
static bool eliminateBackToBackInBB(BasicBlock *BB,
                                    SmallVectorImpl<BarrierDesc> &Barriers,
                                    const BBMemInfoMap &BBMemInfo) {
  SmallVector<BarrierDesc, 8> Survivors;
  bool Changed = false;
  RegionMemScope BlockScope =
      BB ? BBMemInfo.lookup(BB) : RegionMemScope::Unknown;

  // If there are no memory accesses requiring synchronization in this block,
  // collapse all barriers to the single largest one.
  if (BlockScope == RegionMemScope::None) {
    bool HasUnknown = llvm::any_of(Barriers, [](const BarrierDesc &BD) {
      return BD.ExecScope == Scope::Unknown || BD.MemScope == Scope::Unknown;
    });
    if (!HasUnknown) {
      LLVM_DEBUG(
          dbgs() << "Erasing barrier in basic block with no memory accesses\n");
      // Pick the barrier with the widest scope.
      auto Best = std::max_element(
          Barriers.begin(), Barriers.end(), [](auto &A, auto &B) {
            // First prefer the barrier whose semantics fence more memory +
            // stronger ordering.
            if (semanticsSuperset(B.Semantic, A.Semantic) &&
                !semanticsSuperset(A.Semantic, B.Semantic))
              return true;
            if (semanticsSuperset(A.Semantic, B.Semantic) &&
                !semanticsSuperset(B.Semantic, A.Semantic))
              return false;
            // Then fall back to exec/mem‐scope width as before:
            auto CmpExec = compareScopesWithWeights(B.ExecScope, A.ExecScope);
            if (CmpExec != CompareRes::EQUAL)
              return CmpExec == CompareRes::BIGGER;
            auto CmpMem = compareScopesWithWeights(B.MemScope, A.MemScope);
            return CmpMem == CompareRes::BIGGER;
          });

      // Remove all other barriers in the block.
      llvm::erase_if(Barriers, [&](BarrierDesc &BD) {
        if (&BD == &*Best)
          return false;
        Changed |= eraseBarrierWithITT(BD);
        return true;
      });
      return Changed;
    }
  }

  // Otherwise do a sliding window compare of each barrier against the
  // last survivor.
  for (auto &Cur : Barriers) {
    if (!Cur.CI)
      continue; // already removed
    while (!Survivors.empty()) {
      BarrierDesc &Last = Survivors.back();
      uint32_t LastSem = canonicalizeSemantic(Last.Semantic);
      uint32_t CurSem = canonicalizeSemantic(Cur.Semantic);
      uint32_t MergedSem = mergeSemantics(LastSem, CurSem);

      auto CmpExec = compareScopesWithWeights(Last.ExecScope, Cur.ExecScope);
      auto CmpMem = compareScopesWithWeights(Last.MemScope, Cur.MemScope);
      RegionMemScope FenceLast = getBarrierFencedScope(Last);
      RegionMemScope FenceCur = getBarrierFencedScope(Cur);

      // If either scope is unknown, we cannot merge.
      if (CmpExec == CompareRes::UNKNOWN || CmpMem == CompareRes::UNKNOWN ||
          FenceLast == RegionMemScope::Unknown ||
          FenceCur == RegionMemScope::Unknown)
        break;

      auto *Int32Ty = Type::getInt32Ty(Last.CI->getContext());
      // If the execution and memory scopes of the barriers are equal, we can
      // merge them if there are no accesses that only one of the barriers
      // would need to fence.
      RegionMemScope BetweenScope = std::min(FenceLast, FenceCur);
      if (CmpExec == CompareRes::EQUAL && CmpMem == CompareRes::EQUAL) {
        if (semanticsSuperset(LastSem, CurSem) &&
            noFencedMemAccessesBetween(Last.CI, Cur.CI, BetweenScope,
                                       BBMemInfo)) {
          if (MergedSem != LastSem) {
            Last.CI->setArgOperand(2, ConstantInt::get(Int32Ty, MergedSem));
            Last.Semantic = MergedSem;
          }
          Changed |= eraseBarrierWithITT(Cur);
          break;
        }
        if (semanticsSuperset(CurSem, LastSem) &&
            noFencedMemAccessesBetween(Last.CI, Cur.CI, BetweenScope,
                                       BBMemInfo)) {
          if (MergedSem != CurSem) {
            Cur.CI->setArgOperand(2, ConstantInt::get(Int32Ty, MergedSem));
            Cur.Semantic = MergedSem;
          }
          Changed |= eraseBarrierWithITT(Last);
          Survivors.pop_back();
          continue;
        }
        if (noFencedMemAccessesBetween(Last.CI, Cur.CI, BetweenScope,
                                       BBMemInfo)) {
          Last.CI->setArgOperand(2, ConstantInt::get(Int32Ty, MergedSem));
          Last.Semantic = MergedSem;
          Changed |= eraseBarrierWithITT(Cur);
        }
        break;
      }
      // If the execution or memory scope of the barriers is not equal, we
      // can only merge if one is a superset of the other and there are no
      // accesses that only the other barrier would need to fence.
      if ((CmpExec == CompareRes::BIGGER || CmpMem == CompareRes::BIGGER) &&
          semanticsSuperset(LastSem, CurSem) &&
          noFencedMemAccessesBetween(Last.CI, Cur.CI, BetweenScope,
                                     BBMemInfo)) {
        if (MergedSem != LastSem) {
          Last.CI->setArgOperand(2, ConstantInt::get(Int32Ty, MergedSem));
          Last.Semantic = MergedSem;
        }
        Changed |= eraseBarrierWithITT(Cur);
        break;
      }
      if ((CmpExec == CompareRes::SMALLER || CmpMem == CompareRes::SMALLER) &&
          semanticsSuperset(CurSem, LastSem) &&
          noFencedMemAccessesBetween(Last.CI, Cur.CI, BetweenScope,
                                     BBMemInfo)) {
        if (MergedSem != CurSem) {
          Cur.CI->setArgOperand(2, ConstantInt::get(Int32Ty, MergedSem));
          Cur.Semantic = MergedSem;
        }
        Changed |= eraseBarrierWithITT(Last);
        Survivors.pop_back();
        continue;
      }
      break;
    }
    if (Cur.CI) // Still alive?
      Survivors.emplace_back(Cur);
  }

  // If we removed any, replace Barriers with the survivors.
  if (Survivors.size() != Barriers.size()) {
    Barriers.clear();
    Barriers.append(Survivors.begin(), Survivors.end());
    Changed = true;
  }
  return Changed;
}

// Walk the whole CFG once, first trying to erase fully–redundant
// barriers and, if that is impossible, trying to downgrade
// Cross-work-group barriers that are safely covered by another global fence.
static bool optimizeBarriersCFG(SmallVectorImpl<BarrierDesc *> &Barriers,
                                DominatorTree &DT, PostDominatorTree &PDT,
                                const BBMemInfoMap &BBMemInfo) {
  bool Changed = false;

  for (BarrierDesc *B : Barriers) {
    if (!B->CI)
      continue; // Already removed

    bool Removed = false;
    bool IsGlobalB =
        (B->MemScope == Scope::Device || B->MemScope == Scope::CrossDevice ||
         (B->Semantic &
          static_cast<uint32_t>(MemorySemantics::CrossWorkgroupMemory)));
    BarrierDesc *DowngradeCand = nullptr;

    for (BarrierDesc *A : Barriers) {
      if (A == B || !A->CI)
        continue;

      // Elimination check.
      auto ExecCmp = compareScopesWithWeights(A->ExecScope, B->ExecScope);
      auto MemCmp = compareScopesWithWeights(A->MemScope, B->MemScope);
      bool ScopesCover =
          (ExecCmp == CompareRes::BIGGER || ExecCmp == CompareRes::EQUAL) &&
          (MemCmp == CompareRes::BIGGER || MemCmp == CompareRes::EQUAL);
      bool SemCover = (A->Semantic & B->Semantic) == B->Semantic;
      bool ADominatesB = DT.dominates(A->CI, B->CI);
      if (ScopesCover && SemCover) {
        RegionMemScope Fence = getBarrierMaxFencedScope(*A);
        // FIXME: this check is way too conservative.
        if (Fence != RegionMemScope::Unknown && ADominatesB &&
            PDT.dominates(B->CI, A->CI) &&
            noFencedAccessesCFG(A->CI, B->CI, Fence, BBMemInfo)) {
          Changed |= eraseBarrierWithITT(*B);
          Removed = true;
          break;
        }
      }

      // Downgrade check.
      if (!Removed && IsGlobalB && !DowngradeCand) {
        bool IsGlobalA =
            (A->MemScope == Scope::Device ||
             A->MemScope == Scope::CrossDevice ||
             (A->Semantic &
              static_cast<uint32_t>(MemorySemantics::CrossWorkgroupMemory)));
        if (IsGlobalA) {
          if (DT.dominates(A->CI, B->CI) &&
              noFencedAccessesCFG(A->CI, B->CI, RegionMemScope::Global,
                                  BBMemInfo)) {
            DowngradeCand = A;
          } else if (PDT.dominates(A->CI, B->CI) &&
                     noFencedAccessesCFG(B->CI, A->CI, RegionMemScope::Global,
                                         BBMemInfo)) {
            DowngradeCand = A;
          }
        }
      }
    }

    if (Removed)
      continue;

    if (DowngradeCand) {
      BarrierDesc &A = *DowngradeCand;
      BarrierDesc &R = *B;
      uint32_t mergedSem = mergeSemantics(A.Semantic, R.Semantic);
      LLVMContext &Ctx = R.CI->getContext();
      const bool IsControlBarrier =
          R.CI->getCalledFunction()->getName() == CONTROL_BARRIER;
      Type *Int32Ty = Type::getInt32Ty(Ctx);

      // Merge ordering semantics.
      if (mergedSem != R.Semantic) {
        R.CI->setArgOperand(IsControlBarrier ? 2 : 1,
                            ConstantInt::get(Int32Ty, mergedSem));
        R.Semantic = mergedSem;
      }

      // Downgrade CrossWorkgroup -> Workgroup semantics.
      const uint32_t CrossMask =
          static_cast<uint32_t>(MemorySemantics::CrossWorkgroupMemory);
      if (R.Semantic & CrossMask) {
        uint32_t NewSem =
            (R.Semantic & ~CrossMask) |
            static_cast<uint32_t>(MemorySemantics::WorkgroupMemory);
        R.CI->setArgOperand(IsControlBarrier ? 2 : 1,
                            ConstantInt::get(Int32Ty, NewSem));
        R.Semantic = NewSem;
      }

      // Lower the SPIR-V MemScope operand to Workgroup.
      R.CI->setArgOperand(
          IsControlBarrier ? 1 : 0,
          ConstantInt::get(Int32Ty, static_cast<uint32_t>(Scope::Workgroup)));
      R.MemScope = Scope::Workgroup;

      LLVM_DEBUG(dbgs() << "Downgraded global barrier: " << *R.CI << "\n");
      Changed = true;
    }
  }

  return Changed;
}

// True if BD is the first real instruction of the function.
static bool isAtKernelEntry(const BarrierDesc &BD,
                            const BBMemInfoMap &BBMemInfo) {
  BasicBlock &Entry = BD.CI->getFunction()->getEntryBlock();
  if (BD.CI->getParent() != &Entry)
    return false;

  RegionMemScope Fence = getBarrierFencedScope(BD);
  bool EntryHasFenced = hasFencedAccesses(&Entry, Fence, BBMemInfo);

  // Entry block has no such accesses at all -> barrier redundant.
  if (!EntryHasFenced)
    return true;

  // Otherwise it is redundant only if it is the first inst.
  return &*Entry.getFirstNonPHIOrDbgOrAlloca() == BD.CI;
}

// True if BD is immediately before a return/unreachable and nothing follows.
static bool isAtKernelExit(const BarrierDesc &BD,
                           const BBMemInfoMap &BBMemInfo) {
  BasicBlock *BB = BD.CI->getParent();
  Instruction *Term = BB->getTerminator();
  if (!isa<ReturnInst>(Term) && !isa<UnreachableInst>(Term))
    return false;

  RegionMemScope Fence = getBarrierFencedScope(BD);
  bool ExitHasFenced = hasFencedAccesses(BB, Fence, BBMemInfo);

  // Exit block has no such accesses at all -> barrier redundant.
  if (!ExitHasFenced)
    return true;

  // Otherwise it is redundant only if it is the last inst.
  return BD.CI->getNextNode() == Term;
}

// Remove barriers that appear at the very beginning or end of a kernel
// function.
static bool eliminateBoundaryBarriers(SmallVectorImpl<BarrierDesc *> &Barriers,
                                      const BBMemInfoMap &BBMemInfo) {
  bool Changed = false;
  for (auto *BPtr : Barriers) {
    BarrierDesc &B = *BPtr;
    if (!B.CI)
      continue;
    // Only for real SPIR kernels:
    if (B.CI->getFunction()->getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;

    // entry: no fenced accesses at entry BB.
    if (isAtKernelEntry(B, BBMemInfo)) {
      Changed |= eraseBarrierWithITT(B);
      continue;
    }
    // exit: no fenced accesses at termination BB.
    if (isAtKernelExit(B, BBMemInfo)) {
      Changed |= eraseBarrierWithITT(B);
      continue;
    }
  }
  return Changed;
}

} // namespace

PreservedAnalyses SYCLOptimizeBarriersPass::run(Function &F,
                                                FunctionAnalysisManager &AM) {
  if (F.getCallingConv() != CallingConv::SPIR_KERNEL)
    return PreservedAnalyses::none();
  LLVM_DEBUG(dbgs() << "Running SYCLOptimizeBarriers on " << F.getName()
                    << "\n");
  SmallVector<BarrierDesc, 16> Barriers;
  BBMemInfoMap BBMemInfo;
  BarriersMap BarriersByBB;
  SmallVector<BarrierDesc *, 16> BarrierPtrs;

  // Analyse the function gathering barrier and memory scope of the region info.
  collectBarriersAndMemInfo(F, Barriers, BBMemInfo);
  for (auto &B : Barriers)
    BarriersByBB[B.CI->getParent()].emplace_back(B);

  for (auto &Pair : BarriersByBB)
    for (auto &BD : Pair.second)
      BarrierPtrs.push_back(&BD);

  bool Changed = false;
  // First remove 'at entry' and 'at exit' barriers if they fence nothing.
  Changed |= eliminateBoundaryBarriers(BarrierPtrs, BBMemInfo);
  // Then remove redundant barriers within a single basic block.
  for (auto &BarrierBBPair : BarriersByBB)
    Changed |= eliminateBackToBackInBB(BarrierBBPair.first,
                                       BarrierBBPair.second, BBMemInfo);

  // Refresh the list of barriers after back-to-back elimination.
  BarrierPtrs.clear();
  for (auto &Pair : BarriersByBB)
    for (auto &BD : Pair.second)
      BarrierPtrs.push_back(&BD);
  // TODO: hoist 2 barriers with the same predecessor BBs.

  // In the end eliminate or narrow barriers depending on DT and PDT analyses.
  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  PostDominatorTree &PDT = AM.getResult<PostDominatorTreeAnalysis>(F);

  Changed |= optimizeBarriersCFG(BarrierPtrs, DT, PDT, BBMemInfo);

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
