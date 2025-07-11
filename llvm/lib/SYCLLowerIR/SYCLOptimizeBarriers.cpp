//==== SYCLOptimizeBarriers.cpp - SYCL barrier optimization pass ====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
//      loads from __spirv_BuiltIn GVs) – Unknown  : any other
//      mayReadOrWriteMemory() (intrinsics, calls, addrspace generic)
//
// 2) **At Entry and At Exit Elimination**
//    - **Entry**: For each barrier B, if on *every* path from function entry to
//    B there are no
//      accesses >= B.MemScope, then remove B.
//    - **Exit** : For each barrier B, if on *every* path from B to any function
//    return there are no
//      accesses >= B.MemScope, then remove B.
//
// 3) **Back-to-Back Elimination (per-BB)**
//    a) *Pure-Sync Collapse*
//       If BB summary == None (no local/global/unknown accesses):
//         – Find the single barrier with the *widest* (ExecScope, MemScope)
//         (ignore Unknown).
//         – Erase all other barriers (they synchronize
//         nothing).
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
//       intervening
//         code contains nothing that only the earlier barrier needed, the
//         earlier barrier is removed.
//    Any barrier whose execution or memory scope is Unknown is kept
//    conservatively. After a single pass every basic block contains only the
//    minimal set of barriers required to enforce ordering for the memory
//    operations it actually performs.
//
// 4) **CFG-Wide Elimination**
//    a) *Dominator-Based Removal*
//       For each pair (A, B) with identical Exec and Mem scopes where A
//       dominates B:
//         – If *every* path from A to B has no accesses >= A.MemScope, remove
//         B.
//    b) *Post-Dominator-Based Removal*
//       For each pair (A, B) with identical scopes where B post-dominates A:
//         – If *every* path from A to B has no accesses >= A.MemScope, remove
//         A.
//
// 5) **Global -> Local Downgrade**
//    For each global-scope barrier B (MemScope == Device/CrossDevice or
//    CrossWorkgroupMemory semantics):
//      – If there exists another global barrier A that dominates or
//        post-dominates B and no Global/Unknown accesses occur between the two,
//        B can be downgraded to Workgroup scope.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLOptimizeBarriers.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"

#include <algorithm>

using namespace llvm;

namespace {

// Hard-coded special names used in the pass.
// TODO: add MemoryBarrier.
static constexpr char CONTROL_BARRIER[] = "_Z22__spirv_ControlBarrieriii";
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

enum class MemorySemantics {
  SubgroupMemory = 0x80,
  WorkgroupMemory = 0x100,
  CrossWorkgroupMemory = 0x200
};

inline CompareRes compareScopesWithWeights(const Scope LHS, const Scope RHS) {
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
using BarriersMap = DenseMap<BasicBlock *, SmallVector<BarrierDesc, 2>>;

// Map SPIR-V Barrier Scope to the RegionMemScope that a barrier of that kind
// actually fences.
static RegionMemScope getBarrierFencedScope(const Scope BarrierScope) {
  switch (BarrierScope) {
  case Scope::Invocation:
    // 'Invocation' fences nothing but itself — treat them as None.
    return RegionMemScope::None;
  case Scope::Workgroup:
  case Scope::Subgroup:
    // Workgroup and Subgroup barriers orders local memory.
    return RegionMemScope::Local;
  case Scope::Device:
  case Scope::CrossDevice:
    // Orders cross-workgroup/device memory (global).
    return RegionMemScope::Global;
  default:
    return RegionMemScope::Unknown;
  }
}

// Classify a single instruction’s memory scope. Used to set/update memory
// scope of a basic block.
static RegionMemScope classifyMemScope(Instruction *I) {
  if (CallInst *CI = dyn_cast<CallInst>(I)) {
    if (Function *F = CI->getCalledFunction()) {
      if (F->getName() == CONTROL_BARRIER || F->getName() == ITT_BARRIER ||
          F->getName() == ITT_RESUME)
        return RegionMemScope::None;
    }
  }
  // If it doesn’t read or write, it doesn't affect the region memory scope.
  if (!I->mayReadOrWriteMemory())
    return RegionMemScope::None;

  auto resolveGeneric = [&](Value *Pointer) -> RegionMemScope {
    // If generic pointer originates from an alloca instruction within a
    // function - it's safe to assume, that it's a private allocation.
    // FIXME: use more comprehensive analysis.
    Value *Cand = Pointer->stripInBoundsConstantOffsets();
    if (isa<AllocaInst>(Cand))
      return RegionMemScope::None;
    return RegionMemScope::Unknown;
  };

  auto getScopeForPtr = [&](Value *Ptr, unsigned AS) -> RegionMemScope {
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

  // Check for memory instructions. Currently handled: load/store/memory
  // intrinsics.
  // TODO: check for other intrinsics and SPIR-V friendly function calls.
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
  return RegionMemScope::Unknown;
}

// Scan the function and build:
// 1. a list of all BarrierDesc‘s
// 2. a per-BB memory-scope summary
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

        StringRef Name = Callee->getName();
        if (Name == CONTROL_BARRIER) {
          auto getConst = [&](uint32_t idx) -> uint32_t {
            if (auto *C = dyn_cast<ConstantInt>(CI->getArgOperand(idx)))
              return C->getZExtValue();
            return static_cast<uint32_t>(Scope::Unknown);
          };
          BarrierDesc BD = {CI, static_cast<Scope>(getConst(0)),
                            static_cast<Scope>(getConst(1)), getConst(2)};
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

// True if no fenced accesses of MemScope appear in [A->next, B).
static bool noFencedMemAccessesBetween(CallInst *A, CallInst *B,
                                       RegionMemScope Required,
                                       BBMemInfoMap &BBMemInfo) {
  RegionMemScope BBMemScope = BBMemInfo[A->getParent()];
  if (BBMemScope == RegionMemScope::Unknown ||
      Required == RegionMemScope::Unknown)
    return false;
  if (BBMemScope == RegionMemScope::None)
    return true;
  for (auto It = ++BasicBlock::iterator(A), End = BasicBlock::iterator(B);
       It != End; ++It) {
    auto InstScope = classifyMemScope(&*It);
    if (InstScope == RegionMemScope::Unknown || InstScope >= Required)
      return false;
  }
  return true;
}

// Helper to check if a whole block (or a slice) contains accesses fenced by
// 'Required'.
static bool hasFencedAccesses(BasicBlock *BB, RegionMemScope Required,
                              Instruction *Start = nullptr,
                              Instruction *End = nullptr) {
  auto It = Start ? std::next(BasicBlock::iterator(Start)) : BB->begin();
  auto Finish = End ? BasicBlock::iterator(End) : BB->end();
  for (; It != Finish; ++It) {
    RegionMemScope S = classifyMemScope(&*It);
    if (S == RegionMemScope::Unknown || S >= Required)
      return true;
  }
  return false;
}

// Check across basic blocks that no accesses of Required scope happen on any
// path from A to B. A must dominate B.
static bool noFencedAccessesCFG(CallInst *A, CallInst *B,
                                RegionMemScope Required,
                                BBMemInfoMap &BBMemInfo) {
  if (Required == RegionMemScope::Unknown)
    return false;

  if (A->getParent() == B->getParent())
    return noFencedMemAccessesBetween(A, B, Required, BBMemInfo);

  SmallVector<std::pair<BasicBlock *, Instruction *>, 8> Worklist;
  SmallPtrSet<BasicBlock *, 16> Visited;

  Worklist.emplace_back(A->getParent(), A);
  Visited.insert(A->getParent());

  while (!Worklist.empty()) {
    auto [BB, StartInst] = Worklist.pop_back_val();

    if (BB == B->getParent()) {
      if (hasFencedAccesses(BB, Required, StartInst, B))
        return false;
      continue;
    }

    if (hasFencedAccesses(BB, Required, StartInst, nullptr))
      return false;

    for (BasicBlock *Succ : successors(BB))
      if (Visited.insert(Succ).second)
        Worklist.emplace_back(Succ, nullptr);
  }

  return true;
}

// The back-to-back elimination on one BB.
static bool eliminateBackToBackInBB(BasicBlock *BB,
                                    SmallVectorImpl<BarrierDesc> &Barriers,
                                    BBMemInfoMap &BBMemInfo) {
  SmallVector<BarrierDesc, 8> Survivors;
  bool Changed = false;
  RegionMemScope BlockScope = BB ? BBMemInfo[BB] : RegionMemScope::Unknown;

  // If there are no memory accesses requiring synchronization in this block,
  // collapse all barriers to the single largest one.
  if (BlockScope == RegionMemScope::None) {
    bool HasUnknown = llvm::any_of(Barriers, [](const BarrierDesc &BD) {
      return BD.ExecScope == Scope::Unknown || BD.MemScope == Scope::Unknown;
    });
    if (!HasUnknown) {
      // Pick the barrier with the widest scope.
      auto Best = std::max_element(
          Barriers.begin(), Barriers.end(),
          [](const BarrierDesc &A, const BarrierDesc &B) {
            auto CmpExec = compareScopesWithWeights(B.ExecScope, A.ExecScope);
            auto CmpMem = compareScopesWithWeights(B.MemScope, A.MemScope);
            return (CmpExec == CompareRes::BIGGER ||
                    (CmpExec == CompareRes::EQUAL &&
                     CmpMem == CompareRes::BIGGER)) ||
                   (CmpMem == CompareRes::BIGGER);
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
      // Must share semantics to guess.
      // TODO: actually allow semantics missmatch for barriers removal for
      // several cases.
      if (Last.Semantic != Cur.Semantic)
        break;

      auto CmpExec = compareScopesWithWeights(Last.ExecScope, Cur.ExecScope);
      auto CmpMem = compareScopesWithWeights(Last.MemScope, Cur.MemScope);
      RegionMemScope FenceLast = getBarrierFencedScope(Last.MemScope);
      RegionMemScope FenceCur = getBarrierFencedScope(Cur.MemScope);

      if (CmpExec == CompareRes::UNKNOWN || CmpMem == CompareRes::UNKNOWN ||
          FenceLast == RegionMemScope::Unknown ||
          FenceCur == RegionMemScope::Unknown)
        break;

      // If identical then drop Cur.
      if (CmpExec == CompareRes::EQUAL && CmpMem == CompareRes::EQUAL) {
        if (noFencedMemAccessesBetween(Last.CI, Cur.CI, FenceLast, BBMemInfo)) {
          Changed |= eraseBarrierWithITT(Cur);
        }
        break;
      }
      // If Last wider then drop Cur.
      if ((CmpExec == CompareRes::BIGGER || CmpMem == CompareRes::BIGGER) &&
          noFencedMemAccessesBetween(Last.CI, Cur.CI, FenceCur, BBMemInfo)) {
        Changed |= eraseBarrierWithITT(Cur);
        break;
      }
      // If Cur wider then drop Last and retry.
      if ((CmpExec == CompareRes::SMALLER || CmpMem == CompareRes::SMALLER) &&
          noFencedMemAccessesBetween(Last.CI, Cur.CI, FenceLast, BBMemInfo)) {
        Changed |= eraseBarrierWithITT(Last);
        Survivors.pop_back();
        continue;
      }
      // No elimination possible.
      break;
    }
    if (Cur.CI) // still alive?
      Survivors.push_back(Cur);
  }

  // If we removed any, replace Barriers with the survivors
  if (Survivors.size() != Barriers.size()) {
    Barriers.clear();
    Barriers.append(Survivors.begin(), Survivors.end());
    Changed = true;
  }
  return Changed;
}

// Remove barriers that are redundant in the CFG based on dominance relations.
static bool eliminateDominatedBarriers(SmallVectorImpl<BarrierDesc *> &Barriers,
                                       DominatorTree &DT,
                                       PostDominatorTree &PDT,
                                       BBMemInfoMap &BBMemInfo) {
  bool Changed = false;
  for (auto *B1 : Barriers) {
    if (!B1->CI)
      continue;
    for (auto *B2 : Barriers) {
      // Check if the barrier was already removed.
      if (B1 == B2 || !B2->CI)
        continue;

      // Skip barriers with missmatching Semantic, Scopes or Unknown Scope.
      if (B1->Semantic != B2->Semantic)
        continue;
      if (B1->ExecScope != B2->ExecScope || B1->MemScope != B2->MemScope)
        continue;
      if (B1->ExecScope == Scope::Unknown || B1->MemScope == Scope::Unknown)
        continue;

      RegionMemScope Fence = getBarrierFencedScope(B1->MemScope);
      if (Fence == RegionMemScope::Unknown)
        continue;

      if (DT.dominates(B1->CI, B2->CI)) {
        if (noFencedAccessesCFG(B1->CI, B2->CI, Fence, BBMemInfo))
          Changed |= eraseBarrierWithITT(*B2);
      } else if (PDT.dominates(B1->CI->getParent(), B2->CI->getParent())) {
        if (noFencedAccessesCFG(B2->CI, B1->CI, Fence, BBMemInfo))
          Changed |= eraseBarrierWithITT(*B2);
      }
    }
  }
  return Changed;
}

// Downgrade global barriers to workgroup when no global memory is touched
// before the next global barrier.
static bool downgradeGlobalBarriers(SmallVectorImpl<BarrierDesc *> &Barriers,
                                    DominatorTree &DT, PostDominatorTree &PDT,
                                    BBMemInfoMap &BBMemInfo) {
  bool Changed = false;
  // Check for memory scope and Semantics to see, which memory is fenced.
  auto IsGlobalBarrier = [](const BarrierDesc &BD) {
    return BD.MemScope == Scope::Device || BD.MemScope == Scope::CrossDevice ||
           (BD.Semantic &
            static_cast<uint32_t>(MemorySemantics::CrossWorkgroupMemory));
  };

  for (auto *BPtr : Barriers) {
    BarrierDesc &B = *BPtr;
    if (!B.CI || !IsGlobalBarrier(B))
      continue;
    if (B.ExecScope == Scope::Unknown || B.MemScope == Scope::Unknown)
      continue;
    bool CanDowngrade = false;
    for (auto *APtr : Barriers) {
      if (APtr == BPtr)
        continue;
      BarrierDesc &A = *APtr;
      if (!A.CI || !IsGlobalBarrier(A))
        continue;
      // If no path from A to B contains global memory accesses - downgrade
      // the barrier.
      if (DT.dominates(A.CI, B.CI)) {
        if (noFencedAccessesCFG(A.CI, B.CI, RegionMemScope::Global,
                                BBMemInfo)) {
          CanDowngrade = true;
          break;
        }
      } else if (PDT.dominates(A.CI->getParent(), B.CI->getParent())) {
        if (noFencedAccessesCFG(B.CI, A.CI, RegionMemScope::Global,
                                BBMemInfo)) {
          CanDowngrade = true;
          break;
        }
      }
    }

    if (!CanDowngrade) {
      LLVMContext &Ctx = B.CI->getContext();
      Type *Int32Ty = Type::getInt32Ty(Ctx);
      uint32_t OldSem = B.Semantic;
      // Downgrade both scope and semantics.
      if (OldSem &
          static_cast<uint32_t>(MemorySemantics::CrossWorkgroupMemory)) {
        uint32_t NewSem =
            (OldSem &
             ~static_cast<uint32_t>(MemorySemantics::CrossWorkgroupMemory)) |
            static_cast<uint32_t>(MemorySemantics::WorkgroupMemory);
        B.CI->setArgOperand(2, ConstantInt::get(Int32Ty, NewSem));
        B.Semantic = NewSem;
      }
      B.CI->setArgOperand(1, ConstantInt::get(Int32Ty, static_cast<uint32_t>(
                                                           Scope::Workgroup)));
      B.MemScope = Scope::Workgroup;
      Changed = true;
    }
  }

  return Changed;
}

// True if BD is the first real instruction of the function.
static bool isAtKernelEntry(const BarrierDesc &BD) {
  BasicBlock &Entry = BD.CI->getFunction()->getEntryBlock();
  if (BD.CI->getParent() != &Entry)
    return false;

  for (Instruction &I : Entry) {
    if (&I == BD.CI)
      break;
    if (classifyMemScope(&I) != RegionMemScope::None)
      return false;
  }

  return true;
}

// True if BD is immediately before a return/unreachable and nothing follows.
static bool isAtKernelExit(const BarrierDesc &BD) {
  BasicBlock *BB = BD.CI->getParent();
  Instruction *Term = BB->getTerminator();
  if (!isa<ReturnInst>(Term) && !isa<UnreachableInst>(Term))
    return false;

  for (Instruction *I = BD.CI->getNextNode(); I && I != Term;
       I = I->getNextNode())
    if (classifyMemScope(I) != RegionMemScope::None)
      return false;

  return BD.CI->getNextNonDebugInstruction() == Term;
}

// Remove barriers that appear at the very beginning or end of a kernel
// function.
static bool
eliminateBoundaryBarriers(SmallVectorImpl<BarrierDesc *> &Barriers) {
  bool Changed = false;
  for (auto *BPtr : Barriers) {
    BarrierDesc &B = *BPtr;
    if (!B.CI)
      continue;
    // FIXME?: do we _really_ need this restriction? If yes - should it be
    // applied for other transformations done by the pass?
    if (B.CI->getFunction()->getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;
    if (isAtKernelEntry(B) || isAtKernelExit(B))
      Changed |= eraseBarrierWithITT(B);
  }
  return Changed;
}

} // namespace

PreservedAnalyses SYCLOptimizeBarriersPass::run(Function &F,
                                                FunctionAnalysisManager &AM) {
  SmallVector<BarrierDesc, 16> Barriers;
  BBMemInfoMap BBMemInfo;
  BarriersMap BarriersByBB;
  SmallVector<BarrierDesc *, 16> BarrierPtrs;

  // Analyse the function gathering barrier and memory scope of the region info.
  collectBarriersAndMemInfo(F, Barriers, BBMemInfo);
  for (auto &B : Barriers)
    BarriersByBB[B.CI->getParent()].push_back(B);

  for (auto &Pair : BarriersByBB)
    for (auto &BD : Pair.second)
      BarrierPtrs.push_back(&BD);

  bool Changed = false;
  // First remove 'at entry' and 'at exit' barriers if the fence nothing.
  Changed |= eliminateBoundaryBarriers(BarrierPtrs);
  // Then remove redundant barriers within a single basic block.
  for (auto &BarrierBBPair : BarriersByBB)
    Changed = eliminateBackToBackInBB(BarrierBBPair.first, BarrierBBPair.second,
                                      BBMemInfo);

  // In the end eliminate or narrow barriers depending on DT and PDT analyses.
  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  PostDominatorTree &PDT = AM.getResult<PostDominatorTreeAnalysis>(F);

  Changed |= eliminateDominatedBarriers(BarrierPtrs, DT, PDT, BBMemInfo);
  Changed |= downgradeGlobalBarriers(BarrierPtrs, DT, PDT, BBMemInfo);

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
