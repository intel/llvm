//=== SYCLOptimizeBackToBackBarrier.cpp - SYCL barrier optimization pass ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass cleans up back-to-back ControlBarrier calls.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLOptimizeBackToBackBarrier.h"

#include "llvm/IR/IRBuilder.h"

using namespace llvm;

namespace {

static constexpr char CONTROL_BARRIER[] = "_Z22__spirv_ControlBarrieriii";
static constexpr char ITT_BARRIER[] = "__itt_offload_wg_barrier_wrapper";
static constexpr char ITT_RESUME[] = "__itt_offload_wi_resume_wrapper";

// Known scopes in SPIR-V.
enum class Scope {
  CrossDevice = 0,
  Device = 1,
  Workgroup = 2,
  Subgroup = 3,
  Invocation = 4
};

enum class CompareRes { BIGGER = 0, SMALLER = 1, EQUAL = 2, UNKNOWN = 3 };

// This map is added in case of any future scopes are added to SPIR-V and/or
// SYCL.
const std::unordered_map<uint64_t, uint64_t> ScopeWeights = {
    {static_cast<uint64_t>(Scope::CrossDevice), 1000},
    {static_cast<uint64_t>(Scope::Device), 800},
    {static_cast<uint64_t>(Scope::Workgroup), 600},
    {static_cast<uint64_t>(Scope::Subgroup), 400},
    {static_cast<uint64_t>(Scope::Invocation), 10}};

inline CompareRes compareScopesWithWeights(const uint64_t LHS,
                                           const uint64_t RHS) {
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

// The function removes back-to-back ControlBarrier calls in case if they
// have the same memory scope and memory semantics arguments. When two
// back-to-back ControlBarriers are having different execution scope arguments -
// pick the one with the 'bigger' scope.
// It also cleans up ITT annotations surrounding the removed barrier call.
bool processControlBarrier(Function *F) {
  BasicBlock *PrevBB = nullptr;
  llvm::SmallPtrSet<Instruction *, 8> ToErase;
  for (auto I = F->user_begin(), E = F->user_end(); I != E;) {
    User *U = *I++;
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;

    // New basic block - new processing.
    BasicBlock *CurrentBB = CI->getParent();
    if (CurrentBB != PrevBB) {
      PrevBB = CurrentBB;
      continue;
    }

    llvm::SmallPtrSet<Instruction *, 2> ToEraseLocalITT;
    BasicBlock::iterator It(CI);
    // Iterate over the basic block storing back-to-back barriers and their ITT
    // annotations into ToErase container.
    while (It != CurrentBB->begin()) {
      --It;
      auto *Cand = dyn_cast<CallInst>(&*It);
      if (!Cand)
        break;
      CallInst *CIToRemove = Cand;
      StringRef CandName = Cand->getCalledFunction()->getName();
      if (CandName == ITT_RESUME || CandName == ITT_BARRIER) {
        ToEraseLocalITT.insert(Cand);
        continue;
      } else if (CandName == CONTROL_BARRIER) {
        bool EqualOps = true;
        const auto *ExecutionScopeCI = CI->getOperand(0);
        const auto *ExecutionScopeCand = Cand->getOperand(0);
        if (ExecutionScopeCI != ExecutionScopeCand) {
          if (isa<ConstantInt>(ExecutionScopeCI) &&
              isa<ConstantInt>(ExecutionScopeCand)) {
            const auto ConstScopeCI =
                cast<ConstantInt>(ExecutionScopeCI)->getZExtValue();
            const auto ConstScopeCand =
                cast<ConstantInt>(ExecutionScopeCand)->getZExtValue();
            // Pick ControlBarrier with the 'bigger' execution scope.
            const auto Compare =
                compareScopesWithWeights(ConstScopeCI, ConstScopeCand);
            if (Compare == CompareRes::SMALLER)
              CIToRemove = CI;
            else if (Compare == CompareRes::UNKNOWN)
              // Unknown scopes = unknown rules. Keep ControlBarrier call.
              EqualOps = false;
          } else
            EqualOps = false;
        }
        // TODO: may be handle a case with not-matching memory scope and
        // memory semantic arguments in a smart way.
        for (unsigned I = 1; I != CI->getNumOperands(); ++I) {
          if (CI->getOperand(I) != Cand->getOperand(I)) {
            EqualOps = false;
            break;
          }
        }
        if (EqualOps) {
          ToErase.insert(CIToRemove);
          for (auto *ITT : ToEraseLocalITT)
            ToErase.insert(ITT);
          ToEraseLocalITT.clear();
        }
      }
    }
  }

  if (ToErase.empty())
    return false;

  for (auto *I : ToErase) {
    I->dropAllReferences();
    I->eraseFromParent();
  }

  return true;
}

} // namespace

PreservedAnalyses
SYCLOptimizeBackToBackBarrierPass::run(Module &M, ModuleAnalysisManager &MAM) {
  bool ModuleChanged = false;
  for (Function &F : M)
    if (F.isDeclaration())
      if (F.getName() == CONTROL_BARRIER)
        ModuleChanged |= processControlBarrier(&F);

  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
