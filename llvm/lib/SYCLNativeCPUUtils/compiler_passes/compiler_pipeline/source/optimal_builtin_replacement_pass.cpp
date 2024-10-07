// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This pass replaces builtin functions with optimal equivalents.

#include <compiler/utils/builtin_info.h>
#include <compiler/utils/optimal_builtin_replacement_pass.h>
#include <llvm/ADT/PriorityWorklist.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/TargetParser/Triple.h>
#include <multi_llvm/vector_type_helper.h>

#define DEBUG_TYPE "ca-optimal-builtins"

using namespace llvm;

namespace {

void removeCallSite(CallBase &CB, LazyCallGraph &CG) {
  Function *Caller = CB.getCaller();
  Function *Callee = CB.getCaller();
  auto CallerNode = CG.get(*Caller);
  auto CalleeNode = CG.get(*Callee);
  if (auto *CallerRef = CG.lookupRefSCC(CallerNode)) {
    CallerRef->removeOutgoingEdge(CallerNode, CalleeNode);
  }
}

}  // namespace

namespace compiler {
namespace utils {

Value *OptimalBuiltinReplacementPass::replaceAbacusCLZ(
    CallBase &CB, StringRef BaseName, const SmallVectorImpl<Type *> &,
    const SmallVectorImpl<TypeQualifiers> &) {
  if (BaseName != "__abacus_clz") {
    return nullptr;
  }
  Module *M = CB.getModule();
  SmallVector<Value *, 4> Args(CB.args());
  // Get the declaration for the intrinsic
  auto *const ArgTy = Args[0]->getType();
  auto *const Intrinsic = Intrinsic::getDeclaration(M, Intrinsic::ctlz, ArgTy);
  // If we didn't find the intrinsic or the return type isn't what we
  // expect, skip this optimization
  Function *Callee = CB.getCalledFunction();
  assert(Callee);
  if (!Intrinsic || Intrinsic->getReturnType() != Callee->getReturnType()) {
    return nullptr;
  }

  // On 32-bit ARM, the llvm.ctlz intrinsic on 64-bit types is expanded using
  // compiler-rt. Without online linking, we can't support that.
  const Triple TT(CB.getModule()->getTargetTriple());
  if (TT.getArch() == Triple::arm && ArgTy->isIntOrIntVectorTy(64)) {
    return nullptr;
  }

  // LLVM's ctlz has a second argument to specify that zeroes in the first
  // argument produces a defined result.
  LLVMContext &Ctx = M->getContext();
  Args.push_back(ConstantInt::getFalse(Ctx));

  return CallInst::Create(Intrinsic, Args, "", &CB);
}

Value *OptimalBuiltinReplacementPass::replaceAbacusMulhi(
    CallBase &CB, StringRef BaseName, const SmallVectorImpl<Type *> &,
    const SmallVectorImpl<TypeQualifiers> &Quals) {
  if (BaseName != "__abacus_mul_hi") {
    return nullptr;
  }
  IRBuilder<> B(&CB);

  auto I = CB.arg_begin();
  Value *const LHS = *I++;
  Value *const RHS = *I++;

  const auto BitWidth = LHS->getType()->getScalarType()->getIntegerBitWidth();

  // Don't perform this optimization on 64-bit types as 128-bit types aren't
  // generally well supported.
  if (BitWidth == 64) {
    return nullptr;
  }

  unsigned VecWidth = 1;
  if (const auto *VecTy = dyn_cast<VectorType>(LHS->getType())) {
    VecWidth = multi_llvm::getVectorNumElements(VecTy);
  }

  Type *UpTy = B.getIntNTy(BitWidth * 2);
  if (VecWidth != 1) {
    UpTy = FixedVectorType::get(UpTy, VecWidth);
  }

  bool SrcIsSigned = false;
  for (unsigned i = 0, e = Quals[0].getCount(); i != e; i++) {
    if (Quals[0].at(i) == eTypeQualSignedInt) {
      SrcIsSigned = true;
      break;
    }
  }

  const auto CastOp = SrcIsSigned ? Instruction::SExt : Instruction::ZExt;

  auto *const UpLHS = B.CreateCast(CastOp, LHS, UpTy);
  auto *const UpRHS = B.CreateCast(CastOp, RHS, UpTy);

  auto *const Mul = B.CreateMul(UpLHS, UpRHS);

  Constant *ShiftAmt = B.getIntN(BitWidth * 2, BitWidth);
  if (VecWidth != 1) {
    ShiftAmt = ConstantDataVector::getSplat(VecWidth, ShiftAmt);
  }

  auto *const Shift = B.CreateAShr(Mul, ShiftAmt);

  return B.CreateTrunc(Shift, LHS->getType());
}

Value *OptimalBuiltinReplacementPass::replaceAbacusFMinFMax(
    CallBase &CB, StringRef BaseName, const SmallVectorImpl<Type *> &,
    const SmallVectorImpl<TypeQualifiers> &) {
  const bool IsFMin = BaseName == "__abacus_fmin";
  if (!IsFMin && BaseName != "__abacus_fmax") {
    return nullptr;
  }

  const Triple TT(CB.getModule()->getTargetTriple());
  // minnum/maxnum intrinsics fail CTS on arm targets. See
  // https://llvm.org/PR27363.
  if (TT.getArch() == Triple::arm || TT.getArch() == Triple::aarch64) {
    return nullptr;
  }

  IRBuilder<> B(&CB);

  auto I = CB.arg_begin();
  Value *LHS = *I++;
  Value *RHS = *I++;

  const auto *LHSTy = LHS->getType();
  const auto *RHSTy = RHS->getType();

  if (LHSTy->isVectorTy() != RHSTy->isVectorTy()) {
    auto VectorEC =
        multi_llvm::getVectorElementCount(LHSTy->isVectorTy() ? LHSTy : RHSTy);
    if (!LHS->getType()->isVectorTy()) {
      LHS = B.CreateVectorSplat(VectorEC, LHS);
    }
    if (!RHS->getType()->isVectorTy()) {
      RHS = B.CreateVectorSplat(VectorEC, RHS);
    }
  }
  return B.CreateBinaryIntrinsic(IsFMin ? Intrinsic::minnum : Intrinsic::maxnum,
                                 LHS, RHS);
}

OptimalBuiltinReplacementPass::OptimalBuiltinReplacementPass() {
  replacements.emplace_back(replaceAbacusCLZ);
  replacements.emplace_back(replaceAbacusMulhi);
  replacements.emplace_back(replaceAbacusFMinFMax);
}

Value *OptimalBuiltinReplacementPass::replaceBuiltinWithInlineIR(
    CallBase &CB) const {
  auto *M = CB.getModule();
  NameMangler mangler(&M->getContext());

  SmallVector<Type *, 4> Types;
  SmallVector<TypeQualifiers, 4> Quals;
  Function *Callee = CB.getCalledFunction();
  assert(Callee);
  const StringRef BaseName =
      mangler.demangleName(Callee->getName(), Types, Quals);

  for (const auto &replace_fn : replacements) {
    if (replace_fn) {
      if (auto *V = replace_fn(CB, BaseName, Types, Quals)) {
        return V;
      }
    }
  }

  return nullptr;
}

PreservedAnalyses OptimalBuiltinReplacementPass::run(LazyCallGraph::SCC &C,
                                                     CGSCCAnalysisManager &AM,
                                                     LazyCallGraph &CG,
                                                     CGSCCUpdateResult &) {
  // Without the possibility of recursion, we can expect all meaningful
  // OpenCL/ComputeMux programs to be contained within a single singular SCC
  // serving as the entry point. We use this as the root.
  if (C.size() != 1) {
    return PreservedAnalyses::all();
  }
  Module &M = *C.begin()->getFunction().getParent();

  // Check that at least one node in this graph is a kernel.
  if (none_of(C, [](const LazyCallGraph::Node &N) {
        return N.getFunction().getCallingConv() == CallingConv::SPIR_KERNEL;
      })) {
    return PreservedAnalyses::all();
  }

  const auto &MAMProxy = AM.getResult<ModuleAnalysisManagerCGSCCProxy>(C, CG);
  if (auto *BI = MAMProxy.getCachedResult<BuiltinInfoAnalysis>(M)) {
    replacements.emplace_back(
        [BI](CallBase &CB, StringRef, const SmallVectorImpl<Type *> &,
             const SmallVectorImpl<TypeQualifiers> &) -> Value * {
          Function *Callee = CB.getCalledFunction();
          const auto Props = BI->analyzeBuiltin(*Callee).properties;
          if (Props & eBuiltinPropertyCanEmitInline) {
            IRBuilder<> B(&CB);
            const SmallVector<Value *, 4> Args(CB.args());
            if (Value *Impl = BI->emitBuiltinInline(Callee, B, Args)) {
              assert(
                  Impl->getType() == CB.getType() &&
                  "The inlinined function type must match that of the original "
                  "function");
              return Impl;
            }
          }
          return nullptr;
        });
  }

  if (adjustReplacements) {
    adjustReplacements(replacements);
  }

  // If there are no replacements to run, for whatever reason, we can bail
  // early.
  if (replacements.empty()) {
    return PreservedAnalyses::all();
  }

  SmallVector<CallBase *, 4> ToDelete;
  // The SmallPriorityWorklist prioritises nodes which have been inserted
  // multiple times, and avoids duplication of already-inserted items, but
  // *not* ones already visited and popped off.
  SmallPriorityWorklist<LazyCallGraph::Node *, 4> Worklist;
  // Assuming we only have one node to begin with (see above), start off with
  // that.
  Worklist.insert(&*C.begin());
  // While the worklist above prevents re-insertion, we might end up visiting
  // the same function again after already visiting if popping it off the
  // worklist. So we still have to keep track of recursion.
  SmallPtrSet<LazyCallGraph::Node *, 4> Visited;

  // Now visit all nodes in this "root" graph in order. We will visit
  // outer-most functions (kernels) first before descending the call graph.
  // This gives precedence to "outer-most" replacements.
  while (!Worklist.empty()) {
    LazyCallGraph::Node *N = Worklist.pop_back_val();
    LLVM_DEBUG(dbgs() << "OptimalBuiltinReplacement: visiting " << *N << "\n");
    for (Instruction &I : instructions(N->getFunction())) {
      if (auto *CB = dyn_cast<CallBase>(&I)) {
        if (CB->getCalledFunction() && !isa<IntrinsicInst>(I)) {
          if (Value *New = replaceBuiltinWithInlineIR(*CB)) {
            LLVM_DEBUG(dbgs()
                       << "\tOptimalBuiltinReplacement: replacing call to "
                       << CB->getCalledFunction()->getName() << "\n");
            ToDelete.push_back(CB);
            removeCallSite(*CB, CG);
            // Assume that replacements don't introduce new calls, and we can
            // simply mark this one as gone and move on.
            CB->replaceAllUsesWith(New);
          } else if (auto *CalledN = CG.lookup(*CB->getCalledFunction())) {
            if (Visited.insert(CalledN).second) {
              Worklist.insert(CalledN);
            }
          }
        }
      }
    }
  }

  const bool Modified = !ToDelete.empty();

  // Clean up any dead calls.
  while (!ToDelete.empty()) {
    Instruction *I = ToDelete.pop_back_val();
    I->eraseFromParent();
  }

  return Modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
}  // namespace utils
}  // namespace compiler
