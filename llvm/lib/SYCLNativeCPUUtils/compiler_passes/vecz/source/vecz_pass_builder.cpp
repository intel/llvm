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

#include "vecz_pass_builder.h"

#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/DominanceFrontier.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/MemoryDependenceAnalysis.h>
#include <llvm/Analysis/MemorySSA.h>
#include <llvm/Analysis/PhiValues.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/PassManagerImpl.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/ADCE.h>
#include <llvm/Transforms/Scalar/DCE.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/FlattenCFG.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/IndVarSimplify.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Scalar/Sink.h>
#include <llvm/Transforms/Utils/BreakCriticalEdges.h>
#include <llvm/Transforms/Utils/FixIrreducible.h>
#include <llvm/Transforms/Utils/LowerSwitch.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>
#include <llvm/Transforms/Utils/UnifyFunctionExitNodes.h>

#include <cassert>

#include "analysis/control_flow_analysis.h"
#include "analysis/divergence_analysis.h"
#include "analysis/liveness_analysis.h"
#include "analysis/packetization_analysis.h"
#include "analysis/simd_width_analysis.h"
#include "analysis/stride_analysis.h"
#include "analysis/uniform_value_analysis.h"
#include "analysis/vectorizable_function_analysis.h"
#include "analysis/vectorization_unit_analysis.h"
#include "debugging.h"
#include "transform/common_gep_elimination_pass.h"
#include "transform/control_flow_conversion_pass.h"
#include "transform/inline_post_vectorization_pass.h"
#include "transform/interleaved_group_combine_pass.h"
#include "transform/packetization_helpers.h"
#include "transform/packetization_pass.h"
#include "transform/passes.h"
#include "transform/scalarization_pass.h"
#include "transform/ternary_transform_pass.h"

#define DEBUG_TYPE "vecz"
using namespace llvm;
using namespace vecz;

VeczPassMachinery::VeczPassMachinery(
    llvm::LLVMContext &llvmCtx, llvm::TargetMachine *TM,
    VectorizationContext &Ctx, bool verifyEach,
    compiler::utils::DebugLogging debugLogLevel)
    : compiler::utils::PassMachinery(llvmCtx, TM, verifyEach, debugLogLevel),
      Ctx(Ctx) {}

void VeczPassMachinery::registerPasses() {
  // Register standard passes
  compiler::utils::PassMachinery::registerPasses();

  FAM.registerPass([&] { return VectorizationContextAnalysis(Ctx); });
  FAM.registerPass([&] { return VectorizationUnitAnalysis(Ctx); });
  FAM.registerPass([&] { return VectorizableFunctionAnalysis(); });
  FAM.registerPass([] { return StrideAnalysis(); });
  FAM.registerPass([] { return UniformValueAnalysis(); });
  FAM.registerPass([] { return LivenessAnalysis(); });
  FAM.registerPass([] { return PacketizationAnalysis(); });
  FAM.registerPass([] { return CFGAnalysis(); });
  FAM.registerPass([] { return DivergenceAnalysis(); });

  if (!TM) {
    FAM.registerPass([] { return TargetIRAnalysis(); });
  } else {
    FAM.registerPass(
        [this] { return TargetIRAnalysis(TM->getTargetIRAnalysis()); });
    FAM.registerPass([] { return SimdWidthAnalysis(); });
  }
}

void VeczPassMachinery::addClassToPassNames() {
  {
#define MODULE_PASS(NAME, CREATE_PASS) \
  PIC.addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#define FUNCTION_PASS(NAME, CREATE_PASS) \
  PIC.addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#define LOOP_PASS(NAME, CREATE_PASS) \
  PIC.addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#include "passes.def"
  }

  // Register a callback which skips all passes once we've failed to vectorize
  // a function.
  PIC.registerShouldRunOptionalPassCallback([&](StringRef, llvm::Any IR) {
#if LLVM_VERSION_GREATER_EQUAL(16, 0)
    const Function **FPtr = any_cast<const Function *>(&IR);
    const Function *F = FPtr ? *FPtr : nullptr;
    if (!F) {
      if (const auto **L = any_cast<const Loop *>(&IR)) {
        F = (*L)->getHeader()->getParent();
      } else {
        // Always run module passes
        return true;
      }
    }
#else
    const Function *F = nullptr;
    if (any_isa<const Function *>(IR)) {
      F = any_cast<const Function *>(IR);
    } else if (any_isa<const Loop *>(IR)) {
      F = any_cast<const Loop *>(IR)->getHeader()->getParent();
    } else {
      // Always run module passes
      return true;
    }
#endif
    // FIXME: This is repeating the job of the VectorizationUnitAnalysis.
    // We should track 'failure' more directly in the
    // Function/VectorizationContext?
    auto const *const VU = Ctx.getActiveVU(F);
    if (!VU) {
      // Don't run on anything without a VU since it's not currently being
      // vectorized.
      return false;
    }
    return !VU->failed();
  });
}

void VeczPassMachinery::registerPassCallbacks() {
  // Add a backwards-compatible way of supporting simplifycfg, which used
  // to be called simplify-cfg before LLVM 12.
  PB.registerPipelineParsingCallback(
      [](StringRef Name, ModulePassManager &PM,
         ArrayRef<PassBuilder::PipelineElement>) {
#define MODULE_PASS(NAME, CREATE_PASS) \
  if (Name == NAME) {                  \
    PM.addPass(CREATE_PASS);           \
    return true;                       \
  }
#define FUNCTION_PASS(NAME, CREATE_PASS)                        \
  if (Name == NAME) {                                           \
    PM.addPass(createModuleToFunctionPassAdaptor(CREATE_PASS)); \
    return true;                                                \
  }
#define LOOP_PASS(NAME, CREATE_PASS)                    \
  if (Name == NAME) {                                   \
    PM.addPass(createModuleToFunctionPassAdaptor(       \
        createFunctionToLoopPassAdaptor(CREATE_PASS))); \
    return true;                                        \
  }
#include "passes.def"
        return false;
      });
}

bool vecz::buildPassPipeline(ModulePassManager &PM) {
  // Preparation passes
  PM.addPass(BuiltinInliningPass());
  // Lower switches after builtin inlining, incase the builtins had switches.
  PM.addPass(createModuleToFunctionPassAdaptor(LowerSwitchPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(FixIrreduciblePass()));

  // We have to run LLVM's Mem2Reg pass in case the front end didn't
  PM.addPass(createModuleToFunctionPassAdaptor(PromotePass()));
  // LLVM's own Mem2Reg pass doesn't always get everything
  PM.addPass(createModuleToFunctionPassAdaptor(BasicMem2RegPass()));

  PM.addPass(createModuleToFunctionPassAdaptor(InstCombinePass()));
  PM.addPass(createModuleToFunctionPassAdaptor(AggressiveInstCombinePass()));
  PM.addPass(createModuleToFunctionPassAdaptor(DCEPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(PreLinearizePass()));
  // If pre-linearization created any unnecessary Hoist Guards,
  // Instruction Combining Pass will handily clean them up.
  PM.addPass(createModuleToFunctionPassAdaptor(InstCombinePass()));
  PM.addPass(createModuleToFunctionPassAdaptor(SimplifyCFGPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(DCEPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(UnifyFunctionExitNodesPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(LoopSimplifyPass()));
  // Lower switches again because CFG simplifcation can create them.
  PM.addPass(createModuleToFunctionPassAdaptor(LowerSwitchPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(
      createFunctionToLoopPassAdaptor(VeczLoopRotatePass())));
  // IndVarSimplify can create a lot of duplicate instructions when there
  // are unrolled loops. EarlyCSE is there to clear them up. However,
  // this can destroy LCSSA form, so we need to restore it.
  PM.addPass(createModuleToFunctionPassAdaptor(
      createFunctionToLoopPassAdaptor(IndVarSimplifyPass())));

  PM.addPass(createModuleToFunctionPassAdaptor(EarlyCSEPass()));
  // We run this last because EarlyCSE can actually create infinite loops
  // (with a "conditional" branch on true)
  PM.addPass(createModuleToFunctionPassAdaptor(
      createFunctionToLoopPassAdaptor(SimplifyInfiniteLoopPass())));

  // Verify that the preparation passes cleaned up after themselves.
  PM.addPass(VerifierPass());

  PM.addPass(createModuleToFunctionPassAdaptor(RemoveIntPtrPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(SquashSmallVectorsPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(UniformReassociationPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(TernaryTransformPass()));

  PM.addPass(createModuleToFunctionPassAdaptor(BreakCriticalEdgesPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(LCSSAPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(ControlFlowConversionPass()));
  PM.addPass(VerifierPass());
  PM.addPass(createModuleToFunctionPassAdaptor(DivergenceCleanupPass()));

  PM.addPass(createModuleToFunctionPassAdaptor(CommonGEPEliminationPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(ScalarizationPass()));

  PM.addPass(createModuleToFunctionPassAdaptor(AggressiveInstCombinePass()));
  PM.addPass(createModuleToFunctionPassAdaptor(ADCEPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(SimplifyCFGPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(SimplifyMaskedMemOpsPass()));

  // Having multiple GEP instructions that perform the same operation
  // greatly amplifies the code generated by the packetizer as it duplicates
  // the amount of extractelement instructions, so we want to remove what
  // is unnecessary.
  PM.addPass(createModuleToFunctionPassAdaptor(CommonGEPEliminationPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(PacketizationPass()));

  PM.addPass(createModuleToFunctionPassAdaptor(InlinePostVectorizationPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(FlattenCFGPass()));
  PM.addPass(
      createModuleToFunctionPassAdaptor(GVNPass(GVNOptions().setMemDep(true))));
  PM.addPass(createModuleToFunctionPassAdaptor(AggressiveInstCombinePass()));
  PM.addPass(createModuleToFunctionPassAdaptor(ADCEPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(SinkingPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(SimplifyCFGPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(AggressiveInstCombinePass()));

  PM.addPass(createModuleToFunctionPassAdaptor(
      InterleavedGroupCombinePass(eInterleavedStore)));
  PM.addPass(createModuleToFunctionPassAdaptor(
      InterleavedGroupCombinePass(eInterleavedLoad)));
  PM.addPass(createModuleToFunctionPassAdaptor(InstCombinePass()));
  PM.addPass(createModuleToFunctionPassAdaptor(DCEPass()));
  PM.addPass(createModuleToFunctionPassAdaptor(SimplifyMaskedMemOpsPass()));
  PM.addPass(DefineInternalBuiltinsPass());

  PM.addPass(VerifierPass());

  return true;
}
