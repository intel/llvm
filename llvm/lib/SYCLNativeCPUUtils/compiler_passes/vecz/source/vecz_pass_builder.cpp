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
#include <llvm/IR/PassManager.h>
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
#include <llvm/Transforms/Scalar/LoopPassManager.h>
#include <llvm/Transforms/Scalar/SROA.h>
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
#include "multi_llvm/llvm_version.h"
#include "transform/common_gep_elimination_pass.h"
#include "transform/control_flow_conversion_pass.h"
#include "transform/inline_post_vectorization_pass.h"
#include "transform/interleaved_group_combine_pass.h"
#include "transform/packetization_helpers.h"
#include "transform/packetization_pass.h"
#include "transform/passes.h"
#include "transform/scalarization_pass.h"
#include "transform/ternary_transform_pass.h"

#if LLVM_VERSION_GREATER_EQUAL(18, 0)
#include <llvm/Transforms/Scalar/InferAlignment.h>
#endif

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
    // FIXME: This is repeating the job of the VectorizationUnitAnalysis.
    // We should track 'failure' more directly in the
    // Function/VectorizationContext?
    const auto *const VU = Ctx.getActiveVU(F);
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
  {
    FunctionPassManager FPM;
    // Lower switches after builtin inlining, in case the builtins had switches.
    FPM.addPass(LowerSwitchPass());
    FPM.addPass(FixIrreduciblePass());

    // It's helpful to run SROA in case it opens up more opportunities to
    // eliminate aggregates in (particularly SYCL) kernels. This is especially
    // true after inlining - which we've (usually) just performed in the
    // BuiltinInliningPass - because otherwise SROA is unable to analyze the
    // lifetime of allocas due to them being "escaped" by the function callee.
    FPM.addPass(SROAPass(SROAOptions::ModifyCFG));
    // We have to run LLVM's Mem2Reg pass in case the front end didn't. Note
    // that SROA usually runs Mem2Reg internally (unless disabled via a
    // command-line option) though using its own heuristic. We run it
    // unconditionally regardless, just for good measure.
    FPM.addPass(PromotePass());
    // LLVM's own Mem2Reg pass doesn't always get everything
    FPM.addPass(BasicMem2RegPass());

    FPM.addPass(InstCombinePass());
    FPM.addPass(AggressiveInstCombinePass());
    FPM.addPass(DCEPass());
    FPM.addPass(PreLinearizePass());
    // If pre-linearization created any unnecessary Hoist Guards,
    // Instruction Combining Pass will handily clean them up.
    FPM.addPass(InstCombinePass());
    FPM.addPass(SimplifyCFGPass());
    FPM.addPass(DCEPass());
    FPM.addPass(UnifyFunctionExitNodesPass());
    FPM.addPass(LoopSimplifyPass());
    // Lower switches again because CFG simplifcation can create them.
    FPM.addPass(LowerSwitchPass());
    {
      LoopPassManager LPM;
      LPM.addPass(VeczLoopRotatePass());
      // IndVarSimplify can create a lot of duplicate instructions when there
      // are unrolled loops. EarlyCSE is there to clear them up. However,
      // this can destroy LCSSA form, so we need to restore it.
      LPM.addPass(IndVarSimplifyPass());
      FPM.addPass(createFunctionToLoopPassAdaptor(std::move(LPM)));
    }

    FPM.addPass(EarlyCSEPass());
    // We run this last because EarlyCSE can actually create infinite loops
    // (with a "conditional" branch on true)
    FPM.addPass(createFunctionToLoopPassAdaptor(SimplifyInfiniteLoopPass()));

    FPM.addPass(RemoveIntPtrPass());
    FPM.addPass(SquashSmallVectorsPass());
    FPM.addPass(UniformReassociationPass());
    FPM.addPass(TernaryTransformPass());

    FPM.addPass(BreakCriticalEdgesPass());
    FPM.addPass(LCSSAPass());
    FPM.addPass(ControlFlowConversionPass());

    PM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }

  // Verify that the preparation passes (particularly control-flow conversion)
  // have left the module in a correct state.
  PM.addPass(VerifierPass());

  {
    FunctionPassManager FPM;

    FPM.addPass(DivergenceCleanupPass());

    FPM.addPass(CommonGEPEliminationPass());
    FPM.addPass(ScalarizationPass());

    FPM.addPass(AggressiveInstCombinePass());
    FPM.addPass(ADCEPass());
    FPM.addPass(SimplifyCFGPass());
    FPM.addPass(SimplifyMaskedMemOpsPass());

    // Having multiple GEP instructions that perform the same operation
    // greatly amplifies the code generated by the packetizer as it duplicates
    // the amount of extractelement instructions, so we want to remove what
    // is unnecessary.
    FPM.addPass(CommonGEPEliminationPass());

    // The packetizer - the 'main' bit of the vectorization process.
    FPM.addPass(PacketizationPass());

    FPM.addPass(InlinePostVectorizationPass());
    FPM.addPass(FlattenCFGPass());
    FPM.addPass(GVNPass(GVNOptions().setMemDep(true)));
    FPM.addPass(AggressiveInstCombinePass());
    FPM.addPass(ADCEPass());
    FPM.addPass(SinkingPass());
    FPM.addPass(SimplifyCFGPass());
    FPM.addPass(AggressiveInstCombinePass());

    FPM.addPass(InterleavedGroupCombinePass(eInterleavedStore));
    FPM.addPass(InterleavedGroupCombinePass(eInterleavedLoad));
    FPM.addPass(InstCombinePass());
#if LLVM_VERSION_GREATER_EQUAL(18, 0)
    // LLVM 18 split this pass out of InstCombine
    FPM.addPass(InferAlignmentPass());
#endif
    FPM.addPass(DCEPass());
    FPM.addPass(SimplifyMaskedMemOpsPass());

    PM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }

  PM.addPass(DefineInternalBuiltinsPass());
  PM.addPass(VerifierPass());

  return true;
}
