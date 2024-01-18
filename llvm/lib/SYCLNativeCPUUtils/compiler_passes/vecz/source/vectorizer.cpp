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

#include "vectorizer.h"

#include <compiler/utils/metadata.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <multi_llvm/vector_type_helper.h>

#include <memory>
#include <unordered_set>

#include "analysis/vectorizable_function_analysis.h"
#include "debugging.h"
#include "memory_operations.h"
#include "vectorization_context.h"
#include "vectorization_helpers.h"
#include "vectorization_heuristics.h"
#include "vectorization_unit.h"
#include "vecz/pass.h"
#include "vecz/vecz_choices.h"

#define DEBUG_TYPE "vecz"

using namespace vecz;
using namespace llvm;

namespace {
static cl::opt<bool> VeczDumpReport(
    "vecz-dump-report", cl::desc("report the post-vectorization status"));
// static cl options allow us to access these options from other cpp files,
// such as vectorization_unit.cpp

}  // namespace

// Statistics
STATISTIC(VeczSuccess, "Number of kernels successfully vectorized [ID#V80]");
STATISTIC(VeczFail, "Number of kernels that failed to vectorize [ID#V81]");
STATISTIC(VeczBail,
          "Number of kernels where vectorization was not attempted [ID#V82]");

STATISTIC(ScalarInstructions,
          "Number of instructions in the scalar kernel [ID#V00]");
STATISTIC(ScalarLoadStores,
          "Number of loads and stores in the scalar kernel [ID#V01]");
STATISTIC(ScalarVectorInsts,
          "Number of vector instructions in the scalar kernel [ID#V02]");
STATISTIC(ScalarMaxVectorWidth,
          "The width of the bigger vector instruction found in the scalar "
          "kernel [ID#V13]");
STATISTIC(VeczInstructions,
          "Number of instructions in the vectorized kernel [ID#V03]");
STATISTIC(VeczScalarInstructions,
          "Number of scalar instructions in the vectorized kernel [ID#V04]");
STATISTIC(VeczVectorInstructions,
          "Number of vector instructions in the vectorized kernel [ID#V05]");
STATISTIC(VeczInsertExtract,
          "Number of insert/extractelement instructions in the vectorized "
          "kernel [ID#V06]");
STATISTIC(VeczSplats,
          "Number of vector splats in the vectorized kernel [ID#V07]");
STATISTIC(
    VeczScalarMemOp,
    "Number of scalar loads and stores in the vectorized kernel [ID#V0A]");
STATISTIC(
    VeczVectorMemOp,
    "Number of vector loads and stores in the vectorized kernel [ID#V0B]");
STATISTIC(
    VeczMaskedMemOps,
    "Number of masked memory operations in the vectorized kernel [ID#V0C]");
STATISTIC(VeczInterleavedMemOps,
          "Number of interleaved memory operations in the vectorized kernel "
          "[ID#V0D]");
STATISTIC(VeczMaskedInterleavedMemOps,
          "Number of masked interleaved memory operations in the vectorized "
          "kernel [ID#V0E]");
STATISTIC(VeczScatterGatherMemOps,
          "Number of scatter/gather memory operations in the vectorized kernel "
          "[ID#V10]");
STATISTIC(VeczMaskedScatterGatherMemOps,
          "Number of masked scatter/gather operations in the vectorized "
          "kernel [ID#V11]");
STATISTIC(VeczVectorWidth, "Vector width of the vectorized kernel [ID#V12]");
STATISTIC(Ratio, "Normalized ratio of theoretical speedup[ID#V13]");

namespace {
/// @brief Calculate vectorization related statistics from the kernels
///
/// @param[in] VU The Vectorization Unit we are working on
/// @param[in] Scalar The scalar function that we have vectorized
/// @param[in] Vectorized The vectorized version of the scalar function
void collectStatistics(VectorizationUnit &VU, Function *Scalar,
                       Function *Vectorized) {
  // Do not gather statistics if we failed to vectorize, if we're doing
  // scalable vectorization, or if statistics aren't enabled in the first
  // place.
  if (!Scalar || !Vectorized || !AreStatisticsEnabled() ||
      VU.width().isScalable()) {
    return;
  }

  VeczVectorWidth = VU.width().getFixedValue();

  // Function to check if an instruction is a vector instruction or not
  auto isVectorInst = [](Instruction &I) -> bool {
    Type *Ty = I.getType();

    // Insert/extractelement are not really vector instructions
    if (isa<InsertElementInst>(I) || isa<ExtractElementInst>(I)) {
      return false;
    }
    // Instructions that return a vector
    if (isa<FixedVectorType>(Ty)) {
      return true;
    }
    // Store instructions that store a vector value
    if (StoreInst *SI = dyn_cast<StoreInst>(&I)) {
      auto *ValOp = SI->getValueOperand();
      assert(ValOp && "Could not get value operand");
      return isa<FixedVectorType>(ValOp->getType());
    }
    // Internal builtins that work on vectors. This is relevant for stores only,
    // as loads return a vector type and will be caught earlier on.
    if (CallInst *CI = dyn_cast<CallInst>(&I)) {
      if (auto Op = MemOp::get(CI)) {
        // With the exception of masked loads and stores, every other internal
        // builtin works with vectors
        if (!Op->isMaskedMemOp()) {
          return true;
        }
        // Masked loads are handled earlier on as they return a vector type.
        // We need to check if masked stores are storing vectors or not.
        if (Op->isStore() && isa<FixedVectorType>(Op->getDataType())) {
          return true;
        }
      }
    }

    return false;
  };

  unsigned MaxScalarVectorWidth = 0;
  // Collect the scalar kernel's statistics
  for (auto &BB : *Scalar) {
    for (auto &I : BB) {
      ++ScalarInstructions;
      ScalarLoadStores += (isa<LoadInst>(I) || isa<StoreInst>(I));
      ScalarVectorInsts += isVectorInst(I);
      // Find out how wide is the widest vector used in the scalar kernel
      if (auto *VecTy = dyn_cast<FixedVectorType>(I.getType())) {
        if (VecTy->getNumElements() > MaxScalarVectorWidth) {
          MaxScalarVectorWidth = VecTy->getNumElements();
        }
      }
    }
  }
  ScalarMaxVectorWidth = MaxScalarVectorWidth;

  // Collect the vectorized kernel's statistics
  for (auto &BB : *Vectorized) {
    for (auto &I : BB) {
      // Count instructions
      ++VeczInstructions;

      // Detect vector splats
      // Count insert/extractelement instructions
      if (isa<InsertElementInst>(I) || isa<ExtractElementInst>(I)) {
        if (I.getName().starts_with(".splatinsert")) {
          ++VeczSplats;
        }
        ++VeczInsertExtract;
      }

      // Count vector and scalar instructions
      if (isVectorInst(I)) {
        ++VeczVectorInstructions;
      } else {
        ++VeczScalarInstructions;
      }

      // Count memory operation types
      if (isa<LoadInst>(I) || isa<StoreInst>(I)) {
        // Normal scalar/vector loads and stores
        if (isVectorInst(I)) {
          ++VeczVectorMemOp;
        } else {
          ++VeczScalarMemOp;
        }
      } else if (CallInst *CI = dyn_cast<CallInst>(&I)) {
        Function *F = CI->getCalledFunction();
        if (!F) {
          continue;
        }
        // Subtract 1 for the call instruction, since we are inlining
        --VeczInstructions;

        for (auto &BB : *F) {
          for (auto &Inst : BB) {
            VeczInstructions += !isa<CallInst>(&Inst);
          }
        }
        // Internal builtin memory operations
        if (auto Op = MemOp::get(&I)) {
          VeczMaskedMemOps += Op->isMaskedMemOp();
          VeczInterleavedMemOps += Op->getDesc().isInterleavedMemOp();
          VeczMaskedInterleavedMemOps += Op->isMaskedInterleavedMemOp();
          VeczScatterGatherMemOps += Op->getDesc().isScatterGatherMemOp();
          VeczMaskedScatterGatherMemOps += Op->isMaskedScatterGatherMemOp();
        }
      }
    }
  }

  // Ratio = Normalized Scalar Insts / Vector Insts
  // Normalized Scalar Insts = Simd Width * Scalar Insts
  // IK - Input Kernel
  // Scalar Insts = IK's Scalar Insts + IK's Vec Insts * IK's VecWidth
  const unsigned SimdWidth = VU.width().getFixedValue();
  Ratio = (SimdWidth * (ScalarInstructions - ScalarVectorInsts +
                        ScalarVectorInsts * MaxScalarVectorWidth)) /
          VeczInstructions;
}
}  // namespace

VectorizationUnit *vecz::createVectorizationUnit(VectorizationContext &Ctx,
                                                 Function *Kernel,
                                                 const VeczPassOptions &Opts,
                                                 FunctionAnalysisManager &FAM,
                                                 bool Check) {
  const unsigned SimdDimIdx = Opts.vec_dim_idx;
  const unsigned LocalSize = Opts.local_size;
  const bool Auto = Opts.vecz_auto;
  auto VF =
      ElementCount::get(Opts.factor.getKnownMin(), Opts.factor.isScalable());

  if (!Kernel || VF.isScalar()) {
    ++VeczBail;
    VECZ_FAIL();
  }

  // Up to MAX_SIMD_DIM supported dimensions
  VECZ_ERROR_IF(SimdDimIdx >= MAX_SIMD_DIM,
                "Specified vectorization dimension is invalid");

  VECZ_ERROR_IF(VF.getKnownMinValue() == 0, "Vectorization factor of zero");

  // Adjust VF if the local size is known to vectorize more often.
  if (LocalSize && !VF.isScalable()) {
    // If we know the vectorized loop will never be entered, because the
    // vectorization factor is too large, then vectorizing is a waste of time.
    // It is better instead to vectorize by a smaller factor. Keep on halfing
    // the vector width until a useable value is found (worst case this value
    // will be 1, because that evenly divides everything).
    unsigned FixedSimdWidth = VF.getFixedValue();
    // Note FixedSimdWidth is either a power of two or 3. If FixedSimdWidth
    // was 1 then we would not enter the body of the loop (as X%1 is 0 for all
    // X), if FixedSimdWidth is a greater power of two then dividing it by 2
    // gives another power of two, 3 divided by 2 gives 1, a power of two. Thus
    // if this loop runs at least once then FixedSimdWidth will be a power of
    // 2.
    assert(FixedSimdWidth == 3 || llvm::isPowerOf2_32(FixedSimdWidth));
    while (FixedSimdWidth != 1 && FixedSimdWidth > LocalSize) {
      FixedSimdWidth /= 2;
      assert(FixedSimdWidth > 0 && "Cannot vectorize (or modulo) by 0.");
    }
    if (FixedSimdWidth == 1) {
      ++VeczBail;
      emitVeczRemarkMissed(Kernel, nullptr,
                           "requested Vectorization factor of 1");
      return nullptr;
    }
    VF = ElementCount::get(FixedSimdWidth, false);
  }

  bool canVectorize = true;
  if (Check) {
    auto Res = FAM.getResult<VectorizableFunctionAnalysis>(*Kernel);
    canVectorize = Res.canVectorize;
  }

  if (canVectorize &&
      (!Auto || shouldVectorize(*Kernel, Ctx, VF, SimdDimIdx))) {
    auto VU =
        Ctx.createVectorizationUnit(*Kernel, VF, SimdDimIdx, Opts.choices);
    VU->setAutoWidth(Auto);
    VU->setLocalSize(Opts.local_size);
    return VU;
  }
  return nullptr;
}

void vecz::trackVeczSuccessFailure(VectorizationUnit &VU) {
  Function *Fn = VU.scalarFunction();
  Function *vectorizedFn = VU.vectorizedFunction();
  const bool failed = VU.failed();
  VeczFail += failed;
  VeczSuccess += !failed;
  collectStatistics(VU, Fn, vectorizedFn);

  if (VeczDumpReport) {
    const auto VF = VU.width();
    auto FnName = Fn->getName();
    if (vectorizedFn) {
      errs() << "vecz: Vectorization succeeded for kernel '" << FnName
             << "' << (" << (VF.isScalable() ? "scalable-vector" : "SIMD")
             << " factor: " << VF.getKnownMinValue() << ") "
             << *vectorizedFn->getType() << "\n";
    } else {
      errs() << "vecz: Vectorization failed for kernel '" << FnName << "'\n";
    }
  }
}

bool vecz::createVectorizedFunctionMetadata(VectorizationUnit &vu) {
  Function *fn = vu.scalarFunction();
  Function *vectorizedFn = vu.vectorizedFunction();
  if (vu.failed()) {
    vectorizedFn = nullptr;
  } else {
    // If vectorization succeeded, clone the OpenCL related metadata from the
    // scalar kernel. We do not do this while cloning the kernel because if
    // vectorization fails we will have metadata pointing to non-existing
    // kernels.
    cloneOpenCLMetadata(vu);
  }
  const auto vf = vu.width();
  const auto dim = vu.dimension();

  // emit output metadata based on vectorization result
  auto finalVF = compiler::utils::VectorizationFactor(vf.getKnownMinValue(),
                                                      vf.isScalable());

  const compiler::utils::VectorizationInfo info{
      finalVF, dim, vu.choices().vectorPredication()};

  if (vectorizedFn && vectorizedFn != fn) {  // success
    // Link the original function to the vectorized one.
    compiler::utils::linkOrigToVeczFnMetadata(*fn, *vectorizedFn, info);

    // Link the vectorized function back to the original one.
    compiler::utils::linkVeczToOrigFnMetadata(*vectorizedFn, *fn, info);
  } else {  // fail or bail
    compiler::utils::encodeVectorizationFailedMetadata(*fn, info);
  }
  return vectorizedFn;
}
