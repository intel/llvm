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

#include "analysis/simd_width_analysis.h"

#include <compiler/utils/builtin_info.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Function.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <multi_llvm/vector_type_helper.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "analysis/liveness_analysis.h"
#include "analysis/packetization_analysis.h"
#include "analysis/vectorization_unit_analysis.h"
#include "debugging.h"
#include "vectorization_unit.h"
#include "vecz/vecz_target_info.h"

#define DEBUG_TYPE "vecz-simd-width"

using namespace llvm;
using namespace vecz;

llvm::AnalysisKey SimdWidthAnalysis::Key;

namespace {
bool definedOrUsedInLoop(Value *V, Loop *L) {
  if (!L) {
    // We're not in a loop, so consider everything.
    return true;
  }

  const auto *const I = dyn_cast<Instruction>(V);
  if (I && L->contains(I)) {
    // It's defined in the current loop.
    return true;
  }

  // If it's used in the current loop, return true, unless it is a PHI node.
  // Values defined outwith the loop, but used only by a PHI node within it must
  // be loop-carried variable initial values. If these are not otherwise used
  // directly within the loop, then they are not really live inside the loop.
  for (const auto *const U : V->users()) {
    const auto *const I = dyn_cast<Instruction>(U);
    if (I && !isa<PHINode>(I) && L->contains(I)) {
      return true;
    }
  }
  return false;
}
}  // namespace

// Avoid Spill implementation. It focus on avoiding register spill by optimizing
// register pressure.
unsigned SimdWidthAnalysis::avoidSpillImpl(Function &F,
                                           FunctionAnalysisManager &AM,
                                           unsigned MinWidth) {
  VectorizationUnit &VU = AM.getResult<VectorizationUnitAnalysis>(F).getVU();
  const TargetTransformInfo TTI = VU.context().getTargetTransformInfo(F);
  const auto &Liveness = AM.getResult<LivenessAnalysis>(F);
  const auto &PAR = AM.getResult<PacketizationAnalysis>(F);
  const LoopInfo &LI = AM.getResult<LoopAnalysis>(F);
  // Determine the SIMD width based on a live values register usage estimation.
  assert(!VU.width().isScalable() && "Can't handle scalable-vectors");
  unsigned SimdWidth = VU.width().getFixedValue();
  assert(SimdWidth != 0 && "SimdWidthAnalysis: SimdWidth == 0");

  SmallSet<const Value *, 16> OpenIntervals;
  SmallVector<const Value *, 16> IntervalArray;

  auto ShouldConsider = [&](const Value *V) -> bool {
    // Filter out work item builtin calls such as get_local_id()
    if (auto *const CI = dyn_cast<CallInst>(V)) {
      const Function *Callee = CI->getCalledFunction();
      if (Callee &&
          VU.context().builtins().analyzeBuiltin(*Callee).properties ==
              compiler::utils::eBuiltinPropertyWorkItem) {
        return false;
      }
    }
    return true;
  };

  LLVM_DEBUG(dbgs() << "VEC(REG): Calculating max register usage:\n");
  for (const auto &BB : F) {
    // Get the LiveIns for this Basic Block.
    // The principle of the Loop Aware SIMD Width Analysis is that it is not
    // acceptable to spill values in the middle of a loop, however it may be
    // acceptable to spill some values before entering a loop.
    const auto &BI = Liveness.getBlockInfo(&BB);
    OpenIntervals.clear();
    auto *const CurLoop = LI.getLoopFor(&BB);
    for (auto *V : BI.LiveOut) {
      if (ShouldConsider(V) && PAR.needsPacketization(V) &&
          definedOrUsedInLoop(V, CurLoop)) {
        OpenIntervals.insert(V);
      }
    }

    // Walk backwards through instructions in a block to count the maximum
    // number of live values in that block.
    for (auto &inst : make_range(BB.rbegin(), BB.rend())) {
      if (isa<PHINode>(&inst)) {
        break;
      }

      // The first instruction in the reverse range will be the terminator,
      // so we don't really need to consider it. However we do need to consider
      // the live set at the point before the last (i.e. first) instruction, so
      // we deal with the operands first and then process the live set.
      if (PAR.needsPacketization(&inst)) {
        const bool isGEP = isa<GetElementPtrInst>(&inst);
        for (auto operand : inst.operand_values()) {
          if (isa<Instruction>(operand) || isa<Argument>(operand)) {
            if (!isGEP || PAR.needsPacketization(operand)) {
              OpenIntervals.insert(operand);
            }
          }
        }
      }

      OpenIntervals.erase(&inst);
      IntervalArray.assign(OpenIntervals.begin(), OpenIntervals.end());
      SimdWidth = VU.context().targetInfo().estimateSimdWidth(
          TTI, IntervalArray, SimdWidth);
      LLVM_DEBUG(dbgs() << "VEC(REG): Interval # " << OpenIntervals.size()
                        << " at SIMD Width " << SimdWidth << '\n');
      LLVM_DEBUG(
          for (auto OII = OpenIntervals.begin(), OIIE = OpenIntervals.end();
               OII != OIIE; OII++) { dbgs() << "inst:" << **OII << '\n'; });

      if (SimdWidth < MinWidth) {
        return 0;
      }
    }
  }

  LLVM_DEBUG(dbgs() << "VEC(REG): Found widest fitting SIMD width: "
                    << SimdWidth << '\n');
  return SimdWidth;
}

SimdWidthAnalysis::Result SimdWidthAnalysis::run(
    Function &F, llvm::FunctionAnalysisManager &AM) {
  const TargetTransformInfo &TTI = AM.getResult<TargetIRAnalysis>(F);
  const VectorizationUnit &VU =
      AM.getResult<VectorizationUnitAnalysis>(F).getVU();

  // If the target does not provide vector registers, return 0.
  MaxVecRegBitWidth =
      TTI.getRegisterBitWidth(llvm::TargetTransformInfo::RGK_FixedWidthVector)
          .getFixedValue();

  if (MaxVecRegBitWidth == 0) {
    return 0;
  }

  // If the vectorization factor is for scalable vectors, return 0.
  if (VU.width().isScalable()) {
    return 0;
  }

  auto SimdWidth = avoidSpillImpl(F, AM, 1);
  if (SimdWidth != 0 && SimdWidth < 4) {
    // We only return 0 (i.e. don't vectorize) in the case that the packetized
    // values wouldn't fit into vector registers even with a factor of 1. If
    // the packetized values fit into vector registers for any width, we use
    // a baseline factor of 4 since this is empirically better than 2.
    SimdWidth = 4;
  }

  return SimdWidth;
}
