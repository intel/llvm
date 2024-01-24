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

#include "analysis/uniform_value_analysis.h"

#include <compiler/utils/builtin_info.h>
#include <compiler/utils/mangling.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>

#include <cstdlib>

#include "analysis/instantiation_analysis.h"
#include "analysis/vectorization_unit_analysis.h"
#include "debugging.h"
#include "memory_operations.h"
#include "vectorization_unit.h"

#define DEBUG_TYPE "vecz"

using namespace vecz;
using namespace llvm;

namespace {

// Find leaves by recursing through an instruction's uses
bool findStrayLeaves(UniformValueResult &UVR, Instruction &I,
                     DenseSet<Instruction *> &Visited) {
  for (const Use &U : I.uses()) {
    auto *User = U.getUser();
    if (isa<StoreInst>(User) || isa<AtomicRMWInst>(User) ||
        isa<AtomicCmpXchgInst>(User)) {
      if (UVR.isValueOrMaskVarying(User)) {
        return true;
      }
    } else if (auto *CI = dyn_cast<CallInst>(User)) {
      if (CI->use_empty()) {
        // Any call instruction with no uses is counted as a leaf. This case
        // should also cover any kind of masked stores, since masked stores are
        // builtin calls with no uses, there is no need to explicitly check for
        // masked stores.
        if (UVR.isValueOrMaskVarying(CI)) {
          return true;
        }
      }
    } else if (auto *UI = dyn_cast<Instruction>(User)) {
      if (isa<LoadInst>(User)) {
        // Don't trace through loads
      } else if (Visited.insert(UI).second) {
        if (findStrayLeaves(UVR, *UI, Visited)) {
          return true;
        }
      }
    }
  }
  return false;
}

bool isDivergenceReduction(const Function &F) {
  compiler::utils::Lexer L(F.getName());
  return (L.Consume(VectorizationContext::InternalBuiltinPrefix) &&
          L.Consume("divergence_"));
}

bool isTrueUniformInternal(const Value *V, unsigned Depth) {
  if (!V) {
    return false;
  }

  // Constants and Arguments that can't be undef/poison are truly uniform
  if (isa<Constant>(V) || isa<Argument>(V)) {
    return isGuaranteedNotToBePoison(V);
  }

  constexpr unsigned DepthLimit = 6;

  if (Depth < DepthLimit) {
    // For a specific subset of instructions, if all operands are truly
    // uniform, then the instruction is too.
    // FIXME: This is pessimistic. We could improve this by extending the list
    // of instructions covered. We could also use flow-sensitive analysis in
    // isGuaranteedNotToBePoison to enhance its capabilities.
    if (const auto *I = dyn_cast<Instruction>(V)) {
      if (isa<UnaryOperator>(I) || isa<BinaryOperator>(I) || isa<CastInst>(I) ||
          isa<CmpInst>(I) || isa<SelectInst>(I) || isa<PHINode>(I)) {
        return isGuaranteedNotToBePoison(I) &&
               llvm::all_of(I->operands(), [Depth](Value *Op) {
                 return isTrueUniformInternal(Op, Depth + 1);
               });
      }
    }
  }

  return false;
}

}  // namespace

UniformValueResult::UniformValueResult(Function &F, VectorizationUnit &vu)
    : F(F), VU(vu), Ctx(VU.context()), dimension(VU.dimension()) {}

bool UniformValueResult::isVarying(const Value *V) const {
  auto found = varying.find(V);
  if (found == varying.end()) {
    return false;
  }
  return found->second == VaryingKind::eValueVarying;
}

bool UniformValueResult::isMaskVarying(const Value *V) const {
  auto found = varying.find(V);
  if (found == varying.end()) {
    return false;
  }
  return found->second == VaryingKind::eMaskVarying;
}

bool UniformValueResult::isValueOrMaskVarying(const Value *V) const {
  auto found = varying.find(V);
  if (found == varying.end()) {
    return false;
  }
  return found->second != VaryingKind::eValueTrueUniform &&
         found->second != VaryingKind::eValueActiveUniform;
}

bool UniformValueResult::isTrueUniform(const Value *V) {
  auto found = varying.find(V);
  if (found != varying.end()) {
    return found->second == VaryingKind::eValueTrueUniform;
  }
  if (!isTrueUniformInternal(V, /*Depth=*/0)) {
    return false;
  }
  // Cache this result to help speed up future queries
  varying[V] = VaryingKind::eValueTrueUniform;
  return true;
}

/// @brief Utility function to check whether an instruction is a call to a
/// reduction or broadcast operaton.
///
/// @param[in] I Instruction to check
/// @param[in] BI BuiltinInfo for platform-specific builtin IDs
/// @return true if the instruction is a call to a reduction or broadcast
/// builtin.
static bool isGroupBroadcastOrReduction(
    const Instruction &I, const compiler::utils::BuiltinInfo &BI) {
  if (!isa<CallInst>(&I)) {
    return false;
  }
  auto *const CI = cast<CallInst>(&I);
  auto *const Callee = CI->getCalledFunction();
  if (!Callee) {
    return false;
  }
  auto Info = BI.isMuxGroupCollective(BI.analyzeBuiltin(*Callee).ID);
  return Info && (Info->isSubGroupScope() || Info->isWorkGroupScope()) &&
         (Info->isAnyAll() || Info->isReduction() || Info->isBroadcast());
}

void UniformValueResult::findVectorLeaves(
    std::vector<Instruction *> &Leaves) const {
  const compiler::utils::BuiltinInfo &BI = Ctx.builtins();
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      // Reductions and broadcasts are always vector leaves regardless of
      // uniformity.
      if (isGroupBroadcastOrReduction(I, BI)) {
        Leaves.push_back(&I);
        continue;
      }

      if (!isVarying(&I)) {
        if (isMaskVarying(&I)) {
          // it's a leaf if only its mask operand is varying, since the value
          // itself will be uniform and won't propagate "varying" to its users.
          Leaves.push_back(&I);
          continue;
        }
        if (CallInst *CI = dyn_cast<CallInst>(&I)) {
          Function *Callee = CI->getCalledFunction();
          if (!Callee) {
            continue;
          }

          // If its a call to user defined function whose use is empty, and is
          // uniform then add it to the leaves
          if (!Callee->isIntrinsic() && CI->use_empty()) {
            // Try to identify the called function
            const auto Builtin = BI.analyzeBuiltin(*Callee);
            if (!Builtin.isValid()) {
              Leaves.push_back(CI);
            }
          }
        }
        continue;
      }

      if (StoreInst *Store = dyn_cast<StoreInst>(&I)) {
        Instruction *Ptr = dyn_cast<Instruction>(Store->getPointerOperand());
        if (Ptr && isVarying(Ptr)) {
          Leaves.push_back(Store);
        }
        continue;
      }

      if (ReturnInst *Ret = dyn_cast<ReturnInst>(&I)) {
        Leaves.push_back(Ret);
        continue;
      }

      if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(&I)) {
        Leaves.push_back(RMW);
        continue;
      } else if (AtomicCmpXchgInst *CmpXchg = dyn_cast<AtomicCmpXchgInst>(&I)) {
        Leaves.push_back(CmpXchg);
        continue;
      }

      // Functions that have no uses are leaves.
      if (CallInst *CI = dyn_cast<CallInst>(&I)) {
        bool IsCallLeaf = false;
        if (CI->use_empty()) {
          IsCallLeaf = true;
        } else if (auto Op = MemOp::get(CI)) {
          // Handle masked stores.
          if (Op->isStore() &&
              (Op->isMaskedMemOp() || Op->isMaskedInterleavedMemOp() ||
               Op->isMaskedScatterGatherMemOp())) {
            IsCallLeaf = true;
          }
        } else if (Ctx.isMaskedAtomicFunction(*CI->getCalledFunction())) {
          IsCallLeaf = true;
        }
        if (IsCallLeaf) {
          Leaves.push_back(CI);
          continue;
        }
      }
    }
  }
}

void UniformValueResult::findVectorRoots(std::vector<Value *> &Roots) const {
  const compiler::utils::BuiltinInfo &BI = Ctx.builtins();
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      CallInst *CI = dyn_cast<CallInst>(&I);
      if (!CI || !CI->getCalledFunction()) {
        continue;
      }
      const auto Builtin = BI.analyzeBuiltinCall(*CI, dimension);
      const auto Uniformity = Builtin.uniformity;
      if (Uniformity == compiler::utils::eBuiltinUniformityInstanceID ||
          Uniformity == compiler::utils::eBuiltinUniformityMaybeInstanceID) {
        // Calls to `get_global_id`/`get_local_id` are roots.
        Roots.push_back(CI);
      } else if ((Uniformity == compiler::utils::eBuiltinUniformityNever) &&
                 !CI->getType()->isVoidTy()) {
        // Non-void builtins with side-effects are also roots.
        Roots.push_back(CI);
      }
    }
  }

  // Add vectorized arguments to the list of roots.
  for (const VectorizerTargetArgument &TargetArg : VU.arguments()) {
    if (!TargetArg.IsVectorized && !TargetArg.PointerRetPointeeTy) {
      continue;
    }

    if (&F == VU.scalarFunction()) {
      Roots.push_back(TargetArg.OldArg);
    } else if (&F == VU.vectorizedFunction()) {
      if (TargetArg.Placeholder) {
        Roots.push_back(TargetArg.Placeholder);
      } else {
        Roots.push_back(TargetArg.NewArg);
      }
    }
  }
}

AllocaInst *UniformValueResult::findAllocaFromPointer(Value *Pointer) {
  while (Pointer) {
    if (AllocaInst *Alloca = dyn_cast<AllocaInst>(Pointer)) {
      return Alloca;
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Pointer)) {
      Pointer = GEP->getPointerOperand();
    } else if (BitCastInst *BC = dyn_cast<BitCastInst>(Pointer)) {
      Pointer = BC->getOperand(0);
    } else if (LoadInst *Load = dyn_cast<LoadInst>(Pointer)) {
      Pointer = Load->getPointerOperand();
    } else {
      return nullptr;
    }
  }

  return nullptr;
}

void UniformValueResult::markVaryingValues(Value *V, Value *From) {
  auto &vary = varying[V];
  // Do not visit values twice.
  if (vary == VaryingKind::eValueVarying) {
    return;
  }

  if (CallInst *CI = dyn_cast<CallInst>(V)) {
    // Some builtins produce a uniform value regardless of their inputs.
    Function *Callee = CI->getCalledFunction();
    if (Callee) {
      const compiler::utils::BuiltinInfo &BI = Ctx.builtins();
      const auto Builtin = BI.analyzeBuiltinCall(*CI, dimension);
      const auto Uniformity = Builtin.uniformity;
      if (Uniformity == compiler::utils::eBuiltinUniformityAlways) {
        return;
      }

      if (auto Op = MemOp::get(CI)) {
        // The mask cannot affect the MemOp value, even though we may still
        // need to packetize the mask..
        auto *Mask = Op->getMaskOperand();
        if (Mask && From == Mask) {
          vary = VaryingKind::eMaskVarying;
          return;
        }
      } else if (Ctx.isInternalBuiltin(Callee)) {
        // A divergence reduction builtin's value is uniform even though its
        // argument is not, since it is a reduction over the SIMD width.
        if (isDivergenceReduction(*Callee)) {
          vary = VaryingKind::eMaskVarying;
          return;
        }
      }
    }
  }

  // Mark V as being varying.
  vary = VaryingKind::eValueVarying;
  LLVM_DEBUG(dbgs() << "vecz: Needs packetization: " << *V << "\n");

  // Visit all users of V, they are varying too.
  for (const Use &Use : V->uses()) {
    User *User = Use.getUser();
    markVaryingValues(User, V);
  }

  // Mark uses of V for certain kinds of values.
  Instruction *VIns = dyn_cast<Instruction>(V);
  if (!VIns) {
    return;
  }

  if (StoreInst *Store = dyn_cast<StoreInst>(VIns)) {
    // Find the base address for the store. Storing varying values to an
    // alloca location requires the alloca to be vectorized.
    // We don't want to use extractMemOffset here because this requires the
    // uniform value analysis to be finished.
    AllocaInst *Alloca = findAllocaFromPointer(Store->getPointerOperand());
    if (Alloca) {
      markVaryingValues(Alloca);
    }
  } else if (LoadInst *Load = dyn_cast<LoadInst>(VIns)) {
    AllocaInst *Alloca = findAllocaFromPointer(Load->getPointerOperand());
    if (Alloca) {
      markVaryingValues(Alloca);
    }
  } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(VIns)) {
    // Same as with the stores
    AllocaInst *Alloca = findAllocaFromPointer(GEP->getPointerOperand());
    if (Alloca) {
      markVaryingValues(Alloca);
    }
  } else if (BitCastInst *BC = dyn_cast<BitCastInst>(VIns)) {
    // Same as with the stores
    AllocaInst *Alloca = findAllocaFromPointer(BC->getOperand(0));
    if (Alloca) {
      markVaryingValues(Alloca);
    }
  } else if (CallInst *CI = dyn_cast<CallInst>(VIns)) {
    // Stores might be function calls as well
    // Known MemOps have one known pointer operand which we can check.
    if (auto Op = MemOp::get(CI)) {
      if (auto *const Ptr = Op->getPointerOperand()) {
        if (auto *Alloca = findAllocaFromPointer(Ptr)) {
          markVaryingValues(Alloca);
        }
      }
    } else {
      // Check all parameters of unknown calls with pointer arguments.
      for (auto &A : CI->args()) {
        if (A->getType()->isPointerTy()) {
          if (auto *Alloca = findAllocaFromPointer(A)) {
            markVaryingValues(Alloca);
          }
        }
      }
    }
  }
}

Value *UniformValueResult::extractMemBase(Value *Address) {
  if (BitCastInst *BCast = dyn_cast<BitCastInst>(Address)) {
    return extractMemBase(BCast->getOperand(0));
  } else if (auto *ASCast = dyn_cast<AddrSpaceCastInst>(Address)) {
    return extractMemBase(ASCast->getOperand(0));
  } else if (isa<IntToPtrInst>(Address)) {
    return Address;
  } else if (isa<Argument>(Address)) {
    return Address;
  } else if (isa<GlobalVariable>(Address)) {
    return Address;
  } else if (isa<AllocaInst>(Address)) {
    return Address;
  } else if (auto *const Phi = dyn_cast<PHINode>(Address)) {
    // If all the incoming values are the same, we can trace through it. In
    // the general case, it's not trivial to check that the stride is the same
    // from every incoming block, and since incoming values may not dominate
    // the IRBuilder insert point, we might not even be able to build the
    // offset expression instructions there.
    if (auto *const CVal = Phi->hasConstantValue()) {
      return extractMemBase(CVal);
    }

    // In the simple case of a loop-incremented pointer using a GEP, we can
    // handle it thus:
    auto NumIncoming = Phi->getNumIncomingValues();
    if (NumIncoming != 2) {
      // Perhaps we can handle more than one loop latch, but not yet.
      return nullptr;
    }

    if (auto *const GEP =
            dyn_cast<GetElementPtrInst>(Phi->getIncomingValue(1))) {
      // If it's a simple loop iterator, the base can be analyzed from the
      // initial value.
      if (GEP->getPointerOperand() == Phi) {
        for (const auto &index : GEP->indices()) {
          if (isVarying(index.get())) {
            return nullptr;
          }
        }
        return extractMemBase(Phi->getIncomingValue(0));
      }
    }

    return nullptr;
  } else if (auto *GEP = dyn_cast<GetElementPtrInst>(Address)) {
    // Try to recursively extract the base from the GEP base.
    return extractMemBase(GEP->getPointerOperand());
  } else if (isVarying(Address)) {
    // If it's varying we can't analyze it any further.
    return nullptr;
  } else {
    // If it's uniform we can just return the uniform address.
    return Address;
  }
}

////////////////////////////////////////////////////////////////////////////////

llvm::AnalysisKey UniformValueAnalysis::Key;

UniformValueResult UniformValueAnalysis::run(
    llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  VectorizationUnit &VU = AM.getResult<VectorizationUnitAnalysis>(F).getVU();
  UniformValueResult Res(F, VU);
  std::vector<Value *> Roots;
  Res.findVectorRoots(Roots);

  // Mark all roots and their uses as being varying.
  for (Value *Root : Roots) {
    Res.markVaryingValues(Root);
  }

  const compiler::utils::BuiltinInfo &BI = Res.Ctx.builtins();
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      // Find atomic instructions, these are always varying
      if (I.isAtomic()) {
        Res.markVaryingValues(&I);
        continue;
      }

      // The same goes for the atomic builtins as well
      if (CallInst *CI = dyn_cast<CallInst>(&I)) {
        if (Function *Callee = CI->getCalledFunction()) {
          const auto Builtin = BI.analyzeBuiltin(*Callee);
          if (Builtin.properties & compiler::utils::eBuiltinPropertyAtomic) {
            Res.markVaryingValues(&I);
            continue;
          }
        }
      }
    }
  }

  // If an alloca has been initialized with a uniform value, findVectorLeaves()
  // will not pick up the store instruction as a leaf, even when that alloca is
  // used by some other leaves. We have to go through all the allocas and mark
  // them as varying if any varying instructions use them. This is the case
  // also for masked stores where only the mask is varying.
  bool Changed = true;
  while (Changed) {
    DenseSet<Instruction *> Visited;
    Changed = false;
    bool Remaining = false;
    for (Instruction &I : F.front()) {
      if (isa<AllocaInst>(&I)) {
        if (!Res.isVarying(&I)) {
          if (findStrayLeaves(Res, I, Visited)) {
            // We found a varying leaf, so this Alloca is non-uniform.
            Res.markVaryingValues(&I);

            // Marking an alloca as varying could mark a leaf as varying that
            // may also depend on a different alloca, so we have to go again.
            Changed = true;
          } else {
            Remaining = true;
          }
        }
      } else {
        break;
      }
    }
    Changed &= Remaining;
  }

  return Res;
}
