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

#include "transform/interleaved_group_combine_pass.h"

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Debug.h>
#include <llvm/Transforms/Utils/Local.h>

#include <optional>

#include "analysis/uniform_value_analysis.h"
#include "analysis/vectorization_unit_analysis.h"
#include "debugging.h"
#include "ir_cleanup.h"
#include "memory_operations.h"

#define DEBUG_TYPE "vecz"

using namespace llvm;
using namespace vecz;

char InterleavedGroupCombinePass::PassID = 0;

struct GroupMemberInfo {
  int64_t Offset;
  int64_t Order;
  CallInst *MemOp;
  Value *Ptr;
  Type *DataTy;
};

/// @brief Information about an interleaved operation.
struct InterleavedGroupCombinePass::InterleavedOpInfo {
  /// @brief Interleaved operation.
  CallInst *Op;
  /// @brief Kind of interleaved operation.
  InterleavedOperation Kind;
  /// @brief Interleaved stride.
  int Stride;
  /// @brief Whether the operation was removed or not.
  bool Removed;
};

struct InterleavedGroupCombinePass::InterleavedGroupInfo {
  BasicBlock *BB = nullptr;
  SmallVector<Value *, 4> Data;
  SmallVector<GroupMemberInfo, 4> Info;
  Value *Base = nullptr;
  unsigned Stride = 0;
  int Offset = 0;
  InterleavedOperation Kind = eInterleavedInvalid;

  void clear() {
    BB = nullptr;
    Data.clear();
    Info.clear();
    Base = nullptr;
    Stride = 0;
    Offset = 0;
    Kind = eInterleavedInvalid;
  }

  bool isConsecutive() const {
    auto InfoIt = Info.begin();
    auto InfoE = Info.end();
    assert(InfoIt != InfoE);
    int ExpectedOffset = Info.front().Offset;
    for (++InfoIt; InfoIt != InfoE; ++InfoIt) {
      if (InfoIt->Offset != ++ExpectedOffset) {
        return false;
      }
    }
    return true;
  }

  bool canDeinterleaveMask(const Instruction &Mask) const;
};

namespace {

bool canSwap(Instruction *IA, Instruction *IB) {
  // we need to check for usage-relations here, because a load instruction
  // might depend on a mask calculation and its uses that might end up
  // swapped
  for (auto *const Op : IB->operand_values()) {
    if (isa<GetElementPtrInst>(Op)) {
      // GEPs get eliminated later so ignore them for now
      continue;
    }
    if (Op == IA) {
      return false;
    }
  }

  if (IA->mayReadOrWriteMemory()) {
    if (isa<FenceInst>(IB)) {
      // can't swap any memory operation with a fence
      return false;
    }
  } else {
    // if either instruction is not a memory operation, we can swap them.
    return true;
  }

  if (IB->mayReadOrWriteMemory()) {
    if (isa<FenceInst>(IA)) {
      return false;
    }
  } else {
    return true;
  }

  // can't swap a write with a write, or a write with a read,
  // but it should be ok to swap two reads
  if (IA->mayWriteToMemory() || IB->mayWriteToMemory()) {
    return false;
  }

  return true;
}

bool canMoveUp(const SmallVectorImpl<Value *> &Group, Instruction *IB) {
  auto Ig = Group.rbegin();
  auto Ie = Group.rend();
  Instruction *IA = IB;

  // It looks through all preceding instructions, skipping over any that are
  // already in the Group, until it reaches the first member of the group,
  // terminating if it can't move IB through the current instruction.
  // If it reaches the first member of the Group, it is safe to move IB there.
  while ((IA = IA->getPrevNode())) {
    if (IA == *Ig) {
      if (++Ig == Ie) {
        // we met every group member so we're done
        return true;
      }
    } else if (!canSwap(IA, IB)) {
      return false;
    }
  }
  // if we get here, it means we didn't pass any of the other group members,
  // which shouldn't be able to happen.
  assert(false);
  return false;
}

bool canMoveDown(const SmallVectorImpl<Value *> &Group, Instruction *IA) {
  auto Ig = Group.rbegin();
  auto Ie = Group.rend();
  Instruction *IB = IA;

  // It looks through all following instructions, skipping over any that are
  // already in the Group, until it reaches the first member of the group,
  // terminating if it can't move IA through the current instruction.
  // If it reaches the first member of the Group, it is safe to move IA there.
  while ((IB = IB->getNextNode())) {
    if (IB == *Ig) {
      if (++Ig == Ie) {
        // we met every group member so we're done
        return true;
      }
    } else if (!canSwap(IA, IB)) {
      return false;
    }
  }
  // if we get here, it means we didn't pass any of the other group members,
  // which shouldn't be able to happen.
  assert(false);
  return false;
}

}  // namespace

bool InterleavedGroupCombinePass::InterleavedGroupInfo::canDeinterleaveMask(
    const Instruction &Mask) const {
  // If the mask definition is not in the same block as the group members, it
  // is safe to de-interleave.
  if (Mask.getParent() != BB) {
    return true;
  }

  SmallPtrSet<Instruction *, 2> Ops;
  for (auto &Op : Mask.operands()) {
    if (auto *OpI = dyn_cast<Instruction>(Op.get())) {
      // We only care about operands in the same basic block, since otherwise
      // they cannot be group members or in between group members.
      if (OpI->getParent() == BB) {
        Ops.insert(OpI);
      }
    }
  }

  // If the mask has no dependency on anything in the group basic block, it is
  // safe to de-interleave.
  if (Ops.empty()) {
    return true;
  }

  // Note that the mask can hardly depend on the last group member, since it is
  // itself an operand of this member.
  Instruction *IA = cast<Instruction>(Data.back());

  // It looks through all instructions from the last member of the group
  // back to the first, looking to see if the mask depends on any of them.
  // If it reaches the first member of the Group, it is safe to move the mask.
  // If it finds any of the mask's own operands as group members or in
  // between group members, the mask cannot be (trivially) moved.
  while (IA) {
    if (Ops.count(IA)) {
      // We found something the mask depends on, so we can't de-interleave...
      return false;
    } else if (IA == Data.front()) {
      // we met every group member so we're done
      return true;
    }
    IA = IA->getPrevNode();
  }

  // the mask definition was before every group member
  return true;
}

PreservedAnalyses InterleavedGroupCombinePass::run(
    Function &F, FunctionAnalysisManager &AM) {
  auto &Ctx = AM.getResult<VectorizationContextAnalysis>(F).getContext();
  IRCleanup IC;

  const bool IsLoad =
      (Kind == eInterleavedLoad) || (Kind == eMaskedInterleavedLoad);

  LLVM_DEBUG(dbgs() << "vecz: InterleavedGroupCombinePass on " << F.getName()
                    << "\n");

  scalarEvolution = &AM.getResult<ScalarEvolutionAnalysis>(F);

  UniformValueResult &UVR = AM.getResult<UniformValueAnalysis>(F);
  const auto &DL = F.getParent()->getDataLayout();
  std::vector<InterleavedOpInfo> InterleavedOps;
  for (BasicBlock &BB : F) {
    // Look for interleaved operations.
    for (Instruction &I : BB) {
      CallInst *CI = dyn_cast<CallInst>(&I);
      if (!CI) {
        continue;
      }

      std::optional<MemOp> Op = MemOp::get(CI);
      // We can't optimize interleaved memops if we don't know the stride at
      // runtime, since we need to check if the stride and the group size match.
      if (!Op || !Op->isStrideConstantInt()) {
        continue;
      }
      const int64_t Stride = Op->getStrideAsConstantInt();
      if ((Stride == 0) || (Stride == 1)) {
        continue;
      }
      Value *Mask = Op->getMaskOperand();
      InterleavedOpInfo Info;

      const bool OpIsLoad = Op->isLoad();
      Info.Kind = OpIsLoad
                      ? (Mask ? eMaskedInterleavedLoad : eInterleavedLoad)
                      : (Mask ? eMaskedInterleavedStore : eInterleavedStore);
      Info.Op = CI;
      Info.Stride = Stride;
      Info.Removed = false;

      // only add the interleaved operation kinds we actually care about
      if (IsLoad == OpIsLoad) {
        InterleavedOps.push_back(Info);
      }
    }
    if (!InterleavedOps.empty()) {
      if (Kind == eInterleavedStore) {
        // stores are collated downwards, so reverse the list..
        std::reverse(InterleavedOps.begin(), InterleavedOps.end());
      }

      InterleavedGroupInfo Group;
      Group.BB = &BB;

      while (findGroup(InterleavedOps, UVR, Group)) {
        // Loads have their uses afterwards, while stores use preceding values.
        // Group.Info is in forwards order for Loads, reverse order for Stores.
        IRBuilder<> B(Group.Info.front().MemOp);

        Value *Base = Group.Base;
        if (Kind == eInterleavedLoad && Group.Offset != 0) {
          auto *EltTy = Group.Info.front().DataTy->getScalarType();
          // if it's a Load group that was out of order, we have to use the
          // sequentially first GEP in order to preserve use-def ordering,
          // which means we have to offset it with an additional GEP and
          // hope this optimizes out later.
          // Note that this is not necessary for Stores, since instructions
          // are inserted at the last Store.
          Base = Group.Info.front().Ptr;
          auto *Offset = ConstantInt::getSigned(
              DL.getIntPtrType(Base->getType()), Group.Offset);

          Base = B.CreateInBoundsGEP(EltTy, Base, Offset, "reorder_offset");
        }

        SmallVector<Value *, 4> Masks;
        if (Group.Kind == eMaskedInterleavedStore ||
            Group.Kind == eMaskedInterleavedLoad) {
          Masks.reserve(Group.Data.size());
          for (auto *V : Group.Data) {
            std::optional<MemOp> Op = MemOp::get(cast<Instruction>(V));
            assert(Op && "Unanalyzable interleaved access?");
            Masks.push_back(Op->getMaskOperand());
          }
        }
        if (Ctx.targetInfo().optimizeInterleavedGroup(
                B, Group.Kind, Group.Data, Masks, Base, Group.Stride)) {
          for (Value *V : Group.Data) {
            if (Instruction *Ins = dyn_cast<Instruction>(V)) {
              IC.deleteInstructionLater(Ins);
            }
          }
        }

        // Remove the group no matter whether we optimized it or not. Otherwise
        // we will just iterate indefinitely.
        for (const auto &Info : Group.Info) {
          InterleavedOps[Info.Order].Removed = true;
        }
      }
      InterleavedOps.clear();
    }
  }
  IC.deleteInstructions();

  LLVM_DEBUG(dbgs() << "vecz: InterleavedGroupCombinePass done!\n");

  PreservedAnalyses Preserved;
  Preserved.preserve<ScalarEvolutionAnalysis>();
  Preserved.preserve<DominatorTreeAnalysis>();
  Preserved.preserve<LoopAnalysis>();

  return Preserved;
}

bool InterleavedGroupCombinePass::findGroup(
    const std::vector<InterleavedOpInfo> &Ops, UniformValueResult &UVR,
    InterleavedGroupInfo &Group) {
  VECZ_FAIL_IF(Ops.empty());
  // this check keeps clang-tidy happy
  VECZ_FAIL_IF(Kind != eInterleavedStore && Kind != eInterleavedLoad);

  auto &SE = *scalarEvolution;

  for (unsigned i = 0; i < Ops.size(); i++) {
    // Extract the first memory instruction at the given offset.
    const InterleavedOpInfo &Info0 = Ops[i];
    if (Info0.Removed) {
      continue;
    }

    Type *DataType0 = nullptr;
    Value *Ptr0 = nullptr;
    if (Kind == eInterleavedStore) {
      DataType0 = Info0.Op->getOperand(0)->getType();
      Ptr0 = Info0.Op->getOperand(1);
    } else if (Kind == eInterleavedLoad) {
      DataType0 = Info0.Op->getType();
      Ptr0 = Info0.Op->getOperand(0);
    }

    const IRBuilder<> B(cast<Instruction>(Info0.Op));
    Value *Base0 = UVR.extractMemBase(Ptr0);
    if (!Base0) {
      continue;
    }

    PointerType *PtrTy = dyn_cast<PointerType>(Ptr0->getType());
    if (!PtrTy) {
      continue;
    }

    Type *EleTy = DataType0->getScalarType();
    const unsigned Align = EleTy->getScalarSizeInBits() / 8;
    assert(Align != 0 &&
           "interleaved memory operation with zero-sized elements");

    Group.clear();
    Group.Data.push_back(Info0.Op);
    Group.Info.emplace_back(GroupMemberInfo{0, i, Info0.Op, Ptr0, DataType0});
    Group.Kind = Info0.Kind;

    // Try to find others that have the same stride and base pointer.
    for (unsigned j = i + 1; j < Ops.size(); j++) {
      const InterleavedOpInfo &InfoN = Ops[j];
      if (InfoN.Removed) {
        continue;
      }

      if (Group.Kind != InfoN.Kind) {
        continue;
      }

      Type *DataTypeN = nullptr;
      Value *PtrN = nullptr;
      if (Kind == eInterleavedStore) {
        DataTypeN = InfoN.Op->getOperand(0)->getType();
        PtrN = InfoN.Op->getOperand(1);
      } else if (Kind == eInterleavedLoad) {
        DataTypeN = InfoN.Op->getType();
        PtrN = InfoN.Op->getOperand(0);
      }

      if ((InfoN.Stride != Info0.Stride) || (DataTypeN != DataType0)) {
        continue;
      }

      const IRBuilder<> B(cast<Instruction>(InfoN.Op));
      Value *BaseN = UVR.extractMemBase(PtrN);
      if (!BaseN || BaseN != Base0) {
        continue;
      }

      const SCEV *PtrDiff = SE.getMinusSCEV(SE.getSCEV(PtrN), SE.getSCEV(Ptr0));
      const auto *ConstDiff = dyn_cast<SCEVConstant>(PtrDiff);
      if (!ConstDiff) {
        continue;
      }

      // Note that the offset calculated here is a byte offset
      int64_t Offset = ConstDiff->getAPInt().getSExtValue();
      if (Offset % Align == 0) {
        // only add them to the group if it is possible to collate them together
        // at the same place in the function
        bool CanMove = false;
        if (Kind == eInterleavedLoad) {
          CanMove = canMoveUp(Group.Data, cast<Instruction>(InfoN.Op));

          if (InfoN.Kind == eMaskedInterleavedLoad) {
            std::optional<MemOp> Op = MemOp::get(InfoN.Op);
            assert(Op && "Unanalyzable load?");
            if (auto *MaskInst = dyn_cast<Instruction>(Op->getMaskOperand())) {
              CanMove &= Group.canDeinterleaveMask(*MaskInst);
            }
          }
        } else if (Kind == eInterleavedStore) {
          CanMove = canMoveDown(Group.Data, cast<Instruction>(InfoN.Op));
        }

        if (CanMove) {
          Offset /= Align;
          Group.Data.push_back(InfoN.Op);
          Group.Info.emplace_back(
              GroupMemberInfo{Offset, j, InfoN.Op, PtrN, DataTypeN});
        }
      }
    }

    if (Group.Data.size() > 1) {
      auto InfoB = Group.Info.begin();
      auto InfoE = Group.Info.end();

      if (Kind == eInterleavedStore) {
        // In the case of stores, the instructions are processed in reverse
        // order, so this just puts them back in forwards order
        std::reverse(InfoB, InfoE);
      }

      // Sort the group members in order of their offsets. Use a stable sort
      // so that any duplicates don't get re-ordered (important for stores).
      std::stable_sort(
          InfoB, InfoE,
          [](const GroupMemberInfo &a, const GroupMemberInfo &b) -> bool {
            return a.Offset < b.Offset;
          });

      // If the same offset occurs several times, we can still de-interleave
      // the unique ones, and maybe catch the rest the next time round.
      InfoE = Group.Info.erase(
          std::unique(InfoB, InfoE,
                      [](const GroupMemberInfo &a, const GroupMemberInfo &b)
                          -> bool { return a.Offset == b.Offset; }),
          InfoE);

      if (Group.Info.size() <= 1) {
        // This could happen if our entire group has the same address, in
        // which case "std::unique" removes all but the first element and we
        // don't have a Group anymore.
        continue;
      }

      const unsigned Stride = Info0.Stride;
      Group.Stride = Stride;
      // If the group is bigger than the stride we can still de-interleave the
      // first "Stride" members
      if (Group.Info.size() > Stride) {
        Group.Info.resize(Stride);
        InfoB = Group.Info.begin();
        InfoE = Group.Info.end();
      }

      if (!Group.isConsecutive()) {
        // The group of memory instructions was not consecutive, try further.
        continue;
      }

      // Everything is fine, return this group in offset-sorted order.
      {
        Group.Data.resize(Group.Info.size());
        auto InfoIt = InfoB;
        for (auto &Op : Group.Data) {
          assert(InfoIt != InfoE);
          Op = (InfoIt++)->MemOp;
        }
      }

      Group.Base = Group.Info.front().Ptr;
      Group.Offset = Group.Info.front().Offset;

      // Put the Info list back into original Ops vector order
      // (reverse order for Stores)
      std::sort(InfoB, InfoE,
                [](const GroupMemberInfo &a, const GroupMemberInfo &b) -> bool {
                  return a.Order < b.Order;
                });
      return true;
    }
  }
  return false;
}
