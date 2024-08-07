//===------------------ SYCLCreateNVVMAnnotations.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers function metadata to NVVM annotations
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLCreateNVVMAnnotations.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

static void addNVVMMetadata(GlobalValue &GV, StringRef Name, int Operand) {
  Module *M = GV.getParent();
  LLVMContext &Ctx = M->getContext();

  // Get "nvvm.annotations" metadata node
  NamedMDNode *MD = M->getOrInsertNamedMetadata("nvvm.annotations");

  Metadata *MDVals[] = {ConstantAsMetadata::get(&GV), MDString::get(Ctx, Name),
                        ConstantAsMetadata::get(
                            ConstantInt::get(Type::getInt32Ty(Ctx), Operand))};
  // Append metadata to nvvm.annotations
  MD->addOperand(MDNode::get(Ctx, MDVals));
}

static std::optional<uint64_t> getSingleIntMetadata(const MDNode &Node) {
  assert(Node.getNumOperands() == 1 && "Invalid metadata");
  if (auto *C = mdconst::dyn_extract<ConstantInt>(Node.getOperand(0)))
    return C->getZExtValue();
  return std::nullopt;
}

// Returns a three-dimensional array of work-group-related metadata values
// (sizes, etc.). Not all values may be present, but they are assumed to have
// already been index-flipped such that the fastest-moving dimension is
// left-most.
static std::array<std::optional<uint64_t>, 3>
decomposeWGMetadata(const MDNode &Node, const Function &F) {
  assert(Node.getNumOperands() >= 1 && Node.getNumOperands() <= 3 &&
         "Invalid work-group metadata");

  // We optionally check for additional metadata denoting the number of
  // dimensions the user actually specified, ignoring padding. This better
  // preserves the user's intention.
  unsigned NDim = 3;
  if (auto *NDimMD = F.getMetadata("work_group_num_dim"))
    NDim = getSingleIntMetadata(*NDimMD).value_or(3);
  assert(NDim >= 1 && NDim <= 3 && "Invalid work-group dimensionality");

  std::array<std::optional<uint64_t>, 3> Ops;
  for (unsigned I = 0, E = std::min(Node.getNumOperands(), NDim); I != E; I++) {
    if (auto *C = mdconst::dyn_extract<ConstantInt>(Node.getOperand(I)))
      Ops[I] = C->getZExtValue();
  }
  return Ops;
}

PreservedAnalyses
SYCLCreateNVVMAnnotationsPass::run(Module &M, ModuleAnalysisManager &MAM) {
  for (auto &F : M) {
    // Certain functions will never have metadata
    if (F.isIntrinsic())
      continue;

    constexpr static std::pair<const char *, std::array<const char *, 3>>
        WGAnnotations[] = {
            {"reqd_work_group_size", {"reqntidx", "reqntidy", "reqntidx"}},
            {"max_work_group_size", {"maxntidx", "maxntidy", "maxntidz"}}};

    for (auto &[MDName, Annotations] : WGAnnotations) {
      if (MDNode *Node = F.getMetadata(MDName)) {
        auto WGVals = decomposeWGMetadata(*Node, F);
        // Work-group sizes (in NVVM annotations) must be positive and less than
        // INT32_MAX, whereas SYCL can allow for larger work-group sizes (see
        // -fno-sycl-id-queries-fit-in-int). If any dimension is too large for
        // NVPTX, don't emit any annotation at all.
        if (all_of(WGVals, [](std::optional<uint64_t> V) {
              return !V || llvm::isUInt<31>(*V);
            })) {
          if (auto X = WGVals[0])
            addNVVMMetadata(F, Annotations[0], *X);
          if (auto Y = WGVals[1])
            addNVVMMetadata(F, Annotations[1], *Y);
          if (auto Z = WGVals[2])
            addNVVMMetadata(F, Annotations[2], *Z);
        }
      }
    }

    constexpr static std::pair<const char *, const char *>
        SingleValAnnotations[] = {{"min_work_groups_per_cu", "minctasm"},
                                  {"max_work_groups_per_mp", "maxclusterrank"}};

    for (auto &[MDName, AnnotationName] : SingleValAnnotations) {
      if (MDNode *Node = F.getMetadata(MDName)) {
        if (auto Val = getSingleIntMetadata(*Node))
          addNVVMMetadata(F, AnnotationName, *Val);
      }
    }
  }

  return PreservedAnalyses::all();
}
