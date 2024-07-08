//===-- ConvertToMuxBuiltinsSYCLNativeCPU.cpp - Convert to Mux Builtins ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts SPIRV builtins to Mux builtins used by the oneAPI Construction
// Kit for SYCL Native CPU
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/ConvertToMuxBuiltinsSYCLNativeCPU.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Triple.h"
#include <map>

using namespace llvm;

namespace {

static void fixFunctionAttributes(Function *F) {
  // The frame-pointer=all and the "byval" attributes lead to code generation
  // that conflicts with the Kernel declaration that we emit in the Native CPU
  // helper header (in which all the kernel argument are void* or scalars).
  auto AttList = F->getAttributes();
  for (unsigned ArgNo = 0; ArgNo < F->getFunctionType()->getNumParams();
       ArgNo++) {
    if (AttList.hasParamAttr(ArgNo, Attribute::AttrKind::ByVal)) {
      AttList = AttList.removeParamAttribute(F->getContext(), ArgNo,
                                             Attribute::AttrKind::ByVal);
    }
  }
  F->setAttributes(AttList);
  F->addFnAttr("frame-pointer", "none");
}

// Helper macros for constructing builtin MS names
#define GENMS1(builtin_str) "?" builtin_str "@@YA_KXZ"

#define GEN_IT_proc(b_str, len) "_Z" #len b_str "v"
#define GEN_p(b_str, len, ncpu_bstr, num)                                      \
  {                                                                            \
    {([]() { static_assert(sizeof(b_str) == len + 1); },                       \
      GEN_IT_proc(b_str, len)),                                                \
     GENMS1(b_str)},                                                           \
    {                                                                          \
      ncpu_bstr, num                                                           \
    }                                                                          \
  }
#define GEN_xyz(b_name, len, ncpu_name)                                        \
  GEN_p(#b_name "_x", len, #ncpu_name, 0),                                     \
      GEN_p(#b_name "_y", len, #ncpu_name, 1),                                 \
      GEN_p(#b_name "_z", len, #ncpu_name, 2)

// Todo: add support for more SPIRV builtins here
static const std::pair<std::pair<StringRef, StringRef>,
                       std::pair<StringRef, unsigned int>>
    BuiltinNamesMap[] = {
        GEN_xyz(__spirv_GlobalInvocationId, 28, __mux_get_global_id),
        GEN_xyz(__spirv_GlobalSize, 20, __mux_get_global_size),
        GEN_xyz(__spirv_GlobalOffset, 22, __mux_get_global_offset),
        GEN_xyz(__spirv_LocalInvocationId, 27, __mux_get_local_id),
        GEN_xyz(__spirv_NumWorkgroups, 23, __mux_get_num_groups),
        GEN_xyz(__spirv_WorkgroupSize, 23, __mux_get_local_size),
        GEN_xyz(__spirv_WorkgroupId, 21, __mux_get_group_id),
};

static inline bool isForVisualStudio(StringRef TripleStr) {
  llvm::Triple Triple(TripleStr);
  return Triple.isKnownWindowsMSVCEnvironment();
}

static constexpr char SPIRVBarrier[] = "_Z22__spirv_ControlBarrierjjj";
static constexpr char MuxBarrier[] = "__mux_work_group_barrier";

Function *getReplaceFunc(Module &M, StringRef Name) {
  LLVMContext &Ctx = M.getContext();
  auto *MuxFTy =
      FunctionType::get(Type::getInt64Ty(Ctx), {Type::getInt32Ty(Ctx)}, false);
  auto F = M.getOrInsertFunction(Name, MuxFTy);
  return cast<Function>(F.getCallee());
}

Function *getMuxBarrierFunc(Module &M) {
  // void __mux_work_group_barrier(i32 %id, i32 %scope, i32 %semantics)
  LLVMContext &Ctx = M.getContext();
  auto *Int32Ty = Type::getInt32Ty(Ctx);
  auto *MuxFTy = FunctionType::get(Type::getVoidTy(Ctx),
                                   {Int32Ty, Int32Ty, Int32Ty}, false);
  auto FCallee = M.getOrInsertFunction(MuxBarrier, MuxFTy);
  auto *F = dyn_cast<Function>(FCallee.getCallee());
  if (!F) {
    report_fatal_error("Error while inserting mux builtins");
  }
  return F;
}

static constexpr const char *MuxKernelAttrName = "mux-kernel";

void setIsKernelEntryPt(Function &F) {
  F.addFnAttr(MuxKernelAttrName, "entry-point");
}

bool replaceBarriers(Module &M) {
  // DPC++ emits
  //__spirv_ControlBarrier(__spv::Scope Execution, __spv::Scope Memory,
  //                       uint32_t Semantics) noexcept;
  // OCK expects  void __mux_work_group_barrier(i32 %id, i32 %scope, i32
  // %semantics)
  // __spv::Scope is
  // enum Flag : uint32_t {
  //   CrossDevice = 0,
  //   Device = 1,
  //   Workgroup = 2,
  //   Subgroup = 3,
  //   Invocation = 4,
  // };
  auto *SPIRVBarrierFunc = M.getFunction(SPIRVBarrier);
  if (!SPIRVBarrierFunc) {
    // No barriers are found, just return
    return false;
  }
  static auto *MuxBarrierFunc = getMuxBarrierFunc(M);
  SmallVector<std::pair<Instruction *, Instruction *>> ToRemove;
  auto *Zero = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
  for (auto &Use : SPIRVBarrierFunc->uses()) {
    auto *I = dyn_cast<CallInst>(Use.getUser());
    if (!I)
      report_fatal_error("Unsupported Value in SYCL Native CPU\n");
    SmallVector<Value *, 3> Args{Zero, I->getArgOperand(0),
                                 I->getArgOperand(2)}; // todo: check how the
                                                       // args map to each other
    auto *NewI = CallInst::Create(MuxBarrierFunc->getFunctionType(),
                                  MuxBarrierFunc, Args, "", I);
    ToRemove.push_back(std::pair(I, NewI));
  }

  for (auto &El : ToRemove) {
    auto OldI = El.first;
    auto NewI = El.second;
    OldI->replaceAllUsesWith(NewI);
    OldI->eraseFromParent();
  }

  SPIRVBarrierFunc->eraseFromParent();

  return true;
}

} // namespace

PreservedAnalyses
ConvertToMuxBuiltinsSYCLNativeCPUPass::run(Module &M,
                                           ModuleAnalysisManager &MAM) {
  bool ModuleChanged = false;
  for (auto &F : M) {
    if (F.getCallingConv() == llvm::CallingConv::SPIR_KERNEL) {
      fixFunctionAttributes(&F);
      setIsKernelEntryPt(F);
    }
  }
  const bool VisualStudioMangling = isForVisualStudio(M.getTargetTriple());

  // Then we iterate over all the supported builtins, find their uses and
  // replace them with calls to our Native CPU functions.
  for (auto &Entry : BuiltinNamesMap) {
    auto *Glob = M.getFunction(VisualStudioMangling ? Entry.first.second
                                                    : Entry.first.first);
    if (!Glob)
      continue;
    auto *ReplaceFunc = getReplaceFunc(M, Entry.second.first);
    SmallVector<std::pair<Instruction *, Instruction *>> ToRemove;
    for (auto &Use : Glob->uses()) {
      auto *I = dyn_cast<CallInst>(Use.getUser());
      if (!I)
        report_fatal_error("Unsupported Value in SYCL Native CPU\n");
      auto *Arg = ConstantInt::get(Type::getInt32Ty(M.getContext()),
                                   Entry.second.second);
      auto *NewI = CallInst::Create(ReplaceFunc->getFunctionType(), ReplaceFunc,
                                    {Arg}, "mux_call", I);
      ModuleChanged = true;
      ToRemove.push_back(std::make_pair(I, NewI));
    }

    for (auto &El : ToRemove) {
      auto OldI = El.first;
      auto NewI = El.second;
      OldI->replaceAllUsesWith(NewI);
      OldI->eraseFromParent();
    }

    // Finally, we erase the builtin from the module
    Glob->eraseFromParent();
  }

  ModuleChanged |= replaceBarriers(M);
  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
