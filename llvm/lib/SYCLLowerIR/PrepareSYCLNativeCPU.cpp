//===------ PrepareSYCLNativeCPU.cpp - SYCL Native CPU Preparation Pass ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prepares the kernel for SYCL Native CPU:
// * Handles kernel calling convention and attributes.
// * Materializes spirv buitlins.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/PrepareSYCLNativeCPU.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/PassManager.h"
#include "llvm/SYCLLowerIR/SYCLUtils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <functional>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

using namespace llvm;

namespace {

void fixCallingConv(Function *F) {
  F->setCallingConv(llvm::CallingConv::C);
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

// returns the indexes of the used arguments
SmallVector<unsigned> getUsedIndexes(const Function *F, bool useTLS) {
  SmallVector<unsigned> Res;
  auto UsedNode = F->getMetadata("sycl_kernel_omit_args");
  if (!UsedNode) {
    // the metadata node is not available if -fenable-sycl-dae
    // was not set; set everything to true
    // Exclude one arg because we already added the state ptr
    const unsigned first = useTLS ? 0 : 1;
    for (unsigned I = 0, NumP = F->getFunctionType()->getNumParams();
         I + first < NumP; I++) {
      Res.push_back(I);
    }
    return Res;
  }
  auto NumOperands = UsedNode->getNumOperands();
  for (unsigned I = 0; I < NumOperands; I++) {
    auto &Op = UsedNode->getOperand(I);
    if (auto CAM = dyn_cast<ConstantAsMetadata>(Op.get())) {
      if (auto Const = dyn_cast<ConstantInt>(CAM->getValue())) {
        auto Val = Const->getValue();
        if (!Val.getBoolValue()) {
          Res.push_back(I);
        }
      } else {
        report_fatal_error("Unable to retrieve constant int from "
                           "sycl_kernel_omit_args metadata node");
      }
    } else {
      report_fatal_error(
          "Error while processing sycl_kernel_omit_args metadata node");
    }
  }
  return Res;
}

void emitSubkernelForKernel(Function *F, Type *NativeCPUArgDescType,
                            Type *StatePtrType, llvm::Constant *StateArgTLS) {
  LLVMContext &Ctx = F->getContext();
  Type *NativeCPUArgDescPtrType = PointerType::getUnqual(NativeCPUArgDescType);

  // Create function signature
  // Todo: we need to ensure that the kernel name is not mangled as a type
  // name, otherwise this may lead to runtime failures due to *weird*
  // codegen/linking behaviour, we change the name of the kernel, and the
  // subhandler steals its name, we add a suffix to the subhandler later
  // on when lowering the device module
  std::string OldName = F->getName().str();
  auto NewName = Twine(OldName) + ".NativeCPUKernel";
  const StringRef SubHandlerName = OldName;
  F->setName(NewName);
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(Ctx), {NativeCPUArgDescPtrType, StatePtrType}, false);
  auto SubhFCallee = F->getParent()->getOrInsertFunction(SubHandlerName, FTy);
  Function *SubhF = cast<Function>(SubhFCallee.getCallee());

  // Emit function body, unpack kernel args
  auto UsedIndexes = getUsedIndexes(F, StateArgTLS);
  auto *KernelTy = F->getFunctionType();
  // assert(UsedIndexes.size() + 1 == KernelTy->getNumParams() && "mismatch
  // between number of params and used args");
  IRBuilder<> Builder(Ctx);
  BasicBlock *Block = BasicBlock::Create(Ctx, "entry", SubhF);
  Builder.SetInsertPoint(Block);
  unsigned NumArgs = UsedIndexes.size();
  auto *BaseNativeCPUArg = SubhF->getArg(0);
  SmallVector<Value *, 5> KernelArgs;
  for (unsigned I = 0; I < NumArgs; I++) {
    auto *Arg = F->getArg(I);
    auto UsedI = UsedIndexes[I];
    // Load the correct NativeCPUDesc and load the pointer from it
    auto *Addr = Builder.CreateGEP(NativeCPUArgDescType, BaseNativeCPUArg,
                                   {Builder.getInt64(UsedI)});
    if (Arg->getType()->isPointerTy()) {
      // If the arg is a pointer, just use it
      auto *Load = Builder.CreateLoad(Arg->getType(), Addr);
      KernelArgs.push_back(Load);
    } else {
      // Otherwise, load the scalar value and use that
      auto *Load = Builder.CreateLoad(PointerType::getUnqual(Ctx), Addr);
      auto *Scalar = Builder.CreateLoad(Arg->getType(), Load);
      KernelArgs.push_back(Scalar);
    }
  }

  // Call the kernel
  // Add the nativecpu state as arg
  if (StateArgTLS) {
    Value *Addr = Builder.CreateThreadLocalAddress(StateArgTLS);
    Builder.CreateStore(SubhF->getArg(1), Addr);
  } else
    KernelArgs.push_back(SubhF->getArg(1));

  Builder.CreateCall(KernelTy, F, KernelArgs);
  Builder.CreateRetVoid();

  fixCallingConv(F);
  fixCallingConv(SubhF);
  // Add sycl-module-id attribute
  // Todo: we may want to copy other attributes to the subhandler,
  // but we can't simply use setAttributes(F->getAttributes) since
  // the function signatures are different
  if (F->hasFnAttribute(sycl::utils::ATTR_SYCL_MODULE_ID)) {
    Attribute MId = F->getFnAttribute(sycl::utils::ATTR_SYCL_MODULE_ID);
    SubhF->addFnAttr("sycl-module-id", MId.getValueAsString());
  }
}

// Clones the function and returns a new function with a new argument on type T
// added as last argument
Function *cloneFunctionAndAddParam(Function *OldF, Type *T,
                                   llvm::Constant *StateArgTLS) {
  auto *OldT = OldF->getFunctionType();
  auto *RetT = OldT->getReturnType();

  std::vector<Type *> Args;
  for (auto *Arg : OldT->params()) {
    Args.push_back(Arg);
  }
  if (StateArgTLS == nullptr)
    Args.push_back(T);
  auto *NewT = FunctionType::get(RetT, Args, OldF->isVarArg());
  auto *NewF = Function::Create(NewT, OldF->getLinkage(), OldF->getName(),
                                OldF->getParent());
  // Copy the old function's attributes
  NewF->setAttributes(OldF->getAttributes());

  // Map old arguments to new arguments
  ValueToValueMapTy VMap;
  for (const auto &Pair : llvm::zip(OldF->args(), NewF->args())) {
    auto &OldA = std::get<0>(Pair);
    auto &NewA = std::get<1>(Pair);
    VMap[&OldA] = &NewA;
  }

  SmallVector<ReturnInst *, 1> ReturnInst;
  if (!OldF->isDeclaration())
    CloneFunctionInto(NewF, OldF, VMap,
                      CloneFunctionChangeType::LocalChangesOnly, ReturnInst);
  return NewF;
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
        GEN_xyz(__spirv_GlobalInvocationId, 28, __dpcpp_nativecpu_global_id),
        GEN_xyz(__spirv_GlobalSize, 20, __dpcpp_nativecpu_global_range),
        GEN_xyz(__spirv_GlobalOffset, 22, __dpcpp_nativecpu_get_global_offset),
        GEN_xyz(__spirv_LocalInvocationId, 27, __dpcpp_nativecpu_get_local_id),
        GEN_xyz(__spirv_NumWorkgroups, 23, __dpcpp_nativecpu_get_num_groups),
        GEN_xyz(__spirv_WorkgroupSize, 23, __dpcpp_nativecpu_get_wg_size),
        GEN_xyz(__spirv_WorkgroupId, 21, __dpcpp_nativecpu_get_wg_id),
};

static inline bool IsForVisualStudio(StringRef triple_str) {
  llvm::Triple triple(triple_str);
  return triple.isKnownWindowsMSVCEnvironment();
}

static Function *getReplaceFunc(const Module &M, StringRef Name) {
  Function *F = M.getFunction(Name);
  assert(F && "Error retrieving replace function");
  return F;
}

static Value *getStateArg(Function *F, llvm::Constant *StateTLS) {
  if (StateTLS) {
    IRBuilder<> BB(&*F->getEntryBlock().getFirstInsertionPt());
    llvm::Value *V = BB.CreateThreadLocalAddress(StateTLS);
    return BB.CreateLoad(StateTLS->getType(), V);
  }
  auto *FT = F->getFunctionType();
  return F->getArg(FT->getNumParams() - 1);
}

static inline bool IsNativeCPUKernel(const Function *F) {
  return F->getCallingConv() == llvm::CallingConv::SPIR_KERNEL;
}
static constexpr StringRef STATE_TLS_NAME = "_ZL28nativecpu_thread_local_state";

} // namespace
static llvm::Constant *CurrentStatePointerTLS;
PreservedAnalyses PrepareSYCLNativeCPUPass::run(Module &M,
                                                ModuleAnalysisManager &MAM) {
  bool ModuleChanged = false;
  SmallVector<Function *> OldKernels;
  for (auto &F : M) {
    if (F.getCallingConv() == llvm::CallingConv::SPIR_KERNEL)
      OldKernels.push_back(&F);
  }

  // Materialize builtins
  // First we add a pointer to the Native CPU state as arg to all the
  // kernels.
  Type *StateType =
      StructType::getTypeByName(M.getContext(), "struct.__nativecpu_state");
  if (!StateType)
    return PreservedAnalyses::all();
  Type *StatePtrType = PointerType::get(StateType, 1);

  CurrentStatePointerTLS = nullptr;
  const bool VisualStudioMangling = IsForVisualStudio(M.getTargetTriple());

  // Then we iterate over all the supported builtins, find the used ones
  llvm::SmallVector<
      std::pair<llvm::Function *, const std::pair<StringRef, unsigned int> &>>
      UsedBuiltins;
  for (const auto &Entry : BuiltinNamesMap) {
    auto *Glob = M.getFunction(VisualStudioMangling ? Entry.first.second
                                                    : Entry.first.first);
    if (!Glob)
      continue;
    for (const auto &Use : Glob->uses()) {
      auto I = dyn_cast<CallInst>(Use.getUser());
      if (!I)
        report_fatal_error("Unsupported Value in SYCL Native CPU\n");
      if (!IsNativeCPUKernel(I->getFunction())) {
        // only use the threadlocal if we have kernels calling builtins
        // indirectly
        if (CurrentStatePointerTLS == nullptr)
          CurrentStatePointerTLS = M.getOrInsertGlobal(
              STATE_TLS_NAME, StatePtrType, [&M, StatePtrType]() {
                GlobalVariable *p = new GlobalVariable(
                    M, StatePtrType, false,
                    GlobalValue::LinkageTypes::
                        InternalLinkage /*todo: make external linkage to share
                                           variable*/
                    ,
                    nullptr, STATE_TLS_NAME, nullptr,
                    GlobalValue::ThreadLocalMode::GeneralDynamicTLSModel, 1,
                    false);
                p->setInitializer(Constant::getNullValue(StatePtrType));
                return p;
              });
        break;
      }
    }
    UsedBuiltins.push_back({Glob, Entry.second});
  }

  SmallVector<Function *> NewKernels;
  for (auto &OldF : OldKernels) {
    auto *NewF =
        cloneFunctionAndAddParam(OldF, StatePtrType, CurrentStatePointerTLS);
    NewF->takeName(OldF);
    OldF->eraseFromParent();
    NewKernels.push_back(NewF);
    ModuleChanged = true;
  }

  StructType *NativeCPUArgDescType =
      StructType::create({PointerType::getUnqual(M.getContext())});
  for (auto &NewK : NewKernels) {
    emitSubkernelForKernel(NewK, NativeCPUArgDescType, StatePtrType,
                           CurrentStatePointerTLS);
  }

  // Then we iterate over all used builtins and
  // replace them with calls to our Native CPU functions.
  for (const auto &Entry : UsedBuiltins) {
    SmallVector<Instruction *> ToRemove;
    Function *const Glob = Entry.first;
    for (const auto &Use : Glob->uses()) {
      auto *ReplaceFunc = getReplaceFunc(M, Entry.second.first);
      auto I = dyn_cast<CallInst>(Use.getUser());
      if (!I)
        report_fatal_error("Unsupported Value in SYCL Native CPU\n");
      auto *Arg = ConstantInt::get(Type::getInt32Ty(M.getContext()),
                                   Entry.second.second);
      auto *NewI = CallInst::Create(
          ReplaceFunc->getFunctionType(), ReplaceFunc,
          {Arg, getStateArg(I->getFunction(), CurrentStatePointerTLS)},
          "ncpu_call", I);
      if (I->getMetadata("dbg"))
        NewI->setDebugLoc(I->getDebugLoc());
      I->replaceAllUsesWith(NewI);
      ToRemove.push_back(I);
    }

    for (auto &El : ToRemove)
      El->eraseFromParent();

    // Finally, we erase the builtin from the module
    Glob->eraseFromParent();
  }

  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
