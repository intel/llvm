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
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DebugInfoMetadata.h"
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
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/SYCLLowerIR/UtilsSYCLNativeCPU.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <utility>
#include <vector>

#ifdef NATIVECPU_USE_OCK
#include "compiler/utils/attributes.h"
#include "compiler/utils/builtin_info.h"
#include "compiler/utils/metadata.h"
#endif

using namespace llvm;
using namespace sycl;
using namespace sycl::utils;

namespace {

void fixCallingConv(Function *F) { F->setCallingConv(llvm::CallingConv::C); }

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
  auto NewName = Twine(OldName) + sycl::utils::SYCLNATIVECPUKERNEL;
  const StringRef SubHandlerName = OldName;
  F->setName(NewName);
  FunctionType *FTy = FunctionType::get(
      Type::getVoidTy(Ctx), {NativeCPUArgDescPtrType, StatePtrType}, false);
  auto SubhFCallee = F->getParent()->getOrInsertFunction(SubHandlerName, FTy);
  Function *SubhF = cast<Function>(SubhFCallee.getCallee());

  // Emit function body, unpack kernel args
  auto *KernelTy = F->getFunctionType();
  IRBuilder<> Builder(Ctx);
  BasicBlock *Block = BasicBlock::Create(Ctx, "entry", SubhF);
  Builder.SetInsertPoint(Block);
  unsigned NumArgs = F->getFunctionType()->getNumParams();
  auto *BaseNativeCPUArg = SubhF->getArg(0);
  SmallVector<Value *, 5> KernelArgs;
  const unsigned Inc = StateArgTLS == nullptr ? 1 : 0;
  for (unsigned I = 0; I + Inc < NumArgs; I++) {
    auto *Arg = F->getArg(I);
    // Load the correct NativeCPUDesc and load the pointer from it
    auto *Addr = Builder.CreateGEP(NativeCPUArgDescType, BaseNativeCPUArg,
                                   {Builder.getInt64(I)});
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

static const std::pair<StringRef, StringRef> BuiltinNamesMap[]{
    {"__mux_get_global_id", NativeCPUGlobalId},
    {"__mux_get_global_size", NativeCPUGlobaRange},
    {"__mux_get_global_offset", NativeCPUGlobalOffset},
    {"__mux_get_local_id", NativeCPULocalId},
    {"__mux_get_num_groups", NativeCPUNumGroups},
    {"__mux_get_local_size", NativeCPUWGSize},
    {"__mux_get_group_id", NativeCPUWGId},
    {"__mux_set_num_sub_groups", NativeCPUSetNumSubgroups},
    {"__mux_set_sub_group_id", NativeCPUSetSubgroupId},
    {"__mux_set_max_sub_group_size", NativeCPUSetMaxSubgroupSize},
    {"__mux_set_local_id", NativeCPUSetLocalId}};

static constexpr unsigned int NativeCPUGlobalAS = 1;
static constexpr char StateTypeName[] = "struct.__nativecpu_state";

static Type *getStateType(Module &M) {
  // %struct.__nativecpu_state = type { [3 x i64], [3 x i64], [3 x i64], [3 x
  // i64], [3 x i64], [3 x i64], [3 x i64] } Check that there's no
  // __nativecpu_state type
  auto Types = M.getIdentifiedStructTypes();
  bool HasStateT =
      llvm::any_of(Types, [](auto T) { return T->getName() == StateTypeName; });
  if (HasStateT)
    report_fatal_error("Native CPU state unexpectedly found in the module.");
  auto &Ctx = M.getContext();
  auto *I64Ty = Type::getInt64Ty(Ctx);
  auto *Array3dTy = ArrayType::get(I64Ty, 3);
  std::array<Type *, 7> Elements;
  Elements.fill(Array3dTy);
  auto *StateType = StructType::create(Ctx, StateTypeName);
  StateType->setBody(Elements);
  return StateType;
}

static const StringMap<unsigned> OffsetMap{
    {NativeCPUGlobalId, 0},    {NativeCPUGlobaRange, 1},
    {NativeCPUWGSize, 2},      {NativeCPUWGId, 3},
    {NativeCPULocalId, 4},     {NativeCPUNumGroups, 5},
    {NativeCPUGlobalOffset, 6}};

static Function *addSetLocalIdFunc(Module &M, StringRef Name, Type *StateType) {
  /*
  void __dpcpp_nativecpu_set_local_id(unsigned dim, size_t value,
                                 __nativecpu_state *s) {
    s->MLocal_id[dim] = value;
    s->MGlobal_id[dim] = s->MWorkGroup_size[dim] * s->MWorkGroup_id[dim] +
                         s->MLocal_id[dim] + s->MGlobalOffset[dim];
  }
  */
  auto &Ctx = M.getContext();
  Type *I64Ty = Type::getInt64Ty(Ctx);
  Type *I32Ty = Type::getInt32Ty(Ctx);
  Type *RetTy = Type::getVoidTy(Ctx);
  Type *DimTy = I32Ty;
  Type *ValTy = I64Ty;
  Type *PtrTy = PointerType::get(Ctx, NativeCPUGlobalAS);
  FunctionType *FTy = FunctionType::get(RetTy, {DimTy, ValTy, PtrTy}, false);
  auto FCallee = M.getOrInsertFunction(Name, FTy);
  auto *F = cast<Function>(FCallee.getCallee());
  IRBuilder<> Builder(Ctx);
  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  Builder.SetInsertPoint(BB);
  auto *StatePtr = F->getArg(2);
  auto *IdxProm = Builder.CreateZExt(F->getArg(0), DimTy, "idxprom");
  auto *Zero = ConstantInt::get(I64Ty, 0);
  auto *Offset = ConstantInt::get(I32Ty, OffsetMap.at(NativeCPULocalId));
  auto *GEP = Builder.CreateGEP(StateType, StatePtr, {Zero, Offset, IdxProm});
  // store local id
  auto *Val = F->getArg(1);
  Builder.CreateStore(Val, GEP);
  // update global id
  auto loadHelper = [&](const char *BTName) {
    auto *Offset = ConstantInt::get(I32Ty, OffsetMap.at(BTName));
    auto *Addr =
        Builder.CreateGEP(StateType, StatePtr, {Zero, Offset, IdxProm});
    auto *Load = Builder.CreateLoad(I64Ty, Addr);
    return Load;
  };
  auto *WGId = loadHelper(NativeCPUWGId);
  auto *WGSize = loadHelper(NativeCPUWGSize);
  auto *GlobalOffset = loadHelper(NativeCPUGlobalOffset);
  auto *Mul = Builder.CreateMul(WGId, WGSize);
  auto *GId = Builder.CreateAdd(Builder.CreateAdd(Mul, GlobalOffset), Val);
  auto *GIdOffset = ConstantInt::get(I32Ty, OffsetMap.at(NativeCPUGlobalId));
  auto *GIdAddr =
      Builder.CreateGEP(StateType, StatePtr, {Zero, GIdOffset, IdxProm});
  Builder.CreateStore(GId, GIdAddr);
  Builder.CreateRetVoid();
  return F;
}

static Function *addGetFunc(Module &M, StringRef Name, Type *StateType) {
  auto &Ctx = M.getContext();
  Type *I64Ty = Type::getInt64Ty(Ctx);
  Type *I32Ty = Type::getInt32Ty(Ctx);
  Type *RetTy = I64Ty;
  Type *DimTy = I32Ty;
  Type *PtrTy = PointerType::get(Ctx, NativeCPUGlobalAS);
  static FunctionType *FTy = FunctionType::get(RetTy, {DimTy, PtrTy}, false);
  auto FCallee = M.getOrInsertFunction(Name, FTy);
  auto *F = cast<Function>(FCallee.getCallee());
  IRBuilder<> Builder(Ctx);
  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  Builder.SetInsertPoint(BB);
  auto *IdxProm = Builder.CreateZExt(F->getArg(0), DimTy, "idxprom");
  auto *Zero = ConstantInt::get(I64Ty, 0);
  auto *Offset = ConstantInt::get(I32Ty, OffsetMap.at(Name));
  auto *GEP =
      Builder.CreateGEP(StateType, F->getArg(1), {Zero, Offset, IdxProm});
  auto *Load = Builder.CreateLoad(I64Ty, GEP);
  Builder.CreateRet(Load);
  return F;
}

static Function *addReplaceFunc(Module &M, StringRef Name, Type *StateType) {
  Function *Res;
  const char GetPrefix[] = "__dpcpp_nativecpu_get";
  if (Name.starts_with(GetPrefix)) {
    Res = addGetFunc(M, Name, StateType);
  } else if (Name == NativeCPUSetLocalId) {
    Res = addSetLocalIdFunc(M, Name, StateType);
  } else {
    // the other __dpcpp_nativecpu_set* builtins are subgroup-related and
    // not supported yet, emit empty functions for now.
    auto &Ctx = M.getContext();
    Type *I32Ty = Type::getInt32Ty(Ctx);
    Type *RetTy = Type::getVoidTy(Ctx);
    Type *ValTy = I32Ty;
    Type *PtrTy = PointerType::get(Ctx, NativeCPUGlobalAS);
    static FunctionType *FTy = FunctionType::get(RetTy, {ValTy, PtrTy}, false);
    auto FCallee = M.getOrInsertFunction(Name, FTy);
    auto *F = cast<Function>(FCallee.getCallee());
    IRBuilder<> Builder(Ctx);
    BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
    Builder.SetInsertPoint(BB);
    Builder.CreateRetVoid();
    Res = F;
  }
  Res->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
  return Res;
}

static Function *getReplaceFunc(Module &M, StringRef Name, Type *StateType) {
  Function *F = M.getFunction(Name);
  if (!F)
    return addReplaceFunc(M, Name, StateType);
  assert(F && "Error retrieving replace function");
  return F;
}

static Value *getStateArg(Function *F, llvm::Constant *StateTLS) {
  // Todo: we should probably cache the state thread local load here
  // to avoid re-emitting it for each builtin
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
  Type *StateType = getStateType(M);
  // Todo: fix this check since we are emitting the state type in the pass now
  if (!StateType)
    return PreservedAnalyses::all();
  Type *StatePtrType = PointerType::get(StateType, 1);

  CurrentStatePointerTLS = nullptr;

  // check if any of the kernels is called by some other function.
  // This can happen e.g. with OCK, where wrapper functions are
  // created around the original kernel.
  bool KernelIsCalled = false;
  for (auto &K : OldKernels) {
    for (auto &U : K->uses()) {
      if (isa<CallBase>(U.getUser())) {
        KernelIsCalled = true;
      }
    }
  }

  // Then we iterate over all the supported builtins, find the used ones
  llvm::SmallVector<std::pair<llvm::Function *, StringRef>> UsedBuiltins;
  for (const auto &Entry : BuiltinNamesMap) {
    auto *Glob = M.getFunction(Entry.first);
    if (!Glob)
      continue;
    for (const auto &Use : Glob->uses()) {
      auto *I = cast<CallInst>(Use.getUser());
      if (!IsNativeCPUKernel(I->getFunction()) || KernelIsCalled) {
        // only use the threadlocal if we have kernels calling builtins
        // indirectly, or if the kernel is called by some other func.
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
#ifdef NATIVECPU_USE_OCK
    auto Name = compiler::utils::getBaseFnNameOrFnName(*OldF);
    OldF->setName(Name);
    // if vectorization occurred, at this point we have a wrapper function that
    // runs the vectorized kernel and peels using the scalar kernel. We make it
    // so this wrapper steals the original kernel name.
    std::optional<compiler::utils::LinkMetadataResult> veczR =
        compiler::utils::parseVeczToOrigFnLinkMetadata(*OldF);
    if (veczR) {
      auto ScalarF = veczR.value().first;
      OldF->takeName(ScalarF);
      ScalarF->setName(OldF->getName() + "_scalar");
    } else if (Name != OldF->getName()) {
      auto RealKernel = M.getFunction(Name);
      if (RealKernel) {
        // the real kernel was not inlined in the wrapper, steal its name
        OldF->takeName(RealKernel);
      } else {
        // the real kernel has been inlined, just use the name
        OldF->setName(Name);
      }
    }
#endif
    auto *NewF =
        cloneFunctionAndAddParam(OldF, StatePtrType, CurrentStatePointerTLS);
    NewF->takeName(OldF);
    OldF->replaceAllUsesWith(NewF);
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
    SmallVector<std::pair<Instruction *, Instruction *>> ToRemove;
    Function *const Glob = Entry.first;
    for (const auto &Use : Glob->uses()) {
      auto *ReplaceFunc = getReplaceFunc(M, Entry.second, StateType);
      auto I = cast<CallInst>(Use.getUser());
      SmallVector<Value *> Args(I->arg_begin(), I->arg_end());
      Args.push_back(getStateArg(I->getFunction(), CurrentStatePointerTLS));
      auto *NewI = CallInst::Create(ReplaceFunc->getFunctionType(), ReplaceFunc,
                                    Args, "", I);
      // If the parent function has debug info, we need to make sure that the
      // CallInstructions in it have debug info, otherwise we end up with
      // invalid IR after inlining.
      if (I->getFunction()->hasMetadata("dbg")) {
        if (!I->getMetadata("dbg")) {
          I->setDebugLoc(DILocation::get(M.getContext(), 0, 0,
                                         I->getFunction()->getSubprogram()));
        }
        NewI->setDebugLoc(I->getDebugLoc());
      }
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

#ifdef NATIVECPU_USE_OCK
  // Define __mux_mem_barrier here using the OCK
  compiler::utils::BuiltinInfo BI;
  for (auto &F : M) {
    if (F.getName() == compiler::utils::MuxBuiltins::mem_barrier) {
      BI.defineMuxBuiltin(compiler::utils::BaseBuiltinID::eMuxBuiltinMemBarrier,
                          M);
    }
  }
  // if we find calls to mux barrier now, it means that we had SYCL_EXTERNAL
  // functions that called __mux_work_group_barrier, which didn't get processed
  // by the WorkItemLoop pass. This means that the actual function call has been
  // inlined into the kernel, and the call to __mux_work_group_barrier has been
  // removed in the inlined call, but not in the original function. The original
  // function will not be executed (since it has been inlined) and so we can
  // just define __mux_work_group_barrier as a no-op to avoid linker errors.
  // Todo: currently we can't remove the function here even if it has no uses,
  // because we may still emit a declaration for it in the offload-wrapper.
  auto BarrierF =
      M.getFunction(compiler::utils::MuxBuiltins::work_group_barrier);
  if (BarrierF && BarrierF->isDeclaration()) {
    IRBuilder<> Builder(M.getContext());
    auto BB = BasicBlock::Create(M.getContext(), "noop", BarrierF);
    Builder.SetInsertPoint(BB);
    Builder.CreateRetVoid();
  }
#endif
  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
