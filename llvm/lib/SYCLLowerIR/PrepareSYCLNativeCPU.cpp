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
// * Materializes spirv builtins.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/PrepareSYCLNativeCPU.h"
#include "llvm/BinaryFormat/MsgPack.h"
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
#include "llvm/IR/Operator.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/SYCLLowerIR/UtilsSYCLNativeCPU.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <functional>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#ifdef NATIVECPU_USE_OCK
#include "compiler/utils/attributes.h"
#include "compiler/utils/builtin_info.h"
#endif

using namespace llvm;
using namespace sycl;
using namespace sycl::utils;

namespace {

void fixCallingConv(Function *F) {
  F->setCallingConv(llvm::CallingConv::C);
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

/*
uint32_t NumSubGroups,
         SubGroup_id,
         SubGroup_local_id,
         SubGroup_size;*/

#define NCPUPREFIX "__dpcpp_nativecpu"
         // subgroup getters
constexpr const char* NativeCPUNumSubGroups = NCPUPREFIX "_get_num_sub_groups";
constexpr const char* NativeCPUSubGroup_id = NCPUPREFIX "_get_sub_group_id";
constexpr const char* NativeCPUSubGroup_local_id = NCPUPREFIX "_get_sub_group_local_id";
constexpr const char* NativeCPUSubGroup_size = NCPUPREFIX "_get_sub_group_size";
constexpr const char* NativeCPUSubGroup_max_size = NCPUPREFIX "_get_max_sub_group_size";
// subgroup setters
constexpr const char* NativeCPUSetNumSubGroups = NCPUPREFIX "_set_num_sub_groups";
constexpr const char* NativeCPUSetSubGroup_id = NCPUPREFIX "_set_sub_group_id";
constexpr const char* NativeCPUSetSubGroup_size = NCPUPREFIX "_set_sub_group_size";

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
    {"__mux_set_local_id", NativeCPUSetLocalId},
    // subgroup getters
    {"__mux_get_sub_group_local_id", NativeCPUSubGroup_local_id},
    {"__mux_get_max_sub_group_size", NativeCPUSubGroup_max_size},
    {"__mux_get_sub_group_id", NativeCPUSubGroup_id},
    {"__mux_get_num_sub_groups", NativeCPUNumSubGroups},
    {"__mux_get_sub_group_size", NativeCPUSubGroup_size}
};

static constexpr unsigned int NativeCPUGlobalAS = 1;
static constexpr char StateTypeName[] = "struct.__nativecpu_state";

static Type *getStateType(Module &M) {
  // %struct.__nativecpu_state = type { [3 x i64], [3 x i64], [3 x i64], [3 x
  // i64], [3 x i64], [3 x i64], [3 x i64] } Check that there's no
  // __nativecpu_state type
  auto Types = M.getIdentifiedStructTypes();
  auto str = llvm::find_if(Types, [](auto T) { return T->getName() == StateTypeName; });
  if (str == Types.end()) {
    //report_fatal_error("Native CPU state unexpectedly found in the module.");
  } else if ((*str)->isStructTy()) {
    // state struct should come from linked builtin bc file
    return *str;
  }
  // old code, to be removed
  bool HasStateT =
      llvm::any_of(Types, [](auto T) { return T->getName() == StateTypeName; });
  if (HasStateT)
    report_fatal_error("Native CPU state unexpectedly found in the module.");
  auto &Ctx = M.getContext();
  auto *I64Ty = Type::getInt64Ty(Ctx);
  auto *Array3dTy = ArrayType::get(I64Ty, 3);
  std::array<Type *, 7 + 4> Elements;
  std::fill(Elements.begin(), Elements.begin() + 7, Array3dTy);
  auto* I32Ty = Type::getInt32Ty(Ctx);
  std::fill(Elements.begin() + 7, Elements.begin() + 11, I32Ty);
  auto *StateType = StructType::create(Ctx, StateTypeName);
  StateType->setBody(Elements);
  return StateType;
}

static const StringMap<unsigned> OffsetMap{
    {NativeCPUGlobalId, 0},    {NativeCPUGlobaRange, 1},
    {NativeCPUWGSize, 2},      {NativeCPUWGId, 3},
    {NativeCPULocalId, 4},     {NativeCPUNumGroups, 5},
    {NativeCPUGlobalOffset, 6},
// Subgroup offsets
    {NativeCPUNumSubGroups, 7}, {NativeCPUSetNumSubGroups, 7},
    {NativeCPUSubGroup_id, 8}, {NativeCPUSetSubGroup_id, 8},
    {NativeCPUSubGroup_local_id, 9},
    {NativeCPUSubGroup_size, 10}, {NativeCPUSetSubGroup_size, 10},
    {NativeCPUSubGroup_max_size, 10}, {NativeCPUSetMaxSubgroupSize, 10} //todo
};

//declare void @__mux_set_num_sub_groups(i32% val) #2
//declare void @__mux_set_sub_group_id(i32 % val) #2
//declare void @__mux_set_max_sub_group_size(i32 % val)

//declare i32 @__mux_get_sub_group_local_id()
//declare i32 @__mux_get_sub_group_id()
//declare i32 @__mux_get_num_sub_groups()
//declare i32 @__mux_get_sub_group_size()


static Function* addSetSubGroupValFunc(Module& M, StringRef Name, Type* StateType) {
  /*
  void __dpcpp_nativecpu_set_num_sub_groups(size_t value,
                                            __nativecpu_state *s) {
    s->Name = value;
  }
  */
  auto& Ctx = M.getContext();
  Type* I32Ty = Type::getInt32Ty(Ctx);
  Type* I64Ty = Type::getInt64Ty(Ctx);
  Type* RetTy = Type::getVoidTy(Ctx);
  Type* PtrTy = PointerType::get(Ctx, NativeCPUGlobalAS);
  static FunctionType* FTy =
    FunctionType::get(RetTy, { I32Ty, PtrTy }, false);
  auto FCallee = M.getOrInsertFunction(Name, FTy);
  auto* F = dyn_cast<Function>(FCallee.getCallee());
  IRBuilder<> Builder(Ctx);
  BasicBlock* BB = BasicBlock::Create(Ctx, "entry", F);
  Builder.SetInsertPoint(BB);
  auto* StatePtr = F->getArg(1);
  auto* Zero = ConstantInt::get(I64Ty, 0);
  auto* Offset = ConstantInt::get(I32Ty, OffsetMap.at(Name));
  auto* GEP = Builder.CreateGEP(StateType, StatePtr, { Zero, Offset });
  // store local id
  auto* Val = F->getArg(0);
  Builder.CreateStore(Val, GEP);
  Builder.CreateRetVoid();
  return F;
}

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
  static FunctionType *FTy =
      FunctionType::get(RetTy, {DimTy, ValTy, PtrTy}, false);
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

static Function* addSubGroupGetFunc(Module& M, StringRef Name, Type* StateType) {
  auto& Ctx = M.getContext();
  Type* I64Ty = Type::getInt64Ty(Ctx);
  Type* I32Ty = Type::getInt32Ty(Ctx);
  Type* RetTy = I32Ty;
  Type* PtrTy = PointerType::get(Ctx, NativeCPUGlobalAS);
  static FunctionType* FTy = FunctionType::get(RetTy, { PtrTy }, false);
  auto FCallee = M.getOrInsertFunction(Name, FTy);
  auto* F = dyn_cast<Function>(FCallee.getCallee());
  IRBuilder<> Builder(Ctx);
  BasicBlock* BB = BasicBlock::Create(Ctx, "entry", F);
  Builder.SetInsertPoint(BB);
  auto* Zero = ConstantInt::get(I64Ty, 0);
  auto* Offset = ConstantInt::get(I32Ty, OffsetMap.at(Name));
  auto* GEP =
    Builder.CreateGEP(StateType, F->getArg(0), { Zero, Offset});
  auto* Load = Builder.CreateLoad(I32Ty, GEP);
  Builder.CreateRet(Load);
  return F;
}

static Function *addReplaceFunc(Module &M, StringRef Name, Type *StateType) {
  Function *Res;
  const char GetPrefix[] = "__dpcpp_nativecpu_get";
  if (Name == NativeCPUNumSubGroups ||
    Name == NativeCPUSubGroup_id ||
    Name == NativeCPUSubGroup_local_id ||
    Name == NativeCPUSubGroup_max_size ||
    Name == NativeCPUSubGroup_size) {
    Res = addSubGroupGetFunc(M, Name, StateType);
  } else if (Name == NativeCPUSetNumSubgroups ||
             Name == NativeCPUSetSubgroupId ||
             Name == NativeCPUSetMaxSubgroupSize) {
    Res = addSetSubGroupValFunc(M, Name, StateType);
  } else if (Name.startswith(GetPrefix)) {
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
  {
    SmallVector<Function *> ToRemove;
    for (auto &F : M) {
      if (F.getCallingConv() == llvm::CallingConv::SPIR_KERNEL)
        OldKernels.push_back(&F);
      else if (F.getNumUses() == 0) {
        if (F.isDeclaration()) {
        } else if (F.hasFnAttribute(llvm::Attribute::AlwaysInline)) {
          StringRef val = F.getFnAttribute("sycl-module-id").getValueAsString();
          if (val.endswith("libdevice/nativecpu_utils.cpp"))
            // We remove all unused, always-inlined functions llvm-linked from
            // the nativecpu device builtin library from the module.
            ToRemove.push_back(&F);
        }
      }
    }
    for (Function *f : ToRemove) {
      f->eraseFromParent();
      ModuleChanged = true;
    }
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

  // Then we iterate over all the supported builtins, find the used ones
  llvm::SmallVector<std::pair<llvm::Function *, StringRef>> UsedBuiltins;
  for (const auto &Entry : BuiltinNamesMap) {
    auto *Glob = M.getFunction(Entry.first);
    if (!Glob)
      continue;
    for (const auto &Use : Glob->uses()) {
      auto I = cast<CallInst>(Use.getUser());
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
  auto cloneAndAddKernel = [&](Function *OldF, bool TakeName) {
    auto *NewF =
        cloneFunctionAndAddParam(OldF, StatePtrType, CurrentStatePointerTLS);
    if (TakeName)
      NewF->takeName(OldF);
    OldF->replaceAllUsesWith(NewF);
    OldF->eraseFromParent();
    NewKernels.push_back(NewF);
    ModuleChanged = true;
  };

#ifdef NATIVECPU_USE_OCK
  {
    // First we find the original kernels that have the same names
    // as the work item loop kernels. If the original kernel is called
    // by the workitem loop kernel, clone it and change its name so
    // it can't clash with the workitem loop kernel.
    SmallVector<Function *> ProcessedKernels;
    for (auto &OldF : OldKernels) {
      auto Name = compiler::utils::getBaseFnNameOrFnName(*OldF);
      if (Name != OldF->getName()) {
        auto RealKernel = M.getFunction(Name);
        if (RealKernel) {
          ProcessedKernels.push_back(RealKernel);
          if (RealKernel->getNumUses() == 0) {
            // todo: check if this kernel can be safely removed
          }
          cloneAndAddKernel(RealKernel, false);
        }
        OldF->setName(Name);
        assert(OldF->getName() == Name);
      }
    }

    for (Function *f : ProcessedKernels)
      OldKernels.erase(std::remove(OldKernels.begin(), OldKernels.end(), f),
                       OldKernels.end());
  }
#endif

  for (auto &OldF : OldKernels) {
    cloneAndAddKernel(OldF, true);
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
        I->setDebugLoc(DILocation::get(M.getContext(), 0, 0,
                                       I->getFunction()->getSubprogram()));
        if (I->getMetadata("dbg"))
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
  // because we may still emit a declaration for in the offload-wrapper.
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
