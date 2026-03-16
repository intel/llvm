//===--------- GlobalOffset.cpp - Global Offset Support for CUDA --------- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/GlobalOffset.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/SYCLLowerIR/TargetHelpers.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <deque>

using namespace llvm;

#define DEBUG_TYPE "globaloffset"

#include "llvm/Support/CommandLine.h"
static cl::opt<bool>
    EnableGlobalOffset("enable-global-offset", cl::Hidden, cl::init(true),
                       cl::desc("Enable SYCL global offset pass"));
namespace llvm {
ModulePass *createGlobalOffsetPass();
void initializeGlobalOffsetPass(PassRegistry &);
} // end namespace llvm

// Legacy PM wrapper.
namespace {
class GlobalOffsetLegacy : public ModulePass {
public:
  static char ID;
  GlobalOffsetLegacy() : ModulePass(ID) {
    initializeGlobalOffsetLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    ModuleAnalysisManager DummyMAM;
    auto PA = Impl.run(M, DummyMAM);
    return !PA.areAllPreserved();
  }

private:
  GlobalOffsetPass Impl;
};
} // namespace

char GlobalOffsetLegacy::ID = 0;
INITIALIZE_PASS(GlobalOffsetLegacy, "globaloffset",
                "SYCL Add Implicit Global Offset", false, false)

ModulePass *llvm::createGlobalOffsetPassLegacy() {
  return new GlobalOffsetLegacy();
}

static void validateKernels(Module &M, TargetHelpers::KernelCache &KCache) {
  SmallVector<GlobalValue *, 4> Vec;
  collectUsedGlobalVariables(M, Vec, /*CompilerUsed=*/false);
  collectUsedGlobalVariables(M, Vec, /*CompilerUsed=*/true);
  SmallPtrSet<GlobalValue *, 8u> Used = {Vec.begin(), Vec.end()};

  auto HasUseOtherThanLLVMUsed = [&Used](GlobalValue *GV) {
    if (GV->use_empty())
      return false;
    return !GV->hasOneUse() || !Used.count(GV);
  };

  for (auto &F : KCache) {
    if (HasUseOtherThanLLVMUsed(F))
      llvm_unreachable("Kernel entry point can't have uses.");
  }
}

void GlobalOffsetPass::createClonesAndPopulateVMap(
    const TargetHelpers::KernelCache &KCache, Function *GlobalOffsetFunc) {
  std::deque<User *> WorkList;
  for (auto *U : GlobalOffsetFunc->users())
    WorkList.emplace_back(U);

  while (!WorkList.empty()) {
    auto *WI = WorkList.front();
    WorkList.pop_front();
    auto *Call = dyn_cast<CallInst>(WI);
    if (!Call)
      continue; // Not interesting.

    auto *Func = Call->getFunction();
    if (0 != GlobalVMap->count(Func))
      continue; // Already processed.

    const bool IsKernel = KCache.isKernel(*Func);
    FunctionType *FuncTy = Func->getFunctionType();
    Type *ImplicitArgumentType =
        IsKernel ? KernelImplicitArgumentType->getPointerTo()
                 : ImplicitOffsetPtrType;

    // Construct an argument list containing all of the previous arguments.
    SmallVector<Type *, 8> Arguments;
    for (const auto &A : Func->args())
      Arguments.push_back(A.getType());

    // Add the offset argument.
    Arguments.push_back(ImplicitArgumentType);

    // Build the new function.
    if (FuncTy->isVarArg())
      llvm_unreachable("Variadic arguments prohibited in SYCL");
    FunctionType *NewFuncTy = FunctionType::get(FuncTy->getReturnType(),
                                                Arguments, FuncTy->isVarArg());
    Function *NewFunc = Function::Create(NewFuncTy, Func->getLinkage(),
                                         Func->getAddressSpace());
    NewFunc->setName(Func->getName() + "_with_offset");
    // Remove the subprogram, if exists, as it will be pointing to an incorrect
    // data.
    if (Func->getSubprogram())
      NewFunc->setSubprogram(nullptr);

    // Keep original function ordering, clone goes right after the original.
    Func->getParent()->getFunctionList().insertAfter(Func->getIterator(),
                                                     NewFunc);

    // Populate the global value to value map with function arguments as well
    // as the cloned function itself.
    for (Function::arg_iterator FuncArg = Func->arg_begin(),
                                FuncEnd = Func->arg_end(),
                                NewFuncArg = NewFunc->arg_begin();
         FuncArg != FuncEnd; ++FuncArg, ++NewFuncArg) {
      (*GlobalVMap)[FuncArg] = NewFuncArg;
    }
    (*GlobalVMap)[Func] = NewFunc;

    // Extend the work list with the users of the function.
    for (auto *U : Func->users())
      WorkList.emplace_back(U);
  }
}

// New PM implementation.
PreservedAnalyses GlobalOffsetPass::run(Module &M, ModuleAnalysisManager &) {
  // Only run this pass on SYCL device code
  if (!TargetHelpers::isSYCLDevice(M))
    return PreservedAnalyses::all();

  // And only for NVPTX/AMDGCN targets.
  Triple T(M.getTargetTriple());
  if (!T.isNVPTX() && !T.isAMDGCN())
    return PreservedAnalyses::all();

  Function *GlobalOffsetFunc =
      M.getFunction("_Z27__spirv_BuiltInGlobalOffseti");

  if (!GlobalOffsetFunc || GlobalOffsetFunc->use_empty())
    return PreservedAnalyses::all();

  if (EnableGlobalOffset) {
    // For AMD allocas and pointers have to be to CONSTANT_PRIVATE (5), NVVM is
    // happy with ADDRESS_SPACE_GENERIC (0).
    TargetAS = T.isNVPTX() ? 0 : 5;
    KernelImplicitArgumentType =
        ArrayType::get(Type::getInt32Ty(M.getContext()), 3);
    ImplicitOffsetPtrType = PointerType::get(M.getContext(), TargetAS);

    TargetHelpers::KernelCache KCache;
    KCache.populateKernels(M);
    // Validate kernels
    validateKernels(M, KCache);

    createClonesAndPopulateVMap(KCache, GlobalOffsetFunc);

    // Add implicit parameters to all direct and indirect users of the offset
    addImplicitParameterToCallers(M, GlobalOffsetFunc, nullptr, KCache);
  }

  // Replace each use with a Constant 0
  for (auto *U : make_early_inc_range(GlobalOffsetFunc->users())) {
    auto *CI = cast<CallInst>(U);
    CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), 0));
    CI->eraseFromParent();
  }

  GlobalOffsetFunc->eraseFromParent();

  return PreservedAnalyses::none();
}

void GlobalOffsetPass::processKernelEntryPoint(
    Function *Func, TargetHelpers::KernelCache &KCache) {
  auto &M = *Func->getParent();
  Triple T(M.getTargetTriple());
  LLVMContext &Ctx = M.getContext();

  // Already processed.
  if (ProcessedFunctions.count(Func) == 1)
    return;

  // Add the new argument to all other kernel entry points, despite not
  // using the global offset.
  auto *NewFunc = addOffsetArgumentToFunction(
                      M, Func, PointerType::getUnqual(Func->getContext()),
                      /*KeepOriginal=*/true,
                      /*IsKernel=*/true)
                      .first;

  Argument *NewArgument = std::prev(NewFunc->arg_end());
  // Pass byval to the kernel for NVIDIA, AMD's calling convention disallows
  // byval args, use byref.
  auto Attr =
      T.isNVPTX()
          ? Attribute::getWithByValType(Ctx, KernelImplicitArgumentType)
          : Attribute::getWithByRefType(Ctx, KernelImplicitArgumentType);
  NewArgument->addAttr(Attr);

  KCache.handleNewCloneOf(*Func, *NewFunc, /*KernelOnly*/ true);
}

void GlobalOffsetPass::addImplicitParameterToCallers(
    Module &M, Value *Callee, Function *CalleeWithImplicitParam,
    TargetHelpers::KernelCache &KCache) {
  SmallVector<User *, 8> Users{Callee->users()};

  for (User *U : Users) {
    auto *CallToOld = dyn_cast<CallInst>(U);
    if (!CallToOld)
      return;

    auto *Caller = CallToOld->getFunction();

    // Only original function uses are considered.
    // Clones are processed through a global VMap.
    if (Clones.contains(Caller))
      continue;

    // Kernel entry points need additional processing and change Metdadata.
    if (KCache.isKernel(*Caller))
      processKernelEntryPoint(Caller, KCache);

    // Determine if `Caller` needs to be processed or if this is another
    // callsite from a non-offset function or an already-processed function.
    Value *ImplicitOffset = ProcessedFunctions[Caller];
    bool AlreadyProcessed = ImplicitOffset != nullptr;

    Function *NewFunc;
    if (AlreadyProcessed) {
      NewFunc = Caller;
    } else {
      std::tie(NewFunc, ImplicitOffset) = addOffsetArgumentToFunction(
          M, Caller,
          /*KernelImplicitArgumentType*/ nullptr,
          /*KeepOriginal=*/true, /*IsKernel=*/false);
    }
    CallToOld = cast<CallInst>((*GlobalVMap)[CallToOld]);
    if (!CalleeWithImplicitParam) {
      // If the result of the intrinsic is unused, just erase the call and
      // avoid materializing dead memory operations.
      if (CallToOld->use_empty()) {
        CallToOld->eraseFromParent();
        continue;
      }

      // Replace __spirv_BuiltInGlobalOffset call with load from parameter.
      IRBuilder<> Builder(CallToOld);
      Value *Index = CallToOld->getArgOperand(0);
      Value *OffsetPtr = Builder.CreateInBoundsGEP(
          KernelImplicitArgumentType->getElementType(), ImplicitOffset,
          {Index});
      Value *Offset = Builder.CreateLoad(
          KernelImplicitArgumentType->getElementType(), OffsetPtr);
      Value *OffsetCast = Builder.CreateIntCast(Offset, CallToOld->getType(),
                                                /*isSigned*/ false);
      CallToOld->replaceAllUsesWith(OffsetCast);
    } else {
      // Build up a list of arguments to call the modified function using.
      SmallVector<Value *, 8> ImplicitOffsets;
      for (Use &U : CallToOld->args()) {
        ImplicitOffsets.push_back(U);
      }
      ImplicitOffsets.push_back(ImplicitOffset);

      // Replace call to other function (which now has a new parameter),
      // with a call including the new parameter to that same function.
      auto *NewCallInst = CallInst::Create(
          /* Ty= */ CalleeWithImplicitParam->getFunctionType(),
          /* Func= */ CalleeWithImplicitParam,
          /* Args= */ ImplicitOffsets,
          /* NameStr= */ Twine(),
          /* InsertBefore= */ CallToOld->getIterator());
      NewCallInst->setTailCallKind(CallToOld->getTailCallKind());
      NewCallInst->copyMetadata(*CallToOld);
      CallToOld->replaceAllUsesWith(NewCallInst);

      if (CallToOld->hasName()) {
        NewCallInst->takeName(CallToOld);
      }
    }

    // Remove the caller now that it has been replaced.
    CallToOld->eraseFromParent();

    if (AlreadyProcessed)
      continue;

    // Process callers of the old function.
    addImplicitParameterToCallers(M, Caller, NewFunc, KCache);
  }
}

std::pair<Function *, Value *> GlobalOffsetPass::addOffsetArgumentToFunction(
    Module &M, Function *Func, Type *ImplicitArgumentType, bool KeepOriginal,
    bool IsKernel) {
  const AttributeList &FuncAttrs = Func->getAttributes();
  ImplicitArgumentType =
      ImplicitArgumentType ? ImplicitArgumentType : ImplicitOffsetPtrType;

  // Construct an argument list containing all of the previous arguments.
  SmallVector<Type *, 8> Arguments;
  SmallVector<AttributeSet, 8> ArgumentAttributes;
  for (const auto &I : enumerate(Func->args())) {
    Arguments.push_back(I.value().getType());
    ArgumentAttributes.push_back(FuncAttrs.getParamAttrs(I.index()));
  }

  // Add the offset argument.
  Arguments.push_back(ImplicitArgumentType);
  ArgumentAttributes.push_back(AttributeSet());

  // Build the new function.
  AttributeList NAttrs =
      AttributeList::get(Func->getContext(), FuncAttrs.getFnAttrs(),
                         FuncAttrs.getRetAttrs(), ArgumentAttributes);
  assert((*GlobalVMap).count(Func) != 0 &&
         "All relevant functions must be prepared ahead of time.");
  Function *NewFunc = dyn_cast<Function>((*GlobalVMap)[Func]);

  Value *ImplicitOffset = nullptr;
  if (KeepOriginal) {
    SmallVector<ReturnInst *, 8> Returns;
    CloneFunctionInto(NewFunc, Func, *GlobalVMap,
                      CloneFunctionChangeType::GlobalChanges, Returns);

    // In order to keep the signatures of functions called by the kernel
    // unified, the pass has to copy global offset to an array allocated in
    // addrspace(3). This is done as kernels can't allocate and fill the
    // array in constant address space.
    // Not required any longer, but left due to deprecatedness.
    if (IsKernel && Func->getCallingConv() == CallingConv::AMDGPU_KERNEL) {
      BasicBlock *EntryBlock = &NewFunc->getEntryBlock();
      IRBuilder<> Builder(EntryBlock, EntryBlock->getFirstInsertionPt());
      Type *ImplicitOffsetType =
          ArrayType::get(Type::getInt32Ty(M.getContext()), 3);
      Value *OrigImplicitOffset = std::prev(NewFunc->arg_end());
      AllocaInst *ImplicitOffsetAlloca =
          Builder.CreateAlloca(ImplicitOffsetType, TargetAS);
      auto DL = M.getDataLayout();
      uint64_t AllocByteSize =
          ImplicitOffsetAlloca->getAllocationSizeInBits(DL).value() / 8;
      // After AMD's kernel arg lowering pass runs the accesses to arguments
      // are replaced with uses of kernarg.segment.ptr which is in
      // addrspace(4), cast implicit offset arg to constant memory so the
      // memcpy is issued into a correct address space.
      auto *OrigImplicitOffsetAS4 = Builder.CreateAddrSpaceCast(
          OrigImplicitOffset, llvm::PointerType::get(M.getContext(), 4));
      Builder.CreateMemCpy(
          ImplicitOffsetAlloca, ImplicitOffsetAlloca->getAlign(),
          OrigImplicitOffsetAS4, OrigImplicitOffsetAS4->getPointerAlignment(DL),
          AllocByteSize);
      ImplicitOffset = ImplicitOffsetAlloca;
      ImplicitArgumentType = ImplicitOffset->getType();
    } else {
      ImplicitOffset = std::prev(NewFunc->arg_end());
    }
  } else {
    NewFunc->copyAttributesFrom(Func);
    NewFunc->setComdat(Func->getComdat());
    NewFunc->setAttributes(NAttrs);
    NewFunc->takeName(Func);

    // Splice the body of the old function right into the new function.
    NewFunc->splice(NewFunc->begin(), Func);

    for (Function::arg_iterator FuncArg = Func->arg_begin(),
                                FuncEnd = Func->arg_end(),
                                NewFuncArg = NewFunc->arg_begin();
         FuncArg != FuncEnd; ++FuncArg, ++NewFuncArg) {
      FuncArg->replaceAllUsesWith(NewFuncArg);
    }

    // Clone metadata of the old function, including debug info descriptor.
    SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
    Func->getAllMetadata(MDs);
    for (const auto &MD : MDs)
      NewFunc->addMetadata(MD.first, *MD.second);

    ImplicitOffset = std::prev(NewFunc->arg_end());
  }
  assert(ImplicitOffset && "Value of implicit offset must be set.");

  ProcessedFunctions[Func] = ImplicitOffset;
  Clones.insert(NewFunc);
  // Return the new function and the offset argument.
  return {NewFunc, ImplicitOffset};
}
