//===-- LowerWGLocalMemory.cpp - SYCL kernel local memory allocation pass -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See intro comments in the header.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/LowerWGLocalMemory.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Pass.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "sycllowerwglocalmemory"

static constexpr char SYCL_ALLOCLOCALMEM_CALL[] = "__sycl_allocateLocalMemory";
static constexpr char SYCL_DYNAMIC_LOCALMEM_CALL[] =
    "__sycl_dynamicLocalMemoryPlaceholder";
static constexpr char LOCALMEMORY_GV_PREF[] = "WGLocalMem";
static constexpr char DYNAMIC_LOCALMEM_GV[] =
    "__sycl_dynamicLocalMemoryPlaceholder_GV";
static constexpr char WORK_GROUP_STATIC_ATTR[] = "sycl-work-group-static";
static constexpr char WORK_GROUP_STATIC_ARG_ATTR[] = "sycl-implicit-local-arg";

namespace {
class SYCLLowerWGLocalMemoryLegacy : public ModulePass {
public:
  static char ID;

  SYCLLowerWGLocalMemoryLegacy() : ModulePass(ID) {
    initializeSYCLLowerWGLocalMemoryLegacyPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    ModuleAnalysisManager DummyMAM;
    auto PA = Impl.run(M, DummyMAM);
    return !PA.areAllPreserved();
  }

private:
  SYCLLowerWGLocalMemoryPass Impl;
};
} // namespace

std::vector<std::pair<StringRef, int>>
sycl::getKernelNamesUsingImplicitLocalMem(const Module &M) {
  std::vector<std::pair<StringRef, int>> SPIRKernelNames;
  Triple TT(M.getTargetTriple());

  if (TT.isSPIROrSPIRV()) {
    auto GetArgumentPos = [&](const Function &F) -> int {
      for (const Argument &Arg : F.args())
        if (F.getAttributes().hasParamAttr(Arg.getArgNo(),
                                           WORK_GROUP_STATIC_ARG_ATTR))
          return Arg.getArgNo();
      // Not lowered to an implicit arg or DAE.
      return -1;
    };
    llvm::for_each(M.functions(), [&](const Function &F) {
      if (F.getCallingConv() == CallingConv::SPIR_KERNEL &&
          F.hasFnAttribute(WORK_GROUP_STATIC_ATTR)) {
        int ArgPos = GetArgumentPos(F);
        SPIRKernelNames.emplace_back(F.getName(), ArgPos);
      }
    });
  }
  return SPIRKernelNames;
}

char SYCLLowerWGLocalMemoryLegacy::ID = 0;
INITIALIZE_PASS(SYCLLowerWGLocalMemoryLegacy, "sycllowerwglocalmemory",
                "Replace __sycl_allocateLocalMemory with allocation of memory "
                "in local address space",
                false, false)

ModulePass *llvm::createSYCLLowerWGLocalMemoryLegacyPass() {
  return new SYCLLowerWGLocalMemoryLegacy();
}

// In sycl header __sycl_allocateLocalMemory builtin call is wrapped in
// group_local_memory/group_local_memory_for_overwrite functions, which must be
// inlined first before each __sycl_allocateLocalMemory call can be lowered to a
// unique global variable. Inlining them here so that this pass doesn't have
// implicit dependency on AlwaysInlinerPass.
//
// syclcompat::local_mem, which represents a unique allocation, calls
// group_local_memory_for_overwrite. So local_mem should be inlined as well.
static bool inlineGroupLocalMemoryFunc(Module &M) {
  Function *ALMFunc = M.getFunction(SYCL_ALLOCLOCALMEM_CALL);
  if (!ALMFunc || ALMFunc->use_empty())
    return false;

  SmallVector<Function *, 4> WorkList{ALMFunc};
  DenseSet<Function *> Visited;
  while (!WorkList.empty()) {
    auto *F = WorkList.pop_back_val();
    for (auto *U : make_early_inc_range(F->users())) {
      auto *CI = cast<CallInst>(U);
      auto *Caller = CI->getFunction();
      if (Caller->hasFnAttribute("sycl-forceinline") &&
          Visited.insert(Caller).second)
        WorkList.push_back(Caller);
      if (F != ALMFunc) {
        InlineFunctionInfo IFI;
        [[maybe_unused]] auto Result = InlineFunction(*CI, IFI);
        assert(Result.isSuccess() && "inlining failed");
      }
    }
  }
  for (auto *F : Visited)
    F->eraseFromParent();

  return !Visited.empty();
}

// TODO: It should be checked that __sycl_allocateLocalMemory (or its source
// form - group_local_memory) does not occur:
//  - in a function (other than user lambda/functor)
//  - in a loop
//  - in a non-convergent control flow
// to make it consistent with OpenCL restriction.
// But LLVM pass is not the best place to diagnose these cases.
// Error checking should be done in the front-end compiler.
static void lowerAllocaLocalMemCall(CallInst *CI, Module &M) {
  assert(CI);

  Value *ArgSize = CI->getArgOperand(0);
  uint64_t Size = cast<llvm::ConstantInt>(ArgSize)->getZExtValue();
  Value *ArgAlign = CI->getArgOperand(1);
  uint64_t Alignment = cast<llvm::ConstantInt>(ArgAlign)->getZExtValue();

  IRBuilder<> Builder(CI);
  Type *LocalMemArrayTy = ArrayType::get(Builder.getInt8Ty(), Size);
  unsigned LocalAS =
      CI->getFunctionType()->getReturnType()->getPointerAddressSpace();
  auto *LocalMemArrayGV =
      new GlobalVariable(M,                                // module
                         LocalMemArrayTy,                  // type
                         false,                            // isConstant
                         GlobalValue::InternalLinkage,     // Linkage
                         UndefValue::get(LocalMemArrayTy), // Initializer
                         LOCALMEMORY_GV_PREF,              // Name prefix
                         nullptr,                          // InsertBefore
                         GlobalVariable::NotThreadLocal,   // ThreadLocalMode
                         LocalAS                           // AddressSpace
      );
  LocalMemArrayGV->setAlignment(Align(Alignment));

  Value *GVPtr =
      Builder.CreatePointerCast(LocalMemArrayGV, Builder.getInt8PtrTy(LocalAS));
  CI->replaceAllUsesWith(GVPtr);
}

static void
lowerDynamicLocalMemCallDirect(CallInst *CI, Triple TT,
                               GlobalVariable *LocalMemPlaceholder) {
  assert(CI);

  Value *GVPtr = [&]() -> Value * {
    IRBuilder<> Builder(CI);
    if (TT.isSPIROrSPIRV())
      return Builder.CreateLoad(CI->getType(), LocalMemPlaceholder);

    return Builder.CreatePointerCast(LocalMemPlaceholder, CI->getType());
  }();
  CI->replaceAllUsesWith(GVPtr);
}

static void lowerLocalMemCall(Function *LocalMemAllocFunc,
                              std::function<void(CallInst *CI)> TransformCall) {
  SmallVector<CallInst *, 4> DelCalls;
  for (User *U : LocalMemAllocFunc->users()) {
    auto *CI = cast<CallInst>(U);
    TransformCall(CI);
    DelCalls.push_back(CI);
  }

  for (auto *CI : DelCalls) {
    assert(CI->use_empty() && "removing live instruction");
    CI->eraseFromParent();
  }

  // Remove __sycl_allocateLocalMemory declaration.
  assert(LocalMemAllocFunc->use_empty() &&
         "local mem allocation function is still in use");
  LocalMemAllocFunc->eraseFromParent();
}

static bool allocaWGLocalMemory(Module &M) {
  Function *ALMFunc = M.getFunction(SYCL_ALLOCLOCALMEM_CALL);
  if (!ALMFunc)
    return false;

  assert(ALMFunc->isDeclaration() && "should have declaration only");

  lowerLocalMemCall(ALMFunc,
                    [&](CallInst *CI) { lowerAllocaLocalMemCall(CI, M); });

  return true;
}

// For dynamic memory we have 2 case:
//   - Direct for CUDA/HIP: we create a placeholder and set the memory on
//   launch.
//   - Indirect for OpenCL/Level0: we create a shared value holding the pointer
//   to the buffer passed as argument.
static bool dynamicWGLocalMemory(Module &M) {
  Function *DLMFunc = M.getFunction(SYCL_DYNAMIC_LOCALMEM_CALL);
  if (!DLMFunc)
    return false;

  GlobalVariable *LocalMemArrayGV =
      M.getGlobalVariable(DYNAMIC_LOCALMEM_GV, true);
  Triple TT(M.getTargetTriple());
  unsigned LocalAS = DLMFunc->getReturnType()->getPointerAddressSpace();

  if (!LocalMemArrayGV) {
    assert(DLMFunc->isDeclaration() && "should have declaration only");
    Type *LocalMemArrayTy =
        TT.isSPIROrSPIRV()
            ? static_cast<Type *>(PointerType::get(M.getContext(), LocalAS))
            : static_cast<Type *>(
                  ArrayType::get(Type::getInt8Ty(M.getContext()), 0));
    LocalMemArrayGV = new GlobalVariable(
        M,               // module
        LocalMemArrayTy, // type
        false,           // isConstant
        TT.isSPIROrSPIRV() ? GlobalValue::LinkOnceODRLinkage
                           : GlobalValue::ExternalLinkage, // Linkage
        TT.isSPIROrSPIRV() ? UndefValue::get(LocalMemArrayTy)
                           : nullptr,   // Initializer
        DYNAMIC_LOCALMEM_GV,            // Name prefix
        nullptr,                        // InsertBefore
        GlobalVariable::NotThreadLocal, // ThreadLocalMode
        LocalAS                         // AddressSpace
    );
    LocalMemArrayGV->setUnnamedAddr(GlobalVariable::UnnamedAddr::Local);
    constexpr int DefaultMaxAlignment = 128;
    if (!TT.isSPIROrSPIRV())
      LocalMemArrayGV->setAlignment(Align{DefaultMaxAlignment});
  }
  lowerLocalMemCall(DLMFunc, [&](CallInst *CI) {
    lowerDynamicLocalMemCallDirect(CI, TT, LocalMemArrayGV);
  });
  if (TT.isSPIROrSPIRV()) {
    SmallVector<Function *, 4> Kernels;
    llvm::for_each(M.functions(), [&](Function &F) {
      if (F.getCallingConv() == CallingConv::SPIR_KERNEL &&
          F.hasFnAttribute(WORK_GROUP_STATIC_ATTR)) {
        Kernels.push_back(&F);
      }
    });
    for (Function *OldKernel : Kernels) {
      FunctionType *FuncTy = OldKernel->getFunctionType();
      const AttributeList &FuncAttrs = OldKernel->getAttributes();
      Type *ImplicitLocalPtr = PointerType::get(M.getContext(), LocalAS);

      // Construct an argument list containing all of the previous arguments.
      SmallVector<Type *, 8> Arguments;
      SmallVector<AttributeSet, 8> ArgumentAttributes;
      for (const auto &I : enumerate(OldKernel->args())) {
        Arguments.push_back(I.value().getType());
        ArgumentAttributes.push_back(FuncAttrs.getParamAttrs(I.index()));
      }

      Arguments.push_back(ImplicitLocalPtr);
      ArgumentAttributes.push_back(AttributeSet::get(
          M.getContext(),
          ArrayRef<Attribute>{
              Attribute::get(M.getContext(), Attribute::NoAlias),
              Attribute::get(M.getContext(), WORK_GROUP_STATIC_ARG_ATTR)}));

      // Build the new function.
      AttributeList NAttrs =
          AttributeList::get(OldKernel->getContext(), FuncAttrs.getFnAttrs(),
                             FuncAttrs.getRetAttrs(), ArgumentAttributes);
      assert(!FuncTy->isVarArg() && "Variadic arguments prohibited in SYCL");
      FunctionType *NewFuncTy = FunctionType::get(
          FuncTy->getReturnType(), Arguments, FuncTy->isVarArg());

      Function *NewFunc = Function::Create(NewFuncTy, OldKernel->getLinkage(),
                                           OldKernel->getAddressSpace());

      // Keep original function ordering.
      M.getFunctionList().insertAfter(OldKernel->getIterator(), NewFunc);

      NewFunc->copyAttributesFrom(OldKernel);
      NewFunc->setComdat(OldKernel->getComdat());
      NewFunc->setAttributes(NAttrs);
      NewFunc->takeName(OldKernel);

      // Splice the body of the old function right into the new function.
      NewFunc->splice(NewFunc->begin(), OldKernel);

      for (Function::arg_iterator FuncArg = OldKernel->arg_begin(),
                                  FuncEnd = OldKernel->arg_end(),
                                  NewFuncArg = NewFunc->arg_begin();
           FuncArg != FuncEnd; ++FuncArg, ++NewFuncArg) {
        FuncArg->replaceAllUsesWith(NewFuncArg);
      }

      // Clone metadata of the old function, including debug info descriptor.
      SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
      OldKernel->getAllMetadata(MDs);
      for (const auto &MD : MDs)
        NewFunc->addMetadata(MD.first, *MD.second);
      // Store the pointer to the implicit local memory into the global
      // handler.
      IRBuilder<> Builder(&NewFunc->getEntryBlock(),
                          NewFunc->getEntryBlock().getFirstNonPHIIt());
      Builder.CreateStore(NewFunc->getArg(NewFunc->arg_size() - 1),
                          LocalMemArrayGV);
      OldKernel->eraseFromParent();
      auto FixupMetadata = [&](StringRef MDName, Metadata *NewV) {
        auto *Node = NewFunc->getMetadata(MDName);
        if (!Node)
          return;
        SmallVector<Metadata *, 8> NewMD(Node->operands());
        NewMD.emplace_back(NewV);
        NewFunc->setMetadata(MDName,
                             llvm::MDNode::get(NewFunc->getContext(), NewMD));
      };

      FixupMetadata("kernel_arg_buffer_location",
                    ConstantAsMetadata::get(Builder.getInt32(-1)));
      FixupMetadata("kernel_arg_runtime_aligned",
                    ConstantAsMetadata::get(Builder.getFalse()));
      FixupMetadata("kernel_arg_exclusive_ptr",
                    ConstantAsMetadata::get(Builder.getFalse()));

      FixupMetadata("kernel_arg_addr_space",
                    ConstantAsMetadata::get(Builder.getInt32(LocalAS)));
      FixupMetadata("kernel_arg_access_qual",
                    MDString::get(M.getContext(), "read_write"));
      FixupMetadata("kernel_arg_type", MDString::get(M.getContext(), "void*"));
      FixupMetadata("kernel_arg_base_type",
                    MDString::get(M.getContext(), "void*"));
      FixupMetadata("kernel_arg_type_qual", MDString::get(M.getContext(), ""));
      FixupMetadata("kernel_arg_accessor_ptr",
                    ConstantAsMetadata::get(Builder.getFalse()));
    }
  }

  return true;
}

PreservedAnalyses SYCLLowerWGLocalMemoryPass::run(Module &M,
                                                  ModuleAnalysisManager &) {
  bool Changed = inlineGroupLocalMemoryFunc(M);
  Changed |= allocaWGLocalMemory(M);
  Changed |= dynamicWGLocalMemory(M);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
