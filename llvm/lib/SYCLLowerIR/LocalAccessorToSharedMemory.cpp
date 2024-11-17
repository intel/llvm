//===- LocalAccessorToSharedMemory.cpp - Local Accessor Support for CUDA --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/LocalAccessorToSharedMemory.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO.h"

using namespace llvm;

#define DEBUG_TYPE "localaccessortosharedmemory"

// Legacy PM wrapper.
namespace {
class LocalAccessorToSharedMemoryLegacy : public ModulePass {
public:
  static char ID;

  LocalAccessorToSharedMemoryLegacy() : ModulePass(ID) {
    initializeLocalAccessorToSharedMemoryLegacyPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    ModuleAnalysisManager MAM;
    auto PA = Impl.run(M, MAM);
    return !PA.areAllPreserved();
  }

private:
  LocalAccessorToSharedMemoryPass Impl;
};
} // namespace

char LocalAccessorToSharedMemoryLegacy::ID = 0;
INITIALIZE_PASS(LocalAccessorToSharedMemoryLegacy,
                "localaccessortosharedmemory",
                "SYCL Local Accessor to Shared Memory", false, false)

ModulePass *llvm::createLocalAccessorToSharedMemoryPassLegacy() {
  return new LocalAccessorToSharedMemoryLegacy();
}

// New PM implementation.
PreservedAnalyses
LocalAccessorToSharedMemoryPass::run(Module &M, ModuleAnalysisManager &) {
  // Only run this pass on SYCL device code
  if (!TargetHelpers::isSYCLDevice(M))
    return PreservedAnalyses::all();

  // And only for NVPTX/AMDGCN targets.
  Triple T(M.getTargetTriple());
  if (!T.isNVPTX() && !T.isAMDGCN())
    return PreservedAnalyses::all();

  TargetHelpers::KernelCache KCache;
  KCache.populateKernels(M);
  if (KCache.empty())
    return PreservedAnalyses::all();

  DenseMap<Function *, Function *> NewToOldKernels;
  // Process the function and if changed, update the metadata.
  for (const auto &F : KCache) {
    if (auto *NewKernel = processKernel(M, F))
      NewToOldKernels[NewKernel] = F;
  }

  if (NewToOldKernels.empty())
    return PreservedAnalyses::all();

  for (auto &[NewF, F] : NewToOldKernels)
    KCache.handleReplacedWith(*F, *NewF);

  return PreservedAnalyses::none();
}

Function *LocalAccessorToSharedMemoryPass::processKernel(Module &M,
                                                         Function *F) {
  // Check if this function is eligible by having an argument that uses shared
  // memory.
  const bool UsesLocalMemory =
      std::any_of(F->arg_begin(), F->arg_end(), [&](Argument &FA) {
        return FA.getType()->isPointerTy() &&
               FA.getType()->getPointerAddressSpace() == SharedASValue;
      });

  // Skip functions which are not eligible.
  if (!UsesLocalMemory)
    return nullptr;

  // Create a global symbol to CUDA's ADDRESS_SPACE_SHARED or AMD's
  // LOCAL_ADDRESS.
  auto SharedMemGlobalName = F->getName().str();
  SharedMemGlobalName.append("_shared_mem");
  auto *SharedMemGlobalType =
      ArrayType::get(Type::getInt8Ty(M.getContext()), 0);
  auto *SharedMemGlobal = new GlobalVariable(
      /* Module= */ M,
      /* Type= */ &*SharedMemGlobalType,
      /* IsConstant= */ false,
      /* Linkage= */ GlobalValue::ExternalLinkage,
      /* Initializer= */ nullptr,
      /* Name= */ Twine{SharedMemGlobalName},
      /* InsertBefore= */ nullptr,
      /* ThreadLocalMode= */ GlobalValue::NotThreadLocal,
      /* AddressSpace= */ SharedASValue,
      /* IsExternallyInitialized= */ false);
  SharedMemGlobal->setAlignment(Align(4));

  FunctionType *FTy = F->getFunctionType();
  const AttributeList &FAttrs = F->getAttributes();

  // Store the arguments and attributes for the new function, as well as which
  // arguments were replaced.
  std::vector<Type *> Arguments;
  SmallVector<AttributeSet, 8> ArgumentAttributes;
  SmallVector<bool, 10> ArgumentReplaced(FTy->getNumParams(), false);

  for (const auto &I : enumerate(F->args())) {
    const Argument &FA = I.value();
    if (FA.getType()->isPointerTy() &&
        FA.getType()->getPointerAddressSpace() == SharedASValue) {
      // Replace pointers to shared memory with i32 offsets.
      Arguments.push_back(Type::getInt32Ty(M.getContext()));
      ArgumentAttributes.push_back(
          AttributeSet::get(M.getContext(), ArrayRef<Attribute>{}));
      ArgumentReplaced[I.index()] = true;
    } else {
      // Replace other arguments with the same type as before.
      Arguments.push_back(FA.getType());
      ArgumentAttributes.push_back(FAttrs.getParamAttrs(I.index()));
    }
  }

  // Create new function type.
  AttributeList NAttrs =
      AttributeList::get(F->getContext(), FAttrs.getFnAttrs(),
                         FAttrs.getRetAttrs(), ArgumentAttributes);
  FunctionType *NFTy =
      FunctionType::get(FTy->getReturnType(), Arguments, FTy->isVarArg());

  // Create the new function body and insert it into the module.
  Function *NF = Function::Create(NFTy, F->getLinkage(), F->getAddressSpace(),
                                  Twine{""}, &M);
  NF->copyAttributesFrom(F);
  NF->setComdat(F->getComdat());
  NF->setAttributes(NAttrs);
  NF->takeName(F);

  // Splice the body of the old function right into the new function.
  NF->splice(NF->begin(), F);

  unsigned i = 0;
  for (Function::arg_iterator FA = F->arg_begin(), FE = F->arg_end(),
                              NFA = NF->arg_begin();
       FA != FE; ++FA, ++NFA, ++i) {
    Value *NewValueForUse = NFA;
    if (ArgumentReplaced[i]) {
      // If this argument was replaced, then create a `getelementptr`
      // instruction that uses it to recreate the pointer that was replaced.
      auto *InsertBefore = &NF->getEntryBlock().front();
      auto *PtrInst = GetElementPtrInst::CreateInBounds(
          /* PointeeType= */ SharedMemGlobalType,
          /* Ptr= */ SharedMemGlobal,
          /* IdxList= */
          ArrayRef<Value *>{
              ConstantInt::get(Type::getInt32Ty(M.getContext()), 0, false),
              NFA,
          },
          /* NameStr= */ Twine{NFA->getName()}, InsertBefore);
      // Then create a bitcast to make sure the new pointer is the same type
      // as the old one. This will only ever be a `i8 addrspace(3)*` to `i32
      // addrspace(3)*` type of cast.
      auto *CastInst = new BitCastInst(PtrInst, FA->getType());
      CastInst->insertAfter(PtrInst);
      NewValueForUse = CastInst;
    }

    // Replace uses of the old function's argument with the new argument or
    // the result of the `getelementptr`/`bitcast` instructions.
    FA->replaceAllUsesWith(&*NewValueForUse);
    NewValueForUse->takeName(&*FA);
  }

  // There should be no callers of kernel entry points.
  assert(F->use_empty());

  // Clone metadata of the old function, including debug info descriptor.
  SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
  F->getAllMetadata(MDs);
  for (const auto &MD : MDs)
    NF->addMetadata(MD.first, *MD.second);

  // Now that the old function is dead, delete it.
  F->eraseFromParent();

  return NF;
}
