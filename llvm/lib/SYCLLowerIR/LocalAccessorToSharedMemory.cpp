//===- LocalAccessorToSharedMemory.cpp - Local Accessor Support for CUDA --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass operates on SYCL kernels being compiled to CUDA. It modifies
// kernel entry points which take pointers to shared memory and modifies them
// to take offsets into shared memory (represented by a symbol in the shared
// address space). The SYCL runtime is expected to provide offsets rather than
// pointers to these functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/LocalAccessorToSharedMemory.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/SYCLLowerIR/TargetHelpers.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO.h"

using namespace llvm;

#define DEBUG_TYPE "localaccessortosharedmemory"

namespace llvm {
void initializeLocalAccessorToSharedMemoryPass(PassRegistry &);
} // namespace llvm

namespace {

class LocalAccessorToSharedMemory : public ModulePass {
private:
  using KernelPayload = TargetHelpers::KernelPayload;
  using ArchType = TargetHelpers::ArchType;

  // The value for NVVM's ADDRESS_SPACE_SHARED and AMD's LOCAL_ADDRESS happen to
  // be 3.
  const unsigned SharedASValue = 3;

public:
  static char ID;
  LocalAccessorToSharedMemory() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    const auto AT = TargetHelpers::getArchType(M);

    // Invariant: This pass is only intended to operate on SYCL kernels being
    // compiled to either `nvptx{,64}-nvidia-cuda`, or `amdgcn-amd-amdhsa`
    // triples.
    if (ArchType::Unsupported == AT)
      return false;

    if (skipModule(M))
      return false;

    SmallVector<KernelPayload, 4> Kernels;
    TargetHelpers::populateKernels(M, Kernels, AT);
    SmallVector<std::pair<Function *, KernelPayload>, 4> NewToOldKernels;
    if (Kernels.empty())
      return false;

    // Process the function and if changed, update the metadata.
    for (auto K : Kernels) {
      auto *NewKernel = processKernel(M, K.Kernel);
      if (NewKernel)
        NewToOldKernels.push_back(std::make_pair(NewKernel, K));
    }

    if (NewToOldKernels.empty())
      return false;

    postProcessKernels(NewToOldKernels);

    return true;
  }

  virtual llvm::StringRef getPassName() const override {
    return "SYCL Local Accessor to Shared Memory";
  }

private:
  Function *processKernel(Module &M, Function *F) {
    // Check if this function is eligible by having an argument that uses shared
    // memory.
    auto UsesLocalMemory = false;
    for (Function::arg_iterator FA = F->arg_begin(), FE = F->arg_end();
         FA != FE; ++FA) {
      if (FA->getType()->isPointerTy() &&
          FA->getType()->getPointerAddressSpace() == SharedASValue) {
        UsesLocalMemory = true;
        break;
      }
    }

    // Skip functions which are not eligible.
    if (!UsesLocalMemory)
      return nullptr;

    // Create a global symbol to CUDA shared memory.
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

    unsigned i = 0;
    for (Function::arg_iterator FA = F->arg_begin(), FE = F->arg_end();
         FA != FE; ++FA, ++i) {
      if (FA->getType()->isPointerTy() &&
          FA->getType()->getPointerAddressSpace() == SharedASValue) {
        // Replace pointers to shared memory with i32 offsets.
        Arguments.push_back(Type::getInt32Ty(M.getContext()));
        ArgumentAttributes.push_back(
            AttributeSet::get(M.getContext(), ArrayRef<Attribute>{}));
        ArgumentReplaced[i] = true;
      } else {
        // Replace other arguments with the same type as before.
        Arguments.push_back(FA->getType());
        ArgumentAttributes.push_back(FAttrs.getParamAttrs(i));
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
    NF->getBasicBlockList().splice(NF->begin(), F->getBasicBlockList());

    i = 0;
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
    for (auto MD : MDs)
      NF->addMetadata(MD.first, *MD.second);

    // Now that the old function is dead, delete it.
    F->eraseFromParent();

    return NF;
  }

  void postProcessKernels(
      SmallVectorImpl<std::pair<Function *, KernelPayload>> &NewToOldKernels) {
    for (auto &Pair : NewToOldKernels) {
      std::get<1>(Pair).MD->replaceOperandWith(
          0, llvm::ConstantAsMetadata::get(std::get<0>(Pair)));
    }
  }
};
} // end anonymous namespace

char LocalAccessorToSharedMemory::ID = 0;

INITIALIZE_PASS(LocalAccessorToSharedMemory, "localaccessortosharedmemory",
                "SYCL Local Accessor to Shared Memory", false, false)

ModulePass *llvm::createLocalAccessorToSharedMemoryPass() {
  return new LocalAccessorToSharedMemory();
}
