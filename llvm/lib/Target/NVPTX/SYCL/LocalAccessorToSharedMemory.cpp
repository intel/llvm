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

#include "LocalAccessorToSharedMemory.h"
#include "../MCTargetDesc/NVPTXBaseInfo.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/IPO.h"

using namespace llvm;

#define DEBUG_TYPE "localaccessortosharedmemory"

namespace llvm {
void initializeLocalAccessorToSharedMemoryPass(PassRegistry &);
}

namespace {

class LocalAccessorToSharedMemory : public ModulePass {
public:
  static char ID;
  LocalAccessorToSharedMemory() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    // Invariant: This pass is only intended to operate on SYCL kernels being
    // compiled to the `nvptx{,64}-nvidia-cuda` triple.
    // TODO: make sure that non-SYCL kernels are not impacted.
    if (skipModule(M))
      return false;

    // Keep track of whether the module was changed.
    auto Changed = false;

    // Access `nvvm.annotations` to determine which functions are kernel entry
    // points.
    auto NvvmMetadata = M.getNamedMetadata("nvvm.annotations");
    if (!NvvmMetadata)
      return false;

    for (auto MetadataNode : NvvmMetadata->operands()) {
      if (MetadataNode->getNumOperands() != 3)
        continue;

      // NVPTX identifies kernel entry points using metadata nodes of the form:
      //   !X = !{<function>, !"kernel", i32 1}
      const MDOperand &TypeOperand = MetadataNode->getOperand(1);
      auto Type = dyn_cast<MDString>(TypeOperand);
      if (!Type)
        continue;
      // Only process kernel entry points.
      if (Type->getString() != "kernel")
        continue;

      // Get a pointer to the entry point function from the metadata.
      const MDOperand &FuncOperand = MetadataNode->getOperand(0);
      if (!FuncOperand)
        continue;
      auto FuncConstant = dyn_cast<ConstantAsMetadata>(FuncOperand);
      if (!FuncConstant)
        continue;
      auto Func = dyn_cast<Function>(FuncConstant->getValue());
      if (!Func)
        continue;

      // Process the function and if changed, update the metadata.
      auto NewFunc = this->ProcessFunction(M, Func);
      if (NewFunc) {
        Changed = true;
        MetadataNode->replaceOperandWith(
            0, llvm::ConstantAsMetadata::get(NewFunc));
      }
    }

    return Changed;
  }

  Function *ProcessFunction(Module &M, Function *F) {
    // Check if this function is eligible by having an argument that uses shared
    // memory.
    auto UsesLocalMemory = false;
    for (Function::arg_iterator FA = F->arg_begin(), FE = F->arg_end();
         FA != FE; ++FA) {
      if (FA->getType()->isPointerTy()) {
        UsesLocalMemory =
            FA->getType()->getPointerAddressSpace() == ADDRESS_SPACE_SHARED;
      }
      if (UsesLocalMemory) {
        break;
      }
    }

    // Skip functions which are not eligible.
    if (!UsesLocalMemory)
      return nullptr;

    // Create a global symbol to CUDA shared memory.
    auto SharedMemGlobalName = F->getName().str();
    SharedMemGlobalName.append("_shared_mem");
    auto SharedMemGlobalType =
        ArrayType::get(Type::getInt8Ty(M.getContext()), 0);
    auto SharedMemGlobal = new GlobalVariable(
        /* Module= */ M,
        /* Type= */ &*SharedMemGlobalType,
        /* IsConstant= */ false,
        /* Linkage= */ GlobalValue::ExternalLinkage,
        /* Initializer= */ nullptr,
        /* Name= */ Twine{SharedMemGlobalName},
        /* InsertBefore= */ nullptr,
        /* ThreadLocalMode= */ GlobalValue::NotThreadLocal,
        /* AddressSpace= */ ADDRESS_SPACE_SHARED,
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
          FA->getType()->getPointerAddressSpace() == ADDRESS_SPACE_SHARED) {
        // Replace pointers to shared memory with i32 offsets.
        Arguments.push_back(Type::getInt32Ty(M.getContext()));
        ArgumentAttributes.push_back(
            AttributeSet::get(M.getContext(), ArrayRef<Attribute>{}));
        ArgumentReplaced[i] = true;
      } else {
        // Replace other arguments with the same type as before.
        Arguments.push_back(FA->getType());
        ArgumentAttributes.push_back(FAttrs.getParamAttributes(i));
      }
    }

    // Create new function type.
    AttributeList NAttrs =
        AttributeList::get(F->getContext(), FAttrs.getFnAttributes(),
                           FAttrs.getRetAttributes(), ArgumentAttributes);
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
        auto InsertBefore = &NF->getEntryBlock().front();
        auto PtrInst = GetElementPtrInst::CreateInBounds(
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
        auto CastInst = new BitCastInst(PtrInst, FA->getType());
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

  virtual llvm::StringRef getPassName() const {
    return "localaccessortosharedmemory";
  }
};

} // end anonymous namespace

char LocalAccessorToSharedMemory::ID = 0;

INITIALIZE_PASS(LocalAccessorToSharedMemory, "localaccessortosharedmemory",
                "SYCL Local Accessor to Shared Memory", false, false)

ModulePass *llvm::createLocalAccessorToSharedMemoryPass() {
  return new LocalAccessorToSharedMemory();
}
