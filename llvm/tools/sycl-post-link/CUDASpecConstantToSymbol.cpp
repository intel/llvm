//===-- CUDASpecConstantToSymbol.cpp - CUDA Spec Constants To Symbol Pass -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See comments in the header.
//===----------------------------------------------------------------------===//

#include "CUDASpecConstantToSymbol.h"
#include "Support.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/SYCLLowerIR/TargetHelpers.h"
#include "llvm/Support/MathExtras.h"

#ifndef NDEBUG
#include "llvm/Support/Debug.h"
#endif

using namespace llvm;

#define DEBUG_TYPE "CUDASpecConstantToSymbol"

STATISTIC(TotalSymbolSize,
          "amount of global memory allocated for specialization constants.");

static GlobalVariable *createGlobal(Module &M, size_t Size, StringRef Name,
                                    bool RoundUp) {
  size_t AllocSize = PowerOf2Ceil(Size);
  if (RoundUp && AllocSize < 64)
    // cuMemcpyHtoD has problems copying to symbols that are not power of two
    // or less than 64-bit in size.
    AllocSize = 64;

  LLVM_DEBUG(llvm::dbgs() << "\tallocating: " << AllocSize << " bytes\n");
  TotalSymbolSize += AllocSize;

  auto *ArrTy = ArrayType::get(Type::getInt8Ty(M.getContext()), AllocSize);
  const bool IsConstant = true;
  const bool ExternallyInitialized = true;
  const unsigned AS = 4; // ADDRESS_SPACE_CONST
  return new GlobalVariable(M, ArrTy, IsConstant, GlobalValue::WeakODRLinkage,
                            Constant::getNullValue(ArrTy), Name, nullptr,
                            GlobalValue::NotThreadLocal, AS,
                            ExternallyInitialized);
}

static unsigned uintFromMDNode(const MDNode *Node, unsigned OpNumber) {
  auto *CMD =
      cast<ConstantAsMetadata>(Node->getOperand(OpNumber).get())->getValue();
  return static_cast<unsigned>(CMD->getUniqueInteger().getZExtValue());
}

std::pair<MDNode *, Function *>
getMDNodeAndFuncAtIndex(NamedMDNode &AnnotationMD, unsigned I) {
  assert(I < AnnotationMD.getNumOperands() && "Invalid index.");
  auto *MD = AnnotationMD.getOperand(I);
  // Kernel entry points are identified using metadata nodes of the form:
  //   !X = !{<function>, !"kernel", i32 1}
  auto *Type = dyn_cast<MDString>(MD->getOperand(1));
  // Only process kernel entry points.
  if (!Type || Type->getString() != "kernel")
    return {nullptr, nullptr};

  // Get a pointer to the entry point function from the metadata.
  const MDOperand &FuncOperand = MD->getOperand(0);
  if (!FuncOperand)
    return {nullptr, nullptr};
  if (auto *FuncConstant = dyn_cast<ConstantAsMetadata>(FuncOperand))
    if (auto *Func = dyn_cast<Function>(FuncConstant->getValue()))
      return {MD, Func};

  llvm_unreachable("Failed to find the function/node pair.");
  return {nullptr, nullptr};
}

static MDNode *getMDNode(Module &M, StringRef KernelName) {
  NamedMDNode *AnnotationMD = M.getNamedMetadata("nvvm.annotations");
  assert(AnnotationMD);

  for (unsigned I = 0; I < AnnotationMD->getNumOperands(); ++I) {
    auto Res = getMDNodeAndFuncAtIndex(*AnnotationMD, I);
    if (!std::get<0>(Res))
      continue;
    if (std::get<1>(Res)->getName() == KernelName)
      return std::get<0>(Res);
  }

  return nullptr;
}

void CUDASpecConstantToSymbolPass::setUpPlaceholderEntries(Module &M) {
  NamedMDNode *AnnotationMD = M.getNamedMetadata("nvvm.annotations");
  // The module has no kernels, early exit.
  if (!AnnotationMD)
    return;

  for (unsigned I = 0; I < AnnotationMD->getNumOperands(); ++I) {
    auto Res = getMDNodeAndFuncAtIndex(*AnnotationMD, I);
    if (!std::get<0>(Res))
      continue;
    std::string GlobalName{GlobalNamePrefix};
    GlobalName.append(std::get<1>(Res)->getName());
    LLVM_DEBUG(llvm::dbgs() << "Setting up a placeholder for: "
                            << std::get<1>(Res)->getName() << "\n");
    // NOTE: The size of the symbol here - 1 - is important (the value is
    // invalid and would result in a failure from `cuMemcpyHtoD`), instead,
    // it's used as a flag to communicate to the plugin that even though the
    // implicit kernel argument is present, it has no uses and hence there is
    // no need to set up the symbol. See:
    // `cuda_piextProgramSetSpecializationConstant` for pi handling of it.
    createGlobal(M, 1, GlobalName, false);
  }
}

void CUDASpecConstantToSymbolPass::allocatePerKernelGlobals(NamedMDNode *MD) {
  Module &M = *MD->getParent();
  // Calculate the size of global symbol and allocate it for each kernel.
  for (const auto *Node : MD->operands()) {
    const StringRef KernelName =
        cast<MDString>(Node->getOperand(0).get())->getString();
    LLVM_DEBUG(llvm::dbgs() << "Working on: " << KernelName << "\n");
    unsigned PerKernelSize = 0;
    // Loop over all spec constants of a kernel
    for (unsigned i = 1; i < Node->getNumOperands(); ++i) {
      MDNode *SC = dyn_cast<MDNode>(Node->getOperand(i));
      assert(SC && SC->getNumOperands() >= 4 && "Invalid node.");
      // get the size and offset node to calculate the total size of spec
      // constants (size of type + offset in the composite - if any);
      PerKernelSize += uintFromMDNode(SC, SC->getNumOperands() - 1);
      PerKernelSize += uintFromMDNode(SC, SC->getNumOperands() - 2);
      LLVM_DEBUG(llvm::dbgs()
                 << "\tper kernel size calculation: " << PerKernelSize << "\n");
    }
    // finally, allocate per-kernel globals.
    std::string GlobalName{GlobalNamePrefix};
    GlobalName.append(KernelName.str());
    SpecConstGlobals[KernelName] =
        createGlobal(M, PerKernelSize, GlobalName, true);
  }
}

void CUDASpecConstantToSymbolPass::fixupSpecConstantUses(
    Argument *A, const StringRef KernelName) {
  // Re-write uses of, now defunct, spec constant kernel arg to GEPs from the
  // global value.
  IRBuilder B(A->getParent()->getParent()->getContext());
  SmallVector<Instruction *> ToErase;
  for (auto *U : A->users()) {
    auto *I = dyn_cast_or_null<Instruction>(&*U);
    assert(I && "Expected an instruction.");
    switch (I->getOpcode()) {
    default: {
      std::string Str;
      raw_string_ostream Out(Str);
      Out << "Unhandled instruction: ";
      I->print(Out);
      llvm_unreachable(Str.c_str());
    }
    case Instruction::AddrSpaceCast: {
      auto *ASCInst = dyn_cast_or_null<AddrSpaceCastInst>(I);
      assert(ASCInst);
      const unsigned DestAS = ASCInst->getDestAddressSpace();
      auto PTy = PointerType::getWithSamePointeeType(
          cast<PointerType>(ASCInst->getPointerOperand()->getType()), DestAS);
      B.SetInsertPoint(I);
      auto NewASCInst = B.CreateAddrSpaceCast(SpecConstGlobals[KernelName], PTy,
                                              "symbol_ASC");
      ASCInst->replaceAllUsesWith(NewASCInst);
      ToErase.push_back(ASCInst);
      break;
    }
    }
  }
  // Get rid of the defunct instructions.
  for (auto *I : ToErase)
    I->eraseFromParent();
}

void CUDASpecConstantToSymbolPass::rewriteKernelSignature(NamedMDNode *MD) {
  Module &M = *MD->getParent();
  auto &Context = M.getContext();
  for (const auto *Node : MD->operands()) {
    const StringRef KernelName =
        cast<MDString>(Node->getOperand(0).get())->getString();
    auto *F = M.getFunction(KernelName);
    assert(F && "Function not found.");
    auto *SpecConstArg = std::prev(F->arg_end());
    assert(SpecConstArg->getName() == "_arg__specialization_constants_buffer" &&
           "Expected a specialisation constants buffer argument. The function "
           "is not a kernel?");

    // Prepare a new function type, a copy of the original without the last arg.
    std::vector<Type *> Arguments;
    SmallVector<AttributeSet, 8> ArgumentAttributes;
    auto FAttrs =
        F->getAttributes().removeParamAttributes(Context, F->arg_size() - 1);
    for (unsigned I = 0; I < F->arg_size() - 1; ++I) {
      Arguments.push_back(F->getArg(I)->getType());
      ArgumentAttributes.push_back(FAttrs.getParamAttrs(I));
    }
    AttributeList NAttrs = AttributeList::get(
        Context, FAttrs.getFnAttrs(), FAttrs.getRetAttrs(), ArgumentAttributes);
    FunctionType *NFTy =
        FunctionType::get(F->getFunctionType()->getReturnType(), Arguments,
                          F->getFunctionType()->isVarArg());
    // Create a new function body and insert it into the module.
    F->setName(KernelName + "_old");
    Function *NF = Function::Create(NFTy, F->getLinkage(), F->getAddressSpace(),
                                    KernelName, &M);
    NF->setComdat(F->getComdat());
    NF->setDSOLocal(true);
    NF->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);
    NF->setAttributes(NAttrs);

    // Replace the uses of implicit arg with the global variable.
    fixupSpecConstantUses(SpecConstArg, KernelName);

    for (Function::arg_iterator I = F->arg_begin(), E = std::prev(F->arg_end()),
                                I2 = NF->arg_begin();
         I != E; ++I, ++I2) {
      // Move the name and users over to the new version.
      I->replaceAllUsesWith(&*I2);
      I2->takeName(&*I);
    }
    NF->splice(NF->begin(), F);

    // Clone metadata of the old function, including debug info descriptor,
    SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
    F->getAllMetadata(MDs);
    for (auto MD : MDs)
      NF->addMetadata(MD.first, *MD.second);
    // update it.
    std::string KN(KernelName);
    KN.append("_old");
    auto *OldMD = getMDNode(M, KN.c_str());
    assert(OldMD && "Failed to find corresponding MDNode");
    F->eraseFromParent();
    OldMD->replaceOperandWith(0, llvm::ConstantAsMetadata::get(NF));
  }
}

PreservedAnalyses
CUDASpecConstantToSymbolPass::run(Module &M, ModuleAnalysisManager &MAM) {
  NamedMDNode *MD = M.getNamedMetadata(::KERNEL_SPEC_CONST_MD_STRING);
  if (!MD) {
    LLVM_DEBUG(llvm::dbgs()
               << "No sycl.specialization-constants-kernel node found.\n");
    setUpPlaceholderEntries(M);
    return PreservedAnalyses::all();
  }

  // Allocate global variable that corresponds to a kernel's accumulated spec
  // constants.
  allocatePerKernelGlobals(MD);

  // Now, re-write the signature of the kernel to drop the extra spec constant
  // argument and patch in accesses to the global symbol (instead of the arg).
  rewriteKernelSignature(MD);

  SpecConstGlobals.clear();

  return PreservedAnalyses::none();
}
