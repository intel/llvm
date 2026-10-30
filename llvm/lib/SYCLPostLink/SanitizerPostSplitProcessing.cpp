//===------------------ SanitizerPostSplitProcessing.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See comments in the header.
//===----------------------------------------------------------------------===//
#include "llvm/SYCLPostLink/SanitizerPostSplitProcessing.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/SYCLLowerIR/SanitizerUtils.h"

using namespace llvm;

namespace {

constexpr StringRef KernelMetadataPrefixes[] = {
    sycl::utils::ASAN_KERNEL_METADATA_PREFIX,
    sycl::utils::MSAN_KERNEL_METADATA_PREFIX,
    sycl::utils::TSAN_KERNEL_METADATA_PREFIX};

bool isSanitizerKernelMetadata(const GlobalVariable &GV) {
  return any_of(KernelMetadataPrefixes, [&GV](StringRef Prefix) {
    return GV.getName().starts_with(Prefix);
  });
}

// Returns the name of the kernel described by a sanitizer kernel metadata
// entry { i64 ptrtoint (ptr addrspace(2) @__asan_kernel to i64), i64 <size>,
// ... }, or an empty string if it can't be determined.
StringRef getKernelName(Constant *Entry) {
  auto *NameField =
      dyn_cast_or_null<ConstantExpr>(Entry->getAggregateElement(0u));
  if (!NameField || NameField->getOpcode() != Instruction::PtrToInt)
    return {};
  auto *NameGV = dyn_cast<GlobalVariable>(NameField->getOperand(0));
  if (!NameGV || !NameGV->hasInitializer())
    return {};
  auto *Name = dyn_cast<ConstantDataSequential>(NameGV->getInitializer());
  if (!Name || !Name->isCString())
    return {};
  return Name->getAsCString();
}

bool isKernelDefined(const Module &M, StringRef Name) {
  const Function *F = M.getFunction(Name);
  return F && !F->isDeclaration() && F->hasKernelCallingConv();
}

// Module splitting copies the sanitizer kernel metadata of the whole program,
// along with the name strings of all the kernels it references, into every
// split module. With -split=kernel this makes the total size of the device
// images quadratic in the number of kernels, so zero out the entries of the
// kernels which aren't defined in this module, which lets the name strings be
// removed by the module cleanup. The sanitizer runtime skips entries with a
// zero size. The number of entries must be kept as is: the metadata is a
// device global, and all device images must agree on its size.
bool pruneSanitizerKernelMetadata(Module &M) {
  bool Modified = false;
  for (GlobalVariable &KernelMetadata : M.globals()) {
    if (!KernelMetadata.hasInitializer() ||
        !isSanitizerKernelMetadata(KernelMetadata))
      continue;

    // SanitizerPostOptimizerPass has wrapped the array of entries into a
    // structure type.
    auto *Wrapper = dyn_cast<ConstantStruct>(KernelMetadata.getInitializer());
    if (!Wrapper || Wrapper->getNumOperands() != 1)
      continue;
    auto *Entries = dyn_cast<ConstantArray>(Wrapper->getOperand(0));
    if (!Entries)
      continue;

    SmallVector<Constant *, 8> NewEntries;
    bool Changed = false;
    for (Use &Op : Entries->operands()) {
      auto *Entry = cast<Constant>(Op);
      StringRef KernelName = getKernelName(Entry);
      // Conservatively keep the entries which can't be matched to a kernel.
      if (!KernelName.empty() && !isKernelDefined(M, KernelName)) {
        NewEntries.push_back(Constant::getNullValue(Entry->getType()));
        Changed = true;
      } else {
        NewEntries.push_back(Entry);
      }
    }

    if (!Changed)
      continue;

    KernelMetadata.setInitializer(ConstantStruct::get(
        Wrapper->getType(),
        ConstantArray::get(Entries->getType(), NewEntries)));
    Modified = true;
  }

  return Modified;
}

bool createSanitizerModuleID(Module &M) {
  SmallVector<StringRef, 3> ModuleIDs;
  for (GlobalVariable &GV : M.globals()) {
    auto GVName = GV.getName();
    for (StringRef Prefix : KernelMetadataPrefixes) {
      if (GVName.starts_with(Prefix)) {
        ModuleIDs.push_back(GVName.drop_front(Prefix.size()));
        break;
      }
    }
  }

  if (ModuleIDs.empty())
    return false;

  for (StringRef ModuleID : ModuleIDs) {
    FunctionType *FTy =
        FunctionType::get(Type::getVoidTy(M.getContext()), {}, false);
    Function *F = Function::Create(FTy, GlobalValue::WeakODRLinkage,
                                   "__sanitizerModule" + ModuleID, &M);
    F->setCallingConv(CallingConv::SPIR_KERNEL);
    F->setDSOLocal(true);
    F->addFnAttr("sycl-entry-point");
    BasicBlock *BB = BasicBlock::Create(M.getContext(), "entry", F);
    ReturnInst::Create(M.getContext(), BB);
  }

  return true;
}

} // namespace

bool llvm::sycl_post_link::handleSanitizers(
    llvm::SmallVectorImpl<std::unique_ptr<module_split::ModuleDesc>> &MDs) {
  bool Modified = false;
  for (std::unique_ptr<module_split::ModuleDesc> &MD : MDs) {
    Modified |= pruneSanitizerKernelMetadata(MD->getModule());
    Modified |= createSanitizerModuleID(MD->getModule());
  }
  return Modified;
}
