//==------------------------ PostLinkActions.cpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PostLinkActions.h"

#include <llvm/IR/Constants.h>
#include <llvm/SYCLLowerIR/DeviceGlobals.h>
#include <llvm/Transforms/Utils/GlobalStatus.h>

using namespace llvm;

bool jit_compiler::post_link::removeSYCLKernelsConstRefArray(Module &M) {
  GlobalVariable *GV = M.getGlobalVariable("llvm.used");

  if (!GV) {
    return false;
  }
  assert(GV->user_empty() && "Unexpected llvm.used users");
  Constant *Initializer = GV->getInitializer();
  GV->setInitializer(nullptr);
  GV->eraseFromParent();

  // Destroy the initializer and all operands of it.
  SmallVector<Constant *, 8> IOperands;
  for (auto It = Initializer->op_begin(); It != Initializer->op_end(); It++)
    IOperands.push_back(cast<Constant>(*It));
  assert(llvm::isSafeToDestroyConstant(Initializer) &&
         "Cannot remove initializer of llvm.used global");
  Initializer->destroyConstant();
  for (auto It = IOperands.begin(); It != IOperands.end(); It++) {
    auto Op = (*It)->stripPointerCasts();
    auto *F = dyn_cast<Function>(Op);
    if (llvm::isSafeToDestroyConstant(*It)) {
      (*It)->destroyConstant();
    } else if (F && F->getCallingConv() == CallingConv::SPIR_KERNEL &&
               !F->use_empty()) {
      // The element in "llvm.used" array has other users. That is Ok for
      // specialization constants, but is wrong for kernels.
      llvm::report_fatal_error("Unexpected usage of SYCL kernel");
    }

    // Remove unused kernel declarations to avoid LLVM IR check fails.
    if (F && F->isDeclaration() && F->use_empty())
      F->eraseFromParent();
  }
  return true;
}

bool jit_compiler::post_link::removeDeviceGlobalFromCompilerUsed(Module &M) {
  GlobalVariable *GV = M.getGlobalVariable("llvm.compiler.used");
  if (!GV)
    return false;

  // Erase the old llvm.compiler.used. A new one will be created at the end if
  // there are other values in it (other than device_global).
  assert(GV->user_empty() && "Unexpected llvm.compiler.used users");
  Constant *Initializer = GV->getInitializer();
  const auto *VAT = cast<ArrayType>(GV->getValueType());
  GV->setInitializer(nullptr);
  GV->eraseFromParent();

  // Destroy the initializer. Keep the operands so we keep the ones we need.
  SmallVector<Constant *, 8> IOperands;
  for (auto It = Initializer->op_begin(); It != Initializer->op_end(); It++)
    IOperands.push_back(cast<Constant>(*It));
  assert(llvm::isSafeToDestroyConstant(Initializer) &&
         "Cannot remove initializer of llvm.compiler.used global");
  Initializer->destroyConstant();

  // Iterate through all operands. If they are device_global then we drop them
  // and erase them if they have no uses afterwards. All other values are kept.
  SmallVector<Constant *, 8> NewOperands;
  for (auto It = IOperands.begin(); It != IOperands.end(); It++) {
    Constant *Op = *It;
    auto *DG = dyn_cast<GlobalVariable>(Op->stripPointerCasts());

    // If it is not a device_global we keep it.
    if (!DG || !isDeviceGlobalVariable(*DG)) {
      NewOperands.push_back(Op);
      continue;
    }

    // Destroy the device_global operand.
    if (llvm::isSafeToDestroyConstant(Op))
      Op->destroyConstant();

    // Remove device_global if it no longer has any uses.
    if (!DG->isConstantUsed())
      DG->eraseFromParent();
  }

  // If we have any operands left from the original llvm.compiler.used we create
  // a new one with the new size.
  if (!NewOperands.empty()) {
    ArrayType *ATy = ArrayType::get(VAT->getElementType(), NewOperands.size());
    GlobalVariable *NGV =
        new GlobalVariable(M, ATy, false, GlobalValue::AppendingLinkage,
                           ConstantArray::get(ATy, NewOperands), "");
    NGV->setName("llvm.compiler.used");
    NGV->setSection("llvm.metadata");
  }

  return true;
}
