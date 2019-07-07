//===----- CGSYCLRuntime.cpp - Interface to SYCL Runtimes -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides custom clang code generation for SYCL.
//
//===----------------------------------------------------------------------===//

#include "CGSYCLRuntime.h"
#include "CodeGenFunction.h"
#include "clang/AST/Decl.h"
#include "llvm/IR/Instructions.h"
#include <assert.h>

using namespace clang;
using namespace CodeGen;

const char *WG_SCOPE_MD_ID = "work_group_scope";
const char *WI_SCOPE_MD_ID = "work_item_scope";

bool CGSYCLRuntime::actOnFunctionStart(const FunctionDecl &FD,
                                       llvm::Function &F) {
  SYCLScopeAttr *Scope = FD.getAttr<SYCLScopeAttr>();
  if (!Scope)
    return false;
  switch (Scope->getLevel()) {
  case SYCLScopeAttr::Level::WorkGroup:
    F.setMetadata(WG_SCOPE_MD_ID, llvm::MDNode::get(F.getContext(), {}));
    break;
  case SYCLScopeAttr::Level::WorkItem:
    F.setMetadata(WI_SCOPE_MD_ID, llvm::MDNode::get(F.getContext(), {}));
    break;
  default:
    llvm_unreachable("unknown sycl scope");
  }
  return true;
}

void CGSYCLRuntime::emitWorkGroupLocalVarDecl(CodeGenFunction &CGF,
                                              const VarDecl &D) {
#ifndef NDEBUG
  SYCLScopeAttr *Scope = D.getAttr<SYCLScopeAttr>();
  assert(Scope && Scope->isWorkGroup() && "work group scope expected");
#endif // NDEBUG
  // generate global variable in the address space selected by the clang CodeGen
  // (should be local)
  return CGF.EmitStaticVarDecl(D, llvm::GlobalValue::InternalLinkage);
}
