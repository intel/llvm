//===- SemaSYCL.cpp - Semantic Analysis for SYCL constructs ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This implements Semantic Analysis for SYCL constructs.
//===----------------------------------------------------------------------===//

#include "clang/Sema/Sema.h"

using namespace clang;
// -----------------------------------------------------------------------------
// SYCL device specific diagnostics implementation
// -----------------------------------------------------------------------------

// Do we know that we will eventually codegen the given function?
static bool isKnownEmitted(Sema &S, FunctionDecl *FD) {
  assert(FD && "Given function may not be null.");

  if (FD->hasAttr<SYCLKernelAttr>())
    return true;

  // Otherwise, the function is known-emitted if it's in our set of
  // known-emitted functions.
  return S.DeviceKnownEmittedFns.count(FD) > 0;
}

Sema::DeviceDiagBuilder Sema::SYCLDiagIfDeviceCode(SourceLocation Loc,
                                                   unsigned DiagID) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL compilation");
  FunctionDecl *FD = dyn_cast<FunctionDecl>(getCurLexicalContext());
  DeviceDiagBuilder::Kind DiagKind = [this, FD] {
    if (!FD || isUnevaluatedContext())
      return DeviceDiagBuilder::K_Nop;
    if (isKnownEmitted(*this, FD))
      return DeviceDiagBuilder::K_ImmediateWithCallStack;
    return DeviceDiagBuilder::K_Deferred;
  }();
  return DeviceDiagBuilder(DiagKind, Loc, DiagID, FD, *this);
}

void Sema::checkSYCLDeviceFunction(SourceLocation Loc, FunctionDecl *Callee) {
  assert(Callee && "Callee may not be null.");
  FunctionDecl *Caller = dyn_cast<FunctionDecl>(getCurLexicalContext());

  // If the caller is known-emitted, mark the callee as known-emitted.
  // Otherwise, mark the call in our call graph so we can traverse it later.
  if (Caller && isKnownEmitted(*this, Caller))
    markKnownEmitted(*this, Caller, Callee, Loc, isKnownEmitted);
  else if (Caller)
    DeviceCallGraph[Caller].insert({Callee, Loc});
}
