//===----- CGSYCLRuntime.h - Interface to SYCL Runtimes ---------*- C++ -*-===//
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

#ifndef LLVM_CLANG_LIB_CODEGEN_CGSYCLRUNTIME_H
#define LLVM_CLANG_LIB_CODEGEN_CGSYCLRUNTIME_H

#include "CodeGenFunction.h"

#include "clang/AST/Decl.h"
#include "llvm/IR/Value.h"

namespace clang {
namespace CodeGen {

class CodeGenModule;

class CGSYCLRuntime {
protected:
  CodeGenModule &CGM;

public:
  CGSYCLRuntime(CodeGenModule &CGM) : CGM(CGM) {}

  bool actOnFunctionStart(const FunctionDecl &FD, llvm::Function &F);
  void emitWorkGroupLocalVarDecl(CodeGenFunction &CGF, const VarDecl &D);
  bool actOnAutoVarEmit(CodeGenFunction &CGF, const VarDecl &D,
                        llvm::Value *Addr);
};

} // namespace CodeGen
} // namespace clang

#endif // LLVM_CLANG_LIB_CODEGEN_CGSYCLRUNTIME_H
