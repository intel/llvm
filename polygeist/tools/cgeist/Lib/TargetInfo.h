//===---- TargetInfo.h - Encapsulate target details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function definition used
// to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#ifndef CGEIST_LIB_CODEGEN_TARGETINFO_H
#define CGEIST_LIB_CODEGEN_TARGETINFO_H

#include "clang/../../lib/CodeGen/ABIInfo.h"
#include "clang/../../lib/CodeGen/Address.h"

namespace mlirclang {

/// \class
class DefaultABIInfo : public clang::CodeGen::ABIInfo {
public:
  DefaultABIInfo(clang::CodeGen::CodeGenTypes &CGT) : ABIInfo(CGT) {}

  clang::CodeGen::ABIArgInfo classifyReturnType(clang::QualType RetTy) const;
  clang::CodeGen::ABIArgInfo classifyArgumentType(clang::QualType RetTy) const;

  void computeInfo(clang::CodeGen::CGFunctionInfo &FI) const override;

  clang::CodeGen::Address EmitVAArg(clang::CodeGen::CodeGenFunction &CGF,
                                    clang::CodeGen::Address VAListAddr,
                                    clang::QualType Ty) const override;
};

/// \class
class CommonSPIRABIInfo : public DefaultABIInfo {
public:
  CommonSPIRABIInfo(clang::CodeGen::CodeGenTypes &CGT) : DefaultABIInfo(CGT) {
    setCCs();
  }

  clang::CodeGen::ABIArgInfo
  classifyKernelArgumentType(clang::QualType Ty) const;

  // Add new functions rather than overload existing so that these public APIs
  // can't be blindly misused with wrong calling convention.
  clang::CodeGen::ABIArgInfo
  classifyRegcallReturnType(clang::QualType RetTy) const;
  clang::CodeGen::ABIArgInfo
  classifyRegcallArgumentType(clang::QualType RetTy) const;

private:
  void setCCs();
};

} // end namespace mlirclang

#endif // CGEIST_LIB_CODEGEN_TARGETINFO_H
