//===--- CodeGenTypes.h - Type translation for MLIR CodeGen -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the code that handles AST -> MLIR type lowering.
//
//===----------------------------------------------------------------------===//

#ifndef CGEIST_LIB_CODEGEN_CODEGENTYPES_H
#define CGEIST_LIB_CODEGEN_CODEGENTYPES_H

#include "clang/Basic/ABI.h"
#include "clang/CodeGen/CGFunctionInfo.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir {
class FunctionType;
class ModuleOp;
class Type;
} // namespace mlir

namespace clang {
class ASTContext;
template <typename> class CanQual;
class CodeGenOptions;
class QualType;
class RecordType;
// class TargetInfo;
class Type;
typedef CanQual<Type> CanQualType;
class GlobalDecl;

namespace CodeGen {
class ABIInfo;
class CGCXXABI;
// class CGRecordLayout;
class CodeGenModule;
} // namespace CodeGen
} // namespace clang

namespace mlirclang {
namespace CodeGen {

/// This class organizes the cross-module state that is used while lowering
/// AST types to MLIR types.
class CodeGenTypes {
  clang::CodeGen::CodeGenModule &CGM;
  clang::ASTContext &Context;
  mlir::OwningOpRef<mlir::ModuleOp> &TheModule;
  clang::CodeGen::CGCXXABI &TheCXXABI;
  const clang::CodeGen::ABIInfo &TheABIInfo;

  std::map<const clang::RecordType *, mlir::LLVM::LLVMStructType> TypeCache;

public:
  CodeGenTypes(clang::CodeGen::CodeGenModule &CGM,
               mlir::OwningOpRef<mlir::ModuleOp> &Module);

  clang::CodeGen::CodeGenModule &getCGM() const { return CGM; }
  clang::ASTContext &getContext() const { return Context; }
  const clang::CodeGen::ABIInfo &getABIInfo() const { return TheABIInfo; }
  mlir::OwningOpRef<mlir::ModuleOp> &getModule() { return TheModule; }
  clang::CodeGen::CGCXXABI &getCXXABI() const { return TheCXXABI; }
  const clang::CodeGenOptions &getCodeGenOpts() const;

  mlir::FunctionType getFunctionType(const clang::CodeGen::CGFunctionInfo &FI,
                                     const clang::FunctionDecl &FD);

  // TODO: Possibly create a SYCLTypeCache
  mlir::Type getMLIRType(clang::QualType QT, bool *ImplicitRef = nullptr,
                         bool AllowMerge = true);

  const clang::CodeGen::CGFunctionInfo &
  arrangeGlobalDeclaration(clang::GlobalDecl GD);
};

} // namespace CodeGen
} // namespace mlirclang

#endif // CGEIST_LIB_CODEGEN_CODEGENTYPES_H
