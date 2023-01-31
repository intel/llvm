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

#include "Attributes.h"
#include "mlir/IR/OwningOpRef.h"
#include "clang/Basic/ABI.h"
#include <map>

namespace clang {
class ASTContext;
class CodeGenOptions;
class FunctionDecl;
class GlobalDecl;
class RecordDecl;
class BuiltinType;
class QualType;
class RecordType;
class Type;

namespace CodeGen {
class ABIInfo;
class CGCXXABI;
class CGCalleeInfo;
class CGFunctionInfo;
class CodeGenModule;
} // namespace CodeGen
} // namespace clang

namespace mlir {
class FunctionType;
class ModuleOp;
class Type;

namespace LLVM {
class LLVMStructType;
}
} // namespace mlir

namespace mlirclang {
namespace CodeGen {

/// This class organizes the cross-module state that is used while lowering
/// AST types to MLIR types.
class CodeGenTypes {
  clang::CodeGen::CodeGenModule &CGM;
  clang::ASTContext &Context;
  mlir::OwningOpRef<mlir::ModuleOp> &TheModule;
  clang::CodeGen::CGCXXABI &TheCXXABI;

  std::map<const clang::RecordType *, mlir::LLVM::LLVMStructType> TypeCache;
  std::map<const clang::Type *, mlir::Type> BuiltinTypeCache;

public:
  CodeGenTypes(clang::CodeGen::CodeGenModule &CGM,
               mlir::OwningOpRef<mlir::ModuleOp> &Module);

  clang::CodeGen::CodeGenModule &getCGM() const { return CGM; }
  clang::ASTContext &getContext() const { return Context; }
  mlir::OwningOpRef<mlir::ModuleOp> &getModule() const { return TheModule; }
  clang::CodeGen::CGCXXABI &getCXXABI() const { return TheCXXABI; }
  const clang::CodeGenOptions &getCodeGenOpts() const;

  /// Construct the MLIR function type.
  mlir::FunctionType getFunctionType(const clang::CodeGen::CGFunctionInfo &FI,
                                     const clang::FunctionDecl &FD);

  /// Construct the IR attribute list of a function type or function call.
  void constructAttributeList(llvm::StringRef Name,
                              const clang::CodeGen::CGFunctionInfo &FI,
                              clang::CodeGen::CGCalleeInfo CalleeInfo,
                              mlirclang::AttributeList &AttrList,
                              unsigned &CallingConv, bool AttrOnCallSite,
                              bool IsThunk);

  /// Convert type T into an mlir::Type.
  ///
  /// This differs from getMLIRType in that it is used to convert to the memory
  /// representation for a type.  For example, the scalar representation for
  /// _Bool is i1, but the memory representation is usually i8 or i32, depending
  /// on the target.
  mlir::Type getMLIRTypeForMem(clang::QualType QT, bool *ImplicitRef = nullptr,
                               bool AllowMerge = true);
  // TODO: Possibly create a SYCLTypeCache
  mlir::Type getMLIRType(clang::QualType QT, bool *ImplicitRef = nullptr,
                         bool AllowMerge = true);

  mlir::Type getPointerOrMemRefType(mlir::Type Ty, unsigned AddressSpace,
                                    bool IsAlloca = false) const;

  const clang::CodeGen::CGFunctionInfo &
  arrangeGlobalDeclaration(clang::GlobalDecl GD);

  static bool isLLVMStructABI(const clang::RecordDecl *RD,
                              llvm::StructType *ST);

  clang::QualType getPromotionType(clang::QualType Ty) const;

private:
  void getDefaultFunctionAttributes(llvm::StringRef Name, bool HasOptnone,
                                    bool AttrOnCallSite,
                                    mlirclang::AttrBuilder &FuncAttrs) const;

  bool getCPUAndFeaturesAttributes(clang::GlobalDecl GD,
                                   AttrBuilder &Attrs) const;

  mlir::Type getMLIRType(const clang::BuiltinType *BT) const;
};

} // namespace CodeGen
} // namespace mlirclang

#endif // CGEIST_LIB_CODEGEN_CODEGENTYPES_H
