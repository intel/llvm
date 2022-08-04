// Copyright (C) Codeplay Software Limited

//===- clang-mlir.h - Emit MLIR IRs by walking clang AST---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_MLIR_H
#define CLANG_MLIR_H

#include "AffineUtils.h"
#include "ValueCategory.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/TypeFromLLVM.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "polygeist/Ops.h"
#include "pragmaHandler.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/CommandLine.h"

#include "clang/../../lib/CodeGen/CGRecordLayout.h"
#include "clang/../../lib/CodeGen/CodeGenModule.h"
#include "clang/AST/Mangle.h"

using namespace clang;
using namespace mlir;

extern llvm::cl::opt<std::string> PrefixABI;

struct LoopContext {
  mlir::Value keepRunning;
  mlir::Value noBreak;
};

struct MLIRASTConsumer : public ASTConsumer {
  std::set<std::string> &emitIfFound;
  std::set<std::string> &done;
  std::map<std::string, mlir::LLVM::GlobalOp> &llvmStringGlobals;
  std::map<std::string, std::pair<mlir::memref::GlobalOp, bool>> &globals;
  std::map<std::string, mlir::func::FuncOp> &functions;
  std::map<std::string, mlir::LLVM::GlobalOp> &llvmGlobals;
  std::map<std::string, mlir::LLVM::LLVMFuncOp> &llvmFunctions;
  Preprocessor &PP;
  ASTContext &astContext;
  mlir::OwningOpRef<mlir::ModuleOp> &module;
  clang::SourceManager &SM;
  llvm::LLVMContext lcontext;
  llvm::Module llvmMod;
  CodeGenOptions &codegenops;
  CodeGen::CodeGenModule CGM;
  bool error;
  ScopLocList scopLocList;
  LowerToInfo LTInfo;

  /// The stateful type translator (contains named structs).
  LLVM::TypeFromLLVMIRTranslator typeTranslator;
  LLVM::TypeToLLVMIRTranslator reverseTypeTranslator;

  MLIRASTConsumer(
      std::set<std::string> &emitIfFound, std::set<std::string> &done,
      std::map<std::string, mlir::LLVM::GlobalOp> &llvmStringGlobals,
      std::map<std::string, std::pair<mlir::memref::GlobalOp, bool>> &globals,
      std::map<std::string, mlir::func::FuncOp> &functions,
      std::map<std::string, mlir::LLVM::GlobalOp> &llvmGlobals,
      std::map<std::string, mlir::LLVM::LLVMFuncOp> &llvmFunctions,
      Preprocessor &PP, ASTContext &astContext,
      mlir::OwningOpRef<mlir::ModuleOp> &module, clang::SourceManager &SM,
      CodeGenOptions &codegenops)
      : emitIfFound(emitIfFound), done(done),
        llvmStringGlobals(llvmStringGlobals), globals(globals),
        functions(functions), llvmGlobals(llvmGlobals),
        llvmFunctions(llvmFunctions), PP(PP), astContext(astContext),
        module(module), SM(SM), lcontext(), llvmMod("tmp", lcontext),
        codegenops(codegenops),
        CGM(astContext, PP.getHeaderSearchInfo().getHeaderSearchOpts(),
            PP.getPreprocessorOpts(), codegenops, llvmMod, PP.getDiagnostics()),
        error(false), typeTranslator(*module->getContext()),
        reverseTypeTranslator(lcontext) {
    addPragmaScopHandlers(PP, scopLocList);
    addPragmaEndScopHandlers(PP, scopLocList);
    addPragmaLowerToHandlers(PP, LTInfo);
  }

  ~MLIRASTConsumer() {}

  mlir::func::FuncOp GetOrCreateMLIRFunction(const FunctionDecl *FD,
                                             const bool ShouldEmit,
                                             bool getDeviceStub = false);

  mlir::LLVM::LLVMFuncOp GetOrCreateLLVMFunction(const FunctionDecl *FD);
  mlir::LLVM::LLVMFuncOp GetOrCreateMallocFunction();
  mlir::LLVM::LLVMFuncOp GetOrCreateFreeFunction();

  mlir::LLVM::GlobalOp GetOrCreateLLVMGlobal(const ValueDecl *VD,
                                             std::string prefix = "");

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  mlir::Value GetOrCreateGlobalLLVMString(mlir::Location loc,
                                          mlir::OpBuilder &builder,
                                          StringRef value);

  std::pair<mlir::memref::GlobalOp, bool>
  GetOrCreateGlobal(const ValueDecl *VD, std::string prefix,
                    bool tryInit = true);

  std::deque<const FunctionDecl *> functionsToEmit;

  void run();

  void HandleTranslationUnit(clang::ASTContext &Context) override;

  bool HandleTopLevelDecl(DeclGroupRef dg) override;

  void HandleDeclContext(DeclContext *DC);

  std::map<const clang::RecordType *, mlir::LLVM::LLVMStructType> typeCache;
  // JLE_QUEL::TODO: Possibly create a SYCLTypeCache
  mlir::Type getMLIRType(clang::QualType t, bool *implicitRef = nullptr,
                         bool allowMerge = true);
  mlir::Type getSYCLType(const clang::RecordType *RT);
  llvm::Type *getLLVMType(clang::QualType t);

  mlir::Location getMLIRLocation(clang::SourceLocation loc);
};

class MLIRScanner : public StmtVisitor<MLIRScanner, ValueCategory> {
private:
  friend class IfScope;
  MLIRASTConsumer &Glob;
  mlir::func::FuncOp function;
  mlir::OwningOpRef<mlir::ModuleOp> &module;
  mlir::OpBuilder builder;
  mlir::Location loc;
  mlir::Block *entryBlock;
  std::vector<LoopContext> loops;
  mlir::Block *allocationScope;

  // ValueCategory getValue(std::string name);

  std::map<const void *, std::vector<mlir::LLVM::AllocaOp>> bufs;
  mlir::LLVM::AllocaOp allocateBuffer(size_t i, mlir::LLVM::LLVMPointerType t) {
    auto &vec = bufs[t.getAsOpaquePointer()];
    if (i < vec.size())
      return vec[i];

    mlir::OpBuilder subbuilder(builder.getContext());
    subbuilder.setInsertionPointToStart(allocationScope);

    auto one = subbuilder.create<arith::ConstantIntOp>(loc, 1, 64);
    auto rs = subbuilder.create<mlir::LLVM::AllocaOp>(loc, t, one, 0);
    vec.push_back(rs);
    return rs;
  }

  mlir::Location getMLIRLocation(clang::SourceLocation loc);

  llvm::Type *getLLVMType(clang::QualType t);
  mlir::Type getMLIRType(clang::QualType t);

  mlir::Value getTypeSize(clang::QualType t);
  mlir::Value getTypeAlign(clang::QualType t);

  mlir::Value createAllocOp(mlir::Type t, VarDecl *name, uint64_t memspace,
                            bool isArray, bool LLVMABI);

  const clang::FunctionDecl *EmitCallee(const Expr *E);

  mlir::func::FuncOp EmitDirectCallee(const FunctionDecl *FD);

  std::map<int, mlir::Value> constants;

  mlir::Value castToIndex(mlir::Location loc, mlir::Value val);

  bool isTrivialAffineLoop(clang::ForStmt *fors,
                           mlirclang::AffineLoopDescriptor &descr);

  bool getUpperBound(clang::ForStmt *fors,
                     mlirclang::AffineLoopDescriptor &descr);

  bool getLowerBound(clang::ForStmt *fors,
                     mlirclang::AffineLoopDescriptor &descr);

  bool getConstantStep(clang::ForStmt *fors,
                       mlirclang::AffineLoopDescriptor &descr);

  void buildAffineLoop(clang::ForStmt *fors, mlir::Location loc,
                       const mlirclang::AffineLoopDescriptor &descr);

  void buildAffineLoopImpl(clang::ForStmt *fors, mlir::Location loc,
                           mlir::Value lb, mlir::Value ub,
                           const mlirclang::AffineLoopDescriptor &descr);

public:
  const FunctionDecl *EmittingFunctionDecl;
  std::map<const VarDecl *, ValueCategory> params;
  llvm::DenseMap<const VarDecl *, FieldDecl *> Captures;
  llvm::DenseMap<const VarDecl *, LambdaCaptureKind> CaptureKinds;
  FieldDecl *ThisCapture;
  std::vector<mlir::Value> arrayinit;
  ValueCategory ThisVal;
  mlir::Value returnVal;
  LowerToInfo &LTInfo;

  MLIRScanner(MLIRASTConsumer &Glob, mlir::OwningOpRef<mlir::ModuleOp> &module,
              LowerToInfo &LTInfo);

  void init(mlir::func::FuncOp function, const FunctionDecl *fd);

  void setEntryAndAllocBlock(mlir::Block *B) {
    allocationScope = entryBlock = B;
    builder.setInsertionPointToStart(B);
  }

  mlir::OpBuilder &getBuilder();

  mlir::Value getConstantIndex(int x);

  ValueCategory VisitDeclStmt(clang::DeclStmt *decl);

  ValueCategory VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *decl);

  ValueCategory VisitConstantExpr(clang::ConstantExpr *expr);

  ValueCategory VisitAtomicExpr(clang::AtomicExpr *expr);

  ValueCategory VisitTypeTraitExpr(clang::TypeTraitExpr *expr);

  ValueCategory VisitGNUNullExpr(clang::GNUNullExpr *expr);
  ValueCategory VisitIntegerLiteral(clang::IntegerLiteral *expr);

  ValueCategory VisitCharacterLiteral(clang::CharacterLiteral *expr);

  ValueCategory VisitFloatingLiteral(clang::FloatingLiteral *expr);

  ValueCategory VisitImaginaryLiteral(clang::ImaginaryLiteral *expr);

  ValueCategory VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *expr);

  ValueCategory VisitCXXTypeidExpr(clang::CXXTypeidExpr *expr);

  ValueCategory VisitCXXTryStmt(clang::CXXTryStmt *stmt);

  ValueCategory VisitStringLiteral(clang::StringLiteral *expr);

  ValueCategory VisitParenExpr(clang::ParenExpr *expr);

  ValueCategory VisitVarDecl(clang::VarDecl *decl);

  ValueCategory VisitForStmt(clang::ForStmt *fors);

  ValueCategory VisitCXXForRangeStmt(clang::CXXForRangeStmt *fors);

  ValueCategory VisitOMPSingleDirective(clang::OMPSingleDirective *);

  ValueCategory VisitOMPForDirective(clang::OMPForDirective *);

  ValueCategory VisitOMPParallelDirective(clang::OMPParallelDirective *);

  ValueCategory VisitExtVectorElementExpr(clang::ExtVectorElementExpr *);

  ValueCategory
  VisitOMPParallelForDirective(clang::OMPParallelForDirective *fors);

  ValueCategory VisitWhileStmt(clang::WhileStmt *fors);

  ValueCategory VisitDoStmt(clang::DoStmt *fors);

  ValueCategory VisitArraySubscriptExpr(clang::ArraySubscriptExpr *expr);

  ValueCategory VisitCallExpr(clang::CallExpr *expr);

  ValueCategory
  CallHelper(mlir::func::FuncOp tocall, QualType objType,
             ArrayRef<std::pair<ValueCategory, clang::Expr *>> arguments,
             QualType retType, bool retReference, clang::Expr *expr);

  std::pair<ValueCategory, bool>
  EmitClangBuiltinCallExpr(clang::CallExpr *expr);

  std::pair<ValueCategory, bool> EmitGPUCallExpr(clang::CallExpr *expr);

  std::pair<ValueCategory, bool> EmitBuiltinOps(clang::CallExpr *expr);

  mlir::Operation *EmitSYCLOps(const clang::Expr *Expr,
                               const llvm::SmallVectorImpl<mlir::Value> &Args);

  ValueCategory
  VisitCXXScalarValueInitExpr(clang::CXXScalarValueInitExpr *expr);

  ValueCategory
  VisitCXXPseudoDestructorExpr(clang::CXXPseudoDestructorExpr *expr);

  ValueCategory VisitCXXConstructExpr(clang::CXXConstructExpr *expr);

  ValueCategory VisitConstructCommon(clang::CXXConstructExpr *expr,
                                     VarDecl *name, unsigned space,
                                     mlir::Value mem = nullptr,
                                     mlir::Value count = nullptr);

  ValueCategory VisitMSPropertyRefExpr(clang::MSPropertyRefExpr *expr);

  ValueCategory VisitPseudoObjectExpr(clang::PseudoObjectExpr *expr);

  ValueCategory VisitUnaryOperator(clang::UnaryOperator *U);

  ValueCategory
  VisitSubstNonTypeTemplateParmExpr(clang::SubstNonTypeTemplateParmExpr *expr);

  ValueCategory
  VisitUnaryExprOrTypeTraitExpr(clang::UnaryExprOrTypeTraitExpr *Uop);

  ValueCategory VisitBinaryOperator(clang::BinaryOperator *BO);

  ValueCategory VisitCXXNoexceptExpr(clang::CXXNoexceptExpr *AS);

  ValueCategory VisitAttributedStmt(clang::AttributedStmt *AS);

  ValueCategory VisitExprWithCleanups(clang::ExprWithCleanups *E);

  ValueCategory VisitDeclRefExpr(clang::DeclRefExpr *E);

  ValueCategory VisitOpaqueValueExpr(clang::OpaqueValueExpr *E);

  ValueCategory VisitMemberExpr(clang::MemberExpr *ME);

  ValueCategory VisitCastExpr(clang::CastExpr *E);

  mlir::Value GetAddressOfBaseClass(mlir::Value obj,
                                    const CXXRecordDecl *DerivedClass,
                                    ArrayRef<const clang::Type *> BaseTypes,
                                    ArrayRef<bool> BaseVirtuals);

  mlir::Value GetAddressOfDerivedClass(mlir::Value obj,
                                       const CXXRecordDecl *DerivedClass,
                                       CastExpr::path_const_iterator Start,
                                       CastExpr::path_const_iterator End);

  ValueCategory VisitIfStmt(clang::IfStmt *stmt);

  ValueCategory VisitSwitchStmt(clang::SwitchStmt *stmt);

  ValueCategory VisitConditionalOperator(clang::ConditionalOperator *E);

  ValueCategory VisitCompoundStmt(clang::CompoundStmt *stmt);

  ValueCategory VisitBreakStmt(clang::BreakStmt *stmt);

  ValueCategory VisitContinueStmt(clang::ContinueStmt *stmt);

  ValueCategory VisitReturnStmt(clang::ReturnStmt *stmt);

  std::map<LabelStmt *, Block *> labels;
  ValueCategory VisitLabelStmt(clang::LabelStmt *stmt);

  ValueCategory VisitGotoStmt(clang::GotoStmt *stmt);

  ValueCategory VisitStmtExpr(clang::StmtExpr *stmt);

  ValueCategory VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr *expr);

  ValueCategory
  VisitMaterializeTemporaryExpr(clang::MaterializeTemporaryExpr *expr);

  ValueCategory VisitCXXNewExpr(clang::CXXNewExpr *expr);

  ValueCategory VisitCXXDeleteExpr(clang::CXXDeleteExpr *expr);

  ValueCategory VisitCXXDefaultInitExpr(clang::CXXDefaultInitExpr *expr);

  ValueCategory VisitCXXThisExpr(clang::CXXThisExpr *expr);

  ValueCategory VisitPredefinedExpr(clang::PredefinedExpr *expr);

  ValueCategory VisitLambdaExpr(clang::LambdaExpr *expr);

  ValueCategory VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *expr);

  ValueCategory VisitCXXFunctionalCastExpr(clang::CXXFunctionalCastExpr *expr);

  mlir::Attribute InitializeValueByInitListExpr(mlir::Value toInit,
                                                clang::Expr *expr);
  ValueCategory VisitInitListExpr(clang::InitListExpr *expr);
  ValueCategory
  VisitCXXStdInitializerListExpr(clang::CXXStdInitializerListExpr *expr);

  ValueCategory VisitArrayInitLoop(clang::ArrayInitLoopExpr *expr,
                                   ValueCategory tostore);

  ValueCategory VisitArrayInitIndexExpr(clang::ArrayInitIndexExpr *expr);

  ValueCategory CommonFieldLookup(clang::QualType OT, const FieldDecl *FD,
                                  mlir::Value val, bool isLValue);

  ValueCategory CommonArrayLookup(ValueCategory val, mlir::Value idx,
                                  bool isImplicitRefResult,
                                  bool removeIndex = true);

  ValueCategory CommonArrayToPointer(ValueCategory val);
};

#endif
