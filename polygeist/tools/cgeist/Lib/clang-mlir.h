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
#include "Attributes.h"
#include "CodeGenTypes.h"
#include "ValueCategory.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "polygeist/Ops.h"
#include "pragmaHandler.h"

#include "clang/../../lib/CodeGen/CGRecordLayout.h"
#include "clang/../../lib/CodeGen/CodeGenModule.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/CommandLine.h"

extern llvm::cl::opt<std::string> PrefixABI;

namespace mlir {
namespace sycl {
class SYCLMethodOpInterface;
} // namespace sycl
} // namespace mlir

namespace mlirclang {
namespace CodeGen {
class CodeGenTypes;
} // namespace CodeGen
} // namespace mlirclang

struct LoopContext {
  mlir::Value keepRunning;
  mlir::Value noBreak;
};

class BinOpInfo;

/// Context in which a function is located.
enum class FunctionContext {
  Host,      ///< Host function
  SYCLDevice ///< SYCL Device function
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &Out,
                                     const FunctionContext &Context) {
  switch (Context) {
  case FunctionContext::Host:
    Out << "Host";
    break;
  case FunctionContext::SYCLDevice:
    Out << "SYCLDevice";
    break;
  }
  return Out;
}

/// class encapsulating a function declaration and its context.
/// Note: SYCL kernel & device functions should always be in a SYCLDevice
///       context. Any other functions may be in a host or device context.
class FunctionToEmit {
public:
  // Note: the context is determined from the given function declarator.
  explicit FunctionToEmit(const clang::FunctionDecl &FuncDecl)
      : FuncDecl(FuncDecl),
        FuncContext((FuncDecl.hasAttr<clang::SYCLKernelAttr>() ||
                     FuncDecl.hasAttr<clang::SYCLDeviceAttr>())
                        ? FunctionContext::SYCLDevice
                        : FunctionContext::Host) {}

  /// Note: set the context requested, ensuring a host context is not requested
  /// for SYCL kernel/functions.
  FunctionToEmit(const clang::FunctionDecl &FuncDecl,
                 FunctionContext FuncContext)
      : FuncDecl(FuncDecl), FuncContext(FuncContext) {
    bool IsSyclFunc = FuncDecl.hasAttr<clang::SYCLKernelAttr>() ||
                      FuncDecl.hasAttr<clang::SYCLDeviceAttr>();
    (void)IsSyclFunc;

    assert((FuncContext == FunctionContext::SYCLDevice ||
            (FuncContext == FunctionContext::Host && !IsSyclFunc)) &&
           "SYCL kernel/device functions should not have host context");
  }

  const clang::FunctionDecl &getDecl() const { return FuncDecl; }
  FunctionContext getContext() const { return FuncContext; }

private:
  const clang::FunctionDecl &FuncDecl;
  const FunctionContext FuncContext;
};

class MLIRASTConsumer : public clang::ASTConsumer {
private:
  std::set<std::string> &emitIfFound;
  std::set<std::pair<FunctionContext, std::string>> &done;
  std::map<std::string, mlir::LLVM::GlobalOp> &llvmStringGlobals;
  std::map<std::string, std::pair<mlir::memref::GlobalOp, bool>> &globals;
  std::map<std::string, mlir::func::FuncOp> &functions;
  std::map<std::string, mlir::FunctionOpInterface> &deviceFunctions;
  std::map<std::string, mlir::LLVM::GlobalOp> &llvmGlobals;
  std::map<std::string, mlir::LLVM::LLVMFuncOp> &llvmFunctions;
  std::deque<FunctionToEmit> functionsToEmit;
  mlir::OwningOpRef<mlir::ModuleOp> &module;
  clang::SourceManager &SM;
  llvm::LLVMContext lcontext;
  llvm::Module llvmMod;
  clang::CodeGen::CodeGenModule CGM;
  mlirclang::CodeGen::CodeGenTypes CGTypes;
  bool error;
  ScopLocList scopLocList;
  LowerToInfo LTInfo;
  std::map<const clang::FunctionDecl *, const clang::CodeGen::CGFunctionInfo *>
      CGFunctionInfos;

public:
  static constexpr llvm::StringLiteral DeviceModuleName{"device_functions"};

  MLIRASTConsumer(
      std::set<std::string> &EmitIfFound,
      std::set<std::pair<FunctionContext, std::string>> &Done,
      std::map<std::string, mlir::LLVM::GlobalOp> &LlvmStringGlobals,
      std::map<std::string, std::pair<mlir::memref::GlobalOp, bool>> &Globals,
      std::map<std::string, mlir::func::FuncOp> &Functions,
      std::map<std::string, mlir::FunctionOpInterface> &DeviceFunctions,
      std::map<std::string, mlir::LLVM::GlobalOp> &LlvmGlobals,
      std::map<std::string, mlir::LLVM::LLVMFuncOp> &LlvmFunctions,
      clang::Preprocessor &PP, clang::ASTContext &AstContext,
      mlir::OwningOpRef<mlir::ModuleOp> &Module, clang::SourceManager &SM,
      clang::CodeGenOptions &Codegenops, std::string ModuleId)
      : emitIfFound(EmitIfFound), done(Done),
        llvmStringGlobals(LlvmStringGlobals), globals(Globals),
        functions(Functions), deviceFunctions(DeviceFunctions),
        llvmGlobals(LlvmGlobals), llvmFunctions(LlvmFunctions),
        functionsToEmit(), module(Module), SM(SM), lcontext(),
        llvmMod(ModuleId, lcontext),
        CGM(AstContext, &SM.getFileManager().getVirtualFileSystem(),
            PP.getHeaderSearchInfo().getHeaderSearchOpts(),
            PP.getPreprocessorOpts(), Codegenops, llvmMod, PP.getDiagnostics()),
        CGTypes(CGM, Module), error(false), scopLocList(), LTInfo() {
    addPragmaScopHandlers(PP, scopLocList);
    addPragmaEndScopHandlers(PP, scopLocList);
    addPragmaLowerToHandlers(PP, LTInfo);
  }

  clang::CodeGen::CodeGenModule &getCGM() { return CGM; }
  mlirclang::CodeGen::CodeGenTypes &getTypes() { return CGTypes; }
  std::map<std::string, mlir::func::FuncOp> &getFunctions() {
    return functions;
  }
  ScopLocList &getScopLocList() { return scopLocList; }

  mlir::FunctionOpInterface GetOrCreateMLIRFunction(FunctionToEmit &FTE,
                                                    const bool ShouldEmit,
                                                    bool GetDeviceStub = false);
  mlir::LLVM::LLVMFuncOp GetOrCreateLLVMFunction(const clang::FunctionDecl *FD);
  mlir::LLVM::LLVMFuncOp GetOrCreateMallocFunction();
  mlir::LLVM::LLVMFuncOp GetOrCreateFreeFunction();

  mlir::LLVM::GlobalOp GetOrCreateLLVMGlobal(const clang::ValueDecl *VD,
                                             std::string Prefix = "");

  /// Return a value representing an access into a global string with the
  /// given name, creating the string if necessary.
  mlir::Value GetOrCreateGlobalLLVMString(mlir::Location Loc,
                                          mlir::OpBuilder &Builder,
                                          clang::StringRef Value);

  /// Create global variable and initialize it.
  std::pair<mlir::memref::GlobalOp, bool>
  getOrCreateGlobal(const clang::ValueDecl &VD, std::string Prefix,
                    FunctionContext FuncContext);

  const clang::CodeGen::CGFunctionInfo &
  GetOrCreateCGFunctionInfo(const clang::FunctionDecl *FD);

  void run();

  void HandleTranslationUnit(clang::ASTContext &Context) override;

  bool HandleTopLevelDecl(clang::DeclGroupRef dg) override;

  void HandleDeclContext(clang::DeclContext *DC);

  mlir::Location getMLIRLocation(clang::SourceLocation Loc);

private:
  /// Returns the LLVM linkage type of the given function declaration \p FD.
  llvm::GlobalValue::LinkageTypes
  getLLVMLinkageType(const clang::FunctionDecl &FD, bool ShouldEmit);

  /// Returns the MLIR LLVM dialect linkage corresponding to \p LV.
  static mlir::LLVM::Linkage getMLIRLinkage(llvm::GlobalValue::LinkageTypes LV);

  /// Returns the MLIR Function type given clang's CGFunctionInfo \p FI.
  mlir::FunctionType getFunctionType(const clang::CodeGen::CGFunctionInfo &FI,
                                     const clang::FunctionDecl &FD);

  /// Returns the MLIR function corresponding to \p mangledName.
  llvm::Optional<mlir::FunctionOpInterface>
  getMLIRFunction(const std::string &MangledName,
                  FunctionContext Context) const;

  /// Create the MLIR function corresponding to the given \p FTE.
  /// The MLIR function is created in either the device module (GPUModuleOp) or
  /// in the host module (ModuleOp), depending on the calling context embedded
  /// in the FTE).
  mlir::FunctionOpInterface createMLIRFunction(const FunctionToEmit &FTE,
                                               std::string MangledName,
                                               bool ShouldEmit);

  /// Set the symbol visibility on the given \p function.
  void setMLIRFunctionVisibility(mlir::FunctionOpInterface Function,
                                 const FunctionToEmit &FTE, bool ShouldEmit);

  /// Set the MLIR function attributes for the given \p function.
  void setMLIRFunctionAttributes(mlir::FunctionOpInterface Function,
                                 const FunctionToEmit &FTE, bool ShouldEmit);

  void setMLIRFunctionAttributesForDefinition(
      const clang::Decl *D, mlir::FunctionOpInterface Function) const;
};

class MLIRScanner : public clang::StmtVisitor<MLIRScanner, ValueCategory> {
  friend class IfScope;

private:
  MLIRASTConsumer &Glob;
  mlir::FunctionOpInterface function;
  mlir::OwningOpRef<mlir::ModuleOp> &module;
  mlir::OpBuilder builder;
  mlir::Location loc;
  mlir::Block *entryBlock;
  std::vector<LoopContext> loops;
  mlir::Block *allocationScope;
  llvm::SmallSet<std::string, 4> unsupportedFuncs;
  std::map<const void *, std::vector<mlir::LLVM::AllocaOp>> bufs;
  std::map<int, mlir::Value> constants;
  std::map<clang::LabelStmt *, mlir::Block *> labels;
  const clang::FunctionDecl *EmittingFunctionDecl;
  std::map<const clang::ValueDecl *, ValueCategory> params;
  llvm::DenseMap<const clang::ValueDecl *, clang::FieldDecl *> Captures;
  llvm::DenseMap<const clang::ValueDecl *, clang::LambdaCaptureKind>
      CaptureKinds;
  clang::FieldDecl *ThisCapture;
  std::vector<mlir::Value> arrayinit;
  ValueCategory ThisVal;
  mlir::Value returnVal;
  LowerToInfo &LTInfo;

  // Initialize a exclude list of SYCL functions to emit instead just the
  // declaration. Eventually, this list should be removed.
  void initUnsupportedFunctions();
  bool isUnsupportedFunction(std::string Name) const {
    return unsupportedFuncs.contains(Name);
  }

  mlir::LLVM::AllocaOp allocateBuffer(size_t i, mlir::LLVM::LLVMPointerType t) {
    auto &Vec = bufs[t.getAsOpaquePointer()];
    if (i < Vec.size())
      return Vec[i];

    mlir::OpBuilder Subbuilder(builder.getContext());
    Subbuilder.setInsertionPointToStart(allocationScope);

    auto One = Subbuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 64);
    auto Rs = Subbuilder.create<mlir::LLVM::AllocaOp>(loc, t, One, 0);
    Vec.push_back(Rs);
    return Rs;
  }

  mlir::Location getMLIRLocation(clang::SourceLocation Loc);

  mlir::Value getTypeSize(clang::QualType Ty);
  mlir::Value getTypeAlign(clang::QualType Ty);

  mlir::Value createAllocOp(mlir::Type Ty, clang::VarDecl *Name,
                            uint64_t Myspace, bool IsArray, bool LLVMABI);

  const clang::FunctionDecl *EmitCallee(const clang::Expr *E);

  mlir::FunctionOpInterface EmitDirectCallee(const clang::FunctionDecl *FD,
                                             FunctionContext Context);

  mlir::Value castToIndex(mlir::Location Loc, mlir::Value Val);

  /// Converts the \p val to the memory space \p memSpace and returns the
  /// converted value.
  mlir::Value castToMemSpace(mlir::Value Val, unsigned MemSpace);

  /// Converts the \p val to the memory space of \p t and returns the
  /// converted value.
  mlir::Value castToMemSpaceOfType(mlir::Value Val, mlir::Type TargetType);

  bool isTrivialAffineLoop(clang::ForStmt *Fors,
                           mlirclang::AffineLoopDescriptor &Descr);

  bool getUpperBound(clang::ForStmt *Fors,
                     mlirclang::AffineLoopDescriptor &Descr);

  bool getLowerBound(clang::ForStmt *Fors,
                     mlirclang::AffineLoopDescriptor &Descr);

  bool getConstantStep(clang::ForStmt *Fors,
                       mlirclang::AffineLoopDescriptor &Descr);

  void buildAffineLoop(clang::ForStmt *Fors, mlir::Location Loc,
                       const mlirclang::AffineLoopDescriptor &Descr);

  void buildAffineLoopImpl(clang::ForStmt *Fors, mlir::Location Loc,
                           mlir::Value LB, mlir::Value UB,
                           const mlirclang::AffineLoopDescriptor &Descr);

  /// Creates an instance of SYCLMethodOpInterface if the SYCLCallOp with base
  /// type name \param typeName, function name \param functionName, mangled
  /// function name \param mangledFunctionName, parameters \param operands and
  /// (optional) return type \param returnType can be represented as such.
  ///
  /// E.g., the SYCLCallOp to the accessor member function
  /// accessor::operator[] can be represented using a SYCLAccessorSubscriptOp.
  llvm::Optional<mlir::sycl::SYCLMethodOpInterface>
  createSYCLMethodOp(llvm::StringRef TypeName, llvm::StringRef FunctionName,
                     mlir::ValueRange Operands,
                     llvm::Optional<mlir::Type> ReturnType,
                     llvm::StringRef MangledFunctionName);

  // Reshape memref<elemTy> to memref<1 x elemTy>.
  mlir::Value reshapeRanklessGlobal(mlir::memref::GetGlobalOp GV);

  ValueCategory CastToVoidPtr(ValueCategory Ptr);

  /// TODO: Add ScalarConversion options
  ValueCategory EmitScalarCast(mlir::Location Loc, ValueCategory Src,
                               clang::QualType SrcType, clang::QualType DstType,
                               mlir::Type SrcTy, mlir::Type DstTy);
  /// TODO: Add ScalarConversion options
  ValueCategory EmitScalarConversion(ValueCategory Src, clang::QualType SrcType,
                                     clang::QualType DstType,
                                     clang::SourceLocation Loc);

  ValueCategory EmitPointerToBoolConversion(mlir::Location Loc,
                                            ValueCategory Src);
  ValueCategory EmitIntToBoolConversion(mlir::Location Loc, ValueCategory Src);
  ValueCategory EmitFloatToBoolConversion(mlir::Location Loc,
                                          ValueCategory Src);
  ValueCategory EmitConversionToBool(mlir::Location Loc, ValueCategory Src,
                                     clang::QualType SrcType);

public:
  MLIRScanner(MLIRASTConsumer &Glob, mlir::OwningOpRef<mlir::ModuleOp> &Module,
              LowerToInfo &LTInfo);

  void init(mlir::FunctionOpInterface Function, const FunctionToEmit &FTE);

  void setEntryAndAllocBlock(mlir::Block *B) {
    allocationScope = entryBlock = B;
    builder.setInsertionPointToStart(B);
  }

  mlir::OpBuilder &getBuilder() { return builder; };
  std::vector<LoopContext> &getLoops() { return loops; }
  mlir::Location getLoc() const { return loc; }

  mlir::Value getConstantIndex(int X);

  ValueCategory VisitDeclStmt(clang::DeclStmt *Decl);

  ValueCategory VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *Decl);

  ValueCategory VisitConstantExpr(clang::ConstantExpr *Expr);

  ValueCategory VisitAtomicExpr(clang::AtomicExpr *Expr);

  ValueCategory VisitTypeTraitExpr(clang::TypeTraitExpr *Expr);

  ValueCategory VisitGNUNullExpr(clang::GNUNullExpr *Expr);

  ValueCategory VisitIntegerLiteral(clang::IntegerLiteral *Expr);

  ValueCategory VisitCharacterLiteral(clang::CharacterLiteral *Expr);

  ValueCategory VisitFloatingLiteral(clang::FloatingLiteral *Expr);

  ValueCategory VisitImaginaryLiteral(clang::ImaginaryLiteral *Expr);

  ValueCategory VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *Expr);

  ValueCategory VisitCXXTypeidExpr(clang::CXXTypeidExpr *Expr);

  ValueCategory VisitCXXTryStmt(clang::CXXTryStmt *Stmt);

  ValueCategory VisitStringLiteral(clang::StringLiteral *Expr);

  ValueCategory VisitParenExpr(clang::ParenExpr *Expr);

  ValueCategory VisitVarDecl(clang::VarDecl *Decl);

  ValueCategory VisitForStmt(clang::ForStmt *Fors);

  ValueCategory VisitCXXForRangeStmt(clang::CXXForRangeStmt *Fors);

  ValueCategory VisitOMPSingleDirective(clang::OMPSingleDirective *);

  ValueCategory VisitOMPForDirective(clang::OMPForDirective *);

  ValueCategory VisitOMPParallelDirective(clang::OMPParallelDirective *);

  ValueCategory VisitExtVectorElementExpr(clang::ExtVectorElementExpr *);

  ValueCategory
  VisitOMPParallelForDirective(clang::OMPParallelForDirective *Fors);

  ValueCategory VisitWhileStmt(clang::WhileStmt *Fors);

  ValueCategory VisitDoStmt(clang::DoStmt *Fors);

  ValueCategory VisitArraySubscriptExpr(clang::ArraySubscriptExpr *Expr);

  ValueCategory VisitCallExpr(clang::CallExpr *Expr);

  ValueCategory
  callHelper(mlir::func::FuncOp ToCall, clang::QualType ObjType,
             clang::ArrayRef<std::pair<ValueCategory, clang::Expr *>> Arguments,
             clang::QualType RetType, bool RetReference, clang::Expr *Expr,
             const clang::FunctionDecl &Callee);

  std::pair<ValueCategory, bool>
  emitClangBuiltinCallExpr(clang::CallExpr *Expr);

  std::pair<ValueCategory, bool> emitGPUCallExpr(clang::CallExpr *Expr);

  std::pair<ValueCategory, bool> emitBuiltinOps(clang::CallExpr *Expr);

  mlir::Operation *emitSYCLOps(const clang::Expr *Expr,
                               const llvm::SmallVectorImpl<mlir::Value> &Args);

  ValueCategory
  VisitCXXScalarValueInitExpr(clang::CXXScalarValueInitExpr *Expr);

  ValueCategory
  VisitCXXPseudoDestructorExpr(clang::CXXPseudoDestructorExpr *Expr);

  ValueCategory VisitCXXConstructExpr(clang::CXXConstructExpr *Expr);

  ValueCategory VisitConstructCommon(clang::CXXConstructExpr *Expr,
                                     clang::VarDecl *Name, unsigned Space,
                                     mlir::Value Mem = nullptr,
                                     mlir::Value Count = nullptr);

  ValueCategory VisitMSPropertyRefExpr(clang::MSPropertyRefExpr *Expr);

  ValueCategory VisitPseudoObjectExpr(clang::PseudoObjectExpr *Expr);

  ValueCategory VisitUnaryOperator(clang::UnaryOperator *U);

  ValueCategory
  VisitSubstNonTypeTemplateParmExpr(clang::SubstNonTypeTemplateParmExpr *Expr);

  ValueCategory
  VisitUnaryExprOrTypeTraitExpr(clang::UnaryExprOrTypeTraitExpr *Uop);

  ValueCategory VisitBinaryOperator(clang::BinaryOperator *BO);
  ValueCategory VisitBinAssign(clang::BinaryOperator *BO);

  ValueCategory EmitPromoted(clang::Expr *E, clang::QualType PromotionType);
  ValueCategory EmitPromotedScalarExpr(clang::Expr *E,
                                       clang::QualType PromotionType);
  ValueCategory EmitPromotedValue(mlir::Location Loc, ValueCategory Result,
                                  clang::QualType PromotionType);
  ValueCategory EmitUnPromotedValue(mlir::Location Loc, ValueCategory Result,
                                    clang::QualType PromotionType);

  ValueCategory EmitCompoundAssignmentLValue(clang::CompoundAssignOperator *E);

  ValueCategory EmitLValue(clang::Expr *E);
  std::pair<ValueCategory, ValueCategory>
  EmitCompoundAssignLValue(clang::CompoundAssignOperator *E,
                           ValueCategory (MLIRScanner::*F)(const BinOpInfo &));

  ValueCategory
  EmitCompoundAssign(clang::CompoundAssignOperator *E,
                     ValueCategory (MLIRScanner::*F)(const BinOpInfo &));

  ValueCategory EmitCheckedInBoundsGEP(mlir::Type ElemTy, ValueCategory Pointer,
                                       mlir::ValueRange IdxList, bool IsSigned,
                                       bool IsSubtraction);
  ValueCategory EmitPointerArithmetic(const BinOpInfo &Info);

  BinOpInfo EmitBinOps(clang::BinaryOperator *E,
                       clang::QualType PromotionTy = clang::QualType());
#define HANDLEBINOP(OP)                                                        \
  ValueCategory EmitBin##OP(const BinOpInfo &E);                               \
  ValueCategory VisitBin##OP(clang::BinaryOperator *E);                        \
  ValueCategory VisitBin##OP##Assign(clang::BinaryOperator *E);
#include "Expressions.def"
#undef HANDLEBINOP

  ValueCategory VisitCXXNoexceptExpr(clang::CXXNoexceptExpr *AS);

  ValueCategory VisitAttributedStmt(clang::AttributedStmt *AS);

  ValueCategory VisitExprWithCleanups(clang::ExprWithCleanups *E);

  ValueCategory VisitDeclRefExpr(clang::DeclRefExpr *E);

  ValueCategory VisitOpaqueValueExpr(clang::OpaqueValueExpr *E);

  ValueCategory VisitMemberExpr(clang::MemberExpr *ME);

  ValueCategory VisitCastExpr(clang::CastExpr *E);

  mlir::Value
  GetAddressOfBaseClass(mlir::Value Obj,
                        const clang::CXXRecordDecl *DerivedClass,
                        clang::ArrayRef<const clang::Type *> BaseTypes,
                        clang::ArrayRef<bool> BaseVirtuals);

  mlir::Value
  GetAddressOfDerivedClass(mlir::Value Obj,
                           const clang::CXXRecordDecl *DerivedClass,
                           clang::CastExpr::path_const_iterator Start,
                           clang::CastExpr::path_const_iterator End);

  ValueCategory VisitIfStmt(clang::IfStmt *Stmt);

  ValueCategory VisitSwitchStmt(clang::SwitchStmt *Stmt);

  ValueCategory VisitConditionalOperator(clang::ConditionalOperator *E);

  ValueCategory VisitCompoundStmt(clang::CompoundStmt *Stmt);

  ValueCategory VisitBreakStmt(clang::BreakStmt *Stmt);

  ValueCategory VisitContinueStmt(clang::ContinueStmt *Stmt);

  ValueCategory VisitReturnStmt(clang::ReturnStmt *Stmt);

  ValueCategory VisitLabelStmt(clang::LabelStmt *Stmt);

  ValueCategory VisitGotoStmt(clang::GotoStmt *Stmt);

  ValueCategory VisitStmtExpr(clang::StmtExpr *Stmt);

  ValueCategory VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr *Expr);

  ValueCategory
  VisitMaterializeTemporaryExpr(clang::MaterializeTemporaryExpr *Expr);

  ValueCategory VisitCXXNewExpr(clang::CXXNewExpr *Expr);

  ValueCategory VisitCXXDeleteExpr(clang::CXXDeleteExpr *Expr);

  ValueCategory VisitCXXDefaultInitExpr(clang::CXXDefaultInitExpr *Expr);

  ValueCategory VisitCXXThisExpr(clang::CXXThisExpr *Expr);

  ValueCategory VisitPredefinedExpr(clang::PredefinedExpr *Expr);

  ValueCategory VisitLambdaExpr(clang::LambdaExpr *Expr);

  ValueCategory VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *Expr);

  ValueCategory VisitCXXFunctionalCastExpr(clang::CXXFunctionalCastExpr *Expr);

  mlir::Attribute InitializeValueByInitListExpr(mlir::Value ToInit,
                                                clang::Expr *Expr);

  ValueCategory VisitInitListExpr(clang::InitListExpr *Expr);

  ValueCategory
  VisitCXXStdInitializerListExpr(clang::CXXStdInitializerListExpr *Expr);

  ValueCategory VisitArrayInitLoop(clang::ArrayInitLoopExpr *Expr,
                                   ValueCategory Tostore);

  ValueCategory VisitArrayInitIndexExpr(clang::ArrayInitIndexExpr *Expr);

  ValueCategory CommonFieldLookup(clang::QualType OT,
                                  const clang::FieldDecl *FD, mlir::Value Val,
                                  bool IsLValue);

  ValueCategory CommonArrayLookup(ValueCategory Val, mlir::Value Idx,
                                  bool IsImplicitRefResult,
                                  bool RemoveIndex = true);

  ValueCategory CommonArrayToPointer(ValueCategory Val);

  static std::string getMangledFuncName(const clang::FunctionDecl &FD,
                                        clang::CodeGen::CodeGenModule &CGM);
};

#endif /* CLANG_MLIR_H */
