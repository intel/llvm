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
  explicit FunctionToEmit(const clang::FunctionDecl &funcDecl)
      : funcDecl(funcDecl),
        funcContext((funcDecl.hasAttr<clang::SYCLKernelAttr>() ||
                     funcDecl.hasAttr<clang::SYCLDeviceAttr>())
                        ? FunctionContext::SYCLDevice
                        : FunctionContext::Host) {}

  /// Note: set the context requested, ensuring a host context is not requested
  /// for SYCL kernel/functions.
  FunctionToEmit(const clang::FunctionDecl &funcDecl,
                 FunctionContext funcContext)
      : funcDecl(funcDecl), funcContext(funcContext) {
    bool isSYCLFunc = funcDecl.hasAttr<clang::SYCLKernelAttr>() ||
                      funcDecl.hasAttr<clang::SYCLDeviceAttr>();
    (void)isSYCLFunc;

    assert((funcContext == FunctionContext::SYCLDevice ||
            (funcContext == FunctionContext::Host && !isSYCLFunc)) &&
           "SYCL kernel/device functions should not have host context");
  }

  const clang::FunctionDecl &getDecl() const { return funcDecl; }
  FunctionContext getContext() const { return funcContext; }

private:
  const clang::FunctionDecl &funcDecl;
  const FunctionContext funcContext;
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
      std::set<std::string> &emitIfFound,
      std::set<std::pair<FunctionContext, std::string>> &done,
      std::map<std::string, mlir::LLVM::GlobalOp> &llvmStringGlobals,
      std::map<std::string, std::pair<mlir::memref::GlobalOp, bool>> &globals,
      std::map<std::string, mlir::func::FuncOp> &functions,
      std::map<std::string, mlir::FunctionOpInterface> &deviceFunctions,
      std::map<std::string, mlir::LLVM::GlobalOp> &llvmGlobals,
      std::map<std::string, mlir::LLVM::LLVMFuncOp> &llvmFunctions,
      clang::Preprocessor &PP, clang::ASTContext &astContext,
      mlir::OwningOpRef<mlir::ModuleOp> &module, clang::SourceManager &SM,
      clang::CodeGenOptions &codegenops, std::string moduleId)
      : emitIfFound(emitIfFound), done(done),
        llvmStringGlobals(llvmStringGlobals), globals(globals),
        functions(functions), deviceFunctions(deviceFunctions),
        llvmGlobals(llvmGlobals), llvmFunctions(llvmFunctions),
        functionsToEmit(), module(module), SM(SM), lcontext(),
        llvmMod(moduleId, lcontext),
        CGM(astContext, &SM.getFileManager().getVirtualFileSystem(),
            PP.getHeaderSearchInfo().getHeaderSearchOpts(),
            PP.getPreprocessorOpts(), codegenops, llvmMod, PP.getDiagnostics()),
        CGTypes(CGM, module), error(false), scopLocList(), LTInfo() {
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
                                                    bool getDeviceStub = false);
  mlir::LLVM::LLVMFuncOp GetOrCreateLLVMFunction(const clang::FunctionDecl *FD);
  mlir::LLVM::LLVMFuncOp GetOrCreateMallocFunction();
  mlir::LLVM::LLVMFuncOp GetOrCreateFreeFunction();

  mlir::LLVM::GlobalOp GetOrCreateLLVMGlobal(const clang::ValueDecl *VD,
                                             std::string prefix = "");

  /// Return a value representing an access into a global string with the
  /// given name, creating the string if necessary.
  mlir::Value GetOrCreateGlobalLLVMString(mlir::Location loc,
                                          mlir::OpBuilder &builder,
                                          clang::StringRef value);

  std::pair<mlir::memref::GlobalOp, bool>
  GetOrCreateGlobal(const clang::ValueDecl *VD, std::string prefix,
                    bool tryInit = true,
                    FunctionContext funcContext = FunctionContext::Host);

  const clang::CodeGen::CGFunctionInfo &
  GetOrCreateCGFunctionInfo(const clang::FunctionDecl *FD);

  void run();

  void HandleTranslationUnit(clang::ASTContext &Context) override;

  bool HandleTopLevelDecl(clang::DeclGroupRef dg) override;

  void HandleDeclContext(clang::DeclContext *DC);

  mlir::Location getMLIRLocation(clang::SourceLocation loc);

private:
  /// Returns the LLVM linkage type of the given function declaration \p FD.
  llvm::GlobalValue::LinkageTypes
  getLLVMLinkageType(const clang::FunctionDecl &FD, bool shouldEmit);

  /// Returns the MLIR LLVM dialect linkage corresponding to \p LV.
  static mlir::LLVM::Linkage getMLIRLinkage(llvm::GlobalValue::LinkageTypes LV);

  /// Returns the MLIR Function type given clang's CGFunctionInfo \p FI.
  mlir::FunctionType getFunctionType(const clang::CodeGen::CGFunctionInfo &FI,
                                     const clang::FunctionDecl &FD);

  /// Returns the MLIR function corresponding to \p mangledName.
  llvm::Optional<mlir::FunctionOpInterface>
  getMLIRFunction(const std::string &mangledName,
                  FunctionContext context) const;

  /// Create the MLIR function corresponding to the given \p FTE.
  /// The MLIR function is created in either the device module (GPUModuleOp) or
  /// in the host module (ModuleOp), depending on the calling context embedded
  /// in the FTE).
  mlir::FunctionOpInterface createMLIRFunction(const FunctionToEmit &FTE,
                                               std::string mangledName,
                                               bool ShouldEmit);

  /// Set the symbol visibility on the given \p function.
  void setMLIRFunctionVisibility(mlir::FunctionOpInterface function,
                                 const FunctionToEmit &FTE, bool shouldEmit);

  /// Set the MLIR function attributes for the given \p function.
  void setMLIRFunctionAttributes(mlir::FunctionOpInterface function,
                                 const FunctionToEmit &FTE, bool ShouldEmit);

  void setMLIRFunctionAttributesForDefinition(
      const clang::Decl *D, mlir::FunctionOpInterface function) const;
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
  llvm::SmallSet<std::string, 4> supportedFuncs;
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

  // Initialize a whitelist of SYCL functions to emit instead just the
  // declaration. Eventually, this list should be removed.
  void initSupportedFunctions();
  bool isSupportedFunctions(std::string name) const {
    return supportedFuncs.contains(name);
  }

  mlir::LLVM::AllocaOp allocateBuffer(size_t i, mlir::LLVM::LLVMPointerType t) {
    auto &vec = bufs[t.getAsOpaquePointer()];
    if (i < vec.size())
      return vec[i];

    mlir::OpBuilder subbuilder(builder.getContext());
    subbuilder.setInsertionPointToStart(allocationScope);

    auto one = subbuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 64);
    auto rs = subbuilder.create<mlir::LLVM::AllocaOp>(loc, t, one, 0);
    vec.push_back(rs);
    return rs;
  }

  mlir::Location getMLIRLocation(clang::SourceLocation loc);

  mlir::Value getTypeSize(clang::QualType t);
  mlir::Value getTypeAlign(clang::QualType t);

  mlir::Value createAllocOp(mlir::Type t, clang::VarDecl *name,
                            uint64_t memspace, bool isArray, bool LLVMABI);

  const clang::FunctionDecl *EmitCallee(const clang::Expr *E);

  mlir::FunctionOpInterface EmitDirectCallee(const clang::FunctionDecl *FD,
                                             FunctionContext Context);

  mlir::Value castToIndex(mlir::Location loc, mlir::Value val);

  /// Converts the \p val to the memory space \p memSpace and returns the
  /// converted value.
  mlir::Value castToMemSpace(mlir::Value val, unsigned memSpace);

  /// Converts the \p val to the memory space of \p t and returns the
  /// converted value.
  mlir::Value castToMemSpaceOfType(mlir::Value val, mlir::Type targetType);

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

  /// Creates an instance of SYCLMethodOpInterface if the SYCLCallOp with base
  /// type name \param typeName, function name \param functionName, mangled
  /// function name \param mangledFunctionName, parameters \param operands and
  /// (optional) return type \param returnType can be represented as such.
  ///
  /// E.g., the SYCLCallOp to the accessor member function
  /// accessor::operator[] can be represented using a SYCLAccessorSubscriptOp.
  llvm::Optional<mlir::sycl::SYCLMethodOpInterface>
  createSYCLMethodOp(llvm::StringRef typeName, llvm::StringRef functionName,
                     mlir::ValueRange operands,
                     llvm::Optional<mlir::Type> returnType,
                     llvm::StringRef mangledFunctionName);

public:
  MLIRScanner(MLIRASTConsumer &Glob, mlir::OwningOpRef<mlir::ModuleOp> &module,
              LowerToInfo &LTInfo);

  void init(mlir::FunctionOpInterface function, const FunctionToEmit &FTE);

  void setEntryAndAllocBlock(mlir::Block *B) {
    allocationScope = entryBlock = B;
    builder.setInsertionPointToStart(B);
  }

  mlir::OpBuilder &getBuilder() { return builder; };
  std::vector<LoopContext> &getLoops() { return loops; }
  mlir::Location getLoc() const { return loc; }

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
  CallHelper(mlir::func::FuncOp tocall, clang::QualType objType,
             clang::ArrayRef<std::pair<ValueCategory, clang::Expr *>> arguments,
             clang::QualType retType, bool retReference, clang::Expr *expr,
             const clang::FunctionDecl *callee);

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
                                     clang::VarDecl *name, unsigned space,
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
  ValueCategory VisitBinAssign(clang::BinaryOperator *BO);
  BinOpInfo EmitBinOps(clang::BinaryOperator *E);
#define HANDLEBINOP(OP)                                                        \
  ValueCategory EmitBin##OP(const BinOpInfo &E);                               \
  ValueCategory VisitBin##OP(clang::BinaryOperator *E);
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
  GetAddressOfBaseClass(mlir::Value obj,
                        const clang::CXXRecordDecl *DerivedClass,
                        clang::ArrayRef<const clang::Type *> BaseTypes,
                        clang::ArrayRef<bool> BaseVirtuals);

  mlir::Value
  GetAddressOfDerivedClass(mlir::Value obj,
                           const clang::CXXRecordDecl *DerivedClass,
                           clang::CastExpr::path_const_iterator Start,
                           clang::CastExpr::path_const_iterator End);

  ValueCategory VisitIfStmt(clang::IfStmt *stmt);

  ValueCategory VisitSwitchStmt(clang::SwitchStmt *stmt);

  ValueCategory VisitConditionalOperator(clang::ConditionalOperator *E);

  ValueCategory VisitCompoundStmt(clang::CompoundStmt *stmt);

  ValueCategory VisitBreakStmt(clang::BreakStmt *stmt);

  ValueCategory VisitContinueStmt(clang::ContinueStmt *stmt);

  ValueCategory VisitReturnStmt(clang::ReturnStmt *stmt);

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

  ValueCategory CommonFieldLookup(clang::QualType OT,
                                  const clang::FieldDecl *FD, mlir::Value val,
                                  bool isLValue);

  ValueCategory CommonArrayLookup(ValueCategory val, mlir::Value idx,
                                  bool isImplicitRefResult,
                                  bool removeIndex = true);

  ValueCategory CommonArrayToPointer(ValueCategory val);

  static std::string getMangledFuncName(const clang::FunctionDecl &FD,
                                        clang::CodeGen::CodeGenModule &CGM);
};

#endif /* CLANG_MLIR_H */
