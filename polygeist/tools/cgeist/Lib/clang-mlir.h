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
#include "mlir/Dialect/Polygeist/IR/PolygeistOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
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
  mlir::Value KeepRunning;
  mlir::Value NoBreak;
};

class BinOpInfo;

/// Context in which a function or a global is located.
enum class InsertionContext {
  Host,      ///< Host context
  SYCLDevice ///< SYCL Device context
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &Out,
                                     const InsertionContext &Context) {
  switch (Context) {
  case InsertionContext::Host:
    Out << "Host";
    break;
  case InsertionContext::SYCLDevice:
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
                        ? InsertionContext::SYCLDevice
                        : InsertionContext::Host) {}

  /// Note: set the context requested, ensuring a host context is not requested
  /// for SYCL kernel/functions.
  FunctionToEmit(const clang::FunctionDecl &FuncDecl,
                 InsertionContext FuncContext)
      : FuncDecl(FuncDecl), FuncContext(FuncContext) {
    bool IsSyclFunc = FuncDecl.hasAttr<clang::SYCLKernelAttr>() ||
                      FuncDecl.hasAttr<clang::SYCLDeviceAttr>();
    (void)IsSyclFunc;

    assert((FuncContext == InsertionContext::SYCLDevice ||
            (FuncContext == InsertionContext::Host && !IsSyclFunc)) &&
           "SYCL kernel/device functions should not have host context");
  }

  const clang::FunctionDecl &getDecl() const { return FuncDecl; }
  InsertionContext getContext() const { return FuncContext; }

private:
  const clang::FunctionDecl &FuncDecl;
  const InsertionContext FuncContext;
};

class MLIRASTConsumer : public clang::ASTConsumer {
private:
  std::set<std::string> &EmitIfFound;
  std::set<std::pair<InsertionContext, std::string>> &Done;
  std::map<std::string, mlir::LLVM::GlobalOp> &LLVMStringGlobals;
  std::map<std::string, std::pair<mlir::memref::GlobalOp, bool>> &Globals;
  std::map<std::string, mlir::func::FuncOp> &Functions;
  std::map<std::string, mlir::FunctionOpInterface> &DeviceFunctions;
  std::map<std::string, mlir::LLVM::GlobalOp> &LLVMGlobals;
  std::map<std::string, mlir::LLVM::LLVMFuncOp> &LLVMFunctions;
  std::deque<FunctionToEmit> FunctionsToEmit;
  mlir::OwningOpRef<mlir::ModuleOp> &Module;
  clang::SourceManager &SM;
  std::unique_ptr<llvm::LLVMContext> Lcontext;
  llvm::Module LLVMMod;
  clang::CodeGen::CodeGenModule CGM;
  mlirclang::CodeGen::CodeGenTypes CGTypes;
  bool Error;
  ScopLocList ScopLocs;
  LowerToInfo LTInfo;
  std::map<const clang::FunctionDecl *, const clang::CodeGen::CGFunctionInfo *>
      CGFunctionInfos;

public:
  MLIRASTConsumer(
      std::set<std::string> &EmitIfFound,
      std::set<std::pair<InsertionContext, std::string>> &Done,
      std::map<std::string, mlir::LLVM::GlobalOp> &LLVMStringGlobals,
      std::map<std::string, std::pair<mlir::memref::GlobalOp, bool>> &Globals,
      std::map<std::string, mlir::func::FuncOp> &Functions,
      std::map<std::string, mlir::FunctionOpInterface> &DeviceFunctions,
      std::map<std::string, mlir::LLVM::GlobalOp> &LLVMGlobals,
      std::map<std::string, mlir::LLVM::LLVMFuncOp> &LLVMFunctions,
      clang::Preprocessor &PP, clang::ASTContext &AstContext,
      mlir::OwningOpRef<mlir::ModuleOp> &Module, clang::SourceManager &SM,
      std::unique_ptr<llvm::LLVMContext> &&LCtx,
      clang::CodeGenOptions &Codegenops, std::string ModuleId)
      : EmitIfFound(EmitIfFound), Done(Done),
        LLVMStringGlobals(LLVMStringGlobals), Globals(Globals),
        Functions(Functions), DeviceFunctions(DeviceFunctions),
        LLVMGlobals(LLVMGlobals), LLVMFunctions(LLVMFunctions),
        FunctionsToEmit(), Module(Module), SM(SM), Lcontext(std::move(LCtx)),
        LLVMMod(ModuleId, *Lcontext),
        CGM(AstContext, &SM.getFileManager().getVirtualFileSystem(),
            PP.getHeaderSearchInfo().getHeaderSearchOpts(),
            PP.getPreprocessorOpts(), Codegenops, LLVMMod, PP.getDiagnostics()),
        CGTypes(CGM, Module), Error(false), ScopLocs(), LTInfo() {
    addPragmaScopHandlers(PP, ScopLocs);
    addPragmaEndScopHandlers(PP, ScopLocs);
    addPragmaLowerToHandlers(PP, LTInfo);
  }

  clang::CodeGen::CodeGenModule &getCGM() { return CGM; }
  mlirclang::CodeGen::CodeGenTypes &getTypes() { return CGTypes; }
  std::map<std::string, mlir::func::FuncOp> &getFunctions() {
    return Functions;
  }
  ::ScopLocList &getScopLocList() { return ScopLocs; }

  mlir::FunctionOpInterface getOrCreateMLIRFunction(FunctionToEmit &FTE,
                                                    bool GetDeviceStub = false);
  mlir::LLVM::LLVMFuncOp getOrCreateLLVMFunction(const clang::FunctionDecl *FD,
                                                 InsertionContext FuncContext);

  mlir::LLVM::GlobalOp getOrCreateLLVMGlobal(const clang::ValueDecl *VD,
                                             std::string Prefix,
                                             InsertionContext FuncContext);

  /// Return a value representing an access into a global string with the
  /// given name, creating the string if necessary.
  mlir::Value getOrCreateGlobalLLVMString(mlir::Location Loc,
                                          mlir::OpBuilder &Builder,
                                          clang::StringRef Value,
                                          InsertionContext FuncContext);

  /// Create global variable and initialize it.
  std::pair<mlir::memref::GlobalOp, bool>
  getOrCreateGlobal(const clang::ValueDecl &VD, std::string Prefix,
                    InsertionContext FuncContext);

  const clang::CodeGen::CGFunctionInfo &
  getOrCreateCGFunctionInfo(const clang::FunctionDecl *FD);

  void run();

  void HandleTranslationUnit(clang::ASTContext &Context) override;

  bool HandleTopLevelDecl(clang::DeclGroupRef DG) override;

  void HandleDeclContext(clang::DeclContext *DC);

  mlir::Location getMLIRLocation(clang::SourceLocation Loc);

private:
  /// Returns the LLVM linkage type of the given function declaration \p FD.
  llvm::GlobalValue::LinkageTypes
  getLLVMLinkageType(const clang::FunctionDecl &FD);

  /// Returns the MLIR LLVM dialect linkage corresponding to \p LV.
  static mlir::LLVM::Linkage getMLIRLinkage(llvm::GlobalValue::LinkageTypes LV);

  /// Returns the MLIR Function type given clang's CGFunctionInfo \p FI.
  mlir::FunctionType getFunctionType(const clang::CodeGen::CGFunctionInfo &FI,
                                     const clang::FunctionDecl &FD);

  /// Returns the MLIR function corresponding to \p mangledName.
  llvm::Optional<mlir::FunctionOpInterface>
  getMLIRFunction(const std::string &MangledName,
                  InsertionContext Context) const;

  /// Create the MLIR function corresponding to the given \p FTE.
  /// The MLIR function is created in either the device module (GPUModuleOp) or
  /// in the host module (ModuleOp), depending on the calling context embedded
  /// in the FTE).
  mlir::FunctionOpInterface createMLIRFunction(const FunctionToEmit &FTE,
                                               std::string MangledName);

  /// Set the symbol visibility on the given \p function.
  void setMLIRFunctionVisibility(mlir::FunctionOpInterface Function,
                                 const FunctionToEmit &FTE);

  /// Set the MLIR function attributes for the given \p function.
  void setMLIRFunctionAttributes(mlir::FunctionOpInterface Function,
                                 const FunctionToEmit &FTE);

  void setMLIRFunctionAttributesForDefinition(
      const clang::Decl *D, mlir::FunctionOpInterface Function) const;
};

class MLIRScanner : public clang::StmtVisitor<MLIRScanner, ValueCategory> {
  friend class IfScope;

private:
  MLIRASTConsumer &Glob;
  mlir::FunctionOpInterface Function;
  InsertionContext FuncContext;
  mlir::OwningOpRef<mlir::ModuleOp> &Module;
  mlir::OpBuilder Builder;
  mlir::Location Loc;
  mlir::Block *EntryBlock;
  std::vector<LoopContext> Loops;
  mlir::Block *AllocationScope;
  std::map<const void *, std::vector<mlir::LLVM::AllocaOp>> Bufs;
  std::map<int, mlir::Value> Constants;
  std::map<clang::LabelStmt *, mlir::Block *> Labels;
  const clang::FunctionDecl *EmittingFunctionDecl;
  std::map<const clang::ValueDecl *, ValueCategory> Params;
  llvm::DenseMap<const clang::ValueDecl *, clang::FieldDecl *> Captures;
  llvm::DenseMap<const clang::ValueDecl *, clang::LambdaCaptureKind>
      CaptureKinds;
  clang::FieldDecl *ThisCapture;
  std::vector<mlir::Value> ArrayInit;
  ValueCategory ThisVal;
  mlir::Value ReturnVal;
  LowerToInfo &LTInfo;

  // Get the \p FNum field of MemRef Value \p V of element type T. \p Shape is
  // the shape of the result MemRef.
  template <typename T>
  mlir::Value SYCLCommonFieldLookup(mlir::Value V, size_t FNum,
                                    llvm::ArrayRef<int64_t> Shape);

  mlir::LLVM::AllocaOp allocateBuffer(size_t I, mlir::LLVM::LLVMPointerType T,
                                      mlir::Type ElemTy) {
    auto &Vec = Bufs[T.getAsOpaquePointer()];
    if (I < Vec.size())
      return Vec[I];

    mlir::OpBuilder Subbuilder(Builder.getContext());
    Subbuilder.setInsertionPointToStart(AllocationScope);

    auto One = Subbuilder.create<mlir::arith::ConstantIntOp>(Loc, 1, 64);
    auto Rs = Subbuilder.create<mlir::LLVM::AllocaOp>(Loc, T, ElemTy, One, 0);
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
                                             InsertionContext Context);

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

  /// Creates an instance of SYCLMethodOpInterface if the SYCLCallOp with
  /// function name \param functionName, parameters \param operands and
  /// (optional) return type \param returnType can be represented as such.
  ///
  /// E.g., the SYCLCallOp to the accessor member function
  /// accessor::operator[] can be represented using a SYCLAccessorSubscriptOp.
  llvm::Optional<mlir::sycl::SYCLMethodOpInterface>
  createSYCLMethodOp(llvm::StringRef FunctionName, mlir::ValueRange Operands,
                     llvm::Optional<mlir::Type> ReturnType);

  /// Attempts to map a call to \param FunctionName to a math operation in the
  /// `sycl` dialect.
  mlir::Operation *createSYCLMathOp(llvm::StringRef FunctionName,
                                    mlir::ValueRange Operands,
                                    mlir::Type ReturnType);

  /// Creates an instance of a SYCL grid operation replacing a call to \param
  /// Callee.
  mlir::Operation *createSYCLBuiltinOp(const clang::FunctionDecl *Callee,
                                       mlir::Location Loc);

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
  ValueCategory EmitPointerToIntegralConversion(mlir::Location Loc,
                                                mlir::Type DestTy,
                                                ValueCategory Src);
  ValueCategory EmitIntegralToPointerConversion(mlir::Location Loc,
                                                mlir::Type DestTy,
                                                mlir::Type ElemTy,
                                                ValueCategory Src);
  ValueCategory EmitVectorInitList(clang::InitListExpr *Expr,
                                   mlir::VectorType VType);
  ValueCategory EmitVectorSubscript(clang::ArraySubscriptExpr *Expr);
  ValueCategory EmitArraySubscriptExpr(clang::ArraySubscriptExpr *E);
  ValueCategory EmitConditionalOperator(clang::ConditionalOperator *E,
                                        bool isReference);
  ValueCategory EmitConditionalOperatorLValue(clang::ConditionalOperator *E);
  ValueCategory EvaluateExprAsBool(clang::Expr *E);

public:
  MLIRScanner(MLIRASTConsumer &Glob, mlir::OwningOpRef<mlir::ModuleOp> &Module,
              LowerToInfo &LTInfo, InsertionContext FuncContext);

  void init(mlir::FunctionOpInterface Function, const FunctionToEmit &FTE);

  void setEntryAndAllocBlock(mlir::Block *B);

  mlir::OpBuilder &getBuilder() { return Builder; };
  std::vector<LoopContext> &getLoops() { return Loops; }
  mlir::Location getLoc() const { return Loc; }

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

  /// Emit a call to the inherited constructor \p InheritedCtor. \p BasePtr is a
  /// pointer to the base member and \p Args, the list of arguments to the
  /// original constructor.
  void
  emitCallToInheritedCtor(const clang::CXXInheritedCtorInitExpr *InheritedCtor,
                          mlir::Value BasePtr,
                          mlir::FunctionOpInterface::BlockArgListType Args);

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
  ValueCategory EmitBinaryOperatorLValue(clang::BinaryOperator *E);

  ValueCategory
  EmitCompoundAssign(clang::CompoundAssignOperator *E,
                     ValueCategory (MLIRScanner::*F)(const BinOpInfo &));

  ValueCategory EmitCheckedInBoundsPtrOffsetOp(mlir::Type ElemTy,
                                               ValueCategory Pointer,
                                               mlir::ValueRange IdxList,
                                               bool IsSigned,
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

#define VISITCOMP(CODE, UI, SI, FP)                                            \
  ValueCategory VisitBin##CODE(clang::BinaryOperator *E) {                     \
    return EmitCompare(E, mlir::arith::CmpIPredicate::UI,                      \
                       mlir::arith::CmpIPredicate::SI,                         \
                       mlir::arith::CmpFPredicate::FP);                        \
  }
#include "CmpExpressions.def"
#undef VISITCOMP

  ValueCategory EmitCompare(clang::BinaryOperator *E,
                            mlir::arith::CmpIPredicate UICmp,
                            mlir::arith::CmpIPredicate SICmp,
                            mlir::arith::CmpFPredicate FCmp);

#define HANDLEUNARYOP(OP)                                                      \
  ValueCategory VisitUnary##OP(clang::UnaryOperator *E,                        \
                               clang::QualType PromotionTy =                   \
                                   clang::QualType());                         \
  ValueCategory Visit##OP(clang::UnaryOperator *E, clang::QualType PromotionTy);
#include "Expressions.def"
#undef HANDLEUNARYOP

  ValueCategory ConstrainShiftValue(ValueCategory LHS, ValueCategory RHS);

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
                                                mlir::Type ElemTy,
                                                clang::Expr *Expr);

  ValueCategory VisitInitListExpr(clang::InitListExpr *Expr);

  ValueCategory
  VisitCXXStdInitializerListExpr(clang::CXXStdInitializerListExpr *Expr);

  ValueCategory VisitArrayInitLoop(clang::ArrayInitLoopExpr *Expr,
                                   ValueCategory Tostore);

  ValueCategory VisitArrayInitIndexExpr(clang::ArrayInitIndexExpr *Expr);

  ValueCategory VisitSizeOfPackExpr(clang::SizeOfPackExpr *E);

  ValueCategory CommonFieldLookup(clang::QualType OT,
                                  const clang::FieldDecl *FD, mlir::Value Val,
                                  mlir::Type ElementType, bool IsLValue,
                                  mlir::Type BaseType = nullptr);

  ValueCategory CommonArrayLookup(ValueCategory Val, mlir::Value Idx,
                                  bool IsImplicitRefResult,
                                  bool RemoveIndex = true);

  ValueCategory CommonArrayToPointer(ValueCategory Val);

  static std::string getMangledFuncName(const clang::FunctionDecl &FD,
                                        clang::CodeGen::CodeGenModule &CGM);
};

#endif /* CLANG_MLIR_H */
