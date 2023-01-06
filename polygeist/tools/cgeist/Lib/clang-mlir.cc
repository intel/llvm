//===- clang-mlir.cc - Emit MLIR IRs by walking clang AST--------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-mlir.h"
#include "Attributes.h"
#include "CodeGenTypes.h"
#include "Options.h"
#include "TypeUtils.h"
#include "utils.h"

#include "mlir/Conversion/SYCLToLLVM/SYCLFuncRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/TypeFromLLVM.h"

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/Version.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "cgeist"

using namespace mlir;

llvm::cl::opt<std::string>
    PrefixABI("prefix-abi", llvm::cl::init(""),
              llvm::cl::desc("Prefix for emitted symbols"));

llvm::cl::opt<bool>
    GenerateAllSYCLFuncs("gen-all-sycl-funcs", llvm::cl::init(false),
                         llvm::cl::desc("Generate all SYCL functions"));

constexpr llvm::StringLiteral MLIRASTConsumer::DeviceModuleName;

/******************************************************************************/
/*                               MLIRScanner                                  */
/******************************************************************************/

MLIRScanner::MLIRScanner(MLIRASTConsumer &Glob, OwningOpRef<ModuleOp> &Module,
                         LowerToInfo &LTInfo)
    : Glob(Glob), Function(), Module(Module), Builder(Module->getContext()),
      Loc(Builder.getUnknownLoc()), EntryBlock(nullptr), Loops(),
      AllocationScope(nullptr), UnsupportedFuncs(), Bufs(), Constants(),
      Labels(), EmittingFunctionDecl(nullptr), Params(), Captures(),
      CaptureKinds(), ThisCapture(nullptr), ArrayInit(), ThisVal(), ReturnVal(),
      LTInfo(LTInfo) {}

void MLIRScanner::initUnsupportedFunctions() {}

static void checkFunctionParent(const FunctionOpInterface F,
                                FunctionContext Context,
                                const OwningOpRef<ModuleOp> &Module) {
  assert(
      (Context != FunctionContext::Host || F->getParentOp() == Module.get()) &&
      "New function must be inserted into global module");
  assert((Context != FunctionContext::SYCLDevice ||
          F->getParentOfType<gpu::GPUModuleOp>() ==
              mlirclang::getDeviceModule(*Module)) &&
         "New device function must be inserted into device module");
}

/// If \p FD corresponds to a function implementing a SYCL method, registers
/// \p Func as the function implementing the corrsponding operation.
static void tryToRegisterSYCLMethod(const clang::FunctionDecl &FD,
                                    FunctionOpInterface Func) {
  if (mlirclang::getNamespaceKind(FD.getEnclosingNamespaceContext()) ==
          mlirclang::NamespaceKind::Other ||
      !mlirclang::areSYCLMemberFunctionOrConstructorArgs(
          Func.getArgumentTypes()) ||
      !isa<clang::CXXMethodDecl>(FD) ||
      isa<clang::CXXConstructorDecl, clang::CXXDestructorDecl>(FD) ||
      !cast<clang::CXXMethodDecl>(FD).isInstance()) {
    // Only member functions in the sycl namespace need to be registered.
    return;
  }
  const auto FunctionTy = Func.getFunctionType().cast<FunctionType>();
  const auto ThisArg = FunctionTy.getInput(0).dyn_cast<MemRefType>();
  assert(ThisArg &&
         "The first argument is expected to be a reference to `this`");
  auto *SYCLDialect = Func.getContext()->getLoadedDialect<sycl::SYCLDialect>();
  assert(SYCLDialect && "SYCL dialect not loaded");
  const std::string MethodName = FD.getAsFunction()->getNameAsString();
  if (!SYCLDialect->findMethodFromBaseClass(
          ThisArg.getElementType().getTypeID(), MethodName)) {
    // Only add a definition for functions registered as SYCL methods.
    return;
  }
  SYCLDialect->registerMethodDefinition(MethodName, cast<func::FuncOp>(Func));
}

void MLIRScanner::init(FunctionOpInterface Func, const FunctionToEmit &FTE) {
  const clang::FunctionDecl *FD = &FTE.getDecl();

  Function = Func;
  EmittingFunctionDecl = FD;

  if (ShowAST)
    llvm::dbgs() << "Emitting fn: " << Function.getName() << "\n"
                 << "\tfunctionDecl:" << *FD << "\n"
                 << "function:" << Function << "\n";

  initUnsupportedFunctions();
  // This is needed, as GPUFuncOps are already created with an entry Block.
  setEntryAndAllocBlock(isa<gpu::GPUFuncOp>(Function)
                            ? &Function.getBlocks().front()
                            : Function.addEntryBlock());

  unsigned I = 0;
  if (const auto *CM = dyn_cast<clang::CXXMethodDecl>(FD)) {
    if (CM->getParent()->isLambda()) {
      for (auto C : CM->getParent()->captures()) {
        if (C.capturesVariable())
          CaptureKinds[C.getCapturedVar()] = C.getCaptureKind();
      }

      CM->getParent()->getCaptureFields(Captures, ThisCapture);
      if (ThisCapture) {
        llvm::errs() << " ThisCapture:\n";
        ThisCapture->dump();
      }
    }

    if (CM->isInstance()) {
      Value Val = Function.getArgument(I);
      ThisVal = ValueCategory(Val, /*isReference*/ false);
      I++;
    }
  }

  const clang::CodeGen::CGFunctionInfo &FI = Glob.getOrCreateCGFunctionInfo(FD);
  auto FIArgs = FI.arguments();

  for (clang::ParmVarDecl *Parm : FD->parameters()) {
    assert(I != Function.getNumArguments());

    clang::QualType ParmType = Parm->getType();

    bool LLVMABI = false, IsArray = false;
    if (Glob.getTypes()
            .getMLIRType(Glob.getCGM().getContext().getPointerType(ParmType))
            .isa<LLVM::LLVMPointerType>())
      LLVMABI = true;
    else
      Glob.getTypes().getMLIRType(ParmType, &IsArray);

    bool IsReference = IsArray || isa<clang::ReferenceType>(
                                      ParmType->getUnqualifiedDesugaredType());
    IsReference |=
        (FIArgs[I].info.getKind() == clang::CodeGen::ABIArgInfo::Indirect ||
         FIArgs[I].info.getKind() ==
             clang::CodeGen::ABIArgInfo::IndirectAliased);

    Value Val = Function.getArgument(I);
    assert(Val && "Expecting a valid value");

    if (IsReference)
      Params.emplace(Parm, ValueCategory(Val, /*isReference*/ true));
    else {
      Value Alloc =
          createAllocOp(Val.getType(), Parm, /*MemSpace*/ 0, IsArray, LLVMABI);
      ValueCategory(Alloc, /*isReference*/ true).store(Builder, Val);
    }

    I++;
  }

  if (FD->hasAttr<clang::CUDAGlobalAttr>() &&
      Glob.getCGM().getLangOpts().CUDA &&
      !Glob.getCGM().getLangOpts().CUDAIsDevice) {
    FunctionToEmit FTE(*FD);
    auto DeviceStub = cast<func::FuncOp>(
        Glob.getOrCreateMLIRFunction(FTE, true /* ShouldEmit*/,
                                     /* getDeviceStub */ true));

    Builder.create<func::CallOp>(Loc, DeviceStub, Function.getArguments());
    Builder.create<func::ReturnOp>(Loc);
    return;
  }

  if (const auto *CC = dyn_cast<clang::CXXConstructorDecl>(FD)) {
    const clang::CXXRecordDecl *ClassDecl = CC->getParent();
    for (auto *Expr : CC->inits()) {
      if (ShowAST) {
        llvm::errs() << " init: - baseInit:" << (int)Expr->isBaseInitializer()
                     << " memberInit:" << (int)Expr->isMemberInitializer()
                     << " anyMember:" << (int)Expr->isAnyMemberInitializer()
                     << " indirectMember:"
                     << (int)Expr->isIndirectMemberInitializer()
                     << " isinClass:" << (int)Expr->isInClassMemberInitializer()
                     << " delegating:" << (int)Expr->isDelegatingInitializer()
                     << " isPack:" << (int)Expr->isPackExpansion() << "\n";
        if (Expr->getMember())
          Expr->getMember()->dump();
        if (Expr->getInit())
          Expr->getInit()->dump();
      }
      assert(ThisVal.val);

      clang::FieldDecl *Field = Expr->getMember();
      if (!Field) {
        if (Expr->isBaseInitializer()) {
          bool BaseIsVirtual = Expr->isBaseVirtual();
          const auto *BaseType = Expr->getBaseClass();

          // Shift and cast down to the base type.
          // TODO: for complete types, this should be possible with a GEP.
          Value V = ThisVal.val;
          const clang::Type *BaseTypes[] = {BaseType};
          bool BaseVirtual[] = {BaseIsVirtual};

          V = GetAddressOfBaseClass(V, /*derived*/ ClassDecl, BaseTypes,
                                    BaseVirtual);

          clang::Expr *Init = Expr->getInit();
          if (auto *Clean = dyn_cast<clang::ExprWithCleanups>(Init)) {
            CGEIST_WARNING(llvm::WithColor::warning() << "TODO: cleanup\n");
            Init = Clean->getSubExpr();
          }

          VisitConstructCommon(cast<clang::CXXConstructExpr>(Init),
                               /*name*/ nullptr, /*space*/ 0, /*mem*/ V);
          continue;
        }

        if (Expr->isDelegatingInitializer()) {
          clang::Expr *Init = Expr->getInit();
          if (auto *Clean = dyn_cast<clang::ExprWithCleanups>(Init)) {
            CGEIST_WARNING(llvm::WithColor::warning() << "TODO: cleanup\n");
            Init = Clean->getSubExpr();
          }

          VisitConstructCommon(cast<clang::CXXConstructExpr>(Init),
                               /*name*/ nullptr, /*space*/ 0,
                               /*mem*/ ThisVal.val);
          continue;
        }
      }
      assert(Field && "initialiation expression must apply to a field");

      if (auto *AILE = dyn_cast<clang::ArrayInitLoopExpr>(Expr->getInit())) {
        VisitArrayInitLoop(AILE,
                           CommonFieldLookup(CC->getThisObjectType(), Field,
                                             ThisVal.val, /*IsLValue*/ false));
        continue;
      }

      if (auto *Cons = dyn_cast<clang::CXXConstructExpr>(Expr->getInit())) {
        VisitConstructCommon(Cons, /*name*/ nullptr, /*space*/ 0,
                             CommonFieldLookup(CC->getThisObjectType(), Field,
                                               ThisVal.val, /*IsLValue*/ false)
                                 .val);
        continue;
      }

      auto InitExpr = Visit(Expr->getInit());
      if (!InitExpr.val) {
        Expr->getInit()->dump();
        assert(InitExpr.val);
      }

      bool IsArray = false;
      Glob.getTypes().getMLIRType(Expr->getInit()->getType(), &IsArray);

      auto CFL = CommonFieldLookup(CC->getThisObjectType(), Field, ThisVal.val,
                                   /*IsLValue*/ false);
      assert(CFL.val);
      CFL.store(Builder, InitExpr, IsArray);
    }
  }

  CGEIST_WARNING({
    if (auto CC = dyn_cast<clang::CXXDestructorDecl>(FD))
      llvm::WithColor::warning()
          << "destructor not fully handled yet for: " << CC << "\n";
  });

  auto I1Ty = Builder.getIntegerType(1);
  auto Type = MemRefType::get({}, I1Ty, {}, 0);
  auto TrueV = Builder.create<arith::ConstantIntOp>(Loc, true, 1);
  Loops.push_back({Builder.create<memref::AllocaOp>(Loc, Type),
                   Builder.create<memref::AllocaOp>(Loc, Type)});
  Builder.create<memref::StoreOp>(Loc, TrueV, Loops.back().NoBreak);
  Builder.create<memref::StoreOp>(Loc, TrueV, Loops.back().KeepRunning);

  if (Function.getResultTypes().size()) {
    auto Type = MemRefType::get({}, Function.getResultTypes()[0], {}, 0);
    ReturnVal = Builder.create<memref::AllocaOp>(Loc, Type);
    if (Type.getElementType().isa<IntegerType, FloatType>()) {
      Builder.create<memref::StoreOp>(
          Loc,
          ValueCategory::getUndefValue(Builder, Loc, Type.getElementType()).val,
          ReturnVal, std::vector<Value>({}));
    }
  }

  if (const auto *D = dyn_cast<clang::CXXMethodDecl>(FD)) {
    // ClangAST incorrectly does not contain the correct definition
    // of a union move operation and as such we _must_ emit a memcpy
    // for a defaulted union copy or move.
    if (D->getParent()->isUnion() && D->isDefaulted()) {
      Value V = ThisVal.val;
      assert(V);
      if (auto MT = V.getType().dyn_cast<MemRefType>())
        V = Builder.create<polygeist::Pointer2MemrefOp>(
            Loc, LLVM::LLVMPointerType::get(MT.getElementType()), V);

      Value Src = Function.getArgument(1);
      if (auto MT = Src.getType().dyn_cast<MemRefType>())
        Src = Builder.create<polygeist::Pointer2MemrefOp>(
            Loc, LLVM::LLVMPointerType::get(MT.getElementType()), Src);

      Value TypeSize = Builder.create<polygeist::TypeSizeOp>(
          Loc, Builder.getIndexType(),
          TypeAttr::get(
              V.getType().cast<LLVM::LLVMPointerType>().getElementType()));
      TypeSize = Builder.create<arith::IndexCastOp>(Loc, Builder.getI64Type(),
                                                    TypeSize);
      V = Builder.create<LLVM::BitcastOp>(
          Loc,
          LLVM::LLVMPointerType::get(
              Builder.getI8Type(),
              V.getType().cast<LLVM::LLVMPointerType>().getAddressSpace()),
          V);
      Src = Builder.create<LLVM::BitcastOp>(
          Loc,
          LLVM::LLVMPointerType::get(
              Builder.getI8Type(),
              Src.getType().cast<LLVM::LLVMPointerType>().getAddressSpace()),
          Src);
      Value VolatileCpy = Builder.create<arith::ConstantIntOp>(Loc, false, 1);
      Builder.create<LLVM::MemcpyOp>(Loc, V, Src, TypeSize, VolatileCpy);
    }
  }

  clang::Stmt *Stmt = FD->getBody();
  assert(Stmt);
  if (ShowAST)
    Stmt->dump();

  Visit(Stmt);

  if (Function.getResultTypes().size()) {
    assert(!isa<gpu::GPUFuncOp>(Function) &&
           "SYCL kernel functions must always have a void return type.");
    Builder.create<func::ReturnOp>(
        Loc, ArrayRef<Value>({Builder.create<memref::LoadOp>(Loc, ReturnVal)}));
  } else if (isa<gpu::GPUFuncOp>(Function))
    Builder.create<gpu::ReturnOp>(Loc);
  else
    Builder.create<func::ReturnOp>(Loc);

  tryToRegisterSYCLMethod(*FD, Function);

  checkFunctionParent(Function, FTE.getContext(), Module);
}

Value MLIRScanner::createAllocOp(Type T, clang::VarDecl *Name,
                                 uint64_t MemSpace, bool IsArray = false,
                                 bool LLVMABI = false) {
  MemRefType MR;
  Value Alloc = nullptr;
  OpBuilder ABuilder(Builder.getContext());
  ABuilder.setInsertionPointToStart(AllocationScope);
  Location VarLoc = Name ? getMLIRLocation(Name->getBeginLoc()) : Loc;

  if (!IsArray) {
    if (LLVMABI) {
      if (Name)
        if (const auto *Var = dyn_cast<clang::VariableArrayType>(
                Name->getType()->getUnqualifiedDesugaredType())) {
          auto Len = Visit(Var->getSizeExpr()).getValue(Builder);
          Alloc = Builder.create<LLVM::AllocaOp>(
              VarLoc, LLVM::LLVMPointerType::get(T, MemSpace), Len);
          Builder.create<polygeist::TrivialUseOp>(VarLoc, Alloc);
          Alloc = Builder.create<LLVM::BitcastOp>(
              VarLoc,
              LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(T, 0)),
              Alloc);
        }

      if (!Alloc) {
        Alloc = ABuilder.create<LLVM::AllocaOp>(
            VarLoc, LLVM::LLVMPointerType::get(T, MemSpace),
            ABuilder.create<arith::ConstantIntOp>(VarLoc, 1, 64), 0);
        if (T.isa<IntegerType, FloatType>()) {
          ABuilder.create<LLVM::StoreOp>(
              VarLoc, ValueCategory::getUndefValue(ABuilder, VarLoc, T).val,
              Alloc);
        }
      }
    } else {
      MR = MemRefType::get(1, T, {}, MemSpace);
      Alloc = ABuilder.create<memref::AllocaOp>(VarLoc, MR);
      LLVM_DEBUG({
        llvm::dbgs() << "MLIRScanner::createAllocOp: created: ";
        Alloc.dump();
      });

      if (MemSpace != 0) {
        // Note: this code is incorrect because 'alloc' has a MemRefType in
        // memory space that is not zero, therefore is illegal to create a
        // Memref2Pointer operation that yields a result not in the same memory
        // space.
        auto MemRefToPtr = ABuilder.create<polygeist::Memref2PointerOp>(
            VarLoc, LLVM::LLVMPointerType::get(T, 0), Alloc);
        Alloc = ABuilder.create<polygeist::Pointer2MemrefOp>(
            VarLoc, MemRefType::get(ShapedType::kDynamic, T, {}, MemSpace),
            MemRefToPtr);
      }
      Alloc = ABuilder.create<memref::CastOp>(
          VarLoc, MemRefType::get(ShapedType::kDynamic, T, {}, 0), Alloc);
      LLVM_DEBUG({
        llvm::dbgs() << "MLIRScanner::createAllocOp: created: ";
        Alloc.dump();
        llvm::dbgs() << "\n";
      });

      if (T.isa<IntegerType, FloatType, VectorType>())
        ABuilder.create<memref::StoreOp>(
            VarLoc, ValueCategory::getUndefValue(ABuilder, VarLoc, T).val,
            Alloc,
            ArrayRef<Value>({ABuilder.create<arith::ConstantIndexOp>(Loc, 0)}));
    }
  } else {
    auto MT = T.cast<MemRefType>();
    auto Shape = std::vector<int64_t>(MT.getShape());
    auto PShape = Shape[0];

    if (Name)
      if (const auto *Var = dyn_cast<clang::VariableArrayType>(
              Name->getType()->getUnqualifiedDesugaredType())) {
        assert(Shape[0] == ShapedType::kDynamic);
        MR = MemRefType::get(
            Shape, MT.getElementType(), MemRefLayoutAttrInterface(),
            mlirclang::wrapIntegerMemorySpace(MemSpace, MT.getContext()));
        auto Len = Visit(Var->getSizeExpr()).getValue(Builder);
        Len = Builder.create<arith::IndexCastOp>(VarLoc, Builder.getIndexType(),
                                                 Len);
        Alloc = Builder.create<memref::AllocaOp>(VarLoc, MR, Len);
        Builder.create<polygeist::TrivialUseOp>(VarLoc, Alloc);
        if (MemSpace != 0)
          Alloc = ABuilder.create<polygeist::Pointer2MemrefOp>(
              VarLoc, MemRefType::get(Shape, MT.getElementType()),
              ABuilder.create<polygeist::Memref2PointerOp>(
                  VarLoc, LLVM::LLVMPointerType::get(MT.getElementType(), 0),
                  Alloc));
      }

    if (!Alloc) {
      if (PShape == ShapedType::kDynamic)
        Shape[0] = 1;
      MR = MemRefType::get(
          Shape, MT.getElementType(), MemRefLayoutAttrInterface(),
          mlirclang::wrapIntegerMemorySpace(MemSpace, MT.getContext()));
      Alloc = ABuilder.create<memref::AllocaOp>(VarLoc, MR);
      if (MemSpace != 0)
        Alloc = ABuilder.create<polygeist::Pointer2MemrefOp>(
            VarLoc, MemRefType::get(Shape, MT.getElementType()),
            ABuilder.create<polygeist::Memref2PointerOp>(
                VarLoc, LLVM::LLVMPointerType::get(MT.getElementType(), 0),
                Alloc));

      Shape[0] = PShape;
      Alloc = ABuilder.create<memref::CastOp>(
          VarLoc, MemRefType::get(Shape, MT.getElementType()), Alloc);
    }
  }
  assert(Alloc);

  if (Name) {
    if (Params.find(Name) != Params.end())
      Name->dump();
    assert(Params.find(Name) == Params.end());
    Params[Name] = ValueCategory(Alloc, /*isReference*/ true);
  }

  return Alloc;
}

Value add(MLIRScanner &Sc, OpBuilder &Builder, Location Loc, Value LHS,
          Value RHS) {
  assert(LHS);
  assert(RHS);
  if (auto Op = LHS.getDefiningOp<arith::ConstantIntOp>()) {
    if (Op.value() == 0)
      return RHS;
  }

  if (auto Op = LHS.getDefiningOp<arith::ConstantIndexOp>()) {
    if (Op.value() == 0)
      return RHS;
  }

  if (auto Op = RHS.getDefiningOp<arith::ConstantIntOp>()) {
    if (Op.value() == 0)
      return LHS;
  }

  if (auto Op = RHS.getDefiningOp<arith::ConstantIndexOp>()) {
    if (Op.value() == 0)
      return LHS;
  }

  return Builder.create<arith::AddIOp>(Loc, LHS, RHS);
}

Value MLIRScanner::castToIndex(Location Loc, Value Val) {
  assert(Val && "Expect non-null value");

  if (auto Op = Val.getDefiningOp<arith::ConstantIntOp>())
    return getConstantIndex(Op.value());

  return Builder.create<arith::IndexCastOp>(
      Loc, IndexType::get(Val.getContext()), Val);
}

Value MLIRScanner::castToMemSpace(Value Val, unsigned MemSpace) {
  assert(Val && "Expect non-null value");

  return TypeSwitch<Type, Value>(Val.getType())
      .Case<MemRefType>([&](MemRefType ValType) -> Value {
        if (ValType.getMemorySpaceAsInt() == MemSpace)
          return Val;

        Value NewVal = Builder.create<polygeist::Memref2PointerOp>(
            Loc,
            LLVM::LLVMPointerType::get(ValType.getElementType(),
                                       ValType.getMemorySpaceAsInt()),
            Val);
        NewVal = Builder.create<LLVM::AddrSpaceCastOp>(
            Loc, LLVM::LLVMPointerType::get(ValType.getElementType(), MemSpace),
            NewVal);
        return Builder.create<polygeist::Pointer2MemrefOp>(
            Loc,
            MemRefType::get(ValType.getShape(), ValType.getElementType(),
                            MemRefLayoutAttrInterface(),
                            mlirclang::wrapIntegerMemorySpace(
                                MemSpace, ValType.getContext())),
            NewVal);
      })
      .Case<LLVM::LLVMPointerType>([&](LLVM::LLVMPointerType ValType) -> Value {
        if (ValType.getAddressSpace() == MemSpace)
          return Val;

        return Builder.create<LLVM::AddrSpaceCastOp>(
            Loc, LLVM::LLVMPointerType::get(ValType.getElementType(), MemSpace),
            Val);
      });
}

Value MLIRScanner::castToMemSpaceOfType(Value Val, Type T) {
  assert((T.isa<MemRefType>() || T.isa<LLVM::LLVMPointerType>()) &&
         "Unexpected type");
  unsigned MemSpace = T.isa<MemRefType>()
                          ? T.cast<MemRefType>().getMemorySpaceAsInt()
                          : T.cast<LLVM::LLVMPointerType>().getAddressSpace();
  return castToMemSpace(Val, MemSpace);
}

ValueCategory MLIRScanner::CommonArrayToPointer(ValueCategory Scalar) {
  assert(Scalar.val && Scalar.isReference);

  if (auto PT = Scalar.val.getType().dyn_cast<LLVM::LLVMPointerType>()) {
    if (PT.getElementType().isa<LLVM::LLVMPointerType>())
      return ValueCategory(Scalar.val, /*isRef*/ false);

    if (!PT.getElementType().isa<LLVM::LLVMArrayType>()) {
      EmittingFunctionDecl->dump();
      Function.dump();
      llvm::errs() << " sval: " << Scalar.val << "\n";
      llvm::errs() << PT << "\n";
    }

    auto ET = PT.getElementType().cast<LLVM::LLVMArrayType>().getElementType();
    return ValueCategory(
        Builder.create<LLVM::GEPOp>(
            Loc, LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
            Scalar.val,
            ArrayRef<Value>(
                {Builder.create<arith::ConstantIntOp>(Loc, 0, 32),
                 Builder.create<arith::ConstantIntOp>(Loc, 0, 32)})),
        /*isReference*/ false);
  }

  auto MT = Scalar.val.getType().cast<MemRefType>();
  auto Shape = std::vector<int64_t>(MT.getShape());
  Shape[0] = ShapedType::kDynamic;
  auto MT0 = MemRefType::get(Shape, MT.getElementType(),
                             MemRefLayoutAttrInterface(), MT.getMemorySpace());

  auto Post = Builder.create<memref::CastOp>(Loc, MT0, Scalar.val);
  return ValueCategory(Post, /*isReference*/ false);
}

ValueCategory MLIRScanner::CommonArrayLookup(ValueCategory Array, Value Idx,
                                             bool IsImplicitRefResult,
                                             bool RemoveIndex) {
  Value Val = Array.getValue(Builder);
  assert(Val);

  if (Val.getType().isa<LLVM::LLVMPointerType>()) {
    // TODO sub
    return ValueCategory(
        Builder.create<LLVM::GEPOp>(
            Loc, Val.getType(), Val,
            ArrayRef<Value>({Builder.create<arith::IndexCastOp>(
                Loc, Builder.getIntegerType(64), Idx)})),
        /*isReference*/ true);
  }

  if (!Val.getType().isa<MemRefType>()) {
    EmittingFunctionDecl->dump();
    Builder.getInsertionBlock()->dump();
    Function.dump();
    llvm::errs() << "value: " << Val << "\n";
  }

  ValueCategory DRef;
  {
    auto MT = Val.getType().cast<MemRefType>();
    auto Shape = std::vector<int64_t>(MT.getShape());
    Shape[0] = ShapedType::kDynamic;
    auto MT0 =
        MemRefType::get(Shape, MT.getElementType(), MemRefLayoutAttrInterface(),
                        MT.getMemorySpace());
    auto Post = Builder.create<polygeist::SubIndexOp>(Loc, MT0, Val, Idx);
    // TODO sub
    DRef = ValueCategory(Post, /*isReference*/ true);
  }
  assert(DRef.isReference);

  if (!RemoveIndex)
    return DRef;

  auto MT = DRef.val.getType().cast<MemRefType>();
  auto Shape = std::vector<int64_t>(MT.getShape());
  if (Shape.size() == 1 || (Shape.size() == 2 && IsImplicitRefResult))
    Shape[0] = ShapedType::kDynamic;
  else
    Shape.erase(Shape.begin());

  auto MT0 = MemRefType::get(Shape, MT.getElementType(),
                             MemRefLayoutAttrInterface(), MT.getMemorySpace());
  auto Post = Builder.create<polygeist::SubIndexOp>(Loc, MT0, DRef.val,
                                                    getConstantIndex(0));
  return ValueCategory(Post, /*isReference*/ true);
}

Value MLIRScanner::getConstantIndex(int X) {
  if (Constants.find(X) != Constants.end())
    return Constants[X];

  OpBuilder SubBuilder(Builder.getContext());
  SubBuilder.setInsertionPointToStart(EntryBlock);
  return Constants[X] = SubBuilder.create<arith::ConstantIndexOp>(Loc, X);
}

ValueCategory MLIRScanner::VisitUnaryOperator(clang::UnaryOperator *U) {
  Location Loc = getMLIRLocation(U->getExprLoc());
  ValueCategory Sub = Visit(U->getSubExpr());

  switch (U->getOpcode()) {
  case clang::UnaryOperator::Opcode::UO_Extension:
    return Sub;

  case clang::UnaryOperator::Opcode::UO_LNot: {
    assert(Sub.val);
    Value Val = Sub.getValue(Builder);

    if (auto MT = Val.getType().dyn_cast<MemRefType>()) {
      Val = Builder.create<polygeist::Memref2PointerOp>(
          Loc,
          LLVM::LLVMPointerType::get(Builder.getI8Type(),
                                     MT.getMemorySpaceAsInt()),
          Val);
    }

    if (auto LT = Val.getType().dyn_cast<LLVM::LLVMPointerType>()) {
      auto NullOp = Builder.create<LLVM::NullOp>(Loc, LT);
      auto NE = Builder.create<LLVM::ICmpOp>(Loc, LLVM::ICmpPredicate::eq, Val,
                                             NullOp);
      return ValueCategory(NE, /*isReference*/ false);
    }

    if (!Val.getType().isa<IntegerType>()) {
      U->dump();
      Val.dump();
    }

    auto Ty = Val.getType().cast<IntegerType>();
    if (Ty.getWidth() != 1)
      Val = Builder.create<arith::CmpIOp>(
          Loc, arith::CmpIPredicate::ne, Val,
          Builder.create<arith::ConstantIntOp>(Loc, 0, Ty));

    auto C1 = Builder.create<arith::ConstantIntOp>(Loc, 1, Val.getType());
    Value Res = Builder.create<arith::XOrIOp>(Loc, Val, C1);

    auto PostTy = Glob.getTypes().getMLIRType(U->getType()).cast<IntegerType>();
    if (PostTy.getWidth() > 1)
      Res = Builder.create<arith::ExtUIOp>(Loc, PostTy, Res);

    return ValueCategory(Res, /*isReference*/ false);
  }

  case clang::UnaryOperator::Opcode::UO_Not: {
    assert(Sub.val);
    Value Val = Sub.getValue(Builder);

    if (!Val.getType().isa<IntegerType>()) {
      U->dump();
      Val.dump();
    }

    auto Ty = Val.getType().cast<IntegerType>();
    auto C1 = Builder.create<arith::ConstantIntOp>(
        Loc, APInt::getAllOnesValue(Ty.getWidth()).getSExtValue(), Ty);
    return ValueCategory(Builder.create<arith::XOrIOp>(Loc, Val, C1),
                         /*isReference*/ false);
  }

  case clang::UnaryOperator::Opcode::UO_Deref:
    return Sub.dereference(Builder);

  case clang::UnaryOperator::Opcode::UO_AddrOf: {
    assert(Sub.isReference);
    if (Sub.val.getType().isa<LLVM::LLVMPointerType>())
      return ValueCategory(Sub.val, /*isReference*/ false);

    bool IsArray = false;
    Glob.getTypes().getMLIRType(U->getSubExpr()->getType(), &IsArray);
    auto MT = Sub.val.getType().cast<MemRefType>();
    auto Shape = std::vector<int64_t>(MT.getShape());
    Shape[0] = ShapedType::kDynamic;
    auto MT0 =
        MemRefType::get(Shape, MT.getElementType(), MemRefLayoutAttrInterface(),
                        MT.getMemorySpace());
    Value Res = Builder.create<memref::CastOp>(Loc, MT0, Sub.val);
    return ValueCategory(Res,
                         /*isReference*/ false);
  }

  case clang::UnaryOperator::Opcode::UO_PreInc:
  case clang::UnaryOperator::Opcode::UO_PostInc: {
    assert(Sub.isReference);
    Value Prev = Sub.getValue(Builder);
    Type Ty = Prev.getType();

    Value Next;
    if (auto FT = Ty.dyn_cast<FloatType>()) {
      if (Prev.getType() != Ty) {
        U->dump();
        llvm::errs() << " ty: " << Ty << "prev: " << Prev << "\n";
      }
      assert(Prev.getType() == Ty);
      Next = Builder.create<arith::AddFOp>(
          Loc, Prev,
          Builder.create<arith::ConstantFloatOp>(
              Loc, APFloat(FT.getFloatSemantics(), "1"), FT));
    } else if (auto MT = Ty.dyn_cast<MemRefType>()) {
      auto Shape = std::vector<int64_t>(MT.getShape());
      Shape[0] = ShapedType::kDynamic;
      auto MT0 =
          MemRefType::get(Shape, MT.getElementType(),
                          MemRefLayoutAttrInterface(), MT.getMemorySpace());
      Next = Builder.create<polygeist::SubIndexOp>(Loc, MT0, Prev,
                                                   getConstantIndex(1));
    } else if (auto PT = Ty.dyn_cast<LLVM::LLVMPointerType>()) {
      auto ITy = IntegerType::get(Builder.getContext(), 64);
      Next = Builder.create<LLVM::GEPOp>(
          Loc, PT, Prev,
          std::vector<Value>(
              {Builder.create<arith::ConstantIntOp>(Loc, 1, ITy)}));
    } else {
      if (!Ty.isa<IntegerType>()) {
        llvm::errs() << Ty << " - " << Prev << "\n";
        U->dump();
      }
      if (Prev.getType() != Ty) {
        U->dump();
        llvm::errs() << " ty: " << Ty << "prev: " << Prev << "\n";
      }
      assert(Prev.getType() == Ty);
      Next = Builder.create<arith::AddIOp>(
          Loc, Prev,
          Builder.create<arith::ConstantIntOp>(Loc, 1, Ty.cast<IntegerType>()));
    }
    Sub.store(Builder, Next);

    if (U->getOpcode() == clang::UnaryOperator::Opcode::UO_PreInc)
      return Sub;
    return ValueCategory(Prev, /*isReference*/ false);
  }

  case clang::UnaryOperator::Opcode::UO_PreDec:
  case clang::UnaryOperator::Opcode::UO_PostDec: {
    auto Ty = Glob.getTypes().getMLIRType(U->getType());
    assert(Sub.isReference);
    auto Prev = Sub.getValue(Builder);

    Value Next;
    if (auto FT = Ty.dyn_cast<FloatType>()) {
      Next = Builder.create<arith::SubFOp>(
          Loc, Prev,
          Builder.create<arith::ConstantFloatOp>(
              Loc, APFloat(FT.getFloatSemantics(), "1"), FT));
    } else if (auto PT = Ty.dyn_cast<LLVM::LLVMPointerType>()) {
      auto ITy = IntegerType::get(Builder.getContext(), 64);
      Next = Builder.create<LLVM::GEPOp>(
          Loc, PT, Prev,
          std::vector<Value>(
              {Builder.create<arith::ConstantIntOp>(Loc, -1, ITy)}));
    } else if (auto MT = Ty.dyn_cast<MemRefType>()) {
      auto Shape = std::vector<int64_t>(MT.getShape());
      Shape[0] = ShapedType::kDynamic;
      auto MT0 =
          MemRefType::get(Shape, MT.getElementType(),
                          MemRefLayoutAttrInterface(), MT.getMemorySpace());
      Next = Builder.create<polygeist::SubIndexOp>(Loc, MT0, Prev,
                                                   getConstantIndex(-1));
    } else {
      if (!Ty.isa<IntegerType>()) {
        llvm::errs() << Ty << " - " << Prev << "\n";
        U->dump();
      }
      Next = Builder.create<arith::SubIOp>(
          Loc, Prev,
          Builder.create<arith::ConstantIntOp>(Loc, 1, Ty.cast<IntegerType>()));
    }
    Sub.store(Builder, Next);
    return ValueCategory(
        (U->getOpcode() == clang::UnaryOperator::Opcode::UO_PostInc) ? Prev
                                                                     : Next,
        /*isReference*/ false);
  }

  default: {
    U->dump();
    llvm_unreachable("unhandled opcode");
  }
  }
}

bool hasAffineArith(Operation *Op, AffineExpr &Expr, Value &AffineForIndVar) {
  // skip IndexCastOp
  if (isa<arith::IndexCastOp>(Op))
    return hasAffineArith(Op->getOperand(0).getDefiningOp(), Expr,
                          AffineForIndVar);

  // induction variable are modelled as memref<1xType>
  // %1 = index_cast %induction : index to i32
  // %2 = alloca() : memref<1xi32>
  // store %1, %2[0] : memref<1xi32>
  // ...
  // %5 = load %2[0] : memref<1xf32>
  if (isa<memref::LoadOp>(Op)) {
    auto Load = cast<memref::LoadOp>(Op);
    Value LoadOperand = Load.getOperand(0);
    if (LoadOperand.getType().cast<MemRefType>().getShape().size() != 1)
      return false;

    Operation *MaybeAllocaOp = LoadOperand.getDefiningOp();
    if (!isa<memref::AllocaOp>(MaybeAllocaOp))
      return false;

    Operation::user_range AllocaUsers = MaybeAllocaOp->getUsers();
    if (llvm::none_of(AllocaUsers, [](Operation *Op) {
          if (isa<memref::StoreOp>(Op))
            return true;
          return false;
        }))
      return false;

    for (auto *User : AllocaUsers)
      if (auto StoreOp = dyn_cast<memref::StoreOp>(User)) {
        Value StoreOperand = StoreOp.getOperand(0);
        Operation *MaybeIndexCast = StoreOperand.getDefiningOp();
        if (!isa<arith::IndexCastOp>(MaybeIndexCast))
          return false;
        Value IndexCastOperand = MaybeIndexCast->getOperand(0);
        if (auto BlockArg = IndexCastOperand.dyn_cast<BlockArgument>()) {
          if (auto AffineFor =
                  dyn_cast<AffineForOp>(BlockArg.getOwner()->getParentOp()))
            AffineForIndVar = AffineFor.getInductionVar();
          else
            return false;
        }
      }

    return true;
  }

  // at this point we expect only AddIOp or MulIOp
  if (!isa<arith::AddIOp>(Op) && !isa<arith::MulIOp>(Op))
    return false;

  // make sure that the current op has at least one constant operand
  // (ConstantIndexOp or ConstantIntOp)
  if (llvm::none_of(Op->getOperands(), [](Value Operand) {
        return (isa<arith::ConstantIndexOp>(Operand.getDefiningOp()) ||
                isa<arith::ConstantIntOp>(Operand.getDefiningOp()));
      }))
    return false;

  // build affine expression by adding or multiplying Constants.
  // and keep iterating on the non-constant index
  Value NonCstOperand = nullptr;
  for (auto Operand : Op->getOperands()) {
    if (auto ConstantIndexOp =
            dyn_cast<arith::ConstantIndexOp>(Operand.getDefiningOp())) {
      if (isa<arith::AddIOp>(Op))
        Expr = Expr + ConstantIndexOp.value();
      else
        Expr = Expr * ConstantIndexOp.value();
    } else if (auto ConstantIntOp =
                   dyn_cast<arith::ConstantIntOp>(Operand.getDefiningOp())) {
      if (isa<arith::AddIOp>(Op))
        Expr = Expr + ConstantIntOp.value();
      else
        Expr = Expr * ConstantIntOp.value();
    } else
      NonCstOperand = Operand;
  }
  return hasAffineArith(NonCstOperand.getDefiningOp(), Expr, AffineForIndVar);
}

ValueCategory MLIRScanner::VisitBinaryOperator(clang::BinaryOperator *BO) {
  Location Loc = getMLIRLocation(BO->getExprLoc());

  auto FixInteger = [&](Value Res) {
    auto PrevTy = Res.getType().cast<IntegerType>();
    auto PostTy =
        Glob.getTypes().getMLIRType(BO->getType()).cast<IntegerType>();
    bool SignedType = true;
    if (const auto *Bit = dyn_cast<clang::BuiltinType>(&*BO->getType())) {
      if (Bit->isUnsignedInteger())
        SignedType = false;
      if (Bit->isSignedInteger())
        SignedType = true;
    }
    if (PostTy != PrevTy) {
      if (SignedType)
        Res = Builder.create<arith::ExtSIOp>(Loc, PostTy, Res);
      else
        Res = Builder.create<arith::ExtUIOp>(Loc, PostTy, Res);
    }
    return ValueCategory(Res, /*isReference*/ false);
  };

  ValueCategory LHS = Visit(BO->getLHS());
  if (!LHS.val && BO->getOpcode() != clang::BinaryOperator::Opcode::BO_Comma) {
    BO->dump();
    BO->getLHS()->dump();
    assert(LHS.val);
  }

  switch (BO->getOpcode()) {
  case clang::BinaryOperator::Opcode::BO_LAnd: {
    Value Cond = LHS.getValue(Builder);
    if (auto LT = Cond.getType().dyn_cast<LLVM::LLVMPointerType>())
      Cond =
          Builder.create<LLVM::ICmpOp>(Loc, LLVM::ICmpPredicate::ne, Cond,
                                       Builder.create<LLVM::NullOp>(Loc, LT));

    LLVM_DEBUG({if (!Cond.getType().isa<IntegerType>()) {
      BO->dump();
      BO->getType()->dump();
      llvm::dbgs() << "cond: " << Cond << "\n";
    }});

    auto PrevTy = Cond.getType().cast<IntegerType>();
    if (!PrevTy.isInteger(1))
      Cond = Builder.create<arith::CmpIOp>(
          Loc, arith::CmpIPredicate::ne, Cond,
          Builder.create<arith::ConstantIntOp>(Loc, 0, PrevTy));

    auto IfOp = Builder.create<scf::IfOp>(
        Loc, ArrayRef<Type>({Builder.getIntegerType(1)}), Cond,
        /*hasElseRegion*/ true);

    Block::iterator OldPoint = Builder.getInsertionPoint();
    Block *OldBlock = Builder.getInsertionBlock();
    Builder.setInsertionPointToStart(&IfOp.getThenRegion().back());

    Value RHS = Visit(BO->getRHS()).getValue(Builder);
    assert(RHS != nullptr);
    if (auto LT = RHS.getType().dyn_cast<LLVM::LLVMPointerType>())
      RHS = Builder.create<LLVM::ICmpOp>(Loc, LLVM::ICmpPredicate::ne, RHS,
                                         Builder.create<LLVM::NullOp>(Loc, LT));

    if (!RHS.getType().cast<IntegerType>().isInteger(1))
      RHS = Builder.create<arith::CmpIOp>(
          Loc, arith::CmpIPredicate::ne, RHS,
          Builder.create<arith::ConstantIntOp>(Loc, 0, RHS.getType()));

    Builder.create<scf::YieldOp>(Loc, ArrayRef<Value>({RHS}));

    Builder.setInsertionPointToStart(&IfOp.getElseRegion().back());
    Builder.create<scf::YieldOp>(
        Loc, ArrayRef<Value>({Builder.create<arith::ConstantIntOp>(
                 Loc, 0, Builder.getIntegerType(1))}));

    Builder.setInsertionPoint(OldBlock, OldPoint);
    return FixInteger(IfOp.getResult(0));
  }

  case clang::BinaryOperator::Opcode::BO_LOr: {
    Value Cond = LHS.getValue(Builder);
    auto PrevTy = Cond.getType().cast<IntegerType>();
    if (!PrevTy.isInteger(1))
      Cond = Builder.create<arith::CmpIOp>(
          Loc, arith::CmpIPredicate::ne, Cond,
          Builder.create<arith::ConstantIntOp>(Loc, 0, PrevTy));

    auto IfOp = Builder.create<scf::IfOp>(
        Loc, ArrayRef<Type>({Builder.getIntegerType(1)}), Cond,
        /*hasElseRegion*/ true);

    Block::iterator OldPoint = Builder.getInsertionPoint();
    Block *OldBlock = Builder.getInsertionBlock();
    Builder.setInsertionPointToStart(&IfOp.getThenRegion().back());

    Builder.create<scf::YieldOp>(
        Loc, ArrayRef<Value>({Builder.create<arith::ConstantIntOp>(
                 Loc, 1, Builder.getIntegerType(1))}));

    Builder.setInsertionPointToStart(&IfOp.getElseRegion().back());
    Value RHS = Visit(BO->getRHS()).getValue(Builder);
    if (!RHS.getType().cast<IntegerType>().isInteger(1)) {
      RHS = Builder.create<arith::CmpIOp>(
          Loc, arith::CmpIPredicate::ne, RHS,
          Builder.create<arith::ConstantIntOp>(Loc, 0, RHS.getType()));
    }
    assert(RHS != nullptr);
    Builder.create<scf::YieldOp>(Loc, ArrayRef<Value>({RHS}));

    Builder.setInsertionPoint(OldBlock, OldPoint);

    return FixInteger(IfOp.getResult(0));
  }
  default:
    break;
  }

  ValueCategory RHS = Visit(BO->getRHS());
  if (!RHS.val && BO->getOpcode() != clang::BinaryOperator::Opcode::BO_Comma) {
    BO->getRHS()->dump();
    assert(RHS.val);
  }

  // TODO note assumptions made here about unsigned / unordered
  bool SignedType = true;
  if (const auto *Bit = dyn_cast<clang::BuiltinType>(&*BO->getType())) {
    if (Bit->isUnsignedInteger())
      SignedType = false;
    if (Bit->isSignedInteger())
      SignedType = true;
  }

  switch (BO->getOpcode()) {
  case clang::BinaryOperator::Opcode::BO_GT:
  case clang::BinaryOperator::Opcode::BO_GE:
  case clang::BinaryOperator::Opcode::BO_LT:
  case clang::BinaryOperator::Opcode::BO_LE:
  case clang::BinaryOperator::Opcode::BO_EQ:
  case clang::BinaryOperator::Opcode::BO_NE: {
    SignedType = true;
    if (const auto *Bit =
            dyn_cast<clang::BuiltinType>(&*BO->getLHS()->getType())) {
      if (Bit->isUnsignedInteger())
        SignedType = false;
      if (Bit->isSignedInteger())
        SignedType = true;
    }
    arith::CmpFPredicate FPred;
    arith::CmpIPredicate IPred;
    LLVM::ICmpPredicate LPred;
    switch (BO->getOpcode()) {
    case clang::BinaryOperator::Opcode::BO_GT:
      FPred = arith::CmpFPredicate::UGT;
      IPred =
          SignedType ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::ugt,
      LPred = LLVM::ICmpPredicate::ugt;
      break;
    case clang::BinaryOperator::Opcode::BO_GE:
      FPred = arith::CmpFPredicate::UGE;
      IPred =
          SignedType ? arith::CmpIPredicate::sge : arith::CmpIPredicate::uge,
      LPred = LLVM::ICmpPredicate::uge;
      break;
    case clang::BinaryOperator::Opcode::BO_LT:
      FPred = arith::CmpFPredicate::ULT;
      IPred =
          SignedType ? arith::CmpIPredicate::slt : arith::CmpIPredicate::ult,
      LPred = LLVM::ICmpPredicate::ult;
      break;
    case clang::BinaryOperator::Opcode::BO_LE:
      FPred = arith::CmpFPredicate::ULE;
      IPred =
          SignedType ? arith::CmpIPredicate::sle : arith::CmpIPredicate::ule,
      LPred = LLVM::ICmpPredicate::ule;
      break;
    case clang::BinaryOperator::Opcode::BO_EQ:
      FPred = arith::CmpFPredicate::UEQ;
      IPred = arith::CmpIPredicate::eq;
      LPred = LLVM::ICmpPredicate::eq;
      break;
    case clang::BinaryOperator::Opcode::BO_NE:
      FPred = arith::CmpFPredicate::UNE;
      IPred = arith::CmpIPredicate::ne;
      LPred = LLVM::ICmpPredicate::ne;
      break;
    default:
      llvm_unreachable("Unknown op in binary comparision switch");
    }

    Value LHSVal = LHS.getValue(Builder);
    Value RHSVal = RHS.getValue(Builder);
    if (auto MT = LHSVal.getType().dyn_cast<MemRefType>())
      LHSVal = Builder.create<polygeist::Memref2PointerOp>(
          Loc,
          LLVM::LLVMPointerType::get(MT.getElementType(),
                                     MT.getMemorySpaceAsInt()),
          LHSVal);

    if (auto MT = RHSVal.getType().dyn_cast<MemRefType>())
      RHSVal = Builder.create<polygeist::Memref2PointerOp>(
          Loc,
          LLVM::LLVMPointerType::get(MT.getElementType(),
                                     MT.getMemorySpaceAsInt()),
          RHSVal);

    Value Res;
    if (LHSVal.getType().isa<FloatType>())
      Res = Builder.create<arith::CmpFOp>(Loc, FPred, LHSVal, RHSVal);
    else if (LHSVal.getType().isa<LLVM::LLVMPointerType>())
      Res = Builder.create<LLVM::ICmpOp>(Loc, LPred, LHSVal, RHSVal);
    else
      Res = Builder.create<arith::CmpIOp>(Loc, IPred, LHSVal, RHSVal);

    return ValueCategory(Res, /*isReference*/ false);
  }

  case clang::BinaryOperator::Opcode::BO_Comma:
    return RHS;

  default:
    BO->dump();
    llvm_unreachable("unhandled opcode");
  }
}

template <typename T>
Value MLIRScanner::SYCLCommonFieldLookup(Value V, size_t FNum,
                                         llvm::ArrayRef<int64_t> Shape) {
  auto MT = V.getType().cast<MemRefType>();
  Type ElemTy = MT.getElementType();
  assert(ElemTy.isa<T>() && "Expecting element type to be the templated type");
  assert(sycl::isSYCLType(ElemTy) && "Expecting SYCL element type");
  auto SYCLElemTy = ElemTy.cast<T>();
  assert(FNum < SYCLElemTy.getBody().size() && "ERROR");

  const auto ElementType = SYCLElemTy.getBody()[FNum];
  const auto ResultType = MemRefType::get(
      Shape, ElementType, MemRefLayoutAttrInterface(), MT.getMemorySpace());

  return Builder.create<polygeist::SubIndexOp>(Loc, ResultType, V,
                                               getConstantIndex(FNum));
}

ValueCategory MLIRScanner::CommonFieldLookup(clang::QualType CT,
                                             const clang::FieldDecl *FD,
                                             Value Val, bool IsLValue) {
  assert(FD && "Attempting to lookup field of nullptr");

  const clang::RecordDecl *RD = FD->getParent();
  auto *ST = cast<llvm::StructType>(mlirclang::getLLVMType(CT, Glob.getCGM()));
  size_t FNum = 0;
  const auto *CXRD = dyn_cast<clang::CXXRecordDecl>(RD);

  if (mlirclang::CodeGen::CodeGenTypes::isLLVMStructABI(RD, ST)) {
    auto &Layout = Glob.getCGM().getTypes().getCGRecordLayout(RD);
    FNum = Layout.getLLVMFieldNo(FD);
  } else {
    FNum = 0;
    if (CXRD)
      FNum += CXRD->getDefinition()->getNumBases();
    for (auto *Field : RD->fields()) {
      if (Field == FD)
        break;

      ++FNum;
    }
  }

  if (auto PT = Val.getType().dyn_cast<LLVM::LLVMPointerType>()) {
    if (!PT.getElementType().isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>()) {
      llvm::errs() << "Function: " << Function << "\n";
      FD->dump();
      FD->getType()->dump();
      llvm::errs() << " val: " << Val << " - pt: " << PT << " fn: " << FNum
                   << " ST: " << *ST << "\n";
    }

    Type ET = TypeSwitch<Type, Type>(PT.getElementType())
                  .Case<LLVM::LLVMStructType>([FNum](LLVM::LLVMStructType ST) {
                    return ST.getBody()[FNum];
                  })
                  .Case<LLVM::LLVMArrayType, MemRefType>(
                      [](auto Ty) { return Ty.getElementType(); });
    Value CommonGep = Builder.create<LLVM::GEPOp>(
        Loc, LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()), Val,
        ArrayRef<Value>({Builder.create<arith::ConstantIntOp>(Loc, 0, 32),
                         Builder.create<arith::ConstantIntOp>(Loc, FNum, 32)}));

    if (RD->isUnion()) {
      LLVM::TypeFromLLVMIRTranslator TypeTranslator(*Module->getContext());
      auto SubType = TypeTranslator.translateType(
          mlirclang::getLLVMType(FD->getType(), Glob.getCGM()));
      CommonGep = Builder.create<LLVM::BitcastOp>(
          Loc, LLVM::LLVMPointerType::get(SubType, PT.getAddressSpace()),
          CommonGep);
    }

    if (IsLValue)
      CommonGep =
          ValueCategory(CommonGep, /*isReference*/ true).getValue(Builder);

    return ValueCategory(CommonGep, /*isReference*/ true);
  }

  auto MT = Val.getType().cast<MemRefType>();
  llvm::SmallVector<int64_t> Shape(MT.getShape());
  if (Shape.size() > 1)
    Shape.erase(Shape.begin());
  else
    Shape[0] = ShapedType::kDynamic;

  // JLE_QUEL::THOUGHTS
  // This redundancy is here because we might, at some point, create
  // an equivalent GEP or SubIndexOp operation for each sycl types or otherwise
  // clean the redundancy
  Value Result;
  if (auto ST = MT.getElementType().dyn_cast<LLVM::LLVMStructType>()) {
    assert(FNum < ST.getBody().size() && "ERROR");

    const auto ElementType = ST.getBody()[FNum];
    const auto ResultType = MemRefType::get(
        Shape, ElementType, MemRefLayoutAttrInterface(), MT.getMemorySpace());

    Result = Builder.create<polygeist::SubIndexOp>(Loc, ResultType, Val,
                                                   getConstantIndex(FNum));
  } else if (sycl::isSYCLType(MT.getElementType())) {
    Type ElemTy = MT.getElementType();
    Result =
        TypeSwitch<Type, Value>(ElemTy)
            .Case<sycl::ArrayType>([&](sycl::ArrayType AT) {
              assert(FNum < AT.getBody().size() && "ERROR");
              const auto ElemType = AT.getBody()[FNum].cast<MemRefType>();
              const auto ResultType = MemRefType::get(
                  ElemType.getShape(), ElemType.getElementType(),
                  MemRefLayoutAttrInterface(), MT.getMemorySpace());
              return Builder.create<polygeist::SubIndexOp>(
                  Loc, ResultType, Val, getConstantIndex(FNum));
            })
            .Case<sycl::AccessorType, sycl::AccessorImplDeviceType,
                  sycl::AccessorSubscriptType, sycl::AtomicType,
                  sycl::GetScalarOpType, sycl::GroupType, sycl::ItemBaseType,
                  sycl::ItemType, sycl::LocalAccessorBaseDeviceType,
                  sycl::LocalAccessorBaseType, sycl::LocalAccessorType,
                  sycl::MultiPtrType, sycl::NdItemType, sycl::NdRangeType,
                  sycl::StreamType, sycl::SwizzledVecType, sycl::VecType>(
                [&](auto ElemTy) {
                  return SYCLCommonFieldLookup<decltype(ElemTy)>(Val, FNum,
                                                                 Shape);
                });
  } else {
    auto MT0 =
        MemRefType::get(Shape, MT.getElementType(), MemRefLayoutAttrInterface(),
                        MT.getMemorySpace());
    Shape[0] = ShapedType::kDynamic;
    auto MT1 =
        MemRefType::get(Shape, MT.getElementType(), MemRefLayoutAttrInterface(),
                        MT.getMemorySpace());

    Result = Builder.create<polygeist::SubIndexOp>(Loc, MT0, Val,
                                                   getConstantIndex(0));
    Result = Builder.create<polygeist::SubIndexOp>(Loc, MT1, Result,
                                                   getConstantIndex(FNum));
  }

  if (IsLValue)
    Result = ValueCategory(Result, /*isReference*/ true).getValue(Builder);

  return ValueCategory(Result, /*isReference*/ true);
}

static bool isSYCLInheritType(Type &Ty, Value &Val) {
  assert(Val.getType().isa<MemRefType>());
  if (!Ty.isa<MemRefType>())
    return false;

  Type ElemTy = Ty.cast<MemRefType>().getElementType();
  return TypeSwitch<Type, bool>(
             Val.getType().cast<MemRefType>().getElementType())
      .Case<sycl::AccessorType>([&](auto) {
        return ElemTy.isa<sycl::AccessorCommonType>() ||
               ElemTy.isa<sycl::LocalAccessorBaseType>() ||
               ElemTy.isa<sycl::OwnerLessBaseType>();
      })
      .Case<sycl::LocalAccessorBaseType>(
          [&](auto) { return ElemTy.isa<sycl::AccessorCommonType>(); })
      .Case<sycl::LocalAccessorType>(
          [&](auto) { return ElemTy.isa<sycl::LocalAccessorBaseType>(); })
      .Case<sycl::IDType, sycl::RangeType>(
          [&](auto) { return ElemTy.isa<sycl::ArrayType>(); })
      .Default(false);
}

Value MLIRScanner::GetAddressOfDerivedClass(
    Value Val, const clang::CXXRecordDecl *DerivedClass,
    clang::CastExpr::path_const_iterator Start,
    clang::CastExpr::path_const_iterator End) {
  const clang::ASTContext &Context = Glob.getCGM().getContext();

  SmallVector<const clang::CXXRecordDecl *> ToBase = {DerivedClass};
  SmallVector<const clang::CXXBaseSpecifier *> Bases;
  for (const auto *I = Start; I != End; I++) {
    const clang::CXXBaseSpecifier *Base = *I;
    const auto *BaseDecl = cast<clang::CXXRecordDecl>(
        Base->getType()->castAs<clang::RecordType>()->getDecl());
    ToBase.push_back(BaseDecl);
    Bases.push_back(Base);
  }

  for (int I = ToBase.size() - 1; I > 0; I--) {
    const clang::CXXBaseSpecifier *Base = Bases[I - 1];

    const auto *BaseDecl = cast<clang::CXXRecordDecl>(
        Base->getType()->castAs<clang::RecordType>()->getDecl());
    const auto *RD = ToBase[I - 1];
    // Get the layout.
    const clang::ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    assert(!Base->isVirtual() && "Should not see virtual bases here!");

    // Add the offset.
    Type NT = Glob.getTypes().getMLIRType(
        Glob.getCGM().getContext().getLValueReferenceType(Base->getType()));

    Value Offset = nullptr;
    if (mlirclang::CodeGen::CodeGenTypes::isLLVMStructABI(RD, /*ST*/ nullptr)) {
      Offset = Builder.create<arith::ConstantIntOp>(
          Loc, -(ssize_t)Layout.getBaseClassOffset(BaseDecl).getQuantity(), 32);
    } else {
      Offset = Builder.create<arith::ConstantIntOp>(Loc, 0, 32);
      bool Found = false;
      for (auto F : RD->bases()) {
        if (F.getType().getTypePtr()->getUnqualifiedDesugaredType() ==
            Base->getType()->getUnqualifiedDesugaredType()) {
          Found = true;
          break;
        }
        bool SubType = false;
        Type NT = Glob.getTypes().getMLIRType(F.getType(), &SubType, false);
        Offset = Builder.create<arith::SubIOp>(
            Loc, Offset,
            Builder.create<arith::IndexCastOp>(
                Loc, Offset.getType(),
                Builder.create<polygeist::TypeSizeOp>(
                    Loc, Builder.getIndexType(), TypeAttr::get(NT))));
      }
      assert(Found);
    }

    Value Ptr = Val;
    if (auto PT = Ptr.getType().dyn_cast<LLVM::LLVMPointerType>())
      Ptr = Builder.create<LLVM::BitcastOp>(
          Loc,
          LLVM::LLVMPointerType::get(Builder.getI8Type(), PT.getAddressSpace()),
          Ptr);
    else
      Ptr = Builder.create<polygeist::Memref2PointerOp>(
          Loc,
          LLVM::LLVMPointerType::get(Builder.getI8Type(), PT.getAddressSpace()),
          Ptr);

    Ptr = Builder.create<LLVM::GEPOp>(Loc, Ptr.getType(), Ptr,
                                      ArrayRef<Value>({Offset}));

    if (auto PT = NT.dyn_cast<LLVM::LLVMPointerType>())
      Val = Builder.create<LLVM::BitcastOp>(
          Loc,
          LLVM::LLVMPointerType::get(
              PT.getElementType(),
              Ptr.getType().cast<LLVM::LLVMPointerType>().getAddressSpace()),
          Ptr);
    else
      Val = Builder.create<polygeist::Pointer2MemrefOp>(Loc, NT, Ptr);
  }

  return Val;
}

Value MLIRScanner::GetAddressOfBaseClass(
    Value Val, const clang::CXXRecordDecl *DerivedClass,
    ArrayRef<const clang::Type *> BaseTypes, ArrayRef<bool> BaseVirtuals) {
  const clang::CXXRecordDecl *RD = DerivedClass;

  for (auto Tup : llvm::zip(BaseTypes, BaseVirtuals)) {
    const auto *BaseType = std::get<0>(Tup);
    const auto *BaseDecl = cast<clang::CXXRecordDecl>(
        BaseType->castAs<clang::RecordType>()->getDecl());

    // Add the offset.
    Type NT = Glob.getTypes().getMLIRType(
        Glob.getCGM().getContext().getLValueReferenceType(
            clang::QualType(BaseType, 0)));

    size_t FNum;
    bool SubIndex = true;

    if (mlirclang::CodeGen::CodeGenTypes::isLLVMStructABI(RD, /*ST*/ nullptr)) {
      auto &Layout = Glob.getCGM().getTypes().getCGRecordLayout(RD);
      if (std::get<1>(Tup))
        FNum = Layout.getVirtualBaseIndex(BaseDecl);
      else {
        if (!Layout.hasNonVirtualBaseLLVMField(BaseDecl))
          SubIndex = false;
        else
          FNum = Layout.getNonVirtualBaseLLVMFieldNo(BaseDecl);
      }
    } else {
      assert(!std::get<1>(Tup) && "Should not see virtual bases here!");
      FNum = 0;
      bool Found = false;
      for (auto F : RD->bases()) {
        if (F.getType().getTypePtr()->getUnqualifiedDesugaredType() ==
            BaseType->getUnqualifiedDesugaredType()) {
          Found = true;
          break;
        }
        FNum++;
      }
      assert(Found);
    }

    if (SubIndex) {
      if (auto MT = Val.getType().dyn_cast<MemRefType>()) {
        auto Shape = std::vector<int64_t>(MT.getShape());
        // We do not remove dimensions for an id->array or range->array, because
        // the later cast will be incompatible due to dimension mismatch.
        if (!isSYCLInheritType(NT, Val))
          Shape.erase(Shape.begin());
        auto MT0 =
            MemRefType::get(Shape, MT.getElementType(),
                            MemRefLayoutAttrInterface(), MT.getMemorySpace());
        Val = Builder.create<polygeist::SubIndexOp>(Loc, MT0, Val,
                                                    getConstantIndex(FNum));
      } else {
        Value Idx[] = {Builder.create<arith::ConstantIntOp>(Loc, 0, 32),
                       Builder.create<arith::ConstantIntOp>(Loc, FNum, 32)};
        auto PT = Val.getType().cast<LLVM::LLVMPointerType>();
        Type ET = TypeSwitch<Type, Type>(PT.getElementType())
                      .Case<sycl::AccessorType, LLVM::LLVMStructType>(
                          [FNum](auto Ty) { return Ty.getBody()[FNum]; })
                      .Case<LLVM::LLVMArrayType>([](LLVM::LLVMArrayType AT) {
                        return AT.getElementType();
                      });

        Val = Builder.create<LLVM::GEPOp>(
            Loc, LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()), Val,
            Idx);
      }
    }

    auto PT = NT.dyn_cast<LLVM::LLVMPointerType>();
    if (auto Opt = Val.getType().dyn_cast<LLVM::LLVMPointerType>()) {
      if (!PT)
        Val = Builder.create<polygeist::Pointer2MemrefOp>(Loc, NT, Val);
      else {
        if (Val.getType() != NT)
          Val = Builder.create<LLVM::BitcastOp>(Loc, PT, Val);
      }
    } else {
      assert(Val.getType().isa<MemRefType>() &&
             "Expecting value to have MemRefType");
      if (PT) {
        assert(
            Val.getType().cast<MemRefType>().getMemorySpaceAsInt() ==
                PT.getAddressSpace() &&
            "The type of 'value' does not have the same memory space as 'pt'");
        Val = Builder.create<polygeist::Memref2PointerOp>(Loc, PT, Val);
      } else {
        if (Val.getType() != NT) {
          if (isSYCLInheritType(NT, Val))
            Val = Builder.create<sycl::SYCLCastOp>(Loc, NT, Val);
          else
            Val = Builder.create<memref::CastOp>(Loc, NT, Val);
        }
      }
    }

    RD = BaseDecl;
  }

  return Val;
}

Value MLIRScanner::reshapeRanklessGlobal(memref::GetGlobalOp GV) {
  assert(GV.getType().isa<MemRefType>() &&
         "Type of GetGlobalOp should be MemRef");
  MemRefType MT = GV.getType().cast<MemRefType>();
  if (!MT.getShape().empty())
    return GV;

  auto Shape = Builder.create<memref::AllocaOp>(
      Loc, MemRefType::get(1, IndexType::get(Builder.getContext())));
  return Builder.create<memref::ReshapeOp>(
      Loc,
      MemRefType::get(1, MT.getElementType(), MemRefLayoutAttrInterface(),
                      MT.getMemorySpace()),
      GV, Shape);
}

/******************************************************************************/
/*                             MLIRASTConsumer                                */
/******************************************************************************/

LLVM::LLVMFuncOp MLIRASTConsumer::getOrCreateMallocFunction() {
  std::string Name = "malloc";
  if (LLVMFunctions.find(Name) != LLVMFunctions.end())
    return LLVMFunctions[Name];

  MLIRContext *Ctx = Module->getContext();
  auto LLVMFnType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMPointerType::get(IntegerType::get(Ctx, 8)),
      ArrayRef<Type>(IntegerType::get(Ctx, 64)), false);

  LLVM::Linkage Lnk = LLVM::Linkage::External;
  OpBuilder Builder(Module->getContext());
  Builder.setInsertionPointToStart(Module->getBody());
  return LLVMFunctions[Name] = Builder.create<LLVM::LLVMFuncOp>(
             Module->getLoc(), Name, LLVMFnType, Lnk);
}

LLVM::LLVMFuncOp MLIRASTConsumer::getOrCreateFreeFunction() {
  std::string Name = "free";
  if (LLVMFunctions.find(Name) != LLVMFunctions.end())
    return LLVMFunctions[Name];

  MLIRContext *Ctx = Module->getContext();
  auto LLVMFnType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMVoidType::get(Ctx),
      ArrayRef<Type>({LLVM::LLVMPointerType::get(IntegerType::get(Ctx, 8))}),
      false);

  LLVM::Linkage Lnk = LLVM::Linkage::External;
  OpBuilder Builder(Module->getContext());
  Builder.setInsertionPointToStart(Module->getBody());
  return LLVMFunctions[Name] = Builder.create<LLVM::LLVMFuncOp>(
             Module->getLoc(), Name, LLVMFnType, Lnk);
}

LLVM::LLVMFuncOp
MLIRASTConsumer::getOrCreateLLVMFunction(const clang::FunctionDecl *FD) {
  std::string Name = MLIRScanner::getMangledFuncName(*FD, CGM);
  if (Name != "malloc" && Name != "free")
    Name = (PrefixABI + Name);

  if (LLVMFunctions.find(Name) != LLVMFunctions.end())
    return LLVMFunctions[Name];

  LLVM::TypeFromLLVMIRTranslator TypeTranslator(*Module->getContext());

  std::vector<Type> Types;
  if (const auto *CC = dyn_cast<clang::CXXMethodDecl>(FD))
    Types.push_back(TypeTranslator.translateType(
        mlirclang::anonymize(mlirclang::getLLVMType(CC->getThisType(), CGM))));

  for (auto *Parm : FD->parameters())
    Types.push_back(TypeTranslator.translateType(mlirclang::anonymize(
        mlirclang::getLLVMType(Parm->getOriginalType(), CGM))));

  auto RT = TypeTranslator.translateType(
      mlirclang::anonymize(mlirclang::getLLVMType(FD->getReturnType(), CGM)));
  auto LLVMFnType = LLVM::LLVMFunctionType::get(RT, Types,
                                                /*isVarArg=*/FD->isVariadic());
  // Insert the function into the body of the parent module.
  OpBuilder Builder(Module->getContext());
  Builder.setInsertionPointToStart(Module->getBody());

  return LLVMFunctions[Name] = Builder.create<LLVM::LLVMFuncOp>(
             Module->getLoc(), Name, LLVMFnType,
             getMLIRLinkage(CGM.getFunctionLinkage(FD)));
}

LLVM::GlobalOp
MLIRASTConsumer::getOrCreateLLVMGlobal(const clang::ValueDecl *FD,
                                       std::string Prefix) {
  std::string Name = Prefix + CGM.getMangledName(FD).str();
  Name = (PrefixABI + Name);

  if (LLVMGlobals.find(Name) != LLVMGlobals.end())
    return LLVMGlobals[Name];

  const auto *VD = dyn_cast<clang::VarDecl>(FD);
  if (!VD)
    FD->dump();
  VD = VD->getCanonicalDecl();

  auto Linkage = CGM.getLLVMLinkageVarDefinition(VD, /*isConstant*/ false);
  LLVM::Linkage Lnk = getMLIRLinkage(Linkage);
  Type RT = getTypes().getMLIRType(FD->getType());

  OpBuilder Builder(Module->getContext());
  Builder.setInsertionPointToStart(Module->getBody());

  auto Glob = Builder.create<LLVM::GlobalOp>(
      Module->getLoc(), RT, /*constant*/ false, Lnk, Name, Attribute());

  if (VD->getInit() ||
      VD->isThisDeclarationADefinition() == clang::VarDecl::Definition ||
      VD->isThisDeclarationADefinition() ==
          clang::VarDecl::TentativeDefinition) {
    Block *Blk = new Block();
    Builder.setInsertionPointToStart(Blk);
    Value Res;
    if (const auto *Init = VD->getInit()) {
      MLIRScanner MS(*this, Module, LTInfo);
      MS.setEntryAndAllocBlock(Blk);
      Res = MS.Visit(const_cast<clang::Expr *>(Init)).getValue(Builder);
    } else
      Res = ValueCategory::getUndefValue(Builder, Module->getLoc(), RT).val;

    bool Legal = true;
    for (Operation &Op : *Blk) {
      auto Iface = dyn_cast<MemoryEffectOpInterface>(Op);
      if (!Iface || !Iface.hasNoEffect()) {
        Legal = false;
        break;
      }
    }

    if (Legal) {
      Builder.create<LLVM::ReturnOp>(Module->getLoc(),
                                     std::vector<Value>({Res}));
      Glob.getInitializerRegion().push_back(Blk);
    } else {
      Block *Blk2 = new Block();
      Builder.setInsertionPointToEnd(Blk2);
      Value NRes =
          ValueCategory::getUndefValue(Builder, Module->getLoc(), RT).val;
      Builder.create<LLVM::ReturnOp>(Module->getLoc(),
                                     std::vector<Value>({NRes}));
      Glob.getInitializerRegion().push_back(Blk2);

      Builder.setInsertionPointToStart(Module->getBody());
      auto FuncName = Name + "@init";
      LLVM::GlobalCtorsOp Ctors = nullptr;
      for (auto &Op : *Module->getBody()) {
        if (auto C = dyn_cast<LLVM::GlobalCtorsOp>(&Op))
          Ctors = C;
      }

      SmallVector<Attribute> Funcs;
      Funcs.push_back(FlatSymbolRefAttr::get(Module->getContext(), FuncName));
      SmallVector<Attribute> Idxs;
      Idxs.push_back(Builder.getI32IntegerAttr(0));

      if (Ctors) {
        for (auto F : Ctors.getCtors())
          Funcs.push_back(F);
        for (auto V : Ctors.getPriorities())
          Idxs.push_back(V);
        Ctors->erase();
      }

      Builder.create<LLVM::GlobalCtorsOp>(Module->getLoc(),
                                          Builder.getArrayAttr(Funcs),
                                          Builder.getArrayAttr(Idxs));

      auto LLVMFnType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(Module->getContext()), ArrayRef<Type>(),
          false);

      auto Func = Builder.create<LLVM::LLVMFuncOp>(
          Module->getLoc(), FuncName, LLVMFnType, LLVM::Linkage::Private);
      Func.getRegion().push_back(Blk);
      Builder.setInsertionPointToEnd(Blk);
      Builder.create<LLVM::StoreOp>(
          Module->getLoc(), Res,
          Builder.create<LLVM::AddressOfOp>(Module->getLoc(), Glob));
      Builder.create<LLVM::ReturnOp>(Module->getLoc(), ArrayRef<Value>());
    }
  }

  if (Lnk == LLVM::Linkage::Private || Lnk == LLVM::Linkage::Internal)
    SymbolTable::setSymbolVisibility(Glob, SymbolTable::Visibility::Private);

  return LLVMGlobals[Name] = Glob;
}

std::pair<memref::GlobalOp, bool>
MLIRASTConsumer::getOrCreateGlobal(const clang::ValueDecl &VD,
                                   std::string Prefix,
                                   FunctionContext FuncContext) {
  const std::string Name = PrefixABI + Prefix + CGM.getMangledName(&VD).str();
  if (Globals.find(Name) != Globals.end())
    return Globals[Name];

  const bool IsArray = isa<clang::ArrayType>(VD.getType());
  const Type MLIRType = getTypes().getMLIRType(VD.getType());
  const clang::VarDecl *Var = cast<clang::VarDecl>(VD).getCanonicalDecl();
  const unsigned MemSpace =
      CGM.getContext().getTargetAddressSpace(CGM.GetGlobalVarAddressSpace(Var));

  // Note: global scalar variables have always memref type with rank zero.
  auto VarTy =
      (!IsArray) ? MemRefType::get({}, MLIRType, {}, MemSpace)
                 : MemRefType::get(MLIRType.cast<MemRefType>().getShape(),
                                   MLIRType.cast<MemRefType>().getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   mlirclang::wrapIntegerMemorySpace(
                                       MemSpace, Module->getContext()));

  // The insertion point depends on whether the global variable is in the host
  // or the device context.
  OpBuilder Builder(Module->getContext());
  if (FuncContext == FunctionContext::SYCLDevice)
    Builder.setInsertionPointToStart(
        mlirclang::getDeviceModule(*Module).getBody());
  else {
    assert(FuncContext == FunctionContext::Host);
    Builder.setInsertionPointToStart(Module->getBody());
  }

  // Create the global.
  clang::VarDecl::DefinitionKind DefKind = Var->isThisDeclarationADefinition();
  Attribute InitialVal;
  if (DefKind == clang::VarDecl::Definition ||
      DefKind == clang::VarDecl::TentativeDefinition)
    InitialVal = Builder.getUnitAttr();

  const bool IsConst = VD.getType().isConstQualified();
  llvm::Align Align = CGM.getContext().getDeclAlign(&VD).getAsAlign();

  auto GlobalOp = Builder.create<memref::GlobalOp>(
      Module->getLoc(), Name, /*sym_visibility*/ StringAttr(), VarTy,
      InitialVal, IsConst,
      Builder.getIntegerAttr(Builder.getIntegerType(64), Align.value()));

  // Set the visibility.
  switch (CGM.getLLVMLinkageVarDefinition(Var, IsConst)) {
  case llvm::GlobalValue::LinkageTypes::InternalLinkage:
  case llvm::GlobalValue::LinkageTypes::PrivateLinkage:
    SymbolTable::setSymbolVisibility(GlobalOp,
                                     SymbolTable::Visibility::Private);
    break;
  case llvm::GlobalValue::LinkageTypes::ExternalLinkage:
  case llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
  case llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
  case llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
  case llvm::GlobalValue::LinkageTypes::WeakODRLinkage:
  case llvm::GlobalValue::LinkageTypes::CommonLinkage:
  case llvm::GlobalValue::LinkageTypes::AppendingLinkage:
  case llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage:
  case llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
    SymbolTable::setSymbolVisibility(GlobalOp, SymbolTable::Visibility::Public);
    break;
  }

  // Initialize the global.
  const clang::Expr *InitExpr = Var->getAnyInitializer();

  if (!InitExpr) {
    if (DefKind != clang::VarDecl::DeclarationOnly) {
      // Tentative definitions are initialized to {0}.
      assert(!VD.getType()->isIncompleteType() && "Unexpected incomplete type");

      Attribute Zero =
          TypeSwitch<Type, Attribute>(VarTy.getElementType())
              .Case<IntegerType>(
                  [](auto Ty) { return IntegerAttr::get(Ty, 0); })
              .Case<FloatType>([](auto Ty) { return FloatAttr::get(Ty, 0); });
      auto ZeroVal = DenseElementsAttr::get(
          RankedTensorType::get(VarTy.getShape(), VarTy.getElementType()),
          Zero);
      GlobalOp.setInitialValueAttr(ZeroVal);
    }
  } else {
    // explicit initialization.
    assert(DefKind == clang::VarDecl::Definition);

    MLIRScanner MS(*this, Module, LTInfo);
    Block B;
    MS.setEntryAndAllocBlock(&B);

    OpBuilder Builder(Module->getContext());
    Builder.setInsertionPointToEnd(&B);
    auto Op = Builder.create<memref::AllocaOp>(Module->getLoc(), VarTy);

    if (isa<clang::InitListExpr>(InitExpr)) {
      Attribute InitValAttr = MS.InitializeValueByInitListExpr(
          Op, const_cast<clang::Expr *>(InitExpr));
      GlobalOp.setInitialValueAttr(InitValAttr);
    } else {
      ValueCategory VC = MS.Visit(const_cast<clang::Expr *>(InitExpr));
      assert(!VC.isReference && "The initializer should not be a reference");

      auto Op = VC.val.getDefiningOp<arith::ConstantOp>();
      assert(Op && "Could not find the initializer constant expression");

      auto InitialVal = SplatElementsAttr::get(
          RankedTensorType::get(VarTy.getShape(), VarTy.getElementType()),
          Op.getValue());
      GlobalOp.setInitialValueAttr(InitialVal);
    }
  }

  Globals[Name] = std::make_pair(GlobalOp, IsArray);

  return Globals[Name];
}

Value MLIRASTConsumer::getOrCreateGlobalLLVMString(
    Location Loc, OpBuilder &Builder, StringRef Value,
    FunctionContext FuncContext) {
  using namespace mlir;
  // Create the global at the entry of the module.
  if (LLVMStringGlobals.find(Value.str()) == LLVMStringGlobals.end()) {
    OpBuilder::InsertionGuard InsertGuard(Builder);
    switch (FuncContext) {
    case FunctionContext::SYCLDevice:
      Builder.setInsertionPointToStart(
          mlirclang::getDeviceModule(*Module).getBody());
      break;
    case FunctionContext::Host:
      Builder.setInsertionPointToStart(Module->getBody());
      break;
    }

    auto Type = LLVM::LLVMArrayType::get(
        IntegerType::get(Builder.getContext(), 8), Value.size() + 1);
    LLVMStringGlobals[Value.str()] = Builder.create<LLVM::GlobalOp>(
        Loc, Type, /*isConstant=*/true, LLVM::Linkage::Internal,
        "str" + std::to_string(LLVMStringGlobals.size()),
        Builder.getStringAttr(Value.str() + '\0'));
  }

  LLVM::GlobalOp Global = LLVMStringGlobals[Value.str()];
  // Get the pointer to the first character in the global string.
  return Builder.create<LLVM::AddressOfOp>(Loc, Global);
}

FunctionOpInterface
MLIRASTConsumer::getOrCreateMLIRFunction(FunctionToEmit &FTE, bool ShouldEmit,
                                         bool GetDeviceStub) {
  assert(FTE.getDecl().getTemplatedKind() !=
             clang::FunctionDecl::TemplatedKind::TK_FunctionTemplate &&
         FTE.getDecl().getTemplatedKind() !=
             clang::FunctionDecl::TemplatedKind::
                 TK_DependentFunctionTemplateSpecialization &&
         "Unexpected template kind");

  const clang::FunctionDecl &FD = FTE.getDecl();
  const std::string MangledName =
      (GetDeviceStub)
          ? PrefixABI +
                CGM.getMangledName(clang::GlobalDecl(
                                       &FD, clang::KernelReferenceKind::Kernel))
                    .str()
          : PrefixABI + MLIRScanner::getMangledFuncName(FD, CGM);
  assert(MangledName != "free");

  // Early exit if the function has already been generated.
  if (Optional<FunctionOpInterface> OptFunction =
          getMLIRFunction(MangledName, FTE.getContext()))
    return *OptFunction;

  // Create the MLIR function and set its various attributes.
  FunctionOpInterface Function =
      createMLIRFunction(FTE, MangledName, ShouldEmit);
  checkFunctionParent(Function, FTE.getContext(), Module);

  // Decide whether the MLIR function should be emitted.
  const clang::FunctionDecl *Def = nullptr;
  if (!FD.isDefined(Def, /*checkforfriend*/ true))
    Def = &FD;

  if (Def->isThisDeclarationADefinition()) {
    assert(Def->getTemplatedKind() !=
               clang::FunctionDecl::TemplatedKind::TK_FunctionTemplate &&
           Def->getTemplatedKind() !=
               clang::FunctionDecl::TemplatedKind::
                   TK_DependentFunctionTemplateSpecialization);
    if (ShouldEmit) {
      LLVM_DEBUG(llvm::dbgs()
                 << __LINE__ << ": Pushing " << FTE.getContext() << " function "
                 << Def->getNameAsString() << " to FunctionsToEmit\n");
      FunctionsToEmit.emplace_back(*Def, FTE.getContext());
    }
  } else if (ShouldEmit) {
    EmitIfFound.insert(MangledName);
  }

  tryToRegisterSYCLMethod(FD, Function);

  return Function;
}

const clang::CodeGen::CGFunctionInfo &
MLIRASTConsumer::getOrCreateCGFunctionInfo(const clang::FunctionDecl *FD) {
  auto Result = CGFunctionInfos.find(FD);
  if (Result != CGFunctionInfos.end())
    return *Result->second;

  clang::GlobalDecl GD;
  if (const auto *CC = dyn_cast<clang::CXXConstructorDecl>(FD))
    GD = clang::GlobalDecl(CC, clang::CXXCtorType::Ctor_Complete);
  else if (const auto *CC = dyn_cast<clang::CXXDestructorDecl>(FD))
    GD = clang::GlobalDecl(CC, clang::CXXDtorType::Dtor_Complete);
  else
    GD = clang::GlobalDecl(FD);

  CGFunctionInfos[FD] = &getTypes().arrangeGlobalDeclaration(GD);

  return *CGFunctionInfos[FD];
}

void MLIRASTConsumer::run() {
  while (FunctionsToEmit.size()) {
    FunctionToEmit FTE = FunctionsToEmit.front();
    FunctionsToEmit.pop_front();

    const clang::FunctionDecl &FD = FTE.getDecl();

    assert(FD.getBody());
    assert(FD.getTemplatedKind() != clang::FunctionDecl::TK_FunctionTemplate);
    assert(FD.getTemplatedKind() !=
           clang::FunctionDecl::TemplatedKind::
               TK_DependentFunctionTemplateSpecialization);

    std::string MangledName = MLIRScanner::getMangledFuncName(FD, CGM);
    const std::pair<FunctionContext, std::string> DoneKey(FTE.getContext(),
                                                          MangledName);
    if (Done.count(DoneKey))
      continue;

    LLVM_DEBUG({
      StringRef funcKind =
          (FD.hasAttr<clang::SYCLKernelAttr>()   ? "SYCL KERNEL"
           : FD.hasAttr<clang::SYCLDeviceAttr>() ? "SYCL DEVICE"
                                                 : "");
      llvm::dbgs() << "\n-- " << funcKind << " FUNCTION (" << FTE.getContext()
                   << " context) BEING EMITTED: " << FD.getNameAsString()
                   << " --\n\n";
    });

    Done.insert(DoneKey);
    MLIRScanner MS(*this, Module, LTInfo);
    FunctionOpInterface Function =
        getOrCreateMLIRFunction(FTE, true /* ShouldEmit */);
    MS.init(Function, FTE);

    LLVM_DEBUG({
      llvm::dbgs() << "\n";
      Function.dump();
      llvm::dbgs() << "\n";

      if (FunctionsToEmit.size()) {
        llvm::dbgs() << "-- FUNCTION(S) LEFT TO BE EMITTED --\n";

        for (const auto &FTE : FunctionsToEmit) {
          const clang::FunctionDecl &FD = FTE.getDecl();
          llvm::dbgs() << "  [+] " << FD.getNameAsString() << "(";
          for (unsigned int index = 0; index < FD.getNumParams(); index += 1) {
            printf("%s",
                   FD.getParamDecl(index)->getType().getAsString().c_str());
            if (index + 1 != FD.getNumParams())
              llvm::dbgs() << ", ";
          }
          llvm::dbgs() << ")\n";
        }
        llvm::dbgs() << "\n";
      }
    });
  }
}

void MLIRASTConsumer::HandleDeclContext(clang::DeclContext *DC) {
  for (auto *D : DC->decls()) {
    if (auto *NS = dyn_cast<clang::NamespaceDecl>(D)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto *NS = dyn_cast<clang::ExternCContextDecl>(D)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto *NS = dyn_cast<clang::LinkageSpecDecl>(D)) {
      HandleDeclContext(NS);
      continue;
    }
    const auto *FD = dyn_cast<clang::FunctionDecl>(D);
    if (!FD)
      continue;
    if (!FD->doesThisDeclarationHaveABody() &&
        !FD->doesDeclarationForceExternallyVisibleDefinition())
      continue;
    if (!FD->hasBody())
      continue;
    if (FD->isTemplated())
      continue;

    bool ExternLinkage = true;
    if (!CGM.getContext().DeclMustBeEmitted(FD))
      ExternLinkage = false;

    std::string Name = MLIRScanner::getMangledFuncName(*FD, CGM);

    // Don't create std functions unless necessary
    if (StringRef(Name).startswith("_ZNKSt"))
      continue;
    if (StringRef(Name).startswith("_ZSt"))
      continue;
    if (StringRef(Name).startswith("_ZNSt"))
      continue;
    if (StringRef(Name).startswith("_ZN9__gnu"))
      continue;
    if (Name == "cudaGetDevice" || Name == "cudaMalloc")
      continue;

    if ((EmitIfFound.count("*") && Name != "fpclassify" && !FD->isStatic() &&
         ExternLinkage) ||
        EmitIfFound.count(Name)) {
      FunctionToEmit FTE(*FD);
      LLVM_DEBUG(llvm::dbgs()
                 << __LINE__ << ": Pushing " << FTE.getContext() << " function "
                 << FD->getNameAsString() << " to FunctionsToEmit\n");
      FunctionsToEmit.push_back(FTE);
    }
  }
}

bool MLIRASTConsumer::HandleTopLevelDecl(clang::DeclGroupRef DG) {
  if (Error)
    return true;

  clang::DeclGroupRef::iterator It;
  for (It = DG.begin(); It != DG.end(); ++It) {
    if (auto *NS = dyn_cast<clang::NamespaceDecl>(*It)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto *NS = dyn_cast<clang::ExternCContextDecl>(*It)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto *NS = dyn_cast<clang::LinkageSpecDecl>(*It)) {
      HandleDeclContext(NS);
      continue;
    }
    const auto *FD = dyn_cast<clang::FunctionDecl>(*It);
    if (!FD)
      continue;
    if (!FD->doesThisDeclarationHaveABody() &&
        !FD->doesDeclarationForceExternallyVisibleDefinition())
      continue;
    if (!FD->hasBody())
      continue;
    if (FD->isTemplated())
      continue;

    bool ExternLinkage = true;
    if (!CGM.getContext().DeclMustBeEmitted(FD))
      ExternLinkage = false;

    std::string Name = MLIRScanner::getMangledFuncName(*FD, CGM);

    // Don't create std functions unless necessary
    if (StringRef(Name).startswith("_ZNKSt"))
      continue;
    if (StringRef(Name).startswith("_ZSt"))
      continue;
    if (StringRef(Name).startswith("_ZNSt"))
      continue;
    if (StringRef(Name).startswith("_ZN9__gnu"))
      continue;
    if (Name == "cudaGetDevice" || Name == "cudaMalloc")
      continue;

    if ((EmitIfFound.count("*") && Name != "fpclassify" && !FD->isStatic() &&
         ExternLinkage) ||
        EmitIfFound.count(Name) || FD->hasAttr<clang::OpenCLKernelAttr>() ||
        FD->hasAttr<clang::SYCLDeviceAttr>()) {
      FunctionToEmit FTE(*FD);
      LLVM_DEBUG(llvm::dbgs()
                 << __LINE__ << ": Pushing " << FTE.getContext() << " function "
                 << FD->getNameAsString() << " to FunctionsToEmit\n");
      FunctionsToEmit.push_back(FTE);
    }
  }

  return true;
}

// Wait until Sema has instantiated all the relevant code
// before running codegen on the selected functions.
void MLIRASTConsumer::HandleTranslationUnit(clang::ASTContext &C) { run(); }

Location MLIRASTConsumer::getMLIRLocation(clang::SourceLocation Loc) {
  auto SpellingLoc = SM.getSpellingLoc(Loc);
  auto LineNumber = SM.getSpellingLineNumber(SpellingLoc);
  auto ColNumber = SM.getSpellingColumnNumber(SpellingLoc);
  auto FileId = SM.getFilename(SpellingLoc);

  auto *Ctx = Module->getContext();
  return FileLineColLoc::get(Ctx, FileId, LineNumber, ColNumber);
}

llvm::GlobalValue::LinkageTypes
MLIRASTConsumer::getLLVMLinkageType(const clang::FunctionDecl &FD,
                                    bool ShouldEmit) {
  if (!FD.hasBody() || !ShouldEmit)
    return llvm::GlobalValue::LinkageTypes::ExternalLinkage;
  if (const auto *CC = dyn_cast<clang::CXXConstructorDecl>(&FD))
    return CGM.getFunctionLinkage(
        clang::GlobalDecl(CC, clang::CXXCtorType::Ctor_Complete));
  if (const auto *CC = dyn_cast<clang::CXXDestructorDecl>(&FD))
    return CGM.getFunctionLinkage(
        clang::GlobalDecl(CC, clang::CXXDtorType::Dtor_Complete));

  return CGM.getFunctionLinkage(&FD);
}

LLVM::Linkage
MLIRASTConsumer::getMLIRLinkage(llvm::GlobalValue::LinkageTypes LV) {
  switch (LV) {
  case llvm::GlobalValue::LinkageTypes::InternalLinkage:
    return LLVM::Linkage::Internal;
  case llvm::GlobalValue::LinkageTypes::ExternalLinkage:
    return LLVM::Linkage::External;
  case llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
    return LLVM::Linkage::AvailableExternally;
  case llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
    return LLVM::Linkage::Linkonce;
  case llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
    return LLVM::Linkage::Weak;
  case llvm::GlobalValue::LinkageTypes::WeakODRLinkage:
    return LLVM::Linkage::WeakODR;
  case llvm::GlobalValue::LinkageTypes::CommonLinkage:
    return LLVM::Linkage::Common;
  case llvm::GlobalValue::LinkageTypes::AppendingLinkage:
    return LLVM::Linkage::Appending;
  case llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage:
    return LLVM::Linkage::ExternWeak;
  case llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
    return LLVM::Linkage::LinkonceODR;
  case llvm::GlobalValue::LinkageTypes::PrivateLinkage:
    return LLVM::Linkage::Private;
  }

  llvm_unreachable("Unexpected linkage");
}

FunctionOpInterface
MLIRASTConsumer::createMLIRFunction(const FunctionToEmit &FTE,
                                    std::string MangledName, bool ShouldEmit) {
  const clang::FunctionDecl &FD = FTE.getDecl();
  Location Loc = getMLIRLocation(FD.getLocation());
  OpBuilder Builder(Module->getContext());

  const clang::CodeGen::CGFunctionInfo &FI = getOrCreateCGFunctionInfo(&FD);
  FunctionType FuncTy = getTypes().getFunctionType(FI, FD);

  FunctionOpInterface Function =
      FD.hasAttr<clang::SYCLKernelAttr>()
          ? Builder.create<gpu::GPUFuncOp>(Loc, MangledName, FuncTy)
          : Builder.create<func::FuncOp>(Loc, MangledName, FuncTy);

  setMLIRFunctionVisibility(Function, FTE, ShouldEmit);
  setMLIRFunctionAttributes(Function, FTE, ShouldEmit);

  /// Inject the MLIR function created in either the device module or in the
  /// host module, depending on the calling context.
  switch (FTE.getContext()) {
  case FunctionContext::Host:
    Module->push_back(Function);
    Functions[MangledName] = cast<func::FuncOp>(Function);
    break;
  case FunctionContext::SYCLDevice:
    mlirclang::getDeviceModule(*Module).push_back(Function);
    DeviceFunctions[MangledName] = Function;
    break;
  }

  LLVM_DEBUG(llvm::dbgs() << "Created MLIR function: " << Function << "\n");

  return Function;
}

void MLIRASTConsumer::setMLIRFunctionVisibility(FunctionOpInterface Function,
                                                const FunctionToEmit &FTE,
                                                bool ShouldEmit) {
  const clang::FunctionDecl &FD = FTE.getDecl();
  SymbolTable::Visibility Visibility = SymbolTable::Visibility::Public;

  if (!ShouldEmit || !FD.isDefined() || FD.hasAttr<clang::CUDAGlobalAttr>() ||
      FD.hasAttr<clang::CUDADeviceAttr>())
    Visibility = SymbolTable::Visibility::Private;
  else {
    llvm::GlobalValue::LinkageTypes LV = getLLVMLinkageType(FD, ShouldEmit);
    if (LV == llvm::GlobalValue::InternalLinkage ||
        LV == llvm::GlobalValue::PrivateLinkage)
      Visibility = SymbolTable::Visibility::Private;
  }

  SymbolTable::setSymbolVisibility(Function, Visibility);
}

/// Determines whether the language options require us to model unwind
/// exceptions.  We treat -fexceptions as mandating this except under the
/// fragile ObjC ABI with only ObjC exceptions enabled.  This means, for
/// example, that C with -fexceptions enables this.
static bool hasUnwindExceptions(const clang::LangOptions &LangOpts) {
  // If exceptions are completely disabled, obviously this is false.
  if (!LangOpts.Exceptions)
    return false;

  // If C++ exceptions are enabled, this is true.
  if (LangOpts.CXXExceptions)
    return true;

  // If ObjC exceptions are enabled, this depends on the ABI.
  if (LangOpts.ObjCExceptions)
    return LangOpts.ObjCRuntime.hasUnwindExceptions();

  return true;
}

void MLIRASTConsumer::setMLIRFunctionAttributesForDefinition(
    const clang::Decl *D, FunctionOpInterface F) const {
  const clang::CodeGenOptions &CodeGenOpts = CGM.getCodeGenOpts();
  const clang::LangOptions &LangOpts = CGM.getLangOpts();
  MLIRContext *Ctx = Module->getContext();
  mlirclang::AttrBuilder B(*Ctx);

  if ((!D || !D->hasAttr<clang::NoUwtableAttr>()) && CodeGenOpts.UnwindTables)
    B.addPassThroughAttribute(
        llvm::Attribute::UWTable,
        uint64_t(llvm::UWTableKind(CodeGenOpts.UnwindTables)));

  if (CodeGenOpts.StackClashProtector)
    B.addPassThroughAttribute("probe-stack",
                              StringAttr::get(Ctx, "inline-asm"));

  if (!hasUnwindExceptions(LangOpts))
    B.addPassThroughAttribute(llvm::Attribute::NoUnwind);

  if (D && D->hasAttr<clang::NoStackProtectorAttr>())
    ; // Do Nothing.
  else if (D && D->hasAttr<clang::StrictGuardStackCheckAttr>() &&
           LangOpts.getStackProtector() == clang::LangOptions::SSPOn)
    B.addPassThroughAttribute(llvm::Attribute::StackProtectStrong);
  else if (LangOpts.getStackProtector() == clang::LangOptions::SSPOn)
    B.addPassThroughAttribute(llvm::Attribute::StackProtect);
  else if (LangOpts.getStackProtector() == clang::LangOptions::SSPStrong)
    B.addPassThroughAttribute(llvm::Attribute::StackProtectStrong);
  else if (LangOpts.getStackProtector() == clang::LangOptions::SSPReq)
    B.addPassThroughAttribute(llvm::Attribute::StackProtectReq);

  if (!D) {
    // If we don't have a declaration to control inlining, the function isn't
    // explicitly marked as alwaysinline for semantic reasons, and inlining is
    // disabled, mark the function as noinline.
    if (!F->hasAttr(llvm::Attribute::getNameFromAttrKind(
            llvm::Attribute::AlwaysInline)) &&
        CodeGenOpts.getInlining() == clang::CodeGenOptions::OnlyAlwaysInlining)
      B.addPassThroughAttribute(llvm::Attribute::NoInline);

    NamedAttrList Attrs(F->getAttrDictionary());
    Attrs.append(B.getAttributes());
    F->setAttrs(Attrs.getDictionary(Ctx));
    return;
  }

  // Track whether we need to add the optnone LLVM attribute,
  // starting with the default for this optimization level.
  bool ShouldAddOptNone = !CodeGenOpts.DisableO0ImplyOptNone &&
                          (CodeGenOpts.OptimizationLevel == 0u);
  // We can't add optnone in the following cases, it won't pass the verifier.
  ShouldAddOptNone &= !D->hasAttr<clang::MinSizeAttr>();
  ShouldAddOptNone &= !D->hasAttr<clang::AlwaysInlineAttr>();

  // Add optnone, but do so only if the function isn't always_inline.
  if ((ShouldAddOptNone || D->hasAttr<clang::OptimizeNoneAttr>()) &&
      !F->hasAttr(llvm::Attribute::getNameFromAttrKind(
          llvm::Attribute::AlwaysInline))) {
    B.addPassThroughAttribute(llvm::Attribute::OptimizeNone);

    // OptimizeNone implies noinline; we should not be inlining such functions.
    B.addPassThroughAttribute(llvm::Attribute::NoInline);

    // We still need to handle naked functions even though optnone subsumes
    // much of their semantics.
    if (D->hasAttr<clang::NakedAttr>())
      B.addPassThroughAttribute(llvm::Attribute::Naked);

    // OptimizeNone wins over OptimizeForSize and MinSize.
    F->removeAttr(
        llvm::Attribute::getNameFromAttrKind(llvm::Attribute::OptimizeForSize));
    F->removeAttr(
        llvm::Attribute::getNameFromAttrKind(llvm::Attribute::MinSize));
  } else if (D->hasAttr<clang::NakedAttr>()) {
    // Naked implies noinline: we should not be inlining such functions.
    B.addPassThroughAttribute(llvm::Attribute::Naked);
    B.addPassThroughAttribute(llvm::Attribute::NoInline);
  } else if (D->hasAttr<clang::NoDuplicateAttr>()) {
    B.addPassThroughAttribute(llvm::Attribute::NoDuplicate);
  } else if (D->hasAttr<clang::NoInlineAttr>() &&
             !F->hasAttr(llvm::Attribute::getNameFromAttrKind(
                 llvm::Attribute::AlwaysInline))) {
    // Add noinline if the function isn't always_inline.
    B.addPassThroughAttribute(llvm::Attribute::NoInline);
  } else if (D->hasAttr<clang::AlwaysInlineAttr>() &&
             !F->hasAttr(llvm::Attribute::getNameFromAttrKind(
                 llvm::Attribute::NoInline))) {
    // (noinline wins over always_inline, and we can't specify both in IR)
    B.addPassThroughAttribute(llvm::Attribute::AlwaysInline);
  } else if (CodeGenOpts.getInlining() ==
             clang::CodeGenOptions::OnlyAlwaysInlining) {
    // If we're not inlining, then force everything that isn't always_inline to
    // carry an explicit noinline attribute.
    if (!F->hasAttr(llvm::Attribute::getNameFromAttrKind(
            llvm::Attribute::AlwaysInline)))
      B.addPassThroughAttribute(llvm::Attribute::NoInline);
  } else {
    // Otherwise, propagate the inline hint attribute and potentially use its
    // absence to mark things as noinline.
    if (auto *FD = dyn_cast<clang::FunctionDecl>(D)) {
      // Search function and template pattern redeclarations for inline.
      auto CheckForInline = [](const clang::FunctionDecl *FD) {
        auto CheckRedeclForInline = [](const clang::FunctionDecl *Redecl) {
          return Redecl->isInlineSpecified();
        };
        if (any_of(FD->redecls(), CheckRedeclForInline))
          return true;
        const clang::FunctionDecl *Pattern =
            FD->getTemplateInstantiationPattern();
        if (!Pattern)
          return false;
        return any_of(Pattern->redecls(), CheckRedeclForInline);
      };
      if (CheckForInline(FD)) {
        B.addPassThroughAttribute(llvm::Attribute::InlineHint);
      } else if (CodeGenOpts.getInlining() ==
                     clang::CodeGenOptions::OnlyHintInlining &&
                 !FD->isInlined() &&
                 !F->hasAttr(llvm::Attribute::getNameFromAttrKind(
                     llvm::Attribute::AlwaysInline))) {
        B.addPassThroughAttribute(llvm::Attribute::NoInline);
      }
    }
  }

  // Add other optimization related attributes if we are optimizing this
  // Function.
  if (!D->hasAttr<clang::OptimizeNoneAttr>()) {
    if (D->hasAttr<clang::ColdAttr>()) {
      if (!ShouldAddOptNone)
        B.addPassThroughAttribute(llvm::Attribute::OptimizeForSize);
      B.addPassThroughAttribute(llvm::Attribute::Cold);
    }
    if (D->hasAttr<clang::HotAttr>())
      B.addPassThroughAttribute(llvm::Attribute::Hot);
    if (D->hasAttr<clang::MinSizeAttr>())
      B.addPassThroughAttribute(llvm::Attribute::MinSize);
  }

  NamedAttrList Attrs(F->getAttrDictionary());
  Attrs.append(B.getAttributes());
  F->setAttrs(Attrs.getDictionary(Ctx));

  unsigned Alignment = D->getMaxAlignment() / CGM.getContext().getCharWidth();
  if (Alignment) {
    OpBuilder Builder(Ctx);
    F->setAttr(llvm::Attribute::getNameFromAttrKind(llvm::Attribute::Alignment),
               Builder.getIntegerAttr(Builder.getIntegerType(64), Alignment));
  }

  if (!D->hasAttr<clang::AlignedAttr>())
    if (LangOpts.FunctionAlignment) {
      OpBuilder Builder(Ctx);
      F->setAttr(
          llvm::Attribute::getNameFromAttrKind(llvm::Attribute::Alignment),
          Builder.getIntegerAttr(Builder.getIntegerType(64),
                                 1ull << LangOpts.FunctionAlignment));
    }

  // Some C++ ABIs require 2-byte alignment for member functions, in order to
  // reserve a bit for differentiating between virtual and non-virtual member
  // functions. If the current target's C++ ABI requires this and this is a
  // member function, set its alignment accordingly.
  if (CGM.getTarget().getCXXABI().areMemberFunctionsAligned()) {
    StringRef AlignAttrName =
        llvm::Attribute::getNameFromAttrKind(llvm::Attribute::Alignment);

    if (auto AlignmentAttr = F->getAttrOfType<ArrayAttr>(AlignAttrName)) {
      assert(AlignmentAttr.size() == 2);
      unsigned AlignVal = AlignmentAttr[1].cast<IntegerAttr>().getInt();
      if (AlignVal < 2 && isa<clang::CXXMethodDecl>(D)) {
        OpBuilder Builder(Ctx);
        F->setAttr(AlignAttrName,
                   Builder.getIntegerAttr(Builder.getIntegerType(64), 2));
      }
    }
  }

#if 0
  // TODO: handle metadata generation.
  // In the cross-dso CFI mode with canonical jump tables, we want !type
  // attributes on definitions only.
  if (CGM.getCodeGenOpts().SanitizeCfiCrossDso &&
      CGM.getCodeGenOpts().SanitizeCfiCanonicalJumpTables) {
    if (auto *FD = dyn_cast<FunctionDecl>(D)) {
      // Skip available_externally functions. They won't be codegen'ed in the
      // current module anyway.
      if (CGM.getContext().GetGVALinkageForFunction(FD) !=
          GVA_AvailableExternally)
        CreateFunctionTypeMetadataForIcall(FD, F);
    }
  }

  // Emit type metadata on member functions for member function pointer checks.
  // These are only ever necessary on definitions; we're guaranteed that the
  // definition will be present in the LTO unit as a result of LTO visibility.
  auto *MD = dyn_cast<clang::CXXMethodDecl>(D);
  if (MD && requiresMemberFunctionPointerTypeMetadata(*this, MD)) {
    for (const CXXRecordDecl *Base : getMostBaseClasses(MD->getParent())) {
      llvm::Metadata *Id =
          CreateMetadataIdentifierForType(Context.getMemberPointerType(
              MD->getType(), Context.getRecordType(Base).getTypePtr()));
      F->addTypeMetadata(0, Id);
    }
  }
#endif
}

void MLIRASTConsumer::setMLIRFunctionAttributes(FunctionOpInterface Function,
                                                const FunctionToEmit &FTE,
                                                bool ShouldEmit) {
  using Attribute = llvm::Attribute;

  const clang::FunctionDecl &FD = FTE.getDecl();
  MLIRContext *Ctx = Module->getContext();

  bool IsDeviceContext = (FTE.getContext() == FunctionContext::SYCLDevice);
  if (!IsDeviceContext) {
    LLVM_DEBUG(llvm::dbgs()
               << "Not in a device context - skipping setting attributes for "
               << FD.getNameAsString() << "\n");

    mlirclang::AttrBuilder AttrBuilder(*Ctx);
    LLVM::Linkage Lnk = getMLIRLinkage(getLLVMLinkageType(FD, ShouldEmit));
    AttrBuilder.addAttribute("llvm.linkage", LLVM::LinkageAttr::get(Ctx, Lnk));

    // HACK: we want to avoid setting additional attributes on non-sycl
    // functions because we do not want to adjust the test cases at this time
    // (if we did we would have merge conflicts if we ever update polygeist).
    NamedAttrList Attrs(Function->getAttrDictionary());
    Attrs.append(AttrBuilder.getAttributes());
    Function->setAttrs(Attrs.getDictionary(Module->getContext()));
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Setting attributes for " << FD.getNameAsString()
                          << "\n");

  mlirclang::AttributeList PAL;
  {
    const clang::CodeGen::CGFunctionInfo &FI = getOrCreateCGFunctionInfo(&FD);
    const auto *FPT = FD.getType()->getAs<clang::FunctionProtoType>();
    clang::CodeGen::CGCalleeInfo CalleeInfo(FPT);

    unsigned CallingConv;
    getTypes().constructAttributeList(Function.getName(), FI, CalleeInfo, PAL,
                                      CallingConv,
                                      /*AttrOnCallSite*/ false,
                                      /*IsThunk*/ false);

    // Set additional function attributes that are not derivable from the
    // function declaration.
    mlirclang::AttrBuilder AttrBuilder(*Ctx);
    {
      AttrBuilder.addAttribute(
          "llvm.cconv", LLVM::CConvAttr::get(
                            Ctx, static_cast<LLVM::cconv::CConv>(CallingConv)));

      LLVM::Linkage Lnk = getMLIRLinkage(getLLVMLinkageType(FD, ShouldEmit));
      AttrBuilder.addAttribute("llvm.linkage",
                               LLVM::LinkageAttr::get(Ctx, Lnk));

      if (FD.hasAttr<clang::SYCLKernelAttr>())
        AttrBuilder.addAttribute(gpu::GPUDialect::getKernelFuncAttrName(),
                                 UnitAttr::get(Ctx));

      if (CGM.getLangOpts().SYCLIsDevice)
        AttrBuilder.addPassThroughAttribute(
            "sycl-module-id",
            StringAttr::get(Ctx, LLVMMod.getModuleIdentifier()));

      // If we're in C++ mode and the function name is "main", it is
      // guaranteed to be norecurse by the standard (3.6.1.3 "The function
      // main shall not be used within a program").
      //
      // OpenCL C 2.0 v2.2-11 s6.9.i:
      //     Recursion is not supported.
      //
      // SYCL v1.2.1 s3.10:
      //     kernels cannot include RTTI information, exception classes,
      //     recursive code, virtual functions or make use of C++ libraries that
      //     are not compiled for the device.
      if ((CGM.getLangOpts().CPlusPlus && FD.isMain()) ||
          CGM.getLangOpts().OpenCL || CGM.getLangOpts().SYCLIsDevice ||
          (CGM.getLangOpts().CUDA && FD.hasAttr<clang::CUDAGlobalAttr>()))
        AttrBuilder.addPassThroughAttribute(
            llvm::Attribute::AttrKind::NoRecurse);

      // Note: this one is incorrect because we should traverse the function
      // body before setting this attribute. If the body does not contain
      // any infinite loops the attributes can be set.
      auto FunctionMustProgress =
          [](const clang::CodeGen::CodeGenModule &CGM) -> bool {
        if (CGM.getCodeGenOpts().getFiniteLoops() ==
            clang::CodeGenOptions::FiniteLoopsKind::Never)
          return false;
        return CGM.getLangOpts().CPlusPlus11;
      };

      if (FunctionMustProgress(CGM))
        AttrBuilder.addPassThroughAttribute(Attribute::AttrKind::MustProgress);
    }

    setMLIRFunctionAttributesForDefinition(&FD, Function);

    PAL.addFnAttrs(AttrBuilder);
  }

  // Set function attributes.
  mlirclang::AttributeList FnAttrs(Function->getAttrDictionary(), {}, {});
  FnAttrs.addFnAttrs(PAL.getFnAttributes(), *Ctx);
  Function->setAttrs(FnAttrs.getFnAttributes().getDictionary(Ctx));

  // Set parameters attributes.
  const ArrayRef<NamedAttrList> ParamAttrs = PAL.getParamAttributes();
  assert(ParamAttrs.size() == Function.getNumArguments());
  for (unsigned Index : llvm::seq<unsigned>(0, Function.getNumArguments())) {
    for (NamedAttribute Attr : ParamAttrs[Index])
      Function.setArgAttr(Index, Attr.getName(), Attr.getValue());
  }

  // Set function result attributes.
  for (NamedAttribute Attr : PAL.getRetAttributes())
    Function.setResultAttr(0, Attr.getName(), Attr.getValue());
}

llvm::Optional<FunctionOpInterface>
MLIRASTConsumer::getMLIRFunction(const std::string &MangledName,
                                 FunctionContext Context) const {
  const auto Find = [MangledName](const auto &Map) {
    const auto Iter = Map.find(MangledName);
    return Iter == Map.end()
               ? std::nullopt
               : llvm::Optional<FunctionOpInterface>{Iter->second};
  };

  switch (Context) {
  case FunctionContext::Host:
    return Find(Functions);
  case FunctionContext::SYCLDevice:
    return Find(DeviceFunctions);
  }
  llvm_unreachable("Invalid function context");
}

#include "clang/Frontend/FrontendAction.h"
#include "llvm/Support/Host.h"

class MLIRAction : public clang::ASTFrontendAction {
public:
  std::set<std::string> EmitIfFound;
  std::set<std::pair<FunctionContext, std::string>> Done;
  OwningOpRef<ModuleOp> &Module;
  std::map<std::string, LLVM::GlobalOp> LLVMStringGlobals;
  std::map<std::string, std::pair<memref::GlobalOp, bool>> Globals;
  std::map<std::string, func::FuncOp> Functions;
  std::map<std::string, FunctionOpInterface> DeviceFunctions;
  std::map<std::string, LLVM::GlobalOp> LLVMGlobals;
  std::map<std::string, LLVM::LLVMFuncOp> LLVMFunctions;
  std::string ModuleId;
  MLIRAction(std::string Fn, OwningOpRef<ModuleOp> &Module,
             std::string ModuleId)
      : Module(Module), ModuleId(ModuleId) {
    EmitIfFound.insert(Fn);
  }
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI, StringRef InFile) override {
    return std::unique_ptr<clang::ASTConsumer>(new MLIRASTConsumer(
        EmitIfFound, Done, LLVMStringGlobals, Globals, Functions,
        DeviceFunctions, LLVMGlobals, LLVMFunctions, CI.getPreprocessor(),
        CI.getASTContext(), Module, CI.getSourceManager(), CI.getCodeGenOpts(),
        ModuleId));
  }
};

FunctionOpInterface MLIRScanner::EmitDirectCallee(const clang::FunctionDecl *FD,
                                                  FunctionContext Context) {
  FunctionToEmit FTE(*FD, Context);
  return Glob.getOrCreateMLIRFunction(FTE, true /* ShouldEmit */);
}

Location MLIRScanner::getMLIRLocation(clang::SourceLocation Loc) {
  return Glob.getMLIRLocation(Loc);
}

Value MLIRScanner::getTypeSize(clang::QualType QT) {
  // llvm::Type *T = Glob.getCGM().getTypes().ConvertType(t);
  // return (Glob.LLVMMod.getDataLayout().getTypeSizeInBits(T) + 7) / 8;
  bool IsArray = false;
  auto InnerTy = Glob.getTypes().getMLIRType(QT, &IsArray);
  if (IsArray) {
    auto MT = InnerTy.cast<MemRefType>();
    size_t Num = 1;
    for (auto N : MT.getShape()) {
      assert(N > 0);
      Num *= N;
    }
    return Builder.create<arith::MulIOp>(
        Loc,
        Builder.create<polygeist::TypeSizeOp>(
            Loc, Builder.getIndexType(), TypeAttr::get(MT.getElementType())),
        Builder.create<arith::ConstantIndexOp>(Loc, Num));
  }
  assert(!IsArray);
  return Builder.create<polygeist::TypeSizeOp>(
      Loc, Builder.getIndexType(),
      TypeAttr::get(InnerTy)); // DLI.getTypeSize(innerTy);
}

Value MLIRScanner::getTypeAlign(clang::QualType QT) {
  // llvm::Type *T = Glob.getCGM().getTypes().ConvertType(t);
  // return (Glob.LLVMMod.getDataLayout().getTypeSizeInBits(T) + 7) / 8;
  bool IsArray = false;
  auto InnerTy = Glob.getTypes().getMLIRType(QT, &IsArray);
  assert(!IsArray);
  return Builder.create<polygeist::TypeAlignOp>(
      Loc, Builder.getIndexType(),
      TypeAttr::get(InnerTy)); // DLI.getTypeSize(innerTy);
}

std::string
MLIRScanner::getMangledFuncName(const clang::FunctionDecl &FD,
                                clang::CodeGen::CodeGenModule &CGM) {
  if (const auto *CC = dyn_cast<clang::CXXConstructorDecl>(&FD))
    return CGM
        .getMangledName(
            clang::GlobalDecl(CC, clang::CXXCtorType::Ctor_Complete))
        .str();
  if (const auto *CC = dyn_cast<clang::CXXDestructorDecl>(&FD))
    return CGM
        .getMangledName(
            clang::GlobalDecl(CC, clang::CXXDtorType::Dtor_Complete))
        .str();

  return CGM.getMangledName(&FD).str();
}

#include "clang/Frontend/TextDiagnosticBuffer.h"

static bool parseMLIR(const char *Argv0, std::vector<std::string> Filenames,
                      std::string Fn, std::vector<std::string> IncludeDirs,
                      std::vector<std::string> Defines,
                      OwningOpRef<ModuleOp> &Module, llvm::Triple &Triple,
                      llvm::DataLayout &DL,
                      std::vector<std::string> InputCommandArgs) {
  clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagID(
      new clang::DiagnosticIDs());
  // Buffer diagnostics from argument parsing so that we can output them using
  // a well formed diagnostic object.
  auto *DiagsBuffer = new clang::TextDiagnosticBuffer();
  clang::DiagnosticsEngine Diags(DiagID, new clang::DiagnosticOptions(),
                                 DiagsBuffer);

  bool Success;
  const char *Binary = Argv0;
  const std::unique_ptr<clang::driver::Driver> Driver(new clang::driver::Driver(
      Binary, llvm::sys::getDefaultTargetTriple(), Diags));
  ArgumentList Argv;
  Argv.push_back(Binary);
  for (const auto &Filename : Filenames)
    Argv.push_back(Filename);

  if (FOpenMP)
    Argv.push_back("-fopenmp");
  if (TargetTripleOpt != "") {
    Argv.push_back("-target");
    Argv.push_back(TargetTripleOpt);
  }
  if (McpuOpt != "")
    Argv.emplace_back("-mcpu=", McpuOpt);
  if (Standard != "")
    Argv.emplace_back("-std=", Standard);
  if (ResourceDir != "") {
    Argv.push_back("-resource-dir");
    Argv.push_back(ResourceDir);
  }
  if (SysRoot != "") {
    Argv.push_back("--sysroot");
    Argv.push_back(SysRoot);
  }
  if (Verbose)
    Argv.push_back("-v");
  if (NoCUDAInc)
    Argv.push_back("-nocudainc");
  if (NoCUDALib)
    Argv.push_back("-nocudalib");
  if (CUDAGPUArch != "")
    Argv.emplace_back("--cuda-gpu-arch=", CUDAGPUArch);
  if (CUDAPath != "")
    Argv.emplace_back("--cuda-path=", CUDAPath);
  if (MArch != "")
    Argv.emplace_back("-march=", MArch);
  for (const auto &Dir : IncludeDirs) {
    Argv.push_back("-I");
    Argv.push_back(Dir);
  }
  for (const auto &Define : Defines)
    Argv.emplace_back("-D", Define);
  for (const auto &Include : Includes) {
    Argv.push_back("-include");
    Argv.push_back(Include);
  }

  Argv.push_back("-emit-ast");

  llvm::SmallVector<const llvm::opt::ArgStringList *, 4> CommandList;
  llvm::opt::ArgStringList InputCommandArgList;
  std::unique_ptr<clang::driver::Compilation> Compilation;

  if (InputCommandArgs.empty()) {
    Compilation.reset(Driver->BuildCompilation(Argv.getArguments()));

    clang::driver::JobList &Jobs = Compilation->getJobs();
    if (Jobs.size() < 1)
      return false;

    for (auto &Job : Jobs) {
      auto *Cmd = cast<clang::driver::Command>(&Job);
      if (strcmp(Cmd->getCreator().getName(), "clang"))
        return false;

      CommandList.push_back(&Cmd->getArguments());
    }
  } else {
    for (std::string &S : InputCommandArgs)
      InputCommandArgList.push_back(S.c_str());

    CommandList.push_back(&InputCommandArgList);
  }

  MLIRAction Act(Fn, Module,
                 Filenames.size() == 1 ? Filenames[0] : "LLVMDialectModule");

  for (const llvm::opt::ArgStringList *Args : CommandList) {
    std::unique_ptr<clang::CompilerInstance> Clang(
        new clang::CompilerInstance());

    Success = clang::CompilerInvocation::CreateFromArgs(Clang->getInvocation(),
                                                        *Args, Diags);
    Clang->getInvocation().getFrontendOpts().DisableFree = false;

    // Infer the builtin include path if unspecified.
    if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
        Clang->getHeaderSearchOpts().ResourceDir.size() == 0) {
      extern std::string GetExecutablePath(const char *, bool);
      Clang->getHeaderSearchOpts().ResourceDir =
          clang::CompilerInvocation::GetResourcesPath(
              Argv0, (void *)(intptr_t)GetExecutablePath);
    }

    Clang->getInvocation().getFrontendOpts().DisableFree = false;

    // Create the actual diagnostics engine.
    Clang->createDiagnostics();
    if (!Clang->hasDiagnostics())
      return false;

    DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
    if (!Success)
      return false;

    // Create and execute the frontend action.

    // Create the target instance.
    Clang->setTarget(clang::TargetInfo::CreateTargetInfo(
        Clang->getDiagnostics(), Clang->getInvocation().TargetOpts));
    if (!Clang->hasTarget())
      return false;

    // Create TargetInfo for the other side of CUDA and OpenMP compilation.
    if ((Clang->getLangOpts().CUDA || Clang->getLangOpts().OpenMPIsDevice ||
         Clang->getLangOpts().SYCLIsDevice) &&
        !Clang->getFrontendOpts().AuxTriple.empty()) {
      auto TO = std::make_shared<clang::TargetOptions>();
      TO->Triple = llvm::Triple::normalize(Clang->getFrontendOpts().AuxTriple);
      TO->HostTriple = Clang->getTarget().getTriple().str();
      Clang->setAuxTarget(
          clang::TargetInfo::CreateTargetInfo(Clang->getDiagnostics(), TO));
    }

    // Inform the target of the language options.
    //
    // FIXME: We shouldn't need to do this, the target should be immutable once
    // created. This complexity should be lifted elsewhere.
    Clang->getTarget().adjust(Clang->getDiagnostics(), Clang->getLangOpts());

    // Adjust target options based on codegen options.
    Clang->getTarget().adjustTargetOptions(Clang->getCodeGenOpts(),
                                           Clang->getTargetOpts());

    llvm::Triple JobTriple = Clang->getTarget().getTriple();
    if (Triple.str() == "" || !JobTriple.isNVPTX()) {
      Triple = JobTriple;
      Module.get()->setAttr(
          LLVM::LLVMDialect::getTargetTripleAttrName(),
          StringAttr::get(Module->getContext(),
                          Clang->getTarget().getTriple().getTriple()));
      DL = llvm::DataLayout(Clang->getTarget().getDataLayoutString());
      Module.get()->setAttr(
          LLVM::LLVMDialect::getDataLayoutAttrName(),
          StringAttr::get(Module->getContext(),
                          Clang->getTarget().getDataLayoutString()));

      Module.get()->setAttr(("dlti." + DataLayoutSpecAttr::kAttrKeyword).str(),
                            translateDataLayout(DL, Module->getContext()));
    }

    for (const auto &FIF : Clang->getFrontendOpts().Inputs) {
      // Reset the ID tables if we are reusing the SourceManager and parsing
      // regular files.
      if (Clang->hasSourceManager() && !Act.isModelParsingAction())
        Clang->getSourceManager().clearIDTables();
      if (Act.BeginSourceFile(*Clang, FIF)) {
        llvm::Error Err = Act.Execute();
        if (Err) {
          llvm::errs() << "saw error: " << Err << "\n";
          return false;
        }
        assert(Clang->hasSourceManager());

        Act.EndSourceFile();
      }
    }
  }
  return true;
}
