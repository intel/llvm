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
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "cgeist"

using namespace clang;
using namespace llvm;
using namespace clang::driver;
using namespace llvm::opt;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::func;
using namespace mlir::sycl;
using namespace mlirclang;

cl::opt<std::string> PrefixABI("prefix-abi", cl::init(""),
                               cl::desc("Prefix for emitted symbols"));

cl::opt<bool> GenerateAllSYCLFuncs("gen-all-sycl-funcs", cl::init(false),
                                   cl::desc("Generate all SYCL functions"));

constexpr llvm::StringLiteral MLIRASTConsumer::DeviceModuleName;

/******************************************************************************/
/*                               MLIRScanner                                  */
/******************************************************************************/

MLIRScanner::MLIRScanner(MLIRASTConsumer &Glob,
                         mlir::OwningOpRef<mlir::ModuleOp> &module,
                         LowerToInfo &LTInfo)
    : Glob(Glob), function(), module(module), builder(module->getContext()),
      loc(builder.getUnknownLoc()), entryBlock(nullptr), loops(),
      allocationScope(nullptr), supportedFuncs(), bufs(), constants(), labels(),
      EmittingFunctionDecl(nullptr), params(), Captures(), CaptureKinds(),
      ThisCapture(nullptr), arrayinit(), ThisVal(), returnVal(),
      LTInfo(LTInfo) {}

void MLIRScanner::initSupportedFunctions() {
  // Functions needed for single_task with one dimensional write buffer.

  // SYCL constructors:
  supportedFuncs.insert("_ZN4sycl3_V16detail18AccessorImplDeviceILi1EEC1ENS0_"
                        "2idILi1EEENS0_5rangeILi1EEES7_");
  supportedFuncs.insert(
      "_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_"
      "3ext6oneapi22accessor_property_listIJEEEEC1Ev");
  supportedFuncs.insert("_ZN4sycl3_V16detail5arrayILi1EEC1ILi1EEENSt9enable_"
                        "ifIXeqT_Li1EEmE4typeE");
  supportedFuncs.insert("_ZN4sycl3_V16detail5arrayILi1EEC1ERKS3_");
  supportedFuncs.insert("_ZN4sycl3_V12idILi1EEC1Ev");
  supportedFuncs.insert("_ZN4sycl3_V12idILi1EEC1ERKS2_");
  supportedFuncs.insert(
      "_ZN4sycl3_V12idILi1EEC1ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE");
  supportedFuncs.insert("_ZN4sycl3_V15rangeILi1EEC1ERKS2_");
  supportedFuncs.insert(
      "_ZN4sycl3_V15rangeILi1EEC1ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE");

  // Other SYCL functions:
  supportedFuncs.insert(
      "_ZN4sycl3_V13ext6oneapi22accessor_property_listIJEE12has_propertyINS2_"
      "8property9no_offsetEEEbPNSt9enable_ifIXsr24is_compile_time_propertyIT_"
      "EE5valueEvE4typeE");
  supportedFuncs.insert(
      "_ZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEEixILi1EvEERiNS0_2idILi1EEE");
  supportedFuncs.insert(
      "_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEE");
  supportedFuncs.insert(
      "_ZZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE6__initEPU3AS1iNS0_5rangeILi1EEESE_NS0_2idILi1EEEENKUlmE_"
      "clEm");
  supportedFuncs.insert(
      "_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE14getAccessRangeEv");
  supportedFuncs.insert(
      "_ZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE14getLinearIndexILi1EEEmNS0_2idIXT_EEE");
  supportedFuncs.insert(
      "_ZZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE14getLinearIndexILi1EEEmNS0_2idIXT_EEEENKUlmE_clEm");
  supportedFuncs.insert(
      "_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE14getMemoryRangeEv");
  supportedFuncs.insert(
      "_ZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE14getMemoryRangeEv");
  supportedFuncs.insert(
      "_ZN4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE9getOffsetEv");
  supportedFuncs.insert(
      "_ZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE14getTotalOffsetEv");
  supportedFuncs.insert(
      "_ZZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE14getTotalOffsetEvENKUlmE_clEm");
  supportedFuncs.insert(
      "_ZNK4sycl3_V18accessorIiLi1ELNS0_6access4modeE1025ELNS2_"
      "6targetE2014ELNS2_11placeholderE0ENS0_3ext6oneapi22accessor_property_"
      "listIJEEEE15getQualifiedPtrEv");
  supportedFuncs.insert("_ZN4sycl3_V16detail5arrayILi1EEixEi");
  supportedFuncs.insert("_ZNK4sycl3_V16detail5arrayILi1EEixEi");
  supportedFuncs.insert("_ZNK4sycl3_V16detail5arrayILi1EE15check_dimensionEi");
  supportedFuncs.insert("_ZN4sycl3_V16detail14InitializedValILi1ENS0_"
                        "5rangeEE3getILi0EEENS3_ILi1EEEv");
  supportedFuncs.insert(
      "_ZN4sycl3_V16detail8dim_loopILm1EZNS0_8accessorIiLi1ELNS0_"
      "6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_"
      "3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1iNS0_"
      "5rangeILi1EEESG_NS0_2idILi1EEEEUlmE_EEvOT0_");
  supportedFuncs.insert(
      "_ZN4sycl3_V16detail8dim_loopILm1EZNKS0_8accessorIiLi1ELNS0_"
      "6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_"
      "3ext6oneapi22accessor_property_listIJEEEE14getLinearIndexILi1EEEmNS0_"
      "2idIXT_EEEEUlmE_EEvOT0_");
  supportedFuncs.insert(
      "_ZN4sycl3_V16detail8dim_loopILm1EZNKS0_8accessorIiLi1ELNS0_"
      "6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_"
      "3ext6oneapi22accessor_property_listIJEEEE14getTotalOffsetEvEUlmE_"
      "EEvOT0_");
  supportedFuncs.insert(
      "_ZN4sycl3_V16detail13dim_loop_implIJLm0EEZNS0_8accessorIiLi1ELNS0_"
      "6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_"
      "3ext6oneapi22accessor_property_listIJEEEE6__initEPU3AS1iNS0_"
      "5rangeILi1EEESG_NS0_2idILi1EEEEUlmE_EEvSt16integer_sequenceImJXspT_"
      "EEEOT0_");
  supportedFuncs.insert(
      "_ZN4sycl3_V16detail13dim_loop_implIJLm0EEZNKS0_8accessorIiLi1ELNS0_"
      "6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_"
      "3ext6oneapi22accessor_property_listIJEEEE14getLinearIndexILi1EEEmNS0_"
      "2idIXT_EEEEUlmE_EEvSt16integer_sequenceImJXspT_EEEOT0_");
  supportedFuncs.insert(
      "_ZN4sycl3_V16detail13dim_loop_implIJLm0EEZNKS0_8accessorIiLi1ELNS0_"
      "6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_"
      "3ext6oneapi22accessor_property_listIJEEEE14getTotalOffsetEvEUlmE_"
      "EEvSt16integer_sequenceImJXspT_EEEOT0_");
}

static void checkFunctionParent(const FunctionOpInterface F,
                                FunctionContext Context,
                                const OwningOpRef<ModuleOp> &module) {
  assert(
      (Context != FunctionContext::Host || F->getParentOp() == module.get()) &&
      "New function must be inserted into global module");
  assert((Context != FunctionContext::SYCLDevice ||
          F->getParentOfType<gpu::GPUModuleOp>() == getDeviceModule(*module)) &&
         "New device function must be inserted into device module");
}

void MLIRScanner::init(mlir::FunctionOpInterface func,
                       const FunctionToEmit &FTE) {
  const clang::FunctionDecl *FD = &FTE.getDecl();

  function = func;
  EmittingFunctionDecl = FD;

  if (ShowAST)
    llvm::dbgs() << "Emitting fn: " << function.getName() << "\n"
                 << "\tfunctionDecl:" << *FD << "\n"
                 << "function:" << function << "\n";

  initSupportedFunctions();
  // This is needed, as GPUFuncOps are already created with an entry block.
  setEntryAndAllocBlock(isa<gpu::GPUFuncOp>(function)
                            ? &function.getBlocks().front()
                            : function.addEntryBlock());

  unsigned i = 0;
  if (auto CM = dyn_cast<CXXMethodDecl>(FD)) {
    if (CM->getParent()->isLambda()) {
      for (auto C : CM->getParent()->captures()) {
        if (C.capturesVariable())
          CaptureKinds[C.getCapturedVar()] = C.getCaptureKind();
      }

      CM->getParent()->getCaptureFields(Captures, ThisCapture);
      if (ThisCapture) {
        llvm::errs() << " thiscapture:\n";
        ThisCapture->dump();
      }
    }

    if (CM->isInstance()) {
      mlir::Value val = function.getArgument(i);
      ThisVal = ValueCategory(val, /*isReference*/ false);
      i++;
    }
  }

  const clang::CodeGen::CGFunctionInfo &FI = Glob.GetOrCreateCGFunctionInfo(FD);
  auto FIArgs = FI.arguments();

  for (ParmVarDecl *parm : FD->parameters()) {
    assert(i != function.getNumArguments());

    QualType parmType = parm->getType();

    bool LLVMABI = false, isArray = false;
    if (Glob.getTypes()
            .getMLIRType(Glob.getCGM().getContext().getPointerType(parmType))
            .isa<mlir::LLVM::LLVMPointerType>())
      LLVMABI = true;
    else
      Glob.getTypes().getMLIRType(parmType, &isArray);

    bool isReference = isArray || isa<clang::ReferenceType>(
                                      parmType->getUnqualifiedDesugaredType());
    isReference |=
        (FIArgs[i].info.getKind() == clang::CodeGen::ABIArgInfo::Indirect ||
         FIArgs[i].info.getKind() ==
             clang::CodeGen::ABIArgInfo::IndirectAliased);

    mlir::Value val = function.getArgument(i);
    assert(val && "Expecting a valid value");

    if (isReference)
      params.emplace(parm, ValueCategory(val, /*isReference*/ true));
    else {
      mlir::Value alloc =
          createAllocOp(val.getType(), parm, /*memspace*/ 0, isArray, LLVMABI);
      ValueCategory(alloc, /*isReference*/ true).store(builder, val);
    }

    i++;
  }

  if (FD->hasAttr<CUDAGlobalAttr>() && Glob.getCGM().getLangOpts().CUDA &&
      !Glob.getCGM().getLangOpts().CUDAIsDevice) {
    FunctionToEmit FTE(*FD);
    auto deviceStub = cast<func::FuncOp>(
        Glob.GetOrCreateMLIRFunction(FTE, true /* ShouldEmit*/,
                                     /* getDeviceStub */ true));

    builder.create<func::CallOp>(loc, deviceStub, function.getArguments());
    builder.create<ReturnOp>(loc);
    return;
  }

  if (auto CC = dyn_cast<CXXConstructorDecl>(FD)) {
    const CXXRecordDecl *ClassDecl = CC->getParent();
    for (auto expr : CC->inits()) {
      if (ShowAST) {
        llvm::errs() << " init: - baseInit:" << (int)expr->isBaseInitializer()
                     << " memberInit:" << (int)expr->isMemberInitializer()
                     << " anyMember:" << (int)expr->isAnyMemberInitializer()
                     << " indirectMember:"
                     << (int)expr->isIndirectMemberInitializer()
                     << " isinClass:" << (int)expr->isInClassMemberInitializer()
                     << " delegating:" << (int)expr->isDelegatingInitializer()
                     << " isPack:" << (int)expr->isPackExpansion() << "\n";
        if (expr->getMember())
          expr->getMember()->dump();
        if (expr->getInit())
          expr->getInit()->dump();
      }
      assert(ThisVal.val);

      FieldDecl *field = expr->getMember();
      if (!field) {
        if (expr->isBaseInitializer()) {
          bool BaseIsVirtual = expr->isBaseVirtual();
          auto BaseType = expr->getBaseClass();

          // Shift and cast down to the base type.
          // TODO: for complete types, this should be possible with a GEP.
          mlir::Value V = ThisVal.val;
          const clang::Type *BaseTypes[] = {BaseType};
          bool BaseVirtual[] = {BaseIsVirtual};

          V = GetAddressOfBaseClass(V, /*derived*/ ClassDecl, BaseTypes,
                                    BaseVirtual);

          Expr *init = expr->getInit();
          if (auto clean = dyn_cast<ExprWithCleanups>(init)) {
            llvm::errs() << "TODO: cleanup\n";
            init = clean->getSubExpr();
          }

          VisitConstructCommon(cast<clang::CXXConstructExpr>(init),
                               /*name*/ nullptr, /*space*/ 0, /*mem*/ V);
          continue;
        }

        if (expr->isDelegatingInitializer()) {
          Expr *init = expr->getInit();
          if (auto clean = dyn_cast<ExprWithCleanups>(init)) {
            llvm::errs() << "TODO: cleanup\n";
            init = clean->getSubExpr();
          }

          VisitConstructCommon(cast<clang::CXXConstructExpr>(init),
                               /*name*/ nullptr, /*space*/ 0,
                               /*mem*/ ThisVal.val);
          continue;
        }
      }
      assert(field && "initialiation expression must apply to a field");

      if (auto AILE = dyn_cast<ArrayInitLoopExpr>(expr->getInit())) {
        VisitArrayInitLoop(AILE,
                           CommonFieldLookup(CC->getThisObjectType(), field,
                                             ThisVal.val, /*isLValue*/ false));
        continue;
      }

      if (auto cons = dyn_cast<CXXConstructExpr>(expr->getInit())) {
        VisitConstructCommon(cons, /*name*/ nullptr, /*space*/ 0,
                             CommonFieldLookup(CC->getThisObjectType(), field,
                                               ThisVal.val, /*isLValue*/ false)
                                 .val);
        continue;
      }

      auto initexpr = Visit(expr->getInit());
      if (!initexpr.val) {
        expr->getInit()->dump();
        assert(initexpr.val);
      }

      bool isArray = false;
      Glob.getTypes().getMLIRType(expr->getInit()->getType(), &isArray);

      auto cfl = CommonFieldLookup(CC->getThisObjectType(), field, ThisVal.val,
                                   /*isLValue*/ false);
      assert(cfl.val);
      cfl.store(builder, initexpr, isArray);
    }
  }

  if (auto CC = dyn_cast<CXXDestructorDecl>(FD)) {
    CC->dump();
    llvm::errs() << " warning, destructor not fully handled yet\n";
  }

  auto i1Ty = builder.getIntegerType(1);
  auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
  auto truev = builder.create<ConstantIntOp>(loc, true, 1);
  loops.push_back({builder.create<mlir::memref::AllocaOp>(loc, type),
                   builder.create<mlir::memref::AllocaOp>(loc, type)});
  builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);
  builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().keepRunning);

  if (function.getResultTypes().size()) {
    auto type = mlir::MemRefType::get({}, function.getResultTypes()[0], {}, 0);
    returnVal = builder.create<mlir::memref::AllocaOp>(loc, type);
    if (type.getElementType().isa<mlir::IntegerType, mlir::FloatType>()) {
      builder.create<mlir::memref::StoreOp>(
          loc, builder.create<mlir::LLVM::UndefOp>(loc, type.getElementType()),
          returnVal, std::vector<mlir::Value>({}));
    }
  }

  if (auto D = dyn_cast<CXXMethodDecl>(FD)) {
    // ClangAST incorrectly does not contain the correct definition
    // of a union move operation and as such we _must_ emit a memcpy
    // for a defaulted union copy or move.
    if (D->getParent()->isUnion() && D->isDefaulted()) {
      mlir::Value V = ThisVal.val;
      assert(V);
      if (auto MT = V.getType().dyn_cast<MemRefType>())
        V = builder.create<polygeist::Pointer2MemrefOp>(
            loc, LLVM::LLVMPointerType::get(MT.getElementType()), V);

      mlir::Value src = function.getArgument(1);
      if (auto MT = src.getType().dyn_cast<MemRefType>())
        src = builder.create<polygeist::Pointer2MemrefOp>(
            loc, LLVM::LLVMPointerType::get(MT.getElementType()), src);

      mlir::Value typeSize = builder.create<polygeist::TypeSizeOp>(
          loc, builder.getIndexType(),
          mlir::TypeAttr::get(
              V.getType().cast<LLVM::LLVMPointerType>().getElementType()));
      typeSize = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                                    typeSize);
      V = builder.create<LLVM::BitcastOp>(
          loc,
          LLVM::LLVMPointerType::get(
              builder.getI8Type(),
              V.getType().cast<LLVM::LLVMPointerType>().getAddressSpace()),
          V);
      src = builder.create<LLVM::BitcastOp>(
          loc,
          LLVM::LLVMPointerType::get(
              builder.getI8Type(),
              src.getType().cast<LLVM::LLVMPointerType>().getAddressSpace()),
          src);
      mlir::Value volatileCpy = builder.create<ConstantIntOp>(loc, false, 1);
      builder.create<LLVM::MemcpyOp>(loc, V, src, typeSize, volatileCpy);
    }
  }

  Stmt *stmt = FD->getBody();
  assert(stmt);
  if (ShowAST)
    stmt->dump();

  Visit(stmt);

  if (function.getResultTypes().size()) {
    assert(!isa<gpu::GPUFuncOp>(function) &&
           "SYCL kernel functions must always have a void return type.");
    mlir::Value vals[1] = {
        builder.create<mlir::memref::LoadOp>(loc, returnVal)};
    builder.create<ReturnOp>(loc, vals);
  } else if (isa<gpu::GPUFuncOp>(function))
    builder.create<gpu::ReturnOp>(loc);
  else
    builder.create<ReturnOp>(loc);

  checkFunctionParent(function, FTE.getContext(), module);
}

mlir::Value MLIRScanner::createAllocOp(mlir::Type t, VarDecl *name,
                                       uint64_t memspace, bool isArray = false,
                                       bool LLVMABI = false) {
  mlir::MemRefType mr;
  mlir::Value alloc = nullptr;
  OpBuilder abuilder(builder.getContext());
  abuilder.setInsertionPointToStart(allocationScope);
  auto varLoc = name ? getMLIRLocation(name->getBeginLoc()) : loc;
  if (!isArray) {
    if (LLVMABI) {
      if (name)
        if (auto var = dyn_cast<VariableArrayType>(
                name->getType()->getUnqualifiedDesugaredType())) {
          auto len = Visit(var->getSizeExpr()).getValue(builder);
          alloc = builder.create<mlir::LLVM::AllocaOp>(
              varLoc, LLVM::LLVMPointerType::get(t, memspace), len);
          builder.create<polygeist::TrivialUseOp>(varLoc, alloc);
          alloc = builder.create<mlir::LLVM::BitcastOp>(
              varLoc,
              LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(t, 0)),
              alloc);
        }

      if (!alloc) {
        alloc = abuilder.create<mlir::LLVM::AllocaOp>(
            varLoc, mlir::LLVM::LLVMPointerType::get(t, memspace),
            abuilder.create<arith::ConstantIntOp>(varLoc, 1, 64), 0);
        if (t.isa<mlir::IntegerType, mlir::FloatType>()) {
          abuilder.create<LLVM::StoreOp>(
              varLoc, abuilder.create<mlir::LLVM::UndefOp>(varLoc, t), alloc);
        }
        // alloc = builder.create<mlir::LLVM::BitcastOp>(varLoc,
        // LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(t, 1)), alloc);
      }
    } else {
      mr = mlir::MemRefType::get(1, t, {}, memspace);
      alloc = abuilder.create<mlir::memref::AllocaOp>(varLoc, mr);
      LLVM_DEBUG({
        llvm::dbgs() << "MLIRScanner::createAllocOp: created: ";
        alloc.dump();
      });

      if (memspace != 0) {
        // Note: this code is incorrect because 'alloc' has a MemRefType in
        // memory space that is not zero, therefore is illegal to create a
        // Memref2Pointer operation that yields a result not in the same memory
        // space.
        auto memRefToPtr = abuilder.create<polygeist::Memref2PointerOp>(
            varLoc, LLVM::LLVMPointerType::get(t, 0), alloc);
        alloc = abuilder.create<polygeist::Pointer2MemrefOp>(
            varLoc, mlir::MemRefType::get(-1, t, {}, memspace), memRefToPtr);
      }
      alloc = abuilder.create<mlir::memref::CastOp>(
          varLoc, mlir::MemRefType::get(-1, t, {}, 0), alloc);
      LLVM_DEBUG({
        llvm::dbgs() << "MLIRScanner::createAllocOp: created: ";
        alloc.dump();
        llvm::dbgs() << "\n";
      });

      if (t.isa<mlir::IntegerType, mlir::FloatType>()) {
        mlir::Value idxs[] = {abuilder.create<ConstantIndexOp>(loc, 0)};
        abuilder.create<mlir::memref::StoreOp>(
            varLoc, abuilder.create<mlir::LLVM::UndefOp>(varLoc, t), alloc,
            idxs);
      }
    }
  } else {
    auto mt = t.cast<mlir::MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());
    auto pshape = shape[0];

    if (name)
      if (auto var = dyn_cast<VariableArrayType>(
              name->getType()->getUnqualifiedDesugaredType())) {
        assert(shape[0] == -1);
        mr = mlir::MemRefType::get(
            shape, mt.getElementType(), MemRefLayoutAttrInterface(),
            wrapIntegerMemorySpace(memspace, mt.getContext()));
        auto len = Visit(var->getSizeExpr()).getValue(builder);
        len = builder.create<IndexCastOp>(varLoc, builder.getIndexType(), len);
        alloc = builder.create<mlir::memref::AllocaOp>(varLoc, mr, len);
        builder.create<polygeist::TrivialUseOp>(varLoc, alloc);
        if (memspace != 0) {
          alloc = abuilder.create<polygeist::Pointer2MemrefOp>(
              varLoc, mlir::MemRefType::get(shape, mt.getElementType()),
              abuilder.create<polygeist::Memref2PointerOp>(
                  varLoc, LLVM::LLVMPointerType::get(mt.getElementType(), 0),
                  alloc));
        }
      }

    if (!alloc) {
      if (pshape == -1)
        shape[0] = 1;
      mr = mlir::MemRefType::get(
          shape, mt.getElementType(), MemRefLayoutAttrInterface(),
          wrapIntegerMemorySpace(memspace, mt.getContext()));
      alloc = abuilder.create<mlir::memref::AllocaOp>(varLoc, mr);
      if (memspace != 0) {
        alloc = abuilder.create<polygeist::Pointer2MemrefOp>(
            varLoc, mlir::MemRefType::get(shape, mt.getElementType()),
            abuilder.create<polygeist::Memref2PointerOp>(
                varLoc, LLVM::LLVMPointerType::get(mt.getElementType(), 0),
                alloc));
      }
      shape[0] = pshape;
      alloc = abuilder.create<mlir::memref::CastOp>(
          varLoc, mlir::MemRefType::get(shape, mt.getElementType()), alloc);
    }
  }
  assert(alloc);
  // NamedAttribute attrs[] = {NamedAttribute("name", name)};
  if (name) {
    // if (name->getName() == "i")
    //  assert(0 && " not i");
    if (params.find(name) != params.end()) {
      name->dump();
    }
    assert(params.find(name) == params.end());
    params[name] = ValueCategory(alloc, /*isReference*/ true);
  }
  return alloc;
}

ValueCategory MLIRScanner::VisitVarDecl(clang::VarDecl *decl) {
  decl = decl->getCanonicalDecl();
  mlir::Type subType = Glob.getTypes().getMLIRType(decl->getType());
  ValueCategory inite = nullptr;
  unsigned memtype = decl->hasAttr<CUDASharedAttr>() ? 5 : 0;
  bool LLVMABI = false;
  bool isArray = false;

  if (Glob.getTypes()
          .getMLIRType(Glob.getCGM().getContext().getLValueReferenceType(
              decl->getType()))
          .isa<mlir::LLVM::LLVMPointerType>())
    LLVMABI = true;
  else
    Glob.getTypes().getMLIRType(decl->getType(), &isArray);

  if (!LLVMABI && isArray) {
    subType = Glob.getTypes().getMLIRType(
        Glob.getCGM().getContext().getLValueReferenceType(decl->getType()));
  }

  mlir::LLVM::TypeFromLLVMIRTranslator typeTranslator(*module->getContext());

  if (auto init = decl->getInit()) {
    if (!isa<InitListExpr>(init) && !isa<CXXConstructExpr>(init)) {
      auto visit = Visit(init);
      if (!visit.val) {
        decl->dump();
        assert(visit.val);
      }
      bool isReference = init->isLValue() || init->isXValue();
      if (isReference) {
        assert(visit.isReference);
        builder.create<polygeist::TrivialUseOp>(loc, visit.val);
        return params[decl] = visit;
      }
      if (isArray) {
        assert(visit.isReference);
        inite = visit;
      } else {
        inite = ValueCategory(visit.getValue(builder), /*isRef*/ false);
        if (!inite.val) {
          init->dump();
          assert(0 && inite.val);
        }
        subType = inite.val.getType();
      }
    }
  } else if (auto ava = decl->getAttr<AlignValueAttr>()) {
    if (auto algn = dyn_cast<clang::ConstantExpr>(ava->getAlignment())) {
      for (auto a : algn->children()) {
        if (auto IL = dyn_cast<IntegerLiteral>(a)) {
          if (IL->getValue() == 8192) {
            llvm::Type *T =
                Glob.getCGM().getTypes().ConvertType(decl->getType());
            subType = typeTranslator.translateType(T);
            LLVMABI = true;
            break;
          }
        }
      }
    }
  } else if (auto ava = decl->getAttr<InitPriorityAttr>()) {
    if (ava->getPriority() == 8192) {
      llvm::Type *T = Glob.getCGM().getTypes().ConvertType(decl->getType());
      subType = typeTranslator.translateType(T);
      LLVMABI = true;
    }
  }

  mlir::Value op;

  Block *block = nullptr;
  Block::iterator iter;

  if (decl->isStaticLocal() && memtype == 0) {
    OpBuilder abuilder(builder.getContext());
    abuilder.setInsertionPointToStart(allocationScope);
    auto varLoc = getMLIRLocation(decl->getBeginLoc());

    if (Glob.getTypes()
            .getMLIRType(
                Glob.getCGM().getContext().getPointerType(decl->getType()))
            .isa<mlir::LLVM::LLVMPointerType>()) {
      op = abuilder.create<mlir::LLVM::AddressOfOp>(
          varLoc, Glob.GetOrCreateLLVMGlobal(
                      decl, (function.getName() + "@static@").str()));
    } else {
      auto gv =
          Glob.getOrCreateGlobal(*decl, (function.getName() + "@static@").str(),
                                 FunctionContext::Host);
      auto gv2 = abuilder.create<memref::GetGlobalOp>(
          varLoc, gv.first.getType(), gv.first.getName());
      op = reshapeRanklessGlobal(gv2);
    }
    params[decl] = ValueCategory(op, /*isReference*/ true);
    if (decl->getInit()) {
      auto mr = MemRefType::get({}, builder.getI1Type());
      bool inits[1] = {true};
      auto rtt = RankedTensorType::get({}, builder.getI1Type());
      auto init_value = DenseIntElementsAttr::get(rtt, inits);
      OpBuilder gbuilder(builder.getContext());
      gbuilder.setInsertionPointToStart(module->getBody());
      auto name = Glob.getCGM().getMangledName(decl);
      auto globalOp = gbuilder.create<mlir::memref::GlobalOp>(
          module->getLoc(),
          builder.getStringAttr(function.getName() + "@static@" + name +
                                "@init"),
          /*sym_visibility*/ mlir::StringAttr(), mlir::TypeAttr::get(mr),
          init_value, mlir::UnitAttr(), /*alignment*/ nullptr);
      SymbolTable::setSymbolVisibility(globalOp,
                                       mlir::SymbolTable::Visibility::Private);

      auto boolop =
          builder.create<memref::GetGlobalOp>(varLoc, mr, globalOp.getName());
      mlir::Value V = reshapeRanklessGlobal(boolop);
      auto cond = builder.create<memref::LoadOp>(
          varLoc, V, std::vector<mlir::Value>({getConstantIndex(0)}));

      auto ifOp = builder.create<scf::IfOp>(varLoc, cond, /*hasElse*/ false);
      block = builder.getInsertionBlock();
      iter = builder.getInsertionPoint();
      builder.setInsertionPointToStart(&ifOp.getThenRegion().back());
      builder.create<memref::StoreOp>(
          varLoc, builder.create<ConstantIntOp>(varLoc, false, 1), V,
          std::vector<mlir::Value>({getConstantIndex(0)}));
    }
  } else
    op = createAllocOp(subType, decl, memtype, isArray, LLVMABI);

  if (inite.val) {
    ValueCategory(op, /*isReference*/ true).store(builder, inite, isArray);
  } else if (auto init = decl->getInit()) {
    if (isa<InitListExpr>(init)) {
      InitializeValueByInitListExpr(op, init);
    } else if (auto CE = dyn_cast<CXXConstructExpr>(init)) {
      VisitConstructCommon(CE, decl, memtype, op);
    } else
      assert(0 && "unknown init list");
  }
  if (block)
    builder.setInsertionPoint(block, iter);
  return ValueCategory(op, /*isReference*/ true);
}

mlir::Value add(MLIRScanner &sc, mlir::OpBuilder &builder, mlir::Location loc,
                mlir::Value lhs, mlir::Value rhs) {
  assert(lhs);
  assert(rhs);
  if (auto op = lhs.getDefiningOp<ConstantIntOp>()) {
    if (op.value() == 0) {
      return rhs;
    }
  }

  if (auto op = lhs.getDefiningOp<ConstantIndexOp>()) {
    if (op.value() == 0) {
      return rhs;
    }
  }

  if (auto op = rhs.getDefiningOp<ConstantIntOp>()) {
    if (op.value() == 0) {
      return lhs;
    }
  }

  if (auto op = rhs.getDefiningOp<ConstantIndexOp>()) {
    if (op.value() == 0) {
      return lhs;
    }
  }
  return builder.create<AddIOp>(loc, lhs, rhs);
}

mlir::Value MLIRScanner::castToIndex(mlir::Location loc, mlir::Value val) {
  assert(val && "Expect non-null value");

  if (auto op = val.getDefiningOp<ConstantIntOp>())
    return getConstantIndex(op.value());

  return builder.create<arith::IndexCastOp>(
      loc, mlir::IndexType::get(val.getContext()), val);
}

mlir::Value MLIRScanner::castToMemSpace(mlir::Value val, unsigned memSpace) {
  assert(val && "Expect non-null value");

  return mlir::TypeSwitch<mlir::Type, mlir::Value>(val.getType())
      .Case<mlir::MemRefType>([&](mlir::MemRefType valType) -> mlir::Value {
        if (valType.getMemorySpaceAsInt() == memSpace)
          return val;

        mlir::Value newVal = builder.create<polygeist::Memref2PointerOp>(
            loc,
            LLVM::LLVMPointerType::get(valType.getElementType(),
                                       valType.getMemorySpaceAsInt()),
            val);
        newVal = builder.create<LLVM::AddrSpaceCastOp>(
            loc, LLVM::LLVMPointerType::get(valType.getElementType(), memSpace),
            newVal);
        return builder.create<polygeist::Pointer2MemrefOp>(
            loc,
            mlir::MemRefType::get(
                valType.getShape(), valType.getElementType(),
                MemRefLayoutAttrInterface(),
                wrapIntegerMemorySpace(memSpace, valType.getContext())),
            newVal);
      })
      .Case<LLVM::LLVMPointerType>(
          [&](LLVM::LLVMPointerType valType) -> mlir::Value {
            if (valType.getAddressSpace() == memSpace)
              return val;

            return builder.create<LLVM::AddrSpaceCastOp>(
                loc,
                LLVM::LLVMPointerType::get(valType.getElementType(), memSpace),
                val);
          })
      .Default([&](mlir::Type valType) {
        llvm_unreachable("unimplemented");
        return val;
      });
}

mlir::Value MLIRScanner::castToMemSpaceOfType(mlir::Value val, mlir::Type t) {
  assert((t.isa<mlir::MemRefType>() || t.isa<LLVM::LLVMPointerType>()) &&
         "Unexpected type");
  unsigned memSpace = t.isa<mlir::MemRefType>()
                          ? t.cast<mlir::MemRefType>().getMemorySpaceAsInt()
                          : t.cast<LLVM::LLVMPointerType>().getAddressSpace();
  return castToMemSpace(val, memSpace);
}

ValueCategory MLIRScanner::CommonArrayToPointer(ValueCategory scalar) {
  assert(scalar.val);
  assert(scalar.isReference);
  if (auto PT = scalar.val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    if (PT.getElementType().isa<mlir::LLVM::LLVMPointerType>())
      return ValueCategory(scalar.val, /*isRef*/ false);
    mlir::Value vec[2] = {builder.create<ConstantIntOp>(loc, 0, 32),
                          builder.create<ConstantIntOp>(loc, 0, 32)};
    if (!PT.getElementType().isa<mlir::LLVM::LLVMArrayType>()) {
      EmittingFunctionDecl->dump();
      function.dump();
      llvm::errs() << " sval: " << scalar.val << "\n";
      llvm::errs() << PT << "\n";
    }
    auto ET =
        PT.getElementType().cast<mlir::LLVM::LLVMArrayType>().getElementType();
    return ValueCategory(
        builder.create<mlir::LLVM::GEPOp>(
            loc, mlir::LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
            scalar.val, vec),
        /*isReference*/ false);
  }

  auto mt = scalar.val.getType().cast<MemRefType>();
  auto shape = std::vector<int64_t>(mt.getShape());
  // if (shape.size() > 1) {
  //  shape.erase(shape.begin());
  //} else {
  shape[0] = -1;
  //}
  auto mt0 =
      mlir::MemRefType::get(shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(), mt.getMemorySpace());

  auto post = builder.create<memref::CastOp>(loc, mt0, scalar.val);
  return ValueCategory(post, /*isReference*/ false);
}

ValueCategory MLIRScanner::CommonArrayLookup(ValueCategory array,
                                             mlir::Value idx,
                                             bool isImplicitRefResult,
                                             bool removeIndex) {
  mlir::Value val = array.getValue(builder);
  assert(val);

  if (val.getType().isa<LLVM::LLVMPointerType>()) {

    mlir::Value vals[] = {
        builder.create<IndexCastOp>(loc, builder.getIntegerType(64), idx)};
    // TODO sub
    return ValueCategory(
        builder.create<mlir::LLVM::GEPOp>(loc, val.getType(), val, vals),
        /*isReference*/ true);
  }
  if (!val.getType().isa<MemRefType>()) {
    EmittingFunctionDecl->dump();
    builder.getInsertionBlock()->dump();
    function.dump();
    llvm::errs() << "value: " << val << "\n";
  }

  ValueCategory dref;
  {
    auto mt = val.getType().cast<MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());
    shape[0] = -1;
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    auto post = builder.create<polygeist::SubIndexOp>(loc, mt0, val, idx);
    // TODO sub
    dref = ValueCategory(post, /*isReference*/ true);
  }
  assert(dref.isReference);
  if (!removeIndex)
    return dref;

  auto mt = dref.val.getType().cast<MemRefType>();
  auto shape = std::vector<int64_t>(mt.getShape());
  if (shape.size() == 1 || (shape.size() == 2 && isImplicitRefResult)) {
    shape[0] = -1;
  } else {
    shape.erase(shape.begin());
  }
  auto mt0 =
      mlir::MemRefType::get(shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(), mt.getMemorySpace());
  auto post = builder.create<polygeist::SubIndexOp>(loc, mt0, dref.val,
                                                    getConstantIndex(0));
  return ValueCategory(post, /*isReference*/ true);
}

mlir::Value MLIRScanner::getConstantIndex(int x) {
  if (constants.find(x) != constants.end()) {
    return constants[x];
  }
  mlir::OpBuilder subbuilder(builder.getContext());
  subbuilder.setInsertionPointToStart(entryBlock);
  return constants[x] = subbuilder.create<ConstantIndexOp>(loc, x);
}

ValueCategory MLIRScanner::VisitUnaryOperator(clang::UnaryOperator *U) {
  auto loc = getMLIRLocation(U->getExprLoc());
  auto sub = Visit(U->getSubExpr());

  switch (U->getOpcode()) {
  case clang::UnaryOperator::Opcode::UO_Extension: {
    return sub;
  }
  case clang::UnaryOperator::Opcode::UO_LNot: {
    assert(sub.val);
    mlir::Value val = sub.getValue(builder);

    if (auto MT = val.getType().dyn_cast<mlir::MemRefType>()) {
      val = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(builder.getI8Type(),
                                     MT.getMemorySpaceAsInt()),
          val);
    }

    if (auto LT = val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      auto ne = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::eq, val, nullptr_llvm);
      return ValueCategory(ne, /*isReference*/ false);
    }

    if (!val.getType().isa<mlir::IntegerType>()) {
      U->dump();
      val.dump();
    }
    auto ty = val.getType().cast<mlir::IntegerType>();
    if (ty.getWidth() != 1) {
      val = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, val,
          builder.create<ConstantIntOp>(loc, 0, ty));
    }
    auto c1 = builder.create<ConstantIntOp>(loc, 1, val.getType());
    mlir::Value res = builder.create<XOrIOp>(loc, val, c1);

    auto postTy =
        Glob.getTypes().getMLIRType(U->getType()).cast<mlir::IntegerType>();
    if (postTy.getWidth() > 1)
      res = builder.create<arith::ExtUIOp>(loc, postTy, res);
    return ValueCategory(res, /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Not: {
    assert(sub.val);
    mlir::Value val = sub.getValue(builder);

    if (!val.getType().isa<mlir::IntegerType>()) {
      U->dump();
      val.dump();
    }
    auto ty = val.getType().cast<mlir::IntegerType>();
    auto c1 = builder.create<ConstantIntOp>(
        loc, APInt::getAllOnesValue(ty.getWidth()).getSExtValue(), ty);
    return ValueCategory(builder.create<XOrIOp>(loc, val, c1),
                         /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Deref: {
    auto dref = sub.dereference(builder);
    return dref;
  }
  case clang::UnaryOperator::Opcode::UO_AddrOf: {
    assert(sub.isReference);
    if (sub.val.getType().isa<mlir::LLVM::LLVMPointerType>()) {
      return ValueCategory(sub.val, /*isReference*/ false);
    }

    bool isArray = false;
    Glob.getTypes().getMLIRType(U->getSubExpr()->getType(), &isArray);
    auto mt = sub.val.getType().cast<MemRefType>();
    auto shape = std::vector<int64_t>(mt.getShape());
    mlir::Value res;
    shape[0] = -1;
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    res = builder.create<memref::CastOp>(loc, mt0, sub.val);
    return ValueCategory(res,
                         /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Plus: {
    return sub;
  }
  case clang::UnaryOperator::Opcode::UO_Minus: {
    mlir::Value val = sub.getValue(builder);
    auto ty = val.getType();
    if (auto ft = ty.dyn_cast<mlir::FloatType>()) {
      if (auto CI = val.getDefiningOp<ConstantFloatOp>()) {
        auto api = CI.getValue().cast<FloatAttr>().getValue();
        return ValueCategory(builder.create<arith::ConstantOp>(
                                 loc, ty, mlir::FloatAttr::get(ty, -api)),
                             /*isReference*/ false);
      }
      return ValueCategory(builder.create<NegFOp>(loc, val),
                           /*isReference*/ false);
    } else {
      if (auto CI = val.getDefiningOp<ConstantIntOp>()) {
        auto api = CI.getValue().cast<IntegerAttr>().getValue();
        return ValueCategory(builder.create<arith::ConstantOp>(
                                 loc, ty, mlir::IntegerAttr::get(ty, -api)),
                             /*isReference*/ false);
      }
      return ValueCategory(
          builder.create<SubIOp>(loc,
                                 builder.create<ConstantIntOp>(
                                     loc, 0, ty.cast<mlir::IntegerType>()),
                                 val),
          /*isReference*/ false);
    }
  }
  case clang::UnaryOperator::Opcode::UO_PreInc:
  case clang::UnaryOperator::Opcode::UO_PostInc: {
    assert(sub.isReference);
    auto prev = sub.getValue(builder);
    auto ty = prev.getType();

    mlir::Value next;
    if (auto ft = ty.dyn_cast<mlir::FloatType>()) {
      if (prev.getType() != ty) {
        U->dump();
        llvm::errs() << " ty: " << ty << "prev: " << prev << "\n";
      }
      assert(prev.getType() == ty);
      next = builder.create<AddFOp>(
          loc, prev,
          builder.create<ConstantFloatOp>(
              loc, APFloat(ft.getFloatSemantics(), "1"), ft));
    } else if (auto mt = ty.dyn_cast<MemRefType>()) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = -1;
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      next = builder.create<polygeist::SubIndexOp>(loc, mt0, prev,
                                                   getConstantIndex(1));
    } else if (auto pt = ty.dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto ity = mlir::IntegerType::get(builder.getContext(), 64);
      next = builder.create<LLVM::GEPOp>(
          loc, pt, prev,
          std::vector<mlir::Value>(
              {builder.create<ConstantIntOp>(loc, 1, ity)}));
    } else {
      if (!ty.isa<mlir::IntegerType>()) {
        llvm::errs() << ty << " - " << prev << "\n";
        U->dump();
      }
      if (prev.getType() != ty) {
        U->dump();
        llvm::errs() << " ty: " << ty << "prev: " << prev << "\n";
      }
      assert(prev.getType() == ty);
      next = builder.create<AddIOp>(
          loc, prev,
          builder.create<ConstantIntOp>(loc, 1, ty.cast<mlir::IntegerType>()));
    }
    sub.store(builder, next);

    if (U->getOpcode() == clang::UnaryOperator::Opcode::UO_PreInc)
      return sub;
    else
      return ValueCategory(prev, /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_PreDec:
  case clang::UnaryOperator::Opcode::UO_PostDec: {
    auto ty = Glob.getTypes().getMLIRType(U->getType());
    assert(sub.isReference);
    auto prev = sub.getValue(builder);

    mlir::Value next;
    if (auto ft = ty.dyn_cast<mlir::FloatType>()) {
      next = builder.create<SubFOp>(
          loc, prev,
          builder.create<ConstantFloatOp>(
              loc, APFloat(ft.getFloatSemantics(), "1"), ft));
    } else if (auto pt = ty.dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto ity = mlir::IntegerType::get(builder.getContext(), 64);
      next = builder.create<LLVM::GEPOp>(
          loc, pt, prev,
          std::vector<mlir::Value>(
              {builder.create<ConstantIntOp>(loc, -1, ity)}));
    } else if (auto mt = ty.dyn_cast<MemRefType>()) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = -1;
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      next = builder.create<polygeist::SubIndexOp>(loc, mt0, prev,
                                                   getConstantIndex(-1));
    } else {
      if (!ty.isa<mlir::IntegerType>()) {
        llvm::errs() << ty << " - " << prev << "\n";
        U->dump();
      }
      next = builder.create<SubIOp>(
          loc, prev,
          builder.create<ConstantIntOp>(loc, 1, ty.cast<mlir::IntegerType>()));
    }
    sub.store(builder, next);
    return ValueCategory(
        (U->getOpcode() == clang::UnaryOperator::Opcode::UO_PostInc) ? prev
                                                                     : next,
        /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Real:
  case clang::UnaryOperator::Opcode::UO_Imag: {
    int fnum =
        (U->getOpcode() == clang::UnaryOperator::Opcode::UO_Real) ? 0 : 1;
    auto lhs_v = sub.val;
    assert(sub.isReference);
    if (auto mt = lhs_v.getType().dyn_cast<mlir::MemRefType>()) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = -1;
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      return ValueCategory(builder.create<polygeist::SubIndexOp>(
                               loc, mt0, lhs_v, getConstantIndex(fnum)),
                           /*isReference*/ true);
    } else if (auto PT =
                   lhs_v.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      mlir::Type ET;
      if (auto ST =
              PT.getElementType().dyn_cast<mlir::LLVM::LLVMStructType>()) {
        ET = ST.getBody()[fnum];
      } else {
        ET = PT.getElementType()
                 .cast<mlir::LLVM::LLVMArrayType>()
                 .getElementType();
      }
      mlir::Value vec[2] = {builder.create<ConstantIntOp>(loc, 0, 32),
                            builder.create<ConstantIntOp>(loc, fnum, 32)};
      return ValueCategory(
          builder.create<mlir::LLVM::GEPOp>(
              loc, mlir::LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
              lhs_v, vec),
          /*isReference*/ true);
    }

    llvm::errs() << "lhs_v: " << lhs_v << "\n";
    U->dump();
    assert(0 && "unhandled real");
  }
  default: {
    U->dump();
    assert(0 && "unhandled opcode");
  }
  }
}

bool hasAffineArith(Operation *op, AffineExpr &expr,
                    mlir::Value &affineForIndVar) {
  // skip IndexCastOp
  if (isa<IndexCastOp>(op))
    return hasAffineArith(op->getOperand(0).getDefiningOp(), expr,
                          affineForIndVar);

  // induction variable are modelled as memref<1xType>
  // %1 = index_cast %induction : index to i32
  // %2 = alloca() : memref<1xi32>
  // store %1, %2[0] : memref<1xi32>
  // ...
  // %5 = load %2[0] : memref<1xf32>
  if (isa<mlir::memref::LoadOp>(op)) {
    auto load = cast<mlir::memref::LoadOp>(op);
    auto loadOperand = load.getOperand(0);
    if (loadOperand.getType().cast<MemRefType>().getShape().size() != 1)
      return false;
    auto maybeAllocaOp = loadOperand.getDefiningOp();
    if (!isa<mlir::memref::AllocaOp>(maybeAllocaOp))
      return false;
    auto allocaUsers = maybeAllocaOp->getUsers();
    if (llvm::none_of(allocaUsers, [](mlir::Operation *op) {
          if (isa<mlir::memref::StoreOp>(op))
            return true;
          return false;
        }))
      return false;
    for (auto user : allocaUsers)
      if (auto storeOp = dyn_cast<mlir::memref::StoreOp>(user)) {
        auto storeOperand = storeOp.getOperand(0);
        auto maybeIndexCast = storeOperand.getDefiningOp();
        if (!isa<IndexCastOp>(maybeIndexCast))
          return false;
        auto indexCastOperand = maybeIndexCast->getOperand(0);
        if (auto blockArg = indexCastOperand.dyn_cast<mlir::BlockArgument>()) {
          if (auto affineForOp = dyn_cast<mlir::AffineForOp>(
                  blockArg.getOwner()->getParentOp()))
            affineForIndVar = affineForOp.getInductionVar();
          else
            return false;
        }
      }
    return true;
  }

  // at this point we expect only AddIOp or MulIOp
  if ((!isa<AddIOp>(op)) && (!isa<MulIOp>(op))) {
    return false;
  }

  // make sure that the current op has at least one constant operand
  // (ConstantIndexOp or ConstantIntOp)
  if (llvm::none_of(op->getOperands(), [](mlir::Value operand) {
        return (isa<ConstantIndexOp>(operand.getDefiningOp()) ||
                isa<ConstantIntOp>(operand.getDefiningOp()));
      }))
    return false;

  // build affine expression by adding or multiplying constants.
  // and keep iterating on the non-constant index
  mlir::Value nonCstOperand = nullptr;
  for (auto operand : op->getOperands()) {
    if (auto constantIndexOp =
            dyn_cast<ConstantIndexOp>(operand.getDefiningOp())) {
      if (isa<AddIOp>(op))
        expr = expr + constantIndexOp.value();
      else
        expr = expr * constantIndexOp.value();
    } else if (auto constantIntOp =
                   dyn_cast<ConstantIntOp>(operand.getDefiningOp())) {
      if (isa<AddIOp>(op))
        expr = expr + constantIntOp.value();
      else
        expr = expr * constantIntOp.value();
    } else
      nonCstOperand = operand;
  }
  return hasAffineArith(nonCstOperand.getDefiningOp(), expr, affineForIndVar);
}

ValueCategory MLIRScanner::VisitBinaryOperator(clang::BinaryOperator *BO) {
  auto loc = getMLIRLocation(BO->getExprLoc());

  auto fixInteger = [&](mlir::Value res) {
    auto prevTy = res.getType().cast<mlir::IntegerType>();
    auto postTy =
        Glob.getTypes().getMLIRType(BO->getType()).cast<mlir::IntegerType>();
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*BO->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    if (postTy != prevTy) {
      if (signedType) {
        res = builder.create<mlir::arith::ExtSIOp>(loc, postTy, res);
      } else {
        res = builder.create<mlir::arith::ExtUIOp>(loc, postTy, res);
      }
    }
    return ValueCategory(res, /*isReference*/ false);
  };

  auto lhs = Visit(BO->getLHS());
  if (!lhs.val && BO->getOpcode() != clang::BinaryOperator::Opcode::BO_Comma) {
    BO->dump();
    BO->getLHS()->dump();
    assert(lhs.val);
  }

  switch (BO->getOpcode()) {
  case clang::BinaryOperator::Opcode::BO_LAnd: {
    mlir::Type types[] = {builder.getIntegerType(1)};
    auto cond = lhs.getValue(builder);
    if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      cond = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
    }
    if (!cond.getType().isa<mlir::IntegerType>()) {
      BO->dump();
      BO->getType()->dump();
      llvm::errs() << "cond: " << cond << "\n";
    }
    auto prevTy = cond.getType().cast<mlir::IntegerType>();
    if (!prevTy.isInteger(1)) {
      cond = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, cond,
          builder.create<ConstantIntOp>(loc, 0, prevTy));
    }
    auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                                /*hasElseRegion*/ true);

    auto oldpoint = builder.getInsertionPoint();
    auto oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&ifOp.getThenRegion().back());

    auto rhs = Visit(BO->getRHS()).getValue(builder);
    assert(rhs != nullptr);
    if (auto LT = rhs.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      rhs = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, rhs, nullptr_llvm);
    }
    if (!rhs.getType().cast<mlir::IntegerType>().isInteger(1)) {
      rhs = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, rhs,
          builder.create<ConstantIntOp>(loc, 0, rhs.getType()));
    }
    mlir::Value truearray[] = {rhs};
    builder.create<mlir::scf::YieldOp>(loc, truearray);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().back());
    mlir::Value falsearray[] = {
        builder.create<ConstantIntOp>(loc, 0, types[0])};
    builder.create<mlir::scf::YieldOp>(loc, falsearray);

    builder.setInsertionPoint(oldblock, oldpoint);
    return fixInteger(ifOp.getResult(0));
  }
  case clang::BinaryOperator::Opcode::BO_LOr: {
    mlir::Type types[] = {builder.getIntegerType(1)};
    auto cond = lhs.getValue(builder);
    auto prevTy = cond.getType().cast<mlir::IntegerType>();
    if (!prevTy.isInteger(1)) {
      cond = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, cond,
          builder.create<ConstantIntOp>(loc, 0, prevTy));
    }
    auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                                /*hasElseRegion*/ true);

    auto oldpoint = builder.getInsertionPoint();
    auto oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&ifOp.getThenRegion().back());

    mlir::Value truearray[] = {builder.create<ConstantIntOp>(loc, 1, types[0])};
    builder.create<mlir::scf::YieldOp>(loc, truearray);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().back());
    auto rhs = Visit(BO->getRHS()).getValue(builder);
    if (!rhs.getType().cast<mlir::IntegerType>().isInteger(1)) {
      rhs = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, rhs,
          builder.create<ConstantIntOp>(loc, 0, rhs.getType()));
    }
    assert(rhs != nullptr);
    mlir::Value falsearray[] = {rhs};
    builder.create<mlir::scf::YieldOp>(loc, falsearray);

    builder.setInsertionPoint(oldblock, oldpoint);

    return fixInteger(ifOp.getResult(0));
  }
  default:
    break;
  }
  auto rhs = Visit(BO->getRHS());
  if (!rhs.val && BO->getOpcode() != clang::BinaryOperator::Opcode::BO_Comma) {
    BO->getRHS()->dump();
    assert(rhs.val);
  }
  // TODO note assumptions made here about unsigned / unordered
  bool signedType = true;
  if (auto bit = dyn_cast<clang::BuiltinType>(&*BO->getType())) {
    if (bit->isUnsignedInteger())
      signedType = false;
    if (bit->isSignedInteger())
      signedType = true;
  }
  switch (BO->getOpcode()) {
  case clang::BinaryOperator::Opcode::BO_GT:
  case clang::BinaryOperator::Opcode::BO_GE:
  case clang::BinaryOperator::Opcode::BO_LT:
  case clang::BinaryOperator::Opcode::BO_LE:
  case clang::BinaryOperator::Opcode::BO_EQ:
  case clang::BinaryOperator::Opcode::BO_NE: {
    signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*BO->getLHS()->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    CmpFPredicate FPred;
    CmpIPredicate IPred;
    LLVM::ICmpPredicate LPred;
    switch (BO->getOpcode()) {
    case clang::BinaryOperator::Opcode::BO_GT:
      FPred = CmpFPredicate::UGT;
      IPred = signedType ? CmpIPredicate::sgt : CmpIPredicate::ugt,
      LPred = LLVM::ICmpPredicate::ugt;
      break;
    case clang::BinaryOperator::Opcode::BO_GE:
      FPred = CmpFPredicate::UGE;
      IPred = signedType ? CmpIPredicate::sge : CmpIPredicate::uge,
      LPred = LLVM::ICmpPredicate::uge;
      break;
    case clang::BinaryOperator::Opcode::BO_LT:
      FPred = CmpFPredicate::ULT;
      IPred = signedType ? CmpIPredicate::slt : CmpIPredicate::ult,
      LPred = LLVM::ICmpPredicate::ult;
      break;
    case clang::BinaryOperator::Opcode::BO_LE:
      FPred = CmpFPredicate::ULE;
      IPred = signedType ? CmpIPredicate::sle : CmpIPredicate::ule,
      LPred = LLVM::ICmpPredicate::ule;
      break;
    case clang::BinaryOperator::Opcode::BO_EQ:
      FPred = CmpFPredicate::UEQ;
      IPred = CmpIPredicate::eq;
      LPred = LLVM::ICmpPredicate::eq;
      break;
    case clang::BinaryOperator::Opcode::BO_NE:
      FPred = CmpFPredicate::UNE;
      IPred = CmpIPredicate::ne;
      LPred = LLVM::ICmpPredicate::ne;
      break;
    default:
      llvm_unreachable("Unknown op in binary comparision switch");
    }

    auto lhs_v = lhs.getValue(builder);
    auto rhs_v = rhs.getValue(builder);
    if (auto mt = lhs_v.getType().dyn_cast<mlir::MemRefType>()) {
      lhs_v = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(mt.getElementType(),
                                     mt.getMemorySpaceAsInt()),
          lhs_v);
    }
    if (auto mt = rhs_v.getType().dyn_cast<mlir::MemRefType>()) {
      rhs_v = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(mt.getElementType(),
                                     mt.getMemorySpaceAsInt()),
          rhs_v);
    }
    mlir::Value res;
    if (lhs_v.getType().isa<mlir::FloatType>()) {
      res = builder.create<arith::CmpFOp>(loc, FPred, lhs_v, rhs_v);
    } else if (lhs_v.getType().isa<LLVM::LLVMPointerType>()) {
      res = builder.create<LLVM::ICmpOp>(loc, LPred, lhs_v, rhs_v);
    } else {
      res = builder.create<arith::CmpIOp>(loc, IPred, lhs_v, rhs_v);
    }
    return ValueCategory(res, /*isReference*/ false);
  }

  case clang::BinaryOperator::Opcode::BO_Comma: {
    return rhs;
  }

  default: {
    BO->dump();
    assert(0 && "unhandled opcode");
  }
  }
}

ValueCategory MLIRScanner::CommonFieldLookup(clang::QualType CT,
                                             const FieldDecl *FD,
                                             mlir::Value val, bool isLValue) {
  assert(FD && "Attempting to lookup field of nullptr");
  auto rd = FD->getParent();

  auto ST = cast<llvm::StructType>(getLLVMType(CT, Glob.getCGM()));

  size_t fnum = 0;

  auto CXRD = dyn_cast<CXXRecordDecl>(rd);

  if (mlirclang::CodeGen::CodeGenTypes::IsLLVMStructABI(rd, ST)) {
    auto &layout = Glob.getCGM().getTypes().getCGRecordLayout(rd);
    fnum = layout.getLLVMFieldNo(FD);
  } else {
    fnum = 0;
    if (CXRD)
      fnum += CXRD->getDefinition()->getNumBases();
    for (auto field : rd->fields()) {
      if (field == FD) {
        break;
      }
      fnum++;
    }
  }

  if (auto PT = val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    mlir::Value vec[] = {builder.create<ConstantIntOp>(loc, 0, 32),
                         builder.create<ConstantIntOp>(loc, fnum, 32)};
    if (!PT.getElementType()
             .isa<mlir::LLVM::LLVMStructType, mlir::LLVM::LLVMArrayType>()) {
      llvm::errs() << "function: " << function << "\n";
      // rd->dump();
      FD->dump();
      FD->getType()->dump();
      llvm::errs() << " val: " << val << " - pt: " << PT << " fn: " << fnum
                   << " ST: " << *ST << "\n";
    }
    mlir::Type ET =
        mlir::TypeSwitch<mlir::Type, mlir::Type>(PT.getElementType())
            .Case<mlir::LLVM::LLVMStructType>(
                [fnum](mlir::LLVM::LLVMStructType ST) {
                  return ST.getBody()[fnum];
                })
            .Case<mlir::LLVM::LLVMArrayType>([](mlir::LLVM::LLVMArrayType AT) {
              return AT.getElementType();
            })
            .Case<MemRefType>([](MemRefType MT) { return MT.getElementType(); })
            .Default([](mlir::Type T) {
              llvm_unreachable("not implemented");
              return T;
            });

    mlir::Value commonGEP = builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()), val,
        vec);
    if (rd->isUnion()) {
      mlir::LLVM::TypeFromLLVMIRTranslator typeTranslator(
          *module->getContext());
      auto subType = typeTranslator.translateType(
          getLLVMType(FD->getType(), Glob.getCGM()));
      commonGEP = builder.create<mlir::LLVM::BitcastOp>(
          loc, mlir::LLVM::LLVMPointerType::get(subType, PT.getAddressSpace()),
          commonGEP);
    }
    if (isLValue)
      commonGEP =
          ValueCategory(commonGEP, /*isReference*/ true).getValue(builder);
    return ValueCategory(commonGEP, /*isReference*/ true);
  }
  auto mt = val.getType().cast<MemRefType>();
  auto shape = std::vector<int64_t>(mt.getShape());
  if (shape.size() > 1) {
    shape.erase(shape.begin());
  } else {
    shape[0] = -1;
  }

  // JLE_QUEL::THOUGHTS
  // This redundancy is here because we might, at some point, create
  // an equivalent GEP or SubIndexOp operation for each sycl types or otherwise
  // clean the redundancy
  mlir::Value Result;
  if (auto ST = mt.getElementType().dyn_cast<mlir::LLVM::LLVMStructType>()) {
    assert(fnum < ST.getBody().size() && "ERROR");

    const auto ElementType = ST.getBody()[fnum];
    const auto ResultType = mlir::MemRefType::get(
        shape, ElementType, MemRefLayoutAttrInterface(), mt.getMemorySpace());

    Result = builder.create<polygeist::SubIndexOp>(loc, ResultType, val,
                                                   getConstantIndex(fnum));
  } else if (auto AT =
                 mt.getElementType().dyn_cast<mlir::sycl::AccessorType>()) {
    assert(fnum < AT.getBody().size() && "ERROR");

    const auto ElementType = AT.getBody()[fnum];
    const auto ResultType = mlir::MemRefType::get(
        shape, ElementType, MemRefLayoutAttrInterface(), mt.getMemorySpace());

    Result = builder.create<polygeist::SubIndexOp>(loc, ResultType, val,
                                                   getConstantIndex(fnum));
  } else if (auto AT = mt.getElementType()
                           .dyn_cast<mlir::sycl::AccessorImplDeviceType>()) {
    assert(fnum < AT.getBody().size() && "ERROR");

    const auto ElementType = AT.getBody()[fnum];
    const auto ResultType = mlir::MemRefType::get(
        shape, ElementType, MemRefLayoutAttrInterface(), mt.getMemorySpace());

    Result = builder.create<polygeist::SubIndexOp>(loc, ResultType, val,
                                                   getConstantIndex(fnum));
  } else if (auto AT = mt.getElementType().dyn_cast<mlir::sycl::ArrayType>()) {
    assert(fnum < AT.getBody().size() && "ERROR");
    const auto elemType = AT.getBody()[fnum].cast<MemRefType>();
    const auto ResultType =
        mlir::MemRefType::get(elemType.getShape(), elemType.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    Result = builder.create<polygeist::SubIndexOp>(loc, ResultType, val,
                                                   getConstantIndex(fnum));
  } else if (auto IT = mt.getElementType().dyn_cast<mlir::sycl::IDType>()) {
    llvm_unreachable("not implemented");
  } else if (auto RT = mt.getElementType().dyn_cast<mlir::sycl::RangeType>()) {
    llvm_unreachable("not implemented");
  } else if (auto RT =
                 mt.getElementType().dyn_cast<mlir::sycl::NdRangeType>()) {
    llvm_unreachable("not implemented");
  } else if (auto RT = mt.getElementType().dyn_cast<mlir::sycl::ItemType>()) {
    assert(fnum < RT.getBody().size() && "ERROR");

    const auto ElementType = RT.getBody()[fnum];
    const auto ResultType = mlir::MemRefType::get(
        shape, ElementType, MemRefLayoutAttrInterface(), mt.getMemorySpace());

    Result = builder.create<polygeist::SubIndexOp>(loc, ResultType, val,
                                                   getConstantIndex(fnum));
  } else if (auto RT =
                 mt.getElementType().dyn_cast<mlir::sycl::ItemBaseType>()) {
    assert(fnum < RT.getBody().size() && "ERROR");

    const auto ElementType = RT.getBody()[fnum];
    const auto ResultType = mlir::MemRefType::get(
        shape, ElementType, MemRefLayoutAttrInterface(), mt.getMemorySpace());

    Result = builder.create<polygeist::SubIndexOp>(loc, ResultType, val,
                                                   getConstantIndex(fnum));
  } else if (auto RT = mt.getElementType().dyn_cast<mlir::sycl::NdItemType>()) {
    assert(fnum < RT.getBody().size() && "ERROR");

    const auto ElementType = RT.getBody()[fnum];
    const auto ResultType = mlir::MemRefType::get(
        shape, ElementType, MemRefLayoutAttrInterface(), mt.getMemorySpace());

    Result = builder.create<polygeist::SubIndexOp>(loc, ResultType, val,
                                                   getConstantIndex(fnum));
  } else if (auto RT = mt.getElementType().dyn_cast<mlir::sycl::GroupType>()) {
    assert(fnum < RT.getBody().size() && "ERROR");

    const auto ElementType = RT.getBody()[fnum];
    const auto ResultType = mlir::MemRefType::get(
        shape, ElementType, MemRefLayoutAttrInterface(), mt.getMemorySpace());

    Result = builder.create<polygeist::SubIndexOp>(loc, ResultType, val,
                                                   getConstantIndex(fnum));
  } else if (auto RT =
                 mt.getElementType().dyn_cast<mlir::sycl::AtomicType>()) {
    llvm_unreachable("not implemented");
  } else if (auto RT =
                 mt.getElementType().dyn_cast<mlir::sycl::MultiPtrType>()) {
    llvm_unreachable("not implemented");
  } else {
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    shape[0] = -1;
    auto mt1 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());

    Result = builder.create<polygeist::SubIndexOp>(loc, mt0, val,
                                                   getConstantIndex(0));
    Result = builder.create<polygeist::SubIndexOp>(loc, mt1, Result,
                                                   getConstantIndex(fnum));
  }

  if (isLValue) {
    Result = ValueCategory(Result, /*isReference*/ true).getValue(builder);
  }

  return ValueCategory(Result, /*isReference*/ true);
}

static bool isSyclIDorRangetoArray(mlir::Type &nt, mlir::Value &value) {
  mlir::Type elemTy = value.getType().dyn_cast<MemRefType>().getElementType();
  return ((elemTy.isa<sycl::IDType>() || elemTy.isa<sycl::RangeType>()) &&
          nt.dyn_cast<MemRefType>().getElementType().isa<sycl::ArrayType>());
}

mlir::Value MLIRScanner::GetAddressOfDerivedClass(
    mlir::Value value, const CXXRecordDecl *DerivedClass,
    CastExpr::path_const_iterator Start, CastExpr::path_const_iterator End) {
  const ASTContext &Context = Glob.getCGM().getContext();

  SmallVector<const CXXRecordDecl *> ToBase = {DerivedClass};
  SmallVector<const CXXBaseSpecifier *> Bases;
  for (auto I = Start; I != End; I++) {
    const CXXBaseSpecifier *Base = *I;
    const auto *BaseDecl =
        cast<CXXRecordDecl>(Base->getType()->castAs<RecordType>()->getDecl());
    ToBase.push_back(BaseDecl);
    Bases.push_back(Base);
  }

  for (int i = ToBase.size() - 1; i > 0; i--) {
    const CXXBaseSpecifier *Base = Bases[i - 1];

    const auto *BaseDecl =
        cast<CXXRecordDecl>(Base->getType()->castAs<RecordType>()->getDecl());
    const auto *RD = ToBase[i - 1];
    // Get the layout.
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    assert(!Base->isVirtual() && "Should not see virtual bases here!");

    // Add the offset.

    mlir::Type nt = Glob.getTypes().getMLIRType(
        Glob.getCGM().getContext().getLValueReferenceType(Base->getType()));

    mlir::Value Offset = nullptr;
    if (mlirclang::CodeGen::CodeGenTypes::IsLLVMStructABI(RD, /*ST*/ nullptr)) {
      Offset = builder.create<arith::ConstantIntOp>(
          loc, -(ssize_t)Layout.getBaseClassOffset(BaseDecl).getQuantity(), 32);
    } else {
      Offset = builder.create<arith::ConstantIntOp>(loc, 0, 32);
      bool found = false;
      for (auto f : RD->bases()) {
        if (f.getType().getTypePtr()->getUnqualifiedDesugaredType() ==
            Base->getType()->getUnqualifiedDesugaredType()) {
          found = true;
          break;
        }
        bool subType = false;
        mlir::Type nt =
            Glob.getTypes().getMLIRType(f.getType(), &subType, false);
        Offset = builder.create<arith::SubIOp>(
            loc, Offset,
            builder.create<IndexCastOp>(
                loc, Offset.getType(),
                builder.create<polygeist::TypeSizeOp>(
                    loc, builder.getIndexType(), mlir::TypeAttr::get(nt))));
      }
      assert(found);
    }

    mlir::Value ptr = value;
    if (auto PT = ptr.getType().dyn_cast<LLVM::LLVMPointerType>())
      ptr = builder.create<LLVM::BitcastOp>(
          loc,
          LLVM::LLVMPointerType::get(builder.getI8Type(), PT.getAddressSpace()),
          ptr);
    else
      ptr = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(builder.getI8Type(), PT.getAddressSpace()),
          ptr);

    mlir::Value idx[] = {Offset};
    ptr = builder.create<LLVM::GEPOp>(loc, ptr.getType(), ptr, idx);

    if (auto PT = nt.dyn_cast<mlir::LLVM::LLVMPointerType>())
      value = builder.create<LLVM::BitcastOp>(
          loc,
          LLVM::LLVMPointerType::get(
              PT.getElementType(),
              ptr.getType().cast<LLVM::LLVMPointerType>().getAddressSpace()),
          ptr);
    else
      value = builder.create<polygeist::Pointer2MemrefOp>(loc, nt, ptr);
  }

  return value;
}

mlir::Value MLIRScanner::GetAddressOfBaseClass(
    mlir::Value value, const CXXRecordDecl *DerivedClass,
    ArrayRef<const clang::Type *> BaseTypes, ArrayRef<bool> BaseVirtuals) {
  const CXXRecordDecl *RD = DerivedClass;

  for (auto tup : llvm::zip(BaseTypes, BaseVirtuals)) {

    auto BaseType = std::get<0>(tup);

    const auto *BaseDecl =
        cast<CXXRecordDecl>(BaseType->castAs<RecordType>()->getDecl());
    // Add the offset.

    mlir::Type nt = Glob.getTypes().getMLIRType(
        Glob.getCGM().getContext().getLValueReferenceType(
            QualType(BaseType, 0)));

    size_t fnum;
    bool subIndex = true;

    if (mlirclang::CodeGen::CodeGenTypes::IsLLVMStructABI(RD, /*ST*/ nullptr)) {
      auto &layout = Glob.getCGM().getTypes().getCGRecordLayout(RD);
      if (std::get<1>(tup))
        fnum = layout.getVirtualBaseIndex(BaseDecl);
      else {
        if (!layout.hasNonVirtualBaseLLVMField(BaseDecl)) {
          subIndex = false;
        } else {
          fnum = layout.getNonVirtualBaseLLVMFieldNo(BaseDecl);
        }
      }
    } else {
      assert(!std::get<1>(tup) && "Should not see virtual bases here!");
      fnum = 0;
      bool found = false;
      for (auto f : RD->bases()) {
        if (f.getType().getTypePtr()->getUnqualifiedDesugaredType() ==
            BaseType->getUnqualifiedDesugaredType()) {
          found = true;
          break;
        }
        fnum++;
      }
      assert(found);
    }

    if (subIndex) {
      if (auto mt = value.getType().dyn_cast<MemRefType>()) {
        auto shape = std::vector<int64_t>(mt.getShape());
        // We do not remove dimensions for an id->array or range->array, because
        // the later cast will be incompatible due to dimension mismatch.
        if (!isSyclIDorRangetoArray(nt, value))
          shape.erase(shape.begin());
        auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                         MemRefLayoutAttrInterface(),
                                         mt.getMemorySpace());
        value = builder.create<polygeist::SubIndexOp>(loc, mt0, value,
                                                      getConstantIndex(fnum));
      } else {
        mlir::Value idx[] = {
            builder.create<arith::ConstantIntOp>(loc, 0, 32),
            builder.create<arith::ConstantIntOp>(loc, fnum, 32)};
        auto PT = value.getType().cast<LLVM::LLVMPointerType>();
        mlir::Type ET =
            mlir::TypeSwitch<mlir::Type, mlir::Type>(PT.getElementType())
                .Case<mlir::LLVM::LLVMStructType>(
                    [fnum](mlir::LLVM::LLVMStructType ST) {
                      return ST.getBody()[fnum];
                    })
                .Case<mlir::LLVM::LLVMArrayType>(
                    [](mlir::LLVM::LLVMArrayType AT) {
                      return AT.getElementType();
                    })
                .Case<mlir::sycl::AccessorType>(
                    [fnum](mlir::sycl::AccessorType AT) {
                      return AT.getBody()[fnum];
                    })
                .Default([](mlir::Type T) {
                  llvm_unreachable("not implemented");
                  return T;
                });

        value = builder.create<LLVM::GEPOp>(
            loc, LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()), value,
            idx);
      }
    }

    auto pt = nt.dyn_cast<mlir::LLVM::LLVMPointerType>();
    if (auto opt = value.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      if (!pt) {
        value = builder.create<polygeist::Pointer2MemrefOp>(loc, nt, value);
      } else {
        if (value.getType() != nt)
          value = builder.create<mlir::LLVM::BitcastOp>(loc, pt, value);
      }
    } else {
      assert(value.getType().isa<MemRefType>() &&
             "Expecting value to have MemRefType");
      if (pt) {
        assert(
            value.getType().cast<MemRefType>().getMemorySpaceAsInt() ==
                pt.getAddressSpace() &&
            "The type of 'value' does not have the same memory space as 'pt'");
        value = builder.create<polygeist::Memref2PointerOp>(loc, pt, value);
      } else {
        if (value.getType() != nt) {
          if (isSyclIDorRangetoArray(nt, value))
            value = builder.create<sycl::SYCLCastOp>(loc, nt, value);
          else
            value = builder.create<memref::CastOp>(loc, nt, value);
        }
      }
    }

    RD = BaseDecl;
  }

  return value;
}

mlir::Value MLIRScanner::reshapeRanklessGlobal(mlir::memref::GetGlobalOp GV) {
  assert(GV.getType().isa<MemRefType>() &&
         "Type of GetGlobalOp should be MemRef");
  MemRefType MT = GV.getType().cast<MemRefType>();
  if (!MT.getShape().empty())
    return GV;

  auto Shape = builder.create<memref::AllocaOp>(
      loc,
      mlir::MemRefType::get(1, mlir::IndexType::get(builder.getContext())));
  return builder.create<memref::ReshapeOp>(
      loc,
      mlir::MemRefType::get(1, MT.getElementType(), MemRefLayoutAttrInterface(),
                            MT.getMemorySpace()),
      GV, Shape);
}

/******************************************************************************/
/*                             MLIRASTConsumer                                */
/******************************************************************************/

mlir::LLVM::LLVMFuncOp MLIRASTConsumer::GetOrCreateMallocFunction() {
  std::string name = "malloc";
  if (llvmFunctions.find(name) != llvmFunctions.end()) {
    return llvmFunctions[name];
  }
  auto ctx = module->getContext();
  mlir::Type types[] = {mlir::IntegerType::get(ctx, 64)};
  auto llvmFnType = LLVM::LLVMFunctionType::get(
      LLVM::LLVMPointerType::get(mlir::IntegerType::get(ctx, 8)), types, false);

  LLVM::Linkage lnk = LLVM::Linkage::External;
  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module->getBody());
  return llvmFunctions[name] = builder.create<LLVM::LLVMFuncOp>(
             module->getLoc(), name, llvmFnType, lnk);
}
mlir::LLVM::LLVMFuncOp MLIRASTConsumer::GetOrCreateFreeFunction() {
  std::string name = "free";
  if (llvmFunctions.find(name) != llvmFunctions.end()) {
    return llvmFunctions[name];
  }
  auto ctx = module->getContext();
  mlir::Type types[] = {
      LLVM::LLVMPointerType::get(mlir::IntegerType::get(ctx, 8))};
  auto llvmFnType =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), types, false);

  LLVM::Linkage lnk = LLVM::Linkage::External;
  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module->getBody());
  return llvmFunctions[name] = builder.create<LLVM::LLVMFuncOp>(
             module->getLoc(), name, llvmFnType, lnk);
}

mlir::LLVM::LLVMFuncOp
MLIRASTConsumer::GetOrCreateLLVMFunction(const FunctionDecl *FD) {
  std::string name = MLIRScanner::getMangledFuncName(*FD, CGM);

  if (name != "malloc" && name != "free")
    name = (PrefixABI + name);

  if (llvmFunctions.find(name) != llvmFunctions.end())
    return llvmFunctions[name];

  mlir::LLVM::TypeFromLLVMIRTranslator typeTranslator(*module->getContext());

  std::vector<mlir::Type> types;
  if (auto CC = dyn_cast<CXXMethodDecl>(FD)) {
    types.push_back(typeTranslator.translateType(
        anonymize(getLLVMType(CC->getThisType(), CGM))));
  }
  for (auto parm : FD->parameters()) {
    types.push_back(typeTranslator.translateType(
        anonymize(getLLVMType(parm->getOriginalType(), CGM))));
  }

  auto rt = typeTranslator.translateType(
      anonymize(getLLVMType(FD->getReturnType(), CGM)));
  auto llvmFnType = LLVM::LLVMFunctionType::get(rt, types,
                                                /*isVarArg=*/FD->isVariadic());
  // Insert the function into the body of the parent module.
  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module->getBody());
  return llvmFunctions[name] = builder.create<LLVM::LLVMFuncOp>(
             module->getLoc(), name, llvmFnType,
             getMLIRLinkage(CGM.getFunctionLinkage(FD)));
}

mlir::LLVM::GlobalOp
MLIRASTConsumer::GetOrCreateLLVMGlobal(const ValueDecl *FD,
                                       std::string prefix) {
  std::string name = prefix + CGM.getMangledName(FD).str();

  name = (PrefixABI + name);

  if (llvmGlobals.find(name) != llvmGlobals.end()) {
    return llvmGlobals[name];
  }

  auto VD = dyn_cast<VarDecl>(FD);
  if (!VD)
    FD->dump();
  VD = VD->getCanonicalDecl();

  auto linkage = CGM.getLLVMLinkageVarDefinition(VD, /*isConstant*/ false);
  LLVM::Linkage lnk = getMLIRLinkage(linkage);

  mlir::Type rt = getTypes().getMLIRType(FD->getType());

  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module->getBody());

  auto glob = builder.create<LLVM::GlobalOp>(
      module->getLoc(), rt, /*constant*/ false, lnk, name, mlir::Attribute());

  if (VD->getInit() ||
      VD->isThisDeclarationADefinition() == VarDecl::Definition ||
      VD->isThisDeclarationADefinition() == VarDecl::TentativeDefinition) {
    Block *blk = new Block();
    builder.setInsertionPointToStart(blk);
    mlir::Value res;
    if (auto init = VD->getInit()) {
      MLIRScanner ms(*this, module, LTInfo);
      ms.setEntryAndAllocBlock(blk);
      res = ms.Visit(const_cast<Expr *>(init)).getValue(builder);
    } else {
      res = builder.create<LLVM::UndefOp>(module->getLoc(), rt);
    }
    bool legal = true;
    for (Operation &op : *blk) {
      auto iface = dyn_cast<MemoryEffectOpInterface>(op);
      if (!iface || !iface.hasNoEffect()) {
        legal = false;
        break;
      }
    }
    if (legal) {
      builder.create<LLVM::ReturnOp>(module->getLoc(),
                                     std::vector<mlir::Value>({res}));
      glob.getInitializerRegion().push_back(blk);
    } else {
      Block *blk2 = new Block();
      builder.setInsertionPointToEnd(blk2);
      mlir::Value nres = builder.create<LLVM::UndefOp>(module->getLoc(), rt);
      builder.create<LLVM::ReturnOp>(module->getLoc(),
                                     std::vector<mlir::Value>({nres}));
      glob.getInitializerRegion().push_back(blk2);

      builder.setInsertionPointToStart(module->getBody());
      auto funcName = name + "@init";
      LLVM::GlobalCtorsOp ctors = nullptr;
      for (auto &op : *module->getBody()) {
        if (auto c = dyn_cast<LLVM::GlobalCtorsOp>(&op)) {
          ctors = c;
        }
      }
      SmallVector<mlir::Attribute> funcs;
      funcs.push_back(FlatSymbolRefAttr::get(module->getContext(), funcName));
      SmallVector<mlir::Attribute> idxs;
      idxs.push_back(builder.getI32IntegerAttr(0));
      if (ctors) {
        for (auto f : ctors.getCtors())
          funcs.push_back(f);
        for (auto v : ctors.getPriorities())
          idxs.push_back(v);
        ctors->erase();
      }

      builder.create<LLVM::GlobalCtorsOp>(module->getLoc(),
                                          builder.getArrayAttr(funcs),
                                          builder.getArrayAttr(idxs));

      auto llvmFnType = LLVM::LLVMFunctionType::get(
          mlir::LLVM::LLVMVoidType::get(module->getContext()),
          ArrayRef<mlir::Type>(), false);

      auto func = builder.create<LLVM::LLVMFuncOp>(
          module->getLoc(), funcName, llvmFnType, LLVM::Linkage::Private);
      func.getRegion().push_back(blk);
      builder.setInsertionPointToEnd(blk);
      builder.create<LLVM::StoreOp>(
          module->getLoc(), res,
          builder.create<LLVM::AddressOfOp>(module->getLoc(), glob));
      builder.create<LLVM::ReturnOp>(module->getLoc(), ArrayRef<mlir::Value>());
    }
  }
  if (lnk == LLVM::Linkage::Private || lnk == LLVM::Linkage::Internal) {
    SymbolTable::setSymbolVisibility(glob,
                                     mlir::SymbolTable::Visibility::Private);
  }
  return llvmGlobals[name] = glob;
}

std::pair<mlir::memref::GlobalOp, bool>
MLIRASTConsumer::getOrCreateGlobal(const ValueDecl &VD, std::string Prefix,
                                   FunctionContext FuncContext) {
  const std::string Name = PrefixABI + Prefix + CGM.getMangledName(&VD).str();
  if (globals.find(Name) != globals.end())
    return globals[Name];

  const bool IsArray = isa<clang::ArrayType>(VD.getType());
  const mlir::Type MLIRType = getTypes().getMLIRType(VD.getType());
  const clang::VarDecl *Var = cast<VarDecl>(VD).getCanonicalDecl();
  const unsigned MemSpace =
      CGM.getContext().getTargetAddressSpace(CGM.GetGlobalVarAddressSpace(Var));

  // Note: global scalar variables have always memref type with rank zero.
  auto VarTy =
      (!IsArray) ? mlir::MemRefType::get({}, MLIRType, {}, MemSpace)
                 : mlir::MemRefType::get(
                       MLIRType.cast<mlir::MemRefType>().getShape(),
                       MLIRType.cast<mlir::MemRefType>().getElementType(),
                       MemRefLayoutAttrInterface(),
                       wrapIntegerMemorySpace(MemSpace, module->getContext()));

  // The insertion point depends on whether the global variable is in the host
  // or the device context.
  mlir::OpBuilder Builder(module->getContext());
  if (FuncContext == FunctionContext::SYCLDevice)
    Builder.setInsertionPointToStart(getDeviceModule(*module).getBody());
  else {
    assert(FuncContext == FunctionContext::Host);
    Builder.setInsertionPointToStart(module->getBody());
  }

  // Create the global.
  VarDecl::DefinitionKind DefKind = Var->isThisDeclarationADefinition();
  mlir::Attribute InitialVal;
  if (DefKind == VarDecl::Definition || DefKind == VarDecl::TentativeDefinition)
    InitialVal = Builder.getUnitAttr();

  const bool IsConst = VD.getType().isConstQualified();
  llvm::Align Align = CGM.getContext().getDeclAlign(&VD).getAsAlign();

  auto globalOp = Builder.create<mlir::memref::GlobalOp>(
      module->getLoc(), Name, /*sym_visibility*/ mlir::StringAttr(), VarTy,
      InitialVal, IsConst,
      Builder.getIntegerAttr(Builder.getIntegerType(64), Align.value()));

  // Set the visibility.
  switch (CGM.getLLVMLinkageVarDefinition(Var, IsConst)) {
  case llvm::GlobalValue::LinkageTypes::InternalLinkage:
  case llvm::GlobalValue::LinkageTypes::PrivateLinkage:
    SymbolTable::setSymbolVisibility(globalOp,
                                     mlir::SymbolTable::Visibility::Private);
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
    SymbolTable::setSymbolVisibility(globalOp,
                                     mlir::SymbolTable::Visibility::Public);
    break;
  }

  // Initialize the global.
  const Expr *InitExpr = Var->getAnyInitializer();

  if (!InitExpr) {
    if (DefKind != VarDecl::DeclarationOnly) {
      // Tentative definitions are initialized to {0}.
      assert(!VD.getType()->isIncompleteType() && "Unexpected incomplete type");

      mlir::Attribute Zero =
          mlir::TypeSwitch<mlir::Type, mlir::Attribute>(VarTy.getElementType())
              .Case<mlir::IntegerType>(
                  [](mlir::Type Ty) { return IntegerAttr::get(Ty, 0); })
              .Case<mlir::FloatType>(
                  [](mlir::Type Ty) { return FloatAttr::get(Ty, 0); })
              .Default([&](mlir::Type Ty) {
                llvm_unreachable("unimplemented");
                return mlir::Attribute();
              });
      auto ZeroVal = DenseElementsAttr::get(
          RankedTensorType::get(VarTy.getShape(), VarTy.getElementType()),
          Zero);
      globalOp.setInitialValueAttr(ZeroVal);
    }
  } else {
    // explicit initialization.
    assert(DefKind == VarDecl::Definition);

    MLIRScanner MS(*this, module, LTInfo);
    mlir::Block B;
    MS.setEntryAndAllocBlock(&B);

    OpBuilder Builder(module->getContext());
    Builder.setInsertionPointToEnd(&B);
    auto Op = Builder.create<memref::AllocaOp>(module->getLoc(), VarTy);

    if (isa<InitListExpr>(InitExpr)) {
      mlir::Attribute InitValAttr = MS.InitializeValueByInitListExpr(
          Op, const_cast<clang::Expr *>(InitExpr));
      globalOp.setInitialValueAttr(InitValAttr);
    } else {
      ValueCategory VC = MS.Visit(const_cast<clang::Expr *>(InitExpr));
      assert(!VC.isReference && "The initializer should not be a reference");

      auto Op = VC.val.getDefiningOp<arith::ConstantOp>();
      assert(Op && "Could not find the initializer constant expression");

      auto InitialVal = SplatElementsAttr::get(
          RankedTensorType::get(VarTy.getShape(), VarTy.getElementType()),
          Op.getValue());
      globalOp.setInitialValueAttr(InitialVal);
    }
  }

  globals[Name] = std::make_pair(globalOp, IsArray);

  return globals[Name];
}

mlir::Value MLIRASTConsumer::GetOrCreateGlobalLLVMString(
    mlir::Location loc, mlir::OpBuilder &builder, StringRef value) {
  using namespace mlir;
  // Create the global at the entry of the module.
  if (llvmStringGlobals.find(value.str()) == llvmStringGlobals.end()) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module->getBody());
    auto type = LLVM::LLVMArrayType::get(
        mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);
    llvmStringGlobals[value.str()] = builder.create<LLVM::GlobalOp>(
        loc, type, /*isConstant=*/true, LLVM::Linkage::Internal,
        "str" + std::to_string(llvmStringGlobals.size()),
        builder.getStringAttr(value.str() + '\0'));
  }

  LLVM::GlobalOp global = llvmStringGlobals[value.str()];
  // Get the pointer to the first character in the global string.
  mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);
  return globalPtr;
}

mlir::FunctionOpInterface
MLIRASTConsumer::GetOrCreateMLIRFunction(FunctionToEmit &FTE, bool ShouldEmit,
                                         bool getDeviceStub) {
  assert(FTE.getDecl().getTemplatedKind() !=
             FunctionDecl::TemplatedKind::TK_FunctionTemplate &&
         FTE.getDecl().getTemplatedKind() !=
             FunctionDecl::TemplatedKind::
                 TK_DependentFunctionTemplateSpecialization &&
         "Unexpected template kind");

  const clang::FunctionDecl &FD = FTE.getDecl();
  const std::string mangledName =
      (getDeviceStub)
          ? PrefixABI +
                CGM.getMangledName(GlobalDecl(&FD, KernelReferenceKind::Kernel))
                    .str()
          : PrefixABI + MLIRScanner::getMangledFuncName(FD, CGM);
  assert(mangledName != "free");

  // Early exit if the function has already been generated.
  if (Optional<FunctionOpInterface> optFunction =
          getMLIRFunction(mangledName, FTE.getContext()))
    return *optFunction;

  // Create the MLIR function and set its various attributes.
  FunctionOpInterface function =
      createMLIRFunction(FTE, mangledName, ShouldEmit);
  checkFunctionParent(function, FTE.getContext(), module);

  // Decide whether the MLIR function should be emitted.
  const FunctionDecl *Def = nullptr;
  if (!FD.isDefined(Def, /*checkforfriend*/ true))
    Def = &FD;

  if (Def->isThisDeclarationADefinition()) {
    assert(Def->getTemplatedKind() !=
               FunctionDecl::TemplatedKind::TK_FunctionTemplate &&
           Def->getTemplatedKind() !=
               FunctionDecl::TemplatedKind::
                   TK_DependentFunctionTemplateSpecialization);
    if (ShouldEmit) {
      LLVM_DEBUG(llvm::dbgs()
                 << __LINE__ << ": Pushing " << FTE.getContext() << " function "
                 << Def->getNameAsString() << " to functionsToEmit\n");
      functionsToEmit.emplace_back(*Def, FTE.getContext());
    }
  } else if (ShouldEmit) {
    emitIfFound.insert(mangledName);
  }

  return function;
}

const clang::CodeGen::CGFunctionInfo &
MLIRASTConsumer::GetOrCreateCGFunctionInfo(const clang::FunctionDecl *FD) {
  auto result = CGFunctionInfos.find(FD);
  if (result != CGFunctionInfos.end())
    return *result->second;

  GlobalDecl GD;
  if (auto CC = dyn_cast<CXXConstructorDecl>(FD))
    GD = GlobalDecl(CC, CXXCtorType::Ctor_Complete);
  else if (auto CC = dyn_cast<CXXDestructorDecl>(FD))
    GD = GlobalDecl(CC, CXXDtorType::Dtor_Complete);
  else
    GD = GlobalDecl(FD);
  CGFunctionInfos[FD] = &getTypes().arrangeGlobalDeclaration(GD);
  return *CGFunctionInfos[FD];
}

void MLIRASTConsumer::run() {
  while (functionsToEmit.size()) {
    FunctionToEmit &FTE = functionsToEmit.front();
    functionsToEmit.pop_front();

    const FunctionDecl &FD = FTE.getDecl();

    assert(FD.getBody());
    assert(FD.getTemplatedKind() != FunctionDecl::TK_FunctionTemplate);
    assert(FD.getTemplatedKind() !=
           FunctionDecl::TemplatedKind::
               TK_DependentFunctionTemplateSpecialization);

    std::string mangledName = MLIRScanner::getMangledFuncName(FD, CGM);
    const std::pair<FunctionContext, std::string> doneKey(FTE.getContext(),
                                                          mangledName);
    if (done.count(doneKey))
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

    done.insert(doneKey);
    MLIRScanner ms(*this, module, LTInfo);
    FunctionOpInterface function =
        GetOrCreateMLIRFunction(FTE, true /* ShouldEmit */);
    ms.init(function, FTE);

    LLVM_DEBUG({
      llvm::dbgs() << "\n";
      function.dump();
      llvm::dbgs() << "\n";

      if (functionsToEmit.size()) {
        llvm::dbgs() << "-- FUNCTION(S) LEFT TO BE EMITTED --\n";

        for (const auto &FTE : functionsToEmit) {
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

void MLIRASTConsumer::HandleDeclContext(DeclContext *DC) {
  for (auto D : DC->decls()) {
    if (auto NS = dyn_cast<clang::NamespaceDecl>(D)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto NS = dyn_cast<clang::ExternCContextDecl>(D)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto NS = dyn_cast<clang::LinkageSpecDecl>(D)) {
      HandleDeclContext(NS);
      continue;
    }
    const FunctionDecl *fd = dyn_cast<clang::FunctionDecl>(D);
    if (!fd)
      continue;
    if (!fd->doesThisDeclarationHaveABody() &&
        !fd->doesDeclarationForceExternallyVisibleDefinition())
      continue;
    if (!fd->hasBody())
      continue;
    if (fd->isTemplated())
      continue;

    bool externLinkage = true;
    /*
    auto LV = CGM.getFunctionLinkage(fd);
    if (LV == llvm::GlobalValue::InternalLinkage || LV ==
    llvm::GlobalValue::PrivateLinkage) externLinkage = false; if
    (fd->isInlineSpecified()) externLinkage = false;
    */
    if (!CGM.getContext().DeclMustBeEmitted(fd))
      externLinkage = false;

    std::string name = MLIRScanner::getMangledFuncName(*fd, CGM);

    // Don't create std functions unless necessary
    if (StringRef(name).startswith("_ZNKSt"))
      continue;
    if (StringRef(name).startswith("_ZSt"))
      continue;
    if (StringRef(name).startswith("_ZNSt"))
      continue;
    if (StringRef(name).startswith("_ZN9__gnu"))
      continue;
    if (name == "cudaGetDevice" || name == "cudaMalloc")
      continue;

    if ((emitIfFound.count("*") && name != "fpclassify" && !fd->isStatic() &&
         externLinkage) ||
        emitIfFound.count(name)) {
      FunctionToEmit FTE(*fd);
      LLVM_DEBUG(llvm::dbgs()
                 << __LINE__ << ": Pushing " << FTE.getContext() << " function "
                 << fd->getNameAsString() << " to functionsToEmit\n");
      functionsToEmit.push_back(FTE);
    }
  }
}

bool MLIRASTConsumer::HandleTopLevelDecl(DeclGroupRef dg) {
  DeclGroupRef::iterator it;

  if (error)
    return true;

  for (it = dg.begin(); it != dg.end(); ++it) {
    if (auto NS = dyn_cast<clang::NamespaceDecl>(*it)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto NS = dyn_cast<clang::ExternCContextDecl>(*it)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto NS = dyn_cast<clang::LinkageSpecDecl>(*it)) {
      HandleDeclContext(NS);
      continue;
    }
    const FunctionDecl *fd = dyn_cast<clang::FunctionDecl>(*it);
    if (!fd)
      continue;
    if (!fd->doesThisDeclarationHaveABody() &&
        !fd->doesDeclarationForceExternallyVisibleDefinition())
      continue;
    if (!fd->hasBody())
      continue;
    if (fd->isTemplated())
      continue;

    //  if (fd->getIdentifier())
    //    llvm::errs() << "Func name: " << fd->getName() << "\n";
    //  llvm::errs() << "Func Body && Loc " << "\n";
    //  fd->getBody()->dump();
    //  fd->getLocation().dump(SM);

    bool externLinkage = true;
    /*
    auto LV = CGM.getFunctionLinkage(fd);
    if (LV == llvm::GlobalValue::InternalLinkage || LV ==
    llvm::GlobalValue::PrivateLinkage) externLinkage = false; if
    (fd->isInlineSpecified()) externLinkage = false;
    */
    if (!CGM.getContext().DeclMustBeEmitted(fd))
      externLinkage = false;

    std::string name = MLIRScanner::getMangledFuncName(*fd, CGM);

    // Don't create std functions unless necessary
    if (StringRef(name).startswith("_ZNKSt"))
      continue;
    if (StringRef(name).startswith("_ZSt"))
      continue;
    if (StringRef(name).startswith("_ZNSt"))
      continue;
    if (StringRef(name).startswith("_ZN9__gnu"))
      continue;
    if (name == "cudaGetDevice" || name == "cudaMalloc")
      continue;

    if ((emitIfFound.count("*") && name != "fpclassify" && !fd->isStatic() &&
         externLinkage) ||
        emitIfFound.count(name) || fd->hasAttr<OpenCLKernelAttr>() ||
        fd->hasAttr<SYCLDeviceAttr>()) {
      FunctionToEmit FTE(*fd);
      LLVM_DEBUG(llvm::dbgs()
                 << __LINE__ << ": Pushing " << FTE.getContext() << " function "
                 << fd->getNameAsString() << " to functionsToEmit\n");
      functionsToEmit.push_back(FTE);
    }
  }

  return true;
}

// Wait until Sema has instantiated all the relevant code
// before running codegen on the selected functions.
void MLIRASTConsumer::HandleTranslationUnit(ASTContext &C) { run(); }

mlir::Location MLIRASTConsumer::getMLIRLocation(clang::SourceLocation loc) {
  auto spellingLoc = SM.getSpellingLoc(loc);
  auto lineNumber = SM.getSpellingLineNumber(spellingLoc);
  auto colNumber = SM.getSpellingColumnNumber(spellingLoc);
  auto fileId = SM.getFilename(spellingLoc);

  auto ctx = module->getContext();
  return FileLineColLoc::get(ctx, fileId, lineNumber, colNumber);
}

llvm::GlobalValue::LinkageTypes
MLIRASTConsumer::getLLVMLinkageType(const clang::FunctionDecl &FD,
                                    bool shouldEmit) {
  if (!FD.hasBody() || !shouldEmit)
    return llvm::GlobalValue::LinkageTypes::ExternalLinkage;
  if (auto CC = dyn_cast<CXXConstructorDecl>(&FD))
    return CGM.getFunctionLinkage(GlobalDecl(CC, CXXCtorType::Ctor_Complete));
  if (auto CC = dyn_cast<CXXDestructorDecl>(&FD))
    return CGM.getFunctionLinkage(GlobalDecl(CC, CXXDtorType::Dtor_Complete));

  return CGM.getFunctionLinkage(&FD);
}

mlir::LLVM::Linkage
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

mlir::FunctionOpInterface
MLIRASTConsumer::createMLIRFunction(const FunctionToEmit &FTE,
                                    std::string mangledName, bool ShouldEmit) {
  const FunctionDecl &FD = FTE.getDecl();
  Location loc = getMLIRLocation(FD.getLocation());
  mlir::OpBuilder Builder(module->getContext());

  const clang::CodeGen::CGFunctionInfo &FI = GetOrCreateCGFunctionInfo(&FD);
  mlir::FunctionType funcTy = getTypes().getFunctionType(FI, FD);

  mlir::FunctionOpInterface function =
      FD.hasAttr<SYCLKernelAttr>()
          ? Builder.create<gpu::GPUFuncOp>(loc, mangledName, funcTy)
          : Builder.create<func::FuncOp>(loc, mangledName, funcTy);

  setMLIRFunctionVisibility(function, FTE, ShouldEmit);
  setMLIRFunctionAttributes(function, FTE, ShouldEmit);

  /// Inject the MLIR function created in either the device module or in the
  /// host module, depending on the calling context.
  switch (FTE.getContext()) {
  case FunctionContext::Host:
    module->push_back(function);
    functions[mangledName] = cast<func::FuncOp>(function);
    break;
  case FunctionContext::SYCLDevice:
    getDeviceModule(*module).push_back(function);
    deviceFunctions[mangledName] = function;
    break;
  }

  LLVM_DEBUG(llvm::dbgs() << "Created MLIR function: " << function << "\n");

  return function;
}

void MLIRASTConsumer::setMLIRFunctionVisibility(
    mlir::FunctionOpInterface function, const FunctionToEmit &FTE,
    bool shouldEmit) {
  const FunctionDecl &FD = FTE.getDecl();
  SymbolTable::Visibility visibility = SymbolTable::Visibility::Public;

  if (!shouldEmit || !FD.isDefined() || FD.hasAttr<CUDAGlobalAttr>() ||
      FD.hasAttr<CUDADeviceAttr>())
    visibility = SymbolTable::Visibility::Private;
  else {
    llvm::GlobalValue::LinkageTypes LV = getLLVMLinkageType(FD, shouldEmit);
    if (LV == llvm::GlobalValue::InternalLinkage ||
        LV == llvm::GlobalValue::PrivateLinkage)
      visibility = SymbolTable::Visibility::Private;
  }

  SymbolTable::setSymbolVisibility(function, visibility);
}

/// Determines whether the language options require us to model
/// unwind exceptions.  We treat -fexceptions as mandating this
/// except under the fragile ObjC ABI with only ObjC exceptions
/// enabled.  This means, for example, that C with -fexceptions
/// enables this.
static bool hasUnwindExceptions(const LangOptions &LangOpts) {
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
    const Decl *D, mlir::FunctionOpInterface F) const {
  const CodeGenOptions &CodeGenOpts = CGM.getCodeGenOpts();
  const LangOptions &LangOpts = CGM.getLangOpts();
  MLIRContext *Ctx = module->getContext();
  mlirclang::AttrBuilder B(*Ctx);

  if ((!D || !D->hasAttr<NoUwtableAttr>()) && CodeGenOpts.UnwindTables)
    B.addPassThroughAttribute(
        llvm::Attribute::UWTable,
        uint64_t(llvm::UWTableKind(CodeGenOpts.UnwindTables)));

  if (CodeGenOpts.StackClashProtector)
    B.addPassThroughAttribute("probe-stack",
                              mlir::StringAttr::get(Ctx, "inline-asm"));

  if (!hasUnwindExceptions(LangOpts))
    B.addPassThroughAttribute(llvm::Attribute::NoUnwind);

  if (D && D->hasAttr<NoStackProtectorAttr>())
    ; // Do Nothing.
  else if (D && D->hasAttr<StrictGuardStackCheckAttr>() &&
           LangOpts.getStackProtector() == LangOptions::SSPOn)
    B.addPassThroughAttribute(llvm::Attribute::StackProtectStrong);
  else if (LangOpts.getStackProtector() == LangOptions::SSPOn)
    B.addPassThroughAttribute(llvm::Attribute::StackProtect);
  else if (LangOpts.getStackProtector() == LangOptions::SSPStrong)
    B.addPassThroughAttribute(llvm::Attribute::StackProtectStrong);
  else if (LangOpts.getStackProtector() == LangOptions::SSPReq)
    B.addPassThroughAttribute(llvm::Attribute::StackProtectReq);

  if (!D) {
    // If we don't have a declaration to control inlining, the function isn't
    // explicitly marked as alwaysinline for semantic reasons, and inlining is
    // disabled, mark the function as noinline.
    if (!F->hasAttr(llvm::Attribute::getNameFromAttrKind(
            llvm::Attribute::AlwaysInline)) &&
        CodeGenOpts.getInlining() == CodeGenOptions::OnlyAlwaysInlining)
      B.addPassThroughAttribute(llvm::Attribute::NoInline);

    mlir::NamedAttrList attrs(F->getAttrDictionary());
    attrs.append(B.getAttrs());
    F->setAttrs(attrs.getDictionary(Ctx));
    return;
  }

  // Track whether we need to add the optnone LLVM attribute,
  // starting with the default for this optimization level.
  bool ShouldAddOptNone =
      !CodeGenOpts.DisableO0ImplyOptNone && CodeGenOpts.OptimizationLevel == 0;
  // We can't add optnone in the following cases, it won't pass the verifier.
  ShouldAddOptNone &= !D->hasAttr<MinSizeAttr>();
  ShouldAddOptNone &= !D->hasAttr<AlwaysInlineAttr>();

  // Add optnone, but do so only if the function isn't always_inline.
  if ((ShouldAddOptNone || D->hasAttr<OptimizeNoneAttr>()) &&
      !F->hasAttr(llvm::Attribute::getNameFromAttrKind(
          llvm::Attribute::AlwaysInline))) {
    B.addPassThroughAttribute(llvm::Attribute::OptimizeNone);

    // OptimizeNone implies noinline; we should not be inlining such functions.
    B.addPassThroughAttribute(llvm::Attribute::NoInline);

    // We still need to handle naked functions even though optnone subsumes
    // much of their semantics.
    if (D->hasAttr<NakedAttr>())
      B.addPassThroughAttribute(llvm::Attribute::Naked);

    // OptimizeNone wins over OptimizeForSize and MinSize.
    F->removeAttr(
        llvm::Attribute::getNameFromAttrKind(llvm::Attribute::OptimizeForSize));
    F->removeAttr(
        llvm::Attribute::getNameFromAttrKind(llvm::Attribute::MinSize));
  } else if (D->hasAttr<NakedAttr>()) {
    // Naked implies noinline: we should not be inlining such functions.
    B.addPassThroughAttribute(llvm::Attribute::Naked);
    B.addPassThroughAttribute(llvm::Attribute::NoInline);
  } else if (D->hasAttr<NoDuplicateAttr>()) {
    B.addPassThroughAttribute(llvm::Attribute::NoDuplicate);
  } else if (D->hasAttr<NoInlineAttr>() &&
             !F->hasAttr(llvm::Attribute::getNameFromAttrKind(
                 llvm::Attribute::AlwaysInline))) {
    // Add noinline if the function isn't always_inline.
    B.addPassThroughAttribute(llvm::Attribute::NoInline);
  } else if (D->hasAttr<AlwaysInlineAttr>() &&
             !F->hasAttr(llvm::Attribute::getNameFromAttrKind(
                 llvm::Attribute::NoInline))) {
    // (noinline wins over always_inline, and we can't specify both in IR)
    B.addPassThroughAttribute(llvm::Attribute::AlwaysInline);
  } else if (CodeGenOpts.getInlining() == CodeGenOptions::OnlyAlwaysInlining) {
    // If we're not inlining, then force everything that isn't always_inline to
    // carry an explicit noinline attribute.
    if (!F->hasAttr(llvm::Attribute::getNameFromAttrKind(
            llvm::Attribute::AlwaysInline)))
      B.addPassThroughAttribute(llvm::Attribute::NoInline);
  } else {
    // Otherwise, propagate the inline hint attribute and potentially use its
    // absence to mark things as noinline.
    if (auto *FD = dyn_cast<FunctionDecl>(D)) {
      // Search function and template pattern redeclarations for inline.
      auto CheckForInline = [](const FunctionDecl *FD) {
        auto CheckRedeclForInline = [](const FunctionDecl *Redecl) {
          return Redecl->isInlineSpecified();
        };
        if (any_of(FD->redecls(), CheckRedeclForInline))
          return true;
        const FunctionDecl *Pattern = FD->getTemplateInstantiationPattern();
        if (!Pattern)
          return false;
        return any_of(Pattern->redecls(), CheckRedeclForInline);
      };
      if (CheckForInline(FD)) {
        B.addPassThroughAttribute(llvm::Attribute::InlineHint);
      } else if (CodeGenOpts.getInlining() ==
                     CodeGenOptions::OnlyHintInlining &&
                 !FD->isInlined() &&
                 !F->hasAttr(llvm::Attribute::getNameFromAttrKind(
                     llvm::Attribute::AlwaysInline))) {
        B.addPassThroughAttribute(llvm::Attribute::NoInline);
      }
    }
  }

  // Add other optimization related attributes if we are optimizing this
  // function.
  if (!D->hasAttr<OptimizeNoneAttr>()) {
    if (D->hasAttr<ColdAttr>()) {
      if (!ShouldAddOptNone)
        B.addPassThroughAttribute(llvm::Attribute::OptimizeForSize);
      B.addPassThroughAttribute(llvm::Attribute::Cold);
    }
    if (D->hasAttr<HotAttr>())
      B.addPassThroughAttribute(llvm::Attribute::Hot);
    if (D->hasAttr<MinSizeAttr>())
      B.addPassThroughAttribute(llvm::Attribute::MinSize);
  }

  NamedAttrList attrs(F->getAttrDictionary());
  attrs.append(B.getAttrs());
  F->setAttrs(attrs.getDictionary(Ctx));

  unsigned alignment = D->getMaxAlignment() / CGM.getContext().getCharWidth();
  if (alignment) {
    OpBuilder Builder(Ctx);
    F->setAttr(llvm::Attribute::getNameFromAttrKind(llvm::Attribute::Alignment),
               Builder.getIntegerAttr(Builder.getIntegerType(64), alignment));
  }

  if (!D->hasAttr<AlignedAttr>())
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
      unsigned AlignVal = AlignmentAttr[1].cast<mlir::IntegerAttr>().getInt();
      if (AlignVal < 2 && isa<CXXMethodDecl>(D)) {
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
  auto *MD = dyn_cast<CXXMethodDecl>(D);
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

void MLIRASTConsumer::setMLIRFunctionAttributes(
    mlir::FunctionOpInterface function, const FunctionToEmit &FTE,
    bool ShouldEmit) {
  using Attribute = llvm::Attribute;

  const FunctionDecl &FD = FTE.getDecl();
  MLIRContext *Ctx = module->getContext();

  bool isDeviceContext = (FTE.getContext() == FunctionContext::SYCLDevice);
  if (!isDeviceContext) {
    LLVM_DEBUG(llvm::dbgs()
               << "Not in a device context - skipping setting attributes for "
               << FD.getNameAsString() << "\n");

    mlirclang::AttrBuilder attrBuilder(*Ctx);
    LLVM::Linkage lnk = getMLIRLinkage(getLLVMLinkageType(FD, ShouldEmit));
    attrBuilder.addAttribute("llvm.linkage",
                             mlir::LLVM::LinkageAttr::get(Ctx, lnk));

    // HACK: we want to avoid setting additional attributes on non-sycl
    // functions because we do not want to adjust the test cases at this time
    // (if we did we would have merge conflicts if we ever update polygeist).
    NamedAttrList attrs(function->getAttrDictionary());
    attrs.append(attrBuilder.getAttrs());
    function->setAttrs(attrs.getDictionary(module->getContext()));
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Setting attributes for " << FD.getNameAsString()
                          << "\n");

  mlirclang::AttributeList PAL;
  {
    const clang::CodeGen::CGFunctionInfo &FI = GetOrCreateCGFunctionInfo(&FD);
    const auto *FPT = FD.getType()->getAs<FunctionProtoType>();
    clang::CodeGen::CGCalleeInfo CalleeInfo(FPT);

    unsigned CallingConv;
    getTypes().constructAttributeList(function.getName(), FI, CalleeInfo, PAL,
                                      CallingConv,
                                      /*AttrOnCallSite*/ false,
                                      /*IsThunk*/ false);

    // Set additional function attributes that are not derivable from the
    // function declaration.
    mlirclang::AttrBuilder attrBuilder(*Ctx);
    {
      attrBuilder.addAttribute(
          "llvm.cconv",
          mlir::LLVM::CConvAttr::get(
              Ctx, static_cast<mlir::LLVM::cconv::CConv>(CallingConv)));

      LLVM::Linkage Lnk = getMLIRLinkage(getLLVMLinkageType(FD, ShouldEmit));
      attrBuilder.addAttribute("llvm.linkage",
                               mlir::LLVM::LinkageAttr::get(Ctx, Lnk));

      if (FD.hasAttr<SYCLKernelAttr>())
        attrBuilder.addAttribute(gpu::GPUDialect::getKernelFuncAttrName(),
                                 UnitAttr::get(Ctx));

      if (CGM.getLangOpts().SYCLIsDevice)
        attrBuilder.addPassThroughAttribute(
            "sycl-module-id",
            StringAttr::get(Ctx, llvmMod.getModuleIdentifier()));

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
          (CGM.getLangOpts().CUDA && FD.hasAttr<CUDAGlobalAttr>()))
        attrBuilder.addPassThroughAttribute(
            llvm::Attribute::AttrKind::NoRecurse);

      // Note: this one is incorrect because we should traverse the function
      // body before setting this attribute. If the body does not contain
      // any infinite loops the attributes can be set.
      auto functionMustProgress =
          [](const clang::CodeGen::CodeGenModule &CGM) -> bool {
        if (CGM.getCodeGenOpts().getFiniteLoops() ==
            CodeGenOptions::FiniteLoopsKind::Never)
          return false;
        return CGM.getLangOpts().CPlusPlus11;
      };

      if (functionMustProgress(CGM))
        attrBuilder.addPassThroughAttribute(Attribute::AttrKind::MustProgress);
    }

    setMLIRFunctionAttributesForDefinition(&FD, function);

    PAL.addFnAttrs(attrBuilder);
  }

  // Set function attributes.
  mlirclang::AttributeList FnAttrs(function->getAttrDictionary(), {}, {});
  FnAttrs.addFnAttrs(PAL.getFnAttrs(), *Ctx);
  function->setAttrs(FnAttrs.getFnAttrs().getDictionary(Ctx));

  // Set parameters attributes.
  const ArrayRef<NamedAttrList> ParamAttrs = PAL.getParamAttrs();
  assert(ParamAttrs.size() == function.getNumArguments());
  for (unsigned Index : llvm::seq<unsigned>(0, function.getNumArguments())) {
    for (NamedAttribute Attr : ParamAttrs[Index])
      function.setArgAttr(Index, Attr.getName(), Attr.getValue());
  }

  // Set function result attributes.
  for (NamedAttribute Attr : PAL.getRetAttrs())
    function.setResultAttr(0, Attr.getName(), Attr.getValue());
}

llvm::Optional<mlir::FunctionOpInterface>
MLIRASTConsumer::getMLIRFunction(const std::string &mangledName,
                                 FunctionContext context) const {
  const auto find = [&](const auto &map) {
    const auto Iter = map.find(mangledName);
    return Iter == map.end()
               ? llvm::None
               : llvm::Optional<mlir::FunctionOpInterface>{Iter->second};
  };
  switch (context) {
  case FunctionContext::Host:
    return find(functions);
  case FunctionContext::SYCLDevice:
    return find(deviceFunctions);
  }
  llvm_unreachable("Invalid function context");
}

#include "clang/Frontend/FrontendAction.h"
#include "llvm/Support/Host.h"

class MLIRAction : public clang::ASTFrontendAction {
public:
  std::set<std::string> emitIfFound;
  std::set<std::pair<FunctionContext, std::string>> done;
  mlir::OwningOpRef<mlir::ModuleOp> &module;
  std::map<std::string, mlir::LLVM::GlobalOp> llvmStringGlobals;
  std::map<std::string, std::pair<mlir::memref::GlobalOp, bool>> globals;
  std::map<std::string, mlir::func::FuncOp> functions;
  std::map<std::string, mlir::FunctionOpInterface> deviceFunctions;
  std::map<std::string, mlir::LLVM::GlobalOp> llvmGlobals;
  std::map<std::string, mlir::LLVM::LLVMFuncOp> llvmFunctions;
  std::string moduleId;
  MLIRAction(std::string fn, mlir::OwningOpRef<mlir::ModuleOp> &module,
             std::string moduleId)
      : module(module), moduleId(moduleId) {
    emitIfFound.insert(fn);
  }
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
    return std::unique_ptr<clang::ASTConsumer>(new MLIRASTConsumer(
        emitIfFound, done, llvmStringGlobals, globals, functions,
        deviceFunctions, llvmGlobals, llvmFunctions, CI.getPreprocessor(),
        CI.getASTContext(), module, CI.getSourceManager(), CI.getCodeGenOpts(),
        moduleId));
  }
};

mlir::FunctionOpInterface
MLIRScanner::EmitDirectCallee(const FunctionDecl *FD, FunctionContext Context) {
  FunctionToEmit FTE(*FD, Context);
  return Glob.GetOrCreateMLIRFunction(FTE, true /* ShouldEmit */);
}

mlir::Location MLIRScanner::getMLIRLocation(clang::SourceLocation loc) {
  return Glob.getMLIRLocation(loc);
}

mlir::Value MLIRScanner::getTypeSize(clang::QualType t) {
  // llvm::Type *T = Glob.getCGM().getTypes().ConvertType(t);
  // return (Glob.llvmMod.getDataLayout().getTypeSizeInBits(T) + 7) / 8;
  bool isArray = false;
  auto innerTy = Glob.getTypes().getMLIRType(t, &isArray);
  if (isArray) {
    auto MT = innerTy.cast<MemRefType>();
    size_t num = 1;
    for (auto n : MT.getShape()) {
      assert(n > 0);
      num *= n;
    }
    return builder.create<arith::MulIOp>(
        loc,
        builder.create<polygeist::TypeSizeOp>(
            loc, builder.getIndexType(),
            mlir::TypeAttr::get(MT.getElementType())),
        builder.create<arith::ConstantIndexOp>(loc, num));
  }
  assert(!isArray);
  return builder.create<polygeist::TypeSizeOp>(
      loc, builder.getIndexType(),
      mlir::TypeAttr::get(innerTy)); // DLI.getTypeSize(innerTy);
}

mlir::Value MLIRScanner::getTypeAlign(clang::QualType t) {
  // llvm::Type *T = Glob.getCGM().getTypes().ConvertType(t);
  // return (Glob.llvmMod.getDataLayout().getTypeSizeInBits(T) + 7) / 8;
  bool isArray = false;
  auto innerTy = Glob.getTypes().getMLIRType(t, &isArray);
  assert(!isArray);
  return builder.create<polygeist::TypeAlignOp>(
      loc, builder.getIndexType(),
      mlir::TypeAttr::get(innerTy)); // DLI.getTypeSize(innerTy);
}

std::string
MLIRScanner::getMangledFuncName(const FunctionDecl &FD,
                                clang::CodeGen::CodeGenModule &CGM) {
  if (auto CC = dyn_cast<CXXConstructorDecl>(&FD))
    return CGM.getMangledName(GlobalDecl(CC, CXXCtorType::Ctor_Complete)).str();
  if (auto CC = dyn_cast<CXXDestructorDecl>(&FD))
    return CGM.getMangledName(GlobalDecl(CC, CXXDtorType::Dtor_Complete)).str();

  return CGM.getMangledName(&FD).str();
}

#include "clang/Frontend/TextDiagnosticBuffer.h"
static bool parseMLIR(const char *Argv0, std::vector<std::string> filenames,
                      std::string fn, std::vector<std::string> includeDirs,
                      std::vector<std::string> defines,
                      mlir::OwningOpRef<mlir::ModuleOp> &module,
                      llvm::Triple &triple, llvm::DataLayout &DL,
                      std::vector<std::string> InputCommandArgs) {
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  // Buffer diagnostics from argument parsing so that we can output them using
  // a well formed diagnostic object.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagsBuffer);

  bool Success;
  //{
  const char *binary = Argv0;
  const std::unique_ptr<Driver> driver(
      new Driver(binary, llvm::sys::getDefaultTargetTriple(), Diags));
  ArgumentList Argv;
  Argv.push_back(binary);
  for (const auto &filename : filenames) {
    Argv.push_back(filename);
  }
  if (FOpenMP)
    Argv.push_back("-fopenmp");
  if (TargetTripleOpt != "") {
    Argv.push_back("-target");
    Argv.push_back(TargetTripleOpt);
  }
  if (McpuOpt != "") {
    Argv.emplace_back("-mcpu=", McpuOpt);
  }
  if (Standard != "") {
    Argv.emplace_back("-std=", Standard);
  }
  if (ResourceDir != "") {
    Argv.push_back("-resource-dir");
    Argv.push_back(ResourceDir);
  }
  if (SysRoot != "") {
    Argv.push_back("--sysroot");
    Argv.push_back(SysRoot);
  }
  if (Verbose) {
    Argv.push_back("-v");
  }
  if (NoCUDAInc) {
    Argv.push_back("-nocudainc");
  }
  if (NoCUDALib) {
    Argv.push_back("-nocudalib");
  }
  if (CUDAGPUArch != "") {
    Argv.emplace_back("--cuda-gpu-arch=", CUDAGPUArch);
  }
  if (CUDAPath != "") {
    Argv.emplace_back("--cuda-path=", CUDAPath);
  }
  if (MArch != "") {
    Argv.emplace_back("-march=", MArch);
  }
  for (const auto &dir : includeDirs) {
    Argv.push_back("-I");
    Argv.push_back(dir);
  }
  for (const auto &define : defines) {
    Argv.emplace_back("-D", define);
  }
  for (const auto &include : Includes) {
    Argv.push_back("-include");
    Argv.push_back(include);
  }

  Argv.push_back("-emit-ast");

  llvm::SmallVector<const ArgStringList *, 4> CommandList;
  ArgStringList InputCommandArgList;

  std::unique_ptr<Compilation> compilation;

  if (InputCommandArgs.empty()) {
    compilation.reset(driver->BuildCompilation(Argv.getArguments()));

    JobList &Jobs = compilation->getJobs();
    if (Jobs.size() < 1)
      return false;
    for (auto &job : Jobs) {
      Command *cmd = cast<Command>(&job);
      if (strcmp(cmd->getCreator().getName(), "clang"))
        return false;
      CommandList.push_back(&cmd->getArguments());
    }
  } else {
    for (std::string &s : InputCommandArgs) {
      InputCommandArgList.push_back(s.c_str());
    }
    CommandList.push_back(&InputCommandArgList);
  }

  MLIRAction Act(fn, module,
                 filenames.size() == 1 ? filenames[0] : "LLVMDialectModule");

  for (const ArgStringList *args : CommandList) {
    std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());

    Success = CompilerInvocation::CreateFromArgs(Clang->getInvocation(), *args,
                                                 Diags);
    Clang->getInvocation().getFrontendOpts().DisableFree = false;

    void *GetExecutablePathVP = (void *)(intptr_t)GetExecutablePath;
    // Infer the builtin include path if unspecified.
    if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
        Clang->getHeaderSearchOpts().ResourceDir.size() == 0)
      Clang->getHeaderSearchOpts().ResourceDir =
          CompilerInvocation::GetResourcesPath(Argv0, GetExecutablePathVP);

    //}
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
    Clang->setTarget(TargetInfo::CreateTargetInfo(
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
          TargetInfo::CreateTargetInfo(Clang->getDiagnostics(), TO));
    }

    // Inform the target of the language options.
    //
    // FIXME: We shouldn't need to do this, the target should be immutable once
    // created. This complexity should be lifted elsewhere.
    Clang->getTarget().adjust(Clang->getDiagnostics(), Clang->getLangOpts());

    // Adjust target options based on codegen options.
    Clang->getTarget().adjustTargetOptions(Clang->getCodeGenOpts(),
                                           Clang->getTargetOpts());

    llvm::Triple jobTriple = Clang->getTarget().getTriple();
    if (triple.str() == "" || !jobTriple.isNVPTX()) {
      triple = jobTriple;
      module.get()->setAttr(
          LLVM::LLVMDialect::getTargetTripleAttrName(),
          StringAttr::get(module->getContext(),
                          Clang->getTarget().getTriple().getTriple()));
      DL = llvm::DataLayout(Clang->getTarget().getDataLayoutString());
      module.get()->setAttr(
          LLVM::LLVMDialect::getDataLayoutAttrName(),
          StringAttr::get(module->getContext(),
                          Clang->getTarget().getDataLayoutString()));

      module.get()->setAttr(("dlti." + DataLayoutSpecAttr::kAttrKeyword).str(),
                            translateDataLayout(DL, module->getContext()));
    }

    for (const auto &FIF : Clang->getFrontendOpts().Inputs) {
      // Reset the ID tables if we are reusing the SourceManager and parsing
      // regular files.
      if (Clang->hasSourceManager() && !Act.isModelParsingAction())
        Clang->getSourceManager().clearIDTables();
      if (Act.BeginSourceFile(*Clang, FIF)) {

        llvm::Error err = Act.Execute();
        if (err) {
          llvm::errs() << "saw error: " << err << "\n";
          return false;
        }
        assert(Clang->hasSourceManager());

        Act.EndSourceFile();
      }
    }
  }
  return true;
}
