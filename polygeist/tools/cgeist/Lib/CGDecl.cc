//===--- CGDecl.cc - Emit LLVM Code for declarations ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Attributes.h"
#include "CodeGenTypes.h"
#include "TypeUtils.h"
#include "clang-mlir.h"
#include "utils.h"

#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"
#include "mlir/Target/LLVMIR/TypeFromLLVM.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/WithColor.h"

#define DEBUG_TYPE "CGDecl"

using namespace clang;
using namespace mlir;

ValueCategory MLIRScanner::VisitVarDecl(clang::VarDecl *Decl) {
  Decl = Decl->getCanonicalDecl();
  mlir::Type SubType = Glob.getTypes().getMLIRType(Decl->getType());
  const unsigned MemType = Decl->hasAttr<clang::CUDASharedAttr>() ? 5 : 0;
  bool LLVMABI = false, IsArray = false;

  if (Glob.getTypes()
          .getMLIRType(Glob.getCGM().getContext().getLValueReferenceType(
              Decl->getType()))
          .isa<LLVM::LLVMPointerType>())
    LLVMABI = true;
  else
    Glob.getTypes().getMLIRType(Decl->getType(), &IsArray);

  if (!LLVMABI && IsArray)
    SubType = Glob.getTypes().getMLIRType(
        Glob.getCGM().getContext().getLValueReferenceType(Decl->getType()));

  LLVM::TypeFromLLVMIRTranslator TypeTranslator(*Module->getContext());
  ValueCategory InitExpr = nullptr;

  if (Expr *Init = Decl->getInit()) {
    bool IsVectorType =
        Glob.getTypes().getMLIRType(Init->getType()).isa<mlir::VectorType>();
    if ((!isa<clang::InitListExpr>(Init) || IsVectorType) &&
        !isa<clang::CXXConstructExpr>(Init)) {
      auto Res = Visit(Init);
      if (!Res.val) {
        Decl->dump();
        assert(Res.val);
      }

      bool IsReference = Init->isLValue() || Init->isXValue();
      if (IsReference) {
        assert(Res.isReference);
        Builder.create<polygeist::TrivialUseOp>(Loc, Res.val);
        return Params[Decl] = Res;
      }

      if (IsArray) {
        assert(Res.isReference);
        InitExpr = Res;
      } else {
        InitExpr = ValueCategory(Res.getValue(Builder), /*isRef*/ false);
        if (!InitExpr.val) {
          Init->dump();
          assert(false);
        }
        SubType = InitExpr.val.getType();
      }
    }
  } else if (auto *Ava = Decl->getAttr<clang::AlignValueAttr>()) {
    if (auto *Algn = dyn_cast<clang::ConstantExpr>(Ava->getAlignment())) {
      for (auto *A : Algn->children()) {
        if (auto *IL = dyn_cast<clang::IntegerLiteral>(A)) {
          if (IL->getValue() == 8192) {
            llvm::Type *T =
                Glob.getCGM().getTypes().ConvertType(Decl->getType());
            SubType = TypeTranslator.translateType(T);
            LLVMABI = true;
            break;
          }
        }
      }
    }
  } else if (auto *Ava = Decl->getAttr<clang::InitPriorityAttr>()) {
    if (Ava->getPriority() == 8192) {
      llvm::Type *T = Glob.getCGM().getTypes().ConvertType(Decl->getType());
      SubType = TypeTranslator.translateType(T);
      LLVMABI = true;
    }
  }

  Block *Block = nullptr;
  Block::iterator Iter;
  Value Op = createAllocOp(SubType, Decl, MemType, IsArray, LLVMABI);

  if (Decl->isStaticLocal() && MemType == 0) {
    OpBuilder ABuilder(Builder.getContext());
    ABuilder.setInsertionPointToStart(AllocationScope);
    Location VarLoc = getMLIRLocation(Decl->getBeginLoc());

    if (Glob.getTypes()
            .getMLIRType(
                Glob.getCGM().getContext().getPointerType(Decl->getType()))
            .isa<LLVM::LLVMPointerType>())
      Op = ABuilder.create<LLVM::AddressOfOp>(
          VarLoc, Glob.getOrCreateLLVMGlobal(
                      Decl, (Function.getName() + "@static@").str()));
    else {
      auto GV =
          Glob.getOrCreateGlobal(*Decl, (Function.getName() + "@static@").str(),
                                 FunctionContext::Host);
      auto GV2 = ABuilder.create<memref::GetGlobalOp>(
          VarLoc, GV.first.getType(), GV.first.getName());
      Op = reshapeRanklessGlobal(GV2);
    }

    Params[Decl] = ValueCategory(Op, /*isReference*/ true);
    if (Decl->getInit()) {
      auto MR = MemRefType::get({}, Builder.getI1Type());
      auto RTT = RankedTensorType::get({}, Builder.getI1Type());
      auto InitValue = DenseIntElementsAttr::get(RTT, {true});
      OpBuilder GBuilder(Builder.getContext());
      GBuilder.setInsertionPointToStart(Module->getBody());
      StringRef Name = Glob.getCGM().getMangledName(Decl);
      auto GlobalOp = GBuilder.create<memref::GlobalOp>(
          Module->getLoc(),
          Builder.getStringAttr(Function.getName() + "@static@" + Name +
                                "@init"),
          /*sym_visibility*/ StringAttr(), mlir::TypeAttr::get(MR), InitValue,
          UnitAttr(), /*alignment*/ nullptr);
      SymbolTable::setSymbolVisibility(GlobalOp,
                                       SymbolTable::Visibility::Private);

      auto BoolOp =
          Builder.create<memref::GetGlobalOp>(VarLoc, MR, GlobalOp.getName());
      Value V = reshapeRanklessGlobal(BoolOp);
      auto Cond = Builder.create<memref::LoadOp>(
          VarLoc, V, std::vector<Value>({getConstantIndex(0)}));

      auto IfOp = Builder.create<scf::IfOp>(VarLoc, Cond, /*hasElse*/ false);
      Block = Builder.getInsertionBlock();
      Iter = Builder.getInsertionPoint();
      Builder.setInsertionPointToStart(&IfOp.getThenRegion().back());
      Builder.create<memref::StoreOp>(
          VarLoc, Builder.create<arith::ConstantIntOp>(VarLoc, false, 1), V,
          std::vector<Value>({getConstantIndex(0)}));
    }
  }

  if (InitExpr.val)
    ValueCategory(Op, /*isReference*/ true).store(Builder, InitExpr, IsArray);
  else if (auto *Init = Decl->getInit()) {
    if (isa<clang::InitListExpr>(Init))
      InitializeValueByInitListExpr(Op, Init);
    else if (auto *CE = dyn_cast<clang::CXXConstructExpr>(Init))
      VisitConstructCommon(CE, Decl, MemType, Op);
    else
      llvm_unreachable("unknown init list");
  }

  if (Block)
    Builder.setInsertionPoint(Block, Iter);

  return ValueCategory(Op, /*isReference*/ true);
}
