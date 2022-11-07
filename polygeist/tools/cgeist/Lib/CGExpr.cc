//===--- CGExpr.cc - Emit MLIR Code from Expressions ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-mlir.h"
#include "utils.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"

#define DEBUG_TYPE "CGExpr"

using namespace clang;
using namespace mlir;
using namespace mlir::arith;

extern llvm::cl::opt<bool> GenerateAllSYCLFuncs;

ValueCategory
MLIRScanner::VisitExtVectorElementExpr(clang::ExtVectorElementExpr *expr) {
  auto base = Visit(expr->getBase());
  SmallVector<uint32_t, 4> indices;
  expr->getEncodedElementAccess(indices);
  assert(indices.size() == 1 &&
         "The support for higher dimensions to be implemented.");
  auto idx = castToIndex(getMLIRLocation(expr->getAccessorLoc()),
                         builder.create<ConstantIntOp>(loc, indices[0], 32));
  assert(base.isReference);
  base.isReference = false;
  auto mt = base.val.getType().cast<MemRefType>();
  auto shape = std::vector<int64_t>(mt.getShape());
  if (shape.size() == 1) {
    shape[0] = -1;
  } else {
    shape.erase(shape.begin());
  }
  auto mt0 =
      mlir::MemRefType::get(shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(), mt.getMemorySpace());
  base.val = builder.create<polygeist::SubIndexOp>(loc, mt0, base.val,
                                                   getConstantIndex(0));
  return CommonArrayLookup(base, idx, base.isReference);
}

ValueCategory MLIRScanner::VisitConstantExpr(clang::ConstantExpr *expr) {
  auto sv = Visit(expr->getSubExpr());
  if (auto ty = Glob.getTypes()
                    .getMLIRType(expr->getType())
                    .dyn_cast<mlir::IntegerType>()) {
    if (expr->hasAPValueResult()) {
      return ValueCategory(builder.create<arith::ConstantIntOp>(
                               getMLIRLocation(expr->getExprLoc()),
                               expr->getResultAsAPSInt().getExtValue(), ty),
                           /*isReference*/ false);
    }
  }
  assert(sv.val);
  return sv;
}

ValueCategory MLIRScanner::VisitTypeTraitExpr(clang::TypeTraitExpr *expr) {
  auto ty =
      Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(
      builder.create<arith::ConstantIntOp>(getMLIRLocation(expr->getExprLoc()),
                                           expr->getValue(), ty),
      /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitGNUNullExpr(clang::GNUNullExpr *expr) {
  auto ty =
      Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(builder.create<arith::ConstantIntOp>(
                           getMLIRLocation(expr->getExprLoc()), 0, ty),
                       /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitIntegerLiteral(clang::IntegerLiteral *expr) {
  auto ty =
      Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(
      builder.create<arith::ConstantIntOp>(getMLIRLocation(expr->getExprLoc()),
                                           expr->getValue().getSExtValue(), ty),
      /*isReference*/ false);
}

ValueCategory
MLIRScanner::VisitCharacterLiteral(clang::CharacterLiteral *expr) {
  auto ty =
      Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(
      builder.create<arith::ConstantIntOp>(getMLIRLocation(expr->getExprLoc()),
                                           expr->getValue(), ty),
      /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitFloatingLiteral(clang::FloatingLiteral *expr) {
  auto ty =
      Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::FloatType>();
  return ValueCategory(
      builder.create<ConstantFloatOp>(getMLIRLocation(expr->getExprLoc()),
                                      expr->getValue(), ty),
      /*isReference*/ false);
}

ValueCategory
MLIRScanner::VisitImaginaryLiteral(clang::ImaginaryLiteral *expr) {
  auto mt = Glob.getTypes().getMLIRType(expr->getType()).cast<MemRefType>();
  auto ty = mt.getElementType().cast<FloatType>();

  OpBuilder abuilder(builder.getContext());
  abuilder.setInsertionPointToStart(allocationScope);
  auto iloc = getMLIRLocation(expr->getExprLoc());
  auto alloc = abuilder.create<mlir::memref::AllocaOp>(iloc, mt);
  builder.create<mlir::memref::StoreOp>(
      iloc,
      builder.create<ConstantFloatOp>(iloc,
                                      APFloat(ty.getFloatSemantics(), "0"), ty),
      alloc, getConstantIndex(0));
  builder.create<mlir::memref::StoreOp>(
      iloc, Visit(expr->getSubExpr()).getValue(builder), alloc,
      getConstantIndex(1));
  return ValueCategory(alloc,
                       /*isReference*/ true);
}

ValueCategory
MLIRScanner::VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *expr) {
  auto ty =
      Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(
      builder.create<ConstantIntOp>(getMLIRLocation(expr->getExprLoc()),
                                    expr->getValue(), ty),
      /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitStringLiteral(clang::StringLiteral *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  return ValueCategory(
      Glob.GetOrCreateGlobalLLVMString(loc, builder, expr->getString()),
      /*isReference*/ true);
}

ValueCategory MLIRScanner::VisitParenExpr(clang::ParenExpr *expr) {
  return Visit(expr->getSubExpr());
}

ValueCategory
MLIRScanner::VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *decl) {
  mlir::Type Mty = Glob.getTypes().getMLIRType(decl->getType());

  if (auto FT = Mty.dyn_cast<mlir::FloatType>())
    return ValueCategory(builder.create<ConstantFloatOp>(
                             loc, APFloat(FT.getFloatSemantics(), "0"), FT),
                         /*isReference*/ false);
  if (auto IT = Mty.dyn_cast<mlir::IntegerType>())
    return ValueCategory(builder.create<ConstantIntOp>(loc, 0, IT),
                         /*isReference*/ false);
  if (auto MT = Mty.dyn_cast<mlir::MemRefType>())
    return ValueCategory(
        builder.create<polygeist::Pointer2MemrefOp>(
            loc, MT,
            builder.create<mlir::LLVM::NullOp>(
                loc, LLVM::LLVMPointerType::get(builder.getI8Type(),
                                                MT.getMemorySpaceAsInt()))),
        false);
  if (auto PT = Mty.dyn_cast<mlir::LLVM::LLVMPointerType>())
    return ValueCategory(builder.create<mlir::LLVM::NullOp>(loc, PT), false);
  for (auto child : decl->children()) {
    child->dump();
  }
  decl->dump();
  llvm::errs() << " mty: " << Mty << "\n";
  assert(0 && "bad");
}

/// Construct corresponding MLIR operations to initialize the given value by a
/// provided InitListExpr.
mlir::Attribute MLIRScanner::InitializeValueByInitListExpr(mlir::Value toInit,
                                                           clang::Expr *expr) {
  // Struct initializan requires an extra 0, since the first index
  // is the pointer index, and then the struct index.
  auto PTT = expr->getType()->getUnqualifiedDesugaredType();

  bool inner = false;
  if (isa<RecordType>(PTT) || isa<clang::ComplexType>(PTT)) {
    if (auto mt = toInit.getType().dyn_cast<MemRefType>()) {
      inner = true;
    }
  }

  while (auto CO = toInit.getDefiningOp<memref::CastOp>())
    toInit = CO.getSource();

  // Recursively visit the initialization expression following the linear
  // increment of the memory address.
  std::function<mlir::DenseElementsAttr(Expr *, mlir::Value, bool)> helper =
      [&](Expr *expr, mlir::Value toInit,
          bool inner) -> mlir::DenseElementsAttr {
    Location loc = toInit.getLoc();
    if (InitListExpr *initListExpr = dyn_cast<InitListExpr>(expr)) {

      if (inner) {
        if (auto mt = toInit.getType().dyn_cast<MemRefType>()) {
          auto shape = std::vector<int64_t>(mt.getShape());
          assert(!shape.empty());
          if (shape.size() > 1) {
            shape.erase(shape.begin());
          } else {
            shape[0] = -1;
          }
          auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                           MemRefLayoutAttrInterface(),
                                           mt.getMemorySpace());
          toInit = builder.create<polygeist::SubIndexOp>(loc, mt0, toInit,
                                                         getConstantIndex(0));
        }
      }

      unsigned num = 0;
      if (initListExpr->hasArrayFiller()) {
        if (auto MT = toInit.getType().dyn_cast<MemRefType>()) {
          auto shape = MT.getShape();
          assert(shape.size() > 0);
          assert(shape[0] != -1);
          num = shape[0];
        } else if (auto PT =
                       toInit.getType().dyn_cast<LLVM::LLVMPointerType>()) {
          if (auto AT = PT.getElementType().dyn_cast<LLVM::LLVMArrayType>()) {
            num = AT.getNumElements();
          } else if (auto AT =
                         PT.getElementType().dyn_cast<LLVM::LLVMStructType>()) {
            num = AT.getBody().size();
          } else {
            toInit.getType().dump();
            assert(0 && "TODO get number of values in array filler expression");
          }
        } else {
          toInit.getType().dump();
          assert(0 && "TODO get number of values in array filler expression");
        }
      } else {
        num = initListExpr->getNumInits();
      }

      SmallVector<char> attrs;
      bool allSub = true;
      for (unsigned i = 0, e = num; i < e; ++i) {

        mlir::Value next;
        if (auto mt = toInit.getType().dyn_cast<MemRefType>()) {
          auto shape = std::vector<int64_t>(mt.getShape());
          assert(!shape.empty());
          shape[0] = -1;

          if (mt.getElementType()
                  .isa<mlir::sycl::AccessorType,
                       mlir::sycl::AccessorImplDeviceType,
                       mlir::sycl::ArrayType, mlir::sycl::ItemType,
                       mlir::sycl::NdItemType, mlir::sycl::GroupType>()) {
            llvm_unreachable("not implemented yet");
          }

          mlir::Type ET;
          if (auto ST =
                  mt.getElementType().dyn_cast<mlir::LLVM::LLVMStructType>()) {
            ET = mlir::MemRefType::get(shape, ST.getBody()[i],
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
          } else if (auto ST = mt.getElementType()
                                   .dyn_cast<mlir::sycl::ItemBaseType>()) {
            ET = mlir::MemRefType::get(shape, ST.getBody()[i],
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
          } else {
            ET = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
          }
          next = builder.create<polygeist::SubIndexOp>(loc, ET, toInit,
                                                       getConstantIndex(i));
        } else {
          auto PT = toInit.getType().cast<LLVM::LLVMPointerType>();
          auto ET = PT.getElementType();
          mlir::Type nextType;
          if (auto ST = ET.dyn_cast<LLVM::LLVMStructType>())
            nextType = ST.getBody()[i];
          else if (auto AT = ET.dyn_cast<LLVM::LLVMArrayType>())
            nextType = AT.getElementType();
          else
            assert(0 && "unknown inner type");

          mlir::Value idxs[] = {
              builder.create<ConstantIntOp>(loc, 0, 32),
              builder.create<ConstantIntOp>(loc, i, 32),
          };
          next = builder.create<LLVM::GEPOp>(
              loc, LLVM::LLVMPointerType::get(nextType, PT.getAddressSpace()),
              toInit, idxs);
        }

        auto sub =
            helper(initListExpr->hasArrayFiller() ? initListExpr->getInit(0)
                                                  : initListExpr->getInit(i),
                   next, true);
        if (sub) {
          size_t n = 1;
          if (sub.isSplat())
            n = sub.size();
          for (size_t i = 0; i < n; i++)
            for (auto ea : sub.getRawData())
              attrs.push_back(ea);
        } else {
          allSub = false;
        }
      }
      if (!allSub)
        return mlir::DenseElementsAttr();
      if (auto mt = toInit.getType().dyn_cast<MemRefType>()) {
        return DenseElementsAttr::getFromRawBuffer(
            RankedTensorType::get(mt.getShape(), mt.getElementType()), attrs);
      }
      return mlir::DenseElementsAttr();
    } else {
      bool isArray = false;
      Glob.getTypes().getMLIRType(expr->getType(), &isArray);
      ValueCategory sub = Visit(expr);
      ValueCategory(toInit, /*isReference*/ true).store(builder, sub, isArray);
      if (!sub.isReference)
        if (auto mt = toInit.getType().dyn_cast<MemRefType>()) {
          if (auto cop = sub.val.getDefiningOp<ConstantIntOp>())
            return DenseElementsAttr::get(
                RankedTensorType::get(std::vector<int64_t>({1}),
                                      mt.getElementType()),
                cop.getValue());
          if (auto cop = sub.val.getDefiningOp<ConstantFloatOp>())
            return DenseElementsAttr::get(
                RankedTensorType::get(std::vector<int64_t>({1}),
                                      mt.getElementType()),
                cop.getValue());
        }
      return mlir::DenseElementsAttr();
    }
  };

  return helper(expr, toInit, inner);
}

ValueCategory
MLIRScanner::VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr *expr) {
  return Visit(expr->getExpr());
}

ValueCategory MLIRScanner::VisitCXXThisExpr(clang::CXXThisExpr *expr) {
  return ThisVal;
}

ValueCategory MLIRScanner::VisitPredefinedExpr(clang::PredefinedExpr *expr) {
  return VisitStringLiteral(expr->getFunctionName());
}

ValueCategory MLIRScanner::VisitInitListExpr(clang::InitListExpr *expr) {
  mlir::Type subType = Glob.getTypes().getMLIRType(expr->getType());
  bool isArray = false;
  bool LLVMABI = false;

  if (Glob.getTypes()
          .getMLIRType(Glob.getCGM().getContext().getLValueReferenceType(
              expr->getType()))
          .isa<mlir::LLVM::LLVMPointerType>())
    LLVMABI = true;
  else {
    Glob.getTypes().getMLIRType(expr->getType(), &isArray);
    if (isArray)
      subType = Glob.getTypes().getMLIRType(
          Glob.getCGM().getContext().getLValueReferenceType(expr->getType()));
  }
  auto op = createAllocOp(subType, nullptr, /*memtype*/ 0, isArray, LLVMABI);
  InitializeValueByInitListExpr(op, expr);
  return ValueCategory(op, true);
}

ValueCategory MLIRScanner::VisitCXXStdInitializerListExpr(
    clang::CXXStdInitializerListExpr *expr) {

  auto ArrayPtr = Visit(expr->getSubExpr());

  const ConstantArrayType *ArrayType =
      Glob.getCGM().getContext().getAsConstantArrayType(
          expr->getSubExpr()->getType());
  assert(ArrayType && "std::initializer_list constructed from non-array");

  // FIXME: Perform the checks on the field types in SemaInit.
  RecordDecl *Record = expr->getType()->castAs<RecordType>()->getDecl();
  auto Field = Record->field_begin();

  mlir::Type subType = Glob.getTypes().getMLIRType(expr->getType());

  mlir::Value res = builder.create<LLVM::UndefOp>(loc, subType);

  ArrayPtr = CommonArrayToPointer(ArrayPtr);

  res = builder.create<LLVM::InsertValueOp>(loc, res,
                                            ArrayPtr.getValue(builder), 0);
  Field++;
  auto iTy =
      Glob.getTypes().getMLIRType(Field->getType()).cast<mlir::IntegerType>();
  res = builder.create<LLVM::InsertValueOp>(
      loc, res,
      builder.create<arith::ConstantIntOp>(
          loc, ArrayType->getSize().getZExtValue(), iTy.getWidth()),
      1);
  return ValueCategory(res, /*isRef*/ false);
}

ValueCategory
MLIRScanner::VisitArrayInitIndexExpr(clang::ArrayInitIndexExpr *expr) {
  assert(arrayinit.size());
  return ValueCategory(
      builder.create<IndexCastOp>(
          loc, Glob.getTypes().getMLIRType(expr->getType()), arrayinit.back()),
      /*isReference*/ false);
}

static const clang::ConstantArrayType *getCAT(const clang::Type *T) {
  const clang::Type *Child;
  if (auto CAT = dyn_cast<clang::ConstantArrayType>(T)) {
    return CAT;
  } else if (auto ET = dyn_cast<clang::ElaboratedType>(T)) {
    Child = ET->getNamedType().getTypePtr();
  } else if (auto TypeDefT = dyn_cast<clang::TypedefType>(T)) {
    Child = TypeDefT->getUnqualifiedDesugaredType();
  } else {
    llvm_unreachable("Unhandled case\n");
  }
  return getCAT(Child);
}

ValueCategory MLIRScanner::VisitArrayInitLoop(clang::ArrayInitLoopExpr *expr,
                                              ValueCategory tostore) {
  const clang::ConstantArrayType *CAT = getCAT(expr->getType().getTypePtr());
  llvm::errs() << "warning recomputing common in arrayinitloopexpr\n";
  std::vector<mlir::Value> start = {getConstantIndex(0)};
  std::vector<mlir::Value> sizes = {
      getConstantIndex(CAT->getSize().getLimitedValue())};
  AffineMap map = builder.getSymbolIdentityMap();
  auto affineOp = builder.create<AffineForOp>(loc, start, map, sizes, map);

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();

  builder.setInsertionPointToStart(&affineOp.getLoopBody().front());

  arrayinit.push_back(affineOp.getInductionVar());

  auto alu =
      CommonArrayLookup(CommonArrayToPointer(tostore),
                        affineOp.getInductionVar(), /*isImplicitRef*/ false);

  if (auto AILE = dyn_cast<ArrayInitLoopExpr>(expr->getSubExpr())) {
    VisitArrayInitLoop(AILE, alu);
  } else {
    auto val = Visit(expr->getSubExpr());
    if (!val.val) {
      expr->dump();
      expr->getSubExpr()->dump();
    }
    assert(val.val);
    assert(tostore.isReference);
    bool isArray = false;
    Glob.getTypes().getMLIRType(expr->getSubExpr()->getType(), &isArray);
    alu.store(builder, val, isArray);
  }

  arrayinit.pop_back();

  builder.setInsertionPoint(oldblock, oldpoint);
  return nullptr;
}

ValueCategory
MLIRScanner::VisitCXXFunctionalCastExpr(clang::CXXFunctionalCastExpr *expr) {
  if (expr->getType()->isVoidType()) {
    Visit(expr->getSubExpr());
    return nullptr;
  }
  if (expr->getCastKind() == clang::CastKind::CK_NoOp)
    return Visit(expr->getSubExpr());
  if (expr->getCastKind() == clang::CastKind::CK_ConstructorConversion)
    return Visit(expr->getSubExpr());
  return VisitCastExpr(expr);
}

ValueCategory
MLIRScanner::VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *expr) {
  return Visit(expr->getSubExpr());
}

ValueCategory MLIRScanner::VisitLambdaExpr(clang::LambdaExpr *expr) {

  // llvm::DenseMap<const VarDecl *, FieldDecl *> InnerCaptures;
  // FieldDecl *ThisCapture = nullptr;

  // expr->getLambdaClass()->getCaptureFields(InnerCaptures, ThisCapture);

  bool LLVMABI = false;
  mlir::Type t =
      Glob.getTypes().getMLIRType(expr->getCallOperator()->getThisType());

  bool isArray =
      false; // isa<clang::ArrayType>(expr->getCallOperator()->getThisType());
  Glob.getTypes().getMLIRType(expr->getCallOperator()->getThisObjectType(),
                              &isArray);

  if (auto PT = t.dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    LLVMABI = true;
    t = PT.getElementType();
  }
  if (auto mt = t.dyn_cast<MemRefType>()) {
    auto shape = std::vector<int64_t>(mt.getShape());
    if (!isArray)
      shape[0] = 1;
    t = mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
  }
  auto op = createAllocOp(t, nullptr, /*memtype*/ 0, isArray, LLVMABI);

  for (auto tup : llvm::zip(expr->getLambdaClass()->captures(),
                            expr->getLambdaClass()->fields())) {
    auto C = std::get<0>(tup);
    auto field = std::get<1>(tup);
    if (C.capturesThis())
      continue;
    else if (!C.capturesVariable())
      continue;

    auto CK = C.getCaptureKind();
    auto var = C.getCapturedVar();

    ValueCategory result;

    if (params.find(var) != params.end()) {
      result = params[var];
    } else {
      if (auto VD = dyn_cast<VarDecl>(var)) {
        if (Captures.find(VD) != Captures.end()) {
          FieldDecl *field = Captures[VD];
          result = CommonFieldLookup(
              cast<CXXMethodDecl>(EmittingFunctionDecl)->getThisObjectType(),
              field, ThisVal.val, /*isLValue*/ false);
          assert(CaptureKinds.find(VD) != CaptureKinds.end());
          if (CaptureKinds[VD] == LambdaCaptureKind::LCK_ByRef)
            result = result.dereference(builder);
          goto endp;
        }
      }
      EmittingFunctionDecl->dump();
      expr->dump();
      function.dump();
      llvm::errs() << "<pairs>\n";
      for (auto p : params)
        p.first->dump();
      llvm::errs() << "</pairs>";
      var->dump();
    }
  endp:

    bool isArray = false;
    Glob.getTypes().getMLIRType(field->getType(), &isArray);

    if (CK == LambdaCaptureKind::LCK_ByCopy)
      CommonFieldLookup(expr->getCallOperator()->getThisObjectType(), field, op,
                        /*isLValue*/ false)
          .store(builder, result, isArray);
    else {
      assert(CK == LambdaCaptureKind::LCK_ByRef);
      assert(result.isReference);

      auto val = result.val;

      if (auto mt = val.getType().dyn_cast<MemRefType>()) {
        auto shape = std::vector<int64_t>(mt.getShape());
        shape[0] = -1;
        val = builder.create<memref::CastOp>(
            loc,
            MemRefType::get(shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(), mt.getMemorySpace()),
            val);
      }

      CommonFieldLookup(expr->getCallOperator()->getThisObjectType(), field, op,
                        /*isLValue*/ false)
          .store(builder, val);
    }
  }
  return ValueCategory(op, /*isReference*/ true);
}

// TODO actually deallocate
ValueCategory MLIRScanner::VisitMaterializeTemporaryExpr(
    clang::MaterializeTemporaryExpr *expr) {
  auto v = Visit(expr->getSubExpr());
  if (!v.val) {
    expr->dump();
  }
  assert(v.val);

  bool isArray = false;
  bool LLVMABI = false;
  if (Glob.getTypes()
          .getMLIRType(Glob.getCGM().getContext().getLValueReferenceType(
              expr->getSubExpr()->getType()))
          .isa<mlir::LLVM::LLVMPointerType>())
    LLVMABI = true;
  else {
    Glob.getTypes().getMLIRType(expr->getSubExpr()->getType(), &isArray);
  }
  if (isArray)
    return v;

  llvm::errs() << "cleanup of materialized not handled";
  auto op =
      createAllocOp(Glob.getTypes().getMLIRType(expr->getSubExpr()->getType()),
                    nullptr, 0, /*isArray*/ isArray, /*LLVMABI*/ LLVMABI);

  ValueCategory(op, /*isRefererence*/ true).store(builder, v, isArray);
  return ValueCategory(op, /*isRefererence*/ true);
}

ValueCategory MLIRScanner::VisitCXXDeleteExpr(clang::CXXDeleteExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  expr->dump();
  llvm::errs() << "warning not calling destructor on delete\n";

  mlir::Value toDelete = Visit(expr->getArgument()).getValue(builder);

  if (toDelete.getType().isa<mlir::MemRefType>()) {
    builder.create<mlir::memref::DeallocOp>(loc, toDelete);
  } else {
    mlir::Value args[1] = {builder.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(builder.getI8Type()), toDelete)};
    builder.create<mlir::LLVM::CallOp>(loc, Glob.GetOrCreateFreeFunction(),
                                       args);
  }

  return nullptr;
}
ValueCategory MLIRScanner::VisitCXXNewExpr(clang::CXXNewExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());

  mlir::Value count;

  if (expr->isArray()) {
    count = Visit(*expr->raw_arg_begin()).getValue(builder);
    count = builder.create<IndexCastOp>(
        loc, mlir::IndexType::get(builder.getContext()), count);
  } else {
    count = getConstantIndex(1);
  }
  assert(count);

  mlir::Type ty = Glob.getTypes().getMLIRType(expr->getType());

  mlir::Value alloc;
  mlir::Value arrayCons;
  if (!expr->placement_arguments().empty()) {
    mlir::Value val = Visit(*expr->placement_arg_begin()).getValue(builder);
    if (auto mt = ty.dyn_cast<mlir::MemRefType>()) {
      arrayCons = alloc =
          builder.create<polygeist::Pointer2MemrefOp>(loc, mt, val);
    } else {
      arrayCons = alloc = builder.create<mlir::LLVM::BitcastOp>(loc, ty, val);
      auto PT = ty.cast<LLVM::LLVMPointerType>();
      if (expr->isArray())
        arrayCons = builder.create<mlir::LLVM::BitcastOp>(
            loc,
            LLVM::LLVMPointerType::get(
                LLVM::LLVMArrayType::get(PT.getElementType(), 0),
                PT.getAddressSpace()),
            alloc);
    }
  } else if (auto mt = ty.dyn_cast<mlir::MemRefType>()) {
    auto shape = std::vector<int64_t>(mt.getShape());
    mlir::Value args[1] = {count};
    arrayCons = alloc = builder.create<mlir::memref::AllocOp>(loc, mt, args);
    if (expr->hasInitializer() && isa<InitListExpr>(expr->getInitializer()))
      (void)InitializeValueByInitListExpr(alloc, expr->getInitializer());

  } else {
    auto i64 = mlir::IntegerType::get(count.getContext(), 64);
    auto typeSize = getTypeSize(expr->getAllocatedType());
    mlir::Value args[1] = {builder.create<arith::MulIOp>(loc, typeSize, count)};
    args[0] = builder.create<IndexCastOp>(loc, i64, args[0]);
    arrayCons = alloc = builder.create<mlir::LLVM::BitcastOp>(
        loc, ty,
        builder
            .create<mlir::LLVM::CallOp>(loc, Glob.GetOrCreateMallocFunction(),
                                        args)
            ->getResult(0));
    auto PT = ty.cast<LLVM::LLVMPointerType>();
    if (expr->isArray())
      arrayCons = builder.create<mlir::LLVM::BitcastOp>(
          loc,
          LLVM::LLVMPointerType::get(
              LLVM::LLVMArrayType::get(PT.getElementType(), 0),
              PT.getAddressSpace()),
          alloc);
  }
  assert(alloc);

  if (expr->getConstructExpr()) {
    VisitConstructCommon(
        const_cast<CXXConstructExpr *>(expr->getConstructExpr()),
        /*name*/ nullptr, /*memtype*/ 0, arrayCons, count);
  }
  return ValueCategory(alloc, /*isRefererence*/ false);
}

ValueCategory
MLIRScanner::VisitCXXScalarValueInitExpr(clang::CXXScalarValueInitExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());

  bool isArray = false;
  mlir::Type melem = Glob.getTypes().getMLIRType(expr->getType(), &isArray);
  assert(!isArray);

  if (melem.isa<mlir::IntegerType>())
    return ValueCategory(builder.create<ConstantIntOp>(loc, 0, melem), false);
  else if (auto MT = melem.dyn_cast<mlir::MemRefType>())
    return ValueCategory(
        builder.create<polygeist::Pointer2MemrefOp>(
            loc, MT,
            builder.create<mlir::LLVM::NullOp>(
                loc, LLVM::LLVMPointerType::get(builder.getI8Type(),
                                                MT.getMemorySpaceAsInt()))),
        false);
  else if (auto PT = melem.dyn_cast<mlir::LLVM::LLVMPointerType>())
    return ValueCategory(builder.create<mlir::LLVM::NullOp>(loc, PT), false);
  else {
    if (!melem.isa<FloatType>())
      expr->dump();
    auto ft = melem.cast<FloatType>();
    return ValueCategory(builder.create<ConstantFloatOp>(
                             loc, APFloat(ft.getFloatSemantics(), "0"), ft),
                         false);
  }
}

ValueCategory MLIRScanner::VisitCXXPseudoDestructorExpr(
    clang::CXXPseudoDestructorExpr *expr) {
  Visit(expr->getBase());
  llvm::errs() << "not running pseudo destructor\n";
  return nullptr;
}

ValueCategory
MLIRScanner::VisitCXXConstructExpr(clang::CXXConstructExpr *cons) {
  return VisitConstructCommon(cons, /*name*/ nullptr, /*space*/ 0);
}

ValueCategory MLIRScanner::VisitConstructCommon(clang::CXXConstructExpr *cons,
                                                VarDecl *name, unsigned memtype,
                                                mlir::Value op,
                                                mlir::Value count) {
  auto loc = getMLIRLocation(cons->getExprLoc());

  bool isArray = false;
  mlir::Type subType = Glob.getTypes().getMLIRType(cons->getType(), &isArray);

  bool LLVMABI = false;
  mlir::Type ptrty = Glob.getTypes().getMLIRType(
      Glob.getCGM().getContext().getLValueReferenceType(cons->getType()));
  if (ptrty.isa<mlir::LLVM::LLVMPointerType>())
    LLVMABI = true;
  else if (isArray) {
    subType = ptrty;
    isArray = true;
  }
  if (op == nullptr)
    op = createAllocOp(subType, name, memtype, isArray, LLVMABI);

  if (cons->requiresZeroInitialization()) {
    mlir::Value val = op;
    if (val.getType().isa<MemRefType>()) {
      val = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(
              builder.getI8Type(),
              val.getType().cast<MemRefType>().getMemorySpaceAsInt()),
          val);
    } else {
      val = builder.create<LLVM::BitcastOp>(
          loc,
          LLVM::LLVMPointerType::get(
              builder.getI8Type(),
              val.getType().cast<LLVM::LLVMPointerType>().getAddressSpace()),
          val);
    }
    mlir::Value size = getTypeSize(cons->getType());

    auto i8_0 = builder.create<ConstantIntOp>(loc, 0, 8);
    auto sizev =
        builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), size);

    auto falsev = builder.create<ConstantIntOp>(loc, false, 1);
    builder.create<LLVM::MemsetOp>(loc, val, i8_0, sizev, falsev);
  }

  CXXConstructorDecl *ctorDecl = cons->getConstructor();
  if (ctorDecl->isTrivial() && ctorDecl->isDefaultConstructor())
    return ValueCategory(op, /*isReference*/ true);

  mlir::Block::iterator oldpoint;
  mlir::Block *oldblock;
  ValueCategory endobj(op, /*isReference*/ true);

  ValueCategory obj(op, /*isReference*/ true);
  QualType innerType = cons->getType();
  if (auto arrayType =
          Glob.getCGM().getContext().getAsArrayType(cons->getType())) {
    innerType = arrayType->getElementType();
    mlir::Value size;
    if (count)
      size = count;
    else {
      auto CAT = cast<clang::ConstantArrayType>(arrayType);
      size = getConstantIndex(CAT->getSize().getLimitedValue());
    }
    auto forOp = builder.create<scf::ForOp>(loc, getConstantIndex(0), size,
                                            getConstantIndex(1));
    oldpoint = builder.getInsertionPoint();
    oldblock = builder.getInsertionBlock();

    builder.setInsertionPointToStart(&forOp.getLoopBody().front());
    assert(obj.isReference);
    obj = CommonArrayToPointer(obj);
    obj = CommonArrayLookup(obj, forOp.getInductionVar(),
                            /*isImplicitRef*/ false, /*removeIndex*/ false);
    assert(obj.isReference);
  }

  /// If the constructor is part of the SYCL namespace, we may not want the
  /// GetOrCreateMLIRFunction to add this FuncOp to the functionsToEmit deque,
  /// since we will create it's equivalent with SYCL operations. Please note
  /// that we still generate some constructors that we need for lowering some
  /// sycl op.  Therefore, in those case, we set ShouldEmit back to "true" by
  /// looking them up in our "registry" of supported constructors.
  bool isSyclCtor =
      mlirclang::isNamespaceSYCL(ctorDecl->getEnclosingNamespaceContext());
  bool ShouldEmit = !isSyclCtor;

  std::string mangledName = MLIRScanner::getMangledFuncName(
      cast<FunctionDecl>(*ctorDecl), Glob.getCGM());
  mangledName = (PrefixABI + mangledName);
  if (GenerateAllSYCLFuncs || isSupportedFunctions(mangledName))
    ShouldEmit = true;

  FunctionToEmit F(*ctorDecl, mlirclang::getInputContext(builder));
  auto tocall = cast<func::FuncOp>(Glob.GetOrCreateMLIRFunction(F, ShouldEmit));

  SmallVector<std::pair<ValueCategory, clang::Expr *>> args;
  args.emplace_back(std::make_pair(obj, (clang::Expr *)nullptr));
  for (auto a : cons->arguments())
    args.push_back(std::make_pair(Visit(a), a));
  CallHelper(tocall, innerType, args,
             /*retType*/ Glob.getCGM().getContext().VoidTy, false, cons,
             ctorDecl);

  if (Glob.getCGM().getContext().getAsArrayType(cons->getType())) {
    builder.setInsertionPoint(oldblock, oldpoint);
  }
  return endobj;
}

ValueCategory
MLIRScanner::VisitArraySubscriptExpr(clang::ArraySubscriptExpr *expr) {
  auto moo = Visit(expr->getLHS());

  auto rhs = Visit(expr->getRHS()).getValue(builder);
  // Check the RHS has been successfully emitted
  assert(rhs);
  auto idx = castToIndex(getMLIRLocation(expr->getRBracketLoc()), rhs);
  if (isa<clang::VectorType>(
          expr->getLHS()->getType()->getUnqualifiedDesugaredType())) {
    assert(moo.isReference);
    moo.isReference = false;
    auto mt = moo.val.getType().cast<MemRefType>();

    auto shape = std::vector<int64_t>(mt.getShape());
    shape.erase(shape.begin());
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    moo.val = builder.create<polygeist::SubIndexOp>(loc, mt0, moo.val,
                                                    getConstantIndex(0));
  }
  bool isArray = false;
  if (!Glob.getCGM().getContext().getAsArrayType(expr->getType()))
    Glob.getTypes().getMLIRType(expr->getType(), &isArray);
  return CommonArrayLookup(moo, idx, isArray);
}

const clang::FunctionDecl *MLIRScanner::EmitCallee(const Expr *E) {
  E = E->IgnoreParens();
  // Look through function-to-pointer decay.
  if (auto ICE = dyn_cast<ImplicitCastExpr>(E)) {
    if (ICE->getCastKind() == CK_FunctionToPointerDecay ||
        ICE->getCastKind() == CK_BuiltinFnToFnPtr) {
      return EmitCallee(ICE->getSubExpr());
    }

    // Resolve direct calls.
  } else if (auto DRE = dyn_cast<DeclRefExpr>(E)) {
    if (auto FD = dyn_cast<FunctionDecl>(DRE->getDecl())) {
      return FD;
    }

  } else if (auto ME = dyn_cast<MemberExpr>(E)) {
    if (auto FD = dyn_cast<FunctionDecl>(ME->getMemberDecl())) {
      // TODO EmitIgnoredExpr(ME->getBase());
      return FD;
    }

    // Look through template substitutions.
  } else if (auto NTTP = dyn_cast<SubstNonTypeTemplateParmExpr>(E)) {
    return EmitCallee(NTTP->getReplacement());
  } else if (auto UOp = dyn_cast<clang::UnaryOperator>(E)) {
    if (UOp->getOpcode() == UnaryOperatorKind::UO_AddrOf) {
      return EmitCallee(UOp->getSubExpr());
    }
  }

  return nullptr;
}

std::pair<ValueCategory, bool>
MLIRScanner::EmitBuiltinOps(clang::CallExpr *expr) {
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr()))
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__log2f") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
        return std::make_pair(
            ValueCategory(builder.create<mlir::math::Log2Op>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }

  std::vector<mlir::Value> args;
  auto VisitArgs = [&]() {
    assert(args.empty() && "Expecting empty args");
    for (auto a : expr->arguments())
      args.push_back(Visit(a).getValue(builder));
  };
  Optional<Value> V = None;
  switch (expr->getBuiltinCallee()) {
  case Builtin::BIceil: {
    VisitArgs();
    V = builder.create<math::CeilOp>(loc, args[0]);
  } break;
  case Builtin::BIcos: {
    VisitArgs();
    V = builder.create<mlir::math::CosOp>(loc, args[0]);
  } break;
  case Builtin::BIexp:
  case Builtin::BIexpf: {
    VisitArgs();
    V = builder.create<mlir::math::ExpOp>(loc, args[0]);
  } break;
  case Builtin::BIlog: {
    VisitArgs();
    V = builder.create<mlir::math::LogOp>(loc, args[0]);
  } break;
  case Builtin::BIsin: {
    VisitArgs();
    V = builder.create<mlir::math::SinOp>(loc, args[0]);
  } break;
  case Builtin::BIsqrt:
  case Builtin::BIsqrtf: {
    VisitArgs();
    V = builder.create<mlir::math::SqrtOp>(loc, args[0]);
  } break;
  case Builtin::BI__builtin_atanh:
  case Builtin::BI__builtin_atanhf:
  case Builtin::BI__builtin_atanhl: {
    VisitArgs();
    V = builder.create<math::AtanOp>(loc, args[0]);
  } break;
  case Builtin::BI__builtin_copysign:
  case Builtin::BI__builtin_copysignf:
  case Builtin::BI__builtin_copysignl: {
    VisitArgs();
    V = builder.create<LLVM::CopySignOp>(loc, args[0], args[1]);
  } break;
  case Builtin::BI__builtin_exp2:
  case Builtin::BI__builtin_exp2f:
  case Builtin::BI__builtin_exp2l: {
    VisitArgs();
    V = builder.create<math::Exp2Op>(loc, args[0]);
  } break;
  case Builtin::BI__builtin_expm1:
  case Builtin::BI__builtin_expm1f:
  case Builtin::BI__builtin_expm1l: {
    VisitArgs();
    V = builder.create<math::ExpM1Op>(loc, args[0]);
  } break;
  case Builtin::BI__builtin_fma:
  case Builtin::BI__builtin_fmaf:
  case Builtin::BI__builtin_fmal: {
    VisitArgs();
    V = builder.create<LLVM::FMAOp>(loc, args[0], args[1], args[2]);
  } break;
  case Builtin::BI__builtin_fmax:
  case Builtin::BI__builtin_fmaxf:
  case Builtin::BI__builtin_fmaxl: {
    VisitArgs();
    V = builder.create<LLVM::MaxNumOp>(loc, args[0], args[1]);
  } break;
  case Builtin::BI__builtin_fmin:
  case Builtin::BI__builtin_fminf:
  case Builtin::BI__builtin_fminl: {
    VisitArgs();
    V = builder.create<LLVM::MinNumOp>(loc, args[0], args[1]);
  } break;
  case Builtin::BI__builtin_log1p:
  case Builtin::BI__builtin_log1pf:
  case Builtin::BI__builtin_log1pl: {
    VisitArgs();
    V = builder.create<math::Log1pOp>(loc, args[0]);
  } break;
  case Builtin::BI__builtin_pow:
  case Builtin::BI__builtin_powf:
  case Builtin::BI__builtin_powl: {
    VisitArgs();
    V = builder.create<math::PowFOp>(loc, args[0], args[1]);
  } break;
  case Builtin::BI__builtin_assume: {
    VisitArgs();
    V = builder.create<LLVM::AssumeOp>(loc, args[0])->getResult(0);
  } break;
  case Builtin::BI__builtin_isgreater: {
    VisitArgs();
    auto postTy =
        Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    V = builder.create<ExtUIOp>(
        loc, postTy,
        builder.create<CmpFOp>(loc, CmpFPredicate::OGT, args[0], args[1]));
  } break;
  case Builtin::BI__builtin_isgreaterequal: {
    VisitArgs();
    auto postTy =
        Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    V = builder.create<ExtUIOp>(
        loc, postTy,
        builder.create<CmpFOp>(loc, CmpFPredicate::OGE, args[0], args[1]));
  } break;
  case Builtin::BI__builtin_isless: {
    VisitArgs();
    auto postTy =
        Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    V = builder.create<ExtUIOp>(
        loc, postTy,
        builder.create<CmpFOp>(loc, CmpFPredicate::OLT, args[0], args[1]));
  } break;
  case Builtin::BI__builtin_islessequal: {
    VisitArgs();
    auto postTy =
        Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    V = builder.create<ExtUIOp>(
        loc, postTy,
        builder.create<CmpFOp>(loc, CmpFPredicate::OLE, args[0], args[1]));
  } break;
  case Builtin::BI__builtin_islessgreater: {
    VisitArgs();
    auto postTy =
        Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    V = builder.create<ExtUIOp>(
        loc, postTy,
        builder.create<CmpFOp>(loc, CmpFPredicate::ONE, args[0], args[1]));
  } break;
  case Builtin::BI__builtin_isunordered: {
    VisitArgs();
    auto postTy =
        Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    V = builder.create<ExtUIOp>(
        loc, postTy,
        builder.create<CmpFOp>(loc, CmpFPredicate::UNO, args[0], args[1]));
  } break;
  case Builtin::BImemmove:
  case Builtin::BI__builtin_memmove: {
    VisitArgs();
    builder.create<LLVM::MemmoveOp>(
        loc, args[0], args[1], args[2],
        /*isVolatile*/ builder.create<ConstantIntOp>(loc, false, 1));
    V = args[0];
  } break;
  case Builtin::BImemset:
  case Builtin::BI__builtin_memset: {
    VisitArgs();
    builder.create<LLVM::MemsetOp>(
        loc, args[0],
        builder.create<TruncIOp>(loc, builder.getI8Type(), args[1]), args[2],
        /*isVolatile*/ builder.create<ConstantIntOp>(loc, false, 1));
    V = args[0];
  } break;
  case Builtin::BImemcpy:
  case Builtin::BI__builtin_memcpy: {
    VisitArgs();
    builder.create<LLVM::MemcpyOp>(
        loc, args[0], args[1], args[2],
        /*isVolatile*/ builder.create<ConstantIntOp>(loc, false, 1));
    V = args[0];
  } break;
  }
  if (V.has_value())
    return std::make_pair(ValueCategory(V.value(),
                                        /*isReference*/ false),
                          true);

  return std::make_pair(ValueCategory(), false);
}

static NamedAttrList getSYCLMethodOpAttrs(OpBuilder &builder,
                                          mlir::Type baseType,
                                          llvm::StringRef typeName,
                                          llvm::StringRef functionName,
                                          llvm::StringRef mangledFunctionName) {
  NamedAttrList attrs;
  attrs.set(mlir::sycl::SYCLDialect::getBaseTypeAttrName(),
            mlir::TypeAttr::get(baseType));
  attrs.set(mlir::sycl::SYCLDialect::getFunctionNameAttrName(),
            FlatSymbolRefAttr::get(builder.getStringAttr(functionName)));
  attrs.set(mlir::sycl::SYCLDialect::getMangledFunctionNameAttrName(),
            FlatSymbolRefAttr::get(builder.getStringAttr(mangledFunctionName)));
  attrs.set(mlir::sycl::SYCLDialect::getTypeNameAttrName(),
            FlatSymbolRefAttr::get(builder.getStringAttr(typeName)));
  return attrs;
}

/// Returns the SYCL cast originating this value if such operation exists; None
/// otherwise.
///
/// This function relies on how arguments are casted to perform a function call.
/// Should be updated if this changes.
static llvm::Optional<mlir::sycl::SYCLCastOp> trackSYCLCast(Value val) {
  const auto trackWithOperand =
      [](Operation *Op) -> llvm::Optional<mlir::sycl::SYCLCastOp> {
    return trackSYCLCast(Op->getOperand(0));
  };
  const auto DefiningOp = val.getDefiningOp();
  if (!DefiningOp) {
    return llvm::None;
  }
  return TypeSwitch<mlir::Operation *, llvm::Optional<mlir::sycl::SYCLCastOp>>(
             DefiningOp)
      .Case<mlir::sycl::SYCLCastOp>(
          [](auto Cast) -> llvm::Optional<mlir::sycl::SYCLCastOp> {
            return Cast;
          })
      .Case<mlir::LLVM::AddrSpaceCastOp>(trackWithOperand)
      .Case<mlir::polygeist::Memref2PointerOp>(trackWithOperand)
      .Case<mlir::polygeist::Pointer2MemrefOp>(trackWithOperand)
      .Default([](auto) -> llvm::Optional<mlir::sycl::SYCLCastOp> {
        return llvm::None;
      });
}

llvm::Optional<sycl::SYCLMethodOpInterface> MLIRScanner::createSYCLMethodOp(
    llvm::StringRef typeName, llvm::StringRef functionName,
    mlir::ValueRange operands, llvm::Optional<mlir::Type> returnType,
    llvm::StringRef mangledFunctionName) {
  // Expecting a MemRef as the first argument, as the first operand to a method
  // call should be a pointer to `this`.
  if (operands.empty() || !operands[0].getType().isa<MemRefType>()) {
    return llvm::None;
  }

  auto *SYCLDialect =
      operands[0].getContext()->getLoadedDialect<mlir::sycl::SYCLDialect>();
  assert(SYCLDialect && "MLIR-SYCL dialect not loaded.");

  // Need to copy to avoid overriding elements in the input argument.
  SmallVector<mlir::Value> operandsCpy(operands);

  // SYCLCastOps are abstracted to avoid missing method calls due to
  // implementation details.
  if (const llvm::Optional<sycl::SYCLCastOp> Cast =
          trackSYCLCast(operandsCpy[0])) {
    auto NewArg = (*Cast)->getOperand(0);
    // Make sure the memory space is not changed:
    const auto MemSpace =
        operandsCpy[0].getType().cast<MemRefType>().getMemorySpaceAsInt();
    if (NewArg.getType().cast<MemRefType>().getMemorySpaceAsInt() != MemSpace) {
      NewArg = castToMemSpace(NewArg, MemSpace);
    }
    operandsCpy[0] = NewArg;
    LLVM_DEBUG(llvm::dbgs() << "Abstracting cast to " << NewArg.getType()
                            << " to insert a SYCL method\n");
  }

  auto BaseType = operandsCpy[0].getType().cast<MemRefType>();
  const llvm::Optional<llvm::StringRef> OptOpName = SYCLDialect->findMethod(
      BaseType.getElementType().getTypeID(), functionName);

  if (!OptOpName) {
    LLVM_DEBUG(llvm::dbgs() << "SYCL method not inserted. Type: " << BaseType
                            << " Name: " << functionName << "\n");
    return llvm::None;
  }

  LLVM_DEBUG(llvm::dbgs() << "Inserting operation " << OptOpName
                          << " to replace SYCL method call.\n");

  return static_cast<sycl::SYCLMethodOpInterface>(builder.create(
      loc, builder.getStringAttr(*OptOpName), operandsCpy,
      returnType ? mlir::TypeRange{*returnType} : mlir::TypeRange{},
      getSYCLMethodOpAttrs(builder, operands[0].getType(), typeName,
                           functionName, mangledFunctionName)));
}

mlir::Operation *
MLIRScanner::EmitSYCLOps(const clang::Expr *Expr,
                         const llvm::SmallVectorImpl<mlir::Value> &Args) {
  mlir::Operation *Op = nullptr;

  if (const auto *ConsExpr = dyn_cast<clang::CXXConstructExpr>(Expr)) {
    const auto *Func = ConsExpr->getConstructor()->getAsFunction();

    if (mlirclang::isNamespaceSYCL(Func->getEnclosingNamespaceContext())) {
      if (const auto *RD = dyn_cast<clang::CXXRecordDecl>(Func->getParent())) {
        std::string name =
            MLIRScanner::getMangledFuncName(*Func, Glob.getCGM());
        Op = builder.create<mlir::sycl::SYCLConstructorOp>(loc, RD->getName(),
                                                           name, Args);
      }
    }
  } else if (const auto *CallExpr = dyn_cast<clang::CallExpr>(Expr)) {
    const auto *Func = CallExpr->getCalleeDecl()->getAsFunction();

    if (mlirclang::isNamespaceSYCL(Func->getEnclosingNamespaceContext())) {
      auto OptFuncType = llvm::Optional<llvm::StringRef>{llvm::None};
      if (const auto *RD = dyn_cast<clang::CXXRecordDecl>(Func->getParent()))
        if (!RD->getName().empty())
          OptFuncType = RD->getName();

      auto OptRetType = llvm::Optional<mlir::Type>{llvm::None};
      const mlir::Type RetType =
          Glob.getTypes().getMLIRType(Func->getReturnType());
      if (!RetType.isa<mlir::NoneType>()) {
        OptRetType = RetType;
      }

      std::string name = MLIRScanner::getMangledFuncName(*Func, Glob.getCGM());
      if (OptFuncType) {
        // Attempt to create a SYCL method call first, if that fails create a
        // generic SYCLCallOp.
        Op = createSYCLMethodOp(*OptFuncType, Func->getNameAsString(), Args,
                                OptRetType, name)
                 .value_or(nullptr);
      }
      if (!Op) {
        Op = builder.create<mlir::sycl::SYCLCallOp>(
            loc, OptRetType, OptFuncType, Func->getNameAsString(), name, Args);
      }
    }
  }

  return Op;
}

ValueCategory MLIRScanner::VisitMSPropertyRefExpr(MSPropertyRefExpr *expr) {
  assert(0 && "unhandled ms propertyref");
  // TODO obviously fake
  return nullptr;
}

ValueCategory
MLIRScanner::VisitPseudoObjectExpr(clang::PseudoObjectExpr *expr) {
  return Visit(expr->getResultExpr());
}

ValueCategory MLIRScanner::VisitSubstNonTypeTemplateParmExpr(
    SubstNonTypeTemplateParmExpr *expr) {
  return Visit(expr->getReplacement());
}

ValueCategory
MLIRScanner::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Uop) {
  switch (Uop->getKind()) {
  case UETT_SizeOf: {
    auto value = getTypeSize(Uop->getTypeOfArgument());
    auto retTy =
        Glob.getTypes().getMLIRType(Uop->getType()).cast<mlir::IntegerType>();
    return ValueCategory(builder.create<arith::IndexCastOp>(loc, retTy, value),
                         /*isReference*/ false);
  }
  case UETT_AlignOf: {
    auto value = getTypeAlign(Uop->getTypeOfArgument());
    auto retTy =
        Glob.getTypes().getMLIRType(Uop->getType()).cast<mlir::IntegerType>();
    return ValueCategory(builder.create<arith::IndexCastOp>(loc, retTy, value),
                         /*isReference*/ false);
  }
  default:
    Uop->dump();
    assert(0 && "unhandled VisitUnaryExprOrTypeTraitExpr");
  }
}

ValueCategory MLIRScanner::VisitAtomicExpr(clang::AtomicExpr *BO) {
  auto loc = getMLIRLocation(BO->getExprLoc());

  switch (BO->getOp()) {
  case AtomicExpr::AtomicOp::AO__atomic_add_fetch: {
    auto a0 = Visit(BO->getPtr()).getValue(builder);
    auto a1 = Visit(BO->getVal1()).getValue(builder);
    auto ty = a1.getType();
    AtomicRMWKind op;
    LLVM::AtomicBinOp lop;
    if (ty.isa<mlir::IntegerType>()) {
      op = AtomicRMWKind::addi;
      lop = LLVM::AtomicBinOp::add;
    } else {
      op = AtomicRMWKind::addf;
      lop = LLVM::AtomicBinOp::fadd;
    }
    // TODO add atomic ordering
    mlir::Value v;
    if (a0.getType().isa<MemRefType>())
      v = builder.create<memref::AtomicRMWOp>(
          loc, a1.getType(), op, a1, a0,
          std::vector<mlir::Value>({getConstantIndex(0)}));
    else
      v = builder.create<LLVM::AtomicRMWOp>(loc, a1.getType(), lop, a0, a1,
                                            LLVM::AtomicOrdering::acq_rel);

    if (ty.isa<mlir::IntegerType>())
      v = builder.create<arith::AddIOp>(loc, v, a1);
    else
      v = builder.create<arith::AddFOp>(loc, v, a1);

    return ValueCategory(v, false);
  }
  default:
    llvm::errs() << "unhandled atomic:";
    BO->dump();
    assert(0);
  }
}

ValueCategory MLIRScanner::VisitExprWithCleanups(ExprWithCleanups *E) {
  auto ret = Visit(E->getSubExpr());
  for (auto &child : E->children()) {
    child->dump();
    llvm::errs() << "cleanup not handled\n";
  }
  return ret;
}

ValueCategory MLIRScanner::VisitDeclRefExpr(DeclRefExpr *E) {
  LLVM_DEBUG({
    llvm::dbgs() << "VisitDeclRefExpr: ";
    E->dump();
    llvm::dbgs() << "\n";
  });

  auto name = E->getDecl()->getName().str();

  if (auto tocall = dyn_cast<FunctionDecl>(E->getDecl()))
    return ValueCategory(builder.create<LLVM::AddressOfOp>(
                             loc, Glob.GetOrCreateLLVMFunction(tocall)),
                         /*isReference*/ true);

  if (auto VD = dyn_cast<VarDecl>(E->getDecl())) {
    if (Captures.find(VD) != Captures.end()) {
      FieldDecl *field = Captures[VD];
      auto res = CommonFieldLookup(
          cast<CXXMethodDecl>(EmittingFunctionDecl)->getThisObjectType(), field,
          ThisVal.val,
          isa<clang::ReferenceType>(
              field->getType()->getUnqualifiedDesugaredType()));
      assert(CaptureKinds.find(VD) != CaptureKinds.end());
      return res;
    }
  }

  if (auto PD = dyn_cast<VarDecl>(E->getDecl())) {
    auto found = params.find(PD);
    if (found != params.end()) {
      auto res = found->second;
      assert(res.val);
      return res;
    }
  }
  if (auto ED = dyn_cast<EnumConstantDecl>(E->getDecl())) {
    auto ty =
        Glob.getTypes().getMLIRType(E->getType()).cast<mlir::IntegerType>();
    return ValueCategory(
        builder.create<ConstantIntOp>(loc, ED->getInitVal().getExtValue(), ty),
        /*isReference*/ false);

    if (!ED->getInitExpr())
      ED->dump();
    return Visit(ED->getInitExpr());
  }
  if (auto VD = dyn_cast<ValueDecl>(E->getDecl())) {
    if (Glob.getTypes()
            .getMLIRType(
                Glob.getCGM().getContext().getPointerType(E->getType()))
            .isa<mlir::LLVM::LLVMPointerType>() ||
        name == "stderr" || name == "stdout" || name == "stdin" ||
        (E->hasQualifier())) {
      return ValueCategory(builder.create<mlir::LLVM::AddressOfOp>(
                               loc, Glob.GetOrCreateLLVMGlobal(VD)),
                           /*isReference*/ true);
    }

    // We need to decide where to put the Global.  If we are in a device
    // module, the global should be in the gpu module (which is nested inside
    // another main module).
    std::pair<mlir::memref::GlobalOp, bool> gv = Glob.GetOrCreateGlobal(
        VD, /*prefix=*/"", true,
        isa<mlir::gpu::GPUModuleOp>(function->getParentOp())
            ? FunctionContext::SYCLDevice
            : FunctionContext::Host);

    auto gv2 = builder.create<memref::GetGlobalOp>(loc, gv.first.getType(),
                                                   gv.first.getName());
    bool isArray = gv.second;
    // TODO check reference
    if (isArray)
      return ValueCategory(gv2, /*isReference*/ true);
    else
      return ValueCategory(gv2, /*isReference*/ true);
    // return gv2;
  }
  E->dump();
  E->getDecl()->dump();
  llvm::errs() << "couldn't find " << name << "\n";
  assert(0 && "couldnt find value");
  return nullptr;
}

ValueCategory MLIRScanner::VisitOpaqueValueExpr(OpaqueValueExpr *E) {
  if (!E->getSourceExpr()) {
    E->dump();
    assert(E->getSourceExpr());
  }
  auto res = Visit(E->getSourceExpr());
  if (!res.val) {
    E->dump();
    E->getSourceExpr()->dump();
    assert(res.val);
  }
  return res;
}

ValueCategory MLIRScanner::VisitCXXTypeidExpr(clang::CXXTypeidExpr *E) {
  QualType T;
  if (E->isTypeOperand())
    T = E->getTypeOperand(Glob.getCGM().getContext());
  else
    T = E->getExprOperand()->getType();
  llvm::Constant *C = Glob.getCGM().GetAddrOfRTTIDescriptor(T);
  llvm::errs() << *C << "\n";
  mlir::Type ty = Glob.getTypes().getMLIRType(E->getType());
  llvm::errs() << ty << "\n";
  assert(0 && "unhandled typeid");
}

ValueCategory
MLIRScanner::VisitCXXDefaultInitExpr(clang::CXXDefaultInitExpr *expr) {
  assert(ThisVal.val);
  auto toset = Visit(expr->getExpr());
  assert(!ThisVal.isReference);
  assert(toset.val);

  bool isArray = false;
  Glob.getTypes().getMLIRType(expr->getExpr()->getType(), &isArray);

  auto cfl = CommonFieldLookup(
      cast<CXXMethodDecl>(EmittingFunctionDecl)->getThisObjectType(),
      expr->getField(), ThisVal.val, /*isLValue*/ false);
  assert(cfl.val);
  cfl.store(builder, toset, isArray);
  return cfl;
}

ValueCategory MLIRScanner::VisitCXXNoexceptExpr(CXXNoexceptExpr *expr) {
  auto ty =
      Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(
      builder.create<ConstantIntOp>(getMLIRLocation(expr->getExprLoc()),
                                    expr->getValue(), ty),
      /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitMemberExpr(MemberExpr *ME) {
  auto memberName = ME->getMemberDecl()->getName();
  if (auto sr2 = dyn_cast<OpaqueValueExpr>(ME->getBase())) {
    if (auto sr = dyn_cast<DeclRefExpr>(sr2->getSourceExpr())) {
      if (sr->getDecl()->getName() == "blockIdx") {
        if (memberName == "__fetch_builtin_x") {
        }
        llvm::errs() << "known block index";
      }
      if (sr->getDecl()->getName() == "blockDim") {
        llvm::errs() << "known block dim";
      }
      if (sr->getDecl()->getName() == "threadIdx") {
        llvm::errs() << "known thread index";
      }
      if (sr->getDecl()->getName() == "gridDim") {
        llvm::errs() << "known grid index";
      }
    }
  }
  auto base = Visit(ME->getBase());
  clang::QualType OT = ME->getBase()->getType();
  if (ME->isArrow()) {
    if (!base.val) {
      ME->dump();
    }
    base = base.dereference(builder);
    OT = cast<clang::PointerType>(OT->getUnqualifiedDesugaredType())
             ->getPointeeType();
  }
  if (!base.isReference) {
    EmittingFunctionDecl->dump();
    function.dump();
    ME->dump();
    llvm::errs() << "base value: " << base.val << "\n";
  }
  assert(base.isReference);
  const FieldDecl *field = cast<FieldDecl>(ME->getMemberDecl());
  return CommonFieldLookup(
      OT, field, base.val,
      isa<clang::ReferenceType>(
          field->getType()->getUnqualifiedDesugaredType()));
}

ValueCategory MLIRScanner::VisitCastExpr(CastExpr *E) {
  auto loc = getMLIRLocation(E->getExprLoc());
  switch (E->getCastKind()) {

  case clang::CastKind::CK_NullToPointer: {
    mlir::Type llvmType = Glob.getTypes().getMLIRType(E->getType());
    if (llvmType.isa<LLVM::LLVMPointerType>())
      return ValueCategory(builder.create<mlir::LLVM::NullOp>(loc, llvmType),
                           /*isReference*/ false);
    else if (auto MT = llvmType.dyn_cast<MemRefType>())
      return ValueCategory(
          builder.create<polygeist::Pointer2MemrefOp>(
              loc, MT,
              builder.create<mlir::LLVM::NullOp>(
                  loc, LLVM::LLVMPointerType::get(builder.getI8Type(),
                                                  MT.getMemorySpaceAsInt()))),
          false);
    llvm_unreachable("illegal type for cast");
  }
  case clang::CastKind::CK_UserDefinedConversion: {
    return Visit(E->getSubExpr());
  }
  case clang::CastKind::CK_AddressSpaceConversion: {
    auto scalar = Visit(E->getSubExpr());
    // JLE_QUEL::TODO (II-201)
    // assert(scalar.isReference);
    auto postTy = returnVal.getType().cast<MemRefType>().getElementType();
    return ValueCategory(castToMemSpaceOfType(scalar.val, postTy),
                         scalar.isReference);
  }
  case clang::CastKind::CK_Dynamic: {
    E->dump();
    assert(0 && "dynamic cast not handled yet\n");
  }
  case clang::CastKind::CK_UncheckedDerivedToBase:
  case clang::CastKind::CK_DerivedToBase: {
    auto se = Visit(E->getSubExpr());
    if (!se.val) {
      E->dump();
    }
    assert(se.val);
    auto Derived =
        (E->isLValue() || E->isXValue())
            ? cast<CXXRecordDecl>(
                  E->getSubExpr()->getType()->castAs<RecordType>()->getDecl())
            : E->getSubExpr()->getType()->getPointeeCXXRecordDecl();
    SmallVector<const clang::Type *> BaseTypes;
    SmallVector<bool> BaseVirtual;
    for (auto B : E->path()) {
      BaseTypes.push_back(B->getType().getTypePtr());
      BaseVirtual.push_back(B->isVirtual());
    }

    if (auto ut = se.val.getType().dyn_cast<mlir::MemRefType>()) {
      auto mt = Glob.getTypes()
                    .getMLIRType(
                        (E->isLValue() || E->isXValue())
                            ? Glob.getCGM().getContext().getLValueReferenceType(
                                  E->getType())
                            : E->getType())
                    .dyn_cast<mlir::MemRefType>();

      if (ut.getShape().size() != mt.getShape().size()) {
        E->dump();
        llvm::errs() << " se.val: " << se.val << " ut: " << ut << " mt: " << mt
                     << "\n";
      }
      assert(ut.getShape().size() == mt.getShape().size());
      auto ty = mlir::MemRefType::get(mt.getShape(), mt.getElementType(),
                                      MemRefLayoutAttrInterface(),
                                      ut.getMemorySpace());
      if (ty.getElementType().getDialect().getNamespace() ==
              mlir::sycl::SYCLDialect::getDialectNamespace() &&
          ut.getElementType().getDialect().getNamespace() ==
              mlir::sycl::SYCLDialect::getDialectNamespace() &&
          ty.getElementType() != ut.getElementType()) {
        return ValueCategory(
            builder.create<mlir::sycl::SYCLCastOp>(loc, ty, se.val),
            /*isReference*/ se.isReference);
      }
    }

    mlir::Value val =
        GetAddressOfBaseClass(se.val, Derived, BaseTypes, BaseVirtual);
    if (E->getCastKind() != clang::CastKind::CK_UncheckedDerivedToBase &&
        !isa<CXXThisExpr>(E->IgnoreParens())) {
      mlir::Value ptr = val;
      if (auto MT = ptr.getType().dyn_cast<MemRefType>())
        ptr = builder.create<polygeist::Memref2PointerOp>(
            loc,
            LLVM::LLVMPointerType::get(MT.getElementType(),
                                       MT.getMemorySpaceAsInt()),
            ptr);
      mlir::Value nullptr_llvm =
          builder.create<mlir::LLVM::NullOp>(loc, ptr.getType());
      auto ne = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, ptr, nullptr_llvm);
      if (auto MT = ptr.getType().dyn_cast<MemRefType>())
        nullptr_llvm =
            builder.create<polygeist::Pointer2MemrefOp>(loc, MT, nullptr_llvm);
      val = builder.create<arith::SelectOp>(loc, ne, val, nullptr_llvm);
    }

    return ValueCategory(val, se.isReference);
  }
  case clang::CastKind::CK_BaseToDerived: {
    auto se = Visit(E->getSubExpr());
    if (!se.val) {
      E->dump();
    }
    assert(se.val);
    auto Derived =
        (E->isLValue() || E->isXValue())
            ? cast<CXXRecordDecl>(E->getType()->castAs<RecordType>()->getDecl())
            : E->getType()->getPointeeCXXRecordDecl();
    mlir::Value val = GetAddressOfDerivedClass(se.val, Derived, E->path_begin(),
                                               E->path_end());
    /*
    if (ShouldNullCheckClassCastValue(E)) {
        mlir::Value ptr = val;
        if (auto MT = ptr.getType().dyn_cast<MemRefType>())
            ptr = builder.create<polygeist::Memref2PointerOp>(loc,
    LLVM::LLVMPointerType::get(MT.getElementType()), ptr); auto nullptr_llvm =
    builder.create<mlir::LLVM::NullOp>(loc, ptr.getType()); auto ne =
    builder.create<mlir::LLVM::ICmpOp>( loc, mlir::LLVM::ICmpPredicate::ne, ptr,
    nullptr_llvm); if (auto MT = ptr.getType().dyn_cast<MemRefType>())
           nullptr_llvm = builder.create<polygeist::Pointer2MemrefOp>(loc, MT,
    nullptr_llvm); val = builder.create<arith::SelectOp>(loc, ne, val,
    nullptr_llvm);
    }
    */
    return ValueCategory(val, se.isReference);
  }
  case clang::CastKind::CK_BitCast: {

    if (auto CI = dyn_cast<clang::CallExpr>(E->getSubExpr()))
      if (auto ic = dyn_cast<ImplicitCastExpr>(CI->getCallee()))
        if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
          if (sr->getDecl()->getIdentifier() &&
              sr->getDecl()->getName() == "polybench_alloc_data") {
            if (auto mt = Glob.getTypes()
                              .getMLIRType(E->getType())
                              .dyn_cast<mlir::MemRefType>()) {
              auto shape = std::vector<int64_t>(mt.getShape());
              // shape.erase(shape.begin());
              auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                               MemRefLayoutAttrInterface(),
                                               mt.getMemorySpace());

              auto alloc = builder.create<mlir::memref::AllocOp>(loc, mt0);
              return ValueCategory(alloc, /*isReference*/ false);
            }
          }
        }

    if (auto CI = dyn_cast<clang::CallExpr>(E->getSubExpr()))
      if (auto ic = dyn_cast<ImplicitCastExpr>(CI->getCallee()))
        if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
          if (sr->getDecl()->getIdentifier() &&
              (sr->getDecl()->getName() == "malloc" ||
               sr->getDecl()->getName() == "calloc"))
            if (auto mt = Glob.getTypes()
                              .getMLIRType(E->getType())
                              .dyn_cast<mlir::MemRefType>()) {
              auto shape = std::vector<int64_t>(mt.getShape());

              auto elemSize =
                  getTypeSize(cast<clang::PointerType>(
                                  E->getType()->getUnqualifiedDesugaredType())
                                  ->getPointeeType());
              mlir::Value allocSize = builder.create<IndexCastOp>(
                  loc, mlir::IndexType::get(builder.getContext()),
                  Visit(CI->getArg(0)).getValue(builder));
              if (sr->getDecl()->getName() == "calloc") {
                allocSize = builder.create<MulIOp>(
                    loc, allocSize,
                    builder.create<arith::IndexCastOp>(
                        loc, mlir::IndexType::get(builder.getContext()),
                        Visit(CI->getArg(1)).getValue(builder)));
              }
              mlir::Value args[1] = {
                  builder.create<DivUIOp>(loc, allocSize, elemSize)};
              auto alloc = builder.create<mlir::memref::AllocOp>(loc, mt, args);
              if (sr->getDecl()->getName() == "calloc") {
                mlir::Value val = alloc;
                if (val.getType().isa<MemRefType>()) {
                  val = builder.create<polygeist::Memref2PointerOp>(
                      loc,
                      LLVM::LLVMPointerType::get(builder.getI8Type(),
                                                 val.getType()
                                                     .cast<MemRefType>()
                                                     .getMemorySpaceAsInt()),
                      val);
                } else {
                  val = builder.create<LLVM::BitcastOp>(
                      loc,
                      LLVM::LLVMPointerType::get(
                          builder.getI8Type(),
                          val.getType()
                              .cast<LLVM::LLVMPointerType>()
                              .getAddressSpace()),
                      val);
                }
                auto i8_0 = builder.create<arith::ConstantIntOp>(loc, 0, 8);
                auto sizev = builder.create<arith::IndexCastOp>(
                    loc, builder.getI64Type(), allocSize);
                auto falsev =
                    builder.create<arith::ConstantIntOp>(loc, false, 1);
                builder.create<LLVM::MemsetOp>(loc, val, i8_0, sizev, falsev);
              }
              return ValueCategory(alloc, /*isReference*/ false);
            }
        }
    auto se = Visit(E->getSubExpr());
#ifdef DEBUG
    if (!se.val) {
      E->dump();
    }
#endif
    auto scalar = se.getValue(builder);
    if (auto spt = scalar.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      mlir::Type nt = Glob.getTypes().getMLIRType(E->getType());
      LLVM::LLVMPointerType pt = nt.dyn_cast<LLVM::LLVMPointerType>();
      if (!pt) {
        return ValueCategory(
            builder.create<polygeist::Pointer2MemrefOp>(loc, nt, scalar),
            false);
      }
      pt = LLVM::LLVMPointerType::get(pt.getElementType(),
                                      spt.getAddressSpace());
      auto nval = builder.create<mlir::LLVM::BitcastOp>(loc, pt, scalar);
      return ValueCategory(nval, /*isReference*/ false);
    }

#ifdef DEBUG
    if (!scalar.getType().isa<mlir::MemRefType>()) {
      E->dump();
      E->getType()->dump();
      llvm::errs() << "scalar: " << scalar << "\n";
    }
#endif

    assert(scalar.getType().isa<mlir::MemRefType>() &&
           "Expecting 'scalar' to have MemRefType");

    auto scalarTy = scalar.getType().cast<mlir::MemRefType>();
    mlir::Type mlirty = Glob.getTypes().getMLIRType(E->getType());

    if (auto PT = mlirty.dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      assert(
          scalarTy.getMemorySpaceAsInt() == PT.getAddressSpace() &&
          "The type of 'scalar' does not have the same memory space as 'PT'");
      auto val =
          builder.create<mlir::polygeist::Memref2PointerOp>(loc, PT, scalar);
      return ValueCategory(val, /*isReference*/ false);
    }

    if (auto mt = mlirty.dyn_cast<mlir::MemRefType>()) {
      auto ty = mlir::MemRefType::get(mt.getShape(), mt.getElementType(),
                                      MemRefLayoutAttrInterface(),
                                      scalarTy.getMemorySpace());
      if (scalarTy.getShape().size() == mt.getShape().size() + 1)
        return ValueCategory(builder.create<mlir::polygeist::SubIndexOp>(
                                 loc, ty, scalar, getConstantIndex(0)),
                             /*isReference*/ false);

      if (scalarTy.getShape().size() != mt.getShape().size()) {
        auto memRefToPtr = builder.create<polygeist::Memref2PointerOp>(
            loc,
            LLVM::LLVMPointerType::get(builder.getI8Type(),
                                       scalarTy.getMemorySpaceAsInt()),
            scalar);
        assert(ty.getMemorySpaceAsInt() == scalarTy.getMemorySpaceAsInt() &&
               "Expecting 'ty' and 'scalarTy' to have the same memory space");
        auto ptrToMemRef =
            builder.create<polygeist::Pointer2MemrefOp>(loc, ty, memRefToPtr);
        return ValueCategory(ptrToMemRef, /*isReference*/ false);
      }

      return ValueCategory(builder.create<memref::CastOp>(loc, ty, scalar),
                           /*isReference*/ false);
    }

#ifdef DEBUG
    E->dump();
    E->getType()->dump();
    llvm::errs() << " scalar: " << scalar << " mlirty: " << mlirty << "\n";
#endif
    llvm_unreachable("illegal type for cast");
  } break;
  case clang::CastKind::CK_LValueToRValue: {
    if (auto dr = dyn_cast<DeclRefExpr>(E->getSubExpr())) {
      if (auto VD = dyn_cast<VarDecl>(dr->getDecl()->getCanonicalDecl())) {
        if (NOUR_Constant == dr->isNonOdrUse()) {
          auto VarD = cast<VarDecl>(VD);
          if (!VarD->getInit()) {
            E->dump();
            VarD->dump();
          }
          assert(VarD->getInit());
          return Visit(VarD->getInit());
        }
      }
      if (dr->getDecl()->getIdentifier() &&
          dr->getDecl()->getName() == "warpSize") {
        mlir::Type mlirType = Glob.getTypes().getMLIRType(E->getType());
        return ValueCategory(
            builder.create<mlir::NVVM::WarpSizeOp>(loc, mlirType),
            /*isReference*/ false);
      }
      /*
      if (dr->isNonOdrUseReason() == clang::NonOdrUseReason::NOUR_Constant) {
        dr->dump();
        auto VD = cast<VarDecl>(dr->getDecl());
        assert(VD->getInit());
      }
      */
    }
    auto prev = Visit(E->getSubExpr());

    bool isArray = false;
    Glob.getTypes().getMLIRType(E->getType(), &isArray);
    if (isArray)
      return prev;

    auto lres = prev.getValue(builder);
#ifdef DEBUG
    if (!prev.isReference) {
      E->dump();
      lres.dump();
    }
#endif
    assert(prev.isReference);
    return ValueCategory(lres, /*isReference*/ false);
  }
  case clang::CastKind::CK_IntegralToFloating: {
    auto scalar = Visit(E->getSubExpr()).getValue(builder);
    auto ty = Glob.getTypes().getMLIRType(E->getType()).cast<mlir::FloatType>();
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getSubExpr()->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    if (signedType)
      return ValueCategory(
          builder.create<mlir::arith::SIToFPOp>(loc, ty, scalar),
          /*isReference*/ false);

    return ValueCategory(builder.create<mlir::arith::UIToFPOp>(loc, ty, scalar),
                         /*isReference*/ false);
  }
  case clang::CastKind::CK_FloatingToIntegral: {
    auto scalar = Visit(E->getSubExpr()).getValue(builder);
    auto ty =
        Glob.getTypes().getMLIRType(E->getType()).cast<mlir::IntegerType>();
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    if (signedType)
      return ValueCategory(
          builder.create<mlir::arith::FPToSIOp>(loc, ty, scalar),
          /*isReference*/ false);
    return ValueCategory(builder.create<mlir::arith::FPToUIOp>(loc, ty, scalar),
                         /*isReference*/ false);
  }
  case clang::CastKind::CK_IntegralCast: {
    auto scalar = Visit(E->getSubExpr()).getValue(builder);
    assert(scalar);
    auto postTy =
        Glob.getTypes().getMLIRType(E->getType()).cast<mlir::IntegerType>();
    if (scalar.getType().isa<mlir::LLVM::LLVMPointerType>())
      return ValueCategory(
          builder.create<mlir::LLVM::PtrToIntOp>(loc, postTy, scalar),
          /*isReference*/ false);
    if (scalar.getType().isa<mlir::IndexType>() ||
        postTy.isa<mlir::IndexType>())
      return ValueCategory(builder.create<IndexCastOp>(loc, postTy, scalar),
                           false);
#ifdef DEBUG
    if (!scalar.getType().isa<mlir::IntegerType>()) {
      E->dump();
      llvm::errs() << " scalar: " << scalar << "\n";
    }
#endif
    auto prevTy = scalar.getType().cast<mlir::IntegerType>();
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getSubExpr()->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }

    if (prevTy == postTy)
      return ValueCategory(scalar, /*isReference*/ false);
    if (prevTy.getWidth() < postTy.getWidth()) {
      if (signedType) {
        if (auto CI = scalar.getDefiningOp<arith::ConstantIntOp>()) {
          return ValueCategory(
              builder.create<arith::ConstantOp>(
                  loc, postTy,
                  mlir::IntegerAttr::get(
                      postTy, CI.getValue().cast<IntegerAttr>().getValue().sext(
                                  postTy.getWidth()))),
              /*isReference*/ false);
        }
        return ValueCategory(
            builder.create<arith::ExtSIOp>(loc, postTy, scalar),
            /*isReference*/ false);
      }
      if (auto CI = scalar.getDefiningOp<arith::ConstantIntOp>()) {
        return ValueCategory(
            builder.create<arith::ConstantOp>(
                loc, postTy,
                mlir::IntegerAttr::get(
                    postTy, CI.getValue().cast<IntegerAttr>().getValue().zext(
                                postTy.getWidth()))),
            /*isReference*/ false);
      }
      return ValueCategory(builder.create<arith::ExtUIOp>(loc, postTy, scalar),
                           /*isReference*/ false);
    }

    if (auto CI = scalar.getDefiningOp<ConstantIntOp>()) {
      return ValueCategory(
          builder.create<arith::ConstantOp>(
              loc, postTy,
              mlir::IntegerAttr::get(
                  postTy, CI.getValue().cast<IntegerAttr>().getValue().trunc(
                              postTy.getWidth()))),
          /*isReference*/ false);
    }
    return ValueCategory(builder.create<arith::TruncIOp>(loc, postTy, scalar),
                         /*isReference*/ false);
  }
  case clang::CastKind::CK_FloatingCast: {
    auto scalar = Visit(E->getSubExpr()).getValue(builder);
#ifdef DEBUG
    if (!scalar.getType().isa<mlir::FloatType>()) {
      E->dump();
      llvm::errs() << "scalar: " << scalar << "\n";
    }
#endif
    auto prevTy = scalar.getType().cast<mlir::FloatType>();
    auto postTy =
        Glob.getTypes().getMLIRType(E->getType()).cast<mlir::FloatType>();

    if (prevTy == postTy)
      return ValueCategory(scalar, /*isReference*/ false);
    if (auto c = scalar.getDefiningOp<ConstantFloatOp>()) {
      APFloat Val = c.getValue().cast<FloatAttr>().getValue();
      bool ignored;
      Val.convert(postTy.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                  &ignored);
      return ValueCategory(builder.create<arith::ConstantOp>(
                               loc, postTy, mlir::FloatAttr::get(postTy, Val)),
                           false);
    }
    if (prevTy.getWidth() < postTy.getWidth()) {
      return ValueCategory(builder.create<arith::ExtFOp>(loc, postTy, scalar),
                           /*isReference*/ false);
    }
    return ValueCategory(builder.create<arith::TruncFOp>(loc, postTy, scalar),
                         /*isReference*/ false);
  }
  case clang::CastKind::CK_ArrayToPointerDecay: {
    return CommonArrayToPointer(Visit(E->getSubExpr()));

#if 0
    auto mt = scalar.val.getType().cast<mlir::MemRefType>();
    auto shape2 = std::vector<int64_t>(mt.getShape());
    if (shape2.size() == 0) {
      E->dump();
      //nex.dump();
      assert(0);
    }
    shape2[0] = -1;
    auto nex = mlir::MemRefType::get(shape2, mt.getElementType(),
                                     mt.getLayout(), mt.getMemorySpace());
    auto cst = builder.create<mlir::MemRefCastOp>(loc, scalar.val, nex);
    //llvm::errs() << "<ArrayToPtrDecay>\n";
    //E->dump();
    //llvm::errs() << cst << " - " << scalar.val << "\n";
    //auto offs = scalar.offsets;
    //offs.push_back(getConstantIndex(0));
    return ValueCategory(cst, scalar.isReference);
#endif
  }
  case clang::CastKind::CK_FunctionToPointerDecay: {
    auto scalar = Visit(E->getSubExpr());
    assert(scalar.isReference);
    return ValueCategory(scalar.val, /*isReference*/ false);
  }
  case clang::CastKind::CK_ConstructorConversion:
  case clang::CastKind::CK_NoOp: {
    return Visit(E->getSubExpr());
  }
  case clang::CastKind::CK_ToVoid: {
    Visit(E->getSubExpr());
    return nullptr;
  }
  case clang::CastKind::CK_PointerToBoolean: {
    auto scalar = Visit(E->getSubExpr()).getValue(builder);
    if (auto mt = scalar.getType().dyn_cast<mlir::MemRefType>()) {
      scalar = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(mt.getElementType(),
                                     mt.getMemorySpaceAsInt()),
          scalar);
    }
    if (auto LT = scalar.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      auto ne = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, scalar, nullptr_llvm);
      return ValueCategory(ne, /*isReference*/ false);
    }
    function.dump();
    llvm::errs() << "scalar: " << scalar << "\n";
    E->dump();
    assert(0 && "unhandled ptrtobool cast");
  }
  case clang::CastKind::CK_PointerToIntegral: {
    auto scalar = Visit(E->getSubExpr()).getValue(builder);
    if (auto mt = scalar.getType().dyn_cast<mlir::MemRefType>()) {
      scalar = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(mt.getElementType(),
                                     mt.getMemorySpaceAsInt()),
          scalar);
    }
    if (auto LT = scalar.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      mlir::Type mlirType = Glob.getTypes().getMLIRType(E->getType());
      auto val = builder.create<mlir::LLVM::PtrToIntOp>(loc, mlirType, scalar);
      return ValueCategory(val, /*isReference*/ false);
    }
#ifdef DEBUG
    function.dump();
    llvm::errs() << "scalar: " << scalar << "\n";
    E->dump();
#endif
    llvm_unreachable("unhandled ptrtoint cast");
  } break;
  case clang::CastKind::CK_IntegralToBoolean: {
    auto res = Visit(E->getSubExpr()).getValue(builder);
    auto prevTy = res.getType().cast<mlir::IntegerType>();
    res = builder.create<arith::CmpIOp>(
        loc, CmpIPredicate::ne, res,
        builder.create<ConstantIntOp>(loc, 0, prevTy));
    auto postTy =
        Glob.getTypes().getMLIRType(E->getType()).cast<mlir::IntegerType>();
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    if (postTy.getWidth() > 1) {
      if (signedType) {
        res = builder.create<arith::ExtSIOp>(loc, postTy, res);
      } else {
        res = builder.create<arith::ExtUIOp>(loc, postTy, res);
      }
    }
    return ValueCategory(res, /*isReference*/ false);
  }
  case clang::CastKind::CK_FloatingToBoolean: {
    auto res = Visit(E->getSubExpr()).getValue(builder);
    auto prevTy = res.getType().cast<mlir::FloatType>();
    auto postTy =
        Glob.getTypes().getMLIRType(E->getType()).cast<mlir::IntegerType>();
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    auto Zero = builder.create<ConstantFloatOp>(
        loc, APFloat::getZero(prevTy.getFloatSemantics()), prevTy);
    res = builder.create<arith::CmpFOp>(loc, CmpFPredicate::UNE, res, Zero);
    if (1 < postTy.getWidth()) {
      if (signedType) {
        res = builder.create<arith::ExtSIOp>(loc, postTy, res);
      } else {
        res = builder.create<arith::ExtUIOp>(loc, postTy, res);
      }
    }
    return ValueCategory(res, /*isReference*/ false);
  }
  case clang::CastKind::CK_IntegralToPointer: {
    auto vc = Visit(E->getSubExpr());
#ifdef DEBUG
    if (!vc.val) {
      E->dump();
    }
#endif
    assert(vc.val);
    auto res = vc.getValue(builder);
    mlir::Type postTy = Glob.getTypes().getMLIRType(E->getType());
    if (postTy.isa<LLVM::LLVMPointerType>())
      res = builder.create<LLVM::IntToPtrOp>(loc, postTy, res);
    else {
      assert(postTy.isa<MemRefType>());
      res = builder.create<LLVM::IntToPtrOp>(
          loc, LLVM::LLVMPointerType::get(builder.getI8Type()), res);
      res = builder.create<polygeist::Pointer2MemrefOp>(loc, postTy, res);
    }
    return ValueCategory(res, /*isReference*/ false);
  }

  default:
    if (EmittingFunctionDecl)
      EmittingFunctionDecl->dump();
    E->dump();
  }

  llvm_unreachable("unhandled cast");
}

ValueCategory
MLIRScanner::VisitConditionalOperator(clang::ConditionalOperator *E) {
  auto cond = Visit(E->getCond()).getValue(builder);
  assert(cond != nullptr);
  if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
    cond = builder.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
  }
  auto prevTy = cond.getType().cast<mlir::IntegerType>();
  if (!prevTy.isInteger(1)) {
    cond = builder.create<arith::CmpIOp>(
        loc, CmpIPredicate::ne, cond,
        builder.create<ConstantIntOp>(loc, 0, prevTy));
  }
  std::vector<mlir::Type> types;
  if (!E->getType()->isVoidType())
    types.push_back(Glob.getTypes().getMLIRType(E->getType()));
  auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                              /*hasElseRegion*/ true);

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();
  builder.setInsertionPointToStart(&ifOp.getThenRegion().back());

  auto trueExpr = Visit(E->getTrueExpr());

  bool isReference = E->isLValue() || E->isXValue();

  std::vector<mlir::Value> truearray;
  if (!E->getType()->isVoidType()) {
    if (!trueExpr.val) {
      E->dump();
    }
    assert(trueExpr.val);
    mlir::Value truev;
    if (isReference) {
      assert(trueExpr.isReference);
      truev = trueExpr.val;
    } else {
      if (trueExpr.isReference)
        if (auto mt = trueExpr.val.getType().dyn_cast<MemRefType>())
          if (mt.getShape().size() != 1) {
            E->dump();
            E->getTrueExpr()->dump();
            llvm::errs() << " trueExpr: " << trueExpr.val << "\n";
            assert(0);
          }
      truev = trueExpr.getValue(builder);
    }
    assert(truev != nullptr);
    truearray.push_back(truev);
    builder.create<mlir::scf::YieldOp>(loc, truearray);
  }

  builder.setInsertionPointToStart(&ifOp.getElseRegion().back());

  auto falseExpr = Visit(E->getFalseExpr());
  std::vector<mlir::Value> falsearray;
  if (!E->getType()->isVoidType()) {
    mlir::Value falsev;
    if (isReference) {
      assert(falseExpr.isReference);
      falsev = falseExpr.val;
    } else
      falsev = falseExpr.getValue(builder);
    assert(falsev != nullptr);
    falsearray.push_back(falsev);
    builder.create<mlir::scf::YieldOp>(loc, falsearray);
  }

  builder.setInsertionPoint(oldblock, oldpoint);

  for (size_t i = 0; i < truearray.size(); i++)
    types[i] = truearray[i].getType();
  auto newIfOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                                 /*hasElseRegion*/ true);
  newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
  newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
  ifOp.erase();
  return ValueCategory(newIfOp.getResult(0), /*isReference*/ isReference);
}

ValueCategory MLIRScanner::VisitStmtExpr(clang::StmtExpr *stmt) {
  ValueCategory off = nullptr;
  for (auto a : stmt->getSubStmt()->children()) {
    off = Visit(a);
  }
  return off;
}

ValueCategory MLIRScanner::VisitBinAssign(BinaryOperator *e) {
  LLVM_DEBUG({
    llvm::dbgs() << "VisitBinAssign: ";
    e->dump();
    llvm::dbgs() << "\n";
  });
  auto rhs = Visit(e->getRHS());
  auto lhs = Visit(e->getLHS());

  assert(lhs.isReference);
  mlir::Value tostore = rhs.getValue(builder);
  mlir::Type subType;
  if (auto PT = lhs.val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>())
    subType = PT.getElementType();
  else
    subType = lhs.val.getType().cast<MemRefType>().getElementType();
  if (tostore.getType() != subType) {
    if (auto prevTy = tostore.getType().dyn_cast<mlir::IntegerType>()) {
      if (auto postTy = subType.dyn_cast<mlir::IntegerType>()) {
        bool signedType = true;
        if (auto bit = dyn_cast<BuiltinType>(&*e->getType())) {
          if (bit->isUnsignedInteger())
            signedType = false;
          if (bit->isSignedInteger())
            signedType = true;
        }

        if (prevTy.getWidth() < postTy.getWidth()) {
          if (signedType) {
            tostore = builder.create<arith::ExtSIOp>(loc, postTy, tostore);
          } else {
            tostore = builder.create<arith::ExtUIOp>(loc, postTy, tostore);
          }
        } else if (prevTy.getWidth() > postTy.getWidth()) {
          tostore = builder.create<arith::TruncIOp>(loc, postTy, tostore);
        }
      }
    }
  }
  lhs.store(builder, tostore);
  return rhs;
}

static bool isSigned(QualType Ty) {
  // TODO note assumptions made here about unsigned / unordered
  bool signedType = true;
  if (auto bit = dyn_cast<clang::BuiltinType>(Ty)) {
    if (bit->isUnsignedInteger())
      signedType = false;
    if (bit->isSignedInteger())
      signedType = true;
  }
  return signedType;
}

class BinOpInfo {
public:
  BinOpInfo(ValueCategory LHS, ValueCategory RHS, QualType Ty,
            BinaryOperator::Opcode Opcode, const BinaryOperator *Expr)
      : LHS(LHS), RHS(RHS), Ty(Ty), Opcode(Opcode), Expr(Expr) {}

  ValueCategory getLHS() const { return LHS; }
  ValueCategory getRHS() const { return RHS; }
  constexpr QualType getType() const { return Ty; }
  constexpr BinaryOperator::Opcode getOpcode() const { return Opcode; }
  constexpr const BinaryOperator *getExpr() const { return Expr; }

private:
  const ValueCategory LHS;
  const ValueCategory RHS;
  const QualType Ty;                   // Computation Type.
  const BinaryOperator::Opcode Opcode; // Opcode of BinOp to perform
  const BinaryOperator *Expr;
};

ValueCategory MLIRScanner::EmitPromoted(Expr *E, QualType PromotionType) {
  assert(E && "Invalid input expression.");
  E = E->IgnoreParens();
  if (auto *BO = dyn_cast<BinaryOperator>(E)) {
    switch (BO->getOpcode()) {
#define HANDLEBINOP(OP)                                                        \
  case BO_##OP:                                                                \
    return EmitBin##OP(EmitBinOps(BO, PromotionType));

      HANDLEBINOP(Add)
      HANDLEBINOP(Sub)
      HANDLEBINOP(Mul)
      HANDLEBINOP(Div)
#undef HANDLEBINOP
    default:
      break;
    }
  } else if (auto *UO = dyn_cast<UnaryOperator>(E)) {
    switch (UO->getOpcode()) {
    case UO_Imag:
    case UO_Real:
    case UO_Minus:
    case UO_Plus:
      llvm::WithColor::warning() << "Default promotion for unary operation\n";
      LLVM_FALLTHROUGH;
    default:
      break;
    }
  }

  const ValueCategory Res = Visit(E);
  if (Res.val) {
    if (!PromotionType.isNull())
      return EmitPromotedValue(Res, PromotionType);
    return EmitUnPromotedValue(Res, E->getType());
  }
  return Res;
}

ValueCategory MLIRScanner::EmitPromotedValue(ValueCategory Result,
                                             QualType PromotionType) {
  return Result.FPExt(builder, Glob.getTypes().getMLIRType(PromotionType));
}

ValueCategory MLIRScanner::EmitUnPromotedValue(ValueCategory Result,
                                               QualType PromotionType) {
  return Result.FPTrunc(builder, Glob.getTypes().getMLIRType(PromotionType));
}

ValueCategory MLIRScanner::EmitPromotedScalarExpr(Expr *E,
                                                  QualType PromotionType) {
  if (!PromotionType.isNull())
    return EmitPromoted(E, PromotionType);
  return Visit(E);
}

#define HANDLEBINOP(OP)                                                        \
  ValueCategory MLIRScanner::VisitBin##OP(BinaryOperator *E) {                 \
    LLVM_DEBUG({                                                               \
      llvm::dbgs() << "VisitBin" #OP ": ";                                     \
      E->dump();                                                               \
      llvm::dbgs() << "\n";                                                    \
    });                                                                        \
    QualType PromotionType = Glob.getTypes().getPromotionType(E->getType());   \
    ValueCategory Result = EmitBin##OP(EmitBinOps(E, PromotionType));          \
    if (Result.val && !PromotionType.isNull())                                 \
      Result = EmitUnPromotedValue(Result, E->getType());                      \
    return Result;                                                             \
  }
#include "Expressions.def"
#undef HANDLEBINOP

BinOpInfo MLIRScanner::EmitBinOps(BinaryOperator *E, QualType PromotionType) {
  const ValueCategory LHS = EmitPromotedScalarExpr(E->getLHS(), PromotionType);
  const ValueCategory RHS = EmitPromotedScalarExpr(E->getRHS(), PromotionType);
  const QualType Ty = !PromotionType.isNull() ? PromotionType : E->getType();
  const BinaryOperator::Opcode Opcode = E->getOpcode();
  return {LHS, RHS, Ty, Opcode, E};
}

ValueCategory MLIRScanner::EmitBinMul(const BinOpInfo &Info) {
  assert(Info.getOpcode() == BinaryOperator::Opcode::BO_Mul &&
         "Invalid binary expression");
  auto lhs_v = Info.getLHS().getValue(builder);
  auto rhs_v = Info.getRHS().getValue(builder);
  if (lhs_v.getType().isa<mlir::FloatType>()) {
    return ValueCategory(builder.create<arith::MulFOp>(loc, lhs_v, rhs_v),
                         /*isReference*/ false);
  } else {
    return ValueCategory(builder.create<arith::MulIOp>(loc, lhs_v, rhs_v),
                         /*isReference*/ false);
  }
}

ValueCategory MLIRScanner::EmitBinDiv(const BinOpInfo &Info) {
  assert(Info.getOpcode() == BinaryOperator::Opcode::BO_Div &&
         "Invalid binary expression");
  auto lhs_v = Info.getLHS().getValue(builder);
  auto rhs_v = Info.getRHS().getValue(builder);
  if (lhs_v.getType().isa<mlir::FloatType>()) {
    return ValueCategory(builder.create<arith::DivFOp>(loc, lhs_v, rhs_v),
                         /*isReference*/ false);
  } else {
    if (isSigned(Info.getType()))
      return ValueCategory(builder.create<arith::DivSIOp>(loc, lhs_v, rhs_v),
                           /*isReference*/ false);
    else
      return ValueCategory(builder.create<arith::DivUIOp>(loc, lhs_v, rhs_v),
                           /*isReference*/ false);
  }
}

ValueCategory MLIRScanner::EmitBinRem(const BinOpInfo &Info) {
  assert(Info.getOpcode() == BinaryOperator::Opcode::BO_Rem &&
         "Invalid binary expression");
  auto lhs_v = Info.getLHS().getValue(builder);
  auto rhs_v = Info.getRHS().getValue(builder);
  if (lhs_v.getType().isa<mlir::FloatType>()) {
    return ValueCategory(builder.create<arith::RemFOp>(loc, lhs_v, rhs_v),
                         /*isReference*/ false);
  } else {
    if (isSigned(Info.getType()))
      return ValueCategory(builder.create<arith::RemSIOp>(loc, lhs_v, rhs_v),
                           /*isReference*/ false);
    else
      return ValueCategory(builder.create<arith::RemUIOp>(loc, lhs_v, rhs_v),
                           /*isReference*/ false);
  }
}

ValueCategory MLIRScanner::EmitBinAdd(const BinOpInfo &Info) {
  assert(Info.getOpcode() == BinaryOperator::Opcode::BO_Add &&
         "Invalid binary expression");
  auto lhs_v = Info.getLHS().getValue(builder);
  auto rhs_v = Info.getRHS().getValue(builder);
  if (lhs_v.getType().isa<mlir::FloatType>()) {
    return ValueCategory(builder.create<AddFOp>(loc, lhs_v, rhs_v),
                         /*isReference*/ false);
  } else if (auto mt = lhs_v.getType().dyn_cast<mlir::MemRefType>()) {
    auto shape = std::vector<int64_t>(mt.getShape());
    shape[0] = -1;
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    auto ptradd = rhs_v;
    ptradd = castToIndex(loc, ptradd);
    return ValueCategory(
        builder.create<polygeist::SubIndexOp>(loc, mt0, lhs_v, ptradd),
        /*isReference*/ false);
  } else if (auto pt =
                 lhs_v.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    return ValueCategory(builder.create<LLVM::GEPOp>(
                             loc, pt, lhs_v, std::vector<mlir::Value>({rhs_v})),
                         /*isReference*/ false);
  } else {
    if (auto lhs_c = lhs_v.getDefiningOp<ConstantIntOp>()) {
      if (auto rhs_c = rhs_v.getDefiningOp<ConstantIntOp>()) {
        return ValueCategory(
            builder.create<arith::ConstantIntOp>(
                loc, lhs_c.value() + rhs_c.value(), lhs_c.getType()),
            false);
      }
    }
    return ValueCategory(builder.create<AddIOp>(loc, lhs_v, rhs_v),
                         /*isReference*/ false);
  }
}

ValueCategory MLIRScanner::EmitBinSub(const BinOpInfo &Info) {
  assert(Info.getOpcode() == BinaryOperator::Opcode::BO_Sub &&
         "Invalid binary expression");
  auto lhs_v = Info.getLHS().getValue(builder);
  auto rhs_v = Info.getRHS().getValue(builder);
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
  if (lhs_v.getType().isa<mlir::FloatType>()) {
    assert(rhs_v.getType() == lhs_v.getType());
    return ValueCategory(builder.create<SubFOp>(loc, lhs_v, rhs_v),
                         /*isReference*/ false);
  } else if (auto pt =
                 lhs_v.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    if (auto IT = rhs_v.getType().dyn_cast<mlir::IntegerType>()) {
      mlir::Value vals[1] = {builder.create<SubIOp>(
          loc, builder.create<ConstantIntOp>(loc, 0, IT.getWidth()), rhs_v)};
      return ValueCategory(
          builder.create<LLVM::GEPOp>(loc, lhs_v.getType(), lhs_v,
                                      ArrayRef<mlir::Value>(vals)),
          false);
    }
    mlir::Value val = builder.create<SubIOp>(
        loc,
        builder.create<LLVM::PtrToIntOp>(
            loc, Glob.getTypes().getMLIRType(Info.getType()), lhs_v),
        builder.create<LLVM::PtrToIntOp>(
            loc, Glob.getTypes().getMLIRType(Info.getType()), rhs_v));
    val = builder.create<DivSIOp>(
        loc, val,
        builder.create<IndexCastOp>(
            loc, val.getType(),
            builder.create<polygeist::TypeSizeOp>(
                loc, builder.getIndexType(),
                mlir::TypeAttr::get(pt.getElementType()))));
    return ValueCategory(val, /*isReference*/ false);
  } else {
    return ValueCategory(builder.create<SubIOp>(loc, lhs_v, rhs_v),
                         /*isReference*/ false);
  }
}

ValueCategory MLIRScanner::EmitBinShl(const BinOpInfo &Info) {
  assert(Info.getOpcode() == BinaryOperator::Opcode::BO_Shl &&
         "Invalid binary expression");
  auto lhsv = Info.getLHS().getValue(builder);
  auto rhsv = Info.getRHS().getValue(builder);
  auto prevTy = rhsv.getType().cast<mlir::IntegerType>();
  auto postTy = lhsv.getType().cast<mlir::IntegerType>();
  if (prevTy.getWidth() < postTy.getWidth())
    rhsv = builder.create<arith::ExtUIOp>(loc, postTy, rhsv);
  if (prevTy.getWidth() > postTy.getWidth())
    rhsv = builder.create<arith::TruncIOp>(loc, postTy, rhsv);
  assert(lhsv.getType() == rhsv.getType());
  return ValueCategory(builder.create<ShLIOp>(loc, lhsv, rhsv),
                       /*isReference*/ false);
}

ValueCategory MLIRScanner::EmitBinShr(const BinOpInfo &Info) {
  assert(Info.getOpcode() == BinaryOperator::Opcode::BO_Shr &&
         "Invalid binary expression");
  auto lhsv = Info.getLHS().getValue(builder);
  auto rhsv = Info.getRHS().getValue(builder);
  auto prevTy = rhsv.getType().cast<mlir::IntegerType>();
  auto postTy = lhsv.getType().cast<mlir::IntegerType>();
  if (prevTy.getWidth() < postTy.getWidth())
    rhsv = builder.create<mlir::arith::ExtUIOp>(loc, postTy, rhsv);
  if (prevTy.getWidth() > postTy.getWidth())
    rhsv = builder.create<mlir::arith::TruncIOp>(loc, postTy, rhsv);
  assert(lhsv.getType() == rhsv.getType());
  if (isSigned(Info.getExpr()->getType()))
    return ValueCategory(builder.create<ShRSIOp>(loc, lhsv, rhsv),
                         /*isReference*/ false);
  else
    return ValueCategory(builder.create<ShRUIOp>(loc, lhsv, rhsv),
                         /*isReference*/ false);
}

ValueCategory MLIRScanner::EmitBinAnd(const BinOpInfo &Info) {
  assert(Info.getOpcode() == BinaryOperator::Opcode::BO_And &&
         "Invalid binary expression");
  return ValueCategory(builder.create<AndIOp>(loc,
                                              Info.getLHS().getValue(builder),
                                              Info.getRHS().getValue(builder)),
                       /*isReference*/ false);
}

ValueCategory MLIRScanner::EmitBinXor(const BinOpInfo &Info) {
  assert(Info.getOpcode() == BinaryOperator::Opcode::BO_Xor &&
         "Invalid binary expression");
  return ValueCategory(builder.create<XOrIOp>(loc,
                                              Info.getLHS().getValue(builder),
                                              Info.getRHS().getValue(builder)),
                       /*isReference*/ false);
}

ValueCategory MLIRScanner::EmitBinOr(const BinOpInfo &Info) {
  assert(Info.getOpcode() == BinaryOperator::Opcode::BO_Or &&
         "Invalid binary expression");
  // TODO short circuit
  return ValueCategory(builder.create<OrIOp>(loc,
                                             Info.getLHS().getValue(builder),
                                             Info.getRHS().getValue(builder)),
                       /*isReference*/ false);
}
