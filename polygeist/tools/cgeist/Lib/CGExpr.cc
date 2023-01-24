//===--- CGExpr.cc - Emit MLIR Code from Expressions ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeUtils.h"
#include "clang-mlir.h"
#include "mlir/Dialect/SYCL/MethodUtils.h"
#include "utils.h"

#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/WithColor.h"

#include <numeric>

#define DEBUG_TYPE "CGExpr"

using namespace clang;
using namespace mlir;

extern llvm::cl::opt<bool> GenerateAllSYCLFuncs;
extern llvm::cl::opt<bool> OmitOptionalMangledFunctionName;

ValueCategory
MLIRScanner::VisitExtVectorElementExpr(clang::ExtVectorElementExpr *Expr) {
  auto Base = Visit(Expr->getBase());
  SmallVector<uint32_t, 4> Indices;
  Expr->getEncodedElementAccess(Indices);
  assert(Indices.size() == 1 &&
         "The support for higher dimensions to be implemented.");
  assert(Base.isReference);
  assert(Base.val.getType().isa<MemRefType>() &&
         "Expecting ExtVectorElementExpr to have memref type");
  auto MT = Base.val.getType().cast<MemRefType>();
  assert(MT.getElementType().isa<mlir::VectorType>() &&
         "Expecting ExtVectorElementExpr to have memref of vector elements");
  ValueCategory Val{Base.getValue(Builder), false};
  return Val.Extract(Builder, Loc, Indices[0]);
}

ValueCategory MLIRScanner::VisitConstantExpr(clang::ConstantExpr *Expr) {
  auto Sv = Visit(Expr->getSubExpr());
  if (auto Ty = Glob.getTypes()
                    .getMLIRType(Expr->getType())
                    .dyn_cast<mlir::IntegerType>()) {
    if (Expr->hasAPValueResult())
      return ValueCategory(Builder.create<arith::ConstantIntOp>(
                               getMLIRLocation(Expr->getExprLoc()),
                               Expr->getResultAsAPSInt().getExtValue(), Ty),
                           /*isReference*/ false);
  }
  assert(Sv.val);
  return Sv;
}

ValueCategory MLIRScanner::VisitTypeTraitExpr(clang::TypeTraitExpr *Expr) {
  auto Ty =
      Glob.getTypes().getMLIRType(Expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(
      Builder.create<arith::ConstantIntOp>(getMLIRLocation(Expr->getExprLoc()),
                                           Expr->getValue(), Ty),
      /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitGNUNullExpr(clang::GNUNullExpr *Expr) {
  auto Ty =
      Glob.getTypes().getMLIRType(Expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(Builder.create<arith::ConstantIntOp>(
                           getMLIRLocation(Expr->getExprLoc()), 0, Ty),
                       /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitIntegerLiteral(clang::IntegerLiteral *Expr) {
  auto Ty =
      Glob.getTypes().getMLIRType(Expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(
      Builder.create<arith::ConstantIntOp>(getMLIRLocation(Expr->getExprLoc()),
                                           Expr->getValue().getSExtValue(), Ty),
      /*isReference*/ false);
}

ValueCategory
MLIRScanner::VisitCharacterLiteral(clang::CharacterLiteral *Expr) {
  auto Ty =
      Glob.getTypes().getMLIRType(Expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(
      Builder.create<arith::ConstantIntOp>(getMLIRLocation(Expr->getExprLoc()),
                                           Expr->getValue(), Ty),
      /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitFloatingLiteral(clang::FloatingLiteral *Expr) {
  auto Ty =
      Glob.getTypes().getMLIRType(Expr->getType()).cast<mlir::FloatType>();
  return ValueCategory(
      Builder.create<arith::ConstantFloatOp>(
          getMLIRLocation(Expr->getExprLoc()), Expr->getValue(), Ty),
      /*isReference*/ false);
}

ValueCategory
MLIRScanner::VisitImaginaryLiteral(clang::ImaginaryLiteral *Expr) {
  auto MT = Glob.getTypes().getMLIRType(Expr->getType()).cast<MemRefType>();
  auto Ty = MT.getElementType().cast<FloatType>();

  OpBuilder Abuilder(Builder.getContext());
  Abuilder.setInsertionPointToStart(AllocationScope);
  auto Iloc = getMLIRLocation(Expr->getExprLoc());
  auto Alloc = Abuilder.create<mlir::memref::AllocaOp>(Iloc, MT);
  Builder.create<mlir::memref::StoreOp>(
      Iloc,
      Builder.create<arith::ConstantFloatOp>(
          Iloc, APFloat(Ty.getFloatSemantics(), "0"), Ty),
      Alloc, getConstantIndex(0));
  Builder.create<mlir::memref::StoreOp>(
      Iloc, Visit(Expr->getSubExpr()).getValue(Builder), Alloc,
      getConstantIndex(1));
  return ValueCategory(Alloc,
                       /*isReference*/ true);
}

ValueCategory
MLIRScanner::VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *Expr) {
  auto Ty =
      Glob.getTypes().getMLIRType(Expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(
      Builder.create<arith::ConstantIntOp>(getMLIRLocation(Expr->getExprLoc()),
                                           Expr->getValue(), Ty),
      /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitStringLiteral(clang::StringLiteral *Expr) {
  auto Loc = getMLIRLocation(Expr->getExprLoc());
  return ValueCategory(
      Glob.getOrCreateGlobalLLVMString(Loc, Builder, Expr->getString(),
                                       mlirclang::getFuncContext(Function)),
      /*isReference*/ true);
}

ValueCategory MLIRScanner::VisitParenExpr(clang::ParenExpr *Expr) {
  return Visit(Expr->getSubExpr());
}

ValueCategory
MLIRScanner::VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *Decl) {
  mlir::Type MLIRTy = Glob.getTypes().getMLIRType(Decl->getType());

  if (auto FT = MLIRTy.dyn_cast<mlir::FloatType>())
    return ValueCategory(Builder.create<arith::ConstantFloatOp>(
                             Loc, APFloat(FT.getFloatSemantics(), "0"), FT),
                         /*isReference*/ false);

  if (auto IT = MLIRTy.dyn_cast<mlir::IntegerType>())
    return ValueCategory(Builder.create<arith::ConstantIntOp>(Loc, 0, IT),
                         /*isReference*/ false);

  if (auto MT = MLIRTy.dyn_cast<mlir::MemRefType>())
    return ValueCategory(
        Builder.create<polygeist::Pointer2MemrefOp>(
            Loc, MT,
            Builder.create<mlir::LLVM::NullOp>(
                Loc, LLVM::LLVMPointerType::get(Builder.getI8Type(),
                                                MT.getMemorySpaceAsInt()))),
        false);

  if (auto PT = MLIRTy.dyn_cast<mlir::LLVM::LLVMPointerType>())
    return ValueCategory(Builder.create<mlir::LLVM::NullOp>(Loc, PT), false);

  for (auto *Child : Decl->children())
    Child->dump();

  Decl->dump();
  llvm::errs() << " mty: " << MLIRTy << "\n";
  llvm_unreachable("bad");
}

/// Construct corresponding MLIR operations to initialize the given value by a
/// provided InitListExpr.
mlir::Attribute MLIRScanner::InitializeValueByInitListExpr(mlir::Value ToInit,
                                                           clang::Expr *Expr) {
  LLVM_DEBUG({
    llvm::dbgs() << "InitializeValueByInitListExpr: ";
    Expr->dump();
    llvm::dbgs() << "\n";
  });

  // Struct initialization requires an extra 0, since the first index is the
  // pointer index, and then the struct index.
  const clang::Type *PTT = Expr->getType()->getUnqualifiedDesugaredType();

  bool Inner = false;
  if (isa<RecordType>(PTT) || isa<clang::ComplexType>(PTT)) {
    if (auto MT = ToInit.getType().dyn_cast<MemRefType>())
      Inner = true;
  }

  while (auto CO = ToInit.getDefiningOp<memref::CastOp>())
    ToInit = CO.getSource();

  // Recursively visit the initialization expression following the linear
  // increment of the memory address.
  std::function<mlir::DenseElementsAttr(clang::Expr *, mlir::Value, bool)>
      Helper = [&](class Expr *Expr, mlir::Value ToInit,
                   bool Inner) -> mlir::DenseElementsAttr {
    Location Loc = ToInit.getLoc();
    if (auto *InitListExpr = dyn_cast<clang::InitListExpr>(Expr)) {
      if (Inner) {
        if (auto MT = ToInit.getType().dyn_cast<MemRefType>()) {
          auto Shape = std::vector<int64_t>(MT.getShape());
          assert(!Shape.empty());
          if (Shape.size() > 1)
            Shape.erase(Shape.begin());
          else
            Shape[0] = ShapedType::kDynamic;
          auto MT0 = mlir::MemRefType::get(Shape, MT.getElementType(),
                                           MemRefLayoutAttrInterface(),
                                           MT.getMemorySpace());
          ToInit = Builder.create<polygeist::SubIndexOp>(Loc, MT0, ToInit,
                                                         getConstantIndex(0));
        }
      }

      unsigned Num = InitListExpr->getNumInits();
      if (InitListExpr->hasArrayFiller()) {
        if (auto MT = ToInit.getType().dyn_cast<MemRefType>()) {
          auto Shape = MT.getShape();
          assert(Shape.size() > 0);
          assert(Shape[0] != ShapedType::kDynamic);
          Num = Shape[0];
        } else if (auto PT =
                       ToInit.getType().dyn_cast<LLVM::LLVMPointerType>()) {
          if (auto AT = PT.getElementType().dyn_cast<LLVM::LLVMArrayType>())
            Num = AT.getNumElements();
          else if (auto AT =
                       PT.getElementType().dyn_cast<LLVM::LLVMStructType>())
            Num = AT.getBody().size();
          else {
            ToInit.getType().dump();
            llvm_unreachable(
                "TODO get number of values in array filler expression");
          }
        } else {
          ToInit.getType().dump();
          llvm_unreachable(
              "TODO get number of values in array filler expression");
        }
      }

      SmallVector<char> Attrs;
      bool AllSub = true;
      for (unsigned I = 0, E = Num; I < E; ++I) {
        mlir::Value Next;
        if (auto MT = ToInit.getType().dyn_cast<MemRefType>()) {
          llvm::SmallVector<int64_t> Shape(MT.getShape());
          assert(!Shape.empty());
          Shape[0] = ShapedType::kDynamic;

          mlir::Type ElemTy = MT.getElementType();
          mlir::Type ET;
          if (auto ST = ElemTy.dyn_cast<mlir::LLVM::LLVMStructType>())
            ET = mlir::MemRefType::get(Shape, ST.getBody()[I],
                                       MemRefLayoutAttrInterface(),
                                       MT.getMemorySpace());
          else if (sycl::isSYCLType(ElemTy))
            ET =
                TypeSwitch<mlir::Type, mlir::MemRefType>(ElemTy)
                    .Case<mlir::sycl::IDType, mlir::sycl::RangeType>([&](auto) {
                      return mlir::MemRefType::get(Shape, ElemTy,
                                                   MemRefLayoutAttrInterface(),
                                                   MT.getMemorySpace());
                    })
                    .Case<mlir::sycl::ItemBaseType>([&](auto Ty) {
                      return mlir::MemRefType::get(Shape, Ty.getBody()[I],
                                                   MemRefLayoutAttrInterface(),
                                                   MT.getMemorySpace());
                    });
          else
            ET = mlir::MemRefType::get(Shape, ElemTy,
                                       MemRefLayoutAttrInterface(),
                                       MT.getMemorySpace());

          Next = Builder.create<polygeist::SubIndexOp>(Loc, ET, ToInit,
                                                       getConstantIndex(I));
        } else {
          auto PT = ToInit.getType().cast<LLVM::LLVMPointerType>();
          auto ET = PT.getElementType();
          if (auto ST = ET.dyn_cast<LLVM::LLVMStructType>())
            Next = Builder.create<LLVM::GEPOp>(
                Loc,
                LLVM::LLVMPointerType::get(ST.getBody()[I],
                                           PT.getAddressSpace()),
                ToInit, llvm::ArrayRef<mlir::LLVM::GEPArg>{0, I});
          else if (auto AT = ET.dyn_cast<LLVM::LLVMArrayType>())
            Next = Builder.create<LLVM::GEPOp>(
                Loc,
                LLVM::LLVMPointerType::get(AT.getElementType(),
                                           PT.getAddressSpace()),
                ToInit, llvm::ArrayRef<mlir::LLVM::GEPArg>{0, I});
          else if (ET.isInteger(8))
            Next = Builder.create<LLVM::GEPOp>(
                Loc, LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
                ToInit, llvm::ArrayRef<mlir::LLVM::GEPArg>{I});
          else
            llvm_unreachable("unknown inner type");
        }

        auto Sub =
            Helper(InitListExpr->hasArrayFiller() ? InitListExpr->getInit(0)
                                                  : InitListExpr->getInit(I),
                   Next, true);
        if (Sub) {
          size_t N = 1;
          if (Sub.isSplat())
            N = Sub.size();
          for (size_t I = 0; I < N; I++)
            for (auto Ea : Sub.getRawData())
              Attrs.push_back(Ea);
        } else
          AllSub = false;
      }

      if (!AllSub)
        return mlir::DenseElementsAttr();

      if (auto MT = ToInit.getType().dyn_cast<MemRefType>())
        return DenseElementsAttr::getFromRawBuffer(
            RankedTensorType::get(MT.getShape(), MT.getElementType()), Attrs);

      return mlir::DenseElementsAttr();
    }

    bool IsArray = false;
    Glob.getTypes().getMLIRType(Expr->getType(), &IsArray);
    ValueCategory Sub = Visit(Expr);
    ValueCategory(ToInit, /*isReference*/ true).store(Builder, Sub, IsArray);
    if (!Sub.isReference)
      if (auto MT = ToInit.getType().dyn_cast<MemRefType>()) {
        if (auto Cop = Sub.val.getDefiningOp<arith::ConstantIntOp>()) {
          const auto C = Cop.getValue();
          const auto CT = C.getType();
          const auto ET = MT.getElementType();
          assert((CT == ET || (CT.isInteger(1) && ET.isInteger(8))) &&
                 "Expecting same width but for boolean values");
          return DenseElementsAttr::get(RankedTensorType::get(1, CT), C);
        }
        if (auto Cop = Sub.val.getDefiningOp<arith::ConstantFloatOp>())
          return DenseElementsAttr::get(
              RankedTensorType::get(std::vector<int64_t>({1}),
                                    MT.getElementType()),
              Cop.getValue());
      }
    return mlir::DenseElementsAttr();
  };

  return Helper(Expr, ToInit, Inner);
}

ValueCategory
MLIRScanner::VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr *Expr) {
  return Visit(Expr->getExpr());
}

ValueCategory MLIRScanner::VisitCXXThisExpr(clang::CXXThisExpr *Expr) {
  return ThisVal;
}

ValueCategory MLIRScanner::VisitPredefinedExpr(clang::PredefinedExpr *Expr) {
  return VisitStringLiteral(Expr->getFunctionName());
}

ValueCategory MLIRScanner::EmitVectorInitList(InitListExpr *Expr,
                                              mlir::VectorType VType) {
  const Location Loc = getMLIRLocation(Expr->getExprLoc());
  const int64_t ResElts = VType.getNumElements();

  int64_t CurIdx = 0;
  auto V = ValueCategory::getNullValue(Builder, Loc, VType);
  SmallVector<int64_t> ShuffleMask(ResElts);
  for (auto *IE : Expr->children()) {
    ValueCategory Init = Visit(IE);

    auto VVT = Init.val.getType().dyn_cast<mlir::VectorType>();

    // Handle scalar elements.
    if (!VVT) {
      V = V.Insert(Builder, Loc, Init.val, CurIdx);
      ++CurIdx;
      continue;
    }

    // Extend init to result vector length, and then shuffle its contribution
    // to the vector initializer into V.
    Init = Init.Reshape(Builder, Loc, VType.getShape());

    const int64_t InitElts = VVT.getNumElements();

    auto *const Begin = ShuffleMask.begin();
    auto *const EndMask0 = Begin + CurIdx;
    auto *const EndMask1 = EndMask0 + InitElts;
    // Keep the current contents
    std::iota(Begin, EndMask0, 0);
    // Concat the input vector
    std::iota(EndMask0, EndMask1, ResElts);
    // Fill the rest with 0s from the current vector
    std::fill(EndMask1, ShuffleMask.end(), CurIdx);

    V = V.Shuffle(Builder, Loc, Init.val, ShuffleMask);
    CurIdx += InitElts;
  }
  return V;
}

ValueCategory MLIRScanner::VisitInitListExpr(clang::InitListExpr *Expr) {
  LLVM_DEBUG({
    llvm::dbgs() << "VisitInitListExpr: ";
    Expr->dump();
    llvm::dbgs() << "\n";
  });

  assert(!Expr->hadArrayRangeDesignator() && "Unsupported");

  if (auto VType = Glob.getTypes()
                       .getMLIRType(Expr->getType())
                       .dyn_cast<mlir::VectorType>())
    return EmitVectorInitList(Expr, VType);

  mlir::Type SubType = Glob.getTypes().getMLIRType(Expr->getType());
  bool IsArray = false, LLVMABI = false;

  if (Glob.getTypes()
          .getMLIRType(Glob.getCGM().getContext().getLValueReferenceType(
              Expr->getType()))
          .isa<mlir::LLVM::LLVMPointerType>())
    LLVMABI = true;
  else {
    Glob.getTypes().getMLIRType(Expr->getType(), &IsArray);
    if (IsArray)
      SubType = Glob.getTypes().getMLIRType(
          Glob.getCGM().getContext().getLValueReferenceType(Expr->getType()));
  }

  auto Op = createAllocOp(SubType, nullptr, /*memtype*/ 0, IsArray, LLVMABI);
  InitializeValueByInitListExpr(Op, Expr);
  return ValueCategory(Op, true);
}

ValueCategory MLIRScanner::VisitCXXStdInitializerListExpr(
    clang::CXXStdInitializerListExpr *Expr) {
  auto ArrayPtr = Visit(Expr->getSubExpr());
  const ConstantArrayType *ArrayType =
      Glob.getCGM().getContext().getAsConstantArrayType(
          Expr->getSubExpr()->getType());
  assert(ArrayType && "std::initializer_list constructed from non-array");

  // FIXME: Perform the checks on the field types in SemaInit.
  RecordDecl *Record = Expr->getType()->castAs<RecordType>()->getDecl();
  auto Field = Record->field_begin();

  LLVM::LLVMStructType SubType =
      Glob.getTypes().getMLIRType(Expr->getType()).cast<LLVM::LLVMStructType>();
  assert(SubType.getBody().size() == 2 && "Expecting two fields");

  mlir::Value Alloca = createAllocOp(SubType, nullptr, /*memtype*/ 0,
                                     /*isArray*/ false, /*LLVMABI*/ true);

  ArrayPtr = CommonArrayToPointer(ArrayPtr);
  if (SubType.getBody()[0].isa<mlir::MemRefType>())
    ArrayPtr = ArrayPtr.Ptr2MemRef(Builder, Loc);

  auto Zero = Builder.create<arith::ConstantIntOp>(Loc, 0, 32);
  auto GEP0 = Builder.create<LLVM::GEPOp>(
      Loc, LLVM::LLVMPointerType::get(SubType.getBody()[0], 0), Alloca,
      ValueRange({Zero, Zero}));
  Builder.create<LLVM::StoreOp>(Loc, ArrayPtr.getValue(Builder), GEP0);
  auto GEP1 = Builder.create<LLVM::GEPOp>(
      Loc, LLVM::LLVMPointerType::get(SubType.getBody()[1], 0), Alloca,
      ValueRange({Zero, Builder.create<arith::ConstantIntOp>(Loc, 1, 32)}));
  ++Field;
  auto ITy =
      Glob.getTypes().getMLIRType(Field->getType()).cast<mlir::IntegerType>();
  Builder.create<LLVM::StoreOp>(
      Loc,
      Builder.create<arith::ConstantIntOp>(
          Loc, ArrayType->getSize().getZExtValue(), ITy.getWidth()),
      GEP1);
  mlir::Value Load = Builder.create<mlir::LLVM::LoadOp>(Loc, Alloca);
  return ValueCategory(Load, /*isRef*/ false);
}

ValueCategory
MLIRScanner::VisitArrayInitIndexExpr(clang::ArrayInitIndexExpr *Expr) {
  assert(ArrayInit.size());
  return ValueCategory(
      Builder.create<arith::IndexCastOp>(
          Loc, Glob.getTypes().getMLIRType(Expr->getType()), ArrayInit.back()),
      /*isReference*/ false);
}

static const clang::ConstantArrayType *getCAT(const clang::Type *T) {
  if (const auto *CAT = dyn_cast<clang::ConstantArrayType>(T))
    return CAT;

  const clang::Type *Child = nullptr;
  if (const auto *ET = dyn_cast<clang::ElaboratedType>(T))
    Child = ET->getNamedType().getTypePtr();
  else if (const auto *TypeDefT = dyn_cast<clang::TypedefType>(T))
    Child = TypeDefT->getUnqualifiedDesugaredType();
  else
    llvm_unreachable("Unhandled case\n");

  return getCAT(Child);
}

ValueCategory MLIRScanner::VisitArrayInitLoop(clang::ArrayInitLoopExpr *Expr,
                                              ValueCategory ToStore) {
  CGEIST_WARNING(llvm::WithColor::warning()
                 << "recomputing common in arrayinitloopexpr\n");

  const clang::ConstantArrayType *CAT = getCAT(Expr->getType().getTypePtr());
  std::vector<mlir::Value> Start = {getConstantIndex(0)};
  std::vector<mlir::Value> Sizes = {
      getConstantIndex(CAT->getSize().getLimitedValue())};
  AffineMap Map = Builder.getSymbolIdentityMap();
  auto AffineOp = Builder.create<AffineForOp>(Loc, Start, Map, Sizes, Map);

  Block::iterator OldPoint = Builder.getInsertionPoint();
  Block *OldBlock = Builder.getInsertionBlock();

  Builder.setInsertionPointToStart(&AffineOp.getLoopBody().front());

  ArrayInit.push_back(AffineOp.getInductionVar());

  ValueCategory Alu =
      CommonArrayLookup(CommonArrayToPointer(ToStore),
                        AffineOp.getInductionVar(), /*isImplicitRef*/ false);

  if (auto *AILE = dyn_cast<ArrayInitLoopExpr>(Expr->getSubExpr())) {
    VisitArrayInitLoop(AILE, Alu);
  } else {
    auto Val = Visit(Expr->getSubExpr());
    if (!Val.val) {
      Expr->dump();
      Expr->getSubExpr()->dump();
    }
    assert(Val.val);
    assert(ToStore.isReference);
    bool IsArray = false;
    Glob.getTypes().getMLIRType(Expr->getSubExpr()->getType(), &IsArray);
    Alu.store(Builder, Val, IsArray);
  }

  ArrayInit.pop_back();
  Builder.setInsertionPoint(OldBlock, OldPoint);

  return nullptr;
}

ValueCategory
MLIRScanner::VisitCXXFunctionalCastExpr(clang::CXXFunctionalCastExpr *Expr) {
  if (Expr->getType()->isVoidType()) {
    Visit(Expr->getSubExpr());
    return nullptr;
  }
  if (Expr->getCastKind() == clang::CastKind::CK_NoOp)
    return Visit(Expr->getSubExpr());
  if (Expr->getCastKind() == clang::CastKind::CK_ConstructorConversion)
    return Visit(Expr->getSubExpr());
  return VisitCastExpr(Expr);
}

ValueCategory
MLIRScanner::VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *Expr) {
  return Visit(Expr->getSubExpr());
}

ValueCategory MLIRScanner::VisitLambdaExpr(clang::LambdaExpr *Expr) {
  mlir::Type T =
      Glob.getTypes().getMLIRType(Expr->getCallOperator()->getThisType());

  bool LLVMABI = false, IsArray = false;
  Glob.getTypes().getMLIRType(Expr->getCallOperator()->getThisObjectType(),
                              &IsArray);

  if (auto PT = T.dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    LLVMABI = true;
    T = PT.getElementType();
  }

  if (auto MT = T.dyn_cast<MemRefType>()) {
    auto Shape = std::vector<int64_t>(MT.getShape());
    if (!IsArray)
      Shape[0] = 1;
    T = mlir::MemRefType::get(Shape, MT.getElementType(),
                              MemRefLayoutAttrInterface(), MT.getMemorySpace());
  }

  auto Op = createAllocOp(T, nullptr, /*memtype*/ 0, IsArray, LLVMABI);
  for (auto Tup : llvm::zip(Expr->getLambdaClass()->captures(),
                            Expr->getLambdaClass()->fields())) {
    auto C = std::get<0>(Tup);
    auto *Field = std::get<1>(Tup);
    if (C.capturesThis())
      continue;
    if (!C.capturesVariable())
      continue;

    auto CK = C.getCaptureKind();
    auto *Var = C.getCapturedVar();

    ValueCategory Result;
    if (Params.find(Var) != Params.end())
      Result = Params[Var];
    else {
      if (auto *VD = dyn_cast<VarDecl>(Var)) {
        if (Captures.find(VD) != Captures.end()) {
          FieldDecl *Field = Captures[VD];
          Result = CommonFieldLookup(
              cast<CXXMethodDecl>(EmittingFunctionDecl)->getThisObjectType(),
              Field, ThisVal.val, /*isLValue*/ false);
          assert(CaptureKinds.find(VD) != CaptureKinds.end());
          if (CaptureKinds[VD] == LambdaCaptureKind::LCK_ByRef)
            Result = Result.dereference(Builder);
          goto endp;
        }
      }
      EmittingFunctionDecl->dump();
      Expr->dump();
      Function.dump();
      llvm::errs() << "<pairs>\n";
      for (auto P : Params)
        P.first->dump();
      llvm::errs() << "</pairs>";
      Var->dump();
    }
  endp:

    bool IsArray = false;
    Glob.getTypes().getMLIRType(Field->getType(), &IsArray);

    if (CK == LambdaCaptureKind::LCK_ByCopy)
      CommonFieldLookup(Expr->getCallOperator()->getThisObjectType(), Field, Op,
                        /*isLValue*/ false)
          .store(Builder, Result, IsArray);
    else {
      assert(CK == LambdaCaptureKind::LCK_ByRef);
      assert(Result.isReference);

      auto Val = Result.val;

      if (auto MT = Val.getType().dyn_cast<MemRefType>()) {
        auto ET = MT.getElementType();
        if (ET.isInteger(1)) {
          ET = Builder.getIntegerType(8);
          const auto Zero = getConstantIndex(0);
          const auto Scalar =
              ValueCategory(Builder.create<memref::LoadOp>(Loc, Val, Zero),
                            /*IsReference*/ false)
                  .IntCast(Builder, Loc, ET, /*IsSigned*/ false);
          Val = Builder.create<memref::AllocaOp>(
              Loc, MemRefType::get(1, ET, MT.getLayout(), MT.getMemorySpace()));
          Builder.create<memref::StoreOp>(Loc, Scalar.val, Val, Zero);
        }
        auto Shape = std::vector<int64_t>(MT.getShape());
        Shape[0] = ShapedType::kDynamic;
        Val = Builder.create<memref::CastOp>(
            Loc,
            MemRefType::get(Shape, ET, MemRefLayoutAttrInterface(),
                            MT.getMemorySpace()),
            Val);
      }

      CommonFieldLookup(Expr->getCallOperator()->getThisObjectType(), Field, Op,
                        /*isLValue*/ false)
          .store(Builder, Val);
    }
  }
  return ValueCategory(Op, /*isReference*/ true);
}

// TODO actually deallocate
ValueCategory MLIRScanner::VisitMaterializeTemporaryExpr(
    clang::MaterializeTemporaryExpr *Expr) {
  ValueCategory V = Visit(Expr->getSubExpr());
  if (!V.val)
    Expr->dump();
  assert(V.val);

  bool IsArray = false, LLVMABI = false;
  if (Glob.getTypes()
          .getMLIRType(Glob.getCGM().getContext().getLValueReferenceType(
              Expr->getSubExpr()->getType()))
          .isa<mlir::LLVM::LLVMPointerType>())
    LLVMABI = true;
  else
    Glob.getTypes().getMLIRType(Expr->getSubExpr()->getType(), &IsArray);

  if (IsArray)
    return V;

  CGEIST_WARNING(llvm::WithColor::warning()
                 << "cleanup of materialized not handled\n");
  auto Op =
      createAllocOp(Glob.getTypes().getMLIRType(Expr->getSubExpr()->getType()),
                    nullptr, 0, /*isArray*/ IsArray, /*LLVMABI*/ LLVMABI);
  unsigned int AS = Glob.getCGM().getContext().getTargetAddressSpace(
      QualType(Expr->getSubExpr()->getType()).getAddressSpace());
  Op = castToMemSpace(Op, AS);
  ValueCategory(Op, /*isRefererence*/ true).store(Builder, V, IsArray);
  return ValueCategory(Op, /*isRefererence*/ true);
}

ValueCategory MLIRScanner::VisitCXXDeleteExpr(clang::CXXDeleteExpr *Expr) {
  CGEIST_WARNING(llvm::WithColor::warning()
                 << "not calling destructor on delete\n");

  Location Loc = getMLIRLocation(Expr->getExprLoc());
  mlir::Value ToDelete = Visit(Expr->getArgument()).getValue(Builder);

  if (ToDelete.getType().isa<mlir::MemRefType>())
    Builder.create<mlir::memref::DeallocOp>(Loc, ToDelete);
  else {
    mlir::Value Args[1] = {Builder.create<LLVM::BitcastOp>(
        Loc, LLVM::LLVMPointerType::get(Builder.getI8Type()), ToDelete)};
    Builder.create<mlir::LLVM::CallOp>(Loc, Glob.getOrCreateFreeFunction(),
                                       Args);
  }

  return nullptr;
}

ValueCategory MLIRScanner::VisitCXXNewExpr(clang::CXXNewExpr *Expr) {
  LLVM_DEBUG({
    llvm::dbgs() << "VisitCXXNewExpr: ";
    Expr->dump();
    llvm::dbgs() << "\n";
  });

  Location Loc = getMLIRLocation(Expr->getExprLoc());

  mlir::Value Count;
  if (Expr->isArray()) {
    Count = Visit(*Expr->raw_arg_begin()).getValue(Builder);
    Count = Builder.create<arith::IndexCastOp>(
        Loc, mlir::IndexType::get(Builder.getContext()), Count);
  } else
    Count = getConstantIndex(1);
  assert(Count);

  mlir::Type Ty = Glob.getTypes().getMLIRType(Expr->getType());

  mlir::Value Alloc, ArrayCons;
  if (!Expr->placement_arguments().empty()) {
    mlir::Value Val = Visit(*Expr->placement_arg_begin()).getValue(Builder);
    if (auto MT = Ty.dyn_cast<mlir::MemRefType>())
      ArrayCons = Alloc =
          Builder.create<polygeist::Pointer2MemrefOp>(Loc, MT, Val);
    else {
      ArrayCons = Alloc = Builder.create<mlir::LLVM::BitcastOp>(Loc, Ty, Val);
      if (Expr->isArray()) {
        auto PT = Ty.cast<LLVM::LLVMPointerType>();
        ArrayCons = Builder.create<mlir::LLVM::BitcastOp>(
            Loc,
            LLVM::LLVMPointerType::get(
                LLVM::LLVMArrayType::get(PT.getElementType(), 0),
                PT.getAddressSpace()),
            Alloc);
      }
    }
  } else if (auto MT = Ty.dyn_cast<mlir::MemRefType>()) {
    ArrayCons = Alloc =
        Builder.create<mlir::memref::AllocOp>(Loc, MT, ValueRange({Count}));
    if (Expr->hasInitializer() && isa<InitListExpr>(Expr->getInitializer()))
      (void)InitializeValueByInitListExpr(Alloc, Expr->getInitializer());
  } else {
    auto I64 = mlir::IntegerType::get(Count.getContext(), 64);
    Value TypeSize = getTypeSize(Expr->getAllocatedType());
    mlir::Value Args[1] = {Builder.create<arith::MulIOp>(Loc, TypeSize, Count)};
    Args[0] = Builder.create<arith::IndexCastOp>(Loc, I64, Args[0]);
    ArrayCons = Alloc = Builder.create<mlir::LLVM::BitcastOp>(
        Loc, Ty,
        Builder
            .create<mlir::LLVM::CallOp>(Loc, Glob.getOrCreateMallocFunction(),
                                        Args)
            ->getResult(0));

    if (Expr->hasInitializer() && isa<InitListExpr>(Expr->getInitializer()))
      (void)InitializeValueByInitListExpr(Alloc, Expr->getInitializer());

    if (Expr->isArray()) {
      auto PT = Ty.cast<LLVM::LLVMPointerType>();
      ArrayCons = Builder.create<mlir::LLVM::BitcastOp>(
          Loc,
          LLVM::LLVMPointerType::get(
              LLVM::LLVMArrayType::get(PT.getElementType(), 0),
              PT.getAddressSpace()),
          Alloc);
    }
  }
  assert(Alloc);

  if (Expr->getConstructExpr())
    VisitConstructCommon(
        const_cast<CXXConstructExpr *>(Expr->getConstructExpr()),
        /*name*/ nullptr, /*memtype*/ 0, ArrayCons, Count);

  return ValueCategory(Alloc, /*isRefererence*/ false);
}

ValueCategory
MLIRScanner::VisitCXXScalarValueInitExpr(clang::CXXScalarValueInitExpr *Expr) {
  Location Loc = getMLIRLocation(Expr->getExprLoc());

  bool IsArray = false;
  mlir::Type MElem = Glob.getTypes().getMLIRType(Expr->getType(), &IsArray);
  assert(!IsArray);

  if (MElem.isa<mlir::IntegerType>())
    return ValueCategory(Builder.create<arith::ConstantIntOp>(Loc, 0, MElem),
                         false);
  if (auto MT = MElem.dyn_cast<mlir::MemRefType>())
    return ValueCategory(
        Builder.create<polygeist::Pointer2MemrefOp>(
            Loc, MT,
            Builder.create<mlir::LLVM::NullOp>(
                Loc, LLVM::LLVMPointerType::get(Builder.getI8Type(),
                                                MT.getMemorySpaceAsInt()))),
        false);
  if (auto PT = MElem.dyn_cast<mlir::LLVM::LLVMPointerType>())
    return ValueCategory(Builder.create<mlir::LLVM::NullOp>(Loc, PT), false);
  if (!MElem.isa<FloatType>())
    Expr->dump();
  auto Ft = MElem.cast<FloatType>();
  return ValueCategory(Builder.create<arith::ConstantFloatOp>(
                           Loc, APFloat(Ft.getFloatSemantics(), "0"), Ft),
                       false);
}

ValueCategory MLIRScanner::VisitCXXPseudoDestructorExpr(
    clang::CXXPseudoDestructorExpr *Expr) {
  Visit(Expr->getBase());
  CGEIST_WARNING(llvm::WithColor::warning()
                 << "not running pseudo destructor\n");
  return nullptr;
}

ValueCategory
MLIRScanner::VisitCXXConstructExpr(clang::CXXConstructExpr *Cons) {
  return VisitConstructCommon(Cons, /*name*/ nullptr, /*space*/ 0);
}

ValueCategory MLIRScanner::VisitConstructCommon(clang::CXXConstructExpr *Cons,
                                                VarDecl *Name, unsigned Memtype,
                                                mlir::Value Op,
                                                mlir::Value Count) {
  Location Loc = getMLIRLocation(Cons->getExprLoc());

  bool IsArray = false;
  mlir::Type SubType = Glob.getTypes().getMLIRType(Cons->getType(), &IsArray);

  bool LLVMABI = false;
  mlir::Type Ptrty = Glob.getTypes().getMLIRType(
      Glob.getCGM().getContext().getLValueReferenceType(Cons->getType()));
  if (Ptrty.isa<mlir::LLVM::LLVMPointerType>())
    LLVMABI = true;
  else if (IsArray) {
    SubType = Ptrty;
    IsArray = true;
  }
  if (Op == nullptr)
    Op = createAllocOp(SubType, Name, Memtype, IsArray, LLVMABI);

  if (Cons->requiresZeroInitialization()) {
    mlir::Value Val = Op;
    if (Val.getType().isa<MemRefType>())
      Val = Builder.create<polygeist::Memref2PointerOp>(
          Loc,
          LLVM::LLVMPointerType::get(
              Builder.getI8Type(),
              Val.getType().cast<MemRefType>().getMemorySpaceAsInt()),
          Val);
    else
      Val = Builder.create<LLVM::BitcastOp>(
          Loc,
          LLVM::LLVMPointerType::get(
              Builder.getI8Type(),
              Val.getType().cast<LLVM::LLVMPointerType>().getAddressSpace()),
          Val);

    mlir::Value Size = getTypeSize(Cons->getType());
    auto I80 = Builder.create<arith::ConstantIntOp>(Loc, 0, 8);
    auto Sizev =
        Builder.create<arith::IndexCastOp>(Loc, Builder.getI64Type(), Size);
    auto Falsev = Builder.create<arith::ConstantIntOp>(Loc, false, 1);
    Builder.create<LLVM::MemsetOp>(Loc, Val, I80, Sizev, Falsev);
  }

  CXXConstructorDecl *CtorDecl = Cons->getConstructor();
  if (CtorDecl->isTrivial() && CtorDecl->isDefaultConstructor())
    return ValueCategory(Op, /*isReference*/ true);

  mlir::Block::iterator OldPoint;
  mlir::Block *OldBlock;
  ValueCategory EndObj(Op, /*isReference*/ true);

  ValueCategory Obj(Op, /*isReference*/ true);
  QualType InnerType = Cons->getType();
  if (const auto *ArrayType =
          Glob.getCGM().getContext().getAsArrayType(Cons->getType())) {
    InnerType = ArrayType->getElementType();
    mlir::Value Size;
    if (Count)
      Size = Count;
    else {
      const auto *CAT = cast<clang::ConstantArrayType>(ArrayType);
      Size = getConstantIndex(CAT->getSize().getLimitedValue());
    }
    auto ForOp = Builder.create<scf::ForOp>(Loc, getConstantIndex(0), Size,
                                            getConstantIndex(1));
    OldPoint = Builder.getInsertionPoint();
    OldBlock = Builder.getInsertionBlock();

    Builder.setInsertionPointToStart(&ForOp.getLoopBody().front());
    assert(Obj.isReference);
    Obj = CommonArrayToPointer(Obj);
    Obj = CommonArrayLookup(Obj, ForOp.getInductionVar(),
                            /*isImplicitRef*/ false, /*removeIndex*/ false);
    assert(Obj.isReference);
  }

  /// If the constructor is part of the SYCL namespace, we may not want the
  /// GetOrCreateMLIRFunction to add this FuncOp to the functionsToEmit deque,
  /// since we will create it's equivalent with SYCL operations. Please note
  /// that we still generate some constructors that we need for lowering some
  /// sycl op.  Therefore, in those case, we set ShouldEmit back to "true" by
  /// looking them up in our "registry" of supported constructors.
  const auto IsSyclCtor =
      mlirclang::getNamespaceKind(CtorDecl->getEnclosingNamespaceContext()) !=
      mlirclang::NamespaceKind::Other;
  bool ShouldEmit = !IsSyclCtor;

  std::string MangledName = MLIRScanner::getMangledFuncName(
      cast<FunctionDecl>(*CtorDecl), Glob.getCGM());
  MangledName = (PrefixABI + MangledName);
  if (GenerateAllSYCLFuncs || !isUnsupportedFunction(MangledName))
    ShouldEmit = true;

  FunctionToEmit F(*CtorDecl, mlirclang::getInputContext(Builder));
  auto ToCall = cast<func::FuncOp>(Glob.getOrCreateMLIRFunction(F, ShouldEmit));

  SmallVector<std::pair<ValueCategory, clang::Expr *>> Args{{Obj, nullptr}};
  Args.reserve(Cons->getNumArgs() + 1);
  for (auto *A : Cons->arguments())
    Args.emplace_back(Visit(A), A);

  callHelper(ToCall, InnerType, Args,
             /*retType*/ Glob.getCGM().getContext().VoidTy, false, Cons,
             *CtorDecl);

  if (Glob.getCGM().getContext().getAsArrayType(Cons->getType()))
    Builder.setInsertionPoint(OldBlock, OldPoint);

  return EndObj;
}

ValueCategory
MLIRScanner::EmitVectorSubscript(clang::ArraySubscriptExpr *Expr) {
  ValueCategory Base{Visit(Expr->getBase()).getValue(Builder), false};
  auto Idx = Visit(Expr->getIdx());

  CGEIST_WARNING(llvm::WithColor::warning() << "Not emitting bounds check\n");

  return Base.ExtractElement(Builder, getMLIRLocation(Expr->getExprLoc()),
                             Idx.val);
}

ValueCategory
MLIRScanner::EmitArraySubscriptExpr(clang::ArraySubscriptExpr *E) {
  if (!E->getBase()->getType()->isVectorType())
    return Visit(E);

  auto LHS = EmitLValue(E->getBase());
  auto Idx = Visit(E->getIdx());
  return {LHS.val, Idx.val};
}

ValueCategory
MLIRScanner::VisitArraySubscriptExpr(clang::ArraySubscriptExpr *Expr) {
  LLVM_DEBUG({
    llvm::dbgs() << "VisitArraySubscriptExpr: ";
    Expr->dump();
    llvm::dbgs() << "\n";
  });

  assert(!Expr->getBase()->getType()->isVLSTBuiltinType() &&
         "Not supported yet");

  if (Expr->getBase()->getType()->isVectorType())
    return EmitVectorSubscript(Expr);

  auto Moo = Visit(Expr->getLHS());
  auto RHS = Visit(Expr->getRHS()).getValue(Builder);
  assert(RHS);

  auto Idx = castToIndex(getMLIRLocation(Expr->getRBracketLoc()), RHS);
  if (isa<clang::VectorType>(
          Expr->getLHS()->getType()->getUnqualifiedDesugaredType())) {
    assert(Moo.isReference);
    Moo.isReference = false;
    auto MT = Moo.val.getType().cast<MemRefType>();

    auto Shape = std::vector<int64_t>(MT.getShape());
    Shape.erase(Shape.begin());
    auto MT0 =
        mlir::MemRefType::get(Shape, MT.getElementType(),
                              MemRefLayoutAttrInterface(), MT.getMemorySpace());
    Moo.val = Builder.create<polygeist::SubIndexOp>(Loc, MT0, Moo.val,
                                                    getConstantIndex(0));
  }

  bool IsArray = false;
  if (!Glob.getCGM().getContext().getAsArrayType(Expr->getType()))
    Glob.getTypes().getMLIRType(Expr->getType(), &IsArray);

  return CommonArrayLookup(Moo, Idx, IsArray);
}

const clang::FunctionDecl *MLIRScanner::EmitCallee(const Expr *E) {
  E = E->IgnoreParens();
  // Look through function-to-pointer decay.
  if (const auto *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    if (ICE->getCastKind() == CK_FunctionToPointerDecay ||
        ICE->getCastKind() == CK_BuiltinFnToFnPtr)
      return EmitCallee(ICE->getSubExpr());
    // Resolve direct calls.
  } else if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (const auto *FD = dyn_cast<FunctionDecl>(DRE->getDecl()))
      return FD;
  } else if (const auto *ME = dyn_cast<MemberExpr>(E)) {
    if (auto *FD = dyn_cast<FunctionDecl>(ME->getMemberDecl())) {
      // TODO EmitIgnoredExpr(ME->getBase());
      return FD;
    }
    // Look through template substitutions.
  } else if (const auto *NTTP = dyn_cast<SubstNonTypeTemplateParmExpr>(E)) {
    return EmitCallee(NTTP->getReplacement());
  } else if (const auto *UOp = dyn_cast<clang::UnaryOperator>(E)) {
    if (UOp->getOpcode() == UnaryOperatorKind::UO_AddrOf)
      return EmitCallee(UOp->getSubExpr());
  }

  return nullptr;
}

static NamedAttrList getSYCLMethodOpAttrs(OpBuilder &Builder,
                                          TypeRange ArgumentTypes,
                                          llvm::StringRef TypeName,
                                          llvm::StringRef FunctionName,
                                          llvm::StringRef MangledFunctionName) {
  NamedAttrList Attrs;
  Attrs.set(mlir::sycl::SYCLDialect::getArgumentTypesAttrName(),
            Builder.getTypeArrayAttr(ArgumentTypes));
  Attrs.set(mlir::sycl::SYCLDialect::getFunctionNameAttrName(),
            FlatSymbolRefAttr::get(Builder.getStringAttr(FunctionName)));
  if (!OmitOptionalMangledFunctionName)
    Attrs.set(
        mlir::sycl::SYCLDialect::getMangledFunctionNameAttrName(),
        FlatSymbolRefAttr::get(Builder.getStringAttr(MangledFunctionName)));

  Attrs.set(mlir::sycl::SYCLDialect::getTypeNameAttrName(),
            FlatSymbolRefAttr::get(Builder.getStringAttr(TypeName)));
  return Attrs;
}

llvm::Optional<sycl::SYCLMethodOpInterface> MLIRScanner::createSYCLMethodOp(
    llvm::StringRef TypeName, llvm::StringRef FunctionName,
    mlir::ValueRange Operands, llvm::Optional<mlir::Type> ReturnType,
    llvm::StringRef MangledFunctionName) {
  // Expecting a MemRef as the first argument, as the first operand to a method
  // call should be a pointer to `this`.
  if (Operands.empty() || !Operands[0].getType().isa<MemRefType>())
    return std::nullopt;

  auto *SYCLDialect =
      Operands[0].getContext()->getLoadedDialect<mlir::sycl::SYCLDialect>();
  assert(SYCLDialect && "MLIR-SYCL dialect not loaded.");

  // Need to copy to avoid overriding elements in the input argument.
  SmallVector<mlir::Value> OperandsCpy(Operands);

  // Cast operations are abstracted to avoid missing method calls due to
  // implementation details.
  OperandsCpy[0] = sycl::abstractCasts(OperandsCpy[0]);

  auto BaseType = OperandsCpy[0].getType().cast<MemRefType>();
  const llvm::Optional<llvm::StringRef> OptOpName = SYCLDialect->findMethod(
      BaseType.getElementType().getTypeID(), FunctionName);

  if (!OptOpName) {
    LLVM_DEBUG(llvm::dbgs() << "SYCL method not inserted. Type: " << BaseType
                            << " Name: " << FunctionName << "\n");
    return std::nullopt;
  }

  LLVM_DEBUG(llvm::dbgs() << "Inserting operation " << OptOpName
                          << " to replace SYCL method call.\n");

  return static_cast<sycl::SYCLMethodOpInterface>(Builder.create(
      Loc, Builder.getStringAttr(*OptOpName),
      sycl::adaptSYCLMethodOpArguments(Builder, Loc, OperandsCpy),
      ReturnType ? mlir::TypeRange{*ReturnType} : mlir::TypeRange{},
      getSYCLMethodOpAttrs(Builder, Operands.getTypes(), TypeName, FunctionName,
                           MangledFunctionName)));
}

mlir::Operation *
MLIRScanner::emitSYCLOps(const clang::Expr *Expr,
                         const llvm::SmallVectorImpl<mlir::Value> &Args) {
  const FunctionDecl *Func = nullptr;
  if (const auto *ConsExpr = dyn_cast<clang::CXXConstructExpr>(Expr)) {
    Func = ConsExpr->getConstructor()->getAsFunction();

    if (mlirclang::getNamespaceKind(Func->getEnclosingNamespaceContext()) !=
        mlirclang::NamespaceKind::Other) {
      const auto *RD = dyn_cast<clang::CXXRecordDecl>(Func->getParent());
      if (RD &&
          mlirclang::areSYCLMemberFunctionOrConstructorArgs(
              ValueRange{Args}.getTypes()) &&
          !RD->getName().empty()) {
        std::string Name =
            MLIRScanner::getMangledFuncName(*Func, Glob.getCGM());
        return Builder.create<mlir::sycl::SYCLConstructorOp>(Loc, RD->getName(),
                                                             Name, Args);
      }
    }
  }

  mlir::Operation *Op = nullptr;
  if (const auto *CallExpr = dyn_cast<clang::CallExpr>(Expr))
    Func = CallExpr->getCalleeDecl()->getAsFunction();

  if (Func)
    if (mlirclang::getNamespaceKind(Func->getEnclosingNamespaceContext()) !=
        mlirclang::NamespaceKind::Other) {
      auto OptFuncType = llvm::Optional<llvm::StringRef>{std::nullopt};
      if (const auto *RD = dyn_cast<clang::CXXRecordDecl>(Func->getParent()))
        if (!RD->getName().empty())
          OptFuncType = RD->getName();

      auto OptRetType = llvm::Optional<mlir::Type>{std::nullopt};
      const mlir::Type RetType =
          Glob.getTypes().getMLIRType(Func->getReturnType());
      if (!RetType.isa<mlir::NoneType>())
        OptRetType = RetType;

      // Attempt to create a SYCL method call first, if that fails create a
      // generic SYCLCallOp.
      std::string Name = MLIRScanner::getMangledFuncName(*Func, Glob.getCGM());
      if (OptFuncType)
        Op = createSYCLMethodOp(*OptFuncType, Func->getNameAsString(), Args,
                                OptRetType, Name)
                 .value_or(nullptr);
      if (!Op)
        Op = Builder.create<mlir::sycl::SYCLCallOp>(
            Loc, OptRetType, OptFuncType, Func->getNameAsString(), Name, Args);
    }

  return Op;
}

ValueCategory MLIRScanner::VisitMSPropertyRefExpr(MSPropertyRefExpr *Expr) {
  llvm_unreachable("unhandled ms propertyref");
  return nullptr;
}

ValueCategory
MLIRScanner::VisitPseudoObjectExpr(clang::PseudoObjectExpr *Expr) {
  return Visit(Expr->getResultExpr());
}

ValueCategory MLIRScanner::VisitSubstNonTypeTemplateParmExpr(
    SubstNonTypeTemplateParmExpr *Expr) {
  return Visit(Expr->getReplacement());
}

ValueCategory
MLIRScanner::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Uop) {
  switch (Uop->getKind()) {
  case UETT_SizeOf: {
    Value TypeSize = getTypeSize(Uop->getTypeOfArgument());
    auto RetTy =
        Glob.getTypes().getMLIRType(Uop->getType()).cast<mlir::IntegerType>();
    return ValueCategory(
        Builder.create<arith::IndexCastOp>(Loc, RetTy, TypeSize),
        /*isReference*/ false);
  }
  case UETT_AlignOf: {
    Value TypeAlign = getTypeAlign(Uop->getTypeOfArgument());
    auto RetTy =
        Glob.getTypes().getMLIRType(Uop->getType()).cast<mlir::IntegerType>();
    return ValueCategory(
        Builder.create<arith::IndexCastOp>(Loc, RetTy, TypeAlign),
        /*isReference*/ false);
  }
  default:
    Uop->dump();
    llvm_unreachable("unhandled VisitUnaryExprOrTypeTraitExpr");
  }
}

ValueCategory MLIRScanner::VisitAtomicExpr(clang::AtomicExpr *BO) {
  Location Loc = getMLIRLocation(BO->getExprLoc());

  switch (BO->getOp()) {
  case AtomicExpr::AtomicOp::AO__atomic_add_fetch: {
    Value A0 = Visit(BO->getPtr()).getValue(Builder);
    Value A1 = Visit(BO->getVal1()).getValue(Builder);
    mlir::Type Ty = A1.getType();
    arith::AtomicRMWKind Op;
    LLVM::AtomicBinOp Lop;
    if (Ty.isa<mlir::IntegerType>()) {
      Op = arith::AtomicRMWKind::addi;
      Lop = LLVM::AtomicBinOp::add;
    } else {
      Op = arith::AtomicRMWKind::addf;
      Lop = LLVM::AtomicBinOp::fadd;
    }
    // TODO add atomic ordering
    mlir::Value V;
    if (A0.getType().isa<MemRefType>())
      V = Builder.create<memref::AtomicRMWOp>(
          Loc, A1.getType(), Op, A1, A0,
          std::vector<mlir::Value>({getConstantIndex(0)}));
    else
      V = Builder.create<LLVM::AtomicRMWOp>(Loc, A1.getType(), Lop, A0, A1,
                                            LLVM::AtomicOrdering::acq_rel);

    if (Ty.isa<mlir::IntegerType>())
      V = Builder.create<arith::AddIOp>(Loc, V, A1);
    else
      V = Builder.create<arith::AddFOp>(Loc, V, A1);

    return ValueCategory(V, false);
  }
  default:
    llvm::errs() << "unhandled atomic:";
    BO->dump();
    assert(0);
  }
}

ValueCategory MLIRScanner::VisitExprWithCleanups(ExprWithCleanups *E) {
  auto Ret = Visit(E->getSubExpr());
  CGEIST_WARNING({
    for (auto &Child : E->children()) {
      llvm::WithColor::warning() << "cleanup not handled for: ";
      Child->dump(llvm::WithColor::warning(), Glob.getCGM().getContext());
    }
  });
  return Ret;
}

ValueCategory MLIRScanner::VisitDeclRefExpr(DeclRefExpr *E) {
  LLVM_DEBUG({
    llvm::dbgs() << "VisitDeclRefExpr: ";
    E->dump();
    llvm::dbgs() << "\n";
  });

  if (auto *Tocall = dyn_cast<FunctionDecl>(E->getDecl()))
    return ValueCategory(Builder.create<LLVM::AddressOfOp>(
                             Loc, Glob.getOrCreateLLVMFunction(Tocall)),
                         /*isReference*/ true);

  if (auto *VD = dyn_cast<VarDecl>(E->getDecl())) {
    if (Captures.find(VD) != Captures.end()) {
      FieldDecl *Field = Captures[VD];
      ValueCategory Res = CommonFieldLookup(
          cast<CXXMethodDecl>(EmittingFunctionDecl)->getThisObjectType(), Field,
          ThisVal.val,
          isa<clang::ReferenceType>(
              Field->getType()->getUnqualifiedDesugaredType()));
      assert(CaptureKinds.find(VD) != CaptureKinds.end());
      return Res;
    }

    if (Params.find(VD) != Params.end()) {
      ValueCategory Res = Params[VD];
      assert(Res.val);
      return Res;
    }
  }
  if (auto *ED = dyn_cast<EnumConstantDecl>(E->getDecl())) {
    auto Ty =
        Glob.getTypes().getMLIRType(E->getType()).cast<mlir::IntegerType>();
    return ValueCategory(Builder.create<arith::ConstantIntOp>(
                             Loc, ED->getInitVal().getExtValue(), Ty),
                         /*isReference*/ false);

    if (!ED->getInitExpr())
      ED->dump();
    return Visit(ED->getInitExpr());
  }
  if (auto *VD = dyn_cast<ValueDecl>(E->getDecl())) {
    const std::string Name = E->getDecl()->getName().str();
    if (Glob.getTypes()
            .getMLIRType(
                Glob.getCGM().getContext().getPointerType(E->getType()))
            .isa<mlir::LLVM::LLVMPointerType>() ||
        Name == "stderr" || Name == "stdout" || Name == "stdin" ||
        (E->hasQualifier())) {
      return ValueCategory(Builder.create<mlir::LLVM::AddressOfOp>(
                               Loc, Glob.getOrCreateLLVMGlobal(VD)),
                           /*isReference*/ true);
    }

    // We need to decide where to put the Global.  If we are in a device
    // module, the global should be in the gpu module (which is nested inside
    // another main module).
    std::pair<mlir::memref::GlobalOp, bool> Gv = Glob.getOrCreateGlobal(
        *VD, /*prefix=*/"",
        isa<mlir::gpu::GPUModuleOp>(Function->getParentOp())
            ? FunctionContext::SYCLDevice
            : FunctionContext::Host);

    auto Gv2 = Builder.create<memref::GetGlobalOp>(Loc, Gv.first.getType(),
                                                   Gv.first.getName());
    Value V = castToMemSpace(reshapeRanklessGlobal(Gv2),
                             Glob.getCGM().getContext().getTargetAddressSpace(
                                 VD->getType().getAddressSpace()));

    // TODO check reference
    return ValueCategory(V, /*isReference*/ true);
  }

  llvm_unreachable("couldn't find value");
  return nullptr;
}

ValueCategory MLIRScanner::VisitOpaqueValueExpr(OpaqueValueExpr *E) {
  if (!E->getSourceExpr()) {
    E->dump();
    assert(E->getSourceExpr());
  }
  auto Res = Visit(E->getSourceExpr());
  if (!Res.val) {
    E->dump();
    E->getSourceExpr()->dump();
    assert(Res.val);
  }
  return Res;
}

ValueCategory MLIRScanner::VisitCXXTypeidExpr(clang::CXXTypeidExpr *E) {
  QualType T;
  if (E->isTypeOperand())
    T = E->getTypeOperand(Glob.getCGM().getContext());
  else
    T = E->getExprOperand()->getType();
  llvm::Constant *C = Glob.getCGM().GetAddrOfRTTIDescriptor(T);
  llvm::errs() << *C << "\n";
  mlir::Type Ty = Glob.getTypes().getMLIRType(E->getType());
  llvm::errs() << Ty << "\n";
  llvm_unreachable("unhandled typeid");
}

ValueCategory
MLIRScanner::VisitCXXDefaultInitExpr(clang::CXXDefaultInitExpr *Expr) {
  assert(ThisVal.val);
  ValueCategory ToSet = Visit(Expr->getExpr());
  assert(!ThisVal.isReference);
  assert(ToSet.val);

  bool IsArray = false;
  Glob.getTypes().getMLIRType(Expr->getExpr()->getType(), &IsArray);

  ValueCategory CFL = CommonFieldLookup(
      cast<CXXMethodDecl>(EmittingFunctionDecl)->getThisObjectType(),
      Expr->getField(), ThisVal.val, /*isLValue*/ false);
  assert(CFL.val);
  CFL.store(Builder, ToSet, IsArray);
  return CFL;
}

ValueCategory MLIRScanner::VisitCXXNoexceptExpr(CXXNoexceptExpr *Expr) {
  auto Ty =
      Glob.getTypes().getMLIRType(Expr->getType()).cast<mlir::IntegerType>();
  return ValueCategory(
      Builder.create<arith::ConstantIntOp>(getMLIRLocation(Expr->getExprLoc()),
                                           Expr->getValue(), Ty),
      /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitMemberExpr(MemberExpr *ME) {
  LLVM_DEBUG({
    if (auto *Sr2 = dyn_cast<OpaqueValueExpr>(ME->getBase())) {
      if (auto *SR = dyn_cast<DeclRefExpr>(Sr2->getSourceExpr())) {
        if (SR->getDecl()->getName() == "blockIdx")
          llvm::dbgs() << "known block index";
        else if (SR->getDecl()->getName() == "blockDim")
          llvm::dbgs() << "known block dim";
        else if (SR->getDecl()->getName() == "threadIdx")
          llvm::dbgs() << "known thread index";
        else if (SR->getDecl()->getName() == "gridDim")
          llvm::dbgs() << "known grid index";
      }
    }
  });

  ValueCategory Base = Visit(ME->getBase());
  clang::QualType OT = ME->getBase()->getType();
  if (ME->isArrow()) {
    if (!Base.val)
      ME->dump();

    Base = Base.dereference(Builder);
    OT = cast<clang::PointerType>(OT->getUnqualifiedDesugaredType())
             ->getPointeeType();
  }
  if (!Base.isReference) {
    EmittingFunctionDecl->dump();
    Function.dump();
    ME->dump();
    llvm::errs() << "base value: " << Base.val << "\n";
  }
  assert(Base.isReference);
  const FieldDecl *Field = cast<clang::FieldDecl>(ME->getMemberDecl());
  return CommonFieldLookup(
      OT, Field, Base.val,
      isa<clang::ReferenceType>(
          Field->getType()->getUnqualifiedDesugaredType()));
}

ValueCategory MLIRScanner::VisitCastExpr(CastExpr *E) {
  LLVM_DEBUG({
    llvm::dbgs() << "VisitCastExpr: ";
    E->dump();
    llvm::dbgs() << "\n";
  });

  Location Loc = getMLIRLocation(E->getExprLoc());

  switch (E->getCastKind()) {
  case clang::CastKind::CK_NullToPointer: {
    mlir::Type LLVMTy = Glob.getTypes().getMLIRType(E->getType());
    if (LLVMTy.isa<LLVM::LLVMPointerType>())
      return ValueCategory(Builder.create<mlir::LLVM::NullOp>(Loc, LLVMTy),
                           /*isReference*/ false);
    if (auto MT = LLVMTy.dyn_cast<MemRefType>())
      return ValueCategory(
          Builder.create<polygeist::Pointer2MemrefOp>(
              Loc, MT,
              Builder.create<mlir::LLVM::NullOp>(
                  Loc, LLVM::LLVMPointerType::get(Builder.getI8Type(),
                                                  MT.getMemorySpaceAsInt()))),
          false);
    llvm_unreachable("illegal type for cast");
  }
  case clang::CastKind::CK_UserDefinedConversion:
    return Visit(E->getSubExpr());

  case clang::CastKind::CK_AddressSpaceConversion: {
    ValueCategory Scalar = Visit(E->getSubExpr());
    QualType DestTy = E->getType();
    unsigned AS = Glob.getCGM().getContext().getTargetAddressSpace(
        DestTy->isPointerType() ? DestTy->getPointeeType().getAddressSpace()
                                : DestTy.getAddressSpace());
    return ValueCategory(castToMemSpace(Scalar.val, AS), Scalar.isReference);
  }
  case clang::CastKind::CK_Dynamic: {
    E->dump();
    llvm_unreachable("dynamic cast not handled yet\n");
  } break;
  case clang::CastKind::CK_UncheckedDerivedToBase:
  case clang::CastKind::CK_DerivedToBase: {
    auto SE = Visit(E->getSubExpr());
    if (!SE.val)
      E->dump();

    assert(SE.val);
    const auto *Derived =
        (E->isLValue() || E->isXValue())
            ? cast<CXXRecordDecl>(
                  E->getSubExpr()->getType()->castAs<RecordType>()->getDecl())
            : E->getSubExpr()->getType()->getPointeeCXXRecordDecl();
    SmallVector<const clang::Type *> BaseTypes;
    SmallVector<bool> BaseVirtual;
    for (auto *B : E->path()) {
      BaseTypes.push_back(B->getType().getTypePtr());
      BaseVirtual.push_back(B->isVirtual());
    }

    if (auto UT = SE.val.getType().dyn_cast<mlir::MemRefType>()) {
      auto MT = Glob.getTypes()
                    .getMLIRType(
                        (E->isLValue() || E->isXValue())
                            ? Glob.getCGM().getContext().getLValueReferenceType(
                                  E->getType())
                            : E->getType())
                    .dyn_cast<mlir::MemRefType>();

      if (UT.getShape().size() != MT.getShape().size()) {
        E->dump();
        llvm::errs() << " se.val: " << SE.val << " ut: " << UT << " mt: " << MT
                     << "\n";
      }
      assert(UT.getShape().size() == MT.getShape().size());
      auto Ty = mlir::MemRefType::get(MT.getShape(), MT.getElementType(),
                                      MemRefLayoutAttrInterface(),
                                      UT.getMemorySpace());
      if (Ty.getElementType().getDialect().getNamespace() ==
              mlir::sycl::SYCLDialect::getDialectNamespace() &&
          UT.getElementType().getDialect().getNamespace() ==
              mlir::sycl::SYCLDialect::getDialectNamespace() &&
          Ty.getElementType() != UT.getElementType()) {
        return ValueCategory(
            Builder.create<mlir::sycl::SYCLCastOp>(Loc, Ty, SE.val),
            /*isReference*/ SE.isReference);
      }
    }

    mlir::Value Val =
        GetAddressOfBaseClass(SE.val, Derived, BaseTypes, BaseVirtual);
    if (E->getCastKind() != clang::CastKind::CK_UncheckedDerivedToBase &&
        !isa<CXXThisExpr>(E->IgnoreParens())) {
      mlir::Value Ptr = Val;
      if (auto MT = Ptr.getType().dyn_cast<MemRefType>())
        Ptr = Builder.create<polygeist::Memref2PointerOp>(
            Loc,
            LLVM::LLVMPointerType::get(MT.getElementType(),
                                       MT.getMemorySpaceAsInt()),
            Ptr);
      mlir::Value NullptrLlvm =
          Builder.create<mlir::LLVM::NullOp>(Loc, Ptr.getType());
      auto NE = Builder.create<mlir::LLVM::ICmpOp>(
          Loc, mlir::LLVM::ICmpPredicate::ne, Ptr, NullptrLlvm);
      if (auto MT = Ptr.getType().dyn_cast<MemRefType>())
        NullptrLlvm =
            Builder.create<polygeist::Pointer2MemrefOp>(Loc, MT, NullptrLlvm);
      Val = Builder.create<arith::SelectOp>(Loc, NE, Val, NullptrLlvm);
    }

    return ValueCategory(Val, SE.isReference);
  }
  case clang::CastKind::CK_BaseToDerived: {
    auto SE = Visit(E->getSubExpr());
    if (!SE.val)
      E->dump();

    assert(SE.val);
    const auto *Derived =
        (E->isLValue() || E->isXValue())
            ? cast<CXXRecordDecl>(E->getType()->castAs<RecordType>()->getDecl())
            : E->getType()->getPointeeCXXRecordDecl();
    mlir::Value Val = GetAddressOfDerivedClass(SE.val, Derived, E->path_begin(),
                                               E->path_end());
    return ValueCategory(Val, SE.isReference);
  }
  case clang::CastKind::CK_BitCast: {
    if (auto *CI = dyn_cast<clang::CallExpr>(E->getSubExpr()))
      if (auto *IC = dyn_cast<ImplicitCastExpr>(CI->getCallee()))
        if (auto *SR = dyn_cast<DeclRefExpr>(IC->getSubExpr())) {
          if (SR->getDecl()->getIdentifier() &&
              SR->getDecl()->getName() == "polybench_alloc_data") {
            if (auto MT = Glob.getTypes()
                              .getMLIRType(E->getType())
                              .dyn_cast<mlir::MemRefType>()) {
              auto Shape = std::vector<int64_t>(MT.getShape());
              // shape.erase(shape.begin());
              auto MT0 = mlir::MemRefType::get(Shape, MT.getElementType(),
                                               MemRefLayoutAttrInterface(),
                                               MT.getMemorySpace());

              auto Alloc = Builder.create<mlir::memref::AllocOp>(Loc, MT0);
              return ValueCategory(Alloc, /*isReference*/ false);
            }
          }
        }

    if (auto *CI = dyn_cast<clang::CallExpr>(E->getSubExpr()))
      if (auto *IC = dyn_cast<ImplicitCastExpr>(CI->getCallee()))
        if (auto *SR = dyn_cast<DeclRefExpr>(IC->getSubExpr())) {
          if (SR->getDecl()->getIdentifier() &&
              (SR->getDecl()->getName() == "malloc" ||
               SR->getDecl()->getName() == "calloc"))
            if (auto MT = Glob.getTypes()
                              .getMLIRType(E->getType())
                              .dyn_cast<mlir::MemRefType>()) {
              auto Shape = std::vector<int64_t>(MT.getShape());

              auto ElemSize =
                  getTypeSize(cast<clang::PointerType>(
                                  E->getType()->getUnqualifiedDesugaredType())
                                  ->getPointeeType());
              mlir::Value AllocSize = Builder.create<arith::IndexCastOp>(
                  Loc, mlir::IndexType::get(Builder.getContext()),
                  Visit(CI->getArg(0)).getValue(Builder));
              if (SR->getDecl()->getName() == "calloc") {
                AllocSize = Builder.create<arith::MulIOp>(
                    Loc, AllocSize,
                    Builder.create<arith::IndexCastOp>(
                        Loc, mlir::IndexType::get(Builder.getContext()),
                        Visit(CI->getArg(1)).getValue(Builder)));
              }
              mlir::Value Args[1] = {
                  Builder.create<arith::DivUIOp>(Loc, AllocSize, ElemSize)};
              auto Alloc = Builder.create<mlir::memref::AllocOp>(Loc, MT, Args);
              if (SR->getDecl()->getName() == "calloc") {
                mlir::Value Val = Alloc;
                if (Val.getType().isa<MemRefType>()) {
                  Val = Builder.create<polygeist::Memref2PointerOp>(
                      Loc,
                      LLVM::LLVMPointerType::get(Builder.getI8Type(),
                                                 Val.getType()
                                                     .cast<MemRefType>()
                                                     .getMemorySpaceAsInt()),
                      Val);
                } else {
                  Val = Builder.create<LLVM::BitcastOp>(
                      Loc,
                      LLVM::LLVMPointerType::get(
                          Builder.getI8Type(),
                          Val.getType()
                              .cast<LLVM::LLVMPointerType>()
                              .getAddressSpace()),
                      Val);
                }
                auto I80 = Builder.create<arith::ConstantIntOp>(Loc, 0, 8);
                auto Sizev = Builder.create<arith::IndexCastOp>(
                    Loc, Builder.getI64Type(), AllocSize);
                auto Falsev =
                    Builder.create<arith::ConstantIntOp>(Loc, false, 1);
                Builder.create<LLVM::MemsetOp>(Loc, Val, I80, Sizev, Falsev);
              }
              return ValueCategory(Alloc, /*isReference*/ false);
            }
        }
    auto SE = Visit(E->getSubExpr());
    LLVM_DEBUG({
      if (!SE.val)
        E->dump();
    });
    auto Scalar = SE.getValue(Builder);
    if (auto SPT = Scalar.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      mlir::Type NT = Glob.getTypes().getMLIRType(E->getType());
      LLVM::LLVMPointerType PT = NT.dyn_cast<LLVM::LLVMPointerType>();
      if (!PT) {
        return ValueCategory(
            Builder.create<polygeist::Pointer2MemrefOp>(Loc, NT, Scalar),
            false);
      }
      PT = LLVM::LLVMPointerType::get(PT.getElementType(),
                                      SPT.getAddressSpace());
      auto Nval = Builder.create<mlir::LLVM::BitcastOp>(Loc, PT, Scalar);
      return ValueCategory(Nval, /*isReference*/ false);
    }

    LLVM_DEBUG({
      if (!Scalar.getType().isa<mlir::MemRefType>()) {
        E->dump();
        E->getType()->dump();
        llvm::errs() << "Scalar: " << Scalar << "\n";
      }
    });

    assert(Scalar.getType().isa<mlir::MemRefType>() &&
           "Expecting 'Scalar' to have MemRefType");

    auto ScalarTy = Scalar.getType().cast<mlir::MemRefType>();
    mlir::Type MLIRTy = Glob.getTypes().getMLIRType(E->getType());

    if (auto PT = MLIRTy.dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      assert(
          ScalarTy.getMemorySpaceAsInt() == PT.getAddressSpace() &&
          "The type of 'Scalar' does not have the same memory space as 'PT'");
      auto Val =
          Builder.create<mlir::polygeist::Memref2PointerOp>(Loc, PT, Scalar);
      return ValueCategory(Val, /*isReference*/ false);
    }

    if (auto MT = MLIRTy.dyn_cast<mlir::MemRefType>()) {
      auto Ty = mlir::MemRefType::get(MT.getShape(), MT.getElementType(),
                                      MemRefLayoutAttrInterface(),
                                      ScalarTy.getMemorySpace());
      if (ScalarTy.getShape().size() == MT.getShape().size() + 1)
        return ValueCategory(Builder.create<mlir::polygeist::SubIndexOp>(
                                 Loc, Ty, Scalar, getConstantIndex(0)),
                             /*isReference*/ false);

      if (ScalarTy.getShape().size() != MT.getShape().size()) {
        auto MemRefToPtr = Builder.create<polygeist::Memref2PointerOp>(
            Loc,
            LLVM::LLVMPointerType::get(Builder.getI8Type(),
                                       ScalarTy.getMemorySpaceAsInt()),
            Scalar);
        assert(Ty.getMemorySpaceAsInt() == ScalarTy.getMemorySpaceAsInt() &&
               "Expecting 'ty' and 'ScalarTy' to have the same memory space");
        auto PtrToMemRef =
            Builder.create<polygeist::Pointer2MemrefOp>(Loc, Ty, MemRefToPtr);
        return ValueCategory(PtrToMemRef, /*isReference*/ false);
      }

      return ValueCategory(Builder.create<memref::CastOp>(Loc, Ty, Scalar),
                           /*isReference*/ false);
    }
    LLVM_DEBUG({
      E->dump();
      E->getType()->dump();
      llvm::errs() << " Scalar: " << Scalar << " MLIRTy: " << MLIRTy << "\n";
    });
    llvm_unreachable("illegal type for cast");
  } break;
  case clang::CastKind::CK_LValueToRValue: {
    if (auto *DR = dyn_cast<DeclRefExpr>(E->getSubExpr())) {
      if (auto *VD = dyn_cast<VarDecl>(DR->getDecl()->getCanonicalDecl())) {
        if (NOUR_Constant == DR->isNonOdrUse()) {
          if (!VD->getInit()) {
            E->dump();
            VD->dump();
          }
          assert(VD->getInit());
          return Visit(VD->getInit());
        }
      }
      if (DR->getDecl()->getIdentifier() &&
          DR->getDecl()->getName() == "warpSize") {
        mlir::Type MLIRTy = Glob.getTypes().getMLIRType(E->getType());
        return ValueCategory(
            Builder.create<mlir::NVVM::WarpSizeOp>(Loc, MLIRTy),
            /*isReference*/ false);
      }
    }

    ValueCategory Prev = EmitLValue(E->getSubExpr());
    bool IsArray = false;
    Glob.getTypes().getMLIRType(E->getType(), &IsArray);
    if (IsArray)
      return Prev;

    mlir::Value Lres = Prev.getValue(Builder);

    LLVM_DEBUG({
      if (!Prev.isReference) {
        llvm::dbgs() << "LValueToRValue cast performed on an RValue: ";
        E->dump(llvm::dbgs(), Glob.getCGM().getContext());
        Lres.print(llvm::dbgs());
        llvm::dbgs() << "\n";
      }
    });
    return ValueCategory(Lres, /*isReference*/ false);
  }
  case clang::CastKind::CK_IntegralToFloating:
  case clang::CastKind::CK_FloatingToIntegral:
  case clang::CastKind::CK_FloatingCast:
  case clang::CastKind::CK_IntegralCast:
  case clang::CastKind::CK_BooleanToSignedIntegral:
    return EmitScalarConversion(Visit(E->getSubExpr()),
                                E->getSubExpr()->getType(), E->getType(),
                                E->getExprLoc());

  case clang::CastKind::CK_ArrayToPointerDecay:
    return CommonArrayToPointer(Visit(E->getSubExpr()));

  case clang::CastKind::CK_FunctionToPointerDecay: {
    auto Scalar = Visit(E->getSubExpr());
    assert(Scalar.isReference);
    return ValueCategory(Scalar.val, /*isReference*/ false);
  }
  case clang::CastKind::CK_ConstructorConversion:
  case clang::CastKind::CK_NoOp:
    return Visit(E->getSubExpr());

  case clang::CastKind::CK_ToVoid: {
    Visit(E->getSubExpr());
    return nullptr;
  }
  case clang::CastKind::CK_PointerToBoolean:
    return EmitPointerToBoolConversion(Loc, Visit(E->getSubExpr()));
  case clang::CastKind::CK_PointerToIntegral: {
    const auto DestTy = E->getType();
    assert(!DestTy->isBooleanType() && "bool should use PointerToBool");
    const auto PtrExpr = Visit(E->getSubExpr());
    return EmitPointerToIntegralConversion(
        Loc, Glob.getTypes().getMLIRType(DestTy), PtrExpr);
  }
  case clang::CastKind::CK_IntegralToBoolean:
    return EmitIntToBoolConversion(Loc, Visit(E->getSubExpr()));
  case clang::CastKind::CK_FloatingToBoolean:
    return EmitFloatToBoolConversion(Loc, Visit(E->getSubExpr()));
  case clang::CastKind::CK_VectorSplat: {
    const auto DstTy = Glob.getTypes().getMLIRType(E->getType());
    const auto Elt = Visit(E->getSubExpr());
    return Elt.Splat(Builder, Loc, DstTy);
  }
  case clang::CastKind::CK_IntegralToPointer: {
    auto VC = Visit(E->getSubExpr());

    // First convert to the correct width.
    const auto MiddleTy = Builder.getIntegerType(
        Glob.getCGM().getDataLayout().getPointerSizeInBits());
    const auto InputSigned =
        E->getSubExpr()->getType()->isSignedIntegerOrEnumerationType();
    const auto IntResult = VC.IntCast(Builder, Loc, MiddleTy, InputSigned);

    // Now perform the integral to pointer conversion.
    mlir::Type PostTy = Glob.getTypes().getMLIRType(E->getType());
    return EmitIntegralToPointerConversion(Loc, PostTy, IntResult);
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
  auto Cond = Visit(E->getCond()).getValue(Builder);
  assert(Cond != nullptr);
  if (auto LT = Cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    auto NullptrLlvm = Builder.create<mlir::LLVM::NullOp>(Loc, LT);
    Cond = Builder.create<mlir::LLVM::ICmpOp>(
        Loc, mlir::LLVM::ICmpPredicate::ne, Cond, NullptrLlvm);
  }
  auto PrevTy = Cond.getType().cast<mlir::IntegerType>();
  if (!PrevTy.isInteger(1)) {
    Cond = Builder.create<arith::CmpIOp>(
        Loc, arith::CmpIPredicate::ne, Cond,
        Builder.create<arith::ConstantIntOp>(Loc, 0, PrevTy));
  }
  std::vector<mlir::Type> Types;
  if (!E->getType()->isVoidType())
    Types.push_back(Glob.getTypes().getMLIRType(E->getType()));
  auto IfOp = Builder.create<mlir::scf::IfOp>(Loc, Types, Cond,
                                              /*hasElseRegion*/ true);

  auto Oldpoint = Builder.getInsertionPoint();
  auto *Oldblock = Builder.getInsertionBlock();
  Builder.setInsertionPointToStart(&IfOp.getThenRegion().back());

  auto TrueExpr = Visit(E->getTrueExpr());

  bool IsReference = E->isLValue() || E->isXValue();

  std::vector<mlir::Value> TrueArray;
  if (!E->getType()->isVoidType()) {
    if (!TrueExpr.val)
      E->dump();

    assert(TrueExpr.val);
    mlir::Value Truev;
    if (IsReference) {
      assert(TrueExpr.isReference);
      Truev = TrueExpr.val;
    } else {
      if (TrueExpr.isReference)
        if (auto MT = TrueExpr.val.getType().dyn_cast<MemRefType>())
          if (MT.getShape().size() != 1) {
            E->dump();
            E->getTrueExpr()->dump();
            llvm::errs() << " trueExpr: " << TrueExpr.val << "\n";
            assert(0);
          }
      Truev = TrueExpr.getValue(Builder);
    }
    assert(Truev != nullptr);
    TrueArray.push_back(Truev);
    Builder.create<mlir::scf::YieldOp>(Loc, TrueArray);
  }

  Builder.setInsertionPointToStart(&IfOp.getElseRegion().back());

  auto FalseExpr = Visit(E->getFalseExpr());
  std::vector<mlir::Value> Falsearray;
  if (!E->getType()->isVoidType()) {
    mlir::Value Falsev;
    if (IsReference) {
      assert(FalseExpr.isReference);
      Falsev = FalseExpr.val;
    } else
      Falsev = FalseExpr.getValue(Builder);
    assert(Falsev != nullptr);
    Falsearray.push_back(Falsev);
    Builder.create<mlir::scf::YieldOp>(Loc, Falsearray);
  }

  Builder.setInsertionPoint(Oldblock, Oldpoint);

  for (size_t I = 0; I < TrueArray.size(); I++)
    Types[I] = TrueArray[I].getType();
  auto NewIfOp = Builder.create<mlir::scf::IfOp>(Loc, Types, Cond,
                                                 /*hasElseRegion*/ true);
  NewIfOp.getThenRegion().takeBody(IfOp.getThenRegion());
  NewIfOp.getElseRegion().takeBody(IfOp.getElseRegion());
  IfOp.erase();
  return ValueCategory(NewIfOp.getResult(0), /*isReference*/ IsReference);
}

ValueCategory MLIRScanner::VisitStmtExpr(clang::StmtExpr *Stmt) {
  ValueCategory Off = nullptr;
  for (auto *A : Stmt->getSubStmt()->children())
    Off = Visit(A);

  return Off;
}

ValueCategory MLIRScanner::VisitBinAssign(BinaryOperator *E) {
  LLVM_DEBUG({
    llvm::dbgs() << "VisitBinAssign: ";
    E->dump();
    llvm::dbgs() << "\n";
  });

  ValueCategory RHS = Visit(E->getRHS());
  ValueCategory LHS = EmitLValue(E->getLHS());

  assert(LHS.isReference);
  mlir::Value ToStore = RHS.getValue(Builder);
  mlir::Type SubType;
  if (auto PT = LHS.val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>())
    SubType = PT.getElementType();
  else
    SubType = LHS.val.getType().cast<MemRefType>().getElementType();

  if (ToStore.getType() != SubType) {
    if (auto PrevTy = ToStore.getType().dyn_cast<mlir::IntegerType>()) {
      if (auto PostTy = SubType.dyn_cast<mlir::IntegerType>()) {
        bool SignedType = true;
        if (const auto *Bit = dyn_cast<BuiltinType>(&*E->getType())) {
          if (Bit->isUnsignedInteger())
            SignedType = false;
          if (Bit->isSignedInteger())
            SignedType = true;
        }

        if (PrevTy.getWidth() < PostTy.getWidth()) {
          if (SignedType)
            ToStore = Builder.create<arith::ExtSIOp>(Loc, PostTy, ToStore);
          else
            ToStore = Builder.create<arith::ExtUIOp>(Loc, PostTy, ToStore);
        } else if (PrevTy.getWidth() > PostTy.getWidth())
          ToStore = Builder.create<arith::TruncIOp>(Loc, PostTy, ToStore);
      }
    }
  }
  LHS.store(Builder, ToStore);
  return RHS;
}

class BinOpInfo {
public:
  BinOpInfo(ValueCategory LHS, ValueCategory RHS, QualType Ty,
            BinaryOperator::Opcode Opcode, const Expr *Expr)
      : LHS(LHS), RHS(RHS), Ty(Ty), Opcode(Opcode), E(Expr) {}

  ValueCategory getLHS() const { return LHS; }
  ValueCategory getRHS() const { return RHS; }
  constexpr QualType getType() const { return Ty; }
  constexpr BinaryOperator::Opcode getOpcode() const { return Opcode; }
  constexpr const Expr *getExpr() const { return E; }

private:
  const ValueCategory LHS;
  const ValueCategory RHS;
  const QualType Ty;                   // Computation Type.
  const BinaryOperator::Opcode Opcode; // Opcode of BinOp to perform
  const Expr *E;
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
#define HANDLEUNARYOP(OP)                                                      \
  case UO_##OP:                                                                \
    return Visit##OP(UO, PromotionType);

#include "Expressions.def"
#undef HANDLEUNARYOP

    default:
      break;
    }
  }

  const ValueCategory Res = Visit(E);
  if (Res.val) {
    const auto Loc = getMLIRLocation(E->getExprLoc());
    if (!PromotionType.isNull())
      return EmitPromotedValue(Loc, Res, PromotionType);
    return EmitUnPromotedValue(Loc, Res, E->getType());
  }
  return Res;
}

ValueCategory MLIRScanner::CastToVoidPtr(ValueCategory Ptr) {
  assert(mlirclang::isPointerOrMemRefTy(Ptr.val.getType()) &&
         "Expecting pointer or memref");

  const auto DestType =
      mlirclang::getPtrTyWithNewType(Ptr.val.getType(), Builder.getI8Type());

  return Ptr.BitCast(Builder, Loc, DestType);
}

ValueCategory MLIRScanner::EmitPromotedValue(Location Loc, ValueCategory Result,
                                             QualType PromotionType) {
  return Result.FPExt(Builder, Loc, Glob.getTypes().getMLIRType(PromotionType));
}

ValueCategory MLIRScanner::EmitUnPromotedValue(Location Loc,
                                               ValueCategory Result,
                                               QualType PromotionType) {
  return Result.FPTrunc(Builder, Loc,
                        Glob.getTypes().getMLIRType(PromotionType));
}

ValueCategory MLIRScanner::EmitPromotedScalarExpr(Expr *E,
                                                  QualType PromotionType) {
  if (!PromotionType.isNull())
    return EmitPromoted(E, PromotionType);
  return Visit(E);
}

ValueCategory MLIRScanner::EmitScalarCast(mlir::Location Loc, ValueCategory Src,
                                          QualType SrcQT, QualType DstQT,
                                          mlir::Type SrcTy, mlir::Type DstTy) {
  assert(SrcTy != DstTy && "Types should be different when casting");
  assert(!SrcQT->isAnyComplexType() && !DstQT->isAnyComplexType() &&
         "Not supported in cgeist");
  assert(!SrcQT->isMatrixType() && !DstQT->isMatrixType() &&
         "Not supported in cgeist");

  if (SrcTy.isIntOrIndex()) {
    bool InputSigned = SrcQT->isSignedIntegerOrEnumerationType();
    if (SrcQT->isBooleanType()) {
      // TODO: Should check options
      CGEIST_WARNING(llvm::WithColor::warning()
                     << "Treating boolean as unsigned\n");
      InputSigned = false;
    }

    if (DstTy.isIntOrIndex())
      return Src.IntCast(Builder, Loc, DstTy, InputSigned);
    if (InputSigned)
      return Src.SIToFP(Builder, Loc, DstTy);
    return Src.UIToFP(Builder, Loc, DstTy);
  }

  if (DstTy.isIntOrIndex()) {
    assert(SrcTy.isa<FloatType>() && "Unknown real conversion");
    bool IsSigned = DstQT->isSignedIntegerOrEnumerationType();

    // If we can't recognize overflow as undefined behavior, assume that
    // overflow saturates. This protects against normal optimizations if we are
    // compiling with non-standard FP semantics.
    CGEIST_WARNING(llvm::WithColor::warning()
                   << "Performing strict float cast overflow\n");

    if (IsSigned)
      return Src.FPToSI(Builder, Loc, DstTy);
    return Src.FPToUI(Builder, Loc, DstTy);
  }

  if (DstTy.cast<FloatType>().getWidth() < SrcTy.cast<FloatType>().getWidth())
    return Src.FPTrunc(Builder, Loc, DstTy);
  return Src.FPExt(Builder, Loc, DstTy);
}

ValueCategory MLIRScanner::EmitScalarConversion(ValueCategory Src,
                                                QualType SrcQT, QualType DstQT,
                                                SourceLocation Loc) {
  // TODO: Handle fixed points here when supported.
  // TODO: Take into account scalar conversion options.

  assert(!SrcQT->isFixedPointType() &&
         "Not handling conversion from fixed point types");
  assert(!DstQT->isFixedPointType() &&
         "Not handling conversion to fixed point types");

  mlirclang::CodeGen::CodeGenTypes &CGTypes = Glob.getTypes();

  SrcQT = Glob.getCGM().getContext().getCanonicalType(SrcQT);
  DstQT = Glob.getCGM().getContext().getCanonicalType(DstQT);
  if (SrcQT == DstQT)
    return Src;
  if (DstQT->isVoidType())
    return nullptr;

  mlir::Type SrcTy = Src.val.getType();

  const Location MLIRLoc = getMLIRLocation(Loc);
  if (DstQT->isBooleanType())
    return EmitConversionToBool(MLIRLoc, Src, SrcQT);

  mlir::Type DstTy = CGTypes.getMLIRType(DstQT);

  // Cast from half through float if half isn't a native type.
  const CodeGen::CodeGenModule &CGM = Glob.getCGM();
  if (SrcQT->isHalfType() && !CGM.getLangOpts().NativeHalfType) {
    // Cast to FP using the intrinsic if the half type itself isn't supported.
    if (DstTy.isa<FloatType>()) {
      CGEIST_WARNING({
        if (CGM.getContext().getTargetInfo().useFP16ConversionIntrinsics())
          llvm::WithColor::warning()
              << "Should call convert_from_fp16 intrinsic "
                 "to perfom this conversion\n";
      });
    } else {
      // Cast to other types through float, using either the intrinsic or FPExt,
      // depending on whether the half type itself is supported (as opposed to
      // operations on half, available with NativeHalfType).
      CGEIST_WARNING({
        if (CGM.getContext().getTargetInfo().useFP16ConversionIntrinsics())
          llvm::WithColor::warning()
              << "Should call convert_from_fp16 intrinsic "
                 "to perfom this conversion\n";
      });

      Src = Src.FPExt(Builder, MLIRLoc, Builder.getF32Type());
      SrcQT = CGM.getContext().FloatTy;
      SrcTy = Builder.getF32Type();
    }
  }

  // Ignore conversions like int -> uint.
  if (SrcTy == DstTy) {
    CGEIST_WARNING(llvm::WithColor::warning()
                   << "Not emitting implicit integer sign change checks\n");
    return Src;
  }

  // Handle pointer conversions next: pointers can only be converted to/from
  // other pointers and integers. Check for pointer types in terms of LLVM, as
  // some native types (like Obj-C id) may map to a pointer type.
  assert(!(DstTy.isa<MemRefType>() || SrcTy.isa<MemRefType>()) &&
         "Not implemented yet");

  // A scalar can be splatted to an extended vector of the same element type
  assert(!(DstQT->isExtVectorType() && !SrcQT->isVectorType()) &&
         "Not implemented yet");

  assert(!(SrcQT->isMatrixType() && DstQT->isMatrixType()) &&
         "Not implemented yet");

  assert(!(SrcTy.isa<mlir::VectorType>() || DstTy.isa<mlir::VectorType>()) &&
         "Not implemented yet");

  // Finally, we have the arithmetic types: real int/float.
  mlir::Type ResTy = DstTy;
  CGEIST_WARNING(llvm::WithColor::warning() << "Missing overflow checks\n");

  // Cast to half through float if half isn't a native type.
  if (DstQT->isHalfType() && !CGM.getContext().getLangOpts().NativeHalfType) {
    // Make sure we cast in a single step if from another FP type.
    if (SrcTy.isa<FloatType>()) {
      // Use the intrinsic if the half type itself isn't supported
      // (as opposed to operations on half, available with NativeHalfType).
      CGEIST_WARNING({
        if (CGM.getContext().getTargetInfo().useFP16ConversionIntrinsics())
          llvm::WithColor::warning() << "Should call convert_to_fp16 intrinsic "
                                        "to perfom this conversion\n";
      });
      // If the half type is supported, just use an fptrunc.
      return Src.FPTrunc(Builder, MLIRLoc, DstTy);
    }
    DstTy = Builder.getF32Type();
  }

  ValueCategory Res = EmitScalarCast(MLIRLoc, Src, SrcQT, DstQT, SrcTy, DstTy);

  if (DstTy != ResTy) {
    if (CGM.getContext().getTargetInfo().useFP16ConversionIntrinsics()) {
      assert(ResTy.isInteger(16) && "Only half FP requires extra conversion");
      CGEIST_WARNING(llvm::WithColor::warning()
                     << "Should call convert_to_fp16 intrinsic to "
                        "perfom this conversion\n");
    }
    Res = Res.FPTrunc(Builder, MLIRLoc, ResTy);
  }

  CGEIST_WARNING({
    llvm::WithColor::warning() << "Missing truncation checks\n";
    llvm::WithColor::warning() << "Missing integer sign change checks\n";
  });

  return Res;
}

ValueCategory MLIRScanner::EmitFloatToBoolConversion(Location Loc,
                                                     ValueCategory Src) {
  assert(Src.val.getType().isa<FloatType>() && "Expecting a float value");
  mlir::OpBuilder SubBuilder(Builder.getContext());
  SubBuilder.setInsertionPointToStart(EntryBlock);
  auto FloatTy = cast<FloatType>(Src.val.getType());
  auto Zero = SubBuilder.create<arith::ConstantFloatOp>(
      Loc, mlir::APFloat::getZero(FloatTy.getFloatSemantics()), FloatTy);
  return Src.FCmpUNE(Builder, Loc, Zero);
}

ValueCategory MLIRScanner::EmitPointerToBoolConversion(Location Loc,
                                                       ValueCategory Src) {
  if (auto MemRefTy = Src.val.getType().dyn_cast<MemRefType>()) {
    auto ElementTy = MemRefTy.getElementType();
    auto AddressSpace = MemRefTy.getMemorySpaceAsInt();
    Src = {
        Builder.create<polygeist::Memref2PointerOp>(
            Loc, LLVM::LLVMPointerType::get(ElementTy, AddressSpace), Src.val),
        Src.isReference};
  }
  assert(Src.val.getType().isa<LLVM::LLVMPointerType>() &&
         "Expecting a pointer");
  mlir::OpBuilder SubBuilder(Builder.getContext());
  SubBuilder.setInsertionPointToStart(EntryBlock);
  auto Zero = SubBuilder.create<LLVM::NullOp>(Loc, Src.val.getType());
  return {Builder.createOrFold<LLVM::ICmpOp>(Loc, LLVM::ICmpPredicate::ne,
                                             Src.val, Zero),
          false};
}

ValueCategory MLIRScanner::EmitIntToBoolConversion(Location Loc,
                                                   ValueCategory Src) {
  assert(Src.val.getType().isa<IntegerType>() && "Expecting an integer value");
  mlir::OpBuilder SubBuilder(Builder.getContext());
  SubBuilder.setInsertionPointToStart(EntryBlock);
  auto Zero = SubBuilder.create<arith::ConstantIntOp>(
      Loc, 0, Src.val.getType().cast<IntegerType>().getWidth());
  return Src.ICmpNE(Builder, Loc, Zero);
}

ValueCategory MLIRScanner::EmitConversionToBool(Location Loc, ValueCategory Src,
                                                QualType SrcQT) {
  assert(SrcQT.isCanonical() && "EmitScalarConversion strips typedefs");
  assert(!isa<MemberPointerType>(SrcQT) && "Not implemented yet");
  const auto ValTy = Src.val.getType();
  if (ValTy.isa<FloatType>())
    return EmitFloatToBoolConversion(Loc, Src);
  if (ValTy.isa<IntegerType>())
    return EmitIntToBoolConversion(Loc, Src);
  if (ValTy.isa<LLVM::LLVMPointerType, MemRefType>())
    return EmitPointerToBoolConversion(Loc, Src);
  llvm_unreachable("Unknown scalar type to convert");
}

ValueCategory MLIRScanner::EmitPointerToIntegralConversion(Location Loc,
                                                           mlir::Type DestTy,
                                                           ValueCategory Src) {
  assert(DestTy.isa<IntegerType>() && "Expecting integer type");
  assert(mlirclang::isPointerOrMemRefTy(Src.val.getType()) &&
         "Expecting pointer input");

  return Src.MemRef2Ptr(Builder, Loc).PtrToInt(Builder, Loc, DestTy);
}

ValueCategory MLIRScanner::EmitIntegralToPointerConversion(Location Loc,
                                                           mlir::Type DestTy,
                                                           ValueCategory Src) {
  assert(mlirclang::isPointerOrMemRefTy(DestTy) && "Expecting pointer type");
  assert(Src.val.getType().isa<IntegerType>() &&
         Src.val.getType().cast<IntegerType>().getWidth() ==
             Glob.getCGM().getDataLayout().getPointerSizeInBits() &&
         "Expecting pointer-width integer input");

  return TypeSwitch<mlir::Type, ValueCategory>(DestTy)
      .Case<LLVM::LLVMPointerType>(
          [=](auto Ty) { return Src.IntToPtr(Builder, Loc, Ty); })
      .Case<MemRefType>([=](auto Ty) {
        const auto MiddlePtrTy = LLVM::LLVMPointerType::get(
            Ty.getElementType(), Ty.getMemorySpaceAsInt());
        return Src.IntToPtr(Builder, Loc, MiddlePtrTy).Ptr2MemRef(Builder, Loc);
      });
}

ValueCategory
MLIRScanner::EmitCompoundAssignmentLValue(clang::CompoundAssignOperator *E) {
  switch (E->getOpcode()) {
#define HANDLEBINOP(OP)                                                        \
  case BO_##OP##Assign:                                                        \
    return EmitCompoundAssignLValue(E, &MLIRScanner::EmitBin##OP).first;

#include "Expressions.def"
#undef HANDLEBINOP
  case BO_PtrMemD:
  case BO_PtrMemI:
  case BO_Mul:
  case BO_Div:
  case BO_Rem:
  case BO_Add:
  case BO_Sub:
  case BO_Shl:
  case BO_Shr:
  case BO_LT:
  case BO_GT:
  case BO_LE:
  case BO_GE:
  case BO_EQ:
  case BO_NE:
  case BO_Cmp:
  case BO_And:
  case BO_Xor:
  case BO_Or:
  case BO_LAnd:
  case BO_LOr:
  case BO_Assign:
  case BO_Comma:
    llvm_unreachable("Not valid compound assignment operators");
  }

  llvm_unreachable("Unhandled compound assignment operator");
}

ValueCategory MLIRScanner::EmitLValue(Expr *E) {
  switch (E->getStmtClass()) {
  default:
    return Visit(E);

  case Expr::CompoundAssignOperatorClass: {
    QualType Ty = E->getType();
    if (const AtomicType *AT = Ty->getAs<AtomicType>())
      Ty = AT->getValueType();
    auto *CAO = cast<CompoundAssignOperator>(E);
    assert(!Ty->isAnyComplexType() && "Handle complex types.");
    return EmitCompoundAssignmentLValue(CAO);
  }
  case Expr::ParenExprClass:
    return EmitLValue(cast<ParenExpr>(E)->getSubExpr());
  case Expr::ArraySubscriptExprClass:
    return EmitArraySubscriptExpr(cast<ArraySubscriptExpr>(E));
  }
}

std::pair<ValueCategory, ValueCategory> MLIRScanner::EmitCompoundAssignLValue(
    CompoundAssignOperator *E,
    ValueCategory (MLIRScanner::*Func)(const BinOpInfo &)) {
  QualType LHSTy = E->getLHS()->getType();

  CGEIST_WARNING({
    if (E->getComputationResultType()->isAnyComplexType())
      llvm::WithColor::warning() << "Not handling complex types yet\n";
  });

  // Emit the RHS first.  __block variables need to have the rhs evaluated
  // first, plus this should improve codegen a little.

  QualType PromotionTypeCR =
      Glob.getTypes().getPromotionType(E->getComputationResultType());
  if (PromotionTypeCR.isNull())
    PromotionTypeCR = E->getComputationResultType();
  QualType PromotionTypeLHS =
      Glob.getTypes().getPromotionType(E->getComputationLHSType());
  QualType PromotionTypeRHS =
      Glob.getTypes().getPromotionType(E->getRHS()->getType());
  const ValueCategory RHS =
      EmitPromotedScalarExpr(E->getRHS(), PromotionTypeRHS);

  const QualType Ty = PromotionTypeCR;
  const BinaryOperator::Opcode OpCode = E->getOpcode();
  const SourceLocation Loc = E->getExprLoc();

  // Load/convert the LHS.
  CGEIST_WARNING(llvm::WithColor::warning() << "Emitting unchecked LValue\n");

  const ValueCategory LHSLV = EmitLValue(E->getLHS());
  CGEIST_WARNING({
    if (isa<AtomicType>(LHSTy))
      llvm::WithColor::warning()
          << "Not handling atomics. Should perform RMW operation here.\n";
  });

  ValueCategory LHS{LHSLV.getValue(Builder), false};
  if (!PromotionTypeLHS.isNull())
    LHS = EmitScalarConversion(LHS, LHSTy, PromotionTypeLHS, E->getExprLoc());
  else
    LHS = EmitScalarConversion(LHS, LHSTy, E->getComputationLHSType(), Loc);

  // Expand the binary operator.
  ValueCategory Result = (this->*Func)({LHS, RHS, Ty, OpCode, E});
  // Convert the result back to the LHS type,
  // potentially with Implicit Conversion sanitizer check.
  Result = EmitScalarConversion(Result, PromotionTypeCR, LHSTy, Loc);

  LHSLV.store(Builder, Result.val);

  CGEIST_WARNING({
    if (Glob.getCGM().getLangOpts().OpenMP)
      llvm::WithColor::warning()
          << "Should checkAndEmitLastprivateConditional, "
             "but not implemented yet.\n";
  });

  return {LHSLV, Result};
}

ValueCategory MLIRScanner::EmitCompoundAssign(
    CompoundAssignOperator *E,
    ValueCategory (MLIRScanner::*Func)(const BinOpInfo &)) {
  const auto &[LHS, Result] = EmitCompoundAssignLValue(E, Func);
  // The return value is the stored value in C and the LValue in C++.
  return Glob.getCGM().getLangOpts().CPlusPlus ? LHS : Result;
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
      Result = EmitUnPromotedValue(getMLIRLocation(E->getExprLoc()), Result,   \
                                   E->getType());                              \
    return Result;                                                             \
  }                                                                            \
                                                                               \
  ValueCategory MLIRScanner::VisitBin##OP##Assign(BinaryOperator *E) {         \
    LLVM_DEBUG({                                                               \
      llvm::dbgs() << "VisitBin" #OP "Assign: ";                               \
      E->dump();                                                               \
      llvm::dbgs() << "\n";                                                    \
    });                                                                        \
    return EmitCompoundAssign(cast<CompoundAssignOperator>(E),                 \
                              &MLIRScanner::EmitBin##OP);                      \
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

static void informNoOverflowCheck(LangOptions::SignedOverflowBehaviorTy SOB,
                                  llvm::StringRef OpName) {
  if (SOB != clang::LangOptions::SOB_Defined)
    llvm::WithColor::warning()
        << "Not emitting overflow-checked " << OpName << "\n";
}

ValueCategory MLIRScanner::EmitBinMul(const BinOpInfo &Info) {
  const auto Loc = getMLIRLocation(Info.getExpr()->getExprLoc());
  const auto LHS = Info.getLHS();
  const auto RHS = Info.getRHS().val;

  if (Info.getType()->isSignedIntegerOrEnumerationType()) {
    CGEIST_WARNING(informNoOverflowCheck(
        Glob.getCGM().getLangOpts().getSignedOverflowBehavior(), "mul"));
    return LHS.Mul(Builder, Loc, RHS);
  }

  assert(!Info.getType()->isConstantMatrixType() && "Not yet implemented");

  if (mlirclang::isFPOrFPVectorTy(LHS.val.getType()))
    return LHS.FMul(Builder, Loc, RHS);
  return LHS.Mul(Builder, Loc, RHS);
}

ValueCategory MLIRScanner::EmitBinDiv(const BinOpInfo &Info) {
  CGEIST_WARNING(
      llvm::WithColor::warning()
      << "Not checking division by zero nor signed integer overflow.\n");

  assert(!Info.getType()->isConstantMatrixType() && "Not implemented");

  const auto Loc = getMLIRLocation(Info.getExpr()->getExprLoc());
  const auto LHS = Info.getLHS();
  const auto RHS = Info.getRHS().val;
  if (mlirclang::isFPOrFPVectorTy(LHS.val.getType())) {
    const auto &LangOpts = Glob.getCGM().getLangOpts();
    const auto &CodeGenOpts = Glob.getCGM().getCodeGenOpts();
    CGEIST_WARNING({
      if ((LangOpts.OpenCL && !CodeGenOpts.OpenCLCorrectlyRoundedDivSqrt) ||
          (LangOpts.HIP && LangOpts.CUDAIsDevice &&
           !CodeGenOpts.HIPCorrectlyRoundedDivSqrt)) {
        // OpenCL v1.1 s7.4: minimum accuracy of single precision / is 2.5ulp
        // OpenCL v1.2 s5.6.4.2: The -cl-fp32-correctly-rounded-divide-sqrt
        // build option allows an application to specify that single precision
        // floating-point divide (x/y and 1/x) and sqrt used in the program
        // source are correctly rounded.
        llvm::WithColor::warning()
            << "Not applying OpenCL/HIP precision options.\n";
      }
    });
    return LHS.FDiv(Builder, Loc, RHS);
  }
  if (Info.getType()->hasUnsignedIntegerRepresentation())
    return LHS.UDiv(Builder, Loc, RHS);
  return LHS.SDiv(Builder, Loc, RHS);
}

ValueCategory MLIRScanner::EmitBinRem(const BinOpInfo &Info) {
  CGEIST_WARNING(
      llvm::WithColor::warning()
      << "Not checking division by zero nor signed integer overflow.\n");

  const auto Loc = getMLIRLocation(Info.getExpr()->getExprLoc());
  const auto LHS = Info.getLHS();
  const auto RHS = Info.getRHS().val;
  if (Info.getType()->hasUnsignedIntegerRepresentation())
    return LHS.URem(Builder, Loc, RHS);
  return LHS.SRem(Builder, Loc, RHS);
}

/// Casts index of subindex operation conditionally.
static Optional<Value> castSubIndexOpIndex(OpBuilder &Builder, Location Loc,
                                           ValueCategory Pointer,
                                           ValueRange IdxList, bool IsSigned) {
  if (Pointer.val.getType().isa<MemRefType>()) {
    assert(IdxList.size() == 1 && "SubIndexOp accepts just an index");
    return ValueCategory(IdxList.front(), false)
        .IntCast(Builder, Loc, Builder.getIndexType(), IsSigned)
        .val;
  }
  return std::nullopt;
}

ValueCategory MLIRScanner::EmitCheckedInBoundsPtrOffsetOp(mlir::Type ElemTy,
                                                          ValueCategory Pointer,
                                                          ValueRange IdxList,
                                                          bool IsSigned, bool) {
  assert(mlirclang::isPointerOrMemRefTy(Pointer.val.getType()) &&
         "Expecting pointer or MemRef");
  assert(std::all_of(IdxList.begin(), IdxList.end(),
                     [](mlir::Value Val) {
                       return Val.getType().isa<IntegerType>();
                     }) &&
         "Expecting indices list");

  Optional<Value> NewValue =
      castSubIndexOpIndex(Builder, Loc, Pointer, IdxList, IsSigned);
  if (NewValue.has_value())
    IdxList = NewValue.value();

  return Pointer.InBoundsGEPOrSubIndex(Builder, Loc, ElemTy, IdxList);
}

ValueCategory MLIRScanner::EmitPointerArithmetic(const BinOpInfo &Info) {
  const auto *Expr = cast<BinaryOperator>(Info.getExpr());

  ValueCategory Pointer = Info.getLHS();
  clang::Expr *PointerOperand = Expr->getLHS();
  ValueCategory Index = Info.getRHS();
  clang::Expr *IndexOperand = Expr->getRHS();

  const BinaryOperator::Opcode Opcode = Info.getOpcode();
  const bool IsSubtraction =
      Opcode == clang::BO_Sub || Opcode == clang::BO_SubAssign;

  assert((!IsSubtraction ||
          mlirclang::isPointerOrMemRefTy(Pointer.val.getType())) &&
         "The LHS is always a pointer in a subtraction");

  if (!mlirclang::isPointerOrMemRefTy(Pointer.val.getType())) {
    std::swap(Pointer, Index);
    std::swap(PointerOperand, IndexOperand);
  }

  assert(Index.val.getType().isa<IntegerType>() && "Expecting integer type");
  assert(mlirclang::isPointerOrMemRefTy(Pointer.val.getType()) &&
         "Expecting pointer type");

  mlir::Type PtrTy = Pointer.val.getType();
  clang::CodeGen::CodeGenModule &CGM = Glob.getCGM();

  // Some versions of glibc and gcc use idioms (particularly in their malloc
  // routines) that add a pointer-sized integer (known to be a pointer
  // value) to a null pointer in order to cast the value back to an integer
  // or as part of a pointer alignment algorithm.  This is undefined
  // behavior, but we'd like to be able to compile programs that use it.
  //
  // Normally, we'd generate a GEP with a null-pointer base here in response
  // to that code, but it's also UB to dereference a pointer created that
  // way.  Instead (as an acknowledged hack to tolerate the idiom) we will
  // generate a direct cast of the integer value to a pointer.
  //
  // The idiom (p = nullptr + N) is not met if any of the following are
  // true:
  //
  //   The operation is subtraction.
  //   The index is not pointer-sized.
  //   The pointer type is not byte-sized.
  //
  if (BinaryOperator::isNullPointerArithmeticExtension(
          CGM.getContext(), Opcode, PointerOperand, IndexOperand)) {
    return EmitIntegralToPointerConversion(Loc, PtrTy, Index);
  }

  const llvm::DataLayout &DL = CGM.getDataLayout();
  const unsigned IndexTypeSize = DL.getIndexTypeSizeInBits(
      CGM.getTypes().ConvertType(PointerOperand->getType()));
  const bool IsSigned =
      IndexOperand->getType()->isSignedIntegerOrEnumerationType();
  const unsigned Width = Index.val.getType().getIntOrFloatBitWidth();
  if (Width != IndexTypeSize) {
    // Zero-extend or sign-extend the pointer value according to
    // whether the index is signed or not.
    Index = Index.IntCast(Builder, Loc, Builder.getIntegerType(IndexTypeSize),
                          IsSigned);
  }

  // If this is subtraction, negate the index.
  if (IsSubtraction)
    Index = Index.Neg(Builder, Loc);

  const auto *PointerType =
      PointerOperand->getType()->getAs<clang::PointerType>();

  assert(PointerType && "Not pointer type");

  QualType ElementType = PointerType->getPointeeType();
  assert(!CGM.getContext().getAsVariableArrayType(ElementType) &&
         "Not implemented yet");

  // Explicitly handle GNU void* and function pointer arithmetic extensions.
  // The GNU void* casts amount to no-ops since our void* type is i8*, but
  // this is future proof.
  if (ElementType->isVoidType() || ElementType->isFunctionType()) {
    assert(PtrTy.isa<LLVM::LLVMPointerType>() && "Expecting pointer type");
    auto Result = CastToVoidPtr(Pointer);
    Result = Result.GEP(Builder, Loc, Builder.getI8Type(), Index.val);
    return Result.BitCast(Builder, Loc, Pointer.val.getType());
  }

  auto ElemTy = Glob.getTypes().getMLIRTypeForMem(ElementType);
  if (CGM.getLangOpts().isSignedOverflowDefined()) {
    if (Optional<Value> NewIndex =
            castSubIndexOpIndex(Builder, Loc, Pointer, Index.val, IsSigned))
      Index.val = *NewIndex;
    return Pointer.GEPOrSubIndex(Builder, Loc, ElemTy, Index.val);
  }

  return EmitCheckedInBoundsPtrOffsetOp(ElemTy, Pointer, Index.val, IsSigned,
                                        IsSubtraction);
}

ValueCategory MLIRScanner::EmitBinAdd(const BinOpInfo &Info) {
  const Location Loc = getMLIRLocation(Info.getExpr()->getExprLoc());
  const ValueCategory LHS = Info.getLHS();
  const ValueCategory RHS = Info.getRHS();

  if (mlirclang::isPointerOrMemRefTy(LHS.val.getType()) ||
      mlirclang::isPointerOrMemRefTy(RHS.val.getType()))
    return EmitPointerArithmetic(Info);

  if (Info.getType()->isSignedIntegerOrEnumerationType()) {
    CGEIST_WARNING(informNoOverflowCheck(
        Glob.getCGM().getLangOpts().getSignedOverflowBehavior(), "add"));
    return LHS.Add(Builder, Loc, RHS.val);
  }

  assert(!Info.getType()->isConstantMatrixType() && "Not yet implemented");

  if (mlirclang::isFPOrFPVectorTy(LHS.val.getType()))
    return LHS.FAdd(Builder, Loc, RHS.val);

  return LHS.Add(Builder, Loc, RHS.val);
}

ValueCategory MLIRScanner::EmitBinSub(const BinOpInfo &Info) {
  const Location Loc = getMLIRLocation(Info.getExpr()->getExprLoc());
  ValueCategory LHS = Info.getLHS();
  ValueCategory RHS = Info.getRHS();

  // The LHS is always a pointer if either side is.
  if (!mlirclang::isPointerOrMemRefTy(LHS.val.getType())) {
    if (Info.getType()->isSignedIntegerOrEnumerationType()) {
      CGEIST_WARNING(informNoOverflowCheck(
          Glob.getCGM().getLangOpts().getSignedOverflowBehavior(), "sub"));
      return LHS.Sub(Builder, Loc, RHS.val);
    }
    assert(!Info.getType()->isConstantMatrixType() && "Not yet implemented");
    if (mlirclang::isFPOrFPVectorTy(LHS.val.getType()))
      return LHS.FSub(Builder, Loc, RHS.val);
    return LHS.Sub(Builder, Loc, RHS.val);
  }

  // If the RHS is not a pointer, then we have normal pointer
  // arithmetic.
  if (!mlirclang::isPointerOrMemRefTy(RHS.val.getType()))
    return EmitPointerArithmetic(Info);

  // Otherwise, this is a pointer subtraction.

  // Do the raw subtraction part.
  const auto PtrDiffTy = Builder.getIntegerType(
      Glob.getCGM().getDataLayout().getPointerSizeInBits());
  LHS = EmitPointerToIntegralConversion(Loc, PtrDiffTy, LHS);
  RHS = EmitPointerToIntegralConversion(Loc, PtrDiffTy, RHS);
  const auto DiffInChars = LHS.Sub(Builder, Loc, RHS.val);

  // Okay, figure out the element size.
  const QualType ElementType = cast<BinaryOperator>(Info.getExpr())
                                   ->getLHS()
                                   ->getType()
                                   ->getPointeeType();

  assert(!Glob.getCGM().getContext().getAsVariableArrayType(ElementType) &&
         "Not implemented yet");

  const CharUnits ElementSize =
      (ElementType->isVoidType() || ElementType->isFunctionType())
          ? CharUnits::One()
          : Glob.getCGM().getContext().getTypeSizeInChars(ElementType);

  if (ElementSize.isOne())
    return DiffInChars;

  const auto Divisor = Builder.createOrFold<arith::ConstantIntOp>(
      Loc, ElementSize.getQuantity(), PtrDiffTy);

  return DiffInChars.ExactSDiv(Builder, Loc, Divisor);
}

static mlir::Value GetWidthMinusOneValue(mlir::OpBuilder &Builder,
                                         mlir::Location Loc, mlir::Value LHS,
                                         mlir::Value RHS) {
  auto Ty = LHS.getType();
  IntegerType IntTy;
  if (auto VT = Ty.dyn_cast<mlir::VectorType>())
    IntTy = VT.getElementType().cast<IntegerType>();
  else
    IntTy = Ty.cast<IntegerType>();

  const auto WidthMinusOne = IntTy.getWidth() - 1;
  ValueCategory Val{
      Builder.createOrFold<arith::ConstantIntOp>(Loc, WidthMinusOne, IntTy),
      false};
  if (auto VT = Ty.dyn_cast<mlir::VectorType>())
    Val = Val.Splat(Builder, Loc, VT);
  return Val.val;
}

ValueCategory MLIRScanner::ConstrainShiftValue(ValueCategory LHS,
                                               ValueCategory RHS) {
  IntegerType Ty;
  if (auto VT = LHS.val.getType().dyn_cast<mlir::VectorType>())
    Ty = VT.getElementType().cast<IntegerType>();
  else
    Ty = LHS.val.getType().cast<IntegerType>();

  if (llvm::isPowerOf2_64(Ty.getWidth()))
    return RHS.And(Builder, Loc,
                   GetWidthMinusOneValue(Builder, Loc, LHS.val, RHS.val));
  return RHS.URem(Builder, Loc,
                  Builder.createOrFold<arith::ConstantIntOp>(
                      Loc, Ty.getWidth(), RHS.val.getType()));
}

ValueCategory MLIRScanner::EmitBinShl(const BinOpInfo &Info) {
  const auto Loc = getMLIRLocation(Info.getExpr()->getExprLoc());
  auto LHS = Info.getLHS();
  auto RHS = Info.getRHS();

  // LLVM requires the LHS and RHS to be the same type: promote or truncate the
  // RHS to the same size as the LHS.
  if (LHS.val.getType() != RHS.val.getType())
    RHS = RHS.IntCast(Builder, Loc, LHS.val.getType(), /*IsSigned*/ false);

  if (Glob.getCGM().getLangOpts().OpenCL) {
    this->Loc = Loc;
    RHS = ConstrainShiftValue(LHS, RHS);
  } else {
    CGEIST_WARNING(llvm::WithColor::warning() << "Not performing SHL checks\n");
  }

  return LHS.Shl(Builder, Loc, RHS.val);
}

ValueCategory MLIRScanner::EmitBinShr(const BinOpInfo &Info) {
  const auto Loc = getMLIRLocation(Info.getExpr()->getExprLoc());
  auto LHS = Info.getLHS();
  auto RHS = Info.getRHS();

  // LLVM requires the LHS and RHS to be the same type: promote or truncate the
  // RHS to the same size as the LHS.
  if (LHS.val.getType() != RHS.val.getType())
    RHS = RHS.IntCast(Builder, Loc, LHS.val.getType(), /*IsSigned*/ false);

  if (Glob.getCGM().getLangOpts().OpenCL) {
    this->Loc = Loc;
    RHS = ConstrainShiftValue(LHS, RHS);
  } else {
    CGEIST_WARNING(llvm::WithColor::warning() << "Not performing SHR checks\n");
  }

  if (Info.getType()->hasUnsignedIntegerRepresentation())
    return LHS.LShr(Builder, Loc, RHS.val);
  return LHS.AShr(Builder, Loc, RHS.val);
}

ValueCategory MLIRScanner::EmitBinAnd(const BinOpInfo &Info) {
  const auto Loc = getMLIRLocation(Info.getExpr()->getExprLoc());
  auto LHS = Info.getLHS();
  auto RHS = Info.getRHS();
  return LHS.And(Builder, Loc, RHS.val);
}

ValueCategory MLIRScanner::EmitBinXor(const BinOpInfo &Info) {
  const auto Loc = getMLIRLocation(Info.getExpr()->getExprLoc());
  auto LHS = Info.getLHS();
  auto RHS = Info.getRHS();
  return LHS.Xor(Builder, Loc, RHS.val);
}

ValueCategory MLIRScanner::EmitBinOr(const BinOpInfo &Info) {
  const auto Loc = getMLIRLocation(Info.getExpr()->getExprLoc());
  auto LHS = Info.getLHS();
  auto RHS = Info.getRHS();
  return LHS.Or(Builder, Loc, RHS.val);
}

#define HANDLEUNARYOP(OP)                                                      \
  ValueCategory MLIRScanner::VisitUnary##OP(UnaryOperator *E,                  \
                                            QualType PromotionType) {          \
    LLVM_DEBUG({                                                               \
      llvm::dbgs() << "VisitUnary" #OP ": ";                                   \
      E->dump();                                                               \
      llvm::dbgs() << "\n";                                                    \
    });                                                                        \
    QualType promotionTy =                                                     \
        PromotionType.isNull()                                                 \
            ? Glob.getTypes().getPromotionType(E->getSubExpr()->getType())     \
            : PromotionType;                                                   \
    ValueCategory result = Visit##OP(E, promotionTy);                          \
    if (result.val && !promotionTy.isNull())                                   \
      result = EmitUnPromotedValue(getMLIRLocation(E->getExprLoc()), result,   \
                                   E->getType());                              \
    return result;                                                             \
  }
#include "Expressions.def"
#undef HANDLEUNARYOP

ValueCategory MLIRScanner::VisitPlus(UnaryOperator *E, QualType PromotionType) {
  if (!PromotionType.isNull())
    return EmitPromotedScalarExpr(E->getSubExpr(), PromotionType);
  return Visit(E->getSubExpr());
}

ValueCategory MLIRScanner::VisitMinus(UnaryOperator *E,
                                      QualType PromotionType) {
  const Location Loc = getMLIRLocation(E->getExprLoc());
  ValueCategory Op;
  if (!PromotionType.isNull())
    Op = EmitPromotedScalarExpr(E->getSubExpr(), PromotionType);
  else
    Op = Visit(E->getSubExpr());

  // Generate a unary FNeg for FP ops.
  if (mlirclang::isFPOrFPVectorTy(Op.val.getType()))
    return Op.FNeg(Builder, Loc);

  // Emit unary minus with EmitBinSub so we handle overflow cases etc.
  const ValueCategory Zero =
      ValueCategory::getNullValue(Builder, Loc, Op.val.getType());
  return EmitBinSub(
      BinOpInfo{Zero, Op, E->getType(), BinaryOperator::Opcode::BO_Sub, E});
}

ValueCategory MLIRScanner::VisitImag(UnaryOperator *E, QualType PromotionType) {
  Expr *Op = E->getSubExpr();

  assert(!Op->getType()->isAnyComplexType() && "Unsupported");

  // __imag on a scalar returns zero.  Emit the subexpr to ensure side
  // effects are evaluated, but not the actual value.
  if (Op->isGLValue())
    EmitLValue(Op);
  else if (!PromotionType.isNull())
    EmitPromotedScalarExpr(Op, PromotionType);
  else
    Visit(Op);
  auto ResTy = Glob.getTypes().getMLIRType(
      !PromotionType.isNull() ? PromotionType : E->getType());
  return ValueCategory::getNullValue(Builder, getMLIRLocation(E->getExprLoc()),
                                     ResTy);
}

ValueCategory MLIRScanner::VisitReal(UnaryOperator *E, QualType PromotionType) {
  Expr *Op = E->getSubExpr();

  assert(!Op->getType()->isAnyComplexType() && "Unsupported");

  if (!PromotionType.isNull())
    return EmitPromotedScalarExpr(Op, PromotionType);
  return Visit(Op);
}
