//===--- CGCall.cc - Encapsulate calling convention details ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeUtils.h"
#include "clang-mlir.h"
#include "utils.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Target/LLVMIR/TypeFromLLVM.h"

#define DEBUG_TYPE "CGCall"

using namespace clang;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::func;
using namespace mlirclang;

extern llvm::cl::opt<bool> CudaLower;
extern llvm::cl::opt<bool> GenerateAllSYCLFuncs;

/******************************************************************************/
/*                           Utility Functions                                */
/******************************************************************************/

/// Try to typecast the caller arg of type MemRef to fit the corresponding
/// callee arg type. We only deal with the cast where src and dst have the same
/// shape size and elem type, and just the first shape differs: src has -1 and
/// dst has a constant integer.
static mlir::Value castCallerMemRefArg(mlir::Value CallerArg,
                                       mlir::Type CalleeArgType,
                                       mlir::OpBuilder &B) {
  mlir::OpBuilder::InsertionGuard Guard(B);
  mlir::Type CallerArgType = CallerArg.getType();

  if (MemRefType DstTy = CalleeArgType.dyn_cast_or_null<MemRefType>()) {
    MemRefType SrcTy = CallerArgType.dyn_cast<MemRefType>();
    if (SrcTy && DstTy.getElementType() == SrcTy.getElementType() &&
        DstTy.getMemorySpace() == SrcTy.getMemorySpace()) {
      auto SrcShape = SrcTy.getShape();
      auto DstShape = DstTy.getShape();

      if (SrcShape.size() == DstShape.size() && !SrcShape.empty() &&
          SrcShape[0] == -1 &&
          std::equal(std::next(SrcShape.begin()), SrcShape.end(),
                     std::next(DstShape.begin()))) {
        B.setInsertionPointAfterValue(CallerArg);

        return B.create<mlir::memref::CastOp>(CallerArg.getLoc(), CalleeArgType,
                                              CallerArg);
      }
    }
  }

  // Return the original value when casting fails.
  return CallerArg;
}

/// Typecast the caller args to match the callee's signature. Mismatches that
/// cannot be resolved by given rules won't raise exceptions, e.g., if the
/// expected type for an arg is memref<10xi8> while the provided is
/// memref<20xf32>, we will simply ignore the case in this function and wait for
/// the rest of the pipeline to detect it.
static void castCallerArgs(mlir::func::FuncOp Callee,
                           llvm::SmallVectorImpl<mlir::Value> &Args,
                           mlir::OpBuilder &B) {
  mlir::FunctionType FuncTy = Callee.getFunctionType();
  assert(Args.size() == FuncTy.getNumInputs() &&
         "The caller arguments should have the same size as the number of "
         "callee arguments as the interface.");

  LLVM_DEBUG({
    llvm::dbgs() << "FuncTy: " << FuncTy << "\n";
    llvm::dbgs() << "Args: \n";
    for (const mlir::Value &arg : Args)
      llvm::dbgs().indent(2) << arg << "\n";
  });

  for (unsigned I = 0; I < Args.size(); ++I) {
    mlir::Type CalleeArgType = FuncTy.getInput(I);
    mlir::Type CallerArgType = Args[I].getType();

    if (CalleeArgType == CallerArgType)
      continue;

    if (CalleeArgType.isa<MemRefType>())
      Args[I] = castCallerMemRefArg(Args[I], CalleeArgType, B);
    assert(CalleeArgType == Args[I].getType() && "Callsite argument mismatch");
  }
}

/******************************************************************************/
/*                               MLIRScanner                                  */
/******************************************************************************/

ValueCategory MLIRScanner::CallHelper(
    mlir::func::FuncOp Tocall, QualType ObjType,
    ArrayRef<std::pair<ValueCategory, clang::Expr *>> Arguments,
    QualType RetType, bool RetReference, clang::Expr *Expr,
    const FunctionDecl *Callee) {
  SmallVector<mlir::Value, 4> Args;
  auto FnType = Tocall.getFunctionType();
  const clang::CodeGen::CGFunctionInfo &FI =
      Glob.GetOrCreateCGFunctionInfo(Callee);
  auto FIArgs = FI.arguments();

  size_t I = 0;
  // map from declaration name to mlir::value
  std::map<std::string, mlir::Value> MapFuncOperands;

  for (auto Pair : Arguments) {
    ValueCategory Arg = std::get<0>(Pair);
    clang::Expr *A = std::get<1>(Pair);

    LLVM_DEBUG({
      if (!Arg.val) {
        Expr->dump();
        A->dump();
      }
    });
    assert(Arg.val && "expect not null");

    if (auto *Ice = dyn_cast_or_null<ImplicitCastExpr>(A))
      if (auto *Dre = dyn_cast<DeclRefExpr>(Ice->getSubExpr()))
        MapFuncOperands.insert(
            make_pair(Dre->getDecl()->getName().str(), Arg.val));

    if (I >= FnType.getInputs().size() || (I != 0 && A == nullptr)) {
      LLVM_DEBUG({
        Expr->dump();
        Tocall.dump();
        FnType.dump();
        for (auto a : Arguments)
          std::get<1>(a)->dump();
      });
      assert(false && "too many arguments in calls");
    }

    bool IsReference =
        (I == 0 && A == nullptr) || A->isLValue() || A->isXValue();

    bool IsArray = false;
    QualType AType = (I == 0 && A == nullptr) ? ObjType : A->getType();
    mlir::Type ExpectedType = Glob.getTypes().getMLIRType(AType, &IsArray);

    LLVM_DEBUG({
      llvm::dbgs() << "AType: " << AType << "\n";
      llvm::dbgs() << "AType addrspace: "
                   << Glob.getCGM().getContext().getTargetAddressSpace(AType)
                   << "\n";
      llvm::dbgs() << "ExpectedType: " << ExpectedType << "\n";
    });

    if (auto PT = Arg.val.getType().dyn_cast<LLVM::LLVMPointerType>()) {
      if (PT.getAddressSpace() == 5)
        Arg.val = builder.create<LLVM::AddrSpaceCastOp>(
            loc, LLVM::LLVMPointerType::get(PT.getElementType(), 0), Arg.val);
    }

    mlir::Value Val = nullptr;
    if (!IsReference) {
      if (IsArray) {
        LLVM_DEBUG({
          if (!Arg.isReference) {
            Expr->dump();
            A->dump();
            llvm::errs() << " v: " << Arg.val << "\n";
          }
        });
        assert(Arg.isReference);

        auto Mt =
            Glob.getTypes()
                .getMLIRType(
                    Glob.getCGM().getContext().getLValueReferenceType(AType))
                .cast<MemRefType>();

        LLVM_DEBUG({
          llvm::dbgs() << "mt: " << Mt << "\n";
          llvm::dbgs() << "getLValueReferenceType(aType): "
                       << Glob.getCGM().getContext().getLValueReferenceType(
                              AType)
                       << "\n";
        });

        auto Shape = std::vector<int64_t>(Mt.getShape());
        assert(Shape.size() == 2);

        auto Pshape = Shape[0];
        if (Pshape == -1)
          Shape[0] = 1;

        OpBuilder Abuilder(builder.getContext());
        Abuilder.setInsertionPointToStart(allocationScope);
        auto Alloc = Abuilder.create<mlir::memref::AllocaOp>(
            loc, mlir::MemRefType::get(Shape, Mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       Mt.getMemorySpace()));
        ValueCategory(Alloc, /*isRef*/ true)
            .store(builder, Arg, /*isArray*/ IsArray);
        Shape[0] = Pshape;
        Val = builder.create<mlir::memref::CastOp>(
            loc,
            mlir::MemRefType::get(Shape, Mt.getElementType(),
                                  MemRefLayoutAttrInterface(),
                                  Mt.getMemorySpace()),
            Alloc);
      } else {
        if (FIArgs[I].info.getKind() == clang::CodeGen::ABIArgInfo::Indirect ||
            FIArgs[I].info.getKind() ==
                clang::CodeGen::ABIArgInfo::IndirectAliased) {
          OpBuilder Abuilder(builder.getContext());
          Abuilder.setInsertionPointToStart(allocationScope);
          auto Ty = Glob.getTypes().getPointerOrMemRefType(
              Arg.getValue(builder).getType(),
              Glob.getCGM().getDataLayout().getAllocaAddrSpace(),
              /*IsAlloc*/ true);
          if (auto MemRefTy = Ty.dyn_cast<mlir::MemRefType>()) {
            Val = Abuilder.create<mlir::memref::AllocaOp>(loc, MemRefTy);
            Val = Abuilder.create<mlir::memref::CastOp>(
                loc, mlir::MemRefType::get(-1, Arg.getValue(builder).getType()),
                Val);
          } else {
            Val = Abuilder.create<mlir::LLVM::AllocaOp>(
                loc, Ty, Abuilder.create<arith::ConstantIntOp>(loc, 1, 64), 0);
          }
          ValueCategory(Val, /*isRef*/ true)
              .store(builder, Arg.getValue(builder));
        } else
          Val = Arg.getValue(builder);

        if (Val.getType().isa<LLVM::LLVMPointerType>() &&
            ExpectedType.isa<MemRefType>()) {
          Val = builder.create<polygeist::Pointer2MemrefOp>(loc, ExpectedType,
                                                            Val);
        }
        if (auto PrevTy = Val.getType().dyn_cast<mlir::IntegerType>()) {
          auto IpostTy = ExpectedType.cast<mlir::IntegerType>();
          if (PrevTy != IpostTy)
            Val = builder.create<arith::TruncIOp>(loc, IpostTy, Val);
        }
      }
    } else {
      assert(Arg.isReference);

      ExpectedType = Glob.getTypes().getMLIRType(
          Glob.getCGM().getContext().getLValueReferenceType(AType));

      Val = Arg.val;
      if (Val.getType().isa<LLVM::LLVMPointerType>() &&
          ExpectedType.isa<MemRefType>())
        Val =
            builder.create<polygeist::Pointer2MemrefOp>(loc, ExpectedType, Val);

      Val = castToMemSpaceOfType(Val, ExpectedType);
    }
    assert(Val);
    Args.push_back(Val);
    I++;
  }

  // handle lowerto pragma.
  if (LTInfo.SymbolTable.count(Tocall.getName())) {
    SmallVector<mlir::Value> InputOperands;
    SmallVector<mlir::Value> OutputOperands;
    for (StringRef Input : LTInfo.InputSymbol)
      if (MapFuncOperands.find(Input.str()) != MapFuncOperands.end())
        InputOperands.push_back(MapFuncOperands[Input.str()]);
    for (StringRef Output : LTInfo.OutputSymbol)
      if (MapFuncOperands.find(Output.str()) != MapFuncOperands.end())
        OutputOperands.push_back(MapFuncOperands[Output.str()]);

    if (InputOperands.size() == 0)
      InputOperands.append(Args);

    return ValueCategory(mlirclang::replaceFuncByOperation(
                             Tocall, LTInfo.SymbolTable[Tocall.getName()],
                             builder, InputOperands, OutputOperands)
                             ->getResult(0),
                         /*isReference=*/false);
  }

  bool IsArrayReturn = false;
  if (!RetReference)
    Glob.getTypes().getMLIRType(RetType, &IsArrayReturn);

  mlir::Value Alloc;
  if (IsArrayReturn) {
    auto Mt =
        Glob.getTypes()
            .getMLIRType(
                Glob.getCGM().getContext().getLValueReferenceType(RetType))
            .cast<MemRefType>();

    auto Shape = std::vector<int64_t>(Mt.getShape());
    assert(Shape.size() == 2);

    auto Pshape = Shape[0];
    if (Pshape == -1)
      Shape[0] = 1;

    OpBuilder Abuilder(builder.getContext());
    Abuilder.setInsertionPointToStart(allocationScope);
    Alloc = Abuilder.create<mlir::memref::AllocaOp>(
        loc, mlir::MemRefType::get(Shape, Mt.getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   Mt.getMemorySpace()));
    Shape[0] = Pshape;
    Alloc = builder.create<mlir::memref::CastOp>(
        loc,
        mlir::MemRefType::get(Shape, Mt.getElementType(),
                              MemRefLayoutAttrInterface(), Mt.getMemorySpace()),
        Alloc);
    Args.push_back(Alloc);
  }

  if (auto *CU = dyn_cast<CUDAKernelCallExpr>(Expr)) {
    auto L0 = Visit(CU->getConfig()->getArg(0));
    assert(L0.isReference);
    mlir::Value Blocks[3];
    for (int I = 0; I < 3; I++) {
      mlir::Value Val = L0.val;
      if (auto MT = Val.getType().dyn_cast<MemRefType>()) {
        mlir::Value Idx[] = {getConstantIndex(0), getConstantIndex(I)};
        assert(MT.getShape().size() == 2);
        Blocks[I] = builder.create<IndexCastOp>(
            loc, mlir::IndexType::get(builder.getContext()),
            builder.create<mlir::memref::LoadOp>(loc, Val, Idx));
      } else {
        mlir::Value Idx[] = {builder.create<arith::ConstantIntOp>(loc, 0, 32),
                             builder.create<arith::ConstantIntOp>(loc, I, 32)};
        auto PT = Val.getType().cast<LLVM::LLVMPointerType>();
        auto ET = PT.getElementType().cast<LLVM::LLVMStructType>().getBody()[I];
        Blocks[I] = builder.create<IndexCastOp>(
            loc, mlir::IndexType::get(builder.getContext()),
            builder.create<LLVM::LoadOp>(
                loc,
                builder.create<LLVM::GEPOp>(
                    loc, LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
                    Val, Idx)));
      }
    }

    auto T0 = Visit(CU->getConfig()->getArg(1));
    assert(T0.isReference);
    mlir::Value Threads[3];
    for (int I = 0; I < 3; I++) {
      mlir::Value Val = T0.val;
      if (auto MT = Val.getType().dyn_cast<MemRefType>()) {
        mlir::Value Idx[] = {getConstantIndex(0), getConstantIndex(I)};
        assert(MT.getShape().size() == 2);
        Threads[I] = builder.create<IndexCastOp>(
            loc, mlir::IndexType::get(builder.getContext()),
            builder.create<mlir::memref::LoadOp>(loc, Val, Idx));
      } else {
        mlir::Value Idx[] = {builder.create<arith::ConstantIntOp>(loc, 0, 32),
                             builder.create<arith::ConstantIntOp>(loc, I, 32)};
        auto PT = Val.getType().cast<LLVM::LLVMPointerType>();
        auto ET = PT.getElementType().cast<LLVM::LLVMStructType>().getBody()[I];
        Threads[I] = builder.create<IndexCastOp>(
            loc, mlir::IndexType::get(builder.getContext()),
            builder.create<LLVM::LoadOp>(
                loc,
                builder.create<LLVM::GEPOp>(
                    loc, LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
                    Val, Idx)));
      }
    }
    mlir::Value Stream = nullptr;
    SmallVector<mlir::Value, 1> AsyncDependencies;
    if (3 < CU->getConfig()->getNumArgs() &&
        !isa<CXXDefaultArgExpr>(CU->getConfig()->getArg(3))) {
      Stream = Visit(CU->getConfig()->getArg(3)).getValue(builder);
      Stream = builder.create<polygeist::StreamToTokenOp>(
          loc, builder.getType<gpu::AsyncTokenType>(), Stream);
      assert(Stream);
      AsyncDependencies.push_back(Stream);
    }
    auto Op = builder.create<mlir::gpu::LaunchOp>(
        loc, Blocks[0], Blocks[1], Blocks[2], Threads[0], Threads[1],
        Threads[2],
        /*dynamic shmem size*/ nullptr,
        /*token type*/ Stream ? Stream.getType() : nullptr,
        /*dependencies*/ AsyncDependencies);
    auto Oldpoint = builder.getInsertionPoint();
    auto *Oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&Op.getRegion().front());
    builder.create<CallOp>(loc, Tocall, Args);
    builder.create<gpu::TerminatorOp>(loc);
    builder.setInsertionPoint(Oldblock, Oldpoint);
    return nullptr;
  }

  // Try to rescue some mismatched types.
  castCallerArgs(Tocall, Args, builder);

  /// Try to emit SYCL operations before creating a CallOp
  mlir::Operation *Op = EmitSYCLOps(Expr, Args);
  if (!Op)
    Op = builder.create<CallOp>(loc, Tocall, Args);

  if (IsArrayReturn) {
    // TODO remedy return
    if (RetReference)
      Expr->dump();
    assert(!RetReference);
    return ValueCategory(Alloc, /*isReference*/ true);
  }
  if (Op->getNumResults()) {
    return ValueCategory(Op->getResult(0),
                         /*isReference*/ RetReference);
  }
  return nullptr;
  llvm::errs() << "do not support indirecto call of " << Tocall << "\n";
  assert(0 && "no indirect");
}

std::pair<ValueCategory, bool>
MLIRScanner::EmitClangBuiltinCallExpr(clang::CallExpr *Expr) {
  switch (Expr->getBuiltinCallee()) {
  case clang::Builtin::BImove:
  case clang::Builtin::BImove_if_noexcept:
  case clang::Builtin::BIforward:
  case clang::Builtin::BIas_const: {
    auto V = Visit(Expr->getArg(0));
    return std::make_pair(V, true);
  }
  default:
    break;
  }
  return std::make_pair(ValueCategory(), false);
}

ValueCategory MLIRScanner::VisitCallExpr(clang::CallExpr *expr) {
  LLVM_DEBUG({
    llvm::dbgs() << "VisitCallExpr: ";
    expr->dump();
    llvm::dbgs() << "\n";
  });

  auto loc = getMLIRLocation(expr->getExprLoc());
  /*
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__shfl_up_sync") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
        builder.create<gpu::ShuffleOp>(loc, );
        assert(0 && "__shfl_up_sync unhandled");
        return nullptr;
      }
    }
  */

  auto ValEmitted = EmitGPUCallExpr(expr);
  if (ValEmitted.second)
    return ValEmitted.first;

  ValEmitted = EmitBuiltinOps(expr);
  if (ValEmitted.second)
    return ValEmitted.first;

  ValEmitted = EmitClangBuiltinCallExpr(expr);
  if (ValEmitted.second)
    return ValEmitted.first;

  if (auto *Oc = dyn_cast<CXXOperatorCallExpr>(expr)) {
    if (Oc->getOperator() == clang::OO_EqualEqual) {
      if (auto *Lhs = dyn_cast<CXXTypeidExpr>(expr->getArg(0))) {
        if (auto *Rhs = dyn_cast<CXXTypeidExpr>(expr->getArg(1))) {
          QualType LT = Lhs->isTypeOperand()
                            ? Lhs->getTypeOperand(Glob.getCGM().getContext())
                            : Lhs->getExprOperand()->getType();
          QualType RT = Rhs->isTypeOperand()
                            ? Rhs->getTypeOperand(Glob.getCGM().getContext())
                            : Rhs->getExprOperand()->getType();
          llvm::Constant *LC = Glob.getCGM().GetAddrOfRTTIDescriptor(LT);
          llvm::Constant *RC = Glob.getCGM().GetAddrOfRTTIDescriptor(RT);
          auto PostTy = Glob.getTypes()
                            .getMLIRType(expr->getType())
                            .cast<mlir::IntegerType>();
          return ValueCategory(
              builder.create<arith::ConstantIntOp>(loc, LC == RC, PostTy),
              false);
        }
      }
    }
  }

  if (auto *Oc = dyn_cast<CXXMemberCallExpr>(expr)) {
    if (auto *Lhs = dyn_cast<CXXTypeidExpr>(Oc->getImplicitObjectArgument())) {
      expr->getCallee()->dump();
      if (auto *Ic = dyn_cast<MemberExpr>(expr->getCallee()))
        if (auto *Sr = dyn_cast<NamedDecl>(Ic->getMemberDecl())) {
          if (Sr->getIdentifier() && Sr->getName() == "name") {
            QualType LT = Lhs->isTypeOperand()
                              ? Lhs->getTypeOperand(Glob.getCGM().getContext())
                              : Lhs->getExprOperand()->getType();
            llvm::Constant *LC = Glob.getCGM().GetAddrOfRTTIDescriptor(LT);
            while (auto *CE = dyn_cast<llvm::ConstantExpr>(LC))
              LC = CE->getOperand(0);
            std::string Val = cast<llvm::GlobalVariable>(LC)->getName().str();
            return CommonArrayToPointer(ValueCategory(
                Glob.GetOrCreateGlobalLLVMString(loc, builder, Val),
                /*isReference*/ true));
          }
        }
    }
  }

  if (auto *Ps = dyn_cast<CXXPseudoDestructorExpr>(expr->getCallee()))
    return Visit(Ps);

  if (auto *Ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *Sr = dyn_cast<DeclRefExpr>(Ic->getSubExpr())) {
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "atomicAdd" ||
           Sr->getDecl()->getName() == "atomicOr" ||
           Sr->getDecl()->getName() == "atomicAnd")) {
        std::vector<ValueCategory> args;
        for (auto *a : expr->arguments()) {
          args.push_back(Visit(a));
        }
        auto A0 = args[0].getValue(builder);
        auto A1 = args[1].getValue(builder);
        AtomicRMWKind Op;
        LLVM::AtomicBinOp Lop;
        if (Sr->getDecl()->getName() == "atomicAdd") {
          if (A1.getType().isa<mlir::IntegerType>()) {
            Op = AtomicRMWKind::addi;
            Lop = LLVM::AtomicBinOp::add;
          } else {
            Op = AtomicRMWKind::addf;
            Lop = LLVM::AtomicBinOp::fadd;
          }
        } else if (Sr->getDecl()->getName() == "atomicOr") {
          Op = AtomicRMWKind::ori;
          Lop = LLVM::AtomicBinOp::_or;
        } else if (Sr->getDecl()->getName() == "atomicAnd") {
          Op = AtomicRMWKind::andi;
          Lop = LLVM::AtomicBinOp::_and;
        } else
          assert(0);

        if (A0.getType().isa<MemRefType>())
          return ValueCategory(
              builder.create<memref::AtomicRMWOp>(
                  loc, A1.getType(), Op, A1, A0,
                  std::vector<mlir::Value>({getConstantIndex(0)})),
              /*isReference*/ false);
        return ValueCategory(
            builder.create<LLVM::AtomicRMWOp>(loc, A1.getType(), Lop, A0, A1,
                                              LLVM::AtomicOrdering::acq_rel),
            /*isReference*/ false);
      }
    }

  mlir::LLVM::TypeFromLLVMIRTranslator TypeTranslator(*module->getContext());

  auto GetLlvm = [&](Expr *E) -> mlir::Value {
    auto Sub = Visit(E);
    if (!Sub.val) {
      expr->dump();
      E->dump();
    }
    assert(Sub.val);

    bool IsReference = E->isLValue() || E->isXValue();
    if (IsReference) {
      assert(Sub.isReference);
      mlir::Value Val = Sub.val;
      if (auto Mt = Val.getType().dyn_cast<MemRefType>()) {
        Val = builder.create<polygeist::Memref2PointerOp>(
            loc,
            LLVM::LLVMPointerType::get(Mt.getElementType(),
                                       Mt.getMemorySpaceAsInt()),
            Val);
      }
      return Val;
    }

    bool IsArray = false;
    Glob.getTypes().getMLIRType(E->getType(), &IsArray);

    if (IsArray) {
      assert(Sub.isReference);
      auto Mt =
          Glob.getTypes()
              .getMLIRType(Glob.getCGM().getContext().getLValueReferenceType(
                  E->getType()))
              .cast<MemRefType>();
      auto Shape = std::vector<int64_t>(Mt.getShape());
      assert(Shape.size() == 2);

      OpBuilder Abuilder(builder.getContext());
      Abuilder.setInsertionPointToStart(allocationScope);
      auto One = Abuilder.create<ConstantIntOp>(loc, 1, 64);
      auto Alloc = Abuilder.create<mlir::LLVM::AllocaOp>(
          loc,
          LLVM::LLVMPointerType::get(
              TypeTranslator.translateType(
                  anonymize(getLLVMType(E->getType(), Glob.getCGM()))),
              0),
          One, 0);
      ValueCategory(Alloc, /*isRef*/ true)
          .store(builder, Sub, /*isArray*/ IsArray);
      Sub = ValueCategory(Alloc, /*isRef*/ true);
    }
    auto Val = Sub.getValue(builder);
    if (auto mt = Val.getType().dyn_cast<MemRefType>()) {
      auto Nt = TypeTranslator
                    .translateType(
                        anonymize(getLLVMType(E->getType(), Glob.getCGM())))
                    .cast<LLVM::LLVMPointerType>();
      assert(Nt.getAddressSpace() == mt.getMemorySpaceAsInt() &&
             "val does not have the same memory space as nt");
      Val = builder.create<polygeist::Memref2PointerOp>(loc, Nt, Val);
    }
    return Val;
  };

  switch (expr->getBuiltinCallee()) {
  case Builtin::BI__builtin_strlen:
  case Builtin::BIstrlen: {
    mlir::Value V0 = GetLlvm(expr->getArg(0));

    const std::string Name("strlen");

    if (Glob.getFunctions().find(Name) == Glob.getFunctions().end()) {
      std::vector<mlir::Type> Types{V0.getType()};

      mlir::Type RT = Glob.getTypes().getMLIRType(expr->getType());
      std::vector<mlir::Type> RetTypes{RT};
      mlir::OpBuilder Builder(module->getContext());
      auto FuncType = Builder.getFunctionType(Types, RetTypes);
      Glob.getFunctions()[Name] = mlir::func::FuncOp(
          mlir::func::FuncOp::create(builder.getUnknownLoc(), Name, FuncType));
      SymbolTable::setSymbolVisibility(Glob.getFunctions()[Name],
                                       SymbolTable::Visibility::Private);
      module->push_back(Glob.getFunctions()[Name]);
    }

    mlir::Value vals[] = {V0};
    return ValueCategory(
        builder.create<CallOp>(loc, Glob.getFunctions()[Name], vals)
            .getResult(0),
        false);
  }
  case Builtin::BI__builtin_isnormal: {
    mlir::Value V = GetLlvm(expr->getArg(0));
    auto Ty = V.getType().cast<mlir::FloatType>();
    mlir::Value Eq = builder.create<CmpFOp>(loc, CmpFPredicate::OEQ, V, V);

    mlir::Value Abs = builder.create<math::AbsFOp>(loc, V);
    auto Infinity = builder.create<ConstantFloatOp>(
        loc, APFloat::getInf(Ty.getFloatSemantics()), Ty);
    mlir::Value IsLessThanInf =
        builder.create<CmpFOp>(loc, CmpFPredicate::ULT, Abs, Infinity);
    APFloat Smallest = APFloat::getSmallestNormalized(Ty.getFloatSemantics());
    auto SmallestV = builder.create<ConstantFloatOp>(loc, Smallest, Ty);
    mlir::Value IsNormal =
        builder.create<CmpFOp>(loc, CmpFPredicate::UGE, Abs, SmallestV);
    V = builder.create<AndIOp>(loc, Eq, IsLessThanInf);
    V = builder.create<AndIOp>(loc, V, IsNormal);
    auto PostTy =
        Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    mlir::Value Res = builder.create<ExtUIOp>(loc, PostTy, V);
    return ValueCategory(Res, /*isRef*/ false);
  }
  case Builtin::BI__builtin_signbit: {
    mlir::Value V = GetLlvm(expr->getArg(0));
    auto Ty = V.getType().cast<mlir::FloatType>();
    auto ITy = builder.getIntegerType(Ty.getWidth());
    mlir::Value BC = builder.create<BitcastOp>(loc, ITy, V);
    auto ZeroV = builder.create<ConstantIntOp>(loc, 0, ITy);
    V = builder.create<CmpIOp>(loc, CmpIPredicate::slt, BC, ZeroV);
    auto PostTy =
        Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    mlir::Value Res = builder.create<ExtUIOp>(loc, PostTy, V);
    return ValueCategory(Res, /*isRef*/ false);
  }
  case Builtin::BI__builtin_addressof: {
    auto V = Visit(expr->getArg(0));
    assert(V.isReference);
    mlir::Value Val = V.val;
    mlir::Type T = Glob.getTypes().getMLIRType(expr->getType());
    if (T == Val.getType())
      return ValueCategory(Val, /*isRef*/ false);
    if (T.isa<LLVM::LLVMPointerType>()) {
      if (Val.getType().isa<MemRefType>()) {
        assert(Val.getType().cast<MemRefType>().getMemorySpaceAsInt() ==
                   T.cast<LLVM::LLVMPointerType>().getAddressSpace() &&
               "val does not have the same memory space as T");
        Val = builder.create<polygeist::Memref2PointerOp>(loc, T, Val);
      } else if (T != Val.getType())
        Val = builder.create<LLVM::BitcastOp>(loc, T, Val);
      return ValueCategory(Val, /*isRef*/ false);
    }
    assert(T.isa<MemRefType>());

    if (Val.getType().isa<MemRefType>())
      Val = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(
              builder.getI8Type(),
              Val.getType().cast<MemRefType>().getMemorySpaceAsInt()),
          Val);
    if (Val.getType().isa<LLVM::LLVMPointerType>())
      Val = builder.create<polygeist::Pointer2MemrefOp>(loc, T, Val);
    return ValueCategory(Val, /*isRef*/ false);

    expr->dump();
    llvm::errs() << " val: " << Val << " T: " << T << "\n";
    assert(0 && "unhandled builtin addressof");
  }
  }

  auto BuiltinCallee = expr->getBuiltinCallee();

  if (auto *Ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *Sr = dyn_cast<DeclRefExpr>(Ic->getSubExpr())) {
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "__powf" ||
           BuiltinCallee == Builtin::BIpow ||
           Sr->getDecl()->getName() == "__nv_pow" ||
           Sr->getDecl()->getName() == "__nv_powf" ||
           Sr->getDecl()->getName() == "__powi" ||
           Sr->getDecl()->getName() == "powi" ||
           Sr->getDecl()->getName() == "__nv_powi" ||
           Sr->getDecl()->getName() == "__nv_powi" ||
           BuiltinCallee == Builtin::BIpowf)) {
        mlir::Type MlirType = Glob.getTypes().getMLIRType(expr->getType());
        std::vector<mlir::Value> Args;
        for (auto *A : expr->arguments()) {
          Args.push_back(Visit(A).getValue(builder));
        }
        if (Args[1].getType().isa<mlir::IntegerType>())
          return ValueCategory(
              builder.create<LLVM::PowIOp>(loc, MlirType, Args[0], Args[1]),
              /*isReference*/ false);
        return ValueCategory(
            builder.create<math::PowFOp>(loc, MlirType, Args[0], Args[1]),
            /*isReference*/ false);
      }
    }

  if (auto *Ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *Sr = dyn_cast<DeclRefExpr>(Ic->getSubExpr())) {
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "__nv_fabsf" ||
           Sr->getDecl()->getName() == "__nv_fabs" ||
           Sr->getDecl()->getName() == "__nv_abs" ||
           BuiltinCallee == Builtin::BIfabs ||
           BuiltinCallee == Builtin::BIfabsf ||
           BuiltinCallee == Builtin::BI__builtin_fabs ||
           BuiltinCallee == Builtin::BI__builtin_fabsf)) {
        // isinf(x)    --> fabs(x) == infinity
        // isfinite(x) --> fabs(x) != infinity
        // x != NaN via the ordered compare in either case.
        mlir::Value V = GetLlvm(expr->getArg(0));
        mlir::Value Fabs;
        if (V.getType().isa<mlir::FloatType>())
          Fabs = builder.create<math::AbsFOp>(loc, V);
        else {
          auto Zero = builder.create<arith::ConstantIntOp>(
              loc, 0, V.getType().cast<mlir::IntegerType>().getWidth());
          Fabs = builder.create<SelectOp>(
              loc,
              builder.create<arith::CmpIOp>(loc, CmpIPredicate::sge, V, Zero),
              V, builder.create<arith::SubIOp>(loc, Zero, V));
        }
        return ValueCategory(Fabs, /*isRef*/ false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          Sr->getDecl()->getName() == "__nv_mul24") {
        mlir::Value V0 = GetLlvm(expr->getArg(0));
        mlir::Value V1 = GetLlvm(expr->getArg(1));
        auto C8 = builder.create<arith::ConstantIntOp>(loc, 8, 32);
        V0 = builder.create<arith::ShLIOp>(loc, V0, C8);
        V0 = builder.create<arith::ShRUIOp>(loc, V0, C8);
        V1 = builder.create<arith::ShLIOp>(loc, V1, C8);
        V1 = builder.create<arith::ShRUIOp>(loc, V1, C8);
        return ValueCategory(builder.create<MulIOp>(loc, V0, V1), false);
      }
      if (BuiltinCallee == Builtin::BI__builtin_frexp ||
          BuiltinCallee == Builtin::BI__builtin_frexpf ||
          BuiltinCallee == Builtin::BI__builtin_frexpl ||
          BuiltinCallee == Builtin::BI__builtin_frexpf128) {
        mlir::Value V0 = GetLlvm(expr->getArg(0));
        mlir::Value V1 = GetLlvm(expr->getArg(1));

        auto Name = Sr->getDecl()
                        ->getName()
                        .substr(std::string("__builtin_").length())
                        .str();

        if (Glob.getFunctions().find(Name) == Glob.getFunctions().end()) {
          std::vector<mlir::Type> Types{V0.getType(), V1.getType()};

          mlir::Type RT = Glob.getTypes().getMLIRType(expr->getType());
          std::vector<mlir::Type> Rettype{RT};
          mlir::OpBuilder Mbuilder(module->getContext());
          auto FuncType = Mbuilder.getFunctionType(Types, Rettype);
          Glob.getFunctions()[Name] =
              mlir::func::FuncOp(mlir::func::FuncOp::create(
                  builder.getUnknownLoc(), Name, FuncType));
          SymbolTable::setSymbolVisibility(Glob.getFunctions()[Name],
                                           SymbolTable::Visibility::Private);
          module->push_back(Glob.getFunctions()[Name]);
        }

        mlir::Value Vals[] = {V0, V1};
        return ValueCategory(
            builder.create<CallOp>(loc, Glob.getFunctions()[Name], Vals)
                .getResult(0),
            false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (BuiltinCallee == Builtin::BI__builtin_isfinite ||
           BuiltinCallee == Builtin::BI__builtin_isinf ||
           Sr->getDecl()->getName() == "__nv_isinff")) {
        // isinf(x)    --> fabs(x) == infinity
        // isfinite(x) --> fabs(x) != infinity
        // x != NaN via the ordered compare in either case.
        mlir::Value V = GetLlvm(expr->getArg(0));
        auto Ty = V.getType().cast<mlir::FloatType>();
        mlir::Value Fabs = builder.create<math::AbsFOp>(loc, V);
        auto Infinity = builder.create<ConstantFloatOp>(
            loc, APFloat::getInf(Ty.getFloatSemantics()), Ty);
        auto Pred = (BuiltinCallee == Builtin::BI__builtin_isinf ||
                     Sr->getDecl()->getName() == "__nv_isinff")
                        ? CmpFPredicate::OEQ
                        : CmpFPredicate::ONE;
        mlir::Value FCmp = builder.create<CmpFOp>(loc, Pred, Fabs, Infinity);
        auto PostTy = Glob.getTypes()
                          .getMLIRType(expr->getType())
                          .cast<mlir::IntegerType>();
        mlir::Value Res = builder.create<ExtUIOp>(loc, PostTy, FCmp);
        return ValueCategory(Res, /*isRef*/ false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (BuiltinCallee == Builtin::BI__builtin_isnan ||
           Sr->getDecl()->getName() == "__nv_isnanf")) {
        mlir::Value V = GetLlvm(expr->getArg(0));
        mlir::Value Eq = builder.create<CmpFOp>(loc, CmpFPredicate::UNO, V, V);
        auto PostTy = Glob.getTypes()
                          .getMLIRType(expr->getType())
                          .cast<mlir::IntegerType>();
        mlir::Value Res = builder.create<ExtUIOp>(loc, PostTy, Eq);
        return ValueCategory(Res, /*isRef*/ false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "__nv_fmodf")) {
        mlir::Value V = GetLlvm(expr->getArg(0));
        mlir::Value V2 = GetLlvm(expr->getArg(1));
        V = builder.create<mlir::LLVM::FRemOp>(loc, V.getType(), V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "__nv_scalbn" ||
           Sr->getDecl()->getName() == "__nv_scalbnf" ||
           Sr->getDecl()->getName() == "__nv_scalbnl")) {
        mlir::Value V = GetLlvm(expr->getArg(0));
        mlir::Value V2 = GetLlvm(expr->getArg(1));
        auto Name = Sr->getDecl()->getName().substr(5).str();
        std::vector<mlir::Type> Types{V.getType(), V2.getType()};
        mlir::Type RT = Glob.getTypes().getMLIRType(expr->getType());

        std::vector<mlir::Type> Rettypes{RT};

        mlir::OpBuilder Mbuilder(module->getContext());
        auto FuncType = Mbuilder.getFunctionType(Types, Rettypes);
        mlir::func::FuncOp Function =
            mlir::func::FuncOp(mlir::func::FuncOp::create(
                builder.getUnknownLoc(), Name, FuncType));
        SymbolTable::setSymbolVisibility(Function,
                                         SymbolTable::Visibility::Private);

        Glob.getFunctions()[Name] = Function;
        module->push_back(Function);
        mlir::Value Vals[] = {V, V2};
        V = builder.create<CallOp>(loc, Function, Vals).getResult(0);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "__nv_dmul_rn")) {
        mlir::Value V = GetLlvm(expr->getArg(0));
        mlir::Value V2 = GetLlvm(expr->getArg(1));
        V = builder.create<MulFOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "__nv_dadd_rn")) {
        mlir::Value V = GetLlvm(expr->getArg(0));
        mlir::Value V2 = GetLlvm(expr->getArg(1));
        V = builder.create<AddFOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "__nv_dsub_rn")) {
        mlir::Value V = GetLlvm(expr->getArg(0));
        mlir::Value V2 = GetLlvm(expr->getArg(1));
        V = builder.create<SubFOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (BuiltinCallee == Builtin::BI__builtin_log2 ||
           BuiltinCallee == Builtin::BI__builtin_log2f ||
           BuiltinCallee == Builtin::BI__builtin_log2l ||
           Sr->getDecl()->getName() == "__nv_log2" ||
           Sr->getDecl()->getName() == "__nv_log2f" ||
           Sr->getDecl()->getName() == "__nv_log2l")) {
        mlir::Value V = GetLlvm(expr->getArg(0));
        V = builder.create<math::Log2Op>(loc, V);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (BuiltinCallee == Builtin::BI__builtin_log10 ||
           BuiltinCallee == Builtin::BI__builtin_log10f ||
           BuiltinCallee == Builtin::BI__builtin_log10l ||
           Sr->getDecl()->getName() == "__nv_log10" ||
           Sr->getDecl()->getName() == "__nv_log10f" ||
           Sr->getDecl()->getName() == "__nv_log10l")) {
        mlir::Value V = GetLlvm(expr->getArg(0));
        V = builder.create<math::Log10Op>(loc, V);
        return ValueCategory(V, /*isRef*/ false);
      }
    }

  if (auto *Ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *Sr = dyn_cast<DeclRefExpr>(Ic->getSubExpr())) {
      if ((Sr->getDecl()->getIdentifier() &&
           (BuiltinCallee == Builtin::BIfscanf ||
            BuiltinCallee == Builtin::BIscanf ||
            Sr->getDecl()->getName() == "__isoc99_sscanf" ||
            BuiltinCallee == Builtin::BIsscanf)) ||
          (isa<CXXOperatorCallExpr>(expr) &&
           cast<CXXOperatorCallExpr>(expr)->getOperator() ==
               OO_GreaterGreater)) {
        const auto *Tocall = EmitCallee(expr->getCallee());
        auto StrcmpF = Glob.GetOrCreateLLVMFunction(Tocall);

        std::vector<mlir::Value> Args;
        std::vector<std::pair<mlir::Value, mlir::Value>> Ops;
        std::map<const void *, size_t> Counts;
        for (auto *A : expr->arguments()) {
          auto V = GetLlvm(A);
          if (auto Toptr = V.getDefiningOp<polygeist::Memref2PointerOp>()) {
            auto T = Toptr.getType().cast<LLVM::LLVMPointerType>();
            auto Idx = Counts[T.getAsOpaquePointer()]++;
            auto Aop = allocateBuffer(Idx, T);
            Args.push_back(Aop.getResult());
            Ops.emplace_back(Aop.getResult(), Toptr.getSource());
          } else
            Args.push_back(V);
        }
        auto Called = builder.create<mlir::LLVM::CallOp>(loc, StrcmpF, Args);
        for (auto Pair : Ops) {
          auto Lop = builder.create<mlir::LLVM::LoadOp>(loc, Pair.first);
          builder.create<mlir::memref::StoreOp>(
              loc, Lop, Pair.second,
              std::vector<mlir::Value>({getConstantIndex(0)}));
        }
        return ValueCategory(Called.getResult(), /*isReference*/ false);
      }
    }

  if (auto *Ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *Sr = dyn_cast<DeclRefExpr>(Ic->getSubExpr())) {
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "cudaMemcpy" ||
           Sr->getDecl()->getName() == "cudaMemcpyAsync" ||
           Sr->getDecl()->getName() == "cudaMemcpyToSymbol" ||
           BuiltinCallee == Builtin::BImemcpy ||
           BuiltinCallee == Builtin::BI__builtin_memcpy)) {
        auto *DstSub = expr->getArg(0);
        while (auto *BC = dyn_cast<clang::CastExpr>(DstSub))
          DstSub = BC->getSubExpr();
        auto *SrcSub = expr->getArg(1);
        while (auto *BC = dyn_cast<clang::CastExpr>(SrcSub))
          SrcSub = BC->getSubExpr();

#if 0
        auto dstst = dstSub->getType()->getUnqualifiedDesugaredType();
        if (isa<clang::PointerType>(dstst) || isa<clang::ArrayType>(dstst)) {

          auto elem = isa<clang::PointerType>(dstst)
                          ? cast<clang::PointerType>(dstst)
                                ->getPointeeType()
                                ->getUnqualifiedDesugaredType()

                          : cast<clang::ArrayType>(dstst)
                                ->getElementType()
                                ->getUnqualifiedDesugaredType();
          auto melem = elem;
          if (auto BC = dyn_cast<clang::ArrayType>(melem))
            melem = BC->getElementType()->getUnqualifiedDesugaredType();

          auto srcst = srcSub->getType()->getUnqualifiedDesugaredType();
          auto selem = isa<clang::PointerType>(srcst)
                           ? cast<clang::PointerType>(srcst)
                                 ->getPointeeType()
                                 ->getUnqualifiedDesugaredType()

                           : cast<clang::ArrayType>(srcst)
                                 ->getElementType()
                                 ->getUnqualifiedDesugaredType();

          auto mselem = selem;
          if (auto BC = dyn_cast<clang::ArrayType>(mselem))
            mselem = BC->getElementType()->getUnqualifiedDesugaredType();

          if (melem == mselem) {
            mlir::Value dst;
            ValueCategory vdst = Visit(dstSub);
            if (isa<clang::PointerType>(dstst)) {
              dst = vdst.getValue(builder);
            } else {
              assert(vdst.isReference);
              dst = vdst.val;
            }
            // if (dst.getType().isa<MemRefType>())
            {
              mlir::Value src;
              ValueCategory vsrc = Visit(srcSub);
              if (isa<clang::PointerType>(srcst)) {
                src = vsrc.getValue(builder);
              } else {
                assert(vsrc.isReference);
                src = vsrc.val;
              }

              bool dstArray = false;
              Glob.getTypes().getMLIRType(QualType(elem, 0), &dstArray);
              bool srcArray = false;
              Glob.getTypes().getMLIRType(QualType(selem, 0), &srcArray);
              auto elemSize = getTypeSize(QualType(elem, 0));
              if (srcArray && !dstArray)
                elemSize = getTypeSize(QualType(selem, 0));
              mlir::Value size = builder.create<IndexCastOp>(
                  loc, Visit(expr->getArg(2)).getValue(builder),
                  mlir::IndexType::get(builder.getContext()));
              size = builder.create<DivUIOp>(
                  loc, size, builder.create<ConstantIndexOp>(loc, elemSize));

              if (sr->getDecl()->getName() == "cudaMemcpyToSymbol") {
                mlir::Value offset = Visit(expr->getArg(3)).getValue(builder);
                offset = builder.create<IndexCastOp>(
                    loc, offset, mlir::IndexType::get(builder.getContext()));
                offset = builder.create<DivUIOp>(
                    loc, offset,
                    builder.create<ConstantIndexOp>(loc, elemSize));
                // assert(!dstArray);
                if (auto mt = dst.getType().dyn_cast<MemRefType>()) {
                  auto shape = std::vector<int64_t>(mt.getShape());
                  shape[0] = -1;
                  auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                                   MemRefLayoutAttrInterface(),
                                                   mt.getMemorySpace());
                  dst = builder.create<polygeist::SubIndexOp>(loc, mt0, dst,
                                                              offset);
                } else {
                  mlir::Value idxs[] = {offset};
                  dst = builder.create<LLVM::GEPOp>(loc, dst.getType(), dst,
                                                    idxs);
                }
              }

              auto affineOp = builder.create<scf::ForOp>(
                  loc, getConstantIndex(0), size, getConstantIndex(1));

              auto oldpoint = builder.getInsertionPoint();
              auto oldblock = builder.getInsertionBlock();

              std::vector<mlir::Value> dstargs = {affineOp.getInductionVar()};
              std::vector<mlir::Value> srcargs = {affineOp.getInductionVar()};

              builder.setInsertionPointToStart(&affineOp.getLoopBody().front());

              if (dstArray) {
                std::vector<mlir::Value> start = {getConstantIndex(0)};
                auto mt = Glob.getTypes().getMLIRType(Glob.getCGM().getContext().getPointerType(
                                               QualType(elem, 0)))
                              .cast<MemRefType>();
                auto shape = std::vector<int64_t>(mt.getShape());
                assert(shape.size() > 0 && shape.back() != -1);
                auto affineOp = builder.create<scf::ForOp>(
                    loc, getConstantIndex(0), getConstantIndex(shape.back()),
                    getConstantIndex(1));
                dstargs.push_back(affineOp.getInductionVar());
                builder.setInsertionPointToStart(
                    &affineOp.getLoopBody().front());
                if (srcArray) {
                  auto smt =
                      Glob.getTypes().getMLIRType(Glob.getCGM().getContext().getPointerType(
                                           QualType(elem, 0)))
                          .cast<MemRefType>();
                  auto sshape = std::vector<int64_t>(smt.getShape());
                  assert(sshape.size() > 0 && sshape.back() != -1);
                  assert(sshape.back() == shape.back());
                  srcargs.push_back(affineOp.getInductionVar());
                } else {
                  srcargs[0] = builder.create<AddIOp>(
                      loc,
                      builder.create<MulIOp>(loc, srcargs[0],
                                             getConstantIndex(shape.back())),
                      affineOp.getInductionVar());
                }
              } else {
                if (srcArray) {
                  auto smt =
                      Glob.getTypes().getMLIRType(Glob.getCGM().getContext().getPointerType(
                                           QualType(selem, 0)))
                          .cast<MemRefType>();
                  auto sshape = std::vector<int64_t>(smt.getShape());
                  assert(sshape.size() > 0 && sshape.back() != -1);
                  auto affineOp = builder.create<scf::ForOp>(
                      loc, getConstantIndex(0), getConstantIndex(sshape.back()),
                      getConstantIndex(1));
                  srcargs.push_back(affineOp.getInductionVar());
                  builder.setInsertionPointToStart(
                      &affineOp.getLoopBody().front());
                  dstargs[0] = builder.create<AddIOp>(
                      loc,
                      builder.create<MulIOp>(loc, dstargs[0],
                                             getConstantIndex(sshape.back())),
                      affineOp.getInductionVar());
                }
              }

              mlir::Value loaded;
              if (src.getType().isa<MemRefType>())
                loaded = builder.create<memref::LoadOp>(loc, src, srcargs);
              else {
                auto opt = src.getType().cast<LLVM::LLVMPointerType>();
                auto elty = LLVM::LLVMPointerType::get(opt.getElementType(),
                                                       opt.getAddressSpace());
                for (auto &val : srcargs) {
                  val = builder.create<IndexCastOp>(val.getLoc(), val,
                                                    builder.getI32Type());
                }
                loaded = builder.create<LLVM::LoadOp>(
                    loc, builder.create<LLVM::GEPOp>(loc, elty, src, srcargs));
              }
              if (dst.getType().isa<MemRefType>()) {
                builder.create<memref::StoreOp>(loc, loaded, dst, dstargs);
              } else {
                auto opt = dst.getType().cast<LLVM::LLVMPointerType>();
                auto elty = LLVM::LLVMPointerType::get(opt.getElementType(),
                                                       opt.getAddressSpace());
                for (auto &val : dstargs) {
                  val = builder.create<IndexCastOp>(val.getLoc(), val,
                                                    builder.getI32Type());
                }
                builder.create<LLVM::StoreOp>(
                    loc, loaded,
                    builder.create<LLVM::GEPOp>(loc, elty, dst, dstargs));
              }

              // TODO: set the value of the iteration value to the final bound
              // at the end of the loop.
              builder.setInsertionPoint(oldblock, oldpoint);

              mlir::Type retTy = Glob.getTypes().getMLIRType(expr->getType());
              if (BuiltinCallee == Builtin::BI__builtin_memcpy ||
                  retTy.isa<LLVM::LLVMPointerType>()) {
                if (dst.getType().isa<MemRefType>()) {
                  auto mt = dst.getType().cast<MemRefType>();
                  assert(retTy.getAddressSpace() == mt.getMemorySpaceAsInt() &&
                    "dst does not have the same memory space as retTy");
                  dst = builder.create<polygeist::Memref2PointerOp>(loc, retTy,
                                                                    dst);
                }
                else
                  dst = builder.create<LLVM::BitcastOp>(loc, retTy, dst);
                if (dst.getType() != retTy) {
                    expr->dump();
                    llvm::errs() << " retTy: " << retTy << " dst: " << dst << "\n";
                }
                assert(dst.getType() == retTy);
                return ValueCategory(dst, /*isReference*/ false);
              } else {
                if (!retTy.isa<mlir::IntegerType>()) {
                  expr->dump();
                  llvm::errs() << " retTy: " << retTy << "\n";
                }
                return ValueCategory(
                    builder.create<ConstantIntOp>(loc, 0, retTy),
                    /*isReference*/ false);
              }
            }
          }
          /*
          function.dump();
          expr->dump();
          dstSub->dump();
          elem->dump();
          srcSub->dump();
          mselem->dump();
          assert(0 && "unhandled cudaMemcpy");
          */
        }
#endif
      }
    }

#if 0
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "cudaMemset")) {
        if (auto IL = dyn_cast<clang::IntegerLiteral>(expr->getArg(1)))
          if (IL->getValue() == 0) {
            auto dstSub = expr->getArg(0);
            while (auto BC = dyn_cast<clang::CastExpr>(dstSub))
              dstSub = BC->getSubExpr();

            auto dstst = dstSub->getType()->getUnqualifiedDesugaredType();
            auto elem = isa<clang::PointerType>(dstst)
                            ? cast<clang::PointerType>(dstst)->getPointeeType()
                            : cast<clang::ArrayType>(dstst)->getElementType();
            mlir::Value dst;
            ValueCategory vdst = Visit(dstSub);
            if (isa<clang::PointerType>(dstst)) {
              dst = vdst.getValue(builder);
            } else {
              assert(vdst.isReference);
              dst = vdst.val;
            }
            if (dst.getType().isa<MemRefType>()) {

              bool dstArray = false;
              mlir::Type melem = Glob.getTypes().getMLIRType(elem, &dstArray);
              mlir::Value toStore;
              if (melem.isa<mlir::IntegerType>())
                toStore = builder.create<ConstantIntOp>(loc, 0, melem);
              else {
                auto ft = melem.cast<FloatType>();
                toStore = builder.create<ConstantFloatOp>(
                    loc, APFloat(ft.getFloatSemantics(), "0"), ft);
              }

              auto elemSize = getTypeSize(elem);
              mlir::Value size = builder.create<IndexCastOp>(
                  loc, Visit(expr->getArg(2)).getValue(builder),
                  mlir::IndexType::get(builder.getContext()));
              size = builder.create<DivUIOp>(
                  loc, size, builder.create<ConstantIndexOp>(loc, elemSize));

              auto affineOp = builder.create<scf::ForOp>(
                  loc, getConstantIndex(0), size, getConstantIndex(1));

              auto oldpoint = builder.getInsertionPoint();
              auto oldblock = builder.getInsertionBlock();

              std::vector<mlir::Value> args = {affineOp.getInductionVar()};

              builder.setInsertionPointToStart(&affineOp.getLoopBody().front());

              if (dstArray) {
                std::vector<mlir::Value> start = {getConstantIndex(0)};
                auto mt =
                    Glob.getTypes().getMLIRType(Glob.getCGM().getContext().getPointerType(elem))
                        .cast<MemRefType>();
                auto shape = std::vector<int64_t>(mt.getShape());
                auto affineOp = builder.create<scf::ForOp>(
                    loc, getConstantIndex(0), getConstantIndex(shape[1]),
                    getConstantIndex(1));
                args.push_back(affineOp.getInductionVar());
                builder.setInsertionPointToStart(
                    &affineOp.getLoopBody().front());
              }

              builder.create<memref::StoreOp>(loc, toStore, dst, args);

              // TODO: set the value of the iteration value to the final bound
              // at the end of the loop.
              builder.setInsertionPoint(oldblock, oldpoint);

              mlir::Type retTy = Glob.getTypes().getMLIRType(expr->getType());
              return ValueCategory(builder.create<ConstantIntOp>(loc, 0, retTy),
                                   /*isReference*/ false);
            }
          }
      }
    }
#endif

  const FunctionDecl *Callee = EmitCallee(expr->getCallee());

  std::set<std::string> Funcs = {
      "fread",
      "read",
      "strcmp",
      "fputs",
      "puts",
      "memcpy",
      "getenv",
      "strrchr",
      "mkdir",
      "printf",
      "fprintf",
      "sprintf",
      "fwrite",
      "__builtin_memcpy",
      "cudaMemcpy",
      "cudaMemcpyAsync",
      "cudaMalloc",
      "cudaMallocHost",
      "cudaFree",
      "cudaFreeHost",
      "open",
      "gettimeofday",
      "fopen",
      "time",
      "memset",
      "cudaMemset",
      "strcpy",
      "close",
      "fclose",
      "atoi",
      "malloc",
      "calloc",
      "free",
      "fgets",
      "__errno_location",
      "__assert_fail",
      "cudaEventElapsedTime",
      "cudaEventSynchronize",
      "cudaDeviceGetAttribute",
      "cudaFuncGetAttributes",
      "cudaGetDevice",
      "cudaGetDeviceCount",
      "cudaMemGetInfo",
      "clock_gettime",
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
      "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
      "cudaEventRecord"};
  if (auto *Ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *Sr = dyn_cast<DeclRefExpr>(Ic->getSubExpr())) {
      StringRef Name;
      if (auto *CC = dyn_cast<CXXConstructorDecl>(Sr->getDecl()))
        Name = Glob.getCGM().getMangledName(
            GlobalDecl(CC, CXXCtorType::Ctor_Complete));
      else if (auto *CC = dyn_cast<CXXDestructorDecl>(Sr->getDecl()))
        Name = Glob.getCGM().getMangledName(
            GlobalDecl(CC, CXXDtorType::Dtor_Complete));
      else if (Sr->getDecl()->hasAttr<CUDAGlobalAttr>())
        Name = Glob.getCGM().getMangledName(GlobalDecl(
            cast<FunctionDecl>(Sr->getDecl()), KernelReferenceKind::Kernel));
      else
        Name = Glob.getCGM().getMangledName(Sr->getDecl());
      if (Funcs.count(Name.str()) || Name.startswith("mkl_") ||
          Name.startswith("MKL_") || Name.startswith("cublas") ||
          Name.startswith("cblas_")) {

        std::vector<mlir::Value> Args;
        for (auto *A : expr->arguments()) {
          Args.push_back(GetLlvm(A));
        }
        mlir::Value Called;

        if (Callee) {
          auto StrcmpF = Glob.GetOrCreateLLVMFunction(Callee);
          Called = builder.create<mlir::LLVM::CallOp>(loc, StrcmpF, Args)
                       .getResult();
        } else {
          Args.insert(Args.begin(), GetLlvm(expr->getCallee()));
          SmallVector<mlir::Type> RTs = {TypeTranslator.translateType(
              anonymize(getLLVMType(expr->getType(), Glob.getCGM())))};
          if (RTs[0].isa<LLVM::LLVMVoidType>())
            RTs.clear();
          Called =
              builder.create<mlir::LLVM::CallOp>(loc, RTs, Args).getResult();
        }
        return ValueCategory(Called, /*isReference*/ expr->isLValue() ||
                                         expr->isXValue());
      }
    }

  if (!Callee || Callee->isVariadic()) {
    bool IsReference = expr->isLValue() || expr->isXValue();
    std::vector<mlir::Value> Args;
    for (auto *A : expr->arguments()) {
      Args.push_back(GetLlvm(A));
    }
    mlir::Value Called;
    if (Callee) {
      auto StrcmpF = Glob.GetOrCreateLLVMFunction(Callee);
      Called =
          builder.create<mlir::LLVM::CallOp>(loc, StrcmpF, Args).getResult();
    } else {
      Args.insert(Args.begin(), GetLlvm(expr->getCallee()));
      auto CT = expr->getType();
      if (IsReference)
        CT = Glob.getCGM().getContext().getLValueReferenceType(CT);
      SmallVector<mlir::Type> RTs = {TypeTranslator.translateType(
          anonymize(getLLVMType(CT, Glob.getCGM())))};

      auto Ft = Args[0]
                    .getType()
                    .cast<LLVM::LLVMPointerType>()
                    .getElementType()
                    .cast<LLVM::LLVMFunctionType>();
      assert(RTs[0] == Ft.getReturnType());
      if (RTs[0].isa<LLVM::LLVMVoidType>())
        RTs.clear();
      Called = builder.create<mlir::LLVM::CallOp>(loc, RTs, Args).getResult();
    }
    if (IsReference) {
      if (!(Called.getType().isa<LLVM::LLVMPointerType>() ||
            Called.getType().isa<MemRefType>())) {
        expr->dump();
        expr->getType()->dump();
        llvm::errs() << " call: " << Called << "\n";
      }
    }
    if (!Called)
      return nullptr;
    return ValueCategory(Called, IsReference);
  }

  /// If the callee is part of the SYCL namespace, we may not want the
  /// GetOrCreateMLIRFunction to add this FuncOp to the functionsToEmit deque,
  /// since we will create it's equivalent with SYCL operations. Please note
  /// that we still generate some functions that we need for lowering some
  /// sycl op.  Therefore, in those case, we set ShouldEmit back to "true" by
  /// looking them up in our "registry" of supported functions.
  bool IsSyclFunc =
      mlirclang::isNamespaceSYCL(Callee->getEnclosingNamespaceContext());
  bool ShouldEmit = !IsSyclFunc;

  std::string MangledName =
      MLIRScanner::getMangledFuncName(*Callee, Glob.getCGM());
  if (GenerateAllSYCLFuncs || isSupportedFunctions(MangledName))
    ShouldEmit = true;

  FunctionToEmit F(*Callee, mlirclang::getInputContext(builder));
  auto ToCall = cast<func::FuncOp>(Glob.GetOrCreateMLIRFunction(F, ShouldEmit));

  SmallVector<std::pair<ValueCategory, clang::Expr *>> Args;
  QualType ObjType;

  if (auto *CC = dyn_cast<CXXMemberCallExpr>(expr)) {
    ValueCategory Obj = Visit(CC->getImplicitObjectArgument());
    ObjType = CC->getObjectType();
#ifdef DEBUG
    if (!obj.val) {
      function.dump();
      llvm::errs() << " objval: " << obj.val << "\n";
      expr->dump();
      CC->getImplicitObjectArgument()->dump();
    }
#endif
    if (cast<MemberExpr>(CC->getCallee()->IgnoreParens())->isArrow()) {
      Obj = Obj.dereference(builder);
    }
    assert(Obj.val);
    assert(Obj.isReference);
    Args.emplace_back(std::make_pair(Obj, (clang::Expr *)nullptr));
  }

  for (auto *A : expr->arguments())
    Args.push_back(std::make_pair(Visit(A), A));

  return CallHelper(ToCall, ObjType, Args, expr->getType(),
                    expr->isLValue() || expr->isXValue(), expr, Callee);
}

std::pair<ValueCategory, bool>
MLIRScanner::EmitGPUCallExpr(clang::CallExpr *expr) {
  auto Loc = getMLIRLocation(expr->getExprLoc());
  if (auto *Ic = dyn_cast<ImplicitCastExpr>(expr->getCallee())) {
    if (auto *Sr = dyn_cast<DeclRefExpr>(Ic->getSubExpr())) {
      if (Sr->getDecl()->getIdentifier() &&
          Sr->getDecl()->getName() == "__syncthreads") {
        builder.create<mlir::NVVM::Barrier0Op>(Loc);
        return std::make_pair(ValueCategory(), true);
      }
      if (Sr->getDecl()->getIdentifier() &&
          Sr->getDecl()->getName() == "cudaFuncSetCacheConfig") {
        llvm::errs() << " Not emitting GPU option: cudaFuncSetCacheConfig\n";
        return std::make_pair(ValueCategory(), true);
      }
      // TODO move free out.
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "free" ||
           Sr->getDecl()->getName() == "cudaFree" ||
           Sr->getDecl()->getName() == "cudaFreeHost")) {

        auto *Sub = expr->getArg(0);
        while (auto *BC = dyn_cast<clang::CastExpr>(Sub))
          Sub = BC->getSubExpr();
        mlir::Value Arg = Visit(Sub).getValue(builder);

        if (Arg.getType().isa<mlir::LLVM::LLVMPointerType>()) {
          const auto *Callee = EmitCallee(expr->getCallee());
          auto StrcmpF = Glob.GetOrCreateLLVMFunction(Callee);
          mlir::Value Args[] = {builder.create<LLVM::BitcastOp>(
              Loc, LLVM::LLVMPointerType::get(builder.getIntegerType(8)), Arg)};
          builder.create<mlir::LLVM::CallOp>(Loc, StrcmpF, Args);
        } else {
          builder.create<mlir::memref::DeallocOp>(Loc, Arg);
        }
        if (Sr->getDecl()->getName() == "cudaFree" ||
            Sr->getDecl()->getName() == "cudaFreeHost") {
          auto Ty = Glob.getTypes().getMLIRType(expr->getType());
          auto Op = builder.create<ConstantIntOp>(Loc, 0, Ty);
          return std::make_pair(ValueCategory(Op, /*isReference*/ false), true);
        }
        // TODO remove me when the free is removed.
        return std::make_pair(ValueCategory(), true);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "cudaMalloc" ||
           Sr->getDecl()->getName() == "cudaMallocHost" ||
           Sr->getDecl()->getName() == "cudaMallocPitch")) {
        auto *Sub = expr->getArg(0);
        while (auto *BC = dyn_cast<clang::CastExpr>(Sub))
          Sub = BC->getSubExpr();
        {
          auto Dst = Visit(Sub).getValue(builder);
          if (auto Omt = Dst.getType().dyn_cast<MemRefType>()) {
            if (auto mt = Omt.getElementType().dyn_cast<MemRefType>()) {
              auto Shape = std::vector<int64_t>(mt.getShape());

              auto ElemSize = getTypeSize(
                  cast<clang::PointerType>(
                      cast<clang::PointerType>(
                          Sub->getType()->getUnqualifiedDesugaredType())
                          ->getPointeeType())
                      ->getPointeeType());
              mlir::Value AllocSize;
              if (Sr->getDecl()->getName() == "cudaMallocPitch") {
                mlir::Value Width = Visit(expr->getArg(2)).getValue(builder);
                mlir::Value Height = Visit(expr->getArg(3)).getValue(builder);
                // Not changing pitch from provided width here
                // TODO can consider addition alignment considerations
                Visit(expr->getArg(1))
                    .dereference(builder)
                    .store(builder, Width);
                AllocSize = builder.create<MulIOp>(Loc, Width, Height);
              } else
                AllocSize = Visit(expr->getArg(1)).getValue(builder);
              auto IdxType = mlir::IndexType::get(builder.getContext());
              mlir::Value Args[1] = {builder.create<DivUIOp>(
                  Loc, builder.create<IndexCastOp>(Loc, IdxType, AllocSize),
                  ElemSize)};
              auto Alloc = builder.create<mlir::memref::AllocOp>(
                  Loc,
                  (Sr->getDecl()->getName() != "cudaMallocHost" && !CudaLower)
                      ? mlir::MemRefType::get(
                            Shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(),
                            wrapIntegerMemorySpace(1, mt.getContext()))
                      : mt,
                  Args);
              ValueCategory(Dst, /*isReference*/ true)
                  .store(builder,
                         builder.create<mlir::memref::CastOp>(Loc, mt, Alloc));
              auto RetTy = Glob.getTypes().getMLIRType(expr->getType());
              return std::make_pair(
                  ValueCategory(builder.create<ConstantIntOp>(Loc, 0, RetTy),
                                /*isReference*/ false),
                  true);
            }
          }
        }
      }
    }

    auto CreateBlockIdOp = [&](gpu::Dimension Str,
                               mlir::Type MlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          Loc, MlirType,
          builder.create<mlir::gpu::BlockIdOp>(
              Loc, mlir::IndexType::get(builder.getContext()), Str));
    };

    auto CreateBlockDimOp = [&](gpu::Dimension Str,
                                mlir::Type MlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          Loc, MlirType,
          builder.create<mlir::gpu::BlockDimOp>(
              Loc, mlir::IndexType::get(builder.getContext()), Str));
    };

    auto CreateThreadIdOp = [&](gpu::Dimension Str,
                                mlir::Type MlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          Loc, MlirType,
          builder.create<mlir::gpu::ThreadIdOp>(
              Loc, mlir::IndexType::get(builder.getContext()), Str));
    };

    auto CreateGridDimOp = [&](gpu::Dimension Str,
                               mlir::Type MlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          Loc, MlirType,
          builder.create<mlir::gpu::GridDimOp>(
              Loc, mlir::IndexType::get(builder.getContext()), Str));
    };

    if (auto *ME = dyn_cast<MemberExpr>(Ic->getSubExpr())) {
      auto MemberName = ME->getMemberDecl()->getName();

      if (auto *Sr2 = dyn_cast<OpaqueValueExpr>(ME->getBase())) {
        if (auto *Sr = dyn_cast<DeclRefExpr>(Sr2->getSourceExpr())) {
          if (Sr->getDecl()->getName() == "blockIdx") {
            auto MlirType = Glob.getTypes().getMLIRType(expr->getType());
            if (MemberName == "__fetch_builtin_x") {
              return std::make_pair(
                  ValueCategory(CreateBlockIdOp(gpu::Dimension::x, MlirType),
                                /*isReference*/ false),
                  true);
            }
            if (MemberName == "__fetch_builtin_y") {
              return std::make_pair(
                  ValueCategory(CreateBlockIdOp(gpu::Dimension::y, MlirType),
                                /*isReference*/ false),
                  true);
            }
            if (MemberName == "__fetch_builtin_z") {
              return std::make_pair(
                  ValueCategory(CreateBlockIdOp(gpu::Dimension::z, MlirType),
                                /*isReference*/ false),
                  true);
            }
          }
          if (Sr->getDecl()->getName() == "blockDim") {
            auto MlirType = Glob.getTypes().getMLIRType(expr->getType());
            if (MemberName == "__fetch_builtin_x") {
              return std::make_pair(
                  ValueCategory(CreateBlockDimOp(gpu::Dimension::x, MlirType),
                                /*isReference*/ false),
                  true);
            }
            if (MemberName == "__fetch_builtin_y") {
              return std::make_pair(
                  ValueCategory(CreateBlockDimOp(gpu::Dimension::y, MlirType),
                                /*isReference*/ false),
                  true);
            }
            if (MemberName == "__fetch_builtin_z") {
              return std::make_pair(
                  ValueCategory(CreateBlockDimOp(gpu::Dimension::z, MlirType),
                                /*isReference*/ false),
                  true);
            }
          }
          if (Sr->getDecl()->getName() == "threadIdx") {
            auto MlirType = Glob.getTypes().getMLIRType(expr->getType());
            if (MemberName == "__fetch_builtin_x") {
              return std::make_pair(
                  ValueCategory(CreateThreadIdOp(gpu::Dimension::x, MlirType),
                                /*isReference*/ false),
                  true);
            }
            if (MemberName == "__fetch_builtin_y") {
              return std::make_pair(
                  ValueCategory(CreateThreadIdOp(gpu::Dimension::y, MlirType),
                                /*isReference*/ false),
                  true);
            }
            if (MemberName == "__fetch_builtin_z") {
              return std::make_pair(
                  ValueCategory(CreateThreadIdOp(gpu::Dimension::z, MlirType),
                                /*isReference*/ false),
                  true);
            }
          }
          if (Sr->getDecl()->getName() == "gridDim") {
            auto MlirType = Glob.getTypes().getMLIRType(expr->getType());
            if (MemberName == "__fetch_builtin_x") {
              return std::make_pair(
                  ValueCategory(CreateGridDimOp(gpu::Dimension::x, MlirType),
                                /*isReference*/ false),
                  true);
            }
            if (MemberName == "__fetch_builtin_y") {
              return std::make_pair(
                  ValueCategory(CreateGridDimOp(gpu::Dimension::y, MlirType),
                                /*isReference*/ false),
                  true);
            }
            if (MemberName == "__fetch_builtin_z") {
              return std::make_pair(
                  ValueCategory(CreateGridDimOp(gpu::Dimension::z, MlirType),
                                /*isReference*/ false),
                  true);
            }
          }
        }
      }
    }
  }
  return std::make_pair(ValueCategory(), false);
}

std::pair<ValueCategory, bool>
MLIRScanner::EmitBuiltinOps(clang::CallExpr *expr) {
  if (auto *Ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *Sr = dyn_cast<DeclRefExpr>(Ic->getSubExpr()))
      if (Sr->getDecl()->getIdentifier() &&
          Sr->getDecl()->getName() == "__log2f") {
        std::vector<mlir::Value> Args;
        for (auto *A : expr->arguments())
          Args.push_back(Visit(A).getValue(builder));

        return std::make_pair(
            ValueCategory(builder.create<mlir::math::Log2Op>(loc, Args[0]),
                          /*isReference*/ false),
            true);
      }

  std::vector<mlir::Value> Args;
  auto VisitArgs = [&]() {
    assert(Args.empty() && "Expecting empty args");
    for (auto *A : expr->arguments())
      Args.push_back(Visit(A).getValue(builder));
  };
  Optional<Value> V = None;
  switch (expr->getBuiltinCallee()) {
  case Builtin::BIceil: {
    VisitArgs();
    V = builder.create<math::CeilOp>(loc, Args[0]);
  } break;
  case Builtin::BIcos: {
    VisitArgs();
    V = builder.create<mlir::math::CosOp>(loc, Args[0]);
  } break;
  case Builtin::BIexp:
  case Builtin::BIexpf: {
    VisitArgs();
    V = builder.create<mlir::math::ExpOp>(loc, Args[0]);
  } break;
  case Builtin::BIlog: {
    VisitArgs();
    V = builder.create<mlir::math::LogOp>(loc, Args[0]);
  } break;
  case Builtin::BIsin: {
    VisitArgs();
    V = builder.create<mlir::math::SinOp>(loc, Args[0]);
  } break;
  case Builtin::BIsqrt:
  case Builtin::BIsqrtf: {
    VisitArgs();
    V = builder.create<mlir::math::SqrtOp>(loc, Args[0]);
  } break;
  case Builtin::BI__builtin_atanh:
  case Builtin::BI__builtin_atanhf:
  case Builtin::BI__builtin_atanhl: {
    VisitArgs();
    V = builder.create<math::AtanOp>(loc, Args[0]);
  } break;
  case Builtin::BI__builtin_copysign:
  case Builtin::BI__builtin_copysignf:
  case Builtin::BI__builtin_copysignl: {
    VisitArgs();
    V = builder.create<LLVM::CopySignOp>(loc, Args[0], Args[1]);
  } break;
  case Builtin::BI__builtin_exp2:
  case Builtin::BI__builtin_exp2f:
  case Builtin::BI__builtin_exp2l: {
    VisitArgs();
    V = builder.create<math::Exp2Op>(loc, Args[0]);
  } break;
  case Builtin::BI__builtin_expm1:
  case Builtin::BI__builtin_expm1f:
  case Builtin::BI__builtin_expm1l: {
    VisitArgs();
    V = builder.create<math::ExpM1Op>(loc, Args[0]);
  } break;
  case Builtin::BI__builtin_fma:
  case Builtin::BI__builtin_fmaf:
  case Builtin::BI__builtin_fmal: {
    VisitArgs();
    V = builder.create<LLVM::FMAOp>(loc, Args[0], Args[1], Args[2]);
  } break;
  case Builtin::BI__builtin_fmax:
  case Builtin::BI__builtin_fmaxf:
  case Builtin::BI__builtin_fmaxl: {
    VisitArgs();
    V = builder.create<LLVM::MaxNumOp>(loc, Args[0], Args[1]);
  } break;
  case Builtin::BI__builtin_fmin:
  case Builtin::BI__builtin_fminf:
  case Builtin::BI__builtin_fminl: {
    VisitArgs();
    V = builder.create<LLVM::MinNumOp>(loc, Args[0], Args[1]);
  } break;
  case Builtin::BI__builtin_log1p:
  case Builtin::BI__builtin_log1pf:
  case Builtin::BI__builtin_log1pl: {
    VisitArgs();
    V = builder.create<math::Log1pOp>(loc, Args[0]);
  } break;
  case Builtin::BI__builtin_pow:
  case Builtin::BI__builtin_powf:
  case Builtin::BI__builtin_powl: {
    VisitArgs();
    V = builder.create<math::PowFOp>(loc, Args[0], Args[1]);
  } break;
  case Builtin::BI__builtin_assume: {
    VisitArgs();
    V = builder.create<LLVM::AssumeOp>(loc, Args[0])->getResult(0);
  } break;
  case Builtin::BI__builtin_isgreater: {
    VisitArgs();
    auto PostTy =
        Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    V = builder.create<ExtUIOp>(
        loc, PostTy,
        builder.create<CmpFOp>(loc, CmpFPredicate::OGT, Args[0], Args[1]));
  } break;
  case Builtin::BI__builtin_isgreaterequal: {
    VisitArgs();
    auto PostTy =
        Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    V = builder.create<ExtUIOp>(
        loc, PostTy,
        builder.create<CmpFOp>(loc, CmpFPredicate::OGE, Args[0], Args[1]));
  } break;
  case Builtin::BI__builtin_isless: {
    VisitArgs();
    auto PostTy =
        Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    V = builder.create<ExtUIOp>(
        loc, PostTy,
        builder.create<CmpFOp>(loc, CmpFPredicate::OLT, Args[0], Args[1]));
  } break;
  case Builtin::BI__builtin_islessequal: {
    VisitArgs();
    auto PostTy =
        Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    V = builder.create<ExtUIOp>(
        loc, PostTy,
        builder.create<CmpFOp>(loc, CmpFPredicate::OLE, Args[0], Args[1]));
  } break;
  case Builtin::BI__builtin_islessgreater: {
    VisitArgs();
    auto PostTy =
        Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    V = builder.create<ExtUIOp>(
        loc, PostTy,
        builder.create<CmpFOp>(loc, CmpFPredicate::ONE, Args[0], Args[1]));
  } break;
  case Builtin::BI__builtin_isunordered: {
    VisitArgs();
    auto PostTy =
        Glob.getTypes().getMLIRType(expr->getType()).cast<mlir::IntegerType>();
    V = builder.create<ExtUIOp>(
        loc, PostTy,
        builder.create<CmpFOp>(loc, CmpFPredicate::UNO, Args[0], Args[1]));
  } break;
  case Builtin::BImemmove:
  case Builtin::BI__builtin_memmove: {
    VisitArgs();
    builder.create<LLVM::MemmoveOp>(
        loc, Args[0], Args[1], Args[2],
        /*isVolatile*/ builder.create<ConstantIntOp>(loc, false, 1));
    V = Args[0];
  } break;
  case Builtin::BImemset:
  case Builtin::BI__builtin_memset: {
    VisitArgs();
    builder.create<LLVM::MemsetOp>(
        loc, Args[0],
        builder.create<TruncIOp>(loc, builder.getI8Type(), Args[1]), Args[2],
        /*isVolatile*/ builder.create<ConstantIntOp>(loc, false, 1));
    V = Args[0];
  } break;
  case Builtin::BImemcpy:
  case Builtin::BI__builtin_memcpy: {
    VisitArgs();
    builder.create<LLVM::MemcpyOp>(
        loc, Args[0], Args[1], Args[2],
        /*isVolatile*/ builder.create<ConstantIntOp>(loc, false, 1));
    V = Args[0];
  } break;
  }
  if (V.has_value())
    return std::make_pair(ValueCategory(V.value(),
                                        /*isReference*/ false),
                          true);

  return std::make_pair(ValueCategory(), false);
}
