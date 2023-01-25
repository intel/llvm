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
#include "llvm/Support/WithColor.h"

#define DEBUG_TYPE "CGCall"

using namespace mlir;

extern llvm::cl::opt<bool> CudaLower;
extern llvm::cl::opt<bool> GenerateAllSYCLFuncs;

/******************************************************************************/
/*                           Utility Functions                                */
/******************************************************************************/

/// Try to typecast the caller arg of type MemRef to fit the corresponding
/// callee arg type. We only deal with the cast where src and dst have the same
/// shape size and elem type, and just the first shape differs: src has
/// ShapedType::kDynamic and dst has a constant integer.
static Value castCallerMemRefArg(Value CallerArg, Type CalleeArgType,
                                 OpBuilder &B) {
  OpBuilder::InsertionGuard Guard(B);
  Type CallerArgType = CallerArg.getType();

  if (MemRefType DstTy = CalleeArgType.dyn_cast_or_null<MemRefType>()) {
    MemRefType SrcTy = CallerArgType.dyn_cast<MemRefType>();
    if (SrcTy && DstTy.getElementType() == SrcTy.getElementType() &&
        DstTy.getMemorySpace() == SrcTy.getMemorySpace()) {
      auto SrcShape = SrcTy.getShape();
      auto DstShape = DstTy.getShape();

      if (SrcShape.size() == DstShape.size() && !SrcShape.empty() &&
          SrcShape[0] == ShapedType::kDynamic &&
          std::equal(std::next(SrcShape.begin()), SrcShape.end(),
                     std::next(DstShape.begin()))) {
        B.setInsertionPointAfterValue(CallerArg);

        return B.create<memref::CastOp>(CallerArg.getLoc(), CalleeArgType,
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
static void castCallerArgs(func::FuncOp Callee,
                           llvm::SmallVectorImpl<Value> &Args, OpBuilder &B) {
  FunctionType FuncTy = Callee.getFunctionType();
  assert(Args.size() == FuncTy.getNumInputs() &&
         "The caller arguments should have the same size as the number of "
         "callee arguments as the interface.");

  LLVM_DEBUG({
    llvm::dbgs() << "FuncTy: " << FuncTy << "\n";
    llvm::dbgs() << "Args: \n";
    for (const Value &arg : Args)
      llvm::dbgs().indent(2) << arg << "\n";
  });

  for (unsigned I = 0; I < Args.size(); ++I) {
    Type CalleeArgType = FuncTy.getInput(I);
    Type CallerArgType = Args[I].getType();

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

ValueCategory MLIRScanner::callHelper(
    func::FuncOp ToCall, clang::QualType ObjType,
    ArrayRef<std::pair<ValueCategory, clang::Expr *>> Arguments,
    clang::QualType RetType, bool RetReference, clang::Expr *Expr,
    const clang::FunctionDecl &Callee) {
  SmallVector<Value, 4> Args;
  FunctionType FnType = ToCall.getFunctionType();
  const clang::CodeGen::CGFunctionInfo &CalleeInfo =
      Glob.getOrCreateCGFunctionInfo(&Callee);
  auto CalleeArgs = CalleeInfo.arguments();

  size_t I = 0;
  // map from declaration name to value
  std::map<std::string, Value> MapFuncOperands;

  for (const std::pair<ValueCategory, clang::Expr *> &Pair : Arguments) {
    ValueCategory Arg = std::get<0>(Pair);
    clang::Expr *A = std::get<1>(Pair);

    LLVM_DEBUG({
      if (!Arg.val) {
        Expr->dump();
        A->dump();
      }
    });
    assert(Arg.val && "expect not null");

    if (auto *Ice = dyn_cast_or_null<clang::ImplicitCastExpr>(A))
      if (auto *Dre = dyn_cast<clang::DeclRefExpr>(Ice->getSubExpr()))
        MapFuncOperands.insert(
            make_pair(Dre->getDecl()->getName().str(), Arg.val));

    if (I >= FnType.getInputs().size() || (I != 0 && A == nullptr)) {
      LLVM_DEBUG({
        Expr->dump();
        ToCall.dump();
        FnType.dump();
        for (auto A : Arguments)
          std::get<1>(A)->dump();
      });
      assert(false && "too many arguments in calls");
    }

    bool IsReference =
        (I == 0 && A == nullptr) || A->isLValue() || A->isXValue();

    bool IsArray = false;
    clang::QualType AType = (I == 0 && A == nullptr) ? ObjType : A->getType();
    Type ExpectedType = Glob.getTypes().getMLIRType(AType, &IsArray);

    LLVM_DEBUG({
      llvm::dbgs() << "AType: " << AType << "\n";
      llvm::dbgs() << "AType addrspace: "
                   << Glob.getCGM().getContext().getTargetAddressSpace(
                          AType.getAddressSpace())
                   << "\n";
      llvm::dbgs() << "ExpectedType: " << ExpectedType << "\n";
    });

    if (auto PT = Arg.val.getType().dyn_cast<LLVM::LLVMPointerType>()) {
      if (PT.getAddressSpace() == 5)
        Arg.val = Builder.create<LLVM::AddrSpaceCastOp>(
            Loc, LLVM::LLVMPointerType::get(PT.getElementType(), 0), Arg.val);
    }

    Value Val = nullptr;
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

        auto MT =
            Glob.getTypes()
                .getMLIRType(
                    Glob.getCGM().getContext().getLValueReferenceType(AType))
                .cast<MemRefType>();

        LLVM_DEBUG({
          llvm::dbgs() << "MT: " << MT << "\n";
          llvm::dbgs() << "getLValueReferenceType(aType): "
                       << Glob.getCGM().getContext().getLValueReferenceType(
                              AType)
                       << "\n";
        });

        auto Shape = std::vector<int64_t>(MT.getShape());
        assert(Shape.size() == 2);

        auto Pshape = Shape[0];
        if (Pshape == ShapedType::kDynamic)
          Shape[0] = 1;

        OpBuilder ABuilder(Builder.getContext());
        ABuilder.setInsertionPointToStart(AllocationScope);
        auto Alloc = ABuilder.create<memref::AllocaOp>(
            Loc,
            MemRefType::get(Shape, MT.getElementType(),
                            MemRefLayoutAttrInterface(), MT.getMemorySpace()));
        ValueCategory(Alloc, /*isRef*/ true)
            .store(Builder, Arg, /*isArray*/ IsArray);
        Shape[0] = Pshape;
        Val = Builder.create<memref::CastOp>(
            Loc,
            MemRefType::get(Shape, MT.getElementType(),
                            MemRefLayoutAttrInterface(), MT.getMemorySpace()),
            Alloc);
      } else {
        if (CalleeArgs[I].info.getKind() ==
                clang::CodeGen::ABIArgInfo::Indirect ||
            CalleeArgs[I].info.getKind() ==
                clang::CodeGen::ABIArgInfo::IndirectAliased) {
          OpBuilder ABuilder(Builder.getContext());
          ABuilder.setInsertionPointToStart(AllocationScope);
          auto Ty = Glob.getTypes().getPointerOrMemRefType(
              Arg.getValue(Builder).getType(),
              Glob.getCGM().getDataLayout().getAllocaAddrSpace(),
              /*IsAlloc*/ true);
          if (auto MemRefTy = Ty.dyn_cast<MemRefType>()) {
            Val = ABuilder.create<memref::AllocaOp>(Loc, MemRefTy);
            Val = ABuilder.create<memref::CastOp>(
                Loc,
                MemRefType::get(ShapedType::kDynamic,
                                Arg.getValue(Builder).getType()),
                Val);
          } else
            Val = ABuilder.create<LLVM::AllocaOp>(
                Loc, Ty, ABuilder.create<arith::ConstantIntOp>(Loc, 1, 64), 0);

          ValueCategory(Val, /*isRef*/ true)
              .store(Builder, Arg.getValue(Builder));
        } else
          Val = Arg.getValue(Builder);

        if (Val.getType().isa<LLVM::LLVMPointerType>() &&
            ExpectedType.isa<MemRefType>())
          Val = Builder.create<polygeist::Pointer2MemrefOp>(Loc, ExpectedType,
                                                            Val);

        if (auto PrevTy = Val.getType().dyn_cast<IntegerType>()) {
          auto IPostTy = ExpectedType.cast<IntegerType>();
          if (PrevTy != IPostTy)
            Val = Builder.create<arith::TruncIOp>(Loc, IPostTy, Val);
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
            Builder.create<polygeist::Pointer2MemrefOp>(Loc, ExpectedType, Val);
      else if (Val.getType().isa<MemRefType>() &&
               ExpectedType.isa<LLVM::LLVMPointerType>())
        Val =
            Builder.create<polygeist::Memref2PointerOp>(Loc, ExpectedType, Val);

      Val = castToMemSpaceOfType(Val, ExpectedType);
    }
    assert(Val);
    Args.push_back(Val);
    I++;
  }

  // handle lowerto pragma.
  if (LTInfo.SymbolTable.count(ToCall.getName())) {
    SmallVector<Value> InputOperands;
    SmallVector<Value> OutputOperands;
    for (StringRef Input : LTInfo.InputSymbol)
      if (MapFuncOperands.find(Input.str()) != MapFuncOperands.end())
        InputOperands.push_back(MapFuncOperands[Input.str()]);
    for (StringRef Output : LTInfo.OutputSymbol)
      if (MapFuncOperands.find(Output.str()) != MapFuncOperands.end())
        OutputOperands.push_back(MapFuncOperands[Output.str()]);

    if (InputOperands.size() == 0)
      InputOperands.append(Args);

    return ValueCategory(mlirclang::replaceFuncByOperation(
                             ToCall, LTInfo.SymbolTable[ToCall.getName()],
                             Builder, InputOperands, OutputOperands)
                             ->getResult(0),
                         /*isReference=*/false);
  }

  bool IsArrayReturn = false;
  if (!RetReference)
    Glob.getTypes().getMLIRType(RetType, &IsArrayReturn);

  Value Alloc;
  if (IsArrayReturn) {
    auto MT =
        Glob.getTypes()
            .getMLIRType(
                Glob.getCGM().getContext().getLValueReferenceType(RetType))
            .cast<MemRefType>();

    auto Shape = std::vector<int64_t>(MT.getShape());
    assert(Shape.size() == 2);

    auto Pshape = Shape[0];
    if (Pshape == ShapedType::kDynamic)
      Shape[0] = 1;

    OpBuilder ABuilder(Builder.getContext());
    ABuilder.setInsertionPointToStart(AllocationScope);
    Alloc = ABuilder.create<memref::AllocaOp>(
        Loc, MemRefType::get(Shape, MT.getElementType(),
                             MemRefLayoutAttrInterface(), MT.getMemorySpace()));
    Shape[0] = Pshape;
    Alloc = Builder.create<memref::CastOp>(
        Loc,
        MemRefType::get(Shape, MT.getElementType(), MemRefLayoutAttrInterface(),
                        MT.getMemorySpace()),
        Alloc);
    Args.push_back(Alloc);
  }

  if (auto *CU = dyn_cast<clang::CUDAKernelCallExpr>(Expr)) {
    auto L0 = Visit(CU->getConfig()->getArg(0));
    assert(L0.isReference);
    Value Blocks[3];
    for (int I = 0; I < 3; I++) {
      Value Val = L0.val;
      if (auto MT = Val.getType().dyn_cast<MemRefType>()) {
        assert(MT.getShape().size() == 2);
        Blocks[I] = Builder.create<arith::IndexCastOp>(
            Loc, IndexType::get(Builder.getContext()),
            Builder.create<memref::LoadOp>(
                Loc, Val,
                ValueRange({getConstantIndex(0), getConstantIndex(I)})));
      } else {
        auto PT = Val.getType().cast<LLVM::LLVMPointerType>();
        auto ET = PT.getElementType().cast<LLVM::LLVMStructType>().getBody()[I];
        Blocks[I] = Builder.create<arith::IndexCastOp>(
            Loc, IndexType::get(Builder.getContext()),
            Builder.create<LLVM::LoadOp>(
                Loc,
                Builder.create<LLVM::GEPOp>(
                    Loc, LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
                    Val,
                    ValueRange(
                        {Builder.create<arith::ConstantIntOp>(Loc, 0, 32),
                         Builder.create<arith::ConstantIntOp>(Loc, I, 32)}))));
      }
    }

    auto T0 = Visit(CU->getConfig()->getArg(1));
    assert(T0.isReference);
    Value Threads[3];
    for (int I = 0; I < 3; I++) {
      Value Val = T0.val;
      if (auto MT = Val.getType().dyn_cast<MemRefType>()) {
        assert(MT.getShape().size() == 2);
        Threads[I] = Builder.create<arith::IndexCastOp>(
            Loc, IndexType::get(Builder.getContext()),
            Builder.create<memref::LoadOp>(
                Loc, Val,
                ValueRange({getConstantIndex(0), getConstantIndex(I)})));
      } else {
        auto PT = Val.getType().cast<LLVM::LLVMPointerType>();
        auto ET = PT.getElementType().cast<LLVM::LLVMStructType>().getBody()[I];
        Threads[I] = Builder.create<arith::IndexCastOp>(
            Loc, IndexType::get(Builder.getContext()),
            Builder.create<LLVM::LoadOp>(
                Loc,
                Builder.create<LLVM::GEPOp>(
                    Loc, LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
                    Val,
                    ValueRange(
                        {Builder.create<arith::ConstantIntOp>(Loc, 0, 32),
                         Builder.create<arith::ConstantIntOp>(Loc, I, 32)}))));
      }
    }
    Value Stream = nullptr;
    SmallVector<Value, 1> AsyncDependencies;
    if (3 < CU->getConfig()->getNumArgs() &&
        !isa<clang::CXXDefaultArgExpr>(CU->getConfig()->getArg(3))) {
      Stream = Visit(CU->getConfig()->getArg(3)).getValue(Builder);
      Stream = Builder.create<polygeist::StreamToTokenOp>(
          Loc, Builder.getType<gpu::AsyncTokenType>(), Stream);
      assert(Stream);
      AsyncDependencies.push_back(Stream);
    }
    auto Op = Builder.create<gpu::LaunchOp>(
        Loc, Blocks[0], Blocks[1], Blocks[2], Threads[0], Threads[1],
        Threads[2],
        /*dynamic shmem size*/ nullptr,
        /*token type*/ Stream ? Stream.getType() : nullptr,
        /*dependencies*/ AsyncDependencies);
    auto OldPoint = Builder.getInsertionPoint();
    auto *OldBlock = Builder.getInsertionBlock();
    Builder.setInsertionPointToStart(&Op.getRegion().front());
    Builder.create<func::CallOp>(Loc, ToCall, Args);
    Builder.create<gpu::TerminatorOp>(Loc);
    Builder.setInsertionPoint(OldBlock, OldPoint);
    return nullptr;
  }

  // Try to rescue some mismatched types.
  castCallerArgs(ToCall, Args, Builder);

  /// Try to emit SYCL operations before creating a CallOp
  Operation *Op = emitSYCLOps(Expr, Args);
  if (!Op)
    Op = Builder.create<func::CallOp>(Loc, ToCall, Args);

  if (IsArrayReturn) {
    // TODO remedy return
    if (RetReference)
      Expr->dump();
    assert(!RetReference);
    return ValueCategory(Alloc, /*isReference*/ true);
  }

  if (Op->getNumResults())
    return ValueCategory(Op->getResult(0),
                         /*isReference*/ RetReference);

  return nullptr;
}

std::pair<ValueCategory, bool>
MLIRScanner::emitClangBuiltinCallExpr(clang::CallExpr *Expr) {
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

ValueCategory MLIRScanner::VisitCallExpr(clang::CallExpr *Expr) {
  LLVM_DEBUG({
    llvm::dbgs() << "VisitCallExpr: ";
    Expr->dump();
    llvm::dbgs() << "\n";
  });

  Location Loc = getMLIRLocation(Expr->getExprLoc());

  auto ValEmitted = emitGPUCallExpr(Expr);
  if (ValEmitted.second)
    return ValEmitted.first;

  ValEmitted = emitBuiltinOps(Expr);
  if (ValEmitted.second)
    return ValEmitted.first;

  ValEmitted = emitClangBuiltinCallExpr(Expr);
  if (ValEmitted.second)
    return ValEmitted.first;

  if (auto *Oc = dyn_cast<clang::CXXOperatorCallExpr>(Expr)) {
    if (Oc->getOperator() == clang::OO_EqualEqual) {
      if (auto *LHS = dyn_cast<clang::CXXTypeidExpr>(Expr->getArg(0))) {
        if (auto *RHS = dyn_cast<clang::CXXTypeidExpr>(Expr->getArg(1))) {
          clang::QualType LT =
              LHS->isTypeOperand()
                  ? LHS->getTypeOperand(Glob.getCGM().getContext())
                  : LHS->getExprOperand()->getType();
          clang::QualType RT =
              RHS->isTypeOperand()
                  ? RHS->getTypeOperand(Glob.getCGM().getContext())
                  : RHS->getExprOperand()->getType();
          llvm::Constant *LC = Glob.getCGM().GetAddrOfRTTIDescriptor(LT);
          llvm::Constant *RC = Glob.getCGM().GetAddrOfRTTIDescriptor(RT);
          auto PostTy =
              Glob.getTypes().getMLIRType(Expr->getType()).cast<IntegerType>();
          return ValueCategory(
              Builder.create<arith::ConstantIntOp>(Loc, LC == RC, PostTy),
              false);
        }
      }
    }
  }

  if (auto *Oc = dyn_cast<clang::CXXMemberCallExpr>(Expr)) {
    if (auto *LHS =
            dyn_cast<clang::CXXTypeidExpr>(Oc->getImplicitObjectArgument())) {
      if (auto *Ic = dyn_cast<clang::MemberExpr>(Expr->getCallee()))
        if (auto *Sr = dyn_cast<clang::NamedDecl>(Ic->getMemberDecl())) {
          if (Sr->getIdentifier() && Sr->getName() == "name") {
            clang::QualType LT =
                LHS->isTypeOperand()
                    ? LHS->getTypeOperand(Glob.getCGM().getContext())
                    : LHS->getExprOperand()->getType();
            llvm::Constant *LC = Glob.getCGM().GetAddrOfRTTIDescriptor(LT);
            while (auto *CE = dyn_cast<llvm::ConstantExpr>(LC))
              LC = CE->getOperand(0);
            std::string Val = cast<llvm::GlobalVariable>(LC)->getName().str();
            return CommonArrayToPointer(ValueCategory(
                Glob.getOrCreateGlobalLLVMString(
                    Loc, Builder, Val, mlirclang::getFuncContext(Function)),
                /*isReference*/ true));
          }
        }
    }
  }

  if (auto *Ps = dyn_cast<clang::CXXPseudoDestructorExpr>(Expr->getCallee()))
    return Visit(Ps);

  if (auto *Ic = dyn_cast<clang::ImplicitCastExpr>(Expr->getCallee()))
    if (auto *Sr = dyn_cast<clang::DeclRefExpr>(Ic->getSubExpr())) {
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "atomicAdd" ||
           Sr->getDecl()->getName() == "atomicOr" ||
           Sr->getDecl()->getName() == "atomicAnd")) {
        std::vector<ValueCategory> Args;
        for (auto *A : Expr->arguments())
          Args.push_back(Visit(A));

        auto A0 = Args[0].getValue(Builder);
        auto A1 = Args[1].getValue(Builder);
        arith::AtomicRMWKind Op;
        LLVM::AtomicBinOp Lop;
        if (Sr->getDecl()->getName() == "atomicAdd") {
          if (A1.getType().isa<IntegerType>()) {
            Op = arith::AtomicRMWKind::addi;
            Lop = LLVM::AtomicBinOp::add;
          } else {
            Op = arith::AtomicRMWKind::addf;
            Lop = LLVM::AtomicBinOp::fadd;
          }
        } else if (Sr->getDecl()->getName() == "atomicOr") {
          Op = arith::AtomicRMWKind::ori;
          Lop = LLVM::AtomicBinOp::_or;
        } else if (Sr->getDecl()->getName() == "atomicAnd") {
          Op = arith::AtomicRMWKind::andi;
          Lop = LLVM::AtomicBinOp::_and;
        } else
          assert(0);

        if (A0.getType().isa<MemRefType>())
          return ValueCategory(Builder.create<memref::AtomicRMWOp>(
                                   Loc, A1.getType(), Op, A1, A0,
                                   std::vector<Value>({getConstantIndex(0)})),
                               /*isReference*/ false);
        return ValueCategory(
            Builder.create<LLVM::AtomicRMWOp>(Loc, A1.getType(), Lop, A0, A1,
                                              LLVM::AtomicOrdering::acq_rel),
            /*isReference*/ false);
      }
    }

  LLVM::TypeFromLLVMIRTranslator TypeTranslator(*Module->getContext());

  auto GetLLVM = [&](clang::Expr *E) -> Value {
    auto Sub = Visit(E);
    if (!Sub.val) {
      Expr->dump();
      E->dump();
    }
    assert(Sub.val);

    bool IsReference = E->isLValue() || E->isXValue();
    if (IsReference) {
      assert(Sub.isReference);
      Value Val = Sub.val;
      if (auto MT = Val.getType().dyn_cast<MemRefType>()) {
        Val = Builder.create<polygeist::Memref2PointerOp>(
            Loc,
            LLVM::LLVMPointerType::get(MT.getElementType(),
                                       MT.getMemorySpaceAsInt()),
            Val);
      }
      return Val;
    }

    bool IsArray = false;
    Glob.getTypes().getMLIRType(E->getType(), &IsArray);

    if (IsArray) {
      assert(Sub.isReference);
      auto MT =
          Glob.getTypes()
              .getMLIRType(Glob.getCGM().getContext().getLValueReferenceType(
                  E->getType()))
              .cast<MemRefType>();
      auto Shape = std::vector<int64_t>(MT.getShape());
      assert(Shape.size() == 2);

      OpBuilder ABuilder(Builder.getContext());
      ABuilder.setInsertionPointToStart(AllocationScope);
      auto One = ABuilder.create<arith::ConstantIntOp>(Loc, 1, 64);
      auto Alloc = ABuilder.create<LLVM::AllocaOp>(
          Loc,
          LLVM::LLVMPointerType::get(
              TypeTranslator.translateType(mlirclang::anonymize(
                  mlirclang::getLLVMType(E->getType(), Glob.getCGM()))),
              0),
          One, 0);
      ValueCategory(Alloc, /*isRef*/ true)
          .store(Builder, Sub, /*isArray*/ IsArray);
      Sub = ValueCategory(Alloc, /*isRef*/ true);
    }
    auto Val = Sub.getValue(Builder);
    if (auto MT = Val.getType().dyn_cast<MemRefType>()) {
      auto Nt = TypeTranslator
                    .translateType(mlirclang::anonymize(
                        mlirclang::getLLVMType(E->getType(), Glob.getCGM())))
                    .cast<LLVM::LLVMPointerType>();
      assert(Nt.getAddressSpace() == MT.getMemorySpaceAsInt() &&
             "val does not have the same memory space as nt");
      Val = Builder.create<polygeist::Memref2PointerOp>(Loc, Nt, Val);
    }
    return Val;
  };

  switch (Expr->getBuiltinCallee()) {
  case clang::Builtin::BI__builtin_strlen:
  case clang::Builtin::BIstrlen: {
    Value V0 = GetLLVM(Expr->getArg(0));

    const std::string Name("strlen");

    if (Glob.getFunctions().find(Name) == Glob.getFunctions().end()) {
      std::vector<Type> Types{V0.getType()};

      Type RT = Glob.getTypes().getMLIRType(Expr->getType());
      std::vector<Type> RetTypes{RT};
      OpBuilder Builder(Module->getContext());
      auto FuncType = Builder.getFunctionType(Types, RetTypes);
      Glob.getFunctions()[Name] = func::FuncOp(
          func::FuncOp::create(Builder.getUnknownLoc(), Name, FuncType));
      SymbolTable::setSymbolVisibility(Glob.getFunctions()[Name],
                                       SymbolTable::Visibility::Private);
      Module->push_back(Glob.getFunctions()[Name]);
    }

    return ValueCategory(
        Builder.create<func::CallOp>(Loc, Glob.getFunctions()[Name], V0)
            .getResult(0),
        false);
  }
  case clang::Builtin::BI__builtin_isnormal: {
    Value V = GetLLVM(Expr->getArg(0));
    auto Ty = V.getType().cast<FloatType>();
    Value Eq =
        Builder.create<arith::CmpFOp>(Loc, arith::CmpFPredicate::OEQ, V, V);

    Value Abs = Builder.create<math::AbsFOp>(Loc, V);
    auto Infinity = Builder.create<arith::ConstantFloatOp>(
        Loc, APFloat::getInf(Ty.getFloatSemantics()), Ty);
    Value IsLessThanInf = Builder.create<arith::CmpFOp>(
        Loc, arith::CmpFPredicate::ULT, Abs, Infinity);
    APFloat Smallest = APFloat::getSmallestNormalized(Ty.getFloatSemantics());
    auto SmallestV = Builder.create<arith::ConstantFloatOp>(Loc, Smallest, Ty);
    Value IsNormal = Builder.create<arith::CmpFOp>(
        Loc, arith::CmpFPredicate::UGE, Abs, SmallestV);
    V = Builder.create<arith::AndIOp>(Loc, Eq, IsLessThanInf);
    V = Builder.create<arith::AndIOp>(Loc, V, IsNormal);
    auto PostTy =
        Glob.getTypes().getMLIRType(Expr->getType()).cast<IntegerType>();
    Value Res = Builder.create<arith::ExtUIOp>(Loc, PostTy, V);
    return ValueCategory(Res, /*isRef*/ false);
  }
  case clang::Builtin::BI__builtin_signbit: {
    Value V = GetLLVM(Expr->getArg(0));
    auto Ty = V.getType().cast<FloatType>();
    auto ITy = Builder.getIntegerType(Ty.getWidth());
    Value BC = Builder.create<arith::BitcastOp>(Loc, ITy, V);
    auto ZeroV = Builder.create<arith::ConstantIntOp>(Loc, 0, ITy);
    V = Builder.create<arith::CmpIOp>(Loc, arith::CmpIPredicate::slt, BC,
                                      ZeroV);
    auto PostTy =
        Glob.getTypes().getMLIRType(Expr->getType()).cast<IntegerType>();
    Value Res = Builder.create<arith::ExtUIOp>(Loc, PostTy, V);
    return ValueCategory(Res, /*isRef*/ false);
  }
  case clang::Builtin::BI__builtin_addressof: {
    auto V = Visit(Expr->getArg(0));
    assert(V.isReference);
    Value Val = V.val;
    Type T = Glob.getTypes().getMLIRType(Expr->getType());
    if (T == Val.getType())
      return ValueCategory(Val, /*isRef*/ false);
    if (T.isa<LLVM::LLVMPointerType>()) {
      if (Val.getType().isa<MemRefType>()) {
        assert(Val.getType().cast<MemRefType>().getMemorySpaceAsInt() ==
                   T.cast<LLVM::LLVMPointerType>().getAddressSpace() &&
               "val does not have the same memory space as T");
        Val = Builder.create<polygeist::Memref2PointerOp>(Loc, T, Val);
      } else if (T != Val.getType())
        Val = Builder.create<LLVM::BitcastOp>(Loc, T, Val);
      return ValueCategory(Val, /*isRef*/ false);
    }
    assert(T.isa<MemRefType>());

    if (Val.getType().isa<MemRefType>())
      Val = Builder.create<polygeist::Memref2PointerOp>(
          Loc,
          LLVM::LLVMPointerType::get(
              Builder.getI8Type(),
              Val.getType().cast<MemRefType>().getMemorySpaceAsInt()),
          Val);
    if (Val.getType().isa<LLVM::LLVMPointerType>())
      Val = Builder.create<polygeist::Pointer2MemrefOp>(Loc, T, Val);
    return ValueCategory(Val, /*isRef*/ false);

    Expr->dump();
    llvm::errs() << " val: " << Val << " T: " << T << "\n";
    llvm_unreachable("unhandled builtin addressof");
  }
  }

  auto BuiltinCallee = Expr->getBuiltinCallee();

  if (auto *Ic = dyn_cast<clang::ImplicitCastExpr>(Expr->getCallee()))
    if (auto *Sr = dyn_cast<clang::DeclRefExpr>(Ic->getSubExpr())) {
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "__powf" ||
           BuiltinCallee == clang::Builtin::BIpow ||
           Sr->getDecl()->getName() == "__nv_pow" ||
           Sr->getDecl()->getName() == "__nv_powf" ||
           Sr->getDecl()->getName() == "__powi" ||
           Sr->getDecl()->getName() == "powi" ||
           Sr->getDecl()->getName() == "__nv_powi" ||
           Sr->getDecl()->getName() == "__nv_powi" ||
           BuiltinCallee == clang::Builtin::BIpowf)) {
        Type MLIRType = Glob.getTypes().getMLIRType(Expr->getType());
        std::vector<Value> Args;
        for (auto *A : Expr->arguments()) {
          Args.push_back(Visit(A).getValue(Builder));
        }
        if (Args[1].getType().isa<IntegerType>())
          return ValueCategory(
              Builder.create<LLVM::PowIOp>(Loc, MLIRType, Args[0], Args[1]),
              /*isReference*/ false);
        return ValueCategory(
            Builder.create<math::PowFOp>(Loc, MLIRType, Args[0], Args[1]),
            /*isReference*/ false);
      }
    }

  if (auto *Ic = dyn_cast<clang::ImplicitCastExpr>(Expr->getCallee()))
    if (auto *Sr = dyn_cast<clang::DeclRefExpr>(Ic->getSubExpr())) {
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "__nv_fabsf" ||
           Sr->getDecl()->getName() == "__nv_fabs" ||
           Sr->getDecl()->getName() == "__nv_abs" ||
           BuiltinCallee == clang::Builtin::BIfabs ||
           BuiltinCallee == clang::Builtin::BIfabsf ||
           BuiltinCallee == clang::Builtin::BI__builtin_fabs ||
           BuiltinCallee == clang::Builtin::BI__builtin_fabsf)) {
        // isinf(x)    --> fabs(x) == infinity
        // isfinite(x) --> fabs(x) != infinity
        // x != NaN via the ordered compare in either case.
        Value V = GetLLVM(Expr->getArg(0));
        Value Fabs;
        if (V.getType().isa<FloatType>())
          Fabs = Builder.create<math::AbsFOp>(Loc, V);
        else {
          auto Zero = Builder.create<arith::ConstantIntOp>(
              Loc, 0, V.getType().cast<IntegerType>().getWidth());
          Fabs = Builder.create<arith::SelectOp>(
              Loc,
              Builder.create<arith::CmpIOp>(Loc, arith::CmpIPredicate::sge, V,
                                            Zero),
              V, Builder.create<arith::SubIOp>(Loc, Zero, V));
        }
        return ValueCategory(Fabs, /*isRef*/ false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          Sr->getDecl()->getName() == "__nv_mul24") {
        Value V0 = GetLLVM(Expr->getArg(0));
        Value V1 = GetLLVM(Expr->getArg(1));
        auto C8 = Builder.create<arith::ConstantIntOp>(Loc, 8, 32);
        V0 = Builder.create<arith::ShLIOp>(Loc, V0, C8);
        V0 = Builder.create<arith::ShRUIOp>(Loc, V0, C8);
        V1 = Builder.create<arith::ShLIOp>(Loc, V1, C8);
        V1 = Builder.create<arith::ShRUIOp>(Loc, V1, C8);
        return ValueCategory(Builder.create<arith::MulIOp>(Loc, V0, V1), false);
      }
      if (BuiltinCallee == clang::Builtin::BI__builtin_frexp ||
          BuiltinCallee == clang::Builtin::BI__builtin_frexpf ||
          BuiltinCallee == clang::Builtin::BI__builtin_frexpl ||
          BuiltinCallee == clang::Builtin::BI__builtin_frexpf128) {
        Value V0 = GetLLVM(Expr->getArg(0));
        Value V1 = GetLLVM(Expr->getArg(1));

        auto Name = Sr->getDecl()
                        ->getName()
                        .substr(std::string("__builtin_").length())
                        .str();

        if (Glob.getFunctions().find(Name) == Glob.getFunctions().end()) {
          std::vector<Type> Types{V0.getType(), V1.getType()};

          Type RT = Glob.getTypes().getMLIRType(Expr->getType());
          std::vector<Type> Rettype{RT};
          OpBuilder MBuilder(Module->getContext());
          auto FuncType = MBuilder.getFunctionType(Types, Rettype);
          Glob.getFunctions()[Name] = func::FuncOp(
              func::FuncOp::create(Builder.getUnknownLoc(), Name, FuncType));
          SymbolTable::setSymbolVisibility(Glob.getFunctions()[Name],
                                           SymbolTable::Visibility::Private);
          Module->push_back(Glob.getFunctions()[Name]);
        }

        return ValueCategory(
            Builder
                .create<func::CallOp>(Loc, Glob.getFunctions()[Name],
                                      ValueRange({V0, V1}))
                .getResult(0),
            false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (BuiltinCallee == clang::Builtin::BI__builtin_isfinite ||
           BuiltinCallee == clang::Builtin::BI__builtin_isinf ||
           Sr->getDecl()->getName() == "__nv_isinff")) {
        // isinf(x)    --> fabs(x) == infinity
        // isfinite(x) --> fabs(x) != infinity
        // x != NaN via the ordered compare in either case.
        Value V = GetLLVM(Expr->getArg(0));
        auto Ty = V.getType().cast<FloatType>();
        Value Fabs = Builder.create<math::AbsFOp>(Loc, V);
        auto Infinity = Builder.create<arith::ConstantFloatOp>(
            Loc, APFloat::getInf(Ty.getFloatSemantics()), Ty);
        auto Pred = (BuiltinCallee == clang::Builtin::BI__builtin_isinf ||
                     Sr->getDecl()->getName() == "__nv_isinff")
                        ? arith::CmpFPredicate::OEQ
                        : arith::CmpFPredicate::ONE;
        Value FCmp = Builder.create<arith::CmpFOp>(Loc, Pred, Fabs, Infinity);
        auto PostTy =
            Glob.getTypes().getMLIRType(Expr->getType()).cast<IntegerType>();
        Value Res = Builder.create<arith::ExtUIOp>(Loc, PostTy, FCmp);
        return ValueCategory(Res, /*isRef*/ false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (BuiltinCallee == clang::Builtin::BI__builtin_isnan ||
           Sr->getDecl()->getName() == "__nv_isnanf")) {
        Value V = GetLLVM(Expr->getArg(0));
        Value Eq =
            Builder.create<arith::CmpFOp>(Loc, arith::CmpFPredicate::UNO, V, V);
        auto PostTy =
            Glob.getTypes().getMLIRType(Expr->getType()).cast<IntegerType>();
        Value Res = Builder.create<arith::ExtUIOp>(Loc, PostTy, Eq);
        return ValueCategory(Res, /*isRef*/ false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "__nv_fmodf")) {
        Value V = GetLLVM(Expr->getArg(0));
        Value V2 = GetLLVM(Expr->getArg(1));
        V = Builder.create<LLVM::FRemOp>(Loc, V.getType(), V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "__nv_scalbn" ||
           Sr->getDecl()->getName() == "__nv_scalbnf" ||
           Sr->getDecl()->getName() == "__nv_scalbnl")) {
        Value V = GetLLVM(Expr->getArg(0));
        Value V2 = GetLLVM(Expr->getArg(1));
        auto Name = Sr->getDecl()->getName().substr(5).str();
        std::vector<Type> Types{V.getType(), V2.getType()};
        Type RT = Glob.getTypes().getMLIRType(Expr->getType());

        std::vector<Type> Rettypes{RT};

        OpBuilder Builder(Module->getContext());
        auto FuncType = Builder.getFunctionType(Types, Rettypes);
        func::FuncOp Function = func::FuncOp(
            func::FuncOp::create(Builder.getUnknownLoc(), Name, FuncType));
        SymbolTable::setSymbolVisibility(Function,
                                         SymbolTable::Visibility::Private);

        Glob.getFunctions()[Name] = Function;
        Module->push_back(Function);
        V = Builder.create<func::CallOp>(Loc, Function, ValueRange({V, V2}))
                .getResult(0);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "__nv_dmul_rn")) {
        Value V = GetLLVM(Expr->getArg(0));
        Value V2 = GetLLVM(Expr->getArg(1));
        V = Builder.create<arith::MulFOp>(Loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "__nv_dadd_rn")) {
        Value V = GetLLVM(Expr->getArg(0));
        Value V2 = GetLLVM(Expr->getArg(1));
        V = Builder.create<arith::AddFOp>(Loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "__nv_dsub_rn")) {
        Value V = GetLLVM(Expr->getArg(0));
        Value V2 = GetLLVM(Expr->getArg(1));
        V = Builder.create<arith::SubFOp>(Loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (BuiltinCallee == clang::Builtin::BI__builtin_log2 ||
           BuiltinCallee == clang::Builtin::BI__builtin_log2f ||
           BuiltinCallee == clang::Builtin::BI__builtin_log2l ||
           Sr->getDecl()->getName() == "__nv_log2" ||
           Sr->getDecl()->getName() == "__nv_log2f" ||
           Sr->getDecl()->getName() == "__nv_log2l")) {
        Value V = GetLLVM(Expr->getArg(0));
        V = Builder.create<math::Log2Op>(Loc, V);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (BuiltinCallee == clang::Builtin::BI__builtin_log10 ||
           BuiltinCallee == clang::Builtin::BI__builtin_log10f ||
           BuiltinCallee == clang::Builtin::BI__builtin_log10l ||
           Sr->getDecl()->getName() == "__nv_log10" ||
           Sr->getDecl()->getName() == "__nv_log10f" ||
           Sr->getDecl()->getName() == "__nv_log10l")) {
        Value V = GetLLVM(Expr->getArg(0));
        V = Builder.create<math::Log10Op>(Loc, V);
        return ValueCategory(V, /*isRef*/ false);
      }
    }

  FunctionContext FuncContext = mlirclang::getFuncContext(Function);
  if (auto *Ic = dyn_cast<clang::ImplicitCastExpr>(Expr->getCallee()))
    if (auto *Sr = dyn_cast<clang::DeclRefExpr>(Ic->getSubExpr())) {
      if ((Sr->getDecl()->getIdentifier() &&
           (BuiltinCallee == clang::Builtin::BIfscanf ||
            BuiltinCallee == clang::Builtin::BIscanf ||
            Sr->getDecl()->getName() == "__isoc99_sscanf" ||
            BuiltinCallee == clang::Builtin::BIsscanf)) ||
          (isa<clang::CXXOperatorCallExpr>(Expr) &&
           cast<clang::CXXOperatorCallExpr>(Expr)->getOperator() ==
               clang::OO_GreaterGreater)) {
        const auto *ToCall = EmitCallee(Expr->getCallee());
        auto StrcmpF = Glob.getOrCreateLLVMFunction(ToCall, FuncContext);

        std::vector<Value> Args;
        std::vector<std::pair<Value, Value>> Ops;
        std::map<const void *, size_t> Counts;
        for (auto *A : Expr->arguments()) {
          auto V = GetLLVM(A);
          if (auto Toper = V.getDefiningOp<polygeist::Memref2PointerOp>()) {
            auto T = Toper.getType().cast<LLVM::LLVMPointerType>();
            auto Idx = Counts[T.getAsOpaquePointer()]++;
            auto Aop = allocateBuffer(Idx, T);
            Args.push_back(Aop.getResult());
            Ops.emplace_back(Aop.getResult(), Toper.getSource());
          } else
            Args.push_back(V);
        }
        auto Called = Builder.create<LLVM::CallOp>(Loc, StrcmpF, Args);
        for (auto Pair : Ops) {
          auto Lop = Builder.create<LLVM::LoadOp>(Loc, Pair.first);
          Builder.create<memref::StoreOp>(
              Loc, Lop, Pair.second, std::vector<Value>({getConstantIndex(0)}));
        }
        return ValueCategory(Called.getResult(), /*isReference*/ false);
      }
    }

  if (auto *Ic = dyn_cast<clang::ImplicitCastExpr>(Expr->getCallee()))
    if (auto *Sr = dyn_cast<clang::DeclRefExpr>(Ic->getSubExpr())) {
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "cudaMemcpy" ||
           Sr->getDecl()->getName() == "cudaMemcpyAsync" ||
           Sr->getDecl()->getName() == "cudaMemcpyToSymbol" ||
           BuiltinCallee == clang::Builtin::BImemcpy ||
           BuiltinCallee == clang::Builtin::BI__builtin_memcpy)) {
        auto *DstSub = Expr->getArg(0);
        while (auto *BC = dyn_cast<clang::CastExpr>(DstSub))
          DstSub = BC->getSubExpr();
        auto *SrcSub = Expr->getArg(1);
        while (auto *BC = dyn_cast<clang::CastExpr>(SrcSub))
          SrcSub = BC->getSubExpr();
      }
    }

  const clang::FunctionDecl *Callee = EmitCallee(Expr->getCallee());

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
  if (auto *Ic = dyn_cast<clang::ImplicitCastExpr>(Expr->getCallee()))
    if (auto *Sr = dyn_cast<clang::DeclRefExpr>(Ic->getSubExpr())) {
      StringRef Name;
      if (auto *CC = dyn_cast<clang::CXXConstructorDecl>(Sr->getDecl()))
        Name = Glob.getCGM().getMangledName(
            clang::GlobalDecl(CC, clang::CXXCtorType::Ctor_Complete));
      else if (auto *CC = dyn_cast<clang::CXXDestructorDecl>(Sr->getDecl()))
        Name = Glob.getCGM().getMangledName(
            clang::GlobalDecl(CC, clang::CXXDtorType::Dtor_Complete));
      else if (Sr->getDecl()->hasAttr<clang::CUDAGlobalAttr>())
        Name = Glob.getCGM().getMangledName(
            clang::GlobalDecl(cast<clang::FunctionDecl>(Sr->getDecl()),
                              clang::KernelReferenceKind::Kernel));
      else
        Name = Glob.getCGM().getMangledName(Sr->getDecl());
      if (Funcs.count(Name.str()) || Name.startswith("mkl_") ||
          Name.startswith("MKL_") || Name.startswith("cublas") ||
          Name.startswith("cblas_")) {

        std::vector<Value> Args;
        for (auto *A : Expr->arguments())
          Args.push_back(GetLLVM(A));

        Value Called;

        if (Callee) {
          auto StrcmpF = Glob.getOrCreateLLVMFunction(Callee, FuncContext);
          LLVM::LLVMFunctionType FuncTy = StrcmpF.getFunctionType();
          for (unsigned I = 0; I < FuncTy.getNumParams(); ++I) {
            Type CallerArgType = Args[I].getType();
            if (CallerArgType.isa<LLVM::LLVMPointerType>()) {
              Type CalleeArgType = FuncTy.getParamType(I);
              Args[I] = castToMemSpaceOfType(Args[I], CalleeArgType);
            }
          }
          Called = Builder.create<LLVM::CallOp>(Loc, StrcmpF, Args).getResult();
        } else {
          Args.insert(Args.begin(), GetLLVM(Expr->getCallee()));
          SmallVector<Type> RTs = {
              TypeTranslator.translateType(mlirclang::anonymize(
                  mlirclang::getLLVMType(Expr->getType(), Glob.getCGM())))};
          if (RTs[0].isa<LLVM::LLVMVoidType>())
            RTs.clear();
          Called = Builder.create<LLVM::CallOp>(Loc, RTs, Args).getResult();
        }
        if (!Called)
          return nullptr;
        return ValueCategory(Called, /*isReference*/ Expr->isLValue() ||
                                         Expr->isXValue());
      }
    }

  if (!Callee || Callee->isVariadic()) {
    bool IsReference = Expr->isLValue() || Expr->isXValue();
    std::vector<Value> Args;
    for (auto *A : Expr->arguments())
      Args.push_back(GetLLVM(A));

    Value Called;
    if (Callee) {
      auto StrcmpF = Glob.getOrCreateLLVMFunction(Callee, FuncContext);
      Called = Builder.create<LLVM::CallOp>(Loc, StrcmpF, Args).getResult();
    } else {
      Args.insert(Args.begin(), GetLLVM(Expr->getCallee()));
      auto CT = Expr->getType();
      if (IsReference)
        CT = Glob.getCGM().getContext().getLValueReferenceType(CT);
      SmallVector<Type> RTs = {TypeTranslator.translateType(
          mlirclang::anonymize(mlirclang::getLLVMType(CT, Glob.getCGM())))};

      auto Ft = Args[0]
                    .getType()
                    .cast<LLVM::LLVMPointerType>()
                    .getElementType()
                    .cast<LLVM::LLVMFunctionType>();
      assert(RTs[0] == Ft.getReturnType());
      if (RTs[0].isa<LLVM::LLVMVoidType>())
        RTs.clear();
      Called = Builder.create<LLVM::CallOp>(Loc, RTs, Args).getResult();
    }
    if (IsReference) {
      if (!(Called.getType().isa<LLVM::LLVMPointerType>() ||
            Called.getType().isa<MemRefType>())) {
        Expr->dump();
        Expr->getType()->dump();
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
  const auto IsSyclFunc =
      mlirclang::getNamespaceKind(Callee->getEnclosingNamespaceContext()) !=
      mlirclang::NamespaceKind::Other;
  bool ShouldEmit = !IsSyclFunc;

  std::string MangledName =
      MLIRScanner::getMangledFuncName(*Callee, Glob.getCGM());
  if (GenerateAllSYCLFuncs || !isUnsupportedFunction(MangledName))
    ShouldEmit = true;

  FunctionToEmit F(*Callee, mlirclang::getInputContext(Builder));
  auto ToCall = cast<func::FuncOp>(Glob.getOrCreateMLIRFunction(F, ShouldEmit));

  SmallVector<std::pair<ValueCategory, clang::Expr *>> Args;
  clang::QualType ObjType;

  if (auto *CC = dyn_cast<clang::CXXMemberCallExpr>(Expr)) {
    ValueCategory Obj = Visit(CC->getImplicitObjectArgument());
    ObjType = CC->getObjectType();
    LLVM_DEBUG({
      if (!Obj.val) {
        Function.dump();
        llvm::errs() << " objval: " << Obj.val << "\n";
        Expr->dump();
        CC->getImplicitObjectArgument()->dump();
      }
    });

    if (cast<clang::MemberExpr>(CC->getCallee()->IgnoreParens())->isArrow())
      Obj = Obj.dereference(Builder);

    assert(Obj.val);
    assert(Obj.isReference);
    Args.emplace_back(std::make_pair(Obj, (clang::Expr *)nullptr));
  }

  for (auto *A : Expr->arguments())
    Args.push_back(std::make_pair(Visit(A), A));

  return callHelper(ToCall, ObjType, Args, Expr->getType(),
                    Expr->isLValue() || Expr->isXValue(), Expr, *Callee);
}

std::pair<ValueCategory, bool>
MLIRScanner::emitGPUCallExpr(clang::CallExpr *Expr) {
  auto Loc = getMLIRLocation(Expr->getExprLoc());
  if (auto *Ic = dyn_cast<clang::ImplicitCastExpr>(Expr->getCallee())) {
    if (auto *Sr = dyn_cast<clang::DeclRefExpr>(Ic->getSubExpr())) {
      if (Sr->getDecl()->getIdentifier() &&
          Sr->getDecl()->getName() == "__syncthreads") {
        Builder.create<NVVM::Barrier0Op>(Loc);
        return std::make_pair(ValueCategory(), true);
      }
      if (Sr->getDecl()->getIdentifier() &&
          Sr->getDecl()->getName() == "cudaFuncSetCacheConfig") {
        CGEIST_WARNING(llvm::WithColor::warning()
                       << " Not emitting GPU option: cudaFuncSetCacheConfig\n");
        return std::make_pair(ValueCategory(), true);
      }
      // TODO move free out.
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "free" ||
           Sr->getDecl()->getName() == "cudaFree" ||
           Sr->getDecl()->getName() == "cudaFreeHost")) {

        auto *Sub = Expr->getArg(0);
        while (auto *BC = dyn_cast<clang::CastExpr>(Sub))
          Sub = BC->getSubExpr();
        Value Arg = Visit(Sub).getValue(Builder);

        if (Arg.getType().isa<LLVM::LLVMPointerType>()) {
          const clang::FunctionDecl *Callee = EmitCallee(Expr->getCallee());
          LLVM::LLVMFuncOp StrcmpF = Glob.getOrCreateLLVMFunction(
              Callee, mlirclang::getFuncContext(Function));
          Builder.create<LLVM::CallOp>(
              Loc, StrcmpF,
              ValueRange({Builder.create<LLVM::BitcastOp>(
                  Loc, LLVM::LLVMPointerType::get(Builder.getIntegerType(8)),
                  Arg)}));
        } else {
          Builder.create<memref::DeallocOp>(Loc, Arg);
        }
        if (Sr->getDecl()->getName() == "cudaFree" ||
            Sr->getDecl()->getName() == "cudaFreeHost") {
          auto Ty = Glob.getTypes().getMLIRType(Expr->getType());
          auto Op = Builder.create<arith::ConstantIntOp>(Loc, 0, Ty);
          return std::make_pair(ValueCategory(Op, /*isReference*/ false), true);
        }
        // TODO remove me when the free is removed.
        return std::make_pair(ValueCategory(), true);
      }
      if (Sr->getDecl()->getIdentifier() &&
          (Sr->getDecl()->getName() == "cudaMalloc" ||
           Sr->getDecl()->getName() == "cudaMallocHost" ||
           Sr->getDecl()->getName() == "cudaMallocPitch")) {
        auto *Sub = Expr->getArg(0);
        while (auto *BC = dyn_cast<clang::CastExpr>(Sub))
          Sub = BC->getSubExpr();
        {
          auto Dst = Visit(Sub).getValue(Builder);
          if (auto Omt = Dst.getType().dyn_cast<MemRefType>()) {
            if (auto MT = Omt.getElementType().dyn_cast<MemRefType>()) {
              auto Shape = std::vector<int64_t>(MT.getShape());

              auto ElemSize = getTypeSize(
                  cast<clang::PointerType>(
                      cast<clang::PointerType>(
                          Sub->getType()->getUnqualifiedDesugaredType())
                          ->getPointeeType())
                      ->getPointeeType());
              Value AllocSize;
              if (Sr->getDecl()->getName() == "cudaMallocPitch") {
                Value Width = Visit(Expr->getArg(2)).getValue(Builder);
                Value Height = Visit(Expr->getArg(3)).getValue(Builder);
                // Not changing pitch from provided width here
                // TODO can consider addition alignment considerations
                Visit(Expr->getArg(1))
                    .dereference(Builder)
                    .store(Builder, Width);
                AllocSize = Builder.create<arith::MulIOp>(Loc, Width, Height);
              } else
                AllocSize = Visit(Expr->getArg(1)).getValue(Builder);
              auto IdxType = IndexType::get(Builder.getContext());
              ValueRange Args({Builder.create<arith::DivUIOp>(
                  Loc,
                  Builder.create<arith::IndexCastOp>(Loc, IdxType, AllocSize),
                  ElemSize)});
              auto Alloc = Builder.create<memref::AllocOp>(
                  Loc,
                  (Sr->getDecl()->getName() != "cudaMallocHost" && !CudaLower)
                      ? MemRefType::get(Shape, MT.getElementType(),
                                        MemRefLayoutAttrInterface(),
                                        mlirclang::wrapIntegerMemorySpace(
                                            1, MT.getContext()))
                      : MT,
                  Args);
              ValueCategory(Dst, /*isReference*/ true)
                  .store(Builder,
                         Builder.create<memref::CastOp>(Loc, MT, Alloc));
              auto RetTy = Glob.getTypes().getMLIRType(Expr->getType());
              return std::make_pair(
                  ValueCategory(
                      Builder.create<arith::ConstantIntOp>(Loc, 0, RetTy),
                      /*isReference*/ false),
                  true);
            }
          }
        }
      }
    }

    auto CreateBlockIdOp = [&](gpu::Dimension Str, Type MLIRType) -> Value {
      return Builder.create<arith::IndexCastOp>(
          Loc, MLIRType,
          Builder.create<gpu::BlockIdOp>(
              Loc, IndexType::get(Builder.getContext()), Str));
    };

    auto CreateBlockDimOp = [&](gpu::Dimension Str, Type MLIRType) -> Value {
      return Builder.create<arith::IndexCastOp>(
          Loc, MLIRType,
          Builder.create<gpu::BlockDimOp>(
              Loc, IndexType::get(Builder.getContext()), Str));
    };

    auto CreateThreadIdOp = [&](gpu::Dimension Str, Type MLIRType) -> Value {
      return Builder.create<arith::IndexCastOp>(
          Loc, MLIRType,
          Builder.create<gpu::ThreadIdOp>(
              Loc, IndexType::get(Builder.getContext()), Str));
    };

    auto CreateGridDimOp = [&](gpu::Dimension Str, Type MLIRType) -> Value {
      return Builder.create<arith::IndexCastOp>(
          Loc, MLIRType,
          Builder.create<gpu::GridDimOp>(
              Loc, IndexType::get(Builder.getContext()), Str));
    };

    if (auto *ME = dyn_cast<clang::MemberExpr>(Ic->getSubExpr())) {
      auto MemberName = ME->getMemberDecl()->getName();

      if (auto *Sr2 = dyn_cast<clang::OpaqueValueExpr>(ME->getBase())) {
        if (auto *Sr = dyn_cast<clang::DeclRefExpr>(Sr2->getSourceExpr())) {
          if (Sr->getDecl()->getName() == "blockIdx") {
            auto MLIRType = Glob.getTypes().getMLIRType(Expr->getType());
            if (MemberName == "__fetch_builtin_x") {
              return std::make_pair(
                  ValueCategory(CreateBlockIdOp(gpu::Dimension::x, MLIRType),
                                /*isReference*/ false),
                  true);
            }
            if (MemberName == "__fetch_builtin_y") {
              return std::make_pair(
                  ValueCategory(CreateBlockIdOp(gpu::Dimension::y, MLIRType),
                                /*isReference*/ false),
                  true);
            }
            if (MemberName == "__fetch_builtin_z") {
              return std::make_pair(
                  ValueCategory(CreateBlockIdOp(gpu::Dimension::z, MLIRType),
                                /*isReference*/ false),
                  true);
            }
          }
          if (Sr->getDecl()->getName() == "blockDim") {
            auto MLIRType = Glob.getTypes().getMLIRType(Expr->getType());
            if (MemberName == "__fetch_builtin_x") {
              return std::make_pair(
                  ValueCategory(CreateBlockDimOp(gpu::Dimension::x, MLIRType),
                                /*isReference*/ false),
                  true);
            }
            if (MemberName == "__fetch_builtin_y") {
              return std::make_pair(
                  ValueCategory(CreateBlockDimOp(gpu::Dimension::y, MLIRType),
                                /*isReference*/ false),
                  true);
            }
            if (MemberName == "__fetch_builtin_z") {
              return std::make_pair(
                  ValueCategory(CreateBlockDimOp(gpu::Dimension::z, MLIRType),
                                /*isReference*/ false),
                  true);
            }
          }
          if (Sr->getDecl()->getName() == "threadIdx") {
            auto MLIRType = Glob.getTypes().getMLIRType(Expr->getType());
            if (MemberName == "__fetch_builtin_x") {
              return std::make_pair(
                  ValueCategory(CreateThreadIdOp(gpu::Dimension::x, MLIRType),
                                /*isReference*/ false),
                  true);
            }
            if (MemberName == "__fetch_builtin_y") {
              return std::make_pair(
                  ValueCategory(CreateThreadIdOp(gpu::Dimension::y, MLIRType),
                                /*isReference*/ false),
                  true);
            }
            if (MemberName == "__fetch_builtin_z") {
              return std::make_pair(
                  ValueCategory(CreateThreadIdOp(gpu::Dimension::z, MLIRType),
                                /*isReference*/ false),
                  true);
            }
          }
          if (Sr->getDecl()->getName() == "gridDim") {
            auto MLIRType = Glob.getTypes().getMLIRType(Expr->getType());
            if (MemberName == "__fetch_builtin_x") {
              return std::make_pair(
                  ValueCategory(CreateGridDimOp(gpu::Dimension::x, MLIRType),
                                /*isReference*/ false),
                  true);
            }
            if (MemberName == "__fetch_builtin_y") {
              return std::make_pair(
                  ValueCategory(CreateGridDimOp(gpu::Dimension::y, MLIRType),
                                /*isReference*/ false),
                  true);
            }
            if (MemberName == "__fetch_builtin_z") {
              return std::make_pair(
                  ValueCategory(CreateGridDimOp(gpu::Dimension::z, MLIRType),
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
MLIRScanner::emitBuiltinOps(clang::CallExpr *Expr) {
  if (auto *Ic = dyn_cast<clang::ImplicitCastExpr>(Expr->getCallee()))
    if (auto *Sr = dyn_cast<clang::DeclRefExpr>(Ic->getSubExpr()))
      if (Sr->getDecl()->getIdentifier() &&
          Sr->getDecl()->getName() == "__log2f") {
        std::vector<Value> Args;
        for (auto *A : Expr->arguments())
          Args.push_back(Visit(A).getValue(Builder));

        return std::make_pair(
            ValueCategory(Builder.create<math::Log2Op>(Loc, Args[0]),
                          /*isReference*/ false),
            true);
      }

  std::vector<Value> Args;
  auto VisitArgs = [&]() {
    assert(Args.empty() && "Expecting empty args");
    for (auto *A : Expr->arguments())
      Args.push_back(Visit(A).getValue(Builder));
  };
  Optional<Value> V = std::nullopt;
  switch (Expr->getBuiltinCallee()) {
  case clang::Builtin::BIceil: {
    VisitArgs();
    V = Builder.create<math::CeilOp>(Loc, Args[0]);
  } break;
  case clang::Builtin::BIcos: {
    VisitArgs();
    V = Builder.create<math::CosOp>(Loc, Args[0]);
  } break;
  case clang::Builtin::BIexp:
  case clang::Builtin::BIexpf: {
    VisitArgs();
    V = Builder.create<math::ExpOp>(Loc, Args[0]);
  } break;
  case clang::Builtin::BIlog: {
    VisitArgs();
    V = Builder.create<math::LogOp>(Loc, Args[0]);
  } break;
  case clang::Builtin::BIsin: {
    VisitArgs();
    V = Builder.create<math::SinOp>(Loc, Args[0]);
  } break;
  case clang::Builtin::BIsqrt:
  case clang::Builtin::BIsqrtf: {
    VisitArgs();
    V = Builder.create<math::SqrtOp>(Loc, Args[0]);
  } break;
  case clang::Builtin::BI__builtin_atanh:
  case clang::Builtin::BI__builtin_atanhf:
  case clang::Builtin::BI__builtin_atanhl: {
    VisitArgs();
    V = Builder.create<math::AtanOp>(Loc, Args[0]);
  } break;
  case clang::Builtin::BI__builtin_copysign:
  case clang::Builtin::BI__builtin_copysignf:
  case clang::Builtin::BI__builtin_copysignl: {
    VisitArgs();
    V = Builder.create<LLVM::CopySignOp>(Loc, Args[0], Args[1]);
  } break;
  case clang::Builtin::BI__builtin_exp2:
  case clang::Builtin::BI__builtin_exp2f:
  case clang::Builtin::BI__builtin_exp2l: {
    VisitArgs();
    V = Builder.create<math::Exp2Op>(Loc, Args[0]);
  } break;
  case clang::Builtin::BI__builtin_expm1:
  case clang::Builtin::BI__builtin_expm1f:
  case clang::Builtin::BI__builtin_expm1l: {
    VisitArgs();
    V = Builder.create<math::ExpM1Op>(Loc, Args[0]);
  } break;
  case clang::Builtin::BI__builtin_fma:
  case clang::Builtin::BI__builtin_fmaf:
  case clang::Builtin::BI__builtin_fmal: {
    VisitArgs();
    V = Builder.create<LLVM::FMAOp>(Loc, Args[0], Args[1], Args[2]);
  } break;
  case clang::Builtin::BI__builtin_fmax:
  case clang::Builtin::BI__builtin_fmaxf:
  case clang::Builtin::BI__builtin_fmaxl: {
    VisitArgs();
    V = Builder.create<LLVM::MaxNumOp>(Loc, Args[0], Args[1]);
  } break;
  case clang::Builtin::BI__builtin_fmin:
  case clang::Builtin::BI__builtin_fminf:
  case clang::Builtin::BI__builtin_fminl: {
    VisitArgs();
    V = Builder.create<LLVM::MinNumOp>(Loc, Args[0], Args[1]);
  } break;
  case clang::Builtin::BI__builtin_log1p:
  case clang::Builtin::BI__builtin_log1pf:
  case clang::Builtin::BI__builtin_log1pl: {
    VisitArgs();
    V = Builder.create<math::Log1pOp>(Loc, Args[0]);
  } break;
  case clang::Builtin::BI__builtin_pow:
  case clang::Builtin::BI__builtin_powf:
  case clang::Builtin::BI__builtin_powl: {
    VisitArgs();
    V = Builder.create<math::PowFOp>(Loc, Args[0], Args[1]);
  } break;
  case clang::Builtin::BI__builtin_assume: {
    VisitArgs();
    V = Builder.create<LLVM::AssumeOp>(Loc, Args[0])->getResult(0);
  } break;
  case clang::Builtin::BI__builtin_isgreater: {
    VisitArgs();
    auto PostTy =
        Glob.getTypes().getMLIRType(Expr->getType()).cast<IntegerType>();
    V = Builder.create<arith::ExtUIOp>(
        Loc, PostTy,
        Builder.create<arith::CmpFOp>(Loc, arith::CmpFPredicate::OGT, Args[0],
                                      Args[1]));
  } break;
  case clang::Builtin::BI__builtin_isgreaterequal: {
    VisitArgs();
    auto PostTy =
        Glob.getTypes().getMLIRType(Expr->getType()).cast<IntegerType>();
    V = Builder.create<arith::ExtUIOp>(
        Loc, PostTy,
        Builder.create<arith::CmpFOp>(Loc, arith::CmpFPredicate::OGE, Args[0],
                                      Args[1]));
  } break;
  case clang::Builtin::BI__builtin_isless: {
    VisitArgs();
    auto PostTy =
        Glob.getTypes().getMLIRType(Expr->getType()).cast<IntegerType>();
    V = Builder.create<arith::ExtUIOp>(
        Loc, PostTy,
        Builder.create<arith::CmpFOp>(Loc, arith::CmpFPredicate::OLT, Args[0],
                                      Args[1]));
  } break;
  case clang::Builtin::BI__builtin_islessequal: {
    VisitArgs();
    auto PostTy =
        Glob.getTypes().getMLIRType(Expr->getType()).cast<IntegerType>();
    V = Builder.create<arith::ExtUIOp>(
        Loc, PostTy,
        Builder.create<arith::CmpFOp>(Loc, arith::CmpFPredicate::OLE, Args[0],
                                      Args[1]));
  } break;
  case clang::Builtin::BI__builtin_islessgreater: {
    VisitArgs();
    auto PostTy =
        Glob.getTypes().getMLIRType(Expr->getType()).cast<IntegerType>();
    V = Builder.create<arith::ExtUIOp>(
        Loc, PostTy,
        Builder.create<arith::CmpFOp>(Loc, arith::CmpFPredicate::ONE, Args[0],
                                      Args[1]));
  } break;
  case clang::Builtin::BI__builtin_isunordered: {
    VisitArgs();
    auto PostTy =
        Glob.getTypes().getMLIRType(Expr->getType()).cast<IntegerType>();
    V = Builder.create<arith::ExtUIOp>(
        Loc, PostTy,
        Builder.create<arith::CmpFOp>(Loc, arith::CmpFPredicate::UNO, Args[0],
                                      Args[1]));
  } break;
  case clang::Builtin::BImemmove:
  case clang::Builtin::BI__builtin_memmove: {
    VisitArgs();
    Builder.create<LLVM::MemmoveOp>(
        Loc, Args[0], Args[1], Args[2],
        /*isVolatile*/ Builder.create<arith::ConstantIntOp>(Loc, false, 1));
    V = Args[0];
  } break;
  case clang::Builtin::BImemset:
  case clang::Builtin::BI__builtin_memset: {
    VisitArgs();
    Builder.create<LLVM::MemsetOp>(
        Loc, Args[0],
        Builder.create<arith::TruncIOp>(Loc, Builder.getI8Type(), Args[1]),
        Args[2],
        /*isVolatile*/ Builder.create<arith::ConstantIntOp>(Loc, false, 1));
    V = Args[0];
  } break;
  case clang::Builtin::BImemcpy:
  case clang::Builtin::BI__builtin_memcpy: {
    VisitArgs();
    Builder.create<LLVM::MemcpyOp>(
        Loc, Args[0], Args[1], Args[2],
        /*isVolatile*/ Builder.create<arith::ConstantIntOp>(Loc, false, 1));
    V = Args[0];
  } break;
  }
  if (V.has_value())
    return std::make_pair(ValueCategory(V.value(),
                                        /*isReference*/ false),
                          true);

  return std::make_pair(ValueCategory(), false);
}
