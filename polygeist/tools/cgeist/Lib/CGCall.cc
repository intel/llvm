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

extern llvm::cl::opt<bool> GenerateAllSYCLFuncs;

/******************************************************************************/
/*                           Utility Functions                                */
/******************************************************************************/

/// Try to typecast the caller arg of type MemRef to fit the corresponding
/// callee arg type. We only deal with the cast where src and dst have the same
/// shape size and elem type, and just the first shape differs: src has -1 and
/// dst has a constant integer.
static mlir::Value castCallerMemRefArg(mlir::Value callerArg,
                                       mlir::Type calleeArgType,
                                       mlir::OpBuilder &b) {
  mlir::OpBuilder::InsertionGuard guard(b);
  mlir::Type callerArgType = callerArg.getType();

  if (MemRefType dstTy = calleeArgType.dyn_cast_or_null<MemRefType>()) {
    MemRefType srcTy = callerArgType.dyn_cast<MemRefType>();
    if (srcTy && dstTy.getElementType() == srcTy.getElementType() &&
        dstTy.getMemorySpace() == srcTy.getMemorySpace()) {
      auto srcShape = srcTy.getShape();
      auto dstShape = dstTy.getShape();

      if (srcShape.size() == dstShape.size() && !srcShape.empty() &&
          srcShape[0] == -1 &&
          std::equal(std::next(srcShape.begin()), srcShape.end(),
                     std::next(dstShape.begin()))) {
        b.setInsertionPointAfterValue(callerArg);

        return b.create<mlir::memref::CastOp>(callerArg.getLoc(), calleeArgType,
                                              callerArg);
      }
    }
  }

  // Return the original value when casting fails.
  return callerArg;
}

/// Typecast the caller args to match the callee's signature. Mismatches that
/// cannot be resolved by given rules won't raise exceptions, e.g., if the
/// expected type for an arg is memref<10xi8> while the provided is
/// memref<20xf32>, we will simply ignore the case in this function and wait for
/// the rest of the pipeline to detect it.
static void castCallerArgs(mlir::func::FuncOp callee,
                           llvm::SmallVectorImpl<mlir::Value> &args,
                           mlir::OpBuilder &b) {
  mlir::FunctionType funcTy = callee.getFunctionType();
  assert(args.size() == funcTy.getNumInputs() &&
         "The caller arguments should have the same size as the number of "
         "callee arguments as the interface.");

  LLVM_DEBUG({
    llvm::dbgs() << "funcTy: " << funcTy << "\n";
    llvm::dbgs() << "args: \n";
    for (const mlir::Value &arg : args)
      llvm::dbgs().indent(2) << arg << "\n";
  });

  for (unsigned i = 0; i < args.size(); ++i) {
    mlir::Type calleeArgType = funcTy.getInput(i);
    mlir::Type callerArgType = args[i].getType();

    if (calleeArgType == callerArgType)
      continue;

    if (calleeArgType.isa<MemRefType>())
      args[i] = castCallerMemRefArg(args[i], calleeArgType, b);
    assert(calleeArgType == args[i].getType() && "Callsite argument mismatch");
  }
}

/******************************************************************************/
/*                               MLIRScanner                                  */
/******************************************************************************/

ValueCategory MLIRScanner::CallHelper(
    mlir::func::FuncOp tocall, QualType objType,
    ArrayRef<std::pair<ValueCategory, clang::Expr *>> arguments,
    QualType retType, bool retReference, clang::Expr *expr,
    const FunctionDecl *callee) {
  SmallVector<mlir::Value, 4> args;
  auto fnType = tocall.getFunctionType();
  const clang::CodeGen::CGFunctionInfo &FI =
      Glob.GetOrCreateCGFunctionInfo(callee);
  auto FIArgs = FI.arguments();

  size_t i = 0;
  // map from declaration name to mlir::value
  std::map<std::string, mlir::Value> mapFuncOperands;

  for (auto pair : arguments) {

    ValueCategory arg = std::get<0>(pair);
    clang::Expr *a = std::get<1>(pair);

    LLVM_DEBUG({
      if (!arg.val) {
        expr->dump();
        a->dump();
      }
    });
    assert(arg.val && "expect not null");

    if (auto *ice = dyn_cast_or_null<ImplicitCastExpr>(a))
      if (auto *dre = dyn_cast<DeclRefExpr>(ice->getSubExpr()))
        mapFuncOperands.insert(
            make_pair(dre->getDecl()->getName().str(), arg.val));

    if (i >= fnType.getInputs().size() || (i != 0 && a == nullptr)) {
      LLVM_DEBUG({
        expr->dump();
        tocall.dump();
        fnType.dump();
        for (auto a : arguments)
          std::get<1>(a)->dump();
      });
      assert(false && "too many arguments in calls");
    }

    bool isReference =
        (i == 0 && a == nullptr) || a->isLValue() || a->isXValue();

    bool isArray = false;
    QualType aType = (i == 0 && a == nullptr) ? objType : a->getType();
    mlir::Type expectedType = Glob.getTypes().getMLIRType(aType, &isArray);

    LLVM_DEBUG({
      llvm::dbgs() << "aType: " << aType << "\n";
      llvm::dbgs() << "aType addrspace: "
                   << Glob.getCGM().getContext().getTargetAddressSpace(aType)
                   << "\n";
      llvm::dbgs() << "expectedType: " << expectedType << "\n";
    });

    if (auto PT = arg.val.getType().dyn_cast<LLVM::LLVMPointerType>()) {
      if (PT.getAddressSpace() == 5)
        arg.val = builder.create<LLVM::AddrSpaceCastOp>(
            loc, LLVM::LLVMPointerType::get(PT.getElementType(), 0), arg.val);
    }

    mlir::Value val = nullptr;
    if (!isReference) {
      if (isArray) {
        LLVM_DEBUG({
          if (!arg.isReference) {
            expr->dump();
            a->dump();
            llvm::errs() << " v: " << arg.val << "\n";
          }
        });
        assert(arg.isReference);

        auto mt =
            Glob.getTypes()
                .getMLIRType(
                    Glob.getCGM().getContext().getLValueReferenceType(aType))
                .cast<MemRefType>();

        LLVM_DEBUG({
          llvm::dbgs() << "mt: " << mt << "\n";
          llvm::dbgs() << "getLValueReferenceType(aType): "
                       << Glob.getCGM().getContext().getLValueReferenceType(
                              aType)
                       << "\n";
        });

        auto shape = std::vector<int64_t>(mt.getShape());
        assert(shape.size() == 2);

        auto pshape = shape[0];
        if (pshape == -1)
          shape[0] = 1;

        OpBuilder abuilder(builder.getContext());
        abuilder.setInsertionPointToStart(allocationScope);
        auto alloc = abuilder.create<mlir::memref::AllocaOp>(
            loc, mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace()));
        ValueCategory(alloc, /*isRef*/ true)
            .store(builder, arg, /*isArray*/ isArray);
        shape[0] = pshape;
        val = builder.create<mlir::memref::CastOp>(
            loc,
            mlir::MemRefType::get(shape, mt.getElementType(),
                                  MemRefLayoutAttrInterface(),
                                  mt.getMemorySpace()),
            alloc);
      } else {
        if (FIArgs[i].info.getKind() == clang::CodeGen::ABIArgInfo::Indirect ||
            FIArgs[i].info.getKind() ==
                clang::CodeGen::ABIArgInfo::IndirectAliased) {
          OpBuilder abuilder(builder.getContext());
          abuilder.setInsertionPointToStart(allocationScope);
          auto Ty = Glob.getTypes().getPointerOrMemRefType(
              arg.getValue(builder).getType(),
              Glob.getCGM().getDataLayout().getAllocaAddrSpace(),
              /*IsAlloc*/ true);
          if (auto MemRefTy = Ty.dyn_cast<mlir::MemRefType>()) {
            val = abuilder.create<mlir::memref::AllocaOp>(loc, MemRefTy);
            val = abuilder.create<mlir::memref::CastOp>(
                loc, mlir::MemRefType::get(-1, arg.getValue(builder).getType()),
                val);
          } else {
            val = abuilder.create<mlir::LLVM::AllocaOp>(
                loc, Ty, abuilder.create<arith::ConstantIntOp>(loc, 1, 64), 0);
          }
          ValueCategory(val, /*isRef*/ true)
              .store(builder, arg.getValue(builder));
        } else
          val = arg.getValue(builder);

        if (val.getType().isa<LLVM::LLVMPointerType>() &&
            expectedType.isa<MemRefType>()) {
          val = builder.create<polygeist::Pointer2MemrefOp>(loc, expectedType,
                                                            val);
        }
        if (auto prevTy = val.getType().dyn_cast<mlir::IntegerType>()) {
          auto ipostTy = expectedType.cast<mlir::IntegerType>();
          if (prevTy != ipostTy)
            val = builder.create<arith::TruncIOp>(loc, ipostTy, val);
        }
      }
    } else {
      assert(arg.isReference);

      expectedType = Glob.getTypes().getMLIRType(
          Glob.getCGM().getContext().getLValueReferenceType(aType));

      val = arg.val;
      if (val.getType().isa<LLVM::LLVMPointerType>() &&
          expectedType.isa<MemRefType>())
        val =
            builder.create<polygeist::Pointer2MemrefOp>(loc, expectedType, val);

      val = castToMemSpaceOfType(val, expectedType);
    }
    assert(val);
    args.push_back(val);
    i++;
  }

  // handle lowerto pragma.
  if (LTInfo.SymbolTable.count(tocall.getName())) {
    SmallVector<mlir::Value> inputOperands;
    SmallVector<mlir::Value> outputOperands;
    for (StringRef input : LTInfo.InputSymbol)
      if (mapFuncOperands.find(input.str()) != mapFuncOperands.end())
        inputOperands.push_back(mapFuncOperands[input.str()]);
    for (StringRef output : LTInfo.OutputSymbol)
      if (mapFuncOperands.find(output.str()) != mapFuncOperands.end())
        outputOperands.push_back(mapFuncOperands[output.str()]);

    if (inputOperands.size() == 0)
      inputOperands.append(args);

    return ValueCategory(mlirclang::replaceFuncByOperation(
                             tocall, LTInfo.SymbolTable[tocall.getName()],
                             builder, inputOperands, outputOperands)
                             ->getResult(0),
                         /*isReference=*/false);
  }

  bool isArrayReturn = false;
  if (!retReference)
    Glob.getTypes().getMLIRType(retType, &isArrayReturn);

  mlir::Value alloc;
  if (isArrayReturn) {
    auto mt =
        Glob.getTypes()
            .getMLIRType(
                Glob.getCGM().getContext().getLValueReferenceType(retType))
            .cast<MemRefType>();

    auto shape = std::vector<int64_t>(mt.getShape());
    assert(shape.size() == 2);

    auto pshape = shape[0];
    if (pshape == -1)
      shape[0] = 1;

    OpBuilder abuilder(builder.getContext());
    abuilder.setInsertionPointToStart(allocationScope);
    alloc = abuilder.create<mlir::memref::AllocaOp>(
        loc, mlir::MemRefType::get(shape, mt.getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   mt.getMemorySpace()));
    shape[0] = pshape;
    alloc = builder.create<mlir::memref::CastOp>(
        loc,
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace()),
        alloc);
    args.push_back(alloc);
  }

  if (auto *CU = dyn_cast<CUDAKernelCallExpr>(expr)) {
    auto l0 = Visit(CU->getConfig()->getArg(0));
    assert(l0.isReference);
    mlir::Value blocks[3];
    for (int i = 0; i < 3; i++) {
      mlir::Value val = l0.val;
      if (auto MT = val.getType().dyn_cast<MemRefType>()) {
        mlir::Value idx[] = {getConstantIndex(0), getConstantIndex(i)};
        assert(MT.getShape().size() == 2);
        blocks[i] = builder.create<IndexCastOp>(
            loc, mlir::IndexType::get(builder.getContext()),
            builder.create<mlir::memref::LoadOp>(loc, val, idx));
      } else {
        mlir::Value idx[] = {builder.create<arith::ConstantIntOp>(loc, 0, 32),
                             builder.create<arith::ConstantIntOp>(loc, i, 32)};
        auto PT = val.getType().cast<LLVM::LLVMPointerType>();
        auto ET = PT.getElementType().cast<LLVM::LLVMStructType>().getBody()[i];
        blocks[i] = builder.create<IndexCastOp>(
            loc, mlir::IndexType::get(builder.getContext()),
            builder.create<LLVM::LoadOp>(
                loc,
                builder.create<LLVM::GEPOp>(
                    loc, LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
                    val, idx)));
      }
    }

    auto t0 = Visit(CU->getConfig()->getArg(1));
    assert(t0.isReference);
    mlir::Value threads[3];
    for (int i = 0; i < 3; i++) {
      mlir::Value val = t0.val;
      if (auto MT = val.getType().dyn_cast<MemRefType>()) {
        mlir::Value idx[] = {getConstantIndex(0), getConstantIndex(i)};
        assert(MT.getShape().size() == 2);
        threads[i] = builder.create<IndexCastOp>(
            loc, mlir::IndexType::get(builder.getContext()),
            builder.create<mlir::memref::LoadOp>(loc, val, idx));
      } else {
        mlir::Value idx[] = {builder.create<arith::ConstantIntOp>(loc, 0, 32),
                             builder.create<arith::ConstantIntOp>(loc, i, 32)};
        auto PT = val.getType().cast<LLVM::LLVMPointerType>();
        auto ET = PT.getElementType().cast<LLVM::LLVMStructType>().getBody()[i];
        threads[i] = builder.create<IndexCastOp>(
            loc, mlir::IndexType::get(builder.getContext()),
            builder.create<LLVM::LoadOp>(
                loc,
                builder.create<LLVM::GEPOp>(
                    loc, LLVM::LLVMPointerType::get(ET, PT.getAddressSpace()),
                    val, idx)));
      }
    }
    mlir::Value stream = nullptr;
    SmallVector<mlir::Value, 1> asyncDependencies;
    if (3 < CU->getConfig()->getNumArgs() &&
        !isa<CXXDefaultArgExpr>(CU->getConfig()->getArg(3))) {
      stream = Visit(CU->getConfig()->getArg(3)).getValue(builder);
      stream = builder.create<polygeist::StreamToTokenOp>(
          loc, builder.getType<gpu::AsyncTokenType>(), stream);
      assert(stream);
      asyncDependencies.push_back(stream);
    }
    auto op = builder.create<mlir::gpu::LaunchOp>(
        loc, blocks[0], blocks[1], blocks[2], threads[0], threads[1],
        threads[2],
        /*dynamic shmem size*/ nullptr,
        /*token type*/ stream ? stream.getType() : nullptr,
        /*dependencies*/ asyncDependencies);
    auto oldpoint = builder.getInsertionPoint();
    auto *oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&op.getRegion().front());
    builder.create<CallOp>(loc, tocall, args);
    builder.create<gpu::TerminatorOp>(loc);
    builder.setInsertionPoint(oldblock, oldpoint);
    return nullptr;
  }

  // Try to rescue some mismatched types.
  castCallerArgs(tocall, args, builder);

  /// Try to emit SYCL operations before creating a CallOp
  mlir::Operation *op = EmitSYCLOps(expr, args);
  if (!op) {
    op = builder.create<CallOp>(loc, tocall, args);
  }

  if (isArrayReturn) {
    // TODO remedy return
    if (retReference)
      expr->dump();
    assert(!retReference);
    return ValueCategory(alloc, /*isReference*/ true);
  } else if (op->getNumResults()) {
    return ValueCategory(op->getResult(0),
                         /*isReference*/ retReference);
  } else
    return nullptr;
  llvm::errs() << "do not support indirecto call of " << tocall << "\n";
  assert(0 && "no indirect");
}

std::pair<ValueCategory, bool>
MLIRScanner::EmitClangBuiltinCallExpr(clang::CallExpr *expr) {
  switch (expr->getBuiltinCallee()) {
  case clang::Builtin::BImove:
  case clang::Builtin::BImove_if_noexcept:
  case clang::Builtin::BIforward:
  case clang::Builtin::BIas_const: {
    auto V = Visit(expr->getArg(0));
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

  auto valEmitted = EmitGPUCallExpr(expr);
  if (valEmitted.second)
    return valEmitted.first;

  valEmitted = EmitBuiltinOps(expr);
  if (valEmitted.second)
    return valEmitted.first;

  valEmitted = EmitClangBuiltinCallExpr(expr);
  if (valEmitted.second)
    return valEmitted.first;

  if (auto *oc = dyn_cast<CXXOperatorCallExpr>(expr)) {
    if (oc->getOperator() == clang::OO_EqualEqual) {
      if (auto *lhs = dyn_cast<CXXTypeidExpr>(expr->getArg(0))) {
        if (auto *rhs = dyn_cast<CXXTypeidExpr>(expr->getArg(1))) {
          QualType LT = lhs->isTypeOperand()
                            ? lhs->getTypeOperand(Glob.getCGM().getContext())
                            : lhs->getExprOperand()->getType();
          QualType RT = rhs->isTypeOperand()
                            ? rhs->getTypeOperand(Glob.getCGM().getContext())
                            : rhs->getExprOperand()->getType();
          llvm::Constant *LC = Glob.getCGM().GetAddrOfRTTIDescriptor(LT);
          llvm::Constant *RC = Glob.getCGM().GetAddrOfRTTIDescriptor(RT);
          auto postTy = Glob.getTypes()
                            .getMLIRType(expr->getType())
                            .cast<mlir::IntegerType>();
          return ValueCategory(
              builder.create<arith::ConstantIntOp>(loc, LC == RC, postTy),
              false);
        }
      }
    }
  }

  if (auto *oc = dyn_cast<CXXMemberCallExpr>(expr)) {
    if (auto *lhs = dyn_cast<CXXTypeidExpr>(oc->getImplicitObjectArgument())) {
      expr->getCallee()->dump();
      if (auto *ic = dyn_cast<MemberExpr>(expr->getCallee()))
        if (auto *sr = dyn_cast<NamedDecl>(ic->getMemberDecl())) {
          if (sr->getIdentifier() && sr->getName() == "name") {
            QualType LT = lhs->isTypeOperand()
                              ? lhs->getTypeOperand(Glob.getCGM().getContext())
                              : lhs->getExprOperand()->getType();
            llvm::Constant *LC = Glob.getCGM().GetAddrOfRTTIDescriptor(LT);
            while (auto *CE = dyn_cast<llvm::ConstantExpr>(LC))
              LC = CE->getOperand(0);
            std::string val = cast<llvm::GlobalVariable>(LC)->getName().str();
            return CommonArrayToPointer(ValueCategory(
                Glob.GetOrCreateGlobalLLVMString(loc, builder, val),
                /*isReference*/ true));
          }
        }
    }
  }

  if (auto *ps = dyn_cast<CXXPseudoDestructorExpr>(expr->getCallee())) {
    return Visit(ps);
  }

  if (auto *ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "atomicAdd" ||
           sr->getDecl()->getName() == "atomicOr" ||
           sr->getDecl()->getName() == "atomicAnd")) {
        std::vector<ValueCategory> args;
        for (auto *a : expr->arguments()) {
          args.push_back(Visit(a));
        }
        auto a0 = args[0].getValue(builder);
        auto a1 = args[1].getValue(builder);
        AtomicRMWKind op;
        LLVM::AtomicBinOp lop;
        if (sr->getDecl()->getName() == "atomicAdd") {
          if (a1.getType().isa<mlir::IntegerType>()) {
            op = AtomicRMWKind::addi;
            lop = LLVM::AtomicBinOp::add;
          } else {
            op = AtomicRMWKind::addf;
            lop = LLVM::AtomicBinOp::fadd;
          }
        } else if (sr->getDecl()->getName() == "atomicOr") {
          op = AtomicRMWKind::ori;
          lop = LLVM::AtomicBinOp::_or;
        } else if (sr->getDecl()->getName() == "atomicAnd") {
          op = AtomicRMWKind::andi;
          lop = LLVM::AtomicBinOp::_and;
        } else
          assert(0);

        if (a0.getType().isa<MemRefType>())
          return ValueCategory(
              builder.create<memref::AtomicRMWOp>(
                  loc, a1.getType(), op, a1, a0,
                  std::vector<mlir::Value>({getConstantIndex(0)})),
              /*isReference*/ false);
        else
          return ValueCategory(
              builder.create<LLVM::AtomicRMWOp>(loc, a1.getType(), lop, a0, a1,
                                                LLVM::AtomicOrdering::acq_rel),
              /*isReference*/ false);
      }
    }

  mlir::LLVM::TypeFromLLVMIRTranslator typeTranslator(*module->getContext());

  auto getLLVM = [&](Expr *E) -> mlir::Value {
    auto sub = Visit(E);
    if (!sub.val) {
      expr->dump();
      E->dump();
    }
    assert(sub.val);

    bool isReference = E->isLValue() || E->isXValue();
    if (isReference) {
      assert(sub.isReference);
      mlir::Value val = sub.val;
      if (auto mt = val.getType().dyn_cast<MemRefType>()) {
        val = builder.create<polygeist::Memref2PointerOp>(
            loc,
            LLVM::LLVMPointerType::get(mt.getElementType(),
                                       mt.getMemorySpaceAsInt()),
            val);
      }
      return val;
    }

    bool isArray = false;
    Glob.getTypes().getMLIRType(E->getType(), &isArray);

    if (isArray) {
      assert(sub.isReference);
      auto mt =
          Glob.getTypes()
              .getMLIRType(Glob.getCGM().getContext().getLValueReferenceType(
                  E->getType()))
              .cast<MemRefType>();
      auto shape = std::vector<int64_t>(mt.getShape());
      assert(shape.size() == 2);

      OpBuilder abuilder(builder.getContext());
      abuilder.setInsertionPointToStart(allocationScope);
      auto one = abuilder.create<ConstantIntOp>(loc, 1, 64);
      auto alloc = abuilder.create<mlir::LLVM::AllocaOp>(
          loc,
          LLVM::LLVMPointerType::get(
              typeTranslator.translateType(
                  anonymize(getLLVMType(E->getType(), Glob.getCGM()))),
              0),
          one, 0);
      ValueCategory(alloc, /*isRef*/ true)
          .store(builder, sub, /*isArray*/ isArray);
      sub = ValueCategory(alloc, /*isRef*/ true);
    }
    auto val = sub.getValue(builder);
    if (auto mt = val.getType().dyn_cast<MemRefType>()) {
      auto nt = typeTranslator
                    .translateType(
                        anonymize(getLLVMType(E->getType(), Glob.getCGM())))
                    .cast<LLVM::LLVMPointerType>();
      assert(nt.getAddressSpace() == mt.getMemorySpaceAsInt() &&
             "val does not have the same memory space as nt");
      val = builder.create<polygeist::Memref2PointerOp>(loc, nt, val);
    }
    return val;
  };

  switch (expr->getBuiltinCallee()) {
  case clang::Builtin::BI__builtin_assume:
    mlir::Value V0 = getLLVM(expr->getArg(0));
    return ValueCategory(builder.create<LLVM::AssumeOp>(loc, V0)->getResult(0),
                         false);
  };

  if (auto *ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__powf" ||
           sr->getDecl()->getName() == "pow" ||
           sr->getDecl()->getName() == "__nv_pow" ||
           sr->getDecl()->getName() == "__nv_powf" ||
           sr->getDecl()->getName() == "__powi" ||
           sr->getDecl()->getName() == "powi" ||
           sr->getDecl()->getName() == "__nv_powi" ||
           sr->getDecl()->getName() == "__nv_powi" ||
           sr->getDecl()->getName() == "powf")) {
        mlir::Type mlirType = Glob.getTypes().getMLIRType(expr->getType());
        std::vector<mlir::Value> args;
        for (auto *a : expr->arguments()) {
          args.push_back(Visit(a).getValue(builder));
        }
        if (args[1].getType().isa<mlir::IntegerType>())
          return ValueCategory(
              builder.create<LLVM::PowIOp>(loc, mlirType, args[0], args[1]),
              /*isReference*/ false);
        else
          return ValueCategory(
              builder.create<math::PowFOp>(loc, mlirType, args[0], args[1]),
              /*isReference*/ false);
      }
    }

  if (auto *ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_addressof") {
        auto V = Visit(expr->getArg(0));
        assert(V.isReference);
        mlir::Value val = V.val;
        mlir::Type T = Glob.getTypes().getMLIRType(expr->getType());
        if (T == val.getType())
          return ValueCategory(val, /*isRef*/ false);
        if (T.isa<LLVM::LLVMPointerType>()) {
          if (val.getType().isa<MemRefType>()) {
            assert(val.getType().cast<MemRefType>().getMemorySpaceAsInt() ==
                       T.cast<LLVM::LLVMPointerType>().getAddressSpace() &&
                   "val does not have the same memory space as T");
            val = builder.create<polygeist::Memref2PointerOp>(loc, T, val);
          } else if (T != val.getType())
            val = builder.create<LLVM::BitcastOp>(loc, T, val);
          return ValueCategory(val, /*isRef*/ false);
        } else {
          assert(T.isa<MemRefType>());
          if (val.getType().isa<MemRefType>())
            val = builder.create<polygeist::Memref2PointerOp>(
                loc,
                LLVM::LLVMPointerType::get(
                    builder.getI8Type(),
                    val.getType().cast<MemRefType>().getMemorySpaceAsInt()),
                val);
          if (val.getType().isa<LLVM::LLVMPointerType>())
            val = builder.create<polygeist::Pointer2MemrefOp>(loc, T, val);
          return ValueCategory(val, /*isRef*/ false);
        }
        expr->dump();
        llvm::errs() << " val: " << val << " T: " << T << "\n";
        assert(0 && "unhandled builtin addressof");
      }
    }

  if (auto *ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__nv_fabsf" ||
           sr->getDecl()->getName() == "__nv_fabs" ||
           sr->getDecl()->getName() == "__nv_abs" ||
           sr->getDecl()->getName() == "fabs" ||
           sr->getDecl()->getName() == "fabsf" ||
           sr->getDecl()->getName() == "__builtin_fabs" ||
           sr->getDecl()->getName() == "__builtin_fabsf")) {
        // isinf(x)    --> fabs(x) == infinity
        // isfinite(x) --> fabs(x) != infinity
        // x != NaN via the ordered compare in either case.
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value Fabs;
        if (V.getType().isa<mlir::FloatType>())
          Fabs = builder.create<math::AbsFOp>(loc, V);
        else {
          auto zero = builder.create<arith::ConstantIntOp>(
              loc, 0, V.getType().cast<mlir::IntegerType>().getWidth());
          Fabs = builder.create<SelectOp>(
              loc,
              builder.create<arith::CmpIOp>(loc, CmpIPredicate::sge, V, zero),
              V, builder.create<arith::SubIOp>(loc, zero, V));
        }
        return ValueCategory(Fabs, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__nv_mul24") {
        mlir::Value V0 = getLLVM(expr->getArg(0));
        mlir::Value V1 = getLLVM(expr->getArg(1));
        auto c8 = builder.create<arith::ConstantIntOp>(loc, 8, 32);
        V0 = builder.create<arith::ShLIOp>(loc, V0, c8);
        V0 = builder.create<arith::ShRUIOp>(loc, V0, c8);
        V1 = builder.create<arith::ShLIOp>(loc, V1, c8);
        V1 = builder.create<arith::ShRUIOp>(loc, V1, c8);
        return ValueCategory(builder.create<MulIOp>(loc, V0, V1), false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_frexp" ||
           sr->getDecl()->getName() == "__builtin_frexpf" ||
           sr->getDecl()->getName() == "__builtin_frexpl" ||
           sr->getDecl()->getName() == "__builtin_frexpf128")) {
        mlir::Value V0 = getLLVM(expr->getArg(0));
        mlir::Value V1 = getLLVM(expr->getArg(1));

        auto name = sr->getDecl()
                        ->getName()
                        .substr(std::string("__builtin_").length())
                        .str();

        if (Glob.getFunctions().find(name) == Glob.getFunctions().end()) {
          std::vector<mlir::Type> types{V0.getType(), V1.getType()};

          mlir::Type RT = Glob.getTypes().getMLIRType(expr->getType());
          std::vector<mlir::Type> rettypes{RT};
          mlir::OpBuilder mbuilder(module->getContext());
          auto funcType = mbuilder.getFunctionType(types, rettypes);
          Glob.getFunctions()[name] =
              mlir::func::FuncOp(mlir::func::FuncOp::create(
                  builder.getUnknownLoc(), name, funcType));
          SymbolTable::setSymbolVisibility(Glob.getFunctions()[name],
                                           SymbolTable::Visibility::Private);
          module->push_back(Glob.getFunctions()[name]);
        }

        mlir::Value vals[] = {V0, V1};
        return ValueCategory(
            builder.create<CallOp>(loc, Glob.getFunctions()[name], vals)
                .getResult(0),
            false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_strlen" ||
           sr->getDecl()->getName() == "strlen")) {
        mlir::Value V0 = getLLVM(expr->getArg(0));

        const auto *name = "strlen";

        if (Glob.getFunctions().find(name) == Glob.getFunctions().end()) {
          std::vector<mlir::Type> types{V0.getType()};

          mlir::Type RT = Glob.getTypes().getMLIRType(expr->getType());
          std::vector<mlir::Type> rettypes{RT};
          mlir::OpBuilder mbuilder(module->getContext());
          auto funcType = mbuilder.getFunctionType(types, rettypes);
          Glob.getFunctions()[name] =
              mlir::func::FuncOp(mlir::func::FuncOp::create(
                  builder.getUnknownLoc(), name, funcType));
          SymbolTable::setSymbolVisibility(Glob.getFunctions()[name],
                                           SymbolTable::Visibility::Private);
          module->push_back(Glob.getFunctions()[name]);
        }

        mlir::Value vals[] = {V0};
        return ValueCategory(
            builder.create<CallOp>(loc, Glob.getFunctions()[name], vals)
                .getResult(0),
            false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_isfinite" ||
           sr->getDecl()->getName() == "__builtin_isinf" ||
           sr->getDecl()->getName() == "__nv_isinff")) {
        // isinf(x)    --> fabs(x) == infinity
        // isfinite(x) --> fabs(x) != infinity
        // x != NaN via the ordered compare in either case.
        mlir::Value V = getLLVM(expr->getArg(0));
        auto Ty = V.getType().cast<mlir::FloatType>();
        mlir::Value Fabs = builder.create<math::AbsFOp>(loc, V);
        auto Infinity = builder.create<ConstantFloatOp>(
            loc, APFloat::getInf(Ty.getFloatSemantics()), Ty);
        auto Pred = (sr->getDecl()->getName() == "__builtin_isinf" ||
                     sr->getDecl()->getName() == "__nv_isinff")
                        ? CmpFPredicate::OEQ
                        : CmpFPredicate::ONE;
        mlir::Value FCmp = builder.create<CmpFOp>(loc, Pred, Fabs, Infinity);
        auto postTy = Glob.getTypes()
                          .getMLIRType(expr->getType())
                          .cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, FCmp);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_isnan" ||
           sr->getDecl()->getName() == "__nv_isnanf")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value Eq = builder.create<CmpFOp>(loc, CmpFPredicate::UNO, V, V);
        auto postTy = Glob.getTypes()
                          .getMLIRType(expr->getType())
                          .cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, Eq);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_isnormal")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        auto Ty = V.getType().cast<mlir::FloatType>();
        mlir::Value Eq = builder.create<CmpFOp>(loc, CmpFPredicate::OEQ, V, V);

        mlir::Value Abs = builder.create<math::AbsFOp>(loc, V);
        auto Infinity = builder.create<ConstantFloatOp>(
            loc, APFloat::getInf(Ty.getFloatSemantics()), Ty);
        mlir::Value IsLessThanInf =
            builder.create<CmpFOp>(loc, CmpFPredicate::ULT, Abs, Infinity);
        APFloat Smallest =
            APFloat::getSmallestNormalized(Ty.getFloatSemantics());
        auto SmallestV = builder.create<ConstantFloatOp>(loc, Smallest, Ty);
        mlir::Value IsNormal =
            builder.create<CmpFOp>(loc, CmpFPredicate::UGE, Abs, SmallestV);
        V = builder.create<AndIOp>(loc, Eq, IsLessThanInf);
        V = builder.create<AndIOp>(loc, V, IsNormal);
        auto postTy = Glob.getTypes()
                          .getMLIRType(expr->getType())
                          .cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, V);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_signbit") {
        mlir::Value V = getLLVM(expr->getArg(0));
        auto Ty = V.getType().cast<mlir::FloatType>();
        auto ITy = builder.getIntegerType(Ty.getWidth());
        mlir::Value BC = builder.create<BitcastOp>(loc, ITy, V);
        auto ZeroV = builder.create<ConstantIntOp>(loc, 0, ITy);
        V = builder.create<CmpIOp>(loc, CmpIPredicate::slt, BC, ZeroV);
        auto postTy = Glob.getTypes()
                          .getMLIRType(expr->getType())
                          .cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, V);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_isgreater") {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<CmpFOp>(loc, CmpFPredicate::OGT, V, V2);
        auto postTy = Glob.getTypes()
                          .getMLIRType(expr->getType())
                          .cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, V);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_isgreaterequal") {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<CmpFOp>(loc, CmpFPredicate::OGE, V, V2);
        auto postTy = Glob.getTypes()
                          .getMLIRType(expr->getType())
                          .cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, V);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_isless") {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<CmpFOp>(loc, CmpFPredicate::OLT, V, V2);
        auto postTy = Glob.getTypes()
                          .getMLIRType(expr->getType())
                          .cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, V);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_islessequal") {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<CmpFOp>(loc, CmpFPredicate::OLE, V, V2);
        auto postTy = Glob.getTypes()
                          .getMLIRType(expr->getType())
                          .cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, V);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_islessgreater") {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<CmpFOp>(loc, CmpFPredicate::ONE, V, V2);
        auto postTy = Glob.getTypes()
                          .getMLIRType(expr->getType())
                          .cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, V);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__builtin_isunordered") {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<CmpFOp>(loc, CmpFPredicate::UNO, V, V2);
        auto postTy = Glob.getTypes()
                          .getMLIRType(expr->getType())
                          .cast<mlir::IntegerType>();
        mlir::Value res = builder.create<ExtUIOp>(loc, postTy, V);
        return ValueCategory(res, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_pow" ||
           sr->getDecl()->getName() == "__builtin_powf" ||
           sr->getDecl()->getName() == "__builtin_powl")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<math::PowFOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__nv_fmodf")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<mlir::LLVM::FRemOp>(loc, V.getType(), V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_atanh" ||
           sr->getDecl()->getName() == "__builtin_atanhf" ||
           sr->getDecl()->getName() == "__builtin_atanhl")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        V = builder.create<math::AtanOp>(loc, V);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__nv_scalbn" ||
           sr->getDecl()->getName() == "__nv_scalbnf" ||
           sr->getDecl()->getName() == "__nv_scalbnl")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        auto name = sr->getDecl()->getName().substr(5).str();
        std::vector<mlir::Type> types{V.getType(), V2.getType()};
        mlir::Type RT = Glob.getTypes().getMLIRType(expr->getType());

        std::vector<mlir::Type> rettypes{RT};

        mlir::OpBuilder mbuilder(module->getContext());
        auto funcType = mbuilder.getFunctionType(types, rettypes);
        mlir::func::FuncOp function =
            mlir::func::FuncOp(mlir::func::FuncOp::create(
                builder.getUnknownLoc(), name, funcType));
        SymbolTable::setSymbolVisibility(function,
                                         SymbolTable::Visibility::Private);

        Glob.getFunctions()[name] = function;
        module->push_back(function);
        mlir::Value vals[] = {V, V2};
        V = builder.create<CallOp>(loc, function, vals).getResult(0);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__nv_dmul_rn")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<MulFOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__nv_dadd_rn")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<AddFOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__nv_dsub_rn")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<SubFOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_log2" ||
           sr->getDecl()->getName() == "__builtin_log2f" ||
           sr->getDecl()->getName() == "__builtin_log2l" ||
           sr->getDecl()->getName() == "__nv_log2" ||
           sr->getDecl()->getName() == "__nv_log2f" ||
           sr->getDecl()->getName() == "__nv_log2l")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        V = builder.create<math::Log2Op>(loc, V);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_log10" ||
           sr->getDecl()->getName() == "__builtin_log10f" ||
           sr->getDecl()->getName() == "__builtin_log10l" ||
           sr->getDecl()->getName() == "__nv_log10" ||
           sr->getDecl()->getName() == "__nv_log10f" ||
           sr->getDecl()->getName() == "__nv_log10l")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        V = builder.create<math::Log10Op>(loc, V);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_log1p" ||
           sr->getDecl()->getName() == "__builtin_log1pf" ||
           sr->getDecl()->getName() == "__builtin_log1pl")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        V = builder.create<math::Log1pOp>(loc, V);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_exp2" ||
           sr->getDecl()->getName() == "__builtin_exp2f" ||
           sr->getDecl()->getName() == "__builtin_exp2l")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        V = builder.create<math::Exp2Op>(loc, V);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_expm1" ||
           sr->getDecl()->getName() == "__builtin_expm1f" ||
           sr->getDecl()->getName() == "__builtin_expm1l")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        V = builder.create<math::ExpM1Op>(loc, V);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_copysign" ||
           sr->getDecl()->getName() == "__builtin_copysignf" ||
           sr->getDecl()->getName() == "__builtin_copysignl")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<LLVM::CopySignOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_copysign" ||
           sr->getDecl()->getName() == "__builtin_copysignf" ||
           sr->getDecl()->getName() == "__builtin_copysignl")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<LLVM::CopySignOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_fmax" ||
           sr->getDecl()->getName() == "__builtin_fmaxf" ||
           sr->getDecl()->getName() == "__builtin_fmaxl")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<LLVM::MaxNumOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_fmin" ||
           sr->getDecl()->getName() == "__builtin_fminf" ||
           sr->getDecl()->getName() == "__builtin_fminl")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        V = builder.create<LLVM::MinNumOp>(loc, V, V2);
        return ValueCategory(V, /*isRef*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "__builtin_fma" ||
           sr->getDecl()->getName() == "__builtin_fmaf" ||
           sr->getDecl()->getName() == "__builtin_fmal")) {
        mlir::Value V = getLLVM(expr->getArg(0));
        mlir::Value V2 = getLLVM(expr->getArg(1));
        mlir::Value V3 = getLLVM(expr->getArg(2));
        V = builder.create<LLVM::FMAOp>(loc, V, V2, V3);
        return ValueCategory(V, /*isRef*/ false);
      }
    }

  if (auto *ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if ((sr->getDecl()->getIdentifier() &&
           (sr->getDecl()->getName() == "fscanf" ||
            sr->getDecl()->getName() == "scanf" ||
            sr->getDecl()->getName() == "__isoc99_sscanf" ||
            sr->getDecl()->getName() == "sscanf")) ||
          (isa<CXXOperatorCallExpr>(expr) &&
           cast<CXXOperatorCallExpr>(expr)->getOperator() ==
               OO_GreaterGreater)) {
        const auto *tocall = EmitCallee(expr->getCallee());
        auto strcmpF = Glob.GetOrCreateLLVMFunction(tocall);

        std::vector<mlir::Value> args;
        std::vector<std::pair<mlir::Value, mlir::Value>> ops;
        std::map<const void *, size_t> counts;
        for (auto *a : expr->arguments()) {
          auto v = getLLVM(a);
          if (auto toptr = v.getDefiningOp<polygeist::Memref2PointerOp>()) {
            auto T = toptr.getType().cast<LLVM::LLVMPointerType>();
            auto idx = counts[T.getAsOpaquePointer()]++;
            auto aop = allocateBuffer(idx, T);
            args.push_back(aop.getResult());
            ops.emplace_back(aop.getResult(), toptr.getSource());
          } else
            args.push_back(v);
        }
        auto called = builder.create<mlir::LLVM::CallOp>(loc, strcmpF, args);
        for (auto pair : ops) {
          auto lop = builder.create<mlir::LLVM::LoadOp>(loc, pair.first);
          builder.create<mlir::memref::StoreOp>(
              loc, lop, pair.second,
              std::vector<mlir::Value>({getConstantIndex(0)}));
        }
        return ValueCategory(called.getResult(), /*isReference*/ false);
      }
    }

  if (auto *ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "memmove" ||
           sr->getDecl()->getName() == "__builtin_memmove")) {
        std::vector<mlir::Value> args = {
            getLLVM(expr->getArg(0)), getLLVM(expr->getArg(1)),
            getLLVM(expr->getArg(2)), /*isVolatile*/
            builder.create<ConstantIntOp>(loc, false, 1)};
        builder.create<LLVM::MemmoveOp>(loc, args[0], args[1], args[2],
                                        args[3]);
        return ValueCategory(args[0], /*isReference*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "memset" ||
           sr->getDecl()->getName() == "__builtin_memset")) {
        std::vector<mlir::Value> args = {
            getLLVM(expr->getArg(0)), getLLVM(expr->getArg(1)),
            getLLVM(expr->getArg(2)), /*isVolatile*/
            builder.create<ConstantIntOp>(loc, false, 1)};

        args[1] = builder.create<TruncIOp>(loc, builder.getI8Type(), args[1]);
        builder.create<LLVM::MemsetOp>(loc, args[0], args[1], args[2], args[3]);
        return ValueCategory(args[0], /*isReference*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "memcpy" ||
           sr->getDecl()->getName() == "__builtin_memcpy")) {
        std::vector<mlir::Value> args = {
            getLLVM(expr->getArg(0)), getLLVM(expr->getArg(1)),
            getLLVM(expr->getArg(2)), /*isVolatile*/
            builder.create<ConstantIntOp>(loc, false, 1)};
        builder.create<LLVM::MemcpyOp>(loc, args[0], args[1], args[2], args[3]);
        return ValueCategory(args[0], /*isReference*/ false);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "cudaMemcpy" ||
           sr->getDecl()->getName() == "cudaMemcpyAsync" ||
           sr->getDecl()->getName() == "cudaMemcpyToSymbol" ||
           sr->getDecl()->getName() == "memcpy" ||
           sr->getDecl()->getName() == "__builtin_memcpy")) {
        auto *dstSub = expr->getArg(0);
        while (auto *BC = dyn_cast<clang::CastExpr>(dstSub))
          dstSub = BC->getSubExpr();
        auto *srcSub = expr->getArg(1);
        while (auto *BC = dyn_cast<clang::CastExpr>(srcSub))
          srcSub = BC->getSubExpr();

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
              if (sr->getDecl()->getName() == "__builtin_memcpy" ||
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

  const FunctionDecl *callee = EmitCallee(expr->getCallee());

  std::set<std::string> funcs = {
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
  if (auto *ic = dyn_cast<ImplicitCastExpr>(expr->getCallee()))
    if (auto *sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      StringRef name;
      if (auto *CC = dyn_cast<CXXConstructorDecl>(sr->getDecl()))
        name = Glob.getCGM().getMangledName(
            GlobalDecl(CC, CXXCtorType::Ctor_Complete));
      else if (auto *CC = dyn_cast<CXXDestructorDecl>(sr->getDecl()))
        name = Glob.getCGM().getMangledName(
            GlobalDecl(CC, CXXDtorType::Dtor_Complete));
      else if (sr->getDecl()->hasAttr<CUDAGlobalAttr>())
        name = Glob.getCGM().getMangledName(GlobalDecl(
            cast<FunctionDecl>(sr->getDecl()), KernelReferenceKind::Kernel));
      else
        name = Glob.getCGM().getMangledName(sr->getDecl());
      if (funcs.count(name.str()) || name.startswith("mkl_") ||
          name.startswith("MKL_") || name.startswith("cublas") ||
          name.startswith("cblas_")) {

        std::vector<mlir::Value> args;
        for (auto *a : expr->arguments()) {
          args.push_back(getLLVM(a));
        }
        mlir::Value called;

        if (callee) {
          auto strcmpF = Glob.GetOrCreateLLVMFunction(callee);
          called = builder.create<mlir::LLVM::CallOp>(loc, strcmpF, args)
                       .getResult();
        } else {
          args.insert(args.begin(), getLLVM(expr->getCallee()));
          SmallVector<mlir::Type> RTs = {typeTranslator.translateType(
              anonymize(getLLVMType(expr->getType(), Glob.getCGM())))};
          if (RTs[0].isa<LLVM::LLVMVoidType>())
            RTs.clear();
          called =
              builder.create<mlir::LLVM::CallOp>(loc, RTs, args).getResult();
        }
        return ValueCategory(called, /*isReference*/ expr->isLValue() ||
                                         expr->isXValue());
      }
    }

  if (!callee || callee->isVariadic()) {
    bool isReference = expr->isLValue() || expr->isXValue();
    std::vector<mlir::Value> args;
    for (auto *a : expr->arguments()) {
      args.push_back(getLLVM(a));
    }
    mlir::Value called;
    if (callee) {
      auto strcmpF = Glob.GetOrCreateLLVMFunction(callee);
      called =
          builder.create<mlir::LLVM::CallOp>(loc, strcmpF, args).getResult();
    } else {
      args.insert(args.begin(), getLLVM(expr->getCallee()));
      auto CT = expr->getType();
      if (isReference)
        CT = Glob.getCGM().getContext().getLValueReferenceType(CT);
      SmallVector<mlir::Type> RTs = {typeTranslator.translateType(
          anonymize(getLLVMType(CT, Glob.getCGM())))};

      auto ft = args[0]
                    .getType()
                    .cast<LLVM::LLVMPointerType>()
                    .getElementType()
                    .cast<LLVM::LLVMFunctionType>();
      assert(RTs[0] == ft.getReturnType());
      if (RTs[0].isa<LLVM::LLVMVoidType>())
        RTs.clear();
      called = builder.create<mlir::LLVM::CallOp>(loc, RTs, args).getResult();
    }
    if (isReference) {
      if (!(called.getType().isa<LLVM::LLVMPointerType>() ||
            called.getType().isa<MemRefType>())) {
        expr->dump();
        expr->getType()->dump();
        llvm::errs() << " call: " << called << "\n";
      }
    }
    if (!called)
      return nullptr;
    return ValueCategory(called, isReference);
  }

  /// If the callee is part of the SYCL namespace, we may not want the
  /// GetOrCreateMLIRFunction to add this FuncOp to the functionsToEmit deque,
  /// since we will create it's equivalent with SYCL operations. Please note
  /// that we still generate some functions that we need for lowering some
  /// sycl op.  Therefore, in those case, we set ShouldEmit back to "true" by
  /// looking them up in our "registry" of supported functions.
  bool isSyclFunc =
      mlirclang::isNamespaceSYCL(callee->getEnclosingNamespaceContext());
  bool ShouldEmit = !isSyclFunc;

  std::string mangledName =
      MLIRScanner::getMangledFuncName(*callee, Glob.getCGM());
  if (GenerateAllSYCLFuncs || isSupportedFunctions(mangledName))
    ShouldEmit = true;

  FunctionToEmit F(*callee, mlirclang::getInputContext(builder));
  auto ToCall = cast<func::FuncOp>(Glob.GetOrCreateMLIRFunction(F, ShouldEmit));

  SmallVector<std::pair<ValueCategory, clang::Expr *>> args;
  QualType objType;

  if (auto *CC = dyn_cast<CXXMemberCallExpr>(expr)) {
    ValueCategory obj = Visit(CC->getImplicitObjectArgument());
    objType = CC->getObjectType();
#ifdef DEBUG
    if (!obj.val) {
      function.dump();
      llvm::errs() << " objval: " << obj.val << "\n";
      expr->dump();
      CC->getImplicitObjectArgument()->dump();
    }
#endif
    if (cast<MemberExpr>(CC->getCallee()->IgnoreParens())->isArrow()) {
      obj = obj.dereference(builder);
    }
    assert(obj.val);
    assert(obj.isReference);
    args.emplace_back(std::make_pair(obj, (clang::Expr *)nullptr));
  }

  for (auto *a : expr->arguments())
    args.push_back(std::make_pair(Visit(a), a));

  return CallHelper(ToCall, objType, args, expr->getType(),
                    expr->isLValue() || expr->isXValue(), expr, callee);
}
