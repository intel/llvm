//===- CGStmt.cc - Emit MLIR IRs by walking stmt-like AST nodes-*- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IfScope.h"
#include "clang-mlir.h"
#include "utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"

#define DEBUG_TYPE "CGStmt"

using namespace mlir;

extern llvm::cl::opt<bool> SuppressWarnings;

static bool isTerminator(Operation *Op) {
  return Op->mightHaveTrait<OpTrait::IsTerminator>();
}

bool MLIRScanner::getLowerBound(clang::ForStmt *Fors,
                                mlirclang::AffineLoopDescriptor &Descr) {
  auto *Init = Fors->getInit();
  if (auto *DeclStmt = dyn_cast<clang::DeclStmt>(Init))
    if (DeclStmt->isSingleDecl()) {
      auto *Decl = DeclStmt->getSingleDecl();
      if (auto *VarDecl = dyn_cast<clang::VarDecl>(Decl)) {
        if (VarDecl->hasInit()) {
          Value Val = VisitVarDecl(VarDecl).getValue(builder);
          Descr.setName(VarDecl);
          Descr.setType(Val.getType());
          LLVM_DEBUG(Descr.getType().print(llvm::dbgs()));

          if (Descr.getForwardMode())
            Descr.setLowerBound(Val);
          else {
            Val = builder.create<arith::AddIOp>(loc, Val, getConstantIndex(1));
            Descr.setUpperBound(Val);
          }
          return true;
        }
      }
    }

  // BinaryOperator 0x7ff7aa17e938 'int' '='
  // |-DeclRefExpr 0x7ff7aa17e8f8 'int' lvalue Var 0x7ff7aa17e758 'i' 'int'
  // -IntegerLiteral 0x7ff7aa17e918 'int' 0
  if (auto *BinOp = dyn_cast<clang::BinaryOperator>(Init))
    if (BinOp->getOpcode() == clang::BinaryOperator::Opcode::BO_Assign)
      if (auto *DeclRefStmt = dyn_cast<clang::DeclRefExpr>(BinOp->getLHS())) {
        Value Val = Visit(BinOp->getRHS()).getValue(builder);
        Val = builder.create<arith::IndexCastOp>(
            loc, IndexType::get(builder.getContext()), Val);
        Descr.setName(cast<clang::VarDecl>(DeclRefStmt->getDecl()));
        Descr.setType(
            Glob.getTypes().getMLIRType(DeclRefStmt->getDecl()->getType()));
        if (Descr.getForwardMode())
          Descr.setLowerBound(Val);
        else {
          Val = builder.create<arith::AddIOp>(loc, Val, getConstantIndex(1));
          Descr.setUpperBound(Val);
        }
        return true;
      }
  return false;
}

// Make sure that the induction variable initialized in
// the for is the same as the one used in the condition.
bool matchIndvar(const clang::Expr *Expr, clang::VarDecl *IndVar) {
  while (const auto *IC = dyn_cast<clang::ImplicitCastExpr>(Expr)) {
    Expr = IC->getSubExpr();
  }
  if (const auto *DeclRef = dyn_cast<clang::DeclRefExpr>(Expr)) {
    const auto *DeclRefName = DeclRef->getDecl();
    if (DeclRefName == IndVar)
      return true;
  }
  return false;
}

bool MLIRScanner::getUpperBound(clang::ForStmt *Fors,
                                mlirclang::AffineLoopDescriptor &Descr) {
  auto *Cond = Fors->getCond();
  if (auto *BinaryOp = dyn_cast<clang::BinaryOperator>(Cond)) {
    auto *Lhs = BinaryOp->getLHS();
    if (!matchIndvar(Lhs, Descr.getName()))
      return false;

    if (Descr.getForwardMode()) {
      if (BinaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_LT &&
          BinaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_LE)
        return false;

      auto *Rhs = BinaryOp->getRHS();
      Value Val = Visit(Rhs).getValue(builder);
      Val = builder.create<arith::IndexCastOp>(
          loc, IndexType::get(Val.getContext()), Val);
      if (BinaryOp->getOpcode() == clang::BinaryOperator::Opcode::BO_LE)
        Val = builder.create<arith::AddIOp>(loc, Val, getConstantIndex(1));
      Descr.setUpperBound(Val);
      return true;
    }
    if (BinaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_GT &&
        BinaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_GE)
      return false;

    auto *Rhs = BinaryOp->getRHS();
    Value Val = Visit(Rhs).getValue(builder);
    Val = builder.create<arith::IndexCastOp>(
        loc, IndexType::get(Val.getContext()), Val);
    if (BinaryOp->getOpcode() == clang::BinaryOperator::Opcode::BO_GT)
      Val = builder.create<arith::AddIOp>(loc, Val, getConstantIndex(1));
    Descr.setLowerBound(Val);
    return true;
  }
  return false;
}

bool MLIRScanner::getConstantStep(clang::ForStmt *Fors,
                                  mlirclang::AffineLoopDescriptor &Descr) {
  auto *Inc = Fors->getInc();
  if (auto *UnaryOp = dyn_cast<clang::UnaryOperator>(Inc))
    if (UnaryOp->isPrefix() || UnaryOp->isPostfix()) {
      bool ForwardLoop =
          UnaryOp->getOpcode() == clang::UnaryOperator::Opcode::UO_PostInc ||
          UnaryOp->getOpcode() == clang::UnaryOperator::Opcode::UO_PreInc;
      Descr.setStep(1);
      Descr.setForwardMode(ForwardLoop);
      return true;
    }
  return false;
}

bool MLIRScanner::isTrivialAffineLoop(clang::ForStmt *Fors,
                                      mlirclang::AffineLoopDescriptor &Descr) {
  if (!getConstantStep(Fors, Descr)) {
    LLVM_DEBUG(llvm::dbgs() << "getConstantStep -> false\n");
    return false;
  }
  if (!getLowerBound(Fors, Descr)) {
    LLVM_DEBUG(llvm::dbgs() << "getLowerBound -> false\n");
    return false;
  }
  if (!getUpperBound(Fors, Descr)) {
    LLVM_DEBUG(llvm::dbgs() << "getUpperBound -> false\n");
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "isTrivialAffineLoop -> true\n");
  return true;
}

void MLIRScanner::buildAffineLoopImpl(
    clang::ForStmt *Fors, Location Loc, Value Lb, Value Ub,
    const mlirclang::AffineLoopDescriptor &Descr) {
  auto AffineOp = builder.create<AffineForOp>(
      Loc, Lb, builder.getSymbolIdentityMap(), Ub,
      builder.getSymbolIdentityMap(), Descr.getStep(),
      /*iterArgs=*/llvm::None);

  auto &Reg = AffineOp.getLoopBody();

  auto Val = (Value)AffineOp.getInductionVar();

  Reg.front().clear();

  auto OldPoint = builder.getInsertionPoint();
  auto *OldBlock = builder.getInsertionBlock();

  builder.setInsertionPointToEnd(&Reg.front());

  auto Er = builder.create<scf::ExecuteRegionOp>(Loc, ArrayRef<Type>());
  Er.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&Er.getRegion().back());
  builder.create<scf::YieldOp>(Loc);
  builder.setInsertionPointToStart(&Er.getRegion().back());

  if (!Descr.getForwardMode()) {
    Val = builder.create<arith::SubIOp>(Loc, Val, Lb);
    Val = builder.create<arith::SubIOp>(
        Loc, builder.create<arith::SubIOp>(Loc, Ub, getConstantIndex(1)), Val);
  }
  auto Idx = builder.create<arith::IndexCastOp>(Loc, Descr.getType(), Val);
  assert(params.find(Descr.getName()) != params.end());
  params[Descr.getName()].store(builder, Idx);

  // TODO: set loop context.
  Visit(Fors->getBody());

  builder.setInsertionPointToEnd(&Reg.front());
  builder.create<AffineYieldOp>(Loc);

  // TODO: set the value of the iteration value to the final bound at the
  // end of the loop.
  builder.setInsertionPoint(OldBlock, OldPoint);
}

void MLIRScanner::buildAffineLoop(
    clang::ForStmt *Fors, Location Loc,
    const mlirclang::AffineLoopDescriptor &Descr) {
  Value LB = Descr.getLowerBound();
  Value UB = Descr.getUpperBound();
  buildAffineLoopImpl(Fors, Loc, LB, UB, Descr);
}

ValueCategory MLIRScanner::VisitForStmt(clang::ForStmt *Fors) {
  IfScope Scope(*this);

  Location Loc = getMLIRLocation(Fors->getForLoc());

  mlirclang::AffineLoopDescriptor AffineLoopDescr;
  if (Glob.getScopLocList().isInScop(Fors->getForLoc()) &&
      isTrivialAffineLoop(Fors, AffineLoopDescr)) {
    buildAffineLoop(Fors, Loc, AffineLoopDescr);
  } else {

    if (auto *S = Fors->getInit())
      Visit(S);

    auto I1Ty = builder.getIntegerType(1);
    auto Type = MemRefType::get({}, I1Ty, {}, 0);
    auto Truev = builder.create<arith::ConstantIntOp>(Loc, true, 1);

    LoopContext Lctx{builder.create<memref::AllocaOp>(Loc, Type),
                     builder.create<memref::AllocaOp>(Loc, Type)};
    builder.create<memref::StoreOp>(Loc, Truev, Lctx.noBreak);

    auto *ToAdd = builder.getInsertionBlock()->getParent();
    auto &CondB = *(new Block());
    ToAdd->getBlocks().push_back(&CondB);
    auto &BodyB = *(new Block());
    ToAdd->getBlocks().push_back(&BodyB);
    auto &ExitB = *(new Block());
    ToAdd->getBlocks().push_back(&ExitB);

    builder.create<cf::BranchOp>(Loc, &CondB);

    builder.setInsertionPointToStart(&CondB);

    if (auto *S = Fors->getCond()) {
      auto CondRes = Visit(S);
      auto Cond = CondRes.getValue(builder);
      if (auto LT = Cond.getType().dyn_cast<LLVM::LLVMPointerType>()) {
        auto NullptrLlvm = builder.create<LLVM::NullOp>(Loc, LT);
        Cond = builder.create<LLVM::ICmpOp>(Loc, LLVM::ICmpPredicate::ne, Cond,
                                            NullptrLlvm);
      }
      auto Ty = Cond.getType().cast<IntegerType>();
      if (Ty.getWidth() != 1) {
        Cond = builder.create<arith::CmpIOp>(
            Loc, arith::CmpIPredicate::ne, Cond,
            builder.create<arith::ConstantIntOp>(Loc, 0, Ty));
      }
      auto Nb = builder.create<memref::LoadOp>(Loc, Lctx.noBreak,
                                               std::vector<Value>());
      Cond = builder.create<arith::AndIOp>(Loc, Cond, Nb);
      builder.create<cf::CondBranchOp>(Loc, Cond, &BodyB, &ExitB);
    } else {
      auto Cond = builder.create<memref::LoadOp>(Loc, Lctx.noBreak,
                                                 std::vector<Value>());
      builder.create<cf::CondBranchOp>(Loc, Cond, &BodyB, &ExitB);
    }

    builder.setInsertionPointToStart(&BodyB);
    builder.create<memref::StoreOp>(
        Loc,
        builder.create<memref::LoadOp>(Loc, Lctx.noBreak, std::vector<Value>()),
        Lctx.keepRunning, std::vector<Value>());

    loops.push_back(Lctx);
    Visit(Fors->getBody());

    builder.create<memref::StoreOp>(
        Loc,
        builder.create<memref::LoadOp>(Loc, Lctx.noBreak, std::vector<Value>()),
        Lctx.keepRunning, std::vector<Value>());
    if (auto *S = Fors->getInc()) {
      IfScope Scope(*this);
      Visit(S);
    }
    loops.pop_back();
    if (builder.getInsertionBlock()->empty() ||
        !isTerminator(&builder.getInsertionBlock()->back())) {
      builder.create<cf::BranchOp>(Loc, &CondB);
    }

    builder.setInsertionPointToStart(&ExitB);
  }
  return nullptr;
}

ValueCategory MLIRScanner::VisitCXXForRangeStmt(clang::CXXForRangeStmt *Fors) {
  IfScope Scope(*this);

  auto Loc = getMLIRLocation(Fors->getForLoc());

  if (auto *S = Fors->getInit()) {
    Visit(S);
  }
  Visit(Fors->getRangeStmt());
  Visit(Fors->getBeginStmt());
  Visit(Fors->getEndStmt());

  auto I1Ty = builder.getIntegerType(1);
  auto Type = MemRefType::get({}, I1Ty, {}, 0);
  auto Truev = builder.create<arith::ConstantIntOp>(Loc, true, 1);

  LoopContext Lctx{builder.create<memref::AllocaOp>(Loc, Type),
                   builder.create<memref::AllocaOp>(Loc, Type)};
  builder.create<memref::StoreOp>(Loc, Truev, Lctx.noBreak);

  auto *ToAdd = builder.getInsertionBlock()->getParent();
  auto &CondB = *(new Block());
  ToAdd->getBlocks().push_back(&CondB);
  auto &BodyB = *(new Block());
  ToAdd->getBlocks().push_back(&BodyB);
  auto &ExitB = *(new Block());
  ToAdd->getBlocks().push_back(&ExitB);

  builder.create<cf::BranchOp>(Loc, &CondB);

  builder.setInsertionPointToStart(&CondB);

  if (auto *S = Fors->getCond()) {
    auto CondRes = Visit(S);
    auto Cond = CondRes.getValue(builder);
    if (auto LT = Cond.getType().dyn_cast<LLVM::LLVMPointerType>()) {
      auto NullptrLlvm = builder.create<LLVM::NullOp>(Loc, LT);
      Cond = builder.create<LLVM::ICmpOp>(Loc, LLVM::ICmpPredicate::ne, Cond,
                                          NullptrLlvm);
    }
    auto Ty = Cond.getType().cast<IntegerType>();
    if (Ty.getWidth() != 1) {
      Cond = builder.create<arith::CmpIOp>(
          Loc, arith::CmpIPredicate::ne, Cond,
          builder.create<arith::ConstantIntOp>(Loc, 0, Ty));
    }
    auto Nb =
        builder.create<memref::LoadOp>(Loc, Lctx.noBreak, std::vector<Value>());
    Cond = builder.create<arith::AndIOp>(Loc, Cond, Nb);
    builder.create<cf::CondBranchOp>(Loc, Cond, &BodyB, &ExitB);
  } else {
    auto Cond =
        builder.create<memref::LoadOp>(Loc, Lctx.noBreak, std::vector<Value>());
    builder.create<cf::CondBranchOp>(Loc, Cond, &BodyB, &ExitB);
  }

  builder.setInsertionPointToStart(&BodyB);
  builder.create<memref::StoreOp>(
      Loc,
      builder.create<memref::LoadOp>(Loc, Lctx.noBreak, std::vector<Value>()),
      Lctx.keepRunning, std::vector<Value>());

  loops.push_back(Lctx);
  Visit(Fors->getLoopVarStmt());
  Visit(Fors->getBody());

  builder.create<memref::StoreOp>(
      Loc,
      builder.create<memref::LoadOp>(Loc, Lctx.noBreak, std::vector<Value>()),
      Lctx.keepRunning, std::vector<Value>());
  if (auto *S = Fors->getInc()) {
    IfScope Scope(*this);
    Visit(S);
  }
  loops.pop_back();
  if (builder.getInsertionBlock()->empty() ||
      !isTerminator(&builder.getInsertionBlock()->back())) {
    builder.create<cf::BranchOp>(Loc, &CondB);
  }

  builder.setInsertionPointToStart(&ExitB);
  return nullptr;
}

ValueCategory
MLIRScanner::VisitOMPSingleDirective(clang::OMPSingleDirective *Par) {
  IfScope Scope(*this);

  builder.create<omp::BarrierOp>(loc);
  auto AffineOp = builder.create<omp::MasterOp>(loc);
  builder.create<omp::BarrierOp>(loc);

  auto OldPoint = builder.getInsertionPoint();
  auto *OldBlock = builder.getInsertionBlock();

  AffineOp.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&AffineOp.getRegion().front());

  auto ExecuteRegion =
      builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<Type>());
  ExecuteRegion.getRegion().push_back(new Block());
  builder.create<omp::TerminatorOp>(loc);
  builder.setInsertionPointToStart(&ExecuteRegion.getRegion().back());

  auto *OldScope = allocationScope;
  allocationScope = &ExecuteRegion.getRegion().back();

  Visit(cast<clang::CapturedStmt>(Par->getAssociatedStmt())
            ->getCapturedDecl()
            ->getBody());

  builder.create<scf::YieldOp>(loc);
  allocationScope = OldScope;
  builder.setInsertionPoint(OldBlock, OldPoint);
  return nullptr;
}

ValueCategory MLIRScanner::VisitOMPForDirective(clang::OMPForDirective *Fors) {
  IfScope Scope(*this);

  if (Fors->getPreInits()) {
    Visit(Fors->getPreInits());
  }

  SmallVector<Value> Inits;
  for (auto *F : Fors->inits()) {
    assert(F);
    F = cast<clang::BinaryOperator>(F)->getRHS();
    Inits.push_back(builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), Visit(F).getValue(builder)));
  }

  SmallVector<Value> Finals;
  for (auto *F : Fors->finals()) {
    F = cast<clang::BinaryOperator>(F)->getRHS();
    Finals.push_back(builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), Visit(F).getValue(builder)));
  }

  SmallVector<Value> Incs;
  for (auto *F : Fors->updates()) {
    F = cast<clang::BinaryOperator>(F)->getRHS();
    while (auto *CE = dyn_cast<clang::CastExpr>(F))
      F = CE->getSubExpr();
    auto *BO = cast<clang::BinaryOperator>(F);
    assert(BO->getOpcode() == clang::BinaryOperator::Opcode::BO_Add);
    F = BO->getRHS();
    while (auto *CE = dyn_cast<clang::CastExpr>(F))
      F = CE->getSubExpr();
    BO = cast<clang::BinaryOperator>(F);
    assert(BO->getOpcode() == clang::BinaryOperator::Opcode::BO_Mul);
    F = BO->getRHS();
    Incs.push_back(builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), Visit(F).getValue(builder)));
  }

  auto AffineOp = builder.create<omp::WsLoopOp>(loc, Inits, Finals, Incs);
  AffineOp.getRegion().push_back(new Block());
  for (auto Init : Inits)
    AffineOp.getRegion().front().addArgument(Init.getType(), Init.getLoc());
  auto Inds = AffineOp.getRegion().front().getArguments();

  auto OldPoint = builder.getInsertionPoint();
  auto *OldBlock = builder.getInsertionBlock();

  builder.setInsertionPointToStart(&AffineOp.getRegion().front());

  auto ExecuteRegion =
      builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<Type>());
  builder.create<omp::YieldOp>(loc, ValueRange());
  ExecuteRegion.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&ExecuteRegion.getRegion().back());

  auto *OldScope = allocationScope;
  allocationScope = &ExecuteRegion.getRegion().back();

  std::map<clang::VarDecl *, ValueCategory> PrevInduction;
  for (auto Zp : zip(Inds, Fors->counters())) {
    auto Idx = builder.create<arith::IndexCastOp>(
        loc,
        Glob.getTypes().getMLIRType(Fors->getIterationVariable()->getType()),
        std::get<0>(Zp));
    clang::VarDecl *Name = cast<clang::VarDecl>(
        cast<clang::DeclRefExpr>(std::get<1>(Zp))->getDecl());

    if (params.find(Name) != params.end()) {
      PrevInduction[Name] = params[Name];
      params.erase(Name);
    }

    bool LLVMABI = false;
    bool IsArray = false;
    if (Glob.getTypes()
            .getMLIRType(Glob.getCGM().getContext().getLValueReferenceType(
                Name->getType()))
            .isa<LLVM::LLVMPointerType>())
      LLVMABI = true;
    else
      Glob.getTypes().getMLIRType(Name->getType(), &IsArray);

    auto AllocOp = createAllocOp(Idx.getType(), Name, /*memtype*/ 0,
                                 /*isArray*/ IsArray, /*LLVMABI*/ LLVMABI);
    params[Name] = ValueCategory(AllocOp, true);
    params[Name].store(builder, Idx);
  }

  // TODO: set loop context.
  Visit(Fors->getBody());

  builder.create<scf::YieldOp>(loc, ValueRange());

  allocationScope = OldScope;

  // TODO: set the value of the iteration value to the final bound at the
  // end of the loop.
  builder.setInsertionPoint(OldBlock, OldPoint);

  for (auto Pair : PrevInduction)
    params[Pair.first] = Pair.second;

  return nullptr;
}

ValueCategory
MLIRScanner::VisitOMPParallelDirective(clang::OMPParallelDirective *Par) {
  IfScope Scope(*this);

  auto AffineOp = builder.create<omp::ParallelOp>(loc);

  auto OldPoint = builder.getInsertionPoint();
  auto *OldBlock = builder.getInsertionBlock();

  AffineOp.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&AffineOp.getRegion().front());

  auto ExecuteRegion =
      builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<Type>());
  ExecuteRegion.getRegion().push_back(new Block());
  builder.create<omp::TerminatorOp>(loc);
  builder.setInsertionPointToStart(&ExecuteRegion.getRegion().back());

  auto *OldScope = allocationScope;
  allocationScope = &ExecuteRegion.getRegion().back();

  std::map<clang::VarDecl *, ValueCategory> PrevInduction;
  for (auto *F : Par->clauses()) {
    switch (F->getClauseKind()) {
    case llvm::omp::OMPC_private:
      for (auto *Stmt : F->children()) {
        auto *Name =
            cast<clang::VarDecl>(cast<clang::DeclRefExpr>(Stmt)->getDecl());

        PrevInduction[Name] = params[Name];
        params.erase(Name);

        bool LLVMABI = false;
        bool IsArray = false;
        Type Ty;
        if (Glob.getTypes()
                .getMLIRType(Glob.getCGM().getContext().getLValueReferenceType(
                    Name->getType()))
                .isa<LLVM::LLVMPointerType>()) {
          LLVMABI = true;
          bool Undef;
          Ty = Glob.getTypes().getMLIRType(Name->getType(), &Undef);
        } else
          Ty = Glob.getTypes().getMLIRType(Name->getType(), &IsArray);

        auto AllocOp = createAllocOp(Ty, Name, /*memtype*/ 0,
                                     /*isArray*/ IsArray, /*LLVMABI*/ LLVMABI);
        params[Name] = ValueCategory(AllocOp, true);
        params[Name].store(builder, PrevInduction[Name], IsArray);
      }
      break;
    default:
      llvm::errs() << "may not handle omp clause " << (int)F->getClauseKind()
                   << "\n";
    }
  }

  Visit(cast<clang::CapturedStmt>(Par->getAssociatedStmt())
            ->getCapturedDecl()
            ->getBody());

  builder.create<scf::YieldOp>(loc);
  allocationScope = OldScope;
  builder.setInsertionPoint(OldBlock, OldPoint);

  for (auto Pair : PrevInduction)
    params[Pair.first] = Pair.second;
  return nullptr;
}

ValueCategory MLIRScanner::VisitOMPParallelForDirective(
    clang::OMPParallelForDirective *Fors) {
  IfScope Scope(*this);

  if (Fors->getPreInits()) {
    Visit(Fors->getPreInits());
  }

  SmallVector<Value> Inits;
  for (auto *F : Fors->inits()) {
    assert(F);
    F = cast<clang::BinaryOperator>(F)->getRHS();
    Inits.push_back(builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), Visit(F).getValue(builder)));
  }

  SmallVector<Value> Finals;
  for (auto *F : Fors->finals()) {
    F = cast<clang::BinaryOperator>(F)->getRHS();
    Finals.push_back(builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), Visit(F).getValue(builder)));
  }

  SmallVector<Value> Incs;
  for (auto *F : Fors->updates()) {
    F = cast<clang::BinaryOperator>(F)->getRHS();
    while (auto *CE = dyn_cast<clang::CastExpr>(F))
      F = CE->getSubExpr();
    auto *BO = cast<clang::BinaryOperator>(F);
    assert(BO->getOpcode() == clang::BinaryOperator::Opcode::BO_Add);
    F = BO->getRHS();
    while (auto *CE = dyn_cast<clang::CastExpr>(F))
      F = CE->getSubExpr();
    BO = cast<clang::BinaryOperator>(F);
    assert(BO->getOpcode() == clang::BinaryOperator::Opcode::BO_Mul);
    F = BO->getRHS();
    Incs.push_back(builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), Visit(F).getValue(builder)));
  }

  auto AffineOp = builder.create<scf::ParallelOp>(loc, Inits, Finals, Incs);

  auto Inds = AffineOp.getInductionVars();

  auto OldPoint = builder.getInsertionPoint();
  auto *OldBlock = builder.getInsertionBlock();

  builder.setInsertionPointToStart(&AffineOp.getRegion().front());

  auto ExecuteRegion =
      builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<Type>());
  ExecuteRegion.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&ExecuteRegion.getRegion().back());

  auto *OldScope = allocationScope;
  allocationScope = &ExecuteRegion.getRegion().back();

  std::map<clang::VarDecl *, ValueCategory> PrevInduction;
  for (auto Zp : zip(Inds, Fors->counters())) {
    auto Idx = builder.create<arith::IndexCastOp>(
        loc,
        Glob.getTypes().getMLIRType(Fors->getIterationVariable()->getType()),
        std::get<0>(Zp));
    auto *Name = cast<clang::VarDecl>(
        cast<clang::DeclRefExpr>(std::get<1>(Zp))->getDecl());

    if (params.find(Name) != params.end()) {
      PrevInduction[Name] = params[Name];
      params.erase(Name);
    }

    bool LLVMABI = false;
    bool IsArray = false;
    if (Glob.getTypes()
            .getMLIRType(Glob.getCGM().getContext().getLValueReferenceType(
                Name->getType()))
            .isa<LLVM::LLVMPointerType>())
      LLVMABI = true;
    else
      Glob.getTypes().getMLIRType(Name->getType(), &IsArray);

    auto AllocOp = createAllocOp(Idx.getType(), Name, /*memtype*/ 0,
                                 /*isArray*/ IsArray, /*LLVMABI*/ LLVMABI);
    params[Name] = ValueCategory(AllocOp, true);
    params[Name].store(builder, Idx);
  }

  // TODO: set loop context.
  Visit(Fors->getBody());

  builder.create<scf::YieldOp>(loc);

  allocationScope = OldScope;

  // TODO: set the value of the iteration value to the final bound at the
  // end of the loop.
  builder.setInsertionPoint(OldBlock, OldPoint);

  for (auto Pair : PrevInduction)
    params[Pair.first] = Pair.second;

  return nullptr;
}

ValueCategory MLIRScanner::VisitDoStmt(clang::DoStmt *Fors) {
  IfScope Scope(*this);

  Location Loc = getMLIRLocation(Fors->getDoLoc());

  auto I1Ty = builder.getIntegerType(1);
  auto Type = MemRefType::get({}, I1Ty, {}, 0);
  auto Truev = builder.create<arith::ConstantIntOp>(Loc, true, 1);
  loops.push_back({builder.create<memref::AllocaOp>(Loc, Type),
                   builder.create<memref::AllocaOp>(Loc, Type)});
  builder.create<memref::StoreOp>(Loc, Truev, loops.back().noBreak);

  auto *ToAdd = builder.getInsertionBlock()->getParent();
  auto &CondB = *(new Block());
  ToAdd->getBlocks().push_back(&CondB);
  auto &BodyB = *(new Block());
  ToAdd->getBlocks().push_back(&BodyB);
  auto &ExitB = *(new Block());
  ToAdd->getBlocks().push_back(&ExitB);

  builder.create<cf::BranchOp>(Loc, &BodyB);

  builder.setInsertionPointToStart(&CondB);

  if (auto *S = Fors->getCond()) {
    auto CondRes = Visit(S);
    auto Cond = CondRes.getValue(builder);
    if (auto LT = Cond.getType().dyn_cast<LLVM::LLVMPointerType>()) {
      auto NullptrLlvm = builder.create<LLVM::NullOp>(Loc, LT);
      Cond = builder.create<LLVM::ICmpOp>(Loc, LLVM::ICmpPredicate::ne, Cond,
                                          NullptrLlvm);
    }
    auto Ty = Cond.getType().cast<IntegerType>();
    if (Ty.getWidth() != 1) {
      Cond = builder.create<arith::CmpIOp>(
          Loc, arith::CmpIPredicate::ne, Cond,
          builder.create<arith::ConstantIntOp>(Loc, 0, Ty));
    }
    auto Nb = builder.create<memref::LoadOp>(Loc, loops.back().noBreak,
                                             std::vector<Value>());
    Cond = builder.create<arith::AndIOp>(Loc, Cond, Nb);
    builder.create<cf::CondBranchOp>(Loc, Cond, &BodyB, &ExitB);
  }

  builder.setInsertionPointToStart(&BodyB);
  builder.create<memref::StoreOp>(
      Loc,
      builder.create<memref::LoadOp>(Loc, loops.back().noBreak,
                                     std::vector<Value>()),
      loops.back().keepRunning, std::vector<Value>());

  Visit(Fors->getBody());
  loops.pop_back();

  builder.create<cf::BranchOp>(Loc, &CondB);

  builder.setInsertionPointToStart(&ExitB);

  return nullptr;
}

ValueCategory MLIRScanner::VisitWhileStmt(clang::WhileStmt *Fors) {
  IfScope Scope(*this);

  Location Loc = getMLIRLocation(Fors->getLParenLoc());

  auto I1Ty = builder.getIntegerType(1);
  auto Type = MemRefType::get({}, I1Ty, {}, 0);
  auto Truev = builder.create<arith::ConstantIntOp>(Loc, true, 1);
  loops.push_back({builder.create<memref::AllocaOp>(Loc, Type),
                   builder.create<memref::AllocaOp>(Loc, Type)});
  builder.create<memref::StoreOp>(Loc, Truev, loops.back().noBreak);

  auto *ToAdd = builder.getInsertionBlock()->getParent();
  auto &CondB = *(new Block());
  ToAdd->getBlocks().push_back(&CondB);
  auto &BodyB = *(new Block());
  ToAdd->getBlocks().push_back(&BodyB);
  auto &ExitB = *(new Block());
  ToAdd->getBlocks().push_back(&ExitB);

  builder.create<cf::BranchOp>(Loc, &CondB);

  builder.setInsertionPointToStart(&CondB);

  if (auto *S = Fors->getCond()) {
    auto CondRes = Visit(S);
    auto Cond = CondRes.getValue(builder);
    if (auto LT = Cond.getType().dyn_cast<LLVM::LLVMPointerType>()) {
      auto NullptrLlvm = builder.create<LLVM::NullOp>(Loc, LT);
      Cond = builder.create<LLVM::ICmpOp>(Loc, LLVM::ICmpPredicate::ne, Cond,
                                          NullptrLlvm);
    }
    auto Ty = Cond.getType().cast<IntegerType>();
    if (Ty.getWidth() != 1) {
      Cond = builder.create<arith::CmpIOp>(
          Loc, arith::CmpIPredicate::ne, Cond,
          builder.create<arith::ConstantIntOp>(Loc, 0, Ty));
    }
    auto Nb = builder.create<memref::LoadOp>(Loc, loops.back().noBreak,
                                             std::vector<Value>());
    Cond = builder.create<arith::AndIOp>(Loc, Cond, Nb);
    builder.create<cf::CondBranchOp>(Loc, Cond, &BodyB, &ExitB);
  }

  builder.setInsertionPointToStart(&BodyB);
  builder.create<memref::StoreOp>(
      Loc,
      builder.create<memref::LoadOp>(Loc, loops.back().noBreak,
                                     std::vector<Value>()),
      loops.back().keepRunning, std::vector<Value>());

  Visit(Fors->getBody());
  loops.pop_back();

  builder.create<cf::BranchOp>(Loc, &CondB);

  builder.setInsertionPointToStart(&ExitB);

  return nullptr;
}

ValueCategory MLIRScanner::VisitIfStmt(clang::IfStmt *Stmt) {
  IfScope Scope(*this);
  auto Loc = getMLIRLocation(Stmt->getIfLoc());
  auto Cond = Visit(Stmt->getCond()).getValue(builder);
  assert(Cond != nullptr && "must be a non-null");

  auto OldPoint = builder.getInsertionPoint();
  auto *OldBlock = builder.getInsertionBlock();
  if (auto LT = Cond.getType().dyn_cast<MemRefType>()) {
    Cond = builder.create<polygeist::Memref2PointerOp>(
        Loc, LLVM::LLVMPointerType::get(builder.getI8Type()), Cond);
  }
  if (auto LT = Cond.getType().dyn_cast<LLVM::LLVMPointerType>()) {
    auto NullptrLlvm = builder.create<LLVM::NullOp>(Loc, LT);
    Cond = builder.create<LLVM::ICmpOp>(Loc, LLVM::ICmpPredicate::ne, Cond,
                                        NullptrLlvm);
  }
  if (!Cond.getType().isa<IntegerType>()) {
    Stmt->dump();
    llvm::errs() << " cond: " << Cond << " ct: " << Cond.getType() << "\n";
  }
  auto PrevTy = Cond.getType().cast<IntegerType>();
  if (!PrevTy.isInteger(1)) {
    Cond = builder.create<arith::CmpIOp>(
        Loc, arith::CmpIPredicate::ne, Cond,
        builder.create<arith::ConstantIntOp>(Loc, 0, PrevTy));
  }
  bool HasElseRegion = Stmt->getElse();
  auto IfOp = builder.create<scf::IfOp>(Loc, Cond, HasElseRegion);

  IfOp.getThenRegion().back().clear();
  builder.setInsertionPointToStart(&IfOp.getThenRegion().back());
  Visit(Stmt->getThen());
  builder.create<scf::YieldOp>(Loc);
  if (HasElseRegion) {
    IfOp.getElseRegion().back().clear();
    builder.setInsertionPointToStart(&IfOp.getElseRegion().back());
    Visit(Stmt->getElse());
    builder.create<scf::YieldOp>(Loc);
  }

  builder.setInsertionPoint(OldBlock, OldPoint);
  return nullptr;
}

ValueCategory MLIRScanner::VisitSwitchStmt(clang::SwitchStmt *Stmt) {
  IfScope Scope(*this);
  auto Cond = Visit(Stmt->getCond()).getValue(builder);
  assert(Cond != nullptr);
  SmallVector<int64_t> CaseVals;

  auto Er = builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<Type>());
  Er.getRegion().push_back(new Block());
  auto OldPoint2 = builder.getInsertionPoint();
  auto *OldBlock2 = builder.getInsertionBlock();

  auto &ExitB = *(new Block());
  builder.setInsertionPointToStart(&ExitB);
  builder.create<scf::YieldOp>(loc);
  builder.setInsertionPointToStart(&ExitB);

  SmallVector<Block *> Blocks;
  bool InCase = false;

  Block *DefaultB = &ExitB;

  for (auto *Cse : Stmt->getBody()->children()) {
    if (auto *Cses = dyn_cast<clang::CaseStmt>(Cse)) {
      auto &CondB = *(new Block());

      auto Cval = Visit(Cses->getLHS());
      if (!Cval.val) {
        Cses->getLHS()->dump();
      }
      assert(Cval.val);
      auto Cint = Cval.getValue(builder).getDefiningOp<arith::ConstantIntOp>();
      if (!Cint) {
        Cses->getLHS()->dump();
        llvm::errs() << "cval: " << Cval.val << "\n";
      }
      assert(Cint);
      CaseVals.push_back(Cint.value());

      if (InCase) {
        auto NoBreak =
            builder.create<memref::LoadOp>(loc, loops.back().noBreak);
        builder.create<cf::CondBranchOp>(loc, NoBreak, &CondB, &ExitB);
        loops.pop_back();
      }

      InCase = true;
      Er.getRegion().getBlocks().push_back(&CondB);
      Blocks.push_back(&CondB);
      builder.setInsertionPointToStart(&CondB);

      auto I1Ty = builder.getIntegerType(1);
      auto Type = MemRefType::get({}, I1Ty, {}, 0);
      auto Truev = builder.create<arith::ConstantIntOp>(loc, true, 1);
      loops.push_back({builder.create<memref::AllocaOp>(loc, Type),
                       builder.create<memref::AllocaOp>(loc, Type)});
      builder.create<memref::StoreOp>(loc, Truev, loops.back().noBreak);
      builder.create<memref::StoreOp>(loc, Truev, loops.back().keepRunning);
      Visit(Cses->getSubStmt());
    } else if (auto *Cses = dyn_cast<clang::DefaultStmt>(Cse)) {
      auto &CondB = *(new Block());

      if (InCase) {
        auto NoBreak =
            builder.create<memref::LoadOp>(loc, loops.back().noBreak);
        builder.create<cf::CondBranchOp>(loc, NoBreak, &CondB, &ExitB);
        loops.pop_back();
      }

      InCase = true;
      Er.getRegion().getBlocks().push_back(&CondB);
      builder.setInsertionPointToStart(&CondB);

      auto I1Ty = builder.getIntegerType(1);
      auto Type = MemRefType::get({}, I1Ty, {}, 0);
      auto Truev = builder.create<arith::ConstantIntOp>(loc, true, 1);
      loops.push_back({builder.create<memref::AllocaOp>(loc, Type),
                       builder.create<memref::AllocaOp>(loc, Type)});
      builder.create<memref::StoreOp>(loc, Truev, loops.back().noBreak);
      builder.create<memref::StoreOp>(loc, Truev, loops.back().keepRunning);
      DefaultB = &CondB;
      Visit(Cses->getSubStmt());
    } else {
      Visit(Cse);
    }
  }

  if (CaseVals.size() == 0) {
    delete &ExitB;
    Er.erase();
    builder.setInsertionPoint(OldBlock2, OldPoint2);
    return nullptr;
  }

  if (InCase)
    loops.pop_back();
  builder.create<cf::BranchOp>(loc, &ExitB);

  Er.getRegion().getBlocks().push_back(&ExitB);

  DenseIntElementsAttr CaseValuesAttr;
  ShapedType CaseValueType =
      VectorType::get(static_cast<int64_t>(CaseVals.size()), Cond.getType());
  auto Ity = Cond.getType().cast<IntegerType>();
  if (Ity.getWidth() == 64)
    CaseValuesAttr = DenseIntElementsAttr::get(CaseValueType, CaseVals);
  else if (Ity.getWidth() == 32) {
    SmallVector<int32_t> CaseVals32;
    for (auto V : CaseVals)
      CaseVals32.push_back((int32_t)V);
    CaseValuesAttr = DenseIntElementsAttr::get(CaseValueType, CaseVals32);
  } else if (Ity.getWidth() == 16) {
    SmallVector<int16_t> CaseVals16;
    for (auto V : CaseVals)
      CaseVals16.push_back((int16_t)V);
    CaseValuesAttr = DenseIntElementsAttr::get(CaseValueType, CaseVals16);
  } else {
    assert(Ity.getWidth() == 8);
    SmallVector<int8_t> CaseVals8;
    for (auto V : CaseVals)
      CaseVals8.push_back((int8_t)V);
    CaseValuesAttr = DenseIntElementsAttr::get(CaseValueType, CaseVals8);
  }

  builder.setInsertionPointToStart(&Er.getRegion().front());
  builder.create<cf::SwitchOp>(
      loc, Cond, DefaultB, ArrayRef<Value>(), CaseValuesAttr, Blocks,
      SmallVector<ValueRange>(CaseVals.size(), ArrayRef<Value>()));
  builder.setInsertionPoint(OldBlock2, OldPoint2);
  return nullptr;
}

ValueCategory MLIRScanner::VisitDeclStmt(clang::DeclStmt *Decl) {
  LLVM_DEBUG({
    llvm::dbgs() << "VisitDeclStmt: ";
    Decl->dump();
    llvm::dbgs() << "\n";
  });

  IfScope Scope(*this);
  for (auto *Sub : Decl->decls()) {
    if (auto *Vd = dyn_cast<clang::VarDecl>(Sub)) {
      VisitVarDecl(Vd);
    } else if (isa<clang::TypeAliasDecl, clang::RecordDecl,
                   clang::StaticAssertDecl, clang::TypedefDecl,
                   clang::UsingDecl, clang::UsingDirectiveDecl>(Sub)) {
    } else {
      emitError(getMLIRLocation(Decl->getBeginLoc()))
          << " + visiting unknonwn sub decl stmt\n";
      Sub->dump();
      assert(0 && "unknown sub decl");
    }
  }
  return nullptr;
}

ValueCategory MLIRScanner::VisitAttributedStmt(clang::AttributedStmt *AS) {
  if (!SuppressWarnings)
    emitWarning(getMLIRLocation(AS->getAttrLoc())) << "ignoring attributes\n";

  return Visit(AS->getSubStmt());
}

ValueCategory MLIRScanner::VisitCompoundStmt(clang::CompoundStmt *Stmt) {
  for (auto *A : Stmt->children()) {
    IfScope Scope(*this);
    Visit(A);
  }
  return nullptr;
}

ValueCategory MLIRScanner::VisitBreakStmt(clang::BreakStmt *Stmt) {
  IfScope Scope(*this);
  assert(loops.size() && "must be non-empty");
  assert(loops.back().keepRunning && "keep running false");
  assert(loops.back().noBreak && "no break false");
  auto Vfalse =
      builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), false, 1);
  builder.create<memref::StoreOp>(loc, Vfalse, loops.back().keepRunning);
  builder.create<memref::StoreOp>(loc, Vfalse, loops.back().noBreak);

  return nullptr;
}

ValueCategory MLIRScanner::VisitContinueStmt(clang::ContinueStmt *Stmt) {
  IfScope Scope(*this);
  assert(loops.size() && "must be non-empty");
  assert(loops.back().keepRunning && "keep running false");
  auto Vfalse =
      builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), false, 1);
  builder.create<memref::StoreOp>(loc, Vfalse, loops.back().keepRunning);
  return nullptr;
}

ValueCategory MLIRScanner::VisitLabelStmt(clang::LabelStmt *Stmt) {
  auto *ToAdd = builder.getInsertionBlock()->getParent();
  Block *LabelB;
  auto Found = labels.find(Stmt);
  if (Found != labels.end()) {
    LabelB = Found->second;
  } else {
    LabelB = new Block();
    labels[Stmt] = LabelB;
  }
  ToAdd->getBlocks().push_back(LabelB);
  builder.create<cf::BranchOp>(loc, LabelB);
  builder.setInsertionPointToStart(LabelB);
  Visit(Stmt->getSubStmt());
  return nullptr;
}

ValueCategory MLIRScanner::VisitGotoStmt(clang::GotoStmt *Stmt) {
  auto *Labelstmt = Stmt->getLabel()->getStmt();
  Block *LabelB;
  auto Found = labels.find(Labelstmt);
  if (Found != labels.end()) {
    LabelB = Found->second;
  } else {
    LabelB = new Block();
    labels[Labelstmt] = LabelB;
  }
  builder.create<cf::BranchOp>(loc, LabelB);
  return nullptr;
}

ValueCategory MLIRScanner::VisitCXXTryStmt(clang::CXXTryStmt *Stmt) {
  mlirclang::warning() << "not performing catches for try stmt\n";
  return Visit(Stmt->getTryBlock());
}

ValueCategory MLIRScanner::VisitReturnStmt(clang::ReturnStmt *Stmt) {
  IfScope Scope(*this);
  bool IsArrayReturn = false;
  Glob.getTypes().getMLIRType(EmittingFunctionDecl->getReturnType(),
                              &IsArrayReturn);

  if (IsArrayReturn) {
    auto Rv = Visit(Stmt->getRetValue());
    assert(Rv.val && "expect right value to be valid");
    assert(Rv.isReference && "right value must be a reference");
    auto Op = function.getArgument(function.getNumArguments() - 1);
    assert(Rv.val.getType().cast<MemRefType>().getElementType() ==
               Op.getType().cast<MemRefType>().getElementType() &&
           "type mismatch");
    assert(Op.getType().cast<MemRefType>().getShape().size() == 2 &&
           "expect 2d memref");
    assert(Rv.val.getType().cast<MemRefType>().getShape().size() == 2 &&
           "expect 2d memref");
    assert(Rv.val.getType().cast<MemRefType>().getShape()[1] ==
           Op.getType().cast<MemRefType>().getShape()[1]);

    for (int I = 0; I < Op.getType().cast<MemRefType>().getShape()[1]; I++) {
      std::vector<Value> Idx = {getConstantIndex(0), getConstantIndex(I)};
      assert(Rv.val.getType().cast<MemRefType>().getShape().size() == 2);
      builder.create<memref::StoreOp>(
          loc, builder.create<memref::LoadOp>(loc, Rv.val, Idx), Op, Idx);
    }
  } else if (Stmt->getRetValue()) {
    auto Rv = Visit(Stmt->getRetValue());
    if (!Stmt->getRetValue()->getType()->isVoidType()) {
      if (!Rv.val) {
        Stmt->dump();
      }
      assert(Rv.val && "expect right value to be valid");

      Value Val;
      if (Stmt->getRetValue()->isLValue() || Stmt->getRetValue()->isXValue()) {
        assert(Rv.isReference);
        Val = Rv.val;
      } else {
        Val = Rv.getValue(builder);
      }

      auto PostTy = returnVal.getType().cast<MemRefType>().getElementType();
      if (auto PrevTy = Val.getType().dyn_cast<IntegerType>()) {
        const auto SrcTy = Stmt->getRetValue()->getType();
        const auto IsSigned =
            SrcTy->isBooleanType() ? false : SrcTy->isSignedIntegerType();
        Val = Rv.IntCast(builder, getMLIRLocation(Stmt->getReturnLoc()), PostTy,
                         IsSigned)
                  .val;
      } else if (Val.getType().isa<MemRefType>() &&
                 PostTy.isa<LLVM::LLVMPointerType>())
        Val = builder.create<polygeist::Memref2PointerOp>(loc, PostTy, Val);
      else if (Val.getType().isa<LLVM::LLVMPointerType>() &&
               PostTy.isa<MemRefType>())
        Val = builder.create<polygeist::Pointer2MemrefOp>(loc, PostTy, Val);
      if (PostTy != Val.getType()) {
        Stmt->dump();
        llvm::errs() << " val: " << Val << " postTy: " << PostTy
                     << " rv.val: " << Rv.val << " rv.isRef"
                     << (int)Rv.isReference << " mm: "
                     << (int)(Stmt->getRetValue()->isLValue() ||
                              Stmt->getRetValue()->isXValue())
                     << "\n";
      }
      assert(PostTy == Val.getType());
      builder.create<memref::StoreOp>(loc, Val, returnVal);
    }
  }

  assert(loops.size() && "must be non-empty");
  auto Vfalse =
      builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), false, 1);
  for (auto L : loops) {
    builder.create<memref::StoreOp>(loc, Vfalse, L.keepRunning);
    builder.create<memref::StoreOp>(loc, Vfalse, L.noBreak);
  }

  return nullptr;
}
