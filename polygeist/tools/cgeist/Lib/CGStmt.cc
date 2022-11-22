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

using namespace clang;
using namespace mlir;
using namespace mlir::arith;

extern llvm::cl::opt<bool> SuppressWarnings;

static bool isTerminator(Operation *op) {
  return op->mightHaveTrait<OpTrait::IsTerminator>();
}

bool MLIRScanner::getLowerBound(clang::ForStmt *fors,
                                mlirclang::AffineLoopDescriptor &descr) {
  auto *init = fors->getInit();
  if (auto *declStmt = dyn_cast<DeclStmt>(init))
    if (declStmt->isSingleDecl()) {
      auto *decl = declStmt->getSingleDecl();
      if (auto *varDecl = dyn_cast<VarDecl>(decl)) {
        if (varDecl->hasInit()) {
          mlir::Value val = VisitVarDecl(varDecl).getValue(Builder);
          descr.setName(varDecl);
          descr.setType(val.getType());
          LLVM_DEBUG(descr.getType().print(llvm::dbgs()));

          if (descr.getForwardMode())
            descr.setLowerBound(val);
          else {
            val = Builder.create<AddIOp>(Loc, val, getConstantIndex(1));
            descr.setUpperBound(val);
          }
          return true;
        }
      }
    }

  // BinaryOperator 0x7ff7aa17e938 'int' '='
  // |-DeclRefExpr 0x7ff7aa17e8f8 'int' lvalue Var 0x7ff7aa17e758 'i' 'int'
  // -IntegerLiteral 0x7ff7aa17e918 'int' 0
  if (auto *binOp = dyn_cast<clang::BinaryOperator>(init))
    if (binOp->getOpcode() == clang::BinaryOperator::Opcode::BO_Assign)
      if (auto *declRefStmt = dyn_cast<DeclRefExpr>(binOp->getLHS())) {
        mlir::Value val = Visit(binOp->getRHS()).getValue(Builder);
        val = Builder.create<IndexCastOp>(
            Loc, mlir::IndexType::get(Builder.getContext()), val);
        descr.setName(cast<VarDecl>(declRefStmt->getDecl()));
        descr.setType(
            Glob.getTypes().getMLIRType(declRefStmt->getDecl()->getType()));
        if (descr.getForwardMode())
          descr.setLowerBound(val);
        else {
          val = Builder.create<AddIOp>(Loc, val, getConstantIndex(1));
          descr.setUpperBound(val);
        }
        return true;
      }
  return false;
}

// Make sure that the induction variable initialized in
// the for is the same as the one used in the condition.
bool matchIndvar(const Expr *expr, VarDecl *indVar) {
  while (const auto *IC = dyn_cast<ImplicitCastExpr>(expr)) {
    expr = IC->getSubExpr();
  }
  if (const auto *declRef = dyn_cast<DeclRefExpr>(expr)) {
    const auto *declRefName = declRef->getDecl();
    if (declRefName == indVar)
      return true;
  }
  return false;
}

bool MLIRScanner::getUpperBound(clang::ForStmt *fors,
                                mlirclang::AffineLoopDescriptor &descr) {
  auto *cond = fors->getCond();
  if (auto *binaryOp = dyn_cast<clang::BinaryOperator>(cond)) {
    auto *lhs = binaryOp->getLHS();
    if (!matchIndvar(lhs, descr.getName()))
      return false;

    if (descr.getForwardMode()) {
      if (binaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_LT &&
          binaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_LE)
        return false;

      auto *rhs = binaryOp->getRHS();
      mlir::Value val = Visit(rhs).getValue(Builder);
      val = Builder.create<IndexCastOp>(
          Loc, mlir::IndexType::get(val.getContext()), val);
      if (binaryOp->getOpcode() == clang::BinaryOperator::Opcode::BO_LE)
        val = Builder.create<AddIOp>(Loc, val, getConstantIndex(1));
      descr.setUpperBound(val);
      return true;
    } else {
      if (binaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_GT &&
          binaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_GE)
        return false;

      auto *rhs = binaryOp->getRHS();
      mlir::Value val = Visit(rhs).getValue(Builder);
      val = Builder.create<IndexCastOp>(
          Loc, mlir::IndexType::get(val.getContext()), val);
      if (binaryOp->getOpcode() == clang::BinaryOperator::Opcode::BO_GT)
        val = Builder.create<AddIOp>(Loc, val, getConstantIndex(1));
      descr.setLowerBound(val);
      return true;
    }
  }
  return false;
}

bool MLIRScanner::getConstantStep(clang::ForStmt *fors,
                                  mlirclang::AffineLoopDescriptor &descr) {
  auto *inc = fors->getInc();
  if (auto *unaryOp = dyn_cast<clang::UnaryOperator>(inc))
    if (unaryOp->isPrefix() || unaryOp->isPostfix()) {
      bool forwardLoop =
          unaryOp->getOpcode() == clang::UnaryOperator::Opcode::UO_PostInc ||
          unaryOp->getOpcode() == clang::UnaryOperator::Opcode::UO_PreInc;
      descr.setStep(1);
      descr.setForwardMode(forwardLoop);
      return true;
    }
  return false;
}

bool MLIRScanner::isTrivialAffineLoop(clang::ForStmt *fors,
                                      mlirclang::AffineLoopDescriptor &descr) {
  if (!getConstantStep(fors, descr)) {
    LLVM_DEBUG(llvm::dbgs() << "getConstantStep -> false\n");
    return false;
  }
  if (!getLowerBound(fors, descr)) {
    LLVM_DEBUG(llvm::dbgs() << "getLowerBound -> false\n");
    return false;
  }
  if (!getUpperBound(fors, descr)) {
    LLVM_DEBUG(llvm::dbgs() << "getUpperBound -> false\n");
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "isTrivialAffineLoop -> true\n");
  return true;
}

void MLIRScanner::buildAffineLoopImpl(
    clang::ForStmt *fors, mlir::Location Loc, mlir::Value lb, mlir::Value ub,
    const mlirclang::AffineLoopDescriptor &descr) {
  auto affineOp = Builder.create<AffineForOp>(
      Loc, lb, Builder.getSymbolIdentityMap(), ub,
      Builder.getSymbolIdentityMap(), descr.getStep(),
      /*iterArgs=*/llvm::None);

  auto &reg = affineOp.getLoopBody();

  auto val = (mlir::Value)affineOp.getInductionVar();

  reg.front().clear();

  auto oldpoint = Builder.getInsertionPoint();
  auto *oldblock = Builder.getInsertionBlock();

  Builder.setInsertionPointToEnd(&reg.front());

  auto er = Builder.create<scf::ExecuteRegionOp>(Loc, ArrayRef<mlir::Type>());
  er.getRegion().push_back(new Block());
  Builder.setInsertionPointToStart(&er.getRegion().back());
  Builder.create<scf::YieldOp>(Loc);
  Builder.setInsertionPointToStart(&er.getRegion().back());

  if (!descr.getForwardMode()) {
    val = Builder.create<SubIOp>(Loc, val, lb);
    val = Builder.create<SubIOp>(
        Loc, Builder.create<SubIOp>(Loc, ub, getConstantIndex(1)), val);
  }
  auto idx = Builder.create<IndexCastOp>(Loc, descr.getType(), val);
  assert(Params.find(descr.getName()) != Params.end());
  Params[descr.getName()].store(Builder, idx);

  // TODO: set loop context.
  Visit(fors->getBody());

  Builder.setInsertionPointToEnd(&reg.front());
  Builder.create<AffineYieldOp>(Loc);

  // TODO: set the value of the iteration value to the final bound at the
  // end of the loop.
  Builder.setInsertionPoint(oldblock, oldpoint);
}

void MLIRScanner::buildAffineLoop(
    clang::ForStmt *fors, mlir::Location Loc,
    const mlirclang::AffineLoopDescriptor &descr) {
  mlir::Value lb = descr.getLowerBound();
  mlir::Value ub = descr.getUpperBound();
  buildAffineLoopImpl(fors, Loc, lb, ub, descr);
}

ValueCategory MLIRScanner::VisitForStmt(clang::ForStmt *fors) {
  IfScope scope(*this);

  auto Loc = getMLIRLocation(fors->getForLoc());

  mlirclang::AffineLoopDescriptor affineLoopDescr;
  if (Glob.getScopLocList().isInScop(fors->getForLoc()) &&
      isTrivialAffineLoop(fors, affineLoopDescr)) {
    buildAffineLoop(fors, Loc, affineLoopDescr);
  } else {

    if (auto *s = fors->getInit()) {
      Visit(s);
    }

    auto i1Ty = Builder.getIntegerType(1);
    auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
    auto truev = Builder.create<ConstantIntOp>(Loc, true, 1);

    LoopContext lctx{Builder.create<mlir::memref::AllocaOp>(Loc, type),
                     Builder.create<mlir::memref::AllocaOp>(Loc, type)};
    Builder.create<mlir::memref::StoreOp>(Loc, truev, lctx.NoBreak);

    auto *toadd = Builder.getInsertionBlock()->getParent();
    auto &condB = *(new Block());
    toadd->getBlocks().push_back(&condB);
    auto &bodyB = *(new Block());
    toadd->getBlocks().push_back(&bodyB);
    auto &exitB = *(new Block());
    toadd->getBlocks().push_back(&exitB);

    Builder.create<mlir::cf::BranchOp>(Loc, &condB);

    Builder.setInsertionPointToStart(&condB);

    if (auto *s = fors->getCond()) {
      auto condRes = Visit(s);
      auto cond = condRes.getValue(Builder);
      if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
        auto nullptr_llvm = Builder.create<mlir::LLVM::NullOp>(Loc, LT);
        cond = Builder.create<mlir::LLVM::ICmpOp>(
            Loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
      }
      auto ty = cond.getType().cast<mlir::IntegerType>();
      if (ty.getWidth() != 1) {
        cond = Builder.create<arith::CmpIOp>(
            Loc, CmpIPredicate::ne, cond,
            Builder.create<ConstantIntOp>(Loc, 0, ty));
      }
      auto nb = Builder.create<mlir::memref::LoadOp>(
          Loc, lctx.NoBreak, std::vector<mlir::Value>());
      cond = Builder.create<AndIOp>(Loc, cond, nb);
      Builder.create<mlir::cf::CondBranchOp>(Loc, cond, &bodyB, &exitB);
    } else {
      auto cond = Builder.create<mlir::memref::LoadOp>(
          Loc, lctx.NoBreak, std::vector<mlir::Value>());
      Builder.create<mlir::cf::CondBranchOp>(Loc, cond, &bodyB, &exitB);
    }

    Builder.setInsertionPointToStart(&bodyB);
    Builder.create<mlir::memref::StoreOp>(
        Loc,
        Builder.create<mlir::memref::LoadOp>(Loc, lctx.NoBreak,
                                             std::vector<mlir::Value>()),
        lctx.KeepRunning, std::vector<mlir::Value>());

    Loops.push_back(lctx);
    Visit(fors->getBody());

    Builder.create<mlir::memref::StoreOp>(
        Loc,
        Builder.create<mlir::memref::LoadOp>(Loc, lctx.NoBreak,
                                             std::vector<mlir::Value>()),
        lctx.KeepRunning, std::vector<mlir::Value>());
    if (auto *s = fors->getInc()) {
      IfScope scope(*this);
      Visit(s);
    }
    Loops.pop_back();
    if (Builder.getInsertionBlock()->empty() ||
        !isTerminator(&Builder.getInsertionBlock()->back())) {
      Builder.create<mlir::cf::BranchOp>(Loc, &condB);
    }

    Builder.setInsertionPointToStart(&exitB);
  }
  return nullptr;
}

ValueCategory MLIRScanner::VisitCXXForRangeStmt(clang::CXXForRangeStmt *fors) {
  IfScope scope(*this);

  auto Loc = getMLIRLocation(fors->getForLoc());

  if (auto *s = fors->getInit()) {
    Visit(s);
  }
  Visit(fors->getRangeStmt());
  Visit(fors->getBeginStmt());
  Visit(fors->getEndStmt());

  auto i1Ty = Builder.getIntegerType(1);
  auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
  auto truev = Builder.create<ConstantIntOp>(Loc, true, 1);

  LoopContext lctx{Builder.create<mlir::memref::AllocaOp>(Loc, type),
                   Builder.create<mlir::memref::AllocaOp>(Loc, type)};
  Builder.create<mlir::memref::StoreOp>(Loc, truev, lctx.NoBreak);

  auto *toadd = Builder.getInsertionBlock()->getParent();
  auto &condB = *(new Block());
  toadd->getBlocks().push_back(&condB);
  auto &bodyB = *(new Block());
  toadd->getBlocks().push_back(&bodyB);
  auto &exitB = *(new Block());
  toadd->getBlocks().push_back(&exitB);

  Builder.create<mlir::cf::BranchOp>(Loc, &condB);

  Builder.setInsertionPointToStart(&condB);

  if (auto *s = fors->getCond()) {
    auto condRes = Visit(s);
    auto cond = condRes.getValue(Builder);
    if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = Builder.create<mlir::LLVM::NullOp>(Loc, LT);
      cond = Builder.create<mlir::LLVM::ICmpOp>(
          Loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
    }
    auto ty = cond.getType().cast<mlir::IntegerType>();
    if (ty.getWidth() != 1) {
      cond = Builder.create<arith::CmpIOp>(
          Loc, CmpIPredicate::ne, cond,
          Builder.create<ConstantIntOp>(Loc, 0, ty));
    }
    auto nb = Builder.create<mlir::memref::LoadOp>(Loc, lctx.NoBreak,
                                                   std::vector<mlir::Value>());
    cond = Builder.create<AndIOp>(Loc, cond, nb);
    Builder.create<mlir::cf::CondBranchOp>(Loc, cond, &bodyB, &exitB);
  } else {
    auto cond = Builder.create<mlir::memref::LoadOp>(
        Loc, lctx.NoBreak, std::vector<mlir::Value>());
    Builder.create<mlir::cf::CondBranchOp>(Loc, cond, &bodyB, &exitB);
  }

  Builder.setInsertionPointToStart(&bodyB);
  Builder.create<mlir::memref::StoreOp>(
      Loc,
      Builder.create<mlir::memref::LoadOp>(Loc, lctx.NoBreak,
                                           std::vector<mlir::Value>()),
      lctx.KeepRunning, std::vector<mlir::Value>());

  Loops.push_back(lctx);
  Visit(fors->getLoopVarStmt());
  Visit(fors->getBody());

  Builder.create<mlir::memref::StoreOp>(
      Loc,
      Builder.create<mlir::memref::LoadOp>(Loc, lctx.NoBreak,
                                           std::vector<mlir::Value>()),
      lctx.KeepRunning, std::vector<mlir::Value>());
  if (auto *s = fors->getInc()) {
    IfScope scope(*this);
    Visit(s);
  }
  Loops.pop_back();
  if (Builder.getInsertionBlock()->empty() ||
      !isTerminator(&Builder.getInsertionBlock()->back())) {
    Builder.create<mlir::cf::BranchOp>(Loc, &condB);
  }

  Builder.setInsertionPointToStart(&exitB);
  return nullptr;
}

ValueCategory
MLIRScanner::VisitOMPSingleDirective(clang::OMPSingleDirective *par) {
  IfScope scope(*this);

  Builder.create<omp::BarrierOp>(Loc);
  auto affineOp = Builder.create<omp::MasterOp>(Loc);
  Builder.create<omp::BarrierOp>(Loc);

  auto oldpoint = Builder.getInsertionPoint();
  auto *oldblock = Builder.getInsertionBlock();

  affineOp.getRegion().push_back(new Block());
  Builder.setInsertionPointToStart(&affineOp.getRegion().front());

  auto executeRegion =
      Builder.create<scf::ExecuteRegionOp>(Loc, ArrayRef<mlir::Type>());
  executeRegion.getRegion().push_back(new Block());
  Builder.create<omp::TerminatorOp>(Loc);
  Builder.setInsertionPointToStart(&executeRegion.getRegion().back());

  auto *oldScope = AllocationScope;
  AllocationScope = &executeRegion.getRegion().back();

  Visit(cast<CapturedStmt>(par->getAssociatedStmt())
            ->getCapturedDecl()
            ->getBody());

  Builder.create<scf::YieldOp>(Loc);
  AllocationScope = oldScope;
  Builder.setInsertionPoint(oldblock, oldpoint);
  return nullptr;
}

ValueCategory MLIRScanner::VisitOMPForDirective(clang::OMPForDirective *fors) {
  IfScope scope(*this);

  if (fors->getPreInits()) {
    Visit(fors->getPreInits());
  }

  SmallVector<mlir::Value> inits;
  for (auto *f : fors->inits()) {
    assert(f);
    f = cast<clang::BinaryOperator>(f)->getRHS();
    inits.push_back(Builder.create<IndexCastOp>(Loc, Builder.getIndexType(),
                                                Visit(f).getValue(Builder)));
  }

  SmallVector<mlir::Value> finals;
  for (auto *f : fors->finals()) {
    f = cast<clang::BinaryOperator>(f)->getRHS();
    finals.push_back(Builder.create<IndexCastOp>(Loc, Builder.getIndexType(),
                                                 Visit(f).getValue(Builder)));
  }

  SmallVector<mlir::Value> incs;
  for (auto *f : fors->updates()) {
    f = cast<clang::BinaryOperator>(f)->getRHS();
    while (auto *ce = dyn_cast<clang::CastExpr>(f))
      f = ce->getSubExpr();
    auto *bo = cast<clang::BinaryOperator>(f);
    assert(bo->getOpcode() == clang::BinaryOperator::Opcode::BO_Add);
    f = bo->getRHS();
    while (auto *ce = dyn_cast<clang::CastExpr>(f))
      f = ce->getSubExpr();
    bo = cast<clang::BinaryOperator>(f);
    assert(bo->getOpcode() == clang::BinaryOperator::Opcode::BO_Mul);
    f = bo->getRHS();
    incs.push_back(Builder.create<IndexCastOp>(Loc, Builder.getIndexType(),
                                               Visit(f).getValue(Builder)));
  }

  auto affineOp = Builder.create<omp::WsLoopOp>(Loc, inits, finals, incs);
  affineOp.getRegion().push_back(new Block());
  for (auto init : inits)
    affineOp.getRegion().front().addArgument(init.getType(), init.getLoc());
  auto inds = affineOp.getRegion().front().getArguments();

  auto oldpoint = Builder.getInsertionPoint();
  auto *oldblock = Builder.getInsertionBlock();

  Builder.setInsertionPointToStart(&affineOp.getRegion().front());

  auto executeRegion =
      Builder.create<scf::ExecuteRegionOp>(Loc, ArrayRef<mlir::Type>());
  Builder.create<omp::YieldOp>(Loc, ValueRange());
  executeRegion.getRegion().push_back(new Block());
  Builder.setInsertionPointToStart(&executeRegion.getRegion().back());

  auto *oldScope = AllocationScope;
  AllocationScope = &executeRegion.getRegion().back();

  std::map<VarDecl *, ValueCategory> prevInduction;
  for (auto zp : zip(inds, fors->counters())) {
    auto idx = Builder.create<IndexCastOp>(
        Loc,
        Glob.getTypes().getMLIRType(fors->getIterationVariable()->getType()),
        std::get<0>(zp));
    VarDecl *name =
        cast<VarDecl>(cast<DeclRefExpr>(std::get<1>(zp))->getDecl());

    if (Params.find(name) != Params.end()) {
      prevInduction[name] = Params[name];
      Params.erase(name);
    }

    bool LLVMABI = false;
    bool isArray = false;
    if (Glob.getTypes()
            .getMLIRType(Glob.getCGM().getContext().getLValueReferenceType(
                name->getType()))
            .isa<mlir::LLVM::LLVMPointerType>())
      LLVMABI = true;
    else
      Glob.getTypes().getMLIRType(name->getType(), &isArray);

    auto allocop = createAllocOp(idx.getType(), name, /*memtype*/ 0,
                                 /*isArray*/ isArray, /*LLVMABI*/ LLVMABI);
    Params[name] = ValueCategory(allocop, true);
    Params[name].store(Builder, idx);
  }

  // TODO: set loop context.
  Visit(fors->getBody());

  Builder.create<scf::YieldOp>(Loc, ValueRange());

  AllocationScope = oldScope;

  // TODO: set the value of the iteration value to the final bound at the
  // end of the loop.
  Builder.setInsertionPoint(oldblock, oldpoint);

  for (auto pair : prevInduction)
    Params[pair.first] = pair.second;

  return nullptr;
}

ValueCategory
MLIRScanner::VisitOMPParallelDirective(clang::OMPParallelDirective *par) {
  IfScope scope(*this);

  auto affineOp = Builder.create<omp::ParallelOp>(Loc);

  auto oldpoint = Builder.getInsertionPoint();
  auto *oldblock = Builder.getInsertionBlock();

  affineOp.getRegion().push_back(new Block());
  Builder.setInsertionPointToStart(&affineOp.getRegion().front());

  auto executeRegion =
      Builder.create<scf::ExecuteRegionOp>(Loc, ArrayRef<mlir::Type>());
  executeRegion.getRegion().push_back(new Block());
  Builder.create<omp::TerminatorOp>(Loc);
  Builder.setInsertionPointToStart(&executeRegion.getRegion().back());

  auto *oldScope = AllocationScope;
  AllocationScope = &executeRegion.getRegion().back();

  std::map<VarDecl *, ValueCategory> prevInduction;
  for (auto *f : par->clauses()) {
    switch (f->getClauseKind()) {
    case llvm::omp::OMPC_private:
      for (auto *stmt : f->children()) {
        VarDecl *name = cast<VarDecl>(cast<DeclRefExpr>(stmt)->getDecl());

        prevInduction[name] = Params[name];
        Params.erase(name);

        bool LLVMABI = false;
        bool isArray = false;
        mlir::Type ty;
        if (Glob.getTypes()
                .getMLIRType(Glob.getCGM().getContext().getLValueReferenceType(
                    name->getType()))
                .isa<mlir::LLVM::LLVMPointerType>()) {
          LLVMABI = true;
          bool undef;
          ty = Glob.getTypes().getMLIRType(name->getType(), &undef);
        } else
          ty = Glob.getTypes().getMLIRType(name->getType(), &isArray);

        auto allocop = createAllocOp(ty, name, /*memtype*/ 0,
                                     /*isArray*/ isArray, /*LLVMABI*/ LLVMABI);
        Params[name] = ValueCategory(allocop, true);
        Params[name].store(Builder, prevInduction[name], isArray);
      }
      break;
    default:
      llvm::errs() << "may not handle omp clause " << (int)f->getClauseKind()
                   << "\n";
    }
  }

  Visit(cast<CapturedStmt>(par->getAssociatedStmt())
            ->getCapturedDecl()
            ->getBody());

  Builder.create<scf::YieldOp>(Loc);
  AllocationScope = oldScope;
  Builder.setInsertionPoint(oldblock, oldpoint);

  for (auto pair : prevInduction)
    Params[pair.first] = pair.second;
  return nullptr;
}

ValueCategory MLIRScanner::VisitOMPParallelForDirective(
    clang::OMPParallelForDirective *fors) {
  IfScope scope(*this);

  if (fors->getPreInits()) {
    Visit(fors->getPreInits());
  }

  SmallVector<mlir::Value> inits;
  for (auto *f : fors->inits()) {
    assert(f);
    f = cast<clang::BinaryOperator>(f)->getRHS();
    inits.push_back(Builder.create<IndexCastOp>(Loc, Builder.getIndexType(),
                                                Visit(f).getValue(Builder)));
  }

  SmallVector<mlir::Value> finals;
  for (auto *f : fors->finals()) {
    f = cast<clang::BinaryOperator>(f)->getRHS();
    finals.push_back(Builder.create<arith::IndexCastOp>(
        Loc, Builder.getIndexType(), Visit(f).getValue(Builder)));
  }

  SmallVector<mlir::Value> incs;
  for (auto *f : fors->updates()) {
    f = cast<clang::BinaryOperator>(f)->getRHS();
    while (auto *ce = dyn_cast<clang::CastExpr>(f))
      f = ce->getSubExpr();
    auto *bo = cast<clang::BinaryOperator>(f);
    assert(bo->getOpcode() == clang::BinaryOperator::Opcode::BO_Add);
    f = bo->getRHS();
    while (auto *ce = dyn_cast<clang::CastExpr>(f))
      f = ce->getSubExpr();
    bo = cast<clang::BinaryOperator>(f);
    assert(bo->getOpcode() == clang::BinaryOperator::Opcode::BO_Mul);
    f = bo->getRHS();
    incs.push_back(Builder.create<IndexCastOp>(Loc, Builder.getIndexType(),
                                               Visit(f).getValue(Builder)));
  }

  auto affineOp = Builder.create<scf::ParallelOp>(Loc, inits, finals, incs);

  auto inds = affineOp.getInductionVars();

  auto oldpoint = Builder.getInsertionPoint();
  auto *oldblock = Builder.getInsertionBlock();

  Builder.setInsertionPointToStart(&affineOp.getRegion().front());

  auto executeRegion =
      Builder.create<scf::ExecuteRegionOp>(Loc, ArrayRef<mlir::Type>());
  executeRegion.getRegion().push_back(new Block());
  Builder.setInsertionPointToStart(&executeRegion.getRegion().back());

  auto *oldScope = AllocationScope;
  AllocationScope = &executeRegion.getRegion().back();

  std::map<VarDecl *, ValueCategory> prevInduction;
  for (auto zp : zip(inds, fors->counters())) {
    auto idx = Builder.create<IndexCastOp>(
        Loc,
        Glob.getTypes().getMLIRType(fors->getIterationVariable()->getType()),
        std::get<0>(zp));
    VarDecl *name =
        cast<VarDecl>(cast<DeclRefExpr>(std::get<1>(zp))->getDecl());

    if (Params.find(name) != Params.end()) {
      prevInduction[name] = Params[name];
      Params.erase(name);
    }

    bool LLVMABI = false;
    bool isArray = false;
    if (Glob.getTypes()
            .getMLIRType(Glob.getCGM().getContext().getLValueReferenceType(
                name->getType()))
            .isa<mlir::LLVM::LLVMPointerType>())
      LLVMABI = true;
    else
      Glob.getTypes().getMLIRType(name->getType(), &isArray);

    auto allocop = createAllocOp(idx.getType(), name, /*memtype*/ 0,
                                 /*isArray*/ isArray, /*LLVMABI*/ LLVMABI);
    Params[name] = ValueCategory(allocop, true);
    Params[name].store(Builder, idx);
  }

  // TODO: set loop context.
  Visit(fors->getBody());

  Builder.create<scf::YieldOp>(Loc);

  AllocationScope = oldScope;

  // TODO: set the value of the iteration value to the final bound at the
  // end of the loop.
  Builder.setInsertionPoint(oldblock, oldpoint);

  for (auto pair : prevInduction)
    Params[pair.first] = pair.second;

  return nullptr;
}

ValueCategory MLIRScanner::VisitDoStmt(clang::DoStmt *fors) {
  IfScope scope(*this);

  auto Loc = getMLIRLocation(fors->getDoLoc());

  auto i1Ty = Builder.getIntegerType(1);
  auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
  auto truev = Builder.create<ConstantIntOp>(Loc, true, 1);
  Loops.push_back({Builder.create<mlir::memref::AllocaOp>(Loc, type),
                   Builder.create<mlir::memref::AllocaOp>(Loc, type)});
  Builder.create<mlir::memref::StoreOp>(Loc, truev, Loops.back().NoBreak);

  auto *toadd = Builder.getInsertionBlock()->getParent();
  auto &condB = *(new Block());
  toadd->getBlocks().push_back(&condB);
  auto &bodyB = *(new Block());
  toadd->getBlocks().push_back(&bodyB);
  auto &exitB = *(new Block());
  toadd->getBlocks().push_back(&exitB);

  Builder.create<mlir::cf::BranchOp>(Loc, &bodyB);

  Builder.setInsertionPointToStart(&condB);

  if (auto *s = fors->getCond()) {
    auto condRes = Visit(s);
    auto cond = condRes.getValue(Builder);
    if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = Builder.create<mlir::LLVM::NullOp>(Loc, LT);
      cond = Builder.create<mlir::LLVM::ICmpOp>(
          Loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
    }
    auto ty = cond.getType().cast<mlir::IntegerType>();
    if (ty.getWidth() != 1) {
      cond = Builder.create<arith::CmpIOp>(
          Loc, CmpIPredicate::ne, cond,
          Builder.create<ConstantIntOp>(Loc, 0, ty));
    }
    auto nb = Builder.create<mlir::memref::LoadOp>(Loc, Loops.back().NoBreak,
                                                   std::vector<mlir::Value>());
    cond = Builder.create<AndIOp>(Loc, cond, nb);
    Builder.create<mlir::cf::CondBranchOp>(Loc, cond, &bodyB, &exitB);
  }

  Builder.setInsertionPointToStart(&bodyB);
  Builder.create<mlir::memref::StoreOp>(
      Loc,
      Builder.create<mlir::memref::LoadOp>(Loc, Loops.back().NoBreak,
                                           std::vector<mlir::Value>()),
      Loops.back().KeepRunning, std::vector<mlir::Value>());

  Visit(fors->getBody());
  Loops.pop_back();

  Builder.create<mlir::cf::BranchOp>(Loc, &condB);

  Builder.setInsertionPointToStart(&exitB);

  return nullptr;
}

ValueCategory MLIRScanner::VisitWhileStmt(clang::WhileStmt *fors) {
  IfScope scope(*this);

  auto Loc = getMLIRLocation(fors->getLParenLoc());

  auto i1Ty = Builder.getIntegerType(1);
  auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
  auto truev = Builder.create<ConstantIntOp>(Loc, true, 1);
  Loops.push_back({Builder.create<mlir::memref::AllocaOp>(Loc, type),
                   Builder.create<mlir::memref::AllocaOp>(Loc, type)});
  Builder.create<mlir::memref::StoreOp>(Loc, truev, Loops.back().NoBreak);

  auto *toadd = Builder.getInsertionBlock()->getParent();
  auto &condB = *(new Block());
  toadd->getBlocks().push_back(&condB);
  auto &bodyB = *(new Block());
  toadd->getBlocks().push_back(&bodyB);
  auto &exitB = *(new Block());
  toadd->getBlocks().push_back(&exitB);

  Builder.create<mlir::cf::BranchOp>(Loc, &condB);

  Builder.setInsertionPointToStart(&condB);

  if (auto *s = fors->getCond()) {
    auto condRes = Visit(s);
    auto cond = condRes.getValue(Builder);
    if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = Builder.create<mlir::LLVM::NullOp>(Loc, LT);
      cond = Builder.create<mlir::LLVM::ICmpOp>(
          Loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
    }
    auto ty = cond.getType().cast<mlir::IntegerType>();
    if (ty.getWidth() != 1) {
      cond = Builder.create<arith::CmpIOp>(
          Loc, CmpIPredicate::ne, cond,
          Builder.create<ConstantIntOp>(Loc, 0, ty));
    }
    auto nb = Builder.create<mlir::memref::LoadOp>(Loc, Loops.back().NoBreak,
                                                   std::vector<mlir::Value>());
    cond = Builder.create<AndIOp>(Loc, cond, nb);
    Builder.create<mlir::cf::CondBranchOp>(Loc, cond, &bodyB, &exitB);
  }

  Builder.setInsertionPointToStart(&bodyB);
  Builder.create<mlir::memref::StoreOp>(
      Loc,
      Builder.create<mlir::memref::LoadOp>(Loc, Loops.back().NoBreak,
                                           std::vector<mlir::Value>()),
      Loops.back().KeepRunning, std::vector<mlir::Value>());

  Visit(fors->getBody());
  Loops.pop_back();

  Builder.create<mlir::cf::BranchOp>(Loc, &condB);

  Builder.setInsertionPointToStart(&exitB);

  return nullptr;
}

ValueCategory MLIRScanner::VisitIfStmt(clang::IfStmt *stmt) {
  IfScope scope(*this);
  auto Loc = getMLIRLocation(stmt->getIfLoc());
  auto cond = Visit(stmt->getCond()).getValue(Builder);
  assert(cond != nullptr && "must be a non-null");

  auto oldpoint = Builder.getInsertionPoint();
  auto *oldblock = Builder.getInsertionBlock();
  if (auto LT = cond.getType().dyn_cast<MemRefType>()) {
    cond = Builder.create<polygeist::Memref2PointerOp>(
        Loc, LLVM::LLVMPointerType::get(Builder.getI8Type()), cond);
  }
  if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    auto nullptr_llvm = Builder.create<mlir::LLVM::NullOp>(Loc, LT);
    cond = Builder.create<mlir::LLVM::ICmpOp>(
        Loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
  }
  if (!cond.getType().isa<mlir::IntegerType>()) {
    stmt->dump();
    llvm::errs() << " cond: " << cond << " ct: " << cond.getType() << "\n";
  }
  auto prevTy = cond.getType().cast<mlir::IntegerType>();
  if (!prevTy.isInteger(1)) {
    cond = Builder.create<arith::CmpIOp>(
        Loc, CmpIPredicate::ne, cond,
        Builder.create<ConstantIntOp>(Loc, 0, prevTy));
  }
  bool hasElseRegion = stmt->getElse();
  auto ifOp = Builder.create<mlir::scf::IfOp>(Loc, cond, hasElseRegion);

  ifOp.getThenRegion().back().clear();
  Builder.setInsertionPointToStart(&ifOp.getThenRegion().back());
  Visit(stmt->getThen());
  Builder.create<scf::YieldOp>(Loc);
  if (hasElseRegion) {
    ifOp.getElseRegion().back().clear();
    Builder.setInsertionPointToStart(&ifOp.getElseRegion().back());
    Visit(stmt->getElse());
    Builder.create<scf::YieldOp>(Loc);
  }

  Builder.setInsertionPoint(oldblock, oldpoint);
  return nullptr;
}

ValueCategory MLIRScanner::VisitSwitchStmt(clang::SwitchStmt *stmt) {
  IfScope scope(*this);
  auto cond = Visit(stmt->getCond()).getValue(Builder);
  assert(cond != nullptr);
  SmallVector<int64_t> caseVals;

  auto er = Builder.create<scf::ExecuteRegionOp>(Loc, ArrayRef<mlir::Type>());
  er.getRegion().push_back(new Block());
  auto oldpoint2 = Builder.getInsertionPoint();
  auto *oldblock2 = Builder.getInsertionBlock();

  auto &exitB = *(new Block());
  Builder.setInsertionPointToStart(&exitB);
  Builder.create<scf::YieldOp>(Loc);
  Builder.setInsertionPointToStart(&exitB);

  SmallVector<Block *> blocks;
  bool inCase = false;

  Block *defaultB = &exitB;

  for (auto *cse : stmt->getBody()->children()) {
    if (auto *cses = dyn_cast<CaseStmt>(cse)) {
      auto &condB = *(new Block());

      auto cval = Visit(cses->getLHS());
      if (!cval.val) {
        cses->getLHS()->dump();
      }
      assert(cval.val);
      auto cint = cval.getValue(Builder).getDefiningOp<ConstantIntOp>();
      if (!cint) {
        cses->getLHS()->dump();
        llvm::errs() << "cval: " << cval.val << "\n";
      }
      assert(cint);
      caseVals.push_back(cint.value());

      if (inCase) {
        auto noBreak =
            Builder.create<mlir::memref::LoadOp>(Loc, Loops.back().NoBreak);
        Builder.create<mlir::cf::CondBranchOp>(Loc, noBreak, &condB, &exitB);
        Loops.pop_back();
      }

      inCase = true;
      er.getRegion().getBlocks().push_back(&condB);
      blocks.push_back(&condB);
      Builder.setInsertionPointToStart(&condB);

      auto i1Ty = Builder.getIntegerType(1);
      auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
      auto truev = Builder.create<ConstantIntOp>(Loc, true, 1);
      Loops.push_back({Builder.create<mlir::memref::AllocaOp>(Loc, type),
                       Builder.create<mlir::memref::AllocaOp>(Loc, type)});
      Builder.create<mlir::memref::StoreOp>(Loc, truev, Loops.back().NoBreak);
      Builder.create<mlir::memref::StoreOp>(Loc, truev,
                                            Loops.back().KeepRunning);
      Visit(cses->getSubStmt());
    } else if (auto *cses = dyn_cast<DefaultStmt>(cse)) {
      auto &condB = *(new Block());

      if (inCase) {
        auto noBreak =
            Builder.create<mlir::memref::LoadOp>(Loc, Loops.back().NoBreak);
        Builder.create<mlir::cf::CondBranchOp>(Loc, noBreak, &condB, &exitB);
        Loops.pop_back();
      }

      inCase = true;
      er.getRegion().getBlocks().push_back(&condB);
      Builder.setInsertionPointToStart(&condB);

      auto i1Ty = Builder.getIntegerType(1);
      auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
      auto truev = Builder.create<ConstantIntOp>(Loc, true, 1);
      Loops.push_back({Builder.create<mlir::memref::AllocaOp>(Loc, type),
                       Builder.create<mlir::memref::AllocaOp>(Loc, type)});
      Builder.create<mlir::memref::StoreOp>(Loc, truev, Loops.back().NoBreak);
      Builder.create<mlir::memref::StoreOp>(Loc, truev,
                                            Loops.back().KeepRunning);
      defaultB = &condB;
      Visit(cses->getSubStmt());
    } else {
      Visit(cse);
    }
  }

  if (caseVals.size() == 0) {
    delete &exitB;
    er.erase();
    Builder.setInsertionPoint(oldblock2, oldpoint2);
    return nullptr;
  }

  if (inCase)
    Loops.pop_back();
  Builder.create<mlir::cf::BranchOp>(Loc, &exitB);

  er.getRegion().getBlocks().push_back(&exitB);

  DenseIntElementsAttr caseValuesAttr;
  ShapedType caseValueType = mlir::VectorType::get(
      static_cast<int64_t>(caseVals.size()), cond.getType());
  auto ity = cond.getType().cast<mlir::IntegerType>();
  if (ity.getWidth() == 64)
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseVals);
  else if (ity.getWidth() == 32) {
    SmallVector<int32_t> caseVals32;
    for (auto v : caseVals)
      caseVals32.push_back((int32_t)v);
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseVals32);
  } else if (ity.getWidth() == 16) {
    SmallVector<int16_t> caseVals16;
    for (auto v : caseVals)
      caseVals16.push_back((int16_t)v);
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseVals16);
  } else {
    assert(ity.getWidth() == 8);
    SmallVector<int8_t> caseVals8;
    for (auto v : caseVals)
      caseVals8.push_back((int8_t)v);
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseVals8);
  }

  Builder.setInsertionPointToStart(&er.getRegion().front());
  Builder.create<mlir::cf::SwitchOp>(
      Loc, cond, defaultB, ArrayRef<mlir::Value>(), caseValuesAttr, blocks,
      SmallVector<mlir::ValueRange>(caseVals.size(), ArrayRef<mlir::Value>()));
  Builder.setInsertionPoint(oldblock2, oldpoint2);
  return nullptr;
}

ValueCategory MLIRScanner::VisitDeclStmt(clang::DeclStmt *decl) {
  LLVM_DEBUG({
    llvm::dbgs() << "VisitDeclStmt: ";
    decl->dump();
    llvm::dbgs() << "\n";
  });

  IfScope scope(*this);
  for (auto *sub : decl->decls()) {
    if (auto *vd = dyn_cast<VarDecl>(sub)) {
      VisitVarDecl(vd);
    } else if (isa<TypeAliasDecl, RecordDecl, StaticAssertDecl, TypedefDecl,
                   UsingDecl, UsingDirectiveDecl>(sub)) {
    } else {
      emitError(getMLIRLocation(decl->getBeginLoc()))
          << " + visiting unknonwn sub decl stmt\n";
      sub->dump();
      assert(0 && "unknown sub decl");
    }
  }
  return nullptr;
}

ValueCategory MLIRScanner::VisitAttributedStmt(AttributedStmt *AS) {
  if (!SuppressWarnings)
    emitWarning(getMLIRLocation(AS->getAttrLoc())) << "ignoring attributes\n";

  return Visit(AS->getSubStmt());
}

ValueCategory MLIRScanner::VisitCompoundStmt(clang::CompoundStmt *stmt) {
  for (auto *a : stmt->children()) {
    IfScope scope(*this);
    Visit(a);
  }
  return nullptr;
}

ValueCategory MLIRScanner::VisitBreakStmt(clang::BreakStmt *stmt) {
  IfScope scope(*this);
  assert(Loops.size() && "must be non-empty");
  assert(Loops.back().KeepRunning && "keep running false");
  assert(Loops.back().NoBreak && "no break false");
  auto vfalse =
      Builder.create<ConstantIntOp>(Builder.getUnknownLoc(), false, 1);
  Builder.create<mlir::memref::StoreOp>(Loc, vfalse, Loops.back().KeepRunning);
  Builder.create<mlir::memref::StoreOp>(Loc, vfalse, Loops.back().NoBreak);

  return nullptr;
}

ValueCategory MLIRScanner::VisitContinueStmt(clang::ContinueStmt *stmt) {
  IfScope scope(*this);
  assert(Loops.size() && "must be non-empty");
  assert(Loops.back().KeepRunning && "keep running false");
  auto vfalse =
      Builder.create<ConstantIntOp>(Builder.getUnknownLoc(), false, 1);
  Builder.create<mlir::memref::StoreOp>(Loc, vfalse, Loops.back().KeepRunning);
  return nullptr;
}

ValueCategory MLIRScanner::VisitLabelStmt(clang::LabelStmt *stmt) {
  auto *toadd = Builder.getInsertionBlock()->getParent();
  Block *labelB;
  auto found = Labels.find(stmt);
  if (found != Labels.end()) {
    labelB = found->second;
  } else {
    labelB = new Block();
    Labels[stmt] = labelB;
  }
  toadd->getBlocks().push_back(labelB);
  Builder.create<mlir::cf::BranchOp>(Loc, labelB);
  Builder.setInsertionPointToStart(labelB);
  Visit(stmt->getSubStmt());
  return nullptr;
}

ValueCategory MLIRScanner::VisitGotoStmt(clang::GotoStmt *stmt) {
  auto *labelstmt = stmt->getLabel()->getStmt();
  Block *labelB;
  auto found = Labels.find(labelstmt);
  if (found != Labels.end()) {
    labelB = found->second;
  } else {
    labelB = new Block();
    Labels[labelstmt] = labelB;
  }
  Builder.create<mlir::cf::BranchOp>(Loc, labelB);
  return nullptr;
}

ValueCategory MLIRScanner::VisitCXXTryStmt(clang::CXXTryStmt *stmt) {
  mlirclang::warning() << "not performing catches for try stmt\n";

  return Visit(stmt->getTryBlock());
}

ValueCategory MLIRScanner::VisitReturnStmt(clang::ReturnStmt *stmt) {
  IfScope scope(*this);
  bool isArrayReturn = false;
  Glob.getTypes().getMLIRType(EmittingFunctionDecl->getReturnType(),
                              &isArrayReturn);

  if (isArrayReturn) {
    auto rv = Visit(stmt->getRetValue());
    assert(rv.val && "expect right value to be valid");
    assert(rv.isReference && "right value must be a reference");
    auto op = Function.getArgument(Function.getNumArguments() - 1);
    assert(rv.val.getType().cast<MemRefType>().getElementType() ==
               op.getType().cast<MemRefType>().getElementType() &&
           "type mismatch");
    assert(op.getType().cast<MemRefType>().getShape().size() == 2 &&
           "expect 2d memref");
    assert(rv.val.getType().cast<MemRefType>().getShape().size() == 2 &&
           "expect 2d memref");
    assert(rv.val.getType().cast<MemRefType>().getShape()[1] ==
           op.getType().cast<MemRefType>().getShape()[1]);

    for (int i = 0; i < op.getType().cast<MemRefType>().getShape()[1]; i++) {
      std::vector<mlir::Value> idx = {getConstantIndex(0), getConstantIndex(i)};
      assert(rv.val.getType().cast<MemRefType>().getShape().size() == 2);
      Builder.create<mlir::memref::StoreOp>(
          Loc, Builder.create<mlir::memref::LoadOp>(Loc, rv.val, idx), op, idx);
    }
  } else if (stmt->getRetValue()) {
    auto rv = Visit(stmt->getRetValue());
    if (!stmt->getRetValue()->getType()->isVoidType()) {
      if (!rv.val) {
        stmt->dump();
      }
      assert(rv.val && "expect right value to be valid");

      mlir::Value val;
      if (stmt->getRetValue()->isLValue() || stmt->getRetValue()->isXValue()) {
        assert(rv.isReference);
        val = rv.val;
      } else {
        val = rv.getValue(Builder);
      }

      auto postTy = ReturnVal.getType().cast<MemRefType>().getElementType();
      if (auto prevTy = val.getType().dyn_cast<mlir::IntegerType>()) {
        const auto SrcTy = stmt->getRetValue()->getType();
        const auto IsSigned =
            SrcTy->isBooleanType() ? false : SrcTy->isSignedIntegerType();
        val = rv.IntCast(Builder, getMLIRLocation(stmt->getReturnLoc()), postTy,
                         IsSigned)
                  .val;
      } else if (val.getType().isa<MemRefType>() &&
                 postTy.isa<LLVM::LLVMPointerType>())
        val = Builder.create<polygeist::Memref2PointerOp>(Loc, postTy, val);
      else if (val.getType().isa<LLVM::LLVMPointerType>() &&
               postTy.isa<MemRefType>())
        val = Builder.create<polygeist::Pointer2MemrefOp>(Loc, postTy, val);
      if (postTy != val.getType()) {
        stmt->dump();
        llvm::errs() << " val: " << val << " postTy: " << postTy
                     << " rv.val: " << rv.val << " rv.isRef"
                     << (int)rv.isReference << " mm: "
                     << (int)(stmt->getRetValue()->isLValue() ||
                              stmt->getRetValue()->isXValue())
                     << "\n";
      }
      assert(postTy == val.getType());
      Builder.create<mlir::memref::StoreOp>(Loc, val, ReturnVal);
    }
  }

  assert(Loops.size() && "must be non-empty");
  auto vfalse =
      Builder.create<ConstantIntOp>(Builder.getUnknownLoc(), false, 1);
  for (auto l : Loops) {
    Builder.create<mlir::memref::StoreOp>(Loc, vfalse, l.KeepRunning);
    Builder.create<mlir::memref::StoreOp>(Loc, vfalse, l.NoBreak);
  }

  return nullptr;
}
