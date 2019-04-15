//== MIGChecker.cpp - MIG calling convention checker ------------*- C++ -*--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MIGChecker, a Mach Interface Generator calling convention
// checker. Namely, in MIG callback implementation the following rules apply:
// - When a server routine returns an error code that represents success, it
//   must take ownership of resources passed to it (and eventually release
//   them).
// - Additionally, when returning success, all out-parameters must be
//   initialized.
// - When it returns any other error code, it must not take ownership,
//   because the message and its out-of-line parameters will be destroyed
//   by the client that called the function.
// For now we only check the last rule, as its violations lead to dangerous
// use-after-free exploits.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/AnyCall.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
class MIGChecker : public Checker<check::PostCall, check::PreStmt<ReturnStmt>,
                                  check::EndFunction> {
  BugType BT{this, "Use-after-free (MIG calling convention violation)",
             categories::MemoryError};

  // The checker knows that an out-of-line object is deallocated if it is
  // passed as an argument to one of these functions. If this object is
  // additionally an argument of a MIG routine, the checker keeps track of that
  // information and issues a warning when an error is returned from the
  // respective routine.
  std::vector<std::pair<CallDescription, unsigned>> Deallocators = {
#define CALL(required_args, deallocated_arg, ...)                              \
  {{{__VA_ARGS__}, required_args}, deallocated_arg}
      // E.g., if the checker sees a C function 'vm_deallocate' that is
      // defined on class 'IOUserClient' that has exactly 3 parameters, it knows
      // that argument #1 (starting from 0, i.e. the second argument) is going
      // to be consumed in the sense of the MIG consume-on-success convention.
      CALL(3, 1, "vm_deallocate"),
      CALL(3, 1, "mach_vm_deallocate"),
      CALL(2, 0, "mig_deallocate"),
      CALL(2, 1, "mach_port_deallocate"),
      // E.g., if the checker sees a method 'releaseAsyncReference64()' that is
      // defined on class 'IOUserClient' that takes exactly 1 argument, it knows
      // that the argument is going to be consumed in the sense of the MIG
      // consume-on-success convention.
      CALL(1, 0, "IOUserClient", "releaseAsyncReference64"),
#undef CALL
  };

  void checkReturnAux(const ReturnStmt *RS, CheckerContext &C) const;

public:
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;

  // HACK: We're making two attempts to find the bug: checkEndFunction
  // should normally be enough but it fails when the return value is a literal
  // that never gets put into the Environment and ends of function with multiple
  // returns get agglutinated across returns, preventing us from obtaining
  // the return value. The problem is similar to https://reviews.llvm.org/D25326
  // but now we step into it in the top-level function.
  void checkPreStmt(const ReturnStmt *RS, CheckerContext &C) const {
    checkReturnAux(RS, C);
  }
  void checkEndFunction(const ReturnStmt *RS, CheckerContext &C) const {
    checkReturnAux(RS, C);
  }

  class Visitor : public BugReporterVisitor {
  public:
    void Profile(llvm::FoldingSetNodeID &ID) const {
      static int X = 0;
      ID.AddPointer(&X);
    }

    std::shared_ptr<PathDiagnosticPiece> VisitNode(const ExplodedNode *N,
        BugReporterContext &BRC, BugReport &R);
  };
};
} // end anonymous namespace

// FIXME: It's a 'const ParmVarDecl *' but there's no ready-made GDM traits
// specialization for this sort of types.
REGISTER_TRAIT_WITH_PROGRAMSTATE(ReleasedParameter, const void *)

std::shared_ptr<PathDiagnosticPiece>
MIGChecker::Visitor::VisitNode(const ExplodedNode *N, BugReporterContext &BRC,
                               BugReport &R) {
  const auto *NewPVD = static_cast<const ParmVarDecl *>(
      N->getState()->get<ReleasedParameter>());
  const auto *OldPVD = static_cast<const ParmVarDecl *>(
      N->getFirstPred()->getState()->get<ReleasedParameter>());
  if (OldPVD == NewPVD)
    return nullptr;

  assert(NewPVD && "What is deallocated cannot be un-deallocated!");
  SmallString<64> Str;
  llvm::raw_svector_ostream OS(Str);
  OS << "Value passed through parameter '" << NewPVD->getName()
     << "' is deallocated";

  PathDiagnosticLocation Loc =
      PathDiagnosticLocation::create(N->getLocation(), BRC.getSourceManager());
  return std::make_shared<PathDiagnosticEventPiece>(Loc, OS.str());
}

static const ParmVarDecl *getOriginParam(SVal V, CheckerContext &C) {
  SymbolRef Sym = V.getAsSymbol();
  if (!Sym)
    return nullptr;

  // If we optimistically assume that the MIG routine never re-uses the storage
  // that was passed to it as arguments when it invalidates it (but at most when
  // it assigns to parameter variables directly), this procedure correctly
  // determines if the value was loaded from the transitive closure of MIG
  // routine arguments in the heap.
  while (const MemRegion *MR = Sym->getOriginRegion()) {
    const auto *VR = dyn_cast<VarRegion>(MR);
    if (VR && VR->hasStackParametersStorage() &&
           VR->getStackFrame()->inTopFrame())
      return cast<ParmVarDecl>(VR->getDecl());

    const SymbolicRegion *SR = MR->getSymbolicBase();
    if (!SR)
      return nullptr;

    Sym = SR->getSymbol();
  }

  return nullptr;
}

static bool isInMIGCall(CheckerContext &C) {
  const LocationContext *LC = C.getLocationContext();
  const StackFrameContext *SFC;
  // Find the top frame.
  while (LC) {
    SFC = LC->getStackFrame();
    LC = SFC->getParent();
  }

  const Decl *D = SFC->getDecl();

  if (Optional<AnyCall> AC = AnyCall::forDecl(D)) {
    // Even though there's a Sema warning when the return type of an annotated
    // function is not a kern_return_t, this warning isn't an error, so we need
    // an extra sanity check here.
    // FIXME: AnyCall doesn't support blocks yet, so they remain unchecked
    // for now.
    if (!AC->getReturnType(C.getASTContext())
             .getCanonicalType()->isSignedIntegerType())
      return false;
  }

  if (D->hasAttr<MIGServerRoutineAttr>())
    return true;

  // See if there's an annotated method in the superclass.
  if (const auto *MD = dyn_cast<CXXMethodDecl>(D))
    for (const auto *OMD: MD->overridden_methods())
      if (OMD->hasAttr<MIGServerRoutineAttr>())
        return true;

  return false;
}

void MIGChecker::checkPostCall(const CallEvent &Call, CheckerContext &C) const {
  if (!isInMIGCall(C))
    return;

  auto I = std::find_if(Deallocators.begin(), Deallocators.end(),
                        [&](const std::pair<CallDescription, unsigned> &Item) {
                          return Call.isCalled(Item.first);
                        });
  if (I == Deallocators.end())
    return;

  unsigned ArgIdx = I->second;
  SVal Arg = Call.getArgSVal(ArgIdx);
  const ParmVarDecl *PVD = getOriginParam(Arg, C);
  if (!PVD)
    return;

  C.addTransition(C.getState()->set<ReleasedParameter>(PVD));
}

// Returns true if V can potentially represent a "successful" kern_return_t.
static bool mayBeSuccess(SVal V, CheckerContext &C) {
  ProgramStateRef State = C.getState();

  // Can V represent KERN_SUCCESS?
  if (!State->isNull(V).isConstrainedFalse())
    return true;

  SValBuilder &SVB = C.getSValBuilder();
  ASTContext &ACtx = C.getASTContext();

  // Can V represent MIG_NO_REPLY?
  static const int MigNoReply = -305;
  V = SVB.evalEQ(C.getState(), V, SVB.makeIntVal(MigNoReply, ACtx.IntTy));
  if (!State->isNull(V).isConstrainedTrue())
    return true;

  // If none of the above, it's definitely an error.
  return false;
}

void MIGChecker::checkReturnAux(const ReturnStmt *RS, CheckerContext &C) const {
  // It is very unlikely that a MIG callback will be called from anywhere
  // within the project under analysis and the caller isn't itself a routine
  // that follows the MIG calling convention. Therefore we're safe to believe
  // that it's always the top frame that is of interest. There's a slight chance
  // that the user would want to enforce the MIG calling convention upon
  // a random routine in the middle of nowhere, but given that the convention is
  // fairly weird and hard to follow in the first place, there's relatively
  // little motivation to spread it this way.
  if (!C.inTopFrame())
    return;

  if (!isInMIGCall(C))
    return;

  // We know that the function is non-void, but what if the return statement
  // is not there in the code? It's not a compile error, we should not crash.
  if (!RS)
    return;

  ProgramStateRef State = C.getState();
  if (!State->get<ReleasedParameter>())
    return;

  SVal V = C.getSVal(RS);
  if (mayBeSuccess(V, C))
    return;

  ExplodedNode *N = C.generateErrorNode();
  if (!N)
    return;

  auto R = llvm::make_unique<BugReport>(
      BT,
      "MIG callback fails with error after deallocating argument value. "
      "This is a use-after-free vulnerability because the caller will try to "
      "deallocate it again",
      N);

  R->addRange(RS->getSourceRange());
  bugreporter::trackExpressionValue(N, RS->getRetValue(), *R, false);
  R->addVisitor(llvm::make_unique<Visitor>());
  C.emitReport(std::move(R));
}

void ento::registerMIGChecker(CheckerManager &Mgr) {
  Mgr.registerChecker<MIGChecker>();
}

bool ento::shouldRegisterMIGChecker(const LangOptions &LO) {
  return true;
}
