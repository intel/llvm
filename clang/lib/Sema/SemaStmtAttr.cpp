//===--- SemaStmtAttr.cpp - Statement Attribute Handling ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements stmt-related attribute processing.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/DelayedDiagnostic.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/StringExtras.h"
#include <optional>

using namespace clang;
using namespace sema;

static Attr *handleFallThroughAttr(Sema &S, Stmt *St, const ParsedAttr &A,
                                   SourceRange Range) {
  FallThroughAttr Attr(S.Context, A);
  if (isa<SwitchCase>(St)) {
    S.Diag(A.getRange().getBegin(), diag::err_fallthrough_attr_wrong_target)
        << A << St->getBeginLoc();
    SourceLocation L = S.getLocForEndOfToken(Range.getEnd());
    S.Diag(L, diag::note_fallthrough_insert_semi_fixit)
        << FixItHint::CreateInsertion(L, ";");
    return nullptr;
  }
  auto *FnScope = S.getCurFunction();
  if (FnScope->SwitchStack.empty()) {
    S.Diag(A.getRange().getBegin(), diag::err_fallthrough_attr_outside_switch);
    return nullptr;
  }

  // If this is spelled as the standard C++17 attribute, but not in C++17, warn
  // about using it as an extension.
  if (!S.getLangOpts().CPlusPlus17 && A.isCXX11Attribute() &&
      !A.getScopeName())
    S.Diag(A.getLoc(), diag::ext_cxx17_attr) << A;

  FnScope->setHasFallthroughStmt();
  return ::new (S.Context) FallThroughAttr(S.Context, A);
}

static Attr *handleSuppressAttr(Sema &S, Stmt *St, const ParsedAttr &A,
                                SourceRange Range) {
  if (A.getAttributeSpellingListIndex() == SuppressAttr::CXX11_gsl_suppress &&
      A.getNumArgs() < 1) {
    // Suppression attribute with GSL spelling requires at least 1 argument.
    S.Diag(A.getLoc(), diag::err_attribute_too_few_arguments) << A << 1;
    return nullptr;
  }

  std::vector<StringRef> DiagnosticIdentifiers;
  for (unsigned I = 0, E = A.getNumArgs(); I != E; ++I) {
    StringRef RuleName;

    if (!S.checkStringLiteralArgumentAttr(A, I, RuleName, nullptr))
      return nullptr;

    DiagnosticIdentifiers.push_back(RuleName);
  }

  return ::new (S.Context) SuppressAttr(
      S.Context, A, DiagnosticIdentifiers.data(), DiagnosticIdentifiers.size());
}

SYCLIntelMaxConcurrencyAttr *
Sema::BuildSYCLIntelMaxConcurrencyAttr(const AttributeCommonInfo &CI,
                                       Expr *E) {
  if (!E->isValueDependent()) {
    llvm::APSInt ArgVal;
    ExprResult Res = VerifyIntegerConstantExpression(E, &ArgVal);
    if (Res.isInvalid())
      return nullptr;
    E = Res.get();

    // This attribute requires a non-negative value.
    if (ArgVal < 0) {
      Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
          << CI << /*non-negative*/ 1;
      return nullptr;
    }
  }

  return new (Context) SYCLIntelMaxConcurrencyAttr(Context, CI, E);
}

static Attr *handleSYCLIntelMaxConcurrencyAttr(Sema &S, Stmt *St,
                                               const ParsedAttr &A) {
  S.CheckDeprecatedSYCLAttributeSpelling(A);

  Expr *E = A.getArgAsExpr(0);
  return S.BuildSYCLIntelMaxConcurrencyAttr(A, E);
}

SYCLIntelInitiationIntervalAttr *
Sema::BuildSYCLIntelInitiationIntervalAttr(const AttributeCommonInfo &CI,
                                           Expr *E) {
  if (!E->isValueDependent()) {
    llvm::APSInt ArgVal;
    ExprResult Res = VerifyIntegerConstantExpression(E, &ArgVal);
    if (Res.isInvalid())
      return nullptr;
    E = Res.get();

    // This attribute requires a strictly positive value.
    if (ArgVal <= 0) {
      Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
          << CI << /*positive*/ 0;
      return nullptr;
    }
  }

  return new (Context) SYCLIntelInitiationIntervalAttr(Context, CI, E);
}

static Attr *handleSYCLIntelInitiationIntervalAttr(Sema &S, Stmt *St,
                                                   const ParsedAttr &A) {
  S.CheckDeprecatedSYCLAttributeSpelling(A);

  Expr *E = A.getArgAsExpr(0);
  return S.BuildSYCLIntelInitiationIntervalAttr(A, E);
}

SYCLIntelMaxInterleavingAttr *
Sema::BuildSYCLIntelMaxInterleavingAttr(const AttributeCommonInfo &CI,
                       		        Expr *E) {
  if (!E->isValueDependent()) {
    llvm::APSInt ArgVal;
    ExprResult Res = VerifyIntegerConstantExpression(E, &ArgVal);
    if (Res.isInvalid())
      return nullptr;
    E = Res.get();

    // This attribute accepts values 0 and 1 only.
    if (ArgVal < 0 || ArgVal > 1) {
      Diag(E->getBeginLoc(), diag::err_attribute_argument_is_not_valid) << CI;
      return nullptr;
    }
  }

  return new (Context) SYCLIntelMaxInterleavingAttr(Context, CI, E);
}

static Attr *handleSYCLIntelMaxInterleavingAttr(Sema &S, Stmt *St,
                                                const ParsedAttr &A) {
  S.CheckDeprecatedSYCLAttributeSpelling(A);

  Expr *E = A.getArgAsExpr(0);
  return S.BuildSYCLIntelMaxInterleavingAttr(A, E);
}

SYCLIntelLoopCoalesceAttr *
Sema::BuildSYCLIntelLoopCoalesceAttr(const AttributeCommonInfo &CI,
                                     Expr *E) {
  if (E && !E->isValueDependent()) {
    llvm::APSInt ArgVal;
    ExprResult Res = VerifyIntegerConstantExpression(E, &ArgVal);
    if (Res.isInvalid())
      return nullptr;
    E = Res.get();

    // This attribute requires a strictly positive value.
    if (ArgVal <= 0) {
      Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
          << CI << /*positive*/ 0;
      return nullptr;
    }
  }

  return new (Context) SYCLIntelLoopCoalesceAttr(Context, CI, E);
}

static Attr *handleSYCLIntelLoopCoalesceAttr(Sema &S, Stmt *St,
                                             const ParsedAttr &A) {
  S.CheckDeprecatedSYCLAttributeSpelling(A);

  Expr *E = A.isArgExpr(0) ? A.getArgAsExpr(0) : nullptr;
  return S.BuildSYCLIntelLoopCoalesceAttr(A, E);
}

SYCLIntelSpeculatedIterationsAttr *
Sema::BuildSYCLIntelSpeculatedIterationsAttr(const AttributeCommonInfo &CI,
                                             Expr *E) {
  if (!E->isValueDependent()) {
    llvm::APSInt ArgVal;
    ExprResult Res = VerifyIntegerConstantExpression(E, &ArgVal);
    if (Res.isInvalid())
      return nullptr;
    E = Res.get();

    // This attribute requires a non-negative value.
    if (ArgVal < 0) {
      Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
          << CI << /*non-negative*/ 1;
      return nullptr;
    }
  }

  return new (Context) SYCLIntelSpeculatedIterationsAttr(Context, CI, E);
}

static Attr *handleSYCLIntelSpeculatedIterationsAttr(Sema &S, Stmt *St,
                                                     const ParsedAttr &A) {
  S.CheckDeprecatedSYCLAttributeSpelling(A);

  Expr *E = A.getArgAsExpr(0);
  return S.BuildSYCLIntelSpeculatedIterationsAttr(A, E);
}

static Attr *handleSYCLIntelDisableLoopPipeliningAttr(Sema &S, Stmt *,
                                                      const ParsedAttr &A) {
  S.CheckDeprecatedSYCLAttributeSpelling(A);
  return new (S.Context) SYCLIntelDisableLoopPipeliningAttr(S.Context, A);
}

static bool checkSYCLIntelIVDepSafeLen(Sema &S, llvm::APSInt &Value,
                                           Expr *E) {
  // This attribute requires a non-negative value.
  if (!Value.isNonNegative())
    return S.Diag(E->getExprLoc(),
                  diag::err_attribute_requires_positive_integer)
           << "'ivdep'" << /*non-negative*/ 1;
  return false;
}

enum class IVDepExprResult {
  Invalid,
  Null,
  Dependent,
  Array,
  SafeLen,
};

static IVDepExprResult HandleIVDepAttrExpr(Sema &S, Expr *E,
                                           unsigned &SafelenValue) {
  if (!E)
    return IVDepExprResult::Null;

  if (E->isInstantiationDependent())
    return IVDepExprResult::Dependent;

  std::optional<llvm::APSInt> ArgVal = E->getIntegerConstantExpr(S.getASTContext());
  if (ArgVal) {
    if (checkSYCLIntelIVDepSafeLen(S, *ArgVal, E))
      return IVDepExprResult::Invalid;
    SafelenValue = ArgVal->getZExtValue();
    // ivdep attribute allows both safelen = 0 and safelen = 1 with a warning.
    if (SafelenValue == 0 || SafelenValue == 1) {
      S.Diag(E->getExprLoc(), diag::warn_ivdep_attribute_argument)
          << SafelenValue;
      return IVDepExprResult::Invalid;
    }
    return IVDepExprResult::SafeLen;
  }

  if (isa<DeclRefExpr>(E) || isa<MemberExpr>(E)) {
    if (!E->getType()->isArrayType() && !E->getType()->isPointerType()) {
      S.Diag(E->getExprLoc(), diag::err_ivdep_declrefexpr_arg);
      return IVDepExprResult::Invalid;
    }
    return IVDepExprResult::Array;
  }

  S.Diag(E->getExprLoc(), diag::err_ivdep_unknown_arg);
  return IVDepExprResult::Invalid;
}

// Note: At the time of this call, we don't know the order of the expressions,
// so we name them vaguely until we can figure it out.
SYCLIntelIVDepAttr *
Sema::BuildSYCLIntelIVDepAttr(const AttributeCommonInfo &CI, Expr *Expr1,
                                  Expr *Expr2) {
  unsigned SafelenValue = 0;
  IVDepExprResult E1 = HandleIVDepAttrExpr(*this, Expr1, SafelenValue);
  IVDepExprResult E2 = HandleIVDepAttrExpr(*this, Expr2, SafelenValue);

  if (E1 == IVDepExprResult::Invalid || E2 == IVDepExprResult::Invalid)
    return nullptr;

  if (E1 == E2 && E1 != IVDepExprResult::Dependent &&
      E1 != IVDepExprResult::Null) {
    Diag(Expr2->getExprLoc(), diag::err_ivdep_duplicate_arg);
    return nullptr;
  }

  // Try to put Safelen in the 1st one so codegen can count on the ordering.
  Expr *SafeLenExpr = Expr1;
  Expr *ArrayExpr = Expr2;

  // Both can be null or dependent, so swap if we're really sure.
  if (E2 == IVDepExprResult::SafeLen || E1 == IVDepExprResult::Array) {
    SafeLenExpr = Expr2;
    ArrayExpr = Expr1;
  }

  return new (Context)
      SYCLIntelIVDepAttr(Context, CI, SafeLenExpr, ArrayExpr, SafelenValue);
}

// Filters out any attributes from the list that are either not the specified
// type, or whose function isDependent returns true.
template <typename T>
static void FilterAttributeList(ArrayRef<const Attr *> Attrs,
                    SmallVectorImpl<const T *> &FilteredAttrs) {

  llvm::transform(Attrs, std::back_inserter(FilteredAttrs),
                  [](const Attr *A) -> const T * {
                    if (const auto *Cast = dyn_cast<T>(A))
                      return Cast->isDependent() ? nullptr : Cast;
                    return nullptr;
                  });
  FilteredAttrs.erase(
      std::remove(FilteredAttrs.begin(), FilteredAttrs.end(), nullptr),
      FilteredAttrs.end());
}

static void
CheckRedundantSYCLIntelIVDepAttrs(Sema &S, ArrayRef<const Attr *> Attrs) {
  // Skip SEMA if we're in a template, this will be diagnosed later.
  if (S.getCurLexicalContext()->isDependentContext())
    return;

  SmallVector<const SYCLIntelIVDepAttr *, 8> FilteredAttrs;
  // Filter down to just non-dependent ivdeps.
  FilterAttributeList(Attrs, FilteredAttrs);
  if (FilteredAttrs.empty())
    return;

  SmallVector<const SYCLIntelIVDepAttr *, 8> SortedAttrs(FilteredAttrs);
  llvm::stable_sort(SortedAttrs, SYCLIntelIVDepAttr::SafelenCompare);

  // Find the maximum without an array expression, which ends up in the 2nd
  // expr.
  const auto *GlobalMaxItr =
      llvm::find_if(SortedAttrs, [](const SYCLIntelIVDepAttr *A) {
        return !A->getArrayExpr();
      });
  const SYCLIntelIVDepAttr *GlobalMax =
      GlobalMaxItr == SortedAttrs.end() ? nullptr : *GlobalMaxItr;

  for (const auto *A : FilteredAttrs) {
    if (A == GlobalMax)
      continue;

    if (GlobalMax && !SYCLIntelIVDepAttr::SafelenCompare(A, GlobalMax)) {
      S.Diag(A->getLocation(), diag::warn_ivdep_redundant)
          << !GlobalMax->isInf() << GlobalMax->getSafelenValue() << !A->isInf()
          << A->getSafelenValue();
      S.Diag(GlobalMax->getLocation(), diag::note_previous_attribute);
      continue;
    }

    if (!A->getArrayExpr())
      continue;

    const ValueDecl *ArrayDecl = A->getArrayDecl();
    auto Other = llvm::find_if(SortedAttrs,
                               [ArrayDecl](const SYCLIntelIVDepAttr *A) {
                                 return ArrayDecl == A->getArrayDecl();
                               });
    assert(Other != SortedAttrs.end() && "Should find at least itself");

    // Diagnose if lower/equal to the lowest with this array.
    if (*Other != A && !SYCLIntelIVDepAttr::SafelenCompare(A, *Other)) {
      S.Diag(A->getLocation(), diag::warn_ivdep_redundant)
          << !(*Other)->isInf() << (*Other)->getSafelenValue() << !A->isInf()
          << A->getSafelenValue();
      S.Diag((*Other)->getLocation(), diag::note_previous_attribute);
    }
  }
}

static Attr *handleIntelIVDepAttr(Sema &S, Stmt *St, const ParsedAttr &A) {
  unsigned NumArgs = A.getNumArgs();

  S.CheckDeprecatedSYCLAttributeSpelling(A);

  return S.BuildSYCLIntelIVDepAttr(
      A, NumArgs >= 1 ? A.getArgAsExpr(0) : nullptr,
      NumArgs == 2 ? A.getArgAsExpr(1) : nullptr);
}

static void
CheckForDuplicateSYCLIntelLoopCountAttrs(Sema &S,
                                         ArrayRef<const Attr *> Attrs) {
  // Create a list of SYCLIntelLoopCount attributes only.
  SmallVector<const SYCLIntelLoopCountAttr *, 8> OnlyLoopCountAttrs;
  llvm::transform(
      Attrs, std::back_inserter(OnlyLoopCountAttrs), [](const Attr *A) {
        return dyn_cast_or_null<const SYCLIntelLoopCountAttr>(A);
      });
  OnlyLoopCountAttrs.erase(
      std::remove(OnlyLoopCountAttrs.begin(), OnlyLoopCountAttrs.end(),
                  static_cast<const SYCLIntelLoopCountAttr *>(nullptr)),
      OnlyLoopCountAttrs.end());
  if (OnlyLoopCountAttrs.empty())
    return;

  unsigned int MinCount = 0;
  unsigned int MaxCount = 0;
  unsigned int AvgCount = 0;
  unsigned int Count = 0;
  for (const auto *A : OnlyLoopCountAttrs) {
    const auto *At = dyn_cast<SYCLIntelLoopCountAttr>(A);
    At->isMin()   ? MinCount++
    : At->isMax() ? MaxCount++
    : At->isAvg() ? AvgCount++
                  : Count++;
    if (MinCount > 1 || MaxCount > 1 || AvgCount > 1 || Count > 1)
      S.Diag(A->getLocation(), diag::err_sycl_loop_attr_duplication) << 1 << A;
  }
}

SYCLIntelLoopCountAttr *
Sema::BuildSYCLIntelLoopCountAttr(const AttributeCommonInfo &CI, Expr *E) {
  if (!E->isValueDependent()) {
    llvm::APSInt ArgVal;
    ExprResult Res = VerifyIntegerConstantExpression(E, &ArgVal);
    if (Res.isInvalid())
      return nullptr;
    E = Res.get();

    // This attribute requires a non-negative value.
    if (ArgVal < 0) {
      Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
          << CI << /*non-negative*/ 1;
      return nullptr;
    }
  }

  return new (Context) SYCLIntelLoopCountAttr(Context, CI, E);
}

static Attr *handleSYCLIntelLoopCountAttr(Sema &S, Stmt *St,
                                          const ParsedAttr &A) {
  S.CheckDeprecatedSYCLAttributeSpelling(A);

  Expr *E = A.getArgAsExpr(0);
  return S.BuildSYCLIntelLoopCountAttr(A, E);
}

static Attr *handleIntelNofusionAttr(Sema &S, Stmt *St,
                                     const ParsedAttr &A) {
  return new (S.Context) SYCLIntelNofusionAttr(S.Context, A);
}

SYCLIntelMaxReinvocationDelayAttr *
Sema::BuildSYCLIntelMaxReinvocationDelayAttr(const AttributeCommonInfo &CI,
                                             Expr *E) {
  if (!E->isValueDependent()) {
    llvm::APSInt ArgVal;
    ExprResult Res = VerifyIntegerConstantExpression(E, &ArgVal);
    if (Res.isInvalid())
      return nullptr;
    E = Res.get();

    // This attribute requires a strictly positive value.
    if (ArgVal <= 0) {
      Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
          << CI << /*positive*/ 0;
      return nullptr;
    }
  }

  return new (Context) SYCLIntelMaxReinvocationDelayAttr(Context, CI, E);
}

static Attr * handleSYCLIntelMaxReinvocationDelayAttr(Sema &S, Stmt *St,
                                                      const ParsedAttr &A) {
  S.CheckDeprecatedSYCLAttributeSpelling(A);

  Expr *E = A.getArgAsExpr(0);
  return S.BuildSYCLIntelMaxReinvocationDelayAttr(A, E);
}

static Attr *handleSYCLIntelEnableLoopPipeliningAttr(Sema &S, Stmt *,
                                                     const ParsedAttr &A) {
  return new (S.Context) SYCLIntelEnableLoopPipeliningAttr(S.Context, A);
}

static Attr *handleLoopHintAttr(Sema &S, Stmt *St, const ParsedAttr &A,
                                SourceRange) {
  IdentifierLoc *PragmaNameLoc = A.getArgAsIdent(0);
  IdentifierLoc *OptionLoc = A.getArgAsIdent(1);
  IdentifierLoc *StateLoc = A.getArgAsIdent(2);
  Expr *ValueExpr = A.getArgAsExpr(3);

  StringRef PragmaName =
      llvm::StringSwitch<StringRef>(PragmaNameLoc->Ident->getName())
          .Cases("unroll", "nounroll", "unroll_and_jam", "nounroll_and_jam",
                 PragmaNameLoc->Ident->getName())
          .Default("clang loop");

  // This could be handled automatically by adding a Subjects definition in
  // Attr.td, but that would make the diagnostic behavior worse in this case
  // because the user spells this attribute as a pragma.
  if (!isa<DoStmt, ForStmt, CXXForRangeStmt, WhileStmt>(St)) {
    std::string Pragma = "#pragma " + std::string(PragmaName);
    S.Diag(St->getBeginLoc(), diag::err_pragma_loop_precedes_nonloop) << Pragma;
    return nullptr;
  }

  LoopHintAttr::OptionType Option;
  LoopHintAttr::LoopHintState State;

  auto SetHints = [&Option, &State](LoopHintAttr::OptionType O,
                                    LoopHintAttr::LoopHintState S) {
    Option = O;
    State = S;
  };

  if (PragmaName == "nounroll") {
    SetHints(LoopHintAttr::Unroll, LoopHintAttr::Disable);
  } else if (PragmaName == "unroll") {
    // #pragma unroll N
    if (ValueExpr) {
      if (!ValueExpr->isValueDependent()) {
        auto Value = ValueExpr->EvaluateKnownConstInt(S.getASTContext());
        if (Value.isZero() || Value.isOne())
          SetHints(LoopHintAttr::Unroll, LoopHintAttr::Disable);
        else
          SetHints(LoopHintAttr::UnrollCount, LoopHintAttr::Numeric);
      } else
        SetHints(LoopHintAttr::UnrollCount, LoopHintAttr::Numeric);
    } else
      SetHints(LoopHintAttr::Unroll, LoopHintAttr::Enable);
  } else if (PragmaName == "nounroll_and_jam") {
    SetHints(LoopHintAttr::UnrollAndJam, LoopHintAttr::Disable);
  } else if (PragmaName == "unroll_and_jam") {
    // #pragma unroll_and_jam N
    if (ValueExpr)
      SetHints(LoopHintAttr::UnrollAndJamCount, LoopHintAttr::Numeric);
    else
      SetHints(LoopHintAttr::UnrollAndJam, LoopHintAttr::Enable);
  } else {
    // #pragma clang loop ...
    assert(OptionLoc && OptionLoc->Ident &&
           "Attribute must have valid option info.");
    Option = llvm::StringSwitch<LoopHintAttr::OptionType>(
                 OptionLoc->Ident->getName())
                 .Case("vectorize", LoopHintAttr::Vectorize)
                 .Case("vectorize_width", LoopHintAttr::VectorizeWidth)
                 .Case("interleave", LoopHintAttr::Interleave)
                 .Case("vectorize_predicate", LoopHintAttr::VectorizePredicate)
                 .Case("interleave_count", LoopHintAttr::InterleaveCount)
                 .Case("unroll", LoopHintAttr::Unroll)
                 .Case("unroll_count", LoopHintAttr::UnrollCount)
                 .Case("pipeline", LoopHintAttr::PipelineDisabled)
                 .Case("pipeline_initiation_interval",
                       LoopHintAttr::PipelineInitiationInterval)
                 .Case("distribute", LoopHintAttr::Distribute)
                 .Default(LoopHintAttr::Vectorize);
    if (Option == LoopHintAttr::VectorizeWidth) {
      assert((ValueExpr || (StateLoc && StateLoc->Ident)) &&
             "Attribute must have a valid value expression or argument.");
      if (ValueExpr && S.CheckLoopHintExpr(ValueExpr, St->getBeginLoc(),
                                           /*AllowZero=*/false))
        return nullptr;
      if (StateLoc && StateLoc->Ident && StateLoc->Ident->isStr("scalable"))
        State = LoopHintAttr::ScalableWidth;
      else
        State = LoopHintAttr::FixedWidth;
    } else if (Option == LoopHintAttr::InterleaveCount ||
               Option == LoopHintAttr::UnrollCount ||
               Option == LoopHintAttr::PipelineInitiationInterval) {
      assert(ValueExpr && "Attribute must have a valid value expression.");
      if (S.CheckLoopHintExpr(ValueExpr, St->getBeginLoc(),
                              /*AllowZero=*/false))
        return nullptr;
      State = LoopHintAttr::Numeric;
    } else if (Option == LoopHintAttr::Vectorize ||
               Option == LoopHintAttr::Interleave ||
               Option == LoopHintAttr::VectorizePredicate ||
               Option == LoopHintAttr::Unroll ||
               Option == LoopHintAttr::Distribute ||
               Option == LoopHintAttr::PipelineDisabled) {
      assert(StateLoc && StateLoc->Ident && "Loop hint must have an argument");
      if (StateLoc->Ident->isStr("disable"))
        State = LoopHintAttr::Disable;
      else if (StateLoc->Ident->isStr("assume_safety"))
        State = LoopHintAttr::AssumeSafety;
      else if (StateLoc->Ident->isStr("full"))
        State = LoopHintAttr::Full;
      else if (StateLoc->Ident->isStr("enable"))
        State = LoopHintAttr::Enable;
      else
        llvm_unreachable("bad loop hint argument");
    } else
      llvm_unreachable("bad loop hint");
  }

  return LoopHintAttr::CreateImplicit(S.Context, Option, State, ValueExpr, A);
}

namespace {
class CallExprFinder : public ConstEvaluatedExprVisitor<CallExprFinder> {
  bool FoundAsmStmt = false;
  std::vector<const CallExpr *> CallExprs;

public:
  typedef ConstEvaluatedExprVisitor<CallExprFinder> Inherited;

  CallExprFinder(Sema &S, const Stmt *St) : Inherited(S.Context) { Visit(St); }

  bool foundCallExpr() { return !CallExprs.empty(); }
  const std::vector<const CallExpr *> &getCallExprs() { return CallExprs; }

  bool foundAsmStmt() { return FoundAsmStmt; }

  void VisitCallExpr(const CallExpr *E) { CallExprs.push_back(E); }

  void VisitAsmStmt(const AsmStmt *S) { FoundAsmStmt = true; }

  void Visit(const Stmt *St) {
    if (!St)
      return;
    ConstEvaluatedExprVisitor<CallExprFinder>::Visit(St);
  }
};
} // namespace

static Attr *handleNoMergeAttr(Sema &S, Stmt *St, const ParsedAttr &A,
                               SourceRange Range) {
  NoMergeAttr NMA(S.Context, A);
  CallExprFinder CEF(S, St);

  if (!CEF.foundCallExpr() && !CEF.foundAsmStmt()) {
    S.Diag(St->getBeginLoc(), diag::warn_attribute_ignored_no_calls_in_stmt)
        << A;
    return nullptr;
  }

  return ::new (S.Context) NoMergeAttr(S.Context, A);
}

template <typename OtherAttr, int DiagIdx>
static bool CheckStmtInlineAttr(Sema &SemaRef, const Stmt *OrigSt,
                                const Stmt *CurSt,
                                const AttributeCommonInfo &A) {
  CallExprFinder OrigCEF(SemaRef, OrigSt);
  CallExprFinder CEF(SemaRef, CurSt);

  // If the call expressions lists are equal in size, we can skip
  // previously emitted diagnostics. However, if the statement has a pack
  // expansion, we have no way of telling which CallExpr is the instantiated
  // version of the other. In this case, we will end up re-diagnosing in the
  // instantiation.
  // ie: [[clang::always_inline]] non_dependent(), (other_call<Pack>()...)
  // will diagnose nondependent again.
  bool CanSuppressDiag =
      OrigSt && CEF.getCallExprs().size() == OrigCEF.getCallExprs().size();

  if (!CEF.foundCallExpr()) {
    return SemaRef.Diag(CurSt->getBeginLoc(),
                        diag::warn_attribute_ignored_no_calls_in_stmt)
           << A;
  }

  for (const auto &Tup :
       llvm::zip_longest(OrigCEF.getCallExprs(), CEF.getCallExprs())) {
    // If the original call expression already had a callee, we already
    // diagnosed this, so skip it here. We can't skip if there isn't a 1:1
    // relationship between the two lists of call expressions.
    if (!CanSuppressDiag || !(*std::get<0>(Tup))->getCalleeDecl()) {
      const Decl *Callee = (*std::get<1>(Tup))->getCalleeDecl();
      if (Callee &&
          (Callee->hasAttr<OtherAttr>() || Callee->hasAttr<FlattenAttr>())) {
        SemaRef.Diag(CurSt->getBeginLoc(),
                     diag::warn_function_stmt_attribute_precedence)
            << A << (Callee->hasAttr<OtherAttr>() ? DiagIdx : 1);
        SemaRef.Diag(Callee->getBeginLoc(), diag::note_conflicting_attribute);
      }
    }
  }

  return false;
}

bool Sema::CheckNoInlineAttr(const Stmt *OrigSt, const Stmt *CurSt,
                             const AttributeCommonInfo &A) {
  return CheckStmtInlineAttr<AlwaysInlineAttr, 0>(*this, OrigSt, CurSt, A);
}

bool Sema::CheckAlwaysInlineAttr(const Stmt *OrigSt, const Stmt *CurSt,
                                 const AttributeCommonInfo &A) {
  return CheckStmtInlineAttr<NoInlineAttr, 2>(*this, OrigSt, CurSt, A);
}

static Attr *handleNoInlineAttr(Sema &S, Stmt *St, const ParsedAttr &A,
                                SourceRange Range) {
  NoInlineAttr NIA(S.Context, A);
  if (!NIA.isStmtNoInline()) {
    S.Diag(St->getBeginLoc(), diag::warn_function_attribute_ignored_in_stmt)
        << "[[clang::noinline]]";
    return nullptr;
  }

  if (S.CheckNoInlineAttr(/*OrigSt=*/nullptr, St, A))
    return nullptr;

  return ::new (S.Context) NoInlineAttr(S.Context, A);
}

static Attr *handleAlwaysInlineAttr(Sema &S, Stmt *St, const ParsedAttr &A,
                                    SourceRange Range) {
  AlwaysInlineAttr AIA(S.Context, A);
  if (!AIA.isClangAlwaysInline()) {
    S.Diag(St->getBeginLoc(), diag::warn_function_attribute_ignored_in_stmt)
        << "[[clang::always_inline]]";
    return nullptr;
  }

  if (S.CheckAlwaysInlineAttr(/*OrigSt=*/nullptr, St, A))
    return nullptr;

  return ::new (S.Context) AlwaysInlineAttr(S.Context, A);
}

static Attr *handleCXXAssumeAttr(Sema &S, Stmt *St, const ParsedAttr &A,
                                 SourceRange Range) {
  ExprResult Res = S.ActOnCXXAssumeAttr(St, A, Range);
  if (!Res.isUsable())
    return nullptr;

  return ::new (S.Context) CXXAssumeAttr(S.Context, A, Res.get());
}

static Attr *handleMustTailAttr(Sema &S, Stmt *St, const ParsedAttr &A,
                                SourceRange Range) {
  // Validation is in Sema::ActOnAttributedStmt().
  return ::new (S.Context) MustTailAttr(S.Context, A);
}

static Attr *handleLikely(Sema &S, Stmt *St, const ParsedAttr &A,
                          SourceRange Range) {

  if (!S.getLangOpts().CPlusPlus20 && A.isCXX11Attribute() && !A.getScopeName())
    S.Diag(A.getLoc(), diag::ext_cxx20_attr) << A << Range;

  return ::new (S.Context) LikelyAttr(S.Context, A);
}

static Attr *handleUnlikely(Sema &S, Stmt *St, const ParsedAttr &A,
                            SourceRange Range) {

  if (!S.getLangOpts().CPlusPlus20 && A.isCXX11Attribute() && !A.getScopeName())
    S.Diag(A.getLoc(), diag::ext_cxx20_attr) << A << Range;

  return ::new (S.Context) UnlikelyAttr(S.Context, A);
}

CodeAlignAttr *Sema::BuildCodeAlignAttr(const AttributeCommonInfo &CI,
                                        Expr *E) {
  if (!E->isValueDependent()) {
    llvm::APSInt ArgVal;
    ExprResult Res = VerifyIntegerConstantExpression(E, &ArgVal);
    if (Res.isInvalid())
      return nullptr;
    E = Res.get();

    // This attribute requires an integer argument which is a constant power of
    // two between 1 and 4096 inclusive.
    if (ArgVal < CodeAlignAttr::MinimumAlignment ||
        ArgVal > CodeAlignAttr::MaximumAlignment || !ArgVal.isPowerOf2()) {
      if (std::optional<int64_t> Value = ArgVal.trySExtValue())
        Diag(CI.getLoc(), diag::err_attribute_power_of_two_in_range)
            << CI << CodeAlignAttr::MinimumAlignment
            << CodeAlignAttr::MaximumAlignment << Value.value();
      else
        Diag(CI.getLoc(), diag::err_attribute_power_of_two_in_range)
            << CI << CodeAlignAttr::MinimumAlignment
            << CodeAlignAttr::MaximumAlignment << E;
      return nullptr;
    }
  }
  return new (Context) CodeAlignAttr(Context, CI, E);
}

static Attr *handleCodeAlignAttr(Sema &S, Stmt *St, const ParsedAttr &A) {

  Expr *E = A.getArgAsExpr(0);
  return S.BuildCodeAlignAttr(A, E);
}

// Diagnose non-identical duplicates as a 'conflicting' loop attributes
// and suppress duplicate errors in cases where the two match.
template <typename LoopAttrT>
static void CheckForDuplicateLoopAttrs(Sema &S, ArrayRef<const Attr *> Attrs) {
  auto FindFunc = [](const Attr *A) { return isa<const LoopAttrT>(A); };
  const auto *FirstItr = std::find_if(Attrs.begin(), Attrs.end(), FindFunc);

  if (FirstItr == Attrs.end()) // no attributes found
    return;

  const auto *LastFoundItr = FirstItr;
  std::optional<llvm::APSInt> FirstValue;

  const auto *CAFA =
      dyn_cast<ConstantExpr>(cast<LoopAttrT>(*FirstItr)->getAlignment());
  // Return early if first alignment expression is dependent (since we don't
  // know what the effective size will be), and skip the loop entirely.
  if (!CAFA)
    return;

  while (Attrs.end() != (LastFoundItr = std::find_if(LastFoundItr + 1,
                                                     Attrs.end(), FindFunc))) {
    const auto *CASA =
        dyn_cast<ConstantExpr>(cast<LoopAttrT>(*LastFoundItr)->getAlignment());
    // If the value is dependent, we can not test anything.
    if (!CASA)
      return;
    // Test the attribute values.
    llvm::APSInt SecondValue = CASA->getResultAsAPSInt();
    if (!FirstValue)
      FirstValue = CAFA->getResultAsAPSInt();

    if (FirstValue != SecondValue) {
      S.Diag((*LastFoundItr)->getLocation(), diag::err_loop_attr_conflict)
          << *FirstItr;
      S.Diag((*FirstItr)->getLocation(), diag::note_previous_attribute);
    }
  }
  return;
}

static Attr *handleMSConstexprAttr(Sema &S, Stmt *St, const ParsedAttr &A,
                                   SourceRange Range) {
  if (!S.getLangOpts().isCompatibleWithMSVC(LangOptions::MSVC2022_3)) {
    S.Diag(A.getLoc(), diag::warn_unknown_attribute_ignored)
        << A << A.getRange();
    return nullptr;
  }
  return ::new (S.Context) MSConstexprAttr(S.Context, A);
}

#define WANT_STMT_MERGE_LOGIC
#include "clang/Sema/AttrParsedAttrImpl.inc"
#undef WANT_STMT_MERGE_LOGIC

static void
CheckForIncompatibleAttributes(Sema &S,
                               const SmallVectorImpl<const Attr *> &Attrs) {
  // The vast majority of attributed statements will only have one attribute
  // on them, so skip all of the checking in the common case.
  if (Attrs.size() < 2)
    return;

  // First, check for the easy cases that are table-generated for us.
  if (!DiagnoseMutualExclusions(S, Attrs))
    return;

  enum CategoryType {
    // For the following categories, they come in two variants: a state form and
    // a numeric form. The state form may be one of default, enable, and
    // disable. The numeric form provides an integer hint (for example, unroll
    // count) to the transformer.
    Vectorize,
    Interleave,
    UnrollAndJam,
    Pipeline,
    // For unroll, default indicates full unrolling rather than enabling the
    // transformation.
    Unroll,
    // The loop distribution transformation only has a state form that is
    // exposed by #pragma clang loop distribute (enable | disable).
    Distribute,
    // The vector predication only has a state form that is exposed by
    // #pragma clang loop vectorize_predicate (enable | disable).
    VectorizePredicate,
    // This serves as a indicator to how many category are listed in this enum.
    NumberOfCategories
  };
  // The following array accumulates the hints encountered while iterating
  // through the attributes to check for compatibility.
  struct {
    const LoopHintAttr *StateAttr;
    const LoopHintAttr *NumericAttr;
  } HintAttrs[CategoryType::NumberOfCategories] = {};

  for (const auto *I : Attrs) {
    const LoopHintAttr *LH = dyn_cast<LoopHintAttr>(I);

    // Skip non loop hint attributes
    if (!LH)
      continue;

    CategoryType Category = CategoryType::NumberOfCategories;
    LoopHintAttr::OptionType Option = LH->getOption();
    switch (Option) {
    case LoopHintAttr::Vectorize:
    case LoopHintAttr::VectorizeWidth:
      Category = Vectorize;
      break;
    case LoopHintAttr::Interleave:
    case LoopHintAttr::InterleaveCount:
      Category = Interleave;
      break;
    case LoopHintAttr::Unroll:
    case LoopHintAttr::UnrollCount:
      Category = Unroll;
      break;
    case LoopHintAttr::UnrollAndJam:
    case LoopHintAttr::UnrollAndJamCount:
      Category = UnrollAndJam;
      break;
    case LoopHintAttr::Distribute:
      // Perform the check for duplicated 'distribute' hints.
      Category = Distribute;
      break;
    case LoopHintAttr::PipelineDisabled:
    case LoopHintAttr::PipelineInitiationInterval:
      Category = Pipeline;
      break;
    case LoopHintAttr::VectorizePredicate:
      Category = VectorizePredicate;
      break;
    };

    assert(Category != NumberOfCategories && "Unhandled loop hint option");
    auto &CategoryState = HintAttrs[Category];
    const LoopHintAttr *PrevAttr;
    if (Option == LoopHintAttr::Vectorize ||
        Option == LoopHintAttr::Interleave || Option == LoopHintAttr::Unroll ||
        Option == LoopHintAttr::UnrollAndJam ||
        Option == LoopHintAttr::VectorizePredicate ||
        Option == LoopHintAttr::PipelineDisabled ||
        Option == LoopHintAttr::Distribute) {
      // Enable|Disable|AssumeSafety hint.  For example, vectorize(enable).
      PrevAttr = CategoryState.StateAttr;
      CategoryState.StateAttr = LH;
    } else {
      // Numeric hint.  For example, vectorize_width(8).
      PrevAttr = CategoryState.NumericAttr;
      CategoryState.NumericAttr = LH;
    }

    PrintingPolicy Policy(S.Context.getLangOpts());
    SourceLocation OptionLoc = LH->getRange().getBegin();
    if (PrevAttr)
      // Cannot specify same type of attribute twice.
      S.Diag(OptionLoc, diag::err_pragma_loop_compatibility)
          << /*Duplicate=*/true << PrevAttr->getDiagnosticName(Policy)
          << LH->getDiagnosticName(Policy);

    if (CategoryState.StateAttr && CategoryState.NumericAttr &&
        (Category == Unroll || Category == UnrollAndJam ||
         CategoryState.StateAttr->getState() == LoopHintAttr::Disable)) {
      // Disable hints are not compatible with numeric hints of the same
      // category.  As a special case, numeric unroll hints are also not
      // compatible with enable or full form of the unroll pragma because these
      // directives indicate full unrolling.
      S.Diag(OptionLoc, diag::err_pragma_loop_compatibility)
          << /*Duplicate=*/false
          << CategoryState.StateAttr->getDiagnosticName(Policy)
          << CategoryState.NumericAttr->getDiagnosticName(Policy);
    }
  }
}

template <typename LoopAttrT>
static void
CheckForDuplicationSYCLLoopAttribute(Sema &S,
                                     const SmallVectorImpl<const Attr *> &Attrs,
                                     bool isIntelFPGAAttr = true) {
  const LoopAttrT *LoopAttr = nullptr;

  for (const auto *I : Attrs) {
    if (LoopAttr && isa<LoopAttrT>(I)) {
      // Cannot specify same type of attribute twice.
      S.Diag(I->getLocation(), diag::err_sycl_loop_attr_duplication)
          << isIntelFPGAAttr << LoopAttr;
    }
    if (isa<LoopAttrT>(I))
      LoopAttr = cast<LoopAttrT>(I);
  }
}

// Diagnose non-identical duplicates as a 'conflicting' loop attributes
// and suppress duplicate errors in cases where the two match for
// FPGA attributes: 'SYCLIntelMaxInterleavingAttr',
// 'SYCLIntelSpeculatedIterationsAttr', 'SYCLIntelMaxReinvocationDelayAttr',
// 'SYCLIntelInitiationIntervalAttr', and 'SYCLIntelMaxConcurrencyAttr'
template <typename LoopAttrT>
static void CheckForDuplicateAttrs(Sema &S, ArrayRef<const Attr *> Attrs) {
  auto FindFunc = [](const Attr *A) { return isa<const LoopAttrT>(A); };
  const auto *FirstItr = std::find_if(Attrs.begin(), Attrs.end(), FindFunc);

  if (FirstItr == Attrs.end()) // no attributes found
    return;

  const auto *LastFoundItr = FirstItr;
  std::optional<llvm::APSInt> FirstValue;

  const auto *CAFA =
      dyn_cast<ConstantExpr>(cast<LoopAttrT>(*FirstItr)->getNExpr());
  // Return early if first expression is dependent (since we don't
  // know what the effective size will be), and skip the loop entirely.
  if (!CAFA)
    return;

  while (Attrs.end() != (LastFoundItr = std::find_if(LastFoundItr + 1,
                                                     Attrs.end(), FindFunc))) {
    const auto *CASA =
        dyn_cast<ConstantExpr>(cast<LoopAttrT>(*LastFoundItr)->getNExpr());
    // If the value is dependent, we can not test anything.
    if (!CASA)
      return;
    // Test the attribute values.
    llvm::APSInt SecondValue = CASA->getResultAsAPSInt();
    if (!FirstValue)
      FirstValue = CAFA->getResultAsAPSInt();

    if (FirstValue != SecondValue) {
      S.Diag((*LastFoundItr)->getLocation(), diag::err_loop_attr_conflict)
          << *FirstItr;
      S.Diag((*FirstItr)->getLocation(), diag::note_previous_attribute);
    }
  }
}

static void CheckForIncompatibleSYCLLoopAttributes(
    Sema &S, const SmallVectorImpl<const Attr *> &Attrs) {
  CheckForDuplicateAttrs<SYCLIntelInitiationIntervalAttr>(S, Attrs);
  CheckForDuplicateAttrs<SYCLIntelMaxConcurrencyAttr>(S, Attrs);
  CheckForDuplicationSYCLLoopAttribute<SYCLIntelLoopCoalesceAttr>(S, Attrs);
  CheckForDuplicationSYCLLoopAttribute<SYCLIntelDisableLoopPipeliningAttr>(
      S, Attrs);
  CheckForDuplicateAttrs<SYCLIntelMaxInterleavingAttr>(S, Attrs);
  CheckForDuplicateAttrs<SYCLIntelSpeculatedIterationsAttr>(S, Attrs);
  CheckForDuplicateSYCLIntelLoopCountAttrs(S, Attrs);
  CheckForDuplicationSYCLLoopAttribute<LoopUnrollHintAttr>(S, Attrs, false);
  CheckRedundantSYCLIntelIVDepAttrs(S, Attrs);
  CheckForDuplicationSYCLLoopAttribute<SYCLIntelNofusionAttr>(S, Attrs);
  CheckForDuplicateAttrs<SYCLIntelMaxReinvocationDelayAttr>(S, Attrs);
  CheckForDuplicationSYCLLoopAttribute<SYCLIntelEnableLoopPipeliningAttr>(
      S, Attrs);
}

void CheckForIncompatibleUnrollHintAttributes(
    Sema &S, const SmallVectorImpl<const Attr *> &Attrs, SourceRange Range) {

  // This check is entered after it was analyzed that there are no duplicating
  // pragmas and loop attributes. So, let's perform check that there are no
  // conflicting pragma unroll and unroll attribute for the loop.
  const LoopUnrollHintAttr *AttrUnroll = nullptr;
  const LoopHintAttr *PragmaUnroll = nullptr;
  for (const auto *I : Attrs) {
    if (auto *LH = dyn_cast<LoopUnrollHintAttr>(I))
      AttrUnroll = LH;
    if (auto *LH = dyn_cast<LoopHintAttr>(I)) {
      LoopHintAttr::OptionType Opt = LH->getOption();
      if (Opt == LoopHintAttr::Unroll || Opt == LoopHintAttr::UnrollCount)
        PragmaUnroll = LH;
    }
  }

  if (AttrUnroll && PragmaUnroll) {
    PrintingPolicy Policy(S.Context.getLangOpts());
    SourceLocation Loc = Range.getBegin();
    S.Diag(Loc, diag::err_loop_unroll_compatibility)
        << PragmaUnroll->getDiagnosticName(Policy)
        << AttrUnroll->getDiagnosticName(Policy);
  }
}

static bool CheckLoopUnrollAttrExpr(Sema &S, Expr *E,
                                    const AttributeCommonInfo &A,
                                    unsigned *UnrollFactor = nullptr) {
  if (E && !E->isInstantiationDependent()) {
    std::optional<llvm::APSInt> ArgVal = E->getIntegerConstantExpr(S.Context);
    if (!ArgVal)
      return S.Diag(E->getExprLoc(), diag::err_attribute_argument_type)
             << A.getAttrName() << AANT_ArgumentIntegerConstant
             << E->getSourceRange();

    if (ArgVal->isNonPositive())
      return S.Diag(E->getExprLoc(),
                    diag::err_attribute_requires_positive_integer)
             << A.getAttrName() << /* positive */ 0;

    if (UnrollFactor)
      *UnrollFactor = ArgVal->getZExtValue();
  }
  return false;
}

LoopUnrollHintAttr *Sema::BuildLoopUnrollHintAttr(const AttributeCommonInfo &A,
                                                  Expr *E) {
  return !CheckLoopUnrollAttrExpr(*this, E, A)
             ? new (Context) LoopUnrollHintAttr(Context, A, E)
             : nullptr;
}

OpenCLUnrollHintAttr *
Sema::BuildOpenCLLoopUnrollHintAttr(const AttributeCommonInfo &A, Expr *E) {
  unsigned UnrollFactor = 0;
  return !CheckLoopUnrollAttrExpr(*this, E, A, &UnrollFactor)
             ? new (Context) OpenCLUnrollHintAttr(Context, A, UnrollFactor)
             : nullptr;
}

static Attr *handleLoopUnrollHint(Sema &S, Stmt *St, const ParsedAttr &A,
                                  SourceRange Range) {
  // Although the feature was introduced only in OpenCL C v2.0 s6.11.5, it's
  // useful for OpenCL 1.x too and doesn't require HW support.
  // opencl_unroll_hint or clang::unroll can have 0 arguments (compiler
  // determines unrolling factor) or 1 argument (the unroll factor provided
  // by the user).

  Expr *E = A.getNumArgs() ? A.getArgAsExpr(0) : nullptr;
  if (A.getParsedKind() == ParsedAttr::AT_OpenCLUnrollHint)
    return S.BuildOpenCLLoopUnrollHintAttr(A, E);
  if (A.getParsedKind() == ParsedAttr::AT_LoopUnrollHint)
    return S.BuildLoopUnrollHintAttr(A, E);

  llvm_unreachable("Unknown loop unroll hint");
}

static Attr *ProcessStmtAttribute(Sema &S, Stmt *St, const ParsedAttr &A,
                                  SourceRange Range) {
  if (A.isInvalid() || A.getKind() == ParsedAttr::IgnoredAttribute)
    return nullptr;

  // Unknown attributes are automatically warned on. Target-specific attributes
  // which do not apply to the current target architecture are treated as
  // though they were unknown attributes.
  const TargetInfo *Aux = S.Context.getAuxTargetInfo();
  if (A.getKind() == ParsedAttr::UnknownAttribute ||
      !(A.existsInTarget(S.Context.getTargetInfo()) ||
        (S.Context.getLangOpts().SYCLIsDevice && Aux &&
         A.existsInTarget(*Aux)))) {
    S.Diag(A.getLoc(), A.isRegularKeywordAttribute()
                           ? (unsigned)diag::err_keyword_not_supported_on_target
                       : A.isDeclspecAttribute()
                           ? (unsigned)diag::warn_unhandled_ms_attribute_ignored
                           : (unsigned)diag::warn_unknown_attribute_ignored)
        << A << A.getRange();
    return nullptr;
  }

  if (S.checkCommonAttributeFeatures(St, A))
    return nullptr;

  switch (A.getKind()) {
  case ParsedAttr::AT_AlwaysInline:
    return handleAlwaysInlineAttr(S, St, A, Range);
  case ParsedAttr::AT_CXXAssume:
    return handleCXXAssumeAttr(S, St, A, Range);
  case ParsedAttr::AT_FallThrough:
    return handleFallThroughAttr(S, St, A, Range);
  case ParsedAttr::AT_LoopHint:
    return handleLoopHintAttr(S, St, A, Range);
  case ParsedAttr::AT_SYCLIntelIVDep:
    return handleIntelIVDepAttr(S, St, A);
  case ParsedAttr::AT_SYCLIntelInitiationInterval:
    return handleSYCLIntelInitiationIntervalAttr(S, St, A);
  case ParsedAttr::AT_SYCLIntelMaxConcurrency:
    return handleSYCLIntelMaxConcurrencyAttr(S, St, A);
  case ParsedAttr::AT_SYCLIntelLoopCoalesce:
    return handleSYCLIntelLoopCoalesceAttr(S, St, A);
  case ParsedAttr::AT_SYCLIntelDisableLoopPipelining:
    return handleSYCLIntelDisableLoopPipeliningAttr(S, St, A);
  case ParsedAttr::AT_SYCLIntelMaxInterleaving:
    return handleSYCLIntelMaxInterleavingAttr(S, St, A);
  case ParsedAttr::AT_SYCLIntelSpeculatedIterations:
    return handleSYCLIntelSpeculatedIterationsAttr(S, St, A);
  case ParsedAttr::AT_SYCLIntelLoopCount:
    return handleSYCLIntelLoopCountAttr(S, St, A);
  case ParsedAttr::AT_OpenCLUnrollHint:
  case ParsedAttr::AT_LoopUnrollHint:
    return handleLoopUnrollHint(S, St, A, Range);
  case ParsedAttr::AT_Suppress:
    return handleSuppressAttr(S, St, A, Range);
  case ParsedAttr::AT_NoMerge:
    return handleNoMergeAttr(S, St, A, Range);
  case ParsedAttr::AT_NoInline:
    return handleNoInlineAttr(S, St, A, Range);
  case ParsedAttr::AT_MustTail:
    return handleMustTailAttr(S, St, A, Range);
  case ParsedAttr::AT_Likely:
    return handleLikely(S, St, A, Range);
  case ParsedAttr::AT_Unlikely:
    return handleUnlikely(S, St, A, Range);
  case ParsedAttr::AT_SYCLIntelNofusion:
    return handleIntelNofusionAttr(S, St, A);
  case ParsedAttr::AT_SYCLIntelMaxReinvocationDelay:
    return handleSYCLIntelMaxReinvocationDelayAttr(S, St, A);
  case ParsedAttr::AT_SYCLIntelEnableLoopPipelining:
    return handleSYCLIntelEnableLoopPipeliningAttr(S, St, A);
  case ParsedAttr::AT_CodeAlign:
    return handleCodeAlignAttr(S, St, A);
  case ParsedAttr::AT_MSConstexpr:
    return handleMSConstexprAttr(S, St, A, Range);
  default:
    // N.B., ClangAttrEmitter.cpp emits a diagnostic helper that ensures a
    // declaration attribute is not written on a statement, but this code is
    // needed for attributes in Attr.td that do not list any subjects.
    S.Diag(A.getRange().getBegin(), diag::err_decl_attribute_invalid_on_stmt)
        << A << A.isRegularKeywordAttribute() << St->getBeginLoc();
    return nullptr;
  }
}

void Sema::ProcessStmtAttributes(Stmt *S, const ParsedAttributes &InAttrs,
                                 SmallVectorImpl<const Attr *> &OutAttrs) {
  for (const ParsedAttr &AL : InAttrs) {
    if (const Attr *A = ProcessStmtAttribute(*this, S, AL, InAttrs.Range))
      OutAttrs.push_back(A);
  }

  CheckForIncompatibleAttributes(*this, OutAttrs);
  CheckForIncompatibleSYCLLoopAttributes(*this, OutAttrs);
  CheckForIncompatibleUnrollHintAttributes(*this, OutAttrs, InAttrs.Range);
  CheckForDuplicateLoopAttrs<CodeAlignAttr>(*this, OutAttrs);
}

bool Sema::CheckRebuiltAttributedStmtAttributes(ArrayRef<const Attr *> Attrs) {
  CheckRedundantSYCLIntelIVDepAttrs(*this, Attrs);
  CheckForDuplicateAttrs<SYCLIntelSpeculatedIterationsAttr>(*this, Attrs);
  CheckForDuplicateAttrs<SYCLIntelMaxInterleavingAttr>(*this, Attrs);
  CheckForDuplicateAttrs<SYCLIntelMaxReinvocationDelayAttr>(*this, Attrs);
  CheckForDuplicateAttrs<SYCLIntelInitiationIntervalAttr>(*this, Attrs);
  CheckForDuplicateAttrs<SYCLIntelMaxConcurrencyAttr>(*this, Attrs);
  return false;
}

bool Sema::CheckRebuiltStmtAttributes(ArrayRef<const Attr *> Attrs) {
  CheckForDuplicateLoopAttrs<CodeAlignAttr>(*this, Attrs);
  return false;
}

ExprResult Sema::ActOnCXXAssumeAttr(Stmt *St, const ParsedAttr &A,
                                    SourceRange Range) {
  if (A.getNumArgs() != 1 || !A.getArgAsExpr(0)) {
    Diag(A.getLoc(), diag::err_attribute_wrong_number_arguments)
        << A.getAttrName() << 1 << Range;
    return ExprError();
  }

  auto *Assumption = A.getArgAsExpr(0);

  if (DiagnoseUnexpandedParameterPack(Assumption)) {
    return ExprError();
  }

  if (Assumption->getDependence() == ExprDependence::None) {
    ExprResult Res = BuildCXXAssumeExpr(Assumption, A.getAttrName(), Range);
    if (Res.isInvalid())
      return ExprError();
    Assumption = Res.get();
  }

  if (!getLangOpts().CPlusPlus23 &&
      A.getSyntax() == AttributeCommonInfo::AS_CXX11)
    Diag(A.getLoc(), diag::ext_cxx23_attr) << A << Range;

  return Assumption;
}

ExprResult Sema::BuildCXXAssumeExpr(Expr *Assumption,
                                    const IdentifierInfo *AttrName,
                                    SourceRange Range) {
  ExprResult Res = CorrectDelayedTyposInExpr(Assumption);
  if (Res.isInvalid())
    return ExprError();

  Res = CheckPlaceholderExpr(Res.get());
  if (Res.isInvalid())
    return ExprError();

  Res = PerformContextuallyConvertToBool(Res.get());
  if (Res.isInvalid())
    return ExprError();

  Assumption = Res.get();
  if (Assumption->HasSideEffects(Context))
    Diag(Assumption->getBeginLoc(), diag::warn_assume_side_effects)
        << AttrName << Range;

  return Assumption;
}
