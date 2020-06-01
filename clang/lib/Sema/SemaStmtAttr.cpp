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

#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/DelayedDiagnostic.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/ScopeInfo.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang;
using namespace sema;

static Attr *handleFallThroughAttr(Sema &S, Stmt *St, const ParsedAttr &A,
                                   SourceRange Range) {
  FallThroughAttr Attr(S.Context, A);
  if (!isa<NullStmt>(St)) {
    S.Diag(A.getRange().getBegin(), diag::err_fallthrough_attr_wrong_target)
        << Attr.getSpelling() << St->getBeginLoc();
    if (isa<SwitchCase>(St)) {
      SourceLocation L = S.getLocForEndOfToken(Range.getEnd());
      S.Diag(L, diag::note_fallthrough_insert_semi_fixit)
          << FixItHint::CreateInsertion(L, ";");
    }
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
  if (A.getNumArgs() < 1) {
    S.Diag(A.getLoc(), diag::err_attribute_too_few_arguments) << A << 1;
    return nullptr;
  }

  std::vector<StringRef> DiagnosticIdentifiers;
  for (unsigned I = 0, E = A.getNumArgs(); I != E; ++I) {
    StringRef RuleName;

    if (!S.checkStringLiteralArgumentAttr(A, I, RuleName, nullptr))
      return nullptr;

    // FIXME: Warn if the rule name is unknown. This is tricky because only
    // clang-tidy knows about available rules.
    DiagnosticIdentifiers.push_back(RuleName);
  }

  return ::new (S.Context) SuppressAttr(
      S.Context, A, DiagnosticIdentifiers.data(), DiagnosticIdentifiers.size());
}

template <typename FPGALoopAttrT>
static Attr *handleIntelFPGALoopAttr(Sema &S, const ParsedAttr &A) {
  if(S.LangOpts.SYCLIsHost)
    return nullptr;

  unsigned NumArgs = A.getNumArgs();
  if (NumArgs > 1) {
    S.Diag(A.getLoc(), diag::warn_attribute_too_many_arguments) << A << 1;
    return nullptr;
  }

  if (NumArgs == 0) {
    if (A.getKind() == ParsedAttr::AT_SYCLIntelFPGAII ||
        A.getKind() == ParsedAttr::AT_SYCLIntelFPGAMaxConcurrency ||
        A.getKind() == ParsedAttr::AT_SYCLIntelFPGAMaxInterleaving ||
        A.getKind() == ParsedAttr::AT_SYCLIntelFPGASpeculatedIterations) {
      S.Diag(A.getLoc(), diag::warn_attribute_too_few_arguments) << A << 1;
      return nullptr;
    }
  }

  return S.BuildSYCLIntelFPGALoopAttr<FPGALoopAttrT>(
      A, A.getNumArgs() ? A.getArgAsExpr(0) : nullptr);
}

template <>
Attr *handleIntelFPGALoopAttr<SYCLIntelFPGADisableLoopPipeliningAttr>(
    Sema &S, const ParsedAttr &A) {
  if (S.LangOpts.SYCLIsHost)
    return nullptr;

  unsigned NumArgs = A.getNumArgs();
  if (NumArgs > 0) {
    S.Diag(A.getLoc(), diag::warn_attribute_too_many_arguments) << A << 0;
    return nullptr;
  }

  return new (S.Context) SYCLIntelFPGADisableLoopPipeliningAttr(S.Context, A);
}

static bool checkSYCLIntelFPGAIVDepSafeLen(Sema &S, llvm::APSInt &Value,
                                           Expr *E) {
  if (!Value.isStrictlyPositive())
    return S.Diag(E->getExprLoc(),
                  diag::err_attribute_requires_positive_integer)
           << "'ivdep'" << /* positive */ 0;
  return false;
}

enum class IVDepExprResult {
  Invalid,
  Null,
  Dependent,
  Array,
  SafeLen,
};

static IVDepExprResult HandleFPGAIVDepAttrExpr(Sema &S, Expr *E,
                                               unsigned &SafelenValue) {
  if (!E)
    return IVDepExprResult::Null;

  if (E->isInstantiationDependent())
    return IVDepExprResult::Dependent;

  llvm::APSInt ArgVal;

  if (E->isIntegerConstantExpr(ArgVal, S.getASTContext())) {
    if (checkSYCLIntelFPGAIVDepSafeLen(S, ArgVal, E))
      return IVDepExprResult::Invalid;
    SafelenValue = ArgVal.getZExtValue();
    return IVDepExprResult::SafeLen;
  }

  if (isa<DeclRefExpr>(E)) {
    if (!cast<DeclRefExpr>(E)->getType()->isArrayType()) {
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
SYCLIntelFPGAIVDepAttr *
Sema::BuildSYCLIntelFPGAIVDepAttr(const AttributeCommonInfo &CI, Expr *Expr1,
                                  Expr *Expr2) {
  unsigned SafelenValue = 0;
  IVDepExprResult E1 = HandleFPGAIVDepAttrExpr(*this, Expr1, SafelenValue);
  IVDepExprResult E2 = HandleFPGAIVDepAttrExpr(*this, Expr2, SafelenValue);

  if (E1 == IVDepExprResult::Invalid || E2 == IVDepExprResult::Invalid)
    return nullptr;

  if (E1 == E2 && E1 != IVDepExprResult::Dependent &&
      E1 != IVDepExprResult::Null) {
    Diag(Expr2->getExprLoc(), diag::err_ivdep_duplicate_arg);
    return nullptr;
  }

  // Try to put Safelen in the 1st one so codegen can count on the ordering.
  Expr *SafeLenExpr;
  Expr *ArrayExpr;
  if (E1 == IVDepExprResult::SafeLen) {
    SafeLenExpr = Expr1;
    ArrayExpr = Expr2;
  } else {
    SafeLenExpr = Expr2;
    ArrayExpr = Expr1;
  }

  return new (Context)
      SYCLIntelFPGAIVDepAttr(Context, CI, SafeLenExpr, ArrayExpr, SafelenValue);
}

// Filters out any attributes from the list that are either not the specified
// type, or whose function isDependent returns true.
template <typename T>
static void FilterAttributeList(ArrayRef<const Attr *> Attrs,
                    SmallVectorImpl<const T *> &FilteredAttrs) {

  llvm::transform(Attrs, std::back_inserter(FilteredAttrs), [](const Attr *A) {
    if (const auto *Cast = dyn_cast_or_null<const T>(A))
      return Cast->isDependent() ? nullptr : Cast;
    return static_cast<const T*>(nullptr);
  });
  FilteredAttrs.erase(
      std::remove(FilteredAttrs.begin(), FilteredAttrs.end(),
                  static_cast<const T*>(nullptr)),
      FilteredAttrs.end());
}

static void
CheckRedundantSYCLIntelFPGAIVDepAttrs(Sema &S, ArrayRef<const Attr *> Attrs) {
  // Skip SEMA if we're in a template, this will be diagnosed later.
  if (S.getCurLexicalContext()->isDependentContext())
    return;

  SmallVector<const SYCLIntelFPGAIVDepAttr *, 8> FilteredAttrs;
  // Filter down to just non-dependent ivdeps.
  FilterAttributeList(Attrs, FilteredAttrs);
  if (FilteredAttrs.empty())
    return;

  SmallVector<const SYCLIntelFPGAIVDepAttr *, 8> SortedAttrs(FilteredAttrs);
  llvm::stable_sort(SortedAttrs, SYCLIntelFPGAIVDepAttr::SafelenCompare);

  // Find the maximum without an array expression, which ends up in the 2nd
  // expr.
  const auto *GlobalMaxItr =
      llvm::find_if(SortedAttrs, [](const SYCLIntelFPGAIVDepAttr *A) {
        return !A->getArrayExpr();
      });
  const SYCLIntelFPGAIVDepAttr *GlobalMax =
      GlobalMaxItr == SortedAttrs.end() ? nullptr : *GlobalMaxItr;

  for (const auto *A : FilteredAttrs) {
    if (A == GlobalMax)
      continue;

    if (GlobalMax && !SYCLIntelFPGAIVDepAttr::SafelenCompare(A, GlobalMax)) {
      S.Diag(A->getLocation(), diag::warn_ivdep_redundant)
          << !GlobalMax->isInf() << GlobalMax->getSafelenValue() << !A->isInf()
          << A->getSafelenValue();
      S.Diag(GlobalMax->getLocation(), diag::note_previous_attribute);
      continue;
    }

    if (!A->getArrayExpr())
      continue;

    const VarDecl *ArrayDecl = A->getArrayDecl();
    auto Other = llvm::find_if(SortedAttrs,
                               [ArrayDecl](const SYCLIntelFPGAIVDepAttr *A) {
                                 return ArrayDecl == A->getArrayDecl();
                               });
    assert(Other != SortedAttrs.end() && "Should find at least itself");

    // Diagnose if lower/equal to the lowest with this array.
    if (*Other != A && !SYCLIntelFPGAIVDepAttr::SafelenCompare(A, *Other)) {
      S.Diag(A->getLocation(), diag::warn_ivdep_redundant)
          << !(*Other)->isInf() << (*Other)->getSafelenValue() << !A->isInf()
          << A->getSafelenValue();
      S.Diag((*Other)->getLocation(), diag::note_previous_attribute);
    }
  }
}

static Attr *handleIntelFPGAIVDepAttr(Sema &S, const ParsedAttr &A) {
  unsigned NumArgs = A.getNumArgs();
  if (NumArgs > 2) {
    S.Diag(A.getLoc(), diag::err_attribute_too_many_arguments) << A << 2;
    return nullptr;
  }

  return S.BuildSYCLIntelFPGAIVDepAttr(
      A, NumArgs >= 1 ? A.getArgAsExpr(0) : nullptr,
      NumArgs == 2 ? A.getArgAsExpr(1) : nullptr);
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

  if (St->getStmtClass() != Stmt::DoStmtClass &&
      St->getStmtClass() != Stmt::ForStmtClass &&
      St->getStmtClass() != Stmt::CXXForRangeStmtClass &&
      St->getStmtClass() != Stmt::WhileStmtClass) {
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
    if (ValueExpr)
      SetHints(LoopHintAttr::UnrollCount, LoopHintAttr::Numeric);
    else
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
    if (Option == LoopHintAttr::VectorizeWidth ||
        Option == LoopHintAttr::InterleaveCount ||
        Option == LoopHintAttr::UnrollCount ||
        Option == LoopHintAttr::PipelineInitiationInterval) {
      assert(ValueExpr && "Attribute must have a valid value expression.");
      if (S.CheckLoopHintExpr(ValueExpr, St->getBeginLoc()))
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
  bool FoundCallExpr = false;

public:
  typedef ConstEvaluatedExprVisitor<CallExprFinder> Inherited;

  CallExprFinder(Sema &S, const Stmt *St) : Inherited(S.Context) { Visit(St); }

  bool foundCallExpr() { return FoundCallExpr; }

  void VisitCallExpr(const CallExpr *E) { FoundCallExpr = true; }

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
  if (S.CheckAttrNoArgs(A))
    return nullptr;

  CallExprFinder CEF(S, St);

  if (!CEF.foundCallExpr()) {
    S.Diag(St->getBeginLoc(), diag::warn_nomerge_attribute_ignored_in_stmt)
        << NMA.getSpelling();
    return nullptr;
  }

  return ::new (S.Context) NoMergeAttr(S.Context, A);
}

static void
CheckForIncompatibleAttributes(Sema &S,
                               const SmallVectorImpl<const Attr *> &Attrs) {
  // There are 6 categories of loop hints attributes: vectorize, interleave,
  // unroll, unroll_and_jam, pipeline and distribute. Except for distribute they
  // come in two variants: a state form and a numeric form.  The state form
  // selectively defaults/enables/disables the transformation for the loop
  // (for unroll, default indicates full unrolling rather than enabling the
  // transformation). The numeric form form provides an integer hint (for
  // example, unroll count) to the transformer. The following array accumulates
  // the hints encountered while iterating through the attributes to check for
  // compatibility.
  struct {
    const LoopHintAttr *StateAttr;
    const LoopHintAttr *NumericAttr;
  } HintAttrs[] = {{nullptr, nullptr}, {nullptr, nullptr}, {nullptr, nullptr},
                   {nullptr, nullptr}, {nullptr, nullptr}, {nullptr, nullptr},
                   {nullptr, nullptr}};

  for (const auto *I : Attrs) {
    const LoopHintAttr *LH = dyn_cast<LoopHintAttr>(I);

    // Skip non loop hint attributes
    if (!LH)
      continue;

    LoopHintAttr::OptionType Option = LH->getOption();
    enum {
      Vectorize,
      Interleave,
      Unroll,
      UnrollAndJam,
      Distribute,
      Pipeline,
      VectorizePredicate
    } Category;
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

    assert(Category < sizeof(HintAttrs) / sizeof(HintAttrs[0]));
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
static void CheckForDuplicationSYCLLoopAttribute(
    Sema &S, const SmallVectorImpl<const Attr *> &Attrs, SourceRange Range,
    bool isIntelFPGAAttr = true) {
  const LoopAttrT *LoopAttr = nullptr;

  for (const auto *I : Attrs) {
    if (LoopAttr) {
      if (isa<LoopAttrT>(I)) {
        SourceLocation Loc = Range.getBegin();
        // Cannot specify same type of attribute twice.
        S.Diag(Loc, diag::err_sycl_loop_attr_duplication)
            << isIntelFPGAAttr << LoopAttr->getName();
      }
    }
    if (isa<LoopAttrT>(I))
      LoopAttr = cast<LoopAttrT>(I);
  }
}

/// Diagnose mutually exclusive attributes when present on a given
/// declaration. Returns true if diagnosed.
template <typename LoopAttrT, typename LoopAttrT2>
static void CheckMutualExclusionSYCLLoopAttribute(
    Sema &S, const SmallVectorImpl<const Attr *> &Attrs, SourceRange Range) {
  const LoopAttrT *LoopAttr = nullptr;
  const LoopAttrT2 *LoopAttr2 = nullptr;

  for (const auto *I : Attrs) {
    if (isa<LoopAttrT>(I))
      LoopAttr = cast<LoopAttrT>(I);
    if (isa<LoopAttrT2>(I))
      LoopAttr2 = cast<LoopAttrT2>(I);
    if (LoopAttr && LoopAttr2) {
      S.Diag(Range.getBegin(), diag::err_attributes_are_not_compatible)
          << LoopAttr->getSpelling() << LoopAttr2->getSpelling();
    }
  }
}

static void CheckForIncompatibleSYCLLoopAttributes(
    Sema &S, const SmallVectorImpl<const Attr *> &Attrs, SourceRange Range) {
  CheckForDuplicationSYCLLoopAttribute<SYCLIntelFPGAIIAttr>(S, Attrs, Range);
  CheckForDuplicationSYCLLoopAttribute<SYCLIntelFPGAMaxConcurrencyAttr>(
      S, Attrs, Range);
  CheckForDuplicationSYCLLoopAttribute<SYCLIntelFPGALoopCoalesceAttr>(S, Attrs,
                                                                      Range);
  CheckForDuplicationSYCLLoopAttribute<SYCLIntelFPGADisableLoopPipeliningAttr>(
      S, Attrs, Range);
  CheckForDuplicationSYCLLoopAttribute<SYCLIntelFPGAMaxInterleavingAttr>(
      S, Attrs, Range);
  CheckForDuplicationSYCLLoopAttribute<SYCLIntelFPGASpeculatedIterationsAttr>(
      S, Attrs, Range);
  CheckForDuplicationSYCLLoopAttribute<LoopUnrollHintAttr>(S, Attrs, Range,
                                                           false);
  CheckMutualExclusionSYCLLoopAttribute<SYCLIntelFPGADisableLoopPipeliningAttr,
                                        SYCLIntelFPGAMaxInterleavingAttr>(
      S, Attrs, Range);
  CheckMutualExclusionSYCLLoopAttribute<SYCLIntelFPGADisableLoopPipeliningAttr,
                                        SYCLIntelFPGASpeculatedIterationsAttr>(
      S, Attrs, Range);
  CheckMutualExclusionSYCLLoopAttribute<SYCLIntelFPGADisableLoopPipeliningAttr,
                                        SYCLIntelFPGAIIAttr>(S, Attrs, Range);
  CheckMutualExclusionSYCLLoopAttribute<SYCLIntelFPGADisableLoopPipeliningAttr,
                                        SYCLIntelFPGAIVDepAttr>(S, Attrs,
                                                                Range);
  CheckMutualExclusionSYCLLoopAttribute<SYCLIntelFPGADisableLoopPipeliningAttr,
                                        SYCLIntelFPGAMaxConcurrencyAttr>(
      S, Attrs, Range);

  CheckRedundantSYCLIntelFPGAIVDepAttrs(S, Attrs);
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
    llvm::APSInt ArgVal(32);

    if (!E->isIntegerConstantExpr(ArgVal, S.Context))
      return S.Diag(E->getExprLoc(), diag::err_attribute_argument_type)
             << A.getAttrName() << AANT_ArgumentIntegerConstant
             << E->getSourceRange();

    if (ArgVal.isNonPositive())
      return S.Diag(E->getExprLoc(),
                    diag::err_attribute_requires_positive_integer)
             << A.getAttrName() << /* positive */ 0;

    if (UnrollFactor)
      *UnrollFactor = ArgVal.getZExtValue();
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

  unsigned NumArgs = A.getNumArgs();

  if (NumArgs > 1) {
    S.Diag(A.getLoc(), diag::err_attribute_too_many_arguments) << A << 1;
    return nullptr;
  }

  Expr *E = NumArgs ? A.getArgAsExpr(0) : nullptr;
  if (A.getParsedKind() == ParsedAttr::AT_OpenCLUnrollHint)
    return S.BuildOpenCLLoopUnrollHintAttr(A, E);
  else if (A.getParsedKind() == ParsedAttr::AT_LoopUnrollHint)
    return S.BuildLoopUnrollHintAttr(A, E);

  return nullptr;
}

static Attr *ProcessStmtAttribute(Sema &S, Stmt *St, const ParsedAttr &A,
                                  SourceRange Range) {
  switch (A.getKind()) {
  case ParsedAttr::UnknownAttribute:
    S.Diag(A.getLoc(), A.isDeclspecAttribute()
                           ? (unsigned)diag::warn_unhandled_ms_attribute_ignored
                           : (unsigned)diag::warn_unknown_attribute_ignored)
        << A;
    return nullptr;
  case ParsedAttr::AT_FallThrough:
    return handleFallThroughAttr(S, St, A, Range);
  case ParsedAttr::AT_LoopHint:
    return handleLoopHintAttr(S, St, A, Range);
  case ParsedAttr::AT_SYCLIntelFPGAIVDep:
    return handleIntelFPGAIVDepAttr(S, A);
  case ParsedAttr::AT_SYCLIntelFPGAII:
    return handleIntelFPGALoopAttr<SYCLIntelFPGAIIAttr>(S, A);
  case ParsedAttr::AT_SYCLIntelFPGAMaxConcurrency:
    return handleIntelFPGALoopAttr<SYCLIntelFPGAMaxConcurrencyAttr>(S, A);
  case ParsedAttr::AT_SYCLIntelFPGALoopCoalesce:
    return handleIntelFPGALoopAttr<SYCLIntelFPGALoopCoalesceAttr>(S, A);
  case ParsedAttr::AT_SYCLIntelFPGADisableLoopPipelining:
    return handleIntelFPGALoopAttr<SYCLIntelFPGADisableLoopPipeliningAttr>(S,
                                                                           A);
  case ParsedAttr::AT_SYCLIntelFPGAMaxInterleaving:
    return handleIntelFPGALoopAttr<SYCLIntelFPGAMaxInterleavingAttr>(S, A);
  case ParsedAttr::AT_SYCLIntelFPGASpeculatedIterations:
    return handleIntelFPGALoopAttr<SYCLIntelFPGASpeculatedIterationsAttr>(S, A);
  case ParsedAttr::AT_OpenCLUnrollHint:
  case ParsedAttr::AT_LoopUnrollHint:
    return handleLoopUnrollHint(S, St, A, Range);
  case ParsedAttr::AT_Suppress:
    return handleSuppressAttr(S, St, A, Range);
  case ParsedAttr::AT_NoMerge:
    return handleNoMergeAttr(S, St, A, Range);
  default:
    // if we're here, then we parsed a known attribute, but didn't recognize
    // it as a statement attribute => it is declaration attribute
    S.Diag(A.getRange().getBegin(), diag::err_decl_attribute_invalid_on_stmt)
        << A << St->getBeginLoc();
    return nullptr;
  }
}

StmtResult Sema::ProcessStmtAttributes(Stmt *S,
                                       const ParsedAttributesView &AttrList,
                                       SourceRange Range) {
  SmallVector<const Attr*, 8> Attrs;
  for (const ParsedAttr &AL : AttrList) {
    if (Attr *a = ProcessStmtAttribute(*this, S, AL, Range))
      Attrs.push_back(a);
  }

  CheckForIncompatibleAttributes(*this, Attrs);
  CheckForIncompatibleSYCLLoopAttributes(*this, Attrs, Range);
  CheckForIncompatibleUnrollHintAttributes(*this, Attrs, Range);

  if (Attrs.empty())
    return S;

  return ActOnAttributedStmt(Range.getBegin(), Attrs, S);
}
bool Sema::CheckRebuiltAttributedStmtAttributes(ArrayRef<const Attr *> Attrs) {
  CheckRedundantSYCLIntelFPGAIVDepAttrs(*this, Attrs);
  return false;
}
