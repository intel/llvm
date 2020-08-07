//===- SemaSYCL.cpp - Semantic Analysis for SYCL constructs ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This implements Semantic Analysis for SYCL constructs.
//===----------------------------------------------------------------------===//

#include "TreeTransform.h"
#include "clang/AST/AST.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/QualTypeNames.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Analysis/CallGraph.h"
#include "clang/Basic/Attributes.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <functional>
#include <initializer_list>

using namespace clang;
using namespace std::placeholders;

using KernelParamKind = SYCLIntegrationHeader::kernel_param_kind_t;

enum target {
  global_buffer = 2014,
  constant_buffer,
  local,
  image,
  host_buffer,
  host_image,
  image_array
};

using ParamDesc = std::tuple<QualType, IdentifierInfo *, TypeSourceInfo *>;

enum KernelInvocationKind {
  InvokeUnknown,
  InvokeSingleTask,
  InvokeParallelFor,
  InvokeParallelForWorkGroup
};

const static std::string InitMethodName = "__init";
const static std::string FinalizeMethodName = "__finalize";

namespace {

/// Various utilities.
class Util {
public:
  using DeclContextDesc = std::pair<clang::Decl::Kind, StringRef>;

  /// Checks whether given clang type is a full specialization of the SYCL
  /// accessor class.
  static bool isSyclAccessorType(const QualType &Ty);

  /// Checks whether given clang type is a full specialization of the SYCL
  /// sampler class.
  static bool isSyclSamplerType(const QualType &Ty);

  /// Checks whether given clang type is a full specialization of the SYCL
  /// stream class.
  static bool isSyclStreamType(const QualType &Ty);

  /// Checks whether given clang type is a full specialization of the SYCL
  /// half class.
  static bool isSyclHalfType(const QualType &Ty);

  /// Checks whether given clang type is a full specialization of the SYCL
  /// property_list class.
  static bool isPropertyListType(const QualType &Ty);

  /// Checks whether given clang type is a full specialization of the SYCL
  /// buffer_location class.
  static bool isSyclBufferLocationType(const QualType &Ty);

  /// Checks whether given clang type is a standard SYCL API class with given
  /// name.
  /// \param Ty    the clang type being checked
  /// \param Name  the class name checked against
  /// \param Tmpl  whether the class is template instantiation or simple record
  static bool isSyclType(const QualType &Ty, StringRef Name, bool Tmpl = false);

  /// Checks whether given clang type is a full specialization of the SYCL
  /// specialization constant class.
  static bool isSyclSpecConstantType(const QualType &Ty);

  /// Checks whether given clang type is declared in the given hierarchy of
  /// declaration contexts.
  /// \param Ty         the clang type being checked
  /// \param Scopes     the declaration scopes leading from the type to the
  ///     translation unit (excluding the latter)
  static bool matchQualifiedTypeName(const QualType &Ty,
                                     ArrayRef<Util::DeclContextDesc> Scopes);
};

} // anonymous namespace

// This information is from Section 4.13 of the SYCL spec
// https://www.khronos.org/registry/SYCL/specs/sycl-1.2.1.pdf
// This function returns false if the math lib function
// corresponding to the input builtin is not supported
// for SYCL
static bool IsSyclMathFunc(unsigned BuiltinID) {
  switch (BuiltinID) {
  case Builtin::BIlround:
  case Builtin::BI__builtin_lround:
  case Builtin::BIceill:
  case Builtin::BI__builtin_ceill:
  case Builtin::BIcopysignl:
  case Builtin::BI__builtin_copysignl:
  case Builtin::BIcosl:
  case Builtin::BI__builtin_cosl:
  case Builtin::BIexpl:
  case Builtin::BI__builtin_expl:
  case Builtin::BIexp2l:
  case Builtin::BI__builtin_exp2l:
  case Builtin::BIfabsl:
  case Builtin::BI__builtin_fabsl:
  case Builtin::BIfloorl:
  case Builtin::BI__builtin_floorl:
  case Builtin::BIfmal:
  case Builtin::BI__builtin_fmal:
  case Builtin::BIfmaxl:
  case Builtin::BI__builtin_fmaxl:
  case Builtin::BIfminl:
  case Builtin::BI__builtin_fminl:
  case Builtin::BIfmodl:
  case Builtin::BI__builtin_fmodl:
  case Builtin::BIlogl:
  case Builtin::BI__builtin_logl:
  case Builtin::BIlog10l:
  case Builtin::BI__builtin_log10l:
  case Builtin::BIlog2l:
  case Builtin::BI__builtin_log2l:
  case Builtin::BIpowl:
  case Builtin::BI__builtin_powl:
  case Builtin::BIrintl:
  case Builtin::BI__builtin_rintl:
  case Builtin::BIroundl:
  case Builtin::BI__builtin_roundl:
  case Builtin::BIsinl:
  case Builtin::BI__builtin_sinl:
  case Builtin::BIsqrtl:
  case Builtin::BI__builtin_sqrtl:
  case Builtin::BItruncl:
  case Builtin::BI__builtin_truncl:
  case Builtin::BIlroundl:
  case Builtin::BI__builtin_lroundl:
  case Builtin::BIcopysign:
  case Builtin::BI__builtin_copysign:
  case Builtin::BIfloor:
  case Builtin::BI__builtin_floor:
  case Builtin::BIfmax:
  case Builtin::BI__builtin_fmax:
  case Builtin::BIfmin:
  case Builtin::BI__builtin_fmin:
  case Builtin::BInearbyint:
  case Builtin::BI__builtin_nearbyint:
  case Builtin::BIrint:
  case Builtin::BI__builtin_rint:
  case Builtin::BIround:
  case Builtin::BI__builtin_round:
  case Builtin::BItrunc:
  case Builtin::BI__builtin_trunc:
  case Builtin::BIcopysignf:
  case Builtin::BI__builtin_copysignf:
  case Builtin::BIfloorf:
  case Builtin::BI__builtin_floorf:
  case Builtin::BIfmaxf:
  case Builtin::BI__builtin_fmaxf:
  case Builtin::BIfminf:
  case Builtin::BI__builtin_fminf:
  case Builtin::BInearbyintf:
  case Builtin::BI__builtin_nearbyintf:
  case Builtin::BIrintf:
  case Builtin::BI__builtin_rintf:
  case Builtin::BIroundf:
  case Builtin::BI__builtin_roundf:
  case Builtin::BItruncf:
  case Builtin::BI__builtin_truncf:
  case Builtin::BIlroundf:
  case Builtin::BI__builtin_lroundf:
  case Builtin::BI__builtin_fpclassify:
  case Builtin::BI__builtin_isfinite:
  case Builtin::BI__builtin_isinf:
  case Builtin::BI__builtin_isnormal:
    return false;
  default:
    break;
  }
  return true;
}

bool Sema::isKnownGoodSYCLDecl(const Decl *D) {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    const IdentifierInfo *II = FD->getIdentifier();
    const DeclContext *DC = FD->getDeclContext();
    if (II && II->isStr("__spirv_ocl_printf") &&
        !FD->isDefined() &&
        FD->getLanguageLinkage() == CXXLanguageLinkage &&
        DC->getEnclosingNamespaceContext()->isTranslationUnit())
      return true;
  }
  return false;
}

static bool isZeroSizedArray(QualType Ty) {
  if (const auto *CATy = dyn_cast<ConstantArrayType>(Ty))
    return CATy->getSize() == 0;
  return false;
}

static void checkSYCLType(Sema &S, QualType Ty, SourceRange Loc,
                          llvm::DenseSet<QualType> Visited,
                          SourceRange UsedAtLoc = SourceRange()) {
  // Not all variable types are supported inside SYCL kernels,
  // for example the quad type __float128 will cause errors in the
  // SPIR-V translation phase.
  // Here we check any potentially unsupported declaration and issue
  // a deferred diagnostic, which will be emitted iff the declaration
  // is discovered to reside in kernel code.
  // The optional UsedAtLoc param is used when the SYCL usage is at a
  // different location than the variable declaration and we need to
  // inform the user of both, e.g. struct member usage vs declaration.

  bool Emitting = false;

  //--- check types ---

  // zero length arrays
  if (isZeroSizedArray(Ty)) {
    S.SYCLDiagIfDeviceCode(Loc.getBegin(), diag::err_typecheck_zero_array_size);
    Emitting = true;
  }

  // variable length arrays
  if (Ty->isVariableArrayType()) {
    S.SYCLDiagIfDeviceCode(Loc.getBegin(), diag::err_vla_unsupported);
    Emitting = true;
  }

  // Sub-reference array or pointer, then proceed with that type.
  while (Ty->isAnyPointerType() || Ty->isArrayType())
    Ty = QualType{Ty->getPointeeOrArrayElementType(), 0};

  // __int128, __int128_t, __uint128_t, long double, __float128
  if (Ty->isSpecificBuiltinType(BuiltinType::Int128) ||
      Ty->isSpecificBuiltinType(BuiltinType::UInt128) ||
      Ty->isSpecificBuiltinType(BuiltinType::LongDouble) ||
      (Ty->isSpecificBuiltinType(BuiltinType::Float128) &&
       !S.Context.getTargetInfo().hasFloat128Type())) {
    S.SYCLDiagIfDeviceCode(Loc.getBegin(), diag::err_type_unsupported)
        << Ty.getUnqualifiedType().getCanonicalType();
    Emitting = true;
  }

  if (Emitting && UsedAtLoc.isValid())
    S.SYCLDiagIfDeviceCode(UsedAtLoc.getBegin(), diag::note_used_here);

  //--- now recurse ---
  // Pointers complicate recursion. Add this type to Visited.
  // If already there, bail out.
  if (!Visited.insert(Ty).second)
    return;

  if (const auto *ATy = dyn_cast<AttributedType>(Ty))
    return checkSYCLType(S, ATy->getModifiedType(), Loc, Visited);

  if (const auto *RD = Ty->getAsRecordDecl()) {
    for (const auto &Field : RD->fields())
      checkSYCLType(S, Field->getType(), Field->getSourceRange(), Visited, Loc);
  } else if (const auto *FPTy = dyn_cast<FunctionProtoType>(Ty)) {
    for (const auto &ParamTy : FPTy->param_types())
      checkSYCLType(S, ParamTy, Loc, Visited);
    checkSYCLType(S, FPTy->getReturnType(), Loc, Visited);
  }
}

void Sema::checkSYCLDeviceVarDecl(VarDecl *Var) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL compilation");
  QualType Ty = Var->getType();
  SourceRange Loc = Var->getLocation();
  llvm::DenseSet<QualType> Visited;

  checkSYCLType(*this, Ty, Loc, Visited);
}

// Tests whether given function is a lambda function or '()' operator used as
// SYCL kernel body function (e.g. in parallel_for).
// NOTE: This is incomplete implemenation. See TODO in the FE TODO list for the
// ESIMD extension.
static bool isSYCLKernelBodyFunction(FunctionDecl *FD) {
  return FD->getOverloadedOperator() == OO_Call;
}

// Helper function to report conflicting function attributes.
// F - the function, A1 - function attribute, A2 - the attribute it conflicts
// with.
static void reportConflictingAttrs(Sema &S, FunctionDecl *F, const Attr *A1,
                                   const Attr *A2) {
  S.Diag(F->getLocation(), diag::err_conflicting_sycl_kernel_attributes);
  S.Diag(A1->getLocation(), diag::note_conflicting_attribute);
  S.Diag(A2->getLocation(), diag::note_conflicting_attribute);
  F->setInvalidDecl();
}

/// Returns the signed constant integer value represented by given expression
static int64_t getIntExprValue(const Expr *E, ASTContext &Ctx) {
  return E->getIntegerConstantExpr(Ctx)->getSExtValue();
}

class MarkDeviceFunction : public RecursiveASTVisitor<MarkDeviceFunction> {
  // Used to keep track of the constexpr depth, so we know whether to skip
  // diagnostics.
  unsigned ConstexprDepth = 0;
  struct ConstexprDepthRAII {
    MarkDeviceFunction &MDF;
    bool Increment;

    ConstexprDepthRAII(MarkDeviceFunction &MDF, bool Increment = true)
        : MDF(MDF), Increment(Increment) {
      if (Increment)
        ++MDF.ConstexprDepth;
    }
    ~ConstexprDepthRAII() {
      if (Increment)
        --MDF.ConstexprDepth;
    }
  };

public:
  MarkDeviceFunction(Sema &S)
      : RecursiveASTVisitor<MarkDeviceFunction>(), SemaRef(S) {}

  bool VisitCallExpr(CallExpr *e) {
    if (FunctionDecl *Callee = e->getDirectCallee()) {
      Callee = Callee->getCanonicalDecl();
      assert(Callee && "Device function canonical decl must be available");

      // Remember that all SYCL kernel functions have deferred
      // instantiation as template functions. It means that
      // all functions used by kernel have already been parsed and have
      // definitions.
      if (RecursiveSet.count(Callee) && !ConstexprDepth) {
        SemaRef.Diag(e->getExprLoc(), diag::warn_sycl_restrict_recursion);
        SemaRef.Diag(Callee->getSourceRange().getBegin(),
                     diag::note_sycl_recursive_function_declared_here)
            << Sema::KernelCallRecursiveFunction;
      }

      if (const CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Callee))
        if (Method->isVirtual())
          SemaRef.Diag(e->getExprLoc(), diag::err_sycl_restrict)
              << Sema::KernelCallVirtualFunction;

      if (auto const *FD = dyn_cast<FunctionDecl>(Callee)) {
        // FIXME: We need check all target specified attributes for error if
        // that function with attribute can not be called from sycl kernel.  The
        // info is in ParsedAttr. We don't have to map from Attr to ParsedAttr
        // currently. Erich is currently working on that in LLVM, once that is
        // committed we need to change this".
        if (FD->hasAttr<DLLImportAttr>()) {
          SemaRef.Diag(e->getExprLoc(), diag::err_sycl_restrict)
              << Sema::KernelCallDllimportFunction;
          SemaRef.Diag(FD->getLocation(), diag::note_callee_decl) << FD;
        }
      }
      // Specifically check if the math library function corresponding to this
      // builtin is supported for SYCL
      unsigned BuiltinID = Callee->getBuiltinID();
      if (BuiltinID && !IsSyclMathFunc(BuiltinID)) {
        StringRef Name = SemaRef.Context.BuiltinInfo.getName(BuiltinID);
        SemaRef.Diag(e->getExprLoc(), diag::err_builtin_target_unsupported)
            << Name << "SYCL device";
      }
    } else if (!SemaRef.getLangOpts().SYCLAllowFuncPtr &&
               !e->isTypeDependent() &&
               !isa<CXXPseudoDestructorExpr>(e->getCallee()))
      SemaRef.Diag(e->getExprLoc(), diag::err_sycl_restrict)
          << Sema::KernelCallFunctionPointer;
    return true;
  }

  bool VisitCXXTypeidExpr(CXXTypeidExpr *E) {
    SemaRef.Diag(E->getExprLoc(), diag::err_sycl_restrict) << Sema::KernelRTTI;
    return true;
  }

  bool VisitCXXDynamicCastExpr(const CXXDynamicCastExpr *E) {
    SemaRef.Diag(E->getExprLoc(), diag::err_sycl_restrict) << Sema::KernelRTTI;
    return true;
  }

  // Skip checking rules on variables initialized during constant evaluation.
  bool TraverseVarDecl(VarDecl *VD) {
    ConstexprDepthRAII R(*this, VD->isConstexpr());
    return RecursiveASTVisitor::TraverseVarDecl(VD);
  }

  // Skip checking rules on template arguments, since these are constant
  // expressions.
  bool TraverseTemplateArgumentLoc(const TemplateArgumentLoc &ArgLoc) {
    ConstexprDepthRAII R(*this);
    return RecursiveASTVisitor::TraverseTemplateArgumentLoc(ArgLoc);
  }

  // Skip checking the static assert, both components are required to be
  // constant expressions.
  bool TraverseStaticAssertDecl(StaticAssertDecl *D) {
    ConstexprDepthRAII R(*this);
    return RecursiveASTVisitor::TraverseStaticAssertDecl(D);
  }

  // Make sure we skip the condition of the case, since that is a constant
  // expression.
  bool TraverseCaseStmt(CaseStmt *S) {
    {
      ConstexprDepthRAII R(*this);
      if (!TraverseStmt(S->getLHS()))
        return false;
      if (!TraverseStmt(S->getRHS()))
        return false;
    }
    return TraverseStmt(S->getSubStmt());
  }

  // Skip checking the size expr, since a constant array type loc's size expr is
  // a constant expression.
  bool TraverseConstantArrayTypeLoc(const ConstantArrayTypeLoc &ArrLoc) {
    if (!TraverseTypeLoc(ArrLoc.getElementLoc()))
      return false;

    ConstexprDepthRAII R(*this);
    return TraverseStmt(ArrLoc.getSizeExpr());
  }

  // The call graph for this translation unit.
  CallGraph SYCLCG;
  // The set of functions called by a kernel function.
  llvm::SmallPtrSet<FunctionDecl *, 10> KernelSet;
  // The set of recursive functions identified while building the
  // kernel set, this is used for error diagnostics.
  llvm::SmallPtrSet<FunctionDecl *, 10> RecursiveSet;
  // Determines whether the function FD is recursive.
  // CalleeNode is a function which is called either directly
  // or indirectly from FD.  If recursion is detected then create
  // diagnostic notes on each function as the callstack is unwound.
  void CollectKernelSet(FunctionDecl *CalleeNode, FunctionDecl *FD,
                        llvm::SmallPtrSet<FunctionDecl *, 10> VisitedSet) {
    // We're currently checking CalleeNode on a different
    // trace through the CallGraph, we avoid infinite recursion
    // by using KernelSet to keep track of this.
    if (!KernelSet.insert(CalleeNode).second)
      // Previously seen, stop recursion.
      return;
    if (CallGraphNode *N = SYCLCG.getNode(CalleeNode)) {
      for (const CallGraphNode *CI : *N) {
        if (FunctionDecl *Callee = dyn_cast<FunctionDecl>(CI->getDecl())) {
          Callee = Callee->getCanonicalDecl();
          if (VisitedSet.count(Callee)) {
            // There's a stack frame to visit this Callee above
            // this invocation. Do not recurse here.
            RecursiveSet.insert(Callee);
            RecursiveSet.insert(CalleeNode);
          } else {
            VisitedSet.insert(Callee);
            CollectKernelSet(Callee, FD, VisitedSet);
            VisitedSet.erase(Callee);
          }
        }
      }
    }
  }

  // Traverses over CallGraph to collect list of attributes applied to
  // functions called by SYCLKernel (either directly and indirectly) which needs
  // to be propagated down to callers and applied to SYCL kernels.
  // For example, reqd_work_group_size, vec_len_hint, reqd_sub_group_size
  // Attributes applied to SYCLKernel are also included
  // Returns the kernel body function found during traversal.
  FunctionDecl *
  CollectPossibleKernelAttributes(FunctionDecl *SYCLKernel,
                                  llvm::SmallPtrSet<Attr *, 4> &Attrs) {
    typedef std::pair<FunctionDecl *, FunctionDecl *> ChildParentPair;
    llvm::SmallPtrSet<FunctionDecl *, 16> Visited;
    llvm::SmallVector<ChildParentPair, 16> WorkList;
    WorkList.push_back({SYCLKernel, nullptr});
    FunctionDecl *KernelBody = nullptr;

    while (!WorkList.empty()) {
      FunctionDecl *FD = WorkList.back().first;
      FunctionDecl *ParentFD = WorkList.back().second;

      if ((ParentFD == SYCLKernel) && isSYCLKernelBodyFunction(FD)) {
        assert(!KernelBody && "inconsistent call graph - only one kernel body "
                              "function can be called");
        KernelBody = FD;
      }
      WorkList.pop_back();
      if (!Visited.insert(FD).second)
        continue; // We've already seen this Decl

      if (auto *A = FD->getAttr<IntelReqdSubGroupSizeAttr>())
        Attrs.insert(A);

      if (auto *A = FD->getAttr<ReqdWorkGroupSizeAttr>())
        Attrs.insert(A);

      if (auto *A = FD->getAttr<SYCLIntelKernelArgsRestrictAttr>())
        Attrs.insert(A);

      if (auto *A = FD->getAttr<SYCLIntelNumSimdWorkItemsAttr>())
        Attrs.insert(A);

      if (auto *A = FD->getAttr<SYCLIntelMaxWorkGroupSizeAttr>())
        Attrs.insert(A);

      if (auto *A = FD->getAttr<SYCLIntelMaxGlobalWorkDimAttr>())
        Attrs.insert(A);

      if (auto *A = FD->getAttr<SYCLIntelNoGlobalWorkOffsetAttr>())
        Attrs.insert(A);

      if (auto *A = FD->getAttr<SYCLSimdAttr>())
        Attrs.insert(A);
      // Propagate the explicit SIMD attribute through call graph - it is used
      // to distinguish ESIMD code in ESIMD LLVM passes.
      if (KernelBody && KernelBody->hasAttr<SYCLSimdAttr>() &&
          (KernelBody != FD) && !FD->hasAttr<SYCLSimdAttr>())
        FD->addAttr(SYCLSimdAttr::CreateImplicit(SemaRef.getASTContext()));

      // TODO: vec_len_hint should be handled here

      CallGraphNode *N = SYCLCG.getNode(FD);
      if (!N)
        continue;

      for (const CallGraphNode *CI : *N) {
        if (auto *Callee = dyn_cast<FunctionDecl>(CI->getDecl())) {
          Callee = Callee->getCanonicalDecl();
          if (!Visited.count(Callee))
            WorkList.push_back({Callee, FD});
        }
      }
    }
    return KernelBody;
  }

private:
  Sema &SemaRef;
};

class KernelBodyTransform : public TreeTransform<KernelBodyTransform> {
public:
  KernelBodyTransform(std::pair<DeclaratorDecl *, DeclaratorDecl *> &MPair,
                      Sema &S)
      : TreeTransform<KernelBodyTransform>(S), MappingPair(MPair), SemaRef(S) {}
  bool AlwaysRebuild() { return true; }

  ExprResult TransformDeclRefExpr(DeclRefExpr *DRE) {
    auto Ref = dyn_cast<DeclaratorDecl>(DRE->getDecl());
    if (Ref && Ref == MappingPair.first) {
      auto NewDecl = MappingPair.second;
      return DeclRefExpr::Create(
          SemaRef.getASTContext(), DRE->getQualifierLoc(),
          DRE->getTemplateKeywordLoc(), NewDecl, false,
          DeclarationNameInfo(DRE->getNameInfo().getName(), SourceLocation(),
                              DRE->getNameInfo().getInfo()),
          NewDecl->getType(), DRE->getValueKind());
    }
    return DRE;
  }

  StmtResult RebuildCompoundStmt(SourceLocation LBraceLoc,
                                 MultiStmtArg Statements,
                                 SourceLocation RBraceLoc, bool IsStmtExpr) {
    // Build a new compound statement but clear the source locations.
    return getSema().ActOnCompoundStmt(SourceLocation(), SourceLocation(),
                                       Statements, IsStmtExpr);
  }

private:
  std::pair<DeclaratorDecl *, DeclaratorDecl *> MappingPair;
  Sema &SemaRef;
};

// Searches for a call to PFWG lambda function and captures it.
class FindPFWGLambdaFnVisitor
    : public RecursiveASTVisitor<FindPFWGLambdaFnVisitor> {
public:
  // LambdaObjTy - lambda type of the PFWG lambda object
  FindPFWGLambdaFnVisitor(const CXXRecordDecl *LambdaObjTy)
      : LambdaFn(nullptr), LambdaObjTy(LambdaObjTy) {}

  bool VisitCallExpr(CallExpr *Call) {
    auto *M = dyn_cast<CXXMethodDecl>(Call->getDirectCallee());
    if (!M || (M->getOverloadedOperator() != OO_Call))
      return true;
    const int NumPFWGLambdaArgs = 2; // group and lambda obj
    if (Call->getNumArgs() != NumPFWGLambdaArgs)
      return true;
    if (!Util::isSyclType(Call->getArg(1)->getType(), "group", true /*Tmpl*/))
      return true;
    if (Call->getArg(0)->getType()->getAsCXXRecordDecl() != LambdaObjTy)
      return true;
    LambdaFn = M; // call to PFWG lambda found - record the lambda
    return false; // ... and stop searching
  }

  // Returns the captured lambda function or nullptr;
  CXXMethodDecl *getLambdaFn() const { return LambdaFn; }

private:
  CXXMethodDecl *LambdaFn;
  const CXXRecordDecl *LambdaObjTy;
};

class MarkWIScopeFnVisitor : public RecursiveASTVisitor<MarkWIScopeFnVisitor> {
public:
  MarkWIScopeFnVisitor(ASTContext &Ctx) : Ctx(Ctx) {}

  bool VisitCXXMemberCallExpr(CXXMemberCallExpr *Call) {
    FunctionDecl *Callee = Call->getDirectCallee();
    if (!Callee)
      // not a direct call - continue search
      return true;
    QualType Ty = Ctx.getRecordType(Call->getRecordDecl());
    if (!Util::isSyclType(Ty, "group", true /*Tmpl*/))
      // not a member of cl::sycl::group - continue search
      return true;
    auto Name = Callee->getName();
    if (((Name != "parallel_for_work_item") && (Name != "wait_for")) ||
        Callee->hasAttr<SYCLScopeAttr>())
      return true;
    // it is a call to cl::sycl::group::parallel_for_work_item/wait_for -
    // mark the callee
    Callee->addAttr(
        SYCLScopeAttr::CreateImplicit(Ctx, SYCLScopeAttr::Level::WorkItem));
    // continue search as there can be other PFWI or wait_for calls
    return true;
  }

private:
  ASTContext &Ctx;
};

static bool isSYCLPrivateMemoryVar(VarDecl *VD) {
  return Util::isSyclType(VD->getType(), "private_memory", true /*Tmpl*/);
}

static void addScopeAttrToLocalVars(CXXMethodDecl &F) {
  for (Decl *D : F.decls()) {
    VarDecl *VD = dyn_cast<VarDecl>(D);

    if (!VD || isa<ParmVarDecl>(VD) ||
        VD->getStorageDuration() != StorageDuration::SD_Automatic)
      continue;
    // Local variables of private_memory type in the WG scope still have WI
    // scope, all the rest - WG scope. Simple logic
    // "if no scope than it is WG scope" won't work, because compiler may add
    // locals not declared in user code (lambda object parameter, byval
    // arguments) which will result in alloca w/o any attribute, so need WI
    // scope too.
    SYCLScopeAttr::Level L = isSYCLPrivateMemoryVar(VD)
                                 ? SYCLScopeAttr::Level::WorkItem
                                 : SYCLScopeAttr::Level::WorkGroup;
    VD->addAttr(SYCLScopeAttr::CreateImplicit(F.getASTContext(), L));
  }
}

/// Return method by name
static CXXMethodDecl *getMethodByName(const CXXRecordDecl *CRD,
                                      const std::string &MethodName) {
  CXXMethodDecl *Method;
  auto It = std::find_if(CRD->methods().begin(), CRD->methods().end(),
                         [&MethodName](const CXXMethodDecl *Method) {
                           return Method->getNameAsString() == MethodName;
                         });
  Method = (It != CRD->methods().end()) ? *It : nullptr;
  return Method;
}

static KernelInvocationKind
getKernelInvocationKind(FunctionDecl *KernelCallerFunc) {
  return llvm::StringSwitch<KernelInvocationKind>(KernelCallerFunc->getName())
      .Case("kernel_single_task", InvokeSingleTask)
      .Case("kernel_parallel_for", InvokeParallelFor)
      .Case("kernel_parallel_for_work_group", InvokeParallelForWorkGroup)
      .Default(InvokeUnknown);
}

static const CXXRecordDecl *getKernelObjectType(FunctionDecl *Caller) {
  return (*Caller->param_begin())->getType()->getAsCXXRecordDecl();
}

/// Creates a kernel parameter descriptor
/// \param Src  field declaration to construct name from
/// \param Ty   the desired parameter type
/// \return     the constructed descriptor
static ParamDesc makeParamDesc(const FieldDecl *Src, QualType Ty) {
  ASTContext &Ctx = Src->getASTContext();
  std::string Name = (Twine("_arg_") + Src->getName()).str();
  return std::make_tuple(Ty, &Ctx.Idents.get(Name),
                         Ctx.getTrivialTypeSourceInfo(Ty));
}

static ParamDesc makeParamDesc(ASTContext &Ctx, const CXXBaseSpecifier &Src,
                               QualType Ty) {
  // TODO: There is no name for the base available, but duplicate names are
  // seemingly already possible, so we'll give them all the same name for now.
  // This only happens with the accessor types.
  std::string Name = "_arg__base";
  return std::make_tuple(Ty, &Ctx.Idents.get(Name),
                         Ctx.getTrivialTypeSourceInfo(Ty));
}

/// \return the target of given SYCL accessor type
static target getAccessTarget(const ClassTemplateSpecializationDecl *AccTy) {
  return static_cast<target>(
      AccTy->getTemplateArgs()[3].getAsIntegral().getExtValue());
}

// The first template argument to the kernel caller function is used to identify
// the kernel itself.
static QualType calculateKernelNameType(ASTContext &Ctx,
                                        FunctionDecl *KernelCallerFunc) {
  const TemplateArgumentList *TAL =
      KernelCallerFunc->getTemplateSpecializationArgs();
  assert(TAL && "No template argument info");
  return TAL->get(0).getAsType().getCanonicalType();
}

// Gets a name for the OpenCL kernel function, calculated from the first
// template argument of the kernel caller function.
static std::pair<std::string, std::string>
constructKernelName(Sema &S, FunctionDecl *KernelCallerFunc,
                    MangleContext &MC) {
  QualType KernelNameType =
      calculateKernelNameType(S.getASTContext(), KernelCallerFunc);

  SmallString<256> Result;
  llvm::raw_svector_ostream Out(Result);

  MC.mangleTypeName(KernelNameType, Out);

  return {std::string(Out.str()),
          PredefinedExpr::ComputeName(S.getASTContext(),
                                      PredefinedExpr::UniqueStableNameType,
                                      KernelNameType)};
}

// anonymous namespace so these don't get linkage.
namespace {

template <typename T> struct bind_param { using type = T; };

template <> struct bind_param<CXXBaseSpecifier &> {
  using type = const CXXBaseSpecifier &;
};

template <> struct bind_param<FieldDecl *&> { using type = FieldDecl *; };

template <> struct bind_param<FieldDecl *const &> { using type = FieldDecl *; };

template <typename T> using bind_param_t = typename bind_param<T>::type;

class KernelObjVisitor {
  Sema &SemaRef;

public:
  KernelObjVisitor(Sema &S) : SemaRef(S) {}

  // These enable handler execution only when previous handlers succeed.
  template <typename... Tn>
  bool handleField(FieldDecl *FD, QualType FDTy, Tn &&... tn) {
    bool result = true;
    std::initializer_list<int>{(result = result && tn(FD, FDTy), 0)...};
    return result;
  }
  template <typename... Tn>
  bool handleField(const CXXBaseSpecifier &BD, QualType BDTy, Tn &&... tn) {
    bool result = true;
    std::initializer_list<int>{(result = result && tn(BD, BDTy), 0)...};
    return result;
  }

// This definition using std::bind is necessary because of a gcc 7.x bug.
#define KF_FOR_EACH(FUNC, Item, Qt)                                            \
  handleField(                                                                 \
      Item, Qt,                                                                \
      std::bind(static_cast<bool (std::decay_t<decltype(handlers)>::*)(        \
                    bind_param_t<decltype(Item)>, QualType)>(                  \
                    &std::decay_t<decltype(handlers)>::FUNC),                  \
                std::ref(handlers), _1, _2)...)

  // The following simpler definition works with gcc 8.x and later.
  //#define KF_FOR_EACH(FUNC) \
//  handleField(Field, FieldTy, ([&](FieldDecl *FD, QualType FDTy) { \
//                return handlers.f(FD, FDTy); \
//              })...)

  // Implements the 'for-each-visitor'  pattern.
  template <typename... Handlers>
  void VisitElement(CXXRecordDecl *Owner, FieldDecl *ArrayField,
                    QualType ElementTy, Handlers &... handlers) {
    if (Util::isSyclAccessorType(ElementTy))
      KF_FOR_EACH(handleSyclAccessorType, ArrayField, ElementTy);
    else if (Util::isSyclStreamType(ElementTy))
      KF_FOR_EACH(handleSyclStreamType, ArrayField, ElementTy);
    else if (Util::isSyclSamplerType(ElementTy))
      KF_FOR_EACH(handleSyclSamplerType, ArrayField, ElementTy);
    else if (Util::isSyclHalfType(ElementTy))
      KF_FOR_EACH(handleSyclHalfType, ArrayField, ElementTy);
    else if (ElementTy->isStructureOrClassType())
      VisitRecord(Owner, ArrayField, ElementTy->getAsCXXRecordDecl(),
                  handlers...);
    else if (ElementTy->isUnionType())
      // TODO: This check is still necessary I think?! Array seems to handle
      // this differently (see above) for structs I think.
      //if (KF_FOR_EACH(handleUnionType, Field, FieldTy)) {
      VisitUnion(Owner, ArrayField, ElementTy->getAsCXXRecordDecl(),
                  handlers...);
      //}
    else if (ElementTy->isArrayType())
      VisitArrayElements(ArrayField, ElementTy, handlers...);
    else if (ElementTy->isScalarType())
      KF_FOR_EACH(handleScalarType, ArrayField, ElementTy);
  }

  template <typename... Handlers>
  void VisitArrayElements(FieldDecl *FD, QualType FieldTy,
                          Handlers &... handlers) {
    const ConstantArrayType *CAT =
        SemaRef.getASTContext().getAsConstantArrayType(FieldTy);
    assert(CAT && "Should only be called on constant-size array.");
    QualType ET = CAT->getElementType();
    int64_t ElemCount = CAT->getSize().getSExtValue();
    std::initializer_list<int>{(handlers.enterArray(), 0)...};
    for (int64_t Count = 0; Count < ElemCount; Count++) {
      VisitElement(nullptr, FD, ET, handlers...);
      (void)std::initializer_list<int>{(handlers.nextElement(ET), 0)...};
    }
    (void)std::initializer_list<int>{
        (handlers.leaveArray(FD, ET, ElemCount), 0)...};
  }

  template <typename ParentTy, typename... Handlers>
  void VisitRecord(const CXXRecordDecl *Owner, ParentTy &Parent,
                   const CXXRecordDecl *Wrapper, Handlers &... handlers);

  // Base case, only calls these when filtered.
  template <typename... FilteredHandlers, typename ParentTy>
  void VisitUnion(const CXXRecordDecl *Owner, ParentTy &Parent,
                  const CXXRecordDecl *Wrapper,
                  FilteredHandlers &... handlers) {
    (void)std::initializer_list<int>{
        (handlers.enterUnion(Owner, Parent), 0)...};
    VisitRecordHelper(Wrapper, Wrapper->fields(), handlers...);
    (void)std::initializer_list<int>{
        (handlers.leaveUnion(Owner, Parent), 0)...};
  }


  template <typename... FilteredHandlers, typename ParentTy,
            typename CurHandler, typename... Handlers>
  std::enable_if_t<!CurHandler::VisitUnionBody>
  VisitUnion(const CXXRecordDecl *Owner, ParentTy &Parent,
                  const CXXRecordDecl *Wrapper,
                  FilteredHandlers &... filtered_handlers,
                  CurHandler &cur_handler, Handlers &... handlers) {
    VisitUnion<FilteredHandlers...>(
        Owner, Parent, Wrapper, filtered_handlers..., handlers...);
  }

  template <typename... FilteredHandlers, typename ParentTy,
            typename CurHandler, typename... Handlers>
  std::enable_if_t<CurHandler::VisitUnionBody>
  VisitUnion(const CXXRecordDecl *Owner, ParentTy &Parent,
                  const CXXRecordDecl *Wrapper,
                  FilteredHandlers &... filtered_handlers,
                  CurHandler &cur_handler, Handlers &... handlers) {
    VisitUnion<FilteredHandlers..., CurHandler>(
        Owner, Parent, Wrapper, filtered_handlers..., cur_handler, handlers...);
  }

  template <typename... Handlers>
  void VisitRecordHelper(const CXXRecordDecl *Owner,
                         clang::CXXRecordDecl::base_class_const_range Range,
                         Handlers &... handlers) {
    for (const auto &Base : Range) {
      (void)std::initializer_list<int>{
          (handlers.enterField(Owner, Base), 0)...};
      QualType BaseTy = Base.getType();
      // Handle accessor class as base
      if (Util::isSyclAccessorType(BaseTy)) {
        (void)std::initializer_list<int>{
            (handlers.handleSyclAccessorType(Base, BaseTy), 0)...};
      } else if (Util::isSyclStreamType(BaseTy)) {
        // Handle stream class as base
        (void)std::initializer_list<int>{
            (handlers.handleSyclStreamType(Base, BaseTy), 0)...};
      } else
        // For all other bases, visit the record
        VisitRecord(Owner, Base, BaseTy->getAsCXXRecordDecl(), handlers...);
      (void)std::initializer_list<int>{
          (handlers.leaveField(Owner, Base), 0)...};
    }
  }

  template <typename... Handlers>
  void VisitRecordHelper(const CXXRecordDecl *Owner,
                         RecordDecl::field_range Range,
                         Handlers &... handlers) {
    VisitRecordFields(Owner, handlers...);
  }

  // FIXME: Can this be refactored/handled some other way?
  template <typename ParentTy, typename... Handlers>
  void VisitStreamRecord(const CXXRecordDecl *Owner, ParentTy &Parent,
                         CXXRecordDecl *Wrapper, Handlers &... handlers) {
    (void)std::initializer_list<int>{
        (handlers.enterStruct(Owner, Parent), 0)...};
    for (const auto &Field : Wrapper->fields()) {
      QualType FieldTy = Field->getType();
      (void)std::initializer_list<int>{
          (handlers.enterField(Wrapper, Field), 0)...};
      // Required to initialize accessors inside streams.
      if (Util::isSyclAccessorType(FieldTy))
        KF_FOR_EACH(handleSyclAccessorType, Field, FieldTy);
      (void)std::initializer_list<int>{
          (handlers.leaveField(Wrapper, Field), 0)...};
    }
    (void)std::initializer_list<int>{
        (handlers.leaveStruct(Owner, Parent), 0)...};
  }

  template <typename... Handlers>
  void VisitRecordBases(const CXXRecordDecl *KernelFunctor,
                        Handlers &... handlers) {
    VisitRecordHelper(KernelFunctor, KernelFunctor->bases(), handlers...);
  }

  // A visitor function that dispatches to functions as defined in
  // SyclKernelFieldHandler for the purposes of kernel generation.
  template <typename... Handlers>
  void VisitRecordFields(const CXXRecordDecl *Owner, Handlers &... handlers) {

    for (const auto Field : Owner->fields()) {
      (void)std::initializer_list<int>{
          (handlers.enterField(Owner, Field), 0)...};
      QualType FieldTy = Field->getType();

      if (Util::isSyclAccessorType(FieldTy))
        KF_FOR_EACH(handleSyclAccessorType, Field, FieldTy);
      else if (Util::isSyclSamplerType(FieldTy))
        KF_FOR_EACH(handleSyclSamplerType, Field, FieldTy);
      else if (Util::isSyclHalfType(FieldTy))
        KF_FOR_EACH(handleSyclHalfType, Field, FieldTy);
      else if (Util::isSyclSpecConstantType(FieldTy))
        KF_FOR_EACH(handleSyclSpecConstantType, Field, FieldTy);
      else if (Util::isSyclStreamType(FieldTy)) {
        CXXRecordDecl *RD = FieldTy->getAsCXXRecordDecl();
        // Handle accessors in stream class.
        VisitStreamRecord(Owner, Field, RD, handlers...);
        KF_FOR_EACH(handleSyclStreamType, Field, FieldTy);
      } else if (FieldTy->isStructureOrClassType()) {
        if (KF_FOR_EACH(handleStructType, Field, FieldTy)) {
          CXXRecordDecl *RD = FieldTy->getAsCXXRecordDecl();
          VisitRecord(Owner, Field, RD, handlers...);
        }
      } else if (FieldTy->isUnionType()) {
        if (KF_FOR_EACH(handleUnionType, Field, FieldTy)) {
          CXXRecordDecl *RD = FieldTy->getAsCXXRecordDecl();
          VisitUnion(Owner, Field, RD, handlers...);
        }
      } else if (FieldTy->isReferenceType())
        KF_FOR_EACH(handleReferenceType, Field, FieldTy);
      else if (FieldTy->isPointerType())
        KF_FOR_EACH(handlePointerType, Field, FieldTy);
      else if (FieldTy->isArrayType()) {
        if (KF_FOR_EACH(handleArrayType, Field, FieldTy))
          VisitArrayElements(Field, FieldTy, handlers...);
      } else if (FieldTy->isScalarType() || FieldTy->isVectorType())
        KF_FOR_EACH(handleScalarType, Field, FieldTy);
      else
        KF_FOR_EACH(handleOtherType, Field, FieldTy);
      (void)std::initializer_list<int>{
          (handlers.leaveField(Owner, Field), 0)...};
    }
  }
#undef KF_FOR_EACH
};
// Parent contains the FieldDecl or CXXBaseSpecifier that was used to enter
// the Wrapper structure that we're currently visiting. Owner is the parent
// type (which doesn't exist in cases where it is a FieldDecl in the
// 'root'), and Wrapper is the current struct being unwrapped.
template <typename ParentTy, typename... Handlers>
void KernelObjVisitor::VisitRecord(const CXXRecordDecl *Owner, ParentTy &Parent,
                                   const CXXRecordDecl *Wrapper,
                                   Handlers &... handlers) {
  (void)std::initializer_list<int>{(handlers.enterStruct(Owner, Parent), 0)...};
  VisitRecordHelper(Wrapper, Wrapper->bases(), handlers...);
  VisitRecordHelper(Wrapper, Wrapper->fields(), handlers...);
  (void)std::initializer_list<int>{(handlers.leaveStruct(Owner, Parent), 0)...};
}

// A base type that the SYCL OpenCL Kernel construction task uses to implement
// individual tasks.
class SyclKernelFieldHandler {
protected:
  Sema &SemaRef;
  SyclKernelFieldHandler(Sema &S) : SemaRef(S) {}

public:
  // Mark these virtual so that we can use override in the implementer classes,
  // despite virtual dispatch never being used.

  // Accessor can be a base class or a field decl, so both must be handled.
  virtual bool handleSyclAccessorType(const CXXBaseSpecifier &, QualType) {
    return true;
  }
  virtual bool handleSyclAccessorType(FieldDecl *, QualType) { return true; }
  virtual bool handleSyclSamplerType(const CXXBaseSpecifier &, QualType) {
    return true;
  }
  virtual bool handleSyclSamplerType(FieldDecl *, QualType) { return true; }
  virtual bool handleSyclSpecConstantType(FieldDecl *, QualType) {
    return true;
  }
  virtual bool handleSyclStreamType(const CXXBaseSpecifier &, QualType) {
    return true;
  }
  virtual bool handleSyclStreamType(FieldDecl *, QualType) { return true; }
  virtual bool handleSyclHalfType(const CXXBaseSpecifier &, QualType) {
    return true;
  }
  virtual bool handleSyclHalfType(FieldDecl *, QualType) { return true; }
  virtual bool handleStructType(FieldDecl *, QualType) { return true; }
  virtual bool handleUnionType(FieldDecl *, QualType) { return true; }
  virtual bool handleReferenceType(FieldDecl *, QualType) { return true; }
  virtual bool handlePointerType(FieldDecl *, QualType) { return true; }
  virtual bool handleArrayType(FieldDecl *, QualType) { return true; }
  virtual bool handleScalarType(FieldDecl *, QualType) { return true; }
  // Most handlers shouldn't be handling this, just the field checker.
  virtual bool handleOtherType(FieldDecl *, QualType) { return true; }

  // The following are only used for keeping track of where we are in the base
  // class/field graph. Int Headers use this to calculate offset, most others
  // don't have a need for these.

  virtual bool enterStruct(const CXXRecordDecl *, FieldDecl *) { return true; }
  virtual bool leaveStruct(const CXXRecordDecl *, FieldDecl *) { return true; }
  virtual bool enterStruct(const CXXRecordDecl *, const CXXBaseSpecifier &) {
    return true;
  }
  virtual bool leaveStruct(const CXXRecordDecl *, const CXXBaseSpecifier &) {
    return true;
  }
  virtual bool enterUnion(const CXXRecordDecl *, FieldDecl *) { return true; }
  virtual bool leaveUnion(const CXXRecordDecl *, FieldDecl *) { return true; }

  // The following are used for stepping through array elements.

  virtual bool enterField(const CXXRecordDecl *, const CXXBaseSpecifier &) {
    return true;
  }
  virtual bool leaveField(const CXXRecordDecl *, const CXXBaseSpecifier &) {
    return true;
  }
  virtual bool enterField(const CXXRecordDecl *, FieldDecl *) { return true; }
  virtual bool leaveField(const CXXRecordDecl *, FieldDecl *) { return true; }
  virtual bool enterArray() { return true; }
  virtual bool nextElement(QualType) { return true; }
  virtual bool leaveArray(FieldDecl *, QualType, int64_t) { return true; }

  virtual ~SyclKernelFieldHandler() = default;
};

// A type to check the validity of all of the argument types.
class SyclKernelFieldChecker : public SyclKernelFieldHandler {
  bool IsInvalid = false;
  DiagnosticsEngine &Diag;

  // Check whether the object should be disallowed from being copied to kernel.
  // Return true if not copyable, false if copyable.
  bool checkNotCopyableToKernel(const FieldDecl *FD, const QualType &FieldTy) {
    if (FieldTy->isArrayType()) {
      if (const auto *CAT =
              SemaRef.getASTContext().getAsConstantArrayType(FieldTy)) {
        QualType ET = CAT->getElementType();
        return checkNotCopyableToKernel(FD, ET);
      }
      return Diag.Report(FD->getLocation(),
                         diag::err_sycl_non_constant_array_type)
             << FieldTy;
    }

    if (SemaRef.getASTContext().getLangOpts().SYCLStdLayoutKernelParams)
      if (!FieldTy->isStandardLayoutType())
        return Diag.Report(FD->getLocation(),
                           diag::err_sycl_non_std_layout_type)
               << FieldTy;

    if (!FieldTy->isStructureOrClassType())
      return false;

    CXXRecordDecl *RD =
        cast<CXXRecordDecl>(FieldTy->getAs<RecordType>()->getDecl());
    if (!RD->hasTrivialCopyConstructor())
      return Diag.Report(FD->getLocation(),
                         diag::err_sycl_non_trivially_copy_ctor_dtor_type)
             << 0 << FieldTy;
    if (!RD->hasTrivialDestructor())
      return Diag.Report(FD->getLocation(),
                         diag::err_sycl_non_trivially_copy_ctor_dtor_type)
             << 1 << FieldTy;

    return false;
  }

  void checkPropertyListType(TemplateArgument PropList, SourceLocation Loc) {
    if (PropList.getKind() != TemplateArgument::ArgKind::Type) {
      SemaRef.Diag(Loc,
                   diag::err_sycl_invalid_accessor_property_template_param);
      return;
    }
    QualType PropListTy = PropList.getAsType();
    if (!Util::isPropertyListType(PropListTy)) {
      SemaRef.Diag(Loc,
                   diag::err_sycl_invalid_accessor_property_template_param);
      return;
    }
    const auto *PropListDecl =
        cast<ClassTemplateSpecializationDecl>(PropListTy->getAsRecordDecl());
    if (PropListDecl->getTemplateArgs().size() != 1) {
      SemaRef.Diag(Loc, diag::err_sycl_invalid_property_list_param_number)
          << "property_list";
      return;
    }
    const auto TemplArg = PropListDecl->getTemplateArgs()[0];
    if (TemplArg.getKind() != TemplateArgument::ArgKind::Pack) {
      SemaRef.Diag(Loc, diag::err_sycl_invalid_property_list_template_param)
          << /*property_list*/ 0 << /*parameter pack*/ 0;
      return;
    }
    for (TemplateArgument::pack_iterator Prop = TemplArg.pack_begin();
         Prop != TemplArg.pack_end(); ++Prop) {
      if (Prop->getKind() != TemplateArgument::ArgKind::Type) {
        SemaRef.Diag(Loc, diag::err_sycl_invalid_property_list_template_param)
            << /*property_list pack argument*/ 1 << /*type*/ 1;
        return;
      }
      QualType PropTy = Prop->getAsType();
      if (Util::isSyclBufferLocationType(PropTy))
        checkBufferLocationType(PropTy, Loc);
    }
  }

  void checkBufferLocationType(QualType PropTy, SourceLocation Loc) {
    const auto *PropDecl =
        dyn_cast<ClassTemplateSpecializationDecl>(PropTy->getAsRecordDecl());
    if (PropDecl->getTemplateArgs().size() != 1) {
      SemaRef.Diag(Loc, diag::err_sycl_invalid_property_list_param_number)
          << "buffer_location";
      return;
    }
    const auto BufferLoc = PropDecl->getTemplateArgs()[0];
    if (BufferLoc.getKind() != TemplateArgument::ArgKind::Integral) {
      SemaRef.Diag(Loc, diag::err_sycl_invalid_property_list_template_param)
          << /*buffer_location*/ 2 << /*non-negative integer*/ 2;
      return;
    }
    int LocationID = static_cast<int>(BufferLoc.getAsIntegral().getExtValue());
    if (LocationID < 0) {
      SemaRef.Diag(Loc, diag::err_sycl_invalid_property_list_template_param)
          << /*buffer_location*/ 2 << /*non-negative integer*/ 2;
      return;
    }
  }

  void checkAccessorType(QualType Ty, SourceRange Loc) {
    assert(Util::isSyclAccessorType(Ty) &&
           "Should only be called on SYCL accessor types.");

    const RecordDecl *RecD = Ty->getAsRecordDecl();
    if (const ClassTemplateSpecializationDecl *CTSD =
            dyn_cast<ClassTemplateSpecializationDecl>(RecD)) {
      const TemplateArgumentList &TAL = CTSD->getTemplateArgs();
      TemplateArgument TA = TAL.get(0);
      const QualType TemplateArgTy = TA.getAsType();

      if (TAL.size() > 5)
        checkPropertyListType(TAL.get(5), Loc.getBegin());
      llvm::DenseSet<QualType> Visited;
      checkSYCLType(SemaRef, TemplateArgTy, Loc, Visited);
    }
  }

public:
  SyclKernelFieldChecker(Sema &S)
      : SyclKernelFieldHandler(S), Diag(S.getASTContext().getDiagnostics()) {}
  bool isValid() { return !IsInvalid; }

  bool handleReferenceType(FieldDecl *FD, QualType FieldTy) final {
    Diag.Report(FD->getLocation(), diag::err_bad_kernel_param_type) << FieldTy;
    IsInvalid = true;
    return isValid();
  }

  bool handleStructType(FieldDecl *FD, QualType FieldTy) final {
    IsInvalid |= checkNotCopyableToKernel(FD, FieldTy);
    return isValid();
  }

  bool handleSyclAccessorType(const CXXBaseSpecifier &BS,
                              QualType FieldTy) final {
    checkAccessorType(FieldTy, BS.getBeginLoc());
    return isValid();
  }

  bool handleSyclAccessorType(FieldDecl *FD, QualType FieldTy) final {
    checkAccessorType(FieldTy, FD->getLocation());
    return isValid();
  }

  bool handleArrayType(FieldDecl *FD, QualType FieldTy) final {
    IsInvalid |= checkNotCopyableToKernel(FD, FieldTy);
    return isValid();
  }

  bool handleOtherType(FieldDecl *FD, QualType FieldTy) final {
    Diag.Report(FD->getLocation(), diag::err_bad_kernel_param_type) << FieldTy;
    IsInvalid = true;
    return isValid();
  }
};

// A type to check the validity of passing union with accessor/sampler/stream
// member as a kernel argument types.
class SyclKernelUnionBodyChecker : public SyclKernelFieldHandler {
  static constexpr const bool VisitUnionBody = true;
  int UnionCount = 0;
  bool IsInvalid = false;
  DiagnosticsEngine &Diag;

  public:
  SyclKernelUnionBodyChecker(Sema &S)
      : SyclKernelFieldHandler(S), Diag(S.getASTContext().getDiagnostics()) {}
  bool isValid() { return !IsInvalid; }

  bool enterUnion(const CXXRecordDecl *RD, FieldDecl *FD) {
    ++UnionCount;
    return true;
  }

  bool leaveUnion(const CXXRecordDecl *RD, FieldDecl *FD) {
    --UnionCount;
    return true;
  }

  bool handlePointerType(FieldDecl *FD, QualType FieldTy) final {
    if (UnionCount) {
      IsInvalid = true;
      Diag.Report(FD->getLocation(), diag::err_bad_kernel_param_type)
          << FieldTy;
    }
    return isValid();
  }

  bool handleSyclAccessorType(FieldDecl *FD, QualType FieldTy) final {
    if (UnionCount) {
      IsInvalid = true;
      Diag.Report(FD->getLocation(), diag::err_bad_kernel_param_type)
          << FieldTy;
    }
    return isValid();
  }

  bool handleSyclSamplerType(FieldDecl *FD, QualType FieldTy) final {
    if (UnionCount) {
      IsInvalid = true;
      Diag.Report(FD->getLocation(), diag::err_bad_kernel_param_type)
          << FieldTy;
    }
    return isValid();
  }
  bool handleSyclStreamType(FieldDecl *FD, QualType FieldTy) final {
    if (UnionCount) {
      IsInvalid = true;
      Diag.Report(FD->getLocation(), diag::err_bad_kernel_param_type)
          << FieldTy;
    }
    return isValid();
   }
};

// A type to Create and own the FunctionDecl for the kernel.
class SyclKernelDeclCreator : public SyclKernelFieldHandler {
  FunctionDecl *KernelDecl;
  llvm::SmallVector<ParmVarDecl *, 8> Params;
  SyclKernelFieldChecker &ArgChecker;
  Sema::ContextRAII FuncContext;
  // Holds the last handled field's first parameter. This doesn't store an
  // iterator as push_back invalidates iterators.
  size_t LastParamIndex = 0;

  void addParam(const FieldDecl *FD, QualType FieldTy) {
    const ConstantArrayType *CAT =
        SemaRef.getASTContext().getAsConstantArrayType(FieldTy);
    if (CAT)
      FieldTy = CAT->getElementType();
    ParamDesc newParamDesc = makeParamDesc(FD, FieldTy);
    addParam(newParamDesc, FieldTy);
  }

  void addParam(const CXXBaseSpecifier &BS, QualType FieldTy) {
    ParamDesc newParamDesc =
        makeParamDesc(SemaRef.getASTContext(), BS, FieldTy);
    addParam(newParamDesc, FieldTy);
  }

  void addParam(ParamDesc newParamDesc, QualType FieldTy) {
    // Create a new ParmVarDecl based on the new info.
    ASTContext &Ctx = SemaRef.getASTContext();
    auto *NewParam = ParmVarDecl::Create(
        Ctx, KernelDecl, SourceLocation(), SourceLocation(),
        std::get<1>(newParamDesc), std::get<0>(newParamDesc),
        std::get<2>(newParamDesc), SC_None, /*DefArg*/ nullptr);
    NewParam->setScopeInfo(0, Params.size());
    NewParam->setIsUsed();

    LastParamIndex = Params.size();
    Params.push_back(NewParam);
  }

  // Handle accessor properties. If any properties were found in
  // the property_list - add the appropriate attributes to ParmVarDecl.
  void handleAccessorPropertyList(ParmVarDecl *Param,
                                  const CXXRecordDecl *RecordDecl,
                                  SourceLocation Loc) {
    const auto *AccTy = cast<ClassTemplateSpecializationDecl>(RecordDecl);
    // TODO: when SYCL headers' part is ready - replace this 'if' with an error
    if (AccTy->getTemplateArgs().size() < 6)
      return;
    const auto PropList = cast<TemplateArgument>(AccTy->getTemplateArgs()[5]);
    QualType PropListTy = PropList.getAsType();
    const auto *PropListDecl =
        cast<ClassTemplateSpecializationDecl>(PropListTy->getAsRecordDecl());
    const auto TemplArg = PropListDecl->getTemplateArgs()[0];
    // Move through TemplateArgs list of a property list and search for
    // properties. If found - apply the appropriate attribute to ParmVarDecl.
    for (TemplateArgument::pack_iterator Prop = TemplArg.pack_begin();
         Prop != TemplArg.pack_end(); ++Prop) {
      QualType PropTy = Prop->getAsType();
      if (Util::isSyclBufferLocationType(PropTy))
        handleBufferLocationProperty(Param, PropTy, Loc);
    }
  }

  // Obtain an integer value stored in a template parameter of buffer_location
  // property to pass it to buffer_location kernel attribute
  void handleBufferLocationProperty(ParmVarDecl *Param, QualType PropTy,
                                    SourceLocation Loc) {
    // If we have more than 1 buffer_location properties on a single
    // accessor - emit an error
    if (Param->hasAttr<SYCLIntelBufferLocationAttr>()) {
      SemaRef.Diag(Loc, diag::err_sycl_compiletime_property_duplication)
          << "buffer_location";
      return;
    }
    ASTContext &Ctx = SemaRef.getASTContext();
    const auto *PropDecl =
        cast<ClassTemplateSpecializationDecl>(PropTy->getAsRecordDecl());
    const auto BufferLoc = PropDecl->getTemplateArgs()[0];
    int LocationID = static_cast<int>(BufferLoc.getAsIntegral().getExtValue());
    Param->addAttr(
        SYCLIntelBufferLocationAttr::CreateImplicit(Ctx, LocationID));
  }

  // All special SYCL objects must have __init method. We extract types for
  // kernel parameters from __init method parameters. We will use __init method
  // and kernel parameters which we build here to initialize special objects in
  // the kernel body.
  bool handleSpecialType(FieldDecl *FD, QualType FieldTy,
                         bool isAccessorType = false) {
    const auto *RecordDecl = FieldTy->getAsCXXRecordDecl();
    assert(RecordDecl && "The accessor/sampler must be a RecordDecl");
    CXXMethodDecl *InitMethod = getMethodByName(RecordDecl, InitMethodName);
    assert(InitMethod && "The accessor/sampler must have the __init method");

    // Don't do -1 here because we count on this to be the first parameter added
    // (if any).
    size_t ParamIndex = Params.size();
    for (const ParmVarDecl *Param : InitMethod->parameters()) {
      QualType ParamTy = Param->getType();
      addParam(FD, ParamTy.getCanonicalType());
      if (ParamTy.getTypePtr()->isPointerType() && isAccessorType)
        handleAccessorPropertyList(Params.back(), RecordDecl,
                                   FD->getLocation());
    }
    LastParamIndex = ParamIndex;
    return true;
  }

  static void setKernelImplicitAttrs(ASTContext &Context, FunctionDecl *FD,
                                     StringRef Name, bool IsSIMDKernel) {
    // Set implicit attributes.
    FD->addAttr(OpenCLKernelAttr::CreateImplicit(Context));
    FD->addAttr(AsmLabelAttr::CreateImplicit(Context, Name));
    FD->addAttr(ArtificialAttr::CreateImplicit(Context));
    if (IsSIMDKernel)
      FD->addAttr(SYCLSimdAttr::CreateImplicit(Context));
  }

  static FunctionDecl *createKernelDecl(ASTContext &Ctx, StringRef Name,
                                        SourceLocation Loc, bool IsInline,
                                        bool IsSIMDKernel) {
    // Create this with no prototype, and we can fix this up after we've seen
    // all the params.
    FunctionProtoType::ExtProtoInfo Info(CC_OpenCLKernel);
    QualType FuncType = Ctx.getFunctionType(Ctx.VoidTy, {}, Info);

    FunctionDecl *FD = FunctionDecl::Create(
        Ctx, Ctx.getTranslationUnitDecl(), Loc, Loc, &Ctx.Idents.get(Name),
        FuncType, Ctx.getTrivialTypeSourceInfo(Ctx.VoidTy), SC_None);
    FD->setImplicitlyInline(IsInline);
    setKernelImplicitAttrs(Ctx, FD, Name, IsSIMDKernel);

    // Add kernel to translation unit to see it in AST-dump.
    Ctx.getTranslationUnitDecl()->addDecl(FD);
    return FD;
  }

public:
  SyclKernelDeclCreator(Sema &S, SyclKernelFieldChecker &ArgChecker,
                        StringRef Name, SourceLocation Loc, bool IsInline,
                        bool IsSIMDKernel)
      : SyclKernelFieldHandler(S),
        KernelDecl(createKernelDecl(S.getASTContext(), Name, Loc, IsInline,
                                    IsSIMDKernel)),
        ArgChecker(ArgChecker), FuncContext(SemaRef, KernelDecl) {}

  ~SyclKernelDeclCreator() {
    ASTContext &Ctx = SemaRef.getASTContext();
    FunctionProtoType::ExtProtoInfo Info(CC_OpenCLKernel);

    SmallVector<QualType, 8> ArgTys;
    std::transform(std::begin(Params), std::end(Params),
                   std::back_inserter(ArgTys),
                   [](const ParmVarDecl *PVD) { return PVD->getType(); });

    QualType FuncType = Ctx.getFunctionType(Ctx.VoidTy, ArgTys, Info);
    KernelDecl->setType(FuncType);
    KernelDecl->setParams(Params);

    if (ArgChecker.isValid())
      SemaRef.addSyclDeviceDecl(KernelDecl);
  }

  bool handleSyclAccessorType(const CXXBaseSpecifier &BS,
                              QualType FieldTy) final {
    const auto *RecordDecl = FieldTy->getAsCXXRecordDecl();
    assert(RecordDecl && "The accessor/sampler must be a RecordDecl");
    CXXMethodDecl *InitMethod = getMethodByName(RecordDecl, InitMethodName);
    assert(InitMethod && "The accessor/sampler must have the __init method");

    // Don't do -1 here because we count on this to be the first parameter added
    // (if any).
    size_t ParamIndex = Params.size();
    for (const ParmVarDecl *Param : InitMethod->parameters()) {
      QualType ParamTy = Param->getType();
      addParam(BS, ParamTy.getCanonicalType());
      if (ParamTy.getTypePtr()->isPointerType())
        handleAccessorPropertyList(Params.back(), RecordDecl, BS.getBeginLoc());
    }
    LastParamIndex = ParamIndex;
    return true;
  }

  bool handleSyclAccessorType(FieldDecl *FD, QualType FieldTy) final {
    return handleSpecialType(FD, FieldTy, /*isAccessorType*/ true);
  }

  bool handleSyclSamplerType(FieldDecl *FD, QualType FieldTy) final {
    return handleSpecialType(FD, FieldTy);
  }

  bool handlePointerType(FieldDecl *FD, QualType FieldTy) final {
    // USM allows to use raw pointers instead of buffers/accessors, but these
    // pointers point to the specially allocated memory. For pointer fields we
    // add a kernel argument with the same type as field but global address
    // space, because OpenCL requires it.
    QualType PointeeTy = FieldTy->getPointeeType();
    Qualifiers Quals = PointeeTy.getQualifiers();
    auto AS = Quals.getAddressSpace();
    // Leave global_device and global_host address spaces as is to help FPGA
    // device in memory allocations
    if (AS != LangAS::opencl_global_device && AS != LangAS::opencl_global_host)
      Quals.setAddressSpace(LangAS::opencl_global);
    PointeeTy = SemaRef.getASTContext().getQualifiedType(
        PointeeTy.getUnqualifiedType(), Quals);
    QualType ModTy = SemaRef.getASTContext().getPointerType(PointeeTy);
    addParam(FD, ModTy);
    return true;
  }

  bool handleScalarType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FD, FieldTy);
    return true;
  }

  bool handleUnionType(FieldDecl *FD, QualType FieldTy) final {
    return handleScalarType(FD, FieldTy);
  }

  bool handleSyclHalfType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FD, FieldTy);
    return true;
  }

  bool handleSyclStreamType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FD, FieldTy);
    return true;
  }

  bool handleSyclStreamType(const CXXBaseSpecifier &, QualType FieldTy) final {
    // FIXME SYCL stream should be usable as a base type
    // See https://github.com/intel/llvm/issues/1552
    return true;
  }

  void setBody(CompoundStmt *KB) { KernelDecl->setBody(KB); }

  FunctionDecl *getKernelDecl() { return KernelDecl; }

  llvm::ArrayRef<ParmVarDecl *> getParamVarDeclsForCurrentField() {
    return ArrayRef<ParmVarDecl *>(std::begin(Params) + LastParamIndex,
                                   std::end(Params));
  }
  using SyclKernelFieldHandler::handleSyclHalfType;
  using SyclKernelFieldHandler::handleSyclSamplerType;
};

class SyclKernelBodyCreator : public SyclKernelFieldHandler {
  SyclKernelDeclCreator &DeclCreator;
  llvm::SmallVector<Stmt *, 16> BodyStmts;
  llvm::SmallVector<Stmt *, 16> FinalizeStmts;
  llvm::SmallVector<Expr *, 16> InitExprs;
  VarDecl *KernelObjClone;
  InitializedEntity VarEntity;
  const CXXRecordDecl *KernelObj;
  llvm::SmallVector<Expr *, 16> MemberExprBases;
  uint64_t ArrayIndex;
  FunctionDecl *KernelCallerFunc;

  // Using the statements/init expressions that we've created, this generates
  // the kernel body compound stmt. CompoundStmt needs to know its number of
  // statements in advance to allocate it, so we cannot do this as we go along.
  CompoundStmt *createKernelBody() {

    Expr *ILE = new (SemaRef.getASTContext()) InitListExpr(
        SemaRef.getASTContext(), SourceLocation(), InitExprs, SourceLocation());
    ILE->setType(QualType(KernelObj->getTypeForDecl(), 0));
    KernelObjClone->setInit(ILE);
    Stmt *FunctionBody = KernelCallerFunc->getBody();

    ParmVarDecl *KernelObjParam = *(KernelCallerFunc->param_begin());

    // DeclRefExpr with valid source location but with decl which is not marked
    // as used is invalid.
    KernelObjClone->setIsUsed();
    std::pair<DeclaratorDecl *, DeclaratorDecl *> MappingPair =
        std::make_pair(KernelObjParam, KernelObjClone);

    // Push the Kernel function scope to ensure the scope isn't empty
    SemaRef.PushFunctionScope();
    KernelBodyTransform KBT(MappingPair, SemaRef);
    Stmt *NewBody = KBT.TransformStmt(FunctionBody).get();
    BodyStmts.push_back(NewBody);

    BodyStmts.insert(BodyStmts.end(), FinalizeStmts.begin(),
                     FinalizeStmts.end());
    return CompoundStmt::Create(SemaRef.getASTContext(), BodyStmts, {}, {});
  }

  void markParallelWorkItemCalls() {
    if (getKernelInvocationKind(KernelCallerFunc) ==
        InvokeParallelForWorkGroup) {
      FindPFWGLambdaFnVisitor V(KernelObj);
      V.TraverseStmt(KernelCallerFunc->getBody());
      CXXMethodDecl *WGLambdaFn = V.getLambdaFn();
      assert(WGLambdaFn && "PFWG lambda not found");
      // Mark the function that it "works" in a work group scope:
      // NOTE: In case of parallel_for_work_item the marker call itself is
      // marked with work item scope attribute, here  the '()' operator of the
      // object passed as parameter is marked. This is an optimization -
      // there are a lot of locals created at parallel_for_work_group
      // scope before calling the lambda - it is more efficient to have
      // all of them in the private address space rather then sharing via
      // the local AS. See parallel_for_work_group implementation in the
      // SYCL headers.
      if (!WGLambdaFn->hasAttr<SYCLScopeAttr>()) {
        WGLambdaFn->addAttr(SYCLScopeAttr::CreateImplicit(
            SemaRef.getASTContext(), SYCLScopeAttr::Level::WorkGroup));
        // Search and mark parallel_for_work_item calls:
        MarkWIScopeFnVisitor MarkWIScope(SemaRef.getASTContext());
        MarkWIScope.TraverseDecl(WGLambdaFn);
        // Now mark local variables declared in the PFWG lambda with work group
        // scope attribute
        addScopeAttrToLocalVars(*WGLambdaFn);
      }
    }
  }

  MemberExpr *BuildMemberExpr(Expr *Base, ValueDecl *Member) {
    DeclAccessPair MemberDAP = DeclAccessPair::make(Member, AS_none);
    MemberExpr *Result = SemaRef.BuildMemberExpr(
        Base, /*IsArrow */ false, SourceLocation(), NestedNameSpecifierLoc(),
        SourceLocation(), Member, MemberDAP,
        /*HadMultipleCandidates*/ false,
        DeclarationNameInfo(Member->getDeclName(), SourceLocation()),
        Member->getType(), VK_LValue, OK_Ordinary);
    return Result;
  }

  Expr *createInitExpr(FieldDecl *FD) {
    ParmVarDecl *KernelParameter =
        DeclCreator.getParamVarDeclsForCurrentField()[0];
    QualType ParamType = KernelParameter->getOriginalType();
    Expr *DRE = SemaRef.BuildDeclRefExpr(KernelParameter, ParamType, VK_LValue,
                                         SourceLocation());
    if (FD->getType()->isPointerType() &&
        FD->getType()->getPointeeType().getAddressSpace() !=
            ParamType->getPointeeType().getAddressSpace())
      DRE = ImplicitCastExpr::Create(SemaRef.Context, FD->getType(),
                                     CK_AddressSpaceConversion, DRE, nullptr,
                                     VK_RValue);
    return DRE;
  }

  void createExprForStructOrScalar(FieldDecl *FD) {
    InitializedEntity Entity =
        InitializedEntity::InitializeMember(FD, &VarEntity);
    InitializationKind InitKind =
        InitializationKind::CreateCopy(SourceLocation(), SourceLocation());
    Expr *DRE = createInitExpr(FD);
    InitializationSequence InitSeq(SemaRef, Entity, InitKind, DRE);
    ExprResult MemberInit = InitSeq.Perform(SemaRef, Entity, InitKind, DRE);
    InitExprs.push_back(MemberInit.get());
  }

  void createExprForScalarElement(FieldDecl *FD) {
    InitializedEntity ArrayEntity =
        InitializedEntity::InitializeMember(FD, &VarEntity);
    InitializationKind InitKind =
        InitializationKind::CreateCopy(SourceLocation(), SourceLocation());
    Expr *DRE = createInitExpr(FD);
    InitializedEntity Entity = InitializedEntity::InitializeElement(
        SemaRef.getASTContext(), ArrayIndex, ArrayEntity);
    ArrayIndex++;
    InitializationSequence InitSeq(SemaRef, Entity, InitKind, DRE);
    ExprResult MemberInit = InitSeq.Perform(SemaRef, Entity, InitKind, DRE);
    InitExprs.push_back(MemberInit.get());
  }

  void addArrayInit(FieldDecl *FD, int64_t Count) {
    llvm::SmallVector<Expr *, 16> ArrayInitExprs;
    for (int64_t I = 0; I < Count; I++) {
      ArrayInitExprs.push_back(InitExprs.back());
      InitExprs.pop_back();
    }
    std::reverse(ArrayInitExprs.begin(), ArrayInitExprs.end());
    Expr *ILE = new (SemaRef.getASTContext())
        InitListExpr(SemaRef.getASTContext(), SourceLocation(), ArrayInitExprs,
                     SourceLocation());
    ILE->setType(FD->getType());
    InitExprs.push_back(ILE);
  }

  CXXMemberCallExpr *createSpecialMethodCall(Expr *Base, CXXMethodDecl *Method,
                                             FieldDecl *Field) {
    unsigned NumParams = Method->getNumParams();
    llvm::SmallVector<Expr *, 4> ParamDREs(NumParams);
    llvm::ArrayRef<ParmVarDecl *> KernelParameters =
        DeclCreator.getParamVarDeclsForCurrentField();
    for (size_t I = 0; I < NumParams; ++I) {
      QualType ParamType = KernelParameters[I]->getOriginalType();
      ParamDREs[I] = SemaRef.BuildDeclRefExpr(KernelParameters[I], ParamType,
                                              VK_LValue, SourceLocation());
    }
    MemberExpr *MethodME = BuildMemberExpr(Base, Method);

    QualType ResultTy = Method->getReturnType();
    ExprValueKind VK = Expr::getValueKindForType(ResultTy);
    ResultTy = ResultTy.getNonLValueExprType(SemaRef.Context);
    llvm::SmallVector<Expr *, 4> ParamStmts;
    const auto *Proto = cast<FunctionProtoType>(Method->getType());
    SemaRef.GatherArgumentsForCall(SourceLocation(), Method, Proto, 0,
                                   ParamDREs, ParamStmts);
    // [kernel_obj or wrapper object].accessor.__init(_ValueType*,
    // range<int>, range<int>, id<int>)
    CXXMemberCallExpr *Call = CXXMemberCallExpr::Create(
        SemaRef.Context, MethodME, ParamStmts, ResultTy, VK, SourceLocation(),
        FPOptionsOverride());
    return Call;
  }

  // FIXME Avoid creation of kernel obj clone.
  // See https://github.com/intel/llvm/issues/1544 for details.
  static VarDecl *createKernelObjClone(ASTContext &Ctx, DeclContext *DC,
                                       const CXXRecordDecl *KernelObj) {
    TypeSourceInfo *TSInfo =
        KernelObj->isLambda() ? KernelObj->getLambdaTypeInfo() : nullptr;
    VarDecl *VD = VarDecl::Create(
        Ctx, DC, SourceLocation(), SourceLocation(), KernelObj->getIdentifier(),
        QualType(KernelObj->getTypeForDecl(), 0), TSInfo, SC_None);

    return VD;
  }

  bool handleSpecialType(FieldDecl *FD, QualType Ty) {
    const auto *RecordDecl = Ty->getAsCXXRecordDecl();
    // TODO: VarEntity is initialized entity for KernelObjClone, I guess we need
    // to create new one when enter new struct.
    InitializedEntity Entity =
        InitializedEntity::InitializeMember(FD, &VarEntity);
    // Initialize with the default constructor.
    InitializationKind InitKind =
        InitializationKind::CreateDefault(SourceLocation());
    InitializationSequence InitSeq(SemaRef, Entity, InitKind, None);
    ExprResult MemberInit = InitSeq.Perform(SemaRef, Entity, InitKind, None);
    InitExprs.push_back(MemberInit.get());

    CXXMethodDecl *InitMethod = getMethodByName(RecordDecl, InitMethodName);
    if (InitMethod) {
      CXXMemberCallExpr *InitCall =
          createSpecialMethodCall(MemberExprBases.back(), InitMethod, FD);
      BodyStmts.push_back(InitCall);
    }
    return true;
  }

  bool handleSpecialType(const CXXBaseSpecifier &BS, QualType Ty) {
    const auto *RecordDecl = Ty->getAsCXXRecordDecl();
    // TODO: VarEntity is initialized entity for KernelObjClone, I guess we need
    // to create new one when enter new struct.
    InitializedEntity Entity = InitializedEntity::InitializeBase(
        SemaRef.Context, &BS, /*IsInheritedVirtualBase*/ false, &VarEntity);
    // Initialize with the default constructor.
    InitializationKind InitKind =
        InitializationKind::CreateDefault(SourceLocation());
    InitializationSequence InitSeq(SemaRef, Entity, InitKind, None);
    ExprResult MemberInit = InitSeq.Perform(SemaRef, Entity, InitKind, None);
    InitExprs.push_back(MemberInit.get());

    CXXMethodDecl *InitMethod = getMethodByName(RecordDecl, InitMethodName);
    if (InitMethod) {
      CXXMemberCallExpr *InitCall =
          createSpecialMethodCall(MemberExprBases.back(), InitMethod, nullptr);
      BodyStmts.push_back(InitCall);
    }
    return true;
  }

public:
  SyclKernelBodyCreator(Sema &S, SyclKernelDeclCreator &DC,
                        const CXXRecordDecl *KernelObj,
                        FunctionDecl *KernelCallerFunc)
      : SyclKernelFieldHandler(S), DeclCreator(DC),
        KernelObjClone(createKernelObjClone(S.getASTContext(),
                                            DC.getKernelDecl(), KernelObj)),
        VarEntity(InitializedEntity::InitializeVariable(KernelObjClone)),
        KernelObj(KernelObj), KernelCallerFunc(KernelCallerFunc) {
    markParallelWorkItemCalls();

    Stmt *DS = new (S.Context) DeclStmt(DeclGroupRef(KernelObjClone),
                                        SourceLocation(), SourceLocation());
    BodyStmts.push_back(DS);
    DeclRefExpr *KernelObjCloneRef = DeclRefExpr::Create(
        S.Context, NestedNameSpecifierLoc(), SourceLocation(), KernelObjClone,
        false, DeclarationNameInfo(), QualType(KernelObj->getTypeForDecl(), 0),
        VK_LValue);
    MemberExprBases.push_back(KernelObjCloneRef);
  }

  ~SyclKernelBodyCreator() {
    CompoundStmt *KernelBody = createKernelBody();
    DeclCreator.setBody(KernelBody);
  }

  bool handleSyclAccessorType(FieldDecl *FD, QualType Ty) final {
    return handleSpecialType(FD, Ty);
  }

  bool handleSyclAccessorType(const CXXBaseSpecifier &BS, QualType Ty) final {
    return handleSpecialType(BS, Ty);
  }

  bool handleSyclSamplerType(FieldDecl *FD, QualType Ty) final {
    return handleSpecialType(FD, Ty);
  }

  bool handleSyclSpecConstantType(FieldDecl *FD, QualType Ty) final {
    return handleSpecialType(FD, Ty);
  }

  bool handleSyclStreamType(FieldDecl *FD, QualType Ty) final {
    const auto *StreamDecl = Ty->getAsCXXRecordDecl();
    createExprForStructOrScalar(FD);
    size_t NumBases = MemberExprBases.size();
    CXXMethodDecl *InitMethod = getMethodByName(StreamDecl, InitMethodName);
    if (InitMethod) {
      CXXMemberCallExpr *InitCall =
          createSpecialMethodCall(MemberExprBases.back(), InitMethod, FD);
      BodyStmts.push_back(InitCall);
    }
    CXXMethodDecl *FinalizeMethod =
        getMethodByName(StreamDecl, FinalizeMethodName);
    if (FinalizeMethod) {
      CXXMemberCallExpr *FinalizeCall = createSpecialMethodCall(
          MemberExprBases[NumBases - 2], FinalizeMethod, FD);
      FinalizeStmts.push_back(FinalizeCall);
    }
    return true;
  }

  bool handleSyclStreamType(const CXXBaseSpecifier &BS, QualType Ty) final {
    // FIXME SYCL stream should be usable as a base type
    // See https://github.com/intel/llvm/issues/1552
    return true;
  }

  bool handleSyclHalfType(FieldDecl *FD, QualType Ty) final {
    createExprForStructOrScalar(FD);
    return true;
  }

  bool handlePointerType(FieldDecl *FD, QualType FieldTy) final {
    createExprForStructOrScalar(FD);
    return true;
  }

  bool handleScalarType(FieldDecl *FD, QualType FieldTy) final {
    if (dyn_cast<ArraySubscriptExpr>(MemberExprBases.back()))
      createExprForScalarElement(FD);
    else
      createExprForStructOrScalar(FD);
    return true;
  }

  bool handleUnionType(FieldDecl *FD, QualType FieldTy) final {
    return handleScalarType(FD, FieldTy);
  }

  bool enterStruct(const CXXRecordDecl *RD, const CXXBaseSpecifier &BS) final {
    CXXCastPath BasePath;
    QualType DerivedTy(RD->getTypeForDecl(), 0);
    QualType BaseTy = BS.getType();
    SemaRef.CheckDerivedToBaseConversion(DerivedTy, BaseTy, SourceLocation(),
                                         SourceRange(), &BasePath,
                                         /*IgnoreBaseAccess*/ true);
    auto Cast = ImplicitCastExpr::Create(
        SemaRef.Context, BaseTy, CK_DerivedToBase, MemberExprBases.back(),
        /* CXXCastPath=*/&BasePath, VK_LValue);
    MemberExprBases.push_back(Cast);
    return true;
  }

  void addStructInit(const CXXRecordDecl *RD) {
    const ASTRecordLayout &Info =
        SemaRef.getASTContext().getASTRecordLayout(RD);
    int NumberOfFields = Info.getFieldCount();
    int popOut = NumberOfFields + RD->getNumBases();

    llvm::SmallVector<Expr *, 16> BaseInitExprs;
    for (int I = 0; I < popOut; I++) {
      BaseInitExprs.push_back(InitExprs.back());
      InitExprs.pop_back();
    }
    std::reverse(BaseInitExprs.begin(), BaseInitExprs.end());

    Expr *ILE = new (SemaRef.getASTContext())
        InitListExpr(SemaRef.getASTContext(), SourceLocation(), BaseInitExprs,
                     SourceLocation());
    ILE->setType(QualType(RD->getTypeForDecl(), 0));
    InitExprs.push_back(ILE);
  }

  bool leaveStruct(const CXXRecordDecl *, FieldDecl *FD) final {
    // Handle struct when kernel object field is struct type or array of
    // structs.
    const CXXRecordDecl *RD =
        FD->getType()->getBaseElementTypeUnsafe()->getAsCXXRecordDecl();

    // Initializers for accessors inside stream not added.
    if (!Util::isSyclStreamType(FD->getType()))
      addStructInit(RD);
    // Pop out unused initializers created in handleSyclAccesorType
    // for accessors inside stream class.
    else {
      for (const auto &Field : RD->fields()) {
        QualType FieldTy = Field->getType();
        if (Util::isSyclAccessorType(FieldTy))
          InitExprs.pop_back();
      }
    }
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *RD, const CXXBaseSpecifier &BS) final {
    const CXXRecordDecl *BaseClass = BS.getType()->getAsCXXRecordDecl();
    addStructInit(BaseClass);
    MemberExprBases.pop_back();
    return true;
  }

  bool enterField(const CXXRecordDecl *RD, FieldDecl *FD) final {
    if (!FD->getType()->isReferenceType())
      MemberExprBases.push_back(BuildMemberExpr(MemberExprBases.back(), FD));
    return true;
  }

  bool leaveField(const CXXRecordDecl *, FieldDecl *FD) final {
    if (!FD->getType()->isReferenceType())
      MemberExprBases.pop_back();
    return true;
  }

  bool enterArray() final {
    Expr *ArrayBase = MemberExprBases.back();
    ExprResult IndexExpr = SemaRef.ActOnIntegerConstant(SourceLocation(), 0);
    ExprResult ElementBase = SemaRef.CreateBuiltinArraySubscriptExpr(
        ArrayBase, SourceLocation(), IndexExpr.get(), SourceLocation());
    MemberExprBases.push_back(ElementBase.get());
    ArrayIndex = 0;
    return true;
  }

  bool nextElement(QualType ET) final {
    ArraySubscriptExpr *LastArrayRef =
        cast<ArraySubscriptExpr>(MemberExprBases.back());
    MemberExprBases.pop_back();
    Expr *LastIdx = LastArrayRef->getIdx();
    llvm::APSInt Result;
    SemaRef.VerifyIntegerConstantExpression(LastIdx, &Result);
    Expr *ArrayBase = MemberExprBases.back();
    ExprResult IndexExpr = SemaRef.ActOnIntegerConstant(
        SourceLocation(), Result.getExtValue() + 1);
    ExprResult ElementBase = SemaRef.CreateBuiltinArraySubscriptExpr(
        ArrayBase, SourceLocation(), IndexExpr.get(), SourceLocation());
    MemberExprBases.push_back(ElementBase.get());
    return true;
  }

  bool leaveArray(FieldDecl *FD, QualType, int64_t Count) final {
    addArrayInit(FD, Count);
    MemberExprBases.pop_back();
    return true;
  }

  using SyclKernelFieldHandler::enterArray;
  using SyclKernelFieldHandler::enterField;
  using SyclKernelFieldHandler::enterStruct;
  using SyclKernelFieldHandler::handleSyclHalfType;
  using SyclKernelFieldHandler::handleSyclSamplerType;
  using SyclKernelFieldHandler::leaveField;
  using SyclKernelFieldHandler::leaveStruct;
};

class SyclKernelIntHeaderCreator : public SyclKernelFieldHandler {
  SYCLIntegrationHeader &Header;
  int64_t CurOffset = 0;

  void addParam(const FieldDecl *FD, QualType ArgTy,
                SYCLIntegrationHeader::kernel_param_kind_t Kind) {
    uint64_t Size;
    const ConstantArrayType *CAT =
        SemaRef.getASTContext().getAsConstantArrayType(ArgTy);
    if (CAT)
      ArgTy = CAT->getElementType();
    Size = SemaRef.getASTContext().getTypeSizeInChars(ArgTy).getQuantity();
    Header.addParamDesc(Kind, static_cast<unsigned>(Size),
                        static_cast<unsigned>(CurOffset));
  }

public:
  SyclKernelIntHeaderCreator(Sema &S, SYCLIntegrationHeader &H,
                             const CXXRecordDecl *KernelObj, QualType NameType,
                             StringRef Name, StringRef StableName)
      : SyclKernelFieldHandler(S), Header(H) {
    Header.startKernel(Name, NameType, StableName, KernelObj->getLocation());
  }

  bool handleSyclAccessorType(const CXXBaseSpecifier &BC,
                              QualType FieldTy) final {
    const auto *AccTy =
        cast<ClassTemplateSpecializationDecl>(FieldTy->getAsRecordDecl());
    assert(AccTy->getTemplateArgs().size() >= 2 &&
           "Incorrect template args for Accessor Type");
    int Dims = static_cast<int>(
        AccTy->getTemplateArgs()[1].getAsIntegral().getExtValue());
    int Info = getAccessTarget(AccTy) | (Dims << 11);
    Header.addParamDesc(SYCLIntegrationHeader::kind_accessor, Info, CurOffset);
    return true;
  }

  bool handleSyclAccessorType(FieldDecl *FD, QualType FieldTy) final {
    const auto *AccTy =
        cast<ClassTemplateSpecializationDecl>(FieldTy->getAsRecordDecl());
    assert(AccTy->getTemplateArgs().size() >= 2 &&
           "Incorrect template args for Accessor Type");
    int Dims = static_cast<int>(
        AccTy->getTemplateArgs()[1].getAsIntegral().getExtValue());
    int Info = getAccessTarget(AccTy) | (Dims << 11);
    Header.addParamDesc(SYCLIntegrationHeader::kind_accessor, Info, CurOffset);
    return true;
  }

  bool handleSyclSamplerType(FieldDecl *FD, QualType FieldTy) final {
    const auto *SamplerTy = FieldTy->getAsCXXRecordDecl();
    assert(SamplerTy && "Sampler type must be a C++ record type");
    CXXMethodDecl *InitMethod = getMethodByName(SamplerTy, InitMethodName);
    assert(InitMethod && "sampler must have __init method");

    // sampler __init method has only one argument
    const ParmVarDecl *SamplerArg = InitMethod->getParamDecl(0);
    assert(SamplerArg && "sampler __init method must have sampler parameter");

    addParam(FD, SamplerArg->getType(), SYCLIntegrationHeader::kind_sampler);
    return true;
  }

  bool handleSyclSpecConstantType(FieldDecl *FD, QualType FieldTy) final {
    const TemplateArgumentList &TemplateArgs =
        cast<ClassTemplateSpecializationDecl>(FieldTy->getAsRecordDecl())
            ->getTemplateInstantiationArgs();
    assert(TemplateArgs.size() == 2 &&
           "Incorrect template args for spec constant type");
    // Get specialization constant ID type, which is the second template
    // argument.
    QualType SpecConstIDTy = TemplateArgs.get(1).getAsType().getCanonicalType();
    const std::string SpecConstName = PredefinedExpr::ComputeName(
        SemaRef.getASTContext(), PredefinedExpr::UniqueStableNameType,
        SpecConstIDTy);
    Header.addSpecConstant(SpecConstName, SpecConstIDTy);
    return true;
  }

  bool handlePointerType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FD, FieldTy, SYCLIntegrationHeader::kind_pointer);
    return true;
  }

  bool handleScalarType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FD, FieldTy, SYCLIntegrationHeader::kind_std_layout);
    return true;
  }

  bool handleUnionType(FieldDecl *FD, QualType FieldTy) final {
    return handleScalarType(FD, FieldTy);
  }

  bool handleSyclStreamType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FD, FieldTy, SYCLIntegrationHeader::kind_std_layout);
    return true;
  }

  bool handleSyclStreamType(const CXXBaseSpecifier &BC,
                            QualType FieldTy) final {
    // FIXME SYCL stream should be usable as a base type
    // See https://github.com/intel/llvm/issues/1552
    return true;
  }

  bool handleSyclHalfType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FD, FieldTy, SYCLIntegrationHeader::kind_std_layout);
    return true;
  }

  bool enterField(const CXXRecordDecl *RD, FieldDecl *FD) final {
    CurOffset += SemaRef.getASTContext().getFieldOffset(FD) / 8;
    return true;
  }

  bool leaveField(const CXXRecordDecl *, FieldDecl *FD) final {
    CurOffset -= SemaRef.getASTContext().getFieldOffset(FD) / 8;
    return true;
  }

  bool enterField(const CXXRecordDecl *RD, const CXXBaseSpecifier &BS) final {
    const ASTRecordLayout &Layout =
        SemaRef.getASTContext().getASTRecordLayout(RD);
    CurOffset += Layout.getBaseClassOffset(BS.getType()->getAsCXXRecordDecl())
                     .getQuantity();
    return true;
  }

  bool leaveField(const CXXRecordDecl *RD, const CXXBaseSpecifier &BS) final {
    const ASTRecordLayout &Layout =
        SemaRef.getASTContext().getASTRecordLayout(RD);
    CurOffset -= Layout.getBaseClassOffset(BS.getType()->getAsCXXRecordDecl())
                     .getQuantity();
    return true;
  }

  bool nextElement(QualType ET) final {
    CurOffset += SemaRef.getASTContext().getTypeSizeInChars(ET).getQuantity();
    return true;
  }

  bool leaveArray(FieldDecl *, QualType ET, int64_t Count) final {
    int64_t ArraySize =
        SemaRef.getASTContext().getTypeSizeInChars(ET).getQuantity();
    if (!ET->isArrayType()) {
      ArraySize *= Count;
    }
    CurOffset -= ArraySize;
    return true;
  }
  using SyclKernelFieldHandler::handleSyclHalfType;
  using SyclKernelFieldHandler::handleSyclSamplerType;
};
} // namespace

// Generates the OpenCL kernel using KernelCallerFunc (kernel caller
// function) defined is SYCL headers.
// Generated OpenCL kernel contains the body of the kernel caller function,
// receives OpenCL like parameters and additionally does some manipulation to
// initialize captured lambda/functor fields with these parameters.
// SYCL runtime marks kernel caller function with sycl_kernel attribute.
// To be able to generate OpenCL kernel from KernelCallerFunc we put
// the following requirements to the function which SYCL runtime can mark with
// sycl_kernel attribute:
//   - Must be template function with at least two template parameters.
//     First parameter must represent "unique kernel name"
//     Second parameter must be the function object type
//   - Must have only one function parameter - function object.
//
// Example of kernel caller function:
//   template <typename KernelName, typename KernelType/*, ...*/>
//   __attribute__((sycl_kernel)) void kernel_caller_function(KernelType
//                                                            KernelFuncObj) {
//     KernelFuncObj();
//   }
//
//
void Sema::ConstructOpenCLKernel(FunctionDecl *KernelCallerFunc,
                                 MangleContext &MC) {
  // The first argument to the KernelCallerFunc is the lambda object.
  const CXXRecordDecl *KernelObj = getKernelObjectType(KernelCallerFunc);
  assert(KernelObj && "invalid kernel caller");

  // Calculate both names, since Integration headers need both.
  std::string CalculatedName, StableName;
  std::tie(CalculatedName, StableName) =
      constructKernelName(*this, KernelCallerFunc, MC);
  StringRef KernelName(getLangOpts().SYCLUnnamedLambda ? StableName
                                                       : CalculatedName);
  if (KernelObj->isLambda()) {
    for (const LambdaCapture &LC : KernelObj->captures())
      if (LC.capturesThis() && LC.isImplicit())
        Diag(LC.getLocation(), diag::err_implicit_this_capture);
  }
  SyclKernelFieldChecker checker(*this);
  SyclKernelDeclCreator kernel_decl(
      *this, checker, KernelName, KernelObj->getLocation(),
      KernelCallerFunc->isInlined(), KernelCallerFunc->hasAttr<SYCLSimdAttr>());
  SyclKernelBodyCreator kernel_body(*this, kernel_decl, KernelObj,
                                    KernelCallerFunc);
  SyclKernelIntHeaderCreator int_header(
      *this, getSyclIntegrationHeader(), KernelObj,
      calculateKernelNameType(Context, KernelCallerFunc), KernelName,
      StableName);

  ConstructingOpenCLKernel = true;
  KernelObjVisitor Visitor{*this};
  Visitor.VisitRecordBases(KernelObj, checker, kernel_decl, kernel_body,
                           int_header);
  Visitor.VisitRecordFields(KernelObj, checker, kernel_decl, kernel_body,
                            int_header);
  ConstructingOpenCLKernel = false;
}

// This function marks all the callees of explicit SIMD kernel
// with !sycl_explicit_simd. We want to have different semantics
// for functions that are called from SYCL and E-SIMD contexts.
// Later, functions marked with !sycl_explicit_simd will be cloned
// to maintain two different semantics.
void Sema::MarkSyclSimd() {
  for (Decl *D : syclDeviceDecls())
    if (auto SYCLKernel = dyn_cast<FunctionDecl>(D))
      if (SYCLKernel->hasAttr<SYCLSimdAttr>()) {
        MarkDeviceFunction Marker(*this);
        Marker.SYCLCG.addToCallGraph(getASTContext().getTranslationUnitDecl());
        llvm::SmallPtrSet<FunctionDecl *, 10> VisitedSet;
        Marker.CollectKernelSet(SYCLKernel, SYCLKernel, VisitedSet);
        for (const auto &elt : Marker.KernelSet) {
          if (FunctionDecl *Def = elt->getDefinition())
            if (!Def->hasAttr<SYCLSimdAttr>())
              Def->addAttr(SYCLSimdAttr::CreateImplicit(getASTContext()));
        }
      }
}

void Sema::MarkDevice(void) {
  // Create the call graph so we can detect recursion and check the validity
  // of new operator overrides. Add the kernel function itself in case
  // it is recursive.
  MarkDeviceFunction Marker(*this);
  Marker.SYCLCG.addToCallGraph(getASTContext().getTranslationUnitDecl());

  // Iterate through SYCL_EXTERNAL functions and add them to the device decls.
  for (const auto &entry : *Marker.SYCLCG.getRoot()) {
    if (auto *FD = dyn_cast<FunctionDecl>(entry.Callee->getDecl())) {
      if (FD->hasAttr<SYCLDeviceAttr>() && !FD->hasAttr<SYCLKernelAttr>() &&
          FD->hasBody())
        addSyclDeviceDecl(FD);
    }
  }

  for (Decl *D : syclDeviceDecls()) {
    if (auto SYCLKernel = dyn_cast<FunctionDecl>(D)) {
      llvm::SmallPtrSet<FunctionDecl *, 10> VisitedSet;
      Marker.CollectKernelSet(SYCLKernel, SYCLKernel, VisitedSet);

      // Let's propagate attributes from device functions to a SYCL kernels
      llvm::SmallPtrSet<Attr *, 4> Attrs;
      // This function collects all kernel attributes which might be applied to
      // a device functions, but need to be propagated down to callers, i.e.
      // SYCL kernels
      FunctionDecl *KernelBody =
          Marker.CollectPossibleKernelAttributes(SYCLKernel, Attrs);

      for (auto *A : Attrs) {
        switch (A->getKind()) {
        case attr::Kind::IntelReqdSubGroupSize: {
          auto *Attr = cast<IntelReqdSubGroupSizeAttr>(A);
          const auto *KBSimdAttr =
              KernelBody ? KernelBody->getAttr<SYCLSimdAttr>() : nullptr;
          if (auto *Existing =
                  SYCLKernel->getAttr<IntelReqdSubGroupSizeAttr>()) {
            if (getIntExprValue(Existing->getSubGroupSize(), getASTContext()) !=
                getIntExprValue(Attr->getSubGroupSize(), getASTContext())) {
              Diag(SYCLKernel->getLocation(),
                   diag::err_conflicting_sycl_kernel_attributes);
              Diag(Existing->getLocation(), diag::note_conflicting_attribute);
              Diag(Attr->getLocation(), diag::note_conflicting_attribute);
              SYCLKernel->setInvalidDecl();
            }
          } else if (KBSimdAttr && (getIntExprValue(Attr->getSubGroupSize(),
                                                    getASTContext()) != 1)) {
            reportConflictingAttrs(*this, KernelBody, KBSimdAttr, Attr);
          } else {
            SYCLKernel->addAttr(A);
          }
          break;
        }
        case attr::Kind::ReqdWorkGroupSize: {
          auto *Attr = cast<ReqdWorkGroupSizeAttr>(A);
          if (auto *Existing = SYCLKernel->getAttr<ReqdWorkGroupSizeAttr>()) {
            if (Existing->getXDim() != Attr->getXDim() ||
                Existing->getYDim() != Attr->getYDim() ||
                Existing->getZDim() != Attr->getZDim()) {
              Diag(SYCLKernel->getLocation(),
                   diag::err_conflicting_sycl_kernel_attributes);
              Diag(Existing->getLocation(), diag::note_conflicting_attribute);
              Diag(Attr->getLocation(), diag::note_conflicting_attribute);
              SYCLKernel->setInvalidDecl();
            }
          } else if (auto *Existing =
                         SYCLKernel->getAttr<SYCLIntelMaxWorkGroupSizeAttr>()) {
            if (Existing->getXDim() < Attr->getXDim() ||
                Existing->getYDim() < Attr->getYDim() ||
                Existing->getZDim() < Attr->getZDim()) {
              Diag(SYCLKernel->getLocation(),
                   diag::err_conflicting_sycl_kernel_attributes);
              Diag(Existing->getLocation(), diag::note_conflicting_attribute);
              Diag(Attr->getLocation(), diag::note_conflicting_attribute);
              SYCLKernel->setInvalidDecl();
            } else {
              SYCLKernel->addAttr(A);
            }
          } else {
            SYCLKernel->addAttr(A);
          }
          break;
        }
        case attr::Kind::SYCLIntelMaxWorkGroupSize: {
          auto *Attr = cast<SYCLIntelMaxWorkGroupSizeAttr>(A);
          if (auto *Existing = SYCLKernel->getAttr<ReqdWorkGroupSizeAttr>()) {
            if (Existing->getXDim() > Attr->getXDim() ||
                Existing->getYDim() > Attr->getYDim() ||
                Existing->getZDim() > Attr->getZDim()) {
              Diag(SYCLKernel->getLocation(),
                   diag::err_conflicting_sycl_kernel_attributes);
              Diag(Existing->getLocation(), diag::note_conflicting_attribute);
              Diag(Attr->getLocation(), diag::note_conflicting_attribute);
              SYCLKernel->setInvalidDecl();
            } else {
              SYCLKernel->addAttr(A);
            }
          } else {
            SYCLKernel->addAttr(A);
          }
          break;
        }
        case attr::Kind::SYCLIntelKernelArgsRestrict:
        case attr::Kind::SYCLIntelNumSimdWorkItems:
        case attr::Kind::SYCLIntelMaxGlobalWorkDim:
        case attr::Kind::SYCLIntelNoGlobalWorkOffset:
        case attr::Kind::SYCLSimd: {
          if ((A->getKind() == attr::Kind::SYCLSimd) && KernelBody &&
              !KernelBody->getAttr<SYCLSimdAttr>()) {
            // Usual kernel can't call ESIMD functions.
            Diag(KernelBody->getLocation(),
                 diag::err_sycl_function_attribute_mismatch)
                << A;
            Diag(A->getLocation(), diag::note_attribute);
            KernelBody->setInvalidDecl();
          } else
            SYCLKernel->addAttr(A);
          break;
        }
        // TODO: vec_len_hint should be handled here
        default:
          // Seeing this means that CollectPossibleKernelAttributes was
          // updated while this switch wasn't...or something went wrong
          llvm_unreachable("Unexpected attribute was collected by "
                           "CollectPossibleKernelAttributes");
        }
      }
    }
  }
  for (const auto &elt : Marker.KernelSet) {
    if (FunctionDecl *Def = elt->getDefinition())
      Marker.TraverseStmt(Def->getBody());
  }
}

// -----------------------------------------------------------------------------
// SYCL device specific diagnostics implementation
// -----------------------------------------------------------------------------

Sema::DeviceDiagBuilder Sema::SYCLDiagIfDeviceCode(SourceLocation Loc,
                                                   unsigned DiagID) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL compilation");
  FunctionDecl *FD = dyn_cast<FunctionDecl>(getCurLexicalContext());
  DeviceDiagBuilder::Kind DiagKind = [this, FD] {
    if (ConstructingOpenCLKernel)
      return DeviceDiagBuilder::K_ImmediateWithCallStack;
    if (!FD)
      return DeviceDiagBuilder::K_Nop;
    if (getEmissionStatus(FD) == Sema::FunctionEmissionStatus::Emitted)
      return DeviceDiagBuilder::K_ImmediateWithCallStack;
    return DeviceDiagBuilder::K_Deferred;
  }();
  return DeviceDiagBuilder(DiagKind, Loc, DiagID, FD, *this);
}

bool Sema::checkSYCLDeviceFunction(SourceLocation Loc, FunctionDecl *Callee) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL compilation");
  assert(Callee && "Callee may not be null.");

  // Errors in unevaluated context don't need to be generated,
  // so we can safely skip them.
  if (isUnevaluatedContext() || isConstantEvaluated())
    return true;

  FunctionDecl *Caller = dyn_cast<FunctionDecl>(getCurLexicalContext());

  if (!Caller)
    return true;

  DeviceDiagBuilder::Kind DiagKind = DeviceDiagBuilder::K_Nop;

  // TODO Set DiagKind to K_Immediate/K_Deferred to emit diagnostics for Callee

  DeviceDiagBuilder(DiagKind, Loc, diag::err_sycl_restrict, Caller, *this)
      << Sema::KernelCallUndefinedFunction;
  DeviceDiagBuilder(DiagKind, Callee->getLocation(), diag::note_previous_decl,
                    Caller, *this)
      << Callee;

  return DiagKind != DeviceDiagBuilder::K_Immediate &&
         DiagKind != DeviceDiagBuilder::K_ImmediateWithCallStack;
}

void Sema::finalizeSYCLDelayedAnalysis(const FunctionDecl *Caller,
                                       const FunctionDecl *Callee,
                                       SourceLocation Loc) {
  // Somehow an unspecialized template appears to be in callgraph or list of
  // device functions. We don't want to emit diagnostic here.
  if (Callee->getTemplatedKind() == FunctionDecl::TK_FunctionTemplate)
    return;

  Callee = Callee->getMostRecentDecl();
  bool HasAttr =
      Callee->hasAttr<SYCLDeviceAttr>() || Callee->hasAttr<SYCLKernelAttr>();

  // Disallow functions with neither definition nor SYCL_EXTERNAL mark
  bool NotDefinedNoAttr = !Callee->isDefined() && !HasAttr;

  if (NotDefinedNoAttr && !Callee->getBuiltinID()) {
    Diag(Loc, diag::err_sycl_restrict)
        << Sema::KernelCallUndefinedFunction;
    Diag(Callee->getLocation(), diag::note_previous_decl) << Callee;
    Diag(Caller->getLocation(), diag::note_called_by) << Caller;
  }
}

bool Sema::checkAllowedSYCLInitializer(VarDecl *VD, bool CheckValueDependent) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL compilation");

  if (VD->isInvalidDecl() || !VD->hasInit() || !VD->hasGlobalStorage())
    return true;

  const Expr *Init = VD->getInit();
  bool ValueDependent = CheckValueDependent && Init->isValueDependent();
  bool isConstantInit =
      Init && !ValueDependent && Init->isConstantInitializer(Context, false);
  if (!VD->isConstexpr() && Init && !ValueDependent && !isConstantInit)
    return false;

  return true;
}

// -----------------------------------------------------------------------------
// Integration header functionality implementation
// -----------------------------------------------------------------------------

/// Returns a string ID of given parameter kind - used in header
/// emission.
static const char *paramKind2Str(KernelParamKind K) {
#define CASE(x)                                                                \
  case SYCLIntegrationHeader::kind_##x:                                        \
    return "kind_" #x
  switch (K) {
    CASE(accessor);
    CASE(std_layout);
    CASE(sampler);
    CASE(pointer);
  default:
    return "<ERROR>";
  }
#undef CASE
}

// Removes all "(anonymous namespace)::" substrings from given string, and emits
// it.
static void emitWithoutAnonNamespaces(llvm::raw_ostream &OS, StringRef Source) {
  const char S1[] = "(anonymous namespace)::";

  size_t Pos;

  while ((Pos = Source.find(S1)) != StringRef::npos) {
    OS << Source.take_front(Pos);
    Source = Source.drop_front(Pos + sizeof(S1) - 1);
  }

  OS << Source;
}

static bool checkEnumTemplateParameter(const EnumDecl *ED,
                                       DiagnosticsEngine &Diag,
                                       SourceLocation KernelLocation) {
  if (!ED->isScoped() && !ED->isFixed()) {
    Diag.Report(KernelLocation, diag::err_sycl_kernel_incorrectly_named) << 2;
    Diag.Report(ED->getSourceRange().getBegin(), diag::note_entity_declared_at)
        << ED;
    return true;
  }
  return false;
}

// Emits a forward declaration
void SYCLIntegrationHeader::emitFwdDecl(raw_ostream &O, const Decl *D,
                                        SourceLocation KernelLocation) {
  // wrap the declaration into namespaces if needed
  unsigned NamespaceCnt = 0;
  std::string NSStr = "";
  const DeclContext *DC = D->getDeclContext();

  while (DC) {
    auto *NS = dyn_cast_or_null<NamespaceDecl>(DC);

    if (!NS) {
      if (!DC->isTranslationUnit()) {
        const TagDecl *TD = isa<ClassTemplateDecl>(D)
                                ? cast<ClassTemplateDecl>(D)->getTemplatedDecl()
                                : dyn_cast<TagDecl>(D);

        if (TD && !UnnamedLambdaSupport) {
          // defined class constituting the kernel name is not globally
          // accessible - contradicts the spec
          const bool KernelNameIsMissing = TD->getName().empty();
          if (KernelNameIsMissing) {
            Diag.Report(KernelLocation, diag::err_sycl_kernel_incorrectly_named)
                << /* kernel name is missing */ 0;
            // Don't emit note if kernel name was completely omitted
          } else {
            if (TD->isCompleteDefinition())
              Diag.Report(KernelLocation,
                          diag::err_sycl_kernel_incorrectly_named)
                  << /* kernel name is not globally-visible */ 1;
            else
              Diag.Report(KernelLocation, diag::warn_sycl_implicit_decl);
            Diag.Report(D->getSourceRange().getBegin(),
                        diag::note_previous_decl)
                << TD->getName();
          }
        }
      }
      break;
    }
    ++NamespaceCnt;
    const StringRef NSInlinePrefix = NS->isInline() ? "inline " : "";
    NSStr.insert(
        0, Twine(NSInlinePrefix + "namespace " + NS->getName() + " { ").str());
    DC = NS->getDeclContext();
  }
  O << NSStr;
  if (NamespaceCnt > 0)
    O << "\n";
  // print declaration into a string:
  PrintingPolicy P(D->getASTContext().getLangOpts());
  P.adjustForCPlusPlusFwdDecl();
  P.SuppressTypedefs = true;
  P.SuppressUnwrittenScope = true;
  std::string S;
  llvm::raw_string_ostream SO(S);
  D->print(SO, P);
  O << SO.str();

  if (const auto *ED = dyn_cast<EnumDecl>(D)) {
    QualType T = ED->getIntegerType();
    // Backup since getIntegerType() returns null for enum forward
    // declaration with no fixed underlying type
    if (T.isNull())
      T = ED->getPromotionType();
    O << " : " << T.getAsString();
  }

  O << ";\n";

  // print closing braces for namespaces if needed
  for (unsigned I = 0; I < NamespaceCnt; ++I)
    O << "}";
  if (NamespaceCnt > 0)
    O << "\n";
}

// Emits forward declarations of classes and template classes on which
// declaration of given type depends.
// For example, consider SimpleVadd
// class specialization in parallel_for below:
//
//   template <typename T1, unsigned int N, typename ... T2>
//   class SimpleVadd;
//   ...
//   template <unsigned int N, typename T1, typename ... T2>
//   void simple_vadd(const std::array<T1, N>& VA, const std::array<T1, N>&
//   VB,
//     std::array<T1, N>& VC, int param, T2 ... varargs) {
//     ...
//     deviceQueue.submit([&](cl::sycl::handler& cgh) {
//       ...
//       cgh.parallel_for<class SimpleVadd<T1, N, T2...>>(...)
//       ...
//     }
//     ...
//   }
//   ...
//   class MyClass {...};
//   template <typename T> class MyInnerTmplClass { ... }
//   template <typename T> class MyTmplClass { ... }
//   ...
//   MyClass *c = new MyClass();
//   MyInnerTmplClass<MyClass**> c1(&c);
//   simple_vadd(A, B, C, 5, 'a', 1.f,
//     new MyTmplClass<MyInnerTmplClass<MyClass**>>(c1));
//
// it will generate the following forward declarations:
//   class MyClass;
//   template <typename T> class MyInnerTmplClass;
//   template <typename T> class MyTmplClass;
//   template <typename T1, unsigned int N, typename ...T2> class SimpleVadd;
//
void SYCLIntegrationHeader::emitForwardClassDecls(
    raw_ostream &O, QualType T, SourceLocation KernelLocation,
    llvm::SmallPtrSetImpl<const void *> &Printed) {

  // peel off the pointer types and get the class/struct type:
  for (; T->isPointerType(); T = T->getPointeeType())
    ;
  const CXXRecordDecl *RD = T->getAsCXXRecordDecl();

  if (!RD)
    return;

  // see if this is a template specialization ...
  if (const auto *TSD = dyn_cast<ClassTemplateSpecializationDecl>(RD)) {
    // ... yes, it is template specialization:
    // - first, recurse into template parameters and emit needed forward
    //   declarations
    const TemplateArgumentList &Args = TSD->getTemplateArgs();

    for (unsigned I = 0; I < Args.size(); I++) {
      const TemplateArgument &Arg = Args[I];

      switch (Arg.getKind()) {
      case TemplateArgument::ArgKind::Type:
      case TemplateArgument::ArgKind::Integral: {
        QualType T = (Arg.getKind() == TemplateArgument::ArgKind::Type)
                         ? Arg.getAsType()
                         : Arg.getIntegralType();

        // Handle Kernel Name Type templated using enum type and value.
        if (const auto *ET = T->getAs<EnumType>()) {
          const EnumDecl *ED = ET->getDecl();
          if (!checkEnumTemplateParameter(ED, Diag, KernelLocation))
            emitFwdDecl(O, ED, KernelLocation);
        } else if (Arg.getKind() == TemplateArgument::ArgKind::Type)
          emitForwardClassDecls(O, T, KernelLocation, Printed);
        break;
      }
      case TemplateArgument::ArgKind::Pack: {
        ArrayRef<TemplateArgument> Pack = Arg.getPackAsArray();

        for (const auto &T : Pack) {
          if (T.getKind() == TemplateArgument::ArgKind::Type) {
            emitForwardClassDecls(O, T.getAsType(), KernelLocation, Printed);
          }
        }
        break;
      }
      case TemplateArgument::ArgKind::Template: {
        // recursion is not required, since the maximum possible nesting level
        // equals two for template argument
        //
        // for example:
        //   template <typename T> class Bar;
        //   template <template <typename> class> class Baz;
        //   template <template <template <typename> class> class T>
        //   class Foo;
        //
        // The Baz is a template class. The Baz<Bar> is a class. The class Foo
        // should be specialized with template class, not a class. The correct
        // specialization of template class Foo is Foo<Baz>. The incorrect
        // specialization of template class Foo is Foo<Baz<Bar>>. In this case
        // template class Foo specialized by class Baz<Bar>, not a template
        // class template <template <typename> class> class T as it should.
        TemplateDecl *TD = Arg.getAsTemplate().getAsTemplateDecl();
        TemplateParameterList *TemplateParams = TD->getTemplateParameters();
        for (NamedDecl *P : *TemplateParams) {
          // If template template paramter type has an enum value template
          // parameter, forward declaration of enum type is required. Only enum
          // values (not types) need to be handled. For example, consider the
          // following kernel name type:
          //
          // template <typename EnumTypeOut, template <EnumValueIn EnumValue,
          // typename TypeIn> class T> class Foo;
          //
          // The correct specialization for Foo (with enum type) is:
          // Foo<EnumTypeOut, Baz>, where Baz is a template class.
          //
          // Therefore the forward class declarations generated in the
          // integration header are:
          // template <EnumValueIn EnumValue, typename TypeIn> class Baz;
          // template <typename EnumTypeOut, template <EnumValueIn EnumValue,
          // typename EnumTypeIn> class T> class Foo;
          //
          // This requires the following enum forward declarations:
          // enum class EnumTypeOut : int; (Used to template Foo)
          // enum class EnumValueIn : int; (Used to template Baz)
          if (NonTypeTemplateParmDecl *TemplateParam =
                  dyn_cast<NonTypeTemplateParmDecl>(P)) {
            QualType T = TemplateParam->getType();
            if (const auto *ET = T->getAs<EnumType>()) {
              const EnumDecl *ED = ET->getDecl();
              if (!checkEnumTemplateParameter(ED, Diag, KernelLocation))
                emitFwdDecl(O, ED, KernelLocation);
            }
          }
        }
        if (Printed.insert(TD).second) {
          emitFwdDecl(O, TD, KernelLocation);
        }
        break;
      }
      default:
        break; // nop
      }
    }
    // - second, emit forward declaration for the template class being
    //   specialized
    ClassTemplateDecl *CTD = TSD->getSpecializedTemplate();
    assert(CTD && "template declaration must be available");

    if (Printed.insert(CTD).second) {
      emitFwdDecl(O, CTD, KernelLocation);
    }
  } else if (Printed.insert(RD).second) {
    // emit forward declarations for "leaf" classes in the template parameter
    // tree;
    emitFwdDecl(O, RD, KernelLocation);
  }
}

static void emitCPPTypeString(raw_ostream &OS, QualType Ty) {
  LangOptions LO;
  PrintingPolicy P(LO);
  P.SuppressTypedefs = true;
  emitWithoutAnonNamespaces(OS, Ty.getAsString(P));
}

static void printArguments(ASTContext &Ctx, raw_ostream &ArgOS,
                           ArrayRef<TemplateArgument> Args,
                           const PrintingPolicy &P);

static void emitKernelNameType(QualType T, ASTContext &Ctx, raw_ostream &OS,
                               const PrintingPolicy &TypePolicy);

static void printArgument(ASTContext &Ctx, raw_ostream &ArgOS,
                          TemplateArgument Arg, const PrintingPolicy &P) {
  switch (Arg.getKind()) {
  case TemplateArgument::ArgKind::Pack: {
    printArguments(Ctx, ArgOS, Arg.getPackAsArray(), P);
    break;
  }
  case TemplateArgument::ArgKind::Integral: {
    QualType T = Arg.getIntegralType();
    const EnumType *ET = T->getAs<EnumType>();

    if (ET) {
      const llvm::APSInt &Val = Arg.getAsIntegral();
      ArgOS << "static_cast<"
            << ET->getDecl()->getQualifiedNameAsString(
                   /*WithGlobalNsPrefix*/ true)
            << ">"
            << "(" << Val << ")";
    } else {
      Arg.print(P, ArgOS);
    }
    break;
  }
  case TemplateArgument::ArgKind::Type: {
    LangOptions LO;
    PrintingPolicy TypePolicy(LO);
    TypePolicy.SuppressTypedefs = true;
    TypePolicy.SuppressTagKeyword = true;
    QualType T = Arg.getAsType();

    emitKernelNameType(T, Ctx, ArgOS, TypePolicy);
    break;
  }
  case TemplateArgument::ArgKind::Template: {
    TemplateDecl *TD = Arg.getAsTemplate().getAsTemplateDecl();
    ArgOS << TD->getQualifiedNameAsString();
    break;
  }
  default:
    Arg.print(P, ArgOS);
  }
}

static void printArguments(ASTContext &Ctx, raw_ostream &ArgOS,
                           ArrayRef<TemplateArgument> Args,
                           const PrintingPolicy &P) {
  for (unsigned I = 0; I < Args.size(); I++) {
    const TemplateArgument &Arg = Args[I];

    // If argument is an empty pack argument, skip printing comma and argument.
    if (Arg.getKind() == TemplateArgument::ArgKind::Pack && !Arg.pack_size())
      continue;

    if (I != 0)
      ArgOS << ", ";

    printArgument(Ctx, ArgOS, Arg, P);
  }
}

static void printTemplateArguments(ASTContext &Ctx, raw_ostream &ArgOS,
                                   ArrayRef<TemplateArgument> Args,
                                   const PrintingPolicy &P) {
  ArgOS << "<";
  printArguments(Ctx, ArgOS, Args, P);
  ArgOS << ">";
}

static void emitRecordType(raw_ostream &OS, QualType T, const CXXRecordDecl *RD,
                           const PrintingPolicy &TypePolicy) {
  SmallString<64> Buf;
  llvm::raw_svector_ostream RecOS(Buf);
  T.getCanonicalType().getQualifiers().print(RecOS, TypePolicy,
                                             /*appendSpaceIfNotEmpty*/ true);
  if (const auto *TSD = dyn_cast<ClassTemplateSpecializationDecl>(RD)) {

    // Print template class name
    TSD->printQualifiedName(RecOS, TypePolicy, /*WithGlobalNsPrefix*/ true);

    // Print template arguments substituting enumerators
    ASTContext &Ctx = RD->getASTContext();
    const TemplateArgumentList &Args = TSD->getTemplateArgs();
    printTemplateArguments(Ctx, RecOS, Args.asArray(), TypePolicy);

    emitWithoutAnonNamespaces(OS, RecOS.str());
    return;
  }
  if (RD->getDeclContext()->isFunctionOrMethod()) {
    emitWithoutAnonNamespaces(OS, T.getCanonicalType().getAsString(TypePolicy));
    return;
  }

  const NamespaceDecl *NS = dyn_cast<NamespaceDecl>(RD->getDeclContext());
  RD->printQualifiedName(RecOS, TypePolicy,
                         !(NS && NS->isAnonymousNamespace()));
  emitWithoutAnonNamespaces(OS, RecOS.str());
}

static void emitKernelNameType(QualType T, ASTContext &Ctx, raw_ostream &OS,
                               const PrintingPolicy &TypePolicy) {
  if (T->isRecordType()) {
    emitRecordType(OS, T, T->getAsCXXRecordDecl(), TypePolicy);
    return;
  }

  if (T->isEnumeralType())
    OS << "::";
  emitWithoutAnonNamespaces(OS, T.getCanonicalType().getAsString(TypePolicy));
}

void SYCLIntegrationHeader::emit(raw_ostream &O) {
  O << "// This is auto-generated SYCL integration header.\n";
  O << "\n";

  O << "#include <CL/sycl/detail/defines.hpp>\n";
  O << "#include <CL/sycl/detail/kernel_desc.hpp>\n";

  O << "\n";

  if (SpecConsts.size() > 0) {
    // Remove duplicates.
    std::sort(SpecConsts.begin(), SpecConsts.end(),
              [](const SpecConstID &SC1, const SpecConstID &SC2) {
                // Sort by string IDs for stable spec consts order in the
                // header.
                return SC1.second.compare(SC2.second) < 0;
              });
    SpecConstID *End =
        std::unique(SpecConsts.begin(), SpecConsts.end(),
                    [](const SpecConstID &SC1, const SpecConstID &SC2) {
                      // Here can do faster comparison of types.
                      return SC1.first == SC2.first;
                    });
    O << "// Specialization constants IDs:\n";
    for (const auto &P : llvm::make_range(SpecConsts.begin(), End)) {
      O << "template <> struct sycl::detail::SpecConstantInfo<";
      emitCPPTypeString(O, P.first);
      O << "> {\n";
      O << "  static constexpr const char* getName() {\n";
      O << "    return \"" << P.second << "\";\n";
      O << "  }\n";
      O << "};\n";
    }
  }

  if (!UnnamedLambdaSupport) {
    O << "// Forward declarations of templated kernel function types:\n";

    llvm::SmallPtrSet<const void *, 4> Printed;
    for (const KernelDesc &K : KernelDescs) {
      emitForwardClassDecls(O, K.NameType, K.KernelLocation, Printed);
    }
  }
  O << "\n";

  O << "__SYCL_INLINE_NAMESPACE(cl) {\n";
  O << "namespace sycl {\n";
  O << "namespace detail {\n";

  O << "\n";

  O << "// names of all kernels defined in the corresponding source\n";
  O << "static constexpr\n";
  O << "const char* const kernel_names[] = {\n";

  for (unsigned I = 0; I < KernelDescs.size(); I++) {
    O << "  \"" << KernelDescs[I].Name << "\"";

    if (I < KernelDescs.size() - 1)
      O << ",";
    O << "\n";
  }
  O << "};\n\n";

  O << "// array representing signatures of all kernels defined in the\n";
  O << "// corresponding source\n";
  O << "static constexpr\n";
  O << "const kernel_param_desc_t kernel_signatures[] = {\n";

  for (unsigned I = 0; I < KernelDescs.size(); I++) {
    auto &K = KernelDescs[I];
    O << "  //--- " << K.Name << "\n";

    for (const auto &P : K.Params) {
      std::string TyStr = paramKind2Str(P.Kind);
      O << "  { kernel_param_kind_t::" << TyStr << ", ";
      O << P.Info << ", " << P.Offset << " },\n";
    }
    O << "\n";
  }
  O << "};\n\n";

  O << "// indices into the kernel_signatures array, each representing a "
       "start"
       " of\n";
  O << "// kernel signature descriptor subarray of the kernel_signatures"
       " array;\n";
  O << "// the index order in this array corresponds to the kernel name order"
       " in the\n";
  O << "// kernel_names array\n";
  O << "static constexpr\n";
  O << "const unsigned kernel_signature_start[] = {\n";
  unsigned CurStart = 0;

  for (unsigned I = 0; I < KernelDescs.size(); I++) {
    auto &K = KernelDescs[I];
    O << "  " << CurStart;
    if (I < KernelDescs.size() - 1)
      O << ",";
    O << " // " << K.Name << "\n";
    CurStart += K.Params.size() + 1;
  }
  O << "};\n\n";

  O << "// Specializations of KernelInfo for kernel function types:\n";
  CurStart = 0;

  for (const KernelDesc &K : KernelDescs) {
    const size_t N = K.Params.size();
    if (UnnamedLambdaSupport) {
      O << "template <> struct KernelInfoData<";
      O << "'" << K.StableName.front();
      for (char c : StringRef(K.StableName).substr(1))
        O << "', '" << c;
      O << "'> {\n";
    } else {
      LangOptions LO;
      PrintingPolicy P(LO);
      P.SuppressTypedefs = true;
      O << "template <> struct KernelInfo<";
      emitKernelNameType(K.NameType, S.getASTContext(), O, P);
      O << "> {\n";
    }
    O << "  DLL_LOCAL\n";
    O << "  static constexpr const char* getName() { return \"" << K.Name
      << "\"; }\n";
    O << "  DLL_LOCAL\n";
    O << "  static constexpr unsigned getNumParams() { return " << N << "; }\n";
    O << "  DLL_LOCAL\n";
    O << "  static constexpr const kernel_param_desc_t& ";
    O << "getParamDesc(unsigned i) {\n";
    O << "    return kernel_signatures[i+" << CurStart << "];\n";
    O << "  }\n";
    O << "};\n";
    CurStart += N;
  }
  O << "\n";
  O << "} // namespace detail\n";
  O << "} // namespace sycl\n";
  O << "} // __SYCL_INLINE_NAMESPACE(cl)\n";
  O << "\n";
}

bool SYCLIntegrationHeader::emit(const StringRef &IntHeaderName) {
  if (IntHeaderName.empty())
    return false;
  int IntHeaderFD = 0;
  std::error_code EC =
      llvm::sys::fs::openFileForWrite(IntHeaderName, IntHeaderFD);
  if (EC) {
    llvm::errs() << "Error: " << EC.message() << "\n";
    // compilation will fail on absent include file - don't need to fail here
    return false;
  }
  llvm::raw_fd_ostream Out(IntHeaderFD, true /*close in destructor*/);
  emit(Out);
  return true;
}

void SYCLIntegrationHeader::startKernel(StringRef KernelName,
                                        QualType KernelNameType,
                                        StringRef KernelStableName,
                                        SourceLocation KernelLocation) {
  KernelDescs.resize(KernelDescs.size() + 1);
  KernelDescs.back().Name = std::string(KernelName);
  KernelDescs.back().NameType = KernelNameType;
  KernelDescs.back().StableName = std::string(KernelStableName);
  KernelDescs.back().KernelLocation = KernelLocation;
}

void SYCLIntegrationHeader::addParamDesc(kernel_param_kind_t Kind, int Info,
                                         unsigned Offset) {
  auto *K = getCurKernelDesc();
  assert(K && "no kernels");
  K->Params.push_back(KernelParamDesc());
  KernelParamDesc &PD = K->Params.back();
  PD.Kind = Kind;
  PD.Info = Info;
  PD.Offset = Offset;
}

void SYCLIntegrationHeader::endKernel() {
  // nop for now
}

void SYCLIntegrationHeader::addSpecConstant(StringRef IDName, QualType IDType) {
  SpecConsts.emplace_back(std::make_pair(IDType, IDName.str()));
}

SYCLIntegrationHeader::SYCLIntegrationHeader(DiagnosticsEngine &_Diag,
                                             bool _UnnamedLambdaSupport,
                                             Sema &_S)
    : Diag(_Diag), UnnamedLambdaSupport(_UnnamedLambdaSupport), S(_S) {}

// -----------------------------------------------------------------------------
// Utility class methods
// -----------------------------------------------------------------------------

bool Util::isSyclAccessorType(const QualType &Ty) {
  return isSyclType(Ty, "accessor", true /*Tmpl*/);
}

bool Util::isSyclSamplerType(const QualType &Ty) {
  return isSyclType(Ty, "sampler");
}

bool Util::isSyclStreamType(const QualType &Ty) {
  return isSyclType(Ty, "stream");
}

bool Util::isSyclHalfType(const QualType &Ty) {
  const StringRef &Name = "half";
  std::array<DeclContextDesc, 5> Scopes = {
      Util::DeclContextDesc{clang::Decl::Kind::Namespace, "cl"},
      Util::DeclContextDesc{clang::Decl::Kind::Namespace, "sycl"},
      Util::DeclContextDesc{clang::Decl::Kind::Namespace, "detail"},
      Util::DeclContextDesc{clang::Decl::Kind::Namespace, "half_impl"},
      Util::DeclContextDesc{Decl::Kind::CXXRecord, Name}};
  return matchQualifiedTypeName(Ty, Scopes);
}

bool Util::isSyclSpecConstantType(const QualType &Ty) {
  const StringRef &Name = "spec_constant";
  std::array<DeclContextDesc, 4> Scopes = {
      Util::DeclContextDesc{clang::Decl::Kind::Namespace, "cl"},
      Util::DeclContextDesc{clang::Decl::Kind::Namespace, "sycl"},
      Util::DeclContextDesc{clang::Decl::Kind::Namespace, "experimental"},
      Util::DeclContextDesc{Decl::Kind::ClassTemplateSpecialization, Name}};
  return matchQualifiedTypeName(Ty, Scopes);
}

bool Util::isPropertyListType(const QualType &Ty) {
  return isSyclType(Ty, "property_list", true /*Tmpl*/);
}

bool Util::isSyclBufferLocationType(const QualType &Ty) {
  const StringRef &Name = "buffer_location";
  std::array<DeclContextDesc, 4> Scopes = {
      Util::DeclContextDesc{clang::Decl::Kind::Namespace, "cl"},
      Util::DeclContextDesc{clang::Decl::Kind::Namespace, "sycl"},
      // TODO: this doesn't belong to property namespace, instead it shall be
      // in its own namespace. Change it, when the actual implementation in SYCL
      // headers is ready
      Util::DeclContextDesc{clang::Decl::Kind::Namespace, "property"},
      Util::DeclContextDesc{Decl::Kind::ClassTemplateSpecialization, Name}};
  return matchQualifiedTypeName(Ty, Scopes);
}

bool Util::isSyclType(const QualType &Ty, StringRef Name, bool Tmpl) {
  Decl::Kind ClassDeclKind =
      Tmpl ? Decl::Kind::ClassTemplateSpecialization : Decl::Kind::CXXRecord;
  std::array<DeclContextDesc, 3> Scopes = {
      Util::DeclContextDesc{clang::Decl::Kind::Namespace, "cl"},
      Util::DeclContextDesc{clang::Decl::Kind::Namespace, "sycl"},
      Util::DeclContextDesc{ClassDeclKind, Name}};
  return matchQualifiedTypeName(Ty, Scopes);
}

bool Util::matchQualifiedTypeName(const QualType &Ty,
                                  ArrayRef<Util::DeclContextDesc> Scopes) {
  // The idea: check the declaration context chain starting from the type
  // itself. At each step check the context is of expected kind
  // (namespace) and name.
  const CXXRecordDecl *RecTy = Ty->getAsCXXRecordDecl();

  if (!RecTy)
    return false; // only classes/structs supported
  const auto *Ctx = cast<DeclContext>(RecTy);
  StringRef Name = "";

  for (const auto &Scope : llvm::reverse(Scopes)) {
    clang::Decl::Kind DK = Ctx->getDeclKind();

    if (DK != Scope.first)
      return false;

    switch (DK) {
    case clang::Decl::Kind::ClassTemplateSpecialization:
      // ClassTemplateSpecializationDecl inherits from CXXRecordDecl
    case clang::Decl::Kind::CXXRecord:
      Name = cast<CXXRecordDecl>(Ctx)->getName();
      break;
    case clang::Decl::Kind::Namespace:
      Name = cast<NamespaceDecl>(Ctx)->getName();
      break;
    default:
      llvm_unreachable("matchQualifiedTypeName: decl kind not supported");
    }
    if (Name != Scope.second)
      return false;
    Ctx = Ctx->getParent();
  }
  return Ctx->isTranslationUnit();
}
