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
#include "clang/AST/TemplateArgumentVisitor.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/Analysis/CallGraph.h"
#include "clang/Basic/Attributes.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/Version.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/APSInt.h"
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

static constexpr llvm::StringLiteral InitMethodName = "__init";
static constexpr llvm::StringLiteral InitESIMDMethodName = "__init_esimd";
static constexpr llvm::StringLiteral InitSpecConstantsBuffer =
    "__init_specialization_constants_buffer";
static constexpr llvm::StringLiteral FinalizeMethodName = "__finalize";
static constexpr llvm::StringLiteral LibstdcxxFailedAssertion =
    "__failed_assertion";
constexpr unsigned MaxKernelArgsSize = 2048;

namespace {

/// Various utilities.
class Util {
public:
  using DeclContextDesc = std::pair<Decl::Kind, StringRef>;

  template <size_t N>
  static constexpr DeclContextDesc MakeDeclContextDesc(Decl::Kind K,
                                                       const char (&Str)[N]) {
    // FIXME: This SHOULD be able to use the StringLiteral constructor here
    // instead, however this seems to fail with an 'invalid string literal' note
    // on the correct constructor in some build configurations.  We need to
    // figure that out before reverting this to use the StringLiteral
    // constructor.
    return DeclContextDesc{K, StringRef{Str, N - 1}};
  }

  static constexpr DeclContextDesc MakeDeclContextDesc(Decl::Kind K,
                                                       StringRef SR) {
    return DeclContextDesc{K, SR};
  }

  static bool isSyclSpecialType(QualType Ty);

  /// Checks whether given clang type is a full specialization of the SYCL
  /// half class.
  static bool isSyclHalfType(QualType Ty);

  /// Checks whether given clang type is a full specialization of the SYCL
  /// accessor_property_list class.
  static bool isAccessorPropertyListType(QualType Ty);

  /// Checks whether given clang type is a full specialization of the SYCL
  /// no_alias class.
  static bool isSyclAccessorNoAliasPropertyType(QualType Ty);

  /// Checks whether given clang type is a full specialization of the SYCL
  /// buffer_location class.
  static bool isSyclBufferLocationType(QualType Ty);

  /// Checks whether given clang type is a standard SYCL API class with given
  /// name.
  /// \param Ty    the clang type being checked
  /// \param Name  the class name checked against
  /// \param Tmpl  whether the class is template instantiation or simple record
  static bool isSyclType(QualType Ty, StringRef Name, bool Tmpl = false);

  /// Checks whether given clang type is a full specialization of the SYCL
  /// specialization constant class.
  static bool isSyclSpecConstantType(QualType Ty);

  /// Checks whether given clang type is a full specialization of the SYCL
  /// specialization id class.
  static bool isSyclSpecIdType(QualType Ty);

  /// Checks whether given clang type is a full specialization of the SYCL
  /// device_global class.
  static bool isSyclDeviceGlobalType(QualType Ty);

  /// Checks whether given clang type is a full specialization of the SYCL
  /// kernel_handler class.
  static bool isSyclKernelHandlerType(QualType Ty);

  // Checks declaration context hierarchy.
  /// \param DC     the context of the item to be checked.
  /// \param Scopes the declaration scopes leading from the item context to the
  ///               translation unit (excluding the latter)
  static bool matchContext(const DeclContext *DC,
                           ArrayRef<Util::DeclContextDesc> Scopes);

  /// Checks whether given clang type is declared in the given hierarchy of
  /// declaration contexts.
  /// \param Ty         the clang type being checked
  /// \param Scopes     the declaration scopes leading from the type to the
  ///     translation unit (excluding the latter)
  static bool matchQualifiedTypeName(QualType Ty,
                                     ArrayRef<Util::DeclContextDesc> Scopes);
};

} // anonymous namespace

ExprResult Sema::ActOnSYCLBuiltinNumFieldsExpr(ParsedType PT) {
  TypeSourceInfo *TInfo = nullptr;
  QualType QT = GetTypeFromParser(PT, &TInfo);
  assert(TInfo && "couldn't get type info from a type from the parser?");
  SourceLocation TypeLoc = TInfo->getTypeLoc().getBeginLoc();

  return BuildSYCLBuiltinNumFieldsExpr(TypeLoc, QT);
}

ExprResult Sema::BuildSYCLBuiltinNumFieldsExpr(SourceLocation Loc,
                                               QualType SourceTy) {
  if (!SourceTy->isDependentType()) {
    if (RequireCompleteType(Loc, SourceTy,
                            diag::err_sycl_type_trait_requires_complete_type,
                            /*__builtin_num_fields*/ 0))
      return ExprError();

    if (!SourceTy->isRecordType()) {
      Diag(Loc, diag::err_sycl_type_trait_requires_record_type)
          << /*__builtin_num_fields*/ 0;
      return ExprError();
    }
  }
  return new (Context)
      SYCLBuiltinNumFieldsExpr(Loc, SourceTy, Context.getSizeType());
}

ExprResult Sema::ActOnSYCLBuiltinFieldTypeExpr(ParsedType PT, Expr *Idx) {
  TypeSourceInfo *TInfo = nullptr;
  QualType QT = GetTypeFromParser(PT, &TInfo);
  assert(TInfo && "couldn't get type info from a type from the parser?");
  SourceLocation TypeLoc = TInfo->getTypeLoc().getBeginLoc();

  return BuildSYCLBuiltinFieldTypeExpr(TypeLoc, QT, Idx);
}

ExprResult Sema::BuildSYCLBuiltinFieldTypeExpr(SourceLocation Loc,
                                               QualType SourceTy, Expr *Idx) {
  // If the expression appears in an evaluated context, we want to give an
  // error so that users don't attempt to use the value of this expression.
  if (!isUnevaluatedContext()) {
    Diag(Loc, diag::err_sycl_builtin_type_trait_evaluated)
        << /*__builtin_field_type*/ 0;
    return ExprError();
  }

  // We may not be able to calculate the field type (the source type may be a
  // dependent type), so use the source type as a basic fallback. This will
  // ensure that the AST node will have a dependent type that gets resolved
  // later to the real type.
  QualType FieldTy = SourceTy;
  ExprValueKind ValueKind = VK_PRValue;
  if (!SourceTy->isDependentType()) {
    if (RequireCompleteType(Loc, SourceTy,
                            diag::err_sycl_type_trait_requires_complete_type,
                            /*__builtin_field_type*/ 1))
      return ExprError();

    if (!SourceTy->isRecordType()) {
      Diag(Loc, diag::err_sycl_type_trait_requires_record_type)
          << /*__builtin_field_type*/ 1;
      return ExprError();
    }

    if (!Idx->isValueDependent()) {
      Optional<llvm::APSInt> IdxVal = Idx->getIntegerConstantExpr(Context);
      if (IdxVal) {
        RecordDecl *RD = SourceTy->getAsRecordDecl();
        assert(RD && "Record type but no record decl?");
        int64_t Index = IdxVal->getExtValue();

        if (Index < 0) {
          Diag(Idx->getExprLoc(),
               diag::err_sycl_type_trait_requires_nonnegative_index)
              << /*fields*/ 0;
          return ExprError();
        }

        // Ensure that the index is within range.
        int64_t NumFields = std::distance(RD->field_begin(), RD->field_end());
        if (Index >= NumFields) {
          Diag(Idx->getExprLoc(),
               diag::err_sycl_builtin_type_trait_index_out_of_range)
              << toString(*IdxVal, 10) << SourceTy << /*fields*/ 0;
          return ExprError();
        }
        const FieldDecl *FD = *std::next(RD->field_begin(), Index);
        FieldTy = FD->getType();

        // If the field type was a reference type, adjust it now.
        if (FieldTy->isLValueReferenceType()) {
          ValueKind = VK_LValue;
          FieldTy = FieldTy.getNonReferenceType();
        } else if (FieldTy->isRValueReferenceType()) {
          ValueKind = VK_XValue;
          FieldTy = FieldTy.getNonReferenceType();
        }
      }
    }
  }
  return new (Context)
      SYCLBuiltinFieldTypeExpr(Loc, SourceTy, Idx, FieldTy, ValueKind);
}

ExprResult Sema::ActOnSYCLBuiltinNumBasesExpr(ParsedType PT) {
  TypeSourceInfo *TInfo = nullptr;
  QualType QT = GetTypeFromParser(PT, &TInfo);
  assert(TInfo && "couldn't get type info from a type from the parser?");
  SourceLocation TypeLoc = TInfo->getTypeLoc().getBeginLoc();

  return BuildSYCLBuiltinNumBasesExpr(TypeLoc, QT);
}

ExprResult Sema::BuildSYCLBuiltinNumBasesExpr(SourceLocation Loc,
                                              QualType SourceTy) {
  if (!SourceTy->isDependentType()) {
    if (RequireCompleteType(Loc, SourceTy,
                            diag::err_sycl_type_trait_requires_complete_type,
                            /*__builtin_num_bases*/ 2))
      return ExprError();

    if (!SourceTy->isRecordType()) {
      Diag(Loc, diag::err_sycl_type_trait_requires_record_type)
          << /*__builtin_num_bases*/ 2;
      return ExprError();
    }
  }
  return new (Context)
      SYCLBuiltinNumBasesExpr(Loc, SourceTy, Context.getSizeType());
}

ExprResult Sema::ActOnSYCLBuiltinBaseTypeExpr(ParsedType PT, Expr *Idx) {
  TypeSourceInfo *TInfo = nullptr;
  QualType QT = GetTypeFromParser(PT, &TInfo);
  assert(TInfo && "couldn't get type info from a type from the parser?");
  SourceLocation TypeLoc = TInfo->getTypeLoc().getBeginLoc();

  return BuildSYCLBuiltinBaseTypeExpr(TypeLoc, QT, Idx);
}

ExprResult Sema::BuildSYCLBuiltinBaseTypeExpr(SourceLocation Loc,
                                              QualType SourceTy, Expr *Idx) {
  // If the expression appears in an evaluated context, we want to give an
  // error so that users don't attempt to use the value of this expression.
  if (!isUnevaluatedContext()) {
    Diag(Loc, diag::err_sycl_builtin_type_trait_evaluated)
        << /*__builtin_base_type*/ 1;
    return ExprError();
  }

  // We may not be able to calculate the base type (the source type may be a
  // dependent type), so use the source type as a basic fallback. This will
  // ensure that the AST node will have a dependent type that gets resolved
  // later to the real type.
  QualType BaseTy = SourceTy;
  if (!SourceTy->isDependentType()) {
    if (RequireCompleteType(Loc, SourceTy,
                            diag::err_sycl_type_trait_requires_complete_type,
                            /*__builtin_base_type*/ 3))
      return ExprError();

    if (!SourceTy->isRecordType()) {
      Diag(Loc, diag::err_sycl_type_trait_requires_record_type)
          << /*__builtin_base_type*/ 3;
      return ExprError();
    }

    if (!Idx->isValueDependent()) {
      Optional<llvm::APSInt> IdxVal = Idx->getIntegerConstantExpr(Context);
      if (IdxVal) {
        CXXRecordDecl *RD = SourceTy->getAsCXXRecordDecl();
        assert(RD && "Record type but no record decl?");
        int64_t Index = IdxVal->getExtValue();

        if (Index < 0) {
          Diag(Idx->getExprLoc(),
               diag::err_sycl_type_trait_requires_nonnegative_index)
              << /*bases*/ 1;
          return ExprError();
        }

        // Ensure that the index is within range.
        if (Index >= RD->getNumBases()) {
          Diag(Idx->getExprLoc(),
               diag::err_sycl_builtin_type_trait_index_out_of_range)
              << toString(*IdxVal, 10) << SourceTy << /*bases*/ 1;
          return ExprError();
        }

        const CXXBaseSpecifier &Spec = *std::next(RD->bases_begin(), Index);
        BaseTy = Spec.getType();
      }
    }
  }
  return new (Context) SYCLBuiltinBaseTypeExpr(Loc, SourceTy, Idx, BaseTy);
}

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
  case Builtin::BIlroundf:
  case Builtin::BI__builtin_lroundf:
    return false;
  default:
    break;
  }
  return true;
}

bool Sema::isKnownGoodSYCLDecl(const Decl *D) {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    const IdentifierInfo *II = FD->getIdentifier();
    // Allow to use `::printf` only for CUDA.
    if (Context.getTargetInfo().getTriple().isNVPTX()) {
      if (FD->getBuiltinID() == Builtin::BIprintf)
        return true;
    }
    const DeclContext *DC = FD->getDeclContext();
    if (II && II->isStr("__spirv_ocl_printf") &&
        !FD->isDefined() &&
        FD->getLanguageLinkage() == CXXLanguageLinkage &&
        DC->getEnclosingNamespaceContext()->isTranslationUnit())
      return true;
  }
  return false;
}

static bool isZeroSizedArray(Sema &SemaRef, QualType Ty) {
  if (const auto *CAT = SemaRef.getASTContext().getAsConstantArrayType(Ty))
    return CAT->getSize() == 0;
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
  if (isZeroSizedArray(S, Ty)) {
    S.SYCLDiagIfDeviceCode(Loc.getBegin(), diag::err_typecheck_zero_array_size)
        << 1;
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

static bool isSYCLUndefinedAllowed(const FunctionDecl *Callee,
                                   const SourceManager &SrcMgr) {
  if (!Callee)
    return false;

  // The check below requires declaration name, make sure we have it.
  if (!Callee->getIdentifier())
    return false;

  // libstdc++-11 introduced an undefined function "void __failed_assertion()"
  // which may lead to SemaSYCL check failure. However, this undefined function
  // is used to trigger some compilation error when the check fails at compile
  // time and will be ignored when the check succeeds. We allow calls to this
  // function to support some important std functions in SYCL device.
  return (Callee->getName() == LibstdcxxFailedAssertion) &&
         Callee->getNumParams() == 0 && Callee->getReturnType()->isVoidType() &&
         SrcMgr.isInSystemHeader(Callee->getLocation());
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

// Collect function attributes related to SYCL.
static void collectSYCLAttributes(Sema &S, FunctionDecl *FD,
                                  llvm::SmallVectorImpl<Attr *> &Attrs,
                                  bool DirectlyCalled) {
  if (!FD->hasAttrs())
    return;

  // In SYCL 1.2.1 mode, the attributes are propagated from the function they
  // are applied to onto the kernel which calls the function.
  // In SYCL 2020 mode, the attributes are not propagated to the kernel.
  if (DirectlyCalled || S.getASTContext().getLangOpts().getSYCLVersion() <
                            LangOptions::SYCL_2020) {
    llvm::copy_if(FD->getAttrs(), std::back_inserter(Attrs), [](Attr *A) {
      // FIXME: Make this list self-adapt as new SYCL attributes are added.
      return isa<IntelReqdSubGroupSizeAttr, IntelNamedSubGroupSizeAttr,
                 ReqdWorkGroupSizeAttr, WorkGroupSizeHintAttr,
                 SYCLIntelKernelArgsRestrictAttr, SYCLIntelNumSimdWorkItemsAttr,
                 SYCLIntelSchedulerTargetFmaxMhzAttr,
                 SYCLIntelMaxWorkGroupSizeAttr, SYCLIntelMaxGlobalWorkDimAttr,
                 SYCLIntelNoGlobalWorkOffsetAttr, SYCLSimdAttr>(A);
    });
  }

  // Attributes that should not be propagated from device functions to a kernel.
  if (DirectlyCalled) {
    llvm::copy_if(FD->getAttrs(), std::back_inserter(Attrs), [](Attr *A) {
      return isa<SYCLIntelLoopFuseAttr, SYCLIntelFPGAMaxConcurrencyAttr,
                 SYCLIntelFPGADisableLoopPipeliningAttr,
                 SYCLIntelFPGAInitiationIntervalAttr,
                 SYCLIntelUseStallEnableClustersAttr, SYCLDeviceHasAttr>(A);
    });
  }
}

class DiagDeviceFunction : public RecursiveASTVisitor<DiagDeviceFunction> {
  Sema &SemaRef;
  const llvm::SmallPtrSetImpl<const FunctionDecl *> &RecursiveFuncs;

public:
  DiagDeviceFunction(
      Sema &S,
      const llvm::SmallPtrSetImpl<const FunctionDecl *> &RecursiveFuncs)
      : RecursiveASTVisitor(), SemaRef(S), RecursiveFuncs(RecursiveFuncs) {}

  void CheckBody(Stmt *ToBeDiagnosed) { TraverseStmt(ToBeDiagnosed); }

  bool VisitCallExpr(CallExpr *e) {
    if (FunctionDecl *Callee = e->getDirectCallee()) {
      Callee = Callee->getCanonicalDecl();
      assert(Callee && "Device function canonical decl must be available");

      // Remember that all SYCL kernel functions have deferred
      // instantiation as template functions. It means that
      // all functions used by kernel have already been parsed and have
      // definitions.
      if (RecursiveFuncs.count(Callee)) {
        SemaRef.Diag(e->getExprLoc(), diag::err_sycl_restrict)
            << Sema::KernelCallRecursiveFunction;
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
               !isa<CXXPseudoDestructorExpr>(e->getCallee())) {
      bool MaybeConstantExpr = false;
      Expr *NonDirectCallee = e->getCallee();
      if (!NonDirectCallee->isValueDependent())
        MaybeConstantExpr =
            NonDirectCallee->isCXX11ConstantExpr(SemaRef.getASTContext());
      if (!MaybeConstantExpr)
        SemaRef.Diag(e->getExprLoc(), diag::err_sycl_restrict)
            << Sema::KernelCallFunctionPointer;
    }
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
    if (VD->isConstexpr())
      return true;
    return RecursiveASTVisitor::TraverseVarDecl(VD);
  }

  // Skip checking rules on template arguments, since these are constant
  // expressions.
  bool TraverseTemplateArgumentLoc(const TemplateArgumentLoc &ArgLoc) {
    return true;
  }

  // Skip checking the static assert, both components are required to be
  // constant expressions.
  bool TraverseStaticAssertDecl(StaticAssertDecl *D) { return true; }

  // Make sure we skip the condition of the case, since that is a constant
  // expression.
  bool TraverseCaseStmt(CaseStmt *S) {
    return TraverseStmt(S->getSubStmt());
  }

  // Skip checking the size expr, since a constant array type loc's size expr is
  // a constant expression.
  bool TraverseConstantArrayTypeLoc(const ConstantArrayTypeLoc &ArrLoc) {
    return true;
  }

  bool TraverseIfStmt(IfStmt *S) {
    if (llvm::Optional<Stmt *> ActiveStmt =
            S->getNondiscardedCase(SemaRef.Context)) {
      if (*ActiveStmt)
        return TraverseStmt(*ActiveStmt);
      return true;
    }
    return RecursiveASTVisitor::TraverseIfStmt(S);
  }
};

// This type manages the list of device functions and recursive functions, as
// well as an entry point for attribute collection, for the translation unit
// during MarkDevices. On construction, this type makes sure that all of the
// root-device functions, (that is, those marked with SYCL_EXTERNAL) are
// collected.  On destruction, it manages and runs the diagnostics required.
// When processing individual kernel/external functions, the
// SingleDeviceFunctionTracker type updates this type.
class DeviceFunctionTracker {
  friend class SingleDeviceFunctionTracker;
  CallGraph CG;
  Sema &SemaRef;
  // The list of functions used on the device, kept so we can diagnose on them
  // later.
  llvm::SmallPtrSet<FunctionDecl *, 16> DeviceFunctions;
  llvm::SmallPtrSet<const FunctionDecl *, 16> RecursiveFunctions;

  void CollectSyclExternalFuncs() {
    for (CallGraphNode::CallRecord Record : CG.getRoot()->callees())
      if (auto *FD = dyn_cast<FunctionDecl>(Record.Callee->getDecl()))
        if (FD->hasBody() && FD->hasAttr<SYCLDeviceAttr>())
          SemaRef.addSyclDeviceDecl(FD);
  }

  CallGraphNode *getNodeForKernel(FunctionDecl *Kernel) {
    assert(CG.getNode(Kernel) && "No call graph entry for a kernel?");
    return CG.getNode(Kernel);
  }

  void AddSingleFunction(
      const llvm::SmallPtrSetImpl<FunctionDecl *> &DevFuncs,
      const llvm::SmallPtrSetImpl<const FunctionDecl *> &Recursive) {
    DeviceFunctions.insert(DevFuncs.begin(), DevFuncs.end());
    RecursiveFunctions.insert(Recursive.begin(), Recursive.end());
  }

public:
  DeviceFunctionTracker(Sema &S) : SemaRef(S) {
    CG.setSkipConstantExpressions(S.Context);
    CG.addToCallGraph(S.getASTContext().getTranslationUnitDecl());
    CollectSyclExternalFuncs();
  }

  ~DeviceFunctionTracker() {
    DiagDeviceFunction Diagnoser{SemaRef, RecursiveFunctions};
    for (const FunctionDecl *FD : DeviceFunctions)
      if (const FunctionDecl *Def = FD->getDefinition())
        Diagnoser.CheckBody(Def->getBody());
  }
};

// This type does the heavy lifting for the management of device functions,
// recursive function detection, and attribute collection for a single
// kernel/external function. It walks the callgraph to find all functions that
// are called, marks the recursive-functions, and figures out the list of
// attributes that apply to this kernel.
//
// Upon destruction, this type updates the DeviceFunctionTracker.
class SingleDeviceFunctionTracker {
  DeviceFunctionTracker &Parent;
  FunctionDecl *SYCLKernel = nullptr;
  FunctionDecl *KernelBody = nullptr;
  llvm::SmallPtrSet<FunctionDecl *, 16> DeviceFunctions;
  llvm::SmallPtrSet<const FunctionDecl *, 16> RecursiveFunctions;
  llvm::SmallVector<Attr *> CollectedAttributes;

  FunctionDecl *GetFDFromNode(CallGraphNode *Node) {
    FunctionDecl *FD = Node->getDecl()->getAsFunction();
    if (!FD)
      return nullptr;

    return FD->getMostRecentDecl();
  }

  void VisitCallNode(CallGraphNode *Node, FunctionDecl *CurrentDecl,
                     llvm::SmallVectorImpl<FunctionDecl *> &CallStack) {
    // If this isn't a function, I don't think there is anything we can do here.
    if (!CurrentDecl)
      return;

    // Determine if this is a recursive function. If so, we're done.
    if (llvm::is_contained(CallStack, CurrentDecl)) {
      RecursiveFunctions.insert(CurrentDecl->getCanonicalDecl());
      return;
    }

    // If this is a routine that is not defined and it does not have either
    // a SYCLKernel or SYCLDevice attribute on it, add it to the set of
    // routines potentially reachable on device. This is to diagnose such
    // cases later in finalizeSYCLDeviceAnalysis().
    if (!CurrentDecl->isDefined() && !CurrentDecl->hasAttr<SYCLKernelAttr>() &&
        !CurrentDecl->hasAttr<SYCLDeviceAttr>())
      Parent.SemaRef.addFDToReachableFromSyclDevice(CurrentDecl,
                                                    CallStack.back());

    // We previously thought we could skip this function if we'd seen it before,
    // but if we haven't seen it before in this call graph, we can end up
    // missing a recursive call.  SO, we have to revisit call-graphs we've
    // already seen, just in case it ALSO has recursion.  For example:
    // void recurse1();
    // void recurse2() { recurse1(); }
    // void recurse1() { recurse2(); }
    // void CallerInKernel() { recurse1(); recurse2(); }
    // When checking 'recurse1', we'd have ended up 'visiting' recurse2 without
    // realizing it was recursive, since we never went into the
    // child-of-its-child, since THAT was recursive and exited early out of
    // necessity.
    // Then when we go to visit the kernel's call to recurse2, we would
    // immediately escape not noticing it was recursive. SO, we have to do a
    // little extra work in this case, and make sure we visit the entire call
    // graph.
    DeviceFunctions.insert(CurrentDecl);

    // Collect attributes for functions that aren't the root kernel.
    if (!CallStack.empty()) {
      bool DirectlyCalled = CallStack.size() == 1;
      collectSYCLAttributes(Parent.SemaRef, CurrentDecl, CollectedAttributes,
                            DirectlyCalled);
    }

    // Calculate the kernel body.  Note the 'isSYCLKernelBodyFunction' only
    // tests that it is operator(), so hopefully this doesn't get us too many
    // false-positives.
    if (isSYCLKernelBodyFunction(CurrentDecl)) {
      // This is a direct callee of the kernel.
      if (CallStack.size() == 1 &&
          CallStack.back()->hasAttr<SYCLKernelAttr>()) {
        assert(!KernelBody && "inconsistent call graph - only one kernel body "
                              "function can be called");
        KernelBody = CurrentDecl;
      } else if (CallStack.size() == 2 && KernelBody == CallStack.back()) {
        // To implement rounding-up of a parallel-for range the
        // SYCL header implementation modifies the kernel call like this:
        // auto Wrapper = [=](TransformedArgType Arg) {
        //  if (Arg[0] >= NumWorkItems[0])
        //    return;
        //  Arg.set_allowed_range(NumWorkItems);
        //  KernelFunc(Arg);
        // };
        //
        // This transformation leads to a condition where a kernel body
        // function becomes callable from a new kernel body function.
        // Hence this test.
        // FIXME: We need to be more selective here, this can be hit by simply
        // having a kernel lambda with a lambda call inside of it.
        KernelBody = CurrentDecl;
      }
    }

    // Recurse.
    CallStack.push_back(CurrentDecl);
    llvm::SmallPtrSet<FunctionDecl *, 16> SeenCallees;
    for (CallGraphNode *CI : Node->callees()) {
      FunctionDecl *CurFD = GetFDFromNode(CI);

      // Make sure we only visit each callee 1x from this function to avoid very
      // time consuming template recursion cases.
      if (!llvm::is_contained(SeenCallees, CurFD)) {
        VisitCallNode(CI, CurFD, CallStack);
        SeenCallees.insert(CurFD);
      }
    }
    CallStack.pop_back();
  }

  // Function to walk the call graph and identify the important information.
  void Init() {
    CallGraphNode *KernelNode = Parent.getNodeForKernel(SYCLKernel);
    llvm::SmallVector<FunctionDecl *> CallStack;
    VisitCallNode(KernelNode, GetFDFromNode(KernelNode), CallStack);
  }

public:
  SingleDeviceFunctionTracker(DeviceFunctionTracker &P, Decl *Kernel)
      : Parent(P), SYCLKernel(Kernel->getAsFunction()) {
    Init();
  }

  FunctionDecl *GetSYCLKernel() { return SYCLKernel; }

  FunctionDecl *GetKernelBody() { return KernelBody; }

  llvm::SmallVectorImpl<Attr *> &GetCollectedAttributes() {
    return CollectedAttributes;
  }

  llvm::SmallPtrSetImpl<FunctionDecl *> &GetDeviceFunctions() {
    return DeviceFunctions;
  }

  ~SingleDeviceFunctionTracker() {
    Parent.AddSingleFunction(DeviceFunctions, RecursiveFunctions);
  }
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
          DRE->getTemplateKeywordLoc(), NewDecl, false, DRE->getNameInfo(),
          NewDecl->getType(), DRE->getValueKind());
    }
    return DRE;
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

    unsigned int NumPFWGLambdaArgs =
        M->getNumParams() + 1; // group, optional kernel_handler and lambda obj
    if (Call->getNumArgs() != NumPFWGLambdaArgs)
      return true;
    if (!Util::isSyclType(Call->getArg(1)->getType(), "group", true /*Tmpl*/))
      return true;
    if ((Call->getNumArgs() > 2) &&
        !Util::isSyclKernelHandlerType(Call->getArg(2)->getType()))
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
                                      StringRef MethodName) {
  CXXMethodDecl *Method;
  auto It = std::find_if(CRD->methods().begin(), CRD->methods().end(),
                         [MethodName](const CXXMethodDecl *Method) {
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

// The SYCL kernel's 'object type' used for diagnostics and naming/mangling is
// the first parameter to a function template using the sycl_kernel
// attribute. In SYCL 1.2.1, this was passed by value,
// and in SYCL 2020, it is passed by reference.
static QualType GetSYCLKernelObjectType(const FunctionDecl *KernelCaller) {
  assert(KernelCaller->getNumParams() > 0 && "Insufficient kernel parameters");
  QualType KernelParamTy = KernelCaller->getParamDecl(0)->getType();

  // SYCL 2020 kernels are passed by reference.
  if (KernelParamTy->isReferenceType())
    KernelParamTy = KernelParamTy->getPointeeType();

  // SYCL 1.2.1
  return KernelParamTy.getUnqualifiedType();
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

static ParamDesc makeParamDesc(ASTContext &Ctx, StringRef Name, QualType Ty) {
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
                                        const FunctionDecl *KernelCallerFunc) {
  const TemplateArgumentList *TAL =
      KernelCallerFunc->getTemplateSpecializationArgs();
  assert(TAL && "No template argument info");
  return TAL->get(0).getAsType().getCanonicalType();
}

// Gets a name for the OpenCL kernel function, calculated from the first
// template argument of the kernel caller function.
static std::pair<std::string, std::string>
constructKernelName(Sema &S, const FunctionDecl *KernelCallerFunc,
                    MangleContext &MC) {
  QualType KernelNameType =
      calculateKernelNameType(S.getASTContext(), KernelCallerFunc);

  SmallString<256> Result;
  llvm::raw_svector_ostream Out(Result);

  MC.mangleTypeName(KernelNameType, Out);

  return {std::string(Out.str()), SYCLUniqueStableNameExpr::ComputeName(
                                      S.getASTContext(), KernelNameType)};
}

static bool isDefaultSPIRArch(ASTContext &Context) {
  llvm::Triple T = Context.getTargetInfo().getTriple();
  if (T.isSPIR() && T.getSubArch() == llvm::Triple::NoSubArch)
    return true;
  return false;
}

static ParmVarDecl *getSyclKernelHandlerArg(FunctionDecl *KernelCallerFunc) {
  // Specialization constants in SYCL 2020 are not captured by lambda and
  // accessed through new optional lambda argument kernel_handler
  auto IsHandlerLambda = [](ParmVarDecl *PVD) {
    return Util::isSyclKernelHandlerType(PVD->getType());
  };

  assert(llvm::count_if(KernelCallerFunc->parameters(), IsHandlerLambda) <= 1 &&
         "Multiple kernel_handler parameters");

  auto KHArg = llvm::find_if(KernelCallerFunc->parameters(), IsHandlerLambda);

  return (KHArg != KernelCallerFunc->param_end()) ? *KHArg : nullptr;
}

static bool isReadOnlyAccessor(const TemplateArgument &AccessModeArg) {
  const auto *AccessModeArgEnumType =
      AccessModeArg.getIntegralType()->castAs<EnumType>();
  const EnumDecl *ED = AccessModeArgEnumType->getDecl();

  auto ReadOnly =
      llvm::find_if(ED->enumerators(), [&](const EnumConstantDecl *E) {
        return E->getName() == "read";
      });

  return ReadOnly != ED->enumerator_end() &&
         (*ReadOnly)->getInitVal() == AccessModeArg.getAsIntegral();
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

  template <typename ParentTy, typename... HandlerTys>
  void VisitUnionImpl(const CXXRecordDecl *Owner, ParentTy &Parent,
                      const CXXRecordDecl *Wrapper, HandlerTys &... Handlers) {
    (void)std::initializer_list<int>{
        (Handlers.enterUnion(Owner, Parent), 0)...};
    VisitRecordHelper(Wrapper, Wrapper->fields(), Handlers...);
    (void)std::initializer_list<int>{
        (Handlers.leaveUnion(Owner, Parent), 0)...};
  }

  // These enable handler execution only when previous Handlers succeed.
  template <typename... Tn>
  bool handleField(FieldDecl *FD, QualType FDTy, Tn &&... tn) {
    bool result = true;
    (void)std::initializer_list<int>{(result = result && tn(FD, FDTy), 0)...};
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
      std::bind(static_cast<bool (std::decay_t<decltype(Handlers)>::*)(        \
                    bind_param_t<decltype(Item)>, QualType)>(                  \
                    &std::decay_t<decltype(Handlers)>::FUNC),                  \
                std::ref(Handlers), _1, _2)...)

  // The following simpler definition works with gcc 8.x and later.
  //#define KF_FOR_EACH(FUNC) \
//  handleField(Field, FieldTy, ([&](FieldDecl *FD, QualType FDTy) { \
//                return Handlers.f(FD, FDTy); \
//              })...)

  // Parent contains the FieldDecl or CXXBaseSpecifier that was used to enter
  // the Wrapper structure that we're currently visiting. Owner is the parent
  // type (which doesn't exist in cases where it is a FieldDecl in the
  // 'root'), and Wrapper is the current struct being unwrapped.
  template <typename ParentTy, typename... HandlerTys>
  void visitComplexRecord(const CXXRecordDecl *Owner, ParentTy &Parent,
                          const CXXRecordDecl *Wrapper, QualType RecordTy,
                          HandlerTys &... Handlers) {
    (void)std::initializer_list<int>{
        (Handlers.enterStruct(Owner, Parent, RecordTy), 0)...};
    VisitRecordHelper(Wrapper, Wrapper->bases(), Handlers...);
    VisitRecordHelper(Wrapper, Wrapper->fields(), Handlers...);
    (void)std::initializer_list<int>{
        (Handlers.leaveStruct(Owner, Parent, RecordTy), 0)...};
  }

  template <typename ParentTy, typename... HandlerTys>
  void visitSimpleRecord(const CXXRecordDecl *Owner, ParentTy &Parent,
                         const CXXRecordDecl *Wrapper, QualType RecordTy,
                         HandlerTys &... Handlers) {
    (void)std::initializer_list<int>{
        (Handlers.handleNonDecompStruct(Owner, Parent, RecordTy), 0)...};
  }

  template <typename ParentTy, typename... HandlerTys>
  void visitRecord(const CXXRecordDecl *Owner, ParentTy &Parent,
                   const CXXRecordDecl *Wrapper, QualType RecordTy,
                   HandlerTys &... Handlers);

  template <typename ParentTy, typename... HandlerTys>
  void VisitUnion(const CXXRecordDecl *Owner, ParentTy &Parent,
                  const CXXRecordDecl *Wrapper, HandlerTys &... Handlers);

  template <typename... HandlerTys>
  void VisitRecordHelper(const CXXRecordDecl *Owner,
                         clang::CXXRecordDecl::base_class_const_range Range,
                         HandlerTys &... Handlers) {
    for (const auto &Base : Range) {
      QualType BaseTy = Base.getType();
      // Handle accessor class as base
      if (Util::isSyclSpecialType(BaseTy)) {
        (void)std::initializer_list<int>{
            (Handlers.handleSyclSpecialType(Owner, Base, BaseTy), 0)...};
      } else
        // For all other bases, visit the record
        visitRecord(Owner, Base, BaseTy->getAsCXXRecordDecl(), BaseTy,
                    Handlers...);
    }
  }

  template <typename... HandlerTys>
  void VisitRecordHelper(const CXXRecordDecl *Owner,
                         RecordDecl::field_range Range,
                         HandlerTys &... Handlers) {
    VisitRecordFields(Owner, Handlers...);
  }

  template <typename... HandlerTys>
  void visitArrayElementImpl(const CXXRecordDecl *Owner, FieldDecl *ArrayField,
                             QualType ElementTy, uint64_t Index,
                             HandlerTys &... Handlers) {
    (void)std::initializer_list<int>{
        (Handlers.nextElement(ElementTy, Index), 0)...};
    visitField(Owner, ArrayField, ElementTy, Handlers...);
  }

  template <typename... HandlerTys>
  void visitFirstArrayElement(const CXXRecordDecl *Owner, FieldDecl *ArrayField,
                              QualType ElementTy, HandlerTys &... Handlers) {
    visitArrayElementImpl(Owner, ArrayField, ElementTy, 0, Handlers...);
  }
  template <typename... HandlerTys>
  void visitNthArrayElement(const CXXRecordDecl *Owner, FieldDecl *ArrayField,
                            QualType ElementTy, uint64_t Index,
                            HandlerTys &... Handlers);

  template <typename... HandlerTys>
  void visitSimpleArray(const CXXRecordDecl *Owner, FieldDecl *Field,
                        QualType ArrayTy, HandlerTys &... Handlers) {
    (void)std::initializer_list<int>{
        (Handlers.handleSimpleArrayType(Field, ArrayTy), 0)...};
  }

  template <typename... HandlerTys>
  void visitComplexArray(const CXXRecordDecl *Owner, FieldDecl *Field,
                         QualType ArrayTy, HandlerTys &... Handlers) {
    // Array workflow is:
    // handleArrayType
    // enterArray
    // nextElement
    // VisitField (same as before, note that The FieldDecl is the of array
    // itself, not the element)
    // ... repeat per element, opt-out for duplicates.
    // leaveArray

    if (!KF_FOR_EACH(handleArrayType, Field, ArrayTy))
      return;

    const ConstantArrayType *CAT =
        SemaRef.getASTContext().getAsConstantArrayType(ArrayTy);
    assert(CAT && "Should only be called on constant-size array.");
    QualType ET = CAT->getElementType();
    uint64_t ElemCount = CAT->getSize().getZExtValue();

    (void)std::initializer_list<int>{
        (Handlers.enterArray(Field, ArrayTy, ET), 0)...};

    visitFirstArrayElement(Owner, Field, ET, Handlers...);
    for (uint64_t Index = 1; Index < ElemCount; ++Index)
      visitNthArrayElement(Owner, Field, ET, Index, Handlers...);

    (void)std::initializer_list<int>{
        (Handlers.leaveArray(Field, ArrayTy, ET), 0)...};
  }

  template <typename... HandlerTys>
  void visitArray(const CXXRecordDecl *Owner, FieldDecl *Field,
                  QualType ArrayTy, HandlerTys &... Handlers);

  template <typename... HandlerTys>
  void visitField(const CXXRecordDecl *Owner, FieldDecl *Field,
                  QualType FieldTy, HandlerTys &... Handlers) {
    if (Util::isSyclSpecialType(FieldTy))
      KF_FOR_EACH(handleSyclSpecialType, Field, FieldTy);
    else if (Util::isSyclHalfType(FieldTy))
      KF_FOR_EACH(handleSyclHalfType, Field, FieldTy);
    else if (Util::isSyclSpecConstantType(FieldTy))
      KF_FOR_EACH(handleSyclSpecConstantType, Field, FieldTy);
    else if (FieldTy->isStructureOrClassType()) {
      if (KF_FOR_EACH(handleStructType, Field, FieldTy)) {
        CXXRecordDecl *RD = FieldTy->getAsCXXRecordDecl();
        visitRecord(Owner, Field, RD, FieldTy, Handlers...);
      }
    } else if (FieldTy->isUnionType()) {
      if (KF_FOR_EACH(handleUnionType, Field, FieldTy)) {
        CXXRecordDecl *RD = FieldTy->getAsCXXRecordDecl();
        VisitUnion(Owner, Field, RD, Handlers...);
      }
    } else if (FieldTy->isReferenceType())
      KF_FOR_EACH(handleReferenceType, Field, FieldTy);
    else if (FieldTy->isPointerType())
      KF_FOR_EACH(handlePointerType, Field, FieldTy);
    else if (FieldTy->isArrayType())
      visitArray(Owner, Field, FieldTy, Handlers...);
    else if (FieldTy->isScalarType() || FieldTy->isVectorType())
      KF_FOR_EACH(handleScalarType, Field, FieldTy);
    else
      KF_FOR_EACH(handleOtherType, Field, FieldTy);
  }

public:
  KernelObjVisitor(Sema &S) : SemaRef(S) {}

  template <typename... HandlerTys>
  void VisitRecordBases(const CXXRecordDecl *KernelFunctor,
                        HandlerTys &... Handlers) {
    VisitRecordHelper(KernelFunctor, KernelFunctor->bases(), Handlers...);
  }

  // A visitor function that dispatches to functions as defined in
  // SyclKernelFieldHandler for the purposes of kernel generation.
  template <typename... HandlerTys>
  void VisitRecordFields(const CXXRecordDecl *Owner, HandlerTys &... Handlers) {
    for (const auto Field : Owner->fields())
      visitField(Owner, Field, Field->getType(), Handlers...);
  }
#undef KF_FOR_EACH
};

// A base type that the SYCL OpenCL Kernel construction task uses to implement
// individual tasks.
class SyclKernelFieldHandlerBase {
public:
  static constexpr const bool VisitUnionBody = false;
  static constexpr const bool VisitNthArrayElement = true;
  // Opt-in based on whether we should visit inside simple containers (structs,
  // arrays). All of the 'check' types should likely be true, the int-header,
  // and kernel decl creation types should not.
  static constexpr const bool VisitInsideSimpleContainers = true;
  // Mark these virtual so that we can use override in the implementer classes,
  // despite virtual dispatch never being used.

  // SYCL special class can be a base class or a field decl, so both must be
  // handled.
  virtual bool handleSyclSpecialType(const CXXRecordDecl *,
                                     const CXXBaseSpecifier &, QualType) {
    return true;
  }
  virtual bool handleSyclSpecialType(FieldDecl *, QualType) { return true; }

  virtual bool handleSyclSpecConstantType(FieldDecl *, QualType) {
    return true;
  }

  virtual bool handleSyclHalfType(const CXXRecordDecl *,
                                  const CXXBaseSpecifier &, QualType) {
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

  // Handle a simple struct that doesn't need to be decomposed, only called on
  // handlers with VisitInsideSimpleContainers as false.  Replaces
  // handleStructType, enterStruct, leaveStruct, and visiting of sub-elements.
  virtual bool handleNonDecompStruct(const CXXRecordDecl *, FieldDecl *,
                                     QualType) {
    return true;
  }
  virtual bool handleNonDecompStruct(const CXXRecordDecl *,
                                     const CXXBaseSpecifier &, QualType) {
    return true;
  }

  // Instead of handleArrayType, enterArray, leaveArray, and nextElement (plus
  // descending down the elements), this function gets called in the event of an
  // array containing simple elements (even in the case of an MD array).
  virtual bool handleSimpleArrayType(FieldDecl *, QualType) { return true; }

  // The following are only used for keeping track of where we are in the base
  // class/field graph. Int Headers use this to calculate offset, most others
  // don't have a need for these.

  virtual bool enterStruct(const CXXRecordDecl *, FieldDecl *, QualType) {
    return true;
  }
  virtual bool leaveStruct(const CXXRecordDecl *, FieldDecl *, QualType) {
    return true;
  }
  virtual bool enterStruct(const CXXRecordDecl *, const CXXBaseSpecifier &,
                           QualType) {
    return true;
  }
  virtual bool leaveStruct(const CXXRecordDecl *, const CXXBaseSpecifier &,
                           QualType) {
    return true;
  }
  virtual bool enterUnion(const CXXRecordDecl *, FieldDecl *) { return true; }
  virtual bool leaveUnion(const CXXRecordDecl *, FieldDecl *) { return true; }

  // The following are used for stepping through array elements.
  virtual bool enterArray(FieldDecl *, QualType ArrayTy, QualType ElementTy) {
    return true;
  }
  virtual bool leaveArray(FieldDecl *, QualType ArrayTy, QualType ElementTy) {
    return true;
  }

  virtual bool nextElement(QualType, uint64_t) { return true; }

  virtual ~SyclKernelFieldHandlerBase() = default;
};

// A class to act as the direct base for all the SYCL OpenCL Kernel construction
// tasks that contains a reference to Sema (and potentially any other
// universally required data).
class SyclKernelFieldHandler : public SyclKernelFieldHandlerBase {
protected:
  Sema &SemaRef;
  SyclKernelFieldHandler(Sema &S) : SemaRef(S) {}
};

// A class to represent the 'do nothing' case for filtering purposes.
class SyclEmptyHandler final : public SyclKernelFieldHandlerBase {};
SyclEmptyHandler GlobalEmptyHandler;

template <bool Keep, typename H> struct HandlerFilter;
template <typename H> struct HandlerFilter<true, H> {
  H &Handler;
  HandlerFilter(H &Handler) : Handler(Handler) {}
};
template <typename H> struct HandlerFilter<false, H> {
  SyclEmptyHandler &Handler = GlobalEmptyHandler;
  HandlerFilter(H &Handler) {}
};

template <bool B, bool... Rest> struct AnyTrue;

template <bool B> struct AnyTrue<B> { static constexpr bool Value = B; };

template <bool B, bool... Rest> struct AnyTrue {
  static constexpr bool Value = B || AnyTrue<Rest...>::Value;
};

template <bool B, bool... Rest> struct AllTrue;

template <bool B> struct AllTrue<B> { static constexpr bool Value = B; };

template <bool B, bool... Rest> struct AllTrue {
  static constexpr bool Value = B && AllTrue<Rest...>::Value;
};

template <typename ParentTy, typename... Handlers>
void KernelObjVisitor::VisitUnion(const CXXRecordDecl *Owner, ParentTy &Parent,
                                  const CXXRecordDecl *Wrapper,
                                  Handlers &... handlers) {
  // Don't continue descending if none of the handlers 'care'. This could be 'if
  // constexpr' starting in C++17.  Until then, we have to count on the
  // optimizer to realize "if (false)" is a dead branch.
  if (AnyTrue<Handlers::VisitUnionBody...>::Value)
    VisitUnionImpl(
        Owner, Parent, Wrapper,
        HandlerFilter<Handlers::VisitUnionBody, Handlers>(handlers).Handler...);
}

template <typename... Handlers>
void KernelObjVisitor::visitNthArrayElement(const CXXRecordDecl *Owner,
                                            FieldDecl *ArrayField,
                                            QualType ElementTy, uint64_t Index,
                                            Handlers &... handlers) {
  // Don't continue descending if none of the handlers 'care'. This could be 'if
  // constexpr' starting in C++17.  Until then, we have to count on the
  // optimizer to realize "if (false)" is a dead branch.
  if (AnyTrue<Handlers::VisitNthArrayElement...>::Value)
    visitArrayElementImpl(
        Owner, ArrayField, ElementTy, Index,
        HandlerFilter<Handlers::VisitNthArrayElement, Handlers>(handlers)
            .Handler...);
}

template <typename ParentTy, typename... HandlerTys>
void KernelObjVisitor::visitRecord(const CXXRecordDecl *Owner, ParentTy &Parent,
                                   const CXXRecordDecl *Wrapper,
                                   QualType RecordTy,
                                   HandlerTys &... Handlers) {
  RecordDecl *RD = RecordTy->getAsRecordDecl();
  assert(RD && "should not be null.");
  if (RD->hasAttr<SYCLRequiresDecompositionAttr>()) {
    // If this container requires decomposition, we have to visit it as
    // 'complex', so all handlers are called in this case with the 'complex'
    // case.
    visitComplexRecord(Owner, Parent, Wrapper, RecordTy, Handlers...);
  } else {
    // "Simple" Containers are those that do NOT need to be decomposed,
    // "Complex" containers are those that DO. In the case where the container
    // does NOT need to be decomposed, we can call VisitSimpleRecord on the
    // handlers that have opted-out of VisitInsideSimpleContainers. The 'if'
    // makes sure we only do that if at least 1 has opted out.
    if (!AllTrue<HandlerTys::VisitInsideSimpleContainers...>::Value)
      visitSimpleRecord(
          Owner, Parent, Wrapper, RecordTy,
          HandlerFilter<!HandlerTys::VisitInsideSimpleContainers, HandlerTys>(
              Handlers)
              .Handler...);

    // Even though this is a 'simple' container, some handlers (via
    // VisitInsideSimpleContainers = true) need to treat it as if it needs
    // decomposing, so we call VisitComplexRecord iif at least one has.
    if (AnyTrue<HandlerTys::VisitInsideSimpleContainers...>::Value)
      visitComplexRecord(
          Owner, Parent, Wrapper, RecordTy,
          HandlerFilter<HandlerTys::VisitInsideSimpleContainers, HandlerTys>(
              Handlers)
              .Handler...);
  }
}

template <typename... HandlerTys>
void KernelObjVisitor::visitArray(const CXXRecordDecl *Owner, FieldDecl *Field,
                                  QualType ArrayTy, HandlerTys &... Handlers) {

  if (Field->hasAttr<SYCLRequiresDecompositionAttr>()) {
    visitComplexArray(Owner, Field, ArrayTy, Handlers...);
  } else {
    if (!AllTrue<HandlerTys::VisitInsideSimpleContainers...>::Value)
      visitSimpleArray(
          Owner, Field, ArrayTy,
          HandlerFilter<!HandlerTys::VisitInsideSimpleContainers, HandlerTys>(
              Handlers)
              .Handler...);

    if (AnyTrue<HandlerTys::VisitInsideSimpleContainers...>::Value)
      visitComplexArray(
          Owner, Field, ArrayTy,
          HandlerFilter<HandlerTys::VisitInsideSimpleContainers, HandlerTys>(
              Handlers)
              .Handler...);
  }
}

// A type to check the validity of all of the argument types.
class SyclKernelFieldChecker : public SyclKernelFieldHandler {
  bool IsInvalid = false;
  DiagnosticsEngine &Diag;
  bool IsSIMD = false;
  // Check whether the object should be disallowed from being copied to kernel.
  // Return true if not copyable, false if copyable.
  bool checkNotCopyableToKernel(const FieldDecl *FD, QualType FieldTy) {
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

    return false;
  }

  bool checkPropertyListType(TemplateArgument PropList, SourceLocation Loc) {
    if (PropList.getKind() != TemplateArgument::ArgKind::Type)
      return SemaRef.Diag(
          Loc, diag::err_sycl_invalid_accessor_property_template_param);

    QualType PropListTy = PropList.getAsType();
    if (!Util::isAccessorPropertyListType(PropListTy))
      return SemaRef.Diag(
          Loc, diag::err_sycl_invalid_accessor_property_template_param);

    const auto *AccPropListDecl =
        cast<ClassTemplateSpecializationDecl>(PropListTy->getAsRecordDecl());
    if (AccPropListDecl->getTemplateArgs().size() != 1)
      return SemaRef.Diag(Loc,
                          diag::err_sycl_invalid_property_list_param_number)
             << "accessor_property_list";

    const auto TemplArg = AccPropListDecl->getTemplateArgs()[0];
    if (TemplArg.getKind() != TemplateArgument::ArgKind::Pack)
      return SemaRef.Diag(
                 Loc,
                 diag::err_sycl_invalid_accessor_property_list_template_param)
             << /*accessor_property_list*/ 0 << /*parameter pack*/ 0;

    for (TemplateArgument::pack_iterator Prop = TemplArg.pack_begin();
         Prop != TemplArg.pack_end(); ++Prop) {
      if (Prop->getKind() != TemplateArgument::ArgKind::Type)
        return SemaRef.Diag(
                   Loc,
                   diag::err_sycl_invalid_accessor_property_list_template_param)
               << /*accessor_property_list pack argument*/ 1 << /*type*/ 1;
      QualType PropTy = Prop->getAsType();
      if (Util::isSyclBufferLocationType(PropTy) &&
          checkBufferLocationType(PropTy, Loc))
        return true;
    }
    return false;
  }

  bool checkBufferLocationType(QualType PropTy, SourceLocation Loc) {
    const auto *PropDecl =
        cast<ClassTemplateSpecializationDecl>(PropTy->getAsRecordDecl());
    if (PropDecl->getTemplateArgs().size() != 1)
      return SemaRef.Diag(Loc,
                          diag::err_sycl_invalid_property_list_param_number)
             << "buffer_location";

    const auto BufferLoc = PropDecl->getTemplateArgs()[0];
    if (BufferLoc.getKind() != TemplateArgument::ArgKind::Integral)
      return SemaRef.Diag(
                 Loc,
                 diag::err_sycl_invalid_accessor_property_list_template_param)
             << /*buffer_location*/ 2 << /*non-negative integer*/ 2;

    int LocationID = static_cast<int>(BufferLoc.getAsIntegral().getExtValue());
    if (LocationID < 0)
      return SemaRef.Diag(
                 Loc,
                 diag::err_sycl_invalid_accessor_property_list_template_param)
             << /*buffer_location*/ 2 << /*non-negative integer*/ 2;

    return false;
  }

  bool checkSyclSpecialType(QualType Ty, SourceRange Loc) {
    assert(Util::isSyclSpecialType(Ty) &&
           "Should only be called on sycl special class types.");
    const RecordDecl *RecD = Ty->getAsRecordDecl();
    if (IsSIMD && !Util::isSyclType(Ty, "accessor", true /*Tmp*/))
      return SemaRef.Diag(Loc.getBegin(),
                          diag::err_sycl_esimd_not_supported_for_type)
             << RecD;
    if (const ClassTemplateSpecializationDecl *CTSD =
            dyn_cast<ClassTemplateSpecializationDecl>(RecD)) {
      const TemplateArgumentList &TAL = CTSD->getTemplateArgs();
      TemplateArgument TA = TAL.get(0);
      llvm::DenseSet<QualType> Visited;
      checkSYCLType(SemaRef, TA.getAsType(), Loc, Visited);

      if (TAL.size() > 5)
        return checkPropertyListType(TAL.get(5), Loc.getBegin());
    }
    return false;
  }

public:
  SyclKernelFieldChecker(Sema &S, bool isSIMD)
      : SyclKernelFieldHandler(S), Diag(S.getASTContext().getDiagnostics()),
        IsSIMD(isSIMD) {}
  static constexpr const bool VisitNthArrayElement = false;
  bool isValid() { return !IsInvalid; }

  bool handleReferenceType(FieldDecl *FD, QualType FieldTy) final {
    Diag.Report(FD->getLocation(), diag::err_bad_kernel_param_type) << FieldTy;
    IsInvalid = true;
    return isValid();
  }

  bool handleStructType(FieldDecl *FD, QualType FieldTy) final {
    IsInvalid |= checkNotCopyableToKernel(FD, FieldTy);
    CXXRecordDecl *RD = FieldTy->getAsCXXRecordDecl();
    assert(RD && "Not a RecordDecl inside the handler for struct type");
    if (RD->isLambda()) {
      for (const LambdaCapture &LC : RD->captures())
        if (LC.capturesThis() && LC.isImplicit()) {
          SemaRef.Diag(LC.getLocation(), diag::err_implicit_this_capture);
          IsInvalid = true;
        }
    }
    return isValid();
  }

  bool handleSyclSpecialType(const CXXRecordDecl *, const CXXBaseSpecifier &BS,
                             QualType FieldTy) final {
    IsInvalid |= checkSyclSpecialType(FieldTy, BS.getBeginLoc());
    return isValid();
  }

  bool handleSyclSpecialType(FieldDecl *FD, QualType FieldTy) final {
    IsInvalid |= checkSyclSpecialType(FieldTy, FD->getLocation());
    return isValid();
  }

  bool handleArrayType(FieldDecl *FD, QualType FieldTy) final {
    IsInvalid |= checkNotCopyableToKernel(FD, FieldTy);
    return isValid();
  }

  bool handlePointerType(FieldDecl *FD, QualType FieldTy) final {
    while (FieldTy->isAnyPointerType()) {
      FieldTy = QualType{FieldTy->getPointeeOrArrayElementType(), 0};
      if (FieldTy->isVariableArrayType()) {
        Diag.Report(FD->getLocation(), diag::err_vla_unsupported);
        IsInvalid = true;
        break;
      }
    }
    return isValid();
  }

  bool handleOtherType(FieldDecl *FD, QualType FieldTy) final {
    Diag.Report(FD->getLocation(), diag::err_bad_kernel_param_type) << FieldTy;
    IsInvalid = true;
    return isValid();
  }
};

// A type to check the validity of accessing accessor/sampler/stream
// types as kernel parameters inside union.
class SyclKernelUnionChecker : public SyclKernelFieldHandler {
  int UnionCount = 0;
  bool IsInvalid = false;
  DiagnosticsEngine &Diag;

public:
  SyclKernelUnionChecker(Sema &S)
      : SyclKernelFieldHandler(S), Diag(S.getASTContext().getDiagnostics()) {}
  bool isValid() { return !IsInvalid; }
  static constexpr const bool VisitUnionBody = true;
  static constexpr const bool VisitNthArrayElement = false;

  bool checkType(SourceLocation Loc, QualType Ty) {
    if (UnionCount) {
      IsInvalid = true;
      Diag.Report(Loc, diag::err_bad_union_kernel_param_members) << Ty;
    }
    return isValid();
  }

  bool enterUnion(const CXXRecordDecl *RD, FieldDecl *FD) override {
    ++UnionCount;
    return true;
  }

  bool leaveUnion(const CXXRecordDecl *RD, FieldDecl *FD) override {
    --UnionCount;
    return true;
  }

  bool handleSyclSpecialType(FieldDecl *FD, QualType FieldTy) final {
    return checkType(FD->getLocation(), FieldTy);
  }

  bool handleSyclSpecialType(const CXXRecordDecl *, const CXXBaseSpecifier &BS,
                             QualType FieldTy) final {
    return checkType(BS.getBeginLoc(), FieldTy);
  }
};

// A type to mark whether a collection requires decomposition.
class SyclKernelDecompMarker : public SyclKernelFieldHandler {
  llvm::SmallVector<bool, 16> CollectionStack;

public:
  static constexpr const bool VisitUnionBody = false;
  static constexpr const bool VisitNthArrayElement = false;

  SyclKernelDecompMarker(Sema &S) : SyclKernelFieldHandler(S) {
    // In order to prevent checking this over and over, just add a dummy-base
    // entry.
    CollectionStack.push_back(true);
  }

  bool handleSyclSpecialType(const CXXRecordDecl *, const CXXBaseSpecifier &,
                             QualType) final {
    CollectionStack.back() = true;
    return true;
  }
  bool handleSyclSpecialType(FieldDecl *, QualType) final {
    CollectionStack.back() = true;
    return true;
  }

  bool handleSyclSpecConstantType(FieldDecl *, QualType) final {
    CollectionStack.back() = true;
    return true;
  }

  bool handleSyclHalfType(const CXXRecordDecl *, const CXXBaseSpecifier &,
                          QualType) final {
    CollectionStack.back() = true;
    return true;
  }
  bool handleSyclHalfType(FieldDecl *, QualType) final {
    CollectionStack.back() = true;
    return true;
  }

  bool handlePointerType(FieldDecl *, QualType) final {
    CollectionStack.back() = true;
    return true;
  }

  bool enterStruct(const CXXRecordDecl *, FieldDecl *, QualType) final {
    CollectionStack.push_back(false);
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, FieldDecl *, QualType Ty) final {
    if (CollectionStack.pop_back_val()) {
      RecordDecl *RD = Ty->getAsRecordDecl();
      assert(RD && "should not be null.");
      if (!RD->hasAttr<SYCLRequiresDecompositionAttr>())
        RD->addAttr(SYCLRequiresDecompositionAttr::CreateImplicit(
            SemaRef.getASTContext()));
      CollectionStack.back() = true;
    }
    return true;
  }

  bool enterStruct(const CXXRecordDecl *, const CXXBaseSpecifier &,
                   QualType) final {
    CollectionStack.push_back(false);
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, const CXXBaseSpecifier &,
                   QualType Ty) final {
    if (CollectionStack.pop_back_val()) {
      RecordDecl *RD = Ty->getAsRecordDecl();
      assert(RD && "should not be null.");
      if (!RD->hasAttr<SYCLRequiresDecompositionAttr>())
        RD->addAttr(SYCLRequiresDecompositionAttr::CreateImplicit(
            SemaRef.getASTContext()));
      CollectionStack.back() = true;
    }

    return true;
  }

  bool enterArray(FieldDecl *, QualType ArrayTy, QualType ElementTy) final {
    CollectionStack.push_back(false);
    return true;
  }

  bool leaveArray(FieldDecl *FD, QualType ArrayTy, QualType ElementTy) final {
    if (CollectionStack.pop_back_val()) {
      // Cannot assert, since in MD arrays we'll end up marking them multiple
      // times.
      if (!FD->hasAttr<SYCLRequiresDecompositionAttr>())
        FD->addAttr(SYCLRequiresDecompositionAttr::CreateImplicit(
            SemaRef.getASTContext()));
      CollectionStack.back() = true;
    }
    return true;
  }
};

// A type to Create and own the FunctionDecl for the kernel.
class SyclKernelDeclCreator : public SyclKernelFieldHandler {
  FunctionDecl *KernelDecl;
  llvm::SmallVector<ParmVarDecl *, 8> Params;
  Sema::ContextRAII FuncContext;
  // Holds the last handled field's first parameter. This doesn't store an
  // iterator as push_back invalidates iterators.
  size_t LastParamIndex = 0;
  // Keeps track of whether we are currently handling fields inside a struct.
  int StructDepth = 0;

  void addParam(const FieldDecl *FD, QualType FieldTy) {
    ParamDesc newParamDesc = makeParamDesc(FD, FieldTy);
    addParam(newParamDesc, FieldTy);
  }

  void addParam(const CXXBaseSpecifier &BS, QualType FieldTy) {
    // TODO: There is no name for the base available, but duplicate names are
    // seemingly already possible, so we'll give them all the same name for now.
    // This only happens with the accessor types.
    StringRef Name = "_arg__base";
    ParamDesc newParamDesc =
        makeParamDesc(SemaRef.getASTContext(), Name, FieldTy);
    addParam(newParamDesc, FieldTy);
  }
  // Add a parameter with specified name and type
  void addParam(StringRef Name, QualType ParamTy) {
    ParamDesc newParamDesc =
        makeParamDesc(SemaRef.getASTContext(), Name, ParamTy);
    addParam(newParamDesc, ParamTy);
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
  // the accessor_property_list - add the appropriate attributes to ParmVarDecl.
  void handleAccessorPropertyList(ParmVarDecl *Param,
                                  const CXXRecordDecl *RecordDecl,
                                  SourceLocation Loc) {
    const auto *AccTy = cast<ClassTemplateSpecializationDecl>(RecordDecl);
    if (AccTy->getTemplateArgs().size() < 6)
      return;
    const auto PropList = cast<TemplateArgument>(AccTy->getTemplateArgs()[5]);
    QualType PropListTy = PropList.getAsType();
    const auto *AccPropListDecl =
        cast<ClassTemplateSpecializationDecl>(PropListTy->getAsRecordDecl());
    const auto TemplArg = AccPropListDecl->getTemplateArgs()[0];
    // Move through TemplateArgs list of a property list and search for
    // properties. If found - apply the appropriate attribute to ParmVarDecl.
    for (TemplateArgument::pack_iterator Prop = TemplArg.pack_begin();
         Prop != TemplArg.pack_end(); ++Prop) {
      QualType PropTy = Prop->getAsType();
      if (Util::isSyclAccessorNoAliasPropertyType(PropTy))
        handleNoAliasProperty(Param, PropTy, Loc);
      if (Util::isSyclBufferLocationType(PropTy))
        handleBufferLocationProperty(Param, PropTy, Loc);
    }
  }

  void handleNoAliasProperty(ParmVarDecl *Param, QualType PropTy,
                             SourceLocation Loc) {
    ASTContext &Ctx = SemaRef.getASTContext();
    Param->addAttr(RestrictAttr::CreateImplicit(Ctx, Loc));
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

  // Additional processing is required for accessor type.
  void handleAccessorType(const CXXRecordDecl *RecordDecl, SourceLocation Loc) {
    handleAccessorPropertyList(Params.back(), RecordDecl, Loc);
    // Get access mode of accessor.
    const auto *AccessorSpecializationDecl =
        cast<ClassTemplateSpecializationDecl>(RecordDecl);
    const TemplateArgument &AccessModeArg =
        AccessorSpecializationDecl->getTemplateArgs().get(2);

    // Add implicit attribute to parameter decl when it is a read only
    // SYCL accessor.
    if (isReadOnlyAccessor(AccessModeArg))
      Params.back()->addAttr(
          SYCLAccessorReadonlyAttr::CreateImplicit(SemaRef.getASTContext()));
    Params.back()->addAttr(
        SYCLAccessorPtrAttr::CreateImplicit(SemaRef.getASTContext()));
  }

  // All special SYCL objects must have __init method. We extract types for
  // kernel parameters from __init method parameters. We will use __init method
  // and kernel parameters which we build here to initialize special objects in
  // the kernel body.
  bool handleSpecialType(FieldDecl *FD, QualType FieldTy) {
    const auto *RecordDecl = FieldTy->getAsCXXRecordDecl();
    assert(RecordDecl && "The type must be a RecordDecl");
    llvm::StringLiteral MethodName =
        KernelDecl->hasAttr<SYCLSimdAttr>() &&
                Util::isSyclType(FieldTy, "accessor", true /*Tmp*/)
            ? InitESIMDMethodName
            : InitMethodName;
    CXXMethodDecl *InitMethod = getMethodByName(RecordDecl, MethodName);
    assert(InitMethod && "The type must have the __init method");

    // Don't do -1 here because we count on this to be the first parameter added
    // (if any).
    size_t ParamIndex = Params.size();
    for (const ParmVarDecl *Param : InitMethod->parameters()) {
      QualType ParamTy = Param->getType();
      addParam(FD, ParamTy.getCanonicalType());
      // FIXME: This code is temporary, and will be removed once __init_esimd
      // is removed and property list refactored.
      // The function handleAccessorType includes a call to
      // handleAccessorPropertyList. If new classes with property list are
      // added, this code needs to be refactored to call
      // handleAccessorPropertyList for each class which requires it.
      if (ParamTy.getTypePtr()->isPointerType() &&
          Util::isSyclType(FieldTy, "accessor", true /*Tmp*/))
        handleAccessorType(RecordDecl, FD->getBeginLoc());
    }
    LastParamIndex = ParamIndex;
    return true;
  }

  static void setKernelImplicitAttrs(ASTContext &Context, FunctionDecl *FD,
                                     bool IsSIMDKernel) {
    // Set implicit attributes.
    FD->addAttr(OpenCLKernelAttr::CreateImplicit(Context));
    FD->addAttr(ArtificialAttr::CreateImplicit(Context));
    if (IsSIMDKernel)
      FD->addAttr(SYCLSimdAttr::CreateImplicit(Context));
  }

  static FunctionDecl *createKernelDecl(ASTContext &Ctx, SourceLocation Loc,
                                        bool IsInline, bool IsSIMDKernel) {
    // Create this with no prototype, and we can fix this up after we've seen
    // all the params.
    FunctionProtoType::ExtProtoInfo Info(CC_OpenCLKernel);
    QualType FuncType = Ctx.getFunctionType(Ctx.VoidTy, {}, Info);

    FunctionDecl *FD = FunctionDecl::Create(
        Ctx, Ctx.getTranslationUnitDecl(), Loc, Loc, DeclarationName(),
        FuncType, Ctx.getTrivialTypeSourceInfo(Ctx.VoidTy), SC_None);
    FD->setImplicitlyInline(IsInline);
    setKernelImplicitAttrs(Ctx, FD, IsSIMDKernel);

    // Add kernel to translation unit to see it in AST-dump.
    Ctx.getTranslationUnitDecl()->addDecl(FD);
    return FD;
  }

public:
  static constexpr const bool VisitInsideSimpleContainers = false;
  SyclKernelDeclCreator(Sema &S, SourceLocation Loc, bool IsInline,
                        bool IsSIMDKernel, FunctionDecl *SYCLKernel)
      : SyclKernelFieldHandler(S),
        KernelDecl(
            createKernelDecl(S.getASTContext(), Loc, IsInline, IsSIMDKernel)),
        FuncContext(SemaRef, KernelDecl) {
    S.addSyclOpenCLKernel(SYCLKernel, KernelDecl);
  }

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

    // Make sure that this is marked as a kernel so that the code-gen can make
    // decisions based on that. We cannot add this earlier, otherwise the call
    // to TransformStmt in replaceWithLocalClone can diagnose something that got
    // diagnosed on the actual kernel.
    KernelDecl->addAttr(
        SYCLKernelAttr::CreateImplicit(SemaRef.getASTContext()));

    SemaRef.addSyclDeviceDecl(KernelDecl);
  }

  bool enterStruct(const CXXRecordDecl *, FieldDecl *, QualType) final {
    ++StructDepth;
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, FieldDecl *, QualType) final {
    --StructDepth;
    return true;
  }

  bool enterStruct(const CXXRecordDecl *, const CXXBaseSpecifier &BS,
                   QualType FieldTy) final {
    ++StructDepth;
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, const CXXBaseSpecifier &BS,
                   QualType FieldTy) final {
    --StructDepth;
    return true;
  }

  bool handleSyclSpecialType(const CXXRecordDecl *, const CXXBaseSpecifier &BS,
                             QualType FieldTy) final {
    const auto *RecordDecl = FieldTy->getAsCXXRecordDecl();
    assert(RecordDecl && "The type must be a RecordDecl");
    llvm::StringLiteral MethodName =
        KernelDecl->hasAttr<SYCLSimdAttr>() &&
                Util::isSyclType(FieldTy, "accessor", true /*Tmp*/)
            ? InitESIMDMethodName
            : InitMethodName;
    CXXMethodDecl *InitMethod = getMethodByName(RecordDecl, MethodName);
    assert(InitMethod && "The type must have the __init method");

    // Don't do -1 here because we count on this to be the first parameter added
    // (if any).
    size_t ParamIndex = Params.size();
    for (const ParmVarDecl *Param : InitMethod->parameters()) {
      QualType ParamTy = Param->getType();
      addParam(BS, ParamTy.getCanonicalType());
      // FIXME: This code is temporary, and will be removed once __init_esimd
      // is removed and property list refactored.
      // The function handleAccessorType includes a call to
      // handleAccessorPropertyList. If new classes with property list are
      // added, this code needs to be refactored to call
      // handleAccessorPropertyList for each class which requires it.
      if (ParamTy.getTypePtr()->isPointerType() &&
          Util::isSyclType(FieldTy, "accessor", true /*Tmp*/))
        handleAccessorType(RecordDecl, BS.getBeginLoc());
    }
    LastParamIndex = ParamIndex;
    return true;
  }

  bool handleSyclSpecialType(FieldDecl *FD, QualType FieldTy) final {
    return handleSpecialType(FD, FieldTy);
  }

  RecordDecl *wrapField(FieldDecl *Field, QualType FieldTy) {
    RecordDecl *WrapperClass =
        SemaRef.getASTContext().buildImplicitRecord("__wrapper_class");
    WrapperClass->startDefinition();
    Field = FieldDecl::Create(
        SemaRef.getASTContext(), WrapperClass, SourceLocation(),
        SourceLocation(), /*Id=*/nullptr, FieldTy,
        SemaRef.getASTContext().getTrivialTypeSourceInfo(FieldTy,
                                                         SourceLocation()),
        /*BW=*/nullptr, /*Mutable=*/false, /*InitStyle=*/ICIS_NoInit);
    Field->setAccess(AS_public);
    WrapperClass->addDecl(Field);
    WrapperClass->completeDefinition();
    return WrapperClass;
  };

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
    if (AS != LangAS::sycl_global_device && AS != LangAS::sycl_global_host)
      Quals.setAddressSpace(LangAS::sycl_global);
    PointeeTy = SemaRef.getASTContext().getQualifiedType(
        PointeeTy.getUnqualifiedType(), Quals);
    QualType ModTy = SemaRef.getASTContext().getPointerType(PointeeTy);
    // When the kernel is generated, struct type kernel arguments are
    // decomposed; i.e. the parameters of the kernel are the fields of the
    // struct, and not the struct itself. This causes an error in the backend
    // when the struct field is a pointer, since non-USM pointers cannot be
    // passed directly. To work around this issue, all pointers inside the
    // struct are wrapped in a generated '__wrapper_class'.
    if (StructDepth) {
      RecordDecl *WrappedPointer = wrapField(FD, ModTy);
      ModTy = SemaRef.getASTContext().getRecordType(WrappedPointer);
    }

    addParam(FD, ModTy);
    return true;
  }

  bool handleSimpleArrayType(FieldDecl *FD, QualType FieldTy) final {
    // Arrays are always wrapped in a struct since they cannot be passed
    // directly.
    RecordDecl *WrappedArray = wrapField(FD, FieldTy);
    QualType ModTy = SemaRef.getASTContext().getRecordType(WrappedArray);
    addParam(FD, ModTy);
    return true;
  }

  bool handleScalarType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FD, FieldTy);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, FieldDecl *FD,
                             QualType Ty) final {
    addParam(FD, Ty);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *Base,
                             const CXXBaseSpecifier &BS, QualType Ty) final {
    addParam(BS, Ty);
    return true;
  }

  bool handleUnionType(FieldDecl *FD, QualType FieldTy) final {
    return handleScalarType(FD, FieldTy);
  }

  bool handleSyclHalfType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FD, FieldTy);
    return true;
  }

  // Generate kernel argument to initialize specialization constants.
  void handleSyclKernelHandlerType() {
    ASTContext &Context = SemaRef.getASTContext();
    StringRef Name = "_arg__specialization_constants_buffer";
    addParam(Name, Context.getPointerType(Context.getAddrSpaceQualType(
                       Context.CharTy, LangAS::sycl_global)));
  }

  void setBody(CompoundStmt *KB) { KernelDecl->setBody(KB); }

  FunctionDecl *getKernelDecl() { return KernelDecl; }

  llvm::ArrayRef<ParmVarDecl *> getParamVarDeclsForCurrentField() {
    return ArrayRef<ParmVarDecl *>(std::begin(Params) + LastParamIndex,
                                   std::end(Params));
  }
  using SyclKernelFieldHandler::handleSyclHalfType;
};

class SyclKernelArgsSizeChecker : public SyclKernelFieldHandler {
  SourceLocation KernelLoc;
  unsigned SizeOfParams = 0;
  bool IsSIMD = false;

  void addParam(QualType ArgTy) {
    SizeOfParams +=
        SemaRef.getASTContext().getTypeSizeInChars(ArgTy).getQuantity();
  }

  bool handleSpecialType(QualType FieldTy) {
    const CXXRecordDecl *RecordDecl = FieldTy->getAsCXXRecordDecl();
    assert(RecordDecl && "The type must be a RecordDecl");
    llvm::StringLiteral MethodName =
        (IsSIMD && Util::isSyclType(FieldTy, "accessor", true /*Tmp*/))
            ? InitESIMDMethodName
            : InitMethodName;
    CXXMethodDecl *InitMethod = getMethodByName(RecordDecl, MethodName);
    assert(InitMethod && "The type must have the __init method");
    for (const ParmVarDecl *Param : InitMethod->parameters())
      addParam(Param->getType());
    return true;
  }

public:
  static constexpr const bool VisitInsideSimpleContainers = false;
  SyclKernelArgsSizeChecker(Sema &S, SourceLocation Loc, bool IsSIMD)
      : SyclKernelFieldHandler(S), KernelLoc(Loc), IsSIMD(IsSIMD) {}

  ~SyclKernelArgsSizeChecker() {
    if (SizeOfParams > MaxKernelArgsSize)
      SemaRef.Diag(KernelLoc, diag::warn_sycl_kernel_too_big_args)
          << SizeOfParams << MaxKernelArgsSize;
  }

  bool handleSyclSpecialType(FieldDecl *FD, QualType FieldTy) final {
    return handleSpecialType(FieldTy);
  }

  bool handleSyclSpecialType(const CXXRecordDecl *, const CXXBaseSpecifier &BS,
                             QualType FieldTy) final {
    return handleSpecialType(FieldTy);
  }

  bool handlePointerType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FieldTy);
    return true;
  }

  bool handleScalarType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FieldTy);
    return true;
  }

  bool handleSimpleArrayType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FieldTy);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, FieldDecl *FD,
                             QualType Ty) final {
    addParam(Ty);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *Base,
                             const CXXBaseSpecifier &BS, QualType Ty) final {
    addParam(Ty);
    return true;
  }

  bool handleUnionType(FieldDecl *FD, QualType FieldTy) final {
    return handleScalarType(FD, FieldTy);
  }

  bool handleSyclHalfType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FieldTy);
    return true;
  }
  using SyclKernelFieldHandler::handleSyclHalfType;
};

std::string getKernelArgDesc(StringRef KernelArgDescription) {
  if (KernelArgDescription == ":" || KernelArgDescription == "")
    return "";
  return ("Compiler generated argument for " + KernelArgDescription + ",")
      .str();
}

class SyclOptReportCreator : public SyclKernelFieldHandler {
  SyclKernelDeclCreator &DC;
  SourceLocation KernelInvocationLoc;

  void addParam(const FieldDecl *KernelArg, QualType KernelArgType,
                StringRef KernelArgDescription) {
    StringRef NameToEmitInDescription = KernelArg->getName();
    const RecordDecl *KernelArgParent = KernelArg->getParent();
    if (KernelArgParent && KernelArgDescription == "decomposed struct/class")
      NameToEmitInDescription = KernelArgParent->getName();

    bool isWrappedField = KernelArgDescription == "WrappedPointer" ||
                          KernelArgDescription == "WrappedArray";

    KernelArgDescription =
        (KernelArgDescription == "WrappedPointer"
             ? "nested pointer"
             : (KernelArgDescription == "WrappedArray" ? "array"
                                                       : KernelArgDescription));

    unsigned KernelArgSize =
        SemaRef.getASTContext().getTypeSizeInChars(KernelArgType).getQuantity();

    SemaRef.getDiagnostics().getSYCLOptReport().AddKernelArgs(
        DC.getKernelDecl(), NameToEmitInDescription,
        isWrappedField ? "Compiler generated" : KernelArgType.getAsString(),
        KernelInvocationLoc, KernelArgSize,
        getKernelArgDesc(KernelArgDescription),
        (KernelArgDescription == "decomposed struct/class")
            ? ("Field:" + KernelArg->getName().str() + ", ")
            : "");
  }

  void addParam(const FieldDecl *FD, QualType FieldTy) {
    std::string KernelArgDescription = FieldTy.getAsString();
    const RecordDecl *RD = FD->getParent();
    if (FieldTy->isScalarType())
      KernelArgDescription = "";
    if (RD && RD->hasAttr<SYCLRequiresDecompositionAttr>())
      KernelArgDescription = "decomposed struct/class";

    addParam(FD, FieldTy, KernelArgDescription);
  }

  // Handles base classes.
  void addParam(const CXXBaseSpecifier &, QualType KernelArgType,
                StringRef KernelArgDescription) {
    unsigned KernelArgSize =
        SemaRef.getASTContext().getTypeSizeInChars(KernelArgType).getQuantity();
    SemaRef.getDiagnostics().getSYCLOptReport().AddKernelArgs(
        DC.getKernelDecl(), KernelArgType.getAsString(),
        KernelArgType.getAsString(), KernelInvocationLoc, KernelArgSize,
        getKernelArgDesc(KernelArgDescription), "");
  }

  // Handles specialization constants.
  void addParam(QualType KernelArgType, std::string KernelArgDescription) {
    unsigned KernelArgSize =
        SemaRef.getASTContext().getTypeSizeInChars(KernelArgType).getQuantity();
    SemaRef.getDiagnostics().getSYCLOptReport().AddKernelArgs(
        DC.getKernelDecl(), "", KernelArgType.getAsString(),
        KernelInvocationLoc, KernelArgSize,
        getKernelArgDesc(KernelArgDescription), "");
  }

public:
  static constexpr const bool VisitInsideSimpleContainers = false;
  SyclOptReportCreator(Sema &S, SyclKernelDeclCreator &DC, SourceLocation Loc)
      : SyclKernelFieldHandler(S), DC(DC), KernelInvocationLoc(Loc) {}

  bool handleSyclSpecialType(FieldDecl *FD, QualType FieldTy) final {
    for (const auto *Param : DC.getParamVarDeclsForCurrentField())
      addParam(FD, Param->getType(), FieldTy.getAsString());
    return true;
  }

  bool handleSyclSpecialType(const CXXRecordDecl *, const CXXBaseSpecifier &BS,
                             QualType FieldTy) final {
    std::string KernelArgDescription = "base class " + FieldTy.getAsString();
    for (const auto *Param : DC.getParamVarDeclsForCurrentField()) {
      QualType KernelArgType = Param->getType();
      unsigned KernelArgSize = SemaRef.getASTContext()
                                   .getTypeSizeInChars(KernelArgType)
                                   .getQuantity();
      SemaRef.getDiagnostics().getSYCLOptReport().AddKernelArgs(
          DC.getKernelDecl(), FieldTy.getAsString(),
          KernelArgType.getAsString(), KernelInvocationLoc, KernelArgSize,
          getKernelArgDesc(KernelArgDescription), "");
    }
    return true;
  }

  bool handlePointerType(FieldDecl *FD, QualType FieldTy) final {
    std::string KernelArgDescription = ":";
    ParmVarDecl *KernelParameter = DC.getParamVarDeclsForCurrentField()[0];
    // Compiler generated openCL kernel argument for current pointer field
    // is not a pointer. This means we are processing a nested pointer and
    // the openCL kernel argument is of type __wrapper_class.
    if (!KernelParameter->getType()->isPointerType())
      KernelArgDescription = "WrappedPointer";
    for (const auto *Param : DC.getParamVarDeclsForCurrentField())
      addParam(FD, Param->getType(), KernelArgDescription);
    return true;
  }

  bool handleScalarType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FD, FieldTy);
    return true;
  }

  bool handleSimpleArrayType(FieldDecl *FD, QualType FieldTy) final {
    // Simple arrays are always wrapped.
    for (const auto *Param : DC.getParamVarDeclsForCurrentField())
      addParam(FD, Param->getType(), "WrappedArray");
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, FieldDecl *FD,
                             QualType Ty) final {
    addParam(FD, Ty);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *Base,
                             const CXXBaseSpecifier &BS, QualType Ty) final {
    addParam(BS, Ty, "base class");
    return true;
  }

  bool handleUnionType(FieldDecl *FD, QualType FieldTy) final {
    return handleScalarType(FD, FieldTy);
  }

  bool handleSyclHalfType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FD, FieldTy);
    return true;
  }

  void handleSyclKernelHandlerType() {
    addParam(DC.getParamVarDeclsForCurrentField()[0]->getType(),
             "SYCL2020 specialization constant");
  }
  using SyclKernelFieldHandler::handleSyclHalfType;
};

static CXXMethodDecl *getOperatorParens(const CXXRecordDecl *Rec) {
  for (auto *MD : Rec->methods()) {
    if (MD->getOverloadedOperator() == OO_Call)
      return MD;
  }
  return nullptr;
}

static bool isESIMDKernelType(const CXXRecordDecl *KernelObjType) {
  const CXXMethodDecl *OpParens = getOperatorParens(KernelObjType);
  return (OpParens != nullptr) && OpParens->hasAttr<SYCLSimdAttr>();
}

class SyclKernelBodyCreator : public SyclKernelFieldHandler {
  SyclKernelDeclCreator &DeclCreator;
  llvm::SmallVector<Stmt *, 16> BodyStmts;
  llvm::SmallVector<InitListExpr *, 16> CollectionInitExprs;
  llvm::SmallVector<Stmt *, 16> FinalizeStmts;
  // This collection contains the information required to add/remove information
  // about arrays as we enter them.  The InitializedEntity component is
  // necessary for initializing child members.  uin64_t is the index of the
  // current element being worked on, which is updated every time we visit
  // nextElement.
  llvm::SmallVector<std::pair<InitializedEntity, uint64_t>, 8> ArrayInfos;
  VarDecl *KernelObjClone;
  InitializedEntity VarEntity;
  const CXXRecordDecl *KernelObj;
  llvm::SmallVector<Expr *, 16> MemberExprBases;
  FunctionDecl *KernelCallerFunc;
  SourceLocation KernelCallerSrcLoc; // KernelCallerFunc source location.
  // Contains a count of how many containers we're in.  This is used by the
  // pointer-struct-wrapping code to ensure that we don't try to wrap
  // non-top-level pointers.
  uint64_t StructDepth = 0;
  VarDecl *KernelHandlerClone = nullptr;

  Stmt *replaceWithLocalClone(ParmVarDecl *OriginalParam, VarDecl *LocalClone,
                              Stmt *FunctionBody) {
    // DeclRefExpr with valid source location but with decl which is not marked
    // as used is invalid.
    LocalClone->setIsUsed();
    std::pair<DeclaratorDecl *, DeclaratorDecl *> MappingPair =
        std::make_pair(OriginalParam, LocalClone);
    KernelBodyTransform KBT(MappingPair, SemaRef);
    return KBT.TransformStmt(FunctionBody).get();
  }

  // Using the statements/init expressions that we've created, this generates
  // the kernel body compound stmt. CompoundStmt needs to know its number of
  // statements in advance to allocate it, so we cannot do this as we go along.
  CompoundStmt *createKernelBody() {
    // Push the Kernel function scope to ensure the scope isn't empty
    SemaRef.PushFunctionScope();

    // Initialize kernel object local clone
    assert(CollectionInitExprs.size() == 1 &&
           "Should have been popped down to just the first one");
    KernelObjClone->setInit(CollectionInitExprs.back());

    // Replace references to the kernel object in kernel body, to use the
    // compiler generated local clone
    Stmt *NewBody =
        replaceWithLocalClone(KernelCallerFunc->getParamDecl(0), KernelObjClone,
                              KernelCallerFunc->getBody());

    // If kernel_handler argument is passed by SYCL kernel, replace references
    // to this argument in kernel body, to use the compiler generated local
    // clone
    if (ParmVarDecl *KernelHandlerParam =
            getSyclKernelHandlerArg(KernelCallerFunc))
      NewBody = replaceWithLocalClone(KernelHandlerParam, KernelHandlerClone,
                                      NewBody);

    // Use transformed body (with clones) as kernel body
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

  // Creates a DeclRefExpr to the ParmVar that represents the current field.
  Expr *createParamReferenceExpr() {
    ParmVarDecl *KernelParameter =
        DeclCreator.getParamVarDeclsForCurrentField()[0];

    QualType ParamType = KernelParameter->getOriginalType();
    Expr *DRE = SemaRef.BuildDeclRefExpr(KernelParameter, ParamType, VK_LValue,
                                         KernelCallerSrcLoc);
    return DRE;
  }

  // Creates a DeclRefExpr to the ParmVar that represents the current pointer
  // field.
  Expr *createPointerParamReferenceExpr(QualType PointerTy, bool Wrapped) {
    ParmVarDecl *KernelParameter =
        DeclCreator.getParamVarDeclsForCurrentField()[0];

    QualType ParamType = KernelParameter->getOriginalType();
    Expr *DRE = SemaRef.BuildDeclRefExpr(KernelParameter, ParamType, VK_LValue,
                                         KernelCallerSrcLoc);

    // Struct Type kernel arguments are decomposed. The pointer fields are
    // then wrapped inside a compiler generated struct. Therefore when
    // generating the initializers, we have to 'unwrap' the pointer.
    if (Wrapped) {
      CXXRecordDecl *WrapperStruct = ParamType->getAsCXXRecordDecl();
      // Pointer field wrapped inside __wrapper_class
      FieldDecl *Pointer = *(WrapperStruct->field_begin());
      DRE = buildMemberExpr(DRE, Pointer);
      ParamType = Pointer->getType();
    }

    DRE = ImplicitCastExpr::Create(SemaRef.Context, ParamType,
                                   CK_LValueToRValue, DRE, /*BasePath=*/nullptr,
                                   VK_PRValue, FPOptionsOverride());

    if (PointerTy->getPointeeType().getAddressSpace() !=
        ParamType->getPointeeType().getAddressSpace())
      DRE = ImplicitCastExpr::Create(SemaRef.Context, PointerTy,
                                     CK_AddressSpaceConversion, DRE, nullptr,
                                     VK_PRValue, FPOptionsOverride());

    return DRE;
  }

  Expr *createSimpleArrayParamReferenceExpr(QualType ArrayTy) {
    ParmVarDecl *KernelParameter =
        DeclCreator.getParamVarDeclsForCurrentField()[0];
    QualType ParamType = KernelParameter->getOriginalType();
    Expr *DRE = SemaRef.BuildDeclRefExpr(KernelParameter, ParamType, VK_LValue,
                                         KernelCallerSrcLoc);

    // Unwrap the array.
    CXXRecordDecl *WrapperStruct = ParamType->getAsCXXRecordDecl();
    FieldDecl *ArrayField = *(WrapperStruct->field_begin());
    return buildMemberExpr(DRE, ArrayField);
  }

  // Returns 'true' if the thing we're visiting (Based on the FD/QualType pair)
  // is an element of an array.  This will determine whether we do
  // MemberExprBases in some cases or not, AND determines how we initialize
  // values.
  bool isArrayElement(const FieldDecl *FD, QualType Ty) const {
    return !SemaRef.getASTContext().hasSameType(FD->getType(), Ty);
  }

  // Creates an initialized entity for a field/item. In the case where this is a
  // field, returns a normal member initializer, if we're in a sub-array of a MD
  // array, returns an element initializer.
  InitializedEntity getFieldEntity(FieldDecl *FD, QualType Ty) {
    if (isArrayElement(FD, Ty))
      return InitializedEntity::InitializeElement(SemaRef.getASTContext(),
                                                  ArrayInfos.back().second,
                                                  ArrayInfos.back().first);
    return InitializedEntity::InitializeMember(FD, &VarEntity);
  }

  void addFieldInit(FieldDecl *FD, QualType Ty, MultiExprArg ParamRef) {
    InitializationKind InitKind =
        InitializationKind::CreateCopy(KernelCallerSrcLoc, KernelCallerSrcLoc);
    addFieldInit(FD, Ty, ParamRef, InitKind);
  }

  void addFieldInit(FieldDecl *FD, QualType Ty, MultiExprArg ParamRef,
                    InitializationKind InitKind) {
    addFieldInit(FD, Ty, ParamRef, InitKind, getFieldEntity(FD, Ty));
  }

  void addFieldInit(FieldDecl *FD, QualType Ty, MultiExprArg ParamRef,
                    InitializationKind InitKind, InitializedEntity Entity) {
    InitializationSequence InitSeq(SemaRef, Entity, InitKind, ParamRef);
    ExprResult Init = InitSeq.Perform(SemaRef, Entity, InitKind, ParamRef);

    InitListExpr *ParentILE = CollectionInitExprs.back();
    ParentILE->updateInit(SemaRef.getASTContext(), ParentILE->getNumInits(),
                          Init.get());
  }

  void addBaseInit(const CXXBaseSpecifier &BS, QualType Ty,
                   InitializationKind InitKind) {
    InitializedEntity Entity = InitializedEntity::InitializeBase(
        SemaRef.Context, &BS, /*IsInheritedVirtualBase*/ false, &VarEntity);
    InitializationSequence InitSeq(SemaRef, Entity, InitKind, None);
    ExprResult Init = InitSeq.Perform(SemaRef, Entity, InitKind, None);

    InitListExpr *ParentILE = CollectionInitExprs.back();
    ParentILE->updateInit(SemaRef.getASTContext(), ParentILE->getNumInits(),
                          Init.get());
  }

  void addSimpleBaseInit(const CXXBaseSpecifier &BS, QualType Ty) {
    InitializationKind InitKind =
        InitializationKind::CreateCopy(KernelCallerSrcLoc, KernelCallerSrcLoc);

    InitializedEntity Entity = InitializedEntity::InitializeBase(
        SemaRef.Context, &BS, /*IsInheritedVirtualBase*/ false, &VarEntity);

    Expr *ParamRef = createParamReferenceExpr();
    InitializationSequence InitSeq(SemaRef, Entity, InitKind, ParamRef);
    ExprResult Init = InitSeq.Perform(SemaRef, Entity, InitKind, ParamRef);

    InitListExpr *ParentILE = CollectionInitExprs.back();
    ParentILE->updateInit(SemaRef.getASTContext(), ParentILE->getNumInits(),
                          Init.get());
  }

  // Adds an initializer that handles a simple initialization of a field.
  void addSimpleFieldInit(FieldDecl *FD, QualType Ty) {
    Expr *ParamRef = createParamReferenceExpr();
    addFieldInit(FD, Ty, ParamRef);
  }

  MemberExpr *buildMemberExpr(Expr *Base, ValueDecl *Member) {
    DeclAccessPair MemberDAP = DeclAccessPair::make(Member, AS_none);
    MemberExpr *Result = SemaRef.BuildMemberExpr(
        Base, /*IsArrow */ false, KernelCallerSrcLoc, NestedNameSpecifierLoc(),
        KernelCallerSrcLoc, Member, MemberDAP,
        /*HadMultipleCandidates*/ false,
        DeclarationNameInfo(Member->getDeclName(), KernelCallerSrcLoc),
        Member->getType(), VK_LValue, OK_Ordinary);
    return Result;
  }

  void addFieldMemberExpr(FieldDecl *FD, QualType Ty) {
    if (!isArrayElement(FD, Ty))
      MemberExprBases.push_back(buildMemberExpr(MemberExprBases.back(), FD));
  }

  void removeFieldMemberExpr(const FieldDecl *FD, QualType Ty) {
    if (!isArrayElement(FD, Ty))
      MemberExprBases.pop_back();
  }

  void createSpecialMethodCall(const CXXRecordDecl *RD, StringRef MethodName,
                               SmallVectorImpl<Stmt *> &AddTo) {
    CXXMethodDecl *Method = getMethodByName(RD, MethodName);
    if (!Method)
      return;

    unsigned NumParams = Method->getNumParams();
    llvm::SmallVector<Expr *, 4> ParamDREs(NumParams);
    llvm::ArrayRef<ParmVarDecl *> KernelParameters =
        DeclCreator.getParamVarDeclsForCurrentField();
    for (size_t I = 0; I < NumParams; ++I) {
      QualType ParamType = KernelParameters[I]->getOriginalType();
      ParamDREs[I] = SemaRef.BuildDeclRefExpr(KernelParameters[I], ParamType,
                                              VK_LValue, KernelCallerSrcLoc);
    }

    MemberExpr *MethodME = buildMemberExpr(MemberExprBases.back(), Method);

    QualType ResultTy = Method->getReturnType();
    ExprValueKind VK = Expr::getValueKindForType(ResultTy);
    ResultTy = ResultTy.getNonLValueExprType(SemaRef.Context);
    llvm::SmallVector<Expr *, 4> ParamStmts;
    const auto *Proto = cast<FunctionProtoType>(Method->getType());
    SemaRef.GatherArgumentsForCall(KernelCallerSrcLoc, Method, Proto, 0,
                                   ParamDREs, ParamStmts);
    // [kernel_obj or wrapper object].accessor.__init(_ValueType*,
    // range<int>, range<int>, id<int>)
    AddTo.push_back(CXXMemberCallExpr::Create(
        SemaRef.Context, MethodME, ParamStmts, ResultTy, VK, KernelCallerSrcLoc,
        FPOptionsOverride()));
  }

  // Creates an empty InitListExpr of the correct number of child-inits
  // of this to append into.
  void addCollectionInitListExpr(const CXXRecordDecl *RD) {
    const ASTRecordLayout &Info =
        SemaRef.getASTContext().getASTRecordLayout(RD);
    uint64_t NumInitExprs = Info.getFieldCount() + RD->getNumBases();
    addCollectionInitListExpr(QualType(RD->getTypeForDecl(), 0), NumInitExprs);
  }

  InitListExpr *createInitListExpr(const CXXRecordDecl *RD) {
    const ASTRecordLayout &Info =
        SemaRef.getASTContext().getASTRecordLayout(RD);
    uint64_t NumInitExprs = Info.getFieldCount() + RD->getNumBases();
    return createInitListExpr(QualType(RD->getTypeForDecl(), 0), NumInitExprs);
  }

  InitListExpr *createInitListExpr(QualType InitTy, uint64_t NumChildInits) {
    InitListExpr *ILE = new (SemaRef.getASTContext()) InitListExpr(
        SemaRef.getASTContext(), KernelCallerSrcLoc, {}, KernelCallerSrcLoc);
    ILE->reserveInits(SemaRef.getASTContext(), NumChildInits);
    ILE->setType(InitTy);

    return ILE;
  }

  // Create an empty InitListExpr of the type/size for the rest of the visitor
  // to append into.
  void addCollectionInitListExpr(QualType InitTy, uint64_t NumChildInits) {

    InitListExpr *ILE = createInitListExpr(InitTy, NumChildInits);
    InitListExpr *ParentILE = CollectionInitExprs.back();
    ParentILE->updateInit(SemaRef.getASTContext(), ParentILE->getNumInits(),
                          ILE);

    CollectionInitExprs.push_back(ILE);
  }

  static VarDecl *createKernelObjClone(ASTContext &Ctx, DeclContext *DC,
                                       const CXXRecordDecl *KernelObj) {
    TypeSourceInfo *TSInfo =
        KernelObj->isLambda() ? KernelObj->getLambdaTypeInfo() : nullptr;
    VarDecl *VD = VarDecl::Create(
        Ctx, DC, KernelObj->getLocation(), KernelObj->getLocation(),
        KernelObj->getIdentifier(), QualType(KernelObj->getTypeForDecl(), 0),
        TSInfo, SC_None);

    return VD;
  }

  const llvm::StringLiteral getInitMethodName() const {
    bool IsSIMDKernel = isESIMDKernelType(KernelObj);
    return IsSIMDKernel ? InitESIMDMethodName : InitMethodName;
  }

  // Default inits the type, then calls the init-method in the body.
  bool handleSpecialType(FieldDecl *FD, QualType Ty) {
    addFieldInit(FD, Ty, None,
                 InitializationKind::CreateDefault(KernelCallerSrcLoc));

    addFieldMemberExpr(FD, Ty);

    const auto *RecordDecl = Ty->getAsCXXRecordDecl();
    createSpecialMethodCall(RecordDecl, getInitMethodName(), BodyStmts);
    CXXMethodDecl *FinalizeMethod =
        getMethodByName(RecordDecl, FinalizeMethodName);
    // A finalize-method is expected for special type such as stream.
    if (FinalizeMethod)
      createSpecialMethodCall(RecordDecl, FinalizeMethodName, FinalizeStmts);

    removeFieldMemberExpr(FD, Ty);

    return true;
  }

  bool handleSpecialType(const CXXBaseSpecifier &BS, QualType Ty) {
    const auto *RecordDecl = Ty->getAsCXXRecordDecl();
    addBaseInit(BS, Ty, InitializationKind::CreateDefault(KernelCallerSrcLoc));
    createSpecialMethodCall(RecordDecl, getInitMethodName(), BodyStmts);
    return true;
  }

  // Generate __init call for kernel handler argument
  void handleSpecialType(QualType KernelHandlerTy) {
    DeclRefExpr *KernelHandlerCloneRef =
        DeclRefExpr::Create(SemaRef.Context, NestedNameSpecifierLoc(),
                            KernelCallerSrcLoc, KernelHandlerClone, false,
                            DeclarationNameInfo(), KernelHandlerTy, VK_LValue);
    const auto *RecordDecl =
        KernelHandlerClone->getType()->getAsCXXRecordDecl();
    MemberExprBases.push_back(KernelHandlerCloneRef);
    createSpecialMethodCall(RecordDecl, InitSpecConstantsBuffer, BodyStmts);
    MemberExprBases.pop_back();
  }

  void createKernelHandlerClone(ASTContext &Ctx, DeclContext *DC,
                                ParmVarDecl *KernelHandlerArg) {
    QualType Ty = KernelHandlerArg->getType();
    TypeSourceInfo *TSInfo = Ctx.getTrivialTypeSourceInfo(Ty);
    KernelHandlerClone =
        VarDecl::Create(Ctx, DC, KernelCallerSrcLoc, KernelCallerSrcLoc,
                        KernelHandlerArg->getIdentifier(), Ty, TSInfo, SC_None);

    // Default initialize clone
    InitializedEntity VarEntity =
        InitializedEntity::InitializeVariable(KernelHandlerClone);
    InitializationKind InitKind =
        InitializationKind::CreateDefault(KernelCallerSrcLoc);
    InitializationSequence InitSeq(SemaRef, VarEntity, InitKind, None);
    ExprResult Init = InitSeq.Perform(SemaRef, VarEntity, InitKind, None);
    KernelHandlerClone->setInit(
        SemaRef.MaybeCreateExprWithCleanups(Init.get()));
    KernelHandlerClone->setInitStyle(VarDecl::CallInit);
  }

public:
  static constexpr const bool VisitInsideSimpleContainers = false;
  SyclKernelBodyCreator(Sema &S, SyclKernelDeclCreator &DC,
                        const CXXRecordDecl *KernelObj,
                        FunctionDecl *KernelCallerFunc)
      : SyclKernelFieldHandler(S), DeclCreator(DC),
        KernelObjClone(createKernelObjClone(S.getASTContext(),
                                            DC.getKernelDecl(), KernelObj)),
        VarEntity(InitializedEntity::InitializeVariable(KernelObjClone)),
        KernelObj(KernelObj), KernelCallerFunc(KernelCallerFunc),
        KernelCallerSrcLoc(KernelCallerFunc->getLocation()) {
    CollectionInitExprs.push_back(createInitListExpr(KernelObj));
    markParallelWorkItemCalls();

    Stmt *DS = new (S.Context) DeclStmt(DeclGroupRef(KernelObjClone),
                                        KernelCallerSrcLoc, KernelCallerSrcLoc);
    BodyStmts.push_back(DS);
    DeclRefExpr *KernelObjCloneRef = DeclRefExpr::Create(
        S.Context, NestedNameSpecifierLoc(), KernelCallerSrcLoc, KernelObjClone,
        false, DeclarationNameInfo(), QualType(KernelObj->getTypeForDecl(), 0),
        VK_LValue);
    MemberExprBases.push_back(KernelObjCloneRef);
  }

  ~SyclKernelBodyCreator() {
    CompoundStmt *KernelBody = createKernelBody();
    DeclCreator.setBody(KernelBody);
  }

  bool handleSyclSpecialType(FieldDecl *FD, QualType Ty) final {
    return handleSpecialType(FD, Ty);
  }

  bool handleSyclSpecialType(const CXXRecordDecl *, const CXXBaseSpecifier &BS,
                             QualType Ty) final {
    return handleSpecialType(BS, Ty);
  }

  bool handleSyclSpecConstantType(FieldDecl *FD, QualType Ty) final {
    return handleSpecialType(FD, Ty);
  }

  bool handleSyclHalfType(FieldDecl *FD, QualType Ty) final {
    addSimpleFieldInit(FD, Ty);
    return true;
  }

  bool handlePointerType(FieldDecl *FD, QualType FieldTy) final {
    Expr *PointerRef =
        createPointerParamReferenceExpr(FieldTy, StructDepth != 0);
    addFieldInit(FD, FieldTy, PointerRef);
    return true;
  }

  bool handleSimpleArrayType(FieldDecl *FD, QualType FieldTy) final {
    Expr *ArrayRef = createSimpleArrayParamReferenceExpr(FieldTy);
    InitializationKind InitKind = InitializationKind::CreateDirect({}, {}, {});

    InitializedEntity Entity =
        InitializedEntity::InitializeMember(FD, &VarEntity, /*Implicit*/ true);

    addFieldInit(FD, FieldTy, ArrayRef, InitKind, Entity);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, FieldDecl *FD,
                             QualType Ty) final {
    addSimpleFieldInit(FD, Ty);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *Base,
                             const CXXBaseSpecifier &BS, QualType Ty) final {
    addSimpleBaseInit(BS, Ty);
    return true;
  }

  bool handleScalarType(FieldDecl *FD, QualType FieldTy) final {
    addSimpleFieldInit(FD, FieldTy);
    return true;
  }

  bool handleUnionType(FieldDecl *FD, QualType FieldTy) final {
    addSimpleFieldInit(FD, FieldTy);
    return true;
  }

  // Default inits the type, then calls the init-method in the body
  void handleSyclKernelHandlerType(ParmVarDecl *KernelHandlerArg) {

    // Create and default initialize local clone of kernel handler
    createKernelHandlerClone(SemaRef.getASTContext(),
                             DeclCreator.getKernelDecl(), KernelHandlerArg);

    // Add declaration statement to openCL kernel body
    Stmt *DS =
        new (SemaRef.Context) DeclStmt(DeclGroupRef(KernelHandlerClone),
                                       KernelCallerSrcLoc, KernelCallerSrcLoc);
    BodyStmts.push_back(DS);

    // Generate
    // KernelHandlerClone.__init_specialization_constants_buffer(specialization_constants_buffer)
    // call if target does not have native support for specialization constants.
    // Here, specialization_constants_buffer is the compiler generated kernel
    // argument of type char*.
    if (!isDefaultSPIRArch(SemaRef.Context))
      handleSpecialType(KernelHandlerArg->getType());
  }

  bool enterStruct(const CXXRecordDecl *RD, FieldDecl *FD, QualType Ty) final {
    ++StructDepth;
    addCollectionInitListExpr(Ty->getAsCXXRecordDecl());

    addFieldMemberExpr(FD, Ty);
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, FieldDecl *FD, QualType Ty) final {
    --StructDepth;
    CollectionInitExprs.pop_back();

    removeFieldMemberExpr(FD, Ty);
    return true;
  }

  bool enterStruct(const CXXRecordDecl *RD, const CXXBaseSpecifier &BS,
                   QualType) final {
    ++StructDepth;

    CXXCastPath BasePath;
    QualType DerivedTy(RD->getTypeForDecl(), 0);
    QualType BaseTy = BS.getType();
    SemaRef.CheckDerivedToBaseConversion(DerivedTy, BaseTy, KernelCallerSrcLoc,
                                         SourceRange(), &BasePath,
                                         /*IgnoreBaseAccess*/ true);
    auto Cast = ImplicitCastExpr::Create(
        SemaRef.Context, BaseTy, CK_DerivedToBase, MemberExprBases.back(),
        /* CXXCastPath=*/&BasePath, VK_LValue, FPOptionsOverride());
    MemberExprBases.push_back(Cast);

    addCollectionInitListExpr(BaseTy->getAsCXXRecordDecl());
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *RD, const CXXBaseSpecifier &BS,
                   QualType) final {
    --StructDepth;
    MemberExprBases.pop_back();
    CollectionInitExprs.pop_back();
    return true;
  }

  bool enterArray(FieldDecl *FD, QualType ArrayType,
                  QualType ElementType) final {
    uint64_t ArraySize = SemaRef.getASTContext()
                             .getAsConstantArrayType(ArrayType)
                             ->getSize()
                             .getZExtValue();
    addCollectionInitListExpr(ArrayType, ArraySize);
    ArrayInfos.emplace_back(getFieldEntity(FD, ArrayType), 0);

    // If this is the top-level array, we need to make a MemberExpr in addition
    // to an array subscript.
    addFieldMemberExpr(FD, ArrayType);
    return true;
  }

  bool nextElement(QualType, uint64_t Index) final {
    ArrayInfos.back().second = Index;

    // Pop off the last member expr base.
    if (Index != 0)
      MemberExprBases.pop_back();

    QualType SizeT = SemaRef.getASTContext().getSizeType();

    llvm::APInt IndexVal{
        static_cast<unsigned>(SemaRef.getASTContext().getTypeSize(SizeT)),
        Index, SizeT->isSignedIntegerType()};

    auto IndexLiteral = IntegerLiteral::Create(
        SemaRef.getASTContext(), IndexVal, SizeT, KernelCallerSrcLoc);

    ExprResult IndexExpr = SemaRef.CreateBuiltinArraySubscriptExpr(
        MemberExprBases.back(), KernelCallerSrcLoc, IndexLiteral,
        KernelCallerSrcLoc);

    assert(!IndexExpr.isInvalid());
    MemberExprBases.push_back(IndexExpr.get());
    return true;
  }

  bool leaveArray(FieldDecl *FD, QualType ArrayType,
                  QualType ElementType) final {
    CollectionInitExprs.pop_back();
    ArrayInfos.pop_back();

    // Remove the IndexExpr.
    MemberExprBases.pop_back();

    // Remove the field access expr as well.
    removeFieldMemberExpr(FD, ArrayType);
    return true;
  }

  using SyclKernelFieldHandler::handleSyclHalfType;
};

// Kernels are only the unnamed-lambda feature if the feature is enabled, AND
// the first template argument has been corrected by the library to match the
// functor type.
static bool IsSYCLUnnamedKernel(Sema &SemaRef, const FunctionDecl *FD) {
  if (!SemaRef.getLangOpts().SYCLUnnamedLambda)
    return false;
  QualType FunctorTy = GetSYCLKernelObjectType(FD);
  QualType TmplArgTy = calculateKernelNameType(SemaRef.Context, FD);
  return SemaRef.Context.hasSameType(FunctorTy, TmplArgTy);
}

class SyclKernelIntHeaderCreator : public SyclKernelFieldHandler {
  SYCLIntegrationHeader &Header;
  int64_t CurOffset = 0;
  llvm::SmallVector<size_t, 16> ArrayBaseOffsets;
  int StructDepth = 0;

  // A series of functions to calculate the change in offset based on the type.
  int64_t offsetOf(const FieldDecl *FD, QualType ArgTy) const {
    return isArrayElement(FD, ArgTy)
               ? 0
               : SemaRef.getASTContext().getFieldOffset(FD) / 8;
  }

  int64_t offsetOf(const CXXRecordDecl *RD, const CXXRecordDecl *Base) const {
    const ASTRecordLayout &Layout =
        SemaRef.getASTContext().getASTRecordLayout(RD);
    return Layout.getBaseClassOffset(Base).getQuantity();
  }

  void addParam(const FieldDecl *FD, QualType ArgTy,
                SYCLIntegrationHeader::kernel_param_kind_t Kind) {
    addParam(ArgTy, Kind, offsetOf(FD, ArgTy));
  }
  void addParam(QualType ArgTy, SYCLIntegrationHeader::kernel_param_kind_t Kind,
                uint64_t OffsetAdj) {
    uint64_t Size;
    Size = SemaRef.getASTContext().getTypeSizeInChars(ArgTy).getQuantity();
    Header.addParamDesc(Kind, static_cast<unsigned>(Size),
                        static_cast<unsigned>(CurOffset + OffsetAdj));
  }

  // Returns 'true' if the thing we're visiting (Based on the FD/QualType pair)
  // is an element of an array.  This will determine whether we do
  // MemberExprBases in some cases or not, AND determines how we initialize
  // values.
  bool isArrayElement(const FieldDecl *FD, QualType Ty) const {
    return !SemaRef.getASTContext().hasSameType(FD->getType(), Ty);
  }

public:
  static constexpr const bool VisitInsideSimpleContainers = false;
  SyclKernelIntHeaderCreator(Sema &S, SYCLIntegrationHeader &H,
                             const CXXRecordDecl *KernelObj, QualType NameType,
                             FunctionDecl *KernelFunc)
      : SyclKernelFieldHandler(S), Header(H) {
    bool IsSIMDKernel = isESIMDKernelType(KernelObj);
    Header.startKernel(KernelFunc, NameType, KernelObj->getLocation(),
                       IsSIMDKernel, IsSYCLUnnamedKernel(S, KernelFunc));
  }

  bool handleSyclSpecialType(const CXXRecordDecl *RD,
                             const CXXBaseSpecifier &BC,
                             QualType FieldTy) final {
    const auto *AccTy =
        cast<ClassTemplateSpecializationDecl>(FieldTy->getAsRecordDecl());
    assert(AccTy->getTemplateArgs().size() >= 2 &&
           "Incorrect template args for Accessor Type");
    int Dims = static_cast<int>(
        AccTy->getTemplateArgs()[1].getAsIntegral().getExtValue());
    int Info = getAccessTarget(AccTy) | (Dims << 11);
    Header.addParamDesc(SYCLIntegrationHeader::kind_accessor, Info,
                        CurOffset +
                            offsetOf(RD, BC.getType()->getAsCXXRecordDecl()));
    return true;
  }

  bool handleSyclSpecialType(FieldDecl *FD, QualType FieldTy) final {
    const auto *ClassTy = FieldTy->getAsCXXRecordDecl();
    assert(ClassTy && "Type must be a C++ record type");
    if (Util::isSyclType(FieldTy, "accessor", true /*Tmp*/)) {
      const auto *AccTy =
          cast<ClassTemplateSpecializationDecl>(FieldTy->getAsRecordDecl());
      assert(AccTy->getTemplateArgs().size() >= 2 &&
             "Incorrect template args for Accessor Type");
      int Dims = static_cast<int>(
          AccTy->getTemplateArgs()[1].getAsIntegral().getExtValue());
      int Info = getAccessTarget(AccTy) | (Dims << 11);

      Header.addParamDesc(SYCLIntegrationHeader::kind_accessor, Info,
                          CurOffset + offsetOf(FD, FieldTy));
    } else {
      if (getMethodByName(ClassTy, FinalizeMethodName))
        addParam(FD, FieldTy, SYCLIntegrationHeader::kind_stream);
      else {
        CXXMethodDecl *InitMethod = getMethodByName(ClassTy, InitMethodName);
        assert(InitMethod && "type must have __init method");
        const ParmVarDecl *SamplerArg = InitMethod->getParamDecl(0);
        assert(SamplerArg && "Init method must have arguments");
        addParam(SamplerArg->getType(), SYCLIntegrationHeader::kind_sampler,
                 offsetOf(FD, FieldTy));
      }
    }
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
    const std::string SpecConstName = SYCLUniqueStableNameExpr::ComputeName(
        SemaRef.getASTContext(), SpecConstIDTy);
    Header.addSpecConstant(SpecConstName, SpecConstIDTy);
    return true;
  }

  bool handlePointerType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FD, FieldTy,
             ((StructDepth) ? SYCLIntegrationHeader::kind_std_layout
                            : SYCLIntegrationHeader::kind_pointer));
    return true;
  }

  bool handleScalarType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FD, FieldTy, SYCLIntegrationHeader::kind_std_layout);
    return true;
  }

  bool handleSimpleArrayType(FieldDecl *FD, QualType FieldTy) final {
    // Arrays are always wrapped inside of structs, so just treat it as a simple
    // struct.
    addParam(FD, FieldTy, SYCLIntegrationHeader::kind_std_layout);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, FieldDecl *FD,
                             QualType Ty) final {
    addParam(FD, Ty, SYCLIntegrationHeader::kind_std_layout);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *Base,
                             const CXXBaseSpecifier &, QualType Ty) final {
    addParam(Ty, SYCLIntegrationHeader::kind_std_layout,
             offsetOf(Base, Ty->getAsCXXRecordDecl()));
    return true;
  }

  bool handleUnionType(FieldDecl *FD, QualType FieldTy) final {
    return handleScalarType(FD, FieldTy);
  }

  bool handleSyclHalfType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FD, FieldTy, SYCLIntegrationHeader::kind_std_layout);
    return true;
  }

  void handleSyclKernelHandlerType(QualType Ty) {
    // The compiler generated kernel argument used to initialize SYCL 2020
    // specialization constants, `specialization_constants_buffer`, should
    // have corresponding entry in integration header.
    ASTContext &Context = SemaRef.getASTContext();
    // Offset is zero since kernel_handler argument is not part of
    // kernel object (i.e. it is not captured)
    addParam(Context.getPointerType(Context.CharTy),
             SYCLIntegrationHeader::kind_specialization_constants_buffer, 0);
  }

  bool enterStruct(const CXXRecordDecl *, FieldDecl *FD, QualType Ty) final {
    ++StructDepth;
    CurOffset += offsetOf(FD, Ty);
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, FieldDecl *FD, QualType Ty) final {
    --StructDepth;
    CurOffset -= offsetOf(FD, Ty);
    return true;
  }

  bool enterStruct(const CXXRecordDecl *RD, const CXXBaseSpecifier &BS,
                   QualType) final {
    CurOffset += offsetOf(RD, BS.getType()->getAsCXXRecordDecl());
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *RD, const CXXBaseSpecifier &BS,
                   QualType) final {
    CurOffset -= offsetOf(RD, BS.getType()->getAsCXXRecordDecl());
    return true;
  }

  bool enterArray(FieldDecl *FD, QualType ArrayTy, QualType) final {
    ArrayBaseOffsets.push_back(CurOffset + offsetOf(FD, ArrayTy));
    return true;
  }

  bool nextElement(QualType ET, uint64_t Index) final {
    int64_t Size = SemaRef.getASTContext().getTypeSizeInChars(ET).getQuantity();
    CurOffset = ArrayBaseOffsets.back() + Size * Index;
    return true;
  }

  bool leaveArray(FieldDecl *FD, QualType ArrayTy, QualType) final {
    CurOffset = ArrayBaseOffsets.pop_back_val();
    CurOffset -= offsetOf(FD, ArrayTy);
    return true;
  }

  using SyclKernelFieldHandler::enterStruct;
  using SyclKernelFieldHandler::handleSyclHalfType;
  using SyclKernelFieldHandler::leaveStruct;
};

class SyclKernelIntFooterCreator : public SyclKernelFieldHandler {
  SYCLIntegrationFooter &Footer;

public:
  SyclKernelIntFooterCreator(Sema &S, SYCLIntegrationFooter &F)
      : SyclKernelFieldHandler(S), Footer(F) {
    (void)Footer; // workaround for unused field warning
  }
};

} // namespace

class SYCLKernelNameTypeVisitor
    : public TypeVisitor<SYCLKernelNameTypeVisitor>,
      public ConstTemplateArgumentVisitor<SYCLKernelNameTypeVisitor> {
  Sema &S;
  SourceLocation KernelInvocationFuncLoc;
  QualType KernelNameType;
  using InnerTypeVisitor = TypeVisitor<SYCLKernelNameTypeVisitor>;
  using InnerTemplArgVisitor =
      ConstTemplateArgumentVisitor<SYCLKernelNameTypeVisitor>;
  bool IsInvalid = false;
  bool IsUnnamedKernel = false;

  void VisitTemplateArgs(ArrayRef<TemplateArgument> Args) {
    for (auto &A : Args)
      Visit(A);
  }

public:
  SYCLKernelNameTypeVisitor(Sema &S, SourceLocation KernelInvocationFuncLoc,
                            QualType KernelNameType, bool IsUnnamedKernel)
      : S(S), KernelInvocationFuncLoc(KernelInvocationFuncLoc),
        KernelNameType(KernelNameType), IsUnnamedKernel(IsUnnamedKernel) {}

  bool isValid() { return !IsInvalid; }

  void Visit(QualType T) {
    if (T.isNull())
      return;

    const CXXRecordDecl *RD = T->getAsCXXRecordDecl();
    // If KernelNameType has template args visit each template arg via
    // ConstTemplateArgumentVisitor
    if (const auto *TSD =
            dyn_cast_or_null<ClassTemplateSpecializationDecl>(RD)) {
      ArrayRef<TemplateArgument> Args = TSD->getTemplateArgs().asArray();

      VisitTemplateArgs(Args);
    } else {
      InnerTypeVisitor::Visit(T.getTypePtr());
    }
  }

  void Visit(const TemplateArgument &TA) {
    if (TA.isNull())
      return;
    InnerTemplArgVisitor::Visit(TA);
  }

  void VisitBuiltinType(const BuiltinType *TT) {
    if (TT->isNullPtrType()) {
      S.Diag(KernelInvocationFuncLoc, diag::err_nullptr_t_type_in_sycl_kernel)
          << KernelNameType;

      IsInvalid = true;
    }
    return;
  }

  void VisitTagType(const TagType *TT) {
    return DiagnoseKernelNameType(TT->getDecl());
  }

  void DiagnoseKernelNameType(const NamedDecl *DeclNamed) {
    /*
    This is a helper function which throws an error if the kernel name
    declaration is:
      * declared within namespace 'std' (at any level)
        e.g., namespace std { namespace literals { class Whatever; } }
        h.single_task<std::literals::Whatever>([]() {});
      * declared within a function
        e.g., void foo() { struct S { int i; };
        h.single_task<S>([]() {}); }
      * declared within another tag
        e.g., struct S { struct T { int i } t; };
        h.single_task<S::T>([]() {});
    */

    if (const auto *ED = dyn_cast<EnumDecl>(DeclNamed)) {
      if (!ED->isScoped() && !ED->isFixed()) {
        S.Diag(KernelInvocationFuncLoc, diag::err_sycl_kernel_incorrectly_named)
            << /* unscoped enum requires fixed underlying type */ 1
            << DeclNamed;
        IsInvalid = true;
      }
    }

    const DeclContext *DeclCtx = DeclNamed->getDeclContext();
    if (DeclCtx && !IsUnnamedKernel) {

      // Check if the kernel name declaration is declared within namespace
      // "std" (at any level).
      while (!DeclCtx->isTranslationUnit() && isa<NamespaceDecl>(DeclCtx)) {
        const auto *NSDecl = cast<NamespaceDecl>(DeclCtx);
        if (NSDecl->isStdNamespace()) {
          S.Diag(KernelInvocationFuncLoc,
                 diag::err_invalid_std_type_in_sycl_kernel)
              << KernelNameType << DeclNamed;
          IsInvalid = true;
          return;
        }
        DeclCtx = DeclCtx->getParent();
      }

      // Check if the kernel name is a Tag declaration
      // local to a non-namespace scope (i.e. Inside a function or within
      // another Tag etc).
      if (!DeclCtx->isTranslationUnit() && !isa<NamespaceDecl>(DeclCtx)) {
        if (const auto *Tag = dyn_cast<TagDecl>(DeclNamed)) {
          bool UnnamedLambdaUsed = Tag->getIdentifier() == nullptr;

          if (UnnamedLambdaUsed) {
            S.Diag(KernelInvocationFuncLoc,
                   diag::err_sycl_kernel_incorrectly_named)
                << /* unnamed type is invalid */ 2 << KernelNameType;
            IsInvalid = true;
            return;
          }

          // Diagnose used types without complete definition i.e.
          //   int main() {
          //     class KernelName1;
          //     parallel_for<class KernelName1>(..);
          //   }
          // This case can only be diagnosed during host compilation because the
          // integration header is required to distinguish between the invalid
          // code (above) and the following valid code:
          //   int main() {
          //     parallel_for<class KernelName2>(..);
          //   }
          // The device compiler forward declares both KernelName1 and
          // KernelName2 in the integration header as ::KernelName1 and
          // ::KernelName2. The problem with the former case is the additional
          // declaration 'class KernelName1' in non-global scope. Lookup in this
          // case will resolve to ::main::KernelName1 (instead of
          // ::KernelName1). Since this is not visible to runtime code that
          // submits kernels, this is invalid.
          if (Tag->isCompleteDefinition() ||
              S.getLangOpts().SYCLEnableIntHeaderDiags) {
            S.Diag(KernelInvocationFuncLoc,
                   diag::err_sycl_kernel_incorrectly_named)
                << /* kernel name should be forward declarable at namespace
                      scope */
                0 << KernelNameType;
            IsInvalid = true;
          } else {
            S.Diag(KernelInvocationFuncLoc, diag::warn_sycl_implicit_decl);
            S.Diag(DeclNamed->getLocation(), diag::note_previous_decl)
                << DeclNamed->getName();
          }
        }
      }
    }
  }

  void VisitTypeTemplateArgument(const TemplateArgument &TA) {
    QualType T = TA.getAsType();
    if (const auto *ET = T->getAs<EnumType>())
      VisitTagType(ET);
    else
      Visit(T);
  }

  void VisitIntegralTemplateArgument(const TemplateArgument &TA) {
    QualType T = TA.getIntegralType();
    if (const EnumType *ET = T->getAs<EnumType>())
      VisitTagType(ET);
  }

  void VisitTemplateTemplateArgument(const TemplateArgument &TA) {
    TemplateDecl *TD = TA.getAsTemplate().getAsTemplateDecl();
    assert(TD && "template declaration must be available");
    TemplateParameterList *TemplateParams = TD->getTemplateParameters();
    for (NamedDecl *P : *TemplateParams) {
      if (NonTypeTemplateParmDecl *TemplateParam =
              dyn_cast<NonTypeTemplateParmDecl>(P))
        if (const EnumType *ET = TemplateParam->getType()->getAs<EnumType>())
          VisitTagType(ET);
    }
  }

  void VisitPackTemplateArgument(const TemplateArgument &TA) {
    VisitTemplateArgs(TA.getPackAsArray());
  }
};

void Sema::CheckSYCLKernelCall(FunctionDecl *KernelFunc, SourceRange CallLoc,
                               ArrayRef<const Expr *> Args) {
  QualType KernelNameType =
      calculateKernelNameType(getASTContext(), KernelFunc);
  SYCLKernelNameTypeVisitor KernelNameTypeVisitor(
      *this, Args[0]->getExprLoc(), KernelNameType,
      IsSYCLUnnamedKernel(*this, KernelFunc));
  KernelNameTypeVisitor.Visit(KernelNameType.getCanonicalType());

  // FIXME: In place until the library works around its 'host' invocation
  // issues.
  if (!LangOpts.SYCLIsDevice)
    return;

  const CXXRecordDecl *KernelObj =
      GetSYCLKernelObjectType(KernelFunc)->getAsCXXRecordDecl();

  if (!KernelObj) {
    Diag(Args[0]->getExprLoc(), diag::err_sycl_kernel_not_function_object);
    KernelFunc->setInvalidDecl();
    return;
  }

  if (KernelObj->isLambda()) {
    for (const LambdaCapture &LC : KernelObj->captures())
      if (LC.capturesThis() && LC.isImplicit()) {
        Diag(LC.getLocation(), diag::err_implicit_this_capture);
        KernelFunc->setInvalidDecl();
      }
  }

  // check that calling kernel conforms to spec
  QualType KernelParamTy = KernelFunc->getParamDecl(0)->getType();
  if (KernelParamTy->isReferenceType()) {
    // passing by reference, so emit warning if not using SYCL 2020
    if (LangOpts.getSYCLVersion() < LangOptions::SYCL_2020)
      Diag(KernelFunc->getLocation(), diag::warn_sycl_pass_by_reference_future);
  } else {
    // passing by value.  emit warning if using SYCL 2020 or greater
    if (LangOpts.getSYCLVersion() > LangOptions::SYCL_2017)
      Diag(KernelFunc->getLocation(), diag::warn_sycl_pass_by_value_deprecated);
  }

  // Do not visit invalid kernel object.
  if (KernelObj->isInvalidDecl())
    return;

  bool IsSIMDKernel = isESIMDKernelType(KernelObj);

  SyclKernelDecompMarker DecompMarker(*this);
  SyclKernelFieldChecker FieldChecker(*this, IsSIMDKernel);
  SyclKernelUnionChecker UnionChecker(*this);

  SyclKernelArgsSizeChecker ArgsSizeChecker(*this, Args[0]->getExprLoc(),
                                            IsSIMDKernel);

  KernelObjVisitor Visitor{*this};

  DiagnosingSYCLKernel = true;

  // Emit diagnostics for SYCL device kernels only
  Visitor.VisitRecordBases(KernelObj, FieldChecker, UnionChecker, DecompMarker);
  Visitor.VisitRecordFields(KernelObj, FieldChecker, UnionChecker,
                            DecompMarker);
  // ArgSizeChecker needs to happen after DecompMarker has completed, since it
  // cares about the decomp attributes. DecompMarker cannot run before the
  // others, since it counts on the FieldChecker to make sure it is visiting
  // valid arrays/etc. Thus, ArgSizeChecker has its own visitation.
  if (FieldChecker.isValid() && UnionChecker.isValid()) {
    Visitor.VisitRecordBases(KernelObj, ArgsSizeChecker);
    Visitor.VisitRecordFields(KernelObj, ArgsSizeChecker);
  }
  DiagnosingSYCLKernel = false;
  // Set the kernel function as invalid, if any of the checkers fail validation.
  if (!FieldChecker.isValid() || !UnionChecker.isValid() ||
      !KernelNameTypeVisitor.isValid())
    KernelFunc->setInvalidDecl();
}

// For a wrapped parallel_for, copy attributes from original
// kernel to wrapped kernel.
void Sema::copySYCLKernelAttrs(const CXXRecordDecl *KernelObj) {
  // Get the operator() function of the wrapper.
  CXXMethodDecl *OpParens = getOperatorParens(KernelObj);
  assert(OpParens && "invalid kernel object");

  typedef std::pair<FunctionDecl *, FunctionDecl *> ChildParentPair;
  llvm::SmallPtrSet<FunctionDecl *, 16> Visited;
  llvm::SmallVector<ChildParentPair, 16> WorkList;
  WorkList.push_back({OpParens, nullptr});
  FunctionDecl *KernelBody = nullptr;

  CallGraph SYCLCG;
  SYCLCG.addToCallGraph(getASTContext().getTranslationUnitDecl());
  while (!WorkList.empty()) {
    FunctionDecl *FD = WorkList.back().first;
    FunctionDecl *ParentFD = WorkList.back().second;

    if ((ParentFD == OpParens) && isSYCLKernelBodyFunction(FD)) {
      KernelBody = FD;
      break;
    }

    WorkList.pop_back();
    if (!Visited.insert(FD).second)
      continue; // We've already seen this Decl

    CallGraphNode *N = SYCLCG.getNode(FD);
    if (!N)
      continue;

    for (const CallGraphNode *CI : *N) {
      if (auto *Callee = dyn_cast<FunctionDecl>(CI->getDecl())) {
        Callee = Callee->getMostRecentDecl();
        if (!Visited.count(Callee))
          WorkList.push_back({Callee, FD});
      }
    }
  }

  assert(KernelBody && "improper parallel_for wrap");
  if (KernelBody) {
    llvm::SmallVector<Attr *, 4> Attrs;
    collectSYCLAttributes(*this, KernelBody, Attrs, /*DirectlyCalled*/ true);
    if (!Attrs.empty())
      llvm::for_each(Attrs, [OpParens](Attr *A) { OpParens->addAttr(A); });
  }
}

void Sema::SetSYCLKernelNames() {
  std::unique_ptr<MangleContext> MangleCtx(
      getASTContext().createMangleContext());
  // We assume the list of KernelDescs is the complete list of kernels needing
  // to be rewritten.
  for (const std::pair<const FunctionDecl *, FunctionDecl *> &Pair :
       SyclKernelsToOpenCLKernels) {
    std::string CalculatedName, StableName;
    std::tie(CalculatedName, StableName) =
        constructKernelName(*this, Pair.first, *MangleCtx);
    StringRef KernelName(
        IsSYCLUnnamedKernel(*this, Pair.first) ? StableName : CalculatedName);

    getSyclIntegrationHeader().updateKernelNames(Pair.first, KernelName,
                                                 StableName);

    // Set name of generated kernel.
    Pair.second->setDeclName(&Context.Idents.get(KernelName));
    // Update the AsmLabel for this generated kernel.
    Pair.second->addAttr(AsmLabelAttr::CreateImplicit(Context, KernelName));
  }
}

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
  const CXXRecordDecl *KernelObj =
      GetSYCLKernelObjectType(KernelCallerFunc)->getAsCXXRecordDecl();
  assert(KernelObj && "invalid kernel caller");

  // Do not visit invalid kernel object.
  if (KernelObj->isInvalidDecl())
    return;

  {
    // Do enough to calculate the StableName for the purposes of the hackery
    // below for __pf_kernel_wrapper. Placed in a scope so that we don't
    // accidentially use these values below, before the names are stabililzed.
    std::string CalculatedName, StableName;
    std::tie(CalculatedName, StableName) =
        constructKernelName(*this, KernelCallerFunc, MC);

    // Attributes of a user-written SYCL kernel must be copied to the internally
    // generated alternative kernel, identified by a known string in its name.
    if (StableName.find("__pf_kernel_wrapper") != std::string::npos)
      copySYCLKernelAttrs(KernelObj);
  }

  bool IsSIMDKernel = isESIMDKernelType(KernelObj);

  SyclKernelDeclCreator kernel_decl(*this, KernelObj->getLocation(),
                                    KernelCallerFunc->isInlined(), IsSIMDKernel,
                                    KernelCallerFunc);
  SyclKernelBodyCreator kernel_body(*this, kernel_decl, KernelObj,
                                    KernelCallerFunc);
  SyclKernelIntHeaderCreator int_header(
      *this, getSyclIntegrationHeader(), KernelObj,
      calculateKernelNameType(Context, KernelCallerFunc), KernelCallerFunc);

  SyclKernelIntFooterCreator int_footer(*this, getSyclIntegrationFooter());
  SyclOptReportCreator opt_report(*this, kernel_decl, KernelObj->getLocation());

  KernelObjVisitor Visitor{*this};
  Visitor.VisitRecordBases(KernelObj, kernel_decl, kernel_body, int_header,
                           int_footer, opt_report);
  Visitor.VisitRecordFields(KernelObj, kernel_decl, kernel_body, int_header,
                            int_footer, opt_report);

  if (ParmVarDecl *KernelHandlerArg =
          getSyclKernelHandlerArg(KernelCallerFunc)) {
    kernel_decl.handleSyclKernelHandlerType();
    kernel_body.handleSyclKernelHandlerType(KernelHandlerArg);
    int_header.handleSyclKernelHandlerType(KernelHandlerArg->getType());
    opt_report.handleSyclKernelHandlerType();
  }
}

// Figure out the sub-group for the this function.  First we check the
// attributes, then the global settings.
static std::pair<LangOptions::SubGroupSizeType, int64_t>
CalcEffectiveSubGroup(ASTContext &Ctx, const LangOptions &LO,
                      const FunctionDecl *FD) {
  if (const auto *A = FD->getAttr<IntelReqdSubGroupSizeAttr>()) {
    int64_t Val = getIntExprValue(A->getValue(), Ctx);
    return {LangOptions::SubGroupSizeType::Integer, Val};
  }

  if (const auto *A = FD->getAttr<IntelNamedSubGroupSizeAttr>()) {
    if (A->getType() == IntelNamedSubGroupSizeAttr::Primary)
      return {LangOptions::SubGroupSizeType::Primary, 0};
    return {LangOptions::SubGroupSizeType::Auto, 0};
  }

  // Return the global settings.
  return {LO.getDefaultSubGroupSizeType(),
          static_cast<uint64_t>(LO.DefaultSubGroupSize)};
}

static SourceLocation GetSubGroupLoc(const FunctionDecl *FD) {
  if (const auto *A = FD->getAttr<IntelReqdSubGroupSizeAttr>())
    return A->getLocation();
  if (const auto *A = FD->getAttr<IntelNamedSubGroupSizeAttr>())
    return A->getLocation();
  return SourceLocation{};
}

static void CheckSYCL2020SubGroupSizes(Sema &S, FunctionDecl *SYCLKernel,
                                       const FunctionDecl *FD) {
  // If they are the same, no error.
  if (CalcEffectiveSubGroup(S.Context, S.getLangOpts(), SYCLKernel) ==
      CalcEffectiveSubGroup(S.Context, S.getLangOpts(), FD))
    return;

  // Else we need to figure out why they don't match.
  SourceLocation FDAttrLoc = GetSubGroupLoc(FD);
  SourceLocation KernelAttrLoc = GetSubGroupLoc(SYCLKernel);

  if (FDAttrLoc.isValid()) {
    // This side was caused by an attribute.
    S.Diag(FDAttrLoc, diag::err_sycl_mismatch_group_size)
        << /*kernel called*/ 0;

    if (KernelAttrLoc.isValid()) {
      S.Diag(KernelAttrLoc, diag::note_conflicting_attribute);
    } else {
      // Kernel is 'default'.
      S.Diag(SYCLKernel->getLocation(), diag::note_sycl_kernel_declared_here);
    }
    return;
  }

  // Else this doesn't have an attribute, which can only be caused by this being
  // an undefined SYCL_EXTERNAL, and the kernel has an attribute that conflicts.
  if (const auto *A = SYCLKernel->getAttr<IntelReqdSubGroupSizeAttr>()) {
    // Don't diagnose this if the kernel got its size from the 'old' attribute
    // spelling.
    if (!A->isSYCL2020Spelling())
      return;
  }

  assert(KernelAttrLoc.isValid() && "Kernel doesn't have attribute either?");
  S.Diag(FD->getLocation(), diag::err_sycl_mismatch_group_size)
      << /*undefined SYCL_EXTERNAL*/ 1;
  S.Diag(KernelAttrLoc, diag::note_conflicting_attribute);
}

// Check SYCL2020 Attributes.  2020 attributes don't propogate, they are only
// valid if they match the attribute on the kernel. Note that this is a slight
// difference from what the spec says, which says these attributes are only
// valid on SYCL Kernels and SYCL_EXTERNAL, but we felt that for
// self-documentation purposes that it would be nice to be able to repeat these
// on subsequent functions.
static void CheckSYCL2020Attributes(
    Sema &S, FunctionDecl *SYCLKernel, FunctionDecl *KernelBody,
    const llvm::SmallPtrSetImpl<FunctionDecl *> &CalledFuncs) {

  if (KernelBody) {
    // Make sure the kernel itself has all the 2020 attributes, since we don't
    // do propagation of these.
    if (auto *A = KernelBody->getAttr<IntelReqdSubGroupSizeAttr>())
      if (A->isSYCL2020Spelling())
        SYCLKernel->addAttr(A);
    if (auto *A = KernelBody->getAttr<IntelNamedSubGroupSizeAttr>())
      SYCLKernel->addAttr(A);

    // If the kernel has a body, we should get the attributes for the kernel
    // from there instead, so that we get the functor object.
    SYCLKernel = KernelBody;
  }

  for (auto *FD : CalledFuncs) {
    if (FD == SYCLKernel || FD == KernelBody)
      continue;
    for (auto *Attr : FD->attrs()) {
      switch (Attr->getKind()) {
      case attr::Kind::IntelReqdSubGroupSize:
        // Pre SYCL2020 spellings handled during collection.
        if (!cast<IntelReqdSubGroupSizeAttr>(Attr)->isSYCL2020Spelling())
          break;
        LLVM_FALLTHROUGH;
      case attr::Kind::IntelNamedSubGroupSize:
        CheckSYCL2020SubGroupSizes(S, SYCLKernel, FD);
        break;
      case attr::Kind::SYCLDevice:
        // If a SYCL_EXTERNAL function is not defined in this TU, its necessary
        // that it has a compatible sub-group-size. Don't diagnose if it has a
        // sub-group attribute, we can count on the other checks to catch this.
        if (!FD->isDefined() && !FD->hasAttr<IntelReqdSubGroupSizeAttr>() &&
            !FD->hasAttr<IntelNamedSubGroupSizeAttr>())
          CheckSYCL2020SubGroupSizes(S, SYCLKernel, FD);
        break;
      default:
        break;
      }
    }
  }
}

static void PropagateAndDiagnoseDeviceAttr(
    Sema &S, const SingleDeviceFunctionTracker &Tracker, Attr *A,
    FunctionDecl *SYCLKernel, FunctionDecl *KernelBody) {
  switch (A->getKind()) {
  case attr::Kind::IntelReqdSubGroupSize: {
    auto *Attr = cast<IntelReqdSubGroupSizeAttr>(A);

    if (Attr->isSYCL2020Spelling())
      break;
    const auto *KBSimdAttr =
        KernelBody ? KernelBody->getAttr<SYCLSimdAttr>() : nullptr;
    if (auto *Existing = SYCLKernel->getAttr<IntelReqdSubGroupSizeAttr>()) {
      if (getIntExprValue(Existing->getValue(), S.getASTContext()) !=
          getIntExprValue(Attr->getValue(), S.getASTContext())) {
        S.Diag(SYCLKernel->getLocation(),
               diag::err_conflicting_sycl_kernel_attributes);
        S.Diag(Existing->getLocation(), diag::note_conflicting_attribute);
        S.Diag(Attr->getLocation(), diag::note_conflicting_attribute);
        SYCLKernel->setInvalidDecl();
      }
    } else if (KBSimdAttr &&
               (getIntExprValue(Attr->getValue(), S.getASTContext()) != 1)) {
      reportConflictingAttrs(S, KernelBody, KBSimdAttr, Attr);
    } else {
      SYCLKernel->addAttr(A);
    }
    break;
  }
  case attr::Kind::ReqdWorkGroupSize: {
    auto *RWGSA = cast<ReqdWorkGroupSizeAttr>(A);
    if (auto *Existing = SYCLKernel->getAttr<ReqdWorkGroupSizeAttr>()) {
      ASTContext &Ctx = S.getASTContext();
      if (Existing->getXDimVal(Ctx) != RWGSA->getXDimVal(Ctx) ||
          Existing->getYDimVal(Ctx) != RWGSA->getYDimVal(Ctx) ||
          Existing->getZDimVal(Ctx) != RWGSA->getZDimVal(Ctx)) {
        S.Diag(SYCLKernel->getLocation(),
               diag::err_conflicting_sycl_kernel_attributes);
        S.Diag(Existing->getLocation(), diag::note_conflicting_attribute);
        S.Diag(RWGSA->getLocation(), diag::note_conflicting_attribute);
        SYCLKernel->setInvalidDecl();
      }
    } else if (auto *Existing =
                   SYCLKernel->getAttr<SYCLIntelMaxWorkGroupSizeAttr>()) {
      ASTContext &Ctx = S.getASTContext();
      if (*Existing->getXDimVal() < RWGSA->getXDimVal(Ctx) ||
          *Existing->getYDimVal() < RWGSA->getYDimVal(Ctx) ||
          *Existing->getZDimVal() < RWGSA->getZDimVal(Ctx)) {
        S.Diag(SYCLKernel->getLocation(),
               diag::err_conflicting_sycl_kernel_attributes);
        S.Diag(Existing->getLocation(), diag::note_conflicting_attribute);
        S.Diag(RWGSA->getLocation(), diag::note_conflicting_attribute);
        SYCLKernel->setInvalidDecl();
      } else {
        SYCLKernel->addAttr(A);
      }
    } else {
      SYCLKernel->addAttr(A);
    }
    break;
  }
  case attr::Kind::WorkGroupSizeHint: {
    auto *WGSH = cast<WorkGroupSizeHintAttr>(A);
    if (auto *Existing = SYCLKernel->getAttr<WorkGroupSizeHintAttr>()) {
      if (Existing->getXDimVal() != WGSH->getXDimVal() ||
          Existing->getYDimVal() != WGSH->getYDimVal() ||
          Existing->getZDimVal() != WGSH->getZDimVal()) {
        S.Diag(SYCLKernel->getLocation(),
               diag::err_conflicting_sycl_kernel_attributes);
        S.Diag(Existing->getLocation(), diag::note_conflicting_attribute);
        S.Diag(WGSH->getLocation(), diag::note_conflicting_attribute);
        SYCLKernel->setInvalidDecl();
      }
    }
    SYCLKernel->addAttr(A);
    break;
  }
  case attr::Kind::SYCLIntelMaxWorkGroupSize: {
    auto *SIMWGSA = cast<SYCLIntelMaxWorkGroupSizeAttr>(A);
    if (auto *Existing = SYCLKernel->getAttr<ReqdWorkGroupSizeAttr>()) {
      ASTContext &Ctx = S.getASTContext();
      if (Existing->getXDimVal(Ctx) > *SIMWGSA->getXDimVal() ||
          Existing->getYDimVal(Ctx) > *SIMWGSA->getYDimVal() ||
          Existing->getZDimVal(Ctx) > *SIMWGSA->getZDimVal()) {
        S.Diag(SYCLKernel->getLocation(),
               diag::err_conflicting_sycl_kernel_attributes);
        S.Diag(Existing->getLocation(), diag::note_conflicting_attribute);
        S.Diag(SIMWGSA->getLocation(), diag::note_conflicting_attribute);
        SYCLKernel->setInvalidDecl();
      } else {
        SYCLKernel->addAttr(A);
      }
    } else {
      SYCLKernel->addAttr(A);
    }
    break;
  }
  case attr::Kind::SYCLSimd:
    if (KernelBody && !KernelBody->getAttr<SYCLSimdAttr>()) {
      // Usual kernel can't call ESIMD functions.
      S.Diag(KernelBody->getLocation(),
             diag::err_sycl_function_attribute_mismatch)
          << A;
      S.Diag(A->getLocation(), diag::note_attribute);
      KernelBody->setInvalidDecl();
      break;
    }
    LLVM_FALLTHROUGH;
  case attr::Kind::SYCLIntelKernelArgsRestrict:
  case attr::Kind::SYCLIntelNumSimdWorkItems:
  case attr::Kind::SYCLIntelSchedulerTargetFmaxMhz:
  case attr::Kind::SYCLIntelMaxGlobalWorkDim:
  case attr::Kind::SYCLIntelNoGlobalWorkOffset:
  case attr::Kind::SYCLIntelLoopFuse:
  case attr::Kind::SYCLIntelFPGAMaxConcurrency:
  case attr::Kind::SYCLIntelFPGADisableLoopPipelining:
  case attr::Kind::SYCLIntelFPGAInitiationInterval:
  case attr::Kind::SYCLIntelUseStallEnableClusters:
  case attr::Kind::SYCLDeviceHas:
    SYCLKernel->addAttr(A);
    break;
  case attr::Kind::IntelNamedSubGroupSize:
    // Nothing to do here, handled in the SYCL2020 spelling.
    break;
  // TODO: vec_len_hint should be handled here
  default:
    // Seeing this means that CollectPossibleKernelAttributes was
    // updated while this switch wasn't...or something went wrong
    llvm_unreachable("Unexpected attribute was collected by "
                     "CollectPossibleKernelAttributes");
  }
}

void Sema::MarkDevices() {
  // This Tracker object ensures that the SyclDeviceDecls collection includes
  // the SYCL_EXTERNAL functions, and manages the diagnostics for all of the
  // functions in the kernel.
  DeviceFunctionTracker Tracker(*this);

  for (Decl *D : syclDeviceDecls()) {
    auto *SYCLKernel = cast<FunctionDecl>(D);

    // This type does the actual analysis on a per-kernel basis. It does this to
    // make sure that we're only ever dealing with the context of a single
    // kernel at a time.
    SingleDeviceFunctionTracker T{Tracker, SYCLKernel};

    CheckSYCL2020Attributes(*this, T.GetSYCLKernel(), T.GetKernelBody(),
                            T.GetDeviceFunctions());
    for (auto *A : T.GetCollectedAttributes())
      PropagateAndDiagnoseDeviceAttr(*this, T, A, T.GetSYCLKernel(),
                                     T.GetKernelBody());
  }
}

// -----------------------------------------------------------------------------
// SYCL device specific diagnostics implementation
// -----------------------------------------------------------------------------

Sema::SemaDiagnosticBuilder
Sema::SYCLDiagIfDeviceCode(SourceLocation Loc, unsigned DiagID,
                           DeviceDiagnosticReason Reason) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL compilation");
  FunctionDecl *FD = dyn_cast<FunctionDecl>(getCurLexicalContext());
  SemaDiagnosticBuilder::Kind DiagKind = [this, FD, Reason] {
    if (DiagnosingSYCLKernel)
      return SemaDiagnosticBuilder::K_ImmediateWithCallStack;
    if (!FD)
      return SemaDiagnosticBuilder::K_Nop;
    if (getEmissionStatus(FD) == Sema::FunctionEmissionStatus::Emitted) {
      // Skip the diagnostic if we know it won't be emitted.
      if ((getEmissionReason(FD) & Reason) ==
          Sema::DeviceDiagnosticReason::None)
        return SemaDiagnosticBuilder::K_Nop;

      return SemaDiagnosticBuilder::K_ImmediateWithCallStack;
    }
    return SemaDiagnosticBuilder::K_Deferred;
  }();
  return SemaDiagnosticBuilder(DiagKind, Loc, DiagID, FD, *this, Reason);
}

bool Sema::checkSYCLDeviceFunction(SourceLocation Loc, FunctionDecl *Callee) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL compilation");
  assert(Callee && "Callee may not be null.");

  // Errors in an unevaluated context don't need to be generated,
  // so we can safely skip them.
  if (isUnevaluatedContext() || isConstantEvaluated())
    return true;

  FunctionDecl *Caller = dyn_cast<FunctionDecl>(getCurLexicalContext());

  if (!Caller)
    return true;

  SemaDiagnosticBuilder::Kind DiagKind = SemaDiagnosticBuilder::K_Nop;

  // TODO Set DiagKind to K_Immediate/K_Deferred to emit diagnostics for Callee
  SemaDiagnosticBuilder(DiagKind, Loc, diag::err_sycl_restrict, Caller, *this,
                        DeviceDiagnosticReason::Sycl)
      << Sema::KernelCallUndefinedFunction;
  SemaDiagnosticBuilder(DiagKind, Callee->getLocation(),
                        diag::note_previous_decl, Caller, *this,
                        DeviceDiagnosticReason::Sycl)
      << Callee;

  return DiagKind != SemaDiagnosticBuilder::K_Immediate &&
         DiagKind != SemaDiagnosticBuilder::K_ImmediateWithCallStack;
}

void Sema::deepTypeCheckForSYCLDevice(SourceLocation UsedAt,
                                      llvm::DenseSet<QualType> Visited,
                                      ValueDecl *DeclToCheck) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL compilation");
  // Emit notes only for the first discovered declaration of unsupported type
  // to avoid mess of notes. This flag is to track that error already happened.
  bool NeedToEmitNotes = true;

  auto Check = [&](QualType TypeToCheck, const ValueDecl *D) {
    bool ErrorFound = false;
    if (isZeroSizedArray(*this, TypeToCheck)) {
      SYCLDiagIfDeviceCode(UsedAt, diag::err_typecheck_zero_array_size) << 1;
      ErrorFound = true;
    }
    // Checks for other types can also be done here.
    if (ErrorFound) {
      if (NeedToEmitNotes) {
        if (auto *FD = dyn_cast<FieldDecl>(D))
          SYCLDiagIfDeviceCode(FD->getLocation(),
                               diag::note_illegal_field_declared_here)
              << FD->getType()->isPointerType() << FD->getType();
        else
          SYCLDiagIfDeviceCode(D->getLocation(), diag::note_declared_at);
      }
    }

    return ErrorFound;
  };

  // In case we have a Record used do the DFS for a bad field.
  SmallVector<const ValueDecl *, 4> StackForRecursion;
  StackForRecursion.push_back(DeclToCheck);

  // While doing DFS save how we get there to emit a nice set of notes.
  SmallVector<const FieldDecl *, 4> History;
  History.push_back(nullptr);

  do {
    const ValueDecl *Next = StackForRecursion.pop_back_val();
    if (!Next) {
      assert(!History.empty());
      // Found a marker, we have gone up a level.
      History.pop_back();
      continue;
    }
    QualType NextTy = Next->getType();

    if (!Visited.insert(NextTy).second)
      continue;

    auto EmitHistory = [&]() {
      // The first element is always nullptr.
      for (uint64_t Index = 1; Index < History.size(); ++Index) {
        SYCLDiagIfDeviceCode(History[Index]->getLocation(),
                             diag::note_within_field_of_type)
            << History[Index]->getType();
      }
    };

    if (Check(NextTy, Next)) {
      if (NeedToEmitNotes)
        EmitHistory();
      NeedToEmitNotes = false;
    }

    // In case pointer/array/reference type is met get pointee type, then
    // proceed with that type.
    while (NextTy->isAnyPointerType() || NextTy->isArrayType() ||
           NextTy->isReferenceType()) {
      if (NextTy->isArrayType())
        NextTy = QualType{NextTy->getArrayElementTypeNoTypeQual(), 0};
      else
        NextTy = NextTy->getPointeeType();
      if (Check(NextTy, Next)) {
        if (NeedToEmitNotes)
          EmitHistory();
        NeedToEmitNotes = false;
      }
    }

    if (const auto *RecDecl = NextTy->getAsRecordDecl()) {
      if (auto *NextFD = dyn_cast<FieldDecl>(Next))
        History.push_back(NextFD);
      // When nullptr is discovered, this means we've gone back up a level, so
      // the history should be cleaned.
      StackForRecursion.push_back(nullptr);
      llvm::copy(RecDecl->fields(), std::back_inserter(StackForRecursion));
    }
  } while (!StackForRecursion.empty());
}

void Sema::finalizeSYCLDelayedAnalysis(const FunctionDecl *Caller,
                                       const FunctionDecl *Callee,
                                       SourceLocation Loc,
                                       DeviceDiagnosticReason Reason) {
  Callee = Callee->getMostRecentDecl();

  // If the reason for the emission of this diagnostic is not SYCL-specific,
  // and it is not known to be reachable from a routine on device, do not
  // issue a diagnostic.
  if ((Reason & DeviceDiagnosticReason::Sycl) == DeviceDiagnosticReason::None &&
      !isFDReachableFromSyclDevice(Callee, Caller))
    return;

  // If Callee has a SYCL attribute, no diagnostic needed.
  if (Callee->hasAttr<SYCLDeviceAttr>() || Callee->hasAttr<SYCLKernelAttr>())
    return;

  // Diagnose if this is an undefined function and it is not a builtin.
  // Currently, there is an exception of "__failed_assertion" in libstdc++-11,
  // this undefined function is used to trigger a compiling error.
  if (!Callee->isDefined() && !Callee->getBuiltinID() &&
      !Callee->isReplaceableGlobalAllocationFunction() &&
      !isSYCLUndefinedAllowed(Callee, getSourceManager())) {
    Diag(Loc, diag::err_sycl_restrict) << Sema::KernelCallUndefinedFunction;
    Diag(Callee->getLocation(), diag::note_previous_decl) << Callee;
    Diag(Caller->getLocation(), diag::note_called_by) << Caller;
  }
}

bool Sema::checkAllowedSYCLInitializer(VarDecl *VD) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL compilation");

  if (VD->isInvalidDecl() || !VD->hasInit() || !VD->hasGlobalStorage())
    return true;

  const Expr *Init = VD->getInit();
  bool ValueDependent = Init && Init->isValueDependent();
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
    CASE(stream);
    CASE(specialization_constants_buffer);
    CASE(pointer);
  }
  return "<ERROR>";

#undef CASE
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
class SYCLFwdDeclEmitter
    : public TypeVisitor<SYCLFwdDeclEmitter>,
      public ConstTemplateArgumentVisitor<SYCLFwdDeclEmitter> {
  using InnerTypeVisitor = TypeVisitor<SYCLFwdDeclEmitter>;
  using InnerTemplArgVisitor = ConstTemplateArgumentVisitor<SYCLFwdDeclEmitter>;
  raw_ostream &OS;
  llvm::SmallPtrSet<const NamedDecl *, 4> Printed;
  PrintingPolicy Policy;

  void printForwardDecl(NamedDecl *D) {
    // wrap the declaration into namespaces if needed
    unsigned NamespaceCnt = 0;
    std::string NSStr = "";
    const DeclContext *DC = D->getDeclContext();

    while (DC) {
      if (const auto *NS = dyn_cast<NamespaceDecl>(DC)) {
        ++NamespaceCnt;
        StringRef NSInlinePrefix = NS->isInline() ? "inline " : "";
        NSStr.insert(
            0,
            Twine(NSInlinePrefix + "namespace " + NS->getName() + " { ").str());
        DC = NS->getDeclContext();
      } else {
        // We should be able to handle a subset of the decl-context types to
        // make our namespaces for forward declarations as specific as possible,
        // so just skip them here.  We can't use their names, since they would
        // not be forward declarable, but we can try to make them as specific as
        // possible.
        // This permits things such as:
        // namespace N1 { void foo() { kernel<class K>(...); }}
        // and
        // namespace N2 { void foo() { kernel<class K>(...); }}
        // to co-exist, despite technically being against the SYCL rules.
        // See SYCLKernelNameTypePrinter for the corresponding part that prints
        // the kernel information for this type. These two must match.
        if (isa<FunctionDecl, RecordDecl, LinkageSpecDecl>(DC)) {
          DC = cast<Decl>(DC)->getDeclContext();
        } else {
          break;
        }
      }
    }
    OS << NSStr;
    if (NamespaceCnt > 0)
      OS << "\n";

    D->print(OS, Policy);

    if (const auto *ED = dyn_cast<EnumDecl>(D)) {
      QualType T = ED->getIntegerType();
      // Backup since getIntegerType() returns null for enum forward
      // declaration with no fixed underlying type
      if (T.isNull())
        T = ED->getPromotionType();
      OS << " : " << T.getAsString();
    }

    OS << ";\n";

    // print closing braces for namespaces if needed
    for (unsigned I = 0; I < NamespaceCnt; ++I)
      OS << "}";
    if (NamespaceCnt > 0)
      OS << "\n";
  }

  // Checks if we've already printed forward declaration and prints it if not.
  void checkAndEmitForwardDecl(NamedDecl *D) {
    if (Printed.insert(D).second)
      printForwardDecl(D);
  }

  void VisitTemplateArgs(ArrayRef<TemplateArgument> Args) {
    for (size_t I = 0, E = Args.size(); I < E; ++I)
      Visit(Args[I]);
  }

public:
  SYCLFwdDeclEmitter(raw_ostream &OS, LangOptions LO) : OS(OS), Policy(LO) {
    Policy.adjustForCPlusPlusFwdDecl();
    Policy.SuppressTypedefs = true;
    Policy.SuppressUnwrittenScope = true;
    Policy.PrintCanonicalTypes = true;
    Policy.SkipCanonicalizationOfTemplateTypeParms = true;
  }

  void Visit(QualType T) {
    if (T.isNull())
      return;
    InnerTypeVisitor::Visit(T.getTypePtr());
  }

  void VisitReferenceType(const ReferenceType *RT) {
    // Our forward declarations don't care about references, so we should just
    // ignore the reference and continue on.
    Visit(RT->getPointeeType());
  }

  void Visit(const TemplateArgument &TA) {
    if (TA.isNull())
      return;
    InnerTemplArgVisitor::Visit(TA);
  }

  void VisitPointerType(const PointerType *T) {
    // Peel off the pointer types.
    QualType PT = T->getPointeeType();
    while (PT->isPointerType())
      PT = PT->getPointeeType();
    Visit(PT);
  }

  void VisitTagType(const TagType *T) {
    TagDecl *TD = T->getDecl();
    if (const auto *TSD = dyn_cast<ClassTemplateSpecializationDecl>(TD)) {
      // - first, recurse into template parameters and emit needed forward
      //   declarations
      ArrayRef<TemplateArgument> Args = TSD->getTemplateArgs().asArray();
      VisitTemplateArgs(Args);
      // - second, emit forward declaration for the template class being
      //   specialized
      ClassTemplateDecl *CTD = TSD->getSpecializedTemplate();
      assert(CTD && "template declaration must be available");

      checkAndEmitForwardDecl(CTD);
      return;
    }
    checkAndEmitForwardDecl(TD);
  }

  void VisitTypeTemplateArgument(const TemplateArgument &TA) {
    QualType T = TA.getAsType();
    Visit(T);
  }

  void VisitIntegralTemplateArgument(const TemplateArgument &TA) {
    QualType T = TA.getIntegralType();
    if (const EnumType *ET = T->getAs<EnumType>())
      VisitTagType(ET);
  }

  void VisitTemplateTemplateArgument(const TemplateArgument &TA) {
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
    TemplateDecl *TD = TA.getAsTemplate().getAsTemplateDecl();
    assert(TD && "template declaration must be available");
    TemplateParameterList *TemplateParams = TD->getTemplateParameters();
    for (NamedDecl *P : *TemplateParams) {
      // If template template parameter type has an enum value template
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
              dyn_cast<NonTypeTemplateParmDecl>(P))
        if (const EnumType *ET = TemplateParam->getType()->getAs<EnumType>())
          VisitTagType(ET);
    }
    checkAndEmitForwardDecl(TD);
  }

  void VisitPackTemplateArgument(const TemplateArgument &TA) {
    VisitTemplateArgs(TA.getPackAsArray());
  }
};

class SYCLKernelNameTypePrinter
    : public TypeVisitor<SYCLKernelNameTypePrinter>,
      public ConstTemplateArgumentVisitor<SYCLKernelNameTypePrinter> {
  using InnerTypeVisitor = TypeVisitor<SYCLKernelNameTypePrinter>;
  using InnerTemplArgVisitor =
      ConstTemplateArgumentVisitor<SYCLKernelNameTypePrinter>;
  raw_ostream &OS;
  PrintingPolicy &Policy;

  void printTemplateArgs(ArrayRef<TemplateArgument> Args) {
    for (size_t I = 0, E = Args.size(); I < E; ++I) {
      const TemplateArgument &Arg = Args[I];
      // If argument is an empty pack argument, skip printing comma and
      // argument.
      if (Arg.getKind() == TemplateArgument::ArgKind::Pack && !Arg.pack_size())
        continue;

      if (I)
        OS << ", ";

      Visit(Arg);
    }
  }

  void VisitQualifiers(Qualifiers Quals) {
    Quals.print(OS, Policy, /*appendSpaceIfNotEmpty*/ true);
  }

  // Use recursion to print the namespace-qualified name for the purposes of the
  // canonical sycl example of a type being created in the kernel call.
  void PrintNamespaceScopes(const DeclContext *DC) {
    if (isa<NamespaceDecl, FunctionDecl, RecordDecl, LinkageSpecDecl>(DC)) {
      PrintNamespaceScopes(DC->getParent());

      const auto *NS = dyn_cast<NamespaceDecl>(DC);
      if (NS && !NS->isAnonymousNamespace())
        OS << NS->getName() << "::";
    }
  }

public:
  SYCLKernelNameTypePrinter(raw_ostream &OS, PrintingPolicy &Policy)
      : OS(OS), Policy(Policy) {}

  void Visit(QualType T) {
    if (T.isNull())
      return;

    QualType CT = T.getCanonicalType();
    VisitQualifiers(CT.getQualifiers());

    InnerTypeVisitor::Visit(CT.getTypePtr());
  }

  void VisitType(const Type *T) {
    OS << QualType::getAsString(T, Qualifiers(), Policy);
  }

  void Visit(const TemplateArgument &TA) {
    if (TA.isNull())
      return;
    InnerTemplArgVisitor::Visit(TA);
  }

  void VisitTagType(const TagType *T) {
    TagDecl *RD = T->getDecl();
    if (const auto *TSD = dyn_cast<ClassTemplateSpecializationDecl>(RD)) {

      // Print template class name
      TSD->printQualifiedName(OS, Policy, /*WithGlobalNsPrefix*/ true);

      ArrayRef<TemplateArgument> Args = TSD->getTemplateArgs().asArray();
      OS << "<";
      printTemplateArgs(Args);
      OS << ">";

      return;
    }

    // Handle the canonical sycl example where the type is created for the first
    // time in the kernel naming. We want to qualify this as fully as we can,
    // but not in a way that won't be forward declarable.  See
    // SYCLFwdDeclEmitter::printForwardDecl for the corresponding list for
    // printing the forward declaration, these two must match.
    DeclContext *DC = RD->getDeclContext();
    if (isa<FunctionDecl, RecordDecl, LinkageSpecDecl>(DC)) {
      PrintNamespaceScopes(DC);
      RD->printName(OS);
      return;
    }

    const NamespaceDecl *NS = dyn_cast<NamespaceDecl>(RD->getDeclContext());
    RD->printQualifiedName(OS, Policy, !(NS && NS->isAnonymousNamespace()));
  }

  void VisitTemplateArgument(const TemplateArgument &TA) {
    TA.print(Policy, OS, false /* IncludeType */);
  }

  void VisitTypeTemplateArgument(const TemplateArgument &TA) {
    Policy.SuppressTagKeyword = true;
    QualType T = TA.getAsType();
    Visit(T);
    Policy.SuppressTagKeyword = false;
  }

  void VisitIntegralTemplateArgument(const TemplateArgument &TA) {
    QualType T = TA.getIntegralType();
    if (const EnumType *ET = T->getAs<EnumType>()) {
      const llvm::APSInt &Val = TA.getAsIntegral();
      OS << "static_cast<";
      ET->getDecl()->printQualifiedName(OS, Policy,
                                        /*WithGlobalNsPrefix*/ true);
      OS << ">(" << Val << ")";
    } else {
      TA.print(Policy, OS, false /* IncludeType */);
    }
  }

  void VisitTemplateTemplateArgument(const TemplateArgument &TA) {
    TemplateDecl *TD = TA.getAsTemplate().getAsTemplateDecl();
    TD->printQualifiedName(OS, Policy);
  }

  void VisitPackTemplateArgument(const TemplateArgument &TA) {
    printTemplateArgs(TA.getPackAsArray());
  }
};

static void OutputStableNameChar(raw_ostream &O, char C) {
  // If it is reliably printable, print in the integration header as a
  // character. Else just print it as the integral representation.
  if (llvm::isPrint(C))
    O << '\'' << C << '\'';
  else
    O << static_cast<short>(C);
}

static void OutputStableNameInChars(raw_ostream &O, StringRef Name) {
  assert(!Name.empty() && "Expected a nonempty string!");
  OutputStableNameChar(O, Name[0]);

  for (char C : Name.substr(1)) {
    O << ", ";
    OutputStableNameChar(O, C);
  }
}

void SYCLIntegrationHeader::emit(raw_ostream &O) {
  O << "// This is auto-generated SYCL integration header.\n";
  O << "\n";

  O << "#include <CL/sycl/detail/defines_elementary.hpp>\n";
  O << "#include <CL/sycl/detail/kernel_desc.hpp>\n";

  O << "\n";

  LangOptions LO;
  PrintingPolicy Policy(LO);
  Policy.SuppressTypedefs = true;
  Policy.SuppressUnwrittenScope = true;
  SYCLFwdDeclEmitter FwdDeclEmitter(O, S.getLangOpts());

  // Predefines which need to be set for custom host compilation
  // must be defined in integration header.
  for (const std::pair<StringRef, StringRef> &Macro :
       getSYCLVersionMacros(S.getLangOpts())) {
    O << "#ifndef " << Macro.first << '\n';
    O << "#define " << Macro.first << " " << Macro.second << '\n';
    O << "#endif //" << Macro.first << "\n\n";
  }

  if (S.getLangOpts().SYCLDisableRangeRounding) {
    O << "#ifndef __SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ \n";
    O << "#define __SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ 1\n";
    O << "#endif //__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__\n\n";
  }

  if (SpecConsts.size() > 0) {
    O << "// Forward declarations of templated spec constant types:\n";
    for (const auto &SC : SpecConsts)
      FwdDeclEmitter.Visit(SC.first);
    O << "\n";

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
      SYCLKernelNameTypePrinter Printer(O, Policy);
      Printer.Visit(P.first);
      O << "> {\n";
      O << "  static constexpr const char* getName() {\n";
      O << "    return \"" << P.second << "\";\n";
      O << "  }\n";
      O << "};\n";
    }
  }

  O << "// Forward declarations of templated kernel function types:\n";
  for (const KernelDesc &K : KernelDescs)
    if (!K.IsUnnamedKernel)
      FwdDeclEmitter.Visit(K.NameType);
  O << "\n";

  O << "__SYCL_INLINE_NAMESPACE(cl) {\n";
  O << "namespace sycl {\n";
  O << "namespace detail {\n";

  // Generate declaration of variable of type __sycl_device_global_registration
  // whose sole purpose is to run its constructor before the application's
  // main() function.

  if (S.getSyclIntegrationFooter().isDeviceGlobalsEmitted()) {
    O << "namespace {\n";

    O << "class __sycl_device_global_registration {\n";
    O << "public:\n";
    O << "  __sycl_device_global_registration() noexcept;\n";
    O << "};\n";
    O << "__sycl_device_global_registration __sycl_device_global_registrar;\n";

    O << "} // namespace\n";

    O << "\n";
  }

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

  // Sentinel in place for 2 reasons:
  // 1- to make sure we don't get a warning because this collection is empty.
  // 2- to provide an obvious value that we can use when debugging to see that
  //    we have left valid kernel information.
  // integer-field values are negative, so they are obviously invalid, notable
  // enough to 'stick out' and 'negative enough' to not be easily reachable by a
  // mathematical error.
  O << "  { kernel_param_kind_t::kind_invalid, -987654321, -987654321 }, \n";
  O << "};\n\n";

  O << "// Specializations of KernelInfo for kernel function types:\n";
  unsigned CurStart = 0;

  for (const KernelDesc &K : KernelDescs) {
    const size_t N = K.Params.size();
    PresumedLoc PLoc = S.Context.getSourceManager().getPresumedLoc(
        S.Context.getSourceManager()
            .getExpansionRange(K.KernelLocation)
            .getEnd());
    if (K.IsUnnamedKernel) {
      O << "template <> struct KernelInfoData<";
      OutputStableNameInChars(O, K.StableName);
      O << "> {\n";
    } else {
      O << "template <> struct KernelInfo<";
      SYCLKernelNameTypePrinter Printer(O, Policy);
      Printer.Visit(K.NameType);
      O << "> {\n";
    }
    O << "  __SYCL_DLL_LOCAL\n";
    O << "  static constexpr const char* getName() { return \"" << K.Name
      << "\"; }\n";
    O << "  __SYCL_DLL_LOCAL\n";
    O << "  static constexpr unsigned getNumParams() { return " << N << "; }\n";
    O << "  __SYCL_DLL_LOCAL\n";
    O << "  static constexpr const kernel_param_desc_t& ";
    O << "getParamDesc(unsigned i) {\n";
    O << "    return kernel_signatures[i+" << CurStart << "];\n";
    O << "  }\n";
    O << "  __SYCL_DLL_LOCAL\n";
    O << "  static constexpr bool isESIMD() { return " << K.IsESIMDKernel
      << "; }\n";
    O << "  __SYCL_DLL_LOCAL\n";
    O << "  static constexpr const char* getFileName() {\n";
    O << "#ifndef NDEBUG\n";
    O << "    return \""
      << std::string(PLoc.getFilename())
             .substr(std::string(PLoc.getFilename()).find_last_of("/\\") + 1);
    O << "\";\n";
    O << "#else\n";
    O << "    return \"\";\n";
    O << "#endif\n";
    O << "  }\n";
    O << "  __SYCL_DLL_LOCAL\n";
    O << "  static constexpr const char* getFunctionName() {\n";
    O << "#ifndef NDEBUG\n";
    O << "    return \"";
    SYCLKernelNameTypePrinter Printer(O, Policy);
    Printer.Visit(K.NameType);
    O << "\";\n";
    O << "#else\n";
    O << "    return \"\";\n";
    O << "#endif\n";
    O << "  }\n";
    O << "  __SYCL_DLL_LOCAL\n";
    O << "  static constexpr unsigned getLineNumber() {\n";
    O << "#ifndef NDEBUG\n";
    O << "    return " << PLoc.getLine() << ";\n";
    O << "#else\n";
    O << "    return 0;\n";
    O << "#endif\n";
    O << "  }\n";
    O << "  __SYCL_DLL_LOCAL\n";
    O << "  static constexpr unsigned getColumnNumber() {\n";
    O << "#ifndef NDEBUG\n";
    O << "    return " << PLoc.getColumn() << ";\n";
    O << "#else\n";
    O << "    return 0;\n";
    O << "#endif\n";
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

bool SYCLIntegrationHeader::emit(StringRef IntHeaderName) {
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

void SYCLIntegrationHeader::startKernel(const FunctionDecl *SyclKernel,
                                        QualType KernelNameType,
                                        SourceLocation KernelLocation,
                                        bool IsESIMDKernel,
                                        bool IsUnnamedKernel) {
  KernelDescs.emplace_back(SyclKernel, KernelNameType, KernelLocation,
                           IsESIMDKernel, IsUnnamedKernel);
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

SYCLIntegrationHeader::SYCLIntegrationHeader(Sema &S) : S(S) {}

void SYCLIntegrationFooter::addVarDecl(const VarDecl *VD) {
  // Variable template declaration can result in an error case of 'nullptr'
  // here.
  if (!VD)
    return;
  // Skip the dependent version of these variables, we only care about them
  // after instantiation.
  if (VD->getDeclContext()->isDependentContext())
    return;

  // Skip partial specializations of a variable template, treat other variable
  // template instantiations as a VarDecl.
  if (isa<VarTemplatePartialSpecializationDecl>(VD))
    return;
  // Step 1: ensure that this is of the correct type template specialization.
  if (!Util::isSyclSpecIdType(VD->getType()) &&
      !Util::isSyclDeviceGlobalType(VD->getType())) {
    // Handle the case where this could be a deduced type, such as a deduction
    // guide. We have to do this here since this function, unlike most of the
    // rest of this file, is called during Sema instead of after it. We will
    // also have to filter out after deduction later.
    QualType Ty = VD->getType().getCanonicalType();
    if (!Ty->isUndeducedType())
      return;
  }
  // Step 2: ensure that this is a static member, or a namespace-scope.
  // Note that isLocalVarDeclorParm excludes thread-local and static-local
  // intentionally, as there is no way to 'spell' one of those in the
  // specialization. We just don't generate the specialization for those, and
  // let an error happen during host compilation.
  if (!VD->hasGlobalStorage() || VD->isLocalVarDeclOrParm())
    return;
  // Step 3: Add to collection.
  GlobalVars.push_back(VD);
}

// Post-compile integration header support.
bool SYCLIntegrationFooter::emit(StringRef IntHeaderName) {
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
  return emit(Out);
}

template <typename BeforeFn, typename AfterFn>
static void PrintNSHelper(BeforeFn Before, AfterFn After, raw_ostream &OS,
                          const DeclContext *DC) {
  if (DC->isTranslationUnit())
    return;

  const auto *CurDecl = cast<Decl>(DC);
  // Ensure we are in the canonical version, so that we know we have the 'full'
  // name of the thing.
  CurDecl = CurDecl->getCanonicalDecl();

  // We are intentionally skipping linkage decls and record decls.  Namespaces
  // can appear in a linkage decl, but not a record decl, so we don't have to
  // worry about the names getting messed up from that.  We handle record decls
  // later when printing the name of the thing.
  const auto *NS = dyn_cast<NamespaceDecl>(CurDecl);
  if (NS)
    Before(OS, NS);

  if (const DeclContext *NewDC = CurDecl->getDeclContext())
    PrintNSHelper(Before, After, OS, NewDC);

  if (NS)
    After(OS, NS);
}

static void PrintNamespaces(raw_ostream &OS, const DeclContext *DC) {
  PrintNSHelper([](raw_ostream &OS, const NamespaceDecl *NS) {},
                [](raw_ostream &OS, const NamespaceDecl *NS) {
                  if (NS->isInline())
                    OS << "inline ";
                  OS << "namespace ";
                  if (!NS->isAnonymousNamespace())
                    OS << NS->getName() << " ";
                  OS << "{\n";
                },
                OS, DC);
}

static void PrintNSClosingBraces(raw_ostream &OS, const DeclContext *DC) {
  PrintNSHelper(
      [](raw_ostream &OS, const NamespaceDecl *NS) {
        OS << "} // ";
        if (NS->isInline())
          OS << "inline ";

        OS << "namespace ";
        if (!NS->isAnonymousNamespace())
          OS << NS->getName();

        OS << '\n';
      },
      [](raw_ostream &OS, const NamespaceDecl *NS) {}, OS, DC);
}

static std::string EmitShim(raw_ostream &OS, unsigned &ShimCounter,
                            const std::string &LastShim,
                            const NamespaceDecl *AnonNS) {
  std::string NewShimName =
      "__sycl_detail::__shim_" + std::to_string(ShimCounter) + "()";
  // Print opening-namespace
  PrintNamespaces(OS, Decl::castToDeclContext(AnonNS));
  OS << "namespace __sycl_detail {\n";
  OS << "static constexpr decltype(" << LastShim << ") &__shim_" << ShimCounter
     << "() {\n";
  OS << "  return " << LastShim << ";\n";
  OS << "}\n";
  OS << "} // namespace __sycl_detail\n";
  PrintNSClosingBraces(OS, Decl::castToDeclContext(AnonNS));

  ++ShimCounter;
  return NewShimName;
}

// Emit the list of shims required for a DeclContext, calls itself recursively.
static void EmitShims(raw_ostream &OS, unsigned &ShimCounter,
                      const DeclContext *DC, std::string &NameForLastShim) {
  if (DC->isTranslationUnit()) {
    NameForLastShim = "::" + NameForLastShim;
    return;
  }

  const auto *CurDecl = cast<Decl>(DC)->getCanonicalDecl();

  // We skip linkage decls, since they don't modify the Qualified name.
  if (const auto *RD = dyn_cast<RecordDecl>(CurDecl)) {
    NameForLastShim = RD->getNameAsString() + "::" + NameForLastShim;
  } else if (const auto *ND = dyn_cast<NamespaceDecl>(CurDecl)) {
    if (ND->isAnonymousNamespace()) {
      // Print current shim, reset 'name for last shim'.
      NameForLastShim = EmitShim(OS, ShimCounter, NameForLastShim, ND);
    } else {
      NameForLastShim = ND->getNameAsString() + "::" + NameForLastShim;
    }
  } else {
    // FIXME: I don't believe there are other declarations that these variables
    // could possibly find themselves in. LinkageDecls don't change the
    // qualified name, so there is nothing to do here. At one point we should
    // probably convince ourselves that this is entire list and remove this
    // comment.
    assert((isa<LinkageSpecDecl, ExternCContextDecl>(CurDecl)) &&
           "Unhandled decl type");
  }

  EmitShims(OS, ShimCounter, CurDecl->getDeclContext(), NameForLastShim);
}

// Emit the list of shims required for a variable declaration.
// Returns a string containing the FQN of the 'top most' shim, including its
// function call parameters.
static std::string EmitShims(raw_ostream &OS, unsigned &ShimCounter,
                             PrintingPolicy &Policy, const VarDecl *VD) {
  if (!VD->isInAnonymousNamespace())
    return "";
  std::string RelativeName;
  llvm::raw_string_ostream stream(RelativeName);
  VD->getNameForDiagnostic(stream, Policy, false);
  stream.flush();

  EmitShims(OS, ShimCounter, VD->getDeclContext(), RelativeName);
  return RelativeName;
}

bool SYCLIntegrationFooter::emit(raw_ostream &OS) {
  PrintingPolicy Policy{S.getLangOpts()};
  Policy.adjustForCPlusPlusFwdDecl();
  Policy.SuppressTypedefs = true;
  Policy.SuppressUnwrittenScope = true;

  llvm::SmallSet<const VarDecl *, 8> Visited;
  bool EmittedFirstSpecConstant = false;

  // Used to uniquely name the 'shim's as we generate the names in each
  // anonymous namespace.
  unsigned ShimCounter = 0;

  std::string DeviceGlobalsBuf;
  llvm::raw_string_ostream DeviceGlobOS(DeviceGlobalsBuf);
  for (const VarDecl *VD : GlobalVars) {
    VD = VD->getCanonicalDecl();

    // Skip if this isn't a SpecIdType or DeviceGlobal.  This can happen if it
    // was a deduced type.
    if (!Util::isSyclSpecIdType(VD->getType()) &&
        !Util::isSyclDeviceGlobalType(VD->getType()))
      continue;

    // Skip if we've already visited this.
    if (llvm::find(Visited, VD) != Visited.end())
      continue;

    // We only want to emit the #includes if we have a variable that needs
    // them, so emit this one on the first time through the loop.
    if (!EmittedFirstSpecConstant && !DeviceGlobalsEmitted)
      OS << "#include <CL/sycl/detail/defines_elementary.hpp>\n";

    Visited.insert(VD);
    std::string TopShim = EmitShims(OS, ShimCounter, Policy, VD);
    if (Util::isSyclDeviceGlobalType(VD->getType())) {
      DeviceGlobalsEmitted = true;
      DeviceGlobOS << "device_global_map::add(";
      DeviceGlobOS << "(void *)&";
      if (VD->isInAnonymousNamespace()) {
        DeviceGlobOS << TopShim;
      } else {
        DeviceGlobOS << "::";
        VD->getNameForDiagnostic(DeviceGlobOS, Policy, true);
      }
      DeviceGlobOS << ", \"";
      DeviceGlobOS << SYCLUniqueStableIdExpr::ComputeName(S.getASTContext(),
                                                          VD);
      DeviceGlobOS << "\");\n";
    } else {
      EmittedFirstSpecConstant = true;
      OS << "__SYCL_INLINE_NAMESPACE(cl) {\n";
      OS << "namespace sycl {\n";
      OS << "namespace detail {\n";
      OS << "template<>\n";
      OS << "inline const char *get_spec_constant_symbolic_ID_impl<";

      if (VD->isInAnonymousNamespace()) {
        OS << TopShim;
      } else {
        OS << "::";
        VD->getNameForDiagnostic(OS, Policy, true);
      }

      OS << ">() {\n";
      OS << "  return \"";
      OS << SYCLUniqueStableIdExpr::ComputeName(S.getASTContext(), VD);
      OS << "\";\n";
      OS << "}\n";
      OS << "} // namespace detail\n";
      OS << "} // namespace sycl\n";
      OS << "} // __SYCL_INLINE_NAMESPACE(cl)\n";
    }
  }

  if (EmittedFirstSpecConstant)
    OS << "#include <CL/sycl/detail/spec_const_integration.hpp>\n";

  if (DeviceGlobalsEmitted) {
    OS << "#include <CL/sycl/detail/device_global_map.hpp>\n";
    DeviceGlobOS.flush();
    OS << "namespace sycl::detail {\n";
    OS << "namespace {\n";
    OS << "__sycl_device_global_registration::__sycl_device_global_"
          "registration() noexcept {\n";
    OS << DeviceGlobalsBuf;
    OS << "}\n";
    OS << "} // namespace (unnamed)\n";
    OS << "} // namespace sycl::detail\n";
  }
  return true;
}

// -----------------------------------------------------------------------------
// Utility class methods
// -----------------------------------------------------------------------------
bool Util::isSyclSpecialType(const QualType Ty) {
  const CXXRecordDecl *RecTy = Ty->getAsCXXRecordDecl();
  if (!RecTy)
    return false;
  return RecTy->hasAttr<SYCLSpecialClassAttr>();
}

bool Util::isSyclHalfType(QualType Ty) {
  std::array<DeclContextDesc, 5> Scopes = {
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "cl"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "sycl"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "detail"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "half_impl"),
      Util::MakeDeclContextDesc(Decl::Kind::CXXRecord, "half")};
  return matchQualifiedTypeName(Ty, Scopes);
}

bool Util::isSyclSpecConstantType(QualType Ty) {
  std::array<DeclContextDesc, 6> Scopes = {
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "cl"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "sycl"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "ext"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "oneapi"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "experimental"),
      Util::MakeDeclContextDesc(Decl::Kind::ClassTemplateSpecialization,
                                "spec_constant")};
  return matchQualifiedTypeName(Ty, Scopes);
}

bool Util::isSyclSpecIdType(QualType Ty) {
  std::array<DeclContextDesc, 3> Scopes = {
      Util::MakeDeclContextDesc(clang::Decl::Kind::Namespace, "cl"),
      Util::MakeDeclContextDesc(clang::Decl::Kind::Namespace, "sycl"),
      Util::MakeDeclContextDesc(Decl::Kind::ClassTemplateSpecialization,
                                "specialization_id")};
  return matchQualifiedTypeName(Ty, Scopes);
}

bool Util::isSyclDeviceGlobalType(QualType Ty) {
  const CXXRecordDecl *RecTy = Ty->getAsCXXRecordDecl();
  if (!RecTy)
    return false;
  if (auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(RecTy)) {
    ClassTemplateDecl *Template = CTSD->getSpecializedTemplate();
    if (CXXRecordDecl *RD = Template->getTemplatedDecl())
      return RD->hasAttr<SYCLDeviceGlobalAttr>();
  }
  return RecTy->hasAttr<SYCLDeviceGlobalAttr>();
}

bool Util::isSyclKernelHandlerType(QualType Ty) {
  std::array<DeclContextDesc, 3> Scopes = {
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "cl"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "sycl"),
      Util::MakeDeclContextDesc(Decl::Kind::CXXRecord, "kernel_handler")};
  return matchQualifiedTypeName(Ty, Scopes);
}

bool Util::isSyclAccessorNoAliasPropertyType(QualType Ty) {
  std::array<DeclContextDesc, 7> Scopes = {
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "cl"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "sycl"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "ext"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "oneapi"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "property"),
      Util::MakeDeclContextDesc(Decl::Kind::CXXRecord, "no_alias"),
      Util::MakeDeclContextDesc(Decl::Kind::ClassTemplateSpecialization,
                                "instance")};
  return matchQualifiedTypeName(Ty, Scopes);
}

bool Util::isSyclBufferLocationType(QualType Ty) {
  std::array<DeclContextDesc, 7> Scopes = {
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "cl"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "sycl"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "ext"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "intel"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "property"),
      Util::MakeDeclContextDesc(Decl::Kind::CXXRecord, "buffer_location"),
      Util::MakeDeclContextDesc(Decl::Kind::ClassTemplateSpecialization,
                                "instance")};
  return matchQualifiedTypeName(Ty, Scopes);
}

bool Util::isSyclType(QualType Ty, StringRef Name, bool Tmpl) {
  Decl::Kind ClassDeclKind =
      Tmpl ? Decl::Kind::ClassTemplateSpecialization : Decl::Kind::CXXRecord;
  std::array<DeclContextDesc, 3> Scopes = {
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "cl"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "sycl"),
      Util::MakeDeclContextDesc(ClassDeclKind, Name)};
  return matchQualifiedTypeName(Ty, Scopes);
}

bool Util::isAccessorPropertyListType(QualType Ty) {
  std::array<DeclContextDesc, 5> Scopes = {
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "cl"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "sycl"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "ext"),
      Util::MakeDeclContextDesc(Decl::Kind::Namespace, "oneapi"),
      Util::MakeDeclContextDesc(Decl::Kind::ClassTemplateSpecialization,
                                "accessor_property_list")};
  return matchQualifiedTypeName(Ty, Scopes);
}

bool Util::matchContext(const DeclContext *Ctx,
                        ArrayRef<Util::DeclContextDesc> Scopes) {
  // The idea: check the declaration context chain starting from the item
  // itself. At each step check the context is of expected kind
  // (namespace) and name.
  StringRef Name = "";

  for (const auto &Scope : llvm::reverse(Scopes)) {
    Decl::Kind DK = Ctx->getDeclKind();
    if (DK != Scope.first)
      return false;

    switch (DK) {
    case Decl::Kind::ClassTemplateSpecialization:
      // ClassTemplateSpecializationDecl inherits from CXXRecordDecl
    case Decl::Kind::CXXRecord:
      Name = cast<CXXRecordDecl>(Ctx)->getName();
      break;
    case Decl::Kind::Namespace:
      Name = cast<NamespaceDecl>(Ctx)->getName();
      break;
    default:
      llvm_unreachable("matchContext: decl kind not supported");
    }
    if (Name != Scope.second)
      return false;
    Ctx = Ctx->getParent();
  }
  return Ctx->isTranslationUnit() ||
         (Ctx->isExternCXXContext() &&
          Ctx->getEnclosingNamespaceContext()->isTranslationUnit());
}

bool Util::matchQualifiedTypeName(QualType Ty,
                                  ArrayRef<Util::DeclContextDesc> Scopes) {
  const CXXRecordDecl *RecTy = Ty->getAsCXXRecordDecl();

  if (!RecTy)
    return false; // only classes/structs supported
  const auto *Ctx = cast<DeclContext>(RecTy);
  return Util::matchContext(Ctx, Scopes);
}
