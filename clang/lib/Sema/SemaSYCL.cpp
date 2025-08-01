//===- SemaSYCL.cpp - Semantic Analysis for SYCL constructs ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This implements Semantic Analysis for SYCL constructs.
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaSYCL.h"
#include "TreeTransform.h"
#include "clang/AST/AST.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/QualTypeNames.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/SYCLKernelInfo.h"
#include "clang/AST/StmtSYCL.h"
#include "clang/AST/TemplateArgumentVisitor.h"
#include "clang/AST/TypeOrdering.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/Analysis/CallGraph.h"
#include "clang/Basic/Attributes.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "clang/Sema/Attr.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
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
static constexpr llvm::StringLiteral GlibcxxAssertFail =
    "__glibcxx_assert_fail";
constexpr unsigned MaxKernelArgsSize = 2048;

bool SemaSYCL::isSyclType(QualType Ty, SYCLTypeAttr::SYCLType TypeName) {
  const auto *RD = Ty->getAsCXXRecordDecl();
  if (!RD)
    return false;

  if (const auto *Attr = RD->getAttr<SYCLTypeAttr>())
    return Attr->getType() == TypeName;

  if (const auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(RD))
    if (CXXRecordDecl *TemplateDecl =
            CTSD->getSpecializedTemplate()->getTemplatedDecl())
      if (const auto *Attr = TemplateDecl->getAttr<SYCLTypeAttr>())
        return Attr->getType() == TypeName;

  return false;
}

static bool isSyclAccessorType(QualType Ty) {
  return SemaSYCL::isSyclType(Ty, SYCLTypeAttr::accessor) ||
         SemaSYCL::isSyclType(Ty, SYCLTypeAttr::local_accessor) ||
         SemaSYCL::isSyclType(Ty, SYCLTypeAttr::dynamic_local_accessor);
}

// FIXME: Accessor property lists should be modified to use compile-time
// properties. Once implemented, this function (and possibly all/most code
// in SemaSYCL.cpp handling no_alias and buffer_location property) can be
// removed.
static bool isAccessorPropertyType(QualType Ty,
                                   SYCLTypeAttr::SYCLType TypeName) {
  if (const auto *RD = Ty->getAsCXXRecordDecl())
    if (const auto *Parent = dyn_cast<CXXRecordDecl>(RD->getParent()))
      if (const auto *Attr = Parent->getAttr<SYCLTypeAttr>())
        return Attr->getType() == TypeName;

  return false;
}

static bool isSyclSpecialType(QualType Ty, SemaSYCL &S) {
  return S.isTypeDecoratedWithDeclAttribute<SYCLSpecialClassAttr>(Ty);
}

ExprResult SemaSYCL::ActOnSYCLBuiltinNumFieldsExpr(ParsedType PT) {
  TypeSourceInfo *TInfo = nullptr;
  QualType QT = Sema::GetTypeFromParser(PT, &TInfo);
  assert(TInfo && "couldn't get type info from a type from the parser?");
  SourceLocation TypeLoc = TInfo->getTypeLoc().getBeginLoc();

  return BuildSYCLBuiltinNumFieldsExpr(TypeLoc, QT);
}

ExprResult SemaSYCL::BuildSYCLBuiltinNumFieldsExpr(SourceLocation Loc,
                                                   QualType SourceTy) {
  if (!SourceTy->isDependentType()) {
    if (SemaRef.RequireCompleteType(
            Loc, SourceTy, diag::err_sycl_type_trait_requires_complete_type,
            /*__builtin_num_fields*/ 0))
      return ExprError();

    if (!SourceTy->isRecordType()) {
      Diag(Loc, diag::err_sycl_type_trait_requires_record_type)
          << /*__builtin_num_fields*/ 0;
      return ExprError();
    }
  }
  return new (getASTContext())
      SYCLBuiltinNumFieldsExpr(Loc, SourceTy, getASTContext().getSizeType());
}

ExprResult SemaSYCL::ActOnSYCLBuiltinFieldTypeExpr(ParsedType PT, Expr *Idx) {
  TypeSourceInfo *TInfo = nullptr;
  QualType QT = Sema::GetTypeFromParser(PT, &TInfo);
  assert(TInfo && "couldn't get type info from a type from the parser?");
  SourceLocation TypeLoc = TInfo->getTypeLoc().getBeginLoc();

  return BuildSYCLBuiltinFieldTypeExpr(TypeLoc, QT, Idx);
}

ExprResult SemaSYCL::BuildSYCLBuiltinFieldTypeExpr(SourceLocation Loc,
                                                   QualType SourceTy,
                                                   Expr *Idx) {
  // If the expression appears in an evaluated context, we want to give an
  // error so that users don't attempt to use the value of this expression.
  if (!SemaRef.isUnevaluatedContext()) {
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
    if (SemaRef.RequireCompleteType(
            Loc, SourceTy, diag::err_sycl_type_trait_requires_complete_type,
            /*__builtin_field_type*/ 1))
      return ExprError();

    if (!SourceTy->isRecordType()) {
      Diag(Loc, diag::err_sycl_type_trait_requires_record_type)
          << /*__builtin_field_type*/ 1;
      return ExprError();
    }

    if (!Idx->isValueDependent()) {
      std::optional<llvm::APSInt> IdxVal =
          Idx->getIntegerConstantExpr(getASTContext());
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
  return new (getASTContext())
      SYCLBuiltinFieldTypeExpr(Loc, SourceTy, Idx, FieldTy, ValueKind);
}

ExprResult SemaSYCL::ActOnSYCLBuiltinNumBasesExpr(ParsedType PT) {
  TypeSourceInfo *TInfo = nullptr;
  QualType QT = Sema::GetTypeFromParser(PT, &TInfo);
  assert(TInfo && "couldn't get type info from a type from the parser?");
  SourceLocation TypeLoc = TInfo->getTypeLoc().getBeginLoc();

  return BuildSYCLBuiltinNumBasesExpr(TypeLoc, QT);
}

ExprResult SemaSYCL::BuildSYCLBuiltinNumBasesExpr(SourceLocation Loc,
                                                  QualType SourceTy) {
  if (!SourceTy->isDependentType()) {
    if (SemaRef.RequireCompleteType(
            Loc, SourceTy, diag::err_sycl_type_trait_requires_complete_type,
            /*__builtin_num_bases*/ 2))
      return ExprError();

    if (!SourceTy->isRecordType()) {
      Diag(Loc, diag::err_sycl_type_trait_requires_record_type)
          << /*__builtin_num_bases*/ 2;
      return ExprError();
    }
  }
  return new (getASTContext())
      SYCLBuiltinNumBasesExpr(Loc, SourceTy, getASTContext().getSizeType());
}

ExprResult SemaSYCL::ActOnSYCLBuiltinBaseTypeExpr(ParsedType PT, Expr *Idx) {
  TypeSourceInfo *TInfo = nullptr;
  QualType QT = SemaRef.GetTypeFromParser(PT, &TInfo);
  assert(TInfo && "couldn't get type info from a type from the parser?");
  SourceLocation TypeLoc = TInfo->getTypeLoc().getBeginLoc();

  return BuildSYCLBuiltinBaseTypeExpr(TypeLoc, QT, Idx);
}

ExprResult SemaSYCL::BuildSYCLBuiltinBaseTypeExpr(SourceLocation Loc,
                                                  QualType SourceTy,
                                                  Expr *Idx) {
  // If the expression appears in an evaluated context, we want to give an
  // error so that users don't attempt to use the value of this expression.
  if (!SemaRef.isUnevaluatedContext()) {
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
    if (SemaRef.RequireCompleteType(
            Loc, SourceTy, diag::err_sycl_type_trait_requires_complete_type,
            /*__builtin_base_type*/ 3))
      return ExprError();

    if (!SourceTy->isRecordType()) {
      Diag(Loc, diag::err_sycl_type_trait_requires_record_type)
          << /*__builtin_base_type*/ 3;
      return ExprError();
    }

    if (!Idx->isValueDependent()) {
      std::optional<llvm::APSInt> IdxVal =
          Idx->getIntegerConstantExpr(getASTContext());
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
  return new (getASTContext())
      SYCLBuiltinBaseTypeExpr(Loc, SourceTy, Idx, BaseTy);
}

/// Returns true if the target requires a new type.
/// This happens if a pointer to generic cannot be passed
static bool targetRequiresNewType(ASTContext &Context) {
  llvm::Triple T = Context.getTargetInfo().getTriple();
  return !T.isNVPTX();
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

bool SemaSYCL::isDeclAllowedInSYCLDeviceCode(const Decl *D) {
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(D)) {
    const IdentifierInfo *II = FD->getIdentifier();

    // Allow __builtin_assume_aligned and __builtin_printf to be called from
    // within device code.
    if (FD->getBuiltinID() &&
        (FD->getBuiltinID() == Builtin::BI__builtin_assume_aligned ||
         FD->getBuiltinID() == Builtin::BI__builtin_printf))
      return true;

    // Allow to use `::printf` only for CUDA.
    if (getLangOpts().SYCLCUDACompat) {
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

SemaSYCL::SemaSYCL(Sema &S)
    : SemaBase(S), SyclIntHeader(nullptr), SyclIntFooter(nullptr) {}

static bool isZeroSizedArray(SemaSYCL &S, QualType Ty) {
  if (const auto *CAT = S.getASTContext().getAsConstantArrayType(Ty))
    return CAT->isZeroSize();
  return false;
}

static void checkSYCLType(SemaSYCL &S, QualType Ty, SourceRange Loc,
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
  ASTContext &Context = S.getASTContext();

  //--- check types ---
  if (Ty->isDependentType())
    return;

  // zero length arrays
  if (isZeroSizedArray(S, Ty)) {
    S.DiagIfDeviceCode(Loc.getBegin(), diag::err_typecheck_zero_array_size)
        << 1;
    Emitting = true;
  }

  // variable length arrays
  if (Ty->isVariableArrayType()) {
    S.DiagIfDeviceCode(Loc.getBegin(), diag::err_vla_unsupported) << 0;
    Emitting = true;
  }

  // Sub-reference array or pointer, then proceed with that type.
  while (Ty->isAnyPointerType() || Ty->isArrayType())
    Ty = QualType{Ty->getPointeeOrArrayElementType(), 0};

  if (((Ty->isFloat128Type() ||
        (Ty->isRealFloatingType() && Context.getTypeSize(Ty) == 128)) &&
       !Context.getTargetInfo().hasFloat128Type()) ||
      (Ty->isIntegerType() && Context.getTypeSize(Ty) == 128 &&
       !Context.getTargetInfo().hasInt128Type()) ||
      (Ty->isBFloat16Type() && !Context.getTargetInfo().hasBFloat16Type()) ||
      // FIXME: this should have a TI check, but support isn't properly reported
      // ...
      (Ty->isSpecificBuiltinType(BuiltinType::LongDouble))) {
    S.DiagIfDeviceCode(Loc.getBegin(), diag::err_type_unsupported)
        << Ty.getUnqualifiedType().getCanonicalType();
    Emitting = true;
  }

  if (Emitting && UsedAtLoc.isValid())
    S.DiagIfDeviceCode(UsedAtLoc.getBegin(), diag::note_used_here);

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

void SemaSYCL::checkSYCLDeviceVarDecl(VarDecl *Var) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL compilation");
  QualType Ty = Var->getType();
  SourceRange Loc = Var->getLocation();
  llvm::DenseSet<QualType> Visited;

  checkSYCLType(*this, Ty, Loc, Visited);
}

enum NotForwardDeclarableReason {
  UnscopedEnum,
  StdNamespace,
  UnnamedTag,
  NotAtNamespaceScope,
  None
};

// This is a helper function which is used to check if a class declaration is:
//   * declared within namespace 'std' (at any level)
//     e.g., namespace std { namespace literals { class Whatever; } }
//     h.single_task<std::literals::Whatever>([]() {});
//   * declared within a function
//     e.g., void foo() { struct S { int i; };
//     h.single_task<S>([]() {}); }
//   * declared within another tag
//     e.g., struct S { struct T { int i } t; };
//     h.single_task<S::T>([]() {});
//  User for kernel name types and class/struct types used in free function
//  kernel arguments.
static NotForwardDeclarableReason
isForwardDeclarable(const NamedDecl *DeclToCheck, SemaSYCL &S,
                    bool DiagForFreeFunction = false) {
  if (const auto *ED = dyn_cast<EnumDecl>(DeclToCheck);
      ED && !ED->isScoped() && !ED->isFixed())
    return NotForwardDeclarableReason::UnscopedEnum;

  const DeclContext *DeclCtx = DeclToCheck->getDeclContext();
  if (DeclCtx) {
    while (!DeclCtx->isTranslationUnit() &&
           (isa<NamespaceDecl>(DeclCtx) || isa<LinkageSpecDecl>(DeclCtx))) {
      const auto *NSDecl = dyn_cast<NamespaceDecl>(DeclCtx);
      // We don't report free function kernel parameter case because the
      // restriction for the type used there to be forward declarable comes from
      // the need to forward declare it in the integration header. We're safe
      // to do so because the integration header is an implemention detail and
      // is generated by the compiler.
      // We do diagnose case with kernel name type since the spec requires us to
      // do so.
      if (!DiagForFreeFunction && NSDecl && NSDecl->isStdNamespace())
        return NotForwardDeclarableReason::StdNamespace;
      DeclCtx = DeclCtx->getParent();
    }
  }

  // Check if the we've met a Tag declaration local to a non-namespace scope
  // (i.e. Inside a function or within another Tag etc).
  if (const auto *Tag = dyn_cast<TagDecl>(DeclToCheck)) {
    if (Tag->getIdentifier() == nullptr)
      return NotForwardDeclarableReason::UnnamedTag;
    if (!DeclCtx->isTranslationUnit()) {
      // Diagnose used types without complete definition i.e.
      //   int main() {
      //     class KernelName1;
      //     parallel_for<class KernelName1>(..);
      //   }
      // For kernel name type This case can only be diagnosed during host
      // compilation because the integration header is required to distinguish
      // between the invalid code (above) and the following valid code:
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
          S.getLangOpts().SYCLEnableIntHeaderDiags || DiagForFreeFunction)
        return NotForwardDeclarableReason::NotAtNamespaceScope;
    }
  }

  return NotForwardDeclarableReason::None;
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

  bool IsAllowed = false;
  // libstdc++-11 introduced an undefined function "void __failed_assertion()"
  // which may lead to SemaSYCL check failure. However, this undefined function
  // is used to trigger some compilation error when the check fails at compile
  // time and will be ignored when the check succeeds. We allow calls to this
  // function to support some important std functions in SYCL device.
  IsAllowed = (Callee->getName() == LibstdcxxFailedAssertion) &&
              Callee->getNumParams() == 0 &&
              Callee->getReturnType()->isVoidType() &&
              SrcMgr.isInSystemHeader(Callee->getLocation());

  if (IsAllowed)
    return true;

  // GCC-15 introduced "std::__glibcxx_assert_fail" declared c++config.h and
  // extensively used in STL to do runtime check in debug mode. The behavior
  // is similar to "assert", we have supported it in libdevice in the same way
  // as "assert". However, Sema check will report "undefined function without
  // SYCL_EXTERNAL attribute" error in some cases. We have to allow it just as
  // what we did to "__failed_assertion". The prototype is following:
  // void __glibcxx_assert_fail(const char *, int, const char *, const char*);
  IsAllowed = (Callee->getName() == GlibcxxAssertFail) &&
              Callee->getNumParams() == 4 &&
              Callee->getReturnType()->isVoidType() &&
              SrcMgr.isInSystemHeader(Callee->getLocation());

  return IsAllowed;
}

// Helper function to report conflicting function attributes.
// F - the function, A1 - function attribute, A2 - the attribute it conflicts
// with.
static void reportConflictingAttrs(SemaSYCL &S, FunctionDecl *F, const Attr *A1,
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
static void collectSYCLAttributes(FunctionDecl *FD,
                                  llvm::SmallVectorImpl<Attr *> &Attrs,
                                  bool DirectlyCalled) {
  if (!FD->hasAttrs())
    return;

  // In SYCL 2020 mode, the attributes aren't propagated from the function they
  // are applied on to the kernel which calls the function.
  if (DirectlyCalled) {
    llvm::copy_if(FD->getAttrs(), std::back_inserter(Attrs), [](Attr *A) {
      // FIXME: Make this list self-adapt as new SYCL attributes are added.
      return isa<IntelReqdSubGroupSizeAttr, IntelNamedSubGroupSizeAttr,
                 SYCLReqdWorkGroupSizeAttr, SYCLWorkGroupSizeHintAttr,
                 SYCLIntelKernelArgsRestrictAttr, SYCLIntelNumSimdWorkItemsAttr,
                 SYCLIntelSchedulerTargetFmaxMhzAttr,
                 SYCLIntelMaxWorkGroupSizeAttr, SYCLIntelMaxGlobalWorkDimAttr,
                 SYCLIntelMinWorkGroupsPerComputeUnitAttr,
                 SYCLIntelMaxWorkGroupsPerMultiprocessorAttr,
                 SYCLIntelNoGlobalWorkOffsetAttr, SYCLSimdAttr,
                 SYCLIntelLoopFuseAttr, SYCLIntelMaxConcurrencyAttr,
                 SYCLIntelDisableLoopPipeliningAttr,
                 SYCLIntelInitiationIntervalAttr,
                 SYCLIntelUseStallEnableClustersAttr, SYCLDeviceHasAttr,
                 SYCLAddIRAttributesFunctionAttr>(A);
    });
  }
}

class DiagDeviceFunction : public RecursiveASTVisitor<DiagDeviceFunction> {
  SemaSYCL &SemaSYCLRef;
  const llvm::SmallPtrSetImpl<const FunctionDecl *> &RecursiveFuncs;

public:
  DiagDeviceFunction(
      SemaSYCL &S,
      const llvm::SmallPtrSetImpl<const FunctionDecl *> &RecursiveFuncs)
      : RecursiveASTVisitor(), SemaSYCLRef(S), RecursiveFuncs(RecursiveFuncs) {}

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
        SemaSYCLRef.Diag(e->getExprLoc(), diag::err_sycl_restrict)
            << SemaSYCL::KernelCallRecursiveFunction;
        SemaSYCLRef.Diag(Callee->getSourceRange().getBegin(),
                         diag::note_sycl_recursive_function_declared_here)
            << SemaSYCL::KernelCallRecursiveFunction;
      }

      // Specifically check if the math library function corresponding to this
      // builtin is supported for SYCL
      unsigned BuiltinID = Callee->getBuiltinID();
      if (BuiltinID && !IsSyclMathFunc(BuiltinID)) {
        std::string Name =
            SemaSYCLRef.getASTContext().BuiltinInfo.getName(BuiltinID);
        SemaSYCLRef.Diag(e->getExprLoc(), diag::err_builtin_target_unsupported)
            << Name << "SYCL device";
      }
    } else if (!SemaSYCLRef.getLangOpts().SYCLAllowFuncPtr &&
               !e->isTypeDependent() &&
               !isa<CXXPseudoDestructorExpr>(e->getCallee())) {
      bool MaybeConstantExpr = false;
      Expr *NonDirectCallee = e->getCallee();
      if (!NonDirectCallee->isValueDependent())
        MaybeConstantExpr =
            NonDirectCallee->isCXX11ConstantExpr(SemaSYCLRef.getASTContext());
      if (!MaybeConstantExpr)
        SemaSYCLRef.Diag(e->getExprLoc(), diag::err_sycl_restrict)
            << SemaSYCL::KernelCallFunctionPointer;
    }
    return true;
  }

  bool VisitCXXTypeidExpr(CXXTypeidExpr *E) {
    SemaSYCLRef.Diag(E->getExprLoc(), diag::err_sycl_restrict)
        << SemaSYCL::KernelRTTI;
    return true;
  }

  bool VisitCXXDynamicCastExpr(const CXXDynamicCastExpr *E) {
    SemaSYCLRef.Diag(E->getExprLoc(), diag::err_sycl_restrict)
        << SemaSYCL::KernelRTTI;
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
  bool TraverseTemplateArgumentLoc(const TemplateArgumentLoc &) { return true; }

  // Skip checking the static assert, both components are required to be
  // constant expressions.
  bool TraverseStaticAssertDecl(StaticAssertDecl *) { return true; }

  // Make sure we skip the condition of the case, since that is a constant
  // expression.
  bool TraverseCaseStmt(CaseStmt *S) {
    return TraverseStmt(S->getSubStmt());
  }

  // Skip checking the size expr, since a constant array type loc's size expr is
  // a constant expression.
  bool TraverseConstantArrayTypeLoc(const ConstantArrayTypeLoc &) {
    return true;
  }

  bool TraverseIfStmt(IfStmt *S) {
    if (std::optional<Stmt *> ActiveStmt =
            S->getNondiscardedCase(SemaSYCLRef.getASTContext())) {
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
  SemaSYCL &SemaSYCLRef;
  // The list of functions used on the device, kept so we can diagnose on them
  // later.
  llvm::SmallPtrSet<FunctionDecl *, 16> DeviceFunctions;
  llvm::SmallPtrSet<const FunctionDecl *, 16> RecursiveFunctions;

  void CollectSyclExternalFuncs() {
    for (CallGraphNode::CallRecord Record : CG.getRoot()->callees())
      if (auto *FD = dyn_cast<FunctionDecl>(Record.Callee->getDecl()))
        if (FD->hasBody() && FD->hasAttr<SYCLDeviceAttr>())
          SemaSYCLRef.addSyclDeviceDecl(FD);
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
  DeviceFunctionTracker(SemaSYCL &S) : SemaSYCLRef(S) {
    CG.setSkipConstantExpressions(S.getASTContext());
    CG.addToCallGraph(S.getASTContext().getTranslationUnitDecl());
    CollectSyclExternalFuncs();
  }

  ~DeviceFunctionTracker() {
    DiagDeviceFunction Diagnoser{SemaSYCLRef, RecursiveFunctions};
    for (const FunctionDecl *FD : DeviceFunctions)
      if (const FunctionDecl *Def = FD->getDefinition())
        Diagnoser.CheckBody(Def->getBody());
  }
};

/// This function checks whether given DeclContext contains a topmost
/// namespace with name "sycl".
static bool isDeclaredInSYCLNamespace(const Decl *D) {
  const DeclContext *DC = D->getDeclContext()->getEnclosingNamespaceContext();
  const auto *ND = dyn_cast<NamespaceDecl>(DC);
  // If this is not a namespace, then we are done.
  if (!ND)
    return false;

  // While it is a namespace, find its parent scope.
  while (const DeclContext *Parent = ND->getParent()) {
    if (!isa<NamespaceDecl>(Parent))
      break;
    ND = cast<NamespaceDecl>(Parent);
  }

  return ND->getName() == "sycl";
}

static bool isSYCLPrivateMemoryVar(VarDecl *VD) {
  return SemaSYCL::isSyclType(VD->getType(), SYCLTypeAttr::private_memory);
}

static void addScopeAttrToLocalVars(FunctionDecl &F) {
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
    // cases later in finalizeSYCLDelayedAnalysis().
    if (!CurrentDecl->isDefined() && !CurrentDecl->hasAttr<DeviceKernelAttr>() &&
        !CurrentDecl->hasAttr<SYCLDeviceAttr>())
      Parent.SemaSYCLRef.addFDToReachableFromSyclDevice(CurrentDecl,
                                                        CallStack.back());

    // If this is a parallel_for_work_item that is declared in the
    // sycl namespace, mark it with the WorkItem scope attribute.
    // Note: Here, we assume that this is called from within a
    // parallel_for_work_group; it is undefined to call it otherwise.
    // We deliberately do not diagnose a violation.
    // The following changes have also been added:
    // 1. The function inside which the parallel_for_work_item exists is
    //    marked with WorkGroup scope attribute, if not present already.
    // 2. The local variables inside the function are marked with appropriate
    //    scope.
    if (CurrentDecl->getIdentifier() &&
        CurrentDecl->getIdentifier()->getName() == "parallel_for_work_item" &&
        isDeclaredInSYCLNamespace(CurrentDecl) &&
        !CurrentDecl->hasAttr<SYCLScopeAttr>()) {
      CurrentDecl->addAttr(SYCLScopeAttr::CreateImplicit(
          Parent.SemaSYCLRef.getASTContext(), SYCLScopeAttr::Level::WorkItem));
      FunctionDecl *Caller = CallStack.back();
      if (!Caller->hasAttr<SYCLScopeAttr>()) {
        Caller->addAttr(
            SYCLScopeAttr::CreateImplicit(Parent.SemaSYCLRef.getASTContext(),
                                          SYCLScopeAttr::Level::WorkGroup));
        addScopeAttrToLocalVars(*Caller);
      }
    }

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
      collectSYCLAttributes(CurrentDecl, CollectedAttributes, DirectlyCalled);
    }

    // Calculate the kernel body.  Note the 'isSYCLKernelBodyFunction' only
    // tests that it is operator(), so hopefully this doesn't get us too many
    // false-positives.
    if (isSYCLKernelBodyFunction(CurrentDecl)) {
      // This is a direct callee of the kernel.
      if (CallStack.size() == 1 &&
          CallStack.back()->hasAttr<DeviceKernelAttr>()) {
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
      if (KernelBody)
        Parent.SemaSYCLRef.addSYCLKernelFunction(KernelBody);
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

    // Always inline the KernelBody in the kernel entry point. For ESIMD
    // inlining is handled later down the pipeline.
    if (KernelBody &&
        Parent.SemaSYCLRef.getLangOpts().SYCLForceInlineKernelLambda &&
        !KernelBody->hasAttr<NoInlineAttr>() &&
        !KernelBody->hasAttr<AlwaysInlineAttr>() &&
        !KernelBody->hasAttr<SYCLSimdAttr>()) {
      KernelBody->addAttr(AlwaysInlineAttr::CreateImplicit(
          KernelBody->getASTContext(), {}, AlwaysInlineAttr::Keyword_forceinline));
    }
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
static ParamDesc makeParamDesc(const ParmVarDecl *Src, QualType Ty) {
  ASTContext &Ctx = Src->getASTContext();
  std::string Name = (Twine("__arg_") + Src->getName()).str();
  return std::make_tuple(Ty, &Ctx.Idents.get(Name),
                         Ctx.getTrivialTypeSourceInfo(Ty));
}

static ParamDesc makeParamDesc(ASTContext &Ctx, StringRef Name, QualType Ty) {
  return std::make_tuple(Ty, &Ctx.Idents.get(Name),
                         Ctx.getTrivialTypeSourceInfo(Ty));
}

static void unsupportedFreeFunctionParamType() {
  llvm::report_fatal_error("Unsupported free kernel parameter type!");
}

class MarkWIScopeFnVisitor : public RecursiveASTVisitor<MarkWIScopeFnVisitor> {
public:
  MarkWIScopeFnVisitor(ASTContext &Ctx) : Ctx(Ctx) {}

  bool VisitCXXMemberCallExpr(CXXMemberCallExpr *Call) {
    FunctionDecl *Callee = Call->getDirectCallee();
    if (!Callee)
      // not a direct call - continue search
      return true;
    QualType Ty = Ctx.getRecordType(Call->getRecordDecl());
    if (!SemaSYCL::isSyclType(Ty, SYCLTypeAttr::group))
      // not a member of sycl::group - continue search
      return true;
    auto Name = Callee->getName();
    if (Name != "wait_for" ||
        Callee->hasAttr<SYCLScopeAttr>())
      return true;
    // it is a call to sycl::group::wait_for - mark the callee
    Callee->addAttr(
        SYCLScopeAttr::CreateImplicit(Ctx, SYCLScopeAttr::Level::WorkItem));
    // continue search as there can be other wait_for calls
    return true;
  }

private:
  ASTContext &Ctx;
};

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

/// \return the target of given SYCL accessor type
static target getAccessTarget(QualType FieldTy,
                              const ClassTemplateSpecializationDecl *AccTy) {
  if (SemaSYCL::isSyclType(FieldTy, SYCLTypeAttr::local_accessor) ||
      SemaSYCL::isSyclType(FieldTy, SYCLTypeAttr::dynamic_local_accessor))
    return local;

  return static_cast<target>(
      AccTy->getTemplateArgs()[3].getAsIntegral().getExtValue());
}

bool SemaSYCL::isFreeFunction(const FunctionDecl *FD) {
  SourceLocation Loc = FD->getLocation();
  bool NextDeclaredWithAttr = false;
  for (FunctionDecl *Redecl : FD->redecls()) {
    bool IsFreeFunctionAttr = false;
    for (auto *IRAttr :
         Redecl->specific_attrs<SYCLAddIRAttributesFunctionAttr>()) {
      SmallVector<std::pair<std::string, std::string>, 4> NameValuePairs =
          IRAttr->getAttributeNameValuePairs(getASTContext());
      const auto it = std::find_if(
          NameValuePairs.begin(), NameValuePairs.end(),
          [](const auto &NameValuePair) {
            return NameValuePair.first == "sycl-nd-range-kernel" ||
                   NameValuePair.first == "sycl-single-task-kernel";
          });
      IsFreeFunctionAttr = it != NameValuePairs.end();
      if (IsFreeFunctionAttr)
        break;
    }
    if (Redecl->isFirstDecl()) {
      if (IsFreeFunctionAttr)
        return true;
      if (NextDeclaredWithAttr) {
        Diag(Loc, diag::err_free_function_first_occurrence_missing_attr);
        Diag(Redecl->getLocation(), diag::note_previous_declaration);
        return false;
      }
    } else {
      Loc = Redecl->getLocation();
      NextDeclaredWithAttr = IsFreeFunctionAttr;
    }
  }
  return false;
}

static int getFreeFunctionRangeDim(SemaSYCL &SemaSYCLRef,
                                   const FunctionDecl *FD) {
  for (auto *IRAttr : FD->specific_attrs<SYCLAddIRAttributesFunctionAttr>()) {
    SmallVector<std::pair<std::string, std::string>, 4> NameValuePairs =
        IRAttr->getAttributeNameValuePairs(SemaSYCLRef.getASTContext());
    for (const auto &NameValuePair : NameValuePairs) {
      if (NameValuePair.first == "sycl-nd-range-kernel")
        return std::stoi(NameValuePair.second);
      if (NameValuePair.first == "sycl-single-task-kernel")
        return 0;
    }
  }
  return false;
}

// Creates a name for the free function kernel function.
// Consider a free function named "MyFunction". The normal device function will
// be given its mangled name, say "_Z10MyFunctionIiEvPT_S0_". The corresponding
// kernel function for this free function will be named
// "_Z24__sycl_kernel_MyFunctionIiEvPT_S0_". This is the mangled name of a
// fictitious function that has the same template and function parameters as the
// original free function but with identifier prefixed with __sycl_kernel_.
// We generate this name by starting with the mangled name of the free function
// and adjusting it textually to simulate the __sycl_kernel_ prefix.
// Because free functions are allowed only at file scope and cannot be within
// namespaces the mangled name has the format _Z<length><identifier>... where
// length is the identifier's length. The text manipulation inserts the prefix
// __sycl_kernel_ and adjusts the length, leaving the rest of the name as-is.
static std::pair<std::string, std::string>
constructFreeFunctionKernelName(const FunctionDecl *FreeFunc,
                                MangleContext &MC) {
  SmallString<256> Result;
  llvm::raw_svector_ostream Out(Result);
  std::string NewName;
  std::string StableName;

  // Handle extern "C"
  if (FreeFunc->getLanguageLinkage() == CLanguageLinkage) {
    const IdentifierInfo *II = FreeFunc->getIdentifier();
    NewName = "__sycl_kernel_" + II->getName().str();
  } else {
    MC.mangleName(FreeFunc, Out);
    std::string MangledName(Out.str());
    size_t StartNums = MangledName.find_first_of("0123456789");
    size_t EndNums = MangledName.find_first_not_of("0123456789", StartNums);
    size_t NameLength =
        std::stoi(MangledName.substr(StartNums, EndNums - StartNums));
    size_t NewNameLength = 14 /*length of __sycl_kernel_*/ + NameLength;
    NewName = MangledName.substr(0, StartNums) + std::to_string(NewNameLength) +
              "__sycl_kernel_" + MangledName.substr(EndNums);
  }
  StableName = NewName;
  return {NewName, StableName};
}

// The first template argument to the kernel caller function is used to identify
// the kernel itself.
static QualType calculateKernelNameType(const FunctionDecl *KernelCallerFunc) {
  const TemplateArgumentList *TAL =
      KernelCallerFunc->getTemplateSpecializationArgs();
  assert(TAL && "No template argument info");
  return TAL->get(0).getAsType().getCanonicalType();
}

// Gets a name for the OpenCL kernel function, calculated from the first
// template argument of the kernel caller function.
static std::pair<std::string, std::string>
constructKernelName(SemaSYCL &S, const FunctionDecl *KernelCallerFunc,
                    MangleContext &MC) {
  QualType KernelNameType = calculateKernelNameType(KernelCallerFunc);

  SmallString<256> Result;
  llvm::raw_svector_ostream Out(Result);

  MC.mangleCanonicalTypeName(KernelNameType, Out);
  std::string MangledName(Out.str());

  std::string StableName =
      SYCLUniqueStableNameExpr::ComputeName(S.getASTContext(), KernelNameType);

  // For NativeCPU the kernel name is set to the stable GNU-mangled name
  // because the default mangling may be different, for example on Windows.
  // This is needed for compiling kernels for multiple SYCL targets to ensure
  // the same kernel name can be used for kernel lookup in different target
  // binaries. This assumes that all SYCL targets use the same mangling
  // produced for the stable name.
  // Todo: Check if this assumption is valid, and if it would be better
  // instead to always compile the NativeCPU device code in GNU mode which
  // may cause issues when compiling headers with non-standard extensions
  // written for compilers with different C++ ABIs (like MS VS).
  if (S.getASTContext().getTargetInfo().getTriple().isNativeCPU()) {
    MangledName = StableName;
  }

  return {MangledName, StableName};
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
    return SemaSYCL::isSyclType(PVD->getType(), SYCLTypeAttr::kernel_handler);
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
  SemaSYCL &SemaSYCLRef;

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

  // This enables handler execution only when previous Handlers succeed.
  template <typename... Tn>
  bool handleParam(ParmVarDecl *PD, QualType PDTy, Tn &&...tn) {
    bool result = true;
    (void)std::initializer_list<int>{(result = result && tn(PD, PDTy), 0)...};
    return result;
  }

  // This definition using std::bind is necessary because of a gcc 7.x bug.
#define KP_FOR_EACH(FUNC, Item, Qt)                                            \
  handleParam(                                                                 \
      Item, Qt,                                                                \
      std::bind(static_cast<bool (std::decay_t<decltype(Handlers)>::*)(        \
                    bind_param_t<decltype(Item)>, QualType)>(                  \
                    &std::decay_t<decltype(Handlers)>::FUNC),                  \
                std::ref(Handlers), _1, _2)...)

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
                         const CXXRecordDecl *, QualType RecordTy,
                         HandlerTys &...Handlers) {
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
      if (isSyclSpecialType(BaseTy, SemaSYCLRef))
        (void)std::initializer_list<int>{
            (Handlers.handleSyclSpecialType(Owner, Base, BaseTy), 0)...};
      else
        // For all other bases, visit the record
        visitRecord(Owner, Base, BaseTy->getAsCXXRecordDecl(), BaseTy,
                    Handlers...);
    }
  }

  template <typename... HandlerTys>
  void VisitRecordHelper(const CXXRecordDecl *Owner, RecordDecl::field_range,
                         HandlerTys &...Handlers) {
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
  void visitSimpleArray(const CXXRecordDecl *, FieldDecl *Field,
                        QualType ArrayTy, HandlerTys &...Handlers) {
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
        SemaSYCLRef.getASTContext().getAsConstantArrayType(ArrayTy);
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
  void visitField(const CXXRecordDecl *Owner, FieldDecl *Field,
                  QualType FieldTy, HandlerTys &... Handlers) {
    if (isSyclSpecialType(FieldTy, SemaSYCLRef))
      KF_FOR_EACH(handleSyclSpecialType, Field, FieldTy);
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

  template <typename... HandlerTys>
  void visitParam(ParmVarDecl *Param, QualType ParamTy,
                  HandlerTys &...Handlers) {
    if (isSyclSpecialType(ParamTy, SemaSYCLRef))
      KP_FOR_EACH(handleSyclSpecialType, Param, ParamTy);
    else if (ParamTy->isStructureOrClassType()) {
      if (KP_FOR_EACH(handleStructType, Param, ParamTy)) {
        CXXRecordDecl *RD = ParamTy->getAsCXXRecordDecl();
        visitRecord(nullptr, Param, RD, ParamTy, Handlers...);
      }
    } else if (ParamTy->isUnionType())
      KP_FOR_EACH(handleOtherType, Param, ParamTy);
    else if (ParamTy->isReferenceType())
      KP_FOR_EACH(handleOtherType, Param, ParamTy);
    else if (ParamTy->isPointerType())
      KP_FOR_EACH(handlePointerType, Param, ParamTy);
    else if (ParamTy->isArrayType())
      KP_FOR_EACH(handleOtherType, Param, ParamTy);
    else if (ParamTy->isScalarType())
      KP_FOR_EACH(handleScalarType, Param, ParamTy);
    else
      KP_FOR_EACH(handleOtherType, Param, ParamTy);
  }

public:
  KernelObjVisitor(SemaSYCL &S) : SemaSYCLRef(S) {}

  static bool useTopLevelKernelObj(const CXXRecordDecl *KernelObj) {
    // If the kernel is empty, "decompose" it so we don't generate arguments.
    if (KernelObj->isEmpty())
      return false;
    // FIXME: Workaround to not change large number of tests
    // this is covered by the test below.
    if (targetRequiresNewType(KernelObj->getASTContext()))
      return false;
    if (KernelObj->hasAttr<SYCLRequiresDecompositionAttr>() ||
        KernelObj->hasAttr<SYCLGenerateNewTypeAttr>())
      return false;
    return true;
  }

  template <typename... HandlerTys>
  void visitTopLevelRecord(const CXXRecordDecl *Owner, QualType RecordTy,
                           HandlerTys &...Handlers) {
    (void)std::initializer_list<int>{
        (Handlers.handleTopLevelStruct(Owner, RecordTy), 0)...};
  }

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

  template <typename... HandlerTys>
  void visitArray(const CXXRecordDecl *Owner, FieldDecl *Field,
                  QualType ArrayTy, HandlerTys &...Handlers);

  // A visitor for Kernel object to functions as defined in
  // SyclKernelFieldHandler by iterating over fields and bases
  // if they require decomposition or new type.
  template <typename... HandlerTys>
  void VisitKernelRecord(const CXXRecordDecl *KernelObj,
                         QualType KernelFunctorTy, HandlerTys &...Handlers) {
    if (!useTopLevelKernelObj(KernelObj)) {
      VisitRecordBases(KernelObj, Handlers...);
      VisitRecordFields(KernelObj, Handlers...);
    } else {
      visitTopLevelRecord(KernelObj, KernelFunctorTy, Handlers...);
    }
  }

  // A visitor function that dispatches to functions as defined in
  // SyclKernelFieldHandler by iterating over a free function parameter list.
  template <typename... HandlerTys>
  void VisitFunctionParameters(const FunctionDecl *FreeFunc,
                               HandlerTys &...Handlers) {
    for (ParmVarDecl *Param : FreeFunc->parameters())
      visitParam(Param, Param->getType(), Handlers...);
  }

#undef KF_FOR_EACH
#undef KP_FOR_EACH
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
  static constexpr const bool VisitInsideSimpleContainersWithPointer = false;
  // Mark these virtual so that we can use override in the implementer classes,
  // despite virtual dispatch never being used.

  // SYCL special class can be a base class or a field decl, so both must be
  // handled.
  virtual bool handleSyclSpecialType(const CXXRecordDecl *,
                                     const CXXBaseSpecifier &, QualType) {
    return true;
  }
  virtual bool handleSyclSpecialType(FieldDecl *, QualType) { return true; }
  virtual bool handleSyclSpecialType(ParmVarDecl *, QualType) { return true; }

  virtual bool handleStructType(FieldDecl *, QualType) { return true; }
  virtual bool handleStructType(ParmVarDecl *, QualType) { return true; }
  virtual bool handleUnionType(FieldDecl *, QualType) { return true; }
  virtual bool handleUnionType(ParmVarDecl *, QualType) { return true; }
  virtual bool handleReferenceType(FieldDecl *, QualType) { return true; }
  virtual bool handleReferenceType(ParmVarDecl *, QualType) { return true; }
  virtual bool handlePointerType(FieldDecl *, QualType) { return true; }
  virtual bool handlePointerType(ParmVarDecl *, QualType) { return true; }
  virtual bool handleArrayType(FieldDecl *, QualType) { return true; }
  virtual bool handleArrayType(ParmVarDecl *, QualType) { return true; }
  virtual bool handleScalarType(FieldDecl *, QualType) { return true; }
  virtual bool handleScalarType(ParmVarDecl *, QualType) { return true; }
  // Most handlers shouldn't be handling this, just the field checker.
  virtual bool handleOtherType(FieldDecl *, QualType) { return true; }
  virtual bool handleOtherType(ParmVarDecl *, QualType) { return true; }

  // Handle the SYCL kernel as a whole. This applies only when the target can
  // support pointer to the generic address space as arguments and the functor
  // doesn't have any SYCL special types.
  virtual bool handleTopLevelStruct(const CXXRecordDecl *, QualType) {
    return true;
  }

  // Handle a simple struct that doesn't need to be decomposed, only called on
  // handlers with VisitInsideSimpleContainers as false.  Replaces
  // handleStructType, enterStruct, leaveStruct, and visiting of sub-elements.
  virtual bool handleNonDecompStruct(const CXXRecordDecl *, FieldDecl *,
                                     QualType) {
    return true;
  }

  virtual bool handleNonDecompStruct(const CXXRecordDecl *, ParmVarDecl *,
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
  virtual bool enterStruct(const CXXRecordDecl *, ParmVarDecl *, QualType) {
    return true;
  }
  virtual bool leaveStruct(const CXXRecordDecl *, ParmVarDecl *, QualType) {
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
  virtual bool enterUnion(const CXXRecordDecl *, ParmVarDecl *) { return true; }
  virtual bool leaveUnion(const CXXRecordDecl *, ParmVarDecl *) { return true; }

  // The following are used for stepping through array elements.
  virtual bool enterArray(FieldDecl *, QualType, QualType) { return true; }
  virtual bool leaveArray(FieldDecl *, QualType, QualType) { return true; }
  virtual bool enterArray(ParmVarDecl *, QualType, QualType) { return true; }
  virtual bool leaveArray(ParmVarDecl *, QualType, QualType) { return true; }

  virtual bool nextElement(QualType, uint64_t) { return true; }

  virtual ~SyclKernelFieldHandlerBase() = default;
};

// A class to act as the direct base for all the SYCL OpenCL Kernel construction
// tasks that contains a reference to Sema (and potentially any other
// universally required data).
class SyclKernelFieldHandler : public SyclKernelFieldHandlerBase {
protected:
  SemaSYCL &SemaSYCLRef;
  SyclKernelFieldHandler(SemaSYCL &S) : SemaSYCLRef(S) {}

  // Returns 'true' if the thing we're visiting (Based on the FD/QualType pair)
  // is an element of an array. FD will always be the array field. When
  // traversing the array field, Ty will be the type of the array field or the
  // type of array element (or some decomposed type from array).
  bool isArrayElement(const FieldDecl *FD, QualType Ty) const {
    return !SemaSYCLRef.getASTContext().hasSameType(FD->getType(), Ty);
  }
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
  HandlerFilter(H &) {}
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
  } else if (AnyTrue<HandlerTys::VisitInsideSimpleContainersWithPointer...>::
                 Value) {
    // We are currently in PointerHandler visitor.
    if (RD->hasAttr<SYCLGenerateNewTypeAttr>()) {
      // This is a record containing pointers.
      visitComplexRecord(Owner, Parent, Wrapper, RecordTy, Handlers...);
    } else {
      // This is a record without pointers.
      visitSimpleRecord(Owner, Parent, Wrapper, RecordTy, Handlers...);
    }
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
  } else if (AnyTrue<HandlerTys::VisitInsideSimpleContainersWithPointer...>::
                 Value) {
    // We are currently in PointerHandler visitor.
    if (Field->hasAttr<SYCLGenerateNewTypeAttr>()) {
      // This is an array of pointers, or an array of a type containing
      // pointers.
      visitComplexArray(Owner, Field, ArrayTy, Handlers...);
    } else {
      // This is an array which does not contain pointers.
      visitSimpleArray(Owner, Field, ArrayTy, Handlers...);
    }
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
  // Keeps track of whether we are currently handling fields inside a struct.
  // Fields of kernel functor or direct kernel captures will have a depth 0.
  int StructFieldDepth = 0;
  // Initialize with -1 so that fields of a base class of the kernel functor
  // has depth 0. Visitor method enterStruct increments this to 0 when the base
  // class is entered.
  int StructBaseDepth = -1;

  // Check whether the object should be disallowed from being copied to kernel.
  // Return true if not copyable, false if copyable.
  bool checkNotCopyableToKernel(const FieldDecl *FD, QualType FieldTy) {
    if (FieldTy->isArrayType()) {
      if (const auto *CAT =
              SemaSYCLRef.getASTContext().getAsConstantArrayType(FieldTy)) {
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
      return SemaSYCLRef.Diag(
          Loc, diag::err_sycl_invalid_accessor_property_template_param);

    QualType PropListTy = PropList.getAsType();
    if (!SemaSYCL::isSyclType(PropListTy, SYCLTypeAttr::accessor_property_list))
      return SemaSYCLRef.Diag(
          Loc, diag::err_sycl_invalid_accessor_property_template_param);

    const auto *AccPropListDecl =
        cast<ClassTemplateSpecializationDecl>(PropListTy->getAsRecordDecl());
    if (AccPropListDecl->getTemplateArgs().size() != 1)
      return SemaSYCLRef.Diag(Loc,
                              diag::err_sycl_invalid_property_list_param_number)
             << "accessor_property_list";

    const auto TemplArg = AccPropListDecl->getTemplateArgs()[0];
    if (TemplArg.getKind() != TemplateArgument::ArgKind::Pack)
      return SemaSYCLRef.Diag(
                 Loc,
                 diag::err_sycl_invalid_accessor_property_list_template_param)
             << /*accessor_property_list*/ 0 << /*parameter pack*/ 0;

    for (TemplateArgument::pack_iterator Prop = TemplArg.pack_begin();
         Prop != TemplArg.pack_end(); ++Prop) {
      if (Prop->getKind() != TemplateArgument::ArgKind::Type)
        return SemaSYCLRef.Diag(
                   Loc,
                   diag::err_sycl_invalid_accessor_property_list_template_param)
               << /*accessor_property_list pack argument*/ 1 << /*type*/ 1;
      QualType PropTy = Prop->getAsType();
      if (isAccessorPropertyType(PropTy, SYCLTypeAttr::buffer_location) &&
          checkBufferLocationType(PropTy, Loc))
        return true;
    }
    return false;
  }

  bool checkBufferLocationType(QualType PropTy, SourceLocation Loc) {
    const auto *PropDecl =
        cast<ClassTemplateSpecializationDecl>(PropTy->getAsRecordDecl());
    if (PropDecl->getTemplateArgs().size() != 1)
      return SemaSYCLRef.Diag(Loc,
                              diag::err_sycl_invalid_property_list_param_number)
             << "buffer_location";

    const auto BufferLoc = PropDecl->getTemplateArgs()[0];
    if (BufferLoc.getKind() != TemplateArgument::ArgKind::Integral)
      return SemaSYCLRef.Diag(
                 Loc,
                 diag::err_sycl_invalid_accessor_property_list_template_param)
             << /*buffer_location*/ 2 << /*non-negative integer*/ 2;

    int LocationID = static_cast<int>(BufferLoc.getAsIntegral().getExtValue());
    if (LocationID < 0)
      return SemaSYCLRef.Diag(
                 Loc,
                 diag::err_sycl_invalid_accessor_property_list_template_param)
             << /*buffer_location*/ 2 << /*non-negative integer*/ 2;

    return false;
  }

  bool checkSyclSpecialType(QualType Ty, SourceRange Loc) {
    assert(isSyclSpecialType(Ty, SemaSYCLRef) &&
           "Should only be called on sycl special class types.");

    // Annotated pointers and annotated arguments must be captured
    // directly by the SYCL kernel.
    if ((SemaSYCL::isSyclType(Ty, SYCLTypeAttr::annotated_ptr) ||
         SemaSYCL::isSyclType(Ty, SYCLTypeAttr::annotated_arg)) &&
        (StructFieldDepth > 0 || StructBaseDepth > 0))
      return SemaSYCLRef.Diag(Loc.getBegin(),
                              diag::err_bad_kernel_param_data_members)
             << Ty << /*Struct*/ 1;

    const RecordDecl *RecD = Ty->getAsRecordDecl();
    if (const ClassTemplateSpecializationDecl *CTSD =
            dyn_cast<ClassTemplateSpecializationDecl>(RecD)) {
      const TemplateArgumentList &TAL = CTSD->getTemplateArgs();
      TemplateArgument TA = TAL.get(0);

      // Parameter packs are used by properties so they are always valid.
      if (TA.getKind() != TemplateArgument::Pack) {
        llvm::DenseSet<QualType> Visited;
        checkSYCLType(SemaSYCLRef, TA.getAsType(), Loc, Visited);
      }

      if (TAL.size() > 5)
        return checkPropertyListType(TAL.get(5), Loc.getBegin());
    }
    return false;
  }

public:
  SyclKernelFieldChecker(SemaSYCL &S)
      : SyclKernelFieldHandler(S), Diag(S.getASTContext().getDiagnostics()) {}
  static constexpr const bool VisitNthArrayElement = false;
  bool isValid() { return !IsInvalid; }

  bool handleReferenceType(FieldDecl *FD, QualType FieldTy) final {
    Diag.Report(FD->getLocation(), diag::err_bad_kernel_param_type) << FieldTy;
    IsInvalid = true;
    return isValid();
  }

  bool handleReferenceType(ParmVarDecl *PD, QualType ParamTy) final {
    Diag.Report(PD->getLocation(), diag::err_bad_kernel_param_type) << ParamTy;
    IsInvalid = true;
    return isValid();
  }

  bool handleStructType(FieldDecl *, QualType FieldTy) final {
    CXXRecordDecl *RD = FieldTy->getAsCXXRecordDecl();
    assert(RD && "Not a RecordDecl inside the handler for struct type");
    if (RD->isLambda()) {
      for (const LambdaCapture &LC : RD->captures())
        if (LC.capturesThis() && LC.isImplicit()) {
          SemaSYCLRef.Diag(LC.getLocation(), diag::err_implicit_this_capture);
          IsInvalid = true;
        }
    }
    return isValid();
  }

  bool handleStructType(ParmVarDecl *PD, QualType ParamTy) final {
    if (SemaSYCLRef.getLangOpts().SYCLRTCMode) {
      // When compiling in RTC mode, the restriction regarding forward
      // declarations doesn't apply, as we don't need the integration header.
      return isValid();
    }
    CXXRecordDecl *RD = ParamTy->getAsCXXRecordDecl();
    // For free functions all struct/class kernel arguments are forward declared
    // in integration header, that adds additional restrictions for kernel
    // arguments.
    NotForwardDeclarableReason NFDR =
        isForwardDeclarable(RD, SemaSYCLRef, /*DiagForFreeFunction=*/true);
    if (NFDR != NotForwardDeclarableReason::None) {
      Diag.Report(PD->getLocation(),
                  diag::err_bad_kernel_param_type)
          << ParamTy;
      Diag.Report(PD->getLocation(),
                  diag::note_free_function_kernel_param_type_not_fwd_declarable)
          << ParamTy;
      IsInvalid = true;
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

  bool handleSyclSpecialType(ParmVarDecl *PD, QualType ParamTy) final {
    IsInvalid |= checkSyclSpecialType(ParamTy, PD->getLocation());
    return isValid();
  }

  bool handleArrayType(FieldDecl *FD, QualType FieldTy) final {
    IsInvalid |= checkNotCopyableToKernel(FD, FieldTy);
    return isValid();
  }

  bool handleArrayType(ParmVarDecl *PD, QualType ParamTy) final {
    Diag.Report(PD->getLocation(), diag::err_bad_kernel_param_type) << ParamTy;
    IsInvalid = true;
    return isValid();
  }

  bool handlePointerType(FieldDecl *FD, QualType FieldTy) final {
    while (FieldTy->isAnyPointerType()) {
      FieldTy = QualType{FieldTy->getPointeeOrArrayElementType(), 0};
      if (FieldTy->isVariableArrayType()) {
        Diag.Report(FD->getLocation(), diag::err_vla_unsupported) << 0;
        IsInvalid = true;
        break;
      }
    }
    return isValid();
  }

  bool handlePointerType(ParmVarDecl *PD, QualType ParamTy) final {
    while (ParamTy->isAnyPointerType()) {
      ParamTy = QualType{ParamTy->getPointeeOrArrayElementType(), 0};
      if (ParamTy->isVariableArrayType()) {
        Diag.Report(PD->getLocation(), diag::err_vla_unsupported) << 0;
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

  bool handleOtherType(ParmVarDecl *PD, QualType ParamTy) final {
    Diag.Report(PD->getLocation(), diag::err_bad_kernel_param_type) << ParamTy;
    IsInvalid = true;
    return isValid();
  }

  bool enterStruct(const CXXRecordDecl *, FieldDecl *, QualType) final {
    ++StructFieldDepth;
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, FieldDecl *, QualType) final {
    --StructFieldDepth;
    return true;
  }

  bool enterStruct(const CXXRecordDecl *, ParmVarDecl *, QualType) final {
    // TODO manipulate struct depth once special types are supported for free
    // function kernels.
    // ++StructFieldDepth;
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, ParmVarDecl *PD,
                   QualType ParamTy) final {
    // TODO manipulate struct depth once special types are supported for free
    // function kernels.
    // --StructFieldDepth;
    // TODO We don't yet support special types and therefore structs that
    // require decomposition and leaving/entering. Diagnose for better user
    // experience.
    CXXRecordDecl *RD = ParamTy->getAsCXXRecordDecl();
    if (RD->hasAttr<SYCLRequiresDecompositionAttr>()) {
      Diag.Report(PD->getLocation(),
                  diag::err_bad_kernel_param_type)
          << ParamTy;
      Diag.Report(PD->getLocation(),
                  diag::note_free_function_kernel_param_type_not_supported)
          << ParamTy;
      IsInvalid = true;
    }
    return isValid();
  }

  bool enterStruct(const CXXRecordDecl *, const CXXBaseSpecifier &,
                   QualType) final {
    ++StructBaseDepth;
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, const CXXBaseSpecifier &,
                   QualType) final {
    --StructBaseDepth;
    return true;
  }
};

// A type to check the validity of accessing accessor/sampler/stream
// types as kernel parameters inside union.
class SyclKernelUnionChecker : public SyclKernelFieldHandler {
  int UnionCount = 0;
  bool IsInvalid = false;
  DiagnosticsEngine &Diag;

public:
  SyclKernelUnionChecker(SemaSYCL &S)
      : SyclKernelFieldHandler(S), Diag(S.getASTContext().getDiagnostics()) {}
  bool isValid() { return !IsInvalid; }
  static constexpr const bool VisitUnionBody = true;
  static constexpr const bool VisitNthArrayElement = false;

  bool checkType(SourceLocation Loc, QualType Ty) {
    if (UnionCount) {
      IsInvalid = true;
      Diag.Report(Loc, diag::err_bad_kernel_param_data_members)
          << Ty << /*Union*/ 0;
    }
    return isValid();
  }

  bool enterUnion(const CXXRecordDecl *, FieldDecl *) override {
    ++UnionCount;
    return true;
  }

  bool enterUnion(const CXXRecordDecl *, ParmVarDecl *) override {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool leaveUnion(const CXXRecordDecl *, FieldDecl *) override {
    --UnionCount;
    return true;
  }

  bool leaveUnion(const CXXRecordDecl *, ParmVarDecl *) override {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool handleSyclSpecialType(FieldDecl *FD, QualType FieldTy) final {
    return checkType(FD->getLocation(), FieldTy);
  }

  bool handleSyclSpecialType(ParmVarDecl *PD, QualType ParamTy) final {
    return checkType(PD->getLocation(), ParamTy);
  }

  bool handleSyclSpecialType(const CXXRecordDecl *, const CXXBaseSpecifier &BS,
                             QualType FieldTy) final {
    return checkType(BS.getBeginLoc(), FieldTy);
  }
};

// A type to mark whether a collection requires decomposition
// or needs to be transformed to a new type. If a collection
// contains pointers, and is not decomposed, a new type must
// be generated with all pointers in global address space.
class SyclKernelDecompMarker : public SyclKernelFieldHandler {
  llvm::SmallVector<bool, 16> CollectionStack;
  llvm::SmallVector<bool, 16> PointerStack;

public:
  static constexpr const bool VisitUnionBody = false;
  static constexpr const bool VisitNthArrayElement = false;

  SyclKernelDecompMarker(SemaSYCL &S) : SyclKernelFieldHandler(S) {
    // Base entry.
    CollectionStack.push_back(false);
    PointerStack.push_back(false);
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

  bool handleSyclSpecialType(ParmVarDecl *, QualType) final {
    // TODO We don't support special types in free function kernel parameters,
    // but track them to diagnose the case properly.
    CollectionStack.back() = true;
    return true;
  }

  bool handlePointerType(FieldDecl *, QualType) final {
    PointerStack.back() = targetRequiresNewType(SemaSYCLRef.getASTContext());
    return true;
  }

  bool handlePointerType(ParmVarDecl *, QualType) final {
    PointerStack.back() = targetRequiresNewType(SemaSYCLRef.getASTContext());
    return true;
  }

  // Add Top level information to ease checks for processor.
  bool handleTopLevelStruct(const CXXRecordDecl *, QualType Ty) final {
    CXXRecordDecl *RD = Ty->getAsCXXRecordDecl();
    assert(RD && "should not be null.");
    if (CollectionStack.pop_back_val() ||
        SemaSYCLRef.getLangOpts().SYCLDecomposeStruct) {
      if (!RD->hasAttr<SYCLRequiresDecompositionAttr>())
        RD->addAttr(SYCLRequiresDecompositionAttr::CreateImplicit(
            SemaSYCLRef.getASTContext()));
      PointerStack.pop_back();
    } else if (PointerStack.pop_back_val()) {
      if (!RD->hasAttr<SYCLGenerateNewTypeAttr>())
        RD->addAttr(SYCLGenerateNewTypeAttr::CreateImplicit(
            SemaSYCLRef.getASTContext()));
    }
    assert(CollectionStack.size() == 0);
    assert(PointerStack.size() == 0);
    return true;
  }

  bool enterStruct(const CXXRecordDecl *, FieldDecl *, QualType) final {
    CollectionStack.push_back(false);
    PointerStack.push_back(false);
    return true;
  }

  bool enterStruct(const CXXRecordDecl *, ParmVarDecl *, QualType) final {
    CollectionStack.push_back(false);
    PointerStack.push_back(false);
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, FieldDecl *, QualType Ty) final {
    // If a record needs to be decomposed, it is marked with
    // SYCLRequiresDecompositionAttr. Else if a record contains
    // a pointer, it is marked with SYCLGenerateNewTypeAttr. A record
    // will never be marked with both attributes.
    CXXRecordDecl *RD = Ty->getAsCXXRecordDecl();
    assert(RD && "should not be null.");
    if (CollectionStack.pop_back_val()) {
      if (!RD->hasAttr<SYCLRequiresDecompositionAttr>())
        RD->addAttr(SYCLRequiresDecompositionAttr::CreateImplicit(
            SemaSYCLRef.getASTContext()));
      CollectionStack.back() = true;
      PointerStack.pop_back();
    } else if (PointerStack.pop_back_val()) {
      PointerStack.back() = true;
      if (!RD->hasAttr<SYCLGenerateNewTypeAttr>())
        RD->addAttr(SYCLGenerateNewTypeAttr::CreateImplicit(
            SemaSYCLRef.getASTContext()));
    }
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, ParmVarDecl *,
                   QualType ParamTy) final {
    CXXRecordDecl *RD = ParamTy->getAsCXXRecordDecl();
    assert(RD && "should not be null.");
    if (CollectionStack.pop_back_val()) {
      if (!RD->hasAttr<SYCLRequiresDecompositionAttr>())
        RD->addAttr(SYCLRequiresDecompositionAttr::CreateImplicit(
            SemaSYCLRef.getASTContext()));
      CollectionStack.back() = true;
      PointerStack.pop_back();
    } else if (PointerStack.pop_back_val()) {
      PointerStack.back() = true;
      if (!RD->hasAttr<SYCLGenerateNewTypeAttr>())
        RD->addAttr(SYCLGenerateNewTypeAttr::CreateImplicit(
            SemaSYCLRef.getASTContext()));
    }
    return true;
  }

  bool enterStruct(const CXXRecordDecl *, const CXXBaseSpecifier &,
                   QualType) final {
    CollectionStack.push_back(false);
    PointerStack.push_back(false);
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, const CXXBaseSpecifier &,
                   QualType Ty) final {
    // If a record needs to be decomposed, it is marked with
    // SYCLRequiresDecompositionAttr. Else if a record contains
    // a pointer, it is marked with SYCLGenerateNewTypeAttr. A record
    // will never be marked with both attributes.
    CXXRecordDecl *RD = Ty->getAsCXXRecordDecl();
    assert(RD && "should not be null.");
    if (CollectionStack.pop_back_val()) {
      if (!RD->hasAttr<SYCLRequiresDecompositionAttr>())
        RD->addAttr(SYCLRequiresDecompositionAttr::CreateImplicit(
            SemaSYCLRef.getASTContext()));
      CollectionStack.back() = true;
      PointerStack.pop_back();
    } else if (PointerStack.pop_back_val()) {
      PointerStack.back() = true;
      if (!RD->hasAttr<SYCLGenerateNewTypeAttr>())
        RD->addAttr(SYCLGenerateNewTypeAttr::CreateImplicit(
            SemaSYCLRef.getASTContext()));
    }
    return true;
  }

  bool enterArray(FieldDecl *, QualType, QualType) final {
    CollectionStack.push_back(false);
    PointerStack.push_back(false);
    return true;
  }

  bool enterArray(ParmVarDecl *, QualType, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool leaveArray(FieldDecl *FD, QualType, QualType) final {
    // If an array needs to be decomposed, it is marked with
    // SYCLRequiresDecompositionAttr. Else if the array is an array of pointers
    // or an array of structs containing pointers, it is marked with
    // SYCLGenerateNewTypeAttr. An array will never be marked with both
    // attributes.
    if (CollectionStack.pop_back_val()) {
      // Cannot assert, since in MD arrays we'll end up marking them multiple
      // times.
      if (!FD->hasAttr<SYCLRequiresDecompositionAttr>())
        FD->addAttr(SYCLRequiresDecompositionAttr::CreateImplicit(
            SemaSYCLRef.getASTContext()));
      CollectionStack.back() = true;
      PointerStack.pop_back();
    } else if (PointerStack.pop_back_val()) {
      if (!FD->hasAttr<SYCLGenerateNewTypeAttr>())
        FD->addAttr(SYCLGenerateNewTypeAttr::CreateImplicit(
            SemaSYCLRef.getASTContext()));
      PointerStack.back() = true;
    }
    return true;
  }

  bool leaveArray(ParmVarDecl *, QualType, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }
};

static QualType ModifyAddressSpace(SemaSYCL &SemaSYCLRef, QualType Ty) {
  // USM allows to use raw pointers instead of buffers/accessors, but these
  // pointers point to the specially allocated memory. For pointer fields,
  // except for function pointer fields, we add a kernel argument with the
  // same type as field but global address space, because OpenCL requires it.
  // Function pointers should have program address space. This is set in
  // CodeGen.
  QualType PointeeTy = Ty->getPointeeType();
  Qualifiers Quals = PointeeTy.getQualifiers();
  LangAS AS = Quals.getAddressSpace();
  // Leave global_device and global_host address spaces as is to help FPGA
  // device in memory allocations.
  if (!PointeeTy->isFunctionType() && AS != LangAS::sycl_global_device &&
      AS != LangAS::sycl_global_host)
    Quals.setAddressSpace(LangAS::sycl_global);
  PointeeTy = SemaSYCLRef.getASTContext().getQualifiedType(
      PointeeTy.getUnqualifiedType(), Quals);
  return SemaSYCLRef.getASTContext().getPointerType(PointeeTy);
}

// This visitor is used to traverse a non-decomposed record/array to
// generate a new type corresponding to this record/array.
class SyclKernelPointerHandler : public SyclKernelFieldHandler {
  llvm::SmallVector<CXXRecordDecl *, 8> ModifiedRecords;
  SmallVector<CXXBaseSpecifier *, 8> ModifiedBases;
  SmallVector<QualType, 8> ModifiedArrayElementsOrArray;

  IdentifierInfo *getModifiedName(IdentifierInfo *Id) {
    std::string Name =
        Id ? (Twine("__generated_") + Id->getName()).str() : "__generated_";
    return &SemaSYCLRef.getASTContext().Idents.get(Name);
  }

  // Create Decl for the new type we are generating.
  // The fields (and base classes) of this record will be generated as
  // the visitor traverses kernel object record fields.
  void createNewType(const CXXRecordDecl *RD) {
    auto *ModifiedRD = CXXRecordDecl::Create(
        SemaSYCLRef.getASTContext(), RD->getTagKind(),
        const_cast<DeclContext *>(RD->getDeclContext()), SourceLocation(),
        SourceLocation(), getModifiedName(RD->getIdentifier()));
    ModifiedRD->startDefinition();
    if (RD->hasAttrs())
      ModifiedRD->setAttrs(RD->getAttrs());
    ModifiedRecords.push_back(ModifiedRD);
  }

  // Create and add FieldDecl for FieldTy to generated record.
  void addField(const FieldDecl *FD, QualType FieldTy) {
    assert(!ModifiedRecords.empty() &&
           "ModifiedRecords should have at least 1 record");
    ASTContext &Ctx = SemaSYCLRef.getASTContext();
    auto *Field = FieldDecl::Create(
        Ctx, ModifiedRecords.back(), SourceLocation(), SourceLocation(),
        getModifiedName(FD->getIdentifier()), FieldTy,
        Ctx.getTrivialTypeSourceInfo(FieldTy, SourceLocation()), /*BW=*/nullptr,
        /*Mutable=*/false, ICIS_NoInit);
    Field->setAccess(FD->getAccess());
    if (FD->hasAttrs())
      Field->setAttrs(FD->getAttrs());
    // Add generated field to generated record.
    ModifiedRecords.back()->addDecl(Field);
  }

  void createBaseSpecifier(const CXXRecordDecl *Parent, const CXXRecordDecl *RD,
                           const CXXBaseSpecifier &BS) {
    TypeSourceInfo *TInfo =
        SemaSYCLRef.getASTContext().getTrivialTypeSourceInfo(
            QualType(RD->getTypeForDecl(), 0), SourceLocation());
    CXXBaseSpecifier *ModifiedBase = SemaSYCLRef.SemaRef.CheckBaseSpecifier(
        const_cast<CXXRecordDecl *>(Parent), SourceRange(), BS.isVirtual(),
        BS.getAccessSpecifier(), TInfo, SourceLocation());
    ModifiedBases.push_back(ModifiedBase);
  }

  CXXRecordDecl *getGeneratedNewRecord(const CXXRecordDecl *OldBaseDecl) {
    // At this point we have finished generating fields for the new
    // class corresponding to OldBaseDecl. Pop out the generated
    // record.
    CXXRecordDecl *ModifiedRD = ModifiedRecords.pop_back_val();
    ModifiedRD->completeDefinition();
    // Check the 'old' class for base classes.
    // Set bases classes for newly generated class if it has any.
    if (OldBaseDecl->getNumBases() > 0) {
      SmallVector<CXXBaseSpecifier *, 8> BasesForGeneratedClass;
      for (size_t I = 0; I < OldBaseDecl->getNumBases(); ++I)
        BasesForGeneratedClass.insert(BasesForGeneratedClass.begin(),
                                      ModifiedBases.pop_back_val());
      ModifiedRD->setBases(BasesForGeneratedClass.data(),
                           OldBaseDecl->getNumBases());
    }
    return ModifiedRD;
  }

public:
  static constexpr const bool VisitInsideSimpleContainersWithPointer = true;
  static constexpr const bool VisitNthArrayElement = false;
  SyclKernelPointerHandler(SemaSYCL &S, const CXXRecordDecl *RD)
      : SyclKernelFieldHandler(S) {
    createNewType(RD);
  }

  SyclKernelPointerHandler(SemaSYCL &S) : SyclKernelFieldHandler(S) {}

  bool enterStruct(const CXXRecordDecl *, FieldDecl *, QualType Ty) final {
    createNewType(Ty->getAsCXXRecordDecl());
    return true;
  }

  bool enterStruct(const CXXRecordDecl *, ParmVarDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, FieldDecl *FD, QualType Ty) final {
    CXXRecordDecl *ModifiedRD = getGeneratedNewRecord(Ty->getAsCXXRecordDecl());

    // Add this record as a field of it's parent record if it is not an
    // array element.
    if (!isArrayElement(FD, Ty))
      addField(FD, QualType(ModifiedRD->getTypeForDecl(), 0));
    else
      ModifiedArrayElementsOrArray.push_back(
          QualType(ModifiedRD->getTypeForDecl(), 0));

    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, ParmVarDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool enterStruct(const CXXRecordDecl *, const CXXBaseSpecifier &,
                   QualType Ty) final {
    createNewType(Ty->getAsCXXRecordDecl());
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *Parent, const CXXBaseSpecifier &BS,
                   QualType Ty) final {
    CXXRecordDecl *ModifiedRD = getGeneratedNewRecord(Ty->getAsCXXRecordDecl());

    // Create CXXBaseSpecifier for this generated class.
    createBaseSpecifier(Parent, ModifiedRD, BS);
    return true;
  }

  bool leaveArray(FieldDecl *FD, QualType ArrayTy, QualType) final {
    QualType ModifiedArrayElement = ModifiedArrayElementsOrArray.pop_back_val();

    const ConstantArrayType *CAT =
        SemaSYCLRef.getASTContext().getAsConstantArrayType(ArrayTy);
    assert(CAT && "Should only be called on constant-size array.");
    QualType ModifiedArray = SemaSYCLRef.getASTContext().getConstantArrayType(
        ModifiedArrayElement, CAT->getSize(),
        const_cast<Expr *>(CAT->getSizeExpr()), CAT->getSizeModifier(),
        CAT->getIndexTypeCVRQualifiers());

    if (ModifiedRecords.empty()) {
      // This is a top-level kernel argument.
      ModifiedArrayElementsOrArray.push_back(ModifiedArray);
    } else if (!isArrayElement(FD, ArrayTy)) {
      // Add this array field as a field of it's parent record.
      addField(FD, ModifiedArray);
    } else {
      // Multi-dimensional array element.
      ModifiedArrayElementsOrArray.push_back(ModifiedArray);
    }

    return true;
  }

  bool leaveArray(ParmVarDecl *, QualType, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool handlePointerType(FieldDecl *FD, QualType FieldTy) final {
    QualType ModifiedPointerType = ModifyAddressSpace(SemaSYCLRef, FieldTy);
    if (!isArrayElement(FD, FieldTy))
      addField(FD, ModifiedPointerType);
    else
      ModifiedArrayElementsOrArray.push_back(ModifiedPointerType);
    // We do not need to wrap pointers since this is a pointer inside
    // non-decomposed struct.
    return true;
  }

  bool handlePointerType(ParmVarDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool handleScalarType(FieldDecl *FD, QualType FieldTy) final {
    addField(FD, FieldTy);
    return true;
  }

  bool handleScalarType(ParmVarDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool handleUnionType(FieldDecl *FD, QualType FieldTy) final {
    return handleScalarType(FD, FieldTy);
  }

  bool handleUnionType(ParmVarDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, FieldDecl *FD,
                             QualType Ty) final {
    addField(FD, Ty);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, ParmVarDecl *,
                             QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *Parent,
                             const CXXBaseSpecifier &BS, QualType Ty) final {
    createBaseSpecifier(Parent, Ty->getAsCXXRecordDecl(), BS);
    return true;
  }

  bool handleSimpleArrayType(FieldDecl *FD, QualType Ty) final {
    addField(FD, Ty);
    return true;
  }

public:
  QualType getNewRecordType() {
    CXXRecordDecl *ModifiedRD = ModifiedRecords.pop_back_val();
    ModifiedRD->completeDefinition();

    if (!ModifiedBases.empty())
      ModifiedRD->setBases(ModifiedBases.data(), ModifiedBases.size());

    return QualType(ModifiedRD->getTypeForDecl(), 0);
  }
  QualType getNewArrayType() {
    return ModifiedArrayElementsOrArray.pop_back_val();
  }
};

// A type to Create and own the FunctionDecl for the kernel.
class SyclKernelDeclCreator : public SyclKernelFieldHandler {
  FunctionDecl *KernelDecl = nullptr;
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

  void addParam(const ParmVarDecl *PD, QualType ParamTy) {
    ParamDesc newParamDesc = makeParamDesc(PD, ParamTy);
    addParam(newParamDesc, ParamTy);
  }

  void addParam(const CXXBaseSpecifier &, QualType FieldTy) {
    // TODO: There is no name for the base available, but duplicate names are
    // seemingly already possible, so we'll give them all the same name for now.
    // This only happens with the accessor types.
    StringRef Name = "_arg__base";
    ParamDesc newParamDesc =
        makeParamDesc(SemaSYCLRef.getASTContext(), Name, FieldTy);
    addParam(newParamDesc, FieldTy);
  }
  // Add a parameter with specified name and type
  void addParam(StringRef Name, QualType ParamTy) {
    ParamDesc newParamDesc =
        makeParamDesc(SemaSYCLRef.getASTContext(), Name, ParamTy);
    addParam(newParamDesc, ParamTy);
  }

  void addParam(ParamDesc newParamDesc, QualType) {
    // Create a new ParmVarDecl based on the new info.
    ASTContext &Ctx = SemaSYCLRef.getASTContext();
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
      if (isAccessorPropertyType(PropTy, SYCLTypeAttr::no_alias))
        handleNoAliasProperty(Param, PropTy, Loc);
      if (isAccessorPropertyType(PropTy, SYCLTypeAttr::buffer_location))
        handleBufferLocationProperty(Param, PropTy, Loc);
    }
  }

  void handleNoAliasProperty(ParmVarDecl *Param, QualType, SourceLocation Loc) {
    ASTContext &Ctx = SemaSYCLRef.getASTContext();
    Param->addAttr(
        RestrictAttr::CreateImplicit(Ctx, nullptr, ParamIdx(1, Param), Loc));
  }

  // Obtain an integer value stored in a template parameter of buffer_location
  // property to pass it to buffer_location kernel attribute
  void handleBufferLocationProperty(ParmVarDecl *Param, QualType PropTy,
                                    SourceLocation Loc) {
    // If we have more than 1 buffer_location properties on a single
    // accessor - emit an error
    if (Param->hasAttr<SYCLIntelBufferLocationAttr>()) {
      SemaSYCLRef.Diag(Loc, diag::err_sycl_compiletime_property_duplication)
          << "buffer_location";
      return;
    }
    ASTContext &Ctx = SemaSYCLRef.getASTContext();
    const auto *PropDecl =
        cast<ClassTemplateSpecializationDecl>(PropTy->getAsRecordDecl());
    const auto BufferLoc = PropDecl->getTemplateArgs()[0];
    int LocationID = static_cast<int>(BufferLoc.getAsIntegral().getExtValue());
    Param->addAttr(
        SYCLIntelBufferLocationAttr::CreateImplicit(Ctx, LocationID));
  }

  // Additional processing is required for accessor type.
  void handleAccessorType(QualType FieldTy, const CXXRecordDecl *RecordDecl,
                          SourceLocation Loc) {
    handleAccessorPropertyList(Params.back(), RecordDecl, Loc);

    // If "accessor" type check if read only
    if (SemaSYCL::isSyclType(FieldTy, SYCLTypeAttr::accessor)) {
      // Get access mode of accessor.
      const auto *AccessorSpecializationDecl =
          cast<ClassTemplateSpecializationDecl>(RecordDecl);
      const TemplateArgument &AccessModeArg =
          AccessorSpecializationDecl->getTemplateArgs().get(2);
      if (isReadOnlyAccessor(AccessModeArg))
        Params.back()->addAttr(SYCLAccessorReadonlyAttr::CreateImplicit(
            SemaSYCLRef.getASTContext()));
    }

    // Add implicit attribute to parameter decl when it is a read only
    // SYCL accessor.
    Params.back()->addAttr(
        SYCLAccessorPtrAttr::CreateImplicit(SemaSYCLRef.getASTContext()));
  }

  // All special SYCL objects must have __init method. We extract types for
  // kernel parameters from __init method parameters. We will use __init method
  // and kernel parameters which we build here to initialize special objects in
  // the kernel body.
  // ParentDecl parameterizes whether we are in a free function kernel or a
  // lambda kernel by taking the value ParmVarDecl or FieldDecl respectively.
  template <typename ParentDecl>
  bool handleSpecialType(ParentDecl *decl, QualType Ty) {
    const auto *RD = Ty->getAsCXXRecordDecl();
    assert(RD && "The type must be a RecordDecl");
    llvm::StringLiteral MethodName =
        KernelDecl->hasAttr<SYCLSimdAttr>() && isSyclAccessorType(Ty)
            ? InitESIMDMethodName
            : InitMethodName;
    CXXMethodDecl *InitMethod = getMethodByName(RD, MethodName);
    assert(InitMethod && "The type must have the __init method");

    // Don't do -1 here because we count on this to be the first parameter added
    // (if any).
    size_t ParamIndex = Params.size();
    for (const ParmVarDecl *Param : InitMethod->parameters()) {
      QualType ParamTy = Param->getType();
      // For lambda kernels the arguments to the OpenCL kernel are named
      // based on the position they have as fields in the definition of the
      // special type structure i.e __arg_field1, __arg_field2 and so on.
      // For free function kernels the arguments are named in direct mapping
      // with the names they have in the __init method i.e __arg_Ptr for work
      // group memory since its init function takes a parameter with Ptr name.
      if constexpr (std::is_same_v<ParentDecl, FieldDecl>)
        addParam(decl, ParamTy.getCanonicalType());
      else
        addParam(Param, ParamTy.getCanonicalType());
      // Propagate add_ir_attributes_kernel_parameter attribute.
      if (const auto *AddIRAttr =
              Param->getAttr<SYCLAddIRAttributesKernelParameterAttr>())
        Params.back()->addAttr(AddIRAttr->clone(SemaSYCLRef.getASTContext()));

      // FIXME: This code is temporary, and will be removed once __init_esimd
      // is removed and property list refactored.
      // The function handleAccessorType includes a call to
      // handleAccessorPropertyList. If new classes with property list are
      // added, this code needs to be refactored to call
      // handleAccessorPropertyList for each class which requires it.
      if (ParamTy.getTypePtr()->isPointerType() && isSyclAccessorType(Ty))
        handleAccessorType(Ty, RD, decl->getBeginLoc());
    }
    LastParamIndex = ParamIndex;
    return true;
  }

  static void setKernelImplicitAttrs(ASTContext &Context, FunctionDecl *FD,
                                     bool IsSIMDKernel) {
    // Set implicit attributes.
    FD->addAttr(DeviceKernelAttr::CreateImplicit(Context));
    FD->addAttr(ArtificialAttr::CreateImplicit(Context));
    if (IsSIMDKernel)
      FD->addAttr(SYCLSimdAttr::CreateImplicit(Context));
  }

  static FunctionDecl *createKernelDecl(ASTContext &Ctx, SourceLocation Loc,
                                        bool IsInline, bool IsSIMDKernel) {
    // Create this with no prototype, and we can fix this up after we've seen
    // all the params.
    FunctionProtoType::ExtProtoInfo Info(CC_DeviceKernel);
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

  // If the record has been marked with SYCLGenerateNewTypeAttr,
  // it implies that it contains a pointer within. This function
  // defines a PointerHandler visitor which visits this record
  // recursively and modifies the address spaces of any pointer
  // found as required, thereby generating a new record with all
  // pointers in 'right' address space. PointerHandler.getNewRecordType()
  // returns this generated type.
  QualType GenerateNewRecordType(const CXXRecordDecl *RD) {
    SyclKernelPointerHandler PointerHandler(SemaSYCLRef, RD);
    KernelObjVisitor Visitor{SemaSYCLRef};
    Visitor.VisitRecordBases(RD, PointerHandler);
    Visitor.VisitRecordFields(RD, PointerHandler);
    return PointerHandler.getNewRecordType();
  }

  // If the array has been marked with SYCLGenerateNewTypeAttr,
  // it implies that this is an array of pointers, or an array
  // of a type which contains pointers. This function generates
  // a new array with all pointers in the required address space.
  QualType GenerateNewArrayType(FieldDecl *FD, QualType FieldTy) {
    const auto *Owner = dyn_cast<CXXRecordDecl>(FD->getParent());
    SyclKernelPointerHandler PointerHandler(SemaSYCLRef);
    KernelObjVisitor Visitor{SemaSYCLRef};
    Visitor.visitArray(Owner, FD, FieldTy, PointerHandler);
    return PointerHandler.getNewArrayType();
  }

public:
  static constexpr const bool VisitInsideSimpleContainers = false;
  SyclKernelDeclCreator(SemaSYCL &S, SourceLocation Loc, bool IsInline,
                        bool IsSIMDKernel, FunctionDecl *SYCLKernel)
      : SyclKernelFieldHandler(S),
        KernelDecl(
            createKernelDecl(S.getASTContext(), Loc, IsInline, IsSIMDKernel)),
        FuncContext(SemaSYCLRef.SemaRef, KernelDecl) {
    S.addSyclOpenCLKernel(SYCLKernel, KernelDecl);
    for (const auto *IRAttr :
         SYCLKernel->specific_attrs<SYCLAddIRAttributesFunctionAttr>()) {
      KernelDecl->addAttr(IRAttr->clone(SemaSYCLRef.getASTContext()));
    }
  }

  ~SyclKernelDeclCreator() {
    ASTContext &Ctx = SemaSYCLRef.getASTContext();
    FunctionProtoType::ExtProtoInfo Info(CC_DeviceKernel);

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
        DeviceKernelAttr::CreateImplicit(SemaSYCLRef.getASTContext()));

    SemaSYCLRef.addSyclDeviceDecl(KernelDecl);
  }

  bool enterStruct(const CXXRecordDecl *, FieldDecl *, QualType) final {
    ++StructDepth;
    return true;
  }

  bool enterStruct(const CXXRecordDecl *, ParmVarDecl *, QualType) final {
    // TODO
    // ++StructDepth;
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, FieldDecl *, QualType) final {
    --StructDepth;
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, ParmVarDecl *, QualType) final {
    // TODO
    // --StructDepth;
    return true;
  }

  bool enterStruct(const CXXRecordDecl *, const CXXBaseSpecifier &,
                   QualType) final {
    ++StructDepth;
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, const CXXBaseSpecifier &,
                   QualType) final {
    --StructDepth;
    return true;
  }

  bool handleSyclSpecialType(const CXXRecordDecl *, const CXXBaseSpecifier &BS,
                             QualType FieldTy) final {
    const auto *RecordDecl = FieldTy->getAsCXXRecordDecl();
    assert(RecordDecl && "The type must be a RecordDecl");
    llvm::StringLiteral MethodName =
        KernelDecl->hasAttr<SYCLSimdAttr>() && isSyclAccessorType(FieldTy)
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
      if (ParamTy.getTypePtr()->isPointerType() && isSyclAccessorType(FieldTy))
        handleAccessorType(FieldTy, RecordDecl, BS.getBeginLoc());
    }
    LastParamIndex = ParamIndex;
    return true;
  }

  bool handleSyclSpecialType(FieldDecl *FD, QualType FieldTy) final {
    return handleSpecialType(FD, FieldTy);
  }

  bool handleSyclSpecialType(ParmVarDecl *PD, QualType ParamTy) final {
    return handleSpecialType(PD, ParamTy);
  }

  RecordDecl *wrapField(FieldDecl *Field, QualType FieldTy) {
    RecordDecl *WrapperClass =
        SemaSYCLRef.getASTContext().buildImplicitRecord("__wrapper_class");
    WrapperClass->startDefinition();
    Field = FieldDecl::Create(
        SemaSYCLRef.getASTContext(), WrapperClass, SourceLocation(),
        SourceLocation(), /*Id=*/nullptr, FieldTy,
        SemaSYCLRef.getASTContext().getTrivialTypeSourceInfo(FieldTy,
                                                             SourceLocation()),
        /*BW=*/nullptr, /*Mutable=*/false, /*InitStyle=*/ICIS_NoInit);
    Field->setAccess(AS_public);
    WrapperClass->addDecl(Field);
    WrapperClass->completeDefinition();
    return WrapperClass;
  };

  bool handlePointerType(FieldDecl *FD, QualType FieldTy) final {
    QualType ModTy = ModifyAddressSpace(SemaSYCLRef, FieldTy);
    // When the kernel is generated, struct type kernel arguments are
    // decomposed; i.e. the parameters of the kernel are the fields of the
    // struct, and not the struct itself. This causes an error in the backend
    // when the struct field is a pointer, since non-USM pointers cannot be
    // passed directly. To work around this issue, all pointers inside the
    // struct are wrapped in a generated '__wrapper_class'.
    if (StructDepth) {
      RecordDecl *WrappedPointer = wrapField(FD, ModTy);
      ModTy = SemaSYCLRef.getASTContext().getRecordType(WrappedPointer);
    }

    addParam(FD, ModTy);
    return true;
  }

  bool handlePointerType(ParmVarDecl *PD, QualType ParamTy) final {
    QualType ModTy = ModifyAddressSpace(SemaSYCLRef, ParamTy);
    addParam(PD, ModTy);
    return true;
  }

  bool handleSimpleArrayType(FieldDecl *FD, QualType FieldTy) final {
    QualType ArrayTy = FieldTy;

    // This is an array of pointers or an array of a type with pointer.
    if (FD->hasAttr<SYCLGenerateNewTypeAttr>())
      ArrayTy = GenerateNewArrayType(FD, FieldTy);

    // Arrays are wrapped in a struct since they cannot be passed directly.
    RecordDecl *WrappedArray = wrapField(FD, ArrayTy);
    addParam(FD, SemaSYCLRef.getASTContext().getRecordType(WrappedArray));
    return true;
  }

  bool handleScalarType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FD, FieldTy);
    return true;
  }

  bool handleScalarType(ParmVarDecl *PD, QualType ParamTy) final {
    addParam(PD, ParamTy);
    return true;
  }

  bool handleTopLevelStruct(const CXXRecordDecl *, QualType Ty) final {
    StringRef Name = "_arg__sycl_functor";
    addParam(Name, Ty);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, FieldDecl *FD,
                             QualType Ty) final {
    // This is a field which should not be decomposed.
    CXXRecordDecl *FieldRecordDecl = Ty->getAsCXXRecordDecl();
    assert(FieldRecordDecl && "Type must be a C++ record type");
    // Check if we need to generate a new type for this record,
    // i.e. this record contains pointers.
    if (FieldRecordDecl->hasAttr<SYCLGenerateNewTypeAttr>())
      addParam(FD, GenerateNewRecordType(FieldRecordDecl));
    else
      addParam(FD, Ty);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, ParmVarDecl *PD,
                             QualType ParamTy) final {
    // This is a struct parameter which should not be decomposed.
    CXXRecordDecl *ParamRecordDecl = ParamTy->getAsCXXRecordDecl();
    assert(ParamRecordDecl && "Type must be a C++ record type");
    // Check if we need to generate a new type for this record,
    // i.e. this record contains pointers.
    if (ParamRecordDecl->hasAttr<SYCLGenerateNewTypeAttr>())
      addParam(PD, GenerateNewRecordType(ParamRecordDecl));
    else
      addParam(PD, ParamTy);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, const CXXBaseSpecifier &BS,
                             QualType Ty) final {
    // This is a base class which should not be decomposed.
    CXXRecordDecl *BaseRecordDecl = Ty->getAsCXXRecordDecl();
    assert(BaseRecordDecl && "Type must be a C++ record type");
    // Check if we need to generate a new type for this record,
    // i.e. this record contains pointers.
    if (BaseRecordDecl->hasAttr<SYCLGenerateNewTypeAttr>())
      addParam(BS, GenerateNewRecordType(BaseRecordDecl));
    else
      addParam(BS, Ty);
    return true;
  }

  bool handleUnionType(FieldDecl *FD, QualType FieldTy) final {
    return handleScalarType(FD, FieldTy);
  }

  bool handleUnionType(ParmVarDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  // Generate kernel argument to initialize specialization constants.
  void handleSyclKernelHandlerType() {
    ASTContext &Context = SemaSYCLRef.getASTContext();
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
};

// This Visitor traverses the AST of the function with
// `sycl_kernel` attribute and returns the version of “operator()()” that is
// called by KernelFunc. There will only be one call to KernelFunc in that
// AST because the DPC++ headers are structured such that the user’s
// kernel function is only called once. This ensures that the correct
// “operator()()” function call is returned, when a named function object used
// to define a kernel has more than one “operator()()” calls defined in it. For
// example, in the code below, 'operator()(sycl::id<1> id)' is returned based on
// the 'parallel_for' invocation which takes a 'sycl::range<1>(16)' argument.
//   class MyKernel {
//    public:
//      void operator()() const {
//        // code
//      }
//
//      [[sycl::reqd_sub_group_size(4)]] void operator()(sycl::id<1> id) const
//      {
//        // code
//      }
//    };
//
//    int main() {
//
//    Q.submit([&](sycl::handler& cgh) {
//      MyKernel kernelFunctorObject;
//      cgh.parallel_for(sycl::range<1>(16), kernelFunctorObject);
//    });
//      return 0;
//    }

class KernelCallOperatorVisitor
    : public RecursiveASTVisitor<KernelCallOperatorVisitor> {

  FunctionDecl *KernelCallerFunc;

public:
  CXXMethodDecl *CallOperator = nullptr;
  const CXXRecordDecl *KernelObj;

  KernelCallOperatorVisitor(FunctionDecl *KernelCallerFunc,
                            const CXXRecordDecl *KernelObj)
      : KernelCallerFunc(KernelCallerFunc), KernelObj(KernelObj) {}

  bool VisitCallExpr(CallExpr *CE) {
    Decl *CalleeDecl = CE->getCalleeDecl();
    if (isa_and_nonnull<CXXMethodDecl>(CalleeDecl)) {
      CXXMethodDecl *MD = cast<CXXMethodDecl>(CalleeDecl);
      if (MD->getOverloadedOperator() == OO_Call &&
          MD->getParent() == KernelObj) {
        CallOperator = MD;
      }
    }
    return true;
  }

  CXXMethodDecl *getCallOperator() {
    if (CallOperator)
      return CallOperator;

    TraverseDecl(KernelCallerFunc);
    return CallOperator;
  }
};

class ESIMDKernelDiagnostics : public SyclKernelFieldHandler {

  SourceLocation KernelLoc;
  bool IsESIMD = false;

  bool handleSpecialType(QualType FieldTy) {
    const CXXRecordDecl *RecordDecl = FieldTy->getAsCXXRecordDecl();

    if (IsESIMD && !isSyclAccessorType(FieldTy))
      return SemaSYCLRef.Diag(KernelLoc,
                              diag::err_sycl_esimd_not_supported_for_type)
             << RecordDecl;
    return true;
  }

public:
  ESIMDKernelDiagnostics(SemaSYCL &S, SourceLocation Loc, bool IsESIMD)
      : SyclKernelFieldHandler(S), KernelLoc(Loc), IsESIMD(IsESIMD) {}

  bool handleSyclSpecialType(FieldDecl *, QualType FieldTy) final {
    return handleSpecialType(FieldTy);
  }

  bool handleSyclSpecialType(const CXXRecordDecl *, const CXXBaseSpecifier &,
                             QualType FieldTy) final {
    return handleSpecialType(FieldTy);
  }

  using SyclKernelFieldHandler::handleSyclSpecialType;
};

class SyclKernelArgsSizeChecker : public SyclKernelFieldHandler {
  SourceLocation KernelLoc;
  unsigned SizeOfParams = 0;
  bool IsESIMD = false;

  void addParam(QualType ArgTy) {
    SizeOfParams +=
        SemaSYCLRef.getASTContext().getTypeSizeInChars(ArgTy).getQuantity();
  }

  bool handleSpecialType(QualType FieldTy) {
    const CXXRecordDecl *RecordDecl = FieldTy->getAsCXXRecordDecl();
    assert(RecordDecl && "The type must be a RecordDecl");
    llvm::StringLiteral MethodName = (IsESIMD && isSyclAccessorType(FieldTy))
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
  SyclKernelArgsSizeChecker(SemaSYCL &S, SourceLocation Loc, bool IsESIMD)
      : SyclKernelFieldHandler(S), KernelLoc(Loc), IsESIMD(IsESIMD) {}

  ~SyclKernelArgsSizeChecker() {
    if (SizeOfParams > MaxKernelArgsSize)
      SemaSYCLRef.Diag(KernelLoc, diag::warn_sycl_kernel_too_big_args)
          << SizeOfParams << MaxKernelArgsSize;
  }

  bool handleSyclSpecialType(FieldDecl *, QualType FieldTy) final {
    return handleSpecialType(FieldTy);
  }

  bool handleSyclSpecialType(ParmVarDecl *, QualType ParamTy) final {
    return handleSpecialType(ParamTy);
  }

  bool handleSyclSpecialType(const CXXRecordDecl *, const CXXBaseSpecifier &,
                             QualType FieldTy) final {
    return handleSpecialType(FieldTy);
  }

  bool handlePointerType(FieldDecl *, QualType FieldTy) final {
    addParam(FieldTy);
    return true;
  }

  bool handlePointerType(ParmVarDecl *, QualType ParamTy) final {
    addParam(ParamTy);
    return true;
  }

  bool handleScalarType(FieldDecl *, QualType FieldTy) final {
    addParam(FieldTy);
    return true;
  }

  bool handleScalarType(ParmVarDecl *, QualType ParamTy) final {
    addParam(ParamTy);
    return true;
  }

  bool handleSimpleArrayType(FieldDecl *, QualType FieldTy) final {
    addParam(FieldTy);
    return true;
  }

  bool handleTopLevelStruct(const CXXRecordDecl *, QualType Ty) final {
    addParam(Ty);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, FieldDecl *,
                             QualType Ty) final {
    addParam(Ty);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, ParmVarDecl *,
                             QualType ParamTy) final {
    addParam(ParamTy);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, const CXXBaseSpecifier &,
                             QualType Ty) final {
    addParam(Ty);
    return true;
  }

  bool handleUnionType(FieldDecl *FD, QualType FieldTy) final {
    return handleScalarType(FD, FieldTy);
  }

  bool handleUnionType(ParmVarDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }
};

std::string getKernelArgDesc(StringRef KernelArgDescription) {
  if (KernelArgDescription == "")
    return "";
  return ("Compiler generated argument for " + KernelArgDescription + ",")
      .str();
}

class SyclOptReportCreator : public SyclKernelFieldHandler {
  SyclKernelDeclCreator &DC;
  SourceLocation KernelInvocationLoc;

  void addParam(const FieldDecl *KernelArg, QualType KernelArgType,
                StringRef KernelArgDescription,
                bool IsCompilerGeneratedType = false) {
    StringRef NameToEmitInDescription = KernelArg->getName();
    const RecordDecl *KernelArgParent = KernelArg->getParent();
    if (KernelArgParent && KernelArgDescription == "decomposed struct/class")
      NameToEmitInDescription = KernelArgParent->getName();

    unsigned KernelArgSize = SemaSYCLRef.getASTContext()
                                 .getTypeSizeInChars(KernelArgType)
                                 .getQuantity();

    SemaSYCLRef.getDiagnostics().getSYCLOptReport().AddKernelArgs(
        DC.getKernelDecl(), NameToEmitInDescription,
        IsCompilerGeneratedType ? "Compiler generated"
                                : KernelArgType.getAsString(),
        KernelInvocationLoc, KernelArgSize,
        getKernelArgDesc(KernelArgDescription),
        (KernelArgDescription == "decomposed struct/class")
            ? ("Field:" + KernelArg->getName().str() + ", ")
            : "");
  }

  void addParam(const FieldDecl *FD, QualType FieldTy) {
    std::string KernelArgDescription = "";
    const RecordDecl *RD = FD->getParent();
    if (RD && RD->hasAttr<SYCLRequiresDecompositionAttr>())
      KernelArgDescription = "decomposed struct/class";

    addParam(FD, FieldTy, KernelArgDescription);
  }

  // Handles base classes.
  void addParam(const CXXBaseSpecifier &, QualType KernelArgType,
                StringRef KernelArgDescription,
                bool IsCompilerGeneratedType = false) {
    unsigned KernelArgSize = SemaSYCLRef.getASTContext()
                                 .getTypeSizeInChars(KernelArgType)
                                 .getQuantity();
    SemaSYCLRef.getDiagnostics().getSYCLOptReport().AddKernelArgs(
        DC.getKernelDecl(), KernelArgType.getAsString(),
        IsCompilerGeneratedType ? "Compiler generated"
                                : KernelArgType.getAsString(),
        KernelInvocationLoc, KernelArgSize,
        getKernelArgDesc(KernelArgDescription), "");
  }

  // Handles specialization constants.
  void addParam(QualType KernelArgType, std::string KernelArgDescription) {
    unsigned KernelArgSize = SemaSYCLRef.getASTContext()
                                 .getTypeSizeInChars(KernelArgType)
                                 .getQuantity();
    SemaSYCLRef.getDiagnostics().getSYCLOptReport().AddKernelArgs(
        DC.getKernelDecl(), "", KernelArgType.getAsString(),
        KernelInvocationLoc, KernelArgSize,
        getKernelArgDesc(KernelArgDescription), "");
  }

public:
  static constexpr const bool VisitInsideSimpleContainers = false;
  SyclOptReportCreator(SemaSYCL &S, SyclKernelDeclCreator &DC,
                       SourceLocation Loc)
      : SyclKernelFieldHandler(S), DC(DC), KernelInvocationLoc(Loc) {}

  using SyclKernelFieldHandler::handleSyclSpecialType;
  bool handleSyclSpecialType(FieldDecl *FD, QualType FieldTy) final {
    for (const auto *Param : DC.getParamVarDeclsForCurrentField())
      addParam(FD, Param->getType(), FieldTy.getAsString());
    return true;
  }

  bool handleSyclSpecialType(const CXXRecordDecl *, const CXXBaseSpecifier &,
                             QualType FieldTy) final {
    std::string KernelArgDescription = "base class " + FieldTy.getAsString();
    for (const auto *Param : DC.getParamVarDeclsForCurrentField()) {
      QualType KernelArgType = Param->getType();
      unsigned KernelArgSize = SemaSYCLRef.getASTContext()
                                   .getTypeSizeInChars(KernelArgType)
                                   .getQuantity();
      SemaSYCLRef.getDiagnostics().getSYCLOptReport().AddKernelArgs(
          DC.getKernelDecl(), FieldTy.getAsString(),
          KernelArgType.getAsString(), KernelInvocationLoc, KernelArgSize,
          getKernelArgDesc(KernelArgDescription), "");
    }
    return true;
  }

  using SyclKernelFieldHandler::handlePointerType;
  bool handlePointerType(FieldDecl *FD, QualType) final {
    std::string KernelArgDescription = "";
    bool IsCompilerGeneratedType = false;
    ParmVarDecl *KernelParameter = DC.getParamVarDeclsForCurrentField()[0];
    // Compiler generated openCL kernel argument for current pointer field
    // is not a pointer. This means we are processing a nested pointer and
    // the openCL kernel argument is of type __wrapper_class.
    if (!KernelParameter->getType()->isPointerType()) {
      KernelArgDescription = "nested pointer";
      IsCompilerGeneratedType = true;
    }

    for (const auto *Param : DC.getParamVarDeclsForCurrentField())
      addParam(FD, Param->getType(), KernelArgDescription,
               IsCompilerGeneratedType);
    return true;
  }

  using SyclKernelFieldHandler::handleScalarType;
  bool handleScalarType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FD, FieldTy);
    return true;
  }

  using SyclKernelFieldHandler::handleSimpleArrayType;
  bool handleSimpleArrayType(FieldDecl *FD, QualType) final {
    // Simple arrays are always wrapped.
    for (const auto *Param : DC.getParamVarDeclsForCurrentField())
      addParam(FD, Param->getType(), "array", /*IsCompilerGeneratedType*/ true);
    return true;
  }

  bool handleTopLevelStruct(const CXXRecordDecl *, QualType) final {
    addParam(DC.getParamVarDeclsForCurrentField()[0]->getType(),
             "SYCL Functor");
    return true;
  }

  using SyclKernelFieldHandler::handleNonDecompStruct;
  bool handleNonDecompStruct(const CXXRecordDecl *, FieldDecl *FD,
                             QualType Ty) final {
    CXXRecordDecl *RD = Ty->getAsCXXRecordDecl();
    assert(RD && "Type must be a C++ record type");
    if (RD->hasAttr<SYCLGenerateNewTypeAttr>())
      addParam(FD, Ty, "object with pointer", /*IsCompilerGeneratedType*/ true);
    else
      addParam(FD, Ty);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, const CXXBaseSpecifier &BS,
                             QualType Ty) final {
    CXXRecordDecl *RD = Ty->getAsCXXRecordDecl();
    assert(RD && "Type must be a C++ record type");
    if (RD->hasAttr<SYCLGenerateNewTypeAttr>())
      addParam(BS, Ty, "base class with pointer",
               /*IsCompilerGeneratedType*/ true);
    else
      addParam(BS, Ty, "base class");
    return true;
  }

  using SyclKernelFieldHandler::handleUnionType;
  bool handleUnionType(FieldDecl *FD, QualType FieldTy) final {
    return handleScalarType(FD, FieldTy);
  }

  void handleSyclKernelHandlerType() {
    addParam(DC.getParamVarDeclsForCurrentField()[0]->getType(),
             "SYCL2020 specialization constant");
  }
};

static bool isESIMDKernelType(CXXMethodDecl *CallOperator) {
  return (CallOperator != nullptr) && CallOperator->hasAttr<SYCLSimdAttr>();
}

class SyclKernelBodyCreator : public SyclKernelFieldHandler {
  bool UseTopLevelKernelObj;
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
  std::optional<InitializedEntity> VarEntity;
  llvm::SmallVector<Expr *, 16> MemberExprBases;
  llvm::SmallVector<Expr *, 16> ArrayParamBases;
  FunctionDecl *KernelCallerFunc;
  SourceLocation KernelCallerSrcLoc; // KernelCallerFunc source location.
  // Contains a count of how many containers we're in.  This is used by the
  // pointer-struct-wrapping code to ensure that we don't try to wrap
  // top-level pointers.
  uint64_t StructDepth = 0;
  VarDecl *KernelHandlerClone = nullptr;
  bool IsESIMD = false;
  CXXMethodDecl *CallOperator = nullptr;

  Stmt *replaceWithLocalClone(ParmVarDecl *OriginalParam, VarDecl *LocalClone,
                              Stmt *FunctionBody) {
    // DeclRefExpr with valid source location but with decl which is not marked
    // as used is invalid.
    LocalClone->setIsUsed();
    std::pair<DeclaratorDecl *, DeclaratorDecl *> MappingPair =
        std::make_pair(OriginalParam, LocalClone);
    KernelBodyTransform KBT(MappingPair, SemaSYCLRef.SemaRef);
    return KBT.TransformStmt(FunctionBody).get();
  }

  // Using the statements/init expressions that we've created, this generates
  // the kernel body compound stmt. CompoundStmt needs to know its number of
  // statements in advance to allocate it, so we cannot do this as we go along.
  CompoundStmt *createKernelBody() {
    // Push the Kernel function scope to ensure the scope isn't empty
    SemaSYCLRef.SemaRef.PushFunctionScope();

    if (!UseTopLevelKernelObj) {
      // Initialize kernel object local clone
      assert(CollectionInitExprs.size() == 1 &&
             "Should have been popped down to just the first one");
      KernelObjClone->setInit(CollectionInitExprs.back());
    }

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

    SourceLocation LL = NewBody ? NewBody->getBeginLoc() : SourceLocation();
    SourceLocation LR = NewBody ? NewBody->getEndLoc() : SourceLocation();

    return CompoundStmt::Create(SemaSYCLRef.getASTContext(), BodyStmts,
                                FPOptionsOverride(), LL, LR);
  }

  void annotateHierarchicalParallelismAPICalls() {
    // Is this a hierarchical parallelism kernel invocation?
    if (getKernelInvocationKind(KernelCallerFunc) != InvokeParallelForWorkGroup)
      return;

    // Mark kernel object with work-group scope attribute to avoid work-item
    // scope memory allocation.
    KernelObjClone->addAttr(SYCLScopeAttr::CreateImplicit(
        SemaSYCLRef.getASTContext(), SYCLScopeAttr::Level::WorkGroup));

    assert(CallOperator && "non callable object is passed as kernel obj");
    // Mark the function that it "works" in a work group scope:
    // NOTE: In case of wait_for the marker call itself is
    // marked with work item scope attribute, here  the '()' operator of the
    // object passed as parameter is marked. This is an optimization -
    // there are a lot of locals created at parallel_for_work_group
    // scope before calling the lambda - it is more efficient to have
    // all of them in the private address space rather then sharing via
    // the local AS. See parallel_for_work_group implementation in the
    // SYCL headers.
    if (!CallOperator->hasAttr<SYCLScopeAttr>()) {
      CallOperator->addAttr(SYCLScopeAttr::CreateImplicit(
          SemaSYCLRef.getASTContext(), SYCLScopeAttr::Level::WorkGroup));
      // Search and mark wait_for calls:
      MarkWIScopeFnVisitor MarkWIScope(SemaSYCLRef.getASTContext());
      MarkWIScope.TraverseDecl(CallOperator);
      // Now mark local variables declared in the PFWG lambda with work group
      // scope attribute
      addScopeAttrToLocalVars(*CallOperator);
    }
  }

  // Creates a DeclRefExpr to the ParmVar that represents the current field.
  Expr *createParamReferenceExpr() {
    ParmVarDecl *KernelParameter =
        DeclCreator.getParamVarDeclsForCurrentField()[0];

    QualType ParamType = KernelParameter->getOriginalType();
    Expr *DRE = SemaSYCLRef.SemaRef.BuildDeclRefExpr(
        KernelParameter, ParamType, VK_LValue, KernelCallerSrcLoc);
    return DRE;
  }

  // Creates a DeclRefExpr to the ParmVar that represents the current pointer
  // field.
  Expr *createPointerParamReferenceExpr(QualType PointerTy, bool Wrapped) {
    ParmVarDecl *KernelParameter =
        DeclCreator.getParamVarDeclsForCurrentField()[0];

    QualType ParamType = KernelParameter->getOriginalType();
    Expr *DRE = SemaSYCLRef.SemaRef.BuildDeclRefExpr(
        KernelParameter, ParamType, VK_LValue, KernelCallerSrcLoc);

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

    DRE = ImplicitCastExpr::Create(SemaSYCLRef.getASTContext(), ParamType,
                                   CK_LValueToRValue, DRE, /*BasePath=*/nullptr,
                                   VK_PRValue, FPOptionsOverride());

    if (PointerTy->getPointeeType().getAddressSpace() !=
        ParamType->getPointeeType().getAddressSpace())
      DRE = ImplicitCastExpr::Create(SemaSYCLRef.getASTContext(), PointerTy,
                                     CK_AddressSpaceConversion, DRE, nullptr,
                                     VK_PRValue, FPOptionsOverride());

    return DRE;
  }

  Expr *createSimpleArrayParamReferenceExpr(QualType) {
    ParmVarDecl *KernelParameter =
        DeclCreator.getParamVarDeclsForCurrentField()[0];
    QualType ParamType = KernelParameter->getOriginalType();
    Expr *DRE = SemaSYCLRef.SemaRef.BuildDeclRefExpr(
        KernelParameter, ParamType, VK_LValue, KernelCallerSrcLoc);

    // Unwrap the array.
    CXXRecordDecl *WrapperStruct = ParamType->getAsCXXRecordDecl();
    FieldDecl *ArrayField = *(WrapperStruct->field_begin());
    return buildMemberExpr(DRE, ArrayField);
  }

  // Creates an initialized entity for a field/item. In the case where this is a
  // field, returns a normal member initializer, if we're in a sub-array of a MD
  // array, returns an element initializer.
  InitializedEntity getFieldEntity(FieldDecl *FD, QualType Ty) {
    if (isArrayElement(FD, Ty))
      return InitializedEntity::InitializeElement(SemaSYCLRef.getASTContext(),
                                                  ArrayInfos.back().second,
                                                  ArrayInfos.back().first);
    return InitializedEntity::InitializeMember(FD, &VarEntity.value());
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

  void addFieldInit(FieldDecl *, QualType, MultiExprArg ParamRef,
                    InitializationKind InitKind, InitializedEntity Entity) {
    InitializationSequence InitSeq(SemaSYCLRef.SemaRef, Entity, InitKind,
                                   ParamRef);
    ExprResult Init =
        InitSeq.Perform(SemaSYCLRef.SemaRef, Entity, InitKind, ParamRef);

    InitListExpr *ParentILE = CollectionInitExprs.back();
    ParentILE->updateInit(SemaSYCLRef.getASTContext(), ParentILE->getNumInits(),
                          Init.get());
  }

  void addBaseInit(const CXXBaseSpecifier &BS, QualType,
                   InitializationKind InitKind) {
    InitializedEntity Entity = InitializedEntity::InitializeBase(
        SemaSYCLRef.getASTContext(), &BS, /*IsInheritedVirtualBase*/ false,
        &VarEntity.value());
    InitializationSequence InitSeq(SemaSYCLRef.SemaRef, Entity, InitKind,
                                   std::nullopt);
    ExprResult Init =
        InitSeq.Perform(SemaSYCLRef.SemaRef, Entity, InitKind, std::nullopt);

    InitListExpr *ParentILE = CollectionInitExprs.back();
    ParentILE->updateInit(SemaSYCLRef.getASTContext(), ParentILE->getNumInits(),
                          Init.get());
  }

  void addBaseInit(const CXXBaseSpecifier &BS, QualType,
                   InitializationKind InitKind, MultiExprArg Args) {
    InitializedEntity Entity = InitializedEntity::InitializeBase(
        SemaSYCLRef.getASTContext(), &BS, /*IsInheritedVirtualBase*/ false,
        &VarEntity.value());
    InitializationSequence InitSeq(SemaSYCLRef.SemaRef, Entity, InitKind, Args);
    ExprResult Init =
        InitSeq.Perform(SemaSYCLRef.SemaRef, Entity, InitKind, Args);

    InitListExpr *ParentILE = CollectionInitExprs.back();
    ParentILE->updateInit(SemaSYCLRef.getASTContext(), ParentILE->getNumInits(),
                          Init.get());
  }

  void addSimpleBaseInit(const CXXBaseSpecifier &BS, QualType) {
    InitializationKind InitKind =
        InitializationKind::CreateCopy(KernelCallerSrcLoc, KernelCallerSrcLoc);

    InitializedEntity Entity = InitializedEntity::InitializeBase(
        SemaSYCLRef.getASTContext(), &BS, /*IsInheritedVirtualBase*/ false,
        &VarEntity.value());

    Expr *ParamRef = createParamReferenceExpr();
    InitializationSequence InitSeq(SemaSYCLRef.SemaRef, Entity, InitKind,
                                   ParamRef);
    ExprResult Init =
        InitSeq.Perform(SemaSYCLRef.SemaRef, Entity, InitKind, ParamRef);

    InitListExpr *ParentILE = CollectionInitExprs.back();
    ParentILE->updateInit(SemaSYCLRef.getASTContext(), ParentILE->getNumInits(),
                          Init.get());
  }

  // Adds an initializer that handles a simple initialization of a field.
  void addSimpleFieldInit(FieldDecl *FD, QualType Ty) {
    Expr *ParamRef = createParamReferenceExpr();
    addFieldInit(FD, Ty, ParamRef);
  }

  Expr *createGetAddressOf(Expr *E) {
    return UnaryOperator::Create(
        SemaSYCLRef.getASTContext(), E, UO_AddrOf,
        SemaSYCLRef.getASTContext().getPointerType(E->getType()), VK_PRValue,
        OK_Ordinary, KernelCallerSrcLoc, false,
        SemaSYCLRef.SemaRef.CurFPFeatureOverrides());
  }

  Expr *createDerefOp(Expr *E) {
    return UnaryOperator::Create(SemaSYCLRef.getASTContext(), E, UO_Deref,
                                 E->getType()->getPointeeType(), VK_LValue,
                                 OK_Ordinary, KernelCallerSrcLoc, false,
                                 SemaSYCLRef.SemaRef.CurFPFeatureOverrides());
  }

  Expr *createReinterpretCastExpr(Expr *E, QualType To) {
    return CXXReinterpretCastExpr::Create(
        SemaSYCLRef.getASTContext(), To, VK_PRValue, CK_BitCast, E,
        /*Path=*/nullptr,
        SemaSYCLRef.getASTContext().getTrivialTypeSourceInfo(To),
        SourceLocation(), SourceLocation(), SourceRange());
  }

  void handleGeneratedType(FieldDecl *FD, QualType Ty) {
    // Equivalent of the following code is generated here:
    // void ocl_kernel(__generated_type GT) {
    //   Kernel KernelObjClone { *(reinterpret_cast<UsersType*>(&GT)) };
    // }

    Expr *RCE = createReinterpretCastExpr(
        createGetAddressOf(createParamReferenceExpr()),
        SemaSYCLRef.getASTContext().getPointerType(Ty));
    Expr *Initializer = createDerefOp(RCE);
    addFieldInit(FD, Ty, Initializer);
  }

  void handleGeneratedType(const CXXRecordDecl *, const CXXBaseSpecifier &BS,
                           QualType Ty) {
    // Equivalent of the following code is generated here:
    // void ocl_kernel(__generated_type GT) {
    //   Kernel KernelObjClone { *(reinterpret_cast<UsersType*>(&GT)) };
    // }
    Expr *RCE = createReinterpretCastExpr(
        createGetAddressOf(createParamReferenceExpr()),
        SemaSYCLRef.getASTContext().getPointerType(Ty));
    Expr *Initializer = createDerefOp(RCE);
    InitializationKind InitKind =
        InitializationKind::CreateCopy(KernelCallerSrcLoc, KernelCallerSrcLoc);
    addBaseInit(BS, Ty, InitKind, Initializer);
  }

  MemberExpr *buildMemberExpr(Expr *Base, ValueDecl *Member) {
    DeclAccessPair MemberDAP = DeclAccessPair::make(Member, AS_none);
    MemberExpr *Result = SemaSYCLRef.SemaRef.BuildMemberExpr(
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
      ParamDREs[I] = SemaSYCLRef.SemaRef.BuildDeclRefExpr(
          KernelParameters[I], ParamType, VK_LValue, KernelCallerSrcLoc);
    }

    MemberExpr *MethodME = buildMemberExpr(MemberExprBases.back(), Method);

    QualType ResultTy = Method->getReturnType();
    ExprValueKind VK = Expr::getValueKindForType(ResultTy);
    ResultTy = ResultTy.getNonLValueExprType(SemaSYCLRef.getASTContext());
    llvm::SmallVector<Expr *, 4> ParamStmts;
    const auto *Proto = cast<FunctionProtoType>(Method->getType());
    SemaSYCLRef.SemaRef.GatherArgumentsForCall(KernelCallerSrcLoc, Method,
                                               Proto, 0, ParamDREs, ParamStmts);
    // [kernel_obj or wrapper object].accessor.__init(_ValueType*,
    // range<int>, range<int>, id<int>)
    AddTo.push_back(CXXMemberCallExpr::Create(
        SemaSYCLRef.getASTContext(), MethodME, ParamStmts, ResultTy, VK,
        KernelCallerSrcLoc, FPOptionsOverride()));
  }

  // Creates an empty InitListExpr of the correct number of child-inits
  // of this to append into.
  void addCollectionInitListExpr(const CXXRecordDecl *RD) {
    const ASTRecordLayout &Info =
        SemaSYCLRef.getASTContext().getASTRecordLayout(RD);
    uint64_t NumInitExprs = Info.getFieldCount() + RD->getNumBases();
    addCollectionInitListExpr(QualType(RD->getTypeForDecl(), 0), NumInitExprs);
  }

  InitListExpr *createInitListExpr(const CXXRecordDecl *RD) {
    const ASTRecordLayout &Info =
        SemaSYCLRef.getASTContext().getASTRecordLayout(RD);
    uint64_t NumInitExprs = Info.getFieldCount() + RD->getNumBases();
    return createInitListExpr(QualType(RD->getTypeForDecl(), 0), NumInitExprs);
  }

  InitListExpr *createInitListExpr(QualType InitTy, uint64_t NumChildInits) {
    InitListExpr *ILE = new (SemaSYCLRef.getASTContext())
        InitListExpr(SemaSYCLRef.getASTContext(), KernelCallerSrcLoc, {},
                     KernelCallerSrcLoc);
    ILE->reserveInits(SemaSYCLRef.getASTContext(), NumChildInits);
    ILE->setType(InitTy);

    return ILE;
  }

  // Create an empty InitListExpr of the type/size for the rest of the visitor
  // to append into.
  void addCollectionInitListExpr(QualType InitTy, uint64_t NumChildInits) {

    InitListExpr *ILE = createInitListExpr(InitTy, NumChildInits);
    InitListExpr *ParentILE = CollectionInitExprs.back();
    ParentILE->updateInit(SemaSYCLRef.getASTContext(), ParentILE->getNumInits(),
                          ILE);

    CollectionInitExprs.push_back(ILE);
  }

  static VarDecl *createKernelObjClone(ASTContext &Ctx, DeclContext *DC,
                                       const CXXRecordDecl *KernelObj) {
    TypeSourceInfo *TSInfo =
        KernelObj->isLambda() ? KernelObj->getLambdaTypeInfo() : nullptr;
    IdentifierInfo *Ident = KernelObj->getIdentifier();
    if (!Ident)
      Ident = &Ctx.Idents.get("__SYCLKernel");

    VarDecl *VD = VarDecl::Create(
        Ctx, DC, KernelObj->getLocation(), KernelObj->getLocation(), Ident,
        QualType(KernelObj->getTypeForDecl(), 0), TSInfo, SC_None);
    return VD;
  }

  const llvm::StringLiteral getInitMethodName() const {
    return IsESIMD ? InitESIMDMethodName : InitMethodName;
  }

  // Default inits the type, then calls the init-method in the body.
  // A type may not have a public default constructor as per its spec so
  // typically if this is the case the default constructor will be private and
  // in such cases we must manually override the access specifier from private
  // to public just for the duration of this default initialization.
  // TODO: Revisit this approach once https://github.com/intel/llvm/issues/16061
  // is closed.
  bool handleSpecialType(FieldDecl *FD, QualType Ty) {
    const auto *RecordDecl = Ty->getAsCXXRecordDecl();
    AccessSpecifier DefaultConstructorAccess;
    auto DefaultConstructor =
        std::find_if(RecordDecl->ctor_begin(), RecordDecl->ctor_end(),
                     [](auto it) { return it->isDefaultConstructor(); });
    DefaultConstructorAccess = DefaultConstructor->getAccess();
    DefaultConstructor->setAccess(AS_public);

    addFieldInit(FD, Ty, std::nullopt,
                 InitializationKind::CreateDefault(KernelCallerSrcLoc));
    DefaultConstructor->setAccess(DefaultConstructorAccess);
    addFieldMemberExpr(FD, Ty);

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
    const auto *BaseRecordDecl = BS.getType()->getAsCXXRecordDecl();
    AccessSpecifier DefaultConstructorAccess;
    auto DefaultConstructor =
        std::find_if(BaseRecordDecl->ctor_begin(), BaseRecordDecl->ctor_end(),
                     [](auto it) { return it->isDefaultConstructor(); });
    DefaultConstructorAccess = DefaultConstructor->getAccess();
    DefaultConstructor->setAccess(AS_public);

    addBaseInit(BS, Ty, InitializationKind::CreateDefault(KernelCallerSrcLoc));
    DefaultConstructor->setAccess(DefaultConstructorAccess);
    createSpecialMethodCall(BaseRecordDecl, getInitMethodName(), BodyStmts);
    return true;
  }

  // Generate __init call for kernel handler argument
  void handleSpecialType(QualType KernelHandlerTy) {
    DeclRefExpr *KernelHandlerCloneRef = DeclRefExpr::Create(
        SemaSYCLRef.getASTContext(), NestedNameSpecifierLoc(),
        KernelCallerSrcLoc, KernelHandlerClone, false, DeclarationNameInfo(),
        KernelHandlerTy, VK_LValue);
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
    InitializationSequence InitSeq(SemaSYCLRef.SemaRef, VarEntity, InitKind,
                                   std::nullopt);
    ExprResult Init =
        InitSeq.Perform(SemaSYCLRef.SemaRef, VarEntity, InitKind, std::nullopt);
    KernelHandlerClone->setInit(
        SemaSYCLRef.SemaRef.MaybeCreateExprWithCleanups(Init.get()));
    KernelHandlerClone->setInitStyle(VarDecl::CallInit);
  }

  Expr *createArraySubscriptExpr(uint64_t Index, Expr *ArrayRef) {
    QualType SizeT = SemaSYCLRef.getASTContext().getSizeType();
    llvm::APInt IndexVal{
        static_cast<unsigned>(SemaSYCLRef.getASTContext().getTypeSize(SizeT)),
        Index, SizeT->isSignedIntegerType()};
    auto IndexLiteral = IntegerLiteral::Create(
        SemaSYCLRef.getASTContext(), IndexVal, SizeT, KernelCallerSrcLoc);
    ExprResult IndexExpr = SemaSYCLRef.SemaRef.CreateBuiltinArraySubscriptExpr(
        ArrayRef, KernelCallerSrcLoc, IndexLiteral, KernelCallerSrcLoc);
    assert(!IndexExpr.isInvalid());
    return IndexExpr.get();
  }

  void addSimpleArrayInit(FieldDecl *FD, QualType FieldTy) {
    Expr *ArrayRef = createSimpleArrayParamReferenceExpr(FieldTy);
    InitializationKind InitKind = InitializationKind::CreateDirect({}, {}, {});

    InitializedEntity Entity = InitializedEntity::InitializeMember(
        FD, &VarEntity.value(), /*Implicit*/ true);

    addFieldInit(FD, FieldTy, ArrayRef, InitKind, Entity);
  }

  void addArrayElementInit(FieldDecl *FD, QualType T) {
    Expr *RCE = createReinterpretCastExpr(
        createGetAddressOf(ArrayParamBases.pop_back_val()),
        SemaSYCLRef.getASTContext().getPointerType(T));
    Expr *Initializer = createDerefOp(RCE);
    addFieldInit(FD, T, Initializer);
  }

  // This function is recursive in order to handle
  // multi-dimensional arrays. If the array element is
  // an array, it implies that the array is multi-dimensional.
  // We continue recursion till we reach a non-array element to
  // generate required array subscript expressions.
  void createArrayInit(FieldDecl *FD, QualType T) {
    const ConstantArrayType *CAT =
        SemaSYCLRef.getASTContext().getAsConstantArrayType(T);

    if (!CAT) {
      addArrayElementInit(FD, T);
      return;
    }

    QualType ET = CAT->getElementType();
    uint64_t ElemCount = CAT->getSize().getZExtValue();
    enterArray(FD, T, ET);

    for (uint64_t Index = 0; Index < ElemCount; ++Index) {
      ArrayInfos.back().second = Index;
      Expr *ArraySubscriptExpr =
          createArraySubscriptExpr(Index, ArrayParamBases.back());
      ArrayParamBases.push_back(ArraySubscriptExpr);
      createArrayInit(FD, ET);
    }

    leaveArray(FD, T, ET);
  }

  // This function is used to create initializers for a top
  // level array which contains pointers. The openCl kernel
  // parameter for this array will be a wrapper class
  // which contains the generated type. This function generates
  // code equivalent to:
  // void ocl_kernel(__wrapper_class WrappedGT) {
  //   Kernel KernelObjClone {
  //   *reinterpret_cast<UserArrayET*>(&WrappedGT.GeneratedArr[0]),
  //                           *reinterpret_cast<UserArrayET*>(&WrappedGT.GeneratedArr[1]),
  //                           *reinterpret_cast<UserArrayET*>(&WrappedGT.GeneratedArr[2])
  //                         };
  // }
  void handleGeneratedArrayType(FieldDecl *FD, QualType FieldTy) {
    ArrayParamBases.push_back(createSimpleArrayParamReferenceExpr(FieldTy));
    createArrayInit(FD, FieldTy);
  }

public:
  static constexpr const bool VisitInsideSimpleContainers = false;
  SyclKernelBodyCreator(SemaSYCL &S, SyclKernelDeclCreator &DC,
                        const CXXRecordDecl *KernelObj,
                        FunctionDecl *KernelCallerFunc, bool IsSIMDKernel,
                        CXXMethodDecl *CallOperator)
      : SyclKernelFieldHandler(S),
        UseTopLevelKernelObj(KernelObjVisitor::useTopLevelKernelObj(KernelObj)),
        DeclCreator(DC),
        KernelObjClone(UseTopLevelKernelObj
                           ? nullptr
                           : createKernelObjClone(S.getASTContext(),
                                                  DC.getKernelDecl(),
                                                  KernelObj)),
        VarEntity(), KernelCallerFunc(KernelCallerFunc),
        KernelCallerSrcLoc(KernelCallerFunc->getLocation()),
        IsESIMD(IsSIMDKernel), CallOperator(CallOperator) {
    if (!UseTopLevelKernelObj) {
      VarEntity.emplace(InitializedEntity::InitializeVariable(KernelObjClone));
      Stmt *DS = new (S.getASTContext()) DeclStmt(
          DeclGroupRef(KernelObjClone), KernelCallerSrcLoc, KernelCallerSrcLoc);
      BodyStmts.push_back(DS);
      CollectionInitExprs.push_back(createInitListExpr(KernelObj));
      DeclRefExpr *KernelObjCloneRef = DeclRefExpr::Create(
          S.getASTContext(), NestedNameSpecifierLoc(), KernelCallerSrcLoc,
          KernelObjClone, false, DeclarationNameInfo(),
          QualType(KernelObj->getTypeForDecl(), 0), VK_LValue);
      MemberExprBases.push_back(KernelObjCloneRef);
    }
  }

  ~SyclKernelBodyCreator() {
    annotateHierarchicalParallelismAPICalls();
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

  bool handlePointerType(FieldDecl *FD, QualType FieldTy) final {
    Expr *PointerRef =
        createPointerParamReferenceExpr(FieldTy, StructDepth != 0);
    addFieldInit(FD, FieldTy, PointerRef);
    return true;
  }

  bool handleSimpleArrayType(FieldDecl *FD, QualType FieldTy) final {
    if (FD->hasAttr<SYCLGenerateNewTypeAttr>())
      handleGeneratedArrayType(FD, FieldTy);
    else
      addSimpleArrayInit(FD, FieldTy);
    return true;
  }

  bool handleTopLevelStruct(const CXXRecordDecl *, QualType) final {
    // As the functor is passed as a whole, use the param as the vardecl
    // otherwise used as the clone.
    KernelObjClone = DeclCreator.getParamVarDeclsForCurrentField()[0];
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, FieldDecl *FD,
                             QualType Ty) final {
    CXXRecordDecl *RD = Ty->getAsCXXRecordDecl();
    assert(RD && "Type must be a C++ record type");
    if (RD->hasAttr<SYCLGenerateNewTypeAttr>())
      handleGeneratedType(FD, Ty);
    else
      addSimpleFieldInit(FD, Ty);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *RD,
                             const CXXBaseSpecifier &BS, QualType Ty) final {
    CXXRecordDecl *BaseDecl = Ty->getAsCXXRecordDecl();
    assert(BaseDecl && "Type must be a C++ record type");
    if (BaseDecl->hasAttr<SYCLGenerateNewTypeAttr>())
      handleGeneratedType(RD, BS, Ty);
    else
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
    createKernelHandlerClone(SemaSYCLRef.getASTContext(),
                             DeclCreator.getKernelDecl(), KernelHandlerArg);

    // Add declaration statement to openCL kernel body
    Stmt *DS = new (SemaSYCLRef.getASTContext())
        DeclStmt(DeclGroupRef(KernelHandlerClone), KernelCallerSrcLoc,
                 KernelCallerSrcLoc);
    BodyStmts.push_back(DS);

    // Generate
    // KernelHandlerClone.__init_specialization_constants_buffer(specialization_constants_buffer)
    // call if target does not have native support for specialization constants.
    // Here, specialization_constants_buffer is the compiler generated kernel
    // argument of type char*.
    if (!isDefaultSPIRArch(SemaSYCLRef.getASTContext()))
      handleSpecialType(KernelHandlerArg->getType());
  }

  bool enterStruct(const CXXRecordDecl *, FieldDecl *FD, QualType Ty) final {
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
    SemaSYCLRef.SemaRef.CheckDerivedToBaseConversion(
        DerivedTy, BaseTy, KernelCallerSrcLoc, SourceRange(), &BasePath,
        /*IgnoreBaseAccess*/ true);
    auto Cast = ImplicitCastExpr::Create(
        SemaSYCLRef.getASTContext(), BaseTy, CK_DerivedToBase,
        MemberExprBases.back(),
        /* CXXCastPath=*/&BasePath, VK_LValue, FPOptionsOverride());
    MemberExprBases.push_back(Cast);
    addCollectionInitListExpr(BaseTy->getAsCXXRecordDecl());
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, const CXXBaseSpecifier &,
                   QualType) final {
    --StructDepth;
    MemberExprBases.pop_back();
    CollectionInitExprs.pop_back();
    return true;
  }

  bool enterArray(FieldDecl *FD, QualType ArrayType, QualType) final {
    const ConstantArrayType *CAT =
        SemaSYCLRef.getASTContext().getAsConstantArrayType(ArrayType);
    assert(CAT && "Should only be called on constant-size array.");
    uint64_t ArraySize = CAT->getSize().getZExtValue();
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

    MemberExprBases.push_back(
        createArraySubscriptExpr(Index, MemberExprBases.back()));
    return true;
  }

  bool leaveArray(FieldDecl *FD, QualType ArrayType, QualType) final {
    CollectionInitExprs.pop_back();
    ArrayInfos.pop_back();

    // Remove the IndexExpr.
    if (!FD->hasAttr<SYCLGenerateNewTypeAttr>())
      MemberExprBases.pop_back();
    else
      ArrayParamBases.pop_back();

    // Remove the field access expr as well.
    removeFieldMemberExpr(FD, ArrayType);
    return true;
  }
  using SyclKernelFieldHandler::enterArray;
  using SyclKernelFieldHandler::enterStruct;
  using SyclKernelFieldHandler::handleNonDecompStruct;
  using SyclKernelFieldHandler::handlePointerType;
  using SyclKernelFieldHandler::handleScalarType;
  using SyclKernelFieldHandler::handleSyclSpecialType;
  using SyclKernelFieldHandler::handleUnionType;
  using SyclKernelFieldHandler::leaveArray;
  using SyclKernelFieldHandler::leaveStruct;
};

class FreeFunctionKernelBodyCreator : public SyclKernelFieldHandler {
  SyclKernelDeclCreator &DeclCreator;
  llvm::SmallVector<Stmt *, 16> BodyStmts;
  FunctionDecl *FreeFunc = nullptr;
  SourceLocation FreeFunctionSrcLoc; // Free function source location.
  llvm::SmallVector<Expr *, 8> ArgExprs;

  // Creates a DeclRefExpr to the ParmVar that represents the current free
  // function parameter.
  Expr *createParamReferenceExpr() {
    ParmVarDecl *FreeFunctionParameter =
        DeclCreator.getParamVarDeclsForCurrentField()[0];

    QualType FreeFunctionParamType = FreeFunctionParameter->getOriginalType();
    Expr *DRE = SemaSYCLRef.SemaRef.BuildDeclRefExpr(
        FreeFunctionParameter, FreeFunctionParamType, VK_LValue,
        FreeFunctionSrcLoc);
    DRE = SemaSYCLRef.SemaRef.DefaultLvalueConversion(DRE).get();
    return DRE;
  }

  // Creates a DeclRefExpr to the ParmVar that represents the current pointer
  // parameter.
  Expr *createPointerParamReferenceExpr(QualType PointerTy) {
    ParmVarDecl *FreeFunctionParameter =
        DeclCreator.getParamVarDeclsForCurrentField()[0];

    QualType FreeFunctionParamType = FreeFunctionParameter->getOriginalType();
    Expr *DRE = SemaSYCLRef.SemaRef.BuildDeclRefExpr(
        FreeFunctionParameter, FreeFunctionParamType, VK_LValue,
        FreeFunctionSrcLoc);
    DRE = SemaSYCLRef.SemaRef.DefaultLvalueConversion(DRE).get();

    if (PointerTy->getPointeeType().getAddressSpace() !=
        FreeFunctionParamType->getPointeeType().getAddressSpace())
      DRE = ImplicitCastExpr::Create(SemaSYCLRef.getASTContext(), PointerTy,
                                     CK_AddressSpaceConversion, DRE, nullptr,
                                     VK_PRValue, FPOptionsOverride());
    return DRE;
  }

  Expr *createGetAddressOf(Expr *E) {
    return UnaryOperator::Create(
        SemaSYCLRef.getASTContext(), E, UO_AddrOf,
        SemaSYCLRef.getASTContext().getPointerType(E->getType()), VK_PRValue,
        OK_Ordinary, SourceLocation(), false,
        SemaSYCLRef.SemaRef.CurFPFeatureOverrides());
  }

  Expr *createDerefOp(Expr *E) {
    return UnaryOperator::Create(SemaSYCLRef.getASTContext(), E, UO_Deref,
                                 E->getType()->getPointeeType(), VK_LValue,
                                 OK_Ordinary, SourceLocation(), false,
                                 SemaSYCLRef.SemaRef.CurFPFeatureOverrides());
  }

  Expr *createReinterpretCastExpr(Expr *E, QualType To) {
    return CXXReinterpretCastExpr::Create(
        SemaSYCLRef.getASTContext(), To, VK_PRValue, CK_BitCast, E,
        /*Path=*/nullptr,
        SemaSYCLRef.getASTContext().getTrivialTypeSourceInfo(To),
        SourceLocation(), SourceLocation(), SourceRange());
  }

  Expr *createCopyInitExpr(ParmVarDecl *OrigFunctionParameter) {
    Expr *DRE = createParamReferenceExpr();

    assert(OrigFunctionParameter && "no parameter?");

    CXXRecordDecl *RD = OrigFunctionParameter->getType()->getAsCXXRecordDecl();
    InitializedEntity Entity = InitializedEntity::InitializeParameter(
        SemaSYCLRef.getASTContext(), OrigFunctionParameter);

    if (RD->hasAttr<SYCLGenerateNewTypeAttr>()) {
      DRE = createReinterpretCastExpr(
          createGetAddressOf(DRE), SemaSYCLRef.getASTContext().getPointerType(
                                       OrigFunctionParameter->getType()));
      DRE = createDerefOp(DRE);
    }

    ExprResult ArgE = SemaSYCLRef.SemaRef.PerformCopyInitialization(
        Entity, SourceLocation(), DRE, false, false);
    return ArgE.getAs<Expr>();
  }

  // For a free function such as:
  // void f(int i, int* p, struct Simple S) { ... }
  //
  // Keep the function as-is for the version callable from device code.
  // void f(int i, int *p, struct Simple S) { ... }
  //
  // For the host-callable kernel function generate this:
  // void __sycl_kernel_f(int __arg_i, int* __arg_p, struct Simple __arg_S)
  // {
  //   f(__arg_i, __arg_p, __arg_S);
  // }
  CompoundStmt *createFreeFunctionKernelBody() {
    SemaSYCLRef.SemaRef.PushFunctionScope();
    Expr *Fn = SemaSYCLRef.SemaRef.BuildDeclRefExpr(
        FreeFunc, FreeFunc->getType(), VK_LValue, FreeFunctionSrcLoc);
    ASTContext &Context = SemaSYCLRef.getASTContext();
    QualType ResultTy = FreeFunc->getReturnType();
    ExprValueKind VK = Expr::getValueKindForType(ResultTy);
    ResultTy = ResultTy.getNonLValueExprType(Context);
    Fn = ImplicitCastExpr::Create(Context,
                                  Context.getPointerType(FreeFunc->getType()),
                                  CK_FunctionToPointerDecay, Fn, nullptr,
                                  VK_PRValue, FPOptionsOverride());
    auto CallExpr = CallExpr::Create(Context, Fn, ArgExprs, ResultTy, VK,
                                     FreeFunctionSrcLoc, FPOptionsOverride());
    BodyStmts.push_back(CallExpr);
    return CompoundStmt::Create(Context, BodyStmts, FPOptionsOverride(), {},
                                {});
  }

  MemberExpr *buildMemberExpr(Expr *Base, ValueDecl *Member) {
    DeclAccessPair MemberDAP = DeclAccessPair::make(Member, AS_none);
    MemberExpr *Result = SemaSYCLRef.SemaRef.BuildMemberExpr(
        Base, /*IsArrow */ false, FreeFunctionSrcLoc, NestedNameSpecifierLoc(),
        FreeFunctionSrcLoc, Member, MemberDAP,
        /*HadMultipleCandidates*/ false,
        DeclarationNameInfo(Member->getDeclName(), FreeFunctionSrcLoc),
        Member->getType(), VK_LValue, OK_Ordinary);
    return Result;
  }

  void createSpecialMethodCall(const CXXRecordDecl *RD, StringRef MethodName,
                               Expr *MemberBaseExpr,
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
      ParamDREs[I] = SemaSYCLRef.SemaRef.BuildDeclRefExpr(
          KernelParameters[I], ParamType, VK_LValue, FreeFunctionSrcLoc);
    }
    MemberExpr *MethodME = buildMemberExpr(MemberBaseExpr, Method);
    QualType ResultTy = Method->getReturnType();
    ExprValueKind VK = Expr::getValueKindForType(ResultTy);
    ResultTy = ResultTy.getNonLValueExprType(SemaSYCLRef.getASTContext());
    llvm::SmallVector<Expr *, 4> ParamStmts;
    const auto *Proto = cast<FunctionProtoType>(Method->getType());
    SemaSYCLRef.SemaRef.GatherArgumentsForCall(FreeFunctionSrcLoc, Method,
                                               Proto, 0, ParamDREs, ParamStmts);
    AddTo.push_back(CXXMemberCallExpr::Create(
        SemaSYCLRef.getASTContext(), MethodME, ParamStmts, ResultTy, VK,
        FreeFunctionSrcLoc, FPOptionsOverride()));
  }

public:
  static constexpr const bool VisitInsideSimpleContainers = false;

  FreeFunctionKernelBodyCreator(SemaSYCL &S, SyclKernelDeclCreator &DC,
                                FunctionDecl *FF)
      : SyclKernelFieldHandler(S), DeclCreator(DC), FreeFunc(FF),
        FreeFunctionSrcLoc(FF->getLocation()) {}

  ~FreeFunctionKernelBodyCreator() {
    CompoundStmt *KernelBody = createFreeFunctionKernelBody();
    DeclCreator.setBody(KernelBody);
  }

  bool handleSyclSpecialType(FieldDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  // Default inits the type, then calls the init-method in the body.
  // A type may not have a public default constructor as per its spec so
  // typically if this is the case the default constructor will be private and
  // in such cases we must manually override the access specifier from private
  // to public just for the duration of this default initialization.
  // TODO: Revisit this approach once https://github.com/intel/llvm/issues/16061
  // is closed.
  bool handleSyclSpecialType(ParmVarDecl *PD, QualType ParamTy) final {
    // The code produced looks like this in the case of a work group memory
    // parameter:
    // void auto_generated_kernel(__local int * arg) {
    //    work_group_memory wgm;
    //    wgm.__init(arg);
    //    user_kernel(some arguments..., wgm, some arguments...);
    // }
    const auto *RecordDecl = ParamTy->getAsCXXRecordDecl();
    AccessSpecifier DefaultConstructorAccess;
    auto DefaultConstructor =
        std::find_if(RecordDecl->ctor_begin(), RecordDecl->ctor_end(),
                     [](auto it) { return it->isDefaultConstructor(); });
    DefaultConstructorAccess = DefaultConstructor->getAccess();
    DefaultConstructor->setAccess(AS_public);

    ASTContext &Ctx = SemaSYCLRef.SemaRef.getASTContext();
    VarDecl *SpecialObjectClone =
        VarDecl::Create(Ctx, DeclCreator.getKernelDecl(), FreeFunctionSrcLoc,
                        FreeFunctionSrcLoc, PD->getIdentifier(), ParamTy,
                        Ctx.getTrivialTypeSourceInfo(ParamTy), SC_None);
    InitializedEntity VarEntity =
        InitializedEntity::InitializeVariable(SpecialObjectClone);
    InitializationKind InitKind =
        InitializationKind::CreateDefault(FreeFunctionSrcLoc);
    InitializationSequence InitSeq(SemaSYCLRef.SemaRef, VarEntity, InitKind,
                                   std::nullopt);
    ExprResult Init =
        InitSeq.Perform(SemaSYCLRef.SemaRef, VarEntity, InitKind, std::nullopt);
    SpecialObjectClone->setInit(
        SemaSYCLRef.SemaRef.MaybeCreateExprWithCleanups(Init.get()));
    SpecialObjectClone->setInitStyle(VarDecl::CallInit);
    DefaultConstructor->setAccess(DefaultConstructorAccess);

    Stmt *DS = new (SemaSYCLRef.getASTContext())
        DeclStmt(DeclGroupRef(SpecialObjectClone), FreeFunctionSrcLoc,
                 FreeFunctionSrcLoc);
    BodyStmts.push_back(DS);
    Expr *MemberBaseExpr = SemaSYCLRef.SemaRef.BuildDeclRefExpr(
        SpecialObjectClone, ParamTy, VK_PRValue, FreeFunctionSrcLoc);
    createSpecialMethodCall(RecordDecl, InitMethodName, MemberBaseExpr,
                            BodyStmts);
    ArgExprs.push_back(MemberBaseExpr);
    return true;
  }

  bool handleSyclSpecialType(const CXXRecordDecl *, const CXXBaseSpecifier &,
                             QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool handlePointerType(FieldDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool handlePointerType(ParmVarDecl *, QualType ParamTy) final {
    Expr *PointerRef = createPointerParamReferenceExpr(ParamTy);
    ArgExprs.push_back(PointerRef);
    return true;
  }

  bool handleSimpleArrayType(FieldDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, FieldDecl *,
                             QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, ParmVarDecl *PD,
                             QualType) final {
    Expr *TempCopy = createCopyInitExpr(PD);
    ArgExprs.push_back(TempCopy);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, const CXXBaseSpecifier &,
                             QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool handleScalarType(FieldDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool handleScalarType(ParmVarDecl *, QualType) final {
    Expr *ParamRef = createParamReferenceExpr();
    ArgExprs.push_back(ParamRef);
    return true;
  }

  bool handleUnionType(FieldDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool handleUnionType(ParmVarDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool enterStruct(const CXXRecordDecl *, FieldDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool enterStruct(const CXXRecordDecl *, ParmVarDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, FieldDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, ParmVarDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool enterStruct(const CXXRecordDecl *, const CXXBaseSpecifier &,
                   QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, const CXXBaseSpecifier &,
                   QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool enterArray(FieldDecl *, QualType, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool enterArray(ParmVarDecl *, QualType, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool leaveArray(FieldDecl *, QualType, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool leaveArray(ParmVarDecl *, QualType, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }
};

// Kernels are only the unnamed-lambda feature if the feature is enabled, AND
// the first template argument has been corrected by the library to match the
// functor type.
static bool IsSYCLUnnamedKernel(SemaSYCL &SemaSYCLRef, const FunctionDecl *FD) {
  if (!SemaSYCLRef.getLangOpts().SYCLUnnamedLambda)
    return false;

  QualType FunctorTy = GetSYCLKernelObjectType(FD);
  QualType TmplArgTy = calculateKernelNameType(FD);
  return SemaSYCLRef.getASTContext().hasSameType(FunctorTy, TmplArgTy);
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
               : SemaSYCLRef.getASTContext().getFieldOffset(FD) / 8;
  }
  // For free functions each parameter is stand-alone, so offsets within a
  // lambda/function object are not relevant. Therefore offsetOf will always be
  // 0.
  int64_t offsetOf(const ParmVarDecl *, QualType) const { return 0; }

  int64_t offsetOf(const CXXRecordDecl *RD, const CXXRecordDecl *Base) const {
    const ASTRecordLayout &Layout =
        SemaSYCLRef.getASTContext().getASTRecordLayout(RD);
    return Layout.getBaseClassOffset(Base).getQuantity();
  }

  void addParam(const FieldDecl *FD, QualType ArgTy,
                SYCLIntegrationHeader::kernel_param_kind_t Kind) {
    addParam(ArgTy, Kind, offsetOf(FD, ArgTy));
  }

  // For free functions we increment the current offset as each parameter is
  // added.
  void addParam(const ParmVarDecl *PD, QualType ParamTy,
                SYCLIntegrationHeader::kernel_param_kind_t Kind) {
    addParam(ParamTy, Kind, offsetOf(PD, ParamTy));
    CurOffset +=
        SemaSYCLRef.getASTContext().getTypeSizeInChars(ParamTy).getQuantity();
  }

  void addParam(QualType ParamTy,
                SYCLIntegrationHeader::kernel_param_kind_t Kind,
                uint64_t OffsetAdj) {
    uint64_t Size;
    Size =
        SemaSYCLRef.getASTContext().getTypeSizeInChars(ParamTy).getQuantity();
    Header.addParamDesc(Kind, static_cast<unsigned>(Size),
                        static_cast<unsigned>(CurOffset + OffsetAdj));
  }

public:
  static constexpr const bool VisitInsideSimpleContainers = false;
  SyclKernelIntHeaderCreator(bool IsESIMD, SemaSYCL &S,
                             SYCLIntegrationHeader &H,
                             const CXXRecordDecl *KernelObj, QualType NameType,
                             FunctionDecl *KernelFunc)
      : SyclKernelFieldHandler(S), Header(H) {

    // The header needs to access the kernel object size.
    int64_t ObjSize = SemaSYCLRef.getASTContext()
                          .getTypeSizeInChars(KernelObj->getTypeForDecl())
                          .getQuantity();
    Header.startKernel(KernelFunc, NameType, KernelObj->getLocation(), IsESIMD,
                       IsSYCLUnnamedKernel(S, KernelFunc), ObjSize);
  }

  SyclKernelIntHeaderCreator(SemaSYCL &S, SYCLIntegrationHeader &H,
                             QualType NameType, const FunctionDecl *FreeFunc)
      : SyclKernelFieldHandler(S), Header(H) {
    Header.startKernel(FreeFunc, NameType, FreeFunc->getLocation(),
                       false /*IsESIMD*/, true /*IsSYCLUnnamedKernel*/,
                       0 /*ObjSize*/);
  }

  bool handleSyclSpecialType(const CXXRecordDecl *RD,
                             const CXXBaseSpecifier &BC,
                             QualType FieldTy) final {
    if (isSyclAccessorType(FieldTy)) {
      const auto *AccTy =
          cast<ClassTemplateSpecializationDecl>(FieldTy->getAsRecordDecl());
      assert(AccTy->getTemplateArgs().size() >= 2 &&
             "Incorrect template args for Accessor Type");
      int Dims = static_cast<int>(
          AccTy->getTemplateArgs()[1].getAsIntegral().getExtValue());
      int Info = getAccessTarget(FieldTy, AccTy) | (Dims << 11);

      SYCLIntegrationHeader::kernel_param_kind_t ParamKind =
          SemaSYCL::isSyclType(FieldTy, SYCLTypeAttr::dynamic_local_accessor)
              ? SYCLIntegrationHeader::kind_dynamic_accessor
              : SYCLIntegrationHeader::kind_accessor;

      Header.addParamDesc(ParamKind, Info,
                          CurOffset +
                              offsetOf(RD, BC.getType()->getAsCXXRecordDecl()));
    } else if (SemaSYCL::isSyclType(FieldTy, SYCLTypeAttr::work_group_memory)) {
      addParam(FieldTy, SYCLIntegrationHeader::kind_work_group_memory,
               offsetOf(RD, BC.getType()->getAsCXXRecordDecl()));
    } else if (SemaSYCL::isSyclType(FieldTy,
                                    SYCLTypeAttr::dynamic_work_group_memory)) {
      addParam(FieldTy, SYCLIntegrationHeader::kind_dynamic_work_group_memory,
               offsetOf(RD, BC.getType()->getAsCXXRecordDecl()));
    }
    return true;
  }

  bool handleSyclSpecialType(FieldDecl *FD, QualType FieldTy) final {
    const auto *ClassTy = FieldTy->getAsCXXRecordDecl();
    assert(ClassTy && "Type must be a C++ record type");
    if (isSyclAccessorType(FieldTy)) {
      const auto *AccTy =
          cast<ClassTemplateSpecializationDecl>(FieldTy->getAsRecordDecl());
      assert(AccTy->getTemplateArgs().size() >= 2 &&
             "Incorrect template args for Accessor Type");
      int Dims = static_cast<int>(
          AccTy->getTemplateArgs()[1].getAsIntegral().getExtValue());
      int Info = getAccessTarget(FieldTy, AccTy) | (Dims << 11);

      SYCLIntegrationHeader::kernel_param_kind_t ParamKind =
          SemaSYCL::isSyclType(FieldTy, SYCLTypeAttr::dynamic_local_accessor)
              ? SYCLIntegrationHeader::kind_dynamic_accessor
              : SYCLIntegrationHeader::kind_accessor;

      Header.addParamDesc(ParamKind, Info, CurOffset + offsetOf(FD, FieldTy));
    } else if (SemaSYCL::isSyclType(FieldTy, SYCLTypeAttr::stream)) {
      addParam(FD, FieldTy, SYCLIntegrationHeader::kind_stream);
    } else if (SemaSYCL::isSyclType(FieldTy, SYCLTypeAttr::work_group_memory)) {
      addParam(FieldTy, SYCLIntegrationHeader::kind_work_group_memory,
               offsetOf(FD, FieldTy));
    } else if (SemaSYCL::isSyclType(FieldTy,
                                    SYCLTypeAttr::dynamic_work_group_memory)) {
      addParam(FieldTy, SYCLIntegrationHeader::kind_dynamic_work_group_memory,
               offsetOf(FD, FieldTy));
    } else if (SemaSYCL::isSyclType(FieldTy, SYCLTypeAttr::sampler) ||
               SemaSYCL::isSyclType(FieldTy, SYCLTypeAttr::annotated_ptr) ||
               SemaSYCL::isSyclType(FieldTy, SYCLTypeAttr::annotated_arg)) {
      CXXMethodDecl *InitMethod = getMethodByName(ClassTy, InitMethodName);
      assert(InitMethod && "type must have __init method");
      const ParmVarDecl *InitArg = InitMethod->getParamDecl(0);
      assert(InitArg && "Init method must have arguments");
      QualType T = InitArg->getType();
      SYCLIntegrationHeader::kernel_param_kind_t ParamKind =
          SemaSYCL::isSyclType(FieldTy, SYCLTypeAttr::sampler)
              ? SYCLIntegrationHeader::kind_sampler
              : (T->isPointerType() ? SYCLIntegrationHeader::kind_pointer
                                    : SYCLIntegrationHeader::kind_std_layout);
      addParam(T, ParamKind, offsetOf(FD, FieldTy));
    } else {
      llvm_unreachable(
          "Unexpected SYCL special class when generating integration header");
    }
    return true;
  }

  bool handleSyclSpecialType(ParmVarDecl *PD, QualType ParamTy) final {
    const auto *ClassTy = ParamTy->getAsCXXRecordDecl();
    assert(ClassTy && "Type must be a C++ record type");
    if (isSyclAccessorType(ParamTy)) {
      const auto *AccTy =
          cast<ClassTemplateSpecializationDecl>(ParamTy->getAsRecordDecl());
      assert(AccTy->getTemplateArgs().size() >= 2 &&
             "Incorrect template args for Accessor Type");
      int Dims = static_cast<int>(
          AccTy->getTemplateArgs()[1].getAsIntegral().getExtValue());
      int Info = getAccessTarget(ParamTy, AccTy) | (Dims << 11);
      Header.addParamDesc(SYCLIntegrationHeader::kind_accessor, Info,
                          CurOffset);
    } else if (SemaSYCL::isSyclType(ParamTy, SYCLTypeAttr::stream)) {
      addParam(PD, ParamTy, SYCLIntegrationHeader::kind_stream);
    } else if (SemaSYCL::isSyclType(ParamTy, SYCLTypeAttr::work_group_memory)) {
      addParam(PD, ParamTy, SYCLIntegrationHeader::kind_work_group_memory);
    } else if (SemaSYCL::isSyclType(ParamTy, SYCLTypeAttr::sampler) ||
               SemaSYCL::isSyclType(ParamTy, SYCLTypeAttr::annotated_ptr) ||
               SemaSYCL::isSyclType(ParamTy, SYCLTypeAttr::annotated_arg)) {
      CXXMethodDecl *InitMethod = getMethodByName(ClassTy, InitMethodName);
      assert(InitMethod && "type must have __init method");
      const ParmVarDecl *InitArg = InitMethod->getParamDecl(0);
      assert(InitArg && "Init method must have arguments");
      QualType T = InitArg->getType();
      SYCLIntegrationHeader::kernel_param_kind_t ParamKind =
          SemaSYCL::isSyclType(ParamTy, SYCLTypeAttr::sampler)
              ? SYCLIntegrationHeader::kind_sampler
              : (T->isPointerType() ? SYCLIntegrationHeader::kind_pointer
                                    : SYCLIntegrationHeader::kind_std_layout);
      addParam(PD, ParamTy, ParamKind);
    } else if (SemaSYCL::isSyclType(ParamTy,
                                    SYCLTypeAttr::dynamic_work_group_memory))
      addParam(PD, ParamTy,
               SYCLIntegrationHeader::kind_dynamic_work_group_memory);

    else {
      llvm_unreachable(
          "Unexpected SYCL special class when generating integration header");
    }
    return true;
  }

  bool handlePointerType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FD, FieldTy,
             ((StructDepth) ? SYCLIntegrationHeader::kind_std_layout
                            : SYCLIntegrationHeader::kind_pointer));
    return true;
  }

  bool handlePointerType(ParmVarDecl *PD, QualType ParamTy) final {
    addParam(PD, ParamTy, SYCLIntegrationHeader::kind_pointer);
    return true;
  }

  bool handleScalarType(FieldDecl *FD, QualType FieldTy) final {
    addParam(FD, FieldTy, SYCLIntegrationHeader::kind_std_layout);
    return true;
  }

  bool handleScalarType(ParmVarDecl *PD, QualType ParamTy) final {
    addParam(PD, ParamTy, SYCLIntegrationHeader::kind_std_layout);
    return true;
  }

  bool handleSimpleArrayType(FieldDecl *FD, QualType FieldTy) final {
    // Arrays are always wrapped inside of structs, so just treat it as a simple
    // struct.
    addParam(FD, FieldTy, SYCLIntegrationHeader::kind_std_layout);
    return true;
  }

  bool handleTopLevelStruct(const CXXRecordDecl *, QualType Ty) final {
    addParam(Ty, SYCLIntegrationHeader::kind_std_layout, /*Offset=*/0);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, FieldDecl *FD,
                             QualType Ty) final {
    addParam(FD, Ty, SYCLIntegrationHeader::kind_std_layout);
    return true;
  }

  bool handleNonDecompStruct(const CXXRecordDecl *, ParmVarDecl *PD,
                             QualType ParamTy) final {
    addParam(PD, ParamTy, SYCLIntegrationHeader::kind_std_layout);
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

  bool handleUnionType(ParmVarDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  void handleSyclKernelHandlerType(QualType) {
    // The compiler generated kernel argument used to initialize SYCL 2020
    // specialization constants, `specialization_constants_buffer`, should
    // have corresponding entry in integration header.
    ASTContext &Context = SemaSYCLRef.getASTContext();
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

  bool enterStruct(const CXXRecordDecl *, ParmVarDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, FieldDecl *FD, QualType Ty) final {
    --StructDepth;
    CurOffset -= offsetOf(FD, Ty);
    return true;
  }

  bool leaveStruct(const CXXRecordDecl *, ParmVarDecl *, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
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

  bool enterArray(ParmVarDecl *, QualType, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  bool nextElement(QualType ET, uint64_t Index) final {
    int64_t Size =
        SemaSYCLRef.getASTContext().getTypeSizeInChars(ET).getQuantity();
    CurOffset = ArrayBaseOffsets.back() + Size * Index;
    return true;
  }

  bool leaveArray(FieldDecl *FD, QualType ArrayTy, QualType) final {
    CurOffset = ArrayBaseOffsets.pop_back_val();
    CurOffset -= offsetOf(FD, ArrayTy);
    return true;
  }

  bool leaveArray(ParmVarDecl *, QualType, QualType) final {
    // TODO
    unsupportedFreeFunctionParamType();
    return true;
  }

  using SyclKernelFieldHandler::enterStruct;
  using SyclKernelFieldHandler::leaveStruct;
};

class SyclKernelIntFooterCreator : public SyclKernelFieldHandler {
  SYCLIntegrationFooter &Footer;

public:
  SyclKernelIntFooterCreator(SemaSYCL &S, SYCLIntegrationFooter &F)
      : SyclKernelFieldHandler(S), Footer(F) {
    (void)Footer; // workaround for unused field warning
  }
};

} // namespace

class SYCLKernelNameTypeVisitor
    : public TypeVisitor<SYCLKernelNameTypeVisitor>,
      public ConstTemplateArgumentVisitor<SYCLKernelNameTypeVisitor> {
  SemaSYCL &S;
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
  SYCLKernelNameTypeVisitor(SemaSYCL &S, SourceLocation KernelInvocationFuncLoc,
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
    if (!IsUnnamedKernel) {
      NotForwardDeclarableReason NFDR = isForwardDeclarable(DeclNamed, S);
      switch (NFDR) {
      case NotForwardDeclarableReason::UnscopedEnum:
        S.Diag(KernelInvocationFuncLoc, diag::err_sycl_kernel_incorrectly_named)
            << /* unscoped enum requires fixed underlying type */ 1
            << DeclNamed;
        IsInvalid = true;
        return;
      case NotForwardDeclarableReason::StdNamespace:
        S.Diag(KernelInvocationFuncLoc,
               diag::err_invalid_std_type_in_sycl_kernel)
            << KernelNameType << DeclNamed;
        IsInvalid = true;
        return;
      case NotForwardDeclarableReason::UnnamedTag:
        S.Diag(KernelInvocationFuncLoc, diag::err_sycl_kernel_incorrectly_named)
            << /* unnamed type is invalid */ 2 << KernelNameType;
        IsInvalid = true;
        return;
      case NotForwardDeclarableReason::NotAtNamespaceScope:
        S.Diag(KernelInvocationFuncLoc, diag::err_sycl_kernel_incorrectly_named)
            << /* kernel name should be forward declarable at namespace
                  scope */
            0 << KernelNameType;
        IsInvalid = true;
        return;
      case NotForwardDeclarableReason::None:
        // Do nothing, we're fine.
        break;
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

void SemaSYCL::CheckSYCLKernelCall(FunctionDecl *KernelFunc,
                                   ArrayRef<const Expr *> Args) {
  QualType KernelNameType = calculateKernelNameType(KernelFunc);
  SYCLKernelNameTypeVisitor KernelNameTypeVisitor(
      *this, Args[0]->getExprLoc(), KernelNameType,
      IsSYCLUnnamedKernel(*this, KernelFunc));
  KernelNameTypeVisitor.Visit(KernelNameType.getCanonicalType());

  // FIXME: In place until the library works around its 'host' invocation
  // issues.
  if (!SemaRef.LangOpts.SYCLIsDevice)
    return;

  const CXXRecordDecl *KernelObj =
      GetSYCLKernelObjectType(KernelFunc)->getAsCXXRecordDecl();

  if (!KernelObj || (KernelObj && !KernelObj->hasDefinition())) {
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
  if (not KernelParamTy->isReferenceType()) {
    // passing by value.  emit warning if using SYCL 2020 or greater
    if (SemaRef.LangOpts.getSYCLVersion() >= LangOptions::SYCL_2020)
      Diag(KernelFunc->getLocation(), diag::warn_sycl_pass_by_value_deprecated);
  }

  // Do not visit invalid kernel object.
  if (KernelObj->isInvalidDecl())
    return;

  SyclKernelDecompMarker DecompMarker(*this);
  SyclKernelFieldChecker FieldChecker(*this);
  SyclKernelUnionChecker UnionChecker(*this);

  KernelObjVisitor Visitor{*this};

  DiagnosingSYCLKernel = true;

  // Emit diagnostics for SYCL device kernels only
  Visitor.VisitRecordBases(KernelObj, FieldChecker, UnionChecker, DecompMarker);
  Visitor.VisitRecordFields(KernelObj, FieldChecker, UnionChecker,
                            DecompMarker);
  Visitor.visitTopLevelRecord(KernelObj, GetSYCLKernelObjectType(KernelFunc),
                              FieldChecker, UnionChecker, DecompMarker);

  DiagnosingSYCLKernel = false;
  // Set the kernel function as invalid, if any of the checkers fail validation.
  if (!FieldChecker.isValid() || !UnionChecker.isValid() ||
      !KernelNameTypeVisitor.isValid())
    KernelFunc->setInvalidDecl();
}

void SemaSYCL::CheckSYCLScopeAttr(CXXRecordDecl *Decl) {
  assert(Decl->hasAttr<SYCLScopeAttr>());

  bool HasError = false;

  if (Decl->isDependentContext())
    return;

  // We don't emit both diags at the time as note will only be emitted for the
  // first, which is confusing. So we check both cases but only report one.
  if (!Decl->hasTrivialDefaultConstructor()) {
    Diag(Decl->getLocation(), diag::err_sycl_wg_scope) << 0;
    HasError = true;
  } else if (!Decl->hasTrivialDestructor()) {
    Diag(Decl->getLocation(), diag::err_sycl_wg_scope) << 1;
    HasError = true;
  }

  if (HasError)
    Decl->dropAttr<SYCLScopeAttr>();
}

// For a wrapped parallel_for, copy attributes from original
// kernel to wrapped kernel.
void SemaSYCL::copyDeviceKernelAttrs(CXXMethodDecl *CallOperator) {
  // Get the operator() function of the wrapper.
  assert(CallOperator && "invalid kernel object");

  typedef std::pair<FunctionDecl *, FunctionDecl *> ChildParentPair;
  llvm::SmallPtrSet<FunctionDecl *, 16> Visited;
  llvm::SmallVector<ChildParentPair, 16> WorkList;
  WorkList.push_back({CallOperator, nullptr});
  FunctionDecl *KernelBody = nullptr;

  CallGraph SYCLCG;
  SYCLCG.addToCallGraph(CallOperator);
  while (!WorkList.empty()) {
    FunctionDecl *FD = WorkList.back().first;
    FunctionDecl *ParentFD = WorkList.back().second;

    if ((ParentFD == CallOperator) && isSYCLKernelBodyFunction(FD)) {
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
    collectSYCLAttributes(KernelBody, Attrs, /*DirectlyCalled*/ true);
    if (!Attrs.empty())
      llvm::for_each(Attrs,
                     [CallOperator](Attr *A) { CallOperator->addAttr(A); });
  }
}

void SemaSYCL::SetSYCLKernelNames() {
  std::unique_ptr<MangleContext> MangleCtx(
      getASTContext().createMangleContext());
  // We assume the list of KernelDescs is the complete list of kernels needing
  // to be rewritten.
  for (const std::pair<const FunctionDecl *, FunctionDecl *> &Pair :
       SyclKernelsToOpenCLKernels) {
    std::string CalculatedName, StableName;
    StringRef KernelName;
    if (isFreeFunction(Pair.first)) {
      std::tie(CalculatedName, StableName) =
          constructFreeFunctionKernelName(Pair.first, *MangleCtx);
      KernelName = CalculatedName;
    } else {
      std::tie(CalculatedName, StableName) =
          constructKernelName(*this, Pair.first, *MangleCtx);
      KernelName =
          IsSYCLUnnamedKernel(*this, Pair.first) ? StableName : CalculatedName;
    }

    getSyclIntegrationHeader().updateKernelNames(Pair.first, KernelName,
                                                 StableName);

    // Set name of generated kernel.
    Pair.second->setDeclName(&getASTContext().Idents.get(KernelName));
    // Update the AsmLabel for this generated kernel.
    Pair.second->addAttr(
        AsmLabelAttr::CreateImplicit(getASTContext(), KernelName));
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
void SemaSYCL::ConstructOpenCLKernel(FunctionDecl *KernelCallerFunc,
                                     MangleContext &MC) {
  // The first argument to the KernelCallerFunc is the lambda object.
  QualType KernelObjTy = GetSYCLKernelObjectType(KernelCallerFunc);
  const CXXRecordDecl *KernelObj = KernelObjTy->getAsCXXRecordDecl();
  assert(KernelObj && "invalid kernel caller");

  // Do not visit invalid kernel object.
  if (KernelObj->isInvalidDecl())
    return;

  KernelCallOperatorVisitor KernelCallOperator(KernelCallerFunc, KernelObj);
  CXXMethodDecl *CallOperator = nullptr;

  if (KernelObj->isLambda())
    CallOperator = KernelObj->getLambdaCallOperator();
  else
    CallOperator = KernelCallOperator.getCallOperator();

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
      copyDeviceKernelAttrs(CallOperator);
  }

  bool IsSIMDKernel = isESIMDKernelType(CallOperator);

  SyclKernelArgsSizeChecker argsSizeChecker(*this, KernelObj->getLocation(),
                                            IsSIMDKernel);
  ESIMDKernelDiagnostics esimdKernel(*this, KernelObj->getLocation(),
                                     IsSIMDKernel);

  SyclKernelDeclCreator kernel_decl(*this, KernelObj->getLocation(),
                                    KernelCallerFunc->isInlined(), IsSIMDKernel,
                                    KernelCallerFunc);
  SyclKernelBodyCreator kernel_body(*this, kernel_decl, KernelObj,
                                    KernelCallerFunc, IsSIMDKernel,
                                    CallOperator);
  SyclKernelIntHeaderCreator int_header(
      IsSIMDKernel, *this, getSyclIntegrationHeader(), KernelObj,
      calculateKernelNameType(KernelCallerFunc), KernelCallerFunc);

  SyclKernelIntFooterCreator int_footer(*this, getSyclIntegrationFooter());
  SyclOptReportCreator opt_report(*this, kernel_decl, KernelObj->getLocation());

  KernelObjVisitor Visitor{*this};

  // Visit handlers to generate information for optimization record only if
  // optimization record is saved.
  if (!getLangOpts().OptRecordFile.empty()) {
    Visitor.VisitKernelRecord(KernelObj, KernelObjTy, argsSizeChecker,
                              esimdKernel, kernel_decl, kernel_body, int_header,
                              int_footer, opt_report);
  } else {
    Visitor.VisitKernelRecord(KernelObj, KernelObjTy, argsSizeChecker,
                              esimdKernel, kernel_decl, kernel_body, int_header,
                              int_footer);
  }

  if (ParmVarDecl *KernelHandlerArg =
          getSyclKernelHandlerArg(KernelCallerFunc)) {
    kernel_decl.handleSyclKernelHandlerType();
    kernel_body.handleSyclKernelHandlerType(KernelHandlerArg);
    int_header.handleSyclKernelHandlerType(KernelHandlerArg->getType());

    if (!getLangOpts().OptRecordFile.empty())
      opt_report.handleSyclKernelHandlerType();
  }
}

static void addRegisteredKernelName(SemaSYCL &S, StringRef Str,
                                    FunctionDecl *FD, SourceLocation Loc) {
  if (!Str.empty())
    FD->addAttr(SYCLRegisteredKernelNameAttr::CreateImplicit(S.getASTContext(),
                                                             Str, Loc));
}

static bool checkAndAddRegisteredKernelName(SemaSYCL &S, FunctionDecl *FD,
                                            StringRef Str) {
  using KernelPair = std::pair<const FunctionDecl *, FunctionDecl *>;
  for (const KernelPair &Pair : S.getKernelFDPairs()) {
    if (Pair.first == FD) {
      // If the current list of free function entries already contains this
      // free function, apply the name Str as an attribute.  But if it already
      // has an attribute name, issue a diagnostic instead.
      if (!Str.empty()) {
        if (!Pair.second->hasAttr<SYCLRegisteredKernelNameAttr>())
          addRegisteredKernelName(S, Str, Pair.second, FD->getLocation());
        else
          S.Diag(FD->getLocation(),
                 diag::err_registered_kernels_name_already_registered)
              << Pair.second->getAttr<SYCLRegisteredKernelNameAttr>()
                     ->getRegName()
              << Str;
      }
      // An empty name string implies a regular free kernel construction
      // call, so simply return.
      return false;
    }
  }
  return true;
}

void SemaSYCL::constructFreeFunctionKernel(FunctionDecl *FD,
                                           StringRef NameStr) {
  if (!checkAndAddRegisteredKernelName(*this, FD, NameStr))
    return;

  SyclKernelArgsSizeChecker argsSizeChecker(*this, FD->getLocation(),
                                            false /*IsSIMDKernel*/);
  SyclKernelDeclCreator kernel_decl(*this, FD->getLocation(), FD->isInlined(),
                                    false /*IsSIMDKernel */, FD);

  FreeFunctionKernelBodyCreator kernel_body(*this, kernel_decl, FD);

  SyclKernelIntHeaderCreator int_header(*this, getSyclIntegrationHeader(),
                                        FD->getType(), FD);

  SyclKernelIntFooterCreator int_footer(*this, getSyclIntegrationFooter());
  KernelObjVisitor Visitor{*this};

  Visitor.VisitFunctionParameters(FD, argsSizeChecker, kernel_decl, kernel_body,
                                  int_header, int_footer);

  assert(getKernelFDPairs().back().first == FD &&
         "OpenCL Kernel not found for free function entry");
  // Register the kernel name with the OpenCL kernel generated for the
  // free function.
  addRegisteredKernelName(*this, NameStr, getKernelFDPairs().back().second,
                          FD->getLocation());
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

static void CheckSYCL2020SubGroupSizes(SemaSYCL &S, FunctionDecl *SYCLKernel,
                                       const FunctionDecl *FD) {
  // If they are the same, no error.
  if (CalcEffectiveSubGroup(S.getASTContext(), S.getLangOpts(), SYCLKernel) ==
      CalcEffectiveSubGroup(S.getASTContext(), S.getLangOpts(), FD))
    return;

  // No need to validate __spirv routines here since they
  // are mapped to the equivalent SPIRV operations.
  const IdentifierInfo *II = FD->getIdentifier();
  if (II && II->getName().starts_with("__spirv_"))
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
    SemaSYCL &S, FunctionDecl *SYCLKernel, FunctionDecl *KernelBody,
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

static void PropagateAndDiagnoseDeviceAttr(SemaSYCL &S, Attr *A,
                                           FunctionDecl *SYCLKernel,
                                           FunctionDecl *KernelBody) {
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
  case attr::Kind::SYCLReqdWorkGroupSize: {
    auto *RWGSA = cast<SYCLReqdWorkGroupSizeAttr>(A);
    if (auto *Existing = SYCLKernel->getAttr<SYCLReqdWorkGroupSizeAttr>()) {
      if (S.anyWorkGroupSizesDiffer(Existing->getXDim(), Existing->getYDim(),
                                    Existing->getZDim(), RWGSA->getXDim(),
                                    RWGSA->getYDim(), RWGSA->getZDim())) {
        S.Diag(SYCLKernel->getLocation(),
               diag::err_conflicting_sycl_kernel_attributes);
        S.Diag(Existing->getLocation(), diag::note_conflicting_attribute);
        S.Diag(RWGSA->getLocation(), diag::note_conflicting_attribute);
        SYCLKernel->setInvalidDecl();
      }
    } else if (auto *Existing =
                   SYCLKernel->getAttr<SYCLIntelMaxWorkGroupSizeAttr>()) {
      if (S.checkMaxAllowedWorkGroupSize(
              RWGSA->getXDim(), RWGSA->getYDim(), RWGSA->getZDim(),
              Existing->getXDim(), Existing->getYDim(), Existing->getZDim())) {
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
  case attr::Kind::SYCLWorkGroupSizeHint: {
    auto *WGSH = cast<SYCLWorkGroupSizeHintAttr>(A);
    if (auto *Existing = SYCLKernel->getAttr<SYCLWorkGroupSizeHintAttr>()) {
      if (S.anyWorkGroupSizesDiffer(Existing->getXDim(), Existing->getYDim(),
                                    Existing->getZDim(), WGSH->getXDim(),
                                    WGSH->getYDim(), WGSH->getZDim())) {
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
    if (auto *Existing = SYCLKernel->getAttr<SYCLReqdWorkGroupSizeAttr>()) {
      if (S.checkMaxAllowedWorkGroupSize(
              Existing->getXDim(), Existing->getYDim(), Existing->getZDim(),
              SIMWGSA->getXDim(), SIMWGSA->getYDim(), SIMWGSA->getZDim())) {
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
  case attr::Kind::SYCLIntelMinWorkGroupsPerComputeUnit:
  case attr::Kind::SYCLIntelMaxWorkGroupsPerMultiprocessor:
  case attr::Kind::SYCLIntelNoGlobalWorkOffset:
  case attr::Kind::SYCLIntelLoopFuse:
  case attr::Kind::SYCLIntelMaxConcurrency:
  case attr::Kind::SYCLIntelDisableLoopPipelining:
  case attr::Kind::SYCLIntelInitiationInterval:
  case attr::Kind::SYCLIntelUseStallEnableClusters:
  case attr::Kind::SYCLDeviceHas:
  case attr::Kind::SYCLAddIRAttributesFunction:
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

void SemaSYCL::MarkDevices() {
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
      PropagateAndDiagnoseDeviceAttr(*this, A, T.GetSYCLKernel(),
                                     T.GetKernelBody());
    checkSYCLAddIRAttributesFunctionAttrConflicts(T.GetSYCLKernel());
  }
}

static bool CheckFreeFunctionDiagnostics(Sema &S, const FunctionDecl *FD) {
  if (FD->isVariadic()) {
    return S.Diag(FD->getLocation(), diag::err_free_function_variadic_args);
  }

  if (!FD->getReturnType()->isVoidType()) {
    return S.Diag(FD->getLocation(), diag::err_free_function_return_type);
  }

  if (const auto *MD = llvm::dyn_cast<CXXMethodDecl>(FD))
    // The free function extension specification usual methods to be used to
    // define a free function kernel. We also disallow static methods because we
    // use integration header.
    return S.Diag(FD->getLocation(), diag::err_free_function_class_method)
           << !MD->isStatic() << MD->getSourceRange();

  for (ParmVarDecl *Param : FD->parameters()) {
    if (Param->hasDefaultArg()) {
      return S.Diag(Param->getLocation(),
                    diag::err_free_function_with_default_arg)
             << Param->getSourceRange();
    }
  }
  return false;
}

void SemaSYCL::finalizeFreeFunctionKernels() {
  // This is called at the end of the translation unit. The kernels that appear
  // in this list are kernels that have been declared but not defined. Their
  // construction consists only of generating the integration header and setting
  // their names manually. The other steps in constructing the kernel cannot be
  // done because potentially nothing is known about the arguments of the kernel
  // except that they exist.
  for (const FunctionDecl *kernel : FreeFunctionDeclarations) {
    if (CheckFreeFunctionDiagnostics(SemaRef, kernel))
      continue; // Continue in order to diagnose errors in all kernels

    SyclKernelIntHeaderCreator IntHeader(*this, getSyclIntegrationHeader(),
                                         kernel->getType(), kernel);
    KernelObjVisitor Visitor{*this};
    Visitor.VisitFunctionParameters(kernel, IntHeader);
    std::unique_ptr<MangleContext> MangleCtx(
        getASTContext().createMangleContext());
    std::string Name, MangledName;
    std::tie(Name, MangledName) =
        constructFreeFunctionKernelName(kernel, *MangleCtx);
    getSyclIntegrationHeader().updateKernelNames(kernel, Name, MangledName);
  }
}

void SemaSYCL::processFreeFunctionDeclaration(const FunctionDecl *FD) {
  // FD represents a forward declaration of a free function kernel.
  // Save them for the end of the translation unit action. This makes it easier
  // to handle the case where a definition is defined later.
  if (isFreeFunction(FD))
    FreeFunctionDeclarations.insert(FD->getCanonicalDecl());
}

void SemaSYCL::ProcessFreeFunction(FunctionDecl *FD) {
  if (isFreeFunction(FD)) {
    if (CheckFreeFunctionDiagnostics(SemaRef, FD))
      return;

    // In case the free function kernel has already been seen by way of a
    // forward declaration, flush it out because a definition takes priority.
    FreeFunctionDeclarations.erase(FD->getCanonicalDecl());

    SyclKernelDecompMarker DecompMarker(*this);
    SyclKernelFieldChecker FieldChecker(*this);
    SyclKernelUnionChecker UnionChecker(*this);

    KernelObjVisitor Visitor{*this};

    DiagnosingSYCLKernel = true;

    // Check parameters of free function.
    Visitor.VisitFunctionParameters(FD, DecompMarker, FieldChecker,
                                    UnionChecker);

    DiagnosingSYCLKernel = false;

    // Ignore the free function if any of the checkers fail validation.
    if (!FieldChecker.isValid() || !UnionChecker.isValid())
      return;

    constructFreeFunctionKernel(FD);
  }
}

// -----------------------------------------------------------------------------
// SYCL device specific diagnostics implementation
// -----------------------------------------------------------------------------

Sema::SemaDiagnosticBuilder
SemaSYCL::DiagIfDeviceCode(SourceLocation Loc, unsigned DiagID,
                           DeviceDiagnosticReason Reason) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL compilation");
  FunctionDecl *FD = dyn_cast<FunctionDecl>(SemaRef.getCurLexicalContext());
  SemaDiagnosticBuilder::Kind DiagKind = [this, FD, Reason] {
    if (DiagnosingSYCLKernel)
      return SemaDiagnosticBuilder::K_ImmediateWithCallStack;
    if (!FD)
      return SemaDiagnosticBuilder::K_Nop;
    if (SemaRef.isConstantEvaluatedContext() ||
        SemaRef.currentEvaluationContext().isDiscardedStatementContext())
      return SemaDiagnosticBuilder::K_Nop;
    // Defer until we know that the variable's intializer is actually a
    // manifestly constant-evaluated expression.
    if (SemaRef.InConstexprVarInit)
      return SemaDiagnosticBuilder::K_Deferred;
    if (SemaRef.getEmissionStatus(FD) ==
        Sema::FunctionEmissionStatus::Emitted) {
      // Skip the diagnostic if we know it won't be emitted.
      if ((SemaRef.getEmissionReason(FD) & Reason) ==
          Sema::DeviceDiagnosticReason::None)
        return SemaDiagnosticBuilder::K_Nop;

      return SemaDiagnosticBuilder::K_ImmediateWithCallStack;
    }
    return SemaDiagnosticBuilder::K_Deferred;
  }();
  return SemaDiagnosticBuilder(DiagKind, Loc, DiagID, FD, SemaRef, Reason);
}

void SemaSYCL::deepTypeCheckForDevice(SourceLocation UsedAt,
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
      DiagIfDeviceCode(UsedAt, diag::err_typecheck_zero_array_size) << 1;
      ErrorFound = true;
    }
    // Checks for other types can also be done here.
    if (ErrorFound) {
      if (NeedToEmitNotes) {
        if (auto *FD = dyn_cast<FieldDecl>(D))
          DiagIfDeviceCode(FD->getLocation(),
                           diag::note_illegal_field_declared_here)
              << FD->getType()->isPointerType() << FD->getType();
        else
          DiagIfDeviceCode(D->getLocation(), diag::note_declared_at);
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
        DiagIfDeviceCode(History[Index]->getLocation(),
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
      llvm::append_range(StackForRecursion, RecDecl->fields());
    }
  } while (!StackForRecursion.empty());
}

void SemaSYCL::finalizeSYCLDelayedAnalysis(const FunctionDecl *Caller,
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
  if (Callee->hasAttr<SYCLDeviceAttr>() || Callee->hasAttr<DeviceKernelAttr>())
    return;

  // If Callee has a CUDA device attribute, no diagnostic needed.
  if (getLangOpts().CUDA && Callee->hasAttr<CUDADeviceAttr>())
    return;

  // Diagnose if this is an undefined function and it is not a builtin.
  // Currently, there is an exception of "__failed_assertion" in libstdc++-11,
  // this undefined function is used to trigger a compiling error.
  if (!Callee->isDefined() && !Callee->getBuiltinID() &&
      !Callee->isReplaceableGlobalAllocationFunction() &&
      !isSYCLUndefinedAllowed(Callee, SemaRef.getSourceManager())) {
    Diag(Loc, diag::err_sycl_restrict) << SemaSYCL::KernelCallUndefinedFunction;
    Diag(Callee->getLocation(), diag::note_previous_decl) << Callee;
    Diag(Caller->getLocation(), diag::note_called_by) << Caller;
  }
}

bool SemaSYCL::checkAllowedSYCLInitializer(VarDecl *VD) {
  assert(getLangOpts().SYCLIsDevice &&
         "Should only be called during SYCL compilation");

  if (VD->isInvalidDecl() || !VD->hasInit() || !VD->hasGlobalStorage())
    return true;

  const Expr *Init = VD->getInit();
  bool ValueDependent = Init && Init->isValueDependent();
  bool isConstantInit = Init && !ValueDependent &&
                        Init->isConstantInitializer(getASTContext(), false);
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
    CASE(work_group_memory);
    CASE(dynamic_work_group_memory);
    CASE(dynamic_accessor);
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
//     deviceQueue.submit([&](sycl::handler& cgh) {
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
      QualType T = ED->getIntegerType().getCanonicalType();
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
  SYCLFwdDeclEmitter(raw_ostream &OS, const LangOptions &LO)
      : OS(OS), Policy(LO) {
    Policy.adjustForCPlusPlusFwdDecl();
    Policy.SuppressTypedefs = true;
    Policy.SuppressUnwrittenScope = true;
    Policy.PrintAsCanonical = true;
    Policy.SkipCanonicalizationOfTemplateTypeParms = true;
    Policy.SuppressFinalSpecifier = true;
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

  void VisitFunctionProtoType(const FunctionProtoType *T) {
    for (const auto Ty : T->getParamTypes())
      Visit(Ty.getCanonicalType());
    // So far this visitor method is only used for free function kernels whose
    // return type is void anyway, so it is not visited. Otherwise, add if
    // required.
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
      RD->printName(OS, Policy);
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

static void EmitPragmaDiagnosticPush(raw_ostream &O, StringRef DiagName) {
  O << "\n";
  O << "#ifdef __clang__\n";
  O << "#pragma clang diagnostic push\n";
  O << "#pragma clang diagnostic ignored \"" << DiagName.str() << "\"\n";
  O << "#endif // defined(__clang__)\n";
  O << "\n";
}

static void EmitPragmaDiagnosticPop(raw_ostream &O) {
  O << "\n";
  O << "#ifdef __clang__\n";
  O << "#pragma clang diagnostic pop\n";
  O << "#endif // defined(__clang__)\n";
  O << "\n";
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

static void PrintNamespaces(raw_ostream &OS, const DeclContext *DC,
                            bool isPrintNamesOnly = false) {
  PrintNSHelper([](raw_ostream &, const NamespaceDecl *) {},
                [isPrintNamesOnly](raw_ostream &OS, const NamespaceDecl *NS) {
                  if (!isPrintNamesOnly) {
                    if (NS->isInline())
                      OS << "inline ";
                    OS << "namespace ";
                  }
                  if (!NS->isAnonymousNamespace()) {
                    OS << NS->getName();
                    if (isPrintNamesOnly)
                      OS << "::";
                    else
                      OS << " ";
                  }
                  if (!isPrintNamesOnly) {
                    OS << "{\n";
                  }
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
      [](raw_ostream &, const NamespaceDecl *) {}, OS, DC);
}

class FreeFunctionPrinter {
  raw_ostream &O;
  PrintingPolicy &Policy;
  bool NSInserted = false;

public:
  FreeFunctionPrinter(raw_ostream &O, PrintingPolicy &PrintPolicy)
      : O(O), Policy(PrintPolicy) {}

  /// Emits the function declaration of template free function.
  /// \param FTD The function declaration to print.
  /// \param S Sema object.
  void printFreeFunctionDeclaration(FunctionTemplateDecl *FTD) {
    const FunctionDecl *TemplatedDecl = FTD->getTemplatedDecl();
    if (!TemplatedDecl)
      return;
    const std::string TemplatedDeclParams =
        getTemplatedParamList(TemplatedDecl->parameters(), Policy);
    const std::string TemplateParams =
        getTemplateParameters(FTD->getTemplateParameters());
    printFreeFunctionDeclaration(TemplatedDecl, TemplatedDeclParams,
                                 TemplateParams);
  }

  /// Emits the function declaration of a free function.
  /// \param FD The function declaration to print.
  /// \param Args The arguments of the function.
  /// \param TemplateParameters The template parameters of the function.
  void printFreeFunctionDeclaration(const FunctionDecl *FD,
                                    const std::string &Args,
                                    std::string_view TemplateParameters = "") {
    const DeclContext *DC = FD->getDeclContext();
    if (DC) {
      // if function in namespace, print namespace
      if (isa<NamespaceDecl>(DC)) {
        PrintNamespaces(O, FD);
        // Set flag to print closing braces for namespaces and namespace in shim
        // function
        NSInserted = true;
      }
      if (FD->isFunctionTemplateSpecialization() &&
          FD->isThisDeclarationADefinition())
        O << "template <> ";
      O << TemplateParameters;
      O << FD->getReturnType().getAsString() << " ";
      FD->printName(O, Policy);
      if (FD->isFunctionTemplateSpecialization() &&
          FD->isThisDeclarationADefinition())
        O << getTemplateSpecializationArgString(
            FD->getTemplateSpecializationArgs());

      O << "(" << Args << ");";
      if (NSInserted) {
        O << "\n";
        PrintNSClosingBraces(O, FD);
      }
      O << "\n";
    }
  }

  /// Emits free function shim function.
  /// \param FD The function declaration to print.
  /// \param ShimCounter The counter for the shim function.
  /// \param ParmList The parameter list of the function.
  void printFreeFunctionShim(const FunctionDecl *FD, const unsigned ShimCounter,
                             const std::string &ParmList) {
    // Generate a shim function that returns the address of the free function.
    O << "static constexpr auto __sycl_shim" << ShimCounter << "() {\n";
    O << "  return (void (*)(" << ParmList << "))";

    if (NSInserted)
      PrintNamespaces(O, FD, /*isPrintNamesOnly=*/true);
    O << FD->getIdentifier()->getName().data();
    if (FD->getPrimaryTemplate())
      O << getTemplateSpecializationArgString(
          FD->getTemplateSpecializationArgs());
  }

  /// Emits free function kernel info specialization for shimN.
  /// \param ShimCounter The counter for the shim function.
  /// \param KParamsSize The number of kernel free function arguments.
  /// \param KName The name of the kernel free function.
  void printFreeFunctionKernelInfo(const unsigned ShimCounter,
                                   const size_t KParamsSize,
                                   std::string_view KName) {
    O << "\n";
    O << "namespace sycl {\n";
    O << "inline namespace _V1 {\n";
    O << "namespace detail {\n";
    O << "//Free Function Kernel info specialization for shim" << ShimCounter
      << "\n";
    O << "template <> struct FreeFunctionInfoData<__sycl_shim" << ShimCounter
      << "()> {\n";
    O << "  __SYCL_DLL_LOCAL\n";
    O << "  static constexpr unsigned getNumParams() { return " << KParamsSize
      << "; }\n";
    O << "  __SYCL_DLL_LOCAL\n";
    O << "  static constexpr const char *getFunctionName() { return ";
    O << "\"" << KName << "\"; }\n";
    O << "};\n";
    O << "} // namespace detail\n"
      << "} // namespace _V1\n"
      << "} // namespace sycl\n";
    O << "\n";
  }

private:
  /// Helper method to get string with template types
  /// \param TAL The template argument list.
  /// \returns string Example:
  /// \code
  ///  template <typename T1, typename T2>
  ///  void foo(T1 a, T2 b);
  /// \endcode
  /// returns string "<T1, T2>"
  /// If TAL is nullptr, returns empty string.
  std::string
  getTemplateSpecializationArgString(const TemplateArgumentList *TAL) {
    if (!TAL)
      return "";
    std::string Buffer;
    llvm::raw_string_ostream StringStream(Buffer);
    ArrayRef<TemplateArgument> A = TAL->asArray();
    bool FirstParam = true;
    for (const auto &X : A) {
      if (FirstParam)
        FirstParam = false;
      else if (X.getKind() == TemplateArgument::Pack) {
        for (const auto &PackArg : X.pack_elements()) {
          StringStream << ", ";
          PackArg.print(Policy, StringStream, /*IncludeType*/ true);
        }
        continue;
      } else
        StringStream << ", ";

      X.print(Policy, StringStream, /*IncludeType*/ true);
    }
    StringStream.flush();
    if (Buffer.front() != '<')
      Buffer = "<" + Buffer + ">";
    return Buffer;
  }

  /// Helper method to get arguments of templated function as a string
  /// \param Parameters Array of parameters of the function.
  /// \param Policy Printing policy.
  /// returned string Example:
  /// \code
  ///  template <typename T1, typename T2>
  ///  void foo(T1 a, T2 b);
  /// \endcode
  /// returns string "T1 a, T2 b"
  std::string
  getTemplatedParamList(const llvm::ArrayRef<clang::ParmVarDecl *> Parameters,
                        PrintingPolicy Policy) {
    bool FirstParam = true;
    llvm::SmallString<128> ParamList;
    llvm::raw_svector_ostream ParmListOstream{ParamList};
    Policy.SuppressTagKeyword = true;
    for (ParmVarDecl *Param : Parameters) {
      if (FirstParam)
        FirstParam = false;
      else
        ParmListOstream << ", ";
      ParmListOstream << Param->getType().getAsString(Policy);
      ParmListOstream << " " << Param->getNameAsString();
    }
    return ParamList.str().str();
  }

  /// Helper method to get text representation of the template parameters.
  /// Throws an error if the last parameter is a pack.
  /// \param TPL The template parameter list.
  /// \param S The SemaSYCL object.
  /// Example:
  /// \code
  ///  template <typename T1, class T2>
  ///  void foo(T1 a, T2 b);
  /// \endcode
  /// returns string "template <typename T1, class T2> "
  std::string getTemplateParameters(const clang::TemplateParameterList *TPL) {
    std::string TemplateParams{"template <"};
    bool FirstParam{true};
    for (NamedDecl *Param : *TPL) {
      if (!FirstParam)
        TemplateParams += ", ";
      FirstParam = false;
      if (const auto *TemplateParam = dyn_cast<TemplateTypeParmDecl>(Param)) {
        TemplateParams +=
            TemplateParam->wasDeclaredWithTypename() ? "typename " : "class ";
        if (TemplateParam->isParameterPack())
          TemplateParams += "... ";
        TemplateParams += TemplateParam->getNameAsString();
      } else if (const auto *NonTypeParam =
                     dyn_cast<NonTypeTemplateParmDecl>(Param)) {
        TemplateParams += NonTypeParam->getType().getAsString();
        TemplateParams += " ";
        TemplateParams += NonTypeParam->getNameAsString();
      }
    }
    TemplateParams += "> ";
    return TemplateParams;
  }
};

void SYCLIntegrationHeader::emit(raw_ostream &O) {
  O << "// This is auto-generated SYCL integration header.\n";
  O << "\n";

  O << "#include <sycl/detail/defines_elementary.hpp>\n";
  O << "#include <sycl/detail/kernel_desc.hpp>\n";
  O << "#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>\n";
  O << "#include <sycl/access/access.hpp>\n";
  O << "\n";

  LangOptions LO;
  PrintingPolicy Policy(LO);
  Policy.SuppressTypedefs = true;
  Policy.SuppressUnwrittenScope = true;
  // Disable printing anonymous tag locations because on Windows
  // file path separators are treated as escape sequences and cause errors
  // when integration header is compiled with host compiler.
  Policy.AnonymousTagLocations = 0;
  SYCLFwdDeclEmitter FwdDeclEmitter(O, S.getLangOpts());

  // Predefines which need to be set for custom host compilation
  // must be defined in integration header.
  for (const std::pair<StringRef, StringRef> &Macro :
       getSYCLVersionMacros(S.getLangOpts())) {
    O << "#ifndef " << Macro.first << '\n';
    O << "#define " << Macro.first << " " << Macro.second << '\n';
    O << "#endif //" << Macro.first << "\n\n";
  }

  switch (S.getLangOpts().getSYCLRangeRounding()) {
  case LangOptions::SYCLRangeRoundingPreference::Disable:
    O << "#ifndef __SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ \n";
    O << "#define __SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ 1\n";
    O << "#endif //__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__\n\n";
    break;
  case LangOptions::SYCLRangeRoundingPreference::Force:
    O << "#ifndef __SYCL_FORCE_PARALLEL_FOR_RANGE_ROUNDING__ \n";
    O << "#define __SYCL_FORCE_PARALLEL_FOR_RANGE_ROUNDING__ 1\n";
    O << "#endif //__SYCL_FORCE_PARALLEL_FOR_RANGE_ROUNDING__\n\n";
    break;
  default:
    break;
  }

  if (S.getLangOpts().SYCLExperimentalRangeRounding) {
    O << "#ifndef __SYCL_EXP_PARALLEL_FOR_RANGE_ROUNDING__ \n";
    O << "#define __SYCL_EXP_PARALLEL_FOR_RANGE_ROUNDING__ 1\n";
    O << "#endif //__SYCL_EXP_PARALLEL_FOR_RANGE_ROUNDING__\n\n";
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

  O << "namespace sycl {\n";
  O << "inline namespace _V1 {\n";
  O << "namespace detail {\n";

  // Generate declaration of variable of type __sycl_device_global_registration
  // whose sole purpose is to run its constructor before the application's
  // main() function.

  if (NeedToEmitDeviceGlobalRegistration) {
    // Supress the reserved identifier diagnostic that clang generates
    // for the construct below.
    EmitPragmaDiagnosticPush(O, "-Wreserved-identifier");
    O << "namespace {\n";

    O << "class __sycl_device_global_registration {\n";
    O << "public:\n";
    O << "  __sycl_device_global_registration() noexcept;\n";
    O << "};\n";
    O << "__sycl_device_global_registration __sycl_device_global_registrar;\n";

    O << "} // namespace\n";

    O << "\n";
    EmitPragmaDiagnosticPop(O);
  }

  // Generate declaration of variable of type __sycl_host_pipe_registration
  // whose sole purpose is to run its constructor before the application's
  // main() function.
  if (NeedToEmitHostPipeRegistration) {
    // Supress the reserved identifier diagnostic that clang generates
    // for the construct below.
    EmitPragmaDiagnosticPush(O, "-Wreserved-identifier");
    O << "namespace {\n";

    O << "class __sycl_host_pipe_registration {\n";
    O << "public:\n";
    O << "  __sycl_host_pipe_registration() noexcept;\n";
    O << "};\n";
    O << "__sycl_host_pipe_registration __sycl_host_pipe_registrar;\n";

    O << "} // namespace\n";

    O << "\n";
    EmitPragmaDiagnosticPop(O);
  }


  O << "// names of all kernels defined in the corresponding source\n";
  O << "static constexpr\n";
  O << "const char* const kernel_names[] = {\n";

  for (unsigned I = 0; I < KernelDescs.size(); I++) {
    O << "  \"" << KernelDescs[I].Name << "\",\n";
  }
  // Add a sentinel to avoid warning if the collection is empty
  // (similar to what we do for kernel_signatures below).
  O << "  \"\",\n";
  O << "};\n\n";

  O << "static constexpr unsigned kernel_args_sizes[] = {";
  for (unsigned I = 0; I < KernelDescs.size(); I++) {
    O << KernelDescs[I].Params.size() << ", ";
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
    if (S.isFreeFunction(K.SyclKernel)) {
      CurStart += N;
      continue;
    }
    PresumedLoc PLoc = S.getASTContext().getSourceManager().getPresumedLoc(
        S.getASTContext()
            .getSourceManager()
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
    StringRef ReturnType = (S.getASTContext().getTargetInfo().getInt64Type() ==
                            TargetInfo::SignedLong)
                               ? "long"
                               : "long long";
    O << "  // Returns the size of the kernel object in bytes.\n";
    O << "  __SYCL_DLL_LOCAL\n";
    O << "  static constexpr " << ReturnType << " getKernelSize() { return "
      << K.ObjSize << "; }\n";
    O << "};\n";
    CurStart += N;
  }
  O << "\n";
  O << "} // namespace detail\n";
  O << "} // namespace _V1\n";
  O << "} // namespace sycl\n";

  // The rest of this function only applies to free-function kernels. However,
  // in RTC mode, we do not need integration header information for
  // free-function kernels, so we can return early here.
  if (S.getLangOpts().SYCLRTCMode) {
    return;
  }

  unsigned ShimCounter = 1;
  int FreeFunctionCount = 0;
  for (const KernelDesc &K : KernelDescs) {
    if (!S.isFreeFunction(K.SyclKernel))
      continue;
    ++FreeFunctionCount;
    // Generate forward declaration for free function.
    O << "\n// Definition of " << K.Name << " as a free function kernel\n";

    O << "\n";
    O << "// Forward declarations of kernel and its argument types:\n";
    Policy.SuppressDefaultTemplateArgs = false;
    FwdDeclEmitter.Visit(K.SyclKernel->getType());
    O << "\n";

    if (K.SyclKernel->getLanguageLinkage() == CLanguageLinkage)
      O << "extern \"C\" ";
    std::string ParmList;
    std::string ParmListWithNames;
    bool FirstParam = true;
    Policy.SuppressDefaultTemplateArgs = false;
    Policy.PrintAsCanonical = true;
    llvm::raw_string_ostream ParmListWithNamesOstream{ParmListWithNames};
    for (ParmVarDecl *Param : K.SyclKernel->parameters()) {
      if (FirstParam)
        FirstParam = false;
      else {
        ParmList += ", ";
        ParmListWithNamesOstream << ", ";
      }
      if (Param->isParameterPack()) {
        ParmListWithNamesOstream << "Args... args";
        ParmList += "Args ...";
      } else {
        Policy.SuppressTagKeyword = true;
        Param->getType().print(ParmListWithNamesOstream, Policy);
        Policy.SuppressTagKeyword = false;
        ParmListWithNamesOstream << " " << Param->getNameAsString();
        ParmList += Param->getType().getCanonicalType().getAsString(Policy);
      }
    }
    ParmListWithNamesOstream.flush();
    FunctionTemplateDecl *FTD = K.SyclKernel->getPrimaryTemplate();
    Policy.PrintAsCanonical = false;
    Policy.SuppressDefinition = true;
    Policy.PolishForDeclaration = true;
    Policy.FullyQualifiedName = true;
    Policy.EnforceScopeForElaboratedTypes = true;
    Policy.UseFullyQualifiedEnumerators = true;

    // Now we need to print the declaration of the kernel itself.
    // Example:
    // template <typename T, typename = int> struct Arg {
    //   T val;
    // };
    // For the following free function kernel:
    // template <typename = T>
    // SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    //     (ext::oneapi::experimental::nd_range_kernel<1>))
    // void foo(Arg<int> arg) {}
    // Integration header must contain the following declaration:
    // template <typename>
    // void foo(Arg<int, int> arg);
    // SuppressDefaultTemplateArguments is a downstream addition that suppresses
    // default template arguments in the function declaration. It should be set
    // to true to emit function declaration that won't cause any compilation
    // errors when present in the integration header.
    // To print Arg<int, int> in the function declaration and shim functions we
    // need to disable default arguments printing suppression via community flag
    // SuppressDefaultTemplateArgs, otherwise they will be suppressed even for
    // canonical types or if even written in the original source code.
    Policy.SuppressDefaultTemplateArguments = true;
    // EnforceDefaultTemplateArgs is a downstream addition that forces printing
    // template arguments that match default template arguments while printing
    // template-ids, even if the source code doesn't reference them.
    Policy.EnforceDefaultTemplateArgs = true;
    FreeFunctionPrinter FFPrinter(O, Policy);
    if (FTD) {
      FFPrinter.printFreeFunctionDeclaration(FTD);
      if (const auto kind = K.SyclKernel->getTemplateSpecializationKind();
          K.SyclKernel->isFunctionTemplateSpecialization() &&
          kind == TSK_ExplicitSpecialization)
        FFPrinter.printFreeFunctionDeclaration(K.SyclKernel, ParmListWithNames);
    } else {
      FFPrinter.printFreeFunctionDeclaration(K.SyclKernel, ParmListWithNames);
    }

    FFPrinter.printFreeFunctionShim(K.SyclKernel, ShimCounter, ParmList);
    O << ";\n";
    O << "}\n";
    FFPrinter.printFreeFunctionKernelInfo(ShimCounter, K.Params.size(), K.Name);
    Policy.SuppressDefaultTemplateArgs = true;
    Policy.EnforceDefaultTemplateArgs = false;

    // Generate is_kernel, is_single_task_kernel and nd_range_kernel functions.
    O << "namespace sycl {\n";
    O << "template <>\n";
    O << "struct ext::oneapi::experimental::is_kernel<__sycl_shim"
      << ShimCounter << "()";
    O << "> {\n";
    O << "  static constexpr bool value = true;\n";
    O << "};\n";
    int Dim = getFreeFunctionRangeDim(S, K.SyclKernel);
    O << "template <>\n";
    if (Dim > 0)
      O << "struct ext::oneapi::experimental::is_nd_range_kernel<__sycl_shim"
        << ShimCounter << "(), " << Dim;
    else
      O << "struct "
           "ext::oneapi::experimental::is_single_task_kernel<__sycl_shim"
        << ShimCounter << "()";
    O << "> {\n";
    O << "  static constexpr bool value = true;\n";
    O << "};\n";
    O << "}\n";
    ++ShimCounter;
  }

  if (FreeFunctionCount > 0) {
    O << "\n#include <sycl/kernel_bundle.hpp>\n";
    O << "#include <sycl/detail/kernel_global_info.hpp>\n";
    O << "namespace sycl {\n";
    O << "inline namespace _V1 {\n";
    O << "namespace detail {\n";
    O << "struct GlobalMapUpdater {\n";
    O << "  GlobalMapUpdater() {\n";
    O << "    sycl::detail::free_function_info_map::add("
      << "sycl::detail::kernel_names, sycl::detail::kernel_args_sizes, "
      << KernelDescs.size() << ");\n";
    O << "  }\n";
    O << "};\n";
    O << "static GlobalMapUpdater updater;\n";
    O << "} // namespace detail\n";
    O << "} // namespace _V1\n";
    O << "} // namespace sycl\n";
  }
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
                                        bool IsUnnamedKernel, int64_t ObjSize) {
  KernelDescs.emplace_back(SyclKernel, KernelNameType, KernelLocation,
                           IsESIMDKernel, IsUnnamedKernel, ObjSize);
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

SYCLIntegrationHeader::SYCLIntegrationHeader(SemaSYCL &S) : S(S) {}

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
  if (!SemaSYCL::isSyclType(VD->getType(), SYCLTypeAttr::specialization_id) &&
      !SemaSYCL::isSyclType(VD->getType(), SYCLTypeAttr::host_pipe) &&
      !S.isTypeDecoratedWithDeclAttribute<SYCLDeviceGlobalAttr>(
          VD->getType())) {
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
  // let an error happen during host compilation. To avoid multiple entries for
  // redeclarations, variables with external storage are omitted.
  if (VD->hasLocalStorage() || VD->isLocalVarDeclOrParm() ||
      VD->hasExternalStorage())
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
                      const DeclContext *DC, std::string &NameForLastShim,
                      PrintingPolicy &Policy) {
  if (DC->isTranslationUnit()) {
    NameForLastShim = "::" + NameForLastShim;
    return;
  }

  const auto *CurDecl = cast<Decl>(DC)->getCanonicalDecl();

  // We skip linkage decls, since they don't modify the Qualified name.
  if (const auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(CurDecl)) {
    std::string TemplatedName;
    llvm::raw_string_ostream Stream(TemplatedName);
    CTSD->getNameForDiagnostic(Stream, Policy, false);
    Stream.flush();
    NameForLastShim = TemplatedName + "::" + NameForLastShim;
  } else if (const auto *RD = dyn_cast<RecordDecl>(CurDecl)) {
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

  EmitShims(OS, ShimCounter, CurDecl->getDeclContext(), NameForLastShim,
            Policy);
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

  EmitShims(OS, ShimCounter, VD->getDeclContext(), RelativeName, Policy);
  return RelativeName;
}

bool SYCLIntegrationFooter::emit(raw_ostream &OS) {
  PrintingPolicy Policy{S.getLangOpts()};
  Policy.adjustForCPlusPlusFwdDecl();
  Policy.SuppressTypedefs = true;
  Policy.SuppressUnwrittenScope = true;
  Policy.PrintAsCanonical = true;

  llvm::SmallSet<const VarDecl *, 8> Visited;
  bool EmittedFirstSpecConstant = false;
  bool DeviceGlobalsEmitted = false;
  bool HostPipesEmitted = false;

  // Used to uniquely name the 'shim's as we generate the names in each
  // anonymous namespace.
  unsigned ShimCounter = 0;

  std::string DeviceGlobalsBuf;
  llvm::raw_string_ostream DeviceGlobOS(DeviceGlobalsBuf);
  std::string HostPipesBuf;
  llvm::raw_string_ostream HostPipesOS(HostPipesBuf);
  for (const VarDecl *VD : GlobalVars) {
    VD = VD->getCanonicalDecl();

    // Skip if this isn't a SpecIdType, DeviceGlobal, or HostPipe.  This 
    // can happen if it was a deduced type.
    if (!SemaSYCL::isSyclType(VD->getType(), SYCLTypeAttr::specialization_id) &&
        !SemaSYCL::isSyclType(VD->getType(), SYCLTypeAttr::host_pipe) &&
        !S.isTypeDecoratedWithDeclAttribute<SYCLDeviceGlobalAttr>(
            VD->getType()))
      continue;

    // Skip if we've already visited this.
    if (llvm::find(Visited, VD) != Visited.end())
      continue;

    // We only want to emit the #includes if we have a variable that needs
    // them, so emit this one on the first time through the loop.
    if (!EmittedFirstSpecConstant && !DeviceGlobalsEmitted && !HostPipesEmitted)
      OS << "#include <sycl/detail/defines_elementary.hpp>\n";

    Visited.insert(VD);
    std::string TopShim = EmitShims(OS, ShimCounter, Policy, VD);
    if (S.isTypeDecoratedWithDeclAttribute<SYCLDeviceGlobalAttr>(
            VD->getType())) {
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
    } else if (SemaSYCL::isSyclType(VD->getType(), SYCLTypeAttr::host_pipe)) {
      HostPipesEmitted = true;
      HostPipesOS << "host_pipe_map::add(";
      HostPipesOS << "(void *)&";
      if (VD->isInAnonymousNamespace()) {
        HostPipesOS << TopShim;
      } else {
        HostPipesOS << "::";
        VD->getNameForDiagnostic(HostPipesOS, Policy, true);
      }
      HostPipesOS << ", \"";
      HostPipesOS << SYCLUniqueStableIdExpr::ComputeName(S.getASTContext(),
                                                         VD);
      HostPipesOS << "\");\n";
    } else {
      EmittedFirstSpecConstant = true;
      OS << "namespace sycl {\n";
      OS << "inline namespace _V1 {\n";
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
      OS << "} // namespace _V1\n";
      OS << "} // namespace sycl\n";
    }
  }

  if (EmittedFirstSpecConstant)
    OS << "#include <sycl/detail/spec_const_integration.hpp>\n";

  if (DeviceGlobalsEmitted) {
    OS << "#include <sycl/detail/device_global_map.hpp>\n";
    DeviceGlobOS.flush();
    OS << "namespace sycl::detail {\n";
    // Supress the old-style case diagnostic that clang generates
    // for the construct below in DeviceGlobalsBuf.
    EmitPragmaDiagnosticPush(OS, "-Wold-style-cast");
    OS << "namespace {\n";
    OS << "__sycl_device_global_registration::__sycl_device_global_"
          "registration() noexcept {\n";
    OS << DeviceGlobalsBuf;
    OS << "}\n";
    OS << "} // namespace (unnamed)\n";
    EmitPragmaDiagnosticPop(OS);
    OS << "} // namespace sycl::detail\n";

    S.getSyclIntegrationHeader().addDeviceGlobalRegistration();
  }

  if (HostPipesEmitted) {
    OS << "#include <sycl/detail/host_pipe_map.hpp>\n";
    HostPipesOS.flush();
    OS << "namespace sycl::detail {\n";
    // Supress the old-style case diagnostic that clang generates
    // for the construct below in HostPipesBuf.
    EmitPragmaDiagnosticPush(OS, "-Wold-style-cast");
    OS << "namespace {\n";
    OS << "__sycl_host_pipe_registration::__sycl_host_pipe_"
          "registration() noexcept {\n";
    OS << HostPipesBuf;
    OS << "}\n";
    OS << "} // namespace (unnamed)\n";
    EmitPragmaDiagnosticPop(OS);
    OS << "} // namespace sycl::detail\n";

    S.getSyclIntegrationHeader().addHostPipeRegistration();
  }

  return true;
}

ExprResult SemaSYCL::BuildUniqueStableIdExpr(SourceLocation OpLoc,
                                             SourceLocation LParen,
                                             SourceLocation RParen, Expr *E) {
  if (!E->isInstantiationDependent()) {
    // Special handling to get us better error messages for a member variable.
    if (auto *ME = dyn_cast<MemberExpr>(E->IgnoreUnlessSpelledInSource())) {
      if (isa<FieldDecl>(ME->getMemberDecl()))
        Diag(E->getExprLoc(), diag::err_unique_stable_id_global_storage);
      else
        Diag(E->getExprLoc(), diag::err_unique_stable_id_expected_var);
      return ExprError();
    }

    auto *DRE = dyn_cast<DeclRefExpr>(E->IgnoreUnlessSpelledInSource());

    if (!DRE || !isa_and_nonnull<VarDecl>(DRE->getDecl())) {
      Diag(E->getExprLoc(), diag::err_unique_stable_id_expected_var);
      return ExprError();
    }

    auto *Var = cast<VarDecl>(DRE->getDecl());

    if (!Var->hasGlobalStorage()) {
      Diag(E->getExprLoc(), diag::err_unique_stable_id_global_storage);
      return ExprError();
    }
  }

  return SYCLUniqueStableIdExpr::Create(getASTContext(), OpLoc, LParen, RParen,
                                        E);
}

ExprResult SemaSYCL::ActOnUniqueStableIdExpr(SourceLocation OpLoc,
                                             SourceLocation LParen,
                                             SourceLocation RParen, Expr *E) {
  return BuildUniqueStableIdExpr(OpLoc, LParen, RParen, E);
}

ExprResult SemaSYCL::BuildUniqueStableNameExpr(SourceLocation OpLoc,
                                               SourceLocation LParen,
                                               SourceLocation RParen,
                                               TypeSourceInfo *TSI) {
  return SYCLUniqueStableNameExpr::Create(getASTContext(), OpLoc, LParen,
                                          RParen, TSI);
}

ExprResult SemaSYCL::ActOnUniqueStableNameExpr(SourceLocation OpLoc,
                                               SourceLocation LParen,
                                               SourceLocation RParen,
                                               ParsedType ParsedTy) {
  TypeSourceInfo *TSI = nullptr;
  QualType Ty = SemaRef.GetTypeFromParser(ParsedTy, &TSI);

  if (Ty.isNull())
    return ExprError();
  if (!TSI)
    TSI = getASTContext().getTrivialTypeSourceInfo(Ty, LParen);

  return BuildUniqueStableNameExpr(OpLoc, LParen, RParen, TSI);
}

void SemaSYCL::performSYCLDelayedAttributesAnalaysis(const FunctionDecl *FD) {
  if (SYCLKernelFunctions.contains(FD))
    return;

  for (const auto *KernelAttr : std::vector<AttributeCommonInfo *>{
           FD->getAttr<SYCLReqdWorkGroupSizeAttr>(),
           FD->getAttr<IntelReqdSubGroupSizeAttr>(),
           FD->getAttr<SYCLWorkGroupSizeHintAttr>(),
           FD->getAttr<VecTypeHintAttr>()}) {
    if (KernelAttr)
      Diag(KernelAttr->getLoc(),
           diag::warn_sycl_incorrect_use_attribute_non_kernel_function)
          << KernelAttr;
  }
}

void SemaSYCL::handleKernelEntryPointAttr(Decl *D, const ParsedAttr &AL) {
  ParsedType PT = AL.getTypeArg();
  TypeSourceInfo *TSI = nullptr;
  (void)SemaRef.GetTypeFromParser(PT, &TSI);
  assert(TSI && "no type source info for attribute argument");
  D->addAttr(::new (SemaRef.Context)
                 SYCLKernelEntryPointAttr(SemaRef.Context, AL, TSI));
}

// Given a potentially qualified type, SourceLocationForUserDeclaredType()
// returns the source location of the canonical declaration of the unqualified
// desugared user declared type, if any. For non-user declared types, an
// invalid source location is returned. The intended usage of this function
// is to identify an appropriate source location, if any, for a
// "entity declared here" diagnostic note.
static SourceLocation SourceLocationForUserDeclaredType(QualType QT) {
  SourceLocation Loc;
  const Type *T = QT->getUnqualifiedDesugaredType();
  if (const TagType *TT = dyn_cast<TagType>(T))
    Loc = TT->getDecl()->getLocation();
  else if (const ObjCInterfaceType *ObjCIT = dyn_cast<ObjCInterfaceType>(T))
    Loc = ObjCIT->getDecl()->getLocation();
  return Loc;
}

static bool CheckSYCLKernelName(Sema &S, SourceLocation Loc,
                                QualType KernelName) {
  assert(!KernelName->isDependentType());

  if (!KernelName->isStructureOrClassType()) {
    // SYCL 2020 section 5.2, "Naming of kernels", only requires that the
    // kernel name be a C++ typename. However, the definition of "kernel name"
    // in the glossary states that a kernel name is a class type. Neither
    // section explicitly states whether the kernel name type can be
    // cv-qualified. For now, kernel name types are required to be class types
    // and that they may be cv-qualified. The following issue requests
    // clarification from the SYCL WG.
    //   https://github.com/KhronosGroup/SYCL-Docs/issues/568
    S.Diag(Loc, diag::warn_sycl_kernel_name_not_a_class_type) << KernelName;
    SourceLocation DeclTypeLoc = SourceLocationForUserDeclaredType(KernelName);
    if (DeclTypeLoc.isValid())
      S.Diag(DeclTypeLoc, diag::note_entity_declared_at) << KernelName;
    return true;
  }

  return false;
}

void SemaSYCL::CheckSYCLEntryPointFunctionDecl(FunctionDecl *FD) {
  // Ensure that all attributes present on the declaration are consistent
  // and warn about any redundant ones.
  SYCLKernelEntryPointAttr *SKEPAttr = nullptr;
  for (auto *SAI : FD->specific_attrs<SYCLKernelEntryPointAttr>()) {
    if (!SKEPAttr) {
      SKEPAttr = SAI;
      continue;
    }
    if (!getASTContext().hasSameType(SAI->getKernelName(),
                                     SKEPAttr->getKernelName())) {
      Diag(SAI->getLocation(), diag::err_sycl_entry_point_invalid_redeclaration)
          << SAI->getKernelName() << SKEPAttr->getKernelName();
      Diag(SKEPAttr->getLocation(), diag::note_previous_attribute);
      SAI->setInvalidAttr();
    } else {
      Diag(SAI->getLocation(),
           diag::warn_sycl_entry_point_redundant_declaration);
      Diag(SKEPAttr->getLocation(), diag::note_previous_attribute);
    }
  }
  assert(SKEPAttr && "Missing sycl_kernel_entry_point attribute");

  // Ensure the kernel name type is valid.
  if (!SKEPAttr->getKernelName()->isDependentType() &&
      CheckSYCLKernelName(SemaRef, SKEPAttr->getLocation(),
                          SKEPAttr->getKernelName()))
    SKEPAttr->setInvalidAttr();

  // Ensure that an attribute present on the previous declaration
  // matches the one on this declaration.
  FunctionDecl *PrevFD = FD->getPreviousDecl();
  if (PrevFD && !PrevFD->isInvalidDecl()) {
    const auto *PrevSKEPAttr = PrevFD->getAttr<SYCLKernelEntryPointAttr>();
    if (PrevSKEPAttr && !PrevSKEPAttr->isInvalidAttr()) {
      if (!getASTContext().hasSameType(SKEPAttr->getKernelName(),
                                       PrevSKEPAttr->getKernelName())) {
        Diag(SKEPAttr->getLocation(),
             diag::err_sycl_entry_point_invalid_redeclaration)
            << SKEPAttr->getKernelName() << PrevSKEPAttr->getKernelName();
        Diag(PrevSKEPAttr->getLocation(), diag::note_previous_decl) << PrevFD;
        SKEPAttr->setInvalidAttr();
      }
    }
  }

  if (const auto *MD = dyn_cast<CXXMethodDecl>(FD)) {
    if (!MD->isStatic()) {
      Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
          << /*non-static member function*/ 0;
      SKEPAttr->setInvalidAttr();
    }
  }

  if (FD->isVariadic()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << /*variadic function*/ 1;
    SKEPAttr->setInvalidAttr();
  }

  if (FD->isDefaulted()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << /*defaulted function*/ 3;
    SKEPAttr->setInvalidAttr();
  } else if (FD->isDeleted()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << /*deleted function*/ 2;
    SKEPAttr->setInvalidAttr();
  }

  if (FD->isConsteval()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << /*consteval function*/ 5;
    SKEPAttr->setInvalidAttr();
  } else if (FD->isConstexpr()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << /*constexpr function*/ 4;
    SKEPAttr->setInvalidAttr();
  }

  if (FD->isNoReturn()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_invalid)
        << /*function declared with the 'noreturn' attribute*/ 6;
    SKEPAttr->setInvalidAttr();
  }

  if (FD->getReturnType()->isUndeducedType()) {
    Diag(SKEPAttr->getLocation(),
         diag::err_sycl_entry_point_deduced_return_type);
    SKEPAttr->setInvalidAttr();
  } else if (!FD->getReturnType()->isDependentType() &&
             !FD->getReturnType()->isVoidType()) {
    Diag(SKEPAttr->getLocation(), diag::err_sycl_entry_point_return_type);
    SKEPAttr->setInvalidAttr();
  }

  if (!FD->isInvalidDecl() && !FD->isTemplated() &&
      !SKEPAttr->isInvalidAttr()) {
    const SYCLKernelInfo *SKI =
        getASTContext().findSYCLKernelInfo(SKEPAttr->getKernelName());
    if (SKI) {
      if (!declaresSameEntity(FD, SKI->getKernelEntryPointDecl())) {
        // FIXME: This diagnostic should include the origin of the kernel
        // FIXME: names; not just the locations of the conflicting declarations.
        Diag(FD->getLocation(), diag::err_sycl_kernel_name_conflict);
        Diag(SKI->getKernelEntryPointDecl()->getLocation(),
             diag::note_previous_declaration);
        SKEPAttr->setInvalidAttr();
      }
    } else {
      getASTContext().registerSYCLEntryPointFunction(FD);
    }
  }
}

namespace {

// The body of a function declared with the [[sycl_kernel_entry_point]]
// attribute is cloned and transformed to substitute references to the original
// function parameters with references to replacement variables that stand in
// for SYCL kernel parameters or local variables that reconstitute a decomposed
// SYCL kernel argument.
class OutlinedFunctionDeclBodyInstantiator
    : public TreeTransform<OutlinedFunctionDeclBodyInstantiator> {
public:
  using ParmDeclMap = llvm::DenseMap<ParmVarDecl *, VarDecl *>;

  OutlinedFunctionDeclBodyInstantiator(Sema &S, ParmDeclMap &M)
      : TreeTransform<OutlinedFunctionDeclBodyInstantiator>(S), SemaRef(S),
        MapRef(M) {}

  // A new set of AST nodes is always required.
  bool AlwaysRebuild() { return true; }

  // Transform ParmVarDecl references to the supplied replacement variables.
  ExprResult TransformDeclRefExpr(DeclRefExpr *DRE) {
    const ParmVarDecl *PVD = dyn_cast<ParmVarDecl>(DRE->getDecl());
    if (PVD) {
      ParmDeclMap::iterator I = MapRef.find(PVD);
      if (I != MapRef.end()) {
        VarDecl *VD = I->second;
        assert(SemaRef.getASTContext().hasSameUnqualifiedType(PVD->getType(),
                                                              VD->getType()));
        assert(!VD->getType().isMoreQualifiedThan(PVD->getType(),
                                                  SemaRef.getASTContext()));
        VD->setIsUsed();
        return DeclRefExpr::Create(
            SemaRef.getASTContext(), DRE->getQualifierLoc(),
            DRE->getTemplateKeywordLoc(), VD, false, DRE->getNameInfo(),
            DRE->getType(), DRE->getValueKind());
      }
    }
    return DRE;
  }

private:
  Sema &SemaRef;
  ParmDeclMap &MapRef;
};

} // unnamed namespace

StmtResult SemaSYCL::BuildSYCLKernelCallStmt(FunctionDecl *FD,
                                             CompoundStmt *Body) {
  assert(!FD->isInvalidDecl());
  assert(!FD->isTemplated());
  assert(FD->hasPrototype());

  const auto *SKEPAttr = FD->getAttr<SYCLKernelEntryPointAttr>();
  assert(SKEPAttr && "Missing sycl_kernel_entry_point attribute");
  assert(!SKEPAttr->isInvalidAttr() &&
         "sycl_kernel_entry_point attribute is invalid");

  // Ensure that the kernel name was previously registered and that the
  // stored declaration matches.
  const SYCLKernelInfo &SKI =
      getASTContext().getSYCLKernelInfo(SKEPAttr->getKernelName());
  assert(declaresSameEntity(SKI.getKernelEntryPointDecl(), FD) &&
         "SYCL kernel name conflict");
  (void)SKI;

  using ParmDeclMap = OutlinedFunctionDeclBodyInstantiator::ParmDeclMap;
  ParmDeclMap ParmMap;

  assert(SemaRef.CurContext == FD);
  OutlinedFunctionDecl *OFD =
      OutlinedFunctionDecl::Create(getASTContext(), FD, FD->getNumParams());
  unsigned i = 0;
  for (ParmVarDecl *PVD : FD->parameters()) {
    ImplicitParamDecl *IPD = ImplicitParamDecl::Create(
        getASTContext(), OFD, SourceLocation(), PVD->getIdentifier(),
        PVD->getType(), ImplicitParamKind::Other);
    OFD->setParam(i, IPD);
    ParmMap[PVD] = IPD;
    ++i;
  }

  OutlinedFunctionDeclBodyInstantiator OFDBodyInstantiator(SemaRef, ParmMap);
  Stmt *OFDBody = OFDBodyInstantiator.TransformStmt(Body).get();
  OFD->setBody(OFDBody);
  OFD->setNothrow();
  Stmt *NewBody = new (getASTContext()) SYCLKernelCallStmt(Body, OFD);

  return NewBody;
}
