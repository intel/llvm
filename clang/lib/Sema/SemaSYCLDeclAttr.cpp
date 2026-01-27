//===- SemaSYCLDeclAttr.cpp - Semantic Analysis for SYCL attributes -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This implements Semantic Analysis for SYCL attributes.
//===----------------------------------------------------------------------===//

#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/Attr.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaSYCL.h"

using namespace clang;

void SemaSYCL::handleKernelAttr(Decl *D, const ParsedAttr &AL) {
  // The 'sycl_kernel' attribute applies only to function templates.
  const auto *FD = cast<FunctionDecl>(D);
  const FunctionTemplateDecl *FT = FD->getDescribedFunctionTemplate();
  assert(FT && "Function template is expected");

  // Function template must have at least two template parameters so it
  // can be used in OpenCL kernel generation.
  const TemplateParameterList *TL = FT->getTemplateParameters();
  if (TL->size() < 2) {
    Diag(FT->getLocation(), diag::warn_sycl_kernel_num_of_template_params);
    return;
  }

  // The first two template parameters must be typenames.
  for (unsigned I = 0; I < 2 && I < TL->size(); ++I) {
    const NamedDecl *TParam = TL->getParam(I);
    if (isa<NonTypeTemplateParmDecl>(TParam)) {
      Diag(FT->getLocation(),
           diag::warn_sycl_kernel_invalid_template_param_type);
      return;
    }
  }

  // Function must have at least one parameter.
  if (getFunctionOrMethodNumParams(D) < 1) {
    Diag(FT->getLocation(), diag::warn_sycl_kernel_num_of_function_params);
    return;
  }

  // Function must return void.
  QualType RetTy = getFunctionOrMethodResultType(D);
  if (!RetTy->isVoidType()) {
    Diag(FT->getLocation(), diag::warn_sycl_kernel_return_type);
    return;
  }

  handleSimpleAttribute<SYCLKernelAttr>(*this, D, AL);
}

// Returns a DupArgResult value; Same means the args have the same value,
// Different means the args do not have the same value, and Unknown means that
// the args cannot (yet) be compared.
enum class DupArgResult { Unknown, Same, Different };
static DupArgResult areArgValuesIdentical(const Expr *LHS, const Expr *RHS) {
  // If both operands are nullptr they are unspecified and are considered the
  // same.
  if (!LHS && !RHS)
    return DupArgResult::Same;

  // Otherwise, if either operand is nullptr they are considered different.
  if (!LHS || !RHS)
    return DupArgResult::Different;

  // Otherwise, if either operand is still value dependent, we can't test
  // anything.
  const auto *LHSCE = dyn_cast<ConstantExpr>(LHS);
  const auto *RHSCE = dyn_cast<ConstantExpr>(RHS);
  if (!LHSCE || !RHSCE)
    return DupArgResult::Unknown;

  // Otherwise, test that the values.
  return LHSCE->getResultAsAPSInt() == RHSCE->getResultAsAPSInt()
             ? DupArgResult::Same
             : DupArgResult::Different;
}

// Returns true if any of the specified dimensions (X,Y,Z) differ between the
// arguments.
bool SemaSYCL::anyWorkGroupSizesDiffer(const Expr *LHSXDim, const Expr *LHSYDim,
                                       const Expr *LHSZDim, const Expr *RHSXDim,
                                       const Expr *RHSYDim,
                                       const Expr *RHSZDim) {
  DupArgResult Results[] = {areArgValuesIdentical(LHSXDim, RHSXDim),
                            areArgValuesIdentical(LHSYDim, RHSYDim),
                            areArgValuesIdentical(LHSZDim, RHSZDim)};
  return llvm::is_contained(Results, DupArgResult::Different);
}

// Returns true if all of the specified dimensions (X,Y,Z) are the same between
// the arguments.
bool SemaSYCL::allWorkGroupSizesSame(const Expr *LHSXDim, const Expr *LHSYDim,
                                     const Expr *LHSZDim, const Expr *RHSXDim,
                                     const Expr *RHSYDim, const Expr *RHSZDim) {
  DupArgResult Results[] = {areArgValuesIdentical(LHSXDim, RHSXDim),
                            areArgValuesIdentical(LHSYDim, RHSYDim),
                            areArgValuesIdentical(LHSZDim, RHSZDim)};
  return llvm::all_of(Results,
                      [](DupArgResult V) { return V == DupArgResult::Same; });
}

// Helper to get CudaArch.
OffloadArch SemaSYCL::getOffloadArch(const TargetInfo &TI) {
  if (!TI.getTriple().isNVPTX())
    llvm_unreachable("getOffloadArch is only valid for NVPTX triple");
  auto &TO = TI.getTargetOpts();
  return StringToOffloadArch(TO.CPU);
}

bool SemaSYCL::hasDependentExpr(Expr **Exprs, const size_t ExprsSize) {
  return std::any_of(Exprs, Exprs + ExprsSize, [](const Expr *E) {
    return E->isValueDependent() || E->isTypeDependent();
  });
}

void SemaSYCL::checkDeprecatedSYCLAttributeSpelling(const ParsedAttr &A,
                                                    StringRef NewName) {
  // Additionally, diagnose deprecated [[intel::reqd_sub_group_size]] spelling
  if (A.getKind() == ParsedAttr::AT_IntelReqdSubGroupSize && A.getScopeName() &&
      A.getScopeName()->isStr("intel")) {
    diagnoseDeprecatedAttribute(A, "sycl", "reqd_sub_group_size");
    return;
  }

  // Diagnose SYCL 2020 spellings in later SYCL modes.
  if (getLangOpts().getSYCLVersion() >= LangOptions::SYCL_2020) {
    // All attributes in the cl vendor namespace are deprecated in favor of a
    // name in the sycl namespace as of SYCL 2020.
    if (A.hasScope() && A.getScopeName()->isStr("cl")) {
      diagnoseDeprecatedAttribute(A, "sycl", NewName);
      return;
    }

    // All GNU-style spellings are deprecated in favor of a C++-style spelling.
    if (A.getSyntax() == ParsedAttr::AS_GNU) {
      // Note: we cannot suggest an automatic fix-it because GNU-style
      // spellings can appear in locations that are not valid for a C++-style
      // spelling, and the attribute could be part of an attribute list within
      // a single __attribute__ specifier. Just tell the user it's deprecated
      // manually.
      //
      // This currently assumes that the GNU-style spelling is the same as the
      // SYCL 2020 spelling (sans the vendor namespace).
      Diag(A.getLoc(), diag::warn_attribute_spelling_deprecated)
          << "'" + A.getNormalizedFullName() + "'";
      Diag(A.getLoc(), diag::note_spelling_suggestion)
          << "'[[sycl::" + A.getNormalizedFullName() + "]]'";
      return;
    }
  }
}

void SemaSYCL::diagnoseDeprecatedAttribute(const ParsedAttr &A,
                                           StringRef NewScope,
                                           StringRef NewName) {
  assert((!NewName.empty() || !NewScope.empty()) &&
         "Deprecated attribute with no new scope or name?");
  Diag(A.getLoc(), diag::warn_attribute_spelling_deprecated)
      << "'" + A.getNormalizedFullName() + "'";

  FixItHint Fix;
  std::string NewFullName;
  if (NewScope.empty() && !NewName.empty()) {
    // Only have a new name.
    Fix = FixItHint::CreateReplacement(A.getLoc(), NewName);
    NewFullName =
        ((A.hasScope() ? A.getScopeName()->getName() : StringRef("")) +
         "::" + NewName)
            .str();
  } else if (NewName.empty() && !NewScope.empty()) {
    // Only have a new scope.
    Fix = FixItHint::CreateReplacement(A.getScopeLoc(), NewScope);
    NewFullName = (NewScope + "::" + A.getAttrName()->getName()).str();
  } else {
    // Have both a new name and a new scope.
    NewFullName = (NewScope + "::" + NewName).str();
    Fix = FixItHint::CreateReplacement(A.getRange(), NewFullName);
  }

  Diag(A.getLoc(), diag::note_spelling_suggestion)
      << "'" + NewFullName + "'" << Fix;
}

// Checks if FPGA memory attributes apply on valid variables.
// Returns true if an error occured.
bool SemaSYCL::checkValidFPGAMemoryAttributesVar(Decl *D) {
  // Check for SYCL device compilation context.
  if (!getLangOpts().SYCLIsDevice) {
    return false;
  }

  const auto *VD = dyn_cast<VarDecl>(D);
  if (!VD)
    return false;

  // Exclude implicit parameters and non-type template parameters.
  if (VD->getKind() == Decl::ImplicitParam ||
      VD->getKind() == Decl::NonTypeTemplateParm)
    return false;

  // Check for non-static data member.
  if (isa<FieldDecl>(D))
    return false;

  // Check for SYCL device global attribute decoration.
  if (isTypeDecoratedWithDeclAttribute<SYCLDeviceGlobalAttr>(VD->getType()))
    return false;

  // Check for constant variables and variables in the OpenCL constant
  // address space.
  if (VD->getType().isConstQualified() ||
      VD->getType().getAddressSpace() == LangAS::opencl_constant)
    return false;

  // Check for static storage class or local storage.
  if (VD->getStorageClass() == SC_Static || VD->hasLocalStorage())
    return false;

  return true;
}

// Handles reqd_work_group_size.
// If the 'reqd_work_group_size' attribute is specified on a declaration along
// with 'num_simd_work_items' attribute, the required work group size specified
// by 'num_simd_work_items' attribute must evenly divide the index that
// increments fastest in the 'reqd_work_group_size' attribute.
//
// The arguments to reqd_work_group_size are ordered based on which index
// increments the fastest. In OpenCL, the first argument is the index that
// increments the fastest, and in SYCL, the last argument is the index that
// increments the fastest.
//
// __attribute__((reqd_work_group_size)) follows the OpenCL rules in OpenCL
// mode. All spellings of reqd_work_group_size attribute (regardless of
// syntax used) follow the SYCL rules when in SYCL mode.
bool SemaSYCL::checkWorkGroupSize(const Expr *NSWIValue, const Expr *RWGSXDim,
                                  const Expr *RWGSYDim, const Expr *RWGSZDim) {
  // If any of the operand is still value dependent, we can't test anything.
  const auto *NSWIValueExpr = dyn_cast<ConstantExpr>(NSWIValue);
  const auto *RWGSXDimExpr = dyn_cast<ConstantExpr>(RWGSXDim);

  if (!NSWIValueExpr || !RWGSXDimExpr)
    return false;

  // Y and Z may be optional so we allow them to be null and consider them
  // dependent if the original epxression was not null while the result of the
  // cast is.
  const auto *RWGSYDimExpr = dyn_cast_or_null<ConstantExpr>(RWGSYDim);
  const auto *RWGSZDimExpr = dyn_cast_or_null<ConstantExpr>(RWGSZDim);

  if ((!RWGSYDimExpr && RWGSYDim) || (!RWGSZDimExpr && RWGSZDim))
    return false;

  // Otherwise, check which argument increments the fastest.
  const ConstantExpr *LastRWGSDimExpr =
      RWGSZDim ? RWGSZDimExpr : (RWGSYDim ? RWGSYDimExpr : RWGSXDimExpr);
  unsigned WorkGroupSize = LastRWGSDimExpr->getResultAsAPSInt().getZExtValue();

  // Check if the required work group size specified by 'num_simd_work_items'
  // attribute evenly divides the index that increments fastest in the
  // 'reqd_work_group_size' attribute.
  return WorkGroupSize % NSWIValueExpr->getResultAsAPSInt().getZExtValue() != 0;
}

// Checks correctness of mutual usage of different work_group_size attributes:
// reqd_work_group_size and max_work_group_size.
//
// If the 'reqd_work_group_size' attribute is specified on a declaration along
// with 'max_work_group_size' attribute, check to see if values of
// 'reqd_work_group_size' attribute arguments are equal to or less than values
// of 'max_work_group_size' attribute arguments.
//
// The arguments to reqd_work_group_size are ordered based on which index
// increments the fastest. In OpenCL, the first argument is the index that
// increments the fastest, and in SYCL, the last argument is the index that
// increments the fastest.
//
// __attribute__((reqd_work_group_size)) follows the OpenCL rules in OpenCL
// mode. All spellings of reqd_work_group_size attribute (regardless of
// syntax used) follow the SYCL rules when in SYCL mode.
bool SemaSYCL::checkMaxAllowedWorkGroupSize(
    const Expr *RWGSXDim, const Expr *RWGSYDim, const Expr *RWGSZDim,
    const Expr *MWGSXDim, const Expr *MWGSYDim, const Expr *MWGSZDim) {
  // If any of the operand is still value dependent, we can't test anything.
  const auto *RWGSXDimExpr = dyn_cast<ConstantExpr>(RWGSXDim);
  const auto *MWGSXDimExpr = dyn_cast<ConstantExpr>(MWGSXDim);
  const auto *MWGSYDimExpr = dyn_cast<ConstantExpr>(MWGSYDim);
  const auto *MWGSZDimExpr = dyn_cast<ConstantExpr>(MWGSZDim);

  if (!RWGSXDimExpr || !MWGSXDimExpr || !MWGSYDimExpr || !MWGSZDimExpr)
    return false;

  // Y and Z may be optional so we allow them to be null and consider them
  // dependent if the original epxression was not null while the result of the
  // cast is.
  const auto *RWGSYDimExpr = dyn_cast_or_null<ConstantExpr>(RWGSYDim);
  const auto *RWGSZDimExpr = dyn_cast_or_null<ConstantExpr>(RWGSZDim);

  if ((!RWGSYDimExpr && RWGSYDim) || (!RWGSZDimExpr && RWGSZDim))
    return false;

  // SYCL reorders arguments based on the dimensionality.
  // If we only have the X-dimension, there is no change to the expressions,
  // otherwise the last specified dimension acts as the first dimension in the
  // work-group size.
  const ConstantExpr *FirstRWGDimExpr = RWGSXDimExpr;
  const ConstantExpr *SecondRWGDimExpr = RWGSYDimExpr;
  const ConstantExpr *ThirdRWGDimExpr = RWGSZDimExpr;
  if (getLangOpts().SYCLIsDevice && RWGSYDim)
    std::swap(FirstRWGDimExpr, RWGSZDim ? ThirdRWGDimExpr : SecondRWGDimExpr);

  // Check if values of 'reqd_work_group_size' attribute arguments are greater
  // than values of 'max_work_group_size' attribute arguments.
  bool CheckFirstArgument =
      FirstRWGDimExpr->getResultAsAPSInt().getZExtValue() >
      MWGSZDimExpr->getResultAsAPSInt().getZExtValue();

  bool CheckSecondArgument =
      SecondRWGDimExpr && SecondRWGDimExpr->getResultAsAPSInt().getZExtValue() >
                              MWGSYDimExpr->getResultAsAPSInt().getZExtValue();

  bool CheckThirdArgument =
      ThirdRWGDimExpr && ThirdRWGDimExpr->getResultAsAPSInt().getZExtValue() >
                             MWGSXDimExpr->getResultAsAPSInt().getZExtValue();

  return CheckFirstArgument || CheckSecondArgument || CheckThirdArgument;
}

// Checks correctness of mutual usage of different work_group_size attributes:
// reqd_work_group_size, max_work_group_size, and max_global_work_dim.
//
// If [[intel::max_work_group_size(X, Y, Z)]] or
// [[sycl::reqd_work_group_size(X, Y, Z)]] or
// [[cl::reqd_work_group_size(X, Y, Z)]]
// or __attribute__((reqd_work_group_size)) attribute is specified on a
// declaration along with [[intel::max_global_work_dim()]] attribute, check to
// see if all arguments of 'max_work_group_size' or different spellings of
// 'reqd_work_group_size' attribute hold value 1 in case the argument of
// [[intel::max_global_work_dim()]] attribute value equals to 0.
bool SemaSYCL::areInvalidWorkGroupSizeAttrs(const Expr *MGValue,
                                            const Expr *XDim, const Expr *YDim,
                                            const Expr *ZDim) {
  // If any of the operand is still value dependent, we can't test anything.
  const auto *MGValueExpr = dyn_cast<ConstantExpr>(MGValue);
  const auto *XDimExpr = dyn_cast<ConstantExpr>(XDim);

  if (!MGValueExpr || !XDimExpr)
    return false;

  // Y and Z may be optional so we allow them to be null and consider them
  // dependent if the original epxression was not null while the result of the
  // cast is.
  const auto *YDimExpr = dyn_cast_or_null<ConstantExpr>(YDim);
  const auto *ZDimExpr = dyn_cast_or_null<ConstantExpr>(ZDim);

  if ((!YDimExpr && YDim) || (!ZDimExpr && ZDim))
    return false;

  // Otherwise, check if the attribute values are equal to one.
  // Y and Z dimensions are optional and are considered trivially 1 if
  // unspecified.
  return (MGValueExpr->getResultAsAPSInt() == 0 &&
          (XDimExpr->getResultAsAPSInt() != 1 ||
           (YDimExpr && YDimExpr->getResultAsAPSInt() != 1) ||
           (ZDimExpr && ZDimExpr->getResultAsAPSInt() != 1)));
}

bool isDeviceAspectType(const QualType Ty) {
  const EnumType *ET = Ty->getAs<EnumType>();
  if (!ET)
    return false;

  if (const auto *Attr = ET->getOriginalDecl()->getAttr<SYCLTypeAttr>())
    return Attr->getType() == SYCLTypeAttr::aspect;

  return false;
}

void SemaSYCL::addSYCLDeviceHasAttr(Decl *D, const AttributeCommonInfo &CI,
                                    Expr **Exprs, unsigned Size) {
  ASTContext &Context = getASTContext();
  SYCLDeviceHasAttr TmpAttr(Context, CI, Exprs, Size);
  SmallVector<Expr *, 5> Aspects;
  for (auto *E : TmpAttr.aspects())
    if (!isa<PackExpansionExpr>(E) && !isDeviceAspectType(E->getType()))
      Diag(E->getExprLoc(), diag::err_sycl_invalid_aspect_argument) << CI;

  if (const auto *ExistingAttr = D->getAttr<SYCLDeviceHasAttr>()) {
    Diag(CI.getLoc(), diag::warn_duplicate_attribute_exact) << CI;
    Diag(ExistingAttr->getLoc(), diag::note_previous_attribute);
    return;
  }

  D->addAttr(::new (Context) SYCLDeviceHasAttr(Context, CI, Exprs, Size));
}

void SemaSYCL::addSYCLUsesAspectsAttr(Decl *D, const AttributeCommonInfo &CI,
                                      Expr **Exprs, unsigned Size) {
  ASTContext &Context = getASTContext();
  SYCLUsesAspectsAttr TmpAttr(Context, CI, Exprs, Size);
  SmallVector<Expr *, 5> Aspects;
  for (auto *E : TmpAttr.aspects())
    if (!isDeviceAspectType(E->getType()))
      Diag(E->getExprLoc(), diag::err_sycl_invalid_aspect_argument) << CI;

  if (const auto *ExistingAttr = D->getAttr<SYCLUsesAspectsAttr>()) {
    Diag(CI.getLoc(), diag::warn_duplicate_attribute_exact) << CI;
    Diag(ExistingAttr->getLoc(), diag::note_previous_attribute);
    return;
  }

  D->addAttr(::new (Context) SYCLUsesAspectsAttr(Context, CI, Exprs, Size));
}

void SemaSYCL::addIntelReqdSubGroupSizeAttr(Decl *D,
                                            const AttributeCommonInfo &CI,
                                            Expr *E) {
  ASTContext &Context = getASTContext();
  if (!E->isValueDependent()) {
    // Validate that we have an integer constant expression and then store the
    // converted constant expression into the semantic attribute so that we
    // don't have to evaluate it again later.
    llvm::APSInt ArgVal;
    ExprResult Res = SemaRef.VerifyIntegerConstantExpression(E, &ArgVal);
    if (Res.isInvalid())
      return;
    E = Res.get();

    // This attribute requires a strictly positive value.
    if (ArgVal <= 0) {
      Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
          << CI << /*positive*/ 0;
      return;
    }
    auto &TI = Context.getTargetInfo();
    if (TI.getTriple().isNVPTX() && ArgVal != 32)
      Diag(E->getExprLoc(), diag::warn_reqd_sub_group_attribute_n)
          << ArgVal.getSExtValue() << TI.getTriple().getArchName() << 32;
    if (TI.getTriple().isAMDGPU()) {
      const auto HasWaveFrontSize64 =
          TI.getTargetOpts().FeatureMap["wavefrontsize64"];
      const auto HasWaveFrontSize32 =
          TI.getTargetOpts().FeatureMap["wavefrontsize32"];

      // CDNA supports only 64 wave front size, for those GPUs allow subgroup
      // size of 64. Some GPUs support both 32 and 64, for those (and the rest)
      // only allow 32. Warn on incompatible sizes.
      const auto SupportedWaveFrontSize =
          HasWaveFrontSize64 && !HasWaveFrontSize32 ? 64 : 32;
      if (ArgVal != SupportedWaveFrontSize)
        Diag(E->getExprLoc(), diag::warn_reqd_sub_group_attribute_n)
            << ArgVal.getSExtValue() << TI.getTriple().getArchName()
            << SupportedWaveFrontSize;
    }

    // Check to see if there's a duplicate attribute with different values
    // already applied to the declaration.
    if (const auto *DeclAttr = D->getAttr<IntelReqdSubGroupSizeAttr>()) {
      // If the other attribute argument is instantiation dependent, we won't
      // have converted it to a constant expression yet and thus we test
      // whether this is a null pointer.
      if (const auto *DeclExpr = dyn_cast<ConstantExpr>(DeclAttr->getValue())) {
        if (ArgVal != DeclExpr->getResultAsAPSInt()) {
          Diag(CI.getLoc(), diag::warn_duplicate_attribute) << CI;
          Diag(DeclAttr->getLoc(), diag::note_previous_attribute);
        }
        // Drop the duplicate attribute.
        return;
      }
    }
  }

  D->addAttr(::new (Context) IntelReqdSubGroupSizeAttr(Context, CI, E));
}

// Check that the value is a non-negative integer constant that can fit in
// 32-bits. Issue correct error message and return false on failure.
bool static check32BitInt(const Expr *E, SemaSYCL &S, llvm::APSInt &I,
                          const AttributeCommonInfo &CI) {
  if (!I.isIntN(32)) {
    S.Diag(E->getExprLoc(), diag::err_ice_too_large)
        << llvm::toString(I, 10, false) << 32 << /* Unsigned */ 1;
    return false;
  }

  if (I.isSigned() && I.isNegative()) {
    S.Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
        << CI << /* Non-negative */ 1;
    return false;
  }

  return true;
}

void SemaSYCL::addSYCLIntelMinWorkGroupsPerComputeUnitAttr(
    Decl *D, const AttributeCommonInfo &CI, Expr *E) {
  ASTContext &Context = getASTContext();
  if (getLangOpts().SYCLIsDevice) {
    if (!Context.getTargetInfo().getTriple().isNVPTX()) {
      Diag(E->getBeginLoc(), diag::warn_launch_bounds_is_cuda_specific)
          << CI << E->getSourceRange();
      return;
    }

    Diag(CI.getLoc(), diag::warn_launch_bounds_missing_attr) << CI << 0;
    return;
  }
  if (!E->isValueDependent()) {
    // Validate that we have an integer constant expression and then store the
    // converted constant expression into the semantic attribute so that we
    // don't have to evaluate it again later.
    llvm::APSInt ArgVal;
    ExprResult Res = SemaRef.VerifyIntegerConstantExpression(E, &ArgVal);
    if (Res.isInvalid())
      return;
    if (!check32BitInt(E, *this, ArgVal, CI))
      return;
    E = Res.get();

    // Check to see if there's a duplicate attribute with different values
    // already applied to the declaration.
    if (const auto *DeclAttr =
            D->getAttr<SYCLIntelMinWorkGroupsPerComputeUnitAttr>()) {
      // If the other attribute argument is instantiation dependent, we won't
      // have converted it to a constant expression yet and thus we test
      // whether this is a null pointer.
      if (const auto *DeclExpr = dyn_cast<ConstantExpr>(DeclAttr->getValue())) {
        if (ArgVal != DeclExpr->getResultAsAPSInt()) {
          Diag(CI.getLoc(), diag::warn_duplicate_attribute) << CI;
          Diag(DeclAttr->getLoc(), diag::note_previous_attribute);
        }
        // Drop the duplicate attribute.
        return;
      }
    }
  }

  D->addAttr(::new (Context)
                 SYCLIntelMinWorkGroupsPerComputeUnitAttr(Context, CI, E));
}

void SemaSYCL::addSYCLIntelMaxWorkGroupsPerMultiprocessorAttr(
    Decl *D, const AttributeCommonInfo &CI, Expr *E) {
  ASTContext &Context = getASTContext();
  auto &TI = Context.getTargetInfo();
  if (Context.getLangOpts().SYCLIsDevice) {
    if (!TI.getTriple().isNVPTX()) {
      Diag(E->getBeginLoc(), diag::warn_launch_bounds_is_cuda_specific)
          << CI << E->getSourceRange();
      return;
    }

    // Feature '.maxclusterrank' requires .target sm_90 or higher.
    auto SM = getOffloadArch(TI);
    if (SM == OffloadArch::UNKNOWN || SM < OffloadArch::SM_90) {
      Diag(E->getBeginLoc(), diag::warn_cuda_maxclusterrank_sm_90)
          << OffloadArchToString(SM) << CI << E->getSourceRange();
      return;
    }

    Diag(CI.getLoc(), diag::warn_launch_bounds_missing_attr) << CI << 1;
    return;
  }
  if (!E->isValueDependent()) {
    // Validate that we have an integer constant expression and then store the
    // converted constant expression into the semantic attribute so that we
    // don't have to evaluate it again later.
    llvm::APSInt ArgVal;
    ExprResult Res = SemaRef.VerifyIntegerConstantExpression(E, &ArgVal);
    if (Res.isInvalid())
      return;
    if (!check32BitInt(E, *this, ArgVal, CI))
      return;
    E = Res.get();

    // Check to see if there's a duplicate attribute with different values
    // already applied to the declaration.
    if (const auto *DeclAttr =
            D->getAttr<SYCLIntelMaxWorkGroupsPerMultiprocessorAttr>()) {
      // If the other attribute argument is instantiation dependent, we won't
      // have converted it to a constant expression yet and thus we test
      // whether this is a null pointer.
      if (const auto *DeclExpr = dyn_cast<ConstantExpr>(DeclAttr->getValue())) {
        if (ArgVal != DeclExpr->getResultAsAPSInt()) {
          Diag(CI.getLoc(), diag::warn_duplicate_attribute) << CI;
          Diag(DeclAttr->getLoc(), diag::note_previous_attribute);
        }
        // Drop the duplicate attribute.
        return;
      }
    }
  }

  D->addAttr(::new (Context)
                 SYCLIntelMaxWorkGroupsPerMultiprocessorAttr(Context, CI, E));
}

void SemaSYCL::addSYCLIntelESimdVectorizeAttr(Decl *D,
                                              const AttributeCommonInfo &CI,
                                              Expr *E) {
  if (!E->isValueDependent()) {
    // Validate that we have an integer constant expression and then store the
    // converted constant expression into the semantic attribute so that we
    // don't have to evaluate it again later.
    llvm::APSInt ArgVal;
    ExprResult Res = SemaRef.VerifyIntegerConstantExpression(E, &ArgVal);
    if (Res.isInvalid())
      return;
    E = Res.get();

    if (ArgVal != 8 && ArgVal != 16 && ArgVal != 32) {
      Diag(E->getExprLoc(), diag::err_sycl_esimd_vectorize_unsupported_value)
          << CI;
      return;
    }

    // Check to see if there's a duplicate attribute with different values
    // already applied to the declaration.
    if (const auto *DeclAttr = D->getAttr<SYCLIntelESimdVectorizeAttr>()) {
      // If the other attribute argument is instantiation dependent, we won't
      // have converted it to a constant expression yet and thus we test
      // whether this is a null pointer.
      if (const auto *DeclExpr = dyn_cast<ConstantExpr>(DeclAttr->getValue())) {
        if (ArgVal != DeclExpr->getResultAsAPSInt()) {
          Diag(CI.getLoc(), diag::warn_duplicate_attribute) << CI;
          Diag(DeclAttr->getLoc(), diag::note_previous_attribute);
        }
        // Drop the duplicate attribute.
        return;
      }
    }
  }

  ASTContext &Context = getASTContext();
  D->addAttr(::new (Context) SYCLIntelESimdVectorizeAttr(Context, CI, E));
}

// Checks if an expression is a valid filter list for an add_ir_attributes_*
// attribute. Returns true if an error occured.
static bool checkAddIRAttributesFilterListExpr(Expr *FilterListArg, SemaSYCL &S,
                                               const AttributeCommonInfo &CI) {
  const auto *FilterListE = cast<InitListExpr>(FilterListArg);
  for (const Expr *FilterElemE : FilterListE->inits())
    if (!isa<StringLiteral>(FilterElemE))
      return S.Diag(FilterElemE->getBeginLoc(),
                    diag::err_sycl_add_ir_attribute_invalid_filter)
             << CI;
  return false;
}

// Returns true if a type is either an array of char or a pointer to char.
static bool isAddIRAttributesValidStringType(QualType T) {
  if (!T->isArrayType() && !T->isPointerType())
    return false;
  QualType ElemT = T->isArrayType()
                       ? cast<ArrayType>(T.getTypePtr())->getElementType()
                       : T->getPointeeType();
  return ElemT.isConstQualified() && ElemT->isCharType();
}

// Checks if an expression is a valid attribute value for an add_ir_attributes_*
// attribute. Returns true if an error occured.
static bool checkAddIRAttributesValueExpr(Expr *ValArg, SemaSYCL &S,
                                          const AttributeCommonInfo &CI) {
  QualType ValType = ValArg->getType();
  if (isAddIRAttributesValidStringType(ValType) || ValType->isNullPtrType() ||
      ValType->isIntegralOrEnumerationType() || ValType->isFloatingType())
    return false;

  return S.Diag(ValArg->getBeginLoc(),
                diag::err_sycl_add_ir_attribute_invalid_value)
         << CI;
}

// Checks if an expression is a valid attribute name for an add_ir_attributes_*
// attribute. Returns true if an error occured.
static bool checkAddIRAttributesNameExpr(Expr *NameArg, SemaSYCL &S,
                                         const AttributeCommonInfo &CI) {
  // Only strings and const char * are valid name arguments.
  if (isAddIRAttributesValidStringType(NameArg->getType()))
    return false;

  return S.Diag(NameArg->getBeginLoc(),
                diag::err_sycl_add_ir_attribute_invalid_name)
         << CI;
}

// Checks and evaluates arguments of an add_ir_attributes_* attribute. Returns
// true if an error occured.
static bool evaluateAddIRAttributesArgs(Expr **Args, size_t ArgsSize,
                                        SemaSYCL &S,
                                        const AttributeCommonInfo &CI) {
  ASTContext &Context = S.getASTContext();

  // Check filter list if it is the first argument.
  bool HasFilter = ArgsSize && isa<InitListExpr>(Args[0]);
  if (HasFilter && checkAddIRAttributesFilterListExpr(Args[0], S, CI))
    return true;

  llvm::SmallVector<PartialDiagnosticAt, 8> Notes;
  bool HasDependentArg = false;
  for (unsigned I = HasFilter; I < ArgsSize; I++) {
    Expr *&E = Args[I];

    if (isa<InitListExpr>(E))
      return S.Diag(E->getBeginLoc(),
                    diag::err_sycl_add_ir_attr_filter_list_invalid_arg)
             << CI;

    if (E->isValueDependent() || E->isTypeDependent()) {
      HasDependentArg = true;
      continue;
    }

    Expr::EvalResult Eval;
    Eval.Diag = &Notes;
    if (!E->EvaluateAsConstantExpr(Eval, Context) || !Notes.empty()) {
      S.Diag(E->getBeginLoc(), diag::err_attribute_argument_n_type)
          << CI << (I + 1) << AANT_ArgumentConstantExpr;
      for (auto &Note : Notes)
        S.Diag(Note.first, Note.second);
      return true;
    }
    assert(Eval.Val.hasValue());
    E = ConstantExpr::Create(Context, E, Eval.Val);
  }

  // If there are no dependent expressions, check for expected number of args.
  if (!HasDependentArg && ArgsSize && (ArgsSize - HasFilter) & 1)
    return S.Diag(CI.getLoc(), diag::err_sycl_add_ir_attribute_must_have_pairs)
           << CI;

  // If there are no dependent expressions, check argument types.
  // First half of the arguments are names, the second half are values.
  unsigned MidArg = (ArgsSize - HasFilter) / 2 + HasFilter;
  if (!HasDependentArg) {
    for (unsigned I = HasFilter; I < ArgsSize; ++I) {
      if ((I < MidArg && checkAddIRAttributesNameExpr(Args[I], S, CI)) ||
          (I >= MidArg && checkAddIRAttributesValueExpr(Args[I], S, CI)))
        return true;
    }
  }
  return false;
}

void SemaSYCL::addSYCLAddIRAttributesFunctionAttr(
    Decl *D, const AttributeCommonInfo &CI, MutableArrayRef<Expr *> Args) {
  if (const auto *FuncD = dyn_cast<FunctionDecl>(D)) {
    if (FuncD->isDefaulted()) {
      Diag(CI.getLoc(), diag::err_disallow_attribute_on_func) << CI << 0;
      return;
    }
    if (FuncD->isDeleted()) {
      Diag(CI.getLoc(), diag::err_disallow_attribute_on_func) << CI << 1;
      return;
    }
  }

  ASTContext &Context = getASTContext();
  auto *Attr = SYCLAddIRAttributesFunctionAttr::Create(Context, Args.data(),
                                                       Args.size(), CI);
  if (evaluateAddIRAttributesArgs(Attr->args_begin(), Attr->args_size(), *this,
                                  CI))
    return;

  // There could be multiple of the same attribute applied to the same
  // declaration. If so, we want to merge them.
  // If there are still dependent expressions in the attribute, we delay merging
  // till after instantiation.
  if (!hasDependentExpr(Attr->args_begin(), Attr->args_size()) &&
      D->hasAttr<SYCLAddIRAttributesFunctionAttr>()) {
    Attr = mergeSYCLAddIRAttributesFunctionAttr(D, *Attr);

    // If null is returned, the attribute did not change after merge and we can
    // exit.
    if (!Attr)
      return;
  }
  D->addAttr(Attr);

  // There are compile-time SYCL properties which we would like to turn into
  // attributes to enable compiler diagnostics.
  // At the moment the only such property is related to virtual functions and
  // it is turned into sycl_device attribute. This is a tiny optimization to
  // avoid deep dive into the attribute if we already know that a declaration
  // is a device declaration. It may have to be removed later if/when we add
  // handling of more compile-time properties here.
  if (D->hasAttr<SYCLDeviceAttr>())
    return;

  // SYCL Headers use template magic to pass key=value pairs to the attribute
  // and we should make sure that all template instantiations are done before
  // accessing attribute arguments.
  if (hasDependentExpr(Attr->args_begin(), Attr->args_size()))
    return;

  SmallVector<std::pair<std::string, std::string>, 4> Pairs =
      Attr->getFilteredAttributeNameValuePairs(Context);

  for (const auto &[Key, Value] : Pairs) {
    if (Key == "indirectly-callable") {
      D->addAttr(SYCLDeviceAttr::CreateImplicit(Context));
      break;
    }
  }
}

void SemaSYCL::addSYCLAddIRAttributesKernelParameterAttr(
    Decl *D, const AttributeCommonInfo &CI, MutableArrayRef<Expr *> Args) {
  ASTContext &Context = getASTContext();
  auto *Attr = SYCLAddIRAttributesKernelParameterAttr::Create(
      Context, Args.data(), Args.size(), CI);
  if (evaluateAddIRAttributesArgs(Attr->args_begin(), Attr->args_size(), *this,
                                  CI))
    return;

  // There could be multiple of the same attribute applied to the same argument.
  // If so, we want to merge them.
  // If there are still dependent expressions in the attribute, we delay merging
  // till after instantiation.
  if (!hasDependentExpr(Attr->args_begin(), Attr->args_size()) &&
      D->hasAttr<SYCLAddIRAttributesKernelParameterAttr>()) {
    Attr = mergeSYCLAddIRAttributesKernelParameterAttr(D, *Attr);

    // If null is returned, the attribute did not change after merge and we can
    // exit.
    if (!Attr)
      return;
  }
  D->addAttr(Attr);
}

void SemaSYCL::addSYCLAddIRAttributesGlobalVariableAttr(
    Decl *D, const AttributeCommonInfo &CI, MutableArrayRef<Expr *> Args) {
  ASTContext &Context = getASTContext();
  auto *Attr = SYCLAddIRAttributesGlobalVariableAttr::Create(
      Context, Args.data(), Args.size(), CI);
  if (evaluateAddIRAttributesArgs(Attr->args_begin(), Attr->args_size(), *this,
                                  CI))
    return;

  // There could be multiple of the same attribute applied to the same global
  // variable. If so, we want to merge them.
  // If there are still dependent expressions in the attribute, we delay merging
  // till after instantiation.
  if (!hasDependentExpr(Attr->args_begin(), Attr->args_size()) &&
      D->hasAttr<SYCLAddIRAttributesGlobalVariableAttr>()) {
    Attr = mergeSYCLAddIRAttributesGlobalVariableAttr(D, *Attr);

    // If null is returned, the attribute did not change after merge and we can
    // exit.
    if (!Attr)
      return;
  }
  D->addAttr(Attr);
}

void SemaSYCL::addSYCLAddIRAnnotationsMemberAttr(Decl *D,
                                                 const AttributeCommonInfo &CI,
                                                 MutableArrayRef<Expr *> Args) {
  ASTContext &Context = getASTContext();
  auto *Attr = SYCLAddIRAnnotationsMemberAttr::Create(Context, Args.data(),
                                                      Args.size(), CI);
  if (evaluateAddIRAttributesArgs(Attr->args_begin(), Attr->args_size(), *this,
                                  CI))
    return;
  D->addAttr(Attr);
}

void SemaSYCL::addSYCLWorkGroupSizeHintAttr(Decl *D,
                                            const AttributeCommonInfo &CI,
                                            Expr *XDim, Expr *YDim,
                                            Expr *ZDim) {
  // Returns nullptr if diagnosing, otherwise returns the original expression
  // or the original expression converted to a constant expression.
  auto CheckAndConvertArg = [&](Expr *E) -> std::optional<Expr *> {
    // We can only check if the expression is not value dependent.
    if (E && !E->isValueDependent()) {
      llvm::APSInt ArgVal;
      ExprResult Res = SemaRef.VerifyIntegerConstantExpression(E, &ArgVal);
      if (Res.isInvalid())
        return std::nullopt;
      E = Res.get();

      // This attribute requires a strictly positive value.
      if (ArgVal <= 0) {
        Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
            << CI << /*positive*/ 0;
        return std::nullopt;
      }
    }

    return E;
  };

  // Check all three argument values, and if any are bad, bail out. This will
  // convert the given expressions into constant expressions when possible.
  std::optional<Expr *> XDimConvert = CheckAndConvertArg(XDim);
  std::optional<Expr *> YDimConvert = CheckAndConvertArg(YDim);
  std::optional<Expr *> ZDimConvert = CheckAndConvertArg(ZDim);
  if (!XDimConvert || !YDimConvert || !ZDimConvert)
    return;
  XDim = XDimConvert.value();
  YDim = YDimConvert.value();
  ZDim = ZDimConvert.value();

  // If the attribute was already applied with different arguments, then
  // diagnose the second attribute as a duplicate and don't add it.
  if (const auto *Existing = D->getAttr<SYCLWorkGroupSizeHintAttr>()) {
    // If any of the results are known to be different, we can diagnose at this
    // point and drop the attribute.
    if (anyWorkGroupSizesDiffer(XDim, YDim, ZDim, Existing->getXDim(),
                                Existing->getYDim(), Existing->getZDim())) {
      Diag(CI.getLoc(), diag::warn_duplicate_attribute) << CI;
      Diag(Existing->getLoc(), diag::note_previous_attribute);
      return;
    }
    // If all of the results are known to be the same, we can silently drop the
    // attribute. Otherwise, we have to add the attribute and resolve its
    // differences later.
    if (allWorkGroupSizesSame(XDim, YDim, ZDim, Existing->getXDim(),
                              Existing->getYDim(), Existing->getZDim()))
      return;
  }

  ASTContext &Context = getASTContext();
  D->addAttr(::new (Context)
                 SYCLWorkGroupSizeHintAttr(Context, CI, XDim, YDim, ZDim));
}

void SemaSYCL::addSYCLReqdWorkGroupSizeAttr(Decl *D,
                                            const AttributeCommonInfo &CI,
                                            Expr *XDim, Expr *YDim,
                                            Expr *ZDim) {
  // Returns nullptr if diagnosing, otherwise returns the original expression
  // or the original expression converted to a constant expression.
  auto CheckAndConvertArg = [&](Expr *E) -> std::optional<Expr *> {
    // Check if the expression is not value dependent.
    if (E && !E->isValueDependent()) {
      llvm::APSInt ArgVal;
      ExprResult Res = SemaRef.VerifyIntegerConstantExpression(E, &ArgVal);
      if (Res.isInvalid())
        return std::nullopt;
      E = Res.get();

      // This attribute requires a strictly positive value.
      if (ArgVal <= 0) {
        Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
            << CI << /*positive*/ 0;
        return std::nullopt;
      }
    }
    return E;
  };

  // Check all three argument values, and if any are bad, bail out. This will
  // convert the given expressions into constant expressions when possible.
  std::optional<Expr *> XDimConvert = CheckAndConvertArg(XDim);
  std::optional<Expr *> YDimConvert = CheckAndConvertArg(YDim);
  std::optional<Expr *> ZDimConvert = CheckAndConvertArg(ZDim);
  if (!XDimConvert || !YDimConvert || !ZDimConvert)
    return;
  XDim = XDimConvert.value();
  YDim = YDimConvert.value();
  ZDim = ZDimConvert.value();

  // If the attribute was already applied with different arguments, then
  // diagnose the second attribute as a duplicate and don't add it.
  if (const auto *Existing = D->getAttr<SYCLReqdWorkGroupSizeAttr>()) {
    // If any of the results are known to be different, we can diagnose at this
    // point and drop the attribute.
    if (anyWorkGroupSizesDiffer(XDim, YDim, ZDim, Existing->getXDim(),
                                Existing->getYDim(), Existing->getZDim())) {
      Diag(CI.getLoc(), diag::err_duplicate_attribute) << CI;
      Diag(Existing->getLoc(), diag::note_previous_attribute);
      return;
    }

    // If all of the results are known to be the same, we can silently drop the
    // attribute. Otherwise, we have to add the attribute and resolve its
    // differences later.
    if (allWorkGroupSizesSame(XDim, YDim, ZDim, Existing->getXDim(),
                              Existing->getYDim(), Existing->getZDim()))
      return;
  }

  ASTContext &Context = getASTContext();
  D->addAttr(::new (Context)
                 SYCLReqdWorkGroupSizeAttr(Context, CI, XDim, YDim, ZDim));
}

// Handles SYCL work_group_size_hint.
void SemaSYCL::handleSYCLWorkGroupSizeHintAttr(Decl *D, const ParsedAttr &AL) {
  checkDeprecatedSYCLAttributeSpelling(AL);

  // __attribute__((work_group_size_hint) requires exactly three arguments.
  if (AL.getSyntax() == ParsedAttr::AS_GNU || !AL.hasScope() ||
      (AL.hasScope() && !AL.getScopeName()->isStr("sycl"))) {
    if (!AL.checkExactlyNumArgs(SemaRef, 3))
      return;
  } else if (!AL.checkAtLeastNumArgs(SemaRef, 1) ||
             !AL.checkAtMostNumArgs(SemaRef, 3))
    return;

  size_t NumArgs = AL.getNumArgs();
  Expr *XDimExpr = NumArgs > 0 ? AL.getArgAsExpr(0) : nullptr;
  Expr *YDimExpr = NumArgs > 1 ? AL.getArgAsExpr(1) : nullptr;
  Expr *ZDimExpr = NumArgs > 2 ? AL.getArgAsExpr(2) : nullptr;
  addSYCLWorkGroupSizeHintAttr(D, AL, XDimExpr, YDimExpr, ZDimExpr);
}

SYCLWorkGroupSizeHintAttr *
SemaSYCL::mergeSYCLWorkGroupSizeHintAttr(Decl *D,
                                         const SYCLWorkGroupSizeHintAttr &A) {
  // Check to see if there's a duplicate attribute already applied.
  if (const auto *DeclAttr = D->getAttr<SYCLWorkGroupSizeHintAttr>()) {
    // If any of the results are known to be different, we can diagnose at this
    // point and drop the attribute.
    if (anyWorkGroupSizesDiffer(DeclAttr->getXDim(), DeclAttr->getYDim(),
                                DeclAttr->getZDim(), A.getXDim(), A.getYDim(),
                                A.getZDim())) {
      Diag(DeclAttr->getLoc(), diag::warn_duplicate_attribute) << &A;
      Diag(A.getLoc(), diag::note_previous_attribute);
      return nullptr;
    }
    // If all of the results are known to be the same, we can silently drop the
    // attribute. Otherwise, we have to add the attribute and resolve its
    // differences later.
    if (allWorkGroupSizesSame(DeclAttr->getXDim(), DeclAttr->getYDim(),
                              DeclAttr->getZDim(), A.getXDim(), A.getYDim(),
                              A.getZDim()))
      return nullptr;
  }
  ASTContext &Context = getASTContext();
  return ::new (Context) SYCLWorkGroupSizeHintAttr(Context, A, A.getXDim(),
                                                   A.getYDim(), A.getZDim());
}

void SemaSYCL::handleSYCLReqdWorkGroupSizeAttr(Decl *D, const ParsedAttr &AL) {
  checkDeprecatedSYCLAttributeSpelling(AL);

  // __attribute__((reqd_work_group_size)) and [[cl::reqd_work_group_size]]
  // all require exactly three arguments.
  if ((AL.getKind() == ParsedAttr::AT_ReqdWorkGroupSize &&
       AL.getAttributeSpellingListIndex() ==
           SYCLReqdWorkGroupSizeAttr::CXX11_cl_reqd_work_group_size) ||
      AL.getSyntax() == ParsedAttr::AS_GNU) {
    if (!AL.checkExactlyNumArgs(SemaRef, 3))
      return;
  } else if (!AL.checkAtLeastNumArgs(SemaRef, 1) ||
             !AL.checkAtMostNumArgs(SemaRef, 3))
    return;

  size_t NumArgs = AL.getNumArgs();
  Expr *XDimExpr = NumArgs > 0 ? AL.getArgAsExpr(0) : nullptr;
  Expr *YDimExpr = NumArgs > 1 ? AL.getArgAsExpr(1) : nullptr;
  Expr *ZDimExpr = NumArgs > 2 ? AL.getArgAsExpr(2) : nullptr;
  addSYCLReqdWorkGroupSizeAttr(D, AL, XDimExpr, YDimExpr, ZDimExpr);
}

SYCLReqdWorkGroupSizeAttr *
SemaSYCL::mergeSYCLReqdWorkGroupSizeAttr(Decl *D,
                                         const SYCLReqdWorkGroupSizeAttr &A) {
  // Check to see if there's a duplicate attribute already applied.
  if (const auto *DeclAttr = D->getAttr<SYCLReqdWorkGroupSizeAttr>()) {
    // If any of the results are known to be different, we can diagnose at this
    // point and drop the attribute.
    if (anyWorkGroupSizesDiffer(DeclAttr->getXDim(), DeclAttr->getYDim(),
                                DeclAttr->getZDim(), A.getXDim(), A.getYDim(),
                                A.getZDim())) {
      Diag(DeclAttr->getLoc(), diag::err_duplicate_attribute) << &A;
      Diag(A.getLoc(), diag::note_previous_attribute);
      return nullptr;
    }

    // If all of the results are known to be the same, we can silently drop the
    // attribute. Otherwise, we have to add the attribute and resolve its
    // differences later.
    if (allWorkGroupSizesSame(DeclAttr->getXDim(), DeclAttr->getYDim(),
                              DeclAttr->getZDim(), A.getXDim(), A.getYDim(),
                              A.getZDim()))
      return nullptr;
  }

  ASTContext &Context = getASTContext();
  return ::new (Context) SYCLReqdWorkGroupSizeAttr(Context, A, A.getXDim(),
                                                   A.getYDim(), A.getZDim());
}

IntelReqdSubGroupSizeAttr *
SemaSYCL::mergeIntelReqdSubGroupSizeAttr(Decl *D,
                                         const IntelReqdSubGroupSizeAttr &A) {
  // Check to see if there's a duplicate attribute with different values
  // already applied to the declaration.
  if (const auto *DeclAttr = D->getAttr<IntelReqdSubGroupSizeAttr>()) {
    if (const auto *DeclExpr = dyn_cast<ConstantExpr>(DeclAttr->getValue())) {
      if (const auto *MergeExpr = dyn_cast<ConstantExpr>(A.getValue())) {
        if (DeclExpr->getResultAsAPSInt() != MergeExpr->getResultAsAPSInt()) {
          Diag(DeclAttr->getLoc(), diag::warn_duplicate_attribute) << &A;
          Diag(A.getLoc(), diag::note_previous_attribute);
          return nullptr;
        }
        // Do not add a duplicate attribute.
        return nullptr;
      }
    }
  }
  ASTContext &Context = getASTContext();
  return ::new (Context) IntelReqdSubGroupSizeAttr(Context, A, A.getValue());
}

void SemaSYCL::handleIntelReqdSubGroupSizeAttr(Decl *D, const ParsedAttr &AL) {
  checkDeprecatedSYCLAttributeSpelling(AL);

  Expr *E = AL.getArgAsExpr(0);
  addIntelReqdSubGroupSizeAttr(D, AL, E);
}

IntelNamedSubGroupSizeAttr *
SemaSYCL::mergeIntelNamedSubGroupSizeAttr(Decl *D,
                                          const IntelNamedSubGroupSizeAttr &A) {
  // Check to see if there's a duplicate attribute with different values
  // already applied to the declaration.
  if (const auto *DeclAttr = D->getAttr<IntelNamedSubGroupSizeAttr>()) {
    if (DeclAttr->getType() != A.getType()) {
      Diag(DeclAttr->getLoc(), diag::warn_duplicate_attribute) << &A;
      Diag(A.getLoc(), diag::note_previous_attribute);
    }
    return nullptr;
  }

  ASTContext &Context = getASTContext();
  return IntelNamedSubGroupSizeAttr::Create(Context, A.getType(), A);
}

void SemaSYCL::handleIntelNamedSubGroupSizeAttr(Decl *D, const ParsedAttr &AL) {
  StringRef SizeStr;
  SourceLocation Loc;
  if (AL.isArgIdent(0)) {
    IdentifierLoc *IL = AL.getArgAsIdent(0);
    SizeStr = IL->getIdentifierInfo()->getName();
    Loc = IL->getLoc();
  } else if (!SemaRef.checkStringLiteralArgumentAttr(AL, 0, SizeStr, &Loc)) {
    return;
  }

  IntelNamedSubGroupSizeAttr::SubGroupSizeType SizeType;
  if (!IntelNamedSubGroupSizeAttr::ConvertStrToSubGroupSizeType(SizeStr,
                                                                SizeType)) {
    Diag(Loc, diag::warn_attribute_type_not_supported) << AL << SizeStr;
    return;
  }
  D->addAttr(IntelNamedSubGroupSizeAttr::Create(getASTContext(), SizeType, AL));
}

SYCLIntelMinWorkGroupsPerComputeUnitAttr *
SemaSYCL::mergeSYCLIntelMinWorkGroupsPerComputeUnitAttr(
    Decl *D, const SYCLIntelMinWorkGroupsPerComputeUnitAttr &A) {
  // Check to see if there's a duplicate attribute with different values
  // already applied to the declaration.
  if (const auto *DeclAttr =
          D->getAttr<SYCLIntelMinWorkGroupsPerComputeUnitAttr>()) {
    if (const auto *DeclExpr = dyn_cast<ConstantExpr>(DeclAttr->getValue())) {
      if (const auto *MergeExpr = dyn_cast<ConstantExpr>(A.getValue())) {
        if (DeclExpr->getResultAsAPSInt() != MergeExpr->getResultAsAPSInt()) {
          Diag(DeclAttr->getLoc(), diag::warn_duplicate_attribute) << &A;
          Diag(A.getLoc(), diag::note_previous_attribute);
        }
        // Do not add a duplicate attribute.
        return nullptr;
      }
    }
  }

  ASTContext &Context = getASTContext();
  return ::new (Context)
      SYCLIntelMinWorkGroupsPerComputeUnitAttr(Context, A, A.getValue());
}

SYCLIntelMaxWorkGroupsPerMultiprocessorAttr *
SemaSYCL::mergeSYCLIntelMaxWorkGroupsPerMultiprocessorAttr(
    Decl *D, const SYCLIntelMaxWorkGroupsPerMultiprocessorAttr &A) {
  // Check to see if there's a duplicate attribute with different values
  // already applied to the declaration.
  if (const auto *DeclAttr =
          D->getAttr<SYCLIntelMaxWorkGroupsPerMultiprocessorAttr>()) {
    if (const auto *DeclExpr = dyn_cast<ConstantExpr>(DeclAttr->getValue())) {
      if (const auto *MergeExpr = dyn_cast<ConstantExpr>(A.getValue())) {
        if (DeclExpr->getResultAsAPSInt() != MergeExpr->getResultAsAPSInt()) {
          Diag(DeclAttr->getLoc(), diag::warn_duplicate_attribute) << &A;
          Diag(A.getLoc(), diag::note_previous_attribute);
        }
        // Do not add a duplicate attribute.
        return nullptr;
      }
    }
  }

  ASTContext &Context = getASTContext();
  return ::new (Context)
      SYCLIntelMaxWorkGroupsPerMultiprocessorAttr(Context, A, A.getValue());
}

void SemaSYCL::handleSYCLIntelESimdVectorizeAttr(Decl *D, const ParsedAttr &A) {
  checkDeprecatedSYCLAttributeSpelling(A);

  Expr *E = A.getArgAsExpr(0);
  addSYCLIntelESimdVectorizeAttr(D, A, E);
}

SYCLIntelESimdVectorizeAttr *SemaSYCL::mergeSYCLIntelESimdVectorizeAttr(
    Decl *D, const SYCLIntelESimdVectorizeAttr &A) {
  // Check to see if there's a duplicate attribute with different values
  // already applied to the declaration.
  if (const auto *DeclAttr = D->getAttr<SYCLIntelESimdVectorizeAttr>()) {
    if (const auto *DeclExpr = dyn_cast<ConstantExpr>(DeclAttr->getValue())) {
      if (const auto *MergeExpr = dyn_cast<ConstantExpr>(A.getValue())) {
        if (DeclExpr->getResultAsAPSInt() != MergeExpr->getResultAsAPSInt()) {
          Diag(DeclAttr->getLoc(), diag::warn_duplicate_attribute) << &A;
          Diag(A.getLoc(), diag::note_previous_attribute);
        }
        // Do not add a duplicate attribute.
        return nullptr;
      }
    }
  }
  ASTContext &Context = getASTContext();
  return ::new (Context) SYCLIntelESimdVectorizeAttr(Context, A, A.getValue());
}

void SemaSYCL::handleSYCLAddIRAttributesFunctionAttr(Decl *D,
                                                     const ParsedAttr &A) {
  llvm::SmallVector<Expr *, 4> Args;
  Args.reserve(A.getNumArgs() - 1);
  for (unsigned I = 0; I < A.getNumArgs(); I++) {
    assert(A.isArgExpr(I));
    Args.push_back(A.getArgAsExpr(I));
  }

  addSYCLAddIRAttributesFunctionAttr(D, A, Args);
}

static bool hasConflictingSYCLAddIRAttributes(
    const SmallVector<std::pair<std::string, std::string>, 4> &LAttrs,
    const SmallVector<std::pair<std::string, std::string>, 4> &RAttrs) {
  std::unordered_map<std::string, std::string> LNameValMap;
  for (const std::pair<std::string, std::string> &NameValuePair : LAttrs)
    LNameValMap[NameValuePair.first] = NameValuePair.second;

  return std::any_of(
      RAttrs.begin(), RAttrs.end(),
      [&](const std::pair<std::string, std::string> &NameValuePair) {
        auto It = LNameValMap.find(NameValuePair.first);
        return It != LNameValMap.end() && NameValuePair.second != It->second;
      });
}

template <typename AddIRAttrT>
static bool checkSYCLAddIRAttributesMergeability(const AddIRAttrT &NewAttr,
                                                 const AddIRAttrT &ExistingAttr,
                                                 SemaSYCL &S) {
  ASTContext &Context = S.getASTContext();

  // If there are dependent argument expressions, then merging cannot be done
  // yet. In that case, it is deferred till after instantiation.
  if (S.hasDependentExpr(NewAttr.args_begin(), NewAttr.args_size()) ||
      S.hasDependentExpr(ExistingAttr.args_begin(), ExistingAttr.args_size()))
    return true;

  // If the filters differ or the attributes are conflicting, then fail due to
  // differing duplicates.
  if (NewAttr.getAttributeFilter() != ExistingAttr.getAttributeFilter() ||
      hasConflictingSYCLAddIRAttributes(
          NewAttr.getAttributeNameValuePairs(Context),
          ExistingAttr.getAttributeNameValuePairs(Context))) {
    S.Diag(ExistingAttr.getLoc(), diag::err_duplicate_attribute) << &NewAttr;
    S.Diag(NewAttr.getLoc(), diag::note_conflicting_attribute);
    return true;
  }
  return false;
}

template <typename AddIRAttrT>
static AddIRAttrT *getMergedSYCLAddIRAttribute(const AddIRAttrT &Attr1,
                                               const AddIRAttrT &Attr2,
                                               SemaSYCL &S) {
  ASTContext &Context = S.getASTContext();
  bool AttrHasFilterList = Attr1.hasFilterList();

  // Get the vectors of name-value-pairs here so we can create string references
  // to them for the map.
  llvm::SmallVector<std::pair<std::string, std::string>, 4> Attr1NVPairs =
      Attr1.getAttributeNameValuePairs(Context);
  llvm::SmallVector<std::pair<std::string, std::string>, 4> Attr2NVPairs =
      Attr2.getAttributeNameValuePairs(Context);

  // Collect all the unique attribute names and their corresponding values. This
  // relies on the uniqueness having been confirmed first and that the
  // attributes appear in the same order as in the name-value-pairs.
  llvm::SmallMapVector<StringRef, std::pair<Expr *, Expr *>, 4> AttrExprByName;
  for (const auto &[Attr, NVPairs] : {std::make_pair(Attr1, Attr1NVPairs),
                                      std::make_pair(Attr2, Attr2NVPairs)}) {
    for (size_t I = 0; I < NVPairs.size(); ++I) {
      AttrExprByName[NVPairs[I].first] = std::make_pair(
          (Attr.args_begin() + AttrHasFilterList)[I],
          (Attr.args_begin() + AttrHasFilterList + (Attr.args_size() / 2))[I]);
    }
  }

  // Create a list of new arguments, starting with the filter if present.
  llvm::SmallVector<Expr *, 4> NewArgs;
  NewArgs.resize(AttrExprByName.size() * 2 + AttrHasFilterList);
  if (AttrHasFilterList)
    NewArgs[0] = *Attr1.args_begin();

  // Then insert all the unique attributes found previously.
  for (size_t I = 0; I < AttrExprByName.size(); ++I) {
    const std::pair<Expr *, Expr *> &Exprs = AttrExprByName.begin()[I].second;
    NewArgs[I + AttrHasFilterList] = Exprs.first;
    NewArgs[I + AttrExprByName.size() + AttrHasFilterList] = Exprs.second;
  }

  return AddIRAttrT::Create(Context, NewArgs.data(), NewArgs.size(), Attr1);
}

SYCLAddIRAttributesFunctionAttr *SemaSYCL::mergeSYCLAddIRAttributesFunctionAttr(
    Decl *D, const SYCLAddIRAttributesFunctionAttr &A) {
  if (const auto *ExistingAttr =
          D->getAttr<SYCLAddIRAttributesFunctionAttr>()) {
    if (checkSYCLAddIRAttributesMergeability(A, *ExistingAttr, *this))
      return nullptr;

    D->dropAttr<SYCLAddIRAttributesFunctionAttr>();
    return getMergedSYCLAddIRAttribute(A, *ExistingAttr, *this);
  }
  ASTContext &Context = getASTContext();
  return A.clone(Context);
}

void SemaSYCL::handleSYCLAddIRAttributesKernelParameterAttr(
    Decl *D, const ParsedAttr &A) {
  llvm::SmallVector<Expr *, 4> Args;
  Args.reserve(A.getNumArgs() - 1);
  for (unsigned I = 0; I < A.getNumArgs(); I++) {
    assert(A.getArgAsExpr(I));
    Args.push_back(A.getArgAsExpr(I));
  }

  addSYCLAddIRAttributesKernelParameterAttr(D, A, Args);
}

SYCLAddIRAttributesKernelParameterAttr *
SemaSYCL::mergeSYCLAddIRAttributesKernelParameterAttr(
    Decl *D, const SYCLAddIRAttributesKernelParameterAttr &A) {
  if (const auto *ExistingAttr =
          D->getAttr<SYCLAddIRAttributesKernelParameterAttr>()) {
    if (checkSYCLAddIRAttributesMergeability(A, *ExistingAttr, *this))
      return nullptr;

    D->dropAttr<SYCLAddIRAttributesKernelParameterAttr>();
    return getMergedSYCLAddIRAttribute(A, *ExistingAttr, *this);
  }
  ASTContext &Context = getASTContext();
  return A.clone(Context);
}

void SemaSYCL::handleSYCLAddIRAttributesGlobalVariableAttr(
    Decl *D, const ParsedAttr &A) {
  llvm::SmallVector<Expr *, 4> Args;
  Args.reserve(A.getNumArgs() - 1);
  for (unsigned I = 0; I < A.getNumArgs(); I++) {
    assert(A.getArgAsExpr(I));
    Args.push_back(A.getArgAsExpr(I));
  }

  addSYCLAddIRAttributesGlobalVariableAttr(D, A, Args);
}

SYCLAddIRAttributesGlobalVariableAttr *
SemaSYCL::mergeSYCLAddIRAttributesGlobalVariableAttr(
    Decl *D, const SYCLAddIRAttributesGlobalVariableAttr &A) {
  if (const auto *ExistingAttr =
          D->getAttr<SYCLAddIRAttributesGlobalVariableAttr>()) {
    if (checkSYCLAddIRAttributesMergeability(A, *ExistingAttr, *this))
      return nullptr;

    D->dropAttr<SYCLAddIRAttributesGlobalVariableAttr>();
    return getMergedSYCLAddIRAttribute(A, *ExistingAttr, *this);
  }
  ASTContext &Context = getASTContext();
  return A.clone(Context);
}

void SemaSYCL::handleSYCLAddIRAnnotationsMemberAttr(Decl *D,
                                                    const ParsedAttr &A) {
  llvm::SmallVector<Expr *, 4> Args;
  Args.reserve(A.getNumArgs());
  for (unsigned I = 0; I < A.getNumArgs(); I++) {
    assert(A.getArgAsExpr(I));
    Args.push_back(A.getArgAsExpr(I));
  }

  addSYCLAddIRAnnotationsMemberAttr(D, A, Args);
}

SYCLAddIRAnnotationsMemberAttr *SemaSYCL::mergeSYCLAddIRAnnotationsMemberAttr(
    Decl *D, const SYCLAddIRAnnotationsMemberAttr &A) {
  if (const auto *ExistingAttr = D->getAttr<SYCLAddIRAnnotationsMemberAttr>()) {
    checkSYCLAddIRAttributesMergeability(A, *ExistingAttr, *this);
    return nullptr;
  }
  ASTContext &Context = getASTContext();
  return A.clone(Context);
}

void SemaSYCL::handleSYCLDeviceHasAttr(Decl *D, const ParsedAttr &A) {
  // Ignore the attribute if compiling for the host side because aspects may not
  // be marked properly for such compilation
  if (!getLangOpts().SYCLIsDevice)
    return;

  SmallVector<Expr *, 5> Args;
  for (unsigned I = 0; I < A.getNumArgs(); ++I)
    Args.push_back(A.getArgAsExpr(I));

  addSYCLDeviceHasAttr(D, A, Args.data(), Args.size());
}

SYCLDeviceHasAttr *
SemaSYCL::mergeSYCLDeviceHasAttr(Decl *D, const SYCLDeviceHasAttr &A) {
  if (const auto *ExistingAttr = D->getAttr<SYCLDeviceHasAttr>()) {
    Diag(ExistingAttr->getLoc(), diag::warn_duplicate_attribute_exact) << &A;
    Diag(A.getLoc(), diag::note_previous_attribute);
    return nullptr;
  }

  SmallVector<Expr *, 5> Args;
  for (auto *E : A.aspects())
    Args.push_back(E);
  ASTContext &Context = getASTContext();
  return ::new (Context)
      SYCLDeviceHasAttr(Context, A, Args.data(), Args.size());
}

void SemaSYCL::handleSYCLUsesAspectsAttr(Decl *D, const ParsedAttr &A) {
  // Ignore the attribute if compiling for the host because aspects may not be
  // marked properly for such compilation
  if (!getLangOpts().SYCLIsDevice)
    return;

  SmallVector<Expr *, 5> Args;
  for (unsigned I = 0; I < A.getNumArgs(); ++I)
    Args.push_back(A.getArgAsExpr(I));

  addSYCLUsesAspectsAttr(D, A, Args.data(), Args.size());
}

SYCLUsesAspectsAttr *
SemaSYCL::mergeSYCLUsesAspectsAttr(Decl *D, const SYCLUsesAspectsAttr &A) {
  if (const auto *ExistingAttr = D->getAttr<SYCLUsesAspectsAttr>()) {
    Diag(ExistingAttr->getLoc(), diag::warn_duplicate_attribute_exact) << &A;
    Diag(A.getLoc(), diag::note_previous_attribute);
    return nullptr;
  }

  SmallVector<Expr *, 5> Args;
  for (auto *E : A.aspects())
    Args.push_back(E);
  ASTContext &Context = getASTContext();
  return ::new (Context)
      SYCLUsesAspectsAttr(Context, A, Args.data(), Args.size());
}

void SemaSYCL::handleSYCLTypeAttr(Decl *D, const ParsedAttr &AL) {
  if (!AL.isArgIdent(0)) {
    Diag(AL.getLoc(), diag::err_attribute_argument_type)
        << AL << AANT_ArgumentIdentifier;
    return;
  }

  IdentifierInfo *II = AL.getArgAsIdent(0)->getIdentifierInfo();
  SYCLTypeAttr::SYCLType Type;

  if (!SYCLTypeAttr::ConvertStrToSYCLType(II->getName(), Type)) {
    Diag(AL.getLoc(), diag::err_attribute_argument_not_supported) << AL << II;
    return;
  }

  if (SYCLTypeAttr *NewAttr = mergeSYCLTypeAttr(D, AL, Type))
    D->addAttr(NewAttr);
}

SYCLTypeAttr *SemaSYCL::mergeSYCLTypeAttr(Decl *D,
                                          const AttributeCommonInfo &CI,
                                          SYCLTypeAttr::SYCLType TypeName) {
  if (const auto *ExistingAttr = D->getAttr<SYCLTypeAttr>()) {
    if (ExistingAttr->getType() != TypeName) {
      Diag(ExistingAttr->getLoc(), diag::err_duplicate_attribute)
          << ExistingAttr;
      Diag(CI.getLoc(), diag::note_previous_attribute);
    }
    // Do not add duplicate attribute
    return nullptr;
  }
  ASTContext &Context = getASTContext();
  return ::new (Context) SYCLTypeAttr(Context, CI, TypeName);
}

/// Give a warning for duplicate attributes, return true if duplicate.
template <typename AttrType>
static bool checkForDuplicateAttribute(SemaSYCL &S, Decl *D,
                                       const ParsedAttr &Attr) {
  // Give a warning for duplicates but not if it's one we've implicitly added.
  auto *A = D->getAttr<AttrType>();
  if (A && !A->isImplicit()) {
    S.Diag(Attr.getLoc(), diag::warn_duplicate_attribute_exact) << A;
    return true;
  }
  return false;
}

// Handles min_work_groups_per_cu attribute.
void SemaSYCL::handleSYCLIntelMinWorkGroupsPerComputeUnit(
    Decl *D, const ParsedAttr &AL) {
  addSYCLIntelMinWorkGroupsPerComputeUnitAttr(D, AL, AL.getArgAsExpr(0));
}

// Handles max_work_groups_per_mp attribute.
void SemaSYCL::handleSYCLIntelMaxWorkGroupsPerMultiprocessor(
    Decl *D, const ParsedAttr &AL) {
  addSYCLIntelMaxWorkGroupsPerMultiprocessorAttr(D, AL, AL.getArgAsExpr(0));
}

void SemaSYCL::handleSYCLDeviceAttr(Decl *D, const ParsedAttr &AL) {
  auto *ND = cast<NamedDecl>(D);
  if (!ND->isExternallyVisible()) {
    Diag(AL.getLoc(), diag::err_sycl_attribute_internal_decl)
        << AL << !isa<FunctionDecl>(ND);
    return;
  }

  if (auto *VD = dyn_cast<VarDecl>(D)) {
    QualType VarType = VD->getType();
    // Diagnose only for non-dependent types since dependent type don't have
    // attributes applied on them ATM.
    if (!VarType->isDependentType() &&
        !isTypeDecoratedWithDeclAttribute<SYCLDeviceGlobalAttr>(
            VD->getType())) {
      Diag(AL.getLoc(), diag::err_sycl_attribute_not_device_global) << AL;
      return;
    }
  }

  handleSimpleAttribute<SYCLDeviceAttr>(*this, D, AL);
}

void SemaSYCL::handleSYCLDeviceIndirectlyCallableAttr(Decl *D,
                                                      const ParsedAttr &AL) {
  auto *FD = cast<FunctionDecl>(D);
  if (!FD->isExternallyVisible()) {
    Diag(AL.getLoc(), diag::err_sycl_attribute_internal_decl)
        << AL << /*function*/ 0;
    return;
  }

  ASTContext &Context = getASTContext();
  D->addAttr(SYCLDeviceAttr::CreateImplicit(Context));
  handleSimpleAttribute<SYCLDeviceIndirectlyCallableAttr>(*this, D, AL);
}

void SemaSYCL::handleSYCLGlobalVarAttr(Decl *D, const ParsedAttr &AL) {
  ASTContext &Context = getASTContext();
  if (!Context.getSourceManager().isInSystemHeader(D->getLocation())) {
    Diag(AL.getLoc(), diag::err_attribute_only_system_header) << AL;
    return;
  }

  handleSimpleAttribute<SYCLGlobalVarAttr>(*this, D, AL);
}

void SemaSYCL::handleSYCLScopeAttr(Decl *D, const ParsedAttr &AL) {
  if (!AL.checkExactlyNumArgs(SemaRef, 0))
    return;
  if (auto *CRD = dyn_cast<CXXRecordDecl>(D);
      !CRD || !(CRD->isClass() || CRD->isStruct())) {
    SemaRef.Diag(AL.getRange().getBegin(),
                 diag::err_attribute_wrong_decl_type_str)
        << AL << AL.isRegularKeywordAttribute() << "classes";
    return;
  }

  D->addAttr(SYCLScopeAttr::Create(SemaRef.getASTContext(),
                                   SYCLScopeAttr::Level::WorkGroup, AL));
}

void SemaSYCL::checkSYCLAddIRAttributesFunctionAttrConflicts(Decl *D) {
  const auto *AddIRFuncAttr = D->getAttr<SYCLAddIRAttributesFunctionAttr>();

  // If there is no such attribute there is nothing to check. If there are
  // dependent arguments we cannot know the actual number of arguments so we
  // defer the check.
  if (!AddIRFuncAttr ||
      hasDependentExpr(AddIRFuncAttr->args_begin(), AddIRFuncAttr->args_size()))
    return;

  // If there are no name-value pairs in the attribute it will not have an
  // effect and we can skip the check. The filter is ignored.
  size_t NumArgsWithoutFilter =
      AddIRFuncAttr->args_size() - (AddIRFuncAttr->hasFilterList() ? 1 : 0);
  if (NumArgsWithoutFilter == 0)
    return;

  // "sycl-single-task" is present on all single_task invocations, implicitly
  // added by the SYCL headers. It can only conflict with max_global_work_dim,
  // but the value will be the same so there is no need for a warning.
  ASTContext &Context = getASTContext();
  if (NumArgsWithoutFilter == 2) {
    auto NameValuePairs = AddIRFuncAttr->getAttributeNameValuePairs(Context);
    if (NameValuePairs.size() > 0 &&
        NameValuePairs[0].first == "sycl-single-task")
      return;
  }

  // If there are potentially conflicting attributes, we issue a warning.
  for (const auto [Attr, PotentialConflictProp] :
       std::vector<std::pair<AttributeCommonInfo *, StringRef>>{
           {D->getAttr<SYCLReqdWorkGroupSizeAttr>(),
            "sycl::ext::oneapi::experimental::work_group_size"},
           {D->getAttr<IntelReqdSubGroupSizeAttr>(),
            "sycl::ext::oneapi::experimental::sub_group_size"},
           {D->getAttr<SYCLWorkGroupSizeHintAttr>(),
            "sycl::ext::oneapi::experimental::work_group_size_hint"},
           {D->getAttr<SYCLDeviceHasAttr>(),
            "sycl::ext::oneapi::experimental::device_has"}})
    if (Attr)
      Diag(Attr->getLoc(), diag::warn_sycl_old_and_new_kernel_attributes)
          << Attr << PotentialConflictProp;
}

void SemaSYCL::handleSYCLRegisteredKernels(Decl *D, const ParsedAttr &A) {
  // Check for SYCL device compilation context.
  if (!getLangOpts().SYCLIsDevice)
    return;

  unsigned NumArgs = A.getNumArgs();
  // When declared, we expect at least one item in the list.
  if (NumArgs == 0) {
    Diag(A.getLoc(), diag::err_registered_kernels_num_of_args);
    return;
  }

  // Traverse through the items in the list.
  for (unsigned I = 0; I < NumArgs; I++) {
    assert(A.isArgExpr(I) && "Expected expression argument");
    // Each item in the list must be an initializer list expression.
    Expr *ArgExpr = A.getArgAsExpr(I);
    if (!isa<InitListExpr>(ArgExpr)) {
      Diag(ArgExpr->getExprLoc(), diag::err_registered_kernels_init_list);
      return;
    }

    auto *ArgListE = cast<InitListExpr>(ArgExpr);
    unsigned NumInits = ArgListE->getNumInits();
    // Each init-list expression must have a pair of values.
    if (NumInits != 2) {
      Diag(ArgExpr->getExprLoc(),
           diag::err_registered_kernels_init_list_pair_values);
      return;
    }

    // The first value of the pair must be a string.
    Expr *FirstExpr = ArgListE->getInit(0);
    StringRef CurStr;
    SourceLocation Loc = FirstExpr->getExprLoc();
    if (!SemaRef.checkStringLiteralArgumentAttr(A, FirstExpr, CurStr, &Loc))
      return;

    // Resolve the FunctionDecl from the second value of the pair.
    Expr *SecondE = ArgListE->getInit(1);
    FunctionDecl *FD = nullptr;
    if (auto *ULE = dyn_cast<UnresolvedLookupExpr>(SecondE)) {
      FD = SemaRef.ResolveSingleFunctionTemplateSpecialization(ULE, true);
      Loc = ULE->getExprLoc();
    } else {
      SecondE = SecondE->IgnoreParenCasts();
      if (auto *DRE = dyn_cast<DeclRefExpr>(SecondE))
        FD = dyn_cast<FunctionDecl>(DRE->getDecl());
      Loc = SecondE->getExprLoc();
    }
    // Issue a diagnostic if we are unable to resolve the FunctionDecl.
    if (!FD) {
      Diag(Loc, diag::err_registered_kernels_resolve_function) << CurStr;
      return;
    }
    // Issue a diagnostic is the FunctionDecl is not a SYCL free function.
    if (!isFreeFunction(FD)) {
      Diag(FD->getLocation(), diag::err_not_sycl_free_function) << CurStr;
      return;
    }
    // Construct a free function kernel.
    constructFreeFunctionKernel(FD, CurStr);
  }
}
