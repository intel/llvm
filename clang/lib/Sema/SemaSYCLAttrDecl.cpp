//===- SemaSYCLAttrDecl.cpp - Semantic Analysis for SYCL attributes -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This implements Semantic Analysis for SYCL attributes.
//===----------------------------------------------------------------------===//

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
  // Additionally, diagnose the old [[intel::ii]] spelling.
  if (A.getKind() == ParsedAttr::AT_SYCLIntelInitiationInterval &&
      A.getAttrName()->isStr("ii")) {
    diagnoseDeprecatedAttribute(A, "intel", "initiation_interval");
    return;
  }

  // Diagnose SYCL 2017 spellings in later SYCL modes.
  if (getLangOpts().getSYCLVersion() > LangOptions::SYCL_2017) {
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

  // Diagnose SYCL 2020 spellings used in earlier SYCL modes as being an
  // extension.
  if (getLangOpts().getSYCLVersion() == LangOptions::SYCL_2017 &&
      A.hasScope() && A.getScopeName()->isStr("sycl")) {
    Diag(A.getLoc(), diag::ext_sycl_2020_attr_spelling) << A;
    return;
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

void SemaSYCL::addSYCLIntelForcePow2DepthAttr(Decl *D,
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

    // This attribute accepts values 0 and 1 only.
    if (ArgVal < 0 || ArgVal > 1) {
      Diag(E->getBeginLoc(), diag::err_attribute_argument_is_not_valid) << CI;
      return;
    }

    // Check attribute applies to field, constant variables, local variables,
    // static variables, agent memory arguments, non-static data members,
    // and device_global variables for the device compilation.
    if (checkValidFPGAMemoryAttributesVar(D)) {
      Diag(CI.getLoc(), diag::err_fpga_attribute_incorrect_variable)
          << CI << /*agent memory arguments*/ 1;
      return;
    }

    // Check to see if there's a duplicate attribute with different values
    // already applied to the declaration.
    if (const auto *DeclAttr = D->getAttr<SYCLIntelForcePow2DepthAttr>()) {
      // If the other attribute argument is instantiation dependent, we won't
      // have converted it to a constant expression yet and thus we test
      // whether this is a null pointer.
      if (const auto *DeclExpr = dyn_cast<ConstantExpr>(DeclAttr->getValue())) {
        if (ArgVal != DeclExpr->getResultAsAPSInt()) {
          Diag(CI.getLoc(), diag::warn_duplicate_attribute) << CI;
          Diag(DeclAttr->getLoc(), diag::note_previous_attribute);
        }
        // If there is no mismatch, drop any duplicate attributes.
        return;
      }
    }
  }

  // If the declaration does not have an [[intel::fpga_memory]]
  // attribute, this creates one as an implicit attribute.
  ASTContext &Context = getASTContext();
  if (!D->hasAttr<SYCLIntelMemoryAttr>())
    D->addAttr(SYCLIntelMemoryAttr::CreateImplicit(
        Context, SYCLIntelMemoryAttr::Default));

  D->addAttr(::new (Context) SYCLIntelForcePow2DepthAttr(Context, CI, E));
}

/// Handle the [[intel::bankwidth]] and [[intel::numbanks]] attributes.
/// These require a single constant power of two greater than zero.
/// These are incompatible with the register attribute.
/// The numbanks and bank_bits attributes are related.  If bank_bits exists
/// when handling numbanks they are checked for consistency.
void SemaSYCL::addSYCLIntelBankWidthAttr(Decl *D, const AttributeCommonInfo &CI,
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

    // This attribute requires a strictly positive value.
    if (ArgVal <= 0) {
      Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
          << CI << /*positive*/ 0;
      return;
    }

    // This attribute requires a single constant power of two greater than zero.
    if (!ArgVal.isPowerOf2()) {
      Diag(E->getExprLoc(), diag::err_attribute_argument_not_power_of_two)
          << CI;
      return;
    }

    // Check attribute applies to field, constant variables, local variables,
    // static variables, agent memory arguments, non-static data members,
    // and device_global variables for the device compilation.
    if (checkValidFPGAMemoryAttributesVar(D)) {
      Diag(CI.getLoc(), diag::err_fpga_attribute_incorrect_variable)
          << CI << /*agent memory arguments*/ 1;
      return;
    }

    // Check to see if there's a duplicate attribute with different values
    // already applied to the declaration.
    if (const auto *DeclAttr = D->getAttr<SYCLIntelBankWidthAttr>()) {
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

  // If the declaration does not have an [[intel::fpga_memory]]
  // attribute, this creates one as an implicit attribute.
  ASTContext &Context = getASTContext();
  if (!D->hasAttr<SYCLIntelMemoryAttr>())
    D->addAttr(SYCLIntelMemoryAttr::CreateImplicit(
        Context, SYCLIntelMemoryAttr::Default));

  D->addAttr(::new (Context) SYCLIntelBankWidthAttr(Context, CI, E));
}

void SemaSYCL::addSYCLIntelNumBanksAttr(Decl *D, const AttributeCommonInfo &CI,
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

    // This attribute requires a strictly positive value.
    if (ArgVal <= 0) {
      Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
          << CI << /*positive*/ 0;
      return;
    }

    // This attribute requires a single constant power of two greater than zero.
    if (!ArgVal.isPowerOf2()) {
      Diag(E->getExprLoc(), diag::err_attribute_argument_not_power_of_two)
          << CI;
      return;
    }

    // Check or add the related BankBits attribute.
    if (auto *BBA = D->getAttr<SYCLIntelBankBitsAttr>()) {
      unsigned NumBankBits = BBA->args_size();
      if (NumBankBits != ArgVal.ceilLogBase2()) {
        Diag(E->getExprLoc(), diag::err_bankbits_numbanks_conflicting) << CI;
        return;
      }
    }

    // Check attribute applies to constant variables, local variables,
    // static variables, agent memory arguments, non-static data members,
    // and device_global variables for the device compilation.
    if (checkValidFPGAMemoryAttributesVar(D)) {
      Diag(CI.getLoc(), diag::err_fpga_attribute_incorrect_variable)
          << CI << /*agent memory arguments*/ 1;
      return;
    }

    // Check to see if there's a duplicate attribute with different values
    // already applied to the declaration.
    if (const auto *DeclAttr = D->getAttr<SYCLIntelNumBanksAttr>()) {
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

  // If the declaration does not have an [[intel::fpga_memory]]
  // attribute, this creates one as an implicit attribute.
  ASTContext &Context = getASTContext();
  if (!D->hasAttr<SYCLIntelMemoryAttr>())
    D->addAttr(SYCLIntelMemoryAttr::CreateImplicit(
        Context, SYCLIntelMemoryAttr::Default));

  // We are adding a user NumBanks attribute, drop any implicit default.
  if (auto *NBA = D->getAttr<SYCLIntelNumBanksAttr>()) {
    if (NBA->isImplicit())
      D->dropAttr<SYCLIntelNumBanksAttr>();
  }

  D->addAttr(::new (Context) SYCLIntelNumBanksAttr(Context, CI, E));
}

void SemaSYCL::addSYCLIntelBankBitsAttr(Decl *D, const AttributeCommonInfo &CI,
                                        Expr **Exprs, unsigned Size) {
  ASTContext &Context = getASTContext();
  SYCLIntelBankBitsAttr TmpAttr(Context, CI, Exprs, Size);
  SmallVector<Expr *, 8> Args;
  SmallVector<int64_t, 8> Values;
  bool ListIsValueDep = false;
  for (auto *E : TmpAttr.args()) {
    llvm::APSInt Value(32, /*IsUnsigned=*/false);
    Expr::EvalResult Result;
    ListIsValueDep = ListIsValueDep || E->isValueDependent();
    if (!E->isValueDependent()) {
      ExprResult ICE = SemaRef.VerifyIntegerConstantExpression(E, &Value);
      if (ICE.isInvalid())
        return;
      if (!Value.isNonNegative()) {
        Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
            << CI << /*non-negative*/ 1;
        return;
      }
      E = ICE.get();
    }
    Args.push_back(E);
    Values.push_back(Value.getExtValue());
  }

  // Check that the list is consecutive.
  if (!ListIsValueDep && Values.size() > 1) {
    bool ListIsAscending = Values[0] < Values[1];
    for (int I = 0, E = Values.size() - 1; I < E; ++I) {
      if (Values[I + 1] != Values[I] + (ListIsAscending ? 1 : -1)) {
        Diag(CI.getLoc(), diag::err_bankbits_non_consecutive) << &TmpAttr;
        return;
      }
    }
  }

  // Check or add the related numbanks attribute.
  if (auto *NBA = D->getAttr<SYCLIntelNumBanksAttr>()) {
    Expr *E = NBA->getValue();
    if (!E->isValueDependent()) {
      Expr::EvalResult Result;
      E->EvaluateAsInt(Result, Context);
      llvm::APSInt Value = Result.Val.getInt();
      if (Args.size() != Value.ceilLogBase2()) {
        Diag(TmpAttr.getLoc(), diag::err_bankbits_numbanks_conflicting);
        return;
      }
    }
  } else {
    llvm::APInt Num(32, (unsigned)(1 << Args.size()));
    Expr *NBE =
        IntegerLiteral::Create(Context, Num, Context.IntTy, SourceLocation());
    D->addAttr(SYCLIntelNumBanksAttr::CreateImplicit(Context, NBE));
  }

  // Check attribute applies to field, constant variables, local variables,
  // static variables, agent memory arguments, non-static data members,
  // and device_global variables for the device compilation.
  if (checkValidFPGAMemoryAttributesVar(D)) {
    Diag(CI.getLoc(), diag::err_fpga_attribute_incorrect_variable)
        << CI << /*agent memory arguments*/ 1;
    return;
  }

  if (!D->hasAttr<SYCLIntelMemoryAttr>())
    D->addAttr(SYCLIntelMemoryAttr::CreateImplicit(
        Context, SYCLIntelMemoryAttr::Default));

  D->addAttr(::new (Context)
                 SYCLIntelBankBitsAttr(Context, CI, Args.data(), Args.size()));
}

bool isDeviceAspectType(const QualType Ty) {
  const EnumType *ET = Ty->getAs<EnumType>();
  if (!ET)
    return false;

  if (const auto *Attr = ET->getDecl()->getAttr<SYCLTypeAttr>())
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

void SemaSYCL::addSYCLIntelPipeIOAttr(Decl *D, const AttributeCommonInfo &CI,
                                      Expr *E) {
  VarDecl *VD = cast<VarDecl>(D);
  QualType Ty = VD->getType();
  // TODO: Applicable only on pipe storages. Currently they are defined
  // as structures inside of SYCL headers. Add a check for pipe_storage_t
  // when it is ready.
  if (!Ty->isStructureType()) {
    Diag(CI.getLoc(), diag::err_attribute_wrong_decl_type_str)
        << CI << CI.isRegularKeywordAttribute()
        << "SYCL pipe storage declaration";
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
    E = Res.get();

    // This attribute requires a non-negative value.
    if (ArgVal < 0) {
      Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
          << CI << /*non-negative*/ 1;
      return;
    }

    // Check to see if there's a duplicate attribute with different values
    // already applied to the declaration.
    if (const auto *DeclAttr = D->getAttr<SYCLIntelPipeIOAttr>()) {
      // If the other attribute argument is instantiation dependent, we won't
      // have converted it to a constant expression yet and thus we test
      // whether this is a null pointer.
      if (const auto *DeclExpr = dyn_cast<ConstantExpr>(DeclAttr->getID())) {
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
  D->addAttr(::new (Context) SYCLIntelPipeIOAttr(Context, CI, E));
}

// Handles [[intel::loop_fuse]] and [[intel::loop_fuse_independent]].
void SemaSYCL::addSYCLIntelLoopFuseAttr(Decl *D, const AttributeCommonInfo &CI,
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

    // This attribute requires a non-negative value.
    if (ArgVal < 0) {
      Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
          << CI << /*non-negative*/ 1;
      return;
    }
    // Check to see if there's a duplicate attribute with different values
    // already applied to the declaration.
    if (const auto *DeclAttr = D->getAttr<SYCLIntelLoopFuseAttr>()) {
      // [[intel::loop_fuse]] and [[intel::loop_fuse_independent]] are
      // incompatible.
      // FIXME: If additional spellings are provided for this attribute,
      // this code will do the wrong thing.
      if (DeclAttr->getAttributeSpellingListIndex() !=
          CI.getAttributeSpellingListIndex()) {
        Diag(CI.getLoc(), diag::err_attributes_are_not_compatible)
            << CI << DeclAttr << CI.isRegularKeywordAttribute();
        Diag(DeclAttr->getLocation(), diag::note_conflicting_attribute);
        return;
      }
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
  D->addAttr(::new (Context) SYCLIntelLoopFuseAttr(Context, CI, E));
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

void SemaSYCL::addSYCLIntelNumSimdWorkItemsAttr(Decl *D,
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

    // This attribute requires a strictly positive value.
    if (ArgVal <= 0) {
      Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
          << CI << /*positive*/ 0;
      return;
    }

    // Check to see if there's a duplicate attribute with different values
    // already applied to the declaration.
    if (const auto *DeclAttr = D->getAttr<SYCLIntelNumSimdWorkItemsAttr>()) {
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

    // If the 'reqd_work_group_size' attribute is specified on a declaration
    // along with 'num_simd_work_items' attribute, the required work group size
    // specified by 'num_simd_work_items' attribute must evenly divide the index
    // that increments fastest in the 'reqd_work_group_size' attribute.
    if (const auto *DeclAttr = D->getAttr<SYCLReqdWorkGroupSizeAttr>()) {
      if (checkWorkGroupSize(E, DeclAttr->getXDim(), DeclAttr->getYDim(),
                             DeclAttr->getZDim())) {
        Diag(CI.getLoc(), diag::err_sycl_num_kernel_wrong_reqd_wg_size)
            << CI << DeclAttr;
        Diag(DeclAttr->getLoc(), diag::note_conflicting_attribute);
        return;
      }
    }
  }

  ASTContext &Context = getASTContext();
  D->addAttr(::new (Context) SYCLIntelNumSimdWorkItemsAttr(Context, CI, E));
}

// Handle scheduler_target_fmax_mhz
void SemaSYCL::addSYCLIntelSchedulerTargetFmaxMhzAttr(
    Decl *D, const AttributeCommonInfo &CI, Expr *E) {
  if (!E->isValueDependent()) {
    // Validate that we have an integer constant expression and then store the
    // converted constant expression into the semantic attribute so that we
    // don't have to evaluate it again later.
    llvm::APSInt ArgVal;
    ExprResult Res = SemaRef.VerifyIntegerConstantExpression(E, &ArgVal);
    if (Res.isInvalid())
      return;
    E = Res.get();

    // This attribute requires a non-negative value.
    if (ArgVal < 0) {
      Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
          << CI << /*non-negative*/ 1;
      return;
    }
    // Check to see if there's a duplicate attribute with different values
    // already applied to the declaration.
    if (const auto *DeclAttr =
            D->getAttr<SYCLIntelSchedulerTargetFmaxMhzAttr>()) {
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
  D->addAttr(::new (Context)
                 SYCLIntelSchedulerTargetFmaxMhzAttr(Context, CI, E));
}

void SemaSYCL::addSYCLIntelNoGlobalWorkOffsetAttr(Decl *D,
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

    // Check to see if there's a duplicate attribute with different values
    // already applied to the declaration.
    if (const auto *DeclAttr = D->getAttr<SYCLIntelNoGlobalWorkOffsetAttr>()) {
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
  D->addAttr(::new (Context) SYCLIntelNoGlobalWorkOffsetAttr(Context, CI, E));
}

void SemaSYCL::addSYCLIntelMaxGlobalWorkDimAttr(Decl *D,
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

    // This attribute must be in the range [0, 3].
    if (ArgVal < 0 || ArgVal > 3) {
      Diag(E->getBeginLoc(), diag::err_attribute_argument_out_of_range)
          << CI << 0 << 3 << E->getSourceRange();
      return;
    }

    // Check to see if there's a duplicate attribute with different values
    // already applied to the declaration.
    if (const auto *DeclAttr = D->getAttr<SYCLIntelMaxGlobalWorkDimAttr>()) {
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

    // If the declaration has a SYCLIntelMaxWorkGroupSizeAttr or
    // SYCLReqdWorkGroupSizeAttr, check to see if the attribute holds values
    // equal to (1, 1, 1) in case the value of SYCLIntelMaxGlobalWorkDimAttr
    // equals to 0.
    if (ArgVal == 0) {
      if (checkWorkGroupSizeAttrExpr<SYCLIntelMaxWorkGroupSizeAttr>(D, CI) ||
          checkWorkGroupSizeAttrExpr<SYCLReqdWorkGroupSizeAttr>(D, CI))
        return;
    }
  }

  ASTContext &Context = getASTContext();
  D->addAttr(::new (Context) SYCLIntelMaxGlobalWorkDimAttr(Context, CI, E));
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

    if (!D->hasAttr<SYCLIntelMaxWorkGroupSizeAttr>()) {
      Diag(CI.getLoc(), diag::warn_launch_bounds_missing_attr) << CI << 0;
      return;
    }
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

    if (!D->hasAttr<SYCLIntelMaxWorkGroupSizeAttr>() ||
        !D->hasAttr<SYCLIntelMinWorkGroupsPerComputeUnitAttr>()) {
      Diag(CI.getLoc(), diag::warn_launch_bounds_missing_attr) << CI << 1;
      return;
    }
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

void SemaSYCL::addSYCLIntelMaxConcurrencyAttr(Decl *D,
                                              const AttributeCommonInfo &CI,
                                              Expr *E) {
  if (!E->isValueDependent()) {
    llvm::APSInt ArgVal;
    ExprResult Res = SemaRef.VerifyIntegerConstantExpression(E, &ArgVal);
    if (Res.isInvalid())
      return;
    E = Res.get();

    // This attribute requires a non-negative value.
    if (ArgVal < 0) {
      Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
          << CI << /*non-negative*/ 1;
      return;
    }

    // Check to see if there's a duplicate attribute with different values
    // already applied to the declaration.
    if (const auto *DeclAttr = D->getAttr<SYCLIntelMaxConcurrencyAttr>()) {
      // If the other attribute argument is instantiation dependent, we won't
      // have converted it to a constant expression yet and thus we test
      // whether this is a null pointer.
      if (const auto *DeclExpr = dyn_cast<ConstantExpr>(DeclAttr->getNExpr())) {
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
  D->addAttr(::new (Context) SYCLIntelMaxConcurrencyAttr(Context, CI, E));
}

void SemaSYCL::addSYCLIntelPrivateCopiesAttr(Decl *D,
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
    // This attribute requires a non-negative value.
    if (ArgVal < 0) {
      Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
          << CI << /*non-negative*/ 1;
      return;
    }

    // Check attribute applies to field as well as const variables, non-static
    // local variables, non-static data members, and device_global variables.
    // for the device compilation.
    if (const auto *VD = dyn_cast<VarDecl>(D)) {
      if (Context.getLangOpts().SYCLIsDevice &&
          (!(isa<FieldDecl>(D) ||
             (VD->getKind() != Decl::ImplicitParam &&
              VD->getKind() != Decl::NonTypeTemplateParm &&
              VD->getKind() != Decl::ParmVar &&
              (VD->hasLocalStorage() ||
               isTypeDecoratedWithDeclAttribute<SYCLDeviceGlobalAttr>(
                   VD->getType())))))) {
        Diag(CI.getLoc(), diag::err_fpga_attribute_invalid_decl) << CI;
        return;
      }
    }

    // Check to see if there's a duplicate attribute with different values
    // already applied to the declaration.
    if (const auto *DeclAttr = D->getAttr<SYCLIntelPrivateCopiesAttr>()) {
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

  // If the declaration does not have [[intel::fpga_memory]]
  // attribute, this creates default implicit memory.
  if (!D->hasAttr<SYCLIntelMemoryAttr>())
    D->addAttr(SYCLIntelMemoryAttr::CreateImplicit(
        Context, SYCLIntelMemoryAttr::Default));

  D->addAttr(::new (Context) SYCLIntelPrivateCopiesAttr(Context, CI, E));
}

void SemaSYCL::addSYCLIntelMaxReplicatesAttr(Decl *D,
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
    // This attribute requires a strictly positive value.
    if (ArgVal <= 0) {
      Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
          << CI << /*positive*/ 0;
      return;
    }

    // Check attribute applies to field, constant variables, local variables,
    // static variables, agent memory arguments, non-static data members,
    // and device_global variables for the device compilation.
    if (checkValidFPGAMemoryAttributesVar(D)) {
      Diag(CI.getLoc(), diag::err_fpga_attribute_incorrect_variable)
          << CI << /*agent memory arguments*/ 1;
      return;
    }

    // Check to see if there's a duplicate attribute with different values
    // already applied to the declaration.
    if (const auto *DeclAttr = D->getAttr<SYCLIntelMaxReplicatesAttr>()) {
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

  // If the declaration does not have an [[intel::fpga_memory]]
  // attribute, this creates one as an implicit attribute.
  ASTContext &Context = getASTContext();
  if (!D->hasAttr<SYCLIntelMemoryAttr>())
    D->addAttr(SYCLIntelMemoryAttr::CreateImplicit(
        Context, SYCLIntelMemoryAttr::Default));

  D->addAttr(::new (Context) SYCLIntelMaxReplicatesAttr(Context, CI, E));
}

// Handles initiation_interval attribute.
void SemaSYCL::addSYCLIntelInitiationIntervalAttr(Decl *D,
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
    // This attribute requires a strictly positive value.
    if (ArgVal <= 0) {
      Diag(E->getExprLoc(), diag::err_attribute_requires_positive_integer)
          << CI << /*positive*/ 0;
      return;
    }
    // Check to see if there's a duplicate attribute with different values
    // already applied to the declaration.
    if (const auto *DeclAttr = D->getAttr<SYCLIntelInitiationIntervalAttr>()) {
      // If the other attribute argument is instantiation dependent, we won't
      // have converted it to a constant expression yet and thus we test
      // whether this is a null pointer.
      if (const auto *DeclExpr = dyn_cast<ConstantExpr>(DeclAttr->getNExpr())) {
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
  D->addAttr(::new (Context) SYCLIntelInitiationIntervalAttr(Context, CI, E));
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

void SemaSYCL::addSYCLIntelMaxWorkGroupSizeAttr(Decl *D,
                                                const AttributeCommonInfo &CI,
                                                Expr *XDim, Expr *YDim,
                                                Expr *ZDim) {
  // Returns nullptr if diagnosing, otherwise returns the original expression
  // or the original expression converted to a constant expression.
  auto CheckAndConvertArg = [&](Expr *E) -> Expr * {
    // Check if the expression is not value dependent.
    if (!E->isValueDependent()) {
      llvm::APSInt ArgVal;
      ExprResult Res = SemaRef.VerifyIntegerConstantExpression(E, &ArgVal);
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
    return E;
  };

  // Check all three argument values, and if any are bad, bail out. This will
  // convert the given expressions into constant expressions when possible.
  XDim = CheckAndConvertArg(XDim);
  YDim = CheckAndConvertArg(YDim);
  ZDim = CheckAndConvertArg(ZDim);
  if (!XDim || !YDim || !ZDim)
    return;

  // If the 'max_work_group_size' attribute is specified on a declaration along
  // with 'reqd_work_group_size' attribute, check to see if values of
  // 'reqd_work_group_size' attribute arguments are equal to or less than values
  // of 'max_work_group_size' attribute arguments.
  //
  // We emit diagnostic if values of 'reqd_work_group_size' attribute arguments
  // are greater than values of 'max_work_group_size' attribute arguments.
  if (const auto *DeclAttr = D->getAttr<SYCLReqdWorkGroupSizeAttr>()) {
    if (checkMaxAllowedWorkGroupSize(DeclAttr->getXDim(), DeclAttr->getYDim(),
                                     DeclAttr->getZDim(), XDim, YDim, ZDim)) {
      Diag(CI.getLoc(), diag::err_conflicting_sycl_function_attributes)
          << CI << DeclAttr;
      Diag(DeclAttr->getLoc(), diag::note_conflicting_attribute);
      return;
    }
  }

  // If the declaration has a SYCLIntelMaxWorkGroupSizeAttr, check to see if
  // the attribute holds values equal to (1, 1, 1) in case the value of
  // SYCLIntelMaxGlobalWorkDimAttr equals to 0.
  if (const auto *DeclAttr = D->getAttr<SYCLIntelMaxGlobalWorkDimAttr>()) {
    if (areInvalidWorkGroupSizeAttrs(DeclAttr->getValue(), XDim, YDim, ZDim)) {
      Diag(CI.getLoc(), diag::err_sycl_x_y_z_arguments_must_be_one)
          << CI << DeclAttr;
      return;
    }
  }

  // If the attribute was already applied with different arguments, then
  // diagnose the second attribute as a duplicate and don't add it.
  if (const auto *Existing = D->getAttr<SYCLIntelMaxWorkGroupSizeAttr>()) {
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
                 SYCLIntelMaxWorkGroupSizeAttr(Context, CI, XDim, YDim, ZDim));
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

  // If the declaration has a ReqdWorkGroupSizeAttr, check to see if
  // the attribute holds values equal to (1, 1, 1) in case the value of
  // SYCLIntelMaxGlobalWorkDimAttr equals to 0.
  if (const auto *DeclAttr = D->getAttr<SYCLIntelMaxGlobalWorkDimAttr>()) {
    if (areInvalidWorkGroupSizeAttrs(DeclAttr->getValue(), XDim, YDim, ZDim)) {
      Diag(CI.getLoc(), diag::err_sycl_x_y_z_arguments_must_be_one)
          << CI << DeclAttr;
    }
  }

  // If the 'max_work_group_size' attribute is specified on a declaration along
  // with 'reqd_work_group_size' attribute, check to see if values of
  // 'reqd_work_group_size' attribute arguments are equal to or less than values
  // of 'max_work_group_size' attribute arguments.
  //
  // We emit diagnostic if values of 'reqd_work_group_size' attribute arguments
  // are greater than values of 'max_work_group_size' attribute arguments.
  if (const auto *DeclAttr = D->getAttr<SYCLIntelMaxWorkGroupSizeAttr>()) {
    if (checkMaxAllowedWorkGroupSize(XDim, YDim, ZDim, DeclAttr->getXDim(),
                                     DeclAttr->getYDim(),
                                     DeclAttr->getZDim())) {
      Diag(CI.getLoc(), diag::err_conflicting_sycl_function_attributes)
          << CI << DeclAttr;
      Diag(DeclAttr->getLoc(), diag::note_conflicting_attribute);
      return;
    }
  }

  // If the 'reqd_work_group_size' attribute is specified on a declaration
  // along with 'num_simd_work_items' attribute, the required work group size
  // specified by 'num_simd_work_items' attribute must evenly divide the index
  // that increments fastest in the 'reqd_work_group_size' attribute.
  if (const auto *DeclAttr = D->getAttr<SYCLIntelNumSimdWorkItemsAttr>()) {
    if (checkWorkGroupSize(DeclAttr->getValue(), XDim, YDim, ZDim)) {
      Diag(DeclAttr->getLoc(), diag::err_sycl_num_kernel_wrong_reqd_wg_size)
          << DeclAttr << CI;
      Diag(CI.getLoc(), diag::note_conflicting_attribute);
      return;
    }
  }

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
