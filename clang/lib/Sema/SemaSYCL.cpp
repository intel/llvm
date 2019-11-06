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
#include "clang/Basic/Diagnostic.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <array>

using namespace clang;

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

  /// Checks whether given clang type is a standard SYCL API class with given
  /// name.
  /// \param Ty    the clang type being checked
  /// \param Name  the class name checked against
  /// \param Tmpl  whether the class is template instantiation or simple record
  static bool isSyclType(const QualType &Ty, StringRef Name, bool Tmpl = false);

  /// Checks whether given clang type is declared in the given hierarchy of
  /// declaration contexts.
  /// \param Ty         the clang type being checked
  /// \param Scopes     the declaration scopes leading from the type to the
  ///     translation unit (excluding the latter)
  static bool matchQualifiedTypeName(const QualType &Ty,
                                     ArrayRef<Util::DeclContextDesc> Scopes);
};

static CXXRecordDecl *getKernelObjectType(FunctionDecl *Caller) {
  return (*Caller->param_begin())->getType()->getAsCXXRecordDecl();
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
  case Builtin::BIceilf:
  case Builtin::BI__builtin_ceilf:
  case Builtin::BIcopysignf:
  case Builtin::BI__builtin_copysignf:
  case Builtin::BIcosf:
  case Builtin::BI__builtin_cosf:
  case Builtin::BIexpf:
  case Builtin::BI__builtin_expf:
  case Builtin::BIexp2f:
  case Builtin::BI__builtin_exp2f:
  case Builtin::BIfabsf:
  case Builtin::BI__builtin_fabsf:
  case Builtin::BIfloorf:
  case Builtin::BI__builtin_floorf:
  case Builtin::BIfmaf:
  case Builtin::BI__builtin_fmaf:
  case Builtin::BIfmaxf:
  case Builtin::BI__builtin_fmaxf:
  case Builtin::BIfminf:
  case Builtin::BI__builtin_fminf:
  case Builtin::BIfmodf:
  case Builtin::BI__builtin_fmodf:
  case Builtin::BIlogf:
  case Builtin::BI__builtin_logf:
  case Builtin::BIlog10f:
  case Builtin::BI__builtin_log10f:
  case Builtin::BIlog2f:
  case Builtin::BI__builtin_log2f:
  case Builtin::BIpowf:
  case Builtin::BI__builtin_powf:
  case Builtin::BIrintf:
  case Builtin::BI__builtin_rintf:
  case Builtin::BIroundf:
  case Builtin::BI__builtin_roundf:
  case Builtin::BIsinf:
  case Builtin::BI__builtin_sinf:
  case Builtin::BIsqrtf:
  case Builtin::BI__builtin_sqrtf:
  case Builtin::BItruncf:
  case Builtin::BI__builtin_truncf:
  case Builtin::BIlroundf:
  case Builtin::BI__builtin_lroundf:
    return false;
  default:
    break;
  }
  return true;
}

class MarkDeviceFunction : public RecursiveASTVisitor<MarkDeviceFunction> {
public:
  MarkDeviceFunction(Sema &S)
      : RecursiveASTVisitor<MarkDeviceFunction>(), SemaRef(S) {}

  bool VisitCallExpr(CallExpr *e) {
    for (const auto &Arg : e->arguments())
      CheckSYCLType(Arg->getType(), Arg->getSourceRange());

    if (FunctionDecl *Callee = e->getDirectCallee()) {
      Callee = Callee->getCanonicalDecl();
      // Remember that all SYCL kernel functions have deferred
      // instantiation as template functions. It means that
      // all functions used by kernel have already been parsed and have
      // definitions.
      if (RecursiveSet.count(Callee)) {
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

      CheckSYCLType(Callee->getReturnType(), Callee->getSourceRange());

      if (FunctionDecl *Def = Callee->getDefinition()) {
        if (!Def->hasAttr<SYCLDeviceAttr>()) {
          Def->addAttr(SYCLDeviceAttr::CreateImplicit(SemaRef.Context));
          SemaRef.addSyclDeviceDecl(Def);
        }
      }
      if (auto const *FD = dyn_cast<FunctionDecl>(Callee)) {
        //FIXME: We need check all target specified attributes for error if that
        //function with attribute can not be called from sycl kernel.  The info
        //is in ParsedAttr. We don't have to map from Attr to ParsedAttr
        //currently. Erich is currently working on that in LLVM, once that is
        //committed we need to change this".
        if (FD->hasAttr<DLLImportAttr>()) {
          SemaRef.Diag(e->getExprLoc(), diag::err_sycl_restrict)
            << Sema::KernelCallDllimportFunction;
          SemaRef.Diag(FD->getLocation(), diag::note_callee_decl) << FD;
        }
      }
      // Specifically check if the math library function corresponding to this
      // builtin is supported for SYCL
      unsigned BuiltinID = (Callee ? Callee->getBuiltinID() : 0);
      if (BuiltinID && !IsSyclMathFunc(BuiltinID)) {
        StringRef Name = SemaRef.Context.BuiltinInfo.getName(BuiltinID);
        SemaRef.Diag(e->getExprLoc(),
                     diag::err_builtin_target_unsupported)
            << Name << "SYCL device";
      }
    } else if ((!SemaRef.getLangOpts().SYCLAllowFuncPtr) &&
               !e->isTypeDependent())
      SemaRef.Diag(e->getExprLoc(), diag::err_sycl_restrict)
          << Sema::KernelCallFunctionPointer;
    return true;
  }

  bool VisitCXXConstructExpr(CXXConstructExpr *E) {
    for (const auto &Arg : E->arguments())
      CheckSYCLType(Arg->getType(), Arg->getSourceRange());

    CXXConstructorDecl *Ctor = E->getConstructor();

    if (FunctionDecl *Def = Ctor->getDefinition()) {
      Def->addAttr(SYCLDeviceAttr::CreateImplicit(SemaRef.Context));
      SemaRef.addSyclDeviceDecl(Def);
    }

    const auto *ConstructedType = Ctor->getParent();
    if (ConstructedType->hasUserDeclaredDestructor()) {
      CXXDestructorDecl *Dtor = ConstructedType->getDestructor();

      if (FunctionDecl *Def = Dtor->getDefinition()) {
        Def->addAttr(SYCLDeviceAttr::CreateImplicit(SemaRef.Context));
        SemaRef.addSyclDeviceDecl(Def);
      }
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

  bool VisitTypedefNameDecl(TypedefNameDecl *TD) {
    CheckSYCLType(TD->getUnderlyingType(), TD->getLocation());
    return true;
  }

  bool VisitRecordDecl(RecordDecl *RD) {
    CheckSYCLType(QualType{RD->getTypeForDecl(), 0}, RD->getLocation());
    return true;
  }

  bool VisitParmVarDecl(VarDecl *VD) {
    CheckSYCLType(VD->getType(), VD->getLocation());
    return true;
  }

  bool VisitVarDecl(VarDecl *VD) {
    CheckSYCLType(VD->getType(), VD->getLocation());
    return true;
  }

  bool VisitMemberExpr(MemberExpr *E) {
    if (VarDecl *VD = dyn_cast<VarDecl>(E->getMemberDecl())) {
      bool IsConst = VD->getType().getNonReferenceType().isConstQualified();
      if (!IsConst && VD->isStaticDataMember())
        SemaRef.Diag(E->getExprLoc(), diag::err_sycl_restrict)
            << Sema::KernelNonConstStaticDataVariable;
    }
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *E) {
    CheckSYCLType(E->getType(), E->getSourceRange());
    if (VarDecl *VD = dyn_cast<VarDecl>(E->getDecl())) {
      bool IsConst = VD->getType().getNonReferenceType().isConstQualified();
      if (!IsConst && VD->isStaticDataMember())
        SemaRef.Diag(E->getExprLoc(), diag::err_sycl_restrict)
            << Sema::KernelNonConstStaticDataVariable;
      else if (!IsConst && VD->hasGlobalStorage() && !VD->isStaticLocal() &&
          !VD->isStaticDataMember() && !isa<ParmVarDecl>(VD)) {
        if (VD->getTLSKind() != VarDecl::TLS_None)
          SemaRef.Diag(E->getLocation(), diag::err_thread_unsupported);
        SemaRef.Diag(E->getLocation(), diag::err_sycl_restrict)
            << Sema::KernelGlobalVariable;
      }
      if (!VD->isLocalVarDeclOrParm() && VD->hasGlobalStorage()) {
        VD->addAttr(SYCLDeviceAttr::CreateImplicit(SemaRef.Context));
        SemaRef.addSyclDeviceDecl(VD);
      }
    }
    return true;
  }

  bool VisitCXXNewExpr(CXXNewExpr *E) {
    // Memory storage allocation is not allowed in kernels.
    // All memory allocation for the device is done on
    // the host using accessor classes. Consequently, the default
    // allocation operator new overloads that allocate
    // storage are disallowed in a SYCL kernel. The placement
    // new operator and any user-defined overloads that
    // do not allocate storage are permitted.
    if (FunctionDecl *FD = E->getOperatorNew()) {
      if (FD->isReplaceableGlobalAllocationFunction()) {
        SemaRef.Diag(E->getExprLoc(), diag::err_sycl_restrict)
            << Sema::KernelAllocateStorage;
      } else if (FunctionDecl *Def = FD->getDefinition()) {
        if (!Def->hasAttr<SYCLDeviceAttr>()) {
          Def->addAttr(SYCLDeviceAttr::CreateImplicit(SemaRef.Context));
          SemaRef.addSyclDeviceDecl(Def);
        }
      }
    }
    return true;
  }
  
  bool VisitGCCAsmStmt(GCCAsmStmt *S) {
    SemaRef.Diag(S->getBeginLoc(), diag::err_sycl_restrict)
        << Sema::KernelUseAssembly;
    return true;
  }
 
  bool VisitMSAsmStmt(MSAsmStmt *S) {
    SemaRef.Diag(S->getBeginLoc(), diag::err_sycl_restrict)
        << Sema::KernelUseAssembly;
    return true;
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
  void CollectPossibleKernelAttributes(FunctionDecl *SYCLKernel,
                                       llvm::SmallPtrSet<Attr *, 4> &Attrs) {
    typedef std::pair<FunctionDecl *, FunctionDecl *> ChildParentPair;
    llvm::SmallPtrSet<FunctionDecl *, 16> Visited;
    llvm::SmallVector<ChildParentPair, 16> WorkList;
    WorkList.push_back({SYCLKernel, nullptr});

    while (!WorkList.empty()) {
      FunctionDecl *FD = WorkList.back().first;
      FunctionDecl *ParentFD = WorkList.back().second;
      WorkList.pop_back();
      if (!Visited.insert(FD).second)
        continue; // We've already seen this Decl

      if (auto *A = FD->getAttr<IntelReqdSubGroupSizeAttr>())
        Attrs.insert(A);
      else if (auto *A = FD->getAttr<ReqdWorkGroupSizeAttr>())
        Attrs.insert(A);
      else if (auto *A = FD->getAttr<SYCLIntelKernelArgsRestrictAttr>()) {
        // Allow the intel::kernel_args_restrict only on the lambda (function
        // object) function, that is called directly from a kernel (i.e. the one
        // passed to the parallel_for function). Emit a warning and ignore all
        // other cases.
        if (ParentFD == SYCLKernel) {
          Attrs.insert(A);
        } else {
          SemaRef.Diag(A->getLocation(), diag::warn_attribute_ignored) << A;
          FD->dropAttr<SYCLIntelKernelArgsRestrictAttr>();
        }
      }

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
  }

private:
  bool CheckSYCLType(QualType Ty, SourceRange Loc) {
    llvm::DenseSet<QualType> visited;
    return CheckSYCLType(Ty, Loc, visited);
  }

  bool CheckSYCLType(QualType Ty, SourceRange Loc, llvm::DenseSet<QualType> &Visited) {
    if (Ty->isVariableArrayType()) {
      SemaRef.Diag(Loc.getBegin(), diag::err_vla_unsupported);
      return false;
    }

    while (Ty->isAnyPointerType() || Ty->isArrayType())
      Ty = QualType{Ty->getPointeeOrArrayElementType(), 0};

    // Pointers complicate recursion. Add this type to Visited.
    // If already there, bail out.
    if (!Visited.insert(Ty).second)
      return true;
    
    if (const auto *ATy = dyn_cast<AttributedType>(Ty))
      return CheckSYCLType(ATy->getModifiedType(), Loc, Visited);

    if (const auto *CRD = Ty->getAsCXXRecordDecl()) {
      // If the class is a forward declaration - skip it, because otherwise we
      // would query property of class with no definition, which results in
      // clang crash.
      if (!CRD->hasDefinition())
        return true;

      if (CRD->isPolymorphic()) {
        // Exceptions aren't allowed in SYCL device code.
        if (SemaRef.getLangOpts().SYCLIsDevice) {
          SemaRef.SYCLDiagIfDeviceCode(CRD->getLocation(),
			               diag::err_sycl_restrict)
                  << Sema::KernelHavePolymorphicClass;
          SemaRef.SYCLDiagIfDeviceCode(Loc.getBegin(),
                                       diag::note_sycl_used_here);
	}
        return false;
      }

      for (const auto &Field : CRD->fields()) {
        if (!CheckSYCLType(Field->getType(), Field->getSourceRange(), Visited)) {
          if (SemaRef.getLangOpts().SYCLIsDevice)
            SemaRef.SYCLDiagIfDeviceCode(Loc.getBegin(),
	                                 diag::note_sycl_used_here);
          return false;
        }
      }
    } else if (const auto *RD = Ty->getAsRecordDecl()) {
      for (const auto &Field : RD->fields()) {
        if (!CheckSYCLType(Field->getType(), Field->getSourceRange(), Visited)) {
          if (SemaRef.getLangOpts().SYCLIsDevice)
            SemaRef.SYCLDiagIfDeviceCode(Loc.getBegin(), 
	                                 diag::note_sycl_used_here);
          return false;
        }
      }
    } else if (const auto *FPTy = dyn_cast<FunctionProtoType>(Ty)) {
      if (FPTy->isVariadic() && SemaRef.getLangOpts().SYCLIsDevice)
	SemaRef.SYCLDiagIfDeviceCode(Loc.getBegin(), diag::err_sycl_restrict)
            << Sema::KernelCallVariadicFunction;
      for (const auto &ParamTy : FPTy->param_types())
        if (!CheckSYCLType(ParamTy, Loc, Visited))
          return false;
      return CheckSYCLType(FPTy->getReturnType(), Loc, Visited);
    } else if (const auto *FTy = dyn_cast<FunctionType>(Ty)) {
      return CheckSYCLType(FTy->getReturnType(), Loc, Visited);
    }
    return true;
  }
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
                                 SourceLocation RBraceLoc,
                                 bool IsStmtExpr) {
    // Build a new compound statement but clear the source locations.
    return getSema().ActOnCompoundStmt(SourceLocation(), SourceLocation(),
                                       Statements, IsStmtExpr);
  }

private:
  std::pair<DeclaratorDecl *, DeclaratorDecl *> MappingPair;
  Sema &SemaRef;
};

static KernelInvocationKind
getKernelInvocationKind(FunctionDecl *KernelCallerFunc) {
  return llvm::StringSwitch<KernelInvocationKind>(KernelCallerFunc->getName())
      .Case("kernel_single_task", InvokeSingleTask)
      .Case("kernel_parallel_for", InvokeParallelFor)
      .Case("kernel_parallel_for_work_group", InvokeParallelForWorkGroup)
      .Default(InvokeUnknown);
}

static FunctionDecl *
CreateOpenCLKernelDeclaration(ASTContext &Context, StringRef Name,
                              ArrayRef<ParamDesc> ParamDescs) {

  DeclContext *DC = Context.getTranslationUnitDecl();
  QualType RetTy = Context.VoidTy;
  SmallVector<QualType, 8> ArgTys;

  // Extract argument types from the descriptor array:
  std::transform(
      ParamDescs.begin(), ParamDescs.end(), std::back_inserter(ArgTys),
      [](const ParamDesc &PD) -> QualType { return std::get<0>(PD); });
  FunctionProtoType::ExtProtoInfo Info(CC_OpenCLKernel);
  QualType FuncTy = Context.getFunctionType(RetTy, ArgTys, Info);
  DeclarationName DN = DeclarationName(&Context.Idents.get(Name));

  FunctionDecl *OpenCLKernel = FunctionDecl::Create(
      Context, DC, SourceLocation(), SourceLocation(), DN, FuncTy,
      Context.getTrivialTypeSourceInfo(RetTy), SC_None);

  llvm::SmallVector<ParmVarDecl *, 16> Params;
  int i = 0;
  for (const auto &PD : ParamDescs) {
    auto P = ParmVarDecl::Create(Context, OpenCLKernel, SourceLocation(),
                                 SourceLocation(), std::get<1>(PD),
                                 std::get<0>(PD), std::get<2>(PD), SC_None, 0);
    P->setScopeInfo(0, i++);
    P->setIsUsed();
    Params.push_back(P);
  }
  OpenCLKernel->setParams(Params);

  OpenCLKernel->addAttr(SYCLDeviceAttr::CreateImplicit(Context));
  OpenCLKernel->addAttr(OpenCLKernelAttr::CreateImplicit(Context));
  OpenCLKernel->addAttr(AsmLabelAttr::CreateImplicit(Context, Name));
  OpenCLKernel->addAttr(ArtificialAttr::CreateImplicit(Context));

  // Add kernel to translation unit to see it in AST-dump
  DC->addDecl(OpenCLKernel);
  return OpenCLKernel;
}
/// Return __init method
static CXXMethodDecl *getInitMethod(const CXXRecordDecl *CRD) {
  CXXMethodDecl *InitMethod;
  auto It = std::find_if(CRD->methods().begin(), CRD->methods().end(),
                         [](const CXXMethodDecl *Method) {
                           return Method->getNameAsString() == "__init";
                         });
  InitMethod = (It != CRD->methods().end()) ? *It : nullptr;
  return InitMethod;
}

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
    if (!Util::isSyclType(Call->getArg(1)->getType(), "group", true /*tmpl*/))
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

// Creates body for new OpenCL kernel. This body contains initialization of SYCL
// kernel object fields with kernel parameters and a little bit transformed body
// of the kernel caller function.
static CompoundStmt *CreateOpenCLKernelBody(Sema &S,
                                            FunctionDecl *KernelCallerFunc,
                                            DeclContext *KernelDecl) {
  llvm::SmallVector<Stmt *, 16> BodyStmts;
  CXXRecordDecl *LC = getKernelObjectType(KernelCallerFunc);
  assert(LC && "Kernel object must be available");

  if (getKernelInvocationKind(KernelCallerFunc) == InvokeParallelForWorkGroup) {
    CXXRecordDecl *LambdaObjTy =
        KernelCallerFunc->getParamDecl(0)->getType()->getAsCXXRecordDecl();
    assert(LambdaObjTy &&
           "unexpected kernel_parallel_for_work_group parameter type");
    FindPFWGLambdaFnVisitor V(LambdaObjTy);
    V.TraverseStmt(KernelCallerFunc->getBody());
    CXXMethodDecl *WGLambdaFn = V.getLambdaFn();
    assert(WGLambdaFn && "PFWG lambda not found");
    // Mark the function that it "works" in a work group scope:
    // NOTE: In case of parallel_for_work_item the marker call itself is marked
    //       with work item scope attribute, here  the '()' operator of the
    //       object passed as parameter is marked. This is an optimization -
    //       there are a lot of locals created at parallel_for_work_group scope
    //       before calling the lambda - it is more efficient to have all of
    //       them in the private address space rather then sharing via the local
    //       AS. See parallel_for_work_group implementation in the SYCL headers.
    if (!WGLambdaFn->hasAttr<SYCLScopeAttr>()) {
      WGLambdaFn->addAttr(SYCLScopeAttr::CreateImplicit(
          S.getASTContext(), SYCLScopeAttr::Level::WorkGroup));
      // Search and mark parallel_for_work_item calls:
      MarkWIScopeFnVisitor MarkWIScope(S.getASTContext());
      MarkWIScope.TraverseDecl(WGLambdaFn);
      // Now mark local variables declared in the PFWG lambda with work group
      // scope attribute
      addScopeAttrToLocalVars(*WGLambdaFn);
    }
  }

  TypeSourceInfo *TSInfo = LC->isLambda() ? LC->getLambdaTypeInfo() : nullptr;

  // Create a local kernel object (lambda or functor) assembled from the
  // incoming formal parameters
  auto KernelObjClone = VarDecl::Create(
      S.Context, KernelDecl, SourceLocation(), SourceLocation(),
      LC->getIdentifier(), QualType(LC->getTypeForDecl(), 0), TSInfo, SC_None);
  Stmt *DS = new (S.Context) DeclStmt(DeclGroupRef(KernelObjClone),
                                      SourceLocation(), SourceLocation());
  BodyStmts.push_back(DS);
  auto KernelObjCloneRef =
      DeclRefExpr::Create(S.Context, NestedNameSpecifierLoc(), SourceLocation(),
                          KernelObjClone, false, DeclarationNameInfo(),
                          QualType(LC->getTypeForDecl(), 0), VK_LValue);

  auto KernelFuncDecl = cast<FunctionDecl>(KernelDecl);
  auto KernelFuncParam =
      KernelFuncDecl->param_begin(); // Iterator to ParamVarDecl (VarDecl)
  if (KernelFuncParam) {
    llvm::SmallVector<Expr *, 16> InitExprs;
    InitializedEntity VarEntity =
        InitializedEntity::InitializeVariable(KernelObjClone);
    for (auto Field : LC->fields()) {
      // Creates Expression for special SYCL object: accessor or sampler.
      // All special SYCL objects must have __init method, here we use it to
      // initialize them. We create call of __init method and pass built kernel
      // arguments as parameters to the __init method.
      auto getExprForSpecialSYCLObj = [&](const QualType &paramTy,
                                          FieldDecl *Field,
                                          const CXXRecordDecl *CRD,
                                          Expr *Base) {
        // All special SYCL objects must have __init method
        CXXMethodDecl *InitMethod = getInitMethod(CRD);
        assert(InitMethod &&
               "The accessor/sampler/stream must have the __init method");
        unsigned NumParams = InitMethod->getNumParams();
        llvm::SmallVector<Expr *, 4> ParamDREs(NumParams);
        auto KFP = KernelFuncParam;
        for (size_t I = 0; I < NumParams; ++KFP, ++I) {
          QualType ParamType = (*KFP)->getOriginalType();
          ParamDREs[I] = DeclRefExpr::Create(
              S.Context, NestedNameSpecifierLoc(), SourceLocation(), *KFP,
              false, DeclarationNameInfo(), ParamType, VK_LValue);
        }

        if (NumParams)
          std::advance(KernelFuncParam, NumParams - 1);

        DeclAccessPair FieldDAP = DeclAccessPair::make(Field, AS_none);
        // [kernel_obj or wrapper object].special_obj
        auto SpecialObjME = MemberExpr::Create(
            S.Context, Base, false, SourceLocation(), NestedNameSpecifierLoc(),
            SourceLocation(), Field, FieldDAP,
            DeclarationNameInfo(Field->getDeclName(), SourceLocation()),
            nullptr, Field->getType(), VK_LValue, OK_Ordinary, NOUR_None);

        // [kernel_obj or wrapper object].special_obj.__init
        DeclAccessPair MethodDAP = DeclAccessPair::make(InitMethod, AS_none);
        auto ME = MemberExpr::Create(
            S.Context, SpecialObjME, false, SourceLocation(),
            NestedNameSpecifierLoc(), SourceLocation(), InitMethod, MethodDAP,
            DeclarationNameInfo(InitMethod->getDeclName(), SourceLocation()),
            nullptr, InitMethod->getType(), VK_LValue, OK_Ordinary, NOUR_None);

        // Not referenced -> not emitted
        S.MarkFunctionReferenced(SourceLocation(), InitMethod, true);

        QualType ResultTy = InitMethod->getReturnType();
        ExprValueKind VK = Expr::getValueKindForType(ResultTy);
        ResultTy = ResultTy.getNonLValueExprType(S.Context);

        llvm::SmallVector<Expr *, 4> ParamStmts;
        const auto *Proto = cast<FunctionProtoType>(InitMethod->getType());
        S.GatherArgumentsForCall(SourceLocation(), InitMethod, Proto, 0,
                                 ParamDREs, ParamStmts);
        // [kernel_obj or wrapper object].accessor.__init(_ValueType*,
        // range<int>, range<int>, id<int>)
        CXXMemberCallExpr *Call = CXXMemberCallExpr::Create(
            S.Context, ME, ParamStmts, ResultTy, VK, SourceLocation());
        BodyStmts.push_back(Call);
      };

      // Recursively search for accessor fields to initialize them with kernel
      // parameters
      std::function<void(const CXXRecordDecl *, Expr *)>
          getExprForWrappedAccessorInit = [&](const CXXRecordDecl *CRD,
                                              Expr *Base) {
            for (auto *WrapperFld : CRD->fields()) {
              QualType FldType = WrapperFld->getType();
              CXXRecordDecl *WrapperFldCRD = FldType->getAsCXXRecordDecl();
              if (FldType->isStructureOrClassType()) {
                if (Util::isSyclAccessorType(FldType)) {
                  // Accessor field found - create expr to initialize this
                  // accessor object. Need to start from the next target
                  // function parameter, since current one is the wrapper object
                  // or parameter of the previous processed accessor object.
                  KernelFuncParam++;
                  getExprForSpecialSYCLObj(FldType, WrapperFld, WrapperFldCRD,
                                           Base);
                } else {
                  // Field is a structure or class so change the wrapper object
                  // and recursively search for accessor field.
                  DeclAccessPair WrapperFieldDAP =
                      DeclAccessPair::make(WrapperFld, AS_none);
                  auto NewBase = MemberExpr::Create(
                      S.Context, Base, false, SourceLocation(),
                      NestedNameSpecifierLoc(), SourceLocation(), WrapperFld,
                      WrapperFieldDAP,
                      DeclarationNameInfo(WrapperFld->getDeclName(),
                                          SourceLocation()),
                      nullptr, WrapperFld->getType(), VK_LValue, OK_Ordinary,
                      NOUR_None);
                  getExprForWrappedAccessorInit(WrapperFldCRD, NewBase);
                }
              }
            }
          };

      // Run through kernel object fields and add initialization for them using
      // built kernel parameters. There are a several possible cases:
      //   - Kernel object field is a SYCL special object (SYCL accessor or SYCL
      //     sampler). These objects has a special initialization scheme - using
      //     __init method.
      //   - Kernel object field has a scalar type. In this case we should add
      //     simple initialization.
      //   - Kernel object field has a structure or class type. Same handling as
      //     a scalar but we should check if this structure/class contains
      //     accessors and add initialization for them properly.
      QualType FieldType = Field->getType();
      CXXRecordDecl *CRD = FieldType->getAsCXXRecordDecl();
      InitializedEntity Entity =
          InitializedEntity::InitializeMember(Field, &VarEntity);
      if (Util::isSyclAccessorType(FieldType) ||
          Util::isSyclSamplerType(FieldType)) {
        // Initialize with the default constructor.
        InitializationKind InitKind =
            InitializationKind::CreateDefault(SourceLocation());
        InitializationSequence InitSeq(S, Entity, InitKind, None);
        ExprResult MemberInit = InitSeq.Perform(S, Entity, InitKind, None);
        InitExprs.push_back(MemberInit.get());
        getExprForSpecialSYCLObj(FieldType, Field, CRD, KernelObjCloneRef);
      } else if (CRD || FieldType->isScalarType()) {
        // If field has built-in or a structure/class type just initialize
        // this field with corresponding kernel argument using copy
        // initialization.
        QualType ParamType = (*KernelFuncParam)->getOriginalType();
        Expr *DRE =
            DeclRefExpr::Create(S.Context, NestedNameSpecifierLoc(),
                                SourceLocation(), *KernelFuncParam, false,
                                DeclarationNameInfo(), ParamType, VK_LValue);

        if (FieldType->isPointerType() &&
            FieldType->getPointeeType().getAddressSpace() !=
                ParamType->getPointeeType().getAddressSpace())
          DRE = ImplicitCastExpr::Create(S.Context, FieldType,
                                         CK_AddressSpaceConversion, DRE,
                                         nullptr, VK_LValue);
        InitializationKind InitKind =
            InitializationKind::CreateCopy(SourceLocation(), SourceLocation());
        InitializationSequence InitSeq(S, Entity, InitKind, DRE);

        ExprResult MemberInit = InitSeq.Perform(S, Entity, InitKind, DRE);
        InitExprs.push_back(MemberInit.get());

        if (CRD) {
          // If a structure/class type has accessor fields then we need to
          // initialize these accessors in proper way by calling __init method
          // of the accessor and passing corresponding kernel parameters.
          DeclAccessPair FieldDAP = DeclAccessPair::make(Field, AS_none);
          auto Lhs = MemberExpr::Create(
              S.Context, KernelObjCloneRef, false, SourceLocation(),
              NestedNameSpecifierLoc(), SourceLocation(), Field, FieldDAP,
              DeclarationNameInfo(Field->getDeclName(), SourceLocation()),
              nullptr, Field->getType(), VK_LValue, OK_Ordinary, NOUR_None);
          getExprForWrappedAccessorInit(CRD, Lhs);
          if (Util::isSyclStreamType(FieldType)) {
            // Generate call to the __init method of the stream class after
            // initializing accessors wrapped by this stream object
            getExprForSpecialSYCLObj(FieldType, Field, CRD, KernelObjCloneRef);
          }
        }
      } else {
        llvm_unreachable("Unsupported field type");
      }
      KernelFuncParam++;
    }
    Expr *ILE = new (S.Context)
        InitListExpr(S.Context, SourceLocation(), InitExprs, SourceLocation());
    ILE->setType(QualType(LC->getTypeForDecl(), 0));
    KernelObjClone->setInit(ILE);
  }

  // In the kernel caller function kernel object is a function parameter, so we
  // need to replace all refs to this kernel oject with refs to our clone
  // declared inside kernel body.
  Stmt *FunctionBody = KernelCallerFunc->getBody();
  ParmVarDecl *KernelObjParam = *(KernelCallerFunc->param_begin());

  // DeclRefExpr with valid source location but with decl which is not marked
  // as used is invalid.
  KernelObjClone->setIsUsed();
  std::pair<DeclaratorDecl *, DeclaratorDecl *> MappingPair;
  MappingPair.first = KernelObjParam;
  MappingPair.second = KernelObjClone;

  // Function scope might be empty, so we do push
  S.PushFunctionScope();
  KernelBodyTransform KBT(MappingPair, S);
  Stmt *NewBody = KBT.TransformStmt(FunctionBody).get();
  BodyStmts.push_back(NewBody);
  return CompoundStmt::Create(S.Context, BodyStmts, SourceLocation(),
                              SourceLocation());
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

/// \return the target of given SYCL accessor type
static target getAccessTarget(const ClassTemplateSpecializationDecl *AccTy) {
  return static_cast<target>(
      AccTy->getTemplateArgs()[3].getAsIntegral().getExtValue());
}

// Creates list of kernel parameters descriptors using KernelObj (kernel object)
// Fields of kernel object must be initialized with SYCL kernel arguments so
// in the following function we extract types of kernel object fields and add it
// to the array with kernel parameters descriptors.
// Returns true if all arguments are successfully built.
static bool buildArgTys(ASTContext &Context, CXXRecordDecl *KernelObj,
                        SmallVectorImpl<ParamDesc> &ParamDescs) {
  const LambdaCapture *Cpt = KernelObj->captures_begin();
  auto CreateAndAddPrmDsc = [&](const FieldDecl *Fld, const QualType &ArgType) {
    // Create a parameter descriptor and append it to the result
    ParamDescs.push_back(makeParamDesc(Fld, ArgType));
  };

  // Creates a parameter descriptor for SYCL special object - SYCL accessor or
  // sampler.
  // All special SYCL objects must have __init method. We extract types for
  // kernel parameters from __init method parameters. We will use __init method
  // and kernel parameters which we build here to initialize special objects in
  // the kernel body.
  auto createSpecialSYCLObjParamDesc = [&](const FieldDecl *Fld,
                                           const QualType &ArgTy) {
    const auto *RecordDecl = ArgTy->getAsCXXRecordDecl();
    assert(RecordDecl && "Special SYCL object must be of a record type");

    CXXMethodDecl *InitMethod = getInitMethod(RecordDecl);
    assert(InitMethod && "The accessor/sampler must have the __init method");
    unsigned NumParams = InitMethod->getNumParams();
    for (size_t I = 0; I < NumParams; ++I) {
      ParmVarDecl *PD = InitMethod->getParamDecl(I);
      CreateAndAddPrmDsc(Fld, PD->getType().getCanonicalType());
    }
  };

  // Create parameter descriptor for accessor in case when it's wrapped with
  // some class.
  // TODO: Do we need support case when sampler is wrapped with some class or
  // struct?
  std::function<void(const FieldDecl *, const QualType &ArgTy)>
      createParamDescForWrappedAccessors =
          [&](const FieldDecl *Fld, const QualType &ArgTy) {
            const auto *Wrapper = ArgTy->getAsCXXRecordDecl();
            for (const auto *WrapperFld : Wrapper->fields()) {
              QualType FldType = WrapperFld->getType();
              if (FldType->isStructureOrClassType()) {
                if (Util::isSyclAccessorType(FldType)) {
                  // accessor field is found - create descriptor
                  createSpecialSYCLObjParamDesc(WrapperFld, FldType);
                } else {
                  // field is some class or struct - recursively check for
                  // accessor fields
                  createParamDescForWrappedAccessors(WrapperFld, FldType);
                }
              }
            }
          };

  bool AllArgsAreValid = true;
  // Run through kernel object fields and create corresponding kernel
  // parameters descriptors. There are a several possible cases:
  //   - Kernel object field is a SYCL special object (SYCL accessor or SYCL
  //     sampler). These objects has a special initialization scheme - using
  //     __init method.
  //   - Kernel object field has a scalar type. In this case we should add
  //     kernel parameter with the same type.
  //   - Kernel object field has a structure or class type. Same handling as a
  //     scalar but we should check if this structure/class contains accessors
  //     and add parameter decriptor for them properly.
  for (const auto *Fld : KernelObj->fields()) {
    QualType ArgTy = Fld->getType();
    if (Util::isSyclAccessorType(ArgTy) || Util::isSyclSamplerType(ArgTy)) {
      createSpecialSYCLObjParamDesc(Fld, ArgTy);
    } else if (!ArgTy->isStandardLayoutType()) {
      // SYCL v1.2.1 s4.8.10 p5:
      // C++ non-standard layout values must not be passed as arguments to a
      // kernel that is compiled for a device.
      const auto &DiagLocation =
          Cpt ? Cpt->getLocation() : cast<DeclaratorDecl>(Fld)->getLocation();

      Context.getDiagnostics().Report(DiagLocation,
                                      diag::err_sycl_non_std_layout_type);

      // Set the flag and continue processing so we can emit error for each
      // invalid argument.
      AllArgsAreValid = false;
    } else if (ArgTy->isStructureOrClassType()) {
      assert(ArgTy->isStandardLayoutType());

      CreateAndAddPrmDsc(Fld, ArgTy);

      // Create descriptors for each accessor field in the class or struct
      createParamDescForWrappedAccessors(Fld, ArgTy);
    } else if (ArgTy->isPointerType()) {
      // Pointer Arguments need to be in the global address space
      QualType PointeeTy = ArgTy->getPointeeType();
      Qualifiers Quals = PointeeTy.getQualifiers();
      Quals.setAddressSpace(LangAS::opencl_global);
      PointeeTy = Context.getQualifiedType(PointeeTy.getUnqualifiedType(),
                                           Quals);
      QualType ModTy = Context.getPointerType(PointeeTy);

      CreateAndAddPrmDsc(Fld, ModTy);
    } else if (ArgTy->isScalarType()) {
      CreateAndAddPrmDsc(Fld, ArgTy);
    } else {
      llvm_unreachable("Unsupported kernel parameter type");
    }

    // Update capture iterator as we process arguments
    if (Cpt && Cpt != KernelObj->captures_end())
      ++Cpt;
  }

  return AllArgsAreValid;
}

/// Adds necessary data describing given kernel to the integration header.
/// \param H           the integration header object
/// \param Name        kernel name
/// \param NameType    type representing kernel name (first template argument
/// of
///                      single_task, parallel_for, etc)
/// \param KernelObjTy kernel object type
static void populateIntHeader(SYCLIntegrationHeader &H, const StringRef Name,
                              QualType NameType, CXXRecordDecl *KernelObjTy) {

  ASTContext &Ctx = KernelObjTy->getASTContext();
  const ASTRecordLayout &Layout = Ctx.getASTRecordLayout(KernelObjTy);
  const std::string StableName = PredefinedExpr::ComputeName(
      Ctx, PredefinedExpr::UniqueStableNameExpr, NameType);
  H.startKernel(Name, NameType, StableName);

  auto populateHeaderForAccessor = [&](const QualType &ArgTy, uint64_t Offset) {
    // The parameter is a SYCL accessor object.
    // The Info field of the parameter descriptor for accessor contains
    // two template parameters packed into an integer field:
    //   - target (e.g. global_buffer, constant_buffer, local);
    //   - dimension of the accessor.
    const auto *AccTy = ArgTy->getAsCXXRecordDecl();
    assert(AccTy && "accessor must be of a record type");
    const auto *AccTmplTy = cast<ClassTemplateSpecializationDecl>(AccTy);
    int Dims = static_cast<int>(
        AccTmplTy->getTemplateArgs()[1].getAsIntegral().getExtValue());
    int Info = getAccessTarget(AccTmplTy) | (Dims << 11);
    H.addParamDesc(SYCLIntegrationHeader::kind_accessor, Info, Offset);
  };

  std::function<void(const QualType &, uint64_t Offset)>
      populateHeaderForWrappedAccessors = [&](const QualType &ArgTy,
                                              uint64_t Offset) {
        const auto *Wrapper = ArgTy->getAsCXXRecordDecl();
        for (const auto *WrapperFld : Wrapper->fields()) {
          QualType FldType = WrapperFld->getType();
          if (FldType->isStructureOrClassType()) {
            ASTContext &WrapperCtx = Wrapper->getASTContext();
            const ASTRecordLayout &WrapperLayout =
                WrapperCtx.getASTRecordLayout(Wrapper);
            // Get offset (in bytes) of the field in wrapper class or struct
            uint64_t OffsetInWrapper =
                WrapperLayout.getFieldOffset(WrapperFld->getFieldIndex()) / 8;
            if (Util::isSyclAccessorType(FldType)) {
              // This is an accesor - populate the header appropriately
              populateHeaderForAccessor(FldType, Offset + OffsetInWrapper);
            } else {
              // This is an other class or struct - recursively search for an
              // accessor field
              populateHeaderForWrappedAccessors(FldType,
                                                Offset + OffsetInWrapper);
            }
          }
        }
      };

  for (const auto Fld : KernelObjTy->fields()) {
    QualType ActualArgType;
    QualType ArgTy = Fld->getType();

    // Get offset in bytes
    uint64_t Offset = Layout.getFieldOffset(Fld->getFieldIndex()) / 8;

    if (Util::isSyclAccessorType(ArgTy)) {
      populateHeaderForAccessor(ArgTy, Offset);
    } else if (Util::isSyclSamplerType(ArgTy)) {
      // The parameter is a SYCL sampler object
      const auto *SamplerTy = ArgTy->getAsCXXRecordDecl();
      assert(SamplerTy && "sampler must be of a record type");

      CXXMethodDecl *InitMethod = getInitMethod(SamplerTy);
      assert(InitMethod && "sampler must have __init method");

      // sampler __init method has only one argument
      auto *FuncDecl = cast<FunctionDecl>(InitMethod);
      ParmVarDecl *SamplerArg = FuncDecl->getParamDecl(0);
      assert(SamplerArg && "sampler __init method must have sampler parameter");
      uint64_t Sz = Ctx.getTypeSizeInChars(SamplerArg->getType()).getQuantity();
      H.addParamDesc(SYCLIntegrationHeader::kind_sampler,
                     static_cast<unsigned>(Sz), static_cast<unsigned>(Offset));
    } else if (ArgTy->isPointerType()) {
      uint64_t Sz = Ctx.getTypeSizeInChars(Fld->getType()).getQuantity();
      H.addParamDesc(SYCLIntegrationHeader::kind_pointer,
                     static_cast<unsigned>(Sz), static_cast<unsigned>(Offset));
    } else if (ArgTy->isStructureOrClassType() || ArgTy->isScalarType()) {
      // the parameter is an object of standard layout type or scalar;
      // the check for standard layout is done elsewhere
      uint64_t Sz = Ctx.getTypeSizeInChars(Fld->getType()).getQuantity();
      H.addParamDesc(SYCLIntegrationHeader::kind_std_layout,
                     static_cast<unsigned>(Sz), static_cast<unsigned>(Offset));

      // check for accessor fields in structure or class and populate the
      // integration header appropriately
      if (ArgTy->isStructureOrClassType()) {
        populateHeaderForWrappedAccessors(ArgTy, Offset);
      }
    } else {
      llvm_unreachable("unsupported kernel parameter type");
    }
  }
}

// Removes all "(anonymous namespace)::" substrings from given string
static std::string eraseAnonNamespace(std::string S) {
  const char S1[] = "(anonymous namespace)::";

  for (auto Pos = S.find(S1); Pos != StringRef::npos; Pos = S.find(S1, Pos))
    S.erase(Pos, sizeof(S1) - 1);
  return S;
}

// Creates a mangled kernel name for given kernel name type
static std::string constructKernelName(QualType KernelNameType,
                                       MangleContext &MC) {
  SmallString<256> Result;
  llvm::raw_svector_ostream Out(Result);

  MC.mangleTypeName(KernelNameType, Out);
  return Out.str();
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
  CXXRecordDecl *LE = getKernelObjectType(KernelCallerFunc);
  assert(LE && "invalid kernel caller");

  // Build list of kernel arguments
  llvm::SmallVector<ParamDesc, 16> ParamDescs;
  if (!buildArgTys(getASTContext(), LE, ParamDescs))
    return;

  // Extract name from kernel caller parameters and mangle it.
  const TemplateArgumentList *TemplateArgs =
      KernelCallerFunc->getTemplateSpecializationArgs();
  assert(TemplateArgs && "No template argument info");
  QualType KernelNameType = TypeName::getFullyQualifiedType(
      TemplateArgs->get(0).getAsType(), getASTContext(), true);
  std::string Name = constructKernelName(KernelNameType, MC);

  // TODO Maybe don't emit integration header inside the Sema?
  populateIntHeader(getSyclIntegrationHeader(), Name, KernelNameType, LE);

  FunctionDecl *OpenCLKernel =
      CreateOpenCLKernelDeclaration(getASTContext(), Name, ParamDescs);

  // Let's copy source location of a functor/lambda to emit nicer diagnostics
  OpenCLKernel->setLocation(LE->getLocation());

  // If the source function is implicitly inline, the kernel should be marked
  // such as well. This allows the kernel to be ODR'd if there are multiple uses
  // in different translation units.
  OpenCLKernel->setImplicitlyInline(KernelCallerFunc->isInlined());

  CompoundStmt *OpenCLKernelBody =
      CreateOpenCLKernelBody(*this, KernelCallerFunc, OpenCLKernel);
  OpenCLKernel->setBody(OpenCLKernelBody);
  addSyclDeviceDecl(OpenCLKernel);
}

void Sema::MarkDevice(void) {
  // Let's mark all called functions with SYCL Device attribute.
  // Create the call graph so we can detect recursion and check the validity
  // of new operator overrides. Add the kernel function itself in case
  // it is recursive.
  MarkDeviceFunction Marker(*this);
  Marker.SYCLCG.addToCallGraph(getASTContext().getTranslationUnitDecl());
  for (Decl *D : syclDeviceDecls()) {
    if (auto SYCLKernel = dyn_cast<FunctionDecl>(D)) {
      llvm::SmallPtrSet<FunctionDecl *, 10> VisitedSet;
      Marker.CollectKernelSet(SYCLKernel, SYCLKernel, VisitedSet);

      // Let's propagate attributes from device functions to a SYCL kernels
      llvm::SmallPtrSet<Attr *, 4> Attrs;
      // This function collects all kernel attributes which might be applied to
      // a device functions, but need to be propageted down to callers, i.e.
      // SYCL kernels
      Marker.CollectPossibleKernelAttributes(SYCLKernel, Attrs);
      for (auto *A : Attrs) {
        switch (A->getKind()) {
          case attr::Kind::IntelReqdSubGroupSize: {
            auto *Attr = cast<IntelReqdSubGroupSizeAttr>(A);
            if (auto *Existing =
                    SYCLKernel->getAttr<IntelReqdSubGroupSizeAttr>()) {
              if (Existing->getSubGroupSize() != Attr->getSubGroupSize()) {
                Diag(SYCLKernel->getLocation(),
                     diag::err_conflicting_sycl_kernel_attributes);
                Diag(Existing->getLocation(), diag::note_conflicting_attribute);
                Diag(Attr->getLocation(), diag::note_conflicting_attribute);
                SYCLKernel->setInvalidDecl();
              }
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
            } else {
              SYCLKernel->addAttr(A);
            }
            break;
          }
          case attr::Kind::SYCLIntelKernelArgsRestrict: {
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
    if (FunctionDecl *Def = elt->getDefinition()) {
      if (!Def->hasAttr<SYCLDeviceAttr>()) {
        Def->addAttr(SYCLDeviceAttr::CreateImplicit(Context));
        addSyclDeviceDecl(Def);
      }
      Marker.TraverseStmt(Def->getBody());
    }
  }
}
//
// Do we know that we will eventually codegen the given function?
static bool isKnownEmitted(Sema &S, FunctionDecl *FD) {
  if (!FD)
    return true; // Seen in LIT testing

  if (FD->hasAttr<SYCLDeviceAttr>() ||
      FD->hasAttr<SYCLKernelAttr>())
     return true;

  // Templates are emitted when they're instantiated.
  if (FD->isDependentContext())
    return false;

  // Otherwise, the function is known-emitted if it's in our set of
  // known-emitted functions.
  return S.DeviceKnownEmittedFns.count(FD) > 0;
}

Sema::DeviceDiagBuilder Sema::SYCLDiagIfDeviceCode(SourceLocation Loc,
                                                   unsigned DiagID) {
  assert(getLangOpts().SYCLIsDevice &&
    "Should only be called during SYCL compilation");
  DeviceDiagBuilder::Kind DiagKind = [this] {
    if (isKnownEmitted(*this, dyn_cast<FunctionDecl>(CurContext)))
      return DeviceDiagBuilder::K_ImmediateWithCallStack;
    return DeviceDiagBuilder::K_Deferred;
  }();
  return DeviceDiagBuilder(DiagKind, Loc, DiagID,
                           dyn_cast<FunctionDecl>(CurContext), *this);
}

bool Sema::CheckSYCLCall(SourceLocation Loc, FunctionDecl *Callee) {

  assert(Callee && "Callee may not be null.");
  FunctionDecl *Caller = getCurFunctionDecl();

  // If the caller is known-emitted, mark the callee as known-emitted.
  // Otherwise, mark the call in our call graph so we can traverse it later.
  if (//!isOpenMPDeviceDelayedContext(*this) ||
      (Caller && Caller->hasAttr<SYCLKernelAttr>()) ||
      (Caller && Caller->hasAttr<SYCLDeviceAttr>()) ||
      (Caller && isKnownEmitted(*this, Caller)))
    markKnownEmitted(*this, Caller, Callee, Loc, isKnownEmitted);
  else if (Caller)
    DeviceCallGraph[Caller].insert({Callee, Loc});
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

// Emits a forward declaration
void SYCLIntegrationHeader::emitFwdDecl(raw_ostream &O, const Decl *D) {
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

        if (TD && TD->isCompleteDefinition() && !UnnamedLambdaSupport) {
          // defined class constituting the kernel name is not globally
          // accessible - contradicts the spec
          Diag.Report(D->getSourceRange().getBegin(),
                      diag::err_sycl_kernel_name_class_not_top_level);
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
  std::string S;
  llvm::raw_string_ostream SO(S);
  D->print(SO, P);
  O << SO.str() << ";\n";

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
    raw_ostream &O, QualType T, llvm::SmallPtrSetImpl<const void *> &Printed) {

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
        emitForwardClassDecls(O, Arg.getAsType(), Printed);
        break;
      case TemplateArgument::ArgKind::Pack: {
        ArrayRef<TemplateArgument> Pack = Arg.getPackAsArray();

        for (const auto &T : Pack) {
          if (T.getKind() == TemplateArgument::ArgKind::Type) {
            emitForwardClassDecls(O, T.getAsType(), Printed);
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
        if (Printed.insert(TD).second) {
          emitFwdDecl(O, TD);
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
      emitFwdDecl(O, CTD);
    }
  } else if (Printed.insert(RD).second) {
    // emit forward declarations for "leaf" classes in the template parameter
    // tree;
    emitFwdDecl(O, RD);
  }
}

void SYCLIntegrationHeader::emit(raw_ostream &O) {
  O << "// This is auto-generated SYCL integration header.\n";
  O << "\n";

  O << "#include <CL/sycl/detail/kernel_desc.hpp>\n";

  O << "\n";
  if (!UnnamedLambdaSupport) {
    O << "// Forward declarations of templated kernel function types:\n";

    llvm::SmallPtrSet<const void *, 4> Printed;
    for (const KernelDesc &K : KernelDescs) {
      emitForwardClassDecls(O, K.NameType, Printed);
    }
  }
  O << "\n";

  O << "namespace cl {\n";
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
      O << "template <> struct KernelInfo<"
        << eraseAnonNamespace(K.NameType.getAsString(P)) << "> {\n";
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
  O << "} // namespace cl\n";
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
                                        StringRef KernelStableName) {
  KernelDescs.resize(KernelDescs.size() + 1);
  KernelDescs.back().Name = KernelName;
  KernelDescs.back().NameType = KernelNameType;
  KernelDescs.back().StableName = KernelStableName;
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

SYCLIntegrationHeader::SYCLIntegrationHeader(DiagnosticsEngine &_Diag,
                                             bool _UnnamedLambdaSupport)
    : Diag(_Diag), UnnamedLambdaSupport(_UnnamedLambdaSupport) {}

bool Util::isSyclAccessorType(const QualType &Ty) {
  return isSyclType(Ty, "accessor", true /*Tmpl*/);
}

bool Util::isSyclSamplerType(const QualType &Ty) {
  return isSyclType(Ty, "sampler");
}

bool Util::isSyclStreamType(const QualType &Ty) {
  return isSyclType(Ty, "stream");
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
  const auto *Ctx = dyn_cast<DeclContext>(RecTy);
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
