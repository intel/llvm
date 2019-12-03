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
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Sema.h"

using namespace clang;

using ParamDesc = std::tuple<QualType, IdentifierInfo *, TypeSourceInfo *>;

/// Various utilities.
class Util {
public:
  using DeclContextDesc = std::pair<clang::Decl::Kind, StringRef>;

  /// Checks whether given clang type is a full specialization of the SYCL
  /// accessor class.
  static bool isSyclAccessorType(const QualType &Ty);

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

// Creates body for new OpenCL kernel. This body contains initialization of SYCL
// kernel object fields with kernel parameters and a little bit transformed body
// of the kernel caller function.
static CompoundStmt *CreateOpenCLKernelBody(Sema &S,
                                            FunctionDecl *KernelCallerFunc,
                                            DeclContext *KernelDecl) {
  llvm::SmallVector<Stmt *, 16> BodyStmts;
  CXXRecordDecl *LC = getKernelObjectType(KernelCallerFunc);
  assert(LC && "Kernel object must be available");
  TypeSourceInfo *TSInfo = LC->isLambda() ? LC->getLambdaTypeInfo() : nullptr;

  // Create a local kernel object (lambda or functor) assembled from the
  // incoming formal parameters.
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
      // Creates Expression for special SYCL object accessor.
      // All special SYCL objects must have __init method, here we use it to
      // initialize them. We create call of __init method and pass built kernel
      // arguments as parameters to the __init method.
      auto getExprForSpecialSYCLObj = [&](const QualType &paramTy,
                                          FieldDecl *Field,
                                          const CXXRecordDecl *CRD,
                                          Expr *Base) {
        // All special SYCL objects must have __init method.
        CXXMethodDecl *InitMethod = getInitMethod(CRD);
        assert(InitMethod &&
               "__init method is expected.");
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
        // [kernel_obj].special_obj
        auto SpecialObjME = MemberExpr::Create(
            S.Context, Base, false, SourceLocation(), NestedNameSpecifierLoc(),
            SourceLocation(), Field, FieldDAP,
            DeclarationNameInfo(Field->getDeclName(), SourceLocation()),
            nullptr, Field->getType(), VK_LValue, OK_Ordinary, NOUR_None);

        // [kernel_obj].special_obj.__init
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
        // [kernel_obj].special_obj.__init(_ValueType*,
        // range<int>, range<int>, id<int>)
        CXXMemberCallExpr *Call = CXXMemberCallExpr::Create(
            S.Context, ME, ParamStmts, ResultTy, VK, SourceLocation());
        BodyStmts.push_back(Call);
      };

      // Run through kernel object fields and add initialization for them using
      // built kernel parameters. There are a several possible cases:
      //   - Kernel object field is a SYCL special object (SYCL accessor).
      //     These objects has a special initialization scheme - using
      //     __init method.
      //   - Kernel object field has a scalar type. In this case we should add
      //     simple initialization.
      //   - Kernel object field has a structure or class type. Same handling as
      //     a scalar.
      QualType FieldType = Field->getType();
      CXXRecordDecl *CRD = FieldType->getAsCXXRecordDecl();
      InitializedEntity Entity =
          InitializedEntity::InitializeMember(Field, &VarEntity);
      if (Util::isSyclAccessorType(FieldType)) {
        // Initialize kernel object field with the default constructor and
        // construct a call of __init method.
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

        InitializationKind InitKind =
            InitializationKind::CreateCopy(SourceLocation(), SourceLocation());
        InitializationSequence InitSeq(S, Entity, InitKind, DRE);

        ExprResult MemberInit = InitSeq.Perform(S, Entity, InitKind, DRE);
        InitExprs.push_back(MemberInit.get());

      } else
        llvm_unreachable("Unsupported field type");
      KernelFuncParam++;
    }
    Expr *ILE = new (S.Context)
        InitListExpr(S.Context, SourceLocation(), InitExprs, SourceLocation());
    ILE->setType(QualType(LC->getTypeForDecl(), 0));
    KernelObjClone->setInit(ILE);
  }

  // In the kernel caller function kernel object is a function parameter, so we
  // need to replace all refs to this kernel oject with refs to our clone
  // declared inside the kernel body.
  Stmt *FunctionBody = KernelCallerFunc->getBody();
  ParmVarDecl *KernelObjParam = *(KernelCallerFunc->param_begin());

  // DeclRefExpr with a valid source location but with decl which is not marked
  // as used becomes invalid.
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

// Creates list of kernel parameters descriptors using KernelObj (kernel
// object). Fields of kernel object must be initialized with SYCL kernel
// arguments so in the following function we extract types of kernel object
// fields and add it to the array with kernel parameters descriptors.
static void buildArgTys(ASTContext &Context, CXXRecordDecl *KernelObj,
                        SmallVectorImpl<ParamDesc> &ParamDescs) {
  auto CreateAndAddPrmDsc = [&](const FieldDecl *Fld, const QualType &ArgType) {
    // Create a parameter descriptor and append it to the result
    ParamDescs.push_back(makeParamDesc(Fld, ArgType));
  };

  // Creates a parameter descriptor for SYCL special object - SYCL accessor.
  // All special SYCL objects must have __init method. We extract types for
  // kernel parameters from __init method parameters. We will use __init method
  // and kernel parameters which we build here to initialize special objects in
  // the kernel body.
  auto createSpecialSYCLObjParamDesc = [&](const FieldDecl *Fld,
                                           const QualType &ArgTy) {
    const auto *RecordDecl = ArgTy->getAsCXXRecordDecl();
    assert(RecordDecl && "Special SYCL object must be of a record type");

    CXXMethodDecl *InitMethod = getInitMethod(RecordDecl);
    assert(InitMethod && "__init method is expected.");
    unsigned NumParams = InitMethod->getNumParams();
    for (size_t I = 0; I < NumParams; ++I) {
      ParmVarDecl *PD = InitMethod->getParamDecl(I);
      CreateAndAddPrmDsc(Fld, PD->getType().getCanonicalType());
    }
  };

  // Run through kernel object fields and create corresponding kernel
  // parameters descriptors. There are a several possible cases:
  //   - Kernel object field is a SYCL special object (SYCL accessor).
  //     These objects has a special initialization scheme - using
  //     __init method.
  //   - Kernel object field has a scalar type. In this case we should add
  //     kernel parameter with the same type.
  //   - Kernel object field has a structure or class type. Same handling as a
  //     scalar but we should check if this structure/class contains accessors
  //     and add parameter decriptor for them properly.
  for (const auto *Fld : KernelObj->fields()) {
    QualType ArgTy = Fld->getType();
    if (Util::isSyclAccessorType(ArgTy))
      createSpecialSYCLObjParamDesc(Fld, ArgTy);
    else if (ArgTy->isStructureOrClassType())
      CreateAndAddPrmDsc(Fld, ArgTy);
    else if (ArgTy->isScalarType())
      CreateAndAddPrmDsc(Fld, ArgTy);
    else
      llvm_unreachable("Unsupported kernel parameter type");
  }
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
void Sema::constructOpenCLKernel(FunctionDecl *KernelCallerFunc,
                                 MangleContext &MC) {
  CXXRecordDecl *LE = getKernelObjectType(KernelCallerFunc);
  assert(LE && "invalid kernel caller");

  // Build list of kernel arguments.
  llvm::SmallVector<ParamDesc, 16> ParamDescs;
  buildArgTys(getASTContext(), LE, ParamDescs);

  // Extract name from kernel caller parameters and mangle it.
  const TemplateArgumentList *TemplateArgs =
      KernelCallerFunc->getTemplateSpecializationArgs();
  assert(TemplateArgs && "No template argument info");
  QualType KernelNameType = TypeName::getFullyQualifiedType(
      TemplateArgs->get(0).getAsType(), getASTContext(), true);
  std::string Name = constructKernelName(KernelNameType, MC);

  FunctionDecl *OpenCLKernel =
      CreateOpenCLKernelDeclaration(getASTContext(), Name, ParamDescs);

  // Let's copy source location of a functor/lambda to emit nicer diagnostics.
  OpenCLKernel->setLocation(LE->getLocation());

  CompoundStmt *OpenCLKernelBody =
      CreateOpenCLKernelBody(*this, KernelCallerFunc, OpenCLKernel);
  OpenCLKernel->setBody(OpenCLKernelBody);

  addSYCLKernel(OpenCLKernel);
}

// -----------------------------------------------------------------------------
// Utility class methods
// -----------------------------------------------------------------------------

bool Util::isSyclAccessorType(const QualType &Ty) {
  static std::array<DeclContextDesc, 3> Scopes = {
      Util::DeclContextDesc{clang::Decl::Kind::Namespace, "cl"},
      Util::DeclContextDesc{clang::Decl::Kind::Namespace, "sycl"},
      Util::DeclContextDesc{clang::Decl::Kind::ClassTemplateSpecialization,
                            "accessor"}};
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

