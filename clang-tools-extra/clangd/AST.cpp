//===--- AST.cpp - Utility AST functions  -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"

#include "SourceCode.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {

namespace {
llvm::Optional<llvm::ArrayRef<TemplateArgumentLoc>>
getTemplateSpecializationArgLocs(const NamedDecl &ND) {
  if (auto *Func = llvm::dyn_cast<FunctionDecl>(&ND)) {
    if (const ASTTemplateArgumentListInfo *Args =
            Func->getTemplateSpecializationArgsAsWritten())
      return Args->arguments();
  } else if (auto *Cls =
                 llvm::dyn_cast<ClassTemplatePartialSpecializationDecl>(&ND)) {
    if (auto *Args = Cls->getTemplateArgsAsWritten())
      return Args->arguments();
  } else if (auto *Var =
                 llvm::dyn_cast<VarTemplatePartialSpecializationDecl>(&ND)) {
    if (auto *Args = Var->getTemplateArgsAsWritten())
      return Args->arguments();
  } else if (auto *Var = llvm::dyn_cast<VarTemplateSpecializationDecl>(&ND))
    return Var->getTemplateArgsInfo().arguments();
  // We return None for ClassTemplateSpecializationDecls because it does not
  // contain TemplateArgumentLoc information.
  return llvm::None;
}

template <class T>
bool isTemplateSpecializationKind(const NamedDecl *D,
                                  TemplateSpecializationKind Kind) {
  if (const auto *TD = dyn_cast<T>(D))
    return TD->getTemplateSpecializationKind() == Kind;
  return false;
}

bool isTemplateSpecializationKind(const NamedDecl *D,
                                  TemplateSpecializationKind Kind) {
  return isTemplateSpecializationKind<FunctionDecl>(D, Kind) ||
         isTemplateSpecializationKind<CXXRecordDecl>(D, Kind) ||
         isTemplateSpecializationKind<VarDecl>(D, Kind);
}

} // namespace

bool isImplicitTemplateInstantiation(const NamedDecl *D) {
  return isTemplateSpecializationKind(D, TSK_ImplicitInstantiation);
}

bool isExplicitTemplateSpecialization(const NamedDecl *D) {
  return isTemplateSpecializationKind(D, TSK_ExplicitSpecialization);
}

bool isImplementationDetail(const Decl *D) {
  return !isSpelledInSource(D->getLocation(),
                            D->getASTContext().getSourceManager());
}

SourceLocation findName(const clang::Decl *D) {
  return D->getLocation();
}

std::string printQualifiedName(const NamedDecl &ND) {
  std::string QName;
  llvm::raw_string_ostream OS(QName);
  PrintingPolicy Policy(ND.getASTContext().getLangOpts());
  // Note that inline namespaces are treated as transparent scopes. This
  // reflects the way they're most commonly used for lookup. Ideally we'd
  // include them, but at query time it's hard to find all the inline
  // namespaces to query: the preamble doesn't have a dedicated list.
  Policy.SuppressUnwrittenScope = true;
  ND.printQualifiedName(OS, Policy);
  OS.flush();
  assert(!StringRef(QName).startswith("::"));
  return QName;
}

static bool isAnonymous(const DeclarationName &N) {
  return N.isIdentifier() && !N.getAsIdentifierInfo();
}

NestedNameSpecifierLoc getQualifierLoc(const NamedDecl &ND) {
  if (auto *V = llvm::dyn_cast<DeclaratorDecl>(&ND))
    return V->getQualifierLoc();
  if (auto *T = llvm::dyn_cast<TagDecl>(&ND))
    return T->getQualifierLoc();
  return NestedNameSpecifierLoc();
}

std::string printUsingNamespaceName(const ASTContext &Ctx,
                                    const UsingDirectiveDecl &D) {
  PrintingPolicy PP(Ctx.getLangOpts());
  std::string Name;
  llvm::raw_string_ostream Out(Name);

  if (auto *Qual = D.getQualifier())
    Qual->print(Out, PP);
  D.getNominatedNamespaceAsWritten()->printName(Out);
  return Out.str();
}

std::string printName(const ASTContext &Ctx, const NamedDecl &ND) {
  std::string Name;
  llvm::raw_string_ostream Out(Name);
  PrintingPolicy PP(Ctx.getLangOpts());

  // Handle 'using namespace'. They all have the same name - <using-directive>.
  if (auto *UD = llvm::dyn_cast<UsingDirectiveDecl>(&ND)) {
    Out << "using namespace ";
    if (auto *Qual = UD->getQualifier())
      Qual->print(Out, PP);
    UD->getNominatedNamespaceAsWritten()->printName(Out);
    return Out.str();
  }

  if (isAnonymous(ND.getDeclName())) {
    // Come up with a presentation for an anonymous entity.
    if (isa<NamespaceDecl>(ND))
      return "(anonymous namespace)";
    if (auto *Cls = llvm::dyn_cast<RecordDecl>(&ND))
      return ("(anonymous " + Cls->getKindName() + ")").str();
    if (isa<EnumDecl>(ND))
      return "(anonymous enum)";
    return "(anonymous)";
  }

  // Print nested name qualifier if it was written in the source code.
  if (auto *Qualifier = getQualifierLoc(ND).getNestedNameSpecifier())
    Qualifier->print(Out, PP);
  // Print the name itself.
  ND.getDeclName().print(Out, PP);
  // Print template arguments.
  Out << printTemplateSpecializationArgs(ND);

  return Out.str();
}

std::string printTemplateSpecializationArgs(const NamedDecl &ND) {
  std::string TemplateArgs;
  llvm::raw_string_ostream OS(TemplateArgs);
  PrintingPolicy Policy(ND.getASTContext().getLangOpts());
  if (llvm::Optional<llvm::ArrayRef<TemplateArgumentLoc>> Args =
          getTemplateSpecializationArgLocs(ND)) {
    printTemplateArgumentList(OS, *Args, Policy);
  } else if (auto *Cls = llvm::dyn_cast<ClassTemplateSpecializationDecl>(&ND)) {
    if (const TypeSourceInfo *TSI = Cls->getTypeAsWritten()) {
      // ClassTemplateSpecializationDecls do not contain
      // TemplateArgumentTypeLocs, they only have TemplateArgumentTypes. So we
      // create a new argument location list from TypeSourceInfo.
      auto STL = TSI->getTypeLoc().getAs<TemplateSpecializationTypeLoc>();
      llvm::SmallVector<TemplateArgumentLoc, 8> ArgLocs;
      ArgLocs.reserve(STL.getNumArgs());
      for (unsigned I = 0; I < STL.getNumArgs(); ++I)
        ArgLocs.push_back(STL.getArgLoc(I));
      printTemplateArgumentList(OS, ArgLocs, Policy);
    } else {
      // FIXME: Fix cases when getTypeAsWritten returns null inside clang AST,
      // e.g. friend decls. Currently we fallback to Template Arguments without
      // location information.
      printTemplateArgumentList(OS, Cls->getTemplateArgs().asArray(), Policy);
    }
  }
  OS.flush();
  return TemplateArgs;
}

std::string printNamespaceScope(const DeclContext &DC) {
  for (const auto *Ctx = &DC; Ctx != nullptr; Ctx = Ctx->getParent())
    if (const auto *NS = dyn_cast<NamespaceDecl>(Ctx))
      if (!NS->isAnonymousNamespace() && !NS->isInlineNamespace())
        return printQualifiedName(*NS) + "::";
  return "";
}

llvm::Optional<SymbolID> getSymbolID(const Decl *D) {
  llvm::SmallString<128> USR;
  if (index::generateUSRForDecl(D, USR))
    return None;
  return SymbolID(USR);
}

llvm::Optional<SymbolID> getSymbolID(const IdentifierInfo &II,
                                     const MacroInfo *MI,
                                     const SourceManager &SM) {
  if (MI == nullptr)
    return None;
  llvm::SmallString<128> USR;
  if (index::generateUSRForMacro(II.getName(), MI->getDefinitionLoc(), SM, USR))
    return None;
  return SymbolID(USR);
}

std::string shortenNamespace(const llvm::StringRef OriginalName,
                             const llvm::StringRef CurrentNamespace) {
  llvm::SmallVector<llvm::StringRef, 8> OriginalParts;
  llvm::SmallVector<llvm::StringRef, 8> CurrentParts;
  llvm::SmallVector<llvm::StringRef, 8> Result;
  OriginalName.split(OriginalParts, "::");
  CurrentNamespace.split(CurrentParts, "::");
  auto MinLength = std::min(CurrentParts.size(), OriginalParts.size());

  unsigned DifferentAt = 0;
  while (DifferentAt < MinLength &&
      CurrentParts[DifferentAt] == OriginalParts[DifferentAt]) {
    DifferentAt++;
  }

  for (unsigned i = DifferentAt; i < OriginalParts.size(); ++i) {
    Result.push_back(OriginalParts[i]);
  }
  return join(Result, "::");
}

std::string printType(const QualType QT, const DeclContext & Context){
  PrintingPolicy PP(Context.getParentASTContext().getPrintingPolicy());
  PP.SuppressUnwrittenScope = 1;
  PP.SuppressTagKeyword = 1;
  return shortenNamespace(
      QT.getAsString(PP),
      printNamespaceScope(Context) );
}


} // namespace clangd
} // namespace clang
