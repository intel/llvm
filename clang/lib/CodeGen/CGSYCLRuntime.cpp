//===----- CGSYCLRuntime.cpp - Interface to SYCL Runtimes -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides custom clang code generation for SYCL.
//
//===----------------------------------------------------------------------===//

#include "CGSYCLRuntime.h"
#include "CodeGenFunction.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Frontend/Offloading/OffloadWrapper.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <assert.h>

using namespace clang;
using namespace CodeGen;

namespace {

/// Various utilities.
/// TODO partially duplicates functionality from SemaSYCL.cpp, can be shared.
class Util {
public:
  using DeclContextDesc = std::pair<clang::Decl::Kind, StringRef>;

  /// Checks whether given clang type is declared in the given hierarchy of
  /// declaration contexts.
  /// \param RecTy      the clang type being checked
  /// \param Scopes     the declaration scopes leading from the type to the
  ///     translation unit (excluding the latter)
  static bool matchQualifiedTypeName(const CXXRecordDecl *RecTy,
                                     ArrayRef<Util::DeclContextDesc> Scopes);
};

static bool isPFWI(const FunctionDecl &FD) {
  const auto *MD = dyn_cast<CXXMethodDecl>(&FD);
  if (!MD)
    return false;
  static std::array<Util::DeclContextDesc, 3> Scopes = {
      Util::DeclContextDesc{clang::Decl::Kind::Namespace, "sycl"},
      Util::DeclContextDesc{clang::Decl::Kind::Namespace, "_V1"},
      Util::DeclContextDesc{Decl::Kind::ClassTemplateSpecialization, "group"}};
  if (!Util::matchQualifiedTypeName(MD->getParent(), Scopes))
    return false;
  return FD.getName() == "parallel_for_work_item";
}

constexpr char WG_SCOPE_MD_ID[] = "work_group_scope";
constexpr char WI_SCOPE_MD_ID[] = "work_item_scope";
constexpr char PFWI_MD_ID[] = "parallel_for_work_item";
constexpr char ATTR_GENX_VOLATILE[] = "genx_volatile";
constexpr char ATTR_GENX_BYTE_OFFSET[] = "genx_byte_offset";

} // anonymous namespace

bool CGSYCLRuntime::actOnFunctionStart(const FunctionDecl &FD,
                                       llvm::Function &F) {
  // Populate "sycl_explicit_simd" attribute if any.
  if (FD.hasAttr<SYCLSimdAttr>())
    F.setMetadata("sycl_explicit_simd", llvm::MDNode::get(F.getContext(), {}));

  // Set the function attribute expected by the vector backend compiler.
  if (const auto *A = FD.getAttr<SYCLIntelESimdVectorizeAttr>())
    if (const auto *DeclExpr = cast<ConstantExpr>(A->getValue())) {
      SmallString<2> Str;
      DeclExpr->getResultAsAPSInt().toString(Str);
      F.addFnAttr("CMGenxSIMT", Str);
    }

  SYCLScopeAttr *Scope = FD.getAttr<SYCLScopeAttr>();
  if (!Scope)
    return false;
  switch (Scope->getLevel()) {
  case SYCLScopeAttr::Level::WorkGroup:
    F.setMetadata(WG_SCOPE_MD_ID, llvm::MDNode::get(F.getContext(), {}));
    return true;
  case SYCLScopeAttr::Level::WorkItem:
    F.setMetadata(WI_SCOPE_MD_ID, llvm::MDNode::get(F.getContext(), {}));
    if (isPFWI(FD))
      // also emit specific marker for parallel_for_work_item, as it needs to
      // be handled specially in the SYCL lowering pass
      F.setMetadata(PFWI_MD_ID, llvm::MDNode::get(F.getContext(), {}));
    return true;
  }
  llvm_unreachable("unknown sycl scope");
}

void CGSYCLRuntime::emitWorkGroupLocalVarDecl(CodeGenFunction &CGF,
                                              const VarDecl &D) {
#ifndef NDEBUG
  SYCLScopeAttr *Scope = D.getAttr<SYCLScopeAttr>();
  if (!Scope)
    if (auto *RD = D.getType()->getAsCXXRecordDecl())
      Scope = RD->getAttr<SYCLScopeAttr>();
  assert((Scope && Scope->isWorkGroup()) && "work group scope expected");
#endif // NDEBUG
  // generate global variable in the address space selected by the clang CodeGen
  // (should be local)
  CGF.EmitStaticVarDecl(D, llvm::GlobalValue::InternalLinkage);
}

bool CGSYCLRuntime::actOnAutoVarEmit(CodeGenFunction &CGF, const VarDecl &D,
                                     llvm::Value *Addr) {
  SYCLScopeAttr *Scope = D.getAttr<SYCLScopeAttr>();
  if (!Scope)
    return false;
  assert(Scope->isWorkItem() && "auto var must be of work item scope");
  auto *AI = dyn_cast<llvm::AllocaInst>(Addr);
  assert(AI && "AllocaInst expected as local var address");
  AI->setMetadata(WI_SCOPE_MD_ID, llvm::MDNode::get(AI->getContext(), {}));
  return true;
}

bool CGSYCLRuntime::actOnGlobalVarEmit(CodeGenModule &CGM, const VarDecl &D,
                                       llvm::Value *Addr) {
  SYCLRegisterNumAttr *RegAttr = D.getAttr<SYCLRegisterNumAttr>();
  if (!RegAttr)
    return false;
  auto *GlobVar = cast<llvm::GlobalVariable>(Addr);
  GlobVar->addAttribute(ATTR_GENX_VOLATILE);
  GlobVar->addAttribute(ATTR_GENX_BYTE_OFFSET,
                        Twine(RegAttr->getNumber()).str());
  // TODO consider reversing the error/success return values
  return true;
}

void clang::CodeGen::embedSYCLNoRDCBinary(CodeGenModule &CGM) {
  if (!CGM.getLangOpts().SYCLIsHost ||
      CGM.getCodeGenOpts().SYCLTargetBinaryFileName.empty())
    return;
  auto BinaryOrErr = CGM.getFileSystem()->getBufferForFile(
      CGM.getCodeGenOpts().SYCLTargetBinaryFileName, /*MaxSize=*/-1,
      /*RequiresNullTerminator=*/false);
  if (std::error_code EC = BinaryOrErr.getError()) {
    CGM.getDiags().Report(diag::err_cannot_open_file)
        << CGM.getCodeGenOpts().SYCLTargetBinaryFileName << EC.message();
    return;
  }
  // The input is a wrapper bitcode module produced by clang-linker-wrapper
  // --sycl-device-link. It already contains the correct __sycl.tgt_bin_desc
  // structure and __sycl_register_lib constructor. Link it directly into the
  // host module so the SYCL runtime finds the device image at program startup.
  llvm::LLVMContext &Ctx = CGM.getModule().getContext();
  llvm::Expected<std::unique_ptr<llvm::Module>> DevModOrErr =
      llvm::parseBitcodeFile((*BinaryOrErr)->getMemBufferRef(), Ctx);
  if (!DevModOrErr) {
    CGM.getDiags().Report(diag::err_fe_linking_module)
        << CGM.getCodeGenOpts().SYCLTargetBinaryFileName
        << llvm::toString(DevModOrErr.takeError());
    return;
  }
  // Propagate the host data layout so the linker doesn't warn about mismatched
  // layouts (the wrapper .bc is created without a data layout set).
  (*DevModOrErr)->setDataLayout(CGM.getModule().getDataLayout());
  // Install a diagnostic handler that routes DK_Linker diagnostics through
  // clang's diagnostic engine. The default handler asserts CurLinkModule is
  // set, which is only true inside BackendConsumer's LinkModules loop.
  Ctx.setDiagnosticHandler(std::make_unique<llvm::DiagnosticHandler>());
  if (llvm::Linker::linkModules(CGM.getModule(), std::move(*DevModOrErr)))
    CGM.getDiags().Report(diag::err_fe_linking_module)
        << CGM.getCodeGenOpts().SYCLTargetBinaryFileName
        << "linking wrapper bitcode into host module failed";
  Ctx.setDiagnosticHandler(nullptr);
}

bool Util::matchQualifiedTypeName(const CXXRecordDecl *RecTy,
                                  ArrayRef<Util::DeclContextDesc> Scopes) {
  // The idea: check the declaration context chain starting from the type
  // itself. At each step check the context is of expected kind
  // (namespace) and name.
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
