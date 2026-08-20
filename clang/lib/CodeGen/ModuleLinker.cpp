//===--- ModuleLinker.cpp - Shared bitcode link helpers -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/ModuleLinker.h"

#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Module.h"

using namespace clang;

bool clang::loadLinkModules(CompilerInstance &CI, llvm::LLVMContext &Ctx,
                            llvm::SmallVectorImpl<LinkModule> &LinkModules) {
  if (!LinkModules.empty())
    return false;

  for (const CodeGenOptions::BitcodeFileToLink &F :
       CI.getCodeGenOpts().LinkBitcodeFiles) {
    auto BCBuf = CI.getFileManager().getBufferForFile(F.Filename);
    if (!BCBuf) {
      CI.getDiagnostics().Report(diag::err_cannot_open_file)
          << F.Filename << BCBuf.getError().message();
      LinkModules.clear();
      return true;
    }

    llvm::Expected<std::unique_ptr<llvm::Module>> ModuleOrErr =
        llvm::getOwningLazyBitcodeModule(std::move(*BCBuf), Ctx);
    if (!ModuleOrErr) {
      llvm::handleAllErrors(
          ModuleOrErr.takeError(), [&](llvm::ErrorInfoBase &EIB) {
            CI.getDiagnostics().Report(diag::err_cannot_open_file)
                << F.Filename << EIB.message();
          });
      LinkModules.clear();
      return true;
    }

    LinkModules.push_back({std::move(ModuleOrErr.get()), F.PropagateAttrs,
                           F.Internalize, F.LinkFlags});
  }

  // For SYCL no-RDC, link the per-TU device wrapper bitcode produced by
  // clang-linker-wrapper --sycl-device-link into the host module so the SYCL
  // runtime finds the device image at program startup.
  if (CI.getLangOpts().SYCLIsHost &&
      !CI.getCodeGenOpts().OffloadBinaryToEmbedFile.empty()) {
    auto BCBuf = CI.getFileManager().getBufferForFile(
        CI.getCodeGenOpts().OffloadBinaryToEmbedFile);
    if (!BCBuf) {
      CI.getDiagnostics().Report(diag::err_cannot_open_file)
          << CI.getCodeGenOpts().OffloadBinaryToEmbedFile
          << BCBuf.getError().message();
      return true;
    }
    llvm::Expected<std::unique_ptr<llvm::Module>> MOrErr =
        llvm::parseBitcodeFile((*BCBuf)->getMemBufferRef(), Ctx);
    if (!MOrErr) {
      CI.getDiagnostics().Report(diag::err_fe_linking_module)
          << CI.getCodeGenOpts().OffloadBinaryToEmbedFile
          << llvm::toString(MOrErr.takeError());
      return true;
    }
    LinkModules.push_back({std::move(*MOrErr),
                           /*PropagateAttrs=*/false,
                           /*Internalize=*/false,
                           /*LinkFlags=*/0});
  }

  return false;
}
