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
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Frontend/Offloading/SYCLOffloadWrapper.h"
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

  // For SYCL no-RDC, wrap the device images of the fat binary produced by
  // clang-linker-wrapper --emit-fatbin-only into a module that registers them
  // with the SYCL runtime, and link it into the host module. Other binaries
  // are embedded by CodeGenModule::embedSYCLDeviceBinary instead.
  StringRef FatbinFile = CI.getCodeGenOpts().OffloadBinaryToEmbedFile;
  if (CI.getLangOpts().SYCLIsHost && !CI.getLangOpts().CUDA &&
      !FatbinFile.empty()) {
    auto FatbinBuf = CI.getFileManager().getBufferForFile(FatbinFile);
    if (!FatbinBuf) {
      CI.getDiagnostics().Report(diag::err_cannot_open_file)
          << FatbinFile << FatbinBuf.getError().message();
      LinkModules.clear();
      return true;
    }
    if (llvm::identify_magic((*FatbinBuf)->getBuffer()) !=
        llvm::file_magic::offload_binary)
      return false;

    auto ReportError = [&](llvm::Error Err) {
      llvm::handleAllErrors(std::move(Err), [&](llvm::ErrorInfoBase &EIB) {
        CI.getDiagnostics().Report(diag::err_fe_linking_module)
            << FatbinFile << EIB.message();
      });
      LinkModules.clear();
      return true;
    };
    auto ImagesOrErr =
        llvm::offloading::readSYCLImages((*FatbinBuf)->getMemBufferRef());
    if (!ImagesOrErr)
      return ReportError(ImagesOrErr.takeError());

    auto WrapperM =
        std::make_unique<llvm::Module>("offload.wrapper.object", Ctx);
    WrapperM->setTargetTriple(CI.getTarget().getTriple());
    WrapperM->setDataLayout(CI.getTarget().getDataLayoutString());
    if (llvm::Error Err =
            llvm::offloading::wrapSYCLBinaries(*WrapperM, *ImagesOrErr))
      return ReportError(std::move(Err));
    LinkModules.push_back({std::move(WrapperM),
                           /*PropagateAttrs=*/false,
                           /*Internalize=*/false,
                           /*LinkFlags=*/0});
  }

  return false;
}
