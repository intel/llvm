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
#include "clang/CodeGenUtils/BackendDiagnosticHandler.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Frontend/Offloading/SYCLOffloadWrapper.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Transforms/IPO/Internalize.h"

using namespace clang;

bool clang::loadLinkModules(CompilerInstance &CI, llvm::LLVMContext &Ctx,
                            llvm::SmallVectorImpl<LinkModule> &LinkModules,
                            bool OnlyInternalized) {
  if (!LinkModules.empty())
    return false;

  for (const CodeGenOptions::BitcodeFileToLink &F :
       CI.getCodeGenOpts().LinkBitcodeFiles) {
    if (OnlyInternalized && !F.Internalize)
      continue;
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

bool clang::linkInModules(llvm::Module &M,
                          llvm::SmallVectorImpl<LinkModule> &LinkModules,
                          const CodeGenOptions &CodeGenOpts,
                          const LangOptions &LangOpts,
                          const TargetOptions &TargetOpts,
                          BackendDiagnosticConsumer &DiagConsumer) {
  for (LinkModule &LM : LinkModules) {
    assert(LM.Module && "LinkModule does not actually have a module");

    if (LM.PropagateAttrs)
      for (llvm::Function &F : *LM.Module) {
        // Skip intrinsics. Keep consistent with how intrinsics are created
        // in LLVM IR.
        if (F.isIntrinsic())
          continue;
        CodeGen::mergeDefaultFunctionDefinitionAttributes(
            F, CodeGenOpts, LangOpts, TargetOpts, LM.Internalize);
      }

    DiagConsumer.setCurLinkModule(LM.Module.get());
    bool Err;

    if (LM.Internalize) {
      Err = llvm::Linker::linkModules(
          M, std::move(LM.Module), LM.LinkFlags,
          [](llvm::Module &M, const llvm::StringSet<> &GVS) {
            llvm::internalizeModule(M, [&GVS](const llvm::GlobalValue &GV) {
              return !GV.hasName() || (GVS.count(GV.getName()) == 0);
            });
          });
    } else
      Err = llvm::Linker::linkModules(M, std::move(LM.Module), LM.LinkFlags);

    if (Err)
      return true;
  }

  LinkModules.clear();
  return false; // success
}
