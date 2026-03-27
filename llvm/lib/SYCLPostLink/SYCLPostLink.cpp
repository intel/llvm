//=-------- SYCLPostLink.cpp - post-link processing for SYCL offloading -----=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "llvm/SYCLPostLink/SYCLPostLink.h"
#include "llvm/SYCLPostLink/ModuleSplitter.h"
#include "llvm/SYCLPostLink/SpecializationConstants.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/SYCLPostLink/ESIMDPostSplitProcessing.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LineIterator.h"

using namespace llvm;

namespace {

Error saveModuleIRInFile(Module &M, StringRef FilePath, bool OutputAssembly) {
  int FD = -1;
  if (std::error_code EC = sys::fs::openFileForWrite(FilePath, FD))
    return errorCodeToError(EC);

  raw_fd_ostream OS(FD, true);
  ModulePassManager MPM;
  ModuleAnalysisManager MAM;
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  if (OutputAssembly)
    MPM.addPass(PrintModulePass(OS));
  else
    MPM.addPass(BitcodeWriterPass(OS));

  MPM.run(M, MAM);
  return Error::success();
}

} // anonymous namespace

Expected<module_split::SplitModule>
llvm::sycl_post_link::saveModuleDesc(module_split::ModuleDesc &MD,
                                     std::string Prefix, bool OutputAssembly) {
  module_split::SplitModule SM;
  Prefix += OutputAssembly ? ".ll" : ".bc";
  MD.saveSplitInformationAsMetadata();
  Error E = saveModuleIRInFile(MD.getModule(), Prefix, OutputAssembly);
  if (E)
    return E;

  SM.ModuleFilePath = Prefix;
  SM.Symbols = MD.makeSymbolTable();
  return SM;
}

Expected<std::vector<module_split::SplitModule>>
llvm::sycl_post_link::parseSplitModulesFromFile(StringRef File) {
  auto EntriesMBOrErr = llvm::MemoryBuffer::getFile(File);

  if (!EntriesMBOrErr)
    return createFileError(File, EntriesMBOrErr.getError());

  line_iterator LI(**EntriesMBOrErr);
  if (LI.is_at_eof() || *LI != "[Code|Properties|Symbols]")
    return createStringError(inconvertibleErrorCode(),
                             "invalid SYCL Table file.");

  ++LI;
  std::vector<module_split::SplitModule> Modules;
  while (!LI.is_at_eof()) {
    StringRef Line = *LI;
    if (Line.empty())
      return createStringError(inconvertibleErrorCode(),
                               "invalid SYCL table row.");

    SmallVector<StringRef, 3> Parts;
    Line.split(Parts, "|");
    if (Parts.size() != 3)
      return createStringError(inconvertibleErrorCode(),
                               "invalid SYCL Table row.");

    auto [IRFilePath, PropertyFilePath, SymbolsFilePath] =
        std::tie(Parts[0], Parts[1], Parts[2]);
    if (PropertyFilePath.empty() || SymbolsFilePath.empty())
      return createStringError(inconvertibleErrorCode(),
                               "invalid SYCL Table row.");

    auto MBOrErr = MemoryBuffer::getFile(PropertyFilePath);
    if (!MBOrErr)
      return createFileError(PropertyFilePath, MBOrErr.getError());

    auto &MB = **MBOrErr;
    auto PropSetOrErr = llvm::util::PropertySetRegistry::read(&MB);
    if (!PropSetOrErr)
      return PropSetOrErr.takeError();

    llvm::util::PropertySetRegistry Properties = std::move(**PropSetOrErr);
    MBOrErr = MemoryBuffer::getFile(SymbolsFilePath);
    if (!MBOrErr)
      return createFileError(SymbolsFilePath, MBOrErr.getError());

    auto &MB2 = *MBOrErr;
    std::string Symbols =
        std::string(MB2->getBufferStart(), MB2->getBufferEnd());
    Modules.emplace_back(IRFilePath, std::move(Properties), std::move(Symbols));
    ++LI;
  }

  return Modules;
}

std::string llvm::sycl_post_link::convertSettingsToString(
    const llvm::sycl_post_link::PostLinkSettings &Settings) {
  StringRef SpecConstMode =
      Settings.SpecConstMode ? SpecConstantsPass::convertHandlingModeToString(
                                   *Settings.SpecConstMode)
                             : "";

  return formatv(
      "OutputAssembly: {0}, SplitMode: {1}, SpecializationConstantMode: "
      "{2}, GenerateModuleWithDefaultSpecConstValues: {3}, "
      "EmitOnlyKernelsAsEntryPoints: {4}, EmitParamInfo: {5}, "
      "EmitProgramMetadata: {6}, EmitKernelNames: {7}, "
      "EmitExportedSymbols: {8}, EmitImportedSymbols: {9}, "
      "{10}",
      Settings.OutputAssembly,
      module_split::convertSplitModeToString(Settings.SplitMode), SpecConstMode,
      Settings.GenerateModuleDescWithDefaultSpecConsts,
      Settings.EmitOnlyKernelsAsEntryPoints, Settings.EmitParamInfo,
      Settings.EmitProgramMetadata, Settings.EmitKernelNames,
      Settings.EmitExportedSymbols, Settings.EmitImportedSymbols,
      sycl_post_link::convertESIMDOptionsToString(Settings.ESIMDOptions));
}

Expected<std::vector<module_split::SplitModule>>
llvm::sycl_post_link::performPostLinkProcessing(
    std::unique_ptr<Module> M,
    llvm::sycl_post_link::PostLinkSettings Settings) {
  std::vector<module_split::SplitModule> SplitModules;
  auto PostSplitCallback =
      [&SplitModules,
       Settings](std::unique_ptr<module_split::ModuleDesc> M) -> Error {
    M->fixupLinkageOfDirectInvokeSimdTargets();

    bool Modified = false;
    bool SplitOccurred = false;
    auto ModulesOrErr = sycl_post_link::handleESIMD(
        std::move(M), Settings.ESIMDOptions, Modified, SplitOccurred);
    if (!ModulesOrErr)
      return ModulesOrErr.takeError();

    SmallVector<std::unique_ptr<module_split::ModuleDesc>, 2> &Modules =
        *ModulesOrErr;
    assert(Modules.size() &&
           "at least one module is expected after ESIMD split");

    SmallVector<std::unique_ptr<module_split::ModuleDesc>> NewModules;
    if (Settings.SpecConstMode)
      llvm::sycl_post_link::handleSpecializationConstants(
          Modules, *Settings.SpecConstMode, NewModules,
          Settings.GenerateModuleDescWithDefaultSpecConsts);

    for (std::unique_ptr<module_split::ModuleDesc> &MD : NewModules)
      Modules.push_back(std::move(MD));

    for (std::unique_ptr<module_split::ModuleDesc> &MD : Modules) {
      size_t ID = SplitModules.size();
      StringRef Suffix = MD->isESIMD() ? "_esimd" : "";
      std::string OutIRFilename =
          (Settings.OutputPrefix + Suffix + "_" + Twine(ID)).str();
      Expected<module_split::SplitModule> SplitImageOrErr =
          saveModuleDesc(*MD, OutIRFilename, Settings.OutputAssembly);
      if (!SplitImageOrErr)
        return SplitImageOrErr.takeError();

      MD.release();
      SplitModules.push_back(std::move(*SplitImageOrErr));
    }

    return Error::success();
  };

  module_split::ModuleSplitterSettings SplitSettings;
  SplitSettings.Mode = Settings.SplitMode;
  if (Error E = module_split::splitSYCLModule(std::move(M), SplitSettings,
                                              PostSplitCallback))
    return createStringError(
        formatv("sycl post link processing failed. {0}", E));

  return SplitModules;
}
