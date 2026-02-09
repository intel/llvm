#include "llvm/SYCLPostLink/SYCLPostLink.h"
#include "llvm/SYCLPostLink/SpecializationConstants.h"
#include "llvm/SYCLPostLink/ModuleSplitter.h"

#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"

using namespace llvm;

namespace {

Error saveModuleIRInFile(Module &M, StringRef FilePath,
                         bool OutputAssembly) {
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

Expected<std::vector<module_split::SplitModule>> llvm::sycl_post_link::parseSplitModulesFromFile(StringRef File) {
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

Expected<module_split::SplitModule> llvm::sycl_post_link::saveModuleDesc(module_split::ModuleDesc &MD, std::string Prefix,
                                     bool OutputAssembly) {
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

Expected<std::vector<module_split::SplitModule>> llvm::sycl_post_link::PostLinkProcessing(std::unique_ptr<Module> M, llvm::sycl_post_link::PostLinkSettings Settings) {
  std::vector<module_split::SplitModule> SplitModules;
  auto PostSplitCallback =
      [&SplitModules, Settings](
          std::unique_ptr<module_split::ModuleDesc> M) -> Error {
    M->fixupLinkageOfDirectInvokeSimdTargets(); // TODO Test this line;
    SmallVector<std::unique_ptr<module_split::ModuleDesc>> Modules;
    Modules.push_back(std::move(M));
    SmallVector<std::unique_ptr<module_split::ModuleDesc>> NewModules;
    if (Settings.SpecConstMode)
      llvm::handleSpecializationConstants(
          Modules, *Settings.SpecConstMode, NewModules,
          Settings.GenerateModuleDescWithDefaultSpecConsts);

    for (std::unique_ptr<module_split::ModuleDesc> &MD : NewModules)
      Modules.push_back(std::move(MD));

    for (std::unique_ptr<module_split::ModuleDesc> &MD : Modules) {
      size_t ID = SplitModules.size();
      std::string OutIRFilename = (Settings.OutputPrefix + "_" + Twine(ID)).str();
      Expected<module_split::SplitModule> SplitImageOrErr = saveModuleDesc(*MD, OutIRFilename, Settings.OutputAssembly);
      if (!SplitImageOrErr)
        return SplitImageOrErr.takeError();

      MD.release();
      SplitModules.push_back(std::move(*SplitImageOrErr));
    }

    return Error::success();
  };

  module_split::ModuleSplitterSettings SplitSettings;
  SplitSettings.Mode = Settings.SplitMode;
  if (Error E = module_split::splitSYCLModule(std::move(M), SplitSettings, PostSplitCallback); !E)
    return createStringError(formatv("sycl post link processing failed. {0}", E));

  return SplitModules;
}
