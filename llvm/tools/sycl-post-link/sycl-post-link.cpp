//===- sycl-post-link.cpp - SYCL post-link device code processing tool ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This source is a collection of utilities run on device code's LLVM IR before
// handing off to back-end for further compilation or emitting SPIRV. The
// utilities are:
// - module splitter to split a big input module into smaller ones
// - specialization constant intrinsic transformation
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/SYCLLowerIR/CompileTimePropertiesPass.h"
#include "llvm/SYCLLowerIR/DeviceConfigFile.hpp"
#include "llvm/SYCLLowerIR/ESIMD/ESIMDUtils.h"
#include "llvm/SYCLLowerIR/HostPipes.h"
#include "llvm/SYCLLowerIR/LowerInvokeSimd.h"
#include "llvm/SYCLLowerIR/SYCLJointMatrixTransform.h"
#include "llvm/SYCLLowerIR/SYCLUtils.h"
#include "llvm/SYCLLowerIR/SpecConstants.h"
#include "llvm/SYCLLowerIR/Support.h"
#include "llvm/SYCLPostLink/ComputeModuleRuntimeInfo.h"
#include "llvm/SYCLPostLink/ESIMDPostSplitProcessing.h"
#include "llvm/SYCLPostLink/ModuleSplitter.h"
#include "llvm/SYCLPostLink/SpecializationConstants.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SimpleTable.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/WithColor.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;
using namespace llvm::sycl;

namespace {

#ifdef NDEBUG
#define DUMP_ENTRY_POINTS(...)
#else
constexpr int DebugPostLink = 0;

#define DUMP_ENTRY_POINTS(...)                                                 \
  if (DebugPostLink > 0) {                                                     \
    llvm::module_split::dumpEntryPoints(__VA_ARGS__);                          \
  }
#endif // NDEBUG

cl::OptionCategory PostLinkCat{"sycl-post-link options"};

// Column names in the output file table. Must match across tools -
// clang/lib/Driver/Driver.cpp, sycl-post-link.cpp, ClangOffloadWrapper.cpp
constexpr char COL_CODE[] = "Code";
constexpr char COL_SYM[] = "Symbols";
constexpr char COL_PROPS[] = "Properties";

// InputFilename - The filename to read from.
cl::opt<std::string> InputFilename{cl::Positional,
                                   cl::desc("<input bitcode file>"),
                                   cl::init("-"), cl::value_desc("filename")};

cl::opt<std::string> OutputDir{
    "out-dir",
    cl::desc(
        "Directory where files listed in the result file table will be output"),
    cl::value_desc("dirname"), cl::cat(PostLinkCat)};

cl::opt<std::string> DeviceLibDir{
    "device-lib-dir",
    cl::desc("Directory where sycl fallback device libraries reside"),
    cl::value_desc("dirname"), cl::cat(PostLinkCat)};

struct TargetFilenamePair {
  std::string Target;
  std::string Filename;
};

struct TargetFilenamePairParser : public cl::basic_parser<TargetFilenamePair> {
  using cl::basic_parser<TargetFilenamePair>::basic_parser;
  bool parse(cl::Option &O, StringRef ArgName, StringRef &ArgValue,
             TargetFilenamePair &Val) const {
    auto [Target, Filename] = ArgValue.split(",");
    if (Filename == "")
      std::swap(Target, Filename);
    Val = {Target.str(), Filename.str()};
    return false;
  }
};

cl::list<TargetFilenamePair, bool, TargetFilenamePairParser> OutputFiles{
    "o",
    cl::desc(
        "Specifies an output file. Multiple output files can be "
        "specified. Additionally, a target may be specified alongside an "
        "output file, which has the effect that when module splitting is "
        "performed, the modules that are in that output table are filtered "
        "so those modules are compatible with the target."),
    cl::value_desc("target filename pair"), cl::cat(PostLinkCat)};

cl::opt<bool> Force{"f", cl::desc("Enable binary output on terminals"),
                    cl::cat(PostLinkCat)};

cl::opt<bool> IROutputOnly{"ir-output-only", cl::desc("Output single IR file"),
                           cl::cat(PostLinkCat)};

cl::opt<bool> OutputAssembly{"S", cl::desc("Write output as LLVM assembly"),
                             cl::Hidden, cl::cat(PostLinkCat)};

cl::opt<bool> SplitEsimd{"split-esimd",
                         cl::desc("Split SYCL and ESIMD entry points"),
                         cl::cat(PostLinkCat)};

// TODO Design note: sycl-post-link should probably separate different kinds of
// its functionality on logical and source level:
//  - LLVM IR module splitting
//  - Running LLVM IR passes on resulting modules
//  - Generating additional files (like spec constants, dead arg info,...)
// The tool itself could be just a "driver" creating needed pipelines from the
// above actions. This could help make the tool structure clearer and more
// maintainable.

cl::opt<bool> LowerEsimd{"lower-esimd", cl::desc("Lower ESIMD constructs"),
                         cl::cat(PostLinkCat)};

cl::opt<bool> OptLevelO0("O0",
                         cl::desc("Optimization level 0. Similar to clang -O0"),
                         cl::cat(PostLinkCat));

cl::opt<bool> OptLevelO1("O1",
                         cl::desc("Optimization level 1. Similar to clang -O1"),
                         cl::cat(PostLinkCat));

cl::opt<bool> OptLevelO2("O2",
                         cl::desc("Optimization level 2. Similar to clang -O2"),
                         cl::cat(PostLinkCat));

cl::opt<bool> OptLevelOs(
    "Os",
    cl::desc(
        "Like -O2 with extra optimizations for size. Similar to clang -Os"),
    cl::cat(PostLinkCat));

cl::opt<bool> OptLevelOz(
    "Oz",
    cl::desc("Like -Os but reduces code size further. Similar to clang -Oz"),
    cl::cat(PostLinkCat));

cl::opt<bool> OptLevelO3("O3",
                         cl::desc("Optimization level 3. Similar to clang -O3"),
                         cl::cat(PostLinkCat));

cl::opt<bool> ForceDisableESIMDOpt("force-disable-esimd-opt", cl::Hidden,
                                   cl::desc("Force no optimizations."),
                                   cl::cat(PostLinkCat));

cl::opt<module_split::IRSplitMode> SplitMode(
    "split", cl::desc("split input module"), cl::Optional,
    cl::init(module_split::SPLIT_NONE),
    cl::values(clEnumValN(module_split::SPLIT_PER_TU, "source",
                          "1 output module per source (translation unit)"),
               clEnumValN(module_split::SPLIT_PER_KERNEL, "kernel",
                          "1 output module per kernel"),
               clEnumValN(module_split::SPLIT_AUTO, "auto",
                          "Choose split mode automatically")),
    cl::cat(PostLinkCat));

cl::opt<bool> DoSymGen{"symbols", cl::desc("generate exported symbol files"),
                       cl::cat(PostLinkCat)};

cl::opt<bool> DoPropGen{"properties",
                        cl::desc("generate module properties files"),
                        cl::cat(PostLinkCat)};

enum SpecConstLowerMode { SC_NATIVE_MODE, SC_EMULATION_MODE };

cl::opt<SpecConstLowerMode> SpecConstLower{
    "spec-const",
    cl::desc("lower and generate specialization constants information"),
    cl::Optional,
    cl::init(SC_NATIVE_MODE),
    cl::values(
        clEnumValN(SC_NATIVE_MODE, "native",
                   "lower spec constants to native spirv instructions so that "
                   "these values could be set at runtime"),
        clEnumValN(
            SC_EMULATION_MODE, "emulation",
            "remove specialization constants and replace it with emulation")),
    cl::cat(PostLinkCat)};

cl::opt<bool> EmitKernelParamInfo{
    "emit-param-info", cl::desc("emit kernel parameter optimization info"),
    cl::cat(PostLinkCat)};

cl::opt<bool> EmitProgramMetadata{"emit-program-metadata",
                                  cl::desc("emit SYCL program metadata"),
                                  cl::cat(PostLinkCat)};

cl::opt<bool> EmitKernelNames{
    "emit-kernel-names", cl::desc("emit kernel names"), cl::cat(PostLinkCat)};

cl::opt<bool> EmitExportedSymbols{"emit-exported-symbols",
                                  cl::desc("emit exported symbols"),
                                  cl::cat(PostLinkCat)};

cl::opt<bool> EmitImportedSymbols{"emit-imported-symbols",
                                  cl::desc("emit imported symbols"),
                                  cl::cat(PostLinkCat)};

cl::opt<bool> EmitOnlyKernelsAsEntryPoints{
    "emit-only-kernels-as-entry-points",
    cl::desc("Consider only sycl_kernel functions as entry points for "
             "device code split"),
    cl::cat(PostLinkCat), cl::init(false)};

cl::opt<bool> DeviceGlobals{
    "device-globals",
    cl::desc("Lower and generate information about device global variables"),
    cl::cat(PostLinkCat)};

cl::opt<bool> GenerateDeviceImageWithDefaultSpecConsts{
    "generate-device-image-default-spec-consts",
    cl::desc("Generate new device image(s) which is a copy of output images "
             "but contain specialization constants "
             "replaced with default values from specialization id(s)."),
    cl::cat(PostLinkCat)};

cl::opt<bool> AllowDeviceImageDependencies{
    "allow-device-image-dependencies",
    cl::desc("Allow dependencies between device images"), cl::cat(PostLinkCat),
    cl::init(false)};

struct IrPropSymFilenameTriple {
  std::string Ir;
  std::string Prop;
  std::string Sym;
};

unsigned getOptLevel() {
  if (OptLevelO3)
    return 3;
  if (OptLevelO2 || OptLevelOs || OptLevelOz)
    return 2;
  if (OptLevelO1)
    return 1;
  if (OptLevelO0)
    return 0;

  return 2; // default value
}

static void writeToFile(const StringRef Filename, const StringRef Content) {
  std::error_code EC;
  raw_fd_ostream OS{Filename, EC, sys::fs::OpenFlags::OF_None};
  checkError(EC, "error opening the file '" + Filename + "'");
  OS.write(Content.data(), Content.size());
  OS.close();
}

void saveModuleIR(Module &M, const StringRef Filename) {
  std::error_code EC;
  raw_fd_ostream Out{Filename, EC, sys::fs::OF_None};
  checkError(EC, "error opening the file '" + Filename + "'");

  ModulePassManager MPM;
  ModuleAnalysisManager MAM;
  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  if (OutputAssembly)
    MPM.addPass(PrintModulePass(Out));
  else if (Force || !CheckBitcodeOutputToConsole(Out))
    MPM.addPass(BitcodeWriterPass(Out));
  MPM.run(M, MAM);
}

void saveModuleProperties(const module_split::ModuleDesc &MD,
                          const GlobalBinImageProps &GlobProps,
                          const StringRef Filename, StringRef Target = "") {

  PropSetRegTy PropSet;

  // For bf16 devicelib module, no kernel included and no specialization
  // constant used, skip regular Prop emit. However, we have fallback and
  // native version of bf16 devicelib and we need new property values to
  // indicate all exported function.
  if (!MD.isSYCLDeviceLib())
    PropSet = computeModuleProperties(MD.getModule(), MD.entries(), GlobProps,
                                      AllowDeviceImageDependencies);
  else
    PropSet = computeDeviceLibProperties(MD.getModule(), MD.Name);

  // When the split mode is none, the required work group size will be added
  // to the whole module, which will make the runtime unable to
  // launch the other kernels in the module that have different
  // required work group sizes or no required work group sizes. So we need to
  // remove the required work group size metadata in this case.
  if (SplitMode == module_split::SPLIT_NONE)
    PropSet.remove(PropSetRegTy::SYCL_DEVICE_REQUIREMENTS,
                   PropSetRegTy::PROPERTY_REQD_WORK_GROUP_SIZE);

  if (!Target.empty())
    PropSet.add(PropSetRegTy::SYCL_DEVICE_REQUIREMENTS, "compile_target",
                Target);

  std::error_code EC;
  raw_fd_ostream SCOut(Filename, EC);
  checkError(EC, "error opening file '" + Filename + "'");
  PropSet.write(SCOut);
}

// Saves specified collection of symbols to a file.
void saveModuleSymbolTable(const module_split::ModuleDesc &MD,
                           const StringRef Filename) {
  std::string SymT = computeModuleSymbolTable(MD.getModule(), MD.entries());
  writeToFile(Filename, SymT);
}

bool isTargetCompatibleWithModule(const std::string &Target,
                                  module_split::ModuleDesc &IrMD);

void addTableRow(util::SimpleTable &Table,
                 const IrPropSymFilenameTriple &RowData);

/// \param OutTables List of tables (one for each target) to output results
/// \param MD Module descriptor to save
/// \param OutputPrefix Prefix for all generated outputs.
/// \param IRFilename Filename of IR component. If filename is not empty, it
///   is recorded in the OutTable. Otherwise, a new file is created to save
///   the IR component, and the new name is recorded in the OutTable.
void saveModule(
    const std::vector<std::unique_ptr<util::SimpleTable>> &OutTables,
    module_split::ModuleDesc &MD, const int I, const Twine &OutputPrefix,
    const StringRef IRFilename) {
  IrPropSymFilenameTriple BaseTriple;
  StringRef Suffix = MD.isESIMD() ? "_esimd" : "";
  MD.saveSplitInformationAsMetadata();

  if (!IRFilename.empty()) {
    // Don't save IR, just record the filename.
    BaseTriple.Ir = IRFilename.str();
  } else {
    if (!MD.isSYCLDeviceLib()) {
      // For deviceLib Modules, we don't need to do clean up and no entry-point
      // is included. The module only includes a bunch of exported functions
      // intended to be invoked by user's device modules.
      MD.cleanup(AllowDeviceImageDependencies);
    }

    StringRef IRExtension = OutputAssembly ? ".ll" : ".bc";
    BaseTriple.Ir =
        (OutputPrefix + Suffix + "_" + Twine(I) + IRExtension).str();
    saveModuleIR(MD.getModule(), BaseTriple.Ir);
  }

  if (DoSymGen) {
    // Save the names of the entry points - the symbol table.
    BaseTriple.Sym = (OutputPrefix + Suffix + "_" + Twine(I) + ".sym").str();
    saveModuleSymbolTable(MD, BaseTriple.Sym);
  }

  for (const auto &[Table, OutputFile] : zip_equal(OutTables, OutputFiles)) {
    if (!isTargetCompatibleWithModule(OutputFile.Target, MD))
      continue;
    auto CopyTriple = BaseTriple;
    if (DoPropGen) {
      GlobalBinImageProps Props = {EmitKernelParamInfo, EmitProgramMetadata,
                                   EmitKernelNames,     EmitExportedSymbols,
                                   EmitImportedSymbols, DeviceGlobals};
      StringRef Target = OutputFile.Target;
      std::string NewSuff = Suffix.str();
      if (!Target.empty())
        NewSuff = (Twine("_") + Target).str();

      CopyTriple.Prop =
          (OutputPrefix + NewSuff + "_" + Twine(I) + ".prop").str();
      saveModuleProperties(MD, Props, CopyTriple.Prop, Target);
    }
    addTableRow(*Table, CopyTriple);
  }
}

void saveDeviceLibModule(
    const std::vector<std::unique_ptr<util::SimpleTable>> &OutTables,
    const Twine &OutputPrefix, const int I,
    const std::string &DeviceLibFileName) {
  assert(!DeviceLibFileName.empty() &&
         "DeviceLibFileName is expected to be non-empty.");
  SMDiagnostic Err;
  LLVMContext Context;
  StringRef DeviceLibLoc = DeviceLibDir;
  StringRef Sep = llvm::sys::path::get_separator();
  std::string DeviceLibPath =
      (DeviceLibLoc.str() + Sep + DeviceLibFileName).str();
  std::unique_ptr<Module> DeviceLibIR =
      parseIRFile(DeviceLibPath, Err, Context);
  Module *DeviceLibMPtr = DeviceLibIR.get();
  if (!DeviceLibMPtr) {
    errs() << "sycl-post-link NOTE: fail to load bfloat16 device library "
              "modules\n";
    return;
  }
  llvm::module_split::ModuleDesc DeviceLibMD(std::move(DeviceLibIR),
                                             DeviceLibFileName);
  saveModule(OutTables, DeviceLibMD, I, OutputPrefix, "");
}

constexpr int MAX_COLUMNS_IN_FILE_TABLE = 3;

void addTableRow(util::SimpleTable &Table,
                 const IrPropSymFilenameTriple &RowData) {
  SmallVector<StringRef, MAX_COLUMNS_IN_FILE_TABLE> Row;

  for (const std::string *S : {&RowData.Ir, &RowData.Prop, &RowData.Sym}) {
    if (!S->empty()) {
      Row.push_back(StringRef(*S));
    }
  }
  assert(static_cast<size_t>(Table.getNumColumns()) == Row.size());
  Table.addRow(Row);
}

// Checks if the given target and module are compatible.
// A target and module are compatible if all the optional kernel features
// the module uses are supported by that target (i.e. that module can be
// compiled for that target and then be executed on that target). This
// information comes from the device config file (DeviceConfigFile.td).
// For example, the intel_gpu_tgllp target does not support fp64 - therefore,
// a module using fp64 would *not* be compatible with intel_gpu_tgllp.
bool isTargetCompatibleWithModule(const std::string &Target,
                                  module_split::ModuleDesc &IrMD) {
  // When the user does not specify a target,
  // (e.g. -o out.table compared to -o intel_gpu_pvc,out-pvc.table)
  // Target will be empty and we will not want to perform any filtering, so
  // we return true here.
  if (Target.empty())
    return true;

  // TODO: If a target not found in the device config file is passed,
  // to sycl-post-link, then we should probably throw an error. However,
  // since not all the information for all the targets is filled out
  // right now, we return true, having the affect that unrecognized
  // targets have no filtering applied to them.
  if (!is_contained(DeviceConfigFile::TargetTable, Target))
    return true;

  const DeviceConfigFile::TargetInfo &TargetInfo =
      DeviceConfigFile::TargetTable[Target];
  const SYCLDeviceRequirements &ModuleReqs =
      IrMD.getOrComputeDeviceRequirements();

  // Check to see if all the requirements of the input module
  // are compatbile with the target.
  for (const auto &Aspect : ModuleReqs.Aspects) {
    if (!is_contained(TargetInfo.aspects, Aspect.Name))
      return false;
  }

  // Check if module sub group size is compatible with the target.
  // For ESIMD, the reqd_sub_group_size will be 1; this is not
  // a supported by any backend (e.g. no backend can support a kernel
  // with sycl::reqd_sub_group_size(1)), but for ESIMD, this is
  // a special case.
  if (!IrMD.isESIMD() && ModuleReqs.SubGroupSize.has_value() &&
      !is_contained(TargetInfo.subGroupSizes, *ModuleReqs.SubGroupSize))
    return false;

  return true;
}

std::vector<std::unique_ptr<util::SimpleTable>>
processInputModule(std::unique_ptr<Module> M, const StringRef OutputPrefix) {
  // Construct the resulting table which will accumulate all the outputs.
  SmallVector<StringRef, MAX_COLUMNS_IN_FILE_TABLE> ColumnTitles{
      StringRef(COL_CODE)};

  if (DoPropGen)
    ColumnTitles.push_back(COL_PROPS);

  if (DoSymGen)
    ColumnTitles.push_back(COL_SYM);

  Expected<std::unique_ptr<util::SimpleTable>> TableE =
      util::SimpleTable::create(ColumnTitles);
  CHECK_AND_EXIT(TableE.takeError());
  std::vector<std::unique_ptr<util::SimpleTable>> Tables;
  for (size_t i = 0; i < OutputFiles.size(); ++i) {
    Expected<std::unique_ptr<util::SimpleTable>> TableE =
        util::SimpleTable::create(ColumnTitles);
    CHECK_AND_EXIT(TableE.takeError());
    Tables.push_back(std::move(TableE.get()));
  }

  // Used in output filenames generation.
  int ID = 0;
  if (llvm::esimd::moduleContainsInvokeSimdBuiltin(*M) && SplitEsimd)
    error("'invoke_simd' calls detected, '-" + SplitEsimd.ArgStr +
          "' must not be specified");

  // Keeps track of any changes made to the input module and report to the user
  // if none were made.
  bool Modified = llvm::module_split::runPreSplitProcessingPipeline(*M);

  // Keeps track of whether any device image uses bf16 devicelib.
  bool IsBF16DeviceLibUsed = false;

  DUMP_ENTRY_POINTS(*M, EmitOnlyKernelsAsEntryPoints, "Input");

  // -ir-output-only assumes single module output thus no code splitting.
  // Violation of this invariant is user error and must've been reported.
  // However, if split mode is "auto", then entry point filtering is still
  // performed.
  assert((!IROutputOnly || (SplitMode == module_split::SPLIT_NONE) ||
          (SplitMode == module_split::SPLIT_AUTO)) &&
         "invalid split mode for IR-only output");

  std::unique_ptr<module_split::ModuleSplitterBase> Splitter =
      module_split::getDeviceCodeSplitter(
          std::make_unique<module_split::ModuleDesc>(std::move(M)), SplitMode,
          IROutputOnly, EmitOnlyKernelsAsEntryPoints,
          AllowDeviceImageDependencies);
  bool SplitOccurred = Splitter->remainingSplits() > 1;
  Modified |= SplitOccurred;

  // FIXME: this check is not performed for ESIMD splits
  if (DeviceGlobals) {
    auto E = Splitter->verifyNoCrossModuleDeviceGlobalUsage();
    if (E)
      error(toString(std::move(E)));
  }

  std::optional<SpecConstantsPass::HandlingMode> SCMode;
  if (SpecConstLower.getNumOccurrences() > 0)
    SCMode = SpecConstLower == SC_NATIVE_MODE
                 ? SpecConstantsPass::HandlingMode::native
                 : SpecConstantsPass::HandlingMode::emulation;

  // It is important that we *DO NOT* preserve all the splits in memory at the
  // same time, because it leads to a huge RAM consumption by the tool on bigger
  // inputs.
  while (Splitter->hasMoreSplits()) {
    std::unique_ptr<module_split::ModuleDesc> MDesc = Splitter->nextSplit();
    DUMP_ENTRY_POINTS(MDesc->entries(), MDesc->Name.c_str(), 1);

    MDesc->fixupLinkageOfDirectInvokeSimdTargets();

    ESIMDProcessingOptions Options = {SplitMode,
                                      EmitOnlyKernelsAsEntryPoints,
                                      AllowDeviceImageDependencies,
                                      LowerEsimd,
                                      SplitEsimd,
                                      getOptLevel(),
                                      ForceDisableESIMDOpt};
    auto ModulesOrErr =
        handleESIMD(std::move(MDesc), Options, Modified, SplitOccurred);
    CHECK_AND_EXIT(ModulesOrErr.takeError());
    SmallVector<std::unique_ptr<module_split::ModuleDesc>, 2> &MMs =
        *ModulesOrErr;
    assert(MMs.size() && "at least one module is expected after ESIMD split");
    SmallVector<std::unique_ptr<module_split::ModuleDesc>, 2>
        MMsWithDefaultSpecConsts;
    Modified |=
        handleSpecializationConstants(MMs, SCMode, MMsWithDefaultSpecConsts,
                                      GenerateDeviceImageWithDefaultSpecConsts);

    if (IROutputOnly) {
      if (SplitOccurred) {
        error("some modules had to be split, '-" + IROutputOnly.ArgStr +
              "' can't be used");
      }
      MMs.front()->cleanup(AllowDeviceImageDependencies);
      saveModuleIR(MMs.front()->getModule(), OutputFiles[0].Filename);
      return Tables;
    }
    // Empty IR file name directs saveModule to generate one and save IR to
    // it:
    std::string OutIRFileName = "";

    if (!Modified && (OutputFiles.getNumOccurrences() == 0)) {
      assert(!SplitOccurred);
      OutIRFileName = InputFilename; // ... non-empty means "skip IR writing"
      errs() << "sycl-post-link NOTE: no modifications to the input LLVM IR "
                "have been made\n";
    }
    for (const std::unique_ptr<module_split::ModuleDesc> &IrMD : MMs) {
      IsBF16DeviceLibUsed |= isSYCLDeviceLibBF16Used(IrMD->getModule());
      saveModule(Tables, *IrMD, ID, OutputPrefix, OutIRFileName);
    }

    ++ID;

    if (!MMsWithDefaultSpecConsts.empty()) {
      for (size_t i = 0; i != MMsWithDefaultSpecConsts.size(); ++i) {
        const std::unique_ptr<module_split::ModuleDesc> &IrMD =
            MMsWithDefaultSpecConsts[i];
        IsBF16DeviceLibUsed |= isSYCLDeviceLibBF16Used(IrMD->getModule());
        saveModule(Tables, *IrMD, ID, OutputPrefix, OutIRFileName);
      }

      ++ID;
    }
  }

  if (IsBF16DeviceLibUsed && (DeviceLibDir.getNumOccurrences() > 0)) {
    saveDeviceLibModule(Tables, OutputPrefix, ID,
                        "libsycl-fallback-bfloat16.bc");
    saveDeviceLibModule(Tables, OutputPrefix, ID + 1,
                        "libsycl-native-bfloat16.bc");
  }
  return Tables;
}

/// Gets output prefix used for all output files from this tool.
std::string getOutputPrefix() {
  StringRef Dir0 = OutputDir.getNumOccurrences() > 0
                       ? OutputDir
                       : sys::path::parent_path(OutputFiles[0].Filename);
  StringRef Sep = sys::path::get_separator();
  std::string Dir = Dir0.str();
  if (!Dir0.empty() && !Dir0.ends_with(Sep))
    Dir += Sep.str();

  return (Dir + sys::path::stem(OutputFiles[0].Filename)).str();
}

} // namespace

int main(int argc, char **argv) {
  InitLLVM X{argc, argv};

  LLVMContext Context;
  cl::HideUnrelatedOptions({&PostLinkCat});
  cl::ParseCommandLineOptions(
      argc, argv,
      "SYCL post-link device code processing tool.\n"
      "This is a collection of utilities run on device code's LLVM IR before\n"
      "handing off to back-end for further compilation or emitting SPIRV.\n"
      "The utilities are:\n"
      "- SYCL and ESIMD kernels can be split into separate modules with\n"
      "  '-split-esimd' option. The option has no effect when there is only\n"
      "  one type of kernels in the input module. Functions unreachable from\n"
      "  any entry point (kernels and SYCL_EXTERNAL functions) are\n"
      "  dropped from the resulting module(s).\n"
      "- Module splitter to split a big input module into smaller ones.\n"
      "  Groups kernels using function attribute 'sycl-module-id', i.e.\n"
      "  kernels with the same values of the 'sycl-module-id' attribute will\n"
      "  be put into the same module. If -split=kernel option is specified,\n"
      "  one module per kernel will be emitted.\n"
      "  '-split=auto' mode automatically selects the best way of splitting\n"
      "  kernels into modules based on some heuristic.\n"
      "  The '-split' option is compatible with '-split-esimd'. In this case,\n"
      "  first input module will be split according to the '-split' option\n"
      "  processing algorithm, not distinguishing between SYCL and ESIMD\n"
      "  kernels. Then each resulting module is further split into SYCL and\n"
      "  ESIMD parts if the module has both kinds of entry points.\n"
      "- If -symbols options is also specified, then for each produced module\n"
      "  a text file containing names of all spir kernels in it is generated.\n"
      "- Specialization constant intrinsic transformer. Replaces symbolic\n"
      "  ID-based intrinsics to integer ID-based ones to make them friendly\n"
      "  for the SPIRV translator\n"
      "When the tool splits input module into regular SYCL and ESIMD kernels,\n"
      "it performs a set of specific lowering and transformation passes on\n"
      "ESIMD module, which is enabled by the '-lower-esimd' option. Regular\n"
      "optimization level options are supported, e.g. -O[0|1|2|3|s|z].\n"
      "Normally, the tool generates a number of files and \"file table\"\n"
      "file listing all generated files in a table manner. For example, if\n"
      "the input file 'example.bc' contains two kernels, then the command\n"
      "  $ sycl-post-link --properties --split=kernel --symbols \\\n"
      "    --spec-const=native    -o example.table example.bc\n"
      "will produce 'example.table' file with the following content:\n"
      "  [Code|Properties|Symbols]\n"
      "  example_0.bc|example_0.prop|example_0.sym\n"
      "  example_1.bc|example_1.prop|example_1.sym\n"
      "When only specialization constant processing is needed, the tool can\n"
      "output a single transformed IR file if --ir-output-only is specified:\n"
      "  $ sycl-post-link --ir-output-only --spec-const=emulation \\\n"
      "    -o example_p.bc example.bc\n"
      "will produce single output file example_p.bc suitable for SPIRV\n"
      "translation.\n"
      "--ir-output-only option is not not compatible with split modes other\n"
      "than 'auto'.\n");

  bool DoSplit = SplitMode.getNumOccurrences() > 0;
  bool DoSplitEsimd = SplitEsimd.getNumOccurrences() > 0;
  bool DoLowerEsimd = LowerEsimd.getNumOccurrences() > 0;
  bool DoSpecConst = SpecConstLower.getNumOccurrences() > 0;
  bool DoParamInfo = EmitKernelParamInfo.getNumOccurrences() > 0;
  bool DoProgMetadata = EmitProgramMetadata.getNumOccurrences() > 0;
  bool DoKernelNames = EmitKernelNames.getNumOccurrences() > 0;
  bool DoExportedSyms = EmitExportedSymbols.getNumOccurrences() > 0;
  bool DoImportedSyms = EmitImportedSymbols.getNumOccurrences() > 0;
  bool DoDeviceGlobals = DeviceGlobals.getNumOccurrences() > 0;
  bool DoGenerateDeviceImageWithDefaulValues =
      GenerateDeviceImageWithDefaultSpecConsts.getNumOccurrences() > 0;

  if (!DoSplit && !DoSpecConst && !DoSymGen && !DoPropGen && !DoParamInfo &&
      !DoProgMetadata && !DoSplitEsimd && !DoKernelNames && !DoExportedSyms &&
      !DoImportedSyms && !DoDeviceGlobals && !DoLowerEsimd) {
    errs() << "no actions specified; try --help for usage info\n";
    return 1;
  }
  if (IROutputOnly && DoSplit && (SplitMode != module_split::SPLIT_AUTO)) {
    errs() << "error: -" << SplitMode.ArgStr << "=" << SplitMode.ValueStr
           << " can't be used with -" << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  if (IROutputOnly && DoSplitEsimd) {
    errs() << "error: -" << SplitEsimd.ArgStr << " can't be used with -"
           << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  if (IROutputOnly && DoSymGen) {
    errs() << "error: -" << DoSymGen.ArgStr << " can't be used with -"
           << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  if (IROutputOnly && DoPropGen) {
    errs() << "error: -" << DoPropGen.ArgStr << " can't be used with -"
           << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  if (IROutputOnly && DoParamInfo) {
    errs() << "error: -" << EmitKernelParamInfo.ArgStr << " can't be used with"
           << " -" << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  if (IROutputOnly && DoProgMetadata) {
    errs() << "error: -" << EmitProgramMetadata.ArgStr << " can't be used with"
           << " -" << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  if (IROutputOnly && DoKernelNames) {
    errs() << "error: -" << EmitKernelNames.ArgStr << " can't be used with"
           << " -" << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  if (IROutputOnly && DoExportedSyms) {
    errs() << "error: -" << EmitExportedSymbols.ArgStr << " can't be used with"
           << " -" << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  if (IROutputOnly && DoImportedSyms) {
    errs() << "error: -" << EmitImportedSymbols.ArgStr << " can't be used with"
           << " -" << IROutputOnly.ArgStr << "\n";
    return 1;
  }
  if (IROutputOnly && DoGenerateDeviceImageWithDefaulValues) {
    errs() << "error: -" << GenerateDeviceImageWithDefaultSpecConsts.ArgStr
           << " can't be used with -" << IROutputOnly.ArgStr << "\n";
    return 1;
  }

  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);
  // It is OK to use raw pointer here as we control that it does not outlive M
  // or objects it is moved to
  Module *MPtr = M.get();

  if (!MPtr) {
    Err.print(argv[0], errs());
    return 1;
  }

  if (OutputFiles.getNumOccurrences() == 0) {
    StringRef S = IROutputOnly ? ".out" : ".files";
    OutputFiles.push_back({{}, (sys::path::stem(InputFilename) + S).str()});
  }

  std::string OutputPrefix = getOutputPrefix();
  std::vector<std::unique_ptr<util::SimpleTable>> Tables =
      processInputModule(std::move(M), OutputPrefix);

  // Input module was processed and a single output file was requested.
  if (IROutputOnly)
    return 0;

  // Emit the resulting tables
  for (const auto &[Table, OutputFile] : zip_equal(Tables, OutputFiles)) {
    std::error_code EC;
    raw_fd_ostream Out{OutputFile.Filename, EC, sys::fs::OF_None};
    checkError(EC, "error opening file '" + OutputFile.Filename + "'");
    Table->write(Out);
  }

  return 0;
}
