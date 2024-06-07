//==-- sycl-module-split: command line tool for testing SYCL Module Splitting //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This program can be used only to test the SYCL Module Splitting.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/SYCLLowerIR/ModuleSplitter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/PropertySetIO.h"
#include "llvm/Support/SimpleTable.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::util;
using namespace module_split;

static cl::OptionCategory SplitCategory("Split options");

static cl::opt<std::string> InputFilename(cl::Positional, cl::desc(""),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string>
    OutputFilenamePrefix("o", cl::desc("output filename prefix"),
                         cl::value_desc("filename prefix"), cl::init("output"),
                         cl::cat(SplitCategory));

cl::opt<bool> OutputAssembly{"S", cl::desc("Write output as LLVM assembly"),
                             cl::cat(SplitCategory)};

cl::opt<IRSplitMode> SplitMode(
    "split", cl::desc("split input module"), cl::Optional, cl::init(SPLIT_NONE),
    cl::values(clEnumValN(module_split::SPLIT_PER_TU, "source",
                          "1 output module per source (translation unit)"),
               clEnumValN(module_split::SPLIT_PER_KERNEL, "kernel",
                          "1 output module per kernel"),
               clEnumValN(module_split::SPLIT_AUTO, "auto",
                          "Choose split mode automatically")),
    cl::cat(SplitCategory));

void writeStringToFile(const std::string &Content, StringRef Path) {
  std::error_code EC;
  raw_fd_ostream OS(Path, EC);
  if (EC) {
    errs() << formatv("error opening file: {0}\n", Path);
    exit(1);
  }

  OS << Content << "\n";
}

void writePropertiesToFile(const PropertySetRegistry &Properties,
                           StringRef Path) {
  std::error_code EC;
  raw_fd_ostream OS(Path, EC);
  if (EC) {
    errs() << formatv("error opening file: {0}\n", Path);
    exit(1);
  }

  Properties.write(OS);
}

void dumpModulesAsTable(const std::vector<SplitModule> &SplitModules,
                        StringRef Path) {
  std::vector<StringRef> Columns = {"Code", "Properties", "Symbols"};
  auto TableOrErr = SimpleTable::create(Columns);
  if (!TableOrErr) {
    errs() << "can't create a table\n";
    exit(1);
  }

  std::unique_ptr<SimpleTable> Table = std::move(*TableOrErr);
  for (const auto &[I, SM] : enumerate(SplitModules)) {
    std::string SymbolsFile = (Twine(Path) + "_" + Twine(I) + ".sym").str();
    std::string PropertiesFile = (Twine(Path) + "_" + Twine(I) + ".prop").str();
    writePropertiesToFile(SM.Properties, PropertiesFile);
    writeStringToFile(SM.Symbols, SymbolsFile);
    SmallVector<StringRef, 3> Row = {SM.ModuleFilePath, PropertiesFile,
                                     SymbolsFile};
    Table->addRow(Row);
  }

  std::error_code EC;
  raw_fd_ostream OS((Path + ".table").str(), EC);
  if (EC) {
    errs() << formatv("error opening file: {0}\n", Path);
    exit(1);
  }

  Table->write(OS);
}

int main(int argc, char *argv[]) {
  LLVMContext C;
  SMDiagnostic Err;
  cl::ParseCommandLineOptions(argc, argv, "SYCL Module Splitter\n");

  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, C);
  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }

  ModuleSplitterSettings Settings;
  Settings.Mode = SplitMode;
  Settings.OutputAssembly = OutputAssembly;
  Settings.OutputPrefix = OutputFilenamePrefix;
  auto SplitModulesOrErr = splitSYCLModule(std::move(M), Settings);
  if (!SplitModulesOrErr) {
    Err.print(argv[0], errs());
    return 1;
  }

  dumpModulesAsTable(*SplitModulesOrErr, OutputFilenamePrefix);
}
