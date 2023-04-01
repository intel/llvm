//==-------------------------- SYCLKernelInfo.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SYCLKernelInfo.h"

#include "KernelIO.h"

#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace jit_compiler;

static llvm::cl::opt<std::string>
    ModuleInfoFilePath("sycl-info-path",
                       llvm::cl::desc("Path to the SYCL module info YAML file"),
                       llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::AnalysisKey SYCLModuleInfoAnalysis::Key;

void SYCLModuleInfoAnalysis::loadModuleInfoFromFile() {
  DiagnosticPrinterRawOStream Printer{llvm::errs()};
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrError =
      MemoryBuffer::getFile(ModuleInfoFilePath);
  if (std::error_code EC = FileOrError.getError()) {
    Printer << "Could not open file " << ModuleInfoFilePath << " due to error "
            << EC.message() << "\n";
    return;
  }
  llvm::yaml::Input In{FileOrError->get()->getMemBufferRef()};
  ModuleInfo = std::make_unique<SYCLModuleInfo>();
  In >> *ModuleInfo;
  if (In.error()) {
    Printer << "Error parsing YAML from " << ModuleInfoFilePath << ": "
            << In.error().message() << "\n";
    return;
  }
}

SYCLModuleInfoAnalysis::Result
SYCLModuleInfoAnalysis::run(Module &, ModuleAnalysisManager &) {
  if (!ModuleInfo) {
    loadModuleInfoFromFile();
  }
  return {ModuleInfo.get()};
}

PreservedAnalyses SYCLModuleInfoPrinter::run(Module &Mod,
                                             ModuleAnalysisManager &MAM) {
  jit_compiler::SYCLModuleInfo *ModuleInfo =
      MAM.getResult<SYCLModuleInfoAnalysis>(Mod).ModuleInfo;
  if (!ModuleInfo) {
    DiagnosticPrinterRawOStream Printer{llvm::errs()};
    Printer << "Error: No module info available\n";
    return PreservedAnalyses::all();
  }
  llvm::yaml::Output Out{llvm::outs()};
  Out << *ModuleInfo;
  return PreservedAnalyses::all();
}
