// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <compiler/utils/pass_machinery.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <multi_llvm/llvm_version.h>

using namespace llvm;

namespace compiler {
namespace utils {

// Note that Clang has three on/off options for debugging pass managers:
// `-fdebug-pass-manager`, `-fdebug-pass-structure`, and
// `-fdebug-pass-arguments``.
// LLVM's `opt` tool combines them all into one:
//   --debug-pass-manager (Normal)
//   --debug-pass-manager=verbose (Verbose)
//   --debug-pass-manager=quiet (Quiet)
// However, the mapping is not one-to-one:
// opt:
//   PrintPassOptions PrintPassOpts;
//   PrintPassOpts.Verbose = DebugPM == DebugLogging::Verbose;
//   PrintPassOpts.SkipAnalyses = DebugPM == DebugLogging::Quiet;
//   StandardInstrumentations SI(DebugPM != DebugLogging::None, VerifyEachPass,
//                               PrintPassOpts);
// clang:
//   bool DebugPassStructure = CodeGenOpts.DebugPass == "Structure";
//   PrintPassOptions PrintPassOpts;
//   PrintPassOpts.Indent = DebugPassStructure;
//   PrintPassOpts.SkipAnalyses = DebugPassStructure;
//   StandardInstrumentations SI(CodeGenOpts.DebugPassManager ||
//                                   DebugPassStructure,
//                               false, PrintPassOpts);
// While clang also pushes `mdebug-pass` onto LLVM, it only works for the
// legacy pass manager, and so we choose to only support and model the
// `debug-pass-manager` form.
static cl::opt<DebugLogging> DebugPM(
    "debug-pass-manager", cl::Hidden, cl::ValueOptional,
    cl::desc("Print pass management debugging information"),
    cl::init(DebugLogging::None),
    cl::values(
        clEnumValN(DebugLogging::Normal, "", ""),
        clEnumValN(DebugLogging::Quiet, "quiet",
                   "Skip printing info about analyses"),
        clEnumValN(
            DebugLogging::Verbose, "verbose",
            "Print extra information about adaptors and pass managers")));

static cl::opt<bool> VerifyEach("verify-each",
                                cl::desc("Verify after each transform"));

PassMachinery::PassMachinery(LLVMContext &Ctx, TargetMachine *TM,
                             bool VerifyEach, DebugLogging debugLogLevel)
    : TM(TM) {
  llvm::PrintPassOptions PrintPassOpts;
  PrintPassOpts.Verbose = DebugPM == DebugLogging::Verbose;
  PrintPassOpts.SkipAnalyses = DebugPM == DebugLogging::Quiet;
  PrintPassOpts.Indent = debugLogLevel != DebugLogging::None;
  SI = std::make_unique<StandardInstrumentations>(
      Ctx, debugLogLevel != DebugLogging::None, VerifyEach, PrintPassOpts);
}

PassMachinery::~PassMachinery() {}

void PassMachinery::initializeStart(PipelineTuningOptions PTO) {
  const std::optional<PGOOptions> PGOOpt;
  PB = PassBuilder(TM, PTO, PGOOpt, &PIC);
}

void PassMachinery::registerPasses() {
  buildDefaultAAPipeline();
  registerLLVMAnalyses();
}

void PassMachinery::initializeFinish() {
  // Register LLVM analyses now, with the knowledge that users have had the
  // chance to register their own versions of LLVM analyses first.
  registerPasses();
  // With all passes registered, cross-register all the proxies
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Allow registration of callbacks and instrumentation machinery
  addClassToPassNames();
  registerPassCallbacks();

  // Register pass instrumentation
#if LLVM_VERSION_GREATER_EQUAL(17, 0)
  SI->registerCallbacks(PIC, &MAM);
#else
  SI->registerCallbacks(PIC, &FAM);
#endif
}

void PassMachinery::buildDefaultAAPipeline() {
  FAM.registerPass([&] { return PB.buildDefaultAAPipeline(); });
}

void PassMachinery::registerLLVMAnalyses() {
  // Register standard analyses
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
}

}  // namespace utils
}  // namespace compiler

namespace compiler {
namespace utils {
/// Helper functions for printing
void printPassName(StringRef PassName, raw_ostream &OS) {
  OS << "  " << PassName << "\n";
}

void printPassName(StringRef PassName, StringRef Params, raw_ostream &OS) {
  OS << "  " << PassName << "<" << Params << ">\n";
}

}  // namespace utils
}  // namespace compiler
