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

#include <compiler/utils/builtin_info.h>
#include <compiler/utils/device_info.h>
#include <compiler/utils/metadata.h>
#include <compiler/utils/optimal_builtin_replacement_pass.h>
#include <compiler/utils/pass_machinery.h>
#include <compiler/utils/sub_group_analysis.h>
#include <compiler/utils/vectorization_factor.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/InitializePasses.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetLoweringObjectFile.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <multi_llvm/llvm_version.h>

#include <string>

#include "vecz/pass.h"
#include "vecz/vecz_target_info.h"

static llvm::cl::opt<std::string> InputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input .bc file>"),
    llvm::cl::init("-"));

static llvm::cl::opt<std::string> OutputFilename(
    "o", llvm::cl::desc("Override output filename"),
    llvm::cl::value_desc("filename"));
static llvm::cl::opt<bool, false> WriteTextual(
    "S", llvm::cl::desc("Write module as text"));

static llvm::cl::list<std::string> KernelNameSpecs(
    "k", llvm::cl::desc("Kernel to vectorize"), llvm::cl::ZeroOrMore,
    llvm::cl::value_desc("name"));

static llvm::cl::opt<unsigned> SIMDDimIdx(
    "d", llvm::cl::desc("Dimension index to vectorize on"), llvm::cl::init(0),
    llvm::cl::value_desc("dimension"));

static llvm::cl::opt<unsigned> SIMDWidth(
    "w", llvm::cl::desc("Width to vectorize to"), llvm::cl::init(0),
    llvm::cl::value_desc("width"));

static llvm::cl::opt<bool> FailQuietly(
    "vecz-fail-quietly",
    llvm::cl::desc("don't return an error code on vectorization failure"));

static llvm::cl::opt<bool> ChoicesHelp(
    "vecz-choices-help",
    llvm::cl::desc("see information about available choices"));

static llvm::cl::opt<bool> VeczAuto(
    "vecz-auto",
    llvm::cl::desc("run the vectorizer if it is found to be useful"));

static llvm::cl::opt<unsigned, 0> VeczSimdWidth(
    "vecz-simd-width",
    llvm::cl::desc("manually set the SIMD width for the vectorizer"));

static llvm::cl::opt<llvm::cl::boolOrDefault> VeczScalable(
    "vecz-scalable",
    llvm::cl::desc("force scalable vectorization for the vectorizer"));

// Allow the passing of Vecz Choices string on the command line. This is parsed
// after the choices environment variable, thus overriding it.
static llvm::cl::opt<std::string> ChoicesString(
    "vecz-choices", llvm::cl::desc("Set vecz choices"));

static llvm::cl::opt<bool> VeczCollectStats(
    "vecz-llvm-stats", llvm::cl::desc("enable reporting LLVM statistics"));

static llvm::cl::opt<std::string> UserTriple(
    "vecz-target-triple", llvm::cl::desc("the target triple"));
static llvm::cl::opt<std::string> UserCPU("vecz-target-mcpu",
                                          llvm::cl::desc("Set the CPU model"));
static llvm::cl::opt<std::string> CPUFeatures(
    "vecz-target-features", llvm::cl::desc("Set the CPU feature string"));
static llvm::cl::opt<bool> DoubleSupport(
    "vecz-double-support", llvm::cl::init(true),
    llvm::cl::desc(
        "Assume the target has double-precision floating point support"));

static llvm::cl::list<unsigned> SGSizes(
    "device-sg-sizes",
    llvm::cl::desc("Comma-separated list of supported sub-group sizes"),
    llvm::cl::CommaSeparated);

static llvm::TargetMachine *initLLVMTarget(llvm::StringRef triple_string,
                                           llvm::StringRef cpu_model,
                                           llvm::StringRef target_features) {
  llvm::Triple triple(triple_string);
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllDisassemblers();

  llvm::TargetOptions opts;
  opts.DisableIntegratedAS = false;
  std::string e;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple.getTriple(), e);
  if (!target) {
    ::fprintf(stderr, "can't get target %s:%s\n", triple.getTriple().c_str(),
              e.c_str());
    ::exit(1);
  }
  llvm::PassRegistry &registry = *llvm::PassRegistry::getPassRegistry();
  llvm::initializeAlwaysInlinerLegacyPassPass(registry);
  return target->createTargetMachine(triple.getTriple(), cpu_model,
                                     target_features, opts,
                                     llvm::Reloc::Model::Static);
}

static vecz::VeczPassOptions getDefaultPassOptions() {
  // Enable/disable Choices from the CODEPLAY_VECZ_CHOICES environment
  // variable.
  vecz::VectorizationChoices Choices;

  const char *ptr = std::getenv("CODEPLAY_VECZ_CHOICES");
  if (ptr && !Choices.parseChoicesString(ptr)) {
    llvm::errs()
        << "Failed to parse the CODEPLAY_VECZ_CHOICES env variable.\n"
           "Use --vecz-choices-help for available choices and usage info.\n";
    ::exit(1);
  }

  // Parse the Vecz choices given in the command line
  const std::string &ch = ChoicesString;
  if (!ch.empty() && !Choices.parseChoicesString(ch)) {
    llvm::errs()
        << "Failed to parse the --vecz-choices command line option.\n"
           "Use --vecz-choices-help for available choices and usage info.\n";
    ::exit(1);
  }

  if (VeczCollectStats) {
    llvm::EnableStatistics(true);
  }

  auto const factor = SIMDWidth ? SIMDWidth : 4;
  auto VF = compiler::utils::VectorizationFactor::getFixedWidth(factor);
  if (VeczSimdWidth) {
    VF.setKnownMin(VeczSimdWidth);
  }

  if (VeczScalable == llvm::cl::BOU_TRUE) {
    VF.setIsScalable(true);
  } else if (VeczScalable == llvm::cl::BOU_FALSE) {
    VF.setIsScalable(false);
  }
  vecz::VeczPassOptions passOpts;
  passOpts.choices = Choices;
  passOpts.factor = VF;
  passOpts.vecz_auto = VeczAuto;
  passOpts.vec_dim_idx = SIMDDimIdx;
  passOpts.local_size = SIMDWidth;
  return passOpts;
}

// Parse a command line vectorization specification for a given kernel
// <kernel_spec> ::= <kernel_name> ':' <spec>
// <kernel_spec> ::= <kernel_name>
// <spec> ::= <vf><dim>(opt)<width>(opt)
//            <scalable_spec>(opt)<predicated_spec>(opt)
// <spec> ::= <spec> ',' <spec>
// <number> ::= [0-9]+
// <kernel_name> ::= [a-zA-Z_][a-zA-Z_0-9]+
// <dim> ::= '.' [123]
// <vf> ::= <number>
// <vf> ::= 'a' // automatic vectorization factor
// <simd_width> ::= '@' <number>
// <scalable_spec> ::= 's'
// <predicated_spec> ::= 'p'
static bool parsePassOptionsSwitch(
    const llvm::StringRef spec, llvm::StringRef &name,
    llvm::SmallVectorImpl<vecz::VeczPassOptions> &opts) {
  auto pair = spec.split(':');
  name = pair.first;
  auto vals = pair.second;
  auto defaults = getDefaultPassOptions();
  if (!name.size()) {
    return false;
  }
  if (!vals.empty()) {
    do {
      // HEREBEDRAGONS: The return status of `consumeInteger` and
      // `consume_front` are "failed" and "succeeded" respectively. It's
      // opposite day somewhere in llvm land...
      unsigned vf;
      auto opt = defaults;
      if (vals.consume_front("a")) {
        opt.vecz_auto = true;
      } else if (!vals.consumeInteger(10, vf)) {
        opt.factor = compiler::utils::VectorizationFactor::getFixedWidth(vf);
      }
      if (vals.consume_front(".")) {
        unsigned dim;
        if (vals.consumeInteger(10, dim)) {
          return false;
        }
        if (!dim || dim > 3) {
          return false;
        }
        opt.vec_dim_idx = dim;
      }
      if (vals.consume_front("@")) {
        unsigned simd_width;
        if (vals.consumeInteger(10, simd_width)) {
          return false;
        }
        opt.local_size = simd_width;
      }
      // <scalable_spec> ::= 's'
      opt.factor.setIsScalable(vals.consume_front("s"));
      // <predicated_spec> ::= 'p'
      if (vals.consume_front("p")) {
        opt.choices.enableVectorPredication();
      }
      opts.push_back(opt);
    } while (vals.consume_front(",") && !vals.empty());
    if (!vals.empty()) {
      return false;
    }
  } else {
    opts.push_back(defaults);
  }
  return true;
}

using KernelOptMap =
    llvm::SmallDenseMap<llvm::StringRef,
                        llvm::SmallVector<vecz::VeczPassOptions, 1>, 1>;

int main(const int argc, const char *const argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  if (ChoicesHelp) {
    const auto &Infos = vecz::VectorizationChoices::queryAvailableChoices();
    llvm::outs() << "Available Vecz Choices:\n\n";
    for (const auto &Info : Infos) {
      llvm::outs() << "  * " << Info.name << ":\n";
      llvm::outs() << "      " << Info.desc << "\n\n";
    }
    llvm::outs() << "Separate multiple items with any one of [:;,].\n"
                    "Prefix any choice with \"no\" to disable that option.\n";
    return 0;
  }

  // If the user didn't specify an output filename, but is reading from stdin,
  // output to stdout. This may be emitting binary, but trust the user to know
  // what they're doing. We could also emit a warning.
  if (OutputFilename.empty() && InputFilename == "-") {
    OutputFilename = "-";
  }

  if (OutputFilename.empty()) {
    llvm::errs() << "Error: no output filename was given (use -o <file>)\n";
    return 1;
  }

  llvm::SMDiagnostic err;
  llvm::LLVMContext context;
#if LLVM_VERSION_LESS(17, 0)
  context.setOpaquePointers(true);
#endif

  std::unique_ptr<llvm::Module> module =
      llvm::parseIRFile(InputFilename, err, context);

  if (!module) {
    auto errorOrInputFile =
        llvm::MemoryBuffer::getFileOrSTDIN(InputFilename.getValue());

    // If there was an error in getting the input file.
    if (!errorOrInputFile) {
      llvm::errs() << "Error: " << errorOrInputFile.getError().message() << " '"
                   << InputFilename.getValue() << "'\n";
      return 1;
    }

    llvm::errs() << "Error: bitcode file was malformed\n";
    err.print("veczc", llvm::errs(),
              llvm::sys::Process::StandardErrHasColors());
    return 1;
  }

  KernelOptMap kernelOpts;
  if (KernelNameSpecs.empty()) {
    auto defaults = getDefaultPassOptions();
    for (const auto &f : *module) {
      if (f.getCallingConv() != llvm::CallingConv::SPIR_KERNEL) {
        continue;
      }
      kernelOpts[f.getName()].push_back(defaults);
    }
  } else {
    for (const auto &S : KernelNameSpecs) {
      llvm::StringRef name;
      llvm::SmallVector<vecz::VeczPassOptions, 1> opts;
      if (!parsePassOptionsSwitch(S, name, opts)) {
        fprintf(stderr,
                "failed to parse kernel vectorization specification%s\n",
                name.str().c_str());
        return 1;
      }
      if (!module->getFunction(name)) {
        llvm::errs() << "Error: no such kernel to vectorize ('" << name
                     << "')\n";
        return 1;
      }
      kernelOpts[name] = std::move(opts);
    }
  }

  // Open the file.
  std::error_code EC;
  llvm::sys::fs::OpenFlags OpenFlags = llvm::sys::fs::OF_None;
  if (WriteTextual) {
    OpenFlags |= llvm::sys::fs::OF_Text;
  }
  auto Out =
      std::make_unique<llvm::ToolOutputFile>(OutputFilename, EC, OpenFlags);
  if (EC || !Out) {
    llvm::errs() << EC.message() << '\n';
    return 1;
  }

  std::unique_ptr<llvm::TargetMachine> tm(
      UserTriple.size() ? initLLVMTarget(UserTriple, UserCPU, CPUFeatures)
                        : nullptr);
  assert(!UserTriple.size() || tm);
  if (tm) {
    module->setTargetTriple(tm->getTargetTriple().getTriple());
    module->setDataLayout(tm->createDataLayout());
  }

  compiler::utils::PassMachinery passMach(context, tm.get());

  auto TICallback = [&](const llvm::Module &) {
    return vecz::createTargetInfoFromTargetMachine(tm.get());
  };

  passMach.initializeStart();
  passMach.getMAM().registerPass(
      [&] { return vecz::TargetInfoAnalysis(TICallback); });
  passMach.getMAM().registerPass(
      [&] { return compiler::utils::BuiltinInfoAnalysis(); });
  passMach.getMAM().registerPass(
      [&] { return compiler::utils::SubgroupAnalysis(); });
  passMach.getFAM().registerPass([] { return llvm::TargetIRAnalysis(); });
  passMach.getMAM().registerPass([] {
    compiler::utils::DeviceInfo Info{/*half*/ 0, /*float*/ 0, DoubleSupport,
                                     /*MaxWorthWidth*/ 64};
    for (const auto S : SGSizes) {
      Info.reqd_sub_group_sizes.push_back(S);
    }
    return compiler::utils::DeviceInfoAnalysis(Info);
  });
  passMach.getMAM().registerPass([&kernelOpts] {
    return vecz::VeczPassOptionsAnalysis(
        [&kernelOpts](llvm::Function &F, llvm::ModuleAnalysisManager &,
                      llvm::SmallVectorImpl<vecz::VeczPassOptions> &Opts) {
          auto it = kernelOpts.find(F.getName());
          if (it == kernelOpts.end()) {
            return false;
          }
          Opts.assign(it->second.begin(), it->second.end());
          return true;
        });
  });
  passMach.initializeFinish();

  llvm::ModulePassManager PM;

  // Forcibly compute the BuiltinInfoAnalysis so that cached retrievals work.
  PM.addPass(llvm::RequireAnalysisPass<compiler::utils::BuiltinInfoAnalysis,
                                       llvm::Module>());

  PM.addPass(llvm::createModuleToPostOrderCGSCCPassAdaptor(
      compiler::utils::OptimalBuiltinReplacementPass()));
  PM.addPass(vecz::RunVeczPass());
  PM.run(*module, passMach.getMAM());

  // If the user has specified a list of kernels to vectorize, we need to
  // check we've matched their expectations. If they didn't specify we work on
  // a "best-effort" basis
  if (!KernelNameSpecs.empty()) {
    for (auto p : kernelOpts) {
      auto &f = *module->getFunction(p.first);
      const auto &requested = p.getSecond();
      llvm::SmallVector<compiler::utils::LinkMetadataResult, 1> results;
      compiler::utils::parseOrigToVeczFnLinkMetadata(f, results);
      for (auto &expected : requested) {
        if (expected.vecz_auto) {
          continue;
        }
        bool found = false;
        for (auto &result : results) {
          // FIXME this probably not the best way to do this
          found |=
              result.second.vf.getKnownMin() >= expected.factor.getKnownMin();
        }
        if (!found) {
          llvm::errs() << "Error: Failed to vectorize function '" << f.getName()
                       << "'\n";
          return FailQuietly ? 0 : 1;
        }
      }
    }
  }

  // Write the resulting module.
  if (WriteTextual) {
    Out->os() << *module;
  } else {
    llvm::WriteBitcodeToFile(*module, Out->os());
  }

  Out->keep();

  if (llvm::AreStatisticsEnabled()) {
    llvm::PrintStatistics();
  }
  return 0;
}
