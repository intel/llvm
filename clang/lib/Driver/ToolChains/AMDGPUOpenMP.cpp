//===- AMDGPUOpenMP.cpp - AMDGPUOpenMP ToolChain Implementation -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUOpenMP.h"
#include "AMDGPU.h"
#include "clang/Driver/CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Path.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

namespace {

static const char *getOutputFileName(Compilation &C, StringRef Base,
                                     const char *Postfix,
                                     const char *Extension) {
  const char *OutputFileName;
  if (C.getDriver().isSaveTempsEnabled()) {
    OutputFileName =
        C.getArgs().MakeArgString(Base.str() + Postfix + "." + Extension);
  } else {
    std::string TmpName =
        C.getDriver().GetTemporaryPath(Base.str() + Postfix, Extension);
    OutputFileName = C.addTempFile(C.getArgs().MakeArgString(TmpName));
  }
  return OutputFileName;
}

static void addLLCOptArg(const llvm::opt::ArgList &Args,
                         llvm::opt::ArgStringList &CmdArgs) {
  if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    StringRef OOpt = "0";
    if (A->getOption().matches(options::OPT_O4) ||
        A->getOption().matches(options::OPT_Ofast))
      OOpt = "3";
    else if (A->getOption().matches(options::OPT_O0))
      OOpt = "0";
    else if (A->getOption().matches(options::OPT_O)) {
      // Clang and opt support -Os/-Oz; llc only supports -O0, -O1, -O2 and -O3
      // so we map -Os/-Oz to -O2.
      // Only clang supports -Og, and maps it to -O1.
      // We map anything else to -O2.
      OOpt = llvm::StringSwitch<const char *>(A->getValue())
                 .Case("1", "1")
                 .Case("2", "2")
                 .Case("3", "3")
                 .Case("s", "2")
                 .Case("z", "2")
                 .Case("g", "1")
                 .Default("0");
    }
    CmdArgs.push_back(Args.MakeArgString("-O" + OOpt));
  }
}
} // namespace

const char *AMDGCN::OpenMPLinker::constructLLVMLinkCommand(
    const toolchains::AMDGPUOpenMPToolChain &AMDGPUOpenMPTC, Compilation &C,
    const JobAction &JA, const InputInfoList &Inputs, const ArgList &Args,
    StringRef SubArchName, StringRef OutputFilePrefix) const {
  ArgStringList CmdArgs;

  for (const auto &II : Inputs)
    if (II.isFilename())
      CmdArgs.push_back(II.getFilename());

  bool HasLibm = false;
  if (Args.hasArg(options::OPT_l)) {
    auto Lm = Args.getAllArgValues(options::OPT_l);
    for (auto &Lib : Lm) {
      if (Lib == "m") {
        HasLibm = true;
        break;
      }
    }

    if (HasLibm) {
      // This is not certain to work. The device libs added here, and passed to
      // llvm-link, are missing attributes that they expect to be inserted when
      // passed to mlink-builtin-bitcode. The amdgpu backend does not generate
      // conservatively correct code when attributes are missing, so this may
      // be the root cause of miscompilations. Passing via mlink-builtin-bitcode
      // ultimately hits CodeGenModule::addDefaultFunctionDefinitionAttributes
      // on each function, see D28538 for context.
      // Potential workarounds:
      //  - unconditionally link all of the device libs to every translation
      //    unit in clang via mlink-builtin-bitcode
      //  - build a libm bitcode file as part of the DeviceRTL and explictly
      //    mlink-builtin-bitcode the rocm device libs components at build time
      //  - drop this llvm-link fork in favour or some calls into LLVM, chosen
      //    to do basically the same work as llvm-link but with that call first
      //  - write an opt pass that sets that on every function it sees and pipe
      //    the device-libs bitcode through that on the way to this llvm-link
      SmallVector<ToolChain::BitCodeLibraryInfo, 12> BCLibs =
          AMDGPUOpenMPTC.getCommonDeviceLibNames(Args, SubArchName.str(),
                                                 Action::OFK_OpenMP);
      for (auto BCLib : BCLibs)
        CmdArgs.push_back(Args.MakeArgString(BCLib.Path));
    }
  }

  AddStaticDeviceLibsLinking(C, *this, JA, Inputs, Args, CmdArgs, "amdgcn",
                             SubArchName, /*isBitCodeSDL=*/true);
  // Add an intermediate output file.
  CmdArgs.push_back("-o");
  const char *OutputFileName =
      getOutputFileName(C, OutputFilePrefix, "-linked", "bc");
  CmdArgs.push_back(OutputFileName);
  const char *Exec =
      Args.MakeArgString(getToolChain().GetProgramPath("llvm-link"));
  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(), Exec, CmdArgs, Inputs,
      InputInfo(&JA, Args.MakeArgString(OutputFileName))));

  // If we linked in libm definitions late we run another round of optimizations
  // to inline the definitions and fold what is foldable.
  if (HasLibm) {
    ArgStringList OptCmdArgs;
    const char *OptOutputFileName =
        getOutputFileName(C, OutputFilePrefix, "-linked-opt", "bc");
    addLLCOptArg(Args, OptCmdArgs);
    OptCmdArgs.push_back(OutputFileName);
    OptCmdArgs.push_back("-o");
    OptCmdArgs.push_back(OptOutputFileName);
    const char *OptExec =
        Args.MakeArgString(getToolChain().GetProgramPath("opt"));
    C.addCommand(std::make_unique<Command>(
        JA, *this, ResponseFileSupport::AtFileCurCP(), OptExec, OptCmdArgs,
        InputInfo(&JA, Args.MakeArgString(OutputFileName)),
        InputInfo(&JA, Args.MakeArgString(OptOutputFileName))));
    OutputFileName = OptOutputFileName;
  }

  return OutputFileName;
}

const char *AMDGCN::OpenMPLinker::constructLlcCommand(
    Compilation &C, const JobAction &JA, const InputInfoList &Inputs,
    const llvm::opt::ArgList &Args, llvm::StringRef SubArchName,
    llvm::StringRef OutputFilePrefix, const char *InputFileName,
    bool OutputIsAsm) const {
  // Construct llc command.
  ArgStringList LlcArgs;
  // The input to llc is the output from opt.
  LlcArgs.push_back(InputFileName);
  // Pass optimization arg to llc.
  addLLCOptArg(Args, LlcArgs);
  LlcArgs.push_back("-mtriple=amdgcn-amd-amdhsa");
  LlcArgs.push_back(Args.MakeArgString("-mcpu=" + SubArchName));
  LlcArgs.push_back(
      Args.MakeArgString(Twine("-filetype=") + (OutputIsAsm ? "asm" : "obj")));

  for (const Arg *A : Args.filtered(options::OPT_mllvm)) {
    LlcArgs.push_back(A->getValue(0));
  }

  // Add output filename
  LlcArgs.push_back("-o");
  const char *LlcOutputFile =
      getOutputFileName(C, OutputFilePrefix, "", OutputIsAsm ? "s" : "o");
  LlcArgs.push_back(LlcOutputFile);
  const char *Llc = Args.MakeArgString(getToolChain().GetProgramPath("llc"));
  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(), Llc, LlcArgs, Inputs,
      InputInfo(&JA, Args.MakeArgString(LlcOutputFile))));
  return LlcOutputFile;
}

void AMDGCN::OpenMPLinker::constructLldCommand(
    Compilation &C, const JobAction &JA, const InputInfoList &Inputs,
    const InputInfo &Output, const llvm::opt::ArgList &Args,
    const char *InputFileName) const {
  // Construct lld command.
  // The output from ld.lld is an HSA code object file.
  ArgStringList LldArgs{"-flavor",    "gnu", "--no-undefined",
                        "-shared",    "-o",  Output.getFilename(),
                        InputFileName};

  const char *Lld = Args.MakeArgString(getToolChain().GetProgramPath("lld"));
  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(), Lld, LldArgs, Inputs,
      InputInfo(&JA, Args.MakeArgString(Output.getFilename()))));
}

// For amdgcn the inputs of the linker job are device bitcode and output is
// object file. It calls llvm-link, opt, llc, then lld steps.
void AMDGCN::OpenMPLinker::ConstructJob(Compilation &C, const JobAction &JA,
                                        const InputInfo &Output,
                                        const InputInfoList &Inputs,
                                        const ArgList &Args,
                                        const char *LinkingOutput) const {
  const ToolChain &TC = getToolChain();
  assert(getToolChain().getTriple().isAMDGCN() && "Unsupported target");

  const toolchains::AMDGPUOpenMPToolChain &AMDGPUOpenMPTC =
      static_cast<const toolchains::AMDGPUOpenMPToolChain &>(TC);

  StringRef GPUArch = Args.getLastArgValue(options::OPT_march_EQ);
  assert(!GPUArch.empty() && "Must have an explicit GPU arch.");

  // Prefix for temporary file name.
  std::string Prefix;
  for (const auto &II : Inputs)
    if (II.isFilename())
      Prefix =
          llvm::sys::path::stem(II.getFilename()).str() + "-" + GPUArch.str();
  assert(Prefix.length() && "no linker inputs are files ");

  // Each command outputs different files.
  const char *LLVMLinkCommand = constructLLVMLinkCommand(
      AMDGPUOpenMPTC, C, JA, Inputs, Args, GPUArch, Prefix);

  // Produce readable assembly if save-temps is enabled.
  if (C.getDriver().isSaveTempsEnabled())
    constructLlcCommand(C, JA, Inputs, Args, GPUArch, Prefix, LLVMLinkCommand,
                        /*OutputIsAsm=*/true);
  const char *LlcCommand = constructLlcCommand(C, JA, Inputs, Args, GPUArch,
                                               Prefix, LLVMLinkCommand);
  constructLldCommand(C, JA, Inputs, Output, Args, LlcCommand);
}

AMDGPUOpenMPToolChain::AMDGPUOpenMPToolChain(const Driver &D,
                                             const llvm::Triple &Triple,
                                             const ToolChain &HostTC,
                                             const ArgList &Args)
    : ROCMToolChain(D, Triple, Args), HostTC(HostTC) {
  // Lookup binaries into the driver directory, this is used to
  // discover the 'amdgpu-arch' executable.
  getProgramPaths().push_back(getDriver().Dir);
  // Diagnose unsupported sanitizer options only once.
  diagnoseUnsupportedSanitizers(Args);
}

void AMDGPUOpenMPToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {
  HostTC.addClangTargetOptions(DriverArgs, CC1Args, DeviceOffloadingKind);

  assert(DeviceOffloadingKind == Action::OFK_OpenMP &&
         "Only OpenMP offloading kinds are supported.");

  if (!DriverArgs.hasFlag(options::OPT_offloadlib, options::OPT_no_offloadlib,
                          true))
    return;

  for (auto BCFile : getDeviceLibs(DriverArgs, DeviceOffloadingKind)) {
    CC1Args.push_back(BCFile.ShouldInternalize ? "-mlink-builtin-bitcode"
                                               : "-mlink-bitcode-file");
    CC1Args.push_back(DriverArgs.MakeArgString(BCFile.Path));
  }

  // Link the bitcode library late if we're using device LTO.
  if (getDriver().isUsingOffloadLTO())
    return;
}

llvm::opt::DerivedArgList *AMDGPUOpenMPToolChain::TranslateArgs(
    const llvm::opt::DerivedArgList &Args, StringRef BoundArch,
    Action::OffloadKind DeviceOffloadKind) const {
  DerivedArgList *DAL =
      HostTC.TranslateArgs(Args, BoundArch, DeviceOffloadKind);

  if (!DAL)
    DAL = new DerivedArgList(Args.getBaseArgs());

  const OptTable &Opts = getDriver().getOpts();

  // Skip sanitize options passed from the HostTC. Claim them early.
  // The decision to sanitize device code is computed only by
  // 'shouldSkipSanitizeOption'.
  if (DAL->hasArg(options::OPT_fsanitize_EQ))
    DAL->claimAllArgs(options::OPT_fsanitize_EQ);

  for (Arg *A : Args)
    if (!shouldSkipSanitizeOption(*this, Args, BoundArch, A) &&
        !llvm::is_contained(*DAL, A))
      DAL->append(A);

  if (!BoundArch.empty()) {
    DAL->eraseArg(options::OPT_march_EQ);
    DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_march_EQ),
                      BoundArch);
  }

  return DAL;
}

Tool *AMDGPUOpenMPToolChain::buildLinker() const {
  assert(getTriple().isAMDGCN());
  return new tools::AMDGCN::OpenMPLinker(*this);
}

void AMDGPUOpenMPToolChain::addClangWarningOptions(
    ArgStringList &CC1Args) const {
  AMDGPUToolChain::addClangWarningOptions(CC1Args);
  HostTC.addClangWarningOptions(CC1Args);
}

ToolChain::CXXStdlibType
AMDGPUOpenMPToolChain::GetCXXStdlibType(const ArgList &Args) const {
  return HostTC.GetCXXStdlibType(Args);
}

void AMDGPUOpenMPToolChain::AddClangCXXStdlibIncludeArgs(
    const llvm::opt::ArgList &Args, llvm::opt::ArgStringList &CC1Args) const {
  HostTC.AddClangCXXStdlibIncludeArgs(Args, CC1Args);
}

void AMDGPUOpenMPToolChain::AddClangSystemIncludeArgs(
    const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  HostTC.AddClangSystemIncludeArgs(DriverArgs, CC1Args);
}

void AMDGPUOpenMPToolChain::AddIAMCUIncludeArgs(const ArgList &Args,
                                                ArgStringList &CC1Args) const {
  HostTC.AddIAMCUIncludeArgs(Args, CC1Args);
}

SanitizerMask AMDGPUOpenMPToolChain::getSupportedSanitizers() const {
  // The AMDGPUOpenMPToolChain only supports sanitizers in the sense that it
  // allows sanitizer arguments on the command line if they are supported by the
  // host toolchain. The AMDGPUOpenMPToolChain will actually ignore any command
  // line arguments for any of these "supported" sanitizers. That means that no
  // sanitization of device code is actually supported at this time.
  //
  // This behavior is necessary because the host and device toolchains
  // invocations often share the command line, so the device toolchain must
  // tolerate flags meant only for the host toolchain.
  return HostTC.getSupportedSanitizers();
}

VersionTuple
AMDGPUOpenMPToolChain::computeMSVCVersion(const Driver *D,
                                          const ArgList &Args) const {
  return HostTC.computeMSVCVersion(D, Args);
}

llvm::SmallVector<ToolChain::BitCodeLibraryInfo, 12>
AMDGPUOpenMPToolChain::getDeviceLibs(
    const llvm::opt::ArgList &Args,
    const Action::OffloadKind DeviceOffloadingKind) const {
  if (!Args.hasFlag(options::OPT_offloadlib, options::OPT_no_offloadlib, true))
    return {};

  StringRef GpuArch = getProcessorFromTargetID(
      getTriple(), Args.getLastArgValue(options::OPT_march_EQ));

  SmallVector<BitCodeLibraryInfo, 12> BCLibs;
  for (auto BCLib :
       getCommonDeviceLibNames(Args, GpuArch.str(), DeviceOffloadingKind,
                               /*IsOpenMP=*/true))
    BCLibs.emplace_back(BCLib);

  return BCLibs;
}
