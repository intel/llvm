//===--- SYCL.cpp - SYCL Tool and ToolChain Implementations -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SYCL.h"
#include "CommonArgs.h"
#include "InputInfo.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

const char *SYCL::Linker::constructLLVMSpirvCommand(Compilation &C,
    const JobAction &JA, const InputInfo &Output, StringRef OutputFilePrefix,
    bool ToBc, const char *InputFileName) const {
  // Construct llvm-spirv command.
  // The output is a bc file or vice versa depending on the -r option usage
  // llvm-spirv -r -o a_kernel.bc a_kernel.spv
  // llvm-spirv -o a_kernel.spv a_kernel.bc
  ArgStringList CmdArgs;
  const char *OutputFileName = nullptr;
  if (ToBc) {
    std::string TmpName =
      C.getDriver().GetTemporaryPath(OutputFilePrefix.str() + "-spirv", "bc");
    OutputFileName = C.addTempFile(C.getArgs().MakeArgString(TmpName));
    CmdArgs.push_back("-r");
    CmdArgs.push_back("-o");
    CmdArgs.push_back(OutputFileName);
  } else {
    CmdArgs.push_back("-spirv-max-version=1.1");
    CmdArgs.push_back("-spirv-ext=+all");
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  }
  CmdArgs.push_back(InputFileName);

  SmallString<128> LLVMSpirvPath(C.getDriver().Dir);
  llvm::sys::path::append(LLVMSpirvPath, "llvm-spirv");
  const char *LLVMSpirv = C.getArgs().MakeArgString(LLVMSpirvPath);
  C.addCommand(std::make_unique<Command>(JA, *this, LLVMSpirv, CmdArgs, None));
  return OutputFileName;
}

void SYCL::constructLLVMForeachCommand(Compilation &C, const JobAction &JA,
                                       std::unique_ptr<Command> InputCommand,
                                       const InputInfoList &InputFiles,
                                       const InputInfo &Output, const Tool *T,
                                       StringRef Ext = "out") {
  // Construct llvm-foreach command.
  // The llvm-foreach command looks like this:
  // llvm-foreach --in-file-list=a.list --in-replace='{}' -- echo '{}'
  ArgStringList ForeachArgs;
  std::string OutputFileName(Output.getFilename());
  ForeachArgs.push_back(C.getArgs().MakeArgString("--out-ext=" + Ext));
  for (auto &I : InputFiles) {
    std::string Filename(I.getFilename());
    ForeachArgs.push_back(
        C.getArgs().MakeArgString("--in-file-list=" + Filename));
    ForeachArgs.push_back(
        C.getArgs().MakeArgString("--in-replace=" + Filename));
  }

  ForeachArgs.push_back(
      C.getArgs().MakeArgString("--out-file-list=" + OutputFileName));
  ForeachArgs.push_back(
      C.getArgs().MakeArgString("--out-replace=" + OutputFileName));
  ForeachArgs.push_back(C.getArgs().MakeArgString("--"));
  ForeachArgs.push_back(
      C.getArgs().MakeArgString(InputCommand->getExecutable()));

  for (auto &Arg : InputCommand->getArguments())
    ForeachArgs.push_back(Arg);

  SmallString<128> ForeachPath(C.getDriver().Dir);
  llvm::sys::path::append(ForeachPath, "llvm-foreach");
  const char *Foreach = C.getArgs().MakeArgString(ForeachPath);
  C.addCommand(std::make_unique<Command>(JA, *T, Foreach, ForeachArgs, None));
}

const char *SYCL::Linker::constructLLVMLinkCommand(Compilation &C,
    const JobAction &JA, const InputInfo &Output, const ArgList &Args,
    StringRef SubArchName, StringRef OutputFilePrefix,
    const InputInfoList &InputFiles) const {
  ArgStringList CmdArgs;
  // Add the input bc's created by compile step.
  // When offloading, the input file(s) could be from unbundled partially
  // linked archives.  The unbundled information is a list of files and not
  // an actual object/archive.  Take that list and pass those to the linker
  // instead of the original object.
  if (JA.isDeviceOffloading(Action::OFK_SYCL)) {
    // Go through the Inputs to the link.  When a listfile is encountered, we
    // know it is an unbundled generated list.
    for (const auto &II : InputFiles) {
      if (II.getType() == types::TY_Tempfilelist) {
        // Pass the unbundled list with '@' to be processed.
        std::string FileName(II.getFilename());
        CmdArgs.push_back(C.getArgs().MakeArgString("@" + FileName));
      } else
        CmdArgs.push_back(II.getFilename());
    }
  } else
    for (const auto &II : InputFiles)
      CmdArgs.push_back(II.getFilename());

  // Add an intermediate output file.
  CmdArgs.push_back("-o");
  const char *OutputFileName = Output.getFilename();
  CmdArgs.push_back(OutputFileName);
  // TODO: temporary workaround for a problem with warnings reported by
  // llvm-link when driver links LLVM modules with empty modules
  CmdArgs.push_back("--suppress-warnings");
  SmallString<128> ExecPath(C.getDriver().Dir);
  llvm::sys::path::append(ExecPath, "llvm-link");
  const char *Exec = C.getArgs().MakeArgString(ExecPath);
  C.addCommand(std::make_unique<Command>(JA, *this, Exec, CmdArgs, None));
  return OutputFileName;
}

void SYCL::Linker::constructLlcCommand(Compilation &C, const JobAction &JA,
    const InputInfo &Output, const char *InputFileName) const {
  // Construct llc command.
  // The output is an object file.
  ArgStringList LlcArgs{"-filetype=obj", "-o",  Output.getFilename(),
                        InputFileName};
  SmallString<128> LlcPath(C.getDriver().Dir);
  llvm::sys::path::append(LlcPath, "llc");
  const char *Llc = C.getArgs().MakeArgString(LlcPath);
  C.addCommand(std::make_unique<Command>(JA, *this, Llc, LlcArgs, None));
}

// For SYCL the inputs of the linker job are SPIR-V binaries and output is
// a single SPIR-V binary.  Input can also be bitcode when specified by
// the user.
void SYCL::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {

  assert((getToolChain().getTriple().isSPIR() || getToolChain().getTriple().isNVPTX()) &&
         "Unsupported target");

  std::string SubArchName =
      std::string(getToolChain().getTriple().getArchName());

  // Prefix for temporary file name.
  std::string Prefix = std::string(llvm::sys::path::stem(SubArchName));

  // For CUDA, we want to link all BC files before resuming the normal
  // compilation path
  if (getToolChain().getTriple().isNVPTX()) {
    InputInfoList NvptxInputs;
    for (const auto &II : Inputs) {
      if (!II.isFilename())
        continue;
      NvptxInputs.push_back(II);
    }

    constructLLVMLinkCommand(C, JA, Output, Args, SubArchName, Prefix,
                             NvptxInputs);
    return;
  }

  // We want to use llvm-spirv linker to link spirv binaries before putting
  // them into the fat object.
  // Each command outputs different files.
  InputInfoList SpirvInputs;
  for (const auto &II : Inputs) {
    if (!II.isFilename())
      continue;
    if (Args.hasFlag(options::OPT_fsycl_use_bitcode,
                     options::OPT_fno_sycl_use_bitcode, true) ||
        Args.hasArg(options::OPT_foffload_static_lib_EQ))
      SpirvInputs.push_back(II);
    else {
      const char *LLVMSpirvOutputFile =
        constructLLVMSpirvCommand(C, JA, Output, Prefix, true,
                                  II.getFilename());
      SpirvInputs.push_back(InputInfo(types::TY_LLVM_BC, LLVMSpirvOutputFile,
                                      LLVMSpirvOutputFile));
    }
  }

  constructLLVMLinkCommand(C, JA, Output, Args, SubArchName, Prefix,
                           SpirvInputs);
}

void SYCL::fpga::BackendCompiler::ConstructJob(Compilation &C,
                                         const JobAction &JA,
                                         const InputInfo &Output,
                                         const InputInfoList &Inputs,
                                         const ArgList &Args,
                                         const char *LinkingOutput) const {
  assert((getToolChain().getTriple().getArch() == llvm::Triple::spir ||
          getToolChain().getTriple().getArch() == llvm::Triple::spir64) &&
         "Unsupported target");

  InputInfoList ForeachInputs;
  ArgStringList CmdArgs{"-o",  Output.getFilename()};
  for (const auto &II : Inputs) {
    std::string Filename(II.getFilename());
    if (II.getType() == types::TY_Tempfilelist)
      ForeachInputs.push_back(II);
    if (II.getType() == types::TY_TempAOCOfilelist)
      // Add any FPGA library lists.  These come in as special tempfile lists.
      CmdArgs.push_back(Args.MakeArgString(Twine("-library-list=") +
          Filename));
    else
      CmdArgs.push_back(C.getArgs().MakeArgString(Filename));
  }
  CmdArgs.push_back("-sycl");

  StringRef ForeachExt = "aocx";
  if (Arg *A = Args.getLastArg(options::OPT_fsycl_link_EQ))
    if (A->getValue() == StringRef("early")) {
      CmdArgs.push_back("-rtl");
      ForeachExt = "aocr";
    }


  InputInfoList FPGADepFiles;
  for (auto *A : Args) {
    // Any input file is assumed to have a dependency file associated
    if (A->getOption().getKind() == Option::InputClass) {
      SmallString<128> FN(A->getSpelling());
      StringRef Ext(llvm::sys::path::extension(FN));
      if (!Ext.empty()) {
        types::ID Ty = getToolChain().LookupTypeForExtension(Ext.drop_front());
        if (Ty == types::TY_INVALID)
          continue;
        if (types::isSrcFile(Ty) || Ty == types::TY_Object) {
          llvm::sys::path::replace_extension(FN, "d");
          FPGADepFiles.push_back(InputInfo(types::TY_Dependencies,
              Args.MakeArgString(FN), Args.MakeArgString(FN)));
        }
      }
    }
  }

  // Add any dependency files.
  if (!FPGADepFiles.empty()) {
    SmallString<128> DepOpt("-dep-files=");
    for (unsigned I = 0; I < FPGADepFiles.size(); ++I) {
      if (I)
        DepOpt += ',';
      DepOpt += FPGADepFiles[I].getFilename();
    }
    CmdArgs.push_back(C.getArgs().MakeArgString(DepOpt));
  }

  // Depending on output file designations, set the report folder
  SmallString<128> ReportOptArg;
  if (Arg *FinalOutput = Args.getLastArg(options::OPT_o, options::OPT__SLASH_o,
        options::OPT__SLASH_Fe)) {
    SmallString<128> FN(FinalOutput->getValue());
    llvm::sys::path::replace_extension(FN, "prj");
    const char * FolderName = Args.MakeArgString(FN);
    ReportOptArg += FolderName;
  } else {
    // Output directory is based off of the first object name
    for (Arg * Cur : Args) {
      SmallString<128> AN = Cur->getSpelling();
      StringRef Ext(llvm::sys::path::extension(AN));
      if (!Ext.empty()) {
        types::ID Ty = getToolChain().LookupTypeForExtension(Ext.drop_front());
        if (Ty == types::TY_INVALID)
          continue;
        if (types::isSrcFile(Ty) || Ty == types::TY_Object) {
          llvm::sys::path::replace_extension(AN, "prj");
          ReportOptArg += Args.MakeArgString(AN);
          break;
        }
      }
    }
  }
  if (!ReportOptArg.empty())
    CmdArgs.push_back(C.getArgs().MakeArgString(
        Twine("-output-report-folder=") + ReportOptArg));
  // Add -Xsycl-target* options.
  const toolchains::SYCLToolChain &TC =
      static_cast<const toolchains::SYCLToolChain &>(getToolChain());
  TC.TranslateBackendTargetArgs(Args, CmdArgs);
  TC.TranslateLinkerTargetArgs(Args, CmdArgs);
  // Look for -reuse-exe=XX option
  if (Arg *A = Args.getLastArg(options::OPT_reuse_exe_EQ)) {
    Args.ClaimAllArgs(options::OPT_reuse_exe_EQ);
    CmdArgs.push_back(Args.MakeArgString(A->getAsString(Args)));
  }

  SmallString<128> ExecPath(getToolChain().GetProgramPath("aoc"));
  const char *Exec = C.getArgs().MakeArgString(ExecPath);
  auto Cmd = std::make_unique<Command>(JA, *this, Exec, CmdArgs, None);
  if (!ForeachInputs.empty())
    constructLLVMForeachCommand(C, JA, std::move(Cmd), ForeachInputs, Output,
                                this, ForeachExt);
  else
    C.addCommand(std::move(Cmd));
}

void SYCL::gen::BackendCompiler::ConstructJob(Compilation &C,
                                         const JobAction &JA,
                                         const InputInfo &Output,
                                         const InputInfoList &Inputs,
                                         const ArgList &Args,
                                         const char *LinkingOutput) const {
  assert((getToolChain().getTriple().getArch() == llvm::Triple::spir ||
          getToolChain().getTriple().getArch() == llvm::Triple::spir64) &&
         "Unsupported target");
  ArgStringList CmdArgs{"-output",  Output.getFilename()};
  InputInfoList ForeachInputs;
  for (const auto &II : Inputs) {
    CmdArgs.push_back("-file");
    std::string Filename(II.getFilename());
    if (II.getType() == types::TY_Tempfilelist)
      ForeachInputs.push_back(II);
    CmdArgs.push_back(C.getArgs().MakeArgString(Filename));
  }
  // The next line prevents ocloc from modifying the image name
  CmdArgs.push_back("-output_no_suffix");
  CmdArgs.push_back("-spirv_input");
  // Add -Xsycl-target* options.
  const toolchains::SYCLToolChain &TC =
      static_cast<const toolchains::SYCLToolChain &>(getToolChain());
  TC.TranslateBackendTargetArgs(Args, CmdArgs);
  TC.TranslateLinkerTargetArgs(Args, CmdArgs);
  SmallString<128> ExecPath(getToolChain().GetProgramPath("ocloc"));
  const char *Exec = C.getArgs().MakeArgString(ExecPath);
  auto Cmd = std::make_unique<Command>(JA, *this, Exec, CmdArgs, None);
  if (!ForeachInputs.empty())
    constructLLVMForeachCommand(C, JA, std::move(Cmd), ForeachInputs, Output,
                                this);
  else
    C.addCommand(std::move(Cmd));
}

void SYCL::x86_64::BackendCompiler::ConstructJob(Compilation &C,
                                         const JobAction &JA,
                                         const InputInfo &Output,
                                         const InputInfoList &Inputs,
                                         const ArgList &Args,
                                         const char *LinkingOutput) const {
  ArgStringList CmdArgs;
  CmdArgs.push_back(Args.MakeArgString(Twine("-o=") + Output.getFilename()));
  CmdArgs.push_back("--device=cpu");
  InputInfoList ForeachInputs;
  for (const auto &II : Inputs) {
    std::string Filename(II.getFilename());
    if (II.getType() == types::TY_Tempfilelist)
      ForeachInputs.push_back(II);
    CmdArgs.push_back(Args.MakeArgString(Filename));
  }
  // Add -Xsycl-target* options.
  const toolchains::SYCLToolChain &TC =
      static_cast<const toolchains::SYCLToolChain &>(getToolChain());
  TC.TranslateBackendTargetArgs(Args, CmdArgs);
  TC.TranslateLinkerTargetArgs(Args, CmdArgs);
  SmallString<128> ExecPath(getToolChain().GetProgramPath("opencl-aot"));
  const char *Exec = C.getArgs().MakeArgString(ExecPath);
  auto Cmd = std::make_unique<Command>(JA, *this, Exec, CmdArgs, None);
  if (!ForeachInputs.empty())
    constructLLVMForeachCommand(C, JA, std::move(Cmd), ForeachInputs, Output,
                                this);
  else
    C.addCommand(std::move(Cmd));
}

SYCLToolChain::SYCLToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ToolChain &HostTC, const ArgList &Args)
    : ToolChain(D, Triple, Args), HostTC(HostTC) {
  // Lookup binaries into the driver directory, this is used to
  // discover the clang-offload-bundler executable.
  getProgramPaths().push_back(getDriver().Dir);
}

void SYCLToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadingKind) const {
  HostTC.addClangTargetOptions(DriverArgs, CC1Args, DeviceOffloadingKind);
}

llvm::opt::DerivedArgList *
SYCLToolChain::TranslateArgs(const llvm::opt::DerivedArgList &Args,
                             StringRef BoundArch,
                             Action::OffloadKind DeviceOffloadKind) const {
  DerivedArgList *DAL =
      HostTC.TranslateArgs(Args, BoundArch, DeviceOffloadKind);

  if (!DAL) {
    DAL = new DerivedArgList(Args.getBaseArgs());
    for (Arg *A : Args)
      DAL->append(A);
  }

  const OptTable &Opts = getDriver().getOpts();
  if (!BoundArch.empty()) {
    DAL->eraseArg(options::OPT_march_EQ);
    DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_march_EQ),
                      BoundArch);
  }
  return DAL;
}

// Expects a specific type of option (e.g. -Xsycl-target-backend) and will
// extract the arguments.
void SYCLToolChain::TranslateTargetOpt(const llvm::opt::ArgList &Args,
    llvm::opt::ArgStringList &CmdArgs, OptSpecifier Opt,
    OptSpecifier Opt_EQ) const {
  for (auto *A : Args) {
    bool OptNoTriple;
    OptNoTriple = A->getOption().matches(Opt);
    if (A->getOption().matches(Opt_EQ)) {
      // Passing device args: -X<Opt>=<triple> -opt=val.
      if (A->getValue() != getTripleString())
        // Provided triple does not match current tool chain.
        continue;
    } else if (!OptNoTriple)
      // Don't worry about any of the other args, we only want to pass what is
      // passed in -X<Opt>
      continue;

    // Add the argument from -X<Opt>
    StringRef ArgString;
    if (OptNoTriple) {
      // With multiple -fsycl-targets, a triple is required so we know where
      // the options should go.
      if (Args.getAllArgValues(options::OPT_fsycl_targets_EQ).size() != 1) {
        getDriver().Diag(diag::err_drv_Xsycl_target_missing_triple)
            << A->getSpelling();
        continue;
      }
      // No triple, so just add the argument.
      ArgString = A->getValue();
    } else
      // Triple found, add the next argument in line.
      ArgString = A->getValue(1);

    // Do a simple parse of the args to pass back
    SmallVector<StringRef, 16> TargetArgs;
    // TODO: Improve parsing, as this only handles arguments separated by
    // spaces.
    ArgString.split(TargetArgs, ' ', -1, false);
    for (const auto &TA : TargetArgs)
      CmdArgs.push_back(Args.MakeArgString(TA));
    A->claim();
  }
}

void SYCLToolChain::TranslateBackendTargetArgs(const llvm::opt::ArgList &Args,
    llvm::opt::ArgStringList &CmdArgs) const {
  // Handle -Xs flags.
  for (auto *A : Args) {
    // When parsing the target args, the -Xs<opt> type option applies to all
    // target compilations is not associated with a specific triple.  The
    // option can be used in 3 different ways:
    //   -Xs -DFOO -Xs -DBAR
    //   -Xs "-DFOO -DBAR"
    //   -XsDFOO -XsDBAR
    // All of the above examples will pass -DFOO -DBAR to the backend compiler.
    if (A->getOption().matches(options::OPT_Xs)) {
      // Take the arg and create an option out of it.
      CmdArgs.push_back(Args.MakeArgString(Twine("-") + A->getValue()));
      A->claim();
      continue;
    }
    if (A->getOption().matches(options::OPT_Xs_separate)) {
      StringRef ArgString(A->getValue());
      // Do a simple parse of the args to pass back
      SmallVector<StringRef, 16> TargetArgs;
      // TODO: Improve parsing, as this only handles arguments separated by
      // spaces.
      ArgString.split(TargetArgs, ' ', -1, false);
      for (const auto &TA : TargetArgs)
        CmdArgs.push_back(Args.MakeArgString(TA));
      A->claim();
      continue;
    }
  }
  // Handle -Xsycl-target-backend.
  TranslateTargetOpt(Args, CmdArgs, options::OPT_Xsycl_backend,
      options::OPT_Xsycl_backend_EQ);
}

void SYCLToolChain::TranslateLinkerTargetArgs(const llvm::opt::ArgList &Args,
    llvm::opt::ArgStringList &CmdArgs) const {
  // Handle -Xsycl-target-linker.
  TranslateTargetOpt(Args, CmdArgs, options::OPT_Xsycl_linker,
      options::OPT_Xsycl_linker_EQ);
}

Tool *SYCLToolChain::buildBackendCompiler() const {
  if (getTriple().getSubArch() == llvm::Triple::SPIRSubArch_fpga)
    return new tools::SYCL::fpga::BackendCompiler(*this);
  if (getTriple().getSubArch() == llvm::Triple::SPIRSubArch_gen)
    return new tools::SYCL::gen::BackendCompiler(*this);
  // fall through is CPU.
  return new tools::SYCL::x86_64::BackendCompiler(*this);
}

Tool *SYCLToolChain::buildLinker() const {
  assert(getTriple().getArch() == llvm::Triple::spir ||
         getTriple().getArch() == llvm::Triple::spir64);
  return new tools::SYCL::Linker(*this);
}

void SYCLToolChain::addClangWarningOptions(ArgStringList &CC1Args) const {
  HostTC.addClangWarningOptions(CC1Args);
}

ToolChain::CXXStdlibType
SYCLToolChain::GetCXXStdlibType(const ArgList &Args) const {
  return HostTC.GetCXXStdlibType(Args);
}

void SYCLToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                              ArgStringList &CC1Args) const {
  HostTC.AddClangSystemIncludeArgs(DriverArgs, CC1Args);
}

void SYCLToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &Args,
                                                 ArgStringList &CC1Args) const {
  HostTC.AddClangCXXStdlibIncludeArgs(Args, CC1Args);
}
