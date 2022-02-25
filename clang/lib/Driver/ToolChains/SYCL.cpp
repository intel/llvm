//===--- SYCL.cpp - SYCL Tool and ToolChain Implementations -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "SYCL.h"
#include "CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

SYCLInstallationDetector::SYCLInstallationDetector(const Driver &D)
    : D(D), InstallationCandidates() {
  InstallationCandidates.emplace_back(D.Dir + "/..");
}

void SYCLInstallationDetector::getSYCLDeviceLibPath(
    llvm::SmallVector<llvm::SmallString<128>, 4> &DeviceLibPaths) const {
  for (const auto &IC : InstallationCandidates) {
    llvm::SmallString<128> InstallLibPath(IC.str());
    InstallLibPath.append("/lib");
    DeviceLibPaths.emplace_back(InstallLibPath);
  }

  DeviceLibPaths.emplace_back(D.SysRoot + "/lib");
}

void SYCLInstallationDetector::print(llvm::raw_ostream &OS) const {
  if (!InstallationCandidates.size())
    return;
  OS << "SYCL Installation Candidates: \n";
  for (const auto &IC : InstallationCandidates) {
    OS << IC << "\n";
  }
}

static void addFPGATimingDiagnostic(std::unique_ptr<Command> &Cmd,
                                    Compilation &C) {
  const char *Msg = C.getArgs().MakeArgString(
      "The FPGA image generated during this compile contains timing violations "
      "and may produce functional errors if used. Refer to the Intel oneAPI "
      "DPC++ FPGA Optimization Guide section on Timing Failures for more "
      "information.");
  Cmd->addDiagForErrorCode(/*ErrorCode*/ 42, Msg);
  Cmd->addExitForErrorCode(/*ErrorCode*/ 42, false);
}

void SYCL::constructLLVMForeachCommand(Compilation &C, const JobAction &JA,
                                       std::unique_ptr<Command> InputCommand,
                                       const InputInfoList &InputFiles,
                                       const InputInfo &Output, const Tool *T,
                                       StringRef Increment, StringRef Ext,
                                       StringRef ParallelJobs) {
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
  if (!Increment.empty())
    ForeachArgs.push_back(
        C.getArgs().MakeArgString("--out-increment=" + Increment));
  if (!ParallelJobs.empty())
    ForeachArgs.push_back(C.getArgs().MakeArgString("--jobs=" + ParallelJobs));

  if (C.getDriver().isSaveTempsEnabled()) {
    SmallString<128> OutputDirName;
    if (C.getDriver().isSaveTempsObj()) {
      OutputDirName =
          T->getToolChain().GetFilePath(OutputFileName.c_str()).c_str();
      llvm::sys::path::remove_filename(OutputDirName);
    }
    // Use the current dir if the `GetFilePath` returned en empty string, which
    // is the case when the `OutputFileName` does not contain any directory
    // information, or if in CWD mode. This is necessary for `llvm-foreach`, as
    // it would disregard the parameter without it. Otherwise append separator.
    if (OutputDirName.empty())
      llvm::sys::path::native(OutputDirName = "./");
    else
      OutputDirName.append(llvm::sys::path::get_separator());
    ForeachArgs.push_back(
        C.getArgs().MakeArgString("--out-dir=" + OutputDirName));
  }

  ForeachArgs.push_back(C.getArgs().MakeArgString("--"));
  ForeachArgs.push_back(
      C.getArgs().MakeArgString(InputCommand->getExecutable()));

  for (auto &Arg : InputCommand->getArguments())
    ForeachArgs.push_back(Arg);

  SmallString<128> ForeachPath(C.getDriver().Dir);
  llvm::sys::path::append(ForeachPath, "llvm-foreach");
  const char *Foreach = C.getArgs().MakeArgString(ForeachPath);

  auto Cmd = std::make_unique<Command>(JA, *T, ResponseFileSupport::None(),
                                       Foreach, ForeachArgs, None);
  // FIXME: Add the FPGA specific timing diagnostic to the foreach call.
  // The foreach call obscures the return codes from the tool it is calling
  // to the compiler itself.
  addFPGATimingDiagnostic(Cmd, C);
  C.addCommand(std::move(Cmd));
}

// The list should match pre-built SYCL device library files located in
// compiler package. Once we add or remove any SYCL device library files,
// the list should be updated accordingly.
static llvm::SmallVector<StringRef, 16> SYCLDeviceLibList{
    "crt",
    "cmath",
    "cmath-fp64",
    "complex",
    "complex-fp64",
    "itt-compiler-wrappers",
    "itt-stubs",
    "itt-user-wrappers",
    "fallback-cassert",
    "fallback-cstring",
    "fallback-cmath",
    "fallback-cmath-fp64",
    "fallback-complex",
    "fallback-complex-fp64"};

const char *SYCL::Linker::constructLLVMLinkCommand(
    Compilation &C, const JobAction &JA, const InputInfo &Output,
    const ArgList &Args, StringRef SubArchName, StringRef OutputFilePrefix,
    const InputInfoList &InputFiles) const {
  // Split inputs into libraries which have 'archive' type and other inputs
  // which can be either objects or list files. Object files are linked together
  // in a usual way, but the libraries/list files need to be linked differently.
  // We need to fetch only required symbols from the libraries. With the current
  // llvm-link command line interface that can be achieved with two step
  // linking: at the first step we will link objects into an intermediate
  // partially linked image which on the second step will be linked with the
  // libraries with --only-needed option.
  ArgStringList Opts;
  ArgStringList Objs;
  ArgStringList Libs;
  // Add the input bc's created by compile step.
  // When offloading, the input file(s) could be from unbundled partially
  // linked archives.  The unbundled information is a list of files and not
  // an actual object/archive.  Take that list and pass those to the linker
  // instead of the original object.
  if (JA.isDeviceOffloading(Action::OFK_SYCL)) {
    auto isSYCLDeviceLib = [&C](const InputInfo &II) {
      const ToolChain *HostTC = C.getSingleOffloadToolChain<Action::OFK_Host>();
      StringRef LibPostfix = ".o";
      if (HostTC->getTriple().isWindowsMSVCEnvironment() &&
          C.getDriver().IsCLMode())
        LibPostfix = ".obj";
      StringRef InputFilename =
          llvm::sys::path::filename(StringRef(II.getFilename()));
      StringRef LibSyclPrefix("libsycl-");
      if (!InputFilename.startswith(LibSyclPrefix) ||
          !InputFilename.endswith(LibPostfix) || (InputFilename.count('-') < 2))
        return false;
      // Skip the prefix "libsycl-"
      StringRef PureLibName = InputFilename.substr(LibSyclPrefix.size());
      for (const auto &L : SYCLDeviceLibList) {
        if (PureLibName.startswith(L))
          return true;
      }
      return false;
    };
    size_t InputFileNum = InputFiles.size();
    bool LinkSYCLDeviceLibs = (InputFileNum >= 2);
    LinkSYCLDeviceLibs = LinkSYCLDeviceLibs && !isSYCLDeviceLib(InputFiles[0]);
    for (size_t Idx = 1; Idx < InputFileNum; ++Idx)
      LinkSYCLDeviceLibs =
          LinkSYCLDeviceLibs && isSYCLDeviceLib(InputFiles[Idx]);
    // Go through the Inputs to the link.  When a listfile is encountered, we
    // know it is an unbundled generated list.
    if (LinkSYCLDeviceLibs)
      Opts.push_back("-only-needed");
    for (const auto &II : InputFiles) {
      if (II.getType() == types::TY_Tempfilelist) {
        // Pass the unbundled list with '@' to be processed.
        std::string FileName(II.getFilename());
        Libs.push_back(C.getArgs().MakeArgString("@" + FileName));
      } else if (II.getType() == types::TY_Archive && !LinkSYCLDeviceLibs) {
        Libs.push_back(II.getFilename());
      } else
        Objs.push_back(II.getFilename());
    }
  } else
    for (const auto &II : InputFiles)
      Objs.push_back(II.getFilename());

  // Get llvm-link path.
  SmallString<128> ExecPath(C.getDriver().Dir);
  llvm::sys::path::append(ExecPath, "llvm-link");
  const char *Exec = C.getArgs().MakeArgString(ExecPath);

  auto AddLinkCommand = [this, &C, &JA, Exec](const char *Output,
                                              const ArgStringList &Inputs,
                                              const ArgStringList &Options) {
    ArgStringList CmdArgs;
    llvm::copy(Options, std::back_inserter(CmdArgs));
    llvm::copy(Inputs, std::back_inserter(CmdArgs));
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output);
    // TODO: temporary workaround for a problem with warnings reported by
    // llvm-link when driver links LLVM modules with empty modules
    CmdArgs.push_back("--suppress-warnings");
    C.addCommand(std::make_unique<Command>(
        JA, *this, ResponseFileSupport::AtFileUTF8(), Exec, CmdArgs, None));
  };

  // Add an intermediate output file.
  const char *OutputFileName = Output.getFilename();

  if (Libs.empty())
    AddLinkCommand(OutputFileName, Objs, Opts);
  else {
    assert(Opts.empty() && "unexpected options");

    // Linker will be invoked twice if inputs contain libraries. First time we
    // will link input objects into an intermediate temporary file, and on the
    // second invocation intermediate temporary object will be linked with the
    // libraries, but now only required symbols will be added to the final
    // output.
    std::string TempFile =
        C.getDriver().GetTemporaryPath(OutputFilePrefix.str() + "-link", "bc");
    const char *LinkOutput = C.addTempFile(C.getArgs().MakeArgString(TempFile));
    AddLinkCommand(LinkOutput, Objs, {});

    // Now invoke linker for the second time to link required symbols from the
    // input libraries.
    ArgStringList LinkInputs{LinkOutput};
    llvm::copy(Libs, std::back_inserter(LinkInputs));
    AddLinkCommand(OutputFileName, LinkInputs, {"--only-needed"});
  }
  return OutputFileName;
}

void SYCL::Linker::constructLlcCommand(Compilation &C, const JobAction &JA,
                                       const InputInfo &Output,
                                       const char *InputFileName) const {
  // Construct llc command.
  // The output is an object file.
  ArgStringList LlcArgs{"-filetype=obj", "-o", Output.getFilename(),
                        InputFileName};
  SmallString<128> LlcPath(C.getDriver().Dir);
  llvm::sys::path::append(LlcPath, "llc");
  const char *Llc = C.getArgs().MakeArgString(LlcPath);
  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileUTF8(), Llc, LlcArgs, None));
}

// For SYCL the inputs of the linker job are SPIR-V binaries and output is
// a single SPIR-V binary.  Input can also be bitcode when specified by
// the user.
void SYCL::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                const InputInfo &Output,
                                const InputInfoList &Inputs,
                                const ArgList &Args,
                                const char *LinkingOutput) const {

  assert((getToolChain().getTriple().isSPIR() ||
          getToolChain().getTriple().isNVPTX() ||
          getToolChain().getTriple().isAMDGCN()) &&
         "Unsupported target");

  std::string SubArchName =
      std::string(getToolChain().getTriple().getArchName());

  // Prefix for temporary file name.
  std::string Prefix = std::string(llvm::sys::path::stem(SubArchName));

  // For CUDA, we want to link all BC files before resuming the normal
  // compilation path
  if (getToolChain().getTriple().isNVPTX() ||
      getToolChain().getTriple().isAMDGCN()) {
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

  InputInfoList SpirvInputs;
  for (const auto &II : Inputs) {
    if (!II.isFilename())
      continue;
    SpirvInputs.push_back(II);
  }

  constructLLVMLinkCommand(C, JA, Output, Args, SubArchName, Prefix,
                           SpirvInputs);
}

static const char *makeExeName(Compilation &C, StringRef Name) {
  llvm::SmallString<8> ExeName(Name);
  const ToolChain *HostTC = C.getSingleOffloadToolChain<Action::OFK_Host>();
  if (HostTC->getTriple().isWindowsMSVCEnvironment())
    ExeName.append(".exe");
  return C.getArgs().MakeArgString(ExeName);
}

void SYCL::fpga::BackendCompiler::constructOpenCLAOTCommand(
    Compilation &C, const JobAction &JA, const InputInfo &Output,
    const InputInfoList &Inputs, const ArgList &Args) const {
  // Construct opencl-aot command. This is used for FPGA AOT compilations
  // when performing emulation.  Input file will be a SPIR-V binary which
  // will be compiled to an aocx file.
  InputInfoList ForeachInputs;
  InputInfoList FPGADepFiles;
  ArgStringList CmdArgs{"-device=fpga_fast_emu"};

  for (const auto &II : Inputs) {
    if (II.getType() == types::TY_TempAOCOfilelist ||
        II.getType() == types::TY_FPGA_Dependencies ||
        II.getType() == types::TY_FPGA_Dependencies_List)
      continue;
    if (II.getType() == types::TY_Tempfilelist)
      ForeachInputs.push_back(II);
    CmdArgs.push_back(
        C.getArgs().MakeArgString("-spv=" + Twine(II.getFilename())));
  }
  CmdArgs.push_back(
      C.getArgs().MakeArgString("-ir=" + Twine(Output.getFilename())));

  StringRef ForeachExt = "aocx";
  if (Arg *A = Args.getLastArg(options::OPT_fsycl_link_EQ))
    if (A->getValue() == StringRef("early"))
      ForeachExt = "aocr";

  // Add any implied arguments before user defined arguments.
  const toolchains::SYCLToolChain &TC =
      static_cast<const toolchains::SYCLToolChain &>(getToolChain());
  llvm::Triple CPUTriple("spir64_x86_64");
  TC.AddImpliedTargetArgs(CPUTriple, Args, CmdArgs);
  // Add the target args passed in
  TC.TranslateBackendTargetArgs(CPUTriple, Args, CmdArgs);
  TC.TranslateLinkerTargetArgs(CPUTriple, Args, CmdArgs);

  SmallString<128> ExecPath(
      getToolChain().GetProgramPath(makeExeName(C, "opencl-aot")));
  const char *Exec = C.getArgs().MakeArgString(ExecPath);
  auto Cmd = std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                       Exec, CmdArgs, None);
  if (!ForeachInputs.empty()) {
    StringRef ParallelJobs =
        Args.getLastArgValue(options::OPT_fsycl_max_parallel_jobs_EQ);
    constructLLVMForeachCommand(C, JA, std::move(Cmd), ForeachInputs, Output,
                                this, "", ForeachExt, ParallelJobs);
  } else
    C.addCommand(std::move(Cmd));
}

void SYCL::fpga::BackendCompiler::ConstructJob(
    Compilation &C, const JobAction &JA, const InputInfo &Output,
    const InputInfoList &Inputs, const ArgList &Args,
    const char *LinkingOutput) const {
  assert((getToolChain().getTriple().getArch() == llvm::Triple::spir ||
          getToolChain().getTriple().getArch() == llvm::Triple::spir64) &&
         "Unsupported target");

  // Grab the -Xsycl-target* options.
  const toolchains::SYCLToolChain &TC =
      static_cast<const toolchains::SYCLToolChain &>(getToolChain());
  ArgStringList TargetArgs;
  TC.TranslateBackendTargetArgs(TC.getTriple(), Args, TargetArgs);

  // When performing emulation compilations for FPGA AOT, we want to use
  // opencl-aot instead of aoc.
  if (C.getDriver().isFPGAEmulationMode()) {
    constructOpenCLAOTCommand(C, JA, Output, Inputs, Args);
    return;
  }

  InputInfoList ForeachInputs;
  InputInfoList FPGADepFiles;
  StringRef CreatedReportName;
  ArgStringList CmdArgs{"-o", Output.getFilename()};
  for (const auto &II : Inputs) {
    std::string Filename(II.getFilename());
    if (II.getType() == types::TY_Tempfilelist)
      ForeachInputs.push_back(II);
    if (II.getType() == types::TY_TempAOCOfilelist)
      // Add any FPGA library lists.  These come in as special tempfile lists.
      CmdArgs.push_back(Args.MakeArgString(Twine("-library-list=") + Filename));
    else if (II.getType() == types::TY_FPGA_Dependencies ||
             II.getType() == types::TY_FPGA_Dependencies_List)
      FPGADepFiles.push_back(II);
    else
      CmdArgs.push_back(C.getArgs().MakeArgString(Filename));
    // Check for any AOCR input, if found use that as the project report name
    StringRef Ext(llvm::sys::path::extension(Filename));
    if (Ext.empty())
      continue;
    if (getToolChain().LookupTypeForExtension(Ext.drop_front()) ==
        types::TY_FPGA_AOCR) {
      // Keep the base of the .aocr file name.  Input file is a temporary,
      // so we are stripping off the additional naming information for a
      // cleaner name.  The suffix being stripped from the name is the
      // added temporary string and the extension.
      StringRef SuffixFormat("-XXXXXX.aocr");
      SmallString<128> NameBase(
          Filename.substr(0, Filename.length() - SuffixFormat.size()));
      NameBase.append(".prj");
      CreatedReportName =
          Args.MakeArgString(llvm::sys::path::filename(NameBase));
    }
  }
  CmdArgs.push_back("-sycl");

  StringRef ForeachExt = "aocx";
  if (Arg *A = Args.getLastArg(options::OPT_fsycl_link_EQ))
    if (A->getValue() == StringRef("early")) {
      CmdArgs.push_back("-rtl");
      ForeachExt = "aocr";
    }

  for (auto *A : Args) {
    // Any input file is assumed to have a dependency file associated and
    // the report folder can also be named based on the first input.
    if (A->getOption().getKind() != Option::InputClass)
      continue;
    SmallString<128> ArgName(A->getSpelling());
    StringRef Ext(llvm::sys::path::extension(ArgName));
    if (Ext.empty())
      continue;
    types::ID Ty = getToolChain().LookupTypeForExtension(Ext.drop_front());
    if (Ty == types::TY_INVALID)
      continue;
    if (types::isSrcFile(Ty) || Ty == types::TY_Object) {
      // The project report is created in CWD, so strip off any directory
      // information if provided with the input file.
      StringRef TrimmedArgName = llvm::sys::path::filename(ArgName);
      if (types::isSrcFile(Ty)) {
        SmallString<128> DepName(
            C.getDriver().getFPGATempDepFile(std::string(TrimmedArgName)));
        if (!DepName.empty())
          FPGADepFiles.push_back(InputInfo(types::TY_Dependencies,
                                           Args.MakeArgString(DepName),
                                           Args.MakeArgString(DepName)));
      }
      if (CreatedReportName.empty()) {
        // Project report should be saved into CWD, so strip off any
        // directory information if provided with the input file.
        llvm::sys::path::replace_extension(ArgName, "prj");
        CreatedReportName = Args.MakeArgString(ArgName);
      }
    }
  }

  // Add any dependency files.
  if (!FPGADepFiles.empty()) {
    SmallString<128> DepOpt("-dep-files=");
    for (unsigned I = 0; I < FPGADepFiles.size(); ++I) {
      if (I)
        DepOpt += ',';
      if (FPGADepFiles[I].getType() == types::TY_FPGA_Dependencies_List)
        DepOpt += "@";
      DepOpt += FPGADepFiles[I].getFilename();
    }
    CmdArgs.push_back(C.getArgs().MakeArgString(DepOpt));
  }

  // Depending on output file designations, set the report folder
  SmallString<128> ReportOptArg;
  if (Arg *FinalOutput = Args.getLastArg(options::OPT_o, options::OPT__SLASH_o,
                                         options::OPT__SLASH_Fe)) {
    SmallString<128> FN(FinalOutput->getValue());
    // For "-o file.xxx" where the option value has an extension, if the
    // extension is one of .a .o .out .lib .obj .exe, the output project
    // directory name will be file.proj which omits the extension. Otherwise
    // the output project directory name will be file.xxx.prj which keeps
    // the original extension.
    StringRef Ext = llvm::sys::path::extension(FN);
    SmallVector<StringRef, 6> Exts = {".o",   ".a",   ".out",
                                      ".obj", ".lib", ".exe"};
    if (std::find(Exts.begin(), Exts.end(), Ext) != Exts.end())
      llvm::sys::path::replace_extension(FN, "prj");
    else
      FN.append(".prj");
    const char *FolderName = Args.MakeArgString(FN);
    ReportOptArg += FolderName;
  } else {
    // Output directory is based off of the first object name as captured
    // above.
    if (!CreatedReportName.empty())
      ReportOptArg += CreatedReportName;
  }
  if (!ReportOptArg.empty())
    CmdArgs.push_back(C.getArgs().MakeArgString(
        Twine("-output-report-folder=") + ReportOptArg));

  // Add any implied arguments before user defined arguments.
  TC.AddImpliedTargetArgs(getToolChain().getTriple(), Args, CmdArgs);

  // Add -Xsycl-target* options.
  TC.TranslateBackendTargetArgs(getToolChain().getTriple(), Args, CmdArgs);
  TC.TranslateLinkerTargetArgs(getToolChain().getTriple(), Args, CmdArgs);

  // Look for -reuse-exe=XX option
  if (Arg *A = Args.getLastArg(options::OPT_reuse_exe_EQ)) {
    Args.ClaimAllArgs(options::OPT_reuse_exe_EQ);
    CmdArgs.push_back(Args.MakeArgString(A->getAsString(Args)));
  }

  SmallString<128> ExecPath(
      getToolChain().GetProgramPath(makeExeName(C, "aoc")));
  const char *Exec = C.getArgs().MakeArgString(ExecPath);
  auto Cmd = std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                       Exec, CmdArgs, None);
  addFPGATimingDiagnostic(Cmd, C);
  if (!ForeachInputs.empty()) {
    StringRef ParallelJobs =
        Args.getLastArgValue(options::OPT_fsycl_max_parallel_jobs_EQ);
    constructLLVMForeachCommand(C, JA, std::move(Cmd), ForeachInputs, Output,
                                this, ReportOptArg, ForeachExt, ParallelJobs);
  } else
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
  ArgStringList CmdArgs{"-output", Output.getFilename()};
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
  TC.AddImpliedTargetArgs(getToolChain().getTriple(), Args, CmdArgs);
  TC.TranslateBackendTargetArgs(getToolChain().getTriple(), Args, CmdArgs);
  TC.TranslateLinkerTargetArgs(getToolChain().getTriple(), Args, CmdArgs);
  SmallString<128> ExecPath(
      getToolChain().GetProgramPath(makeExeName(C, "ocloc")));
  const char *Exec = C.getArgs().MakeArgString(ExecPath);
  auto Cmd = std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                       Exec, CmdArgs, None);
  if (!ForeachInputs.empty()) {
    StringRef ParallelJobs =
        Args.getLastArgValue(options::OPT_fsycl_max_parallel_jobs_EQ);
    constructLLVMForeachCommand(C, JA, std::move(Cmd), ForeachInputs, Output,
                                this, "", "out", ParallelJobs);
  } else
    C.addCommand(std::move(Cmd));
}

void SYCL::x86_64::BackendCompiler::ConstructJob(
    Compilation &C, const JobAction &JA, const InputInfo &Output,
    const InputInfoList &Inputs, const ArgList &Args,
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

  TC.AddImpliedTargetArgs(getToolChain().getTriple(), Args, CmdArgs);
  TC.TranslateBackendTargetArgs(getToolChain().getTriple(), Args, CmdArgs);
  TC.TranslateLinkerTargetArgs(getToolChain().getTriple(), Args, CmdArgs);
  SmallString<128> ExecPath(
      getToolChain().GetProgramPath(makeExeName(C, "opencl-aot")));
  const char *Exec = C.getArgs().MakeArgString(ExecPath);
  auto Cmd = std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                       Exec, CmdArgs, None);
  if (!ForeachInputs.empty()) {
    StringRef ParallelJobs =
        Args.getLastArgValue(options::OPT_fsycl_max_parallel_jobs_EQ);
    constructLLVMForeachCommand(C, JA, std::move(Cmd), ForeachInputs, Output,
                                this, "", "out", ParallelJobs);
  } else
    C.addCommand(std::move(Cmd));
}

SYCLToolChain::SYCLToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ToolChain &HostTC, const ArgList &Args)
    : ToolChain(D, Triple, Args), HostTC(HostTC), SYCLInstallation(D) {
  // Lookup binaries into the driver directory, this is used to
  // discover the clang-offload-bundler executable.
  getProgramPaths().push_back(getDriver().Dir);
}

void SYCLToolChain::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args,
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
    for (Arg *A : Args) {
      // Filter out any options we do not want to pass along to the device
      // compilation.
      switch ((options::ID)A->getOption().getID()) {
      default:
        DAL->append(A);
        break;
      }
    }
  }
  // Strip out -O0 for FPGA Hardware device compilation.
  if (!getDriver().isFPGAEmulationMode() &&
      getTriple().getSubArch() == llvm::Triple::SPIRSubArch_fpga)
    DAL->eraseArg(options::OPT_O0);

  const OptTable &Opts = getDriver().getOpts();
  if (!BoundArch.empty()) {
    DAL->eraseArg(options::OPT_march_EQ);
    DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_march_EQ),
                      BoundArch);
  }
  return DAL;
}

static void parseTargetOpts(StringRef ArgString, const llvm::opt::ArgList &Args,
                            llvm::opt::ArgStringList &CmdArgs) {
  // Tokenize the string.
  SmallVector<const char *, 8> TargetArgs;
  llvm::BumpPtrAllocator A;
  llvm::StringSaver S(A);
  llvm::cl::TokenizeGNUCommandLine(ArgString, S, TargetArgs);
  for (StringRef TA : TargetArgs)
    CmdArgs.push_back(Args.MakeArgString(TA));
}

// Expects a specific type of option (e.g. -Xsycl-target-backend) and will
// extract the arguments.
void SYCLToolChain::TranslateTargetOpt(const llvm::opt::ArgList &Args,
                                       llvm::opt::ArgStringList &CmdArgs,
                                       OptSpecifier Opt,
                                       OptSpecifier Opt_EQ) const {
  for (auto *A : Args) {
    bool OptNoTriple;
    OptNoTriple = A->getOption().matches(Opt);
    if (A->getOption().matches(Opt_EQ)) {
      // Passing device args: -X<Opt>=<triple> -opt=val.
      if (getDriver().MakeSYCLDeviceTriple(A->getValue()) != getTriple())
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
      const Arg *TargetArg = Args.getLastArg(options::OPT_fsycl_targets_EQ);
      if (TargetArg && TargetArg->getValues().size() != 1) {
        getDriver().Diag(diag::err_drv_Xsycl_target_missing_triple)
            << A->getSpelling();
        continue;
      }
      // No triple, so just add the argument.
      ArgString = A->getValue();
    } else
      // Triple found, add the next argument in line.
      ArgString = A->getValue(1);

    parseTargetOpts(ArgString, Args, CmdArgs);
    A->claim();
  }
}

void SYCLToolChain::AddImpliedTargetArgs(
    const llvm::Triple &Triple, const llvm::opt::ArgList &Args,
    llvm::opt::ArgStringList &CmdArgs) const {
  // Current implied args are for debug information and disabling of
  // optimizations.  They are passed along to the respective areas as follows:
  //  FPGA and default device:  -g -cl-opt-disable
  //  GEN:  -options "-g -O0"
  //  CPU:  "--bo=-g -cl-opt-disable"
  llvm::opt::ArgStringList BeArgs;
  bool IsGen = Triple.getSubArch() == llvm::Triple::SPIRSubArch_gen;
  if (Arg *A = Args.getLastArg(options::OPT_g_Group, options::OPT__SLASH_Z7))
    if (!A->getOption().matches(options::OPT_g0))
      BeArgs.push_back("-g");
  if (Args.getLastArg(options::OPT_O0))
    BeArgs.push_back("-cl-opt-disable");
  if (BeArgs.empty())
    return;
  if (Triple.getSubArch() == llvm::Triple::NoSubArch ||
      Triple.getSubArch() == llvm::Triple::SPIRSubArch_fpga) {
    for (StringRef A : BeArgs)
      CmdArgs.push_back(Args.MakeArgString(A));
    return;
  }
  SmallString<128> BeOpt;
  if (IsGen)
    CmdArgs.push_back("-options");
  else
    BeOpt = "--bo=";
  for (unsigned I = 0; I < BeArgs.size(); ++I) {
    if (I)
      BeOpt += ' ';
    BeOpt += BeArgs[I];
  }
  CmdArgs.push_back(Args.MakeArgString(BeOpt));
}

void SYCLToolChain::TranslateBackendTargetArgs(
    const llvm::Triple &Triple, const llvm::opt::ArgList &Args,
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

    // Do not add the -Xs to the default SYCL triple (spir64) when we know we
    // have implied the setting.
    if ((A->getOption().matches(options::OPT_Xs) ||
         A->getOption().matches(options::OPT_Xs_separate)) &&
        Triple.getSubArch() == llvm::Triple::NoSubArch && Triple.isSPIR() &&
        getDriver().isSYCLDefaultTripleImplied())
      continue;

    if (A->getOption().matches(options::OPT_Xs)) {
      // Take the arg and create an option out of it.
      CmdArgs.push_back(Args.MakeArgString(Twine("-") + A->getValue()));
      A->claim();
      continue;
    }
    if (A->getOption().matches(options::OPT_Xs_separate)) {
      StringRef ArgString(A->getValue());
      parseTargetOpts(ArgString, Args, CmdArgs);
      A->claim();
      continue;
    }
  }
  // Do not process -Xsycl-target-backend for implied spir64
  if (Triple.getSubArch() == llvm::Triple::NoSubArch && Triple.isSPIR() &&
      getDriver().isSYCLDefaultTripleImplied())
    return;
  // Handle -Xsycl-target-backend.
  TranslateTargetOpt(Args, CmdArgs, options::OPT_Xsycl_backend,
                     options::OPT_Xsycl_backend_EQ);
}

void SYCLToolChain::TranslateLinkerTargetArgs(
    const llvm::Triple &Triple, const llvm::opt::ArgList &Args,
    llvm::opt::ArgStringList &CmdArgs) const {
  // Do not process -Xsycl-target-linker for implied spir64
  if (Triple.getSubArch() == llvm::Triple::NoSubArch && Triple.isSPIR() &&
      getDriver().isSYCLDefaultTripleImplied())
    return;
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

void SYCLToolChain::AddSYCLIncludeArgs(const clang::driver::Driver &Driver,
                                       const ArgList &DriverArgs,
                                       ArgStringList &CC1Args) {
  // Add ../include/sycl and ../include (in that order)
  SmallString<128> P(Driver.getInstalledDir());
  llvm::sys::path::append(P, "..");
  llvm::sys::path::append(P, "include");
  SmallString<128> SYCLP(P);
  llvm::sys::path::append(SYCLP, "sycl");
  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(SYCLP));
  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(P));
}

void SYCLToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                              ArgStringList &CC1Args) const {
  HostTC.AddClangSystemIncludeArgs(DriverArgs, CC1Args);
}

void SYCLToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &Args,
                                                 ArgStringList &CC1Args) const {
  HostTC.AddClangCXXStdlibIncludeArgs(Args, CC1Args);
}
