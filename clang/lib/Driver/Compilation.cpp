//===- Compilation.cpp - Compilation Task Implementation ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Compilation.h"
#include "clang/Basic/LLVM.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Util.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptSpecifier.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SimpleTable.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <cassert>
#include <fstream>
#include <string>
#include <system_error>
#include <utility>

using namespace clang;
using namespace driver;
using namespace llvm::opt;

Compilation::Compilation(const Driver &D, const ToolChain &_DefaultToolChain,
                         InputArgList *_Args, DerivedArgList *_TranslatedArgs,
                         bool ContainsError)
    : TheDriver(D), DefaultToolChain(_DefaultToolChain), Args(_Args),
      TranslatedArgs(_TranslatedArgs), ContainsError(ContainsError) {
  // The offloading host toolchain is the default toolchain.
  OrderedOffloadingToolchains.insert(
      std::make_pair(Action::OFK_Host, &DefaultToolChain));
}

Compilation::~Compilation() {
  // Remove temporary files. This must be done before arguments are freed, as
  // the file names might be derived from the input arguments.
  if (!TheDriver.isSaveTempsEnabled() && !ForceKeepTempFiles)
    CleanupFileList(TempFiles);

  delete TranslatedArgs;
  delete Args;

  // Free any derived arg lists.
  for (auto Arg : TCArgs)
    if (Arg.second != TranslatedArgs)
      delete Arg.second;
}

static void HandleXarchArgs(DerivedArgList *OffloadArgList, const Driver &D,
                            bool IsDevice) {
  if (!OffloadArgList)
    return;

  if (IsDevice && !OffloadArgList->hasArg(options::OPT_Xarch_device))
    return;

  if (!IsDevice && !OffloadArgList->hasArg(options::OPT_Xarch_host))
    return;

  bool NeedHandle = false;
  std::vector<std::string> XarchValues;
  XarchValues = IsDevice
                    ? OffloadArgList->getAllArgValues(options::OPT_Xarch_device)
                    : OffloadArgList->getAllArgValues(options::OPT_Xarch_host);
  SmallVector<StringRef, 8> XarchValueRefs;
  for (auto XarchV : XarchValues) {
    if (XarchV.find(' ') != std::string::npos) {
      NeedHandle = true;
      StringRef XarchVRef(XarchV);
      SmallVector<StringRef, 8> XarchVecs;
      XarchVRef.trim().split(XarchVecs, ' ', -1, false);
      size_t Index;
      const size_t XSize = XarchVecs.size();
      for (Index = 0; Index < XSize; ++Index) {
        if (XarchVecs[Index].compare("-mllvm") == 0) {
          if (Index < (XSize - 1)) {
            XarchValueRefs.push_back(OffloadArgList->MakeArgStringRef(
                (StringRef("-mllvm=") + XarchVecs[Index + 1]).str()));
            Index++;
            continue;
          } else
            D.Diag(clang::diag::err_drv_missing_argument) << "-mllvm" << 1;
        } else
          XarchValueRefs.push_back(
              OffloadArgList->MakeArgStringRef(XarchVecs[Index]));
      }
    } else
      XarchValueRefs.push_back(OffloadArgList->MakeArgString(XarchV));
  }

  if (NeedHandle) {
    auto Xarch_OPT =
        IsDevice ? options::OPT_Xarch_device : options::OPT_Xarch_host;
    OffloadArgList->eraseArg(Xarch_OPT);
    for (auto XarchV : XarchValueRefs) {
      Arg *A = OffloadArgList->MakeSeparateArg(
          nullptr, D.getOpts().getOption(Xarch_OPT), XarchV);
      A->claim();
      OffloadArgList->append(A);
    }
  }
}

const DerivedArgList &
Compilation::getArgsForToolChain(const ToolChain *TC, StringRef BoundArch,
                                 Action::OffloadKind DeviceOffloadKind) {
  if (!TC)
    TC = &DefaultToolChain;

  DerivedArgList *&Entry = TCArgs[{TC, BoundArch, DeviceOffloadKind}];
  if (!Entry) {
    SmallVector<Arg *, 4> AllocatedArgs;
    DerivedArgList *OffloadArgs = nullptr;
    // Translate offload toolchain arguments provided via the -Xopenmp-target
    // or -Xsycl-target-frontend flags.
    if (DeviceOffloadKind == Action::OFK_OpenMP ||
        DeviceOffloadKind == Action::OFK_SYCL) {
      const ToolChain *HostTC = getSingleOffloadToolChain<Action::OFK_Host>();
      bool SameTripleAsHost = (TC->getTriple() == HostTC->getTriple());
      OffloadArgs = TC->TranslateOffloadTargetArgs(
          *TranslatedArgs, SameTripleAsHost, AllocatedArgs, DeviceOffloadKind);
    }

    DerivedArgList *NewDAL = nullptr;
    if (!OffloadArgs) {
      HandleXarchArgs(TranslatedArgs, getDriver(), false);
      NewDAL = TC->TranslateXarchArgs(*TranslatedArgs, BoundArch,
                                      DeviceOffloadKind, &AllocatedArgs);
    } else {
      HandleXarchArgs(OffloadArgs, getDriver(), true);
      NewDAL = TC->TranslateXarchArgs(*OffloadArgs, BoundArch, DeviceOffloadKind,
                                      &AllocatedArgs);
      if (!NewDAL)
        NewDAL = OffloadArgs;
      else
        delete OffloadArgs;
    }

    if (!NewDAL) {
      Entry = TC->TranslateArgs(*TranslatedArgs, BoundArch, DeviceOffloadKind);
      if (!Entry)
        Entry = TranslatedArgs;
    } else {
      Entry = TC->TranslateArgs(*NewDAL, BoundArch, DeviceOffloadKind);
      if (!Entry)
        Entry = NewDAL;
      else
        delete NewDAL;
    }

    // Add allocated arguments to the final DAL.
    for (auto *ArgPtr : AllocatedArgs)
      Entry->AddSynthesizedArg(ArgPtr);
  }

  return *Entry;
}

bool Compilation::CleanupFile(const char *File, bool IssueErrors) const {
  // FIXME: Why are we trying to remove files that we have not created? For
  // example we should only try to remove a temporary assembly file if
  // "clang -cc1" succeed in writing it. Was this a workaround for when
  // clang was writing directly to a .s file and sometimes leaving it behind
  // during a failure?

  // FIXME: If this is necessary, we can still try to split
  // llvm::sys::fs::remove into a removeFile and a removeDir and avoid the
  // duplicated stat from is_regular_file.

  // Don't try to remove files which we don't have write access to (but may be
  // able to remove), or non-regular files. Underlying tools may have
  // intentionally not overwritten them.

  // Save the device code files if -save-offload-code option is enabled.
  if (TheDriver.isSaveOffloadCodeEnabled()) {
    Arg *SaveOffloadCodeArg =
        getArgs().getLastArg(options::OPT_save_offload_code_EQ);
    std::string ExpectedDir =
        SaveOffloadCodeArg ? SaveOffloadCodeArg->getValue() : "";
    std::string ActualFile(File);

    if (ActualFile.find(ExpectedDir) != std::string::npos) {
      // Save PTX files generated by LLVM NVPTX Back-End,
      // when the nvptx*-nvidia-cuda is passed to -fsycl-targets.
      if (DefaultToolChain.getTriple().isNVPTX())
        return false;
      if (llvm::sys::path::extension(ActualFile) == ".spv")
        return false;
    }
  }

  if (!llvm::sys::fs::can_write(File) || !llvm::sys::fs::is_regular_file(File))
    return true;

  if (std::error_code EC = llvm::sys::fs::remove(File)) {
    // Failure is only failure if the file exists and is "regular". We checked
    // for it being regular before, and llvm::sys::fs::remove ignores ENOENT,
    // so we don't need to check again.

    if (IssueErrors)
      getDriver().Diag(diag::err_drv_unable_to_remove_file)
        << EC.message();
    return false;
  }
  return true;
}

bool Compilation::CleanupFileList(const TempFileList &Files,
                                  bool IssueErrors) const {
  bool Success = true;
  for (const auto &File : Files) {
    // Temporary file lists contain files that need to be cleaned. The
    // file containing the information is also removed
    if (File.second == types::TY_Tempfilelist ||
        File.second == types::TY_Tempfiletable) {
      // These are temporary files and need to be removed.
      bool IsTable = File.second == types::TY_Tempfiletable;

      if (IsTable) {
        if (llvm::sys::fs::exists(File.first)) {
          auto T = llvm::util::SimpleTable::read(File.first);
          if (!T) {
            Success = false;
            continue;
          }

          std::vector<std::string> TmpFileNames;
          T->get()->linearize(TmpFileNames);

          for (const auto &TmpFileName : TmpFileNames) {
            if (!TmpFileName.empty())
              Success &= CleanupFile(TmpFileName.c_str(), IssueErrors);
          }
        }
      } else {
        std::ifstream ListFile(File.first);
        std::string TmpFileName;
        while (std::getline(ListFile, TmpFileName))
          Success &= CleanupFile(TmpFileName.c_str(), IssueErrors);
      }
    }
    Success &= CleanupFile(File.first, IssueErrors);
  }
  return Success;
}

bool Compilation::CleanupFileMap(const ArgStringMap &Files,
                                 const JobAction *JA,
                                 bool IssueErrors) const {
  bool Success = true;
  for (const auto &File : Files) {
    // If specified, only delete the files associated with the JobAction.
    // Otherwise, delete all files in the map.
    if (JA && File.first != JA)
      continue;
    Success &= CleanupFile(File.second, IssueErrors);
  }
  return Success;
}

int Compilation::ExecuteCommand(const Command &C,
                                const Command *&FailingCommand,
                                bool LogOnly) const {
  if ((getDriver().CCPrintOptions ||
       getArgs().hasArg(options::OPT_v)) && !getDriver().CCGenDiagnostics) {
    raw_ostream *OS = &llvm::errs();
    std::unique_ptr<llvm::raw_fd_ostream> OwnedStream;

    // Follow gcc implementation of CC_PRINT_OPTIONS; we could also cache the
    // output stream.
    if (getDriver().CCPrintOptions &&
        !getDriver().CCPrintOptionsFilename.empty()) {
      std::error_code EC;
      OwnedStream.reset(new llvm::raw_fd_ostream(
          getDriver().CCPrintOptionsFilename, EC,
          llvm::sys::fs::OF_Append | llvm::sys::fs::OF_TextWithCRLF));
      if (EC) {
        getDriver().Diag(diag::err_drv_cc_print_options_failure)
            << EC.message();
        FailingCommand = &C;
        return 1;
      }
      OS = OwnedStream.get();
    }

    if (getDriver().CCPrintOptions)
      *OS << "[Logging clang options]\n";

    C.Print(*OS, "\n", /*Quote=*/getDriver().CCPrintOptions);
  }

  if (LogOnly)
    return 0;

  std::string Error;
  bool ExecutionFailed;
  int Res = C.Execute(Redirects, &Error, &ExecutionFailed);
  if (PostCallback)
    PostCallback(C, Res);
  if (!Error.empty()) {
    assert(Res && "Error string set with 0 result code!");
    getDriver().Diag(diag::err_drv_command_failure) << Error;
  }

  // When performing preprocessing, we need to be able to produce the
  // preprocessed output even if the compilation is not valid.  If
  // the device compilation fails for SYCL allow the failure to pass
  // through so we can still generate the expected preprocessed files.
  if (Res && C.getSource().isDeviceOffloading(Action::OFK_SYCL) &&
      getArgs().hasArg(options::OPT_E))
    return 0;

  if (Res)
    FailingCommand = &C;

  return ExecutionFailed ? 1 : Res;
}

using FailingCommandList = SmallVectorImpl<std::pair<int, const Command *>>;

static bool ActionFailed(const Action *A,
                         const FailingCommandList &FailingCommands) {
  if (FailingCommands.empty())
    return false;

  for (const auto &CI : FailingCommands)
    if (!CI.second->getWillExitForErrorCode(CI.first))
      return false;

  // CUDA/HIP/SYCL can have the same input source code compiled multiple times
  // so do not compile again if there are already failures. It is OK to abort
  // the CUDA/HIP/SYCL pipeline on errors.
  if (A->isOffloading(Action::OFK_Cuda) || A->isOffloading(Action::OFK_HIP) ||
      A->isOffloading(Action::OFK_SYCL))
    return true;

  for (const auto &CI : FailingCommands)
    if (A == &(CI.second->getSource()))
      return true;

  for (const auto *AI : A->inputs())
    if (ActionFailed(AI, FailingCommands))
      return true;

  return false;
}

static bool InputsOk(const Command &C,
                     const FailingCommandList &FailingCommands) {
  return !ActionFailed(&C.getSource(), FailingCommands);
}

void Compilation::ExecuteJobs(const JobList &Jobs,
                              FailingCommandList &FailingCommands,
                              bool LogOnly) const {
  // According to UNIX standard, driver need to continue compiling all the
  // inputs on the command line even one of them failed.
  // In all but CLMode, execute all the jobs unless the necessary inputs for the
  // job is missing due to previous failures.
  for (const auto &Job : Jobs) {
    if (!InputsOk(Job, FailingCommands))
      continue;
    const Command *FailingCommand = nullptr;
    if (int Res = ExecuteCommand(Job, FailingCommand, LogOnly)) {
      FailingCommands.push_back(std::make_pair(Res, FailingCommand));
      // Bail as soon as one command fails in cl driver mode.
      // Do not bail when the tool is setup to allow for continuation upon
      // failure.
      if (TheDriver.IsCLMode() && FailingCommand->getWillExitForErrorCode(Res))
        return;
    }
  }
}

void Compilation::initCompilationForDiagnostics() {
  ForDiagnostics = true;

  // Free actions and jobs.
  Actions.clear();
  AllActions.clear();
  Jobs.clear();

  // Remove temporary files.
  if (!TheDriver.isSaveTempsEnabled() && !ForceKeepTempFiles)
    CleanupFileList(TempFiles);

  // Clear temporary/results file lists.
  TempFiles.clear();
  ResultFiles.clear();
  FailureResultFiles.clear();

  // Remove any user specified output.  Claim any unclaimed arguments, so as
  // to avoid emitting warnings about unused args.
  OptSpecifier OutputOpts[] = {
      options::OPT_o,  options::OPT_MD, options::OPT_MMD, options::OPT_M,
      options::OPT_MM, options::OPT_MF, options::OPT_MG,  options::OPT_MJ,
      options::OPT_MQ, options::OPT_MT, options::OPT_MV};
  for (const auto &Opt : OutputOpts) {
    if (TranslatedArgs->hasArg(Opt))
      TranslatedArgs->eraseArg(Opt);
  }
  TranslatedArgs->ClaimAllArgs();

  // Force re-creation of the toolchain Args, otherwise our modifications just
  // above will have no effect.
  for (auto Arg : TCArgs)
    if (Arg.second != TranslatedArgs)
      delete Arg.second;
  TCArgs.clear();

  // Redirect stdout/stderr to /dev/null.
  Redirects = {std::nullopt, {""}, {""}};

  // Temporary files added by diagnostics should be kept.
  ForceKeepTempFiles = true;
}

StringRef Compilation::getSysRoot() const {
  return getDriver().SysRoot;
}

void Compilation::Redirect(ArrayRef<std::optional<StringRef>> Redirects) {
  this->Redirects = Redirects;
}
