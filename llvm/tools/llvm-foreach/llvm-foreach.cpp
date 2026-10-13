//===- llvm-foreach.cpp - Command lines execution ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This source is utility to execute command lines. The specified command will
// be invoked as many times as necessary to use up the list of input items.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#include <list>
#include <thread>
#include <vector>

using namespace llvm;

static cl::list<std::string> InputFileLists{
    "in-file-list", cl::OneOrMore,
    cl::desc("Input list of file names, file names must be delimited by a "
             "newline character."),
    cl::value_desc("filename")};

static cl::list<std::string> InputCommandArgs{
    cl::Positional, cl::OneOrMore, cl::desc("<command>"),
    cl::value_desc("command")};

static cl::list<std::string> Replaces{
    "in-replace", cl::OneOrMore,
    cl::desc("Specify input path in input command, this will be replaced with "
             "names read from corresponding input list of files."),
    cl::value_desc("R")};

static cl::opt<std::string> OutReplace{
    "out-replace",
    cl::desc("Specify output path in input command, this will be replaced with "
             "name of temporary file created for writing command's outputs."),
    cl::init(""), cl::value_desc("R")};

static cl::opt<std::string> OutDirectory{
    "out-dir",
    cl::desc("Specify directory for output files; If unspecified, assume "
             "system temporary directory."),
    cl::init(""), cl::value_desc("R")};

static cl::opt<std::string> OutFilesExt{
    "out-ext",
    cl::desc("Specify extension for output files; If unspecified, assume "
             ".out"),
    cl::init("out"), cl::value_desc("R")};

// Emit list of produced files for better integration with other tools.
static cl::opt<std::string> OutputFileList{
    "out-file-list", cl::desc("Specify filename for list of outputs."),
    cl::value_desc("filename"), cl::init("")};

static cl::opt<std::string> OutIncrement{
    "out-increment",
    cl::desc(
        "Specify output file which should be incrementally named with each "
        "pass."),
    cl::init(""), cl::value_desc("R")};

static cl::opt<unsigned int> JobsInParallel{
    "jobs",
    cl::Optional,
    cl::init(1),
    cl::desc("Specify the number of threads for launching input commands in "
             "parallel mode"),
};

static cl::alias JobsInParallelShort{"j", cl::desc("Alias for --jobs"),
                                     cl::aliasopt(JobsInParallel)};

static constexpr int JobPollSleepIntervalMS = 100;

namespace {

struct RunningJob {
  sys::ProcessInfo Process;
  std::string Command;
};

static std::string formatCommand(ArrayRef<StringRef> Args) {
  std::string Command;
  raw_string_ostream OS(Command);
  for (size_t I = 0; I < Args.size(); ++I) {
    if (I)
      OS << ' ';
    sys::printArg(OS, Args[I], /*Quote=*/true);
  }
  return Command;
}

static void printCommandFailure(int ReturnCode, StringRef ErrMsg,
                                StringRef Command) {
  errs() << "llvm-foreach: command failed";
  if (ReturnCode != 0)
    errs() << " with error code: " << ReturnCode;
  if (!ErrMsg.empty())
    errs() << ": " << ErrMsg;
  errs() << "\n"
         << "llvm-foreach: command: " << Command << '\n';
}

static void error(const Twine &Msg) {
  errs() << "llvm-foreach: " << Msg << '\n';
  exit(1);
}

static void error(std::error_code EC, const Twine &Prefix) {
  if (EC)
    error(Prefix + ": " + EC.message());
}

// Poll all running jobs.
// Try to find a finished job with a non-blocking wait.
// If no job found then sleep and repeat.
// If found one job, removes it from Jobs and returns its exit code..
static int waitForAnyJob(std::list<RunningJob> &Jobs) {
  while (true) {
    std::string ErrMsg;
    for (auto It = Jobs.begin(); It != Jobs.end(); ++It) {
      // Non-blocking wait: returns Pid==0 if still running.
      sys::ProcessInfo WaitResult =
          sys::Wait(It->Process, /*SecondsToWait=*/0, &ErrMsg);
      if (WaitResult.Pid == 0)
        continue; // Job is in progress. Move on.

      if (WaitResult.ReturnCode != 0)
        printCommandFailure(WaitResult.ReturnCode, ErrMsg, It->Command);

      int RC = WaitResult.ReturnCode;
      Jobs.erase(It);
      return RC;
    }

    std::this_thread::sleep_for(
        std::chrono::milliseconds(JobPollSleepIntervalMS));
  }
}

} // namespace

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(
      argc, argv,
      "llvm-foreach: Execute specified command as many times as\n"
      "necessary to use up the list of input items.\n"
      "Usage:\n"
      "llvm-foreach --in-file-list=a.list --in-replace='{}' -- echo '{}'\n"
      "NOTE: commands containig redirects are not supported by llvm-foreach\n"
      "yet.\n");

  ExitOnError ExitOnErr("llvm-foreach: ");

  if (InputFileLists.size() != Replaces.size())
    error("Number of input file lists and input path replaces don't match.");

  std::vector<std::unique_ptr<MemoryBuffer>> MBs;
  std::vector<line_iterator> LineIterators;
  for (auto &InputFileList : InputFileLists) {
    std::unique_ptr<MemoryBuffer> MB = ExitOnErr(
        errorOrToExpected(MemoryBuffer::getFileOrSTDIN(InputFileList)));
    LineIterators.push_back(line_iterator(*MB));
    MBs.push_back(std::move(MB));
  }

  SmallVector<StringRef, 8> Args(InputCommandArgs.begin(),
                                 InputCommandArgs.end());

  if (Args.empty())
    error("No command?");

  struct ArgumentReplace {
    size_t ArgNum = 0;
    // Index in argument string where replace length starts.
    size_t Start = 0;
    size_t ReplaceLen = 0;
  };

  // Find args to replace with filenames from input list.
  std::vector<ArgumentReplace> InReplaceArgs;
  ArgumentReplace OutReplaceArg;
  ArgumentReplace OutIncrementArg;
  for (size_t I = 1; I < Args.size(); ++I) {
    for (auto &Replace : Replaces) {
      size_t ReplaceStart = Args[I].find(Replace);
      if (ReplaceStart != StringRef::npos)
        InReplaceArgs.push_back({I, ReplaceStart, Replace.size()});
    }

    if (!OutReplace.empty() && Args[I].contains(OutReplace)) {
      size_t ReplaceStart = Args[I].find(OutReplace);
      if (ReplaceStart != StringRef::npos)
        OutReplaceArg = {I, ReplaceStart, OutReplace.size()};
    }

    if (!OutIncrement.empty() && Args[I].contains(OutIncrement)) {
      size_t IncrementStart = Args[I].find(OutIncrement);
      if (IncrementStart != StringRef::npos)
        OutIncrementArg = {I, IncrementStart, OutIncrement.size()};
    }
  }

  // Emit an error if user requested replace output file in the command but
  // replace string is not found.
  if (!OutReplace.empty() && OutReplaceArg.ArgNum == 0)
    error("Couldn't find replace string for output in the command.");

  // Make sure that specified program exists, emit an error if not.
  ErrorOr<std::string> ProgOrErr = sys::findProgramByName(Args[0]);
  if (!ProgOrErr) {
    printCommandFailure(/*ReturnCode=*/0, ProgOrErr.getError().message(),
                        formatCommand(Args));
    return 1;
  }
  std::string Prog = *ProgOrErr;

  std::vector<std::vector<std::string>> FileLists(LineIterators.size());
  size_t PrevNumOfLines = 0;
  for (size_t I = 0; I < FileLists.size(); ++I) {
    for (; !LineIterators[I].is_at_eof(); ++LineIterators[I]) {
      FileLists[I].push_back(LineIterators[I]->str());
    }
    if (I != 0 && FileLists[I].size() != PrevNumOfLines)
      error("All input file lists must have same number of lines!");
    PrevNumOfLines = FileLists[I].size();
  }

  if (!JobsInParallel)
    error("Number of parallel threads should be a positive integer");

  size_t MaxSafeNumThreads = optimal_concurrency().compute_thread_count();
  if (JobsInParallel > MaxSafeNumThreads) {
    JobsInParallel = MaxSafeNumThreads;
    outs() << "llvm-foreach: adjusted number of threads to "
           << MaxSafeNumThreads << " (max safe available).\n";
  }

  std::error_code EC;
  raw_fd_ostream OS{OutputFileList, EC, sys::fs::OpenFlags::OF_None};
  if (!OutputFileList.empty())
    error(EC, "error opening the file '" + OutputFileList + "'");

  int Res = 0;
  std::string ResOutArg;
  std::string IncOutArg;
  std::vector<std::string> ResInArgs(InReplaceArgs.size());
  std::string ResFileList = "";
  std::list<RunningJob> JobsSubmitted;
  for (size_t J = 0; J != FileLists[0].size(); ++J) {
    for (size_t I = 0; I < InReplaceArgs.size(); ++I) {
      ArgumentReplace CurReplace = InReplaceArgs[I];
      std::string OriginalString = InputCommandArgs[CurReplace.ArgNum];
      ResInArgs[I] = (Twine(OriginalString.substr(0, CurReplace.Start)) +
                      Twine(FileLists[I][J]) +
                      Twine(OriginalString.substr(CurReplace.Start +
                                                  CurReplace.ReplaceLen)))
                         .str();
      Args[CurReplace.ArgNum] = ResInArgs[I];
    }

    SmallString<128> Path;
    if (!OutReplace.empty()) {
      // Create a file for command result. Add file name to output
      // file list if needed.
      std::string TempFileNameBase = std::string(sys::path::stem(OutReplace));
      if (OutDirectory.empty())
        EC = sys::fs::createTemporaryFile(TempFileNameBase, OutFilesExt, Path);
      else {
        SmallString<128> PathPrefix(OutDirectory);
        // "CreateUniqueFile" functions accepts "Model" - special string with
        // substring containing sequence of "%" symbols. In the resulting
        // filename "%" symbols sequence from "Model" string will be replaced
        // with random chars to make it unique.
        llvm::sys::path::append(PathPrefix,
                                TempFileNameBase + "-%%%%%%." + OutFilesExt);
        EC = sys::fs::createUniqueFile(PathPrefix, Path);
      }
      error(EC, "Could not create a file for command output.");

      std::string OriginalString = InputCommandArgs[OutReplaceArg.ArgNum];
      ResOutArg =
          (Twine(OriginalString.substr(0, OutReplaceArg.Start)) + Twine(Path) +
           Twine(OriginalString.substr(OutReplaceArg.Start +
                                       OutReplaceArg.ReplaceLen)))
              .str();
      Args[OutReplaceArg.ArgNum] = ResOutArg;

      if (!OutputFileList.empty())
        OS << Path << "\n";
    }

    if (!OutIncrement.empty()) {
      // Name the file by adding the current file list index to the name.
      IncOutArg = InputCommandArgs[OutIncrementArg.ArgNum];
      if (J > 0)
        IncOutArg += ("_" + Twine(J)).str();
      Args[OutIncrementArg.ArgNum] = IncOutArg;
    }
    // Do not start execution of a new job until previous one(s) are finished,
    // if the maximum number of parallel workers is reached.
    if (JobsSubmitted.size() == JobsInParallel) {
      if (int ReturnCode = waitForAnyJob(JobsSubmitted); ReturnCode)
        Res = ReturnCode;
    }

    std::string CommandToRun = formatCommand(Args);
    std::string ErrMsg;
    bool ExecutionFailed = false;
    sys::ProcessInfo PI =
        sys::ExecuteNoWait(Prog, Args, /*Env=*/std::nullopt, /*Redirects=*/{},
                           /*MemoryLimit=*/0, &ErrMsg, &ExecutionFailed);
    if (ExecutionFailed) {
      printCommandFailure(/*ReturnCode=*/0, ErrMsg, CommandToRun);
      Res = 1;
      continue;
    }

    JobsSubmitted.emplace_back(RunningJob{PI, std::move(CommandToRun)});
  }

  // Wait for all commands to be executed (blocking wait, no sleep needed).
  while (!JobsSubmitted.empty()) {
    std::string ErrMsg;
    RunningJob &Job = JobsSubmitted.front();
    sys::ProcessInfo WaitResult =
        sys::Wait(Job.Process, /*SecondsToWait=*/std::nullopt, &ErrMsg);
    assert(WaitResult.Pid && "sys::Wait should return positive pid");
    if (WaitResult.ReturnCode != 0) {
      printCommandFailure(WaitResult.ReturnCode, ErrMsg, Job.Command);
      Res = WaitResult.ReturnCode;
    }

    JobsSubmitted.pop_front();
  }

  if (!OutputFileList.empty()) {
    OS.close();
  }

  return Res;
}
