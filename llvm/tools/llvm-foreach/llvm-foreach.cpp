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

#include <vector>

using namespace llvm;

static cl::list<std::string> InputFileLists{"in-file-list", cl::OneOrMore,
                                            cl::desc("<input list of files>"),
                                            cl::value_desc("filename")};

static cl::list<std::string> InputCommandArgs{
    cl::Positional, cl::OneOrMore, cl::desc("<command>"),
    cl::value_desc("command")};

static cl::list<std::string> Replaces{
    "in-replace", cl::OneOrMore,
    cl::desc("Specify input path in input command, this will be replaced with "
             "names read from corresponding input list of files;"),
    cl::value_desc("R")};

static cl::opt<std::string> OutReplace{
    "out-replace",
    cl::desc("Specify output path in input command, this will be replaced with "
             "name of temporary file created for writing command's outputs."),
    cl::init(""), cl::value_desc("R")};

static cl::opt<std::string> OutFilesExt{
    "out-ext",
    cl::desc("Specify extenstion for output files; If unspecified, assume "
             ".out"),
    cl::init("out"), cl::value_desc("R")};

// Emit list of produced files for better integration with other tools.
static cl::opt<std::string> OutputFileList{
    "out-file-list", cl::desc("Specify filename for list of outputs."),
    cl::value_desc("filename"), cl::init("")};

static void error(const Twine &Msg) {
  errs() << "llvm-foreach: " << Msg << '\n';
  exit(1);
}

static void error(std::error_code EC, const Twine &Prefix) {
  if (EC)
    error(Prefix + ": " + EC.message());
}

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
  for (auto &InputFileList : InputFileLists)
    MBs.push_back(ExitOnErr(
        errorOrToExpected(MemoryBuffer::getFileOrSTDIN(InputFileList))));


  SmallVector<StringRef, 8> Args(InputCommandArgs.begin(),
                                 InputCommandArgs.end());

  if (Args.empty())
    error("No command?");

  // Find args to replace with filenames from input list.
  std::vector<int> InReplaceArgs;
  int OutReplaceArg = -1;
  for (size_t i = 0; i < Args.size(); ++i) {
    StringRef PossibleReplace;
    if (Args[i].contains("=")) {
      std::pair<StringRef, StringRef> Split = Args[i].split("=");
      PossibleReplace = Split.second;
    } else
      PossibleReplace = Args[i];

    for (auto &Replace : Replaces)
      if (PossibleReplace == Replace)
        InReplaceArgs.push_back(i);

    if (PossibleReplace == OutReplace)
      OutReplaceArg = i;
  }

  // Emit an error if user requested replace output file in the command but
  // replace string is not found.
  if (!OutReplace.empty() && OutReplaceArg < 0)
    error("Couldn't find replace string for output in the command");

  // Make sure that specified program exists, emit an error if not.
  std::string Prog =
      ExitOnErr(errorOrToExpected(sys::findProgramByName(Args[0])));

  std::vector<line_iterator> LineIterators;
  for (auto &MB : MBs)
    LineIterators.push_back(line_iterator(*MB));


  std::error_code EC;
  // TODO: find a way to check that all file lists have same number of lines
  line_iterator LI = LineIterators[0];
  std::vector<std::string> FileNames(LineIterators.size());
  std::string ResOutArg;
  std::vector<std::string> ResInArgs(InReplaceArgs.size());
  std::string FileList = "";
  for (; !LI.is_at_eof(); ++LI) {
    for (int i = 0; i < FileNames.size(); ++i) {
      FileNames[i] = (LineIterators[i]->str());
      ++LineIterators[i];
    }

    for (int i = 0; i < InReplaceArgs.size(); ++i) {
      if (Args[InReplaceArgs[i]].contains("=")) {
        std::pair<StringRef, StringRef> Split =
            Args[InReplaceArgs[i]].split("=");
        ResInArgs[i] =
            Twine((Split.first) + Twine("=") + Twine(FileNames[i])).str();
        Args[InReplaceArgs[i]] = ResInArgs[i];
      } else
        ResInArgs[i] = FileNames[i];
      Args[InReplaceArgs[i]] = ResInArgs[i];
    }

    SmallString<128> Path;
    if (!OutReplace.empty()) {
      // Create a temporary file for command result. Add file name to output
      // file list if needed.
      std::string TempFileNameBase = sys::path::stem(OutReplace);
      EC = sys::fs::createTemporaryFile(TempFileNameBase, OutFilesExt, Path);
      error(EC, "Could not create a temporary file");
      if (Args[OutReplaceArg].contains("=")) {
        std::pair<StringRef, StringRef> Split = Args[OutReplaceArg].split("=");
        ResOutArg = (Twine(Split.first) + Twine("=") + Twine(Path)).str();
      } else
        ResOutArg = Path;
      Args[OutReplaceArg] = ResOutArg;

      if (!OutputFileList.empty())
        FileList = (Twine(FileList) + Twine(Path) + Twine("\n")).str();
    }

    std::string ErrMsg;
    // TODO: Add possibility to execute commands in parallel.
    int Result =
        sys::ExecuteAndWait(Prog, Args, /*Env=*/None, /*Redirects=*/None,
                            /*SecondsToWait=*/0, /*MemoryLimit=*/0, &ErrMsg);
    if (Result < 0)
      error(ErrMsg);
  }

  // Save file list if needed.
  if (!OutputFileList.empty()) {
    raw_fd_ostream OS{OutputFileList, EC, sys::fs::OpenFlags::OF_None};
    error(EC, "error opening the file '" + OutputFileList + "'");
    OS.write(FileList.data(), FileList.size());
    OS.close();
  }

  return 0;
}
