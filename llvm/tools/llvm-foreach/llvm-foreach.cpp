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
    int ArgNum = -1;
    std::string Prefix;
    std::string Postfix;
  };
  auto CreateArgumentReplace = [](int ArgNum, StringRef Arg,
                                  StringRef Replace) -> ArgumentReplace {
    std::string Prefix = Arg.substr(/*Start*/ 0, Arg.find(Replace));
    std::string Postfix = Arg.substr(Arg.find_last_of(Replace) + 1);
    return {ArgNum, Prefix, Postfix};
  };

  // Find args to replace with filenames from input list.
  std::vector<ArgumentReplace> InReplaceArgs;
  ArgumentReplace OutReplaceArg;
  for (size_t i = 0; i < Args.size(); ++i) {
    for (auto &Replace : Replaces)
      if (Args[i].contains(Replace)) {
        InReplaceArgs.push_back(CreateArgumentReplace(i, Args[i], Replace));
      }

    if (!OutReplace.empty() && Args[i].contains(OutReplace))
      OutReplaceArg = CreateArgumentReplace(i, Args[i], OutReplace);
  }

  // Emit an error if user requested replace output file in the command but
  // replace string is not found.
  if (!OutReplace.empty() && OutReplaceArg.ArgNum < 0)
    error("Couldn't find replace string for output in the command.");

  // Make sure that specified program exists, emit an error if not.
  std::string Prog =
      ExitOnErr(errorOrToExpected(sys::findProgramByName(Args[0])));

  std::vector<std::vector<std::string>> FileLists(LineIterators.size());
  int PrevNumOfLines = 0;
  for (size_t i = 0; i < FileLists.size(); ++i) {
    int NumOfLines = 0;
    for (; !LineIterators[i].is_at_eof(); ++LineIterators[i]) {
      FileLists[i].push_back(LineIterators[i]->str());
      NumOfLines++;
    }
    if (i != 0 && NumOfLines != PrevNumOfLines)
      error("All input file lists must have same number of lines!");
    PrevNumOfLines = NumOfLines;
  }

  std::error_code EC;
  std::string ResOutArg;
  std::vector<std::string> ResInArgs(InReplaceArgs.size());
  std::string ResFileList = "";
  for (size_t j = 0; j != FileLists[0].size(); ++j) {
    for (size_t i = 0; i < InReplaceArgs.size(); ++i) {
      ArgumentReplace CurReplace = InReplaceArgs[i];
      ResInArgs[i] = (Twine(CurReplace.Prefix) + Twine(FileLists[i][j]) +
                      Twine(CurReplace.Postfix))
                         .str();
      Args[CurReplace.ArgNum] = ResInArgs[i];
    }

    SmallString<128> Path;
    if (!OutReplace.empty()) {
      // Create a file for command result. Add file name to output
      // file list if needed.
      std::string TempFileNameBase = sys::path::stem(OutReplace);
      if (OutDirectory.empty())
        EC =
            sys::fs::createTemporaryFile(TempFileNameBase, /*Suffix*/ "", Path);
      else {
        SmallString<128> PathPrefix(OutDirectory);
        llvm::sys::path::append(PathPrefix, TempFileNameBase + "-%%%%%%");
        EC = sys::fs::createUniqueFile(PathPrefix, Path);
      }
      error(EC, "Could not create a file for command output.");

      ResOutArg = (Twine(OutReplaceArg.Prefix) + Twine(Path) +
                   Twine(OutReplaceArg.Postfix))
                      .str();
      Args[OutReplaceArg.ArgNum] = ResOutArg;

      if (!OutputFileList.empty())
        ResFileList = (Twine(ResFileList) + Twine(Path) + Twine("\n")).str();
    }

    std::string ErrMsg;
    // TODO: Add possibility to execute commands in parallel.
    int Result =
        sys::ExecuteAndWait(Prog, Args, /*Env=*/None, /*Redirects=*/None,
                            /*SecondsToWait=*/0, /*MemoryLimit=*/0, &ErrMsg);
    if (Result != 0)
      error(ErrMsg);
  }

  // Save file list if needed.
  if (!OutputFileList.empty()) {
    raw_fd_ostream OS{OutputFileList, EC, sys::fs::OpenFlags::OF_None};
    error(EC, "error opening the file '" + OutputFileList + "'");
    OS.write(ResFileList.data(), ResFileList.size());
    OS.close();
  }

  return 0;
}
