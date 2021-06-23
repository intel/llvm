//===-------------------- append-file/append-file.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the append-file tool. This tool is intended to be
/// used by the clang driver to create an intermediate source file that is
/// used during the host compilation for a SYCL offloading compile.  This
/// intermediate file consists of the original preprocess source file with
/// an additional 'integration footer' appended to the end.
///
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"

#include <fstream>

using namespace llvm;

static cl::OptionCategory AppendFileCategory("append-file options");

static cl::opt<std::string> Output("output", cl::Required,
                                   cl::desc("output file"),
                                   cl::cat(AppendFileCategory));

static cl::opt<std::string> Input(cl::Positional, cl::Required,
                                  cl::desc("<input file>"),
                                  cl::cat(AppendFileCategory));

static cl::opt<std::string>
    AppendFile("append", cl::ZeroOrMore,
               cl::desc("file which is appended to the input file"),
               cl::cat(AppendFileCategory));

static void error(const Twine &Msg) {
  errs() << "append-file: " << Msg << '\n';
  exit(1);
}

int main(int argc, const char **argv) {
  cl::ParseCommandLineOptions(
      argc, argv,
      "A tool for appending files. Takes input file and appends additional\n"
      "input file.\n");

  // Input file has to exist.
  if (!llvm::sys::fs::exists(Input))
    error("input file not found");

  // Copy the input file to the output file
  llvm::sys::fs::copy_file(Input, Output);
  if (!AppendFile.empty()) {
    // Append the to the output file.
    std::ofstream OutFile(Output, std::ios_base::binary | std::ios_base::app |
                                      std::ios_base::ate);
    std::ifstream FooterFile(AppendFile, std::ios_base::binary);
    OutFile << FooterFile.rdbuf();
    OutFile.close();
    FooterFile.close();
  }

  return 0;
}
