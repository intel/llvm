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

static cl::opt<std::string>
    OriginalFile("orig-filename", cl::ZeroOrMore,
                 cl::desc("original filename, when specified is prepended "
                          "as a line directive to the source file"),
                 cl::cat(AppendFileCategory));

static cl::opt<bool>
    UseInclude("use-include", cl::ZeroOrMore,
               cl::desc("appended file is included via #include directive"),
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

  // Open the original source file stream.
  std::ifstream InputFile(Input, std::ios_base::binary);
  // Open the output file stream.
  std::ofstream OutFile(Output, std::ios_base::binary);

  // Add the BOM from the original source (if it exists).  We will handle
  // UTF-8, UTF-16LE and UTF-16BE
  unsigned char char1 = InputFile.get();
  unsigned char char2 = InputFile.get();
  if ((char1 == 0xFF && char2 == 0xFE) || (char1 == 0xFE && char2 == 0xFF)) {
    unsigned char BOM[] = {char1, char2};
    OutFile.write((char *)BOM, sizeof(BOM));
  } else {
    unsigned char char3 = InputFile.get();
    if (char1 == 0xEF && char2 == 0xBB && char3 == 0xBF) {
      unsigned char BOM[] = {char1, char2, char3};
      OutFile.write((char *)BOM, sizeof(BOM));
    } else
      InputFile.seekg(0);
  }

  if (!OriginalFile.empty())
    OutFile << "#line 1 \"" << OriginalFile << "\"\n";

  // Add the original source file contents.
  OutFile << InputFile.rdbuf();
  InputFile.close();

  if (!AppendFile.empty()) {
    if (UseInclude)
      OutFile << "\n#include \"" << AppendFile << "\"\n";
    else {
      // Append the to the output file.
      std::ifstream FooterFile(AppendFile, std::ios_base::binary);
      OutFile << FooterFile.rdbuf();
      FooterFile.close();
    }
  }

  OutFile.close();
  return 0;
}
