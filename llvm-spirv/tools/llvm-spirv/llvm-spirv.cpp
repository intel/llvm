//===-- llvm-spirv.cpp - The LLVM/SPIR-V translator utility -----*- C++ -*-===//
//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
/// \file
///
///  Common Usage:
///  llvm-spirv          - Read LLVM bitcode from stdin, write SPIR-V to stdout
///  llvm-spirv x.bc     - Read LLVM bitcode from the x.bc file, write SPIR-V
///                        to x.bil file
///  llvm-spirv -r       - Read SPIR-V from stdin, write LLVM bitcode to stdout
///  llvm-spirv -r x.bil - Read SPIR-V from the x.bil file, write SPIR-V to
///                        the x.bc file
///
///  Options:
///      --help   - Output command line options
///
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ToolOutputFile.h"

#ifndef _SPIRV_SUPPORT_TEXT_FMT
#define _SPIRV_SUPPORT_TEXT_FMT
#endif

#include "LLVMSPIRVLib.h"

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>

#define DEBUG_TYPE "spirv"

namespace kExt {
const char SpirvBinary[] = ".spv";
const char SpirvText[] = ".spt";
const char LLVMBinary[] = ".bc";
} // namespace kExt

using namespace llvm;

static cl::opt<std::string> InputFile(cl::Positional, cl::desc("<input file>"),
                                      cl::init("-"));

static cl::opt<std::string> OutputFile("o",
                                       cl::desc("Override output filename"),
                                       cl::value_desc("filename"));

static cl::opt<bool>
    IsReverse("r", cl::desc("Reverse translation (SPIR-V to LLVM)"));

static cl::opt<bool>
    IsRegularization("s",
                     cl::desc("Regularize LLVM to be representable by SPIR-V"));

using SPIRV::VersionNumber;

static cl::opt<VersionNumber> MaxSPIRVVersion(
    "spirv-max-version",
    cl::desc("Choose maximum SPIR-V version which can be emitted"),
    cl::values(clEnumValN(VersionNumber::SPIRV_1_0, "1.0", "SPIR-V 1.0"),
               clEnumValN(VersionNumber::SPIRV_1_1, "1.1", "SPIR-V 1.1")),
    cl::init(VersionNumber::MaximumVersion));

static cl::list<std::string>
    SPVExt("spirv-ext", cl::CommaSeparated,
           cl::desc("Specify list of allowed/disallowed extensions"),
           cl::value_desc("+SPV_extenstion1_name,-SPV_extension2_name"),
           cl::ValueRequired);

using SPIRV::ExtensionID;

#ifdef _SPIRV_SUPPORT_TEXT_FMT
namespace SPIRV {
// Use textual format for SPIRV.
extern bool SPIRVUseTextFormat;
} // namespace SPIRV

static cl::opt<bool>
    ToText("to-text",
           cl::desc("Convert input SPIR-V binary to internal textual format"));

static cl::opt<bool> ToBinary(
    "to-binary",
    cl::desc("Convert input SPIR-V in internal textual format to binary"));
#endif

static std::string removeExt(const std::string &FileName) {
  size_t Pos = FileName.find_last_of(".");
  if (Pos != std::string::npos)
    return FileName.substr(0, Pos);
  return FileName;
}

static ExitOnError ExitOnErr;

static int convertLLVMToSPIRV(const SPIRV::TranslatorOpts &Opts) {
  LLVMContext Context;

  std::unique_ptr<MemoryBuffer> MB =
      ExitOnErr(errorOrToExpected(MemoryBuffer::getFileOrSTDIN(InputFile)));
  std::unique_ptr<Module> M =
      ExitOnErr(getOwningLazyBitcodeModule(std::move(MB), Context,
                                           /*ShouldLazyLoadMetadata=*/true));
  ExitOnErr(M->materializeAll());

  if (OutputFile.empty()) {
    if (InputFile == "-")
      OutputFile = "-";
    else
      OutputFile =
          removeExt(InputFile) +
          (SPIRV::SPIRVUseTextFormat ? kExt::SpirvText : kExt::SpirvBinary);
  }

  std::string Err;
  bool Success = false;
  if (OutputFile != "-") {
    std::ofstream OutFile(OutputFile, std::ios::binary);
    Success = writeSpirv(M.get(), Opts, OutFile, Err);
  } else {
    Success = writeSpirv(M.get(), Opts, std::cout, Err);
  }

  if (!Success) {
    errs() << "Fails to save LLVM as SPIR-V: " << Err << '\n';
    return -1;
  }
  return 0;
}

static int convertSPIRVToLLVM(const SPIRV::TranslatorOpts &Opts) {
  LLVMContext Context;
  std::ifstream IFS(InputFile, std::ios::binary);
  Module *M;
  std::string Err;

  if (!readSpirv(Context, Opts, IFS, M, Err)) {
    errs() << "Fails to load SPIR-V as LLVM Module: " << Err << '\n';
    return -1;
  }

  LLVM_DEBUG(dbgs() << "Converted LLVM module:\n" << *M);

  raw_string_ostream ErrorOS(Err);
  if (verifyModule(*M, &ErrorOS)) {
    errs() << "Fails to verify module: " << ErrorOS.str();
    return -1;
  }

  if (OutputFile.empty()) {
    if (InputFile == "-")
      OutputFile = "-";
    else
      OutputFile = removeExt(InputFile) + kExt::LLVMBinary;
  }

  std::error_code EC;
  ToolOutputFile Out(OutputFile.c_str(), EC, sys::fs::F_None);
  if (EC) {
    errs() << "Fails to open output file: " << EC.message();
    return -1;
  }

  WriteBitcodeToFile(*M, Out.os());
  Out.keep();
  delete M;
  return 0;
}

#ifdef _SPIRV_SUPPORT_TEXT_FMT
static int convertSPIRV() {
  if (ToBinary == ToText) {
    errs() << "Invalid arguments\n";
    return -1;
  }
  std::ifstream IFS(InputFile, std::ios::binary);

  if (OutputFile.empty()) {
    if (InputFile == "-")
      OutputFile = "-";
    else {
      OutputFile = removeExt(InputFile) +
                   (ToBinary ? kExt::SpirvBinary : kExt::SpirvText);
    }
  }

  auto Action = [&](std::ostream &OFS) {
    std::string Err;
    if (!SPIRV::convertSpirv(IFS, OFS, Err, ToBinary, ToText)) {
      errs() << "Fails to convert SPIR-V : " << Err << '\n';
      return -1;
    }
    return 0;
  };
  if (OutputFile == "-")
    return Action(std::cout);

  // Open the output file in binary mode in case we convert text to SPIRV binary
  if (ToBinary) {
    std::ofstream OFS(OutputFile, std::ios::binary);
    return Action(OFS);
  }

  // Convert SPIRV binary to text
  std::ofstream OFS(OutputFile);
  return Action(OFS);
}
#endif

static int regularizeLLVM() {
  LLVMContext Context;

  std::unique_ptr<MemoryBuffer> MB =
      ExitOnErr(errorOrToExpected(MemoryBuffer::getFileOrSTDIN(InputFile)));
  std::unique_ptr<Module> M =
      ExitOnErr(getOwningLazyBitcodeModule(std::move(MB), Context,
                                           /*ShouldLazyLoadMetadata=*/true));
  ExitOnErr(M->materializeAll());

  if (OutputFile.empty()) {
    if (InputFile == "-")
      OutputFile = "-";
    else
      OutputFile = removeExt(InputFile) + ".regularized.bc";
  }

  std::string Err;
  if (!regularizeLlvmForSpirv(M.get(), Err)) {
    errs() << "Fails to save LLVM as SPIR-V: " << Err << '\n';
    return -1;
  }

  std::error_code EC;
  ToolOutputFile Out(OutputFile.c_str(), EC, sys::fs::F_None);
  if (EC) {
    errs() << "Fails to open output file: " << EC.message();
    return -1;
  }

  WriteBitcodeToFile(*M.get(), Out.os());
  Out.keep();
  return 0;
}

static int parseSPVExtOption(
    SPIRV::TranslatorOpts::ExtensionsStatusMap &ExtensionsStatus) {
  // Map name -> id for known extensions
  std::map<std::string, ExtensionID> ExtensionNamesMap;
#define _STRINGIFY(X) #X
#define STRINGIFY(X) _STRINGIFY(X)
#define EXT(X) ExtensionNamesMap[STRINGIFY(X)] = ExtensionID::X;
#include "LLVMSPIRVExtensions.inc"
#undef EXT
#undef STRINGIFY
#undef _STRINGIFY

  // Set the initial state:
  //  - during SPIR-V consumption, assume that any known extension is allowed.
  //  - during SPIR-V generation, assume that any known extension is disallowed.
  //  - during conversion to/from SPIR-V text representation, assume that any
  //    known extension is allowed.
  for (const auto &It : ExtensionNamesMap)
    ExtensionsStatus[It.second] = IsReverse;

  if (SPVExt.empty())
    return 0; // Nothing to do

  for (unsigned i = 0; i < SPVExt.size(); ++i) {
    const std::string &ExtString = SPVExt[i];
    if ('+' != ExtString.front() && '-' != ExtString.front()) {
      errs() << "Invalid value of --spirv-ext, expected format is:\n"
             << "\t--spirv-ext=+EXT_NAME,-EXT_NAME\n";
      return -1;
    }

    auto ExtName = ExtString.substr(1);

    if (ExtName.empty()) {
      errs() << "Invalid value of --spirv-ext, expected format is:\n"
             << "\t--spirv-ext=+EXT_NAME,-EXT_NAME\n";
      return -1;
    }

    bool ExtStatus = ('+' == ExtString.front());
    if ("all" == ExtName) {
      // Update status for all known extensions
      for (const auto &It : ExtensionNamesMap)
        ExtensionsStatus[It.second] = ExtStatus;
    } else {
      // Reject unknown extensions
      const auto &It = ExtensionNamesMap.find(ExtName);
      if (ExtensionNamesMap.end() == It) {
        errs() << "Unknown extension '" << ExtName << "' was specified via "
               << "--spirv-ext option\n";
        return -1;
      }

      ExtensionsStatus[It->second] = ExtStatus;
    }
  }

  return 0;
}

int main(int Ac, char **Av) {
  EnablePrettyStackTrace();
  sys::PrintStackTraceOnErrorSignal(Av[0]);
  PrettyStackTraceProgram X(Ac, Av);

  cl::ParseCommandLineOptions(Ac, Av, "LLVM/SPIR-V translator");

  SPIRV::TranslatorOpts::ExtensionsStatusMap ExtensionsStatus;
  // ExtensionsStatus will be properly initialized and update according to
  // values passed via --spirv-ext option in parseSPVExtOption function.
  int Ret = parseSPVExtOption(ExtensionsStatus);
  if (0 != Ret)
    return Ret;

  SPIRV::TranslatorOpts Opts(MaxSPIRVVersion, ExtensionsStatus);

#ifdef _SPIRV_SUPPORT_TEXT_FMT
  if (ToText && (ToBinary || IsReverse || IsRegularization)) {
    errs() << "Cannot use -to-text with -to-binary, -r, -s\n";
    return -1;
  }

  if (ToBinary && (ToText || IsReverse || IsRegularization)) {
    errs() << "Cannot use -to-binary with -to-text, -r, -s\n";
    return -1;
  }

  if (ToBinary || ToText)
    return convertSPIRV();
#endif

  if (!IsReverse && !IsRegularization)
    return convertLLVMToSPIRV(Opts);

  if (IsReverse && IsRegularization) {
    errs() << "Cannot have both -r and -s options\n";
    return -1;
  }
  if (IsReverse)
    return convertSPIRVToLLVM(Opts);

  if (IsRegularization)
    return regularizeLLVM();

  return 0;
}
