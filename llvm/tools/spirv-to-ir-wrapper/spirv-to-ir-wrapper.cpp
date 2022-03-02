//===--- spirv-to-ir-wrapper.cpp - Utility to convert to ir if needed -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This utility checks if the input file is SPIR-V based.  If so, convert to IR
// The input can be either SPIR-V or LLVM-IR.  When LLVM-IR, copy the file to
// the specified output.
//
// Uses llvm-spirv to perform the conversion if needed.
//
// The output file is used to allow for proper input and output flow within
// the driver toolchain.
//
// Usage: spirv-to-ir-wrapper input.spv -o output.bc
//
//===----------------------------------------------------------------------===//

#include "LLVMSPIRVLib.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;

// InputFilename - The filename to read from.
static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::value_desc("<input spv file>"),
                                          cl::desc("<input file>"));

// Output - The filename to output to.
static cl::opt<std::string> Output("o", cl::value_desc("output IR filename"),
                                   cl::desc("output filename"));

// LlvmSpirvOpts - The filename to output to.
static cl::opt<std::string>
    LlvmSpirvOpts("llvm-spirv-opts", cl::value_desc("llvm-spirv options"),
                  cl::desc("options to pass to llvm-spirv"));

// SkipUnknown - Skip unknown files (create empty output instead)
static cl::opt<bool>
    SkipUnknown("skip-unknown-input",
                cl::desc("Only pass through files that are LLVM-IR or "
                         "converted from SPIR-V"));

static void error(const Twine &Message) {
  llvm::errs() << "spirv-to-ir-wrapper: " << Message << '\n';
  exit(1);
}

// Convert the SPIR-V to LLVM-IR.
static int convertSPIRVToLLVMIR(const char *Argv0) {
  // Find llvm-spirv.  It is expected this resides in the same directory
  // as spirv-to-ir-wrapper.
  StringRef ParentPath = llvm::sys::path::parent_path(Argv0);
  llvm::ErrorOr<std::string> LlvmSpirvBinary =
      llvm::sys::findProgramByName("llvm-spirv", ParentPath);
  if (!LlvmSpirvBinary)
    LlvmSpirvBinary = llvm::sys::findProgramByName("llvm-spirv");

  SmallVector<StringRef, 6> LlvmSpirvArgs = {"llvm-spirv", "-r", InputFilename,
                                             "-o", Output};

  // Add any additional options specified by the user.
  SmallVector<const char *, 8> TargetArgs;
  llvm::BumpPtrAllocator BPA;
  llvm::StringSaver S(BPA);
  if (!LlvmSpirvOpts.empty()) {
    // Tokenize the string.
    llvm::cl::TokenizeGNUCommandLine(LlvmSpirvOpts, S, TargetArgs);
    std::copy(TargetArgs.begin(), TargetArgs.end(),
              std::back_inserter(LlvmSpirvArgs));
  }

  return llvm::sys::ExecuteAndWait(LlvmSpirvBinary.get(), LlvmSpirvArgs);
}

static int copyInputToOutput() {
  return llvm::sys::fs::copy_file(InputFilename, Output).value();
}

static int createEmptyOutput() {
  int FD;
  if (std::error_code EC = openFileForWrite(
          Output, FD, sys::fs::CD_CreateAlways, sys::fs::OF_None))
    return EC.value();
  return llvm::sys::Process::SafelyCloseFileDescriptor(FD).value();
}

static bool isSPIRVBinary(const std::string &File) {
  auto FileOrError = MemoryBuffer::getFile(File, /*IsText=*/false,
                                           /*RequiresNullTerminator=*/false);
  if (!FileOrError)
    return false;
  std::unique_ptr<MemoryBuffer> FileBuffer = std::move(*FileOrError);
  return SPIRV::isSpirvBinary(FileBuffer->getBuffer().str());
}

static bool isLLVMIRBinary(const std::string &File) {
  if (File.size() < sizeof(unsigned))
    return false;

  StringRef Ext = llvm::sys::path::has_extension(File)
                      ? llvm::sys::path::extension(File).drop_front()
                      : "";
  llvm::file_magic Magic;
  llvm::identify_magic(File, Magic);

  // Only .bc and bitcode files are to be considered.
  return (Ext == "bc" || Magic == llvm::file_magic::bitcode);
}

static int checkInputFileIsAlreadyLLVM(const char *Argv0) {
  StringRef Ext = llvm::sys::path::has_extension(InputFilename)
                      ? llvm::sys::path::extension(InputFilename).drop_front()
                      : "";
  if (Ext == "bc" || isLLVMIRBinary(InputFilename))
    return copyInputToOutput();
  if (Ext == "spv" || isSPIRVBinary(InputFilename))
    return convertSPIRVToLLVMIR(Argv0);

  if (SkipUnknown)
    return createEmptyOutput();

  // We could not directly determine the input file, so we just copy it
  // to the output file.
  return copyInputToOutput();
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  LLVMContext Context;
  cl::ParseCommandLineOptions(argc, argv, "spirv-to-ir-wrapper\n");

  if (InputFilename.empty())
    error("No input file provided");

  if (!llvm::sys::fs::exists(InputFilename))
    error("Input file \'" + InputFilename + "\' not found");

  if (Output.empty())
    error("Output file not provided");

  return checkInputFileIsAlreadyLLVM(argv[0]);
}
