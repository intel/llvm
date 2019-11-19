//===--- llvm-no-spir-kernel.cpp - Utility check spir kernel entry point --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This utility checks if the input module contains functions that is a spir
// kernel. Return 0 if no, return 1 if yes. Use of an output file is not
// required for a successful check. It is used to allow for proper input and
// output flow within the driver toolchain.
//
// Usage: llvm-no-spir-kernel input.bc/input.ll -o output.bc/output.ll
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
using namespace llvm;

// InputFilename - The filename to read from.
static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input bitcode file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

// Output - The filename to output to.
static cl::opt<std::string> Output("o",
                                   cl::desc("<output filename>"),
                                   cl::value_desc("filename"));


int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  LLVMContext Context;
  cl::ParseCommandLineOptions(argc, argv, "llvm no spir kernel\n");

  // Use lazy loading, since we only care about function calling convention
  SMDiagnostic Err;
  std::unique_ptr<Module> M = getLazyIRFileModule(InputFilename, Err, Context);

  if (!M.get()) {
    Err.print(argv[0], errs());
    return 1;
  }

  for (auto &F : *M) {
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
      return 1;
    }
  }

  // When given an output file, just copy the input to the output
  if (!Output.empty() && !InputFilename.empty()) {
    llvm::sys::fs::copy_file(InputFilename, Output);
  }

  return 0;
}
