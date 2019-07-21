//===--- llvm-no-spir-kernel.cpp - Utility check spir kernel entry point --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This utility checks if the input module contains functions that is a spir
// kernel. Return 0 if no, return 1 if yes.
// Usage: llvm-no-spir-kernel input.bc/input.ll
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

  return 0;
}