//==------------ main.cpp - SYCL Profiler Tool -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "launch.hpp"
#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <string>

using namespace llvm;

enum OutputFormatKind { JSON };

int main(int argc, char **argv, char *env[]) {
  cl::opt<OutputFormatKind> OutputFormat(
      "format", cl::desc("Set profiler output format:"),
      cl::values(
          // TODO performance summary
          clEnumValN(JSON, "json",
                     "JSON file, compatible with chrome://tracing")));
  cl::opt<std::string> OutputFilename("o", cl::desc("Specify output filename"),
                                      cl::value_desc("filename"), cl::Required);
  cl::opt<std::string> TargetExecutable(
      cl::Positional, cl::desc("<target executable>"), cl::Required);
  cl::list<std::string> Argv(cl::ConsumeAfter,
                             cl::desc("<program arguments>..."));

  cl::ParseCommandLineOptions(argc, argv);

  std::vector<std::string> NewEnv;

  {
    size_t I = 0;
    while (env[I] != nullptr)
      NewEnv.emplace_back(env[I++]);
  }

  std::string ProfOutFile = "SYCL_PROF_OUT_FILE=" + OutputFilename;
  NewEnv.push_back(ProfOutFile);
  NewEnv.push_back("XPTI_FRAMEWORK_DISPATCHER=libxptifw.so");
  NewEnv.push_back("XPTI_SUBSCRIBERS=libsycl_profiler_collector.so");
  NewEnv.push_back("XPTI_TRACE_ENABLE=1");

  std::vector<std::string> Args;

  Args.push_back(TargetExecutable);
  std::copy(Argv.begin(), Argv.end(), std::back_inserter(Args));

  int Err = launch(TargetExecutable, Args, NewEnv);

  if (Err) {
    std::cerr << "Failed to launch target application. Error code " << Err
              << "\n";
    return Err;
  }

  return 0;
}
