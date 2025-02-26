//==------------ main.cpp - SYCL Tracing Tool ------------------------------==//
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

enum ModeKind { UR, ZE, CU, SYCL, VERIFY };
enum PrintFormatKind { PRETTY_COMPACT, PRETTY_VERBOSE, CLASSIC };

int main(int argc, char **argv, char *env[]) {
  cl::list<ModeKind> Modes(
      cl::desc("Available tracing modes:"),
      cl::values(
          // TODO graph dot
          clEnumValN(UR, "ur.call", "Trace Unified Runtime calls"),
          clEnumValN(ZE, "level_zero", "Trace Level Zero calls"),
          clEnumValN(CU, "cuda", "Trace CUDA Driver API calls"),
          clEnumValN(SYCL, "sycl", "Trace SYCL API calls"),
          clEnumValN(VERIFY, "verify",
                     "Experimental. Verify PI API call parameters")));
  cl::opt<PrintFormatKind> PrintFormat(
      "print-format", cl::desc("Print format"),
      cl::values(
          clEnumValN(PRETTY_COMPACT, "compact", "Human readable compact"),
          clEnumValN(PRETTY_VERBOSE, "verbose", "Human readable verbose"),
          clEnumValN(
              CLASSIC, "classic",
              "Similar to SYCL_PI_TRACE, only compatible with PI layer")));
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

#ifdef __linux__
  NewEnv.push_back("XPTI_FRAMEWORK_DISPATCHER=libxptifw.so");
  NewEnv.push_back("XPTI_SUBSCRIBERS=libsycl_ur_trace_collector.so");
#elif defined(__APPLE__)
  NewEnv.push_back("XPTI_FRAMEWORK_DISPATCHER=libxptifw.dylib");
  NewEnv.push_back("XPTI_SUBSCRIBERS=libsycl_ur_trace_collector.dylib");
#endif
  NewEnv.push_back("XPTI_TRACE_ENABLE=1");

  const auto EnableURTrace = [&]() {
    NewEnv.push_back("SYCL_TRACE_UR_ENABLE=1");
    NewEnv.push_back("UR_ENABLE_LAYERS=UR_LAYER_TRACING");
  };
  const auto EnableZETrace = [&]() {
    NewEnv.push_back("SYCL_TRACE_ZE_ENABLE=1");
    NewEnv.push_back("ZE_ENABLE_TRACING_LAYER=1");
  };
  const auto EnableCUTrace = [&]() {
    NewEnv.push_back("SYCL_TRACE_CU_ENABLE=1");
  };
  const auto EnableSYCLTrace = [&]() {
    NewEnv.push_back("SYCL_TRACE_API_ENABLE=1");
  };
  const auto EnableVerificationTrace = [&]() {
    NewEnv.push_back("SYCL_TRACE_VERIFICATION_ENABLE=1");
    NewEnv.push_back("UR_ENABLE_LAYERS=UR_LAYER_TRACING");
  };

  for (auto Mode : Modes) {
    switch (Mode) {
    case UR:
      EnableURTrace();
      break;
    case ZE:
      EnableZETrace();
      break;
    case CU:
      EnableCUTrace();
      break;
    case SYCL:
      EnableSYCLTrace();
      break;
    case VERIFY:
      EnableVerificationTrace();
      break;
    }
  }

  if (PrintFormat == CLASSIC) {
    NewEnv.push_back("SYCL_TRACE_PRINT_FORMAT=classic");
  } else if (PrintFormat == PRETTY_VERBOSE) {
    NewEnv.push_back("SYCL_TRACE_PRINT_FORMAT=verbose");
  } else {
    NewEnv.push_back("SYCL_TRACE_PRINT_FORMAT=compact");
  }

  if (Modes.size() == 0) {
    EnableURTrace();
    EnableZETrace();
    EnableCUTrace();
    // Intentionally do not enable SYCL API traces -> to not break existing
    // tests.
    // EnableSYTrace();
  }

  std::vector<std::string> Args;

  Args.push_back(TargetExecutable);
  std::copy(Argv.begin(), Argv.end(), std::back_inserter(Args));

  int Err = launch(TargetExecutable.c_str(), Args, NewEnv);

  if (Err) {
    std::cerr << "Failed to launch target application. Error code " << Err
              << "\n";
    return Err;
  }

  return 0;
}
