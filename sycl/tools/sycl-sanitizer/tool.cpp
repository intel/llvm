//==----------------- tool.cpp ---------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

void showHelp() {
  std::cout << "Sample usage: sycl-sanitizer application.exe --arg1 --arg2\n";
}

int launch(const char *Cmd, const std::vector<const char *> &Args,
           const std::vector<const char*> &Env) {
#ifdef _WIN32
  _spawnve(_P_WAIT, Cmd, const_cast<char *const *>(Args.data()),
          const_cast<char *const *>(Env.data()));
  return 0;
#else
  return execve(Cmd, const_cast<char *const *>(Args.data()),
                const_cast<char *const *>(NewEnv.data()));
#endif
}

int main(int argc, char *argv[], char *env[]) {
  if (argc < 2) {
    showHelp();
    return 0;
  }

  if (std::string_view(argv[1]) == "--help") {
    showHelp();
    return 0;
  }

  std::vector<const char *> NewEnv;

  {
    size_t I = 0;
    while (env[I] != nullptr)
      NewEnv.push_back(env[I++]);
  }

#ifdef _WIN32
  NewEnv.push_back("XPTI_FRAMEWORK_DISPATCHER=xptifw.dll");
  NewEnv.push_back("XPTI_SUBSCRIBERS=sycl_sanitizer_collector.dll");
#else
  NewEnv.push_back("XPTI_FRAMEWORK_DISPATCHER=libxptifw.so");
  NewEnv.push_back("XPTI_SUBSCRIBERS=libsycl_sanitizer_collector.so");
#endif
  NewEnv.push_back("XPTI_TRACE_ENABLE=1");
  NewEnv.push_back(nullptr);

  std::vector<const char *> Args;

  for (size_t I = 1; I < static_cast<size_t>(argc); I++)
    Args.push_back(argv[I]);

  Args.push_back(nullptr);

  int Err = launch(argv[1], Args, NewEnv);

  if (Err) {
    std::cerr << "Failed to launch target application. Error code " << Err
              << "\n";
    return Err;
  }

  return 0;
}
