#include "launch.hpp"
#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <string>

using namespace llvm;

enum ModeKind { PI };

int main(int argc, char **argv, char *env[]) {
  cl::opt<ModeKind> Mode(
      "mode", cl::desc("Set tracing mode:"),
      cl::values(
          // TODO graph dot
          clEnumValN(PI, "plugin", "Trace Plugin Interface calls")));
  cl::opt<std::string> TargetExecutable(
      cl::Positional, cl::desc("<target executable>"), cl::Required);
  cl::list<std::string> Argv(cl::ConsumeAfter,
                             cl::desc("<program arguments>..."));

  cl::ParseCommandLineOptions(argc, argv);

  std::vector<const char *> NewEnv;

  {
    size_t I = 0;
    while (env[I] != nullptr)
      NewEnv.push_back(env[I++]);
  }

  NewEnv.push_back("XPTI_FRAMEWORK_DISPATCHER=libxptifw.so");
  NewEnv.push_back("XPTI_SUBSCRIBERS=libsycl_pi_trace_collector.so");
  NewEnv.push_back("XPTI_TRACE_ENABLE=1");
  NewEnv.push_back(nullptr);

  std::vector<const char *> Args;

  Args.push_back(TargetExecutable.c_str());

  for (auto Arg : Argv) {
    Args.push_back(Arg.c_str());
  }

  Args.push_back(nullptr);

  int Err = launch(TargetExecutable.c_str(), Args, NewEnv);

  if (Err) {
    std::cerr << "Failed to launch target application. Error code " << Err
              << "\n";
    return Err;
  }

  return 0;
}
