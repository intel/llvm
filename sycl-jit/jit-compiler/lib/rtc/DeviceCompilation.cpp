//==---------------------- DeviceCompilation.cpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeviceCompilation.h"

#include <clang/Basic/Version.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Driver/Compilation.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Tooling/CompilationDatabase.h>
#include <clang/Tooling/Tooling.h>

#include <llvm/IR/Module.h>

namespace {
using namespace clang;
using namespace clang::tooling;
using namespace clang::driver;

struct GetLLVMModuleAction : public ToolAction {
  // Code adapted from `FrontendActionFactory::runInvocation`.
  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *Files,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override {
    assert(!Module && "Action should only be invoked on a single file");

    // Create a compiler instance to handle the actual work.
    CompilerInstance Compiler(std::move(PCHContainerOps));
    Compiler.setInvocation(std::move(Invocation));
    Compiler.setFileManager(Files);

    // Create the compiler's actual diagnostics engine.
    Compiler.createDiagnostics(DiagConsumer, /*ShouldOwnClient=*/false);
    if (!Compiler.hasDiagnostics()) {
      return false;
    }

    Compiler.createSourceManager(*Files);

    // Ignore `Compiler.getFrontendOpts().ProgramAction` (would be `EmitBC`) and
    // create/execute an `EmitLLVMOnlyAction` (= codegen to LLVM module without
    // emitting anything) instead.
    EmitLLVMOnlyAction ELOA;
    const bool Success = Compiler.ExecuteAction(ELOA);
    Files->clearStatCache();
    if (!Success) {
      return false;
    }

    // Take the module and its context to extend the objects' lifetime.
    Module = ELOA.takeModule();
    ELOA.takeLLVMContext();

    return true;
  }

  std::unique_ptr<llvm::Module> Module;
};

} // anonymous namespace

std::unique_ptr<llvm::Module> jit_compiler::compileDeviceCode(
    const char *SYCLSource, View<IncludePair> IncludePairs,
    View<const char *> UserArgs, const char *DPCPPRoot) {

  SmallVector<std::string> CommandLine = {"-fsycl-device-only"};
  // TODO: Allow instrumentation again when device library linking is
  //       implemented.
  CommandLine.push_back("-fno-sycl-instrument-device-code");
  CommandLine.append(UserArgs.begin(), UserArgs.end());
  clang::tooling::FixedCompilationDatabase DB{"./", CommandLine};

  constexpr auto SourcePath = "rtc.cpp";
  clang::tooling::ClangTool Tool{DB, {SourcePath}};

  // Set up in-memory filesystem.
  Tool.mapVirtualFile(SourcePath, SYCLSource);
  for (const auto &IP : IncludePairs) {
    Tool.mapVirtualFile(IP.Path, IP.Contents);
  }

  // Reset argument adjusters to drop the `-fsyntax-only` flag which is added by
  // default by this API.
  Tool.clearArgumentsAdjusters();
  // Then, modify argv[0] and set the resource directory so that the driver
  // picks up the correct SYCL environment.
  Tool.appendArgumentsAdjuster(
      [&DPCPPRoot](const CommandLineArguments &Args,
                   StringRef Filename) -> CommandLineArguments {
        (void)Filename;
        CommandLineArguments NewArgs = Args;
        NewArgs[0] = (Twine(DPCPPRoot) + "/bin/clang++").str();
        NewArgs.push_back((Twine("-resource-dir=") + DPCPPRoot + "/lib/clang/" +
                           Twine(CLANG_VERSION_MAJOR))
                              .str());
        return NewArgs;
      });

  GetLLVMModuleAction Action;
  if (!Tool.run(&Action)) {
    return std::move(Action.Module);
  }

  return {};
}
