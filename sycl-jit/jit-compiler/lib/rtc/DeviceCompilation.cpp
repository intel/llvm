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

#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>

#include <array>

using namespace clang;
using namespace clang::tooling;
using namespace clang::driver;
using namespace llvm;

#ifdef _GNU_SOURCE
#include <dlfcn.h>
static char X; // Dummy symbol, used as an anchor for `dlinfo` below.
#endif

static constexpr auto InvalidDPCPPRoot = "<invalid>";

static const std::string &getDPCPPRoot() {
  thread_local std::string DPCPPRoot;

  if (!DPCPPRoot.empty()) {
    return DPCPPRoot;
  }
  DPCPPRoot = InvalidDPCPPRoot;

#ifdef _GNU_SOURCE
  static constexpr auto JITLibraryPathSuffix = "/lib/libsycl-jit.so";
  Dl_info Info;
  if (dladdr(&X, &Info)) {
    std::string LoadedLibraryPath = Info.dli_fname;
    auto Pos = LoadedLibraryPath.rfind(JITLibraryPathSuffix);
    if (Pos != std::string::npos) {
      DPCPPRoot = LoadedLibraryPath.substr(0, Pos);
    }
  }
#endif // _GNU_SOURCE

  // TODO: Implemenent other means of determining the DPCPP root, e.g.
  //       evaluating the `CMPLR_ROOT` env.

  return DPCPPRoot;
}

namespace {

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

Expected<std::unique_ptr<llvm::Module>>
jit_compiler::compileDeviceCode(InMemoryFile SourceFile,
                                View<InMemoryFile> IncludeFiles,
                                View<const char *> UserArgs) {
  const std::string &DPCPPRoot = getDPCPPRoot();
  if (DPCPPRoot == InvalidDPCPPRoot) {
    return createStringError("Could not locate DPCPP root directory");
  }

  SmallVector<std::string> CommandLine = {"-fsycl-device-only"};
  CommandLine.append(UserArgs.begin(), UserArgs.end());
  FixedCompilationDatabase DB{".", CommandLine};

  ClangTool Tool{DB, {SourceFile.Path}};

  // Set up in-memory filesystem.
  Tool.mapVirtualFile(SourceFile.Path, SourceFile.Contents);
  for (const auto &IF : IncludeFiles) {
    Tool.mapVirtualFile(IF.Path, IF.Contents);
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

  // TODO: Capture compiler errors from the ClangTool.
  return createStringError("Unable to obtain LLVM module");
}

Error jit_compiler::linkDefaultDeviceLibraries(llvm::Module &Module,
                                               View<const char *> UserArgs) {
  // This function mimics the device library selection process
  // `clang::driver::tools::SYCL::getDeviceLibraries`, assuming a SPIR-V target
  // (no AoT, no third-party GPUs, no native CPU).

  bool DeviceInstrumentationEnabled = true;
  for (StringRef UA : UserArgs) {
    // Check instrumentation-related flags (last occurence determines outcome).
    if (UA == "-fno-sycl-instrument-device-code") {
      DeviceInstrumentationEnabled = false;
      continue;
    }
    if (UA == "-fsycl-instrument-device-code") {
      DeviceInstrumentationEnabled = true;
      continue;
    }

    // Issue warning for `-fsycl-device-lib` or `-fno-sycl-device-lib`.
    // TODO: Is it worth supporting these flags? We're using `LinkOnlyNeeded`
    //       mode anyways!
    // TODO: If we keep the warning, it must go into the build log, not onto the
    //       console.
    // TODO: The FrontendAction emits a warning that these flags are unused. We
    //       should probably silence that by removing the argument occurence for
    //       the compilation step.
    if (UA.contains("sycl-device-lib")) {
      errs() << "warning: device library selection with '" << UA
             << "' is ignored\n";
    }

    // TODO: Presence of `-fsanitize=address` would require linking
    //       `libsycl-sanitizer`, but currently compilation fails earlier.
    assert(!UA.contains("-fsanitize=address") && "Device ASAN unsupported");
  }

  const std::string &DPCPPRoot = getDPCPPRoot();
  if (DPCPPRoot == InvalidDPCPPRoot) {
    return createStringError("Could not locate DPCPP root directory");
  }

  constexpr std::array<llvm::StringLiteral, 8> SYCLDeviceWrapperLibs = {
      "libsycl-crt",      "libsycl-complex",    "libsycl-complex-fp64",
      "libsycl-cmath",    "libsycl-cmath-fp64", "libsycl-imf",
      "libsycl-imf-fp64", "libsycl-imf-bf16"};

  constexpr std::array<llvm::StringLiteral, 3> SYCLDeviceAnnotationLibs = {
      "libsycl-itt-user-wrappers", "libsycl-itt-compiler-wrappers",
      "libsycl-itt-stubs"};

  LLVMContext &Context = Module.getContext();
  auto Link = [&](ArrayRef<llvm::StringLiteral> LibNames) -> Error {
    for (const auto &LibName : LibNames) {
      std::string LibPath = (DPCPPRoot + "/lib/" + LibName + ".bc").str();

      SMDiagnostic Diag;
      std::unique_ptr<llvm::Module> Lib = parseIRFile(LibPath, Diag, Context);
      if (!Lib) {
        std::string DiagMsg;
        raw_string_ostream SOS(DiagMsg);
        Diag.print(/*ProgName=*/nullptr, SOS);
        return createStringError(DiagMsg);
      }

      if (Linker::linkModules(Module, std::move(Lib), Linker::LinkOnlyNeeded)) {
        // TODO: `linkModules` always prints errors to the console.
        return createStringError("Unable to link device library: %s",
                                 LibPath.c_str());
      }
    }

    return Error::success();
  };

  if (auto Error = Link(SYCLDeviceWrapperLibs)) {
    return Error;
  }
  if (DeviceInstrumentationEnabled) {
    if (auto Error = Link(SYCLDeviceAnnotationLibs)) {
      return Error;
    }
  }
  return Error::success();
}
