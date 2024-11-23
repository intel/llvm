//==---------------------- DeviceCompilation.cpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeviceCompilation.h"

#include <clang/Basic/DiagnosticDriver.h>
#include <clang/Basic/Version.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Options.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/TextDiagnosticBuffer.h>
#include <clang/Tooling/CompilationDatabase.h>
#include <clang/Tooling/Tooling.h>

#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>

#include <array>

using namespace clang;
using namespace clang::tooling;
using namespace clang::driver;
using namespace clang::driver::options;
using namespace llvm;
using namespace llvm::opt;

#ifdef _GNU_SOURCE
#include <dlfcn.h>
static char X; // Dummy symbol, used as an anchor for `dlinfo` below.
#endif

#ifdef _WIN32
#include <filesystem> // For std::filesystem::path ( C++17 only )
#include <shlwapi.h>  // For PathRemoveFileSpec
#include <windows.h>  // For GetModuleFileName, HMODULE, DWORD, MAX_PATH

// cribbed from sycl/source/detail/os_util.cpp
using OSModuleHandle = intptr_t;
static constexpr OSModuleHandle ExeModuleHandle = -1;
static OSModuleHandle getOSModuleHandle(const void *VirtAddr) {
  HMODULE PhModule;
  DWORD Flag = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
  auto LpModuleAddr = reinterpret_cast<LPCSTR>(VirtAddr);
  if (!GetModuleHandleExA(Flag, LpModuleAddr, &PhModule)) {
    // Expect the caller to check for zero and take
    // necessary action
    return 0;
  }
  if (PhModule == GetModuleHandleA(nullptr))
    return ExeModuleHandle;
  return reinterpret_cast<OSModuleHandle>(PhModule);
}

// cribbed from sycl/source/detail/os_util.cpp
/// Returns an absolute path where the object was found.
std::wstring getCurrentDSODir() {
  wchar_t Path[MAX_PATH];
  auto Handle = getOSModuleHandle(reinterpret_cast<void *>(&getCurrentDSODir));
  DWORD Ret = GetModuleFileName(
      reinterpret_cast<HMODULE>(ExeModuleHandle == Handle ? 0 : Handle), Path,
      MAX_PATH);
  assert(Ret < MAX_PATH && "Path is longer than MAX_PATH?");
  assert(Ret > 0 && "GetModuleFileName failed");
  (void)Ret;

  BOOL RetCode = PathRemoveFileSpec(Path);
  assert(RetCode && "PathRemoveFileSpec failed");
  (void)RetCode;

  return Path;
}
#endif // _WIN32

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

#ifdef _WIN32
  DPCPPRoot = std::filesystem::path(getCurrentDSODir()).parent_path().string();
#endif // _WIN32

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
                                const InputArgList &UserArgList) {
  const std::string &DPCPPRoot = getDPCPPRoot();
  if (DPCPPRoot == InvalidDPCPPRoot) {
    return createStringError("Could not locate DPCPP root directory");
  }

  DerivedArgList DAL{UserArgList};
  const auto &OptTable = getDriverOptTable();
  DAL.AddFlagArg(nullptr, OptTable.getOption(OPT_fsycl_device_only));
  DAL.AddJoinedArg(
      nullptr, OptTable.getOption(OPT_resource_dir_EQ),
      (DPCPPRoot + "/lib/clang/" + Twine(CLANG_VERSION_MAJOR)).str());
  for (auto *Arg : UserArgList) {
    DAL.append(Arg);
  }
  // Remove args that will trigger an unused command line argument warning for
  // the FrontendAction invocation, but are handled later (e.g. during device
  // linking).
  DAL.eraseArg(OPT_fsycl_device_lib_EQ);
  DAL.eraseArg(OPT_fno_sycl_device_lib_EQ);

  SmallVector<std::string> CommandLine;
  for (auto *Arg : DAL) {
    CommandLine.emplace_back(Arg->getAsString(DAL));
  }

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
  // Then, modify argv[0] so that the driver picks up the correct SYCL
  // environment. We've already set the resource directory above.
  Tool.appendArgumentsAdjuster(
      [&DPCPPRoot](const CommandLineArguments &Args,
                   StringRef Filename) -> CommandLineArguments {
        (void)Filename;
        CommandLineArguments NewArgs = Args;
        NewArgs[0] = (Twine(DPCPPRoot) + "/bin/clang++").str();
        return NewArgs;
      });

  GetLLVMModuleAction Action;
  if (!Tool.run(&Action)) {
    return std::move(Action.Module);
  }

  // TODO: Capture compiler errors from the ClangTool.
  return createStringError("Unable to obtain LLVM module");
}

// This function is a simplified copy of the device library selection process in
// `clang::driver::tools::SYCL::getDeviceLibraries`, assuming a SPIR-V target
// (no AoT, no third-party GPUs, no native CPU). Keep in sync!
static SmallVector<std::string, 8>
getDeviceLibraries(const ArgList &Args, DiagnosticsEngine &Diags) {
  struct DeviceLibOptInfo {
    StringRef DeviceLibName;
    StringRef DeviceLibOption;
  };

  // Currently, all SYCL device libraries will be linked by default.
  llvm::StringMap<bool> DeviceLibLinkInfo = {
      {"libc", true},          {"libm-fp32", true},   {"libm-fp64", true},
      {"libimf-fp32", true},   {"libimf-fp64", true}, {"libimf-bf16", true},
      {"libm-bfloat16", true}, {"internal", true}};

  // If -fno-sycl-device-lib is specified, its values will be used to exclude
  // linkage of libraries specified by DeviceLibLinkInfo. Linkage of "internal"
  // libraries cannot be affected via -fno-sycl-device-lib.
  bool ExcludeDeviceLibs = false;

  if (Arg *A = Args.getLastArg(OPT_fsycl_device_lib_EQ,
                               OPT_fno_sycl_device_lib_EQ)) {
    if (A->getValues().size() == 0) {
      Diags.Report(diag::warn_drv_empty_joined_argument)
          << A->getAsString(Args);
    } else {
      if (A->getOption().matches(OPT_fno_sycl_device_lib_EQ)) {
        ExcludeDeviceLibs = true;
      }

      for (StringRef Val : A->getValues()) {
        if (Val == "all") {
          for (const auto &K : DeviceLibLinkInfo.keys()) {
            DeviceLibLinkInfo[K] = (K == "internal") || !ExcludeDeviceLibs;
          }
          break;
        }
        auto LinkInfoIter = DeviceLibLinkInfo.find(Val);
        if (LinkInfoIter == DeviceLibLinkInfo.end() || Val == "internal") {
          Diags.Report(diag::err_drv_unsupported_option_argument)
              << A->getSpelling() << Val;
        }
        DeviceLibLinkInfo[Val] = !ExcludeDeviceLibs;
      }
    }
  }

  using SYCLDeviceLibsList = SmallVector<DeviceLibOptInfo, 5>;

  const SYCLDeviceLibsList SYCLDeviceWrapperLibs = {
      {"libsycl-crt", "libc"},
      {"libsycl-complex", "libm-fp32"},
      {"libsycl-complex-fp64", "libm-fp64"},
      {"libsycl-cmath", "libm-fp32"},
      {"libsycl-cmath-fp64", "libm-fp64"},
      {"libsycl-imf", "libimf-fp32"},
      {"libsycl-imf-fp64", "libimf-fp64"},
      {"libsycl-imf-bf16", "libimf-bf16"}};
  // ITT annotation libraries are linked in separately whenever the device
  // code instrumentation is enabled.
  const SYCLDeviceLibsList SYCLDeviceAnnotationLibs = {
      {"libsycl-itt-user-wrappers", "internal"},
      {"libsycl-itt-compiler-wrappers", "internal"},
      {"libsycl-itt-stubs", "internal"}};

  SmallVector<std::string, 8> LibraryList;
  StringRef LibSuffix = ".bc";
  auto AddLibraries = [&](const SYCLDeviceLibsList &LibsList) {
    for (const DeviceLibOptInfo &Lib : LibsList) {
      if (!DeviceLibLinkInfo[Lib.DeviceLibOption]) {
        continue;
      }
      SmallString<128> LibName(Lib.DeviceLibName);
      llvm::sys::path::replace_extension(LibName, LibSuffix);
      LibraryList.push_back(Args.MakeArgString(LibName));
    }
  };

  AddLibraries(SYCLDeviceWrapperLibs);

  if (Args.hasFlag(OPT_fsycl_instrument_device_code,
                   OPT_fno_sycl_instrument_device_code, false)) {
    AddLibraries(SYCLDeviceAnnotationLibs);
  }

  return LibraryList;
}

Error jit_compiler::linkDeviceLibraries(llvm::Module &Module,
                                        const InputArgList &UserArgList) {
  const std::string &DPCPPRoot = getDPCPPRoot();
  if (DPCPPRoot == InvalidDPCPPRoot) {
    return createStringError("Could not locate DPCPP root directory");
  }

  // TODO: Seems a bit excessive to set up this machinery for one warning and
  //       one error. Rethink when implementing the build log/error reporting as
  //       mandated by the extension.
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID{new DiagnosticIDs};
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts{new DiagnosticOptions};
  TextDiagnosticBuffer *DiagBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, DiagOpts, DiagBuffer);

  auto LibNames = getDeviceLibraries(UserArgList, Diags);
  if (std::distance(DiagBuffer->err_begin(), DiagBuffer->err_end()) > 0) {
    std::string DiagMsg;
    raw_string_ostream SOS{DiagMsg};
    interleave(
        DiagBuffer->err_begin(), DiagBuffer->err_end(),
        [&](const auto &D) { SOS << D.second; }, [&]() { SOS << '\n'; });
    return createStringError("Could not determine list of device libraries: %s",
                             DiagMsg.c_str());
  }
  // TODO: Add warnings to build log.

  LLVMContext &Context = Module.getContext();
  for (const std::string &LibName : LibNames) {
    std::string LibPath = DPCPPRoot + "/lib/" + LibName;

    SMDiagnostic Diag;
    std::unique_ptr<llvm::Module> Lib = parseIRFile(LibPath, Diag, Context);
    if (!Lib) {
      std::string DiagMsg;
      raw_string_ostream SOS(DiagMsg);
      Diag.print(/*ProgName=*/nullptr, SOS);
      return createStringError(DiagMsg);
    }

    if (Linker::linkModules(Module, std::move(Lib), Linker::LinkOnlyNeeded)) {
      // TODO: Obtain detailed error message from the context's diagnostics
      //       handler.
      return createStringError("Unable to link device library: %s",
                               LibPath.c_str());
    }
  }

  return Error::success();
}

Expected<InputArgList>
jit_compiler::parseUserArgs(View<const char *> UserArgs) {
  unsigned MissingArgIndex, MissingArgCount;
  auto UserArgsRef = UserArgs.to<ArrayRef>();
  auto AL = getDriverOptTable().ParseArgs(UserArgsRef, MissingArgIndex,
                                          MissingArgCount);
  if (MissingArgCount) {
    return createStringError(
        "User option '%s' at index %d is missing an argument",
        UserArgsRef[MissingArgIndex], MissingArgIndex);
  }

  // Check for unsupported options.
  // TODO: There are probably more, e.g. requesting non-SPIR-V targets.
  {
    // -fsanitize=address
    bool IsDeviceAsanEnabled = false;
    if (Arg *A = AL.getLastArg(OPT_fsanitize_EQ, OPT_fno_sanitize_EQ)) {
      if (A->getOption().matches(OPT_fsanitize_EQ) &&
          A->getValues().size() == 1) {
        std::string SanitizeVal = A->getValue();
        IsDeviceAsanEnabled = SanitizeVal == "address";
      }
    } else {
      // User can pass -fsanitize=address to device compiler via
      // -Xsycl-target-frontend.
      auto SyclFEArg = AL.getAllArgValues(OPT_Xsycl_frontend);
      IsDeviceAsanEnabled = (std::count(SyclFEArg.begin(), SyclFEArg.end(),
                                        "-fsanitize=address") > 0);
      if (!IsDeviceAsanEnabled) {
        auto SyclFEArgEq = AL.getAllArgValues(OPT_Xsycl_frontend_EQ);
        IsDeviceAsanEnabled =
            (std::count(SyclFEArgEq.begin(), SyclFEArgEq.end(),
                        "-fsanitize=address") > 0);
      }

      // User can also enable asan for SYCL device via -Xarch_device option.
      if (!IsDeviceAsanEnabled) {
        auto DeviceArchVals = AL.getAllArgValues(OPT_Xarch_device);
        for (auto DArchVal : DeviceArchVals) {
          if (DArchVal.find("-fsanitize=address") != std::string::npos) {
            IsDeviceAsanEnabled = true;
            break;
          }
        }
      }
    }

    if (IsDeviceAsanEnabled) {
      return createStringError(
          "Device ASAN is not supported for runtime compilation");
    }
  }

  return Expected<InputArgList>{std::move(AL)};
}
