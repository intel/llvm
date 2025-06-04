//===- DeviceCompilation.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeviceCompilation.h"
#include "ESIMD.h"

#include <clang/Basic/DiagnosticDriver.h>
#include <clang/Basic/Version.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Options.h>
#include <clang/Frontend/ChainedDiagnosticConsumer.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/TextDiagnosticBuffer.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>
#include <clang/Tooling/CompilationDatabase.h>
#include <clang/Tooling/Tooling.h>

#include <llvm/IR/DiagnosticInfo.h>
#include <llvm/IR/DiagnosticPrinter.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/SYCLLowerIR/ESIMD/LowerESIMD.h>
#include <llvm/SYCLLowerIR/LowerInvokeSimd.h>
#include <llvm/SYCLLowerIR/SYCLJointMatrixTransform.h>
#include <llvm/SYCLPostLink/ComputeModuleRuntimeInfo.h>
#include <llvm/SYCLPostLink/ModuleSplitter.h>
#include <llvm/Support/BLAKE3.h>
#include <llvm/Support/Base64.h>
#include <llvm/Support/PropertySetIO.h>
#include <llvm/Support/TimeProfiler.h>

#include <algorithm>
#include <array>
#include <sstream>

using namespace clang;
using namespace clang::tooling;
using namespace clang::driver;
using namespace clang::driver::options;
using namespace llvm;
using namespace llvm::opt;
using namespace llvm::sycl;
using namespace llvm::module_split;
using namespace llvm::util;
using namespace jit_compiler;

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

class HashPreprocessedAction : public PreprocessorFrontendAction {
protected:
  void ExecuteAction() override {
    CompilerInstance &CI = getCompilerInstance();

    std::string PreprocessedSource;
    raw_string_ostream PreprocessStream(PreprocessedSource);

    PreprocessorOutputOptions Opts;
    Opts.ShowCPP = 1;
    Opts.MinimizeWhitespace = 1;
    // Make cache key insensitive to virtual source file and header locations.
    Opts.ShowLineMarkers = 0;

    DoPrintPreprocessedInput(CI.getPreprocessor(), &PreprocessStream, Opts);

    Hash = BLAKE3::hash(arrayRefFromStringRef(PreprocessedSource));
    Executed = true;
  }

public:
  BLAKE3Result<> takeHash() {
    assert(Executed);
    Executed = false;
    return std::move(Hash);
  }

private:
  BLAKE3Result<> Hash;
  bool Executed = false;
};

class RTCToolActionBase : public ToolAction {
public:
  // Code adapted from `FrontendActionFactory::runInvocation`.
  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *Files,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override {
    assert(!hasExecuted() && "Action should only be invoked on a single file");

    // Create a compiler instance to handle the actual work.
    CompilerInstance Compiler(std::move(Invocation), std::move(PCHContainerOps));
    Compiler.setFileManager(Files);
    // Suppress summary with number of warnings and errors being printed to
    // stdout.
    Compiler.setVerboseOutputStream(std::make_unique<llvm::raw_null_ostream>());

    // Create the compiler's actual diagnostics engine.
    Compiler.createDiagnostics(Files->getVirtualFileSystem(), DiagConsumer,
                               /*ShouldOwnClient=*/false);
    if (!Compiler.hasDiagnostics()) {
      return false;
    }

    Compiler.createSourceManager(*Files);

    return executeAction(Compiler, Files);
  }

  virtual ~RTCToolActionBase() = default;

protected:
  virtual bool hasExecuted() = 0;
  virtual bool executeAction(CompilerInstance &, FileManager *) = 0;
};

class GetSourceHashAction : public RTCToolActionBase {
protected:
  bool executeAction(CompilerInstance &CI, FileManager *Files) override {
    HashPreprocessedAction HPA;
    const bool Success = CI.ExecuteAction(HPA);
    Files->clearStatCache();
    if (!Success) {
      return false;
    }

    Hash = HPA.takeHash();
    Executed = true;
    return true;
  }

  bool hasExecuted() override { return Executed; }

public:
  BLAKE3Result<> takeHash() {
    assert(Executed);
    Executed = false;
    return std::move(Hash);
  }

private:
  BLAKE3Result<> Hash;
  bool Executed = false;
};

struct GetLLVMModuleAction : public RTCToolActionBase {
protected:
  bool executeAction(CompilerInstance &CI, FileManager *Files) override {
    // Ignore `Compiler.getFrontendOpts().ProgramAction` (would be `EmitBC`) and
    // create/execute an `EmitLLVMOnlyAction` (= codegen to LLVM module without
    // emitting anything) instead.
    EmitLLVMOnlyAction ELOA{&Context};
    const bool Success = CI.ExecuteAction(ELOA);
    Files->clearStatCache();
    if (!Success) {
      return false;
    }

    // Take the module to extend its lifetime.
    Module = ELOA.takeModule();

    return true;
  }

  bool hasExecuted() override { return static_cast<bool>(Module); }

public:
  GetLLVMModuleAction(LLVMContext &Context) : Context{Context}, Module{} {}
  ModuleUPtr takeModule() {
    assert(Module);
    return std::move(Module);
  }

private:
  LLVMContext &Context;
  ModuleUPtr Module;
};

class ClangDiagnosticWrapper {

  llvm::raw_string_ostream LogStream;

  std::unique_ptr<clang::TextDiagnosticPrinter> LogPrinter;

public:
  ClangDiagnosticWrapper(std::string &LogString, DiagnosticOptions *DiagOpts)
      : LogStream(LogString),
        LogPrinter(
            std::make_unique<TextDiagnosticPrinter>(LogStream, DiagOpts)) {}

  clang::TextDiagnosticPrinter *consumer() { return LogPrinter.get(); }

  llvm::raw_ostream &stream() { return LogStream; }
};

class LLVMDiagnosticWrapper : public llvm::DiagnosticHandler {
  llvm::raw_string_ostream LogStream;

  DiagnosticPrinterRawOStream LogPrinter;

public:
  LLVMDiagnosticWrapper(std::string &BuildLog)
      : LogStream(BuildLog), LogPrinter(LogStream) {}

  bool handleDiagnostics(const DiagnosticInfo &DI) override {
    auto Prefix = [](DiagnosticSeverity Severity) -> llvm::StringLiteral {
      switch (Severity) {
      case llvm::DiagnosticSeverity::DS_Error:
        return "ERROR:";
      case llvm::DiagnosticSeverity::DS_Warning:
        return "WARNING:";
      default:
        return "NOTE:";
      }
    }(DI.getSeverity());
    LogPrinter << Prefix;
    DI.print(LogPrinter);
    LogPrinter << "\n";
    return true;
  }
};

} // anonymous namespace

static void adjustArgs(const InputArgList &UserArgList,
                       const std::string &DPCPPRoot,
                       SmallVectorImpl<std::string> &CommandLine) {
  DerivedArgList DAL{UserArgList};
  const auto &OptTable = getDriverOptTable();
  DAL.AddFlagArg(nullptr, OptTable.getOption(OPT_fsycl_device_only));
  DAL.AddJoinedArg(
      nullptr, OptTable.getOption(OPT_resource_dir_EQ),
      (DPCPPRoot + "/lib/clang/" + Twine(CLANG_VERSION_MAJOR)).str());
  // User args may contain options not intended for the frontend, but we can't
  // claim them here to tell the driver they're used later. Hence, suppress the
  // unused argument warning.
  DAL.AddFlagArg(nullptr, OptTable.getOption(OPT_Qunused_arguments));

  ArgStringList ASL;
  for_each(DAL, [&DAL, &ASL](Arg *A) { A->render(DAL, ASL); });
  for_each(UserArgList,
           [&UserArgList, &ASL](Arg *A) { A->render(UserArgList, ASL); });
  transform(ASL, std::back_inserter(CommandLine),
            [](const char *AS) { return std::string{AS}; });
}

static void setupTool(ClangTool &Tool, const std::string &DPCPPRoot,
                      InMemoryFile SourceFile, View<InMemoryFile> IncludeFiles,
                      DiagnosticConsumer *Consumer) {
  Tool.setDiagnosticConsumer(Consumer);
  // Suppress message "Error while processing" being printed to stdout.
  Tool.setPrintErrorMessage(false);

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
}

Expected<std::string>
jit_compiler::calculateHash(InMemoryFile SourceFile,
                            View<InMemoryFile> IncludeFiles,
                            const InputArgList &UserArgList) {
  TimeTraceScope TTS{"calculateHash"};

  const std::string &DPCPPRoot = getDPCPPRoot();
  if (DPCPPRoot == InvalidDPCPPRoot) {
    return createStringError("Could not locate DPCPP root directory");
  }

  SmallVector<std::string> CommandLine;
  adjustArgs(UserArgList, DPCPPRoot, CommandLine);

  FixedCompilationDatabase DB{".", CommandLine};
  ClangTool Tool{DB, {SourceFile.Path}};

  clang::IgnoringDiagConsumer DiagConsumer;
  setupTool(Tool, DPCPPRoot, SourceFile, IncludeFiles, &DiagConsumer);

  GetSourceHashAction Action;
  if (!Tool.run(&Action)) {
    BLAKE3Result<> SourceHash = Action.takeHash();
    // The adjusted command line contains the DPCPP root and clang major
    // version.
    BLAKE3Result<> CommandLineHash =
        BLAKE3::hash(arrayRefFromStringRef(join(CommandLine, ",")));

    std::string EncodedHash =
        encodeBase64(SourceHash) + encodeBase64(CommandLineHash);
    // Make the encoding filesystem-friendly.
    std::replace(EncodedHash.begin(), EncodedHash.end(), '/', '-');
    return std::move(EncodedHash);
  }

  return createStringError("Calculating source hash failed");
}

Expected<ModuleUPtr>
jit_compiler::compileDeviceCode(InMemoryFile SourceFile,
                                View<InMemoryFile> IncludeFiles,
                                const InputArgList &UserArgList,
                                std::string &BuildLog, LLVMContext &Context) {
  TimeTraceScope TTS{"compileDeviceCode"};

  const std::string &DPCPPRoot = getDPCPPRoot();
  if (DPCPPRoot == InvalidDPCPPRoot) {
    return createStringError("Could not locate DPCPP root directory");
  }

  SmallVector<std::string> CommandLine;
  adjustArgs(UserArgList, DPCPPRoot, CommandLine);

  FixedCompilationDatabase DB{".", CommandLine};
  ClangTool Tool{DB, {SourceFile.Path}};

  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts{new DiagnosticOptions};
  ClangDiagnosticWrapper Wrapper(BuildLog, DiagOpts.get());

  setupTool(Tool, DPCPPRoot, SourceFile, IncludeFiles, Wrapper.consumer());

  GetLLVMModuleAction Action{Context};
  if (!Tool.run(&Action)) {
    return Action.takeModule();
  }

  return createStringError(BuildLog);
}

// This function is a simplified copy of the device library selection process in
// `clang::driver::tools::SYCL::getDeviceLibraries`, assuming a SPIR-V target
// (no AoT, no third-party GPUs, no native CPU). Keep in sync!
static bool getDeviceLibraries(const ArgList &Args,
                               SmallVectorImpl<std::string> &LibraryList,
                               DiagnosticsEngine &Diags) {
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

  bool FoundUnknownLib = false;

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
          FoundUnknownLib = true;
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
#if defined(_WIN32)
      {"libsycl-msvc-math", "libm-fp32"},
#endif
      {"libsycl-imf", "libimf-fp32"},
      {"libsycl-imf-fp64", "libimf-fp64"},
      {"libsycl-imf-bf16", "libimf-bf16"}};
  // ITT annotation libraries are linked in separately whenever the device
  // code instrumentation is enabled.
  const SYCLDeviceLibsList SYCLDeviceAnnotationLibs = {
      {"libsycl-itt-user-wrappers", "internal"},
      {"libsycl-itt-compiler-wrappers", "internal"},
      {"libsycl-itt-stubs", "internal"}};

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

  return FoundUnknownLib;
}

static Expected<ModuleUPtr> loadBitcodeLibrary(StringRef LibPath,
                                               LLVMContext &Context) {
  SMDiagnostic Diag;
  ModuleUPtr Lib = parseIRFile(LibPath, Diag, Context);
  if (!Lib) {
    std::string DiagMsg;
    raw_string_ostream SOS(DiagMsg);
    Diag.print(/*ProgName=*/nullptr, SOS);
    return createStringError(DiagMsg);
  }
  return std::move(Lib);
}

Error jit_compiler::linkDeviceLibraries(llvm::Module &Module,
                                        const InputArgList &UserArgList,
                                        std::string &BuildLog) {
  TimeTraceScope TTS{"linkDeviceLibraries"};

  const std::string &DPCPPRoot = getDPCPPRoot();
  if (DPCPPRoot == InvalidDPCPPRoot) {
    return createStringError("Could not locate DPCPP root directory");
  }

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID{new DiagnosticIDs};
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts{new DiagnosticOptions};
  ClangDiagnosticWrapper Wrapper(BuildLog, DiagOpts.get());
  DiagnosticsEngine Diags(DiagID, DiagOpts, Wrapper.consumer(),
                          /* ShouldOwnClient=*/false);

  SmallVector<std::string> LibNames;
  bool FoundUnknownLib = getDeviceLibraries(UserArgList, LibNames, Diags);
  if (FoundUnknownLib) {
    return createStringError("Could not determine list of device libraries: %s",
                             BuildLog.c_str());
  }

  LLVMContext &Context = Module.getContext();
  for (const std::string &LibName : LibNames) {
    std::string LibPath = DPCPPRoot + "/lib/" + LibName;

    ModuleUPtr LibModule;
    if (auto Error = loadBitcodeLibrary(LibPath, Context).moveInto(LibModule)) {
      return Error;
    }

    if (Linker::linkModules(Module, std::move(LibModule),
                            Linker::LinkOnlyNeeded)) {
      return createStringError("Unable to link device library %s: %s",
                               LibPath.c_str(), BuildLog.c_str());
    }
  }

  return Error::success();
}

template <class PassClass> static bool runModulePass(llvm::Module &M) {
  ModulePassManager MPM;
  ModuleAnalysisManager MAM;
  // Register required analysis
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  MPM.addPass(PassClass{});
  PreservedAnalyses Res = MPM.run(M, MAM);
  return !Res.areAllPreserved();
}

static IRSplitMode getDeviceCodeSplitMode(const InputArgList &UserArgList) {
  // This is the (combined) logic from
  // `get[NonTriple|Triple]BasedSYCLPostLinkOpts` in
  // `clang/lib/Driver/ToolChains/Clang.cpp`: Default is auto mode, but the user
  // can override it by specifying the `-fsycl-device-code-split=` option. The
  // no-argument variant `-fsycl-device-code-split` is ignored.
  if (auto *Arg = UserArgList.getLastArg(OPT_fsycl_device_code_split_EQ)) {
    StringRef ArgVal{Arg->getValue()};
    if (ArgVal == "per_kernel") {
      return SPLIT_PER_KERNEL;
    }
    if (ArgVal == "per_source") {
      return SPLIT_PER_TU;
    }
    if (ArgVal == "off") {
      return SPLIT_NONE;
    }
  }
  return SPLIT_AUTO;
}

static void encodeProperties(PropertySetRegistry &Properties,
                             RTCDevImgInfo &DevImgInfo) {
  const auto &PropertySets = Properties.getPropSets();

  DevImgInfo.Properties = FrozenPropertyRegistry{PropertySets.size()};
  for (auto [KV, FrozenPropSet] :
       zip_equal(PropertySets, DevImgInfo.Properties)) {
    const auto &PropertySetName = KV.first;
    const auto &PropertySet = KV.second;
    FrozenPropSet =
        FrozenPropertySet{PropertySetName.str(), PropertySet.size()};
    for (auto [KV2, FrozenProp] :
         zip_equal(PropertySet, FrozenPropSet.Values)) {
      const auto &PropertyName = KV2.first;
      const auto &PropertyValue = KV2.second;
      FrozenProp = PropertyValue.getType() == PropertyValue::Type::UINT32
                       ? FrozenPropertyValue{PropertyName.str(),
                                             PropertyValue.asUint32()}
                       : FrozenPropertyValue{
                             PropertyName.str(), PropertyValue.asRawByteArray(),
                             PropertyValue.getRawByteArraySize()};
    }
  };
}

Expected<PostLinkResult>
jit_compiler::performPostLink(ModuleUPtr Module,
                              const InputArgList &UserArgList) {
  TimeTraceScope TTS{"performPostLink"};

  // This is a simplified version of `processInputModule` in
  // `llvm/tools/sycl-post-link.cpp`. Assertions/TODOs point to functionality
  // left out of the algorithm for now.

  const auto SplitMode = getDeviceCodeSplitMode(UserArgList);

  const bool AllowDeviceImageDependencies = UserArgList.hasFlag(
      options::OPT_fsycl_allow_device_image_dependencies,
      options::OPT_fno_sycl_allow_device_image_dependencies, false);

  // TODO: EmitOnlyKernelsAsEntryPoints is controlled by
  //       `shouldEmitOnlyKernelsAsEntryPoints` in
  //       `clang/lib/Driver/ToolChains/Clang.cpp`.
  // If we allow device image dependencies, we should definitely not only emit
  // kernels as entry points.
  const bool EmitOnlyKernelsAsEntryPoints = !AllowDeviceImageDependencies;

  // TODO: The optlevel passed to `sycl-post-link` is determined by
  //       `getSYCLPostLinkOptimizationLevel` in
  //       `clang/lib/Driver/ToolChains/Clang.cpp`.
  const bool PerformOpts = true;

  // Propagate ESIMD attribute to wrapper functions to prevent spurious splits
  // and kernel link errors.
  runModulePass<SYCLFixupESIMDKernelWrapperMDPass>(*Module);

  assert(!Module->getGlobalVariable("llvm.used") &&
         !Module->getGlobalVariable("llvm.compiler.used"));
  // Otherwise: Port over the `removeSYCLKernelsConstRefArray` and
  // `removeDeviceGlobalFromCompilerUsed` methods.

  assert(!(isModuleUsingAsan(*Module) || isModuleUsingMsan(*Module) ||
           isModuleUsingTsan(*Module)));
  // Otherwise: Run `SanitizerKernelMetadataPass`.

  // Transform Joint Matrix builtin calls to align them with SPIR-V friendly
  // LLVM IR specification.
  runModulePass<SYCLJointMatrixTransformPass>(*Module);

  // Do invoke_simd processing before splitting because this:
  // - saves processing time (the pass is run once, even though on larger IR)
  // - doing it before SYCL/ESIMD splitting is required for correctness
  if (runModulePass<SYCLLowerInvokeSimdPass>(*Module)) {
    return createStringError("`invoke_simd` calls detected");
  }

  std::unique_ptr<ModuleSplitterBase> Splitter = getDeviceCodeSplitter(
      ModuleDesc{std::move(Module)}, SplitMode,
      /*IROutputOnly=*/false, EmitOnlyKernelsAsEntryPoints,
      AllowDeviceImageDependencies);
  assert(Splitter->hasMoreSplits());

  if (auto Err = Splitter->verifyNoCrossModuleDeviceGlobalUsage()) {
    return std::move(Err);
  }

  SmallVector<RTCDevImgInfo> DevImgInfoVec;
  SmallVector<ModuleUPtr> Modules;

  // TODO: The following logic is missing the ability to link ESIMD and SYCL
  //       modules back together, which would be requested via
  //       `-fno-sycl-device-code-split-esimd` as a prerequisite for compiling
  //       `invoke_simd` code.

  bool IsBF16DeviceLibUsed = false;
  while (Splitter->hasMoreSplits()) {
    ModuleDesc MDesc = Splitter->nextSplit();

    // TODO: Call `MDesc.fixupLinkageOfDirectInvokeSimdTargets()` when
    //       `invoke_simd` is supported.

    SmallVector<ModuleDesc, 2> ESIMDSplits =
        splitByESIMD(std::move(MDesc), EmitOnlyKernelsAsEntryPoints,
                     AllowDeviceImageDependencies);
    for (auto &ES : ESIMDSplits) {
      MDesc = std::move(ES);

      if (MDesc.isESIMD()) {
        // `sycl-post-link` has a `-lower-esimd` option, but there's no clang
        // driver option to influence it. Rather, the driver sets it
        // unconditionally in the multi-file output mode, which we are mimicking
        // here.
        lowerEsimdConstructs(MDesc, PerformOpts);
      }

      MDesc.saveSplitInformationAsMetadata();

      RTCDevImgInfo &DevImgInfo = DevImgInfoVec.emplace_back();
      DevImgInfo.SymbolTable = FrozenSymbolTable{MDesc.entries().size()};
      transform(MDesc.entries(), DevImgInfo.SymbolTable.begin(),
                [](Function *F) { return F->getName(); });

      // TODO: Determine what is requested.
      GlobalBinImageProps PropReq{
          /*EmitKernelParamInfo=*/true, /*EmitProgramMetadata=*/true,
          /*EmitExportedSymbols=*/true, /*EmitImportedSymbols=*/true,
          /*DeviceGlobals=*/true};
      PropertySetRegistry Properties =
          computeModuleProperties(MDesc.getModule(), MDesc.entries(), PropReq,
                                  AllowDeviceImageDependencies);

      // When the split mode is none, the required work group size will be added
      // to the whole module, which will make the runtime unable to launch the
      // other kernels in the module that have different required work group
      // sizes or no required work group sizes. So we need to remove the
      // required work group size metadata in this case.
      if (SplitMode == module_split::SPLIT_NONE) {
        Properties.remove(PropSetRegTy::SYCL_DEVICE_REQUIREMENTS,
                          PropSetRegTy::PROPERTY_REQD_WORK_GROUP_SIZE);
      }

      // TODO: Manually add `compile_target` property as in
      //       `saveModuleProperties`?

      encodeProperties(Properties, DevImgInfo);

      IsBF16DeviceLibUsed |= isSYCLDeviceLibBF16Used(MDesc.getModule());
      Modules.push_back(MDesc.releaseModulePtr());
    }
  }

  if (IsBF16DeviceLibUsed) {
    const std::string &DPCPPRoot = getDPCPPRoot();
    if (DPCPPRoot == InvalidDPCPPRoot) {
      return createStringError("Could not locate DPCPP root directory");
    }

    auto &Ctx = Modules.front()->getContext();
    auto WrapLibraryInDevImg = [&](const std::string &LibName) -> Error {
      std::string LibPath = DPCPPRoot + "/lib/" + LibName;
      ModuleUPtr LibModule;
      if (auto Error = loadBitcodeLibrary(LibPath, Ctx).moveInto(LibModule)) {
        return Error;
      }

      PropertySetRegistry Properties =
          computeDeviceLibProperties(*LibModule, LibName);
      encodeProperties(Properties, DevImgInfoVec.emplace_back());
      Modules.push_back(std::move(LibModule));

      return Error::success();
    };

    if (auto Err = WrapLibraryInDevImg("libsycl-fallback-bfloat16.bc")) {
      return std::move(Err);
    }
    if (auto Err = WrapLibraryInDevImg("libsycl-native-bfloat16.bc")) {
      return std::move(Err);
    }
  }

  assert(DevImgInfoVec.size() == Modules.size());
  RTCBundleInfo BundleInfo;
  BundleInfo.DevImgInfos = DynArray<RTCDevImgInfo>{DevImgInfoVec.size()};
  std::move(DevImgInfoVec.begin(), DevImgInfoVec.end(),
            BundleInfo.DevImgInfos.begin());

  return PostLinkResult{std::move(BundleInfo), std::move(Modules)};
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

  // Check for options that are unsupported because they would interfere with
  // the in-memory pipeline.
  Arg *UnsupportedArg =
      AL.getLastArg(OPT_Action_Group,     // Actions like -c or -S
                    OPT_Link_Group,       // Linker flags
                    OPT_o,                // Output file
                    OPT_fsycl_targets_EQ, // AoT compilation
                    OPT_fsycl_link_EQ,    // SYCL linker
                    OPT_fno_sycl_device_code_split_esimd, // invoke_simd
                    OPT_fsanitize_EQ                      // Sanitizer
      );
  if (UnsupportedArg) {
    return createStringError(
        "Option '%s' is not supported for SYCL runtime compilation",
        UnsupportedArg->getAsString(AL).c_str());
  }

  return std::move(AL);
}

void jit_compiler::encodeBuildOptions(RTCBundleInfo &BundleInfo,
                                      const InputArgList &UserArgList) {
  std::string CompileOptions;
  raw_string_ostream COSOS{CompileOptions};

  for (Arg *A : UserArgList.filtered(OPT_Xs, OPT_Xs_separate)) {
    if (!CompileOptions.empty()) {
      COSOS << ' ';
    }
    if (A->getOption().matches(OPT_Xs)) {
      COSOS << '-';
    }
    COSOS << A->getValue();
  }

  if (!CompileOptions.empty()) {
    BundleInfo.CompileOptions = CompileOptions;
  }
}

void jit_compiler::configureDiagnostics(LLVMContext &Context,
                                        std::string &BuildLog) {
  Context.setDiagnosticHandler(
      std::make_unique<LLVMDiagnosticWrapper>(BuildLog));
}
