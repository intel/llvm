//===- DeviceCompilation.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DeviceCompilation.h"
#include "ESIMD.h"
#include "JITBinaryInfo.h"
#include "Resource.h"
#include "translation/Translation.h"

#include <clang/Basic/DiagnosticDriver.h>
#include <clang/Basic/Version.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/CudaInstallationDetector.h>
#include <clang/Driver/Driver.h>
#include <clang/Driver/LazyDetector.h>
#include <clang/Driver/Options.h>
#include <clang/Driver/RocmInstallationDetector.h>
#include <clang/Driver/ToolChain.h>
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
#include <llvm/TargetParser/TargetParser.h>

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
using namespace llvm::vfs;
using namespace jit_compiler;

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

class SYCLToolchain {
  SYCLToolchain() {
    using namespace jit_compiler::resource;

    for (size_t i = 0; i < NumToolchainFiles; ++i) {
      resource_file RF = ToolchainFiles[i];
      std::string_view Path{RF.Path.S, RF.Path.Size};
      std::string_view Content{RF.Content.S, RF.Content.Size};
      ToolchainFS->addFile(Path, 0, llvm::MemoryBuffer::getMemBuffer(Content));
    }
  }

  // Similar to FrontendActionFactory, but we don't take ownership of
  // `FrontendAction`, nor do we create copies of it as we only perform a single
  // `ToolInvocation`.
  class Action : public ToolAction {
    FrontendAction &FEAction;

  public:
    Action(FrontendAction &FEAction) : FEAction(FEAction) {}
    ~Action() override = default;

    // Code adapted from `FrontendActionFactory::runInvocation`:
    bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                       FileManager *Files,
                       std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                       DiagnosticConsumer *DiagConsumer) override {
      // Create a compiler instance to handle the actual work.
      CompilerInstance Compiler(std::move(Invocation),
                                std::move(PCHContainerOps));
      Compiler.setFileManager(Files);
      // Suppress summary with number of warnings and errors being printed to
      // stdout.
      Compiler.setVerboseOutputStream(
          std::make_unique<llvm::raw_null_ostream>());

      // Create the compiler's actual diagnostics engine.
      Compiler.createDiagnostics(DiagConsumer, /*ShouldOwnClient=*/false);
      if (!Compiler.hasDiagnostics())
        return false;

      Compiler.createSourceManager(*Files);

      const bool Success = Compiler.ExecuteAction(FEAction);

      Files->clearStatCache();
      return Success;
    }
  };

public:
  static SYCLToolchain &instance() {
    static SYCLToolchain Instance;
    return Instance;
  }

  bool run(const std::vector<std::string> &CommandLine,
           FrontendAction &FEAction,
           IntrusiveRefCntPtr<FileSystem> FSOverlay = nullptr,
           DiagnosticConsumer *DiagConsumer = nullptr) {
    auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(
        llvm::vfs::getRealFileSystem());
    FS->pushOverlay(ToolchainFS);
    if (FSOverlay)
      FS->pushOverlay(FSOverlay);

    auto Files = llvm::makeIntrusiveRefCnt<clang::FileManager>(
        clang::FileSystemOptions{"." /* WorkingDir */}, FS);

    Action A{FEAction};
    ToolInvocation TI{CommandLine, &A, Files.get(),
                      std::make_shared<PCHContainerOperations>()};
    TI.setDiagnosticConsumer(DiagConsumer ? DiagConsumer : &IgnoreDiag);

    return TI.run();
  }

  Expected<ModuleUPtr> loadBitcodeLibrary(StringRef LibPath,
                                          LLVMContext &Context) {
    auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(
        llvm::vfs::getRealFileSystem());
    FS->pushOverlay(ToolchainFS);

    auto MemBuf = FS->getBufferForFile(LibPath, /*FileSize*/ -1,
                                       /*RequiresNullTerminator*/ false);
    if (!MemBuf) {
      return createStringError("Error opening file %s: %s", LibPath.data(),
                               MemBuf.getError().message().c_str());
    }

    SMDiagnostic Diag;
    ModuleUPtr Lib = parseIR(*MemBuf->get(), Diag, Context);
    if (!Lib) {
      std::string DiagMsg;
      raw_string_ostream SOS(DiagMsg);
      Diag.print(/*ProgName=*/nullptr, SOS);
      return createStringError(DiagMsg);
    }
    return std::move(Lib);
  }

  std::string_view getPrefix() const { return Prefix; }
  std::string_view getClangXXExe() const { return ClangXXExe; }

private:
  clang::IgnoringDiagConsumer IgnoreDiag;
  std::string_view Prefix{jit_compiler::resource::ToolchainPrefix.S,
                          jit_compiler::resource::ToolchainPrefix.Size};
  std::string ClangXXExe = (Prefix + "/bin/clang++").str();
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> ToolchainFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
};

class ClangDiagnosticWrapper {

  llvm::raw_string_ostream LogStream;

  std::unique_ptr<clang::TextDiagnosticPrinter> LogPrinter;

public:
  ClangDiagnosticWrapper(std::string &LogString, DiagnosticOptions *DiagOpts)
      : LogStream(LogString),
        LogPrinter(
            std::make_unique<TextDiagnosticPrinter>(LogStream, *DiagOpts)) {}

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

static std::vector<std::string>
createCommandLine(const InputArgList &UserArgList, BinaryFormat Format,
                  std::string_view SourceFilePath) {
  DerivedArgList DAL{UserArgList};
  const auto &OptTable = getDriverOptTable();
  DAL.AddFlagArg(nullptr, OptTable.getOption(OPT_fsycl_device_only));
  // User args may contain options not intended for the frontend, but we can't
  // claim them here to tell the driver they're used later. Hence, suppress the
  // unused argument warning.
  DAL.AddFlagArg(nullptr, OptTable.getOption(OPT_Qunused_arguments));

  if (Format == BinaryFormat::PTX || Format == BinaryFormat::AMDGCN) {
    auto [CPU, Features] =
        Translator::getTargetCPUAndFeatureAttrs(nullptr, "", Format);
    (void)Features;
    StringRef OT = Format == BinaryFormat::PTX ? "nvptx64-nvidia-cuda"
                                               : "amdgcn-amd-amdhsa";
    DAL.AddJoinedArg(nullptr, OptTable.getOption(OPT_fsycl_targets_EQ), OT);
    DAL.AddJoinedArg(nullptr, OptTable.getOption(OPT_Xsycl_backend_EQ), OT);
    DAL.AddJoinedArg(nullptr, OptTable.getOption(OPT_offload_arch_EQ), CPU);
  }

  ArgStringList ASL;
  for_each(DAL, [&DAL, &ASL](Arg *A) { A->render(DAL, ASL); });
  for_each(UserArgList,
           [&UserArgList, &ASL](Arg *A) { A->render(UserArgList, ASL); });

  std::vector<std::string> CommandLine;
  CommandLine.reserve(ASL.size() + 2);
  CommandLine.emplace_back(SYCLToolchain::instance().getClangXXExe());
  transform(ASL, std::back_inserter(CommandLine),
            [](const char *AS) { return std::string{AS}; });
  CommandLine.emplace_back(SourceFilePath);
  return CommandLine;
}

static llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem>
getInMemoryFS(InMemoryFile SourceFile, View<InMemoryFile> IncludeFiles) {
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();

  InMemoryFS->setCurrentWorkingDirectory(
      *llvm::vfs::getRealFileSystem()->getCurrentWorkingDirectory());

  InMemoryFS->addFile(SourceFile.Path, 0,
                      llvm::MemoryBuffer::getMemBuffer(SourceFile.Contents));
  for (InMemoryFile F : IncludeFiles)
    InMemoryFS->addFile(F.Path, 0,
                        llvm::MemoryBuffer::getMemBuffer(F.Contents));

  return InMemoryFS;
}

Expected<std::string> jit_compiler::calculateHash(
    InMemoryFile SourceFile, View<InMemoryFile> IncludeFiles,
    const InputArgList &UserArgList, BinaryFormat Format) {
  TimeTraceScope TTS{"calculateHash"};

  std::vector<std::string> CommandLine =
      createCommandLine(UserArgList, Format, SourceFile.Path);

  HashPreprocessedAction HashAction;

  if (SYCLToolchain::instance().run(CommandLine, HashAction,
                                    getInMemoryFS(SourceFile, IncludeFiles))) {
    BLAKE3Result<> SourceHash = HashAction.takeHash();
    // Last argument is the source file in the format `rtc_N.cpp` which is
    // unique for each query, so drop it:
    CommandLine.pop_back();

    // TODO: Include hash of the current libsycl-jit.so/.dll somehow...
    BLAKE3Result<> CommandLineHash =
        BLAKE3::hash(arrayRefFromStringRef(join(CommandLine, ",")));

    std::string EncodedHash =
        encodeBase64(SourceHash) + encodeBase64(CommandLineHash);
    // Make the encoding filesystem-friendly.
    std::replace(EncodedHash.begin(), EncodedHash.end(), '/', '-');
    return std::move(EncodedHash);

  } else {
    return createStringError("Calculating source hash failed");
  }
}

Expected<ModuleUPtr> jit_compiler::compileDeviceCode(
    InMemoryFile SourceFile, View<InMemoryFile> IncludeFiles,
    const InputArgList &UserArgList, std::string &BuildLog,
    LLVMContext &Context, BinaryFormat Format) {
  TimeTraceScope TTS{"compileDeviceCode"};

  EmitLLVMOnlyAction ELOA{&Context};
  DiagnosticOptions DiagOpts;
  ClangDiagnosticWrapper Wrapper(BuildLog, &DiagOpts);

  if (SYCLToolchain::instance().run(
          createCommandLine(UserArgList, Format, SourceFile.Path), ELOA,
          getInMemoryFS(SourceFile, IncludeFiles), Wrapper.consumer())) {
    return ELOA.takeModule();
  } else {
    return createStringError(BuildLog);
  }
}

// This function is a simplified copy of the device library selection process
// in `clang::driver::tools::SYCL::getDeviceLibraries`, assuming a SPIR-V, or
// GPU targets (no AoT, no native CPU). Keep in sync!
static bool getDeviceLibraries(const ArgList &Args,
                               SmallVectorImpl<std::string> &LibraryList,
                               DiagnosticsEngine &Diags, BinaryFormat Format) {
  // For CUDA/HIP we only need devicelib, early exit here.
  if (Format == BinaryFormat::PTX) {
    LibraryList.push_back(
        Args.MakeArgString("devicelib-nvptx64-nvidia-cuda.bc"));
    return false;
  } else if (Format == BinaryFormat::AMDGCN) {
    LibraryList.push_back(Args.MakeArgString("devicelib-amdgcn-amd-amdhsa.bc"));
    return false;
  }

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

Error jit_compiler::linkDeviceLibraries(llvm::Module &Module,
                                        const InputArgList &UserArgList,
                                        std::string &BuildLog,
                                        BinaryFormat Format) {
  TimeTraceScope TTS{"linkDeviceLibraries"};

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID{new DiagnosticIDs};
  DiagnosticOptions DiagOpts;
  ClangDiagnosticWrapper Wrapper(BuildLog, &DiagOpts);
  DiagnosticsEngine Diags(DiagID, DiagOpts, Wrapper.consumer(),
                          /* ShouldOwnClient=*/false);

  SmallVector<std::string> LibNames;
  const bool FoundUnknownLib =
      getDeviceLibraries(UserArgList, LibNames, Diags, Format);
  if (FoundUnknownLib) {
    return createStringError("Could not determine list of device libraries: %s",
                             BuildLog.c_str());
  }
  const bool IsCudaHIP =
      Format == BinaryFormat::PTX || Format == BinaryFormat::AMDGCN;
  if (IsCudaHIP) {
    // Based on the OS and the format decide on the version of libspirv.
    // NOTE: this will be problematic if cross-compiling between OSes.
    std::string Libclc{"clc/"};
    Libclc.append(
#ifdef _WIN32
        "remangled-l32-signed_char.libspirv-"
#else
        "remangled-l64-signed_char.libspirv-"
#endif
    );
    Libclc.append(Format == BinaryFormat::PTX ? "nvptx64-nvidia-cuda.bc"
                                              : "amdgcn-amd-amdhsa.bc");
    LibNames.push_back(Libclc);
  }

  LLVMContext &Context = Module.getContext();
  for (const std::string &LibName : LibNames) {
    std::string LibPath =
        (SYCLToolchain::instance().getPrefix() + "/lib/" + LibName).str();

    ModuleUPtr LibModule;
    if (auto Error = SYCLToolchain::instance()
                         .loadBitcodeLibrary(LibPath, Context)
                         .moveInto(LibModule)) {
      return Error;
    }

    if (Linker::linkModules(Module, std::move(LibModule),
                            Linker::LinkOnlyNeeded)) {
      return createStringError("Unable to link device library %s: %s",
                               LibPath.c_str(), BuildLog.c_str());
    }
  }

  // For GPU targets we need to link against vendor provided libdevice.
  if (IsCudaHIP) {
    Triple T{Module.getTargetTriple()};
    Driver D{(SYCLToolchain::instance().getPrefix() + "/bin/clang++").str(),
             T.getTriple(), Diags};
    auto [CPU, Features] =
        Translator::getTargetCPUAndFeatureAttrs(&Module, "", Format);
    (void)Features;
    // Helper lambda to link modules.
    auto LinkInLib = [&](const StringRef LibDevice) -> Error {
      ModuleUPtr LibDeviceModule;
      if (auto Error = SYCLToolchain::instance()
                           .loadBitcodeLibrary(LibDevice, Context)
                           .moveInto(LibDeviceModule)) {
        return Error;
      }
      if (Linker::linkModules(Module, std::move(LibDeviceModule),
                              Linker::LinkOnlyNeeded)) {
        return createStringError("Unable to link libdevice: %s",
                                 BuildLog.c_str());
      }
      return Error::success();
    };
    SmallVector<std::string, 12> LibDeviceFiles;
    if (Format == BinaryFormat::PTX) {
      // For NVPTX we can get away with CudaInstallationDetector.
      LazyDetector<CudaInstallationDetector> CudaInstallation{D, T,
                                                              UserArgList};
      auto LibDevice = CudaInstallation->getLibDeviceFile(CPU);
      if (LibDevice.empty()) {
        return createStringError("Unable to find Cuda libdevice");
      }
      LibDeviceFiles.push_back(LibDevice);
    } else {
      LazyDetector<RocmInstallationDetector> RocmInstallation{D, T,
                                                              UserArgList};
      RocmInstallation->detectDeviceLibrary();
      StringRef CanonArch =
          llvm::AMDGPU::getArchNameAMDGCN(llvm::AMDGPU::parseArchAMDGCN(CPU));
      StringRef LibDeviceFile = RocmInstallation->getLibDeviceFile(CanonArch);
      auto CommonBCLibs = RocmInstallation->getCommonBitcodeLibs(
          UserArgList, LibDeviceFile, CPU, Action::OFK_SYCL,
          /*NeedsASanRT=*/false);
      if (CommonBCLibs.empty()) {
        return createStringError("Unable to find ROCm bitcode libraries");
      }
      for (auto &Lib : CommonBCLibs) {
        LibDeviceFiles.push_back(Lib.Path);
      }
    }
    for (auto &LibDeviceFile : LibDeviceFiles) {
      // llvm::Error converts to false on success.
      if (auto Error = LinkInLib(LibDeviceFile)) {
        return Error;
      }
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
      GlobalBinImageProps PropReq{/*EmitKernelParamInfo=*/true,
                                  /*EmitProgramMetadata=*/true,
                                  /*EmitKernelNames=*/true,
                                  /*EmitExportedSymbols=*/true,
                                  /*EmitImportedSymbols=*/true,
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
    auto &Ctx = Modules.front()->getContext();
    auto WrapLibraryInDevImg = [&](const std::string &LibName) -> Error {
      std::string LibPath =
          (SYCLToolchain::instance().getPrefix() + "/lib/" + LibName).str();
      ModuleUPtr LibModule;
      if (auto Error = SYCLToolchain::instance()
                           .loadBitcodeLibrary(LibPath, Ctx)
                           .moveInto(LibModule)) {
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
