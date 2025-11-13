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

#include "clang/Lex/PreprocessorOptions.h"
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
#include <clang/Frontend/PrecompiledPreamble.h>
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
#include <llvm/Support/BinaryByteStream.h>
#include <llvm/Support/BinaryStreamReader.h>
#include <llvm/Support/BinaryStreamWriter.h>
#include <llvm/Support/Caching.h>
#include <llvm/Support/PropertySetIO.h>
#include <llvm/Support/TimeProfiler.h>
#include <llvm/TargetParser/TargetParser.h>

#include <algorithm>
#include <array>
#include <memory>
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
struct AutoPCHError : public ErrorInfo<AutoPCHError> {
public:
  static char ID;

  std::error_code convertToErrorCode() const override {
    assert(false && "AutoPCHError doesn't support convertToErrorCode!");
    return {};
  }

  void log(raw_ostream &OS) const override { OS << "auto-pch error"; }
};

char AutoPCHError::ID = 0;

// This key is the same for both in-memory and persistent auto-pch.
struct auto_pch_key {
  std::string Opts;
  std::string Preamble;

  void update_hasher(BLAKE3 &Hasher) const {
    Hasher.update(Opts);
    Hasher.update(Preamble);
  }

  Error write(raw_pwrite_stream &OS) const {
    AppendingBinaryByteStream Stream(llvm::endianness::little);
    BinaryStreamWriter Writer(Stream);
    if (auto Error = Writer.writeInteger(Opts.size()))
      return Error;
    if (auto Error = Writer.writeFixedString(Opts))
      return Error;
    if (auto Error = Writer.writeInteger(Preamble.size()))
      return Error;
    if (auto Error = Writer.writeFixedString(Preamble))
      return Error;

    OS.SetBuffered();
    for (uint8_t x : Stream.data())
      OS << x;
    return Error::success();
  }

  Error read(llvm::BinaryStreamReader &Reader) {
    (void)AutoPCHError::ID;
    auto ReadStr = [&](std::string &Out) -> Error {
      std::string::size_type StrLen = 0;

      if (auto Err = Reader.readInteger(StrLen))
        return Err;

      if (StrLen >= std::numeric_limits<uint32_t>::max())
        return make_error<AutoPCHError>();

      StringRef Str;
      if (auto Err = Reader.readFixedString(Str, (uint32_t)StrLen))
        return Err;

      Out = Str.str();
      return Error::success();
    };

    if (auto Err = ReadStr(Opts))
      return Err;

    return ReadStr(Preamble);
  }

  friend bool operator==(const auto_pch_key &lhs, const auto_pch_key &rhs) {
    return std::tie(lhs.Opts, lhs.Preamble) == std::tie(rhs.Opts, rhs.Preamble);
  }
  friend bool operator!=(const auto_pch_key &lhs, const auto_pch_key &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const auto_pch_key &lhs, const auto_pch_key &rhs) {
    return std::tie(lhs.Opts, lhs.Preamble) < std::tie(rhs.Opts, rhs.Preamble);
  }
};
} // namespace

template <> struct std::hash<auto_pch_key> {
  size_t operator()(const auto_pch_key &key) const {
    BLAKE3 Hasher;
    key.update_hasher(Hasher);

    // No `std::bit_cast` in c++17, emulate:
    auto Hash = Hasher.result<sizeof(size_t)>();
    size_t Result;
    static_assert(sizeof(Hash) == sizeof(size_t));
    std::memcpy(&Result, &Hash, sizeof(size_t));
    return Result;
  }
};

namespace {
class SYCLToolchain {
  static auto &getToolchainFS() {
    // TODO: For some reason, removing `thread_local` results in data races
    // leading to memory corruption (e.g., ::free() report errors). I'm not sure
    // if that's a bug somewhere in clang tooling/LLVMSupport or intentional
    // limitation (or maybe a bug in this file, but I can't imagine how could
    // that be).
    //
    // For single thread compilation this gives us [almost] the same performance
    // as if there was no `thread_local` so performing very time-consuming
    // investigation wouldn't give a justifiable ROI at this moment.
    static thread_local const auto ToolchainFS = []() {
      llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> ToolchainFS =
          llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
      using namespace jit_compiler::resource;

      for (size_t i = 0; i < NumToolchainFiles; ++i) {
        resource_file RF = ToolchainFiles[i];
        std::string_view Path{RF.Path.S, RF.Path.Size};
        std::string_view Content{RF.Content.S, RF.Content.Size};
        ToolchainFS->addFile(Path, 0,
                             llvm::MemoryBuffer::getMemBuffer(Content));
      }
      return ToolchainFS;
    }();
    return ToolchainFS;
  }

  SYCLToolchain() = default;

  struct PrecompiledPreambles {
    std::mutex Mutex;
    std::map<auto_pch_key, std::shared_ptr<PrecompiledPreamble>> PreamblesMap;
  };

  // Similar to FrontendActionFactory, but we don't take ownership of
  // `FrontendAction`, nor do we create copies of it as we only perform a
  // single `ToolInvocation`.
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
      Compiler.setVirtualFileSystem(Files->getVirtualFileSystemPtr());
      Compiler.setFileManager(Files);
      // Suppress summary with number of warnings and errors being printed to
      // stdout.
      Compiler.setVerboseOutputStream(
          std::make_unique<llvm::raw_null_ostream>());

      // Create the compiler's actual diagnostics engine.
      Compiler.createDiagnostics(DiagConsumer, /*ShouldOwnClient=*/false);
      if (!Compiler.hasDiagnostics())
        return false;

      Compiler.createSourceManager();

      const bool Success = Compiler.ExecuteAction(FEAction);

      Files->clearStatCache();
      return Success;
    }
  };

  std::vector<std::string> createCommandLine(const InputArgList &UserArgList,
                                             BinaryFormat Format,
                                             std::string_view SourceFilePath) {
    DerivedArgList DAL{UserArgList};
    const auto &OptTable = getDriverOptTable();
    DAL.AddFlagArg(nullptr, OptTable.getOption(OPT_fsycl_device_only));
    // User args may contain options not intended for the frontend, but we
    // can't claim them here to tell the driver they're used later. Hence,
    // suppress the unused argument warning.
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
    for (Arg *A : DAL)
      A->render(DAL, ASL);
    for (Arg *A : UserArgList) {
      Option Group = A->getOption().getGroup();
      if (Group.isValid() && Group.getID() == OPT_sycl_rtc_only_Group)
        continue;

      A->render(UserArgList, ASL);
    }

    std::vector<std::string> CommandLine;
    CommandLine.reserve(ASL.size() + 2);
    CommandLine.emplace_back(ClangXXExe);
    transform(ASL, std::back_inserter(CommandLine),
              [](const char *AS) { return std::string{AS}; });
    CommandLine.emplace_back(SourceFilePath);
    return CommandLine;
  }

  template <bool Persistent = false>
  class ActionWithPCHPreamble : public Action {
    std::string CmdLineOpts;
    std::string PersistentPCHDir; // Empty if !Persistent.

    static void addImplicitPersistentPreamble(
        std::unique_ptr<MemoryBuffer> PrecompiledPreamble,
        const PreambleBounds &Bounds, CompilerInvocation &CI,
        IntrusiveRefCntPtr<llvm::vfs::FileSystem> &VFS) {

      // Processing similar to PrecompiledPreamble::configurePreamble.

      auto &PreprocessorOpts = CI.getPreprocessorOpts();
      PreprocessorOpts.PrecompiledPreambleBytes.first = Bounds.Size;
      PreprocessorOpts.PrecompiledPreambleBytes.second =
          Bounds.PreambleEndsAtStartOfLine;
      PreprocessorOpts.DisablePCHOrModuleValidation =
          DisableValidationForModuleKind::PCH;

      std::string PCHPath = (SYCLToolchain::instance().getPrefix() +
                             "/remapped_persistent_preamble")
                                .str();
      PreprocessorOpts.ImplicitPCHInclude = PCHPath;

      auto PCHFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
      PCHFS->addFile(PCHPath, 0, std::move(PrecompiledPreamble));
      auto OverlayFS =
          llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(VFS);
      OverlayFS->pushOverlay(PCHFS);
      VFS = std::move(OverlayFS);
    }

  public:
    ActionWithPCHPreamble(FrontendAction &FEAction, std::string &&CmdLineOpts,
                          std::string PersistentPCHDir = {})
        : Action(FEAction), CmdLineOpts(std::move(CmdLineOpts)),
          PersistentPCHDir(std::move(PersistentPCHDir)) {
      assert(this->PersistentPCHDir.empty() || Persistent);
    }

    bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                       FileManager *Files,
                       std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                       DiagnosticConsumer *DiagConsumer) override {
      auto MainFilePath = Invocation->getFrontendOpts().Inputs[0].getFile();
      auto MainFileBuffer = Files->getBufferForFile(MainFilePath);
      assert(MainFileBuffer && "Can't get memory buffer for in-memory source?");

      PreambleBounds Bounds = ComputePreambleBounds(
          Invocation->getLangOpts(), **MainFileBuffer, 100 /* MaxLines */);

      auto_pch_key key{
          std::move(CmdLineOpts),
          (*MainFileBuffer)->getBuffer().substr(0, Bounds.Size).str()};

      // In-memory for both `Persistent` and not because PrecompiledPreamble's
      // `StorePreamblesInMemory==false` would create a *temporary* pch file
      // on the file system, it will still be removed once preamble object
      // dies.
      auto BuildPreamble = [&]() {
        PreambleCallbacks Callbacks;
        auto DiagIds = llvm::makeIntrusiveRefCnt<DiagnosticIDs>();
        auto DiagOpts = Invocation->getDiagnosticOpts();
        auto Diags = llvm::makeIntrusiveRefCnt<DiagnosticsEngine>(
            DiagIds, DiagOpts, DiagConsumer, false);

        static std::string StoragePath =
            (SYCLToolchain::instance().getPrefix() + "/preambles").str();
        return PrecompiledPreamble::Build(
            *Invocation, MainFileBuffer->get(), Bounds, Diags,
            Files->getVirtualFileSystemPtr(), PCHContainerOps,
            /*StorePreamblesInMemory*/ true, StoragePath, Callbacks,
            /*AllowASTWithErrors=*/false);
      };

      if constexpr (Persistent) {
        BLAKE3 Hasher;
        key.update_hasher(Hasher);

        std::string EncodedHash = encodeBase64(Hasher.result());
        // Make the encoding filesystem-friendly.
        std::replace(EncodedHash.begin(), EncodedHash.end(), '/', '-');

        // `llvm::localCache`'s API uses a callback to process cached data and
        // the callback's return value (if any) is effectively ignored, so we
        // need this extra `Success` variable to be able to properly return
        // compilation status.
        bool Success = false;
        auto RunWithoutPCH = [&]() -> bool {
          // Run original invocation:
          Success =
              Action::runInvocation(std::move(Invocation), Files,
                                    std::move(PCHContainerOps), DiagConsumer);
          return Success;
        };

        auto UseCachedPreamble = [&](StringRef PCHContent) {
          std::unique_ptr<MemoryBuffer> PCHMemBuf =
              MemoryBuffer::getMemBufferCopy(PCHContent);

          auto VFS = Files->getVirtualFileSystemPtr();
          addImplicitPersistentPreamble(std::move(PCHMemBuf), Bounds,
                                        *Invocation, VFS);

          auto NewFiles = makeIntrusiveRefCnt<FileManager>(
              Files->getFileSystemOpts(), std::move(VFS));

          Success =
              Action::runInvocation(std::move(Invocation), NewFiles.get(),
                                    std::move(PCHContainerOps), DiagConsumer);
          return Success;
        };

        // `llvm::localCache` calls the callback on either succesful cache read
        // or during "commit" if an entry is being created. The problem is that
        // commit might fail and the callback won't be called at all. It's
        // easier to just don't rely on it on cache miss and perform compilation
        // with newly generated preamble ourselves.
        bool CacheHit = true;

        auto CacheCallback = [&](size_t, const Twine &,
                                 std::unique_ptr<MemoryBuffer> MB) -> void {
          if (!CacheHit)
            return; // See above.

          llvm::MemoryBufferByteStream MemBufStream{std::move(MB),
                                                    llvm::endianness::little};
          llvm::BinaryStreamReader Reader(MemBufStream);

          auto_pch_key persistent_key;
          // In case of any errors reading the cache, treat it as a hash
          // collision and just compile without using PCH.
          if (errorToBool(persistent_key.read(Reader)))
            return (void)RunWithoutPCH();

          // Hash collision, **very** unlikely.
          if (key != persistent_key)
            return (void)RunWithoutPCH();

          StringRef PCHStorage;

          // This restriction is simply due to the `BinaryStreamReader|Writer`
          // APIs. Pre-compiled preambles in tests seem to be low double digits
          // megabytes which is well under 4GB limit imposed here.
          if (Reader.bytesRemaining() >= std::numeric_limits<uint32_t>::max())
            return (void)RunWithoutPCH();
          if (errorToBool(Reader.readFixedString(
                  PCHStorage, static_cast<uint32_t>(Reader.bytesRemaining()))))
            return (void)RunWithoutPCH();

          return (void)UseCachedPreamble(PCHStorage);
        };

        llvm::Expected<llvm::FileCache> CacheOrErr =
            llvm::localCache("SYCL RTC Persistent Preambles", "syclrtc-tmp-",
                             PersistentPCHDir, CacheCallback);

        assert(CacheOrErr && "Don't see any code path returning Error");
        llvm::Expected<llvm::AddStreamFn> AddStreamOrErr =
            (*CacheOrErr)(0, EncodedHash, "");
        if (!AddStreamOrErr) {
          // Not a hit, but we won't be able to store the data in the cache, so
          // no need to generate precompiled preamble.
          consumeError(AddStreamOrErr.takeError());
          return RunWithoutPCH();
        }
        llvm::AddStreamFn &AddStream = *AddStreamOrErr;
        if (!AddStream) {
          // UseCachedPreamble was called by the cache after successfully
          // reading persistent auto-pch file.
          return Success;
        }
        CacheHit = false;

        llvm::ErrorOr<PrecompiledPreamble> NewPreamble = BuildPreamble();
        if (!NewPreamble) {
          return false;
        }

        // We could have used `NewPreamble`'s `AddImplicitPreamble` (i.e., as on
        // the in-memory/non-persistent path) here but I think it's better to
        // use the same code on cache read/miss:
        UseCachedPreamble(NewPreamble->memoryContents());

        // Any errors updating the persistent preambles cache won't affect
        // current compilation, so ignore any error below:

        llvm::Expected<std::unique_ptr<llvm::CachedFileStream>> FileOrErr =
            AddStream(1, "");
        if (!FileOrErr) {
          consumeError(FileOrErr.takeError());
          return Success;
        }

        llvm::CachedFileStream *CFS = FileOrErr->get();
        raw_pwrite_stream &OS = *CFS->OS;
        consumeError(key.write(OS));

        OS << NewPreamble->memoryContents();

        consumeError(CFS->commit());

        return Success;
      } else {
        std::shared_ptr<PrecompiledPreamble> Preamble;
        {
          PrecompiledPreambles &Preambles = SYCLToolchain::instance().Preambles;
          std::lock_guard<std::mutex> Lock{Preambles.Mutex};
          auto [It, Inserted] = Preambles.PreamblesMap.try_emplace(key);

          if (Inserted) {
            llvm::ErrorOr<PrecompiledPreamble> NewPreamble = BuildPreamble();

            if (!NewPreamble)
              return false;

            It->second = std::make_shared<PrecompiledPreamble>(
                std::move(NewPreamble.get()));
          }

          Preamble = It->second;
        } // End lock

        assert(Preamble);
        assert(Preamble->CanReuse(*Invocation, **MainFileBuffer, Bounds,
                                  Files->getVirtualFileSystem()));

        assert(Invocation->getPreprocessorOpts().RetainRemappedFileBuffers ==
               false);
        // `PreprocessorOptions::RetainRemappedFileBuffers` defaults to false,
        // so MemoryBuffer will be cleaned up by the CompilerInstance, thus
        // `std::unique_ptr::release`.
        auto Buf = llvm::MemoryBuffer::getMemBufferCopy(
                       (*MainFileBuffer)->getBuffer(), MainFilePath)
                       .release();

        auto VFS = Files->getVirtualFileSystemPtr();
        Preamble->AddImplicitPreamble(*Invocation, VFS, Buf);
        auto NewFiles = makeIntrusiveRefCnt<FileManager>(
            Files->getFileSystemOpts(), std::move(VFS));

        return Action::runInvocation(std::move(Invocation), NewFiles.get(),
                                     std::move(PCHContainerOps), DiagConsumer);
      }
    }
  };

public:
  static SYCLToolchain &instance() {
    static SYCLToolchain Instance;
    return Instance;
  }

  bool run(const InputArgList &UserArgList, BinaryFormat Format,
           const char *SourceFilePath, FrontendAction &FEAction,
           IntrusiveRefCntPtr<FileSystem> FSOverlay = nullptr,
           DiagnosticConsumer *DiagConsumer = nullptr,
           bool EnableAutoPCHOpts = false) {
    std::vector<std::string> CommandLine =
        createCommandLine(UserArgList, Format, SourceFilePath);

    auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(
        llvm::vfs::getRealFileSystem());
    FS->pushOverlay(getToolchainFS());
    if (FSOverlay)
      FS->pushOverlay(std::move(FSOverlay));

    auto Files = llvm::makeIntrusiveRefCnt<clang::FileManager>(
        clang::FileSystemOptions{"." /* WorkingDir */}, FS);

    auto Run = [&](auto &Action) {
      ToolInvocation TI{std::move(CommandLine), &Action, Files.get(),
                        std::make_shared<PCHContainerOperations>()};

      TI.setDiagnosticConsumer(DiagConsumer ? DiagConsumer : &IgnoreDiag);

      return TI.run();
    };

    if (!EnableAutoPCHOpts) {
      Action A{FEAction};
      return Run(A);
    }
    if (UserArgList.hasArg(OPT_auto_pch)) {
      // User compilation options must be part of the key in the preambles map.
      // We can either use "raw" user options or the "processed" from
      // `createCommandLine` as long as we're consistent in what we're using.
      // Current internal APIs pass `InputArgList` around instead of a single
      // `std::string`, so it's easier to use `CommandLine`. Just make sure to
      // drop `rtc_N.cpp` that is always different:
      ActionWithPCHPreamble<false /* Persistent */> WithPreamble{
          FEAction, join(drop_end(CommandLine, 1), " ")};
      return Run(WithPreamble);
    }
    if (UserArgList.hasArg(OPT_persistent_auto_pch_EQ)) {
      // The comment above applies here as well.
      ActionWithPCHPreamble<true /* Persistent */> WithPreamble{
          FEAction, join(drop_end(CommandLine, 1), " "),
          UserArgList.getLastArgValue(OPT_persistent_auto_pch_EQ).str()};
      return Run(WithPreamble);
    }

    // Auto-PCH allowed for this FEAction but not requested by the user:
    Action A{FEAction};
    return Run(A);
  }

  Expected<ModuleUPtr> loadBitcodeLibrary(StringRef LibPath,
                                          LLVMContext &Context) {
    auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(
        llvm::vfs::getRealFileSystem());
    FS->pushOverlay(getToolchainFS());

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

  PrecompiledPreambles Preambles;
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

      Hasher.update(PreprocessedSource);
    }

  public:
    HashPreprocessedAction(BLAKE3 &Hasher) : Hasher(Hasher) {}

  private:
    BLAKE3 &Hasher;
  };

  BLAKE3 Hasher;
  HashPreprocessedAction HashAction{Hasher};

  if (!SYCLToolchain::instance().run(UserArgList, Format, SourceFile.Path,
                                     HashAction,
                                     getInMemoryFS(SourceFile, IncludeFiles)))
    return createStringError("Calculating source hash failed");

  Hasher.update(CLANG_VERSION_STRING);
  Hasher.update(
      ArrayRef<uint8_t>{reinterpret_cast<const uint8_t *>(&Format),
                        reinterpret_cast<const uint8_t *>(&Format + 1)});

  for (Arg *Opt : UserArgList) {
    Hasher.update(Opt->getSpelling());
    for (const char *Val : Opt->getValues())
      Hasher.update(Val);
  }

  std::string EncodedHash = encodeBase64(Hasher.result());

  // Make the encoding filesystem-friendly.
  std::replace(EncodedHash.begin(), EncodedHash.end(), '/', '-');
  return std::move(EncodedHash);
}

Expected<ModuleUPtr> jit_compiler::compileDeviceCode(
    InMemoryFile SourceFile, View<InMemoryFile> IncludeFiles,
    const InputArgList &UserArgList, std::string &BuildLog,
    LLVMContext &Context, BinaryFormat Format) {
  TimeTraceScope TTS{"compileDeviceCode"};

  EmitLLVMOnlyAction ELOA{&Context};
  DiagnosticOptions DiagOpts;
  ClangDiagnosticWrapper Wrapper(BuildLog, &DiagOpts);

  if (SYCLToolchain::instance().run(UserArgList, Format, SourceFile.Path, ELOA,
                                    getInMemoryFS(SourceFile, IncludeFiles),
                                    Wrapper.consumer(),
                                    true /* EnableAutoPCHOpts */)) {
    return ELOA.takeModule();
  } else {
    return createStringError(BuildLog);
  }
}

// This function is a simplified copy of the device library selection process
// in `clang::driver::tools::SYCL::getDeviceLibraries`, assuming a SPIR-V, or
// GPU targets (no native CPU). Keep in sync!
static void getDeviceLibraries(const ArgList &Args,
                               SmallVectorImpl<std::string> &LibraryList,
                               BinaryFormat Format) {
  // For CUDA/HIP we only need devicelib, early exit here.
  if (Format == BinaryFormat::PTX) {
    LibraryList.push_back(
        Args.MakeArgString("devicelib-nvptx64-nvidia-cuda.bc"));
    return;
  } else if (Format == BinaryFormat::AMDGCN) {
    LibraryList.push_back(Args.MakeArgString("devicelib-amdgcn-amd-amdhsa.bc"));
    return;
  }

  using SYCLDeviceLibsList = SmallVector<StringRef>;
  const SYCLDeviceLibsList SYCLDeviceLibs = {"libsycl-crt",
                                             "libsycl-complex",
                                             "libsycl-complex-fp64",
                                             "libsycl-cmath",
                                             "libsycl-cmath-fp64",
#if defined(_WIN32)
                                             "libsycl-msvc-math",
#endif
                                             "libsycl-imf",
                                             "libsycl-imf-fp64",
                                             "libsycl-imf-bf16",
                                             "libsycl-fallback-cassert",
                                             "libsycl-fallback-cstring",
                                             "libsycl-fallback-complex",
                                             "libsycl-fallback-complex-fp64",
                                             "libsycl-fallback-cmath",
                                             "libsycl-fallback-cmath-fp64",
                                             "libsycl-fallback-imf",
                                             "libsycl-fallback-imf-fp64",
                                             "libsycl-fallback-imf-bf16"};

  StringRef LibSuffix = ".bc";
  auto AddLibraries = [&](const SYCLDeviceLibsList &LibsList) {
    for (const StringRef &Lib : LibsList) {
      LibraryList.push_back(Args.MakeArgString(Twine(Lib) + LibSuffix));
    }
  };

  AddLibraries(SYCLDeviceLibs);

  const SYCLDeviceLibsList SYCLDeviceAnnotationLibs = {
      "libsycl-itt-user-wrappers", "libsycl-itt-compiler-wrappers",
      "libsycl-itt-stubs"};
  if (Args.hasFlag(OPT_fsycl_instrument_device_code,
                   OPT_fno_sycl_instrument_device_code, false)) {
    AddLibraries(SYCLDeviceAnnotationLibs);
  }
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
  getDeviceLibraries(UserArgList, LibNames, Format);
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
      std::make_unique<ModuleDesc>(std::move(Module)), SplitMode,
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
    std::unique_ptr<ModuleDesc> MDesc = Splitter->nextSplit();

    // TODO: Call `MDesc.fixupLinkageOfDirectInvokeSimdTargets()` when
    //       `invoke_simd` is supported.

    SmallVector<std::unique_ptr<ModuleDesc>, 2> ESIMDSplits =
        splitByESIMD(std::move(MDesc), EmitOnlyKernelsAsEntryPoints,
                     AllowDeviceImageDependencies);
    for (auto &ES : ESIMDSplits) {
      MDesc = std::move(ES);

      if (MDesc->isESIMD()) {
        // `sycl-post-link` has a `-lower-esimd` option, but there's no clang
        // driver option to influence it. Rather, the driver sets it
        // unconditionally in the multi-file output mode, which we are mimicking
        // here.
        lowerEsimdConstructs(*MDesc, PerformOpts);
      }

      MDesc->saveSplitInformationAsMetadata();

      RTCDevImgInfo &DevImgInfo = DevImgInfoVec.emplace_back();
      DevImgInfo.SymbolTable = FrozenSymbolTable{MDesc->entries().size()};
      transform(MDesc->entries(), DevImgInfo.SymbolTable.begin(),
                [](Function *F) { return F->getName(); });

      // TODO: Determine what is requested.
      GlobalBinImageProps PropReq{/*EmitKernelParamInfo=*/true,
                                  /*EmitProgramMetadata=*/true,
                                  /*EmitKernelNames=*/true,
                                  /*EmitExportedSymbols=*/true,
                                  /*EmitImportedSymbols=*/true,
                                  /*DeviceGlobals=*/true};
      PropertySetRegistry Properties =
          computeModuleProperties(MDesc->getModule(), MDesc->entries(), PropReq,
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

      IsBF16DeviceLibUsed |= isSYCLDeviceLibBF16Used(MDesc->getModule());
      Modules.push_back(MDesc->releaseModulePtr());
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
      AL.getLastArg(OPT_Action_Group,       // Actions like -c or -S
                    OPT_Link_Group,         // Linker flags
                    OPT_o,                  // Output file
                    OPT_fsycl_targets_EQ,   // AoT compilation
                    OPT_offload_targets_EQ, // AoT compilation
                    OPT_fsycl_link_EQ,      // SYCL linker
                    OPT_fno_sycl_device_code_split_esimd, // invoke_simd
                    OPT_fsanitize_EQ                      // Sanitizer
      );
  if (UnsupportedArg) {
    return createStringError(
        "Option '%s' is not supported for SYCL runtime compilation",
        UnsupportedArg->getAsString(AL).c_str());
  }

  if (AL.hasArg(OPT_auto_pch) && AL.hasArg(OPT_persistent_auto_pch_EQ)) {
    return createStringError(
        "--auto-pch and --persistent-auto-pch= cannot be used together");
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
