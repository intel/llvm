//==-------------------------- KernelFusion.cpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "KernelFusion.h"
#include "Kernel.h"
#include "Options.h"
#include "fusion/FusionPipeline.h"
#include "helper/ConfigHelper.h"
#include "rtc/DeviceCompilation.h"
#include "translation/KernelTranslation.h"
#include "translation/SPIRVLLVMTranslation.h"

#include <llvm/ADT/StringExtras.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TimeProfiler.h>

#include <clang/Driver/Options.h>

#include <chrono>
#include <sstream>

using namespace jit_compiler;

static std::string formatError(llvm::Error &&Err, const std::string &Msg) {
  std::stringstream ErrMsg;
  ErrMsg << Msg << "\nDetailed information:\n";
  llvm::handleAllErrors(std::move(Err),
                        [&ErrMsg](const llvm::StringError &StrErr) {
                          ErrMsg << "\t" << StrErr.getMessage() << "\n";
                        });
  return ErrMsg.str();
}

template <typename ResultType, typename... ExtraArgTypes>
static ResultType errorTo(llvm::Error &&Err, const std::string &Msg,
                          ExtraArgTypes... ExtraArgs) {
  // Cannot throw an exception here if LLVM itself is compiled without exception
  // support.
  return ResultType{formatError(std::move(Err), Msg).c_str(), ExtraArgs...};
}

extern "C" KF_EXPORT_SYMBOL JITResult materializeSpecConstants(
    const char *KernelName, jit_compiler::SYCLKernelBinaryInfo &BinInfo,
    View<unsigned char> SpecConstBlob) {
  auto &JITCtx = JITContext::getInstance();

  TargetInfo TargetInfo = ConfigHelper::get<option::JITTargetInfo>();
  BinaryFormat TargetFormat = TargetInfo.getFormat();
  if (TargetFormat != BinaryFormat::PTX &&
      TargetFormat != BinaryFormat::AMDGCN) {
    return JITResult("Output target format not supported by this build. "
                     "Available targets are: PTX or AMDGCN.");
  }

  std::vector<::jit_compiler::SYCLKernelInfo> KernelInfos;
  KernelInfos.emplace_back(KernelName, BinInfo);
  // Load all input kernels from their respective modules into a single
  // LLVM IR module.
  llvm::Expected<std::unique_ptr<llvm::Module>> ModOrError =
      translation::KernelTranslator::loadKernels(*JITCtx.getLLVMContext(),
                                                 KernelInfos);
  if (auto Error = ModOrError.takeError()) {
    return errorTo<JITResult>(std::move(Error), "Failed to load kernels");
  }
  std::unique_ptr<llvm::Module> NewMod = std::move(*ModOrError);
  if (!fusion::FusionPipeline::runMaterializerPasses(
          *NewMod, SpecConstBlob.to<llvm::ArrayRef>()) ||
      !NewMod->getFunction(KernelName)) {
    return JITResult{"Materializer passes should not fail"};
  }

  SYCLKernelInfo &MaterializerKernelInfo = KernelInfos.front();
  if (auto Error = translation::KernelTranslator::translateKernel(
          MaterializerKernelInfo, *NewMod, JITCtx, TargetFormat)) {
    return errorTo<JITResult>(std::move(Error),
                              "Translation to output format failed");
  }

  return JITResult{MaterializerKernelInfo};
}

extern "C" KF_EXPORT_SYMBOL RTCHashResult
calculateHash(InMemoryFile SourceFile, View<InMemoryFile> IncludeFiles,
              View<const char *> UserArgs) {
  auto UserArgListOrErr = parseUserArgs(UserArgs);
  if (!UserArgListOrErr) {
    return RTCHashResult::failure(
        formatError(UserArgListOrErr.takeError(),
                    "Parsing of user arguments failed")
            .c_str());
  }
  llvm::opt::InputArgList UserArgList = std::move(*UserArgListOrErr);

  auto Start = std::chrono::high_resolution_clock::now();
  auto HashOrError = calculateHash(SourceFile, IncludeFiles, UserArgList);
  if (!HashOrError) {
    return RTCHashResult::failure(
        formatError(HashOrError.takeError(), "Hashing failed").c_str());
  }
  auto Hash = *HashOrError;
  auto Stop = std::chrono::high_resolution_clock::now();

  if (UserArgList.hasArg(clang::driver::options::OPT_ftime_trace_EQ)) {
    std::chrono::duration<double, std::milli> HashTime = Stop - Start;
    llvm::dbgs() << "Hashing of " << SourceFile.Path << " took "
                 << int(HashTime.count()) << " ms\n";
  }

  return RTCHashResult::success(Hash.c_str());
}

extern "C" KF_EXPORT_SYMBOL RTCResult
compileSYCL(InMemoryFile SourceFile, View<InMemoryFile> IncludeFiles,
            View<const char *> UserArgs, View<char> CachedIR, bool SaveIR) {
  llvm::LLVMContext Context;
  std::string BuildLog;
  configureDiagnostics(Context, BuildLog);

  auto UserArgListOrErr = parseUserArgs(UserArgs);
  if (!UserArgListOrErr) {
    return errorTo<RTCResult>(UserArgListOrErr.takeError(),
                              "Parsing of user arguments failed",
                              RTCErrorCode::INVALID);
  }
  llvm::opt::InputArgList UserArgList = std::move(*UserArgListOrErr);

  llvm::StringRef TraceFileName;
  if (auto *Arg =
          UserArgList.getLastArg(clang::driver::options::OPT_ftime_trace_EQ)) {
    TraceFileName = Arg->getValue();
    int Granularity =
        500; // microseconds. Same default as in `clang::FrontendOptions`.
    if (auto *Arg = UserArgList.getLastArg(
            clang::driver::options::OPT_ftime_trace_granularity_EQ)) {
      if (!llvm::to_integer(Arg->getValue(), Granularity)) {
        BuildLog += "warning: ignoring malformed argument: '" +
                    Arg->getAsString(UserArgList) + "'\n";
      }
    }
    bool Verbose =
        UserArgList.hasArg(clang::driver::options::OPT_ftime_trace_verbose);

    llvm::timeTraceProfilerInitialize(Granularity, /*ProcName=*/"sycl-rtc",
                                      Verbose);
  }

  std::unique_ptr<llvm::Module> Module;

  if (CachedIR.size() > 0) {
    llvm::StringRef IRStr{CachedIR.begin(), CachedIR.size()};
    std::unique_ptr<llvm::MemoryBuffer> IRBuf =
        llvm::MemoryBuffer::getMemBuffer(IRStr, /*BufferName=*/"",
                                         /*RequiresNullTerminator=*/false);
    auto ModuleOrError = llvm::parseBitcodeFile(*IRBuf, Context);
    if (!ModuleOrError) {
      // Not a fatal error, we'll just compile the source string normally.
      BuildLog.append(formatError(ModuleOrError.takeError(),
                                  "Loading of cached device code failed"));
    } else {
      Module = std::move(*ModuleOrError);
    }
  }

  bool FromSource = false;
  if (!Module) {
    auto ModuleOrErr = compileDeviceCode(SourceFile, IncludeFiles, UserArgList,
                                         BuildLog, Context);
    if (!ModuleOrErr) {
      return errorTo<RTCResult>(ModuleOrErr.takeError(),
                                "Device compilation failed");
    }

    Module = std::move(*ModuleOrErr);
    FromSource = true;
  }

  RTCDeviceCodeIR IR;
  if (SaveIR && FromSource) {
    std::string BCString;
    llvm::raw_string_ostream BCStream{BCString};
    llvm::WriteBitcodeToFile(*Module, BCStream);
    IR = RTCDeviceCodeIR{BCString.data(), BCString.data() + BCString.size()};
  }

  if (auto Error = linkDeviceLibraries(*Module, UserArgList, BuildLog)) {
    return errorTo<RTCResult>(std::move(Error), "Device linking failed");
  }

  auto PostLinkResultOrError = performPostLink(std::move(Module), UserArgList);
  if (!PostLinkResultOrError) {
    return errorTo<RTCResult>(PostLinkResultOrError.takeError(),
                              "Post-link phase failed");
  }
  auto [BundleInfo, Modules] = std::move(*PostLinkResultOrError);

  for (auto [DevImgInfo, Module] :
       llvm::zip_equal(BundleInfo.DevImgInfos, Modules)) {
    auto BinaryInfoOrError =
        translation::KernelTranslator::translateDevImgToSPIRV(
            *Module, JITContext::getInstance());
    if (!BinaryInfoOrError) {
      return errorTo<RTCResult>(BinaryInfoOrError.takeError(),
                                "SPIR-V translation failed");
    }
    DevImgInfo.BinaryInfo = std::move(*BinaryInfoOrError);
  }

  encodeBuildOptions(BundleInfo, UserArgList);

  if (llvm::timeTraceProfilerEnabled()) {
    auto Error = llvm::timeTraceProfilerWrite(
        TraceFileName, /*FallbackFileName=*/"trace.json");
    llvm::timeTraceProfilerCleanup();
    if (Error) {
      return errorTo<RTCResult>(std::move(Error), "Trace file writing failed");
    }
  }

  return RTCResult{std::move(BundleInfo), std::move(IR), BuildLog.c_str()};
}

extern "C" KF_EXPORT_SYMBOL void destroyBinary(BinaryAddress Address) {
  JITContext::getInstance().destroyKernelBinary(Address);
}

extern "C" KF_EXPORT_SYMBOL void resetJITConfiguration() {
  ConfigHelper::reset();
}

extern "C" KF_EXPORT_SYMBOL void addToJITConfiguration(OptionStorage &&Opt) {
  ConfigHelper::getConfig().set(std::move(Opt));
}
