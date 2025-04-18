//===- RTC.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RTC.h"
#include "helper/ErrorHelper.h"
#include "rtc/DeviceCompilation.h"
#include "translation/SPIRVLLVMTranslation.h"
#include "translation/Translation.h"

#include <llvm/ADT/StringExtras.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TimeProfiler.h>

#include <clang/Driver/Options.h>

#include <chrono>

using namespace jit_compiler;

JIT_EXPORT_SYMBOL RTCHashResult calculateHash(InMemoryFile SourceFile,
                                              View<InMemoryFile> IncludeFiles,
                                              View<const char *> UserArgs) {
  llvm::opt::InputArgList UserArgList;
  if (auto Error = parseUserArgs(UserArgs).moveInto(UserArgList)) {
    return errorTo<RTCHashResult>(std::move(Error),
                                  "Parsing of user arguments failed",
                                  /*IsHash=*/false);
  }

  auto Start = std::chrono::high_resolution_clock::now();
  std::string Hash;
  if (auto Error =
          calculateHash(SourceFile, IncludeFiles, UserArgList).moveInto(Hash)) {
    return errorTo<RTCHashResult>(std::move(Error), "Hashing failed",
                                  /*IsHash=*/false);
  }
  auto Stop = std::chrono::high_resolution_clock::now();

  if (UserArgList.hasArg(clang::driver::options::OPT_ftime_trace_EQ)) {
    std::chrono::duration<double, std::milli> HashTime = Stop - Start;
    llvm::dbgs() << "Hashing of " << SourceFile.Path << " took "
                 << int(HashTime.count()) << " ms\n";
  }

  return RTCHashResult{Hash.c_str()};
}

JIT_EXPORT_SYMBOL RTCResult compileSYCL(InMemoryFile SourceFile,
                                        View<InMemoryFile> IncludeFiles,
                                        View<const char *> UserArgs,
                                        View<char> CachedIR, bool SaveIR) {
  llvm::LLVMContext Context;
  std::string BuildLog;
  configureDiagnostics(Context, BuildLog);

  llvm::opt::InputArgList UserArgList;
  if (auto Error = parseUserArgs(UserArgs).moveInto(UserArgList)) {
    return errorTo<RTCResult>(std::move(Error),
                              "Parsing of user arguments failed",
                              RTCResult::RTCErrorCode::INVALID);
  }

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
    if (auto Error = llvm::parseBitcodeFile(*IRBuf, Context).moveInto(Module)) {
      // Not a fatal error, we'll just compile the source string normally.
      BuildLog.append(formatError(std::move(Error),
                                  "Loading of cached device code failed"));
    }
  }

  bool FromSource = !Module;
  if (FromSource) {
    if (auto Error = compileDeviceCode(SourceFile, IncludeFiles, UserArgList,
                                       BuildLog, Context)
                         .moveInto(Module)) {
      return errorTo<RTCResult>(std::move(Error), "Device compilation failed");
    }
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
    if (auto Error = Translator::translate(*Module, JITContext::getInstance(),
                                           BinaryFormat::SPIRV)
                         .moveInto(DevImgInfo.BinaryInfo)) {
      return errorTo<RTCResult>(std::move(Error), "SPIR-V translation failed");
    }
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

JIT_EXPORT_SYMBOL void destroyBinary(BinaryAddress Address) {
  JITContext::getInstance().destroyBinary(Address);
}
