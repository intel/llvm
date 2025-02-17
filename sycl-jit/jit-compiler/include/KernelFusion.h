//==- KernelFusion.h - Public interface of JIT compiler for kernel fusion --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_JIT_COMPILER_KERNELFUSION_H
#define SYCL_FUSION_JIT_COMPILER_KERNELFUSION_H

#ifdef _WIN32
#define KF_EXPORT_SYMBOL __declspec(dllexport)
#else
#define KF_EXPORT_SYMBOL
#endif

#include "Kernel.h"
#include "Options.h"
#include "Parameter.h"
#include "View.h"
#include "sycl/detail/string.hpp"

#include <cassert>

namespace jit_compiler {

class JITResult {
public:
  explicit JITResult(const char *ErrorMessage)
      : Type{JITResultType::FAILED}, KernelInfo{}, ErrorMessage{ErrorMessage} {}

  explicit JITResult(const SYCLKernelInfo &KernelInfo, bool Cached = false)
      : Type{(Cached) ? JITResultType::CACHED : JITResultType::NEW},
        KernelInfo(KernelInfo), ErrorMessage{} {}

  bool failed() const { return Type == JITResultType::FAILED; }

  bool cached() const { return Type == JITResultType::CACHED; }

  const char *getErrorMessage() const {
    assert(failed() && "No error message present");
    return ErrorMessage.c_str();
  }

  const SYCLKernelInfo &getKernelInfo() const {
    assert(!failed() && "No kernel info");
    return KernelInfo;
  }

private:
  enum class JITResultType { FAILED, CACHED, NEW };

  JITResultType Type;
  SYCLKernelInfo KernelInfo;
  sycl::detail::string ErrorMessage;
};

class RTCHashResult {
public:
  static RTCHashResult success(const char *Hash) {
    return RTCHashResult{/*Failed=*/false, Hash};
  }

  static RTCHashResult failure(const char *PreprocLog) {
    return RTCHashResult{/*Failed=*/true, PreprocLog};
  }

  bool failed() { return Failed; }

  const char *getPreprocLog() {
    assert(failed() && "No preprocessor log");
    return HashOrLog.c_str();
  }

  const char *getHash() {
    assert(!failed() && "No hash");
    return HashOrLog.c_str();
  }

private:
  RTCHashResult(bool Failed, const char *HashOrLog)
      : Failed(Failed), HashOrLog(HashOrLog) {}

  bool Failed;
  sycl::detail::string HashOrLog;
};

class RTCResult {
public:
  explicit RTCResult(const char *BuildLog)
      : Failed{true}, BundleInfo{}, BuildLog{BuildLog} {}

  RTCResult(RTCBundleInfo &&BundleInfo, RTCDeviceCodeIR &&DeviceCodeIR,
            const char *BuildLog)
      : Failed{false}, BundleInfo{std::move(BundleInfo)},
        DeviceCodeIR(std::move(DeviceCodeIR)), BuildLog{BuildLog} {}

  bool failed() const { return Failed; }

  const char *getBuildLog() const { return BuildLog.c_str(); }

  const RTCBundleInfo &getBundleInfo() const {
    assert(!failed() && "No bundle info");
    return BundleInfo;
  }

  const RTCDeviceCodeIR &getDeviceCodeIR() const {
    assert(!failed() && "No device code IR");
    return DeviceCodeIR;
  }

private:
  bool Failed;
  RTCBundleInfo BundleInfo;
  RTCDeviceCodeIR DeviceCodeIR;
  sycl::detail::string BuildLog;
};

extern "C" {

#ifdef __clang__
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif // __clang__

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4190)
#endif // _MSC_VER

KF_EXPORT_SYMBOL JITResult
fuseKernels(View<SYCLKernelInfo> KernelInformation, const char *FusedKernelName,
            View<ParameterIdentity> Identities, BarrierFlags BarriersFlags,
            View<ParameterInternalization> Internalization,
            View<jit_compiler::JITConstant> JITConstants);

KF_EXPORT_SYMBOL JITResult materializeSpecConstants(
    const char *KernelName, jit_compiler::SYCLKernelBinaryInfo &BinInfo,
    View<unsigned char> SpecConstBlob);

KF_EXPORT_SYMBOL RTCHashResult calculateHash(InMemoryFile SourceFile,
                                             View<InMemoryFile> IncludeFiles,
                                             View<const char *> UserArgs);

KF_EXPORT_SYMBOL RTCResult compileSYCL(InMemoryFile SourceFile,
                                       View<InMemoryFile> IncludeFiles,
                                       View<const char *> UserArgs,
                                       View<char> CachedIR, bool SaveIR);

/// Clear all previously set options.
KF_EXPORT_SYMBOL void resetJITConfiguration();

/// Add an option to the configuration.
KF_EXPORT_SYMBOL void addToJITConfiguration(OptionStorage &&Opt);

} // end of extern "C"

} // namespace jit_compiler

#endif // SYCL_FUSION_JIT_COMPILER_KERNELFUSION_H
