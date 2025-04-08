//==--- RTC.h - Public interface for SYCL runtime compilation --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifdef _WIN32
#define RTC_EXPORT_SYMBOL __declspec(dllexport)
#else
#define RTC_EXPORT_SYMBOL
#endif

#include "DynArray.h"
#include "JITBinaryInfo.h"
#include "View.h"
#include "sycl/detail/string.hpp"

#include <cassert>

namespace jit_compiler {

struct InMemoryFile {
  const char *Path;
  const char *Contents;
};

using RTCDevImgBinaryInfo = JITBinaryInfo;
using FrozenSymbolTable = DynArray<sycl::detail::string>;

// Note: `FrozenPropertyValue` and `FrozenPropertySet` constructors take
// `std::string_view` arguments instead of `const char *` because they will be
// created from `llvm::SmallString`s, which don't contain the trailing '\0'
// byte. Hence obtaining a C-string would cause an additional copy.

struct FrozenPropertyValue {
  sycl::detail::string Name;
  bool IsUIntValue;
  uint32_t UIntValue;
  DynArray<uint8_t> Bytes;

  FrozenPropertyValue() = default;
  FrozenPropertyValue(FrozenPropertyValue &&) = default;
  FrozenPropertyValue &operator=(FrozenPropertyValue &&) = default;

  FrozenPropertyValue(std::string_view Name, uint32_t Value)
      : Name{Name}, IsUIntValue{true}, UIntValue{Value}, Bytes{0} {}
  FrozenPropertyValue(std::string_view Name, const uint8_t *Ptr, size_t Size)
      : Name{Name}, IsUIntValue{false}, UIntValue{0}, Bytes{Size} {
    std::memcpy(Bytes.begin(), Ptr, Size);
  }
};

struct FrozenPropertySet {
  sycl::detail::string Name;
  DynArray<FrozenPropertyValue> Values;

  FrozenPropertySet() = default;
  FrozenPropertySet(FrozenPropertySet &&) = default;
  FrozenPropertySet &operator=(FrozenPropertySet &&) = default;

  FrozenPropertySet(std::string_view Name, size_t Size)
      : Name{Name}, Values{Size} {}
};

using FrozenPropertyRegistry = DynArray<FrozenPropertySet>;

struct RTCDevImgInfo {
  RTCDevImgBinaryInfo BinaryInfo;
  FrozenSymbolTable SymbolTable;
  FrozenPropertyRegistry Properties;

  RTCDevImgInfo() = default;
  RTCDevImgInfo(RTCDevImgInfo &&) = default;
  RTCDevImgInfo &operator=(RTCDevImgInfo &&) = default;
};

struct RTCBundleInfo {
  DynArray<RTCDevImgInfo> DevImgInfos;
  sycl::detail::string CompileOptions;

  RTCBundleInfo() = default;
  RTCBundleInfo(RTCBundleInfo &&) = default;
  RTCBundleInfo &operator=(RTCBundleInfo &&) = default;
};

// LLVM's APIs prefer `char *` for byte buffers.
using RTCDeviceCodeIR = DynArray<char>;

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

enum class RTCErrorCode { SUCCESS, BUILD, INVALID };

class RTCResult {
public:
  explicit RTCResult(const char *BuildLog,
                     RTCErrorCode ErrorCode = RTCErrorCode::BUILD)
      : ErrorCode{ErrorCode}, BundleInfo{}, BuildLog{BuildLog} {
    assert(ErrorCode != RTCErrorCode::SUCCESS);
  }

  RTCResult(RTCBundleInfo &&BundleInfo, RTCDeviceCodeIR &&DeviceCodeIR,
            const char *BuildLog)
      : ErrorCode{RTCErrorCode::SUCCESS}, BundleInfo{std::move(BundleInfo)},
        DeviceCodeIR(std::move(DeviceCodeIR)), BuildLog{BuildLog} {}

  RTCErrorCode getErrorCode() const { return ErrorCode; }

  const char *getBuildLog() const { return BuildLog.c_str(); }

  const RTCBundleInfo &getBundleInfo() const {
    assert(ErrorCode == RTCErrorCode::SUCCESS && "No bundle info");
    return BundleInfo;
  }

  const RTCDeviceCodeIR &getDeviceCodeIR() const {
    assert(ErrorCode == RTCErrorCode::SUCCESS && "No device code IR");
    return DeviceCodeIR;
  }

private:
  RTCErrorCode ErrorCode;
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

RTC_EXPORT_SYMBOL RTCHashResult calculateHash(InMemoryFile SourceFile,
                                              View<InMemoryFile> IncludeFiles,
                                              View<const char *> UserArgs);

RTC_EXPORT_SYMBOL RTCResult compileSYCL(InMemoryFile SourceFile,
                                        View<InMemoryFile> IncludeFiles,
                                        View<const char *> UserArgs,
                                        View<char> CachedIR, bool SaveIR);

RTC_EXPORT_SYMBOL void destroyBinary(BinaryAddress Address);

} // end of extern "C"

} // namespace jit_compiler
