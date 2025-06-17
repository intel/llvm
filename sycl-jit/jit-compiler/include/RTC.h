//===- RTC.h - Public interface for SYCL runtime compilation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "DynArray.h"
#include "JITBinaryInfo.h"
#include "Macros.h"
#include "View.h"
#include "sycl/detail/string.hpp"

#include <cassert>

namespace jit_compiler {

//===- Data structures ----------------------------------------------------===//

/// Descriptor tuple for file intended to be placed into a virtual filesystem
/// overlay.
struct InMemoryFile {
  const char *Path;
  const char *Contents;
};

/// A non-owning descriptor of a device binary managed by the `JITContext`.
using RTCDevImgBinaryInfo = JITBinaryInfo;

/// Conceptually a vector of strings, but without using STL types.
using FrozenSymbolTable = DynArray<sycl::detail::string>;

// Note: `FrozenPropertyValue` and `FrozenPropertySet` constructors take
// `std::string_view` arguments instead of `const char *` because they will be
// created from `llvm::SmallString`s, which don't contain the trailing '\0'
// byte. Hence obtaining a C-string would cause an additional copy.

/// Represents an `llvm::util::PropertyValue` without using LLVM data types.
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

/// Represents an `llvm::util::PropertySet` without using LLVM data types.
struct FrozenPropertySet {
  sycl::detail::string Name;
  DynArray<FrozenPropertyValue> Values;

  FrozenPropertySet() = default;
  FrozenPropertySet(FrozenPropertySet &&) = default;
  FrozenPropertySet &operator=(FrozenPropertySet &&) = default;

  FrozenPropertySet(std::string_view Name, size_t Size)
      : Name{Name}, Values{Size} {}
};

/// Represents an `llvm::util::PropertySetRegistry` without using LLVM data
/// types.
using FrozenPropertyRegistry = DynArray<FrozenPropertySet>;

/// The counterpart to `sycl_device_binary_struct` in the SYCL runtime.
struct RTCDevImgInfo {
  RTCDevImgBinaryInfo BinaryInfo;
  FrozenSymbolTable SymbolTable;
  FrozenPropertyRegistry Properties;

  RTCDevImgInfo() = default;
  RTCDevImgInfo(RTCDevImgInfo &&) = default;
  RTCDevImgInfo &operator=(RTCDevImgInfo &&) = default;
};

/// The counterpart to `sycl_device_binaries_struct` in the SYCL runtime.
struct RTCBundleInfo {
  DynArray<RTCDevImgInfo> DevImgInfos;
  sycl::detail::string CompileOptions;

  RTCBundleInfo() = default;
  RTCBundleInfo(RTCBundleInfo &&) = default;
  RTCBundleInfo &operator=(RTCBundleInfo &&) = default;
};

/// Represents an LLVM bitcode blob. Note that LLVM's APIs prefer `char *` for
/// byte buffers.
using RTCDeviceCodeIR = DynArray<char>;

//===- Result types -------------------------------------------------------===//

/// Result type for hash calculation: Wraps a string that is either the hash
/// result or the preprocessor log.
class RTCHashResult {
public:
  /// Constructs a result that indicates success iff \p IsHash is true.
  explicit RTCHashResult(const char *HashOrLog, bool IsHash = true)
      : HashOrLog(HashOrLog), IsHash(IsHash) {}

  bool failed() const noexcept { return !IsHash; }

  const char *getPreprocLog() const noexcept {
    assert(failed() && "No preprocessor log");
    return HashOrLog.c_str();
  }

  const char *getHash() const noexcept {
    assert(!failed() && "No hash");
    return HashOrLog.c_str();
  }

private:
  const sycl::detail::string HashOrLog;
  const bool IsHash;
};

/// Result type for SYCL runtime compilation. A successful result contains the
/// build log, the bundle info and optionally a device code IR blob suitable for
/// persistent caching. In case of failure, only the build log and desired error
/// code are wrapped.
class RTCResult {
public:
  /// Enum modelling a subset of `sycl::errc`, used to signal to the runtime
  /// which `sycl::exception` should be thrown in case of failure. NB: LLVM, and
  /// in consequence the sycl-jit library, are usually compiled without support
  /// for C++ exceptions.
  enum class RTCErrorCode { SUCCESS, BUILD, INVALID };

  /// Constructs a result that indicates failure.
  explicit RTCResult(const char *BuildLog,
                     RTCErrorCode ErrorCode = RTCErrorCode::BUILD)
      : ErrorCode{ErrorCode}, BundleInfo{}, BuildLog{BuildLog} {
    assert(ErrorCode != RTCErrorCode::SUCCESS);
  }

  /// Constructs a result that indicates success.
  RTCResult(RTCBundleInfo &&BundleInfo, RTCDeviceCodeIR &&DeviceCodeIR,
            const char *BuildLog)
      : ErrorCode{RTCErrorCode::SUCCESS}, BundleInfo{std::move(BundleInfo)},
        DeviceCodeIR(std::move(DeviceCodeIR)), BuildLog{BuildLog} {}

  RTCErrorCode getErrorCode() const noexcept { return ErrorCode; }

  const char *getBuildLog() const noexcept { return BuildLog.c_str(); }

  const RTCBundleInfo &getBundleInfo() const noexcept {
    assert(ErrorCode == RTCErrorCode::SUCCESS && "No bundle info");
    return BundleInfo;
  }

  const RTCDeviceCodeIR &getDeviceCodeIR() const noexcept {
    assert(ErrorCode == RTCErrorCode::SUCCESS && "No device code IR");
    return DeviceCodeIR;
  }

private:
  const RTCErrorCode ErrorCode;
  const RTCBundleInfo BundleInfo;
  const RTCDeviceCodeIR DeviceCodeIR;
  const sycl::detail::string BuildLog;
};

//===- Entrypoints --------------------------------------------------------===//

/// Calculates a BLAKE3 hash of the pre-processed source string described by
/// \p SourceFile (considering any additional \p IncludeFiles) and the
/// concatenation of the \p UserArgs.
JIT_EXPORT_SYMBOL RTCHashResult calculateHash(InMemoryFile SourceFile,
                                              View<InMemoryFile> IncludeFiles,
                                              View<const char *> UserArgs);

/// Compiles, links against device libraries, and finalizes the device code in
/// the source string described by \p SourceFile, considering any additional \p
/// IncludeFiles as well as the \p UserArgs.
///
/// \p CachedIR can be either empty or an LLVM bitcode blob. If it is the
/// latter, the corresponding module is used instead of invoking the clang
/// frontend.
///
/// If \p SaveIR is true and \p CachedIR is empty, the LLVM module obtained from
/// the frontend invocation is wrapped in bitcode format in the result object.
JIT_EXPORT_SYMBOL RTCResult compileSYCL(InMemoryFile SourceFile,
                                        View<InMemoryFile> IncludeFiles,
                                        View<const char *> UserArgs,
                                        View<char> CachedIR, bool SaveIR);

/// Requests that the JIT binary referenced by \p Address is deleted from the
/// `JITContext`.
JIT_EXPORT_SYMBOL void destroyBinary(BinaryAddress Address);

} // namespace jit_compiler
