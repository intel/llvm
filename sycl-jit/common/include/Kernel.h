//==------- Kernel.h - Representation of a SYCL kernel for JIT compiler ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "DynArray.h"
#include "sycl/detail/string.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string_view>
#include <type_traits>

namespace jit_compiler {

using BinaryAddress = const uint8_t *;

/// Different binary formats supported as input to the JIT compiler.
enum class BinaryFormat : uint32_t { INVALID, LLVM, SPIRV, PTX, AMDGCN };

/// Unique ID for each supported architecture in the SYCL implementation.
///
/// Values of this type will only be used in the kernel fusion non-persistent
/// JIT. There is no guarantee for backwards compatibility, so this should not
/// be used in persistent caches.
using DeviceArchitecture = unsigned;

class TargetInfo {
public:
  static constexpr TargetInfo get(BinaryFormat Format,
                                  DeviceArchitecture Arch) {
    if (Format == BinaryFormat::SPIRV) {
      /// As an exception, SPIR-V targets have a single common ID (-1), as fused
      /// kernels will be reused across SPIR-V devices.
      return {Format, DeviceArchitecture(-1)};
    }
    return {Format, Arch};
  }

  TargetInfo() = default;

  constexpr BinaryFormat getFormat() const { return Format; }
  constexpr DeviceArchitecture getArch() const { return Arch; }

private:
  constexpr TargetInfo(BinaryFormat Format, DeviceArchitecture Arch)
      : Format(Format), Arch(Arch) {}

  BinaryFormat Format;
  DeviceArchitecture Arch;
};

/// Information about a device intermediate representation module (e.g., SPIR-V,
/// LLVM IR) from DPC++.
struct SYCLKernelBinaryInfo {

  BinaryFormat Format = BinaryFormat::INVALID;

  uint64_t AddressBits = 0;

  BinaryAddress BinaryStart = nullptr;

  uint64_t BinarySize = 0;
};

// RTC-related datastructures
// TODO: Consider moving into separate header.

struct InMemoryFile {
  const char *Path;
  const char *Contents;
};

using RTCDevImgBinaryInfo = SYCLKernelBinaryInfo;
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

} // namespace jit_compiler
