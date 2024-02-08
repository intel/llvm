//==------- Kernel.h - Representation of a SYCL kernel for JIT compiler ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_COMMON_KERNEL_H
#define SYCL_FUSION_COMMON_KERNEL_H

#include "DynArray.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <type_traits>

namespace jit_compiler {

using BinaryAddress = const uint8_t *;

/// Possible barrier flags
enum class BarrierFlags : uint32_t {
  None = 0,   // Do not insert barrier
  Local = 1,  // Ensure correct ordering of memory operations to local memory
  Global = 2, // Ensure correct ordering of memory operations to global memory
  LocalAndGlobal = Local | Global
};

constexpr BarrierFlags getNoBarrierFlag() { return BarrierFlags::None; }
constexpr BarrierFlags getLocalAndGlobalBarrierFlag() {
  return BarrierFlags::LocalAndGlobal;
}
constexpr bool isNoBarrierFlag(BarrierFlags Flag) {
  return Flag == BarrierFlags::None;
}
constexpr bool hasLocalBarrierFlag(BarrierFlags Flag) {
  return static_cast<uint32_t>(Flag) &
         static_cast<uint32_t>(BarrierFlags::Local);
}
constexpr bool hasGlobalBarrierFlag(BarrierFlags Flag) {
  return static_cast<uint32_t>(Flag) &
         static_cast<uint32_t>(BarrierFlags::Global);
}

///
/// Enumerate possible kinds of parameters.
/// 1:1 correspondence with the definition in kernel_desc.hpp in the DPC++ SYCL
/// runtime.
enum class ParameterKind : uint32_t {
  Accessor = 0,
  StdLayout = 1,
  Sampler = 2,
  Pointer = 3,
  SpecConstBuffer = 4,
  Stream = 5,
  Invalid = 0xF,
};

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

///
/// Encode usage of parameters for the actual kernel function.
enum ArgUsage : uint8_t {
  // Used to indicate that an argument is not used by the kernel
  Unused = 0,
  // Used to indicate that an argument is used by the kernel
  Used = 1u,
  // Used to indicate that the accessor/pointer argument has been promoted to
  // private memory
  PromotedPrivate = 1u << 4,
  // Used to indicate that the accessor/pointer argument has been promoted to
  // local memory
  PromotedLocal = 1u << 5,
};

///
/// Expose the enum's underlying type because it simplifies bitwise operations.
using ArgUsageUT = std::underlying_type_t<ArgUsage>;

///
/// Describe the list of arguments by their kind and usage.
struct SYCLArgumentDescriptor {
  explicit SYCLArgumentDescriptor(size_t NumArgs = 0)
      : Kinds(NumArgs), UsageMask(NumArgs){};

  DynArray<ParameterKind> Kinds;
  DynArray<ArgUsageUT> UsageMask;
};

///
/// Class to model a three-dimensional index.
class Indices {
public:
  static constexpr size_t size() { return Size; }

  constexpr Indices() : Values{0, 0, 0} {}
  constexpr Indices(size_t V1, size_t V2, size_t V3) : Values{V1, V2, V3} {}

  constexpr const size_t *begin() const { return Values; }
  constexpr const size_t *end() const { return Values + Size; }
  constexpr size_t *begin() { return Values; }
  constexpr size_t *end() { return Values + Size; }

  constexpr const size_t &operator[](int Idx) const { return Values[Idx]; }
  constexpr size_t &operator[](int Idx) { return Values[Idx]; }

  friend bool operator==(const Indices &A, const Indices &B) {
    return std::equal(A.begin(), A.end(), B.begin());
  }

  friend bool operator!=(const Indices &A, const Indices &B) {
    return !(A == B);
  }

  friend bool operator<(const Indices &A, const Indices &B) {
    return std::lexicographical_compare(A.begin(), A.end(), B.begin(), B.end(),
                                        std::less<size_t>{});
  }

  friend bool operator>(const Indices &A, const Indices &B) {
    return std::lexicographical_compare(A.begin(), A.end(), B.begin(), B.end(),
                                        std::greater<size_t>{});
  }

private:
  static constexpr size_t Size = 3;
  size_t Values[Size];
};

///
/// Describe a SYCL/OpenCL kernel attribute by its kind and values.
struct SYCLKernelAttribute {
  enum class AttrKind { Invalid, ReqdWorkGroupSize, WorkGroupSizeHint };

  static constexpr auto ReqdWorkGroupSizeName = "reqd_work_group_size";
  static constexpr auto WorkGroupSizeHintName = "work_group_size_hint";

  static AttrKind parseKind(const char *Name) {
    auto Kind = AttrKind::Invalid;
    if (std::strcmp(Name, ReqdWorkGroupSizeName) == 0) {
      Kind = AttrKind::ReqdWorkGroupSize;
    } else if (std::strcmp(Name, WorkGroupSizeHintName) == 0) {
      Kind = AttrKind::WorkGroupSizeHint;
    }
    return Kind;
  }

  AttrKind Kind;
  Indices Values;

  SYCLKernelAttribute() : Kind(AttrKind::Invalid) {}
  SYCLKernelAttribute(AttrKind Kind, const Indices &Values)
      : Kind(Kind), Values(Values) {}

  const char *getName() const {
    assert(Kind != AttrKind::Invalid);
    switch (Kind) {
    case AttrKind::ReqdWorkGroupSize:
      return ReqdWorkGroupSizeName;
    case AttrKind::WorkGroupSizeHint:
      return WorkGroupSizeHintName;
    default:
      return "__invalid__";
    }
  }
};

///
/// List of SYCL/OpenCL kernel attributes.
using SYCLAttributeList = DynArray<SYCLKernelAttribute>;

///
/// Class to model SYCL nd_range
class NDRange {
public:
  constexpr static Indices AllZeros{0, 0, 0};

  ///
  /// Return the product of each index in an indices array.
  constexpr static size_t linearize(const jit_compiler::Indices &I) {
    return I[0] * I[1] * I[2];
  }

  NDRange() : NDRange{1, {1, 1, 1}} {}

  NDRange(int Dimensions, const Indices &GlobalSize,
          const Indices &LocalSize = {1, 1, 1},
          const Indices &Offset = {0, 0, 0})
      : Dimensions{Dimensions},
        GlobalSize{GlobalSize}, LocalSize{LocalSize}, Offset{Offset} {
#ifndef NDEBUG
    const auto CheckDim = [Dimensions](const Indices &Range) {
      return std::all_of(Range.begin() + Dimensions, Range.end(),
                         [](auto D) { return D == 1; });
    };
    const auto CheckOffsetDim = [Dimensions](const Indices &Offset) {
      return std::all_of(Offset.begin() + Dimensions, Offset.end(),

                         [](auto D) { return D == 0; });
    };
#endif // NDEBUG
    assert(CheckDim(GlobalSize) &&
           "Invalid global range for number of dimensions");
    assert(
        (CheckDim(LocalSize) || std::all_of(LocalSize.begin(), LocalSize.end(),
                                            [](auto D) { return D == 0; })) &&
        "Invalid local range for number of dimensions");
    assert(CheckOffsetDim(Offset) && "Invalid offset for number of dimensions");
  }

  constexpr const Indices &getGlobalSize() const { return GlobalSize; }
  constexpr const Indices &getLocalSize() const { return LocalSize; }
  constexpr const Indices &getOffset() const { return Offset; }
  constexpr int getDimensions() const { return Dimensions; }

  bool hasSpecificLocalSize() const { return LocalSize != AllZeros; }
  bool hasUniformWorkGroupSizes() const {
    assert(hasSpecificLocalSize() && "Local size must be specified");
    for (int I = 0; I < Dimensions; ++I) {
      if (GlobalSize[I] % LocalSize[I] != 0) {
        return false;
      }
    }
    return true;
  }

  friend constexpr bool operator==(const NDRange &LHS, const NDRange &RHS) {
    return LHS.Dimensions == RHS.Dimensions &&
           LHS.GlobalSize == RHS.GlobalSize && LHS.LocalSize == RHS.LocalSize &&
           LHS.Offset == RHS.Offset;
  }

  friend constexpr bool operator!=(const NDRange &LHS, const NDRange &RHS) {
    return !(LHS == RHS);
  }

  friend bool operator<(const NDRange &LHS, const NDRange &RHS) {
    if (LHS.Dimensions < RHS.Dimensions) {
      return true;
    }
    if (LHS.Dimensions > RHS.Dimensions) {
      return false;
    }

    if (LHS.GlobalSize < RHS.GlobalSize) {
      return true;
    }
    if (LHS.GlobalSize > RHS.GlobalSize) {
      return false;
    }

    if (LHS.LocalSize < RHS.LocalSize) {
      return true;
    }
    if (LHS.LocalSize > RHS.LocalSize) {
      return false;
    }

    return LHS.Offset < RHS.Offset;
  }

  friend bool operator>(const NDRange &LHS, const NDRange &RHS) {
    return RHS < LHS;
  }

private:
  /** @brief The number of dimensions. */
  int Dimensions;
  /** @brief The local range. */
  Indices GlobalSize;
  /** @brief The local range. */
  Indices LocalSize;
  /** @brief The offet. */
  Indices Offset;
};

/// Information about a kernel from DPC++.
struct SYCLKernelInfo {

  DynString Name;

  SYCLArgumentDescriptor Args;

  SYCLAttributeList Attributes;

  NDRange NDR;

  SYCLKernelBinaryInfo BinaryInfo;

  SYCLKernelInfo() = default;

  SYCLKernelInfo(const char *KernelName, const SYCLArgumentDescriptor &ArgDesc,
                 const NDRange &NDR, const SYCLKernelBinaryInfo &BinInfo)
      : Name{KernelName}, Args{ArgDesc}, Attributes{}, NDR{NDR},
        BinaryInfo{BinInfo} {}

  SYCLKernelInfo(const char *KernelName, size_t NumArgs)
      : Name{KernelName}, Args{NumArgs}, Attributes{}, NDR{}, BinaryInfo{} {}
};

} // namespace jit_compiler

#endif // SYCL_FUSION_COMMON_KERNEL_H
