//==------- Kernel.h - Representation of a SYCL kernel for JIT compiler ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_COMMON_KERNEL_H
#define SYCL_FUSION_COMMON_KERNEL_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

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

/// Information about a device intermediate representation module (e.g., SPIR-V,
/// LLVM IR) from DPC++.
struct SYCLKernelBinaryInfo {

  BinaryFormat Format = BinaryFormat::INVALID;

  uint64_t AddressBits = 0;

  BinaryAddress BinaryStart = nullptr;

  uint64_t BinarySize = 0;
};

///
/// Describe a SYCL/OpenCL kernel attribute by its name and values.
struct SYCLKernelAttribute {
  using AttributeValueList = std::vector<std::string>;

  // Explicit constructor for compatibility with LLVM YAML I/O.
  SYCLKernelAttribute() : Values{} {};
  SYCLKernelAttribute(std::string Name)
      : AttributeName{std::move(Name)}, Values{} {}

  std::string AttributeName;
  AttributeValueList Values;
};

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
/// Encode usage of parameters for the actual kernel function.
// This is a vector of unsigned char, because std::vector<bool> is a weird
// construct and unlike all other std::vectors, and LLVM YAML I/O is having a
// hard time coping with it.
using ArgUsageMask = std::vector<std::underlying_type_t<ArgUsage>>;

///
/// Describe the list of arguments by their kind.
struct SYCLArgumentDescriptor {

  // Explicit constructor for compatibility with LLVM YAML I/O.
  SYCLArgumentDescriptor() : Kinds{}, UsageMask{} {}

  std::vector<ParameterKind> Kinds;

  ArgUsageMask UsageMask;
};

///
/// List of SYCL/OpenCL kernel attributes.
using AttributeList = std::vector<SYCLKernelAttribute>;

using Indices = std::array<size_t, 3>;

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

  friend constexpr bool operator==(const NDRange &LHS, const NDRange &RHS) {
    return LHS.Dimensions == RHS.Dimensions &&
           LHS.GlobalSize == RHS.GlobalSize &&
           (!LHS.hasSpecificLocalSize() || !RHS.hasSpecificLocalSize() ||
            LHS.LocalSize == RHS.LocalSize) &&
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

    if (!LHS.hasSpecificLocalSize() && RHS.hasSpecificLocalSize()) {
      return true;
    }
    if (LHS.hasSpecificLocalSize() && !RHS.hasSpecificLocalSize()) {
      return false;
    }
    if (LHS.hasSpecificLocalSize() && RHS.hasSpecificLocalSize()) {
      if (LHS.LocalSize < RHS.LocalSize) {
        return true;
      }
      if (LHS.LocalSize > RHS.LocalSize) {
        return false;
      }
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

  std::string Name;

  SYCLArgumentDescriptor Args;

  AttributeList Attributes;

  NDRange NDR;

  SYCLKernelBinaryInfo BinaryInfo;

  //// Explicit constructor for compatibility with LLVM YAML I/O.
  SYCLKernelInfo() : Name{}, Args{}, Attributes{}, NDR{}, BinaryInfo{} {}

  SYCLKernelInfo(const std::string &KernelName,
                 const SYCLArgumentDescriptor &ArgDesc, const NDRange &NDR,
                 const SYCLKernelBinaryInfo &BinInfo)
      : Name{KernelName}, Args{ArgDesc}, Attributes{}, NDR{NDR}, BinaryInfo{
                                                                     BinInfo} {}

  explicit SYCLKernelInfo(const std::string &KernelName)
      : Name{KernelName}, Args{}, Attributes{}, NDR{}, BinaryInfo{} {}
};

///
/// Represents a SPIR-V translation unit containing SYCL kernels by the
/// KernelInfo for each of the contained kernels.
class SYCLModuleInfo {
public:
  using KernelInfoList = std::vector<SYCLKernelInfo>;

  void addKernel(SYCLKernelInfo &Kernel) { Kernels.push_back(Kernel); }

  KernelInfoList &kernels() { return Kernels; }

  bool hasKernelFor(const std::string &KernelName) {
    return getKernelFor(KernelName) != nullptr;
  }

  SYCLKernelInfo *getKernelFor(const std::string &KernelName) {
    auto It =
        std::find_if(Kernels.begin(), Kernels.end(),
                     [&](SYCLKernelInfo &K) { return K.Name == KernelName; });
    return (It != Kernels.end()) ? &*It : nullptr;
  }

private:
  KernelInfoList Kernels;
};

} // namespace jit_compiler

#endif // SYCL_FUSION_COMMON_KERNEL_H
