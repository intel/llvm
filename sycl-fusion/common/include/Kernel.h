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
#include <string>
#include <vector>

namespace jit_compiler {

using BinaryAddress = const uint8_t *;

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
enum class BinaryFormat : uint32_t { INVALID, LLVM, SPIRV };

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

/// Information about a kernel from DPC++.
struct SYCLKernelInfo {

  std::string Name;

  SYCLArgumentDescriptor Args;

  AttributeList Attributes;

  SYCLKernelBinaryInfo BinaryInfo;

  //// Explicit constructor for compatibility with LLVM YAML I/O.
  SYCLKernelInfo() : Name{}, Args{}, Attributes{}, BinaryInfo{} {}

  SYCLKernelInfo(const std::string &KernelName,
                 const SYCLArgumentDescriptor &ArgDesc,
                 const SYCLKernelBinaryInfo &BinInfo)
      : Name{KernelName}, Args{ArgDesc}, Attributes{}, BinaryInfo{BinInfo} {}

  explicit SYCLKernelInfo(const std::string &KernelName)
      : Name{KernelName}, Args{}, Attributes{}, BinaryInfo{} {}
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
