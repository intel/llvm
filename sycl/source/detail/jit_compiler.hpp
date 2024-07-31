//==--- jit_compiler.hpp - SYCL runtime JIT compiler for kernel fusion -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/jit_device_binaries.hpp>
#include <detail/scheduler/commands.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <sycl/feature_test.hpp>
#if SYCL_EXT_CODEPLAY_KERNEL_FUSION
#include <KernelFusion.h>
#endif // SYCL_EXT_CODEPLAY_KERNEL_FUSION

#include <unordered_map>

namespace jit_compiler {
enum class BinaryFormat : uint32_t;
class JITContext;
struct SYCLKernelInfo;
struct SYCLKernelAttribute;
template <typename T> class DynArray;
using ArgUsageMask = DynArray<uint8_t>;
} // namespace jit_compiler

namespace sycl {
inline namespace _V1 {
namespace detail {

class jit_compiler {

public:
  std::unique_ptr<detail::CG>
  fuseKernels(QueueImplPtr Queue, std::vector<ExecCGCommand *> &InputKernels,
              const property_list &);
  ur_kernel_handle_t
  materializeSpecConstants(QueueImplPtr Queue,
                           const RTDeviceBinaryImage *BinImage,
                           const std::string &KernelName,
                           const std::vector<unsigned char> &SpecConstBlob);

  bool isAvailable() { return Available; }

  static jit_compiler &get_instance() {
    static jit_compiler instance{};
    return instance;
  }

private:
  jit_compiler();
  ~jit_compiler() = default;
  jit_compiler(const jit_compiler &) = delete;
  jit_compiler(jit_compiler &&) = delete;
  jit_compiler &operator=(const jit_compiler &) = delete;
  jit_compiler &operator=(const jit_compiler &&) = delete;

  sycl_device_binaries
  createPIDeviceBinary(const ::jit_compiler::SYCLKernelInfo &FusedKernelInfo,
                       ::jit_compiler::BinaryFormat Format);

  std::vector<uint8_t>
  encodeArgUsageMask(const ::jit_compiler::ArgUsageMask &Mask) const;

  std::vector<uint8_t> encodeReqdWorkGroupSize(
      const ::jit_compiler::SYCLKernelAttribute &Attr) const;

  // Indicate availability of the JIT compiler
  bool Available;

  // Manages the lifetime of the UR structs for device binaries.
  std::vector<DeviceBinariesCollection> JITDeviceBinaries;

#if SYCL_EXT_CODEPLAY_KERNEL_FUSION
  // Handles to the entry points of the lazily loaded JIT library.
  using FuseKernelsFuncT = decltype(::jit_compiler::fuseKernels) *;
  using MaterializeSpecConstFuncT =
      decltype(::jit_compiler::materializeSpecConstants) *;
  using ResetConfigFuncT = decltype(::jit_compiler::resetJITConfiguration) *;
  using AddToConfigFuncT = decltype(::jit_compiler::addToJITConfiguration) *;
  FuseKernelsFuncT FuseKernelsHandle = nullptr;
  MaterializeSpecConstFuncT MaterializeSpecConstHandle = nullptr;
  ResetConfigFuncT ResetConfigHandle = nullptr;
  AddToConfigFuncT AddToConfigHandle = nullptr;
#endif // SYCL_EXT_CODEPLAY_KERNEL_FUSION
};

} // namespace detail
} // namespace _V1
} // namespace sycl
