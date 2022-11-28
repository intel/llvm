//==----------------- host_pipe_map_entry.hpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/device_binary_image.hpp>
#include <cstdint>
#include <unordered_map>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

struct HostPipeMapEntry {
  std::string MUniqueId;
  // Pointer to the host_pipe on host.
  const void *MHostPipePtr;
  // Size of the underlying type in the host_pipe.
  std::uint32_t MHostPipeTSize;
  // The device image that pipe is associated with
  const RTDeviceBinaryImage *mDeviceImage;

  // Constructor only initializes with the pointer and ID.
  // Other members will be initialized later
  HostPipeMapEntry(std::string UniqueId, const void *HostPipePtr)
      : MUniqueId(UniqueId), MHostPipePtr(HostPipePtr), MHostPipeTSize(0) {}

  // Constructor only initializes with the size and ID.
  // Other members will be initialized later
  HostPipeMapEntry(std::string UniqueId, std::uint32_t HostPipeTSize)
      : MUniqueId(UniqueId), MHostPipePtr(nullptr),
        MHostPipeTSize(HostPipeTSize) {}

  void initialize(std::uint32_t HostPipeTSize) {
    assert(HostPipeTSize != 0 && "Host pipe initialized with 0 size.");
    assert(MHostPipeTSize == 0 && "Host pipe has already been initialized.");
    MHostPipeTSize = HostPipeTSize;
  }

  void initialize(const void *HostPipePtr) {
    assert(!MHostPipePtr && "Host pipe pointer has already been initialized.");
    MHostPipePtr = HostPipePtr;
  }

  void initialize(const RTDeviceBinaryImage *DeviceImage) {
    mDeviceImage = DeviceImage;
  }
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
