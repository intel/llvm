//==----------------- host_pipe_map_entry.hpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <detail/device_binary_image.hpp>
#include <unordered_map>

namespace sycl {
inline namespace _V1 {
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
    // If there are multiple translation units using the host pipe then
    // initialize function will be called multiple times with the same pointer
    // because special function for pipe registration and initialization has to
    // be emitted by frontend for each translation unit. Just make sure that
    // pointer is the same.
    if (MHostPipePtr) {
      assert(MHostPipePtr == HostPipePtr &&
             "Host pipe intializations disagree on address of the host pipe on "
             "host.");
      return;
    }

    MHostPipePtr = HostPipePtr;
  }

  void initialize(const RTDeviceBinaryImage *DeviceImage) {
    mDeviceImage = DeviceImage;
  }

  RTDeviceBinaryImage *getDevBinImage() {
    return const_cast<RTDeviceBinaryImage *>(mDeviceImage);
  }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
