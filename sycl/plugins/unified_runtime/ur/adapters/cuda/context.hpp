//===--------- context.hpp - CUDA Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <cuda.h>
#include <ur_api.h>

#include <atomic>
#include <mutex>
#include <vector>

#include "common.hpp"
#include "device.hpp"

typedef void (*ur_context_extended_deleter_t)(void *user_data);

/// UR context mapping to a CUDA context object.
///
/// There is no direct mapping between a CUDA context and a UR context.
/// The main differences are described below:
///
/// <b> CUDA context vs UR context </b>
///
/// One of the main differences between the UR API and the CUDA driver API is
/// that the second modifies the state of the threads by assigning
/// `CUcontext` objects to threads. `CUcontext` objects store data associated
/// with a given device and control access to said device from the user side.
/// UR API context are objects that are passed to functions, and not bound
/// to threads.
///
/// Since the ur_context_handle_t can contain multiple devices, and a CUcontext
/// refers to only a single device, the CUcontext is more tightly coupled to a
/// ur_device_handle_t than a ur_context_handle_t. In order to remove some
/// ambiguities about the different semantics of ur_context_handle_t s and
/// native CUcontext, we access the native CUcontext solely through the
/// ur_device_handle_t class, by using the RAII object \ref ScopedDevice, which
/// sets the active device (by setting the active native CUcontext).
///
/// <b> Primary vs User-defined CUcontext </b>
///
/// CUDA has two different types of CUcontext, the Primary context, which is
/// usable by all threads on a given process for a given device, and the
/// aforementioned custom CUcontexts. The CUDA documentation, confirmed with
/// performance analysis, suggest using the Primary context whenever possible.
/// The Primary context is also used by the CUDA Runtime API. For UR
/// applications to interop with CUDA Runtime API, they have to use the primary
/// context - and make that active in the thread.
///
///  <b> Destructor callback </b>
///
///  Required to implement CP023, SYCL Extended Context Destruction,
///  the PI Context can store a number of callback functions that will be
///  called upon destruction of the UR Context.
///  See proposal for details.
///  https://github.com/codeplaysoftware/standards-proposals/blob/master/extended-context-destruction/index.md
///
///  <b> Memory Management for Devices in a Context <\b>
///
///  A ur_buffer_ is associated with a ur_context_handle_t_, which may refer to
///  multiple devices. Therefore the ur_buffer_ must handle a native allocation
///  for each device in the context. UR is responsible for automatically
///  handling event dependencies for kernels writing to or reading from the
///  same ur_buffer_ and migrating memory between native allocations for
///  devices in the same ur_context_handle_t_ if necessary.
///
///  TODO: This management of memory for devices in the same
///  ur_context_handle_t_ is currently only valid for buffers and not for
///  images.
///
///
struct ur_context_handle_t_ {

  struct deleter_data {
    ur_context_extended_deleter_t Function;
    void *UserData;

    void operator()() { Function(UserData); }
  };

  using native_type = CUcontext;

  std::vector<ur_device_handle_t> Devices;
  uint32_t NumDevices{};

  std::atomic_uint32_t RefCount;

  ur_context_handle_t_(const ur_device_handle_t *Devs, uint32_t NumDevices)
      : Devices{Devs, Devs + NumDevices}, NumDevices{NumDevices}, RefCount{1} {
    for (auto &Dev : Devices) {
      urDeviceRetain(Dev);
    }
  };

  ~ur_context_handle_t_() {
    for (auto &Dev : Devices) {
      urDeviceRelease(Dev);
    }
  }

  void invokeExtendedDeleters() {
    std::lock_guard<std::mutex> Guard(Mutex);
    for (auto &Deleter : ExtendedDeleters) {
      Deleter();
    }
  }

  void setExtendedDeleter(ur_context_extended_deleter_t Function,
                          void *UserData) {
    std::lock_guard<std::mutex> Guard(Mutex);
    ExtendedDeleters.emplace_back(deleter_data{Function, UserData});
  }

  std::vector<ur_device_handle_t> getDevices() const noexcept {
    return Devices;
  }

  uint32_t incrementReferenceCount() noexcept { return ++RefCount; }

  uint32_t decrementReferenceCount() noexcept { return --RefCount; }

  uint32_t getReferenceCount() const noexcept { return RefCount; }

private:
  std::mutex Mutex;
  std::vector<deleter_data> ExtendedDeleters;
};
