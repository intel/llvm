//===---------- pi_esimd_emulator.hpp - CM Emulation Plugin ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file pi_esimd_emulator.hpp
/// Declarations for CM Emulation Plugin. It is the interface between the
/// device-agnostic SYCL runtime layer and underlying CM Emulation
///
/// \ingroup sycl_pi_esimd_emulator

#pragma once

#include <CL/sycl/detail/pi.h>
#include <atomic>
#include <cassert>
#include <iostream>
#include <mutex>
#include <unordered_map>

#include <malloc.h>

namespace cm_support {
#include <cm_rt.h>
} // namespace cm_support

template <class To, class From> To pi_cast(From Value) {
  // TODO: see if more sanity checks are possible.
  assert(sizeof(From) == sizeof(To));
  return (To)(Value);
}

template <> uint32_t pi_cast(uint64_t Value) {
  // Cast value and check that we don't lose any information.
  uint32_t CastedValue = (uint32_t)(Value);
  assert((uint64_t)CastedValue == Value);
  return CastedValue;
}

// TODO: Currently die is defined in each plugin. Probably some
// common header file with utilities should be created.
[[noreturn]] void die(const char *Message) {
  std::cerr << "die: " << Message << std::endl;
  std::terminate();
}

// Base class to store common data
struct _pi_object {
  _pi_object() : RefCount{1} {}

  std::atomic<pi_uint32> RefCount;
};
struct _pi_platform {
  _pi_platform() = default;

  // Single-entry Cache pi_devices for reuse
  std::unique_ptr<_pi_device> PiDeviceCache;
  std::mutex PiDeviceCacheMutex;
  bool DeviceCachePopulated = false;

  // Check the device cache and load it if necessary.
  pi_result populateDeviceCacheIfNeeded();

  // Keep Version information.
  std::string CmEmuVersion;
};

struct _pi_device : _pi_object {
  _pi_device(pi_platform ArgPlt, cm_support::CmDevice *ArgCmDev,
             std::string ArgVersionStr)
      : Platform{ArgPlt}, CmDevicePtr{ArgCmDev}, VersionStr{ArgVersionStr} {}

  pi_platform Platform;
  cm_support::CmDevice *CmDevicePtr = nullptr;

  std::string VersionStr;
};

struct _pi_context : _pi_object {
  _pi_context(pi_device ArgDevice) : Device{ArgDevice} {}

  /// One-to-one mapping between Context and Device
  pi_device Device;

  /// Map SVM memory starting address to corresponding
  /// CmBufferSVM object. CmBufferSVM object is needed to release memory.
  std::unordered_map<void *, cm_support::CmBufferSVM *> Addr2CmBufferSVM;
  // Thread-safe mapping management of Addr2CmBufferSVM
  std::mutex CmSVMMapMutex;
};

struct _pi_queue : _pi_object {
  _pi_queue(pi_context ContextArg, cm_support::CmQueue *CmQueueArg)
      : Context{ContextArg}, CmQueuePtr{CmQueueArg} {}

  // Keeps the PI context to which this queue belongs.
  pi_context Context = nullptr;
  cm_support::CmQueue *CmQueuePtr = nullptr;
};

struct _pi_mem : _pi_object {
  static const int HOST_SURFACE_INDEX = (-1);
  _pi_mem() = default;

  pi_context Context;

  // To be used for piEnqueueMemBufferMap
  char *MapHostPtr = nullptr;

  // Mutex for load/store accessing
  std::mutex mutexLock;

  // Surface index used by CM
  int SurfaceIndex;

  // Supplementary data to keep track of the mappings of this memory
  // created with piEnqueueMemBufferMap
  struct Mapping {
    size_t Offset;
    size_t Size;
  };

  virtual ~_pi_mem() = default;

  _pi_mem_type getMemType() const { return MemType; };

  pi_result addMapping(void *MappedTo, size_t SizeArg, size_t OffsetArg);
  pi_result removeMapping(void *MappedTo, Mapping &MapInfo);

protected:
  _pi_mem(pi_context ctxt, char *HostPtr, _pi_mem_type MemTypeArg,
          int SurfaceIdxArg)
      : Context{ctxt}, MapHostPtr{HostPtr},
        SurfaceIndex{SurfaceIdxArg}, MemType{MemTypeArg} {}

private:
  _pi_mem_type MemType;

  std::unordered_map<void *, Mapping> Mappings;
  // Thread-safe mapping management for piEnqueueMemBufferMap
  std::mutex MappingsMutex;
};

struct _pi_buffer final : _pi_mem {
  // Buffer/Sub-buffer constructor
  _pi_buffer(pi_context ctxt, char *HostPtr, cm_support::CmBuffer *CmBufArg,
             int SurfaceIdxArg, size_t SizeArg)
      : _pi_mem(ctxt, HostPtr, PI_MEM_TYPE_BUFFER, SurfaceIdxArg),
        CmBufferPtr{CmBufArg}, Size{SizeArg} {}

  cm_support::CmBuffer *CmBufferPtr;
  size_t Size;
};

struct _pi_image final : _pi_mem {
  // Image constructor
  _pi_image(pi_context ctxt, char *HostPtr, cm_support::CmSurface2D *CmSurfArg,
            int SurfaceIdxArg, size_t WidthArg, size_t HeightArg, size_t BPPArg)
      : _pi_mem(ctxt, HostPtr, PI_MEM_TYPE_IMAGE2D, SurfaceIdxArg),
        CmSurfacePtr{CmSurfArg}, Width{WidthArg}, Height{HeightArg},
        BytesPerPixel{BPPArg} {}

  cm_support::CmSurface2D *CmSurfacePtr;
  size_t Width;
  size_t Height;
  size_t BytesPerPixel;
};

struct _pi_event : _pi_object {
  _pi_event() {}

  cm_support::CmEvent *CmEventPtr = nullptr;
  cm_support::CmQueue *OwnerQueue = nullptr;
  pi_context Context = nullptr;
  bool IsDummyEvent = false;
};

struct _pi_program : _pi_object {
  _pi_program() {}

  // Keep the context of the program.
  pi_context Context;
};

struct _pi_kernel : _pi_object {
  _pi_kernel() {}
};

#include <sycl/ext/intel/experimental/esimd/emu/detail/esimd_emulator_device_interface.hpp>
