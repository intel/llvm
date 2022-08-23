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

#include <sycl/detail/iostream_proxy.hpp>
#include <sycl/detail/pi.h>

#include <atomic>
#include <cassert>
#include <mutex>
#include <unordered_map>

#include <malloc.h>

// This version should be incremented for any change made to this file or its
// corresponding .cpp file.
#define _PI_ESIMD_PLUGIN_VERSION 1

#define _PI_ESIMD_PLUGIN_VERSION_STRING                                        \
  _PI_PLUGIN_VERSION_STRING(_PI_ESIMD_PLUGIN_VERSION)

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

  virtual ~_pi_object() = default;
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
  // TODO: Check if serialization is required when ESIMD_EMULATOR
  // plug-in calls CM runtime functions
  cm_support::CmDevice *CmDevicePtr = nullptr;

  std::string VersionStr;
};

struct _pi_context : _pi_object {
  _pi_context(pi_device ArgDevice) : Device{ArgDevice} {}

  // One-to-one mapping between Context and Device
  pi_device Device;

  // Map SVM memory starting address to corresponding
  // CmBufferSVM object. CmBufferSVM object is needed to release memory.
  std::unordered_map<void *, cm_support::CmBufferSVM *> Addr2CmBufferSVM;
  // A lock guarding access to Addr2CmBufferSVM
  std::mutex Addr2CmBufferSVMLock;

  bool checkSurfaceArgument(pi_mem_flags Flags, void *HostPtr);
};

struct _pi_queue : _pi_object {
  _pi_queue(pi_context ContextArg, cm_support::CmQueue *CmQueueArg)
      : Context{ContextArg}, CmQueuePtr{CmQueueArg} {}

  // Keeps the PI context to which this queue belongs.
  pi_context Context = nullptr;
  cm_support::CmQueue *CmQueuePtr = nullptr;
};

struct cm_surface_ptr_t {
  // 'UP' means 'User-Provided' in CM Lib - corresponding to
  // Buffer/Image created with PI_MEM_FLAGS_HOST_PTR_USE option in
  // SYCL
  enum SurfaceType {
    TypeNone,
    TypeRegularBuffer,
    TypeUserProvidedBuffer,
    TypeRegularImage,
    TypeUserProvidedImage
  };
  SurfaceType tag = TypeNone;

  union {
    cm_support::CmBuffer *RegularBufPtr = nullptr;
    cm_support::CmBufferUP *UPBufPtr;
    cm_support::CmSurface2D *RegularImgPtr;
    cm_support::CmSurface2DUP *UPImgPtr;
  };
};

struct _pi_mem : _pi_object {
  pi_context Context;

  // To be used for piEnqueueMemBufferMap
  char *MapHostPtr = nullptr;

  std::mutex SurfaceLock;

  // Surface index

  unsigned int SurfaceIndex;
  // Supplementary data to keep track of the mappings of this memory
  // created with piEnqueueMemBufferMap
  struct Mapping {
    // The offset in the buffer giving the start of the mapped region.
    size_t Offset = 0;
    // The size of the mapped region.
    size_t Size = 0;
  };

  // The key is the host pointer representing an active mapping.
  // The value is the information needed to maintain/undo the mapping.
  // TODO : std::unordered_map is imported from L0.
  // Use std::stack for strict LIFO behavior checking?
  std::unordered_map<void *, Mapping> Mappings;
  // Supporing multi-threaded mapping/unmapping calls
  std::mutex MappingsMutex;

  cm_surface_ptr_t SurfacePtr;

  // Destructor for invoking buffer/image destory calls in CM Runtime
  ~_pi_mem();

protected:
  _pi_mem(pi_context ctxt, char *HostPtr, cm_surface_ptr_t SurfacePtrArg,
          unsigned int SurfaceIdxArg)
      : Context{ctxt}, MapHostPtr{HostPtr}, SurfaceIndex{SurfaceIdxArg},
        SurfacePtr{SurfacePtrArg} {}
};

struct _pi_buffer final : _pi_mem {
  // Buffer/Sub-buffer constructor
  _pi_buffer(pi_context ctxt, char *HostPtr, cm_surface_ptr_t SurfacePtrArg,
             unsigned int SurfaceIdxArg, size_t SizeArg)
      : _pi_mem(ctxt, HostPtr, SurfacePtrArg, SurfaceIdxArg), Size{SizeArg} {}

  size_t Size;
};

struct _pi_image final : _pi_mem {
  // Image constructor
  _pi_image(pi_context ctxt, char *HostPtr, cm_surface_ptr_t SurfacePtrArg,
            unsigned int SurfaceIdxArg, size_t WidthArg, size_t HeightArg,
            size_t BPPArg)
      : _pi_mem(ctxt, HostPtr, SurfacePtrArg, SurfaceIdxArg), Width{WidthArg},
        Height{HeightArg}, BytesPerPixel{BPPArg} {}

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

#include <sycl/ext/intel/esimd/emu/detail/esimd_emulator_device_interface.hpp>
