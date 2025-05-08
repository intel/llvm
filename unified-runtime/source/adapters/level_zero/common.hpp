//===--------- common.hpp - Level Zero Adapter ----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cassert>
#include <list>
#include <map>
#include <mutex>
#include <stdarg.h>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include "windows.h"
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

#include <loader/ze_loader.h>
#include <ur/ur.hpp>
#include <ur_ddi.h>
#include <ze_api.h>
#include <zes_api.h>

#include <level_zero/ze_intel_gpu.h>
#include <umf_pools/disjoint_pool_config_parser.hpp>

#include "logger/ur_logger.hpp"

struct _ur_platform_handle_t;

[[maybe_unused]] static bool checkL0LoaderTeardown() {
  try {
    if (!zelCheckIsLoaderInTearDown()) {
      return true;
    }
  } catch (...) {
  }
  UR_LOG(DEBUG,
         "ZE ---> checkL0LoaderTeardown: Loader is in teardown or is unstable");
  return false;
}

// Controls UR L0 calls tracing.
enum UrDebugLevel {
  UR_L0_DEBUG_NONE = 0x0,
  UR_L0_DEBUG_BASIC = 0x1,
  UR_L0_DEBUG_VALIDATION = 0x2,
  UR_L0_DEBUG_ALL = -1
};

const int UrL0Debug = [] {
  const char *ZeDebugMode = std::getenv("ZE_DEBUG");
  const char *UrL0DebugMode = std::getenv("UR_L0_DEBUG");
  uint32_t DebugMode = 0;
  if (UrL0DebugMode) {
    DebugMode = std::atoi(UrL0DebugMode);
  } else if (ZeDebugMode) {
    DebugMode = std::atoi(ZeDebugMode);
  }
  return DebugMode;
}();

const int UrL0LeaksDebug = [] {
  const char *UrRet = std::getenv("UR_L0_LEAKS_DEBUG");
  if (!UrRet)
    return 0;

  return std::atoi(UrRet);
}();

// Enable for UR L0 Adapter to Init all L0 Drivers on the system with filtering
// in place for only currently used Drivers.
const int UrL0InitAllDrivers = [] {
  const char *UrRet = std::getenv("UR_L0_INIT_ALL_DRIVERS");
  if (!UrRet)
    return 0;
  return std::atoi(UrRet);
}();

// Controls Level Zero calls serialization to w/a Level Zero driver being not MT
// ready. Recognized values (can be used as a bit mask):
enum {
  UrL0SerializeNone =
      0, // no locking or blocking (except when SYCL RT requested blocking)
  UrL0SerializeLock = 1, // locking around each UR_CALL
  UrL0SerializeBlock =
      2, // blocking UR calls, where supported (usually in enqueue commands)
};

static const uint32_t UrL0Serialize = [] {
  const char *ZeSerializeMode = std::getenv("ZE_SERIALIZE");
  const char *UrL0SerializeMode = std::getenv("UR_L0_SERIALIZE");
  uint32_t SerializeModeValue = 0;
  if (UrL0SerializeMode) {
    SerializeModeValue = std::atoi(UrL0SerializeMode);
  } else if (ZeSerializeMode) {
    SerializeModeValue = std::atoi(ZeSerializeMode);
  }
  return SerializeModeValue;
}();

static const uint32_t UrL0QueueSyncNonBlocking = [] {
  const char *UrL0QueueSyncNonBlocking =
      std::getenv("UR_L0_QUEUE_SYNCHRONIZE_NON_BLOCKING");
  uint32_t L0QueueSyncLockingModeValue = 1;
  if (UrL0QueueSyncNonBlocking) {
    L0QueueSyncLockingModeValue = std::atoi(UrL0QueueSyncNonBlocking);
  }
  return L0QueueSyncLockingModeValue;
}();

// Controls whether the L0 Adapter creates signal events for commands on
// integrated gpu devices.
static const uint32_t UrL0OutOfOrderIntegratedSignalEvent = [] {
  const char *UrL0OutOfOrderIntegratedSignalEventEnv =
      std::getenv("UR_L0_OOQ_INTEGRATED_SIGNAL_EVENT");
  uint32_t UrL0OutOfOrderIntegratedSignalEventValue = 1;
  if (UrL0OutOfOrderIntegratedSignalEventEnv) {
    UrL0OutOfOrderIntegratedSignalEventValue =
        std::atoi(UrL0OutOfOrderIntegratedSignalEventEnv);
  }
  return UrL0OutOfOrderIntegratedSignalEventValue;
}();

// This class encapsulates actions taken along with a call to Level Zero API.
class ZeCall {
private:
  // The global mutex that is used for total serialization of Level Zero calls.
  static std::mutex GlobalLock;

public:
  ZeCall() {
    if ((UrL0Serialize & UrL0SerializeLock) != 0) {
      GlobalLock.lock();
    }
  }
  ~ZeCall() {
    if ((UrL0Serialize & UrL0SerializeLock) != 0) {
      GlobalLock.unlock();
    }
  }

  // The non-static version just calls static one.
  ze_result_t doCall(ze_result_t ZeResult, const char *ZeName,
                     const char *ZeArgs, bool TraceError = true);
};

// This function will ensure compatibility with both Linux and Windows for
// setting environment variables.
bool setEnvVar(const char *name, const char *value);

// Returns the ze_structure_type_t to use in .stype of a structured descriptor.
// Intentionally not defined; will give an error if no proper specialization
template <class T> ze_structure_type_t getZeStructureType();
template <class T> zes_structure_type_t getZesStructureType();

// The helpers to properly default initialize Level-Zero descriptor and
// properties structures.
template <class T> struct ZeStruct : public T {
  ZeStruct() : T{} { // zero initializes base struct
    this->stype = getZeStructureType<T>();
    this->pNext = nullptr;
  }
};

template <class T> struct ZesStruct : public T {
  ZesStruct() : T{} { // zero initializes base struct
    this->stype = getZesStructureType<T>();
    this->pNext = nullptr;
  }
};

// This function will ensure compatibility with both Linux and Windows for
// setting environment variables.
bool setEnvVar(const char *name, const char *value);

// Map Level Zero runtime error code to UR error code.
ur_result_t ze2urResult(ze_result_t ZeResult);

// Parse Level Zero error code and return the error string.
void zeParseError(ze_result_t ZeError, const char *&ErrorString);

// Trace a call to Level-Zero RT
#define ZE2UR_CALL(ZeName, ZeArgs)                                             \
  {                                                                            \
    ze_result_t ZeResult = ZeName ZeArgs;                                      \
    if (auto Result = ZeCall().doCall(ZeResult, #ZeName, #ZeArgs, true))       \
      return ze2urResult(Result);                                              \
  }

// Trace a call to Level-Zero RT, throw on error
#define ZE2UR_CALL_THROWS(ZeName, ZeArgs)                                      \
  {                                                                            \
    ze_result_t ZeResult = ZeName ZeArgs;                                      \
    if (auto Result = ZeCall().doCall(ZeResult, #ZeName, #ZeArgs, true))       \
      throw ze2urResult(Result);                                               \
  }

// Perform traced call to L0 without checking for errors
#define ZE_CALL_NOCHECK(ZeName, ZeArgs)                                        \
  ZeCall().doCall(ZeName ZeArgs, #ZeName, #ZeArgs, false)

#define ZE_CALL_NOCHECK_NAME(ZeName, ZeArgs, callName)                         \
  ZeCall().doCall(ZeName ZeArgs, callName, #ZeArgs, false)

// This wrapper around std::atomic is created to limit operations with reference
// counter and to make allowed operations more transparent in terms of
// thread-safety in the plugin. increment() and load() operations do not need a
// mutex guard around them since the underlying data is already atomic.
// decrementAndTest() method is used to guard a code which needs to be
// executed when object's ref count becomes zero after release. This method also
// doesn't need a mutex guard because decrement operation is atomic and only one
// thread can reach ref count equal to zero, i.e. only a single thread can pass
// through this check.
struct ReferenceCounter {
  ReferenceCounter() : RefCount{1} {}

  // Reset the counter to the initial value.
  void reset() { RefCount = 1; }

  // Used when retaining an object.
  void increment() { RefCount++; }

  // Supposed to be used in ur*GetInfo* methods where ref count value is
  // requested.
  uint32_t load() { return RefCount.load(); }

  // This method allows to guard a code which needs to be executed when object's
  // ref count becomes zero after release. It is important to notice that only a
  // single thread can pass through this check. This is true because of several
  // reasons:
  //   1. Decrement operation is executed atomically.
  //   2. It is not allowed to retain an object after its refcount reaches zero.
  //   3. It is not allowed to release an object more times than the value of
  //   the ref count.
  // 2. and 3. basically means that we can't use an object at all as soon as its
  // refcount reaches zero. Using this check guarantees that code for deleting
  // an object and releasing its resources is executed once by a single thread
  // and we don't need to use any mutexes to guard access to this object in the
  // scope after this check. Of course if we access another objects in this code
  // (not the one which is being deleted) then access to these objects must be
  // guarded, for example with a mutex.
  bool decrementAndTest() { return --RefCount == 0; }

private:
  std::atomic<uint32_t> RefCount;
};

// Base class to store common data
struct ur_object {
  ur_object() : RefCount{} {}

  // Must be atomic to prevent data race when incrementing/decrementing.
  ReferenceCounter RefCount;

  // This mutex protects accesses to all the non-const member variables.
  // Exclusive access is required to modify any of these members.
  //
  // To get shared access to the object in a scope use std::shared_lock:
  //    std::shared_lock Lock(Obj->Mutex);
  // To get exclusive access to the object in a scope use std::scoped_lock:
  //    std::scoped_lock Lock(Obj->Mutex);
  //
  // If several UR objects are accessed in a scope then each object's mutex must
  // be locked. For example, to get write access to Obj1 and Obj2 and read
  // access to Obj3 in a scope use the following approach:
  //   std::shared_lock Obj3Lock(Obj3->Mutex, std::defer_lock);
  //   std::scoped_lock LockAll(Obj1->Mutex, Obj2->Mutex, Obj3Lock);
  ur_shared_mutex Mutex;

  // Indicates if we own the native handle or it came from interop that
  // asked to not transfer the ownership to SYCL RT.
  bool OwnNativeHandle = false;
};

// Record for a memory allocation. This structure is used to keep information
// for each memory allocation.
struct MemAllocRecord : ur_object {
  MemAllocRecord(ur_context_handle_t Context, bool OwnZeMemHandle = true)
      : Context(Context) {
    OwnNativeHandle = OwnZeMemHandle;
  }
  // Currently kernel can reference memory allocations from different contexts
  // and we need to know the context of a memory allocation when we release it
  // in piKernelRelease.
  // TODO: this should go away when memory isolation issue is fixed in the Level
  // Zero runtime.
  ur_context_handle_t Context;
};

extern usm::DisjointPoolAllConfigs DisjointPoolConfigInstance;
extern const bool UseUSMAllocator;

// Controls support of the indirect access kernels and deferred memory release.
const bool IndirectAccessTrackingEnabled = [] {
  char *UrRet = std::getenv("UR_L0_TRACK_INDIRECT_ACCESS_MEMORY");
  char *PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_TRACK_INDIRECT_ACCESS_MEMORY");
  const bool RetVal = UrRet ? std::stoi(UrRet) : (PiRet ? std::stoi(PiRet) : 0);
  return RetVal;
}();

extern const bool UseUSMAllocator;

const bool ExposeCSliceInAffinityPartitioning = [] {
  char *UrRet = std::getenv("UR_L0_EXPOSE_CSLICE_IN_AFFINITY_PARTITIONING");
  char *PiRet =
      std::getenv("SYCL_PI_LEVEL_ZERO_EXPOSE_CSLICE_IN_AFFINITY_PARTITIONING");
  const char *Flag = UrRet ? UrRet : (PiRet ? PiRet : 0);
  return Flag ? std::atoi(Flag) != 0 : false;
}();

// TODO: make it into a ur_device_handle_t class member
const std::pair<int, int>
getRangeOfAllowedCopyEngines(const ur_device_handle_t &Device);

class ZeDriverVersionStringExtension {
  // Pointer to function for Intel Driver Version String
  ze_result_t (*zeIntelGetDriverVersionStringPointer)(
      ze_driver_handle_t hDriver, char *, size_t *) = nullptr;

public:
  // Whether platform supports Intel Driver Version String.
  bool Supported;

  ZeDriverVersionStringExtension() : Supported{false} {}

  void setZeDriverVersionString(ur_platform_handle_t_ *Platform);
  void getDriverVersionString(ze_driver_handle_t DriverHandle,
                              char *pDriverVersion, size_t *pVersionSize);
};

class ZeUSMImportExtension {
  // Pointers to functions that import/release host memory into USM
  ze_result_t (*zexDriverImportExternalPointer)(ze_driver_handle_t hDriver,
                                                void *, size_t) = nullptr;
  ze_result_t (*zexDriverReleaseImportedPointer)(ze_driver_handle_t,
                                                 void *) = nullptr;

public:
  // Whether platform supports Import/Release.
  bool Supported;

  // Whether user has requested Import/Release for buffers.
  bool Enabled;

  ZeUSMImportExtension() : Supported{false}, Enabled{false} {}

  void setZeUSMImport(ur_platform_handle_t_ *Platform);
  void doZeUSMImport(ze_driver_handle_t DriverHandle, void *HostPtr,
                     size_t Size);
  void doZeUSMRelease(ze_driver_handle_t DriverHandle, void *HostPtr);
};

// Helper wrapper for working with USM import extension in Level Zero.
extern ZeUSMImportExtension ZeUSMImport;

// This will count the calls to Level-Zero
extern std::map<std::string, int> *ZeCallCount;

// Some opencl extensions we know are supported by all Level Zero devices.
constexpr char ZE_SUPPORTED_EXTENSIONS[] =
    "cl_khr_il_program cl_khr_subgroups cl_intel_subgroups "
    "cl_intel_subgroups_short cl_intel_required_subgroup_size ";

// Global variables for ZER_EXT_RESULT_ADAPTER_SPECIFIC_ERROR
constexpr size_t MaxMessageSize = 256;
extern thread_local int32_t ErrorMessageCode;
extern thread_local char ErrorMessage[MaxMessageSize];
extern thread_local int32_t ErrorAdapterNativeCode;

// Utility function for setting a message and warning
[[maybe_unused]] void setErrorMessage(const char *pMessage, int32_t ErrorCode,
                                      int32_t AdapterErrorCode);
