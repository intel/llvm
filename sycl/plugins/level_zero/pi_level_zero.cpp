//===-------- pi_level_zero.cpp - Level Zero Plugin --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//

/// \file pi_level_zero.cpp
/// Implementation of Level Zero Plugin.
///
/// \ingroup sycl_pi_level_zero

#include "pi_level_zero.hpp"
#include <CL/sycl/detail/spinlock.hpp>
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#include <level_zero/zet_api.h>

#include "usm_allocator.hpp"

extern "C" {
// Forward declarartions.
static pi_result EventRelease(pi_event Event, pi_queue LockedQueue);
static pi_result QueueRelease(pi_queue Queue, pi_queue LockedQueue);
static pi_result EventCreate(pi_context Context, bool HostVisible,
                             pi_event *RetEvent);
}

namespace {

// Controls Level Zero calls serialization to w/a Level Zero driver being not MT
// ready. Recognized values (can be used as a bit mask):
enum {
  ZeSerializeNone =
      0, // no locking or blocking (except when SYCL RT requested blocking)
  ZeSerializeLock = 1, // locking around each ZE_CALL
  ZeSerializeBlock =
      2, // blocking ZE calls, where supported (usually in enqueue commands)
};
static const pi_uint32 ZeSerialize = [] {
  const char *SerializeMode = std::getenv("ZE_SERIALIZE");
  const pi_uint32 SerializeModeValue =
      SerializeMode ? std::atoi(SerializeMode) : 0;
  return SerializeModeValue;
}();

// This is an experimental option to test performance of device to device copy
// operations on copy engines (versus compute engine)
static const bool UseCopyEngineForD2DCopy = [] {
  const char *CopyEngineForD2DCopy =
      std::getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY");
  return (CopyEngineForD2DCopy && (std::stoi(CopyEngineForD2DCopy) != 0));
}();

// This is an experimental option that allows the use of copy engine, if
// available in the device, in Level Zero plugin for copy operations submitted
// to an in-order queue. The default is 1.
static const bool UseCopyEngineForInOrderQueue = [] {
  const char *CopyEngineForInOrderQueue =
      std::getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE");
  return (!CopyEngineForInOrderQueue ||
          (std::stoi(CopyEngineForInOrderQueue) != 0));
}();

// This class encapsulates actions taken along with a call to Level Zero API.
class ZeCall {
private:
  // The global mutex that is used for total serialization of Level Zero calls.
  static std::mutex GlobalLock;

public:
  ZeCall() {
    if ((ZeSerialize & ZeSerializeLock) != 0) {
      GlobalLock.lock();
    }
  }
  ~ZeCall() {
    if ((ZeSerialize & ZeSerializeLock) != 0) {
      GlobalLock.unlock();
    }
  }

  // The non-static version just calls static one.
  ze_result_t doCall(ze_result_t ZeResult, const char *ZeName,
                     const char *ZeArgs, bool TraceError = true);
};
std::mutex ZeCall::GlobalLock;

// Controls PI level tracing prints.
static bool PrintPiTrace = false;

// Controls support of the indirect access kernels and deferred memory release.
static const bool IndirectAccessTrackingEnabled = [] {
  return std::getenv("SYCL_PI_LEVEL_ZERO_TRACK_INDIRECT_ACCESS_MEMORY") !=
         nullptr;
}();

// Map Level Zero runtime error code to PI error code.
static pi_result mapError(ze_result_t ZeResult) {
  // TODO: these mapping need to be clarified and synced with the PI API return
  // values, which is TBD.
  static std::unordered_map<ze_result_t, pi_result> ErrorMapping = {
      {ZE_RESULT_SUCCESS, PI_SUCCESS},
      {ZE_RESULT_ERROR_DEVICE_LOST, PI_DEVICE_NOT_FOUND},
      {ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS, PI_INVALID_OPERATION},
      {ZE_RESULT_ERROR_NOT_AVAILABLE, PI_INVALID_OPERATION},
      {ZE_RESULT_ERROR_UNINITIALIZED, PI_INVALID_PLATFORM},
      {ZE_RESULT_ERROR_INVALID_ARGUMENT, PI_INVALID_ARG_VALUE},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, PI_INVALID_VALUE},
      {ZE_RESULT_ERROR_INVALID_SIZE, PI_INVALID_VALUE},
      {ZE_RESULT_ERROR_UNSUPPORTED_SIZE, PI_INVALID_VALUE},
      {ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT, PI_INVALID_VALUE},
      {ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT, PI_INVALID_EVENT},
      {ZE_RESULT_ERROR_INVALID_ENUMERATION, PI_INVALID_VALUE},
      {ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION, PI_INVALID_VALUE},
      {ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT, PI_INVALID_VALUE},
      {ZE_RESULT_ERROR_INVALID_NATIVE_BINARY, PI_INVALID_BINARY},
      {ZE_RESULT_ERROR_INVALID_KERNEL_NAME, PI_INVALID_KERNEL_NAME},
      {ZE_RESULT_ERROR_INVALID_FUNCTION_NAME, PI_BUILD_PROGRAM_FAILURE},
      {ZE_RESULT_ERROR_OVERLAPPING_REGIONS, PI_INVALID_OPERATION},
      {ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION,
       PI_INVALID_WORK_GROUP_SIZE},
      {ZE_RESULT_ERROR_MODULE_BUILD_FAILURE, PI_BUILD_PROGRAM_FAILURE},
      {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY, PI_OUT_OF_RESOURCES},
      {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, PI_OUT_OF_HOST_MEMORY}};

  auto It = ErrorMapping.find(ZeResult);
  if (It == ErrorMapping.end()) {
    return PI_ERROR_UNKNOWN;
  }
  return It->second;
}

// This will count the calls to Level-Zero
static std::map<const char *, int> *ZeCallCount = nullptr;

// Trace a call to Level-Zero RT
#define ZE_CALL(ZeName, ZeArgs)                                                \
  {                                                                            \
    ze_result_t ZeResult = ZeName ZeArgs;                                      \
    if (auto Result = ZeCall().doCall(ZeResult, #ZeName, #ZeArgs, true))       \
      return mapError(Result);                                                 \
  }

#define ZE_CALL_NOCHECK(ZeName, ZeArgs)                                        \
  ZeCall().doCall(ZeName ZeArgs, #ZeName, #ZeArgs, false)

// Trace an internal PI call; returns in case of an error.
#define PI_CALL(Call)                                                          \
  {                                                                            \
    if (PrintPiTrace)                                                          \
      fprintf(stderr, "PI ---> %s\n", #Call);                                  \
    pi_result Result = (Call);                                                 \
    if (Result != PI_SUCCESS)                                                  \
      return Result;                                                           \
  }

enum DebugLevel {
  ZE_DEBUG_NONE = 0x0,
  ZE_DEBUG_BASIC = 0x1,
  ZE_DEBUG_VALIDATION = 0x2,
  ZE_DEBUG_CALL_COUNT = 0x4,
  ZE_DEBUG_ALL = -1
};

// Controls Level Zero calls tracing.
static const int ZeDebug = [] {
  const char *DebugMode = std::getenv("ZE_DEBUG");
  return DebugMode ? std::atoi(DebugMode) : ZE_DEBUG_NONE;
}();

static void zePrint(const char *Format, ...) {
  if (ZeDebug & ZE_DEBUG_BASIC) {
    va_list Args;
    va_start(Args, Format);
    vfprintf(stderr, Format, Args);
    va_end(Args);
  }
}

// Controls whether device-scope events are used, and how.
static const enum EventsScope {
  // All events are created host-visible (the default mode)
  AllHostVisible,
  // All events are created with device-scope and only when
  // host waits them or queries their status that a proxy
  // host-visible event is created and set to signal after
  // original event signals.
  OnDemandHostVisibleProxy,
  // All events are created with device-scope and only
  // when a batch of commands is submitted for execution a
  // last command in that batch is added to signal host-visible
  // completion of each command in this batch.
  LastCommandInBatchHostVisible
} EventsScope = [] {
  const auto DeviceEventsStr =
      std::getenv("SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS");

  switch (DeviceEventsStr ? std::atoi(DeviceEventsStr) : 0) {
  case 1:
    return OnDemandHostVisibleProxy;
  case 2:
    return LastCommandInBatchHostVisible;
  }
  return AllHostVisible;
}();

// Maximum number of events that can be present in an event ZePool is captured
// here. Setting it to 256 gave best possible performance for several
// benchmarks.
static const pi_uint32 MaxNumEventsPerPool = [] {
  const auto MaxNumEventsPerPoolEnv =
      std::getenv("ZE_MAX_NUMBER_OF_EVENTS_PER_EVENT_POOL");
  pi_uint32 Result =
      MaxNumEventsPerPoolEnv ? std::atoi(MaxNumEventsPerPoolEnv) : 256;
  if (Result <= 0)
    Result = 256;
  return Result;
}();

// Helper function to implement zeHostSynchronize.
// The behavior is to avoid infinite wait during host sync under ZE_DEBUG.
// This allows for a much more responsive debugging of hangs.
//
template <typename T, typename Func>
ze_result_t zeHostSynchronizeImpl(Func Api, T Handle) {
  if (!ZeDebug) {
    return Api(Handle, UINT64_MAX);
  }

  ze_result_t R;
  while ((R = Api(Handle, 1000)) == ZE_RESULT_NOT_READY)
    ;
  return R;
}

// Template function to do various types of host synchronizations.
// This is intended to be used instead of direct calls to specific
// Level-Zero synchronization APIs.
//
template <typename T> ze_result_t zeHostSynchronize(T Handle);
template <> ze_result_t zeHostSynchronize(ze_event_handle_t Handle) {
  return zeHostSynchronizeImpl(zeEventHostSynchronize, Handle);
}
template <> ze_result_t zeHostSynchronize(ze_command_queue_handle_t Handle) {
  return zeHostSynchronizeImpl(zeCommandQueueSynchronize, Handle);
}
template <> ze_result_t zeHostSynchronize(ze_fence_handle_t Handle) {
  return zeHostSynchronizeImpl(zeFenceHostSynchronize, Handle);
}

template <typename T, typename Assign>
pi_result getInfoImpl(size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret, T value, size_t value_size,
                      Assign &&assign_func) {

  if (param_value != nullptr) {

    if (param_value_size < value_size) {
      return PI_INVALID_VALUE;
    }

    assign_func(param_value, value, value_size);
  }

  if (param_value_size_ret != nullptr) {
    *param_value_size_ret = value_size;
  }

  return PI_SUCCESS;
}

template <typename T>
pi_result getInfo(size_t param_value_size, void *param_value,
                  size_t *param_value_size_ret, T value) {

  auto assignment = [](void *param_value, T value, size_t value_size) {
    (void)value_size;
    *static_cast<T *>(param_value) = value;
  };

  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     sizeof(T), assignment);
}

template <typename T>
pi_result getInfoArray(size_t array_length, size_t param_value_size,
                       void *param_value, size_t *param_value_size_ret,
                       T *value) {
  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     array_length * sizeof(T), memcpy);
}

template <typename T, typename RetType>
pi_result getInfoArray(size_t array_length, size_t param_value_size,
                       void *param_value, size_t *param_value_size_ret,
                       T *value) {
  if (param_value) {
    memset(param_value, 0, param_value_size);
    for (uint32_t I = 0; I < array_length; I++)
      ((RetType *)param_value)[I] = (RetType)value[I];
  }
  if (param_value_size_ret)
    *param_value_size_ret = array_length * sizeof(RetType);
  return PI_SUCCESS;
}

template <>
pi_result getInfo<const char *>(size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret,
                                const char *value) {
  return getInfoArray(strlen(value) + 1, param_value_size, param_value,
                      param_value_size_ret, value);
}

class ReturnHelper {
public:
  ReturnHelper(size_t param_value_size, void *param_value,
               size_t *param_value_size_ret)
      : param_value_size(param_value_size), param_value(param_value),
        param_value_size_ret(param_value_size_ret) {}

  template <class T> pi_result operator()(const T &t) {
    return getInfo(param_value_size, param_value, param_value_size_ret, t);
  }

private:
  size_t param_value_size;
  void *param_value;
  size_t *param_value_size_ret;
};

} // anonymous namespace

// SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE can be set to an integer value, or
// a pair of integer values of the form "lower_index:upper_index".
// Here, the indices point to copy engines in a list of all available copy
// engines.
// This functions returns this pair of indices.
// If the user specifies only a single integer, a value of 0 indicates that
// the copy engines will not be used at all. A value of 1 indicates that all
// available copy engines can be used.
static const std::pair<int, int> getRangeOfAllowedCopyEngines = [] {
  const char *EnvVar = std::getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE");
  // If the environment variable is not set, all available copy engines can be
  // used.
  if (!EnvVar)
    return std::pair<int, int>(0, INT_MAX);
  std::string CopyEngineRange = EnvVar;
  // Environment variable can be a single integer or a pair of integers
  // separated by ":"
  auto pos = CopyEngineRange.find(":");
  if (pos == std::string::npos) {
    bool UseCopyEngine = (std::stoi(CopyEngineRange) != 0);
    if (UseCopyEngine)
      return std::pair<int, int>(0, INT_MAX); // All copy engines can be used.
    return std::pair<int, int>(-1, -1);       // No copy engines will be used.
  }
  int LowerCopyEngineIndex = std::stoi(CopyEngineRange.substr(0, pos));
  int UpperCopyEngineIndex = std::stoi(CopyEngineRange.substr(pos + 1));
  if ((LowerCopyEngineIndex > UpperCopyEngineIndex) ||
      (LowerCopyEngineIndex < -1) || (UpperCopyEngineIndex < -1)) {
    zePrint("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE: invalid value provided, "
            "default set.\n");
    LowerCopyEngineIndex = 0;
    UpperCopyEngineIndex = INT_MAX;
  }
  return std::pair<int, int>(LowerCopyEngineIndex, UpperCopyEngineIndex);
}();

static const bool CopyEngineRequested = [] {
  int LowerCopyQueueIndex = getRangeOfAllowedCopyEngines.first;
  int UpperCopyQueueIndex = getRangeOfAllowedCopyEngines.second;
  return ((LowerCopyQueueIndex != -1) || (UpperCopyQueueIndex != -1));
}();

// Global variables used in PI_Level_Zero
// Note we only create a simple pointer variables such that C++ RT won't
// deallocate them automatically at the end of the main program.
// The heap memory allocated for these global variables reclaimed only when
// Sycl RT calls piTearDown().
static std::vector<pi_platform> *PiPlatformsCache =
    new std::vector<pi_platform>;
static sycl::detail::SpinLock *PiPlatformsCacheMutex =
    new sycl::detail::SpinLock;
static bool PiPlatformCachePopulated = false;

// Flags which tell whether various Level Zero extensions are available.
static bool PiDriverGlobalOffsetExtensionFound = false;
static bool PiDriverModuleProgramExtensionFound = false;

// TODO:: In the following 4 methods we may want to distinguish read access vs.
// write (as it is OK for multiple threads to read the map without locking it).

pi_result _pi_mem::addMapping(void *MappedTo, size_t Offset, size_t Size) {
  std::lock_guard<std::mutex> Lock(MappingsMutex);
  auto Res = Mappings.insert({MappedTo, {Offset, Size}});
  // False as the second value in pair means that mapping was not inserted
  // because mapping already exists.
  if (!Res.second) {
    zePrint("piEnqueueMemBufferMap: duplicate mapping detected\n");
    return PI_INVALID_VALUE;
  }
  return PI_SUCCESS;
}

pi_result _pi_mem::removeMapping(void *MappedTo, Mapping &MapInfo) {
  std::lock_guard<std::mutex> Lock(MappingsMutex);
  auto It = Mappings.find(MappedTo);
  if (It == Mappings.end()) {
    zePrint("piEnqueueMemUnmap: unknown memory mapping\n");
    return PI_INVALID_VALUE;
  }
  MapInfo = It->second;
  Mappings.erase(It);
  return PI_SUCCESS;
}

pi_result
_pi_context::getFreeSlotInExistingOrNewPool(ze_event_pool_handle_t &Pool,
                                            size_t &Index, bool HostVisible) {
  // Lock while updating event pool machinery.
  std::lock_guard<std::mutex> Lock(ZeEventPoolCacheMutex);

  // Setup for host-visible pool as needed.
  ze_event_pool_flag_t ZePoolFlag = {};
  std::list<ze_event_pool_handle_t> *ZePoolCache;

  if (HostVisible) {
    ZePoolFlag = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    ZePoolCache = &ZeHostVisibleEventPoolCache;
  } else {
    ZePoolCache = &ZeDeviceScopeEventPoolCache;
  }

  // Remove full pool from the cache.
  if (!ZePoolCache->empty()) {
    if (NumEventsAvailableInEventPool[ZePoolCache->front()] == 0) {
      ZePoolCache->erase(ZePoolCache->begin());
    }
  }
  if (ZePoolCache->empty()) {
    ZePoolCache->push_back(nullptr);
  }

  // We shall be adding an event to the front pool.
  ze_event_pool_handle_t *ZePool = &ZePoolCache->front();
  Index = 0;
  // Create one event ZePool per MaxNumEventsPerPool events
  if (*ZePool == nullptr) {
    ZeStruct<ze_event_pool_desc_t> ZeEventPoolDesc;
    ZeEventPoolDesc.count = MaxNumEventsPerPool;
    ZeEventPoolDesc.flags = ZePoolFlag | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;

    std::vector<ze_device_handle_t> ZeDevices;
    std::for_each(Devices.begin(), Devices.end(),
                  [&](pi_device &D) { ZeDevices.push_back(D->ZeDevice); });

    ZE_CALL(zeEventPoolCreate, (ZeContext, &ZeEventPoolDesc, ZeDevices.size(),
                                &ZeDevices[0], ZePool));
    NumEventsAvailableInEventPool[*ZePool] = MaxNumEventsPerPool - 1;
    NumEventsUnreleasedInEventPool[*ZePool] = 1;
  } else {
    Index = MaxNumEventsPerPool - NumEventsAvailableInEventPool[*ZePool];
    --NumEventsAvailableInEventPool[*ZePool];
    ++NumEventsUnreleasedInEventPool[*ZePool];
  }
  Pool = *ZePool;
  return PI_SUCCESS;
}

pi_result _pi_context::decrementUnreleasedEventsInPool(pi_event Event) {
  if (!Event->ZeEventPool) {
    // This must be an interop event created on a users's pool.
    // Do nothing.
    return PI_SUCCESS;
  }

  std::list<ze_event_pool_handle_t> *ZePoolCache;
  if (Event->IsHostVisible()) {
    ZePoolCache = &ZeHostVisibleEventPoolCache;
  } else {
    ZePoolCache = &ZeDeviceScopeEventPoolCache;
  }

  // Put the empty pool to the cache of the pools.
  std::lock_guard<std::mutex> Lock(ZeEventPoolCacheMutex);
  if (NumEventsUnreleasedInEventPool[Event->ZeEventPool] == 0)
    die("Invalid event release: event pool doesn't have unreleased events");
  if (--NumEventsUnreleasedInEventPool[Event->ZeEventPool] == 0) {
    if (ZePoolCache->front() != Event->ZeEventPool) {
      ZePoolCache->push_back(Event->ZeEventPool);
    }
    NumEventsAvailableInEventPool[Event->ZeEventPool] = MaxNumEventsPerPool;
  }

  return PI_SUCCESS;
}

// Some opencl extensions we know are supported by all Level Zero devices.
constexpr char ZE_SUPPORTED_EXTENSIONS[] =
    "cl_khr_il_program cl_khr_subgroups cl_intel_subgroups "
    "cl_intel_subgroups_short cl_intel_required_subgroup_size ";

// Forward declarations
static pi_result
enqueueMemCopyHelper(pi_command_type CommandType, pi_queue Queue, void *Dst,
                     pi_bool BlockingWrite, size_t Size, const void *Src,
                     pi_uint32 NumEventsInWaitList,
                     const pi_event *EventWaitList, pi_event *Event,
                     bool PreferCopyEngine = false);

static pi_result enqueueMemCopyRectHelper(
    pi_command_type CommandType, pi_queue Queue, void *SrcBuffer,
    void *DstBuffer, pi_buff_rect_offset SrcOrigin,
    pi_buff_rect_offset DstOrigin, pi_buff_rect_region Region,
    size_t SrcRowPitch, size_t SrcSlicePitch, size_t DstRowPitch,
    size_t DstSlicePitch, pi_bool Blocking, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *Event,
    bool PreferCopyEngine = false);

inline void zeParseError(ze_result_t ZeError, const char *&ErrorString) {
  switch (ZeError) {
#define ZE_ERRCASE(ERR)                                                        \
  case ERR:                                                                    \
    ErrorString = "" #ERR;                                                     \
    break;

    ZE_ERRCASE(ZE_RESULT_SUCCESS)
    ZE_ERRCASE(ZE_RESULT_NOT_READY)
    ZE_ERRCASE(ZE_RESULT_ERROR_DEVICE_LOST)
    ZE_ERRCASE(ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY)
    ZE_ERRCASE(ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
    ZE_ERRCASE(ZE_RESULT_ERROR_MODULE_BUILD_FAILURE)
    ZE_ERRCASE(ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS)
    ZE_ERRCASE(ZE_RESULT_ERROR_NOT_AVAILABLE)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNINITIALIZED)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNSUPPORTED_VERSION)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNSUPPORTED_FEATURE)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_ARGUMENT)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_NULL_HANDLE)
    ZE_ERRCASE(ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_NULL_POINTER)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_SIZE)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNSUPPORTED_SIZE)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_ENUMERATION)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_NATIVE_BINARY)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_GLOBAL_NAME)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_KERNEL_NAME)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_FUNCTION_NAME)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE)
    ZE_ERRCASE(ZE_RESULT_ERROR_OVERLAPPING_REGIONS)
    ZE_ERRCASE(ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED)
    ZE_ERRCASE(ZE_RESULT_ERROR_UNKNOWN)

#undef ZE_ERRCASE
  default:
    assert(false && "Unexpected Error code");
  } // switch
}

ze_result_t ZeCall::doCall(ze_result_t ZeResult, const char *ZeName,
                           const char *ZeArgs, bool TraceError) {
  zePrint("ZE ---> %s%s\n", ZeName, ZeArgs);

  if (ZeDebug & ZE_DEBUG_CALL_COUNT) {
    ++(*ZeCallCount)[ZeName];
  }

  if (ZeResult && TraceError) {
    const char *ErrorString = "Unknown";
    zeParseError(ZeResult, ErrorString);
    zePrint("Error (%s) in %s\n", ErrorString, ZeName);
  }
  return ZeResult;
}

#define PI_ASSERT(condition, error)                                            \
  if (!(condition))                                                            \
    return error;

// This helper function increments the reference counter of the Queue
// without guarding with a lock.
// It is the caller's responsibility to make sure the lock is acquired
// on the Queue that is passed in.
inline static void piQueueRetainNoLock(pi_queue Queue) { Queue->RefCount++; }

// This helper function creates a pi_event and associate a pi_queue.
// Note that the caller of this function must have acquired lock on the Queue
// that is passed in.
// \param Queue pi_queue to associate with a new event.
// \param Event a pointer to hold the newly created pi_event
// \param CommandType various command type determined by the caller
// \param CommandList is the command list where the event is added
inline static pi_result
createEventAndAssociateQueue(pi_queue Queue, pi_event *Event,
                             pi_command_type CommandType,
                             pi_command_list_ptr_t CommandList) {
  pi_result Res = piEventCreate(Queue->Context, Event);
  if (Res != PI_SUCCESS)
    return Res;

  (*Event)->Queue = Queue;
  (*Event)->CommandType = CommandType;

  // Append this Event to the CommandList, if any
  if (CommandList != Queue->CommandListMap.end()) {
    (*Event)->ZeCommandList = CommandList->first;
    CommandList->second.append(*Event);
    PI_CALL(piEventRetain(*Event));
  } else {
    (*Event)->ZeCommandList = nullptr;
  }

  // We need to increment the reference counter here to avoid pi_queue
  // being released before the associated pi_event is released because
  // piEventRelease requires access to the associated pi_queue.
  // In piEventRelease, the reference counter of the Queue is decremented
  // to release it.
  piQueueRetainNoLock(Queue);

  // SYCL RT does not track completion of the events, so it could
  // release a PI event as soon as that's not being waited in the app.
  // But we have to ensure that the event is not destroyed before
  // it is really signalled, so retain it explicitly here and
  // release in Event->cleanup().
  //
  PI_CALL(piEventRetain(*Event));

  return PI_SUCCESS;
}

pi_result _pi_device::initialize(int SubSubDeviceOrdinal,
                                 int SubSubDeviceIndex) {
  uint32_t numQueueGroups = 0;
  ZE_CALL(zeDeviceGetCommandQueueGroupProperties,
          (ZeDevice, &numQueueGroups, nullptr));
  if (numQueueGroups == 0) {
    return PI_ERROR_UNKNOWN;
  }
  zePrint("NOTE: Number of queue groups = %d\n", numQueueGroups);
  std::vector<ZeStruct<ze_command_queue_group_properties_t>>
      QueueGroupProperties(numQueueGroups);
  ZE_CALL(zeDeviceGetCommandQueueGroupProperties,
          (ZeDevice, &numQueueGroups, QueueGroupProperties.data()));

  int ComputeGroupIndex = -1;

  // Initialize a sub-sub-device with its own ordinal and index
  if (SubSubDeviceOrdinal >= 0) {
    ComputeGroupIndex = SubSubDeviceOrdinal;
    ZeComputeEngineIndex = SubSubDeviceIndex;
  } else { // This is a root or a sub-device
    for (uint32_t i = 0; i < numQueueGroups; i++) {
      if (QueueGroupProperties[i].flags &
          ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
        ComputeGroupIndex = i;
        break;
      }
    }
    // How is it possible that there are no "compute" capabilities?
    if (ComputeGroupIndex < 0) {
      return PI_ERROR_UNKNOWN;
    }
    // The index for a root or a sub-device is always 0.
    ZeComputeEngineIndex = 0;

    int MainCopyGroupIndex = -1;
    int LinkCopyGroupIndex = -1;
    if (CopyEngineRequested) {
      for (uint32_t i = 0; i < numQueueGroups; i++) {
        if (((QueueGroupProperties[i].flags &
              ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) == 0) &&
            (QueueGroupProperties[i].flags &
             ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY)) {
          if (QueueGroupProperties[i].numQueues == 1)
            MainCopyGroupIndex = i;
          else {
            LinkCopyGroupIndex = i;
            break;
          }
        }
      }
      if (MainCopyGroupIndex < 0)
        zePrint("NOTE: main blitter/copy engine is not available\n");
      else
        zePrint("NOTE: main blitter/copy engine is available\n");

      if (LinkCopyGroupIndex < 0)
        zePrint("NOTE: link blitter/copy engines are not available\n");
      else
        zePrint("NOTE: link blitter/copy engines are available\n");
    }

    ZeMainCopyQueueGroupIndex = MainCopyGroupIndex;
    if (MainCopyGroupIndex >= 0) {
      ZeMainCopyQueueGroupProperties = QueueGroupProperties[MainCopyGroupIndex];
    }

    ZeLinkCopyQueueGroupIndex = LinkCopyGroupIndex;
    if (LinkCopyGroupIndex >= 0) {
      ZeLinkCopyQueueGroupProperties = QueueGroupProperties[LinkCopyGroupIndex];
    }
  }

  ZeComputeQueueGroupIndex = ComputeGroupIndex;
  ZeComputeQueueGroupProperties = QueueGroupProperties[ComputeGroupIndex];

  // Maintain various device properties cache.
  // Note that we just describe here how to compute the data.
  // The real initialization is upon first access.
  //
  auto ZeDevice = this->ZeDevice;
  ZeDeviceProperties.Compute = [ZeDevice](ze_device_properties_t &Properties) {
    ZE_CALL_NOCHECK(zeDeviceGetProperties, (ZeDevice, &Properties));
  };

  ZeDeviceComputeProperties.Compute =
      [ZeDevice](ze_device_compute_properties_t &Properties) {
        ZE_CALL_NOCHECK(zeDeviceGetComputeProperties, (ZeDevice, &Properties));
      };

  ZeDeviceImageProperties.Compute =
      [ZeDevice](ze_device_image_properties_t &Properties) {
        ZE_CALL_NOCHECK(zeDeviceGetImageProperties, (ZeDevice, &Properties));
      };

  ZeDeviceModuleProperties.Compute =
      [ZeDevice](ze_device_module_properties_t &Properties) {
        ZE_CALL_NOCHECK(zeDeviceGetModuleProperties, (ZeDevice, &Properties));
      };

  ZeDeviceMemoryProperties.Compute =
      [ZeDevice](
          std::vector<ZeStruct<ze_device_memory_properties_t>> &Properties) {
        uint32_t Count = 0;
        ZE_CALL_NOCHECK(zeDeviceGetMemoryProperties,
                        (ZeDevice, &Count, nullptr));

        Properties.resize(Count);
        ZE_CALL_NOCHECK(zeDeviceGetMemoryProperties,
                        (ZeDevice, &Count, Properties.data()));
      };

  ZeDeviceMemoryAccessProperties.Compute =
      [ZeDevice](ze_device_memory_access_properties_t &Properties) {
        ZE_CALL_NOCHECK(zeDeviceGetMemoryAccessProperties,
                        (ZeDevice, &Properties));
      };

  ZeDeviceCacheProperties.Compute =
      [ZeDevice](ze_device_cache_properties_t &Properties) {
        // TODO: Since v1.0 there can be multiple cache properties.
        // For now remember the first one, if any.
        uint32_t Count = 0;
        ZE_CALL_NOCHECK(zeDeviceGetCacheProperties,
                        (ZeDevice, &Count, nullptr));
        if (Count > 0)
          Count = 1;
        ZE_CALL_NOCHECK(zeDeviceGetCacheProperties,
                        (ZeDevice, &Count, &Properties));
      };
  return PI_SUCCESS;
}

pi_result _pi_context::initialize() {
  // Create the immediate command list to be used for initializations
  // Created as synchronous so level-zero performs implicit synchronization and
  // there is no need to query for completion in the plugin
  //
  // TODO: get rid of using Devices[0] for the context with multiple
  // root-devices. We should somehow make the data initialized on all devices.
  pi_device Device = SingleRootDevice ? SingleRootDevice : Devices[0];
  ZeStruct<ze_command_queue_desc_t> ZeCommandQueueDesc;
  ZeCommandQueueDesc.ordinal = Device->ZeComputeQueueGroupIndex;
  ZeCommandQueueDesc.index = Device->ZeComputeEngineIndex;
  ZeCommandQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
  ZE_CALL(
      zeCommandListCreateImmediate,
      (ZeContext, Device->ZeDevice, &ZeCommandQueueDesc, &ZeCommandListInit));
  return PI_SUCCESS;
}

pi_result _pi_context::finalize() {
  // This function is called when pi_context is deallocated, piContextRelease.
  // There could be some memory that may have not been deallocated.
  // For example, event pool caches would be still alive.
  {
    std::lock_guard<std::mutex> Lock(ZeEventPoolCacheMutex);
    for (auto &ZePool : ZeDeviceScopeEventPoolCache)
      ZE_CALL(zeEventPoolDestroy, (ZePool));
    for (auto &ZePool : ZeHostVisibleEventPoolCache)
      ZE_CALL(zeEventPoolDestroy, (ZePool));

    ZeDeviceScopeEventPoolCache.clear();
    ZeHostVisibleEventPoolCache.clear();
  }

  // Destroy the command list used for initializations
  ZE_CALL(zeCommandListDestroy, (ZeCommandListInit));

  // Adjust the number of command lists created on this platform.
  auto Platform = Devices[0]->Platform;

  std::lock_guard<std::mutex> Lock(ZeCommandListCacheMutex);
  for (auto &List : ZeComputeCommandListCache) {
    for (ze_command_list_handle_t &ZeCommandList : List.second) {
      if (ZeCommandList)
        ZE_CALL(zeCommandListDestroy, (ZeCommandList));
    }
    Platform->ZeGlobalCommandListCount -= List.second.size();
  }
  for (auto &List : ZeCopyCommandListCache) {
    for (ze_command_list_handle_t &ZeCommandList : List.second) {
      if (ZeCommandList)
        ZE_CALL(zeCommandListDestroy, (ZeCommandList));
    }
    Platform->ZeGlobalCommandListCount -= List.second.size();
  }
  return PI_SUCCESS;
}

bool _pi_queue::isInOrderQueue() const {
  // If out-of-order queue property is not set, then this is a in-order queue.
  return ((this->PiQueueProperties & PI_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ==
          0);
}

pi_result _pi_queue::resetCommandList(pi_command_list_ptr_t CommandList,
                                      bool MakeAvailable) {
  bool UseCopyEngine = CommandList->second.isCopy();
  auto &ZeCommandListCache =
      UseCopyEngine
          ? this->Context->ZeCopyCommandListCache[this->Device->ZeDevice]
          : this->Context->ZeComputeCommandListCache[this->Device->ZeDevice];

  // Fence had been signalled meaning the associated command-list completed.
  // Reset the fence and put the command list into a cache for reuse in PI
  // calls.
  ZE_CALL(zeFenceReset, (CommandList->second.ZeFence));
  ZE_CALL(zeCommandListReset, (CommandList->first));
  CommandList->second.InUse = false;

  // Finally release/cleanup all the events in this command list.
  // Note, we don't need to synchronize the events since the fence
  // synchronized above already does that.
  auto &EventList = CommandList->second.EventList;
  for (auto &Event : EventList) {
    // All events in this loop are in the same command list which has been just
    // reset above. We don't want cleanup() to reset same command list again for
    // all events in the loop so set it to nullptr.
    Event->ZeCommandList = nullptr;
    if (!Event->CleanedUp) {
      Event->cleanup(this);
    }
    PI_CALL(EventRelease(Event, this));
  }
  EventList.clear();

  if (MakeAvailable) {
    std::lock_guard<std::mutex> lock(this->Context->ZeCommandListCacheMutex);
    ZeCommandListCache.push_back(CommandList->first);
  }

  return PI_SUCCESS;
}

// Maximum Number of Command Lists that can be created.
// This Value is initialized to 20000, but can be changed by the user
// thru the environment variable SYCL_PI_LEVEL_ZERO_MAX_COMMAND_LIST_CACHE
// ie SYCL_PI_LEVEL_ZERO_MAX_COMMAND_LIST_CACHE =10000.
static const int ZeMaxCommandListCacheSize = [] {
  const char *CommandListCacheSize =
      std::getenv("SYCL_PI_LEVEL_ZERO_MAX_COMMAND_LIST_CACHE");
  pi_uint32 CommandListCacheSizeValue;
  try {
    CommandListCacheSizeValue =
        CommandListCacheSize ? std::stoi(CommandListCacheSize) : 20000;
  } catch (std::exception const &) {
    zePrint(
        "SYCL_PI_LEVEL_ZERO_MAX_COMMAND_LIST_CACHE: invalid value provided, "
        "default set.\n");
    CommandListCacheSizeValue = 20000;
  }
  return CommandListCacheSizeValue;
}();

// Configuration of the command-list batching.
typedef struct CommandListBatchConfig {
  // Default value of 0. This specifies to use dynamic batch size adjustment.
  // Other values will try to collect specified amount of commands.
  pi_uint32 Size{0};

  // If doing dynamic batching, specifies start batch size.
  pi_uint32 DynamicSizeStart{4};

  // The maximum size for dynamic batch.
  pi_uint32 DynamicSizeMax{64};

  // The step size for dynamic batch increases.
  pi_uint32 DynamicSizeStep{1};

  // Thresholds for when increase batch size (number of closed early is small
  // and number of closed full is high).
  pi_uint32 NumTimesClosedEarlyThreshold{2};
  pi_uint32 NumTimesClosedFullThreshold{10};

  // Tells the starting size of a batch.
  pi_uint32 startSize() const { return Size > 0 ? Size : DynamicSizeStart; }
  // Tells is we are doing dynamic batch size adjustment.
  bool dynamic() const { return Size == 0; }
} zeCommandListBatchConfig;

// Helper function to initialize static variables that holds batch config info
// for compute and copy command batching.
static const zeCommandListBatchConfig ZeCommandListBatchConfig(bool IsCopy) {
  zeCommandListBatchConfig Config{}; // default initialize

  // Default value of 0. This specifies to use dynamic batch size adjustment.
  const auto BatchSizeStr =
      (IsCopy) ? std::getenv("SYCL_PI_LEVEL_ZERO_COPY_BATCH_SIZE")
               : std::getenv("SYCL_PI_LEVEL_ZERO_BATCH_SIZE");
  if (BatchSizeStr) {
    pi_int32 BatchSizeStrVal = std::atoi(BatchSizeStr);
    // Level Zero may only support a limted number of commands per command
    // list.  The actual upper limit is not specified by the Level Zero
    // Specification.  For now we allow an arbitrary upper limit.
    if (BatchSizeStrVal > 0) {
      Config.Size = BatchSizeStrVal;
    } else if (BatchSizeStrVal == 0) {
      Config.Size = 0;
      // We are requested to do dynamic batching. Collect specifics, if any.
      // The extended format supported is ":" separated values.
      //
      // NOTE: these extra settings are experimental and are intended to
      // be used only for finding a better default heuristic.
      //
      std::string BatchConfig(BatchSizeStr);
      size_t Ord = 0;
      size_t Pos = 0;
      while (true) {
        if (++Ord > 5)
          break;

        Pos = BatchConfig.find(":", Pos);
        if (Pos == std::string::npos)
          break;
        ++Pos; // past the ":"

        pi_uint32 Val;
        try {
          Val = std::stoi(BatchConfig.substr(Pos));
        } catch (...) {
          if (IsCopy)
            zePrint(
                "SYCL_PI_LEVEL_ZERO_COPY_BATCH_SIZE: failed to parse value\n");
          else
            zePrint("SYCL_PI_LEVEL_ZERO_BATCH_SIZE: failed to parse value\n");
          break;
        }
        switch (Ord) {
        case 1:
          Config.DynamicSizeStart = Val;
          break;
        case 2:
          Config.DynamicSizeMax = Val;
          break;
        case 3:
          Config.DynamicSizeStep = Val;
          break;
        case 4:
          Config.NumTimesClosedEarlyThreshold = Val;
          break;
        case 5:
          Config.NumTimesClosedFullThreshold = Val;
          break;
        default:
          die("Unexpected batch config");
        }
        if (IsCopy)
          zePrint("SYCL_PI_LEVEL_ZERO_COPY_BATCH_SIZE: dynamic batch param "
                  "#%d: %d\n",
                  (int)Ord, (int)Val);
        else
          zePrint(
              "SYCL_PI_LEVEL_ZERO_BATCH_SIZE: dynamic batch param #%d: %d\n",
              (int)Ord, (int)Val);
      };

    } else {
      // Negative batch sizes are silently ignored.
      if (IsCopy)
        zePrint("SYCL_PI_LEVEL_ZERO_COPY_BATCH_SIZE: ignored negative value\n");
      else
        zePrint("SYCL_PI_LEVEL_ZERO_BATCH_SIZE: ignored negative value\n");
    }
  }
  return Config;
}

// Static variable that holds batch config info for compute command batching.
static const zeCommandListBatchConfig ZeCommandListBatchComputeConfig = [] {
  using IsCopy = bool;
  return ZeCommandListBatchConfig(IsCopy{false});
}();

// Static variable that holds batch config info for copy command batching.
static const zeCommandListBatchConfig ZeCommandListBatchCopyConfig = [] {
  using IsCopy = bool;
  return ZeCommandListBatchConfig(IsCopy{true});
}();

_pi_queue::_pi_queue(ze_command_queue_handle_t Queue,
                     std::vector<ze_command_queue_handle_t> &CopyQueues,
                     pi_context Context, pi_device Device,
                     bool OwnZeCommandQueue,
                     pi_queue_properties PiQueueProperties)
    : ZeComputeCommandQueue{Queue}, ZeCopyCommandQueues{CopyQueues},
      Context{Context}, Device{Device}, OwnZeCommandQueue{OwnZeCommandQueue},
      PiQueueProperties(PiQueueProperties) {
  ComputeCommandBatch.OpenCommandList = CommandListMap.end();
  CopyCommandBatch.OpenCommandList = CommandListMap.end();
  ComputeCommandBatch.QueueBatchSize =
      ZeCommandListBatchComputeConfig.startSize();
  CopyCommandBatch.QueueBatchSize = ZeCommandListBatchCopyConfig.startSize();
}

// Retrieve an available command list to be used in a PI call
// Caller must hold a lock on the Queue passed in.
pi_result
_pi_context::getAvailableCommandList(pi_queue Queue,
                                     pi_command_list_ptr_t &CommandList,
                                     bool UseCopyEngine, bool AllowBatching) {
  auto &CommandBatch =
      UseCopyEngine ? Queue->CopyCommandBatch : Queue->ComputeCommandBatch;
  // Handle batching of commands
  // First see if there is an command-list open for batching commands
  // for this queue.
  if (Queue->hasOpenCommandList(UseCopyEngine)) {
    if (AllowBatching) {
      CommandList = CommandBatch.OpenCommandList;
      return PI_SUCCESS;
    }
    // If this command isn't allowed to be batched, then we need to
    // go ahead and execute what is already in the batched list,
    // and then go on to process this. On exit from executeOpenCommandList
    // OpenCommandList will be invalidated.
    if (auto Res = Queue->executeOpenCommandList(UseCopyEngine))
      return Res;
  }

  // Create/Reuse the command list, because in Level Zero commands are added to
  // the command lists, and later are then added to the command queue.
  // Each command list is paired with an associated fence to track when the
  // command list is available for reuse.
  _pi_result pi_result = PI_OUT_OF_RESOURCES;
  ZeStruct<ze_fence_desc_t> ZeFenceDesc;

  auto &ZeCommandListCache =
      UseCopyEngine
          ? Queue->Context->ZeCopyCommandListCache[Queue->Device->ZeDevice]
          : Queue->Context->ZeComputeCommandListCache[Queue->Device->ZeDevice];

  // Initally, we need to check if a command list has already been created
  // on this device that is available for use. If so, then reuse that
  // Level-Zero Command List and Fence for this PI call.
  {
    // Make sure to acquire the lock before checking the size, or there
    // will be a race condition.
    std::lock_guard<std::mutex> lock(Queue->Context->ZeCommandListCacheMutex);

    if (ZeCommandListCache.size() > 0) {
      auto &ZeCommandList = ZeCommandListCache.front();
      auto it = Queue->CommandListMap.find(ZeCommandList);
      if (it != Queue->CommandListMap.end()) {
        CommandList = it;
        CommandList->second.InUse = true;
      } else {
        // If there is a command list available on this context, but it
        // wasn't yet used in this queue then create a new entry in this
        // queue's map to hold the fence and other associated command
        // list information.

        // If needed, get the next available copy command queue
        int CopyQueueIndex = -1;
        ze_command_queue_handle_t ZeCopyCommandQueue = nullptr;
        if (UseCopyEngine)
          ZeCopyCommandQueue = Queue->getZeCopyCommandQueue(&CopyQueueIndex);
        auto &ZeCommandQueue =
            UseCopyEngine ? ZeCopyCommandQueue : Queue->ZeComputeCommandQueue;

        ze_fence_handle_t ZeFence;
        ZE_CALL(zeFenceCreate, (ZeCommandQueue, &ZeFenceDesc, &ZeFence));
        CommandList =
            Queue->CommandListMap
                .emplace(ZeCommandList,
                         pi_command_list_info_t{ZeFence, true, CopyQueueIndex})
                .first;
      }
      ZeCommandListCache.pop_front();
      return PI_SUCCESS;
    }
  }

  // If there are no available command lists in the cache, then we check for
  // command lists that have already signalled, but have not been added to the
  // available list yet. Each command list has a fence associated which tracks
  // if a command list has completed dispatch of its commands and is ready for
  // reuse. If a command list is found to have been signalled, then the
  // command list & fence are reset and we return.
  for (auto it = Queue->CommandListMap.begin();
       it != Queue->CommandListMap.end(); ++it) {
    // Make sure this is the command list type needed.
    if (UseCopyEngine != it->second.isCopy())
      continue;

    ze_result_t ZeResult =
        ZE_CALL_NOCHECK(zeFenceQueryStatus, (it->second.ZeFence));
    if (ZeResult == ZE_RESULT_SUCCESS) {
      Queue->resetCommandList(it, false);
      CommandList = it;
      CommandList->second.InUse = true;
      return PI_SUCCESS;
    }
  }

  // If there are no available command lists nor signalled command lists, then
  // we must create another command list if we have not exceed the maximum
  // command lists we can create.
  // Once created, this command list & fence are added to the command list fence
  // map.
  if (Queue->Device->Platform->ZeGlobalCommandListCount <
      ZeMaxCommandListCacheSize) {
    ze_command_list_handle_t ZeCommandList;
    ze_fence_handle_t ZeFence;

    int CopyQueueIndex = -1;
    ze_command_queue_handle_t ZeCopyCommandQueue = nullptr;
    int ZeCopyCommandQueueGroupIndex = -1;
    if (UseCopyEngine) {
      ZeCopyCommandQueue = Queue->getZeCopyCommandQueue(
          &CopyQueueIndex, &ZeCopyCommandQueueGroupIndex);
    }
    ZeStruct<ze_command_list_desc_t> ZeCommandListDesc;
    ZeCommandListDesc.commandQueueGroupOrdinal =
        UseCopyEngine ? ZeCopyCommandQueueGroupIndex
                      : Queue->Device->ZeComputeQueueGroupIndex;

    ZE_CALL(zeCommandListCreate,
            (Queue->Context->ZeContext, Queue->Device->ZeDevice,
             &ZeCommandListDesc, &ZeCommandList));
    // Increments the total number of command lists created on this platform.
    Queue->Device->Platform->ZeGlobalCommandListCount++;

    auto &ZeCommandQueue =
        UseCopyEngine ? ZeCopyCommandQueue : Queue->ZeComputeCommandQueue;
    ZE_CALL(zeFenceCreate, (ZeCommandQueue, &ZeFenceDesc, &ZeFence));
    std::tie(CommandList, std::ignore) = Queue->CommandListMap.insert(
        std::pair<ze_command_list_handle_t, pi_command_list_info_t>(
            ZeCommandList, {ZeFence, true, CopyQueueIndex}));
    pi_result = PI_SUCCESS;
  }

  return pi_result;
}

void _pi_queue::adjustBatchSizeForFullBatch(bool IsCopy) {
  auto &CommandBatch = IsCopy ? CopyCommandBatch : ComputeCommandBatch;
  auto &ZeCommandListBatchConfig =
      IsCopy ? ZeCommandListBatchCopyConfig : ZeCommandListBatchComputeConfig;
  pi_uint32 &QueueBatchSize = CommandBatch.QueueBatchSize;
  // QueueBatchSize of 0 means never allow batching.
  if (QueueBatchSize == 0 || !ZeCommandListBatchConfig.dynamic())
    return;
  CommandBatch.NumTimesClosedFull += 1;

  // If the number of times the list has been closed early is low, and
  // the number of times it has been closed full is high, then raise
  // the batching size slowly. Don't raise it if it is already pretty
  // high.
  if (CommandBatch.NumTimesClosedEarly <=
          ZeCommandListBatchConfig.NumTimesClosedEarlyThreshold &&
      CommandBatch.NumTimesClosedFull >
          ZeCommandListBatchConfig.NumTimesClosedFullThreshold) {
    if (QueueBatchSize < ZeCommandListBatchConfig.DynamicSizeMax) {
      QueueBatchSize += ZeCommandListBatchConfig.DynamicSizeStep;
      zePrint("Raising QueueBatchSize to %d\n", QueueBatchSize);
    }
    CommandBatch.NumTimesClosedEarly = 0;
    CommandBatch.NumTimesClosedFull = 0;
  }
}

void _pi_queue::adjustBatchSizeForPartialBatch(bool IsCopy) {
  auto &CommandBatch = IsCopy ? CopyCommandBatch : ComputeCommandBatch;
  auto &ZeCommandListBatchConfig =
      IsCopy ? ZeCommandListBatchCopyConfig : ZeCommandListBatchComputeConfig;
  pi_uint32 &QueueBatchSize = CommandBatch.QueueBatchSize;
  // QueueBatchSize of 0 means never allow batching.
  if (QueueBatchSize == 0 || !ZeCommandListBatchConfig.dynamic())
    return;
  CommandBatch.NumTimesClosedEarly += 1;

  // If we are closing early more than about 3x the number of times
  // it is closing full, lower the batch size to the value of the
  // current open command list. This is trying to quickly get to a
  // batch size that will be able to be closed full at least once
  // in a while.
  if (CommandBatch.NumTimesClosedEarly >
      (CommandBatch.NumTimesClosedFull + 1) * 3) {
    QueueBatchSize = CommandBatch.OpenCommandList->second.size() - 1;
    if (QueueBatchSize < 1)
      QueueBatchSize = 1;
    zePrint("Lowering QueueBatchSize to %d\n", QueueBatchSize);
    CommandBatch.NumTimesClosedEarly = 0;
    CommandBatch.NumTimesClosedFull = 0;
  }
}

pi_result _pi_queue::executeCommandList(pi_command_list_ptr_t CommandList,
                                        bool IsBlocking,
                                        bool OKToBatchCommand) {
  int Index = CommandList->second.CopyQueueIndex;
  bool UseCopyEngine = (Index != -1);
  if (UseCopyEngine)
    zePrint("Command list to be executed on copy engine %d\n", Index);

  // If the current LastCommandEvent is the nullptr, then it means
  // either that no command has ever been issued to the queue
  // or it means that the LastCommandEvent has been signalled and
  // therefore that this Queue is idle.
  //
  // NOTE: this behavior adds some flakyness to the batching
  // since last command's event may or may not be completed by the
  // time we get here depending on timings and system/gpu load.
  // So, disable it for modes where we print PI traces. Printing
  // traces incurs much different timings than real execution
  // ansyway, and many regression tests use it.
  //
  bool CurrentlyEmpty = !PrintPiTrace && this->LastCommandEvent == nullptr;

  // The list can be empty if command-list only contains signals of proxy
  // events.
  if (!CommandList->second.EventList.empty())
    this->LastCommandEvent = CommandList->second.EventList.back();

  // Batch if allowed to, but don't batch if we know there are no kernels
  // from this queue that are currently executing.  This is intended to get
  // kernels started as soon as possible when there are no kernels from this
  // queue awaiting execution, while allowing batching to occur when there
  // are kernels already executing. Also, if we are using fixed size batching,
  // as indicated by !ZeCommandListBatch.dynamic(), then just ignore
  // CurrentlyEmpty as we want to strictly follow the batching the user
  // specified.
  auto &CommandBatch = UseCopyEngine ? CopyCommandBatch : ComputeCommandBatch;
  auto &ZeCommandListBatchConfig = UseCopyEngine
                                       ? ZeCommandListBatchCopyConfig
                                       : ZeCommandListBatchComputeConfig;
  if (OKToBatchCommand && this->isBatchingAllowed(UseCopyEngine) &&
      (!ZeCommandListBatchConfig.dynamic() || !CurrentlyEmpty)) {

    if (hasOpenCommandList(UseCopyEngine) &&
        CommandBatch.OpenCommandList != CommandList)
      die("executeCommandList: OpenCommandList should be equal to"
          "null or CommandList");

    if (CommandList->second.size() < CommandBatch.QueueBatchSize) {
      CommandBatch.OpenCommandList = CommandList;
      return PI_SUCCESS;
    }

    adjustBatchSizeForFullBatch(UseCopyEngine);
    CommandBatch.OpenCommandList = CommandListMap.end();
  }

  // If available, get the copy command queue assosciated with
  // ZeCommandList
  ze_command_queue_handle_t ZeCopyCommandQueue = nullptr;
  if (Index != -1) {
    if (auto Res = getOrCreateCopyCommandQueue(Index, ZeCopyCommandQueue))
      return Res;
  }
  auto &ZeCommandQueue =
      UseCopyEngine ? ZeCopyCommandQueue : ZeComputeCommandQueue;
  // Scope of the lock must be till the end of the function, otherwise new mem
  // allocs can be created between the moment when we made a snapshot and the
  // moment when command list is closed and executed. But mutex is locked only
  // if indirect access tracking enabled, because std::defer_lock is used.
  // unique_lock destructor at the end of the function will unlock the mutex if
  // it was locked (which happens only if IndirectAccessTrackingEnabled is
  // true).
  std::unique_lock<std::mutex> ContextsLock(Device->Platform->ContextsMutex,
                                            std::defer_lock);
  if (IndirectAccessTrackingEnabled) {
    // We are going to submit kernels for execution. If indirect access flag is
    // set for a kernel then we need to make a snapshot of existing memory
    // allocations in all contexts in the platform. We need to lock the mutex
    // guarding the list of contexts in the platform to prevent creation of new
    // memory alocations in any context before we submit the kernel for
    // execution.
    ContextsLock.lock();
    for (auto &Kernel : KernelsToBeSubmitted) {
      if (!Kernel->hasIndirectAccess())
        continue;

      auto &Contexts = Device->Platform->Contexts;
      for (auto &Ctx : Contexts) {
        for (auto &Elem : Ctx->MemAllocs) {
          const auto &Pair = Kernel->MemAllocs.insert(&Elem);
          // Kernel is referencing this memory allocation from now.
          // If this memory allocation was already captured for this kernel, it
          // means that kernel is submitted several times. Increase reference
          // count only once because we release all allocations only when
          // SubmissionsCount turns to 0. We don't want to know how many times
          // allocation was retained by each submission.
          if (Pair.second)
            Elem.second.RefCount++;
        }
      }
      Kernel->SubmissionsCount++;
    }
    KernelsToBeSubmitted.clear();
  }

  // In this mode all inner-batch events have device visibility only,
  // and we want the last command in the batch to signal a host-visible
  // event that anybody waiting for any event in the batch will
  // really be using.
  //
  if (EventsScope == LastCommandInBatchHostVisible) {
    // Create a "proxy" host-visible event.
    //
    pi_event HostVisibleEvent;
    PI_CALL(EventCreate(Context, true, &HostVisibleEvent));

    // Update each command's event in the command-list to "see" this
    // proxy event as a host-visible counterpart.
    for (auto &Event : CommandList->second.EventList) {
      Event->HostVisibleEvent = HostVisibleEvent;
      PI_CALL(piEventRetain(HostVisibleEvent));
    }

    // Decrement the reference count by 1 so all the remaining references
    // are from the other commands in this batch. This host-visible event
    // will be destroyed after all events in the batch are gone.
    PI_CALL(piEventRelease(HostVisibleEvent));
    // Indicate no cleanup is needed for this PI event as it is special.
    HostVisibleEvent->CleanedUp = true;

    // Finally set to signal the host-visible event at the end of the
    // command-list.
    // TODO: see if we need a barrier here (or explicit wait for all events in
    // the batch).
    ZE_CALL(zeCommandListAppendSignalEvent,
            (CommandList->first, HostVisibleEvent->ZeEvent));
  }

  // Close the command list and have it ready for dispatch.
  ZE_CALL(zeCommandListClose, (CommandList->first));
  // Offload command list to the GPU for asynchronous execution
  auto ZeCommandList = CommandList->first;
  zePrint("Calling zeCommandQueueExecuteCommandLists with Index = %d\n", Index);
  auto ZeResult = ZE_CALL_NOCHECK(
      zeCommandQueueExecuteCommandLists,
      (ZeCommandQueue, 1, &ZeCommandList, CommandList->second.ZeFence));
  if (ZeResult != ZE_RESULT_SUCCESS) {
    this->Healthy = false;
    if (ZeResult == ZE_RESULT_ERROR_UNKNOWN) {
      // Turn into a more informative end-user error.
      return PI_COMMAND_EXECUTION_FAILURE;
    }
    return mapError(ZeResult);
  }

  // Check global control to make every command blocking for debugging.
  if (IsBlocking || (ZeSerialize & ZeSerializeBlock) != 0) {
    // Wait until command lists attached to the command queue are executed.
    ZE_CALL(zeHostSynchronize, (ZeCommandQueue));
  }
  return PI_SUCCESS;
}

bool _pi_queue::isBatchingAllowed(bool IsCopy) const {
  auto &CommandBatch = IsCopy ? CopyCommandBatch : ComputeCommandBatch;
  return (CommandBatch.QueueBatchSize > 0 &&
          ((ZeSerialize & ZeSerializeBlock) == 0));
}

pi_result _pi_queue::getOrCreateCopyCommandQueue(
    int Index, ze_command_queue_handle_t &ZeCopyCommandQueue) {
  ZeCopyCommandQueue = nullptr;

  // Make sure 'Index' is within limits
  PI_ASSERT((Index >= 0) && (Index < (int)(ZeCopyCommandQueues.size())),
            PI_INVALID_VALUE);

  // Return the Ze copy command queue, if already available
  if (ZeCopyCommandQueues[Index]) {
    ZeCopyCommandQueue = ZeCopyCommandQueues[Index];
    return PI_SUCCESS;
  }

  // Ze copy command queue is not available at 'Index'. So we create it below.
  ZeStruct<ze_command_queue_desc_t> ZeCommandQueueDesc;
  // There are two copy queues: main copy queues and link copy queues.
  // Index is the 'index' into the overall list of copy queues
  // (one queue per copy engine).
  // ZeCommandQueueDesc.ordinal specifies the copy group (main or link)
  // ZeCommandQueueDesc.index specifies the copy queue/engine within a group
  // Following are possible scenarios:
  // 1. (Index == 0) and main copy engine is available:
  //    ZeCommandQueueDesc.ordinal = Device->ZeMainCopyQueueGroupIndex
  //    ZeCommandQueueDesc.index = 0
  // 2. (Index == 0) and main copy engine is not available:
  //    ZeCommandQueueDesc.ordinal = Device->ZeLinkCopyQueueGroupIndex
  //    ZeCommandQueueDesc.index = 0
  // 3. (Index != 0) and main copy engine is available:
  //    ZeCommandQueueDesc.ordinal = Device->ZeLinkCopyQueueGroupIndex
  //    ZeCommandQueueDesc.index = Index - 1
  // 4. (Index != 0) and main copy engine is not available:
  //    ZeCommandQueueDesc.ordinal = Device->ZeLinkCopyQueueGroupIndex
  //    ZeCommandQueueDesc.index = Index
  ZeCommandQueueDesc.ordinal = (Index == 0 && Device->hasMainCopyEngine())
                                   ? Device->ZeMainCopyQueueGroupIndex
                                   : Device->ZeLinkCopyQueueGroupIndex;
  ZeCommandQueueDesc.index =
      (Index != 0 && Device->hasMainCopyEngine()) ? Index - 1 : Index;
  zePrint("NOTE: Copy Engine ZeCommandQueueDesc.ordinal = %d, "
          "ZeCommandQueueDesc.index = %d\n",
          ZeCommandQueueDesc.ordinal, ZeCommandQueueDesc.index);
  ZE_CALL(zeCommandQueueCreate,
          (Context->ZeContext, Device->ZeDevice,
           &ZeCommandQueueDesc, // TODO: translate properties
           &ZeCopyCommandQueue));
  ZeCopyCommandQueues[Index] = ZeCopyCommandQueue;
  return PI_SUCCESS;
}

// This function will return one of possibly multiple available copy queues.
// Currently, a round robin strategy is used.
// This function also sends back the value of CopyQueueIndex and
// CopyQueueGroupIndex (optional)
ze_command_queue_handle_t
_pi_queue::getZeCopyCommandQueue(int *CopyQueueIndex,
                                 int *CopyQueueGroupIndex) {
  assert(CopyQueueIndex);
  int n = ZeCopyCommandQueues.size();
  int LowerCopyQueueIndex = getRangeOfAllowedCopyEngines.first;
  int UpperCopyQueueIndex = getRangeOfAllowedCopyEngines.second;

  // Return nullptr when no copy command queues are allowed to be used or if
  // no copy command queues are available.
  if ((LowerCopyQueueIndex == -1) || (UpperCopyQueueIndex == -1) || (n == 0)) {
    if (CopyQueueGroupIndex)
      *CopyQueueGroupIndex = -1;
    *CopyQueueIndex = -1;
    return nullptr;
  }

  LowerCopyQueueIndex = std::max(0, LowerCopyQueueIndex);
  UpperCopyQueueIndex = std::min(UpperCopyQueueIndex, n - 1);

  // If there is only one copy queue, it is the main copy queue (if available),
  // or the first link copy queue in ZeCopyCommandQueues.
  if (n == 1) {
    *CopyQueueIndex = 0;
    if (CopyQueueGroupIndex) {
      if (Device->hasMainCopyEngine())
        *CopyQueueGroupIndex = Device->ZeMainCopyQueueGroupIndex;
      else
        *CopyQueueGroupIndex = Device->ZeLinkCopyQueueGroupIndex;
      zePrint("Note: CopyQueueGroupIndex = %d\n", *CopyQueueGroupIndex);
    }
    zePrint("Note: CopyQueueIndex = %d\n", *CopyQueueIndex);
    ze_command_queue_handle_t ZeCopyCommandQueue = nullptr;
    if (getOrCreateCopyCommandQueue(0, ZeCopyCommandQueue))
      return nullptr;
    return ZeCopyCommandQueue;
  }

  // Round robin logic is used here to access copy command queues.
  // Initial value of LastUsedCopyCommandQueueIndex is -1.
  // So, the round robin logic will start its access at 'LowerCopyQueueIndex'
  // queue.
  // TODO: In this implementation, all the copy engines (main and link)
  // have equal priority. It is expected that main copy engine will be
  // advantageous for H2D and D2H copies, whereas the link copy engines will
  // be advantageous for D2D. We will perform experiments and then assign
  // priority to different copy engines for different types of copy operations.
  if ((LastUsedCopyCommandQueueIndex == -1) ||
      (LastUsedCopyCommandQueueIndex == UpperCopyQueueIndex))
    *CopyQueueIndex = LowerCopyQueueIndex;
  else
    *CopyQueueIndex = LastUsedCopyCommandQueueIndex + 1;
  LastUsedCopyCommandQueueIndex = *CopyQueueIndex;
  if (CopyQueueGroupIndex)
    // First queue in the vector of copy queues is the main copy queue,
    // if available. Otherwise it's a link copy queue.
    *CopyQueueGroupIndex =
        ((*CopyQueueIndex == 0) && Device->hasMainCopyEngine())
            ? Device->ZeMainCopyQueueGroupIndex
            : Device->ZeLinkCopyQueueGroupIndex;
  ze_command_queue_handle_t ZeCopyCommandQueue = nullptr;
  if (getOrCreateCopyCommandQueue(*CopyQueueIndex, ZeCopyCommandQueue))
    return nullptr;
  return ZeCopyCommandQueue;
}

pi_result _pi_queue::executeOpenCommandListWithEvent(pi_event Event) {
  // TODO: see if we can reliably tell if the event is copy or compute.
  // Meanwhile check both open command-lists.
  using IsCopy = bool;
  if (hasOpenCommandList(IsCopy{false}) &&
      ComputeCommandBatch.OpenCommandList->first == Event->ZeCommandList) {
    if (auto Res = executeOpenCommandList(IsCopy{false}))
      return Res;
  }
  if (hasOpenCommandList(IsCopy{true}) &&
      CopyCommandBatch.OpenCommandList->first == Event->ZeCommandList) {
    if (auto Res = executeOpenCommandList(IsCopy{true}))
      return Res;
  }
  return PI_SUCCESS;
}

pi_result _pi_queue::executeOpenCommandList(bool IsCopy) {
  auto &CommandBatch = IsCopy ? CopyCommandBatch : ComputeCommandBatch;
  // If there are any commands still in the open command list for this
  // queue, then close and execute that command list now.
  if (hasOpenCommandList(IsCopy)) {
    adjustBatchSizeForPartialBatch(IsCopy);
    auto Res = executeCommandList(CommandBatch.OpenCommandList, false, false);
    CommandBatch.OpenCommandList = CommandListMap.end();
    return Res;
  }

  return PI_SUCCESS;
}

static const bool FilterEventWaitList = [] {
  const char *Ret = std::getenv("SYCL_PI_LEVEL_ZERO_FILTER_EVENT_WAIT_LIST");
  const bool RetVal = Ret ? std::stoi(Ret) : 1;
  return RetVal;
}();

pi_result _pi_ze_event_list_t::createAndRetainPiZeEventList(
    pi_uint32 EventListLength, const pi_event *EventList, pi_queue CurQueue) {
  this->Length = 0;
  this->ZeEventList = nullptr;
  this->PiEventList = nullptr;

  try {
    if (CurQueue->isInOrderQueue() && CurQueue->LastCommandEvent != nullptr) {
      this->ZeEventList = new ze_event_handle_t[EventListLength + 1];
      this->PiEventList = new pi_event[EventListLength + 1];
    } else if (EventListLength > 0) {
      this->ZeEventList = new ze_event_handle_t[EventListLength];
      this->PiEventList = new pi_event[EventListLength];
    }

    pi_uint32 TmpListLength = 0;

    if (EventListLength > 0) {
      for (pi_uint32 I = 0; I < EventListLength; I++) {
        PI_ASSERT(EventList[I] != nullptr, PI_INVALID_VALUE);
        auto ZeEvent = EventList[I]->ZeEvent;

        // Poll of the host-visible events.
        auto HostVisibleEvent = EventList[I]->HostVisibleEvent;
        if (FilterEventWaitList && HostVisibleEvent) {
          auto Res =
              ZE_CALL_NOCHECK(zeEventQueryStatus, (HostVisibleEvent->ZeEvent));
          if (Res == ZE_RESULT_SUCCESS) {
            // Event has already completed, don't put it into the list
            continue;
          }
        }

        auto Queue = EventList[I]->Queue;

        if (Queue && Queue != CurQueue) {
          // If the event that is going to be waited on is in a
          // different queue, then any open command list in
          // that queue must be closed and executed because
          // the event being waited on could be for a command
          // in the queue's batch.

          // Lock automatically releases when this goes out of scope.
          std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

          if (auto Res = Queue->executeAllOpenCommandLists())
            return Res;
        }

        this->ZeEventList[TmpListLength] = ZeEvent;
        this->PiEventList[TmpListLength] = EventList[I];
        TmpListLength += 1;
      }
    }

    // For in-order queues, every command should be executed only after the
    // previous command has finished. The event associated with the last
    // enqueued command is added into the waitlist to ensure in-order semantics.
    if (CurQueue->isInOrderQueue() && CurQueue->LastCommandEvent != nullptr) {
      this->ZeEventList[TmpListLength] = CurQueue->LastCommandEvent->ZeEvent;
      this->PiEventList[TmpListLength] = CurQueue->LastCommandEvent;
      TmpListLength += 1;
    }

    this->Length = TmpListLength;

  } catch (...) {
    return PI_OUT_OF_HOST_MEMORY;
  }

  for (pi_uint32 I = 0; I < this->Length; I++) {
    PI_CALL(piEventRetain(this->PiEventList[I]));
  }

  return PI_SUCCESS;
}

static void printZeEventList(const _pi_ze_event_list_t &PiZeEventList) {
  zePrint("  NumEventsInWaitList %d:", PiZeEventList.Length);

  for (pi_uint32 I = 0; I < PiZeEventList.Length; I++) {
    zePrint(" %#lx", pi_cast<std::uintptr_t>(PiZeEventList.ZeEventList[I]));
  }

  zePrint("\n");
}

pi_result _pi_ze_event_list_t::collectEventsForReleaseAndDestroyPiZeEventList(
    std::list<pi_event> &EventsToBeReleased) {
  // acquire a lock before reading the length and list fields.
  // Acquire the lock, copy the needed data locally, and reset
  // the fields, then release the lock.
  // Only then do we do the actual actions to release and destroy,
  // holding the lock for the minimum time necessary.
  pi_uint32 LocLength = 0;
  ze_event_handle_t *LocZeEventList = nullptr;
  pi_event *LocPiEventList = nullptr;

  {
    // acquire the lock and copy fields locally
    // Lock automatically releases when this goes out of scope.
    std::lock_guard<std::mutex> lock(this->PiZeEventListMutex);

    LocLength = Length;
    LocZeEventList = ZeEventList;
    LocPiEventList = PiEventList;

    Length = 0;
    ZeEventList = nullptr;
    PiEventList = nullptr;

    // release lock by ending scope.
  }

  for (pi_uint32 I = 0; I < LocLength; I++) {
    // Add the event to be released to the list
    EventsToBeReleased.push_back(LocPiEventList[I]);
  }

  if (LocZeEventList != nullptr) {
    delete[] LocZeEventList;
  }
  if (LocPiEventList != nullptr) {
    delete[] LocPiEventList;
  }

  return PI_SUCCESS;
}

extern "C" {

// Forward declarations
decltype(piEventCreate) piEventCreate;

static ze_result_t
checkUnresolvedSymbols(ze_module_handle_t ZeModule,
                       ze_module_build_log_handle_t *ZeBuildLog);

// This function will ensure compatibility with both Linux and Windows for
// setting environment variables.
static bool setEnvVar(const char *name, const char *value) {
#ifdef _WIN32
  int Res = _putenv_s(name, value);
#else
  int Res = setenv(name, value, 1);
#endif
  if (Res != 0) {
    zePrint(
        "Level Zero plugin was unable to set the environment variable: %s\n",
        name);
    return false;
  }
  return true;
}

static class ZeUSMImportExtension {
  // Pointers to functions that import/release host memory into USM
  ze_result_t (*zexDriverImportExternalPointer)(ze_driver_handle_t hDriver,
                                                void *, size_t);
  ze_result_t (*zexDriverReleaseImportedPointer)(ze_driver_handle_t, void *);

public:
  // Whether user has requested Import/Release, and platform supports it.
  bool Enabled;

  ZeUSMImportExtension() : Enabled{false} {}

  void setZeUSMImport(pi_platform Platform) {
    // Whether env var SYCL_USM_HOSTPTR_IMPORT has been set requesting
    // host ptr import during buffer creation.
    const char *USMHostPtrImportStr = std::getenv("SYCL_USM_HOSTPTR_IMPORT");
    if (!USMHostPtrImportStr || std::atoi(USMHostPtrImportStr) == 0)
      return;

    // Check if USM hostptr import feature is available.
    ze_driver_handle_t driverHandle = Platform->ZeDriver;
    if (ZE_CALL_NOCHECK(zeDriverGetExtensionFunctionAddress,
                        (driverHandle, "zexDriverImportExternalPointer",
                         reinterpret_cast<void **>(
                             &zexDriverImportExternalPointer))) == 0) {
      ZE_CALL_NOCHECK(
          zeDriverGetExtensionFunctionAddress,
          (driverHandle, "zexDriverReleaseImportedPointer",
           reinterpret_cast<void **>(&zexDriverReleaseImportedPointer)));
      // Hostptr import/release is turned on because it has been requested
      // by the env var, and this platform supports the APIs.
      Enabled = true;
      // Hostptr import is only possible if piMemBufferCreate receives a
      // hostptr as an argument. The SYCL runtime passes a host ptr
      // only when SYCL_HOST_UNIFIED_MEMORY is enabled. Therefore we turn it on.
      setEnvVar("SYCL_HOST_UNIFIED_MEMORY", "1");
    }
  }
  void doZeUSMImport(ze_driver_handle_t driverHandle, void *HostPtr,
                     size_t Size) {
    ZE_CALL_NOCHECK(zexDriverImportExternalPointer,
                    (driverHandle, HostPtr, Size));
  }
  void doZeUSMRelease(ze_driver_handle_t driverHandle, void *HostPtr) {
    ZE_CALL_NOCHECK(zexDriverReleaseImportedPointer, (driverHandle, HostPtr));
  }
} ZeUSMImport;

pi_result _pi_platform::initialize() {
  // Cache driver properties
  ZeStruct<ze_driver_properties_t> ZeDriverProperties;
  ZE_CALL(zeDriverGetProperties, (ZeDriver, &ZeDriverProperties));
  uint32_t DriverVersion = ZeDriverProperties.driverVersion;
  // Intel Level-Zero GPU driver stores version as:
  // | 31 - 24 | 23 - 16 | 15 - 0 |
  // |  Major  |  Minor  | Build  |
  auto VersionMajor = std::to_string((DriverVersion & 0xFF000000) >> 24);
  auto VersionMinor = std::to_string((DriverVersion & 0x00FF0000) >> 16);
  auto VersionBuild = std::to_string(DriverVersion & 0x0000FFFF);
  ZeDriverVersion = VersionMajor + "." + VersionMinor + "." + VersionBuild;

  ZE_CALL(zeDriverGetApiVersion, (ZeDriver, &ZeApiVersion));
  ZeDriverApiVersion = std::to_string(ZE_MAJOR_VERSION(ZeApiVersion)) + "." +
                       std::to_string(ZE_MINOR_VERSION(ZeApiVersion));

  // Cache driver extension properties
  uint32_t Count = 0;
  ZE_CALL(zeDriverGetExtensionProperties, (ZeDriver, &Count, nullptr));

  std::vector<ze_driver_extension_properties_t> zeExtensions(Count);

  ZE_CALL(zeDriverGetExtensionProperties,
          (ZeDriver, &Count, zeExtensions.data()));

  for (auto extension : zeExtensions) {
    // Check if global offset extension is available
    if (strncmp(extension.name, ZE_GLOBAL_OFFSET_EXP_NAME,
                strlen(ZE_GLOBAL_OFFSET_EXP_NAME) + 1) == 0) {
      if (extension.version == ZE_GLOBAL_OFFSET_EXP_VERSION_1_0) {
        PiDriverGlobalOffsetExtensionFound = true;
      }
    }
    // Check if extension is available for "static linking" (compiling multiple
    // SPIR-V modules together into one Level Zero module).
    if (strncmp(extension.name, ZE_MODULE_PROGRAM_EXP_NAME,
                strlen(ZE_MODULE_PROGRAM_EXP_NAME) + 1) == 0) {
      if (extension.version == ZE_MODULE_PROGRAM_EXP_VERSION_1_0) {
        PiDriverModuleProgramExtensionFound = true;
      }
    }
    zeDriverExtensionMap[extension.name] = extension.version;
  }

  // Check if import user ptr into USM feature has been requested.
  // If yes, then set up L0 API pointers if the platform supports it.
  ZeUSMImport.setZeUSMImport(this);

  return PI_SUCCESS;
}

pi_result piPlatformsGet(pi_uint32 NumEntries, pi_platform *Platforms,
                         pi_uint32 *NumPlatforms) {

  static const char *PiTrace = std::getenv("SYCL_PI_TRACE");
  static const int PiTraceValue = PiTrace ? std::stoi(PiTrace) : 0;
  if (PiTraceValue == -1 || PiTraceValue == 2) { // Means print all PI traces
    PrintPiTrace = true;
  }

  static std::once_flag ZeCallCountInitialized;
  try {
    std::call_once(ZeCallCountInitialized, []() {
      if (ZeDebug & ZE_DEBUG_CALL_COUNT) {
        ZeCallCount = new std::map<const char *, int>;
      }
    });
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  if (NumEntries == 0 && Platforms != nullptr) {
    return PI_INVALID_VALUE;
  }
  if (Platforms == nullptr && NumPlatforms == nullptr) {
    return PI_INVALID_VALUE;
  }

  // Setting these environment variables before running zeInit will enable the
  // validation layer in the Level Zero loader.
  if (ZeDebug & ZE_DEBUG_VALIDATION) {
    setEnvVar("ZE_ENABLE_VALIDATION_LAYER", "1");
    setEnvVar("ZE_ENABLE_PARAMETER_VALIDATION", "1");
  }

  // Enable SYSMAN support for obtaining the PCI address
  // and maximum memory bandwidth.
  if (getenv("SYCL_ENABLE_PCI") != nullptr) {
    setEnvVar("ZES_ENABLE_SYSMAN", "1");
  }

  // TODO: We can still safely recover if something goes wrong during the init.
  // Implement handling segfault using sigaction.

  // We must only initialize the driver once, even if piPlatformsGet() is called
  // multiple times.  Declaring the return value as "static" ensures it's only
  // called once.
  static ze_result_t ZeResult = ZE_CALL_NOCHECK(zeInit, (0));

  // Absorb the ZE_RESULT_ERROR_UNINITIALIZED and just return 0 Platforms.
  if (ZeResult == ZE_RESULT_ERROR_UNINITIALIZED) {
    PI_ASSERT(NumPlatforms != 0, PI_INVALID_VALUE);
    *NumPlatforms = 0;
    return PI_SUCCESS;
  }

  if (ZeResult != ZE_RESULT_SUCCESS) {
    zePrint("zeInit: Level Zero initialization failure\n");
    return mapError(ZeResult);
  }

  // Cache pi_platforms for reuse in the future
  // It solves two problems;
  // 1. sycl::platform equality issue; we always return the same pi_platform.
  // 2. performance; we can save time by immediately return from cache.
  //

  const std::lock_guard<sycl::detail::SpinLock> Lock{*PiPlatformsCacheMutex};
  if (!PiPlatformCachePopulated) {
    try {
      // Level Zero does not have concept of Platforms, but Level Zero driver is
      // the closest match.
      uint32_t ZeDriverCount = 0;
      ZE_CALL(zeDriverGet, (&ZeDriverCount, nullptr));
      if (ZeDriverCount == 0) {
        PiPlatformCachePopulated = true;
      } else {
        std::vector<ze_driver_handle_t> ZeDrivers;
        ZeDrivers.resize(ZeDriverCount);

        ZE_CALL(zeDriverGet, (&ZeDriverCount, ZeDrivers.data()));
        for (uint32_t I = 0; I < ZeDriverCount; ++I) {
          pi_platform Platform = new _pi_platform(ZeDrivers[I]);
          pi_result Result = Platform->initialize();
          if (Result != PI_SUCCESS) {
            return Result;
          }
          // Save a copy in the cache for future uses.
          PiPlatformsCache->push_back(Platform);
        }
        PiPlatformCachePopulated = true;
      }
    } catch (const std::bad_alloc &) {
      return PI_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return PI_ERROR_UNKNOWN;
    }
  }

  // Populate returned platforms from the cache.
  if (Platforms) {
    PI_ASSERT(NumEntries <= PiPlatformsCache->size(), PI_INVALID_PLATFORM);
    std::copy_n(PiPlatformsCache->begin(), NumEntries, Platforms);
  }

  if (NumPlatforms) {
    *NumPlatforms = PiPlatformsCache->size();
  }

  zePrint("Using events scope: %s\n",
          EventsScope == AllHostVisible ? "all host-visible"
          : EventsScope == OnDemandHostVisibleProxy
              ? "on demand host-visible proxy"
              : "only last command in a batch is host-visible");
  return PI_SUCCESS;
}

pi_result piPlatformGetInfo(pi_platform Platform, pi_platform_info ParamName,
                            size_t ParamValueSize, void *ParamValue,
                            size_t *ParamValueSizeRet) {

  PI_ASSERT(Platform, PI_INVALID_PLATFORM);

  zePrint("==========================\n");
  zePrint("SYCL over Level-Zero %s\n", Platform->ZeDriverVersion.c_str());
  zePrint("==========================\n");

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  switch (ParamName) {
  case PI_PLATFORM_INFO_NAME:
    // TODO: Query Level Zero driver when relevant info is added there.
    return ReturnValue("Intel(R) Level-Zero");
  case PI_PLATFORM_INFO_VENDOR:
    // TODO: Query Level Zero driver when relevant info is added there.
    return ReturnValue("Intel(R) Corporation");
  case PI_PLATFORM_INFO_EXTENSIONS:
    // Convention adopted from OpenCL:
    //     "Returns a space-separated list of extension names (the extension
    // names themselves do not contain any spaces) supported by the platform.
    // Extensions defined here must be supported by all devices associated
    // with this platform."
    //
    // TODO: Check the common extensions supported by all connected devices and
    // return them. For now, hardcoding some extensions we know are supported by
    // all Level Zero devices.
    return ReturnValue(ZE_SUPPORTED_EXTENSIONS);
  case PI_PLATFORM_INFO_PROFILE:
    // TODO: figure out what this means and how is this used
    return ReturnValue("FULL_PROFILE");
  case PI_PLATFORM_INFO_VERSION:
    // TODO: this should query to zeDriverGetDriverVersion
    // but we don't yet have the driver handle here.
    //
    // From OpenCL 2.1: "This version string has the following format:
    // OpenCL<space><major_version.minor_version><space><platform-specific
    // information>. Follow the same notation here.
    //
    return ReturnValue(Platform->ZeDriverApiVersion.c_str());
  default:
    zePrint("piPlatformGetInfo: unrecognized ParamName\n");
    return PI_INVALID_VALUE;
  }

  return PI_SUCCESS;
}

pi_result piextPlatformGetNativeHandle(pi_platform Platform,
                                       pi_native_handle *NativeHandle) {
  PI_ASSERT(Platform, PI_INVALID_PLATFORM);
  PI_ASSERT(NativeHandle, PI_INVALID_VALUE);

  auto ZeDriver = pi_cast<ze_driver_handle_t *>(NativeHandle);
  // Extract the Level Zero driver handle from the given PI platform
  *ZeDriver = Platform->ZeDriver;
  return PI_SUCCESS;
}

pi_result piextPlatformCreateWithNativeHandle(pi_native_handle NativeHandle,
                                              pi_platform *Platform) {
  PI_ASSERT(Platform, PI_INVALID_PLATFORM);
  PI_ASSERT(NativeHandle, PI_INVALID_VALUE);

  auto ZeDriver = pi_cast<ze_driver_handle_t>(NativeHandle);

  pi_uint32 NumPlatforms = 0;
  pi_result Res = piPlatformsGet(0, nullptr, &NumPlatforms);
  if (Res != PI_SUCCESS) {
    return Res;
  }

  if (NumPlatforms) {
    std::vector<pi_platform> Platforms(NumPlatforms);
    PI_CALL(piPlatformsGet(NumPlatforms, Platforms.data(), nullptr));

    // The SYCL spec requires that the set of platforms must remain fixed for
    // the duration of the application's execution. We assume that we found all
    // of the Level Zero drivers when we initialized the platform cache, so the
    // "NativeHandle" must already be in the cache. If it is not, this must not
    // be a valid Level Zero driver.
    for (const pi_platform &CachedPlatform : Platforms) {
      if (CachedPlatform->ZeDriver == ZeDriver) {
        *Platform = CachedPlatform;
        return PI_SUCCESS;
      }
    }
  }

  return PI_INVALID_VALUE;
}

// Get the cahched PI device created for the L0 device handle.
// Return NULL if no such PI device found.
pi_device _pi_platform::getDeviceFromNativeHandle(ze_device_handle_t ZeDevice) {

  pi_result Res = populateDeviceCacheIfNeeded();
  if (Res != PI_SUCCESS) {
    return nullptr;
  }

  // TODO: our sub-sub-device representation is currently [Level-Zero device
  // handle + Level-Zero compute group/engine index], so there is now no 1:1
  // mapping from L0 device handle to PI device assumed in this function. Until
  // Level-Zero adds unique ze_device_handle_t for sub-sub-devices, here we
  // filter out PI sub-sub-devices.
  auto it = std::find_if(PiDevicesCache.begin(), PiDevicesCache.end(),
                         [&](std::unique_ptr<_pi_device> &D) {
                           return D.get()->ZeDevice == ZeDevice &&
                                  (D.get()->RootDevice == nullptr ||
                                   D.get()->RootDevice->RootDevice == nullptr);
                         });
  if (it != PiDevicesCache.end()) {
    return (*it).get();
  }
  return nullptr;
}

pi_result piDevicesGet(pi_platform Platform, pi_device_type DeviceType,
                       pi_uint32 NumEntries, pi_device *Devices,
                       pi_uint32 *NumDevices) {

  PI_ASSERT(Platform, PI_INVALID_PLATFORM);

  pi_result Res = Platform->populateDeviceCacheIfNeeded();
  if (Res != PI_SUCCESS) {
    return Res;
  }

  // Filter available devices based on input DeviceType
  std::vector<pi_device> MatchedDevices;
  for (auto &D : Platform->PiDevicesCache) {
    // Only ever return root-devices from piDevicesGet, but the
    // devices cache also keeps sub-devices.
    if (D->isSubDevice())
      continue;

    bool Matched = false;
    switch (DeviceType) {
    case PI_DEVICE_TYPE_ALL:
      Matched = true;
      break;
    case PI_DEVICE_TYPE_GPU:
    case PI_DEVICE_TYPE_DEFAULT:
      Matched = (D->ZeDeviceProperties->type == ZE_DEVICE_TYPE_GPU);
      break;
    case PI_DEVICE_TYPE_CPU:
      Matched = (D->ZeDeviceProperties->type == ZE_DEVICE_TYPE_CPU);
      break;
    case PI_DEVICE_TYPE_ACC:
      Matched = (D->ZeDeviceProperties->type == ZE_DEVICE_TYPE_MCA ||
                 D->ZeDeviceProperties->type == ZE_DEVICE_TYPE_FPGA);
      break;
    default:
      Matched = false;
      zePrint("Unknown device type");
      break;
    }
    if (Matched)
      MatchedDevices.push_back(D.get());
  }

  uint32_t ZeDeviceCount = MatchedDevices.size();

  if (NumDevices)
    *NumDevices = ZeDeviceCount;

  if (NumEntries == 0) {
    // Devices should be nullptr when querying the number of devices
    PI_ASSERT(Devices == nullptr, PI_INVALID_VALUE);
    return PI_SUCCESS;
  }

  // Return the devices from the cache.
  if (Devices) {
    PI_ASSERT(NumEntries <= ZeDeviceCount, PI_INVALID_DEVICE);
    std::copy_n(MatchedDevices.begin(), NumEntries, Devices);
  }

  return PI_SUCCESS;
}

// Check the device cache and load it if necessary.
pi_result _pi_platform::populateDeviceCacheIfNeeded() {
  std::lock_guard<std::mutex> Lock(PiDevicesCacheMutex);

  if (DeviceCachePopulated) {
    return PI_SUCCESS;
  }

  uint32_t ZeDeviceCount = 0;
  ZE_CALL(zeDeviceGet, (ZeDriver, &ZeDeviceCount, nullptr));

  try {
    std::vector<ze_device_handle_t> ZeDevices(ZeDeviceCount);
    ZE_CALL(zeDeviceGet, (ZeDriver, &ZeDeviceCount, ZeDevices.data()));

    for (uint32_t I = 0; I < ZeDeviceCount; ++I) {
      std::unique_ptr<_pi_device> Device(new _pi_device(ZeDevices[I], this));
      pi_result Result = Device->initialize();
      if (Result != PI_SUCCESS) {
        return Result;
      }

      // Additionally we need to cache all sub-devices too, such that they
      // are readily visible to the piextDeviceCreateWithNativeHandle.
      //
      pi_uint32 SubDevicesCount = 0;
      ZE_CALL(zeDeviceGetSubDevices,
              (Device->ZeDevice, &SubDevicesCount, nullptr));

      auto ZeSubdevices = new ze_device_handle_t[SubDevicesCount];
      ZE_CALL(zeDeviceGetSubDevices,
              (Device->ZeDevice, &SubDevicesCount, ZeSubdevices));

      // Wrap the Level Zero sub-devices into PI sub-devices, and add them to
      // cache.
      for (uint32_t I = 0; I < SubDevicesCount; ++I) {
        std::unique_ptr<_pi_device> PiSubDevice(
            new _pi_device(ZeSubdevices[I], this, Device.get()));
        pi_result Result = PiSubDevice->initialize();
        if (Result != PI_SUCCESS) {
          delete[] ZeSubdevices;
          return Result;
        }

        // collect all the ordinals for the sub-sub-devices
        std::vector<int> Ordinals;

        uint32_t numQueueGroups = 0;
        ZE_CALL(zeDeviceGetCommandQueueGroupProperties,
                (PiSubDevice->ZeDevice, &numQueueGroups, nullptr));
        if (numQueueGroups == 0) {
          return PI_ERROR_UNKNOWN;
        }
        std::vector<ze_command_queue_group_properties_t> QueueGroupProperties(
            numQueueGroups);
        ZE_CALL(zeDeviceGetCommandQueueGroupProperties,
                (PiSubDevice->ZeDevice, &numQueueGroups,
                 QueueGroupProperties.data()));

        for (uint32_t i = 0; i < numQueueGroups; i++) {
          if (QueueGroupProperties[i].flags &
                  ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE &&
              QueueGroupProperties[i].numQueues > 1) {
            Ordinals.push_back(i);
          }
        }

        // Create PI sub-sub-devices with the sub-device for all the ordinals.
        // Each {ordinal, index} points to a specific CCS which constructs
        // a sub-sub-device at this point.
        for (uint32_t J = 0; J < Ordinals.size(); ++J) {
          for (uint32_t K = 0; K < QueueGroupProperties[Ordinals[J]].numQueues;
               ++K) {
            std::unique_ptr<_pi_device> PiSubSubDevice(
                new _pi_device(ZeSubdevices[I], this, PiSubDevice.get()));
            pi_result Result = PiSubSubDevice->initialize(Ordinals[J], K);
            if (Result != PI_SUCCESS) {
              return Result;
            }

            // save pointers to sub-sub-devices for quick retrieval in the
            // future.
            PiSubDevice->SubDevices.push_back(PiSubSubDevice.get());
            PiDevicesCache.push_back(std::move(PiSubSubDevice));
          }
        }

        // save pointers to sub-devices for quick retrieval in the future.
        Device->SubDevices.push_back(PiSubDevice.get());
        PiDevicesCache.push_back(std::move(PiSubDevice));
      }
      delete[] ZeSubdevices;

      // Save the root device in the cache for future uses.
      PiDevicesCache.push_back(std::move(Device));
    }
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  DeviceCachePopulated = true;
  return PI_SUCCESS;
}

pi_result piDeviceRetain(pi_device Device) {
  PI_ASSERT(Device, PI_INVALID_DEVICE);

  // The root-device ref-count remains unchanged (always 1).
  if (Device->isSubDevice()) {
    ++(Device->RefCount);
  }
  return PI_SUCCESS;
}

pi_result piDeviceRelease(pi_device Device) {
  PI_ASSERT(Device, PI_INVALID_DEVICE);

  // Check if the device is already released
  if (Device->RefCount <= 0)
    die("piDeviceRelease: the device has been already released");

  // Root devices are destroyed during the piTearDown process.
  if (Device->isSubDevice()) {
    if (--(Device->RefCount) == 0) {
      delete Device;
    }
  }

  return PI_SUCCESS;
}

pi_result piDeviceGetInfo(pi_device Device, pi_device_info ParamName,
                          size_t ParamValueSize, void *ParamValue,
                          size_t *ParamValueSizeRet) {

  PI_ASSERT(Device, PI_INVALID_DEVICE);

  ze_device_handle_t ZeDevice = Device->ZeDevice;

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  switch (ParamName) {
  case PI_DEVICE_INFO_TYPE: {
    switch (Device->ZeDeviceProperties->type) {
    case ZE_DEVICE_TYPE_GPU:
      return ReturnValue(PI_DEVICE_TYPE_GPU);
    case ZE_DEVICE_TYPE_CPU:
      return ReturnValue(PI_DEVICE_TYPE_CPU);
    case ZE_DEVICE_TYPE_MCA:
    case ZE_DEVICE_TYPE_FPGA:
      return ReturnValue(PI_DEVICE_TYPE_ACC);
    default:
      zePrint("This device type is not supported\n");
      return PI_INVALID_VALUE;
    }
  }
  case PI_DEVICE_INFO_PARENT_DEVICE:
    // TODO: all Level Zero devices are parent ?
    return ReturnValue(pi_device{0});
  case PI_DEVICE_INFO_PLATFORM:
    return ReturnValue(Device->Platform);
  case PI_DEVICE_INFO_VENDOR_ID:
    return ReturnValue(pi_uint32{Device->ZeDeviceProperties->vendorId});
  case PI_DEVICE_INFO_UUID:
    // Intel extension for device UUID. This returns the UUID as
    // std::array<std::byte, 16>. For details about this extension,
    // see sycl/doc/extensions/supported/SYCL_EXT_INTEL_DEVICE_INFO.md.
    return ReturnValue(Device->ZeDeviceProperties->uuid.id);
  case PI_DEVICE_INFO_EXTENSIONS: {
    // Convention adopted from OpenCL:
    //     "Returns a space separated list of extension names (the extension
    // names themselves do not contain any spaces) supported by the device."
    //
    // TODO: Use proper mechanism to get this information from Level Zero after
    // it is added to Level Zero.
    // Hardcoding the few we know are supported by the current hardware.
    //
    //
    std::string SupportedExtensions;

    // cl_khr_il_program - OpenCL 2.0 KHR extension for SPIR-V support. Core
    //   feature in >OpenCL 2.1
    // cl_khr_subgroups - Extension adds support for implementation-controlled
    //   subgroups.
    // cl_intel_subgroups - Extension adds subgroup features, defined by Intel.
    // cl_intel_subgroups_short - Extension adds subgroup functions described in
    //   the cl_intel_subgroups extension to support 16-bit integer data types
    //   for performance.
    // cl_intel_required_subgroup_size - Extension to allow programmers to
    //   optionally specify the required subgroup size for a kernel function.
    // cl_khr_fp16 - Optional half floating-point support.
    // cl_khr_fp64 - Support for double floating-point precision.
    // cl_khr_int64_base_atomics, cl_khr_int64_extended_atomics - Optional
    //   extensions that implement atomic operations on 64-bit signed and
    //   unsigned integers to locations in __global and __local memory.
    // cl_khr_3d_image_writes - Extension to enable writes to 3D image memory
    //   objects.
    //
    // Hardcoding some extensions we know are supported by all Level Zero
    // devices.
    SupportedExtensions += (ZE_SUPPORTED_EXTENSIONS);
    if (Device->ZeDeviceModuleProperties->flags & ZE_DEVICE_MODULE_FLAG_FP16)
      SupportedExtensions += ("cl_khr_fp16 ");
    if (Device->ZeDeviceModuleProperties->flags & ZE_DEVICE_MODULE_FLAG_FP64)
      SupportedExtensions += ("cl_khr_fp64 ");
    if (Device->ZeDeviceModuleProperties->flags &
        ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS)
      // int64AtomicsSupported indicates support for both.
      SupportedExtensions +=
          ("cl_khr_int64_base_atomics cl_khr_int64_extended_atomics ");
    if (Device->ZeDeviceImageProperties->maxImageDims3D > 0)
      // Supports reading and writing of images.
      SupportedExtensions += ("cl_khr_3d_image_writes ");

    return ReturnValue(SupportedExtensions.c_str());
  }
  case PI_DEVICE_INFO_NAME:
    return ReturnValue(Device->ZeDeviceProperties->name);
  case PI_DEVICE_INFO_COMPILER_AVAILABLE:
    return ReturnValue(pi_bool{1});
  case PI_DEVICE_INFO_LINKER_AVAILABLE:
    return ReturnValue(pi_bool{1});
  case PI_DEVICE_INFO_MAX_COMPUTE_UNITS: {
    pi_uint32 MaxComputeUnits =
        Device->ZeDeviceProperties->numEUsPerSubslice *
        Device->ZeDeviceProperties->numSubslicesPerSlice *
        Device->ZeDeviceProperties->numSlices;
    return ReturnValue(pi_uint32{MaxComputeUnits});
  }
  case PI_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS:
    // Level Zero spec defines only three dimensions
    return ReturnValue(pi_uint32{3});
  case PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE:
    return ReturnValue(
        pi_uint64{Device->ZeDeviceComputeProperties->maxTotalGroupSize});
  case PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES: {
    struct {
      size_t Arr[3];
    } MaxGroupSize = {{Device->ZeDeviceComputeProperties->maxGroupSizeX,
                       Device->ZeDeviceComputeProperties->maxGroupSizeY,
                       Device->ZeDeviceComputeProperties->maxGroupSizeZ}};
    return ReturnValue(MaxGroupSize);
  }
  case PI_EXT_ONEAPI_DEVICE_INFO_MAX_WORK_GROUPS_3D: {
    struct {
      size_t Arr[3];
    } MaxGroupCounts = {{Device->ZeDeviceComputeProperties->maxGroupCountX,
                         Device->ZeDeviceComputeProperties->maxGroupCountY,
                         Device->ZeDeviceComputeProperties->maxGroupCountZ}};
    return ReturnValue(MaxGroupCounts);
  }
  case PI_DEVICE_INFO_MAX_CLOCK_FREQUENCY:
    return ReturnValue(pi_uint32{Device->ZeDeviceProperties->coreClockRate});
  case PI_DEVICE_INFO_ADDRESS_BITS: {
    // TODO: To confirm with spec.
    return ReturnValue(pi_uint32{64});
  }
  case PI_DEVICE_INFO_MAX_MEM_ALLOC_SIZE:
    return ReturnValue(pi_uint64{Device->ZeDeviceProperties->maxMemAllocSize});
  case PI_DEVICE_INFO_GLOBAL_MEM_SIZE: {
    uint64_t GlobalMemSize = 0;
    for (uint32_t I = 0; I < Device->ZeDeviceMemoryProperties->size(); I++) {
      GlobalMemSize +=
          (*Device->ZeDeviceMemoryProperties.operator->())[I].totalSize;
    }
    return ReturnValue(pi_uint64{GlobalMemSize});
  }
  case PI_DEVICE_INFO_LOCAL_MEM_SIZE:
    return ReturnValue(
        pi_uint64{Device->ZeDeviceComputeProperties->maxSharedLocalMemory});
  case PI_DEVICE_INFO_IMAGE_SUPPORT:
    return ReturnValue(
        pi_bool{Device->ZeDeviceImageProperties->maxImageDims1D > 0});
  case PI_DEVICE_INFO_HOST_UNIFIED_MEMORY:
    return ReturnValue(pi_bool{(Device->ZeDeviceProperties->flags &
                                ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) != 0});
  case PI_DEVICE_INFO_AVAILABLE:
    return ReturnValue(pi_bool{ZeDevice ? true : false});
  case PI_DEVICE_INFO_VENDOR:
    // TODO: Level-Zero does not return vendor's name at the moment
    // only the ID.
    return ReturnValue("Intel(R) Corporation");
  case PI_DEVICE_INFO_DRIVER_VERSION:
    return ReturnValue(Device->Platform->ZeDriverVersion.c_str());
  case PI_DEVICE_INFO_VERSION:
    return ReturnValue(Device->Platform->ZeDriverApiVersion.c_str());
  case PI_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES: {
    pi_result Res = Device->Platform->populateDeviceCacheIfNeeded();
    if (Res != PI_SUCCESS) {
      return Res;
    }
    return ReturnValue(pi_uint32{(unsigned int)(Device->SubDevices.size())});
  }
  case PI_DEVICE_INFO_REFERENCE_COUNT:
    return ReturnValue(pi_uint32{Device->RefCount});
  case PI_DEVICE_INFO_PARTITION_PROPERTIES: {
    // SYCL spec says: if this SYCL device cannot be partitioned into at least
    // two sub devices then the returned vector must be empty.
    pi_result Res = Device->Platform->populateDeviceCacheIfNeeded();
    if (Res != PI_SUCCESS) {
      return Res;
    }

    uint32_t ZeSubDeviceCount = Device->SubDevices.size();
    if (ZeSubDeviceCount < 2) {
      return ReturnValue(pi_device_partition_property{0});
    }
    // It is debatable if SYCL sub-device and partitioning APIs sufficient to
    // expose Level Zero sub-devices?  We start with support of
    // "partition_by_affinity_domain" and "next_partitionable" but if that
    // doesn't seem to be a good fit we could look at adding a more descriptive
    // partitioning type.
    struct {
      pi_device_partition_property Arr[2];
    } PartitionProperties = {{PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN, 0}};
    return ReturnValue(PartitionProperties);
  }
  case PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN:
    return ReturnValue(pi_device_affinity_domain{
        PI_DEVICE_AFFINITY_DOMAIN_NUMA |
        PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE});
  case PI_DEVICE_INFO_PARTITION_TYPE: {
    if (Device->isSubDevice()) {
      struct {
        pi_device_partition_property Arr[3];
      } PartitionProperties = {{PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
                                PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE,
                                0}};
      return ReturnValue(PartitionProperties);
    }
    // For root-device there is no partitioning to report.
    return ReturnValue(pi_device_partition_property{0});
  }

    // Everything under here is not supported yet

  case PI_DEVICE_INFO_OPENCL_C_VERSION:
    return ReturnValue("");
  case PI_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC:
    return ReturnValue(pi_bool{true});
  case PI_DEVICE_INFO_PRINTF_BUFFER_SIZE:
    return ReturnValue(
        size_t{Device->ZeDeviceModuleProperties->printfBufferSize});
  case PI_DEVICE_INFO_PROFILE:
    return ReturnValue("FULL_PROFILE");
  case PI_DEVICE_INFO_BUILT_IN_KERNELS:
    // TODO: To find out correct value
    return ReturnValue("");
  case PI_DEVICE_INFO_QUEUE_PROPERTIES:
    return ReturnValue(pi_queue_properties{
        PI_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | PI_QUEUE_PROFILING_ENABLE});
  case PI_DEVICE_INFO_EXECUTION_CAPABILITIES:
    return ReturnValue(
        pi_device_exec_capabilities{PI_DEVICE_EXEC_CAPABILITIES_NATIVE_KERNEL});
  case PI_DEVICE_INFO_ENDIAN_LITTLE:
    return ReturnValue(pi_bool{true});
  case PI_DEVICE_INFO_ERROR_CORRECTION_SUPPORT:
    return ReturnValue(pi_bool{Device->ZeDeviceProperties->flags &
                               ZE_DEVICE_PROPERTY_FLAG_ECC});
  case PI_DEVICE_INFO_PROFILING_TIMER_RESOLUTION:
    return ReturnValue(size_t{Device->ZeDeviceProperties->timerResolution});
  case PI_DEVICE_INFO_LOCAL_MEM_TYPE:
    return ReturnValue(PI_DEVICE_LOCAL_MEM_TYPE_LOCAL);
  case PI_DEVICE_INFO_MAX_CONSTANT_ARGS:
    return ReturnValue(pi_uint32{64});
  case PI_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE:
    return ReturnValue(
        pi_uint64{Device->ZeDeviceImageProperties->maxImageBufferSize});
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE:
    return ReturnValue(PI_DEVICE_MEM_CACHE_TYPE_READ_WRITE_CACHE);
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE:
    return ReturnValue(
        // TODO[1.0]: how to query cache line-size?
        pi_uint32{1});
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE:
    return ReturnValue(pi_uint64{Device->ZeDeviceCacheProperties->cacheSize});
  case PI_DEVICE_INFO_MAX_PARAMETER_SIZE:
    return ReturnValue(
        size_t{Device->ZeDeviceModuleProperties->maxArgumentsSize});
  case PI_DEVICE_INFO_MEM_BASE_ADDR_ALIGN:
    // SYCL/OpenCL spec is vague on what this means exactly, but seems to
    // be for "alignment requirement (in bits) for sub-buffer offsets."
    // An OpenCL implementation returns 8*128, but Level Zero can do just 8,
    // meaning unaligned access for values of types larger than 8 bits.
    return ReturnValue(pi_uint32{8});
  case PI_DEVICE_INFO_MAX_SAMPLERS:
    return ReturnValue(pi_uint32{Device->ZeDeviceImageProperties->maxSamplers});
  case PI_DEVICE_INFO_MAX_READ_IMAGE_ARGS:
    return ReturnValue(
        pi_uint32{Device->ZeDeviceImageProperties->maxReadImageArgs});
  case PI_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS:
    return ReturnValue(
        pi_uint32{Device->ZeDeviceImageProperties->maxWriteImageArgs});
  case PI_DEVICE_INFO_SINGLE_FP_CONFIG: {
    uint64_t SingleFPValue = 0;
    ze_device_fp_flags_t ZeSingleFPCapabilities =
        Device->ZeDeviceModuleProperties->fp32flags;
    if (ZE_DEVICE_FP_FLAG_DENORM & ZeSingleFPCapabilities) {
      SingleFPValue |= PI_FP_DENORM;
    }
    if (ZE_DEVICE_FP_FLAG_INF_NAN & ZeSingleFPCapabilities) {
      SingleFPValue |= PI_FP_INF_NAN;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST & ZeSingleFPCapabilities) {
      SingleFPValue |= PI_FP_ROUND_TO_NEAREST;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO & ZeSingleFPCapabilities) {
      SingleFPValue |= PI_FP_ROUND_TO_ZERO;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_INF & ZeSingleFPCapabilities) {
      SingleFPValue |= PI_FP_ROUND_TO_INF;
    }
    if (ZE_DEVICE_FP_FLAG_FMA & ZeSingleFPCapabilities) {
      SingleFPValue |= PI_FP_FMA;
    }
    if (ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT & ZeSingleFPCapabilities) {
      SingleFPValue |= PI_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    }
    return ReturnValue(pi_uint64{SingleFPValue});
  }
  case PI_DEVICE_INFO_HALF_FP_CONFIG: {
    uint64_t HalfFPValue = 0;
    ze_device_fp_flags_t ZeHalfFPCapabilities =
        Device->ZeDeviceModuleProperties->fp16flags;
    if (ZE_DEVICE_FP_FLAG_DENORM & ZeHalfFPCapabilities) {
      HalfFPValue |= PI_FP_DENORM;
    }
    if (ZE_DEVICE_FP_FLAG_INF_NAN & ZeHalfFPCapabilities) {
      HalfFPValue |= PI_FP_INF_NAN;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST & ZeHalfFPCapabilities) {
      HalfFPValue |= PI_FP_ROUND_TO_NEAREST;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO & ZeHalfFPCapabilities) {
      HalfFPValue |= PI_FP_ROUND_TO_ZERO;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_INF & ZeHalfFPCapabilities) {
      HalfFPValue |= PI_FP_ROUND_TO_INF;
    }
    if (ZE_DEVICE_FP_FLAG_FMA & ZeHalfFPCapabilities) {
      HalfFPValue |= PI_FP_FMA;
    }
    if (ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT & ZeHalfFPCapabilities) {
      HalfFPValue |= PI_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    }
    return ReturnValue(pi_uint64{HalfFPValue});
  }
  case PI_DEVICE_INFO_DOUBLE_FP_CONFIG: {
    uint64_t DoubleFPValue = 0;
    ze_device_fp_flags_t ZeDoubleFPCapabilities =
        Device->ZeDeviceModuleProperties->fp64flags;
    if (ZE_DEVICE_FP_FLAG_DENORM & ZeDoubleFPCapabilities) {
      DoubleFPValue |= PI_FP_DENORM;
    }
    if (ZE_DEVICE_FP_FLAG_INF_NAN & ZeDoubleFPCapabilities) {
      DoubleFPValue |= PI_FP_INF_NAN;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_NEAREST & ZeDoubleFPCapabilities) {
      DoubleFPValue |= PI_FP_ROUND_TO_NEAREST;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_ZERO & ZeDoubleFPCapabilities) {
      DoubleFPValue |= PI_FP_ROUND_TO_ZERO;
    }
    if (ZE_DEVICE_FP_FLAG_ROUND_TO_INF & ZeDoubleFPCapabilities) {
      DoubleFPValue |= PI_FP_ROUND_TO_INF;
    }
    if (ZE_DEVICE_FP_FLAG_FMA & ZeDoubleFPCapabilities) {
      DoubleFPValue |= PI_FP_FMA;
    }
    if (ZE_DEVICE_FP_FLAG_ROUNDED_DIVIDE_SQRT & ZeDoubleFPCapabilities) {
      DoubleFPValue |= PI_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    }
    return ReturnValue(pi_uint64{DoubleFPValue});
  }
  case PI_DEVICE_INFO_IMAGE2D_MAX_WIDTH:
    // Until Level Zero provides needed info, hardcode default minimum values
    // required by the SYCL specification.
    return ReturnValue(size_t{8192});
  case PI_DEVICE_INFO_IMAGE2D_MAX_HEIGHT:
    // Until Level Zero provides needed info, hardcode default minimum values
    // required by the SYCL specification.
    return ReturnValue(size_t{8192});
  case PI_DEVICE_INFO_IMAGE3D_MAX_WIDTH:
    // Until Level Zero provides needed info, hardcode default minimum values
    // required by the SYCL specification.
    return ReturnValue(size_t{2048});
  case PI_DEVICE_INFO_IMAGE3D_MAX_HEIGHT:
    // Until Level Zero provides needed info, hardcode default minimum values
    // required by the SYCL specification.
    return ReturnValue(size_t{2048});
  case PI_DEVICE_INFO_IMAGE3D_MAX_DEPTH:
    // Until Level Zero provides needed info, hardcode default minimum values
    // required by the SYCL specification.
    return ReturnValue(size_t{2048});
  case PI_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE:
    return ReturnValue(
        size_t{Device->ZeDeviceImageProperties->maxImageBufferSize});
  case PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE:
    return ReturnValue(
        size_t{Device->ZeDeviceImageProperties->maxImageArraySlices});
  // Handle SIMD widths.
  // TODO: can we do better than this?
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 1);
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 2);
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 4);
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 8);
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 4);
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 8);
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF:
    return ReturnValue(Device->ZeDeviceProperties->physicalEUSimdWidth / 2);
  case PI_DEVICE_INFO_MAX_NUM_SUB_GROUPS: {
    // Max_num_sub_Groups = maxTotalGroupSize/min(set of subGroupSizes);
    uint32_t MinSubGroupSize =
        Device->ZeDeviceComputeProperties->subGroupSizes[0];
    for (uint32_t I = 1;
         I < Device->ZeDeviceComputeProperties->numSubGroupSizes; I++) {
      if (MinSubGroupSize > Device->ZeDeviceComputeProperties->subGroupSizes[I])
        MinSubGroupSize = Device->ZeDeviceComputeProperties->subGroupSizes[I];
    }
    return ReturnValue(Device->ZeDeviceComputeProperties->maxTotalGroupSize /
                       MinSubGroupSize);
  }
  case PI_DEVICE_INFO_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS: {
    // TODO: Not supported yet. Needs to be updated after support is added.
    return ReturnValue(pi_bool{false});
  }
  case PI_DEVICE_INFO_SUB_GROUP_SIZES_INTEL: {
    // ze_device_compute_properties.subGroupSizes is in uint32_t whereas the
    // expected return is size_t datatype. size_t can be 8 bytes of data.
    return getInfoArray<uint32_t, size_t>(
        Device->ZeDeviceComputeProperties->numSubGroupSizes, ParamValueSize,
        ParamValue, ParamValueSizeRet,
        Device->ZeDeviceComputeProperties->subGroupSizes);
  }
  case PI_DEVICE_INFO_IL_VERSION: {
    // Set to a space separated list of IL version strings of the form
    // <IL_Prefix>_<Major_version>.<Minor_version>.
    // "SPIR-V" is a required IL prefix when cl_khr_il_progam extension is
    // reported.
    uint32_t SpirvVersion =
        Device->ZeDeviceModuleProperties->spirvVersionSupported;
    uint32_t SpirvVersionMajor = ZE_MAJOR_VERSION(SpirvVersion);
    uint32_t SpirvVersionMinor = ZE_MINOR_VERSION(SpirvVersion);

    char SpirvVersionString[50];
    int Len = sprintf(SpirvVersionString, "SPIR-V_%d.%d ", SpirvVersionMajor,
                      SpirvVersionMinor);
    // returned string to contain only len number of characters.
    std::string ILVersion(SpirvVersionString, Len);
    return ReturnValue(ILVersion.c_str());
  }
  case PI_DEVICE_INFO_USM_HOST_SUPPORT:
  case PI_DEVICE_INFO_USM_DEVICE_SUPPORT:
  case PI_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT:
  case PI_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT:
  case PI_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT: {
    auto MapCaps = [](const ze_memory_access_cap_flags_t &ZeCapabilities) {
      pi_uint64 Capabilities = 0;
      if (ZeCapabilities & ZE_MEMORY_ACCESS_CAP_FLAG_RW)
        Capabilities |= PI_USM_ACCESS;
      if (ZeCapabilities & ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC)
        Capabilities |= PI_USM_ATOMIC_ACCESS;
      if (ZeCapabilities & ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT)
        Capabilities |= PI_USM_CONCURRENT_ACCESS;
      if (ZeCapabilities & ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT_ATOMIC)
        Capabilities |= PI_USM_CONCURRENT_ATOMIC_ACCESS;
      return Capabilities;
    };
    auto &Props = Device->ZeDeviceMemoryAccessProperties;
    switch (ParamName) {
    case PI_DEVICE_INFO_USM_HOST_SUPPORT:
      return ReturnValue(MapCaps(Props->hostAllocCapabilities));
    case PI_DEVICE_INFO_USM_DEVICE_SUPPORT:
      return ReturnValue(MapCaps(Props->deviceAllocCapabilities));
    case PI_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT:
      return ReturnValue(MapCaps(Props->sharedSingleDeviceAllocCapabilities));
    case PI_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT:
      return ReturnValue(MapCaps(Props->sharedCrossDeviceAllocCapabilities));
    case PI_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT:
      return ReturnValue(MapCaps(Props->sharedSystemAllocCapabilities));
    default:
      die("piDeviceGetInfo: enexpected ParamName.");
    }
  }

    // intel extensions for GPU information
  case PI_DEVICE_INFO_PCI_ADDRESS: {
    if (getenv("ZES_ENABLE_SYSMAN") == nullptr) {
      zePrint("Set SYCL_ENABLE_PCI=1 to obtain PCI data.\n");
      return PI_INVALID_VALUE;
    }
    ZesStruct<zes_pci_properties_t> ZeDevicePciProperties;
    ZE_CALL(zesDevicePciGetProperties, (ZeDevice, &ZeDevicePciProperties));
    std::stringstream ss;
    ss << ZeDevicePciProperties.address.domain << ":"
       << ZeDevicePciProperties.address.bus << ":"
       << ZeDevicePciProperties.address.device << "."
       << ZeDevicePciProperties.address.function;
    return ReturnValue(ss.str().c_str());
  }
  case PI_DEVICE_INFO_GPU_EU_COUNT: {
    pi_uint32 count = Device->ZeDeviceProperties->numEUsPerSubslice *
                      Device->ZeDeviceProperties->numSubslicesPerSlice *
                      Device->ZeDeviceProperties->numSlices;
    return ReturnValue(pi_uint32{count});
  }
  case PI_DEVICE_INFO_GPU_EU_SIMD_WIDTH:
    return ReturnValue(
        pi_uint32{Device->ZeDeviceProperties->physicalEUSimdWidth});
  case PI_DEVICE_INFO_GPU_SLICES:
    return ReturnValue(pi_uint32{Device->ZeDeviceProperties->numSlices});
  case PI_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE:
    return ReturnValue(
        pi_uint32{Device->ZeDeviceProperties->numSubslicesPerSlice});
  case PI_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE:
    return ReturnValue(
        pi_uint32{Device->ZeDeviceProperties->numEUsPerSubslice});
  case PI_DEVICE_INFO_GPU_HW_THREADS_PER_EU:
    return ReturnValue(pi_uint32{Device->ZeDeviceProperties->numThreadsPerEU});
  case PI_DEVICE_INFO_MAX_MEM_BANDWIDTH:
    // currently not supported in level zero runtime
    return PI_INVALID_VALUE;

  // TODO: Implement.
  case PI_DEVICE_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES:
  default:
    zePrint("Unsupported ParamName in piGetDeviceInfo\n");
    zePrint("ParamName=%d(0x%x)\n", ParamName, ParamName);
    return PI_INVALID_VALUE;
  }

  return PI_SUCCESS;
}

pi_result piDevicePartition(pi_device Device,
                            const pi_device_partition_property *Properties,
                            pi_uint32 NumDevices, pi_device *OutDevices,
                            pi_uint32 *OutNumDevices) {
  // Other partitioning ways are not supported by Level Zero
  if (Properties[0] != PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN ||
      (Properties[1] != PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE &&
       Properties[1] != PI_DEVICE_AFFINITY_DOMAIN_NUMA)) {
    return PI_INVALID_VALUE;
  }

  PI_ASSERT(Device, PI_INVALID_DEVICE);

  // Devices cache is normally created in piDevicesGet but still make
  // sure that cache is populated.
  //
  pi_result Res = Device->Platform->populateDeviceCacheIfNeeded();
  if (Res != PI_SUCCESS) {
    return Res;
  }

  if (OutNumDevices) {
    *OutNumDevices = Device->SubDevices.size();
  }

  if (OutDevices) {
    // TODO: Consider support for partitioning to <= total sub-devices.
    // Currently supported partitioning (by affinity domain/numa) would always
    // partition to all sub-devices.
    //
    PI_ASSERT(NumDevices == Device->SubDevices.size(), PI_INVALID_VALUE);

    for (uint32_t I = 0; I < NumDevices; I++) {
      OutDevices[I] = Device->SubDevices[I];
      // reusing the same pi_device needs to increment the reference count
      PI_CALL(piDeviceRetain(OutDevices[I]));
    }
  }
  return PI_SUCCESS;
}

pi_result
piextDeviceSelectBinary(pi_device Device, // TODO: does this need to be context?
                        pi_device_binary *Binaries, pi_uint32 NumBinaries,
                        pi_uint32 *SelectedBinaryInd) {

  PI_ASSERT(Device, PI_INVALID_DEVICE);
  PI_ASSERT(SelectedBinaryInd, PI_INVALID_VALUE);
  PI_ASSERT(NumBinaries == 0 || Binaries, PI_INVALID_VALUE);

  // TODO: this is a bare-bones implementation for choosing a device image
  // that would be compatible with the targeted device. An AOT-compiled
  // image is preferred over SPIR-V for known devices (i.e. Intel devices)
  // The implementation makes no effort to differentiate between multiple images
  // for the given device, and simply picks the first one compatible.
  //
  // Real implementation will use the same mechanism OpenCL ICD dispatcher
  // uses. Something like:
  //   PI_VALIDATE_HANDLE_RETURN_HANDLE(ctx, PI_INVALID_CONTEXT);
  //     return context->dispatch->piextDeviceSelectIR(
  //       ctx, images, num_images, selected_image);
  // where context->dispatch is set to the dispatch table provided by PI
  // plugin for platform/device the ctx was created for.

  // Look for GEN binary, which we known can only be handled by Level-Zero now.
  const char *BinaryTarget = __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_GEN;

  // Find the appropriate device image, fallback to spirv if not found
  constexpr pi_uint32 InvalidInd = std::numeric_limits<pi_uint32>::max();
  pi_uint32 Spirv = InvalidInd;

  for (pi_uint32 i = 0; i < NumBinaries; ++i) {
    if (strcmp(Binaries[i]->DeviceTargetSpec, BinaryTarget) == 0) {
      *SelectedBinaryInd = i;
      return PI_SUCCESS;
    }
    if (strcmp(Binaries[i]->DeviceTargetSpec,
               __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64) == 0)
      Spirv = i;
  }
  // Points to a spirv image, if such indeed was found
  if ((*SelectedBinaryInd = Spirv) != InvalidInd)
    return PI_SUCCESS;

  // No image can be loaded for the given device
  return PI_INVALID_BINARY;
}

pi_result piextDeviceGetNativeHandle(pi_device Device,
                                     pi_native_handle *NativeHandle) {
  PI_ASSERT(Device, PI_INVALID_DEVICE);
  PI_ASSERT(NativeHandle, PI_INVALID_VALUE);

  auto ZeDevice = pi_cast<ze_device_handle_t *>(NativeHandle);
  // Extract the Level Zero module handle from the given PI device
  *ZeDevice = Device->ZeDevice;
  return PI_SUCCESS;
}

pi_result piextDeviceCreateWithNativeHandle(pi_native_handle NativeHandle,
                                            pi_platform Platform,
                                            pi_device *Device) {
  PI_ASSERT(Device, PI_INVALID_DEVICE);
  PI_ASSERT(NativeHandle, PI_INVALID_VALUE);

  auto ZeDevice = pi_cast<ze_device_handle_t>(NativeHandle);

  // The SYCL spec requires that the set of devices must remain fixed for the
  // duration of the application's execution. We assume that we found all of the
  // Level Zero devices when we initialized the platforms/devices cache, so the
  // "NativeHandle" must already be in the cache. If it is not, this must not be
  // a valid Level Zero device.
  //
  // TODO: maybe we should populate cache of platforms if it wasn't already.
  // For now assert that is was populated.
  PI_ASSERT(PiPlatformCachePopulated, PI_INVALID_VALUE);
  const std::lock_guard<sycl::detail::SpinLock> Lock{*PiPlatformsCacheMutex};

  pi_device Dev = nullptr;
  for (auto &ThePlatform : *PiPlatformsCache) {
    Dev = ThePlatform->getDeviceFromNativeHandle(ZeDevice);
    if (Dev) {
      // Check that the input Platform, if was given, matches the found one.
      PI_ASSERT(!Platform || Platform == ThePlatform, PI_INVALID_PLATFORM);
      break;
    }
  }

  if (Dev == nullptr)
    return PI_INVALID_VALUE;

  *Device = Dev;
  return PI_SUCCESS;
}

pi_result piContextCreate(const pi_context_properties *Properties,
                          pi_uint32 NumDevices, const pi_device *Devices,
                          void (*PFnNotify)(const char *ErrInfo,
                                            const void *PrivateInfo, size_t CB,
                                            void *UserData),
                          void *UserData, pi_context *RetContext) {
  (void)Properties;
  (void)PFnNotify;
  (void)UserData;
  PI_ASSERT(NumDevices, PI_INVALID_VALUE);
  PI_ASSERT(Devices, PI_INVALID_DEVICE);
  PI_ASSERT(RetContext, PI_INVALID_VALUE);

  pi_platform Platform = (*Devices)->Platform;
  ZeStruct<ze_context_desc_t> ContextDesc;
  ContextDesc.flags = 0;

  ze_context_handle_t ZeContext;
  ZE_CALL(zeContextCreate, (Platform->ZeDriver, &ContextDesc, &ZeContext));
  try {
    *RetContext = new _pi_context(ZeContext, NumDevices, Devices, true);
    (*RetContext)->initialize();
    if (IndirectAccessTrackingEnabled) {
      std::lock_guard<std::mutex> Lock(Platform->ContextsMutex);
      Platform->Contexts.push_back(*RetContext);
    }
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

pi_result piContextGetInfo(pi_context Context, pi_context_info ParamName,
                           size_t ParamValueSize, void *ParamValue,
                           size_t *ParamValueSizeRet) {

  PI_ASSERT(Context, PI_INVALID_CONTEXT);

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  switch (ParamName) {
  case PI_CONTEXT_INFO_DEVICES:
    return getInfoArray(Context->Devices.size(), ParamValueSize, ParamValue,
                        ParamValueSizeRet, &Context->Devices[0]);
  case PI_CONTEXT_INFO_NUM_DEVICES:
    return ReturnValue(pi_uint32(Context->Devices.size()));
  case PI_CONTEXT_INFO_REFERENCE_COUNT:
    return ReturnValue(pi_uint32{Context->RefCount});
  case PI_CONTEXT_INFO_ATOMIC_MEMORY_SCOPE_CAPABILITIES:
  default:
    // TODO: implement other parameters
    die("piGetContextInfo: unsuppported ParamName.");
  }

  return PI_SUCCESS;
}

// FIXME: Dummy implementation to prevent link fail
pi_result piextContextSetExtendedDeleter(pi_context Context,
                                         pi_context_extended_deleter Function,
                                         void *UserData) {
  (void)Context;
  (void)Function;
  (void)UserData;
  die("piextContextSetExtendedDeleter: not supported");
  return PI_SUCCESS;
}

pi_result piextContextGetNativeHandle(pi_context Context,
                                      pi_native_handle *NativeHandle) {
  PI_ASSERT(Context, PI_INVALID_CONTEXT);
  PI_ASSERT(NativeHandle, PI_INVALID_VALUE);

  auto ZeContext = pi_cast<ze_context_handle_t *>(NativeHandle);
  // Extract the Level Zero queue handle from the given PI queue
  *ZeContext = Context->ZeContext;
  return PI_SUCCESS;
}

pi_result piextContextCreateWithNativeHandle(pi_native_handle NativeHandle,
                                             pi_uint32 NumDevices,
                                             const pi_device *Devices,
                                             bool OwnNativeHandle,
                                             pi_context *RetContext) {
  PI_ASSERT(NativeHandle, PI_INVALID_VALUE);
  PI_ASSERT(Devices, PI_INVALID_DEVICE);
  PI_ASSERT(RetContext, PI_INVALID_VALUE);
  PI_ASSERT(NumDevices, PI_INVALID_VALUE);

  try {
    *RetContext = new _pi_context(pi_cast<ze_context_handle_t>(NativeHandle),
                                  NumDevices, Devices, OwnNativeHandle);
    (*RetContext)->initialize();
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

pi_result piContextRetain(pi_context Context) {

  PI_ASSERT(Context, PI_INVALID_CONTEXT);

  ++(Context->RefCount);
  return PI_SUCCESS;
}

// Helper function to release the context, a caller must lock the platform-level
// mutex guarding the container with contexts because the context can be removed
// from the list of tracked contexts.
pi_result ContextReleaseHelper(pi_context Context) {

  PI_ASSERT(Context, PI_INVALID_CONTEXT);

  if (--(Context->RefCount) == 0) {
    if (IndirectAccessTrackingEnabled) {
      pi_platform Plt = Context->Devices[0]->Platform;
      auto &Contexts = Plt->Contexts;
      auto It = std::find(Contexts.begin(), Contexts.end(), Context);
      if (It != Contexts.end())
        Contexts.erase(It);
    }
    ze_context_handle_t DestoryZeContext =
        Context->OwnZeContext ? Context->ZeContext : nullptr;

    // Clean up any live memory associated with Context
    pi_result Result = Context->finalize();

    // We must delete Context first and then destroy zeContext because
    // Context deallocation requires ZeContext in some member deallocation of
    // pi_context.
    delete Context;

    // Destruction of some members of pi_context uses L0 context
    // and therefore it must be valid at that point.
    // Technically it should be placed to the destructor of pi_context
    // but this makes API error handling more complex.
    if (DestoryZeContext)
      ZE_CALL(zeContextDestroy, (DestoryZeContext));

    return Result;
  }
  return PI_SUCCESS;
}

pi_result piContextRelease(pi_context Context) {
  pi_platform Plt = Context->Devices[0]->Platform;
  std::unique_lock<std::mutex> ContextsLock(Plt->ContextsMutex,
                                            std::defer_lock);
  if (IndirectAccessTrackingEnabled)
    ContextsLock.lock();

  return ContextReleaseHelper(Context);
}

pi_result piQueueCreate(pi_context Context, pi_device Device,
                        pi_queue_properties Properties, pi_queue *Queue) {

  // Check that unexpected bits are not set.
  PI_ASSERT(!(Properties & ~(PI_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                             PI_QUEUE_PROFILING_ENABLE | PI_QUEUE_ON_DEVICE |
                             PI_QUEUE_ON_DEVICE_DEFAULT)),
            PI_INVALID_VALUE);

  ze_device_handle_t ZeDevice;
  ze_command_queue_handle_t ZeComputeCommandQueue;
  PI_ASSERT(Context, PI_INVALID_CONTEXT);

  if (std::find(Context->Devices.begin(), Context->Devices.end(), Device) ==
      Context->Devices.end()) {
    return PI_INVALID_DEVICE;
  }

  PI_ASSERT(Device, PI_INVALID_DEVICE);

  ZeDevice = Device->ZeDevice;
  ZeStruct<ze_command_queue_desc_t> ZeCommandQueueDesc;
  ZeCommandQueueDesc.ordinal = Device->ZeComputeQueueGroupIndex;
  ZeCommandQueueDesc.index = Device->ZeComputeEngineIndex;
  ZeCommandQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;

  ZE_CALL(zeCommandQueueCreate,
          (Context->ZeContext, ZeDevice,
           &ZeCommandQueueDesc, // TODO: translate properties
           &ZeComputeCommandQueue));

  std::vector<ze_command_queue_handle_t> ZeCopyCommandQueues;

  // Create a placeholder in ZeCopyCommandQueues for a queue that will be used
  // to submit commands to main copy engine. This queue is initially NULL and
  // will be replaced by the Ze Command Queue which gets created just before its
  // first use.
  ze_command_queue_handle_t ZeMainCopyCommandQueue = nullptr;
  if (Device->hasMainCopyEngine()) {
    ZeCopyCommandQueues.push_back(ZeMainCopyCommandQueue);
  }

  // Create additional 'placeholder queues' to link copy engines and push them
  // into ZeCopyCommandQueues.
  if (Device->hasLinkCopyEngine()) {
    auto ZeNumLinkCopyQueues = Device->ZeLinkCopyQueueGroupProperties.numQueues;
    for (uint32_t i = 0; i < ZeNumLinkCopyQueues; ++i) {
      ze_command_queue_handle_t ZeLinkCopyCommandQueue = nullptr;
      ZeCopyCommandQueues.push_back(ZeLinkCopyCommandQueue);
    }
  }
  PI_ASSERT(Queue, PI_INVALID_QUEUE);

  try {
    *Queue = new _pi_queue(ZeComputeCommandQueue, ZeCopyCommandQueues, Context,
                           Device, true, Properties);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  return PI_SUCCESS;
}

pi_result piQueueGetInfo(pi_queue Queue, pi_queue_info ParamName,
                         size_t ParamValueSize, void *ParamValue,
                         size_t *ParamValueSizeRet) {

  PI_ASSERT(Queue, PI_INVALID_QUEUE);

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  // TODO: consider support for queue properties and size
  switch (ParamName) {
  case PI_QUEUE_INFO_CONTEXT:
    return ReturnValue(Queue->Context);
  case PI_QUEUE_INFO_DEVICE:
    return ReturnValue(Queue->Device);
  case PI_QUEUE_INFO_REFERENCE_COUNT:
    return ReturnValue(pi_uint32{Queue->RefCount});
  case PI_QUEUE_INFO_PROPERTIES:
    die("PI_QUEUE_INFO_PROPERTIES in piQueueGetInfo not implemented\n");
    break;
  case PI_QUEUE_INFO_SIZE:
    die("PI_QUEUE_INFO_SIZE in piQueueGetInfo not implemented\n");
    break;
  case PI_QUEUE_INFO_DEVICE_DEFAULT:
    die("PI_QUEUE_INFO_DEVICE_DEFAULT in piQueueGetInfo not implemented\n");
    break;
  default:
    zePrint("Unsupported ParamName in piQueueGetInfo: ParamName=%d(0x%x)\n",
            ParamName, ParamName);
    return PI_INVALID_VALUE;
  }

  return PI_SUCCESS;
}

pi_result piQueueRetain(pi_queue Queue) {
  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);
  Queue->RefCountExternal++;
  piQueueRetainNoLock(Queue);
  return PI_SUCCESS;
}

pi_result piQueueRelease(pi_queue Queue) {
  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  {
    // Have this scope such that the lock is released before the
    // queue is potentially deleted in QueueRelease.
    // Lock automatically releases when this goes out of scope.
    std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

    Queue->RefCountExternal--;
    if (Queue->RefCountExternal == 0) {
      // When external reference count goes to zero it is still possible
      // that internal references still exists, e.g. command-lists that
      // are not yet completed. So do full queue synchronization here
      // and perform proper cleanup.
      //
      // It is possible to get to here and still have an open command list
      // if no wait or finish ever occurred for this queue.
      if (auto Res = Queue->executeAllOpenCommandLists())
        return Res;

      // Make sure all commands get executed.
      // Only do so for a healthy queue as otherwise sync may not be valid.
      if (Queue->Healthy) {
        ZE_CALL(zeHostSynchronize, (Queue->ZeComputeCommandQueue));
        for (uint32_t i = 0; i < Queue->ZeCopyCommandQueues.size(); ++i) {
          if (Queue->ZeCopyCommandQueues[i])
            ZE_CALL(zeHostSynchronize, (Queue->ZeCopyCommandQueues[i]));
        }
      }

      // Destroy all the fences created associated with this queue.
      for (auto it = Queue->CommandListMap.begin();
           it != Queue->CommandListMap.end(); ++it) {
        // This fence wasn't yet signalled when we polled it for recycling
        // the command-list, so need to release the command-list too.
        if (it->second.InUse) {
          Queue->resetCommandList(it, true);
        }
        // TODO: remove "if" when the problem is fixed in the level zero
        // runtime. Destroy only if a queue is healthy. Destroying a fence may
        // cause a hang otherwise.
        if (Queue->Healthy)
          ZE_CALL(zeFenceDestroy, (it->second.ZeFence));
      }
      Queue->CommandListMap.clear();
    }
  }
  PI_CALL(QueueRelease(Queue, nullptr));
  return PI_SUCCESS;
}

static pi_result QueueRelease(pi_queue Queue, pi_queue LockedQueue) {
  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  PI_ASSERT(Queue->RefCount, PI_INVALID_QUEUE);

  // We need to use a bool variable here to check the condition that
  // RefCount becomes zero atomically with PiQueueMutex lock.
  // Then, we can release the lock before we remove the Queue below.
  bool RefCountZero = false;
  {
    // Lock automatically releases when this goes out of scope.
    auto Lock = ((Queue == LockedQueue)
                     ? std::unique_lock<std::mutex>()
                     : std::unique_lock<std::mutex>(Queue->PiQueueMutex));

    Queue->RefCount--;
    if (Queue->RefCount == 0) {
      RefCountZero = true;

      if (Queue->OwnZeCommandQueue) {
        ZE_CALL(zeCommandQueueDestroy, (Queue->ZeComputeCommandQueue));
        for (uint32_t i = 0; i < Queue->ZeCopyCommandQueues.size(); ++i) {
          if (Queue->ZeCopyCommandQueues[i])
            ZE_CALL(zeCommandQueueDestroy, (Queue->ZeCopyCommandQueues[i]));
        }
      }

      Queue->ZeComputeCommandQueue = nullptr;
      for (uint32_t i = 0; i < Queue->ZeCopyCommandQueues.size(); ++i) {
        Queue->ZeCopyCommandQueues[i] = nullptr;
      }
      Queue->ZeCopyCommandQueues.clear();

      zePrint("piQueueRelease(compute) NumTimesClosedFull %d, "
              "NumTimesClosedEarly %d\n",
              Queue->ComputeCommandBatch.NumTimesClosedFull,
              Queue->ComputeCommandBatch.NumTimesClosedEarly);
      zePrint("piQueueRelease(copy) NumTimesClosedFull %d, NumTimesClosedEarly "
              "%d\n",
              Queue->CopyCommandBatch.NumTimesClosedFull,
              Queue->CopyCommandBatch.NumTimesClosedEarly);
    }
  }
  if (RefCountZero)
    delete Queue;

  return PI_SUCCESS;
}

pi_result piQueueFinish(pi_queue Queue) {
  // Wait until command lists attached to the command queue are executed.
  PI_ASSERT(Queue, PI_INVALID_QUEUE);

  std::vector<ze_command_queue_handle_t> ZeQueues;
  {
    // Lock automatically releases when this goes out of scope.
    std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

    // execute any command list that may still be open.
    if (auto Res = Queue->executeAllOpenCommandLists())
      return Res;

    ZeQueues = Queue->ZeCopyCommandQueues;
    ZeQueues.push_back(Queue->ZeComputeCommandQueue);
  }

  // Don't hold a lock to the queue's mutex while waiting.
  // This allows continue working with the queue from other threads.
  for (auto ZeQueue : ZeQueues) {
    if (ZeQueue)
      ZE_CALL(zeHostSynchronize, (ZeQueue));
  }

  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);
  // Prevent unneeded already finished events to show up in the wait list.
  Queue->LastCommandEvent = nullptr;

  return PI_SUCCESS;
}

// Flushing cross-queue dependencies is covered by createAndRetainPiZeEventList,
// so this can be left as a no-op.
pi_result piQueueFlush(pi_queue Queue) {
  (void)Queue;
  return PI_SUCCESS;
}

pi_result piextQueueGetNativeHandle(pi_queue Queue,
                                    pi_native_handle *NativeHandle) {
  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  PI_ASSERT(NativeHandle, PI_INVALID_VALUE);

  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

  auto ZeQueue = pi_cast<ze_command_queue_handle_t *>(NativeHandle);
  // Extract the Level Zero queue handle from the given PI queue
  *ZeQueue = Queue->ZeComputeCommandQueue; // TODO: Can we try to return copy
                                           // command queue here?
  return PI_SUCCESS;
}

pi_result piextQueueCreateWithNativeHandle(pi_native_handle NativeHandle,
                                           pi_context Context, pi_queue *Queue,
                                           bool OwnNativeHandle) {
  PI_ASSERT(Context, PI_INVALID_CONTEXT);
  PI_ASSERT(NativeHandle, PI_INVALID_VALUE);
  PI_ASSERT(Queue, PI_INVALID_QUEUE);

  auto ZeQueue = pi_cast<ze_command_queue_handle_t>(NativeHandle);

  // Attach the queue to the "0" device.
  // TODO: see if we need to let user choose the device.
  pi_device Device = Context->Devices[0];
  // TODO: see what we can do to correctly initialize PI queue for
  // compute vs. copy Level-Zero queue.
  std::vector<ze_command_queue_handle_t> ZeroCopyQueues;
  *Queue =
      new _pi_queue(ZeQueue, ZeroCopyQueues, Context, Device, OwnNativeHandle);
  return PI_SUCCESS;
}

// If indirect access tracking is enabled then performs reference counting,
// otherwise just calls zeMemAllocDevice.
static pi_result ZeDeviceMemAllocHelper(void **ResultPtr, pi_context Context,
                                        pi_device Device, size_t Size) {
  pi_platform Plt = Device->Platform;
  std::unique_lock<std::mutex> ContextsLock(Plt->ContextsMutex,
                                            std::defer_lock);
  if (IndirectAccessTrackingEnabled) {
    // Lock the mutex which is guarding contexts container in the platform.
    // This prevents new kernels from being submitted in any context while
    // we are in the process of allocating a memory, this is needed to
    // properly capture allocations by kernels with indirect access.
    ContextsLock.lock();
    // We are going to defer memory release if there are kernels with
    // indirect access, that is why explicitly retain context to be sure
    // that it is released after all memory allocations in this context are
    // released.
    PI_CALL(piContextRetain(Context));
  }

  ze_device_mem_alloc_desc_t ZeDesc = {};
  ZeDesc.flags = 0;
  ZeDesc.ordinal = 0;
  ZE_CALL(zeMemAllocDevice,
          (Context->ZeContext, &ZeDesc, Size, 1, Device->ZeDevice, ResultPtr));

  if (IndirectAccessTrackingEnabled) {
    // Keep track of all memory allocations in the context
    Context->MemAllocs.emplace(std::piecewise_construct,
                               std::forward_as_tuple(*ResultPtr),
                               std::forward_as_tuple(Context));
  }
  return PI_SUCCESS;
}

// If indirect access tracking is enabled then performs reference counting,
// otherwise just calls zeMemAllocHost.
static pi_result ZeHostMemAllocHelper(void **ResultPtr, pi_context Context,
                                      size_t Size) {
  pi_platform Plt = Context->Devices[0]->Platform;
  std::unique_lock<std::mutex> ContextsLock(Plt->ContextsMutex,
                                            std::defer_lock);
  if (IndirectAccessTrackingEnabled) {
    // Lock the mutex which is guarding contexts container in the platform.
    // This prevents new kernels from being submitted in any context while
    // we are in the process of allocating a memory, this is needed to
    // properly capture allocations by kernels with indirect access.
    ContextsLock.lock();
    // We are going to defer memory release if there are kernels with
    // indirect access, that is why explicitly retain context to be sure
    // that it is released after all memory allocations in this context are
    // released.
    PI_CALL(piContextRetain(Context));
  }

  ze_host_mem_alloc_desc_t ZeDesc = {};
  ZeDesc.flags = 0;
  ZE_CALL(zeMemAllocHost, (Context->ZeContext, &ZeDesc, Size, 1, ResultPtr));

  if (IndirectAccessTrackingEnabled) {
    // Keep track of all memory allocations in the context
    Context->MemAllocs.emplace(std::piecewise_construct,
                               std::forward_as_tuple(*ResultPtr),
                               std::forward_as_tuple(Context));
  }
  return PI_SUCCESS;
}

pi_result piMemBufferCreate(pi_context Context, pi_mem_flags Flags, size_t Size,
                            void *HostPtr, pi_mem *RetMem,
                            const pi_mem_properties *properties) {

  // TODO: implement support for more access modes
  if (!((Flags & PI_MEM_FLAGS_ACCESS_RW) ||
        (Flags & PI_MEM_ACCESS_READ_ONLY))) {
    die("piMemBufferCreate: Level-Zero supports read-write and read-only "
        "buffer,"
        "but not other accesses (such as write-only) yet.");
  }

  PI_ASSERT(Context, PI_INVALID_CONTEXT);
  PI_ASSERT(RetMem, PI_INVALID_VALUE);

  if (properties != nullptr) {
    die("piMemBufferCreate: no mem properties goes to Level-Zero RT yet");
  }

  void *Ptr = nullptr;

  // We treat integrated devices (physical memory shared with the CPU)
  // differently from discrete devices (those with distinct memories).
  // For integrated devices, allocating the buffer in host shared memory
  // enables automatic access from the device, and makes copying
  // unnecessary in the map/unmap operations. This improves performance.
  bool DeviceIsIntegrated = Context->Devices.size() == 1 &&
                            Context->Devices[0]->ZeDeviceProperties->flags &
                                ZE_DEVICE_PROPERTY_FLAG_INTEGRATED;

  if (Flags & PI_MEM_FLAGS_HOST_PTR_ALLOC) {
    // Having PI_MEM_FLAGS_HOST_PTR_ALLOC for buffer requires allocation of
    // pinned host memory, see:
    // sycl/doc/extensions/supported/SYCL_EXT_ONEAPI_USE_PINNED_HOST_MEMORY_PROPERTY.asciidoc
    // We are however missing such functionality in Level Zero, so we just
    // ignore the flag for now.
    //
  }

  // Choose an alignment that is at most 64 and is the next power of 2 for sizes
  // less than 64.
  auto Alignment = Size;
  if (Alignment > 32UL)
    Alignment = 64UL;
  else if (Alignment > 16UL)
    Alignment = 32UL;
  else if (Alignment > 8UL)
    Alignment = 16UL;
  else if (Alignment > 4UL)
    Alignment = 8UL;
  else if (Alignment > 2UL)
    Alignment = 4UL;
  else if (Alignment > 1UL)
    Alignment = 2UL;
  else
    Alignment = 1UL;

  // If USM Import feature is enabled and hostptr is supplied,
  // import the hostptr if not already imported into USM.
  // Data transfer rate is maximized when both source and destination
  // are USM pointers. Promotion of the host pointer to USM thus
  // optimizes data transfer performance.
  bool HostPtrImported = false;
  if (ZeUSMImport.Enabled && HostPtr != nullptr &&
      (Flags & PI_MEM_FLAGS_HOST_PTR_USE) != 0) {
    // Query memory type of the host pointer
    ze_device_handle_t ZeDeviceHandle;
    ZeStruct<ze_memory_allocation_properties_t> ZeMemoryAllocationProperties;
    ZE_CALL(zeMemGetAllocProperties,
            (Context->ZeContext, HostPtr, &ZeMemoryAllocationProperties,
             &ZeDeviceHandle));

    // If not shared of any type, we can import the ptr
    if (ZeMemoryAllocationProperties.type == ZE_MEMORY_TYPE_UNKNOWN) {
      // Promote the host ptr to USM host memory
      ze_driver_handle_t driverHandle = Context->Devices[0]->Platform->ZeDriver;
      ZeUSMImport.doZeUSMImport(driverHandle, HostPtr, Size);
      HostPtrImported = true;
    }
  }

  pi_result Result = PI_SUCCESS;
  if (DeviceIsIntegrated) {
    if (HostPtrImported) {
      // When HostPtr is imported we use it for the buffer.
      Ptr = HostPtr;
    } else {
      if (enableBufferPooling()) {
        PI_CALL(piextUSMHostAlloc(&Ptr, Context, nullptr, Size, Alignment));
      } else {
        Result = ZeHostMemAllocHelper(&Ptr, Context, Size);
      }
    }
  } else if (Context->SingleRootDevice) {
    // If we have a single discrete device or all devices in the context are
    // sub-devices of the same device then we can allocate on device
    if (enableBufferPooling()) {
      PI_CALL(piextUSMDeviceAlloc(&Ptr, Context, Context->SingleRootDevice,
                                  nullptr, Size, Alignment));
    } else {
      Result = ZeDeviceMemAllocHelper(&Ptr, Context, Context->SingleRootDevice,
                                      Size);
    }
  } else {
    // Context with several gpu cards. Temporarily use host allocation because
    // it is accessible by all devices. But it is not good in terms of
    // performance.
    // TODO: We need to either allow remote access to device memory using IPC,
    // or do explicit memory transfers from one device to another using host
    // resources as backing buffers to allow those transfers.
    if (HostPtrImported) {
      // When HostPtr is imported we use it for the buffer.
      Ptr = HostPtr;
    } else {
      if (enableBufferPooling()) {
        PI_CALL(piextUSMHostAlloc(&Ptr, Context, nullptr, Size, Alignment));
      } else {
        Result = ZeHostMemAllocHelper(&Ptr, Context, Size);
      }
    }
  }

  if (Result != PI_SUCCESS)
    return Result;

  if (HostPtr) {
    if ((Flags & PI_MEM_FLAGS_HOST_PTR_USE) != 0 ||
        (Flags & PI_MEM_FLAGS_HOST_PTR_COPY) != 0) {
      // Initialize the buffer with user data
      if (DeviceIsIntegrated) {
        // Do a host to host copy.
        // For an imported HostPtr the copy is unneeded.
        if (!HostPtrImported)
          memcpy(Ptr, HostPtr, Size);
      } else if (Context->SingleRootDevice) {
        // Initialize the buffer synchronously with immediate offload
        ZE_CALL(zeCommandListAppendMemoryCopy,
                (Context->ZeCommandListInit, Ptr, HostPtr, Size, nullptr, 0,
                 nullptr));
      } else {
        // Multiple root devices, do a host to host copy because we use a host
        // allocation for this case.
        // For an imported HostPtr the copy is unneeded.
        if (!HostPtrImported)
          memcpy(Ptr, HostPtr, Size);
      }
    } else if (Flags == 0 || (Flags == PI_MEM_FLAGS_ACCESS_RW)) {
      // Nothing more to do.
    } else {
      die("piMemBufferCreate: not implemented");
    }
  }

  auto HostPtrOrNull =
      (Flags & PI_MEM_FLAGS_HOST_PTR_USE) ? pi_cast<char *>(HostPtr) : nullptr;
  try {
    *RetMem = new _pi_buffer(
        Context, pi_cast<char *>(Ptr) /* Level Zero Memory Handle */,
        HostPtrOrNull, nullptr, 0, 0,
        DeviceIsIntegrated /* allocation in host memory */, HostPtrImported);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

pi_result piMemGetInfo(pi_mem Mem,
                       cl_mem_info ParamName, // TODO: untie from OpenCL
                       size_t ParamValueSize, void *ParamValue,
                       size_t *ParamValueSizeRet) {
  (void)Mem;
  (void)ParamName;
  (void)ParamValueSize;
  (void)ParamValue;
  (void)ParamValueSizeRet;
  die("piMemGetInfo: not implemented");
  return {};
}

pi_result piMemRetain(pi_mem Mem) {
  PI_ASSERT(Mem, PI_INVALID_MEM_OBJECT);

  ++(Mem->RefCount);
  return PI_SUCCESS;
}

// If indirect access tracking is not enabled then this functions just performs
// zeMemFree. If indirect access tracking is enabled then reference counting is
// performed.
static pi_result ZeMemFreeHelper(pi_context Context, void *Ptr) {
  pi_platform Plt = Context->Devices[0]->Platform;
  std::unique_lock<std::mutex> ContextsLock(Plt->ContextsMutex,
                                            std::defer_lock);
  if (IndirectAccessTrackingEnabled) {
    ContextsLock.lock();
    auto It = Context->MemAllocs.find(Ptr);
    if (It == std::end(Context->MemAllocs)) {
      die("All memory allocations must be tracked!");
    }
    if (--(It->second.RefCount) != 0) {
      // Memory can't be deallocated yet.
      return PI_SUCCESS;
    }

    // Reference count is zero, it is ok to free memory.
    // We don't need to track this allocation anymore.
    Context->MemAllocs.erase(It);
  }

  ZE_CALL(zeMemFree, (Context->ZeContext, Ptr));

  if (IndirectAccessTrackingEnabled)
    PI_CALL(ContextReleaseHelper(Context));

  return PI_SUCCESS;
}

pi_result piMemRelease(pi_mem Mem) {
  PI_ASSERT(Mem, PI_INVALID_MEM_OBJECT);

  if (--(Mem->RefCount) == 0) {
    if (Mem->isImage()) {
      ZE_CALL(zeImageDestroy, (pi_cast<ze_image_handle_t>(Mem->getZeHandle())));
    } else {
      auto Buf = static_cast<_pi_buffer *>(Mem);
      if (!Buf->isSubBuffer()) {
        if (Mem->HostPtrImported) {
          ze_driver_handle_t driverHandle =
              Mem->Context->Devices[0]->Platform->ZeDriver;
          ZeUSMImport.doZeUSMRelease(driverHandle, Mem->MapHostPtr);
        } else {
          if (enableBufferPooling()) {
            PI_CALL(piextUSMFree(Mem->Context, Mem->getZeHandle()));
          } else {
            if (auto Res = ZeMemFreeHelper(Mem->Context, Mem->getZeHandle()))
              return Res;
          }
        }
      }
    }
    delete Mem;
  }
  return PI_SUCCESS;
}

pi_result piMemImageCreate(pi_context Context, pi_mem_flags Flags,
                           const pi_image_format *ImageFormat,
                           const pi_image_desc *ImageDesc, void *HostPtr,
                           pi_mem *RetImage) {

  // TODO: implement read-only, write-only
  if ((Flags & PI_MEM_FLAGS_ACCESS_RW) == 0) {
    die("piMemImageCreate: Level-Zero implements only read-write buffer,"
        "no read-only or write-only yet.");
  }
  PI_ASSERT(Context, PI_INVALID_CONTEXT);
  PI_ASSERT(RetImage, PI_INVALID_VALUE);
  PI_ASSERT(ImageFormat, PI_INVALID_IMAGE_FORMAT_DESCRIPTOR);

  ze_image_format_type_t ZeImageFormatType;
  size_t ZeImageFormatTypeSize;
  switch (ImageFormat->image_channel_data_type) {
  case CL_FLOAT:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_FLOAT;
    ZeImageFormatTypeSize = 32;
    break;
  case CL_HALF_FLOAT:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_FLOAT;
    ZeImageFormatTypeSize = 16;
    break;
  case CL_UNSIGNED_INT32:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UINT;
    ZeImageFormatTypeSize = 32;
    break;
  case CL_UNSIGNED_INT16:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UINT;
    ZeImageFormatTypeSize = 16;
    break;
  case CL_UNSIGNED_INT8:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UINT;
    ZeImageFormatTypeSize = 8;
    break;
  case CL_UNORM_INT16:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UNORM;
    ZeImageFormatTypeSize = 16;
    break;
  case CL_UNORM_INT8:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_UNORM;
    ZeImageFormatTypeSize = 8;
    break;
  case CL_SIGNED_INT32:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SINT;
    ZeImageFormatTypeSize = 32;
    break;
  case CL_SIGNED_INT16:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SINT;
    ZeImageFormatTypeSize = 16;
    break;
  case CL_SIGNED_INT8:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SINT;
    ZeImageFormatTypeSize = 8;
    break;
  case CL_SNORM_INT16:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SNORM;
    ZeImageFormatTypeSize = 16;
    break;
  case CL_SNORM_INT8:
    ZeImageFormatType = ZE_IMAGE_FORMAT_TYPE_SNORM;
    ZeImageFormatTypeSize = 8;
    break;
  default:
    zePrint("piMemImageCreate: unsupported image data type: data type = %d\n",
            ImageFormat->image_channel_data_type);
    return PI_INVALID_VALUE;
  }

  // TODO: populate the layout mapping
  ze_image_format_layout_t ZeImageFormatLayout;
  switch (ImageFormat->image_channel_order) {
  case CL_RGBA:
    switch (ZeImageFormatTypeSize) {
    case 8:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8;
      break;
    case 16:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16;
      break;
    case 32:
      ZeImageFormatLayout = ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32;
      break;
    default:
      zePrint("piMemImageCreate: unexpected data type Size\n");
      return PI_INVALID_VALUE;
    }
    break;
  default:
    zePrint("format layout = %d\n", ImageFormat->image_channel_order);
    die("piMemImageCreate: unsupported image format layout\n");
    break;
  }

  ze_image_format_t ZeFormatDesc = {
      ZeImageFormatLayout, ZeImageFormatType,
      // TODO: are swizzles deducted from image_format->image_channel_order?
      ZE_IMAGE_FORMAT_SWIZZLE_R, ZE_IMAGE_FORMAT_SWIZZLE_G,
      ZE_IMAGE_FORMAT_SWIZZLE_B, ZE_IMAGE_FORMAT_SWIZZLE_A};

  ze_image_type_t ZeImageType;
  switch (ImageDesc->image_type) {
  case PI_MEM_TYPE_IMAGE1D:
    ZeImageType = ZE_IMAGE_TYPE_1D;
    break;
  case PI_MEM_TYPE_IMAGE2D:
    ZeImageType = ZE_IMAGE_TYPE_2D;
    break;
  case PI_MEM_TYPE_IMAGE3D:
    ZeImageType = ZE_IMAGE_TYPE_3D;
    break;
  case PI_MEM_TYPE_IMAGE1D_ARRAY:
    ZeImageType = ZE_IMAGE_TYPE_1DARRAY;
    break;
  case PI_MEM_TYPE_IMAGE2D_ARRAY:
    ZeImageType = ZE_IMAGE_TYPE_2DARRAY;
    break;
  default:
    zePrint("piMemImageCreate: unsupported image type\n");
    return PI_INVALID_VALUE;
  }

  ZeStruct<ze_image_desc_t> ZeImageDesc;
  ZeImageDesc.arraylevels = ZeImageDesc.flags = 0;
  ZeImageDesc.type = ZeImageType;
  ZeImageDesc.format = ZeFormatDesc;
  ZeImageDesc.width = pi_cast<uint32_t>(ImageDesc->image_width);
  ZeImageDesc.height = pi_cast<uint32_t>(ImageDesc->image_height);
  ZeImageDesc.depth = pi_cast<uint32_t>(ImageDesc->image_depth);
  ZeImageDesc.arraylevels = pi_cast<uint32_t>(ImageDesc->image_array_size);
  ZeImageDesc.miplevels = ImageDesc->num_mip_levels;

  // Currently we have the "0" device in context with mutliple root devices to
  // own the image.
  // TODO: Implement explicit copying for acessing the image from other devices
  // in the context.
  pi_device Device = Context->SingleRootDevice ? Context->SingleRootDevice
                                               : Context->Devices[0];
  ze_image_handle_t ZeHImage;
  ZE_CALL(zeImageCreate,
          (Context->ZeContext, Device->ZeDevice, &ZeImageDesc, &ZeHImage));

  auto HostPtrOrNull =
      (Flags & PI_MEM_FLAGS_HOST_PTR_USE) ? pi_cast<char *>(HostPtr) : nullptr;

  try {
    auto ZePIImage = new _pi_image(Context, ZeHImage, HostPtrOrNull);

#ifndef NDEBUG
    ZePIImage->ZeImageDesc = ZeImageDesc;
#endif // !NDEBUG

    if ((Flags & PI_MEM_FLAGS_HOST_PTR_USE) != 0 ||
        (Flags & PI_MEM_FLAGS_HOST_PTR_COPY) != 0) {
      // Initialize image synchronously with immediate offload
      ZE_CALL(zeCommandListAppendImageCopyFromMemory,
              (Context->ZeCommandListInit, ZeHImage, HostPtr, nullptr, nullptr,
               0, nullptr));
    }

    *RetImage = ZePIImage;
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  return PI_SUCCESS;
}

pi_result piextMemGetNativeHandle(pi_mem Mem, pi_native_handle *NativeHandle) {
  PI_ASSERT(Mem, PI_INVALID_MEM_OBJECT);
  *NativeHandle = pi_cast<pi_native_handle>(Mem->getZeHandle());
  return PI_SUCCESS;
}

pi_result piextMemCreateWithNativeHandle(pi_native_handle NativeHandle,
                                         pi_mem *Mem) {
  (void)NativeHandle;
  (void)Mem;
  die("piextMemCreateWithNativeHandle: not supported");
  return PI_SUCCESS;
}

pi_result piProgramCreate(pi_context Context, const void *ILBytes,
                          size_t Length, pi_program *Program) {

  PI_ASSERT(Context, PI_INVALID_CONTEXT);
  PI_ASSERT(ILBytes && Length, PI_INVALID_VALUE);
  PI_ASSERT(Program, PI_INVALID_PROGRAM);

  // NOTE: the Level Zero module creation is also building the program, so we
  // are deferring it until the program is ready to be built.

  try {
    *Program = new _pi_program(_pi_program::IL, Context, ILBytes, Length);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  return PI_SUCCESS;
}

pi_result piProgramCreateWithBinary(
    pi_context Context, pi_uint32 NumDevices, const pi_device *DeviceList,
    const size_t *Lengths, const unsigned char **Binaries,
    size_t NumMetadataEntries, const pi_device_binary_property *Metadata,
    pi_int32 *BinaryStatus, pi_program *Program) {
  (void)Metadata;
  (void)NumMetadataEntries;

  PI_ASSERT(Context, PI_INVALID_CONTEXT);
  PI_ASSERT(DeviceList && NumDevices, PI_INVALID_VALUE);
  PI_ASSERT(Binaries && Lengths, PI_INVALID_VALUE);
  PI_ASSERT(Program, PI_INVALID_PROGRAM);

  // For now we support only one device.
  if (NumDevices != 1) {
    zePrint("piProgramCreateWithBinary: level_zero supports only one device.");
    return PI_INVALID_VALUE;
  }
  if (!Binaries[0] || !Lengths[0]) {
    if (BinaryStatus)
      *BinaryStatus = PI_INVALID_VALUE;
    return PI_INVALID_VALUE;
  }

  size_t Length = Lengths[0];
  auto Binary = Binaries[0];

  // In OpenCL, clCreateProgramWithBinary() can be used to load any of the
  // following: "program executable", "compiled program", or "library of
  // compiled programs".  In addition, the loaded program can be either
  // IL (SPIR-v) or native device code.  For now, we assume that
  // piProgramCreateWithBinary() is only used to load a "program executable"
  // as native device code.
  // If we wanted to support all the same cases as OpenCL, we would need to
  // somehow examine the binary image to distinguish the cases.  Alternatively,
  // we could change the PI interface and have the caller pass additional
  // information to distinguish the cases.

  try {
    *Program = new _pi_program(_pi_program::Native, Context, Binary, Length);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  if (BinaryStatus)
    *BinaryStatus = PI_SUCCESS;
  return PI_SUCCESS;
}

pi_result piclProgramCreateWithSource(pi_context Context, pi_uint32 Count,
                                      const char **Strings,
                                      const size_t *Lengths,
                                      pi_program *RetProgram) {

  (void)Context;
  (void)Count;
  (void)Strings;
  (void)Lengths;
  (void)RetProgram;
  zePrint("piclProgramCreateWithSource: not supported in Level Zero\n");
  return PI_INVALID_OPERATION;
}

pi_result piProgramGetInfo(pi_program Program, pi_program_info ParamName,
                           size_t ParamValueSize, void *ParamValue,
                           size_t *ParamValueSizeRet) {

  PI_ASSERT(Program, PI_INVALID_PROGRAM);

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  switch (ParamName) {
  case PI_PROGRAM_INFO_REFERENCE_COUNT:
    return ReturnValue(pi_uint32{Program->RefCount});
  case PI_PROGRAM_INFO_NUM_DEVICES:
    // TODO: return true number of devices this program exists for.
    return ReturnValue(pi_uint32{1});
  case PI_PROGRAM_INFO_DEVICES:
    // TODO: return all devices this program exists for.
    return ReturnValue(Program->Context->Devices[0]);
  case PI_PROGRAM_INFO_BINARY_SIZES: {
    std::shared_lock Guard(Program->Mutex);
    size_t SzBinary;
    if (Program->State == _pi_program::IL ||
        Program->State == _pi_program::Native ||
        Program->State == _pi_program::Object) {
      SzBinary = Program->CodeLength;
    } else if (Program->State == _pi_program::Exe) {
      ZE_CALL(zeModuleGetNativeBinary, (Program->ZeModule, &SzBinary, nullptr));
    } else {
      return PI_INVALID_PROGRAM;
    }
    // This is an array of 1 element, initialized as if it were scalar.
    return ReturnValue(size_t{SzBinary});
  }
  case PI_PROGRAM_INFO_BINARIES: {
    // The caller sets "ParamValue" to an array of pointers, one for each
    // device.  Since Level Zero supports only one device, there is only one
    // pointer.  If the pointer is NULL, we don't do anything.  Otherwise, we
    // copy the program's binary image to the buffer at that pointer.
    uint8_t **PBinary = pi_cast<uint8_t **>(ParamValue);
    if (!PBinary[0])
      break;

    std::shared_lock Guard(Program->Mutex);
    if (Program->State == _pi_program::IL ||
        Program->State == _pi_program::Native ||
        Program->State == _pi_program::Object) {
      std::memcpy(PBinary[0], Program->Code.get(), Program->CodeLength);
    } else if (Program->State == _pi_program::Exe) {
      size_t SzBinary = 0;
      ZE_CALL(zeModuleGetNativeBinary,
              (Program->ZeModule, &SzBinary, PBinary[0]));
    } else {
      return PI_INVALID_PROGRAM;
    }
    break;
  }
  case PI_PROGRAM_INFO_NUM_KERNELS: {
    std::shared_lock Guard(Program->Mutex);
    uint32_t NumKernels;
    if (Program->State == _pi_program::IL ||
        Program->State == _pi_program::Native ||
        Program->State == _pi_program::Object) {
      return PI_INVALID_PROGRAM_EXECUTABLE;
    } else if (Program->State == _pi_program::Exe) {
      NumKernels = 0;
      ZE_CALL(zeModuleGetKernelNames,
              (Program->ZeModule, &NumKernels, nullptr));
    } else {
      return PI_INVALID_PROGRAM;
    }
    return ReturnValue(size_t{NumKernels});
  }
  case PI_PROGRAM_INFO_KERNEL_NAMES:
    try {
      std::shared_lock Guard(Program->Mutex);
      std::string PINames{""};
      if (Program->State == _pi_program::IL ||
          Program->State == _pi_program::Native ||
          Program->State == _pi_program::Object) {
        return PI_INVALID_PROGRAM_EXECUTABLE;
      } else if (Program->State == _pi_program::Exe) {
        uint32_t Count = 0;
        ZE_CALL(zeModuleGetKernelNames, (Program->ZeModule, &Count, nullptr));
        std::unique_ptr<const char *[]> PNames(new const char *[Count]);
        ZE_CALL(zeModuleGetKernelNames,
                (Program->ZeModule, &Count, PNames.get()));
        for (uint32_t I = 0; I < Count; ++I) {
          PINames += (I > 0 ? ";" : "");
          PINames += PNames[I];
        }
      } else {
        return PI_INVALID_PROGRAM;
      }
      return ReturnValue(PINames.c_str());
    } catch (const std::bad_alloc &) {
      return PI_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return PI_ERROR_UNKNOWN;
    }
  default:
    die("piProgramGetInfo: not implemented");
  }

  return PI_SUCCESS;
}

pi_result piProgramLink(pi_context Context, pi_uint32 NumDevices,
                        const pi_device *DeviceList, const char *Options,
                        pi_uint32 NumInputPrograms,
                        const pi_program *InputPrograms,
                        void (*PFnNotify)(pi_program Program, void *UserData),
                        void *UserData, pi_program *RetProgram) {
  // We only support one device with Level Zero currently.
  if (NumDevices != 1) {
    zePrint("piProgramLink: level_zero supports only one device.");
    return PI_INVALID_VALUE;
  }

  // We do not support any link flags at this time because the Level Zero API
  // does not have any way to pass flags that are specific to linking.
  if (Options && *Options != '\0') {
    std::string ErrorMessage(
        "Level Zero does not support kernel link flags: \"");
    ErrorMessage.append(Options);
    ErrorMessage.push_back('\"');
    pi_program Program =
        new _pi_program(_pi_program::Invalid, Context, ErrorMessage);
    *RetProgram = Program;
    return PI_LINK_PROGRAM_FAILURE;
  }

  // Validate input parameters.
  PI_ASSERT(DeviceList, PI_INVALID_DEVICE);
  {
    auto DeviceEntry =
        find(Context->Devices.begin(), Context->Devices.end(), DeviceList[0]);
    if (DeviceEntry == Context->Devices.end())
      return PI_INVALID_DEVICE;
  }
  PI_ASSERT(!PFnNotify && !UserData, PI_INVALID_VALUE);
  if (NumInputPrograms == 0 || InputPrograms == nullptr)
    return PI_INVALID_VALUE;

  pi_result PiResult = PI_SUCCESS;
  try {
    // Acquire a "shared" lock on each of the input programs, and also validate
    // that they are all in Object state.
    //
    // There is no danger of deadlock here even if two threads call
    // piProgramLink simultaneously with the same input programs in a different
    // order.  If we were acquiring these with "exclusive" access, this could
    // lead to a classic lock ordering deadlock.  However, there is no such
    // deadlock potential with "shared" access.  There could also be a deadlock
    // potential if there was some other code that holds more than one of these
    // locks simultaneously with "exclusive" access.  However, there is no such
    // code like that, so this is also not a danger.
    std::vector<std::shared_lock<std::shared_mutex>> Guards(NumInputPrograms);
    for (pi_uint32 I = 0; I < NumInputPrograms; I++) {
      std::shared_lock Guard(InputPrograms[I]->Mutex);
      Guards[I].swap(Guard);
      if (InputPrograms[I]->State != _pi_program::Object) {
        return PI_INVALID_OPERATION;
      }
    }

    // Previous calls to piProgramCompile did not actually compile the SPIR-V.
    // Instead, we postpone compilation until this point, when all the modules
    // are linked together.  By doing compilation and linking together, the JIT
    // compiler is able see all modules and do cross-module optimizations.
    //
    // Construct a ze_module_program_exp_desc_t which contains information about
    // all of the modules that will be linked together.
    ZeStruct<ze_module_program_exp_desc_t> ZeExtModuleDesc;
    std::vector<size_t> CodeSizes(NumInputPrograms);
    std::vector<const uint8_t *> CodeBufs(NumInputPrograms);
    std::vector<const char *> BuildFlagPtrs(NumInputPrograms);
    std::vector<const ze_module_constants_t *> SpecConstPtrs(NumInputPrograms);
    std::vector<_pi_program::SpecConstantShim> SpecConstShims;
    SpecConstShims.reserve(NumInputPrograms);

    for (pi_uint32 I = 0; I < NumInputPrograms; I++) {
      pi_program Program = InputPrograms[I];
      CodeSizes[I] = Program->CodeLength;
      CodeBufs[I] = Program->Code.get();
      BuildFlagPtrs[I] = Program->BuildFlags.c_str();
      SpecConstShims.emplace_back(Program);
      SpecConstPtrs[I] = SpecConstShims[I].ze();
    }

    ZeExtModuleDesc.count = NumInputPrograms;
    ZeExtModuleDesc.inputSizes = CodeSizes.data();
    ZeExtModuleDesc.pInputModules = CodeBufs.data();
    ZeExtModuleDesc.pBuildFlags = BuildFlagPtrs.data();
    ZeExtModuleDesc.pConstants = SpecConstPtrs.data();

    ZeStruct<ze_module_desc_t> ZeModuleDesc;
    ZeModuleDesc.pNext = &ZeExtModuleDesc;
    ZeModuleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;

    // This works around a bug in the Level Zero driver.  When "ZE_DEBUG=-1",
    // the driver does validation of the API calls, and it expects
    // "pInputModule" to be non-NULL and "inputSize" to be non-zero.  This
    // validation is wrong when using the "ze_module_program_exp_desc_t"
    // extension because those fields are supposed to be ignored.  As a
    // workaround, set both fields to 1.
    //
    // TODO: Remove this workaround when the driver is fixed.
    ZeModuleDesc.pInputModule = reinterpret_cast<const uint8_t *>(1);
    ZeModuleDesc.inputSize = 1;

    // We need a Level Zero extension to compile multiple programs together into
    // a single Level Zero module.  However, we don't need that extension if
    // there happens to be only one input program.
    //
    // The "|| (NumInputPrograms == 1)" term is a workaround for a bug in the
    // Level Zero driver.  The driver's "ze_module_program_exp_desc_t"
    // extension should work even in the case when there is just one input
    // module.  However, there is currently a bug in the driver that leads to a
    // crash.  As a workaround, do not use the extension when there is one
    // input module.
    //
    // TODO: Remove this workaround when the driver is fixed.
    if (!PiDriverModuleProgramExtensionFound || (NumInputPrograms == 1)) {
      if (NumInputPrograms == 1) {
        ZeModuleDesc.pNext = nullptr;
        ZeModuleDesc.inputSize = ZeExtModuleDesc.inputSizes[0];
        ZeModuleDesc.pInputModule = ZeExtModuleDesc.pInputModules[0];
        ZeModuleDesc.pBuildFlags = ZeExtModuleDesc.pBuildFlags[0];
        ZeModuleDesc.pConstants = ZeExtModuleDesc.pConstants[0];
      } else {
        zePrint("piProgramLink: level_zero driver does not have static linking "
                "support.");
        return PI_INVALID_VALUE;
      }
    }

    // Call the Level Zero API to compile, link, and create the module.
    ze_device_handle_t ZeDevice = DeviceList[0]->ZeDevice;
    ze_context_handle_t ZeContext = Context->ZeContext;
    ze_module_handle_t ZeModule = nullptr;
    ze_module_build_log_handle_t ZeBuildLog = nullptr;
    ze_result_t ZeResult =
        ZE_CALL_NOCHECK(zeModuleCreate, (ZeContext, ZeDevice, &ZeModuleDesc,
                                         &ZeModule, &ZeBuildLog));

    // We still create a _pi_program object even if there is a BUILD_FAILURE
    // because we need the object to hold the ZeBuildLog.  There is no build
    // log created for other errors, so we don't create an object.
    PiResult = mapError(ZeResult);
    if (ZeResult != ZE_RESULT_SUCCESS &&
        ZeResult != ZE_RESULT_ERROR_MODULE_BUILD_FAILURE) {
      return PiResult;
    }

    // The call to zeModuleCreate does not report an error if there are
    // unresolved symbols because it thinks these could be resolved later via a
    // call to zeModuleDynamicLink.  However, modules created with piProgramLink
    // are supposed to be fully linked and ready to use.  Therefore, do an extra
    // check now for unresolved symbols.  Note that we still create a
    // _pi_program if there are unresolved symbols because the ZeBuildLog tells
    // which symbols are unresolved.
    if (ZeResult == ZE_RESULT_SUCCESS) {
      ZeResult = checkUnresolvedSymbols(ZeModule, &ZeBuildLog);
      if (ZeResult == ZE_RESULT_ERROR_MODULE_LINK_FAILURE) {
        PiResult = PI_LINK_PROGRAM_FAILURE;
      } else if (ZeResult != ZE_RESULT_SUCCESS) {
        return mapError(ZeResult);
      }
    }

    _pi_program::state State =
        (PiResult == PI_SUCCESS) ? _pi_program::Exe : _pi_program::Invalid;
    *RetProgram = new _pi_program(State, Context, ZeModule, ZeBuildLog);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  return PiResult;
}

pi_result piProgramCompile(
    pi_program Program, pi_uint32 NumDevices, const pi_device *DeviceList,
    const char *Options, pi_uint32 NumInputHeaders,
    const pi_program *InputHeaders, const char **HeaderIncludeNames,
    void (*PFnNotify)(pi_program Program, void *UserData), void *UserData) {
  (void)NumInputHeaders;
  (void)InputHeaders;
  (void)HeaderIncludeNames;

  PI_ASSERT(Program, PI_INVALID_PROGRAM);

  if ((NumDevices && !DeviceList) || (!NumDevices && DeviceList))
    return PI_INVALID_VALUE;

  // These aren't supported.
  PI_ASSERT(!PFnNotify && !UserData, PI_INVALID_VALUE);

  std::scoped_lock Guard(Program->Mutex);

  // It's only valid to compile a program created from IL (we don't support
  // programs created from source code).
  //
  // The OpenCL spec says that the header parameters are ignored when compiling
  // IL programs, so we don't validate them.
  if (Program->State != _pi_program::IL)
    return PI_INVALID_OPERATION;

  // We don't compile anything now.  Instead, we delay compilation until
  // piProgramLink, where we do both compilation and linking as a single step.
  // This produces better code because the driver can do cross-module
  // optimizations.  Therefore, we just remember the compilation flags, so we
  // can use them later.
  if (Options)
    Program->BuildFlags = Options;
  Program->State = _pi_program::Object;

  return PI_SUCCESS;
}

pi_result piProgramBuild(pi_program Program, pi_uint32 NumDevices,
                         const pi_device *DeviceList, const char *Options,
                         void (*PFnNotify)(pi_program Program, void *UserData),
                         void *UserData) {

  PI_ASSERT(Program, PI_INVALID_PROGRAM);
  if ((NumDevices && !DeviceList) || (!NumDevices && DeviceList))
    return PI_INVALID_VALUE;

  // We only support build to one device with Level Zero now.
  // TODO: we should eventually build to the possibly multiple root
  // devices in the context.
  if (NumDevices != 1) {
    zePrint("piProgramBuild: level_zero supports only one device.");
    return PI_INVALID_VALUE;
  }

  // These aren't supported.
  PI_ASSERT(!PFnNotify && !UserData, PI_INVALID_VALUE);

  std::scoped_lock Guard(Program->Mutex);
  // Check if device belongs to associated context.
  PI_ASSERT(Program->Context, PI_INVALID_PROGRAM);
  {
    auto DeviceEntry = find(Program->Context->Devices.begin(),
                            Program->Context->Devices.end(), DeviceList[0]);
    if (DeviceEntry == Program->Context->Devices.end())
      return PI_INVALID_VALUE;
  }
  // It is legal to build a program created from either IL or from native
  // device code.
  if (Program->State != _pi_program::IL &&
      Program->State != _pi_program::Native)
    return PI_INVALID_OPERATION;

  // We should have either IL or native device code.
  PI_ASSERT(Program->Code, PI_INVALID_PROGRAM);

  // Ask Level Zero to build and load the native code onto the device.
  ZeStruct<ze_module_desc_t> ZeModuleDesc;
  _pi_program::SpecConstantShim Shim(Program);
  ZeModuleDesc.format = (Program->State == _pi_program::IL)
                            ? ZE_MODULE_FORMAT_IL_SPIRV
                            : ZE_MODULE_FORMAT_NATIVE;
  ZeModuleDesc.inputSize = Program->CodeLength;
  ZeModuleDesc.pInputModule = Program->Code.get();
  ZeModuleDesc.pBuildFlags = Options;
  ZeModuleDesc.pConstants = Shim.ze();

  ze_device_handle_t ZeDevice = DeviceList[0]->ZeDevice;
  ze_context_handle_t ZeContext = Program->Context->ZeContext;
  ze_module_handle_t ZeModule = nullptr;
  ZE_CALL(zeModuleCreate, (ZeContext, ZeDevice, &ZeModuleDesc, &ZeModule,
                           &Program->ZeBuildLog));

  // The call to zeModuleCreate does not report an error if there are
  // unresolved symbols because it thinks these could be resolved later via a
  // call to zeModuleDynamicLink.  However, modules created with piProgramBuild
  // are supposed to be fully linked and ready to use.  Therefore, do an extra
  // check now for unresolved symbols.
  ze_result_t ZeResult = checkUnresolvedSymbols(ZeModule, &Program->ZeBuildLog);
  if (ZeResult == ZE_RESULT_ERROR_MODULE_LINK_FAILURE) {
    return PI_BUILD_PROGRAM_FAILURE;
  } else if (ZeResult != ZE_RESULT_SUCCESS) {
    return mapError(ZeResult);
  }

  // We no longer need the IL / native code.
  Program->Code.reset();

  Program->ZeModule = ZeModule;
  Program->State = _pi_program::Exe;

  return PI_SUCCESS;
}

pi_result piProgramGetBuildInfo(pi_program Program, pi_device Device,
                                cl_program_build_info ParamName,
                                size_t ParamValueSize, void *ParamValue,
                                size_t *ParamValueSizeRet) {
  (void)Device;

  std::shared_lock Guard(Program->Mutex);
  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  if (ParamName == CL_PROGRAM_BINARY_TYPE) {
    cl_program_binary_type Type = CL_PROGRAM_BINARY_TYPE_NONE;
    if (Program->State == _pi_program::Object) {
      Type = CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT;
    } else if (Program->State == _pi_program::Exe) {
      Type = CL_PROGRAM_BINARY_TYPE_EXECUTABLE;
    }
    return ReturnValue(cl_program_binary_type{Type});
  }
  if (ParamName == CL_PROGRAM_BUILD_OPTIONS) {
    // TODO: how to get module build options out of Level Zero?
    // For the programs that we compiled we can remember the options
    // passed with piProgramCompile/piProgramBuild, but what can we
    // return for programs that were built outside and registered
    // with piProgramRegister?
    return ReturnValue("");
  } else if (ParamName == CL_PROGRAM_BUILD_LOG) {
    // Check first to see if the plugin code recorded an error message.
    if (!Program->ErrorMessage.empty()) {
      return ReturnValue(Program->ErrorMessage.c_str());
    }

    // Next check if there is a Level Zero build log.
    if (Program->ZeBuildLog) {
      size_t LogSize = ParamValueSize;
      ZE_CALL(zeModuleBuildLogGetString,
              (Program->ZeBuildLog, &LogSize, pi_cast<char *>(ParamValue)));
      if (ParamValueSizeRet) {
        *ParamValueSizeRet = LogSize;
      }
      return PI_SUCCESS;
    }

    // Otherwise, there is no error.  The OpenCL spec says to return an empty
    // string if there ws no previous attempt to compile, build, or link the
    // program.
    return ReturnValue("");
  } else {
    zePrint("piProgramGetBuildInfo: unsupported ParamName\n");
    return PI_INVALID_VALUE;
  }
  return PI_SUCCESS;
}

pi_result piProgramRetain(pi_program Program) {
  PI_ASSERT(Program, PI_INVALID_PROGRAM);
  ++(Program->RefCount);
  return PI_SUCCESS;
}

pi_result piProgramRelease(pi_program Program) {
  PI_ASSERT(Program, PI_INVALID_PROGRAM);
  // Check if the program is already released
  PI_ASSERT(Program->RefCount > 0, PI_INVALID_VALUE);
  if (--(Program->RefCount) == 0) {
    delete Program;
  }
  return PI_SUCCESS;
}

pi_result piextProgramGetNativeHandle(pi_program Program,
                                      pi_native_handle *NativeHandle) {
  PI_ASSERT(Program, PI_INVALID_PROGRAM);
  PI_ASSERT(NativeHandle, PI_INVALID_VALUE);

  auto ZeModule = pi_cast<ze_module_handle_t *>(NativeHandle);

  std::shared_lock Guard(Program->Mutex);
  switch (Program->State) {
  case _pi_program::Exe: {
    *ZeModule = Program->ZeModule;
    break;
  }

  default:
    return PI_INVALID_OPERATION;
  }

  return PI_SUCCESS;
}

pi_result piextProgramCreateWithNativeHandle(pi_native_handle NativeHandle,
                                             pi_context Context,
                                             bool ownNativeHandle,
                                             pi_program *Program) {
  PI_ASSERT(Program, PI_INVALID_PROGRAM);
  PI_ASSERT(NativeHandle, PI_INVALID_VALUE);
  PI_ASSERT(Context, PI_INVALID_CONTEXT);

  auto ZeModule = pi_cast<ze_module_handle_t>(NativeHandle);

  // We assume here that programs created from a native handle always
  // represent a fully linked executable (state Exe) and not an unlinked
  // executable (state Object).

  try {
    *Program =
        new _pi_program(_pi_program::Exe, Context, ZeModule, ownNativeHandle);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  return PI_SUCCESS;
}

_pi_program::~_pi_program() {
  // According to Level Zero Specification, all kernels and build logs
  // must be destroyed before the Module can be destroyed.  So, be sure
  // to destroy build log before destroying the module.
  if (ZeBuildLog) {
    ZE_CALL_NOCHECK(zeModuleBuildLogDestroy, (ZeBuildLog));
  }

  if (ZeModule && OwnZeModule) {
    ZE_CALL_NOCHECK(zeModuleDestroy, (ZeModule));
  }
}

// Check to see if a Level Zero module has any unresolved symbols.
//
// @param ZeModule    The module handle to check.
// @param ZeBuildLog  If there are unresolved symbols, this build log handle is
//                     modified to receive information telling which symbols
//                     are unresolved.
//
// @return ZE_RESULT_ERROR_MODULE_LINK_FAILURE indicates there are unresolved
//  symbols.  ZE_RESULT_SUCCESS indicates all symbols are resolved.  Any other
//  value indicates there was an error and we cannot tell if symbols are
//  resolved.
static ze_result_t
checkUnresolvedSymbols(ze_module_handle_t ZeModule,
                       ze_module_build_log_handle_t *ZeBuildLog) {

  // First check to see if the module has any imported symbols.  If there are
  // no imported symbols, it's not possible to have any unresolved symbols.  We
  // do this check first because we assume it's faster than the call to
  // zeModuleDynamicLink below.
  ZeStruct<ze_module_properties_t> ZeModuleProps;
  ze_result_t ZeResult =
      ZE_CALL_NOCHECK(zeModuleGetProperties, (ZeModule, &ZeModuleProps));
  if (ZeResult != ZE_RESULT_SUCCESS)
    return ZeResult;

  // If there are imported symbols, attempt to "link" the module with itself.
  // As a side effect, this will return the error
  // ZE_RESULT_ERROR_MODULE_LINK_FAILURE if there are any unresolved symbols.
  if (ZeModuleProps.flags & ZE_MODULE_PROPERTY_FLAG_IMPORTS) {
    return ZE_CALL_NOCHECK(zeModuleDynamicLink, (1, &ZeModule, ZeBuildLog));
  }
  return ZE_RESULT_SUCCESS;
}

pi_result piKernelCreate(pi_program Program, const char *KernelName,
                         pi_kernel *RetKernel) {

  PI_ASSERT(Program, PI_INVALID_PROGRAM);
  PI_ASSERT(RetKernel, PI_INVALID_VALUE);
  PI_ASSERT(KernelName, PI_INVALID_VALUE);

  std::shared_lock Guard(Program->Mutex);
  if (Program->State != _pi_program::Exe) {
    return PI_INVALID_PROGRAM_EXECUTABLE;
  }

  ZeStruct<ze_kernel_desc_t> ZeKernelDesc;
  ZeKernelDesc.flags = 0;
  ZeKernelDesc.pKernelName = KernelName;

  ze_kernel_handle_t ZeKernel;
  ZE_CALL(zeKernelCreate, (Program->ZeModule, &ZeKernelDesc, &ZeKernel));

  try {
    *RetKernel = new _pi_kernel(ZeKernel, true, Program);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  // Update the refcount of the program and context to show it's used by this
  // kernel.
  PI_CALL(piProgramRetain(Program));
  if (IndirectAccessTrackingEnabled)
    // TODO: do piContextRetain without the guard
    PI_CALL(piContextRetain(Program->Context));

  // Set up how to obtain kernel properties when needed.
  (*RetKernel)->ZeKernelProperties.Compute =
      [ZeKernel](ze_kernel_properties_t &Properties) {
        ZE_CALL_NOCHECK(zeKernelGetProperties, (ZeKernel, &Properties));
      };

  return PI_SUCCESS;
}

pi_result piKernelSetArg(pi_kernel Kernel, pi_uint32 ArgIndex, size_t ArgSize,
                         const void *ArgValue) {

  // OpenCL: "the arg_value pointer can be NULL or point to a NULL value
  // in which case a NULL value will be used as the value for the argument
  // declared as a pointer to global or constant memory in the kernel"
  //
  // We don't know the type of the argument but it seems that the only time
  // SYCL RT would send a pointer to NULL in 'arg_value' is when the argument
  // is a NULL pointer. Treat a pointer to NULL in 'arg_value' as a NULL.
  if (ArgSize == sizeof(void *) && ArgValue &&
      *(void **)(const_cast<void *>(ArgValue)) == nullptr) {
    ArgValue = nullptr;
  }

  PI_ASSERT(Kernel, PI_INVALID_KERNEL);

  ZE_CALL(zeKernelSetArgumentValue,
          (pi_cast<ze_kernel_handle_t>(Kernel->ZeKernel),
           pi_cast<uint32_t>(ArgIndex), pi_cast<size_t>(ArgSize),
           pi_cast<const void *>(ArgValue)));

  return PI_SUCCESS;
}

// Special version of piKernelSetArg to accept pi_mem.
pi_result piextKernelSetArgMemObj(pi_kernel Kernel, pi_uint32 ArgIndex,
                                  const pi_mem *ArgValue) {
  // TODO: the better way would probably be to add a new PI API for
  // extracting native PI object from PI handle, and have SYCL
  // RT pass that directly to the regular piKernelSetArg (and
  // then remove this piextKernelSetArgMemObj).

  PI_ASSERT(Kernel, PI_INVALID_KERNEL);

  ZE_CALL(zeKernelSetArgumentValue,
          (pi_cast<ze_kernel_handle_t>(Kernel->ZeKernel),
           pi_cast<uint32_t>(ArgIndex), sizeof(void *),
           (*ArgValue)->getZeHandlePtr()));

  return PI_SUCCESS;
}

// Special version of piKernelSetArg to accept pi_sampler.
pi_result piextKernelSetArgSampler(pi_kernel Kernel, pi_uint32 ArgIndex,
                                   const pi_sampler *ArgValue) {
  PI_ASSERT(Kernel, PI_INVALID_KERNEL);

  ZE_CALL(zeKernelSetArgumentValue,
          (pi_cast<ze_kernel_handle_t>(Kernel->ZeKernel),
           pi_cast<uint32_t>(ArgIndex), sizeof(void *),
           &(*ArgValue)->ZeSampler));

  return PI_SUCCESS;
}

pi_result piKernelGetInfo(pi_kernel Kernel, pi_kernel_info ParamName,
                          size_t ParamValueSize, void *ParamValue,
                          size_t *ParamValueSizeRet) {
  PI_ASSERT(Kernel, PI_INVALID_KERNEL);

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  switch (ParamName) {
  case PI_KERNEL_INFO_CONTEXT:
    return ReturnValue(pi_context{Kernel->Program->Context});
  case PI_KERNEL_INFO_PROGRAM:
    return ReturnValue(pi_program{Kernel->Program});
  case PI_KERNEL_INFO_FUNCTION_NAME:
    try {
      size_t Size = 0;
      ZE_CALL(zeKernelGetName, (Kernel->ZeKernel, &Size, nullptr));
      char *KernelName = new char[Size];
      ZE_CALL(zeKernelGetName, (Kernel->ZeKernel, &Size, KernelName));
      pi_result Res = ReturnValue(static_cast<const char *>(KernelName));
      delete[] KernelName;
      return Res;
    } catch (const std::bad_alloc &) {
      return PI_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return PI_ERROR_UNKNOWN;
    }
  case PI_KERNEL_INFO_NUM_ARGS:
    return ReturnValue(pi_uint32{Kernel->ZeKernelProperties->numKernelArgs});
  case PI_KERNEL_INFO_REFERENCE_COUNT:
    return ReturnValue(pi_uint32{Kernel->RefCount});
  case PI_KERNEL_INFO_ATTRIBUTES:
    try {
      uint32_t Size;
      ZE_CALL(zeKernelGetSourceAttributes, (Kernel->ZeKernel, &Size, nullptr));
      char *attributes = new char[Size];
      ZE_CALL(zeKernelGetSourceAttributes,
              (Kernel->ZeKernel, &Size, &attributes));
      auto Res = ReturnValue(attributes);
      delete[] attributes;
      return Res;
    } catch (const std::bad_alloc &) {
      return PI_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return PI_ERROR_UNKNOWN;
    }
  default:
    zePrint("Unsupported ParamName in piKernelGetInfo: ParamName=%d(0x%x)\n",
            ParamName, ParamName);
    return PI_INVALID_VALUE;
  }

  return PI_SUCCESS;
}

pi_result piKernelGetGroupInfo(pi_kernel Kernel, pi_device Device,
                               pi_kernel_group_info ParamName,
                               size_t ParamValueSize, void *ParamValue,
                               size_t *ParamValueSizeRet) {
  PI_ASSERT(Kernel, PI_INVALID_KERNEL);
  PI_ASSERT(Device, PI_INVALID_DEVICE);

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  switch (ParamName) {
  case PI_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE: {
    // TODO: To revisit after level_zero/issues/262 is resolved
    struct {
      size_t Arr[3];
    } WorkSize = {{Device->ZeDeviceComputeProperties->maxGroupSizeX,
                   Device->ZeDeviceComputeProperties->maxGroupSizeY,
                   Device->ZeDeviceComputeProperties->maxGroupSizeZ}};
    return ReturnValue(WorkSize);
  }
  case PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE: {
    uint32_t X, Y, Z;
    ZE_CALL(zeKernelSuggestGroupSize,
            (Kernel->ZeKernel, 10000, 10000, 10000, &X, &Y, &Z));
    return ReturnValue(size_t{X * Y * Z});
  }
  case PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE: {
    struct {
      size_t Arr[3];
    } WgSize = {{Kernel->ZeKernelProperties->requiredGroupSizeX,
                 Kernel->ZeKernelProperties->requiredGroupSizeY,
                 Kernel->ZeKernelProperties->requiredGroupSizeZ}};
    return ReturnValue(WgSize);
  }
  case PI_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE:
    return ReturnValue(pi_uint32{Kernel->ZeKernelProperties->localMemSize});
  case PI_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
    return ReturnValue(size_t{Device->ZeDeviceProperties->physicalEUSimdWidth});
  }
  case PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE:
    return ReturnValue(pi_uint32{Kernel->ZeKernelProperties->privateMemSize});
  case PI_KERNEL_GROUP_INFO_NUM_REGS: {
    die("PI_KERNEL_GROUP_INFO_NUM_REGS in piKernelGetGroupInfo not "
        "implemented\n");
    break;
  }
  default:
    zePrint("Unknown ParamName in piKernelGetGroupInfo: ParamName=%d(0x%x)\n",
            ParamName, ParamName);
    return PI_INVALID_VALUE;
  }
  return PI_SUCCESS;
}

pi_result piKernelGetSubGroupInfo(pi_kernel Kernel, pi_device Device,
                                  pi_kernel_sub_group_info ParamName,
                                  size_t InputValueSize, const void *InputValue,
                                  size_t ParamValueSize, void *ParamValue,
                                  size_t *ParamValueSizeRet) {
  (void)Device;
  (void)InputValueSize;
  (void)InputValue;

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  if (ParamName == PI_KERNEL_MAX_SUB_GROUP_SIZE) {
    ReturnValue(uint32_t{Kernel->ZeKernelProperties->maxSubgroupSize});
  } else if (ParamName == PI_KERNEL_MAX_NUM_SUB_GROUPS) {
    ReturnValue(uint32_t{Kernel->ZeKernelProperties->maxNumSubgroups});
  } else if (ParamName == PI_KERNEL_COMPILE_NUM_SUB_GROUPS) {
    ReturnValue(uint32_t{Kernel->ZeKernelProperties->requiredNumSubGroups});
  } else if (ParamName == PI_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL) {
    ReturnValue(uint32_t{Kernel->ZeKernelProperties->requiredSubgroupSize});
  } else {
    die("piKernelGetSubGroupInfo: parameter not implemented");
    return {};
  }
  return PI_SUCCESS;
}

pi_result piKernelRetain(pi_kernel Kernel) {

  PI_ASSERT(Kernel, PI_INVALID_KERNEL);

  ++(Kernel->RefCount);
  // When retaining a kernel, you are also retaining the program it is part of.
  PI_CALL(piProgramRetain(Kernel->Program));
  return PI_SUCCESS;
}

static pi_result USMFreeHelper(pi_context Context, void *Ptr);

pi_result piKernelRelease(pi_kernel Kernel) {

  PI_ASSERT(Kernel, PI_INVALID_KERNEL);

  if (IndirectAccessTrackingEnabled) {
    // piKernelRelease is called by Event->cleanup() as soon as kernel
    // execution has finished. This is the place where we need to release memory
    // allocations. If kernel is not in use (not submitted by some other thread)
    // then release referenced memory allocations. As a result, memory can be
    // deallocated and context can be removed from container in the platform.
    // That's why we need to lock a mutex here.
    pi_platform Plt = Kernel->Program->Context->Devices[0]->Platform;
    std::lock_guard<std::mutex> ContextsLock(Plt->ContextsMutex);

    if (--Kernel->SubmissionsCount == 0) {
      // Kernel is not submitted for execution, release referenced memory
      // allocations.
      for (auto &MemAlloc : Kernel->MemAllocs) {
        USMFreeHelper(MemAlloc->second.Context, MemAlloc->first);
      }
      Kernel->MemAllocs.clear();
    }
  }

  auto KernelProgram = Kernel->Program;
  if (--(Kernel->RefCount) == 0) {
    if (Kernel->OwnZeKernel)
      ZE_CALL(zeKernelDestroy, (Kernel->ZeKernel));
    if (IndirectAccessTrackingEnabled) {
      PI_CALL(piContextRelease(KernelProgram->Context));
    }
    delete Kernel;
  }

  // do a release on the program this kernel was part of
  PI_CALL(piProgramRelease(KernelProgram));

  return PI_SUCCESS;
}

pi_result
piEnqueueKernelLaunch(pi_queue Queue, pi_kernel Kernel, pi_uint32 WorkDim,
                      const size_t *GlobalWorkOffset,
                      const size_t *GlobalWorkSize, const size_t *LocalWorkSize,
                      pi_uint32 NumEventsInWaitList,
                      const pi_event *EventWaitList, pi_event *Event) {
  PI_ASSERT(Kernel, PI_INVALID_KERNEL);
  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  PI_ASSERT(Event, PI_INVALID_EVENT);
  PI_ASSERT((WorkDim > 0) && (WorkDim < 4), PI_INVALID_WORK_DIMENSION);

  if (GlobalWorkOffset != NULL) {
    if (!PiDriverGlobalOffsetExtensionFound) {
      zePrint("No global offset extension found on this driver\n");
      return PI_INVALID_VALUE;
    }

    ZE_CALL(zeKernelSetGlobalOffsetExp,
            (Kernel->ZeKernel, GlobalWorkOffset[0], GlobalWorkOffset[1],
             GlobalWorkOffset[2]));
  }

  ze_group_count_t ZeThreadGroupDimensions{1, 1, 1};
  uint32_t WG[3];

  // global_work_size of unused dimensions must be set to 1
  PI_ASSERT(WorkDim == 3 || GlobalWorkSize[2] == 1, PI_INVALID_VALUE);
  PI_ASSERT(WorkDim >= 2 || GlobalWorkSize[1] == 1, PI_INVALID_VALUE);

  if (LocalWorkSize) {
    WG[0] = pi_cast<uint32_t>(LocalWorkSize[0]);
    WG[1] = pi_cast<uint32_t>(LocalWorkSize[1]);
    WG[2] = pi_cast<uint32_t>(LocalWorkSize[2]);
  } else {
    ZE_CALL(zeKernelSuggestGroupSize,
            (Kernel->ZeKernel, GlobalWorkSize[0], GlobalWorkSize[1],
             GlobalWorkSize[2], &WG[0], &WG[1], &WG[2]));
  }

  // TODO: assert if sizes do not fit into 32-bit?
  switch (WorkDim) {
  case 3:
    ZeThreadGroupDimensions.groupCountX =
        pi_cast<uint32_t>(GlobalWorkSize[0] / WG[0]);
    ZeThreadGroupDimensions.groupCountY =
        pi_cast<uint32_t>(GlobalWorkSize[1] / WG[1]);
    ZeThreadGroupDimensions.groupCountZ =
        pi_cast<uint32_t>(GlobalWorkSize[2] / WG[2]);
    break;
  case 2:
    ZeThreadGroupDimensions.groupCountX =
        pi_cast<uint32_t>(GlobalWorkSize[0] / WG[0]);
    ZeThreadGroupDimensions.groupCountY =
        pi_cast<uint32_t>(GlobalWorkSize[1] / WG[1]);
    WG[2] = 1;
    break;
  case 1:
    ZeThreadGroupDimensions.groupCountX =
        pi_cast<uint32_t>(GlobalWorkSize[0] / WG[0]);
    WG[1] = WG[2] = 1;
    break;

  default:
    zePrint("piEnqueueKernelLaunch: unsupported work_dim\n");
    return PI_INVALID_VALUE;
  }

  // Error handling for non-uniform group size case
  if (GlobalWorkSize[0] != (ZeThreadGroupDimensions.groupCountX * WG[0])) {
    zePrint("piEnqueueKernelLaunch: invalid work_dim. The range is not a "
            "multiple of the group size in the 1st dimension\n");
    return PI_INVALID_WORK_GROUP_SIZE;
  }
  if (GlobalWorkSize[1] != (ZeThreadGroupDimensions.groupCountY * WG[1])) {
    zePrint("piEnqueueKernelLaunch: invalid work_dim. The range is not a "
            "multiple of the group size in the 2nd dimension\n");
    return PI_INVALID_WORK_GROUP_SIZE;
  }
  if (GlobalWorkSize[2] != (ZeThreadGroupDimensions.groupCountZ * WG[2])) {
    zePrint("piEnqueueKernelLaunch: invalid work_dim. The range is not a "
            "multiple of the group size in the 3rd dimension\n");
    return PI_INVALID_WORK_GROUP_SIZE;
  }

  ZE_CALL(zeKernelSetGroupSize, (Kernel->ZeKernel, WG[0], WG[1], WG[2]));

  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> QueueLock(Queue->PiQueueMutex);

  _pi_ze_event_list_t TmpWaitList;

  if (auto Res = TmpWaitList.createAndRetainPiZeEventList(NumEventsInWaitList,
                                                          EventWaitList, Queue))
    return Res;

  // Get a new command list to be used on this call
  pi_command_list_ptr_t CommandList{};
  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, CommandList, false /* UseCopyEngine */,
          true /* AllowBatching */))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  pi_result Res = createEventAndAssociateQueue(
      Queue, Event, PI_COMMAND_TYPE_NDRANGE_KERNEL, CommandList);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  // Save the kernel in the event, so that when the event is signalled
  // the code can do a piKernelRelease on this kernel.
  (*Event)->CommandData = (void *)Kernel;

  // Use piKernelRetain to increment the reference count and indicate
  // that the Kernel is in use. Once the event has been signalled, the
  // code in Event.cleanup() will do a piReleaseKernel to update
  // the reference count on the kernel, using the kernel saved
  // in CommandData.
  PI_CALL(piKernelRetain(Kernel));

  // Add the command to the command list
  ZE_CALL(zeCommandListAppendLaunchKernel,
          (CommandList->first, Kernel->ZeKernel, &ZeThreadGroupDimensions,
           ZeEvent, (*Event)->WaitList.Length, (*Event)->WaitList.ZeEventList));

  zePrint("calling zeCommandListAppendLaunchKernel() with"
          "  ZeEvent %#lx\n",
          pi_cast<std::uintptr_t>(ZeEvent));
  printZeEventList((*Event)->WaitList);

  if (IndirectAccessTrackingEnabled)
    Queue->KernelsToBeSubmitted.push_back(Kernel);

  // Execute command list asynchronously, as the event will be used
  // to track down its completion.
  if (auto Res = Queue->executeCommandList(CommandList, false, true))
    return Res;

  return PI_SUCCESS;
}

pi_result piextKernelCreateWithNativeHandle(pi_native_handle NativeHandle,
                                            pi_context Context,
                                            pi_program Program,
                                            bool OwnNativeHandle,
                                            pi_kernel *Kernel) {
  PI_ASSERT(Context, PI_INVALID_CONTEXT);
  PI_ASSERT(Program, PI_INVALID_PROGRAM);
  PI_ASSERT(NativeHandle, PI_INVALID_VALUE);
  PI_ASSERT(Kernel, PI_INVALID_KERNEL);

  auto ZeKernel = pi_cast<ze_kernel_handle_t>(NativeHandle);
  *Kernel = new _pi_kernel(ZeKernel, OwnNativeHandle, Program);

  // Update the refcount of the program and context to show it's used by this
  // kernel.
  PI_CALL(piProgramRetain(Program));
  if (IndirectAccessTrackingEnabled)
    // TODO: do piContextRetain without the guard
    PI_CALL(piContextRetain(Program->Context));

  // Set up how to obtain kernel properties when needed.
  (*Kernel)->ZeKernelProperties.Compute =
      [ZeKernel](ze_kernel_properties_t &Properties) {
        ZE_CALL_NOCHECK(zeKernelGetProperties, (ZeKernel, &Properties));
      };

  return PI_SUCCESS;
}

pi_result piextKernelGetNativeHandle(pi_kernel Kernel,
                                     pi_native_handle *NativeHandle) {
  PI_ASSERT(Kernel, PI_INVALID_KERNEL);
  PI_ASSERT(NativeHandle, PI_INVALID_VALUE);

  auto *ZeKernel = pi_cast<ze_kernel_handle_t *>(NativeHandle);
  *ZeKernel = Kernel->ZeKernel;
  return PI_SUCCESS;
}

//
// Events
//
pi_result
_pi_event::getOrCreateHostVisibleEvent(ze_event_handle_t &ZeHostVisibleEvent) {

  if (!HostVisibleEvent) {
    if (EventsScope != OnDemandHostVisibleProxy)
      die("getOrCreateHostVisibleEvent: missing host-visible event");

    // Create a "proxy" host-visible event on demand.
    PI_CALL(EventCreate(Context, true, &HostVisibleEvent));
    HostVisibleEvent->CleanedUp = true;

    // Submit the command(s) signalling the proxy event to the queue.
    // We have to first submit a wait for the device-only event for which this
    // proxy is created.
    //
    // Get a new command list to be used on this call
    {
      std::lock_guard<std::mutex> Lock(Queue->PiQueueMutex);

      // We want to batch these commands to avoid extra submissions (costly)
      bool OkToBatch = true;

      pi_command_list_ptr_t CommandList{};
      if (auto Res = Queue->Context->getAvailableCommandList(
              Queue, CommandList, false /* UseCopyEngine */, OkToBatch))
        return Res;

      ZE_CALL(zeCommandListAppendWaitOnEvents,
              (CommandList->first, 1, &ZeEvent));
      ZE_CALL(zeCommandListAppendSignalEvent,
              (CommandList->first, HostVisibleEvent->ZeEvent));

      if (auto Res = Queue->executeCommandList(CommandList, false, OkToBatch))
        return Res;
    }
  }

  ZeHostVisibleEvent = HostVisibleEvent->ZeEvent;
  return PI_SUCCESS;
}

static pi_result EventCreate(pi_context Context, bool HostVisible,
                             pi_event *RetEvent) {
  size_t Index = 0;
  ze_event_pool_handle_t ZeEventPool = {};
  if (auto Res = Context->getFreeSlotInExistingOrNewPool(ZeEventPool, Index,
                                                         HostVisible))
    return Res;

  ze_event_handle_t ZeEvent;
  ZeStruct<ze_event_desc_t> ZeEventDesc;
  ZeEventDesc.index = Index;
  ZeEventDesc.wait = 0;

  if (HostVisible) {
    ZeEventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  } else {
    //
    // Set the scope to "device" for every event. This is sufficient for global
    // device access and peer device access. If needed to be seen on the host
    // we are doing special handling, see EventsScope options.
    //
    // TODO: see if "sub-device" (ZE_EVENT_SCOPE_FLAG_SUBDEVICE) can better be
    //       used in some circumstances.
    //
    ZeEventDesc.signal = 0;
  }

  ZE_CALL(zeEventCreate, (ZeEventPool, &ZeEventDesc, &ZeEvent));

  try {
    PI_ASSERT(RetEvent, PI_INVALID_VALUE);

    *RetEvent = new _pi_event(ZeEvent, ZeEventPool, Context,
                              PI_COMMAND_TYPE_USER, true);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  if (HostVisible)
    (*RetEvent)->HostVisibleEvent = *RetEvent;

  return PI_SUCCESS;
}

pi_result piEventCreate(pi_context Context, pi_event *RetEvent) {
  return EventCreate(Context, EventsScope == AllHostVisible, RetEvent);
}

pi_result piEventGetInfo(pi_event Event, pi_event_info ParamName,
                         size_t ParamValueSize, void *ParamValue,
                         size_t *ParamValueSizeRet) {

  PI_ASSERT(Event, PI_INVALID_EVENT);

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  switch (ParamName) {
  case PI_EVENT_INFO_COMMAND_QUEUE:
    return ReturnValue(pi_queue{Event->Queue});
  case PI_EVENT_INFO_CONTEXT:
    return ReturnValue(pi_context{Event->Context});
  case PI_EVENT_INFO_COMMAND_TYPE:
    return ReturnValue(pi_cast<pi_uint64>(Event->CommandType));
  case PI_EVENT_INFO_COMMAND_EXECUTION_STATUS: {
    // Check to see if the event's Queue has an open command list due to
    // batching. If so, go ahead and close and submit it, because it is
    // possible that this is trying to query some event's status that
    // is part of the batch.  This isn't strictly required, but it seems
    // like a reasonable thing to do.
    if (Event->Queue) {
      // Lock automatically releases when this goes out of scope.
      std::lock_guard<std::mutex> lock(Event->Queue->PiQueueMutex);
      Event->Queue->executeOpenCommandListWithEvent(Event);
    }

    // Make sure that we query a host-visible event only.
    // If one wasn't yet created then don't create it here as well, and
    // just conservatively return that event is not yet completed.
    auto HostVisibleEvent = Event->HostVisibleEvent;
    if (HostVisibleEvent) {
      ze_result_t ZeResult;
      ZeResult =
          ZE_CALL_NOCHECK(zeEventQueryStatus, (HostVisibleEvent->ZeEvent));
      if (ZeResult == ZE_RESULT_SUCCESS) {
        return getInfo(ParamValueSize, ParamValue, ParamValueSizeRet,
                       pi_int32{CL_COMPLETE}); // Untie from OpenCL
      }
    }

    // TODO: We don't know if the status is queued, submitted or running.
    //       For now return "running", as others are unlikely to be of
    //       interest.
    return getInfo(ParamValueSize, ParamValue, ParamValueSizeRet,
                   pi_int32{CL_RUNNING});
  }
  case PI_EVENT_INFO_REFERENCE_COUNT:
    return ReturnValue(pi_uint32{Event->RefCount});
  default:
    zePrint("Unsupported ParamName in piEventGetInfo: ParamName=%d(%x)\n",
            ParamName, ParamName);
    return PI_INVALID_VALUE;
  }

  return PI_SUCCESS;
}

pi_result piEventGetProfilingInfo(pi_event Event, pi_profiling_info ParamName,
                                  size_t ParamValueSize, void *ParamValue,
                                  size_t *ParamValueSizeRet) {

  PI_ASSERT(Event, PI_INVALID_EVENT);

  uint64_t ZeTimerResolution =
      Event->Queue
          ? Event->Queue->Device->ZeDeviceProperties->timerResolution
          : Event->Context->Devices[0]->ZeDeviceProperties->timerResolution;
  // Get timestamp frequency
  const double ZeTimerFreq = 1E09 / ZeTimerResolution;

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  ze_kernel_timestamp_result_t tsResult;

  switch (ParamName) {
  case PI_PROFILING_INFO_COMMAND_START: {
    ZE_CALL(zeEventQueryKernelTimestamp, (Event->ZeEvent, &tsResult));
    uint64_t ContextStartTime = tsResult.context.kernelStart * ZeTimerFreq;
    return ReturnValue(ContextStartTime);
  }
  case PI_PROFILING_INFO_COMMAND_END: {
    ZE_CALL(zeEventQueryKernelTimestamp, (Event->ZeEvent, &tsResult));

    uint64_t ContextStartTime = tsResult.context.kernelStart;
    uint64_t ContextEndTime = tsResult.context.kernelEnd;
    //
    // Handle a possible wrap-around (the underlying HW counter is < 64-bit).
    // Note, it will not report correct time if there were multiple wrap
    // arounds, and the longer term plan is to enlarge the capacity of the
    // HW timestamps.
    //
    if (ContextEndTime <= ContextStartTime) {
      pi_device Device = Event->Context->Devices[0];
      const uint64_t TimestampMaxValue =
          (1LL << Device->ZeDeviceProperties->kernelTimestampValidBits) - 1;
      ContextEndTime += TimestampMaxValue - ContextStartTime;
    }
    ContextEndTime *= ZeTimerFreq;
    return ReturnValue(ContextEndTime);
  }
  case PI_PROFILING_INFO_COMMAND_QUEUED:
  case PI_PROFILING_INFO_COMMAND_SUBMIT:
    // TODO: Support these when Level Zero supported is added.
    return ReturnValue(uint64_t{0});
  default:
    zePrint("piEventGetProfilingInfo: not supported ParamName\n");
    return PI_INVALID_VALUE;
  }

  return PI_SUCCESS;
}

// Perform any necessary cleanup after an event has been signalled.
// This currently recycles the associate command list, and also makes
// sure to release any kernel that may have been used by the event.
pi_result _pi_event::cleanup(pi_queue LockedQueue) {
  // The implementation of this is slightly tricky.  The same event
  // can be referred to by multiple threads, so it is possible to
  // have a race condition between the read of fields of the event,
  // and reseting those fields in some other thread.
  // But, since the event is uniquely associated with the queue
  // for the event, we use the locking that we already have to do on the
  // queue to also serve as the thread safety mechanism for the
  // any of the Event's data members that need to be read/reset as
  // part of the cleanup operations.
  if (Queue) {
    // Lock automatically releases when this goes out of scope.
    auto Lock = ((Queue == LockedQueue)
                     ? std::unique_lock<std::mutex>()
                     : std::unique_lock<std::mutex>(Queue->PiQueueMutex));

    if (ZeCommandList) {
      // Event has been signalled: If the fence for the associated command list
      // is signalled, then reset the fence and command list and add them to the
      // available list for reuse in PI calls.
      if (Queue->RefCount > 0) {
        auto it = Queue->CommandListMap.find(ZeCommandList);
        if (it == Queue->CommandListMap.end()) {
          die("Missing command-list completition fence");
        }

        // It is possible that the fence was already noted as signalled and
        // reset.  In that case the InUse flag will be false, and
        // we shouldn't query it, synchronize on it, or try to reset it.
        if (it->second.InUse) {
          // Workaround for VM_BIND mode.
          // Make sure that the command-list doing memcpy is reset before
          // non-USM host memory potentially involved in the memcpy is freed.
          //
          // NOTE: it is valid to wait for the fence here as long as we aren't
          // doing batching on the involved command-list. Today memcpy goes by
          // itself in a command list.
          //
          // TODO: this will unnecessarily(?) wait for non-USM memory buffers
          // too, so we might need to add a new command type to differentiate.
          //
          ze_result_t ZeResult =
              (CommandType == PI_COMMAND_TYPE_MEM_BUFFER_COPY)
                  ? ZE_CALL_NOCHECK(zeHostSynchronize, (it->second.ZeFence))
                  : ZE_CALL_NOCHECK(zeFenceQueryStatus, (it->second.ZeFence));

          if (ZeResult == ZE_RESULT_SUCCESS) {
            Queue->resetCommandList(it, true);
            ZeCommandList = nullptr;
          }
        }
      }
    }

    // Release the kernel associated with this event if there is one.
    if (CommandType == PI_COMMAND_TYPE_NDRANGE_KERNEL && CommandData) {
      PI_CALL(piKernelRelease(pi_cast<pi_kernel>(CommandData)));
      CommandData = nullptr;
    }

    // If this event was the LastCommandEvent in the queue, being used
    // to make sure that commands were executed in-order, remove this.
    // If we don't do this, the event can get released and freed leaving
    // a dangling pointer to this event.  It could also cause unneeded
    // already finished events to show up in the wait list.
    if (Queue->LastCommandEvent == this) {
      Queue->LastCommandEvent = nullptr;
    }
  }

  if (!CleanedUp) {
    CleanedUp = true;
    // Release this event since we explicitly retained it on creation.
    // NOTE: that this needs to be done only once for an event so
    // this is guarded with the CleanedUp flag.
    //
    PI_CALL(EventRelease(this, LockedQueue));
  }

  // Make a list of all the dependent events that must have signalled
  // because this event was dependent on them.  This list will be appended
  // to as we walk it so that this algorithm doesn't go recursive
  // due to dependent events themselves being dependent on other events
  // forming a potentially very deep tree, and deep recursion.  That
  // turned out to be a significant problem with the recursive code
  // that preceded this implementation.

  std::list<pi_event> EventsToBeReleased;

  WaitList.collectEventsForReleaseAndDestroyPiZeEventList(EventsToBeReleased);

  while (!EventsToBeReleased.empty()) {
    pi_event DepEvent = EventsToBeReleased.front();
    EventsToBeReleased.pop_front();

    DepEvent->WaitList.collectEventsForReleaseAndDestroyPiZeEventList(
        EventsToBeReleased);
    if (IndirectAccessTrackingEnabled && DepEvent->Queue) {
      // DepEvent has finished, we can release the associated kernel if there is
      // one. This is the earliest place we can do this and it can't be done
      // twice, so it is safe. Lock automatically releases when this goes out of
      // scope.
      // TODO: this code needs to be moved out of the guard.
      auto Lock =
          ((DepEvent->Queue == LockedQueue)
               ? std::unique_lock<std::mutex>()
               : std::unique_lock<std::mutex>(DepEvent->Queue->PiQueueMutex));

      if (DepEvent->CommandType == PI_COMMAND_TYPE_NDRANGE_KERNEL &&
          DepEvent->CommandData) {
        PI_CALL(piKernelRelease(pi_cast<pi_kernel>(DepEvent->CommandData)));
        DepEvent->CommandData = nullptr;
      }
    }
    PI_CALL(EventRelease(DepEvent, LockedQueue));
  }

  return PI_SUCCESS;
}

pi_result piEventsWait(pi_uint32 NumEvents, const pi_event *EventList) {

  if (NumEvents && !EventList) {
    return PI_INVALID_EVENT;
  }
  if (EventsScope == OnDemandHostVisibleProxy) {
    // Make sure to add all host-visible "proxy" event signals if needed.
    // This ensures that all signalling commands are submitted below and
    // thus proxy events can be waited without a deadlock.
    //
    for (uint32_t I = 0; I < NumEvents; I++) {
      ze_event_handle_t ZeHostVisibleEvent;
      if (auto Res =
              EventList[I]->getOrCreateHostVisibleEvent(ZeHostVisibleEvent))
        return Res;
    }
  }
  // Submit dependent open command lists for execution, if any
  for (uint32_t I = 0; I < NumEvents; I++) {
    auto Queue = EventList[I]->Queue;
    if (Queue) {
      // Lock automatically releases when this goes out of scope.
      std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

      if (Queue->RefCount > 0) {
        if (auto Res = Queue->executeAllOpenCommandLists())
          return Res;
      }
    }
  }
  for (uint32_t I = 0; I < NumEvents; I++) {
    auto HostVisibleEvent = EventList[I]->HostVisibleEvent;
    if (!HostVisibleEvent)
      die("The host-visible proxy event missing");

    ze_event_handle_t ZeEvent = HostVisibleEvent->ZeEvent;
    zePrint("ZeEvent = %#lx\n", pi_cast<std::uintptr_t>(ZeEvent));
    ZE_CALL(zeHostSynchronize, (ZeEvent));

    // NOTE: we are cleaning up after the event here to free resources
    // sooner in case run-time is not calling piEventRelease soon enough.
    EventList[I]->cleanup();
  }
  return PI_SUCCESS;
}

pi_result piEventSetCallback(pi_event Event, pi_int32 CommandExecCallbackType,
                             void (*PFnNotify)(pi_event Event,
                                               pi_int32 EventCommandStatus,
                                               void *UserData),
                             void *UserData) {
  (void)Event;
  (void)CommandExecCallbackType;
  (void)PFnNotify;
  (void)UserData;
  die("piEventSetCallback: deprecated, to be removed");
  return PI_SUCCESS;
}

pi_result piEventSetStatus(pi_event Event, pi_int32 ExecutionStatus) {
  (void)Event;
  (void)ExecutionStatus;
  die("piEventSetStatus: deprecated, to be removed");
  return PI_SUCCESS;
}

pi_result piEventRetain(pi_event Event) {
  ++(Event->RefCount);
  return PI_SUCCESS;
}

pi_result piEventRelease(pi_event Event) {
  return EventRelease(Event, nullptr);
}

static pi_result EventRelease(pi_event Event, pi_queue LockedQueue) {
  PI_ASSERT(Event, PI_INVALID_EVENT);
  if (!Event->RefCount) {
    die("piEventRelease: called on a destroyed event");
  }

  if (--(Event->RefCount) == 0) {
    if (!Event->CleanedUp)
      Event->cleanup(LockedQueue);

    if (Event->CommandType == PI_COMMAND_TYPE_MEM_BUFFER_UNMAP &&
        Event->CommandData) {
      // Free the memory allocated in the piEnqueueMemBufferMap.
      if (auto Res = ZeMemFreeHelper(Event->Context, Event->CommandData))
        return Res;
      Event->CommandData = nullptr;
    }
    if (Event->OwnZeEvent) {
      ZE_CALL(zeEventDestroy, (Event->ZeEvent));
    }
    // It is possible that host-visible event was never created.
    // In case it was check if that's different from this same event
    // and release a reference to it.
    if (Event->HostVisibleEvent && Event->HostVisibleEvent != Event) {
      // Decrement ref-count of the host-visible proxy event.
      PI_CALL(piEventRelease(Event->HostVisibleEvent));
    }

    auto Context = Event->Context;
    if (auto Res = Context->decrementUnreleasedEventsInPool(Event))
      return Res;

    // We intentionally incremented the reference counter when an event is
    // created so that we can avoid pi_queue is released before the associated
    // pi_event is released. Here we have to decrement it so pi_queue
    // can be released successfully.
    if (Event->Queue) {
      PI_CALL(QueueRelease(Event->Queue, LockedQueue));
    }
    delete Event;
  }
  return PI_SUCCESS;
}

pi_result piextEventGetNativeHandle(pi_event Event,
                                    pi_native_handle *NativeHandle) {
  PI_ASSERT(Event, PI_INVALID_EVENT);
  PI_ASSERT(NativeHandle, PI_INVALID_VALUE);

  auto *ZeEvent = pi_cast<ze_event_handle_t *>(NativeHandle);
  *ZeEvent = Event->ZeEvent;

  // Event can potentially be in an open command-list, make sure that
  // it is submitted for execution to avoid potential deadlock if
  // interop app is going to wait for it.
  if (Event->Queue) {
    std::lock_guard<std::mutex> lock(Event->Queue->PiQueueMutex);
    Event->Queue->executeOpenCommandListWithEvent(Event);
  }
  return PI_SUCCESS;
}

pi_result piextEventCreateWithNativeHandle(pi_native_handle NativeHandle,
                                           pi_context Context,
                                           bool OwnNativeHandle,
                                           pi_event *Event) {
  PI_ASSERT(Context, PI_INVALID_CONTEXT);
  PI_ASSERT(Event, PI_INVALID_EVENT);
  PI_ASSERT(NativeHandle, PI_INVALID_VALUE);

  auto ZeEvent = pi_cast<ze_event_handle_t>(NativeHandle);
  *Event = new _pi_event(ZeEvent, nullptr /* ZeEventPool */, Context,
                         PI_COMMAND_TYPE_USER, OwnNativeHandle);

  // Assume native event is host-visible, or otherwise we'd
  // need to create a host-visible proxy for it.
  (*Event)->HostVisibleEvent = *Event;

  return PI_SUCCESS;
}

//
// Sampler
//
pi_result piSamplerCreate(pi_context Context,
                          const pi_sampler_properties *SamplerProperties,
                          pi_sampler *RetSampler) {

  PI_ASSERT(Context, PI_INVALID_CONTEXT);
  PI_ASSERT(RetSampler, PI_INVALID_VALUE);

  // Have the "0" device in context to own the sampler. Rely on Level-Zero
  // drivers to perform migration as necessary for sharing it across multiple
  // devices in the context.
  //
  // TODO: figure out if we instead need explicit copying for acessing
  // the sampler from other devices in the context.
  //
  pi_device Device = Context->Devices[0];

  ze_sampler_handle_t ZeSampler;
  ZeStruct<ze_sampler_desc_t> ZeSamplerDesc;

  // Set the default values for the ZeSamplerDesc.
  ZeSamplerDesc.isNormalized = PI_TRUE;
  ZeSamplerDesc.addressMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP;
  ZeSamplerDesc.filterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;

  // Update the values of the ZeSamplerDesc from the pi_sampler_properties list.
  // Default values will be used if any of the following is true:
  //   a) SamplerProperties list is NULL
  //   b) SamplerProperties list is missing any properties

  if (SamplerProperties) {
    const pi_sampler_properties *CurProperty = SamplerProperties;

    while (*CurProperty != 0) {
      switch (*CurProperty) {
      case PI_SAMPLER_PROPERTIES_NORMALIZED_COORDS: {
        pi_bool CurValueBool = pi_cast<pi_bool>(*(++CurProperty));

        if (CurValueBool == PI_TRUE)
          ZeSamplerDesc.isNormalized = PI_TRUE;
        else if (CurValueBool == PI_FALSE)
          ZeSamplerDesc.isNormalized = PI_FALSE;
        else {
          zePrint("piSamplerCreate: unsupported "
                  "PI_SAMPLER_NORMALIZED_COORDS value\n");
          return PI_INVALID_VALUE;
        }
      } break;

      case PI_SAMPLER_PROPERTIES_ADDRESSING_MODE: {
        pi_sampler_addressing_mode CurValueAddressingMode =
            pi_cast<pi_sampler_addressing_mode>(
                pi_cast<pi_uint32>(*(++CurProperty)));

        // Level Zero runtime with API version 1.2 and lower has a bug:
        // ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER is implemented as "clamp to
        // edge" and ZE_SAMPLER_ADDRESS_MODE_CLAMP is implemented as "clamp to
        // border", i.e. logic is flipped. Starting from API version 1.3 this
        // problem is going to be fixed. That's why check for API version to set
        // an address mode.
        ze_api_version_t ZeApiVersion =
            Context->Devices[0]->Platform->ZeApiVersion;
        // TODO: add support for PI_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE
        switch (CurValueAddressingMode) {
        case PI_SAMPLER_ADDRESSING_MODE_NONE:
          ZeSamplerDesc.addressMode = ZE_SAMPLER_ADDRESS_MODE_NONE;
          break;
        case PI_SAMPLER_ADDRESSING_MODE_REPEAT:
          ZeSamplerDesc.addressMode = ZE_SAMPLER_ADDRESS_MODE_REPEAT;
          break;
        case PI_SAMPLER_ADDRESSING_MODE_CLAMP:
          ZeSamplerDesc.addressMode =
              ZeApiVersion < ZE_MAKE_VERSION(1, 3)
                  ? ZE_SAMPLER_ADDRESS_MODE_CLAMP
                  : ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
          break;
        case PI_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE:
          ZeSamplerDesc.addressMode =
              ZeApiVersion < ZE_MAKE_VERSION(1, 3)
                  ? ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER
                  : ZE_SAMPLER_ADDRESS_MODE_CLAMP;
          break;
        case PI_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT:
          ZeSamplerDesc.addressMode = ZE_SAMPLER_ADDRESS_MODE_MIRROR;
          break;
        default:
          zePrint("piSamplerCreate: unsupported PI_SAMPLER_ADDRESSING_MODE "
                  "value\n");
          zePrint("PI_SAMPLER_ADDRESSING_MODE=%d\n", CurValueAddressingMode);
          return PI_INVALID_VALUE;
        }
      } break;

      case PI_SAMPLER_PROPERTIES_FILTER_MODE: {
        pi_sampler_filter_mode CurValueFilterMode =
            pi_cast<pi_sampler_filter_mode>(
                pi_cast<pi_uint32>(*(++CurProperty)));

        if (CurValueFilterMode == PI_SAMPLER_FILTER_MODE_NEAREST)
          ZeSamplerDesc.filterMode = ZE_SAMPLER_FILTER_MODE_NEAREST;
        else if (CurValueFilterMode == PI_SAMPLER_FILTER_MODE_LINEAR)
          ZeSamplerDesc.filterMode = ZE_SAMPLER_FILTER_MODE_LINEAR;
        else {
          zePrint("PI_SAMPLER_FILTER_MODE=%d\n", CurValueFilterMode);
          zePrint(
              "piSamplerCreate: unsupported PI_SAMPLER_FILTER_MODE value\n");
          return PI_INVALID_VALUE;
        }
      } break;

      default:
        break;
      }
      CurProperty++;
    }
  }

  ZE_CALL(zeSamplerCreate, (Context->ZeContext, Device->ZeDevice,
                            &ZeSamplerDesc, // TODO: translate properties
                            &ZeSampler));

  try {
    *RetSampler = new _pi_sampler(ZeSampler);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  return PI_SUCCESS;
}

pi_result piSamplerGetInfo(pi_sampler Sampler, pi_sampler_info ParamName,
                           size_t ParamValueSize, void *ParamValue,
                           size_t *ParamValueSizeRet) {
  (void)Sampler;
  (void)ParamName;
  (void)ParamValueSize;
  (void)ParamValue;
  (void)ParamValueSizeRet;

  die("piSamplerGetInfo: not implemented");
  return {};
}

pi_result piSamplerRetain(pi_sampler Sampler) {
  PI_ASSERT(Sampler, PI_INVALID_SAMPLER);

  ++(Sampler->RefCount);
  return PI_SUCCESS;
}

pi_result piSamplerRelease(pi_sampler Sampler) {
  PI_ASSERT(Sampler, PI_INVALID_SAMPLER);

  if (--(Sampler->RefCount) == 0) {
    ZE_CALL(zeSamplerDestroy, (Sampler->ZeSampler));
    delete Sampler;
  }
  return PI_SUCCESS;
}

//
// Queue Commands
//
pi_result piEnqueueEventsWait(pi_queue Queue, pi_uint32 NumEventsInWaitList,
                              const pi_event *EventWaitList, pi_event *Event) {

  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  PI_ASSERT(Event, PI_INVALID_EVENT);

  if (EventWaitList) {
    PI_ASSERT(NumEventsInWaitList > 0, PI_INVALID_VALUE);

    // Lock automatically releases when this goes out of scope.
    std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

    _pi_ze_event_list_t TmpWaitList = {};
    if (auto Res = TmpWaitList.createAndRetainPiZeEventList(
            NumEventsInWaitList, EventWaitList, Queue))
      return Res;

    // Get a new command list to be used on this call
    pi_command_list_ptr_t CommandList{};
    if (auto Res = Queue->Context->getAvailableCommandList(Queue, CommandList))
      return Res;

    ze_event_handle_t ZeEvent = nullptr;
    auto Res = createEventAndAssociateQueue(Queue, Event, PI_COMMAND_TYPE_USER,
                                            CommandList);
    if (Res != PI_SUCCESS)
      return Res;
    ZeEvent = (*Event)->ZeEvent;
    (*Event)->WaitList = TmpWaitList;

    const auto &WaitList = (*Event)->WaitList;
    auto ZeCommandList = CommandList->first;
    ZE_CALL(zeCommandListAppendWaitOnEvents,
            (ZeCommandList, WaitList.Length, WaitList.ZeEventList));

    ZE_CALL(zeCommandListAppendSignalEvent, (ZeCommandList, ZeEvent));

    // Execute command list asynchronously as the event will be used
    // to track down its completion.
    return Queue->executeCommandList(CommandList);
  }

  // If wait-list is empty, then this particular command should wait until
  // all previous enqueued commands to the command-queue have completed.
  //
  // TODO: find a way to do that without blocking the host.

  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

  auto Res = createEventAndAssociateQueue(Queue, Event, PI_COMMAND_TYPE_USER,
                                          Queue->CommandListMap.end());
  if (Res != PI_SUCCESS)
    return Res;

  ZE_CALL(zeHostSynchronize, (Queue->ZeComputeCommandQueue));
  for (uint32_t i = 0; i < Queue->ZeCopyCommandQueues.size(); ++i) {
    if (Queue->ZeCopyCommandQueues[i])
      ZE_CALL(zeHostSynchronize, (Queue->ZeCopyCommandQueues[i]));
  }

  Queue->LastCommandEvent = *Event;

  ZE_CALL(zeEventHostSignal, ((*Event)->ZeEvent));
  return PI_SUCCESS;
}

pi_result piEnqueueEventsWaitWithBarrier(pi_queue Queue,
                                         pi_uint32 NumEventsInWaitList,
                                         const pi_event *EventWaitList,
                                         pi_event *Event) {
  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  PI_ASSERT(Event, PI_INVALID_EVENT);

  // Submit dependent open command lists for execution, if any
  // Only do it for queues other than the current, since the barrier
  // will go into current queue submission together with the waited event.
  for (uint32_t I = 0; I < NumEventsInWaitList; I++) {
    auto EventQueue = EventWaitList[I]->Queue;
    if (EventQueue && EventQueue != Queue) {
      // Lock automatically releases when this goes out of scope.
      std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

      if (EventQueue->RefCount > 0) {
        if (auto Res = EventQueue->executeAllOpenCommandLists())
          return Res;
      }
    }
  }

  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

  _pi_ze_event_list_t TmpWaitList;
  if (auto Res = TmpWaitList.createAndRetainPiZeEventList(NumEventsInWaitList,
                                                          EventWaitList, Queue))
    return Res;

  // Get a new command list to be used on this call
  bool OkToBatch = true;
  pi_command_list_ptr_t CommandList{};
  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, CommandList, false /*copy*/, OkToBatch))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  auto Res = createEventAndAssociateQueue(Queue, Event, PI_COMMAND_TYPE_USER,
                                          CommandList);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  ZE_CALL(zeCommandListAppendBarrier,
          (CommandList->first, ZeEvent, (*Event)->WaitList.Length,
           (*Event)->WaitList.ZeEventList));

  // Execute command list asynchronously as the event will be used
  // to track down its completion.
  return Queue->executeCommandList(CommandList, false, OkToBatch);
}

pi_result piEnqueueMemBufferRead(pi_queue Queue, pi_mem Src,
                                 pi_bool BlockingRead, size_t Offset,
                                 size_t Size, void *Dst,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {
  PI_ASSERT(Src, PI_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_INVALID_QUEUE);

  return enqueueMemCopyHelper(PI_COMMAND_TYPE_MEM_BUFFER_READ, Queue, Dst,
                              BlockingRead, Size,
                              pi_cast<char *>(Src->getZeHandle()) + Offset,
                              NumEventsInWaitList, EventWaitList, Event);
}

pi_result piEnqueueMemBufferReadRect(
    pi_queue Queue, pi_mem Buffer, pi_bool BlockingRead,
    pi_buff_rect_offset BufferOffset, pi_buff_rect_offset HostOffset,
    pi_buff_rect_region Region, size_t BufferRowPitch, size_t BufferSlicePitch,
    size_t HostRowPitch, size_t HostSlicePitch, void *Ptr,
    pi_uint32 NumEventsInWaitList, const pi_event *EventWaitList,
    pi_event *Event) {

  PI_ASSERT(Buffer, PI_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_INVALID_QUEUE);

  return enqueueMemCopyRectHelper(
      PI_COMMAND_TYPE_MEM_BUFFER_READ_RECT, Queue, Buffer->getZeHandle(),
      static_cast<char *>(Ptr), BufferOffset, HostOffset, Region,
      BufferRowPitch, HostRowPitch, BufferSlicePitch, HostSlicePitch,
      BlockingRead, NumEventsInWaitList, EventWaitList, Event);
}

} // extern "C"

bool _pi_queue::useCopyEngine(bool PreferCopyEngine) const {
  return (!isInOrderQueue() || UseCopyEngineForInOrderQueue) &&
         PreferCopyEngine && Device->hasCopyEngine();
}

// Shared by all memory read/write/copy PI interfaces.
// PI interfaces must not have queue's mutex locked on entry.
static pi_result enqueueMemCopyHelper(pi_command_type CommandType,
                                      pi_queue Queue, void *Dst,
                                      pi_bool BlockingWrite, size_t Size,
                                      const void *Src,
                                      pi_uint32 NumEventsInWaitList,
                                      const pi_event *EventWaitList,
                                      pi_event *Event, bool PreferCopyEngine) {
  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  PI_ASSERT(Event, PI_INVALID_EVENT);
  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

  _pi_ze_event_list_t TmpWaitList;
  if (auto Res = TmpWaitList.createAndRetainPiZeEventList(NumEventsInWaitList,
                                                          EventWaitList, Queue))
    return Res;

  // Get a new command list to be used on this call
  pi_command_list_ptr_t CommandList{};
  bool UseCopyEngine = Queue->useCopyEngine(PreferCopyEngine);

  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;

  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, CommandList, UseCopyEngine, OkToBatch))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  auto Res =
      createEventAndAssociateQueue(Queue, Event, CommandType, CommandList);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;
  if (WaitList.Length) {

    ZE_CALL(zeCommandListAppendWaitOnEvents,
            (ZeCommandList, WaitList.Length, WaitList.ZeEventList));
  }

  ZE_CALL(zeCommandListAppendMemoryCopy,
          (ZeCommandList, Dst, Src, Size, ZeEvent, 0, nullptr));

  zePrint("calling zeCommandListAppendMemoryCopy() with\n"
          "  ZeEvent %#lx\n",
          pi_cast<std::uintptr_t>(ZeEvent));
  printZeEventList(WaitList);

  if (auto Res =
          Queue->executeCommandList(CommandList, BlockingWrite, OkToBatch))
    return Res;

  return PI_SUCCESS;
}

// Shared by all memory read/write/copy rect PI interfaces.
// PI interfaces must not have queue's mutex locked on entry.
static pi_result enqueueMemCopyRectHelper(
    pi_command_type CommandType, pi_queue Queue, void *SrcBuffer,
    void *DstBuffer, pi_buff_rect_offset SrcOrigin,
    pi_buff_rect_offset DstOrigin, pi_buff_rect_region Region,
    size_t SrcRowPitch, size_t DstRowPitch, size_t SrcSlicePitch,
    size_t DstSlicePitch, pi_bool Blocking, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *Event, bool PreferCopyEngine) {

  PI_ASSERT(Region && SrcOrigin && DstOrigin && Queue, PI_INVALID_VALUE);
  PI_ASSERT(Event, PI_INVALID_EVENT);

  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

  _pi_ze_event_list_t TmpWaitList;
  if (auto Res = TmpWaitList.createAndRetainPiZeEventList(NumEventsInWaitList,
                                                          EventWaitList, Queue))
    return Res;

  // Get a new command list to be used on this call
  pi_command_list_ptr_t CommandList{};
  bool UseCopyEngine = Queue->useCopyEngine(PreferCopyEngine);
  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;
  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, CommandList, UseCopyEngine, OkToBatch))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  auto Res =
      createEventAndAssociateQueue(Queue, Event, CommandType, CommandList);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  if (WaitList.Length) {
    ZE_CALL(zeCommandListAppendWaitOnEvents,
            (ZeCommandList, WaitList.Length, WaitList.ZeEventList));
  }
  zePrint("calling zeCommandListAppendMemoryCopy() with\n"
          "  ZeEvent %#lx\n",
          pi_cast<std::uintptr_t>(ZeEvent));
  printZeEventList(WaitList);

  uint32_t SrcOriginX = pi_cast<uint32_t>(SrcOrigin->x_bytes);
  uint32_t SrcOriginY = pi_cast<uint32_t>(SrcOrigin->y_scalar);
  uint32_t SrcOriginZ = pi_cast<uint32_t>(SrcOrigin->z_scalar);

  uint32_t SrcPitch = SrcRowPitch;
  if (SrcPitch == 0)
    SrcPitch = pi_cast<uint32_t>(Region->width_bytes);

  if (SrcSlicePitch == 0)
    SrcSlicePitch = pi_cast<uint32_t>(Region->height_scalar) * SrcPitch;

  uint32_t DstOriginX = pi_cast<uint32_t>(DstOrigin->x_bytes);
  uint32_t DstOriginY = pi_cast<uint32_t>(DstOrigin->y_scalar);
  uint32_t DstOriginZ = pi_cast<uint32_t>(DstOrigin->z_scalar);

  uint32_t DstPitch = DstRowPitch;
  if (DstPitch == 0)
    DstPitch = pi_cast<uint32_t>(Region->width_bytes);

  if (DstSlicePitch == 0)
    DstSlicePitch = pi_cast<uint32_t>(Region->height_scalar) * DstPitch;

  uint32_t Width = pi_cast<uint32_t>(Region->width_bytes);
  uint32_t Height = pi_cast<uint32_t>(Region->height_scalar);
  uint32_t Depth = pi_cast<uint32_t>(Region->depth_scalar);

  const ze_copy_region_t ZeSrcRegion = {SrcOriginX, SrcOriginY, SrcOriginZ,
                                        Width,      Height,     Depth};
  const ze_copy_region_t ZeDstRegion = {DstOriginX, DstOriginY, DstOriginZ,
                                        Width,      Height,     Depth};

  ZE_CALL(zeCommandListAppendMemoryCopyRegion,
          (ZeCommandList, DstBuffer, &ZeDstRegion, DstPitch, DstSlicePitch,
           SrcBuffer, &ZeSrcRegion, SrcPitch, SrcSlicePitch, nullptr, 0,
           nullptr));

  zePrint("calling zeCommandListAppendMemoryCopyRegion()\n");

  ZE_CALL(zeCommandListAppendBarrier, (ZeCommandList, ZeEvent, 0, nullptr));

  zePrint("calling zeCommandListAppendBarrier() with Event %#lx\n",
          pi_cast<std::uintptr_t>(ZeEvent));

  if (auto Res = Queue->executeCommandList(CommandList, Blocking, OkToBatch))
    return Res;

  return PI_SUCCESS;
}

extern "C" {

pi_result piEnqueueMemBufferWrite(pi_queue Queue, pi_mem Buffer,
                                  pi_bool BlockingWrite, size_t Offset,
                                  size_t Size, const void *Ptr,
                                  pi_uint32 NumEventsInWaitList,
                                  const pi_event *EventWaitList,
                                  pi_event *Event) {

  PI_ASSERT(Buffer, PI_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_INVALID_QUEUE);

  return enqueueMemCopyHelper(PI_COMMAND_TYPE_MEM_BUFFER_WRITE, Queue,
                              pi_cast<char *>(Buffer->getZeHandle()) +
                                  Offset, // dst
                              BlockingWrite, Size,
                              Ptr, // src
                              NumEventsInWaitList, EventWaitList, Event);
}

pi_result piEnqueueMemBufferWriteRect(
    pi_queue Queue, pi_mem Buffer, pi_bool BlockingWrite,
    pi_buff_rect_offset BufferOffset, pi_buff_rect_offset HostOffset,
    pi_buff_rect_region Region, size_t BufferRowPitch, size_t BufferSlicePitch,
    size_t HostRowPitch, size_t HostSlicePitch, const void *Ptr,
    pi_uint32 NumEventsInWaitList, const pi_event *EventWaitList,
    pi_event *Event) {

  PI_ASSERT(Buffer, PI_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_INVALID_QUEUE);

  return enqueueMemCopyRectHelper(
      PI_COMMAND_TYPE_MEM_BUFFER_WRITE_RECT, Queue,
      const_cast<char *>(static_cast<const char *>(Ptr)), Buffer->getZeHandle(),
      HostOffset, BufferOffset, Region, HostRowPitch, BufferRowPitch,
      HostSlicePitch, BufferSlicePitch, BlockingWrite, NumEventsInWaitList,
      EventWaitList, Event);
}

pi_result piEnqueueMemBufferCopy(pi_queue Queue, pi_mem SrcBuffer,
                                 pi_mem DstBuffer, size_t SrcOffset,
                                 size_t DstOffset, size_t Size,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {
  PI_ASSERT(SrcBuffer && DstBuffer, PI_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  // Copy engine is preferred only for host to device transfer.
  // Device to device transfers run faster on compute engines.
  bool PreferCopyEngine = (SrcBuffer->OnHost || DstBuffer->OnHost);

  // Temporary option added to use copy engine for D2D copy
  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  return enqueueMemCopyHelper(
      PI_COMMAND_TYPE_MEM_BUFFER_COPY, Queue,
      pi_cast<char *>(DstBuffer->getZeHandle()) + DstOffset,
      false, // blocking
      Size, pi_cast<char *>(SrcBuffer->getZeHandle()) + SrcOffset,
      NumEventsInWaitList, EventWaitList, Event, PreferCopyEngine);
}

pi_result piEnqueueMemBufferCopyRect(
    pi_queue Queue, pi_mem SrcBuffer, pi_mem DstBuffer,
    pi_buff_rect_offset SrcOrigin, pi_buff_rect_offset DstOrigin,
    pi_buff_rect_region Region, size_t SrcRowPitch, size_t SrcSlicePitch,
    size_t DstRowPitch, size_t DstSlicePitch, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *Event) {
  PI_ASSERT(SrcBuffer && DstBuffer, PI_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  // Copy engine is preferred only for host to device transfer.
  // Device to device transfers run faster on compute engines.
  bool PreferCopyEngine = (SrcBuffer->OnHost || DstBuffer->OnHost);
  return enqueueMemCopyRectHelper(
      PI_COMMAND_TYPE_MEM_BUFFER_COPY_RECT, Queue, SrcBuffer->getZeHandle(),
      DstBuffer->getZeHandle(), SrcOrigin, DstOrigin, Region, SrcRowPitch,
      DstRowPitch, SrcSlicePitch, DstSlicePitch,
      false, // blocking
      NumEventsInWaitList, EventWaitList, Event, PreferCopyEngine);
}

} // extern "C"

//
// Caller of this must assure that the Queue is non-null and has not
// acquired the lock.
static pi_result
enqueueMemFillHelper(pi_command_type CommandType, pi_queue Queue, void *Ptr,
                     const void *Pattern, size_t PatternSize, size_t Size,
                     pi_uint32 NumEventsInWaitList,
                     const pi_event *EventWaitList, pi_event *Event) {
  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  PI_ASSERT(Event, PI_INVALID_EVENT);

  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

  _pi_ze_event_list_t TmpWaitList;
  if (auto Res = TmpWaitList.createAndRetainPiZeEventList(NumEventsInWaitList,
                                                          EventWaitList, Queue))
    return Res;

  bool PreferCopyEngine = true;
  size_t MaxPatternSize =
      Queue->Device->ZeComputeQueueGroupProperties.maxMemoryFillPatternSize;

  // Performance analysis on a simple SYCL data "fill" test shows copy engine
  // is faster than compute engine for such operations.
  //
  // Make sure that pattern size matches the capability of the copy queue.
  //
  if (PreferCopyEngine && Queue->Device->hasCopyEngine() &&
      PatternSize <= Queue->Device->ZeMainCopyQueueGroupProperties
                         .maxMemoryFillPatternSize) {
    MaxPatternSize =
        Queue->Device->ZeMainCopyQueueGroupProperties.maxMemoryFillPatternSize;
  } else {
    // pattern size does not fit within capability of copy queue.
    // Try compute queue instead.
    PreferCopyEngine = false;
  }
  // Pattern size must fit the queue.
  PI_ASSERT(PatternSize <= MaxPatternSize, PI_INVALID_VALUE);
  // Pattern size must be a power of two.
  PI_ASSERT((PatternSize > 0) && ((PatternSize & (PatternSize - 1)) == 0),
            PI_INVALID_VALUE);

  pi_command_list_ptr_t CommandList{};
  bool UseCopyEngine = Queue->useCopyEngine(PreferCopyEngine);
  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;
  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, CommandList, UseCopyEngine, OkToBatch))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  auto Res =
      createEventAndAssociateQueue(Queue, Event, CommandType, CommandList);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  if (WaitList.Length) {
    ZE_CALL(zeCommandListAppendWaitOnEvents,
            (ZeCommandList, WaitList.Length, WaitList.ZeEventList));
  }

  ZE_CALL(
      zeCommandListAppendMemoryFill,
      (ZeCommandList, Ptr, Pattern, PatternSize, Size, ZeEvent, 0, nullptr));

  zePrint("calling zeCommandListAppendMemoryFill() with\n"
          "  ZeEvent %#lx\n",
          pi_cast<pi_uint64>(ZeEvent));
  printZeEventList(WaitList);

  // Execute command list asynchronously, as the event will be used
  // to track down its completion.
  if (auto Res = Queue->executeCommandList(CommandList, false, OkToBatch))
    return Res;

  return PI_SUCCESS;
}

extern "C" {

pi_result piEnqueueMemBufferFill(pi_queue Queue, pi_mem Buffer,
                                 const void *Pattern, size_t PatternSize,
                                 size_t Offset, size_t Size,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {

  PI_ASSERT(Buffer, PI_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_INVALID_QUEUE);

  return enqueueMemFillHelper(PI_COMMAND_TYPE_MEM_BUFFER_FILL, Queue,
                              pi_cast<char *>(Buffer->getZeHandle()) + Offset,
                              Pattern, PatternSize, Size, NumEventsInWaitList,
                              EventWaitList, Event);
}

static pi_result USMHostAllocImpl(void **ResultPtr, pi_context Context,
                                  pi_usm_mem_properties *Properties,
                                  size_t Size, pi_uint32 Alignment);

pi_result piEnqueueMemBufferMap(pi_queue Queue, pi_mem Buffer,
                                pi_bool BlockingMap, pi_map_flags MapFlags,
                                size_t Offset, size_t Size,
                                pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList, pi_event *Event,
                                void **RetMap) {

  // TODO: we don't implement read-only or write-only, always read-write.
  // assert((map_flags & PI_MAP_READ) != 0);
  // assert((map_flags & PI_MAP_WRITE) != 0);
  PI_ASSERT(Buffer, PI_INVALID_MEM_OBJECT);
  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  PI_ASSERT(Event, PI_INVALID_EVENT);

  ze_event_handle_t ZeEvent = nullptr;

  {
    // Lock automatically releases when this goes out of scope.
    std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

    _pi_ze_event_list_t TmpWaitList;
    if (auto Res = TmpWaitList.createAndRetainPiZeEventList(
            NumEventsInWaitList, EventWaitList, Queue))
      return Res;

    auto Res = createEventAndAssociateQueue(Queue, Event,
                                            PI_COMMAND_TYPE_MEM_BUFFER_MAP,
                                            Queue->CommandListMap.end());
    if (Res != PI_SUCCESS)
      return Res;
    ZeEvent = (*Event)->ZeEvent;
    (*Event)->WaitList = TmpWaitList;
  }

  // TODO: Level Zero is missing the memory "mapping" capabilities, so we are
  // left to doing new memory allocation and a copy (read) on discrete devices.
  // For integrated devices, we have allocated the buffer in host memory so no
  // actions are needed here except for synchronizing on incoming events.
  // A host-to-host copy is done if a host pointer had been supplied during
  // buffer creation on integrated devices.
  //
  // TODO: for discrete, check if the input buffer is already allocated
  // in shared memory and thus is accessible from the host as is.
  // Can we get SYCL RT to predict/allocate in shared memory
  // from the beginning?

  // For integrated devices the buffer has been allocated in host memory.
  if (Buffer->OnHost) {
    // Wait on incoming events before doing the copy
    PI_CALL(piEventsWait(NumEventsInWaitList, EventWaitList));

    if (Queue->isInOrderQueue()) {
      pi_event TmpLastCommandEvent = nullptr;

      {
        // Lock automatically releases when this goes out of scope.
        std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);
        TmpLastCommandEvent = Queue->LastCommandEvent;
      }

      if (TmpLastCommandEvent != nullptr) {
        PI_CALL(piEventsWait(1, &TmpLastCommandEvent));
      }
    }

    if (Buffer->MapHostPtr) {
      *RetMap = Buffer->MapHostPtr + Offset;
      if (!Buffer->HostPtrImported &&
          !(MapFlags & PI_MAP_WRITE_INVALIDATE_REGION))
        memcpy(*RetMap, pi_cast<char *>(Buffer->getZeHandle()) + Offset, Size);
    } else {
      *RetMap = pi_cast<char *>(Buffer->getZeHandle()) + Offset;
    }

    // Signal this event
    ZE_CALL(zeEventHostSignal, (ZeEvent));

    return Buffer->addMapping(*RetMap, Offset, Size);
  }

  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

  // For discrete devices we need a command list
  pi_command_list_ptr_t CommandList{};
  if (auto Res = Queue->Context->getAvailableCommandList(Queue, CommandList))
    return Res;

  // Set the commandlist in the event
  if (Event) {
    (*Event)->ZeCommandList = CommandList->first;
    CommandList->second.append(*Event);
    PI_CALL(piEventRetain(*Event));
  }

  if (Buffer->MapHostPtr) {
    *RetMap = Buffer->MapHostPtr + Offset;
  } else {
    if (auto Res = ZeHostMemAllocHelper(RetMap, Queue->Context, Size))
      return Res;
  }
  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  if (WaitList.Length) {
    ZE_CALL(zeCommandListAppendWaitOnEvents,
            (ZeCommandList, WaitList.Length, WaitList.ZeEventList));
  }

  ZE_CALL(zeCommandListAppendMemoryCopy,
          (ZeCommandList, *RetMap,
           pi_cast<char *>(Buffer->getZeHandle()) + Offset, Size, ZeEvent, 0,
           nullptr));

  if (auto Res = Queue->executeCommandList(CommandList, BlockingMap))
    return Res;

  return Buffer->addMapping(*RetMap, Offset, Size);
}

pi_result piEnqueueMemUnmap(pi_queue Queue, pi_mem MemObj, void *MappedPtr,
                            pi_uint32 NumEventsInWaitList,
                            const pi_event *EventWaitList, pi_event *Event) {
  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  PI_ASSERT(Event, PI_INVALID_EVENT);

  // TODO: handle the case when user does not care to follow the event
  // of unmap completion.
  PI_ASSERT(Event, PI_INVALID_EVENT);

  ze_event_handle_t ZeEvent = nullptr;
  {
    // Lock automatically releases when this goes out of scope.
    std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

    _pi_ze_event_list_t TmpWaitList;
    if (auto Res = TmpWaitList.createAndRetainPiZeEventList(
            NumEventsInWaitList, EventWaitList, Queue))
      return Res;

    auto Res = createEventAndAssociateQueue(Queue, Event,
                                            PI_COMMAND_TYPE_MEM_BUFFER_UNMAP,
                                            Queue->CommandListMap.end());
    if (Res != PI_SUCCESS)
      return Res;
    ZeEvent = (*Event)->ZeEvent;
    (*Event)->WaitList = TmpWaitList;
  }

  _pi_mem::Mapping MapInfo = {};
  if (pi_result Res = MemObj->removeMapping(MappedPtr, MapInfo))
    return Res;

  // NOTE: we still have to free the host memory allocated/returned by
  // piEnqueueMemBufferMap, but can only do so after the above copy
  // is completed. Instead of waiting for It here (blocking), we shall
  // do so in piEventRelease called for the pi_event tracking the unmap.
  // In the case of an integrated device, the map operation does not allocate
  // any memory, so there is nothing to free. This is indicated by a nullptr.
  if (Event)
    (*Event)->CommandData =
        (MemObj->OnHost ? nullptr : (MemObj->MapHostPtr ? nullptr : MappedPtr));

  // For integrated devices the buffer is allocated in host memory.
  if (MemObj->OnHost) {
    // Wait on incoming events before doing the copy
    PI_CALL(piEventsWait(NumEventsInWaitList, EventWaitList));

    if (Queue->isInOrderQueue()) {
      pi_event TmpLastCommandEvent = nullptr;

      {
        // Lock automatically releases when this goes out of scope.
        std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);
        TmpLastCommandEvent = Queue->LastCommandEvent;
      }

      if (TmpLastCommandEvent != nullptr) {
        PI_CALL(piEventsWait(1, &TmpLastCommandEvent));
      }
    }

    if (MemObj->MapHostPtr)
      memcpy(pi_cast<char *>(MemObj->getZeHandle()) + MapInfo.Offset, MappedPtr,
             MapInfo.Size);

    // Signal this event
    ZE_CALL(zeEventHostSignal, (ZeEvent));

    return PI_SUCCESS;
  }

  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

  pi_command_list_ptr_t CommandList{};
  if (auto Res = Queue->Context->getAvailableCommandList(Queue, CommandList))
    return Res;

  // Set the commandlist in the event
  (*Event)->ZeCommandList = CommandList->first;
  CommandList->second.append(*Event);
  PI_CALL(piEventRetain(*Event));

  const auto &ZeCommandList = CommandList->first;
  if ((*Event)->WaitList.Length) {
    ZE_CALL(zeCommandListAppendWaitOnEvents,
            (ZeCommandList, (*Event)->WaitList.Length,
             (*Event)->WaitList.ZeEventList));
  }

  // TODO: Level Zero is missing the memory "mapping" capabilities, so we are
  // left to doing copy (write back to the device).
  //
  // NOTE: Keep this in sync with the implementation of
  // piEnqueueMemBufferMap/piEnqueueMemImageMap.

  ZE_CALL(zeCommandListAppendMemoryCopy,
          (ZeCommandList,
           pi_cast<char *>(MemObj->getZeHandle()) + MapInfo.Offset, MappedPtr,
           MapInfo.Size, ZeEvent, 0, nullptr));

  // Execute command list asynchronously, as the event will be used
  // to track down its completion.
  if (auto Res = Queue->executeCommandList(CommandList))
    return Res;

  return PI_SUCCESS;
}

pi_result piMemImageGetInfo(pi_mem Image, pi_image_info ParamName,
                            size_t ParamValueSize, void *ParamValue,
                            size_t *ParamValueSizeRet) {
  (void)Image;
  (void)ParamName;
  (void)ParamValueSize;
  (void)ParamValue;
  (void)ParamValueSizeRet;

  die("piMemImageGetInfo: not implemented");
  return {};
}

} // extern "C"

static pi_result getImageRegionHelper(pi_mem Mem, pi_image_offset Origin,
                                      pi_image_region Region,
                                      ze_image_region_t &ZeRegion) {

  PI_ASSERT(Mem, PI_INVALID_MEM_OBJECT);
  PI_ASSERT(Origin, PI_INVALID_VALUE);

#ifndef NDEBUG
  PI_ASSERT(Mem->isImage(), PI_INVALID_MEM_OBJECT);
  auto Image = static_cast<_pi_image *>(Mem);
  ze_image_desc_t &ZeImageDesc = Image->ZeImageDesc;

  PI_ASSERT((ZeImageDesc.type == ZE_IMAGE_TYPE_1D && Origin->y == 0 &&
             Origin->z == 0) ||
                (ZeImageDesc.type == ZE_IMAGE_TYPE_1DARRAY && Origin->z == 0) ||
                (ZeImageDesc.type == ZE_IMAGE_TYPE_2D && Origin->z == 0) ||
                (ZeImageDesc.type == ZE_IMAGE_TYPE_3D),
            PI_INVALID_VALUE);

  PI_ASSERT(Region->width && Region->height && Region->depth, PI_INVALID_VALUE);
  PI_ASSERT(
      (ZeImageDesc.type == ZE_IMAGE_TYPE_1D && Region->height == 1 &&
       Region->depth == 1) ||
          (ZeImageDesc.type == ZE_IMAGE_TYPE_1DARRAY && Region->depth == 1) ||
          (ZeImageDesc.type == ZE_IMAGE_TYPE_2D && Region->depth == 1) ||
          (ZeImageDesc.type == ZE_IMAGE_TYPE_3D),
      PI_INVALID_VALUE);
#endif // !NDEBUG

  uint32_t OriginX = pi_cast<uint32_t>(Origin->x);
  uint32_t OriginY = pi_cast<uint32_t>(Origin->y);
  uint32_t OriginZ = pi_cast<uint32_t>(Origin->z);

  uint32_t Width = pi_cast<uint32_t>(Region->width);
  uint32_t Height = pi_cast<uint32_t>(Region->height);
  uint32_t Depth = pi_cast<uint32_t>(Region->depth);

  ZeRegion = {OriginX, OriginY, OriginZ, Width, Height, Depth};

  return PI_SUCCESS;
}

// Helper function to implement image read/write/copy.
// Caller must not hold a lock on the Queue passed in.
static pi_result enqueueMemImageCommandHelper(
    pi_command_type CommandType, pi_queue Queue,
    const void *Src, // image or ptr
    void *Dst,       // image or ptr
    pi_bool IsBlocking, pi_image_offset SrcOrigin, pi_image_offset DstOrigin,
    pi_image_region Region, size_t RowPitch, size_t SlicePitch,
    pi_uint32 NumEventsInWaitList, const pi_event *EventWaitList,
    pi_event *Event, bool PreferCopyEngine = false) {
  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  PI_ASSERT(Event, PI_INVALID_EVENT);

  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

  _pi_ze_event_list_t TmpWaitList;
  if (auto Res = TmpWaitList.createAndRetainPiZeEventList(NumEventsInWaitList,
                                                          EventWaitList, Queue))
    return Res;

  // Get a new command list to be used on this call
  pi_command_list_ptr_t CommandList{};
  bool UseCopyEngine = Queue->useCopyEngine(PreferCopyEngine);
  // We want to batch these commands to avoid extra submissions (costly)
  bool OkToBatch = true;
  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, CommandList, UseCopyEngine, OkToBatch))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  auto Res =
      createEventAndAssociateQueue(Queue, Event, CommandType, CommandList);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  if (WaitList.Length) {
    ZE_CALL(zeCommandListAppendWaitOnEvents,
            (ZeCommandList, WaitList.Length, WaitList.ZeEventList));
  }
  if (CommandType == PI_COMMAND_TYPE_IMAGE_READ) {
    pi_mem SrcMem = pi_cast<pi_mem>(const_cast<void *>(Src));

    ze_image_region_t ZeSrcRegion;
    auto Result = getImageRegionHelper(SrcMem, SrcOrigin, Region, ZeSrcRegion);
    if (Result != PI_SUCCESS)
      return Result;

    // TODO: Level Zero does not support row_pitch/slice_pitch for images yet.
    // Check that SYCL RT did not want pitch larger than default.
    (void)RowPitch;
    (void)SlicePitch;
#ifndef NDEBUG
    PI_ASSERT(SrcMem->isImage(), PI_INVALID_MEM_OBJECT);

    auto SrcImage = static_cast<_pi_image *>(SrcMem);
    const ze_image_desc_t &ZeImageDesc = SrcImage->ZeImageDesc;
    PI_ASSERT(
        RowPitch == 0 ||
            // special case RGBA image pitch equal to region's width
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32 &&
             RowPitch == 4 * 4 * ZeSrcRegion.width) ||
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16 &&
             RowPitch == 4 * 2 * ZeSrcRegion.width) ||
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8 &&
             RowPitch == 4 * ZeSrcRegion.width),
        PI_INVALID_IMAGE_SIZE);
    PI_ASSERT(SlicePitch == 0 || SlicePitch == RowPitch * ZeSrcRegion.height,
              PI_INVALID_IMAGE_SIZE);
#endif // !NDEBUG

    ZE_CALL(zeCommandListAppendImageCopyToMemory,
            (ZeCommandList, Dst,
             pi_cast<ze_image_handle_t>(SrcMem->getZeHandle()), &ZeSrcRegion,
             ZeEvent, 0, nullptr));
  } else if (CommandType == PI_COMMAND_TYPE_IMAGE_WRITE) {
    pi_mem DstMem = pi_cast<pi_mem>(Dst);
    ze_image_region_t ZeDstRegion;
    auto Result = getImageRegionHelper(DstMem, DstOrigin, Region, ZeDstRegion);
    if (Result != PI_SUCCESS)
      return Result;

      // TODO: Level Zero does not support row_pitch/slice_pitch for images yet.
      // Check that SYCL RT did not want pitch larger than default.
#ifndef NDEBUG
    PI_ASSERT(DstMem->isImage(), PI_INVALID_MEM_OBJECT);

    auto DstImage = static_cast<_pi_image *>(DstMem);
    const ze_image_desc_t &ZeImageDesc = DstImage->ZeImageDesc;
    PI_ASSERT(
        RowPitch == 0 ||
            // special case RGBA image pitch equal to region's width
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32 &&
             RowPitch == 4 * 4 * ZeDstRegion.width) ||
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16 &&
             RowPitch == 4 * 2 * ZeDstRegion.width) ||
            (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8 &&
             RowPitch == 4 * ZeDstRegion.width),
        PI_INVALID_IMAGE_SIZE);
    PI_ASSERT(SlicePitch == 0 || SlicePitch == RowPitch * ZeDstRegion.height,
              PI_INVALID_IMAGE_SIZE);
#endif // !NDEBUG

    ZE_CALL(zeCommandListAppendImageCopyFromMemory,
            (ZeCommandList, pi_cast<ze_image_handle_t>(DstMem->getZeHandle()),
             Src, &ZeDstRegion, ZeEvent, 0, nullptr));
  } else if (CommandType == PI_COMMAND_TYPE_IMAGE_COPY) {
    pi_mem SrcImage = pi_cast<pi_mem>(const_cast<void *>(Src));
    pi_mem DstImage = pi_cast<pi_mem>(Dst);

    ze_image_region_t ZeSrcRegion;
    auto Result =
        getImageRegionHelper(SrcImage, SrcOrigin, Region, ZeSrcRegion);
    if (Result != PI_SUCCESS)
      return Result;
    ze_image_region_t ZeDstRegion;
    Result = getImageRegionHelper(DstImage, DstOrigin, Region, ZeDstRegion);
    if (Result != PI_SUCCESS)
      return Result;

    ZE_CALL(zeCommandListAppendImageCopyRegion,
            (ZeCommandList, pi_cast<ze_image_handle_t>(DstImage->getZeHandle()),
             pi_cast<ze_image_handle_t>(SrcImage->getZeHandle()), &ZeDstRegion,
             &ZeSrcRegion, ZeEvent, 0, nullptr));
  } else {
    zePrint("enqueueMemImageUpdate: unsupported image command type\n");
    return PI_INVALID_OPERATION;
  }

  if (auto Res = Queue->executeCommandList(CommandList, IsBlocking, OkToBatch))
    return Res;

  return PI_SUCCESS;
}

extern "C" {

pi_result piEnqueueMemImageRead(pi_queue Queue, pi_mem Image,
                                pi_bool BlockingRead, pi_image_offset Origin,
                                pi_image_region Region, size_t RowPitch,
                                size_t SlicePitch, void *Ptr,
                                pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList,
                                pi_event *Event) {
  PI_ASSERT(Queue, PI_INVALID_QUEUE);

  return enqueueMemImageCommandHelper(
      PI_COMMAND_TYPE_IMAGE_READ, Queue,
      Image, // src
      Ptr,   // dst
      BlockingRead,
      Origin,  // SrcOrigin
      nullptr, // DstOrigin
      Region, RowPitch, SlicePitch, NumEventsInWaitList, EventWaitList, Event);
}

pi_result piEnqueueMemImageWrite(pi_queue Queue, pi_mem Image,
                                 pi_bool BlockingWrite, pi_image_offset Origin,
                                 pi_image_region Region, size_t InputRowPitch,
                                 size_t InputSlicePitch, const void *Ptr,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {

  PI_ASSERT(Queue, PI_INVALID_QUEUE);

  return enqueueMemImageCommandHelper(PI_COMMAND_TYPE_IMAGE_WRITE, Queue,
                                      Ptr,   // src
                                      Image, // dst
                                      BlockingWrite,
                                      nullptr, // SrcOrigin
                                      Origin,  // DstOrigin
                                      Region, InputRowPitch, InputSlicePitch,
                                      NumEventsInWaitList, EventWaitList,
                                      Event);
}

pi_result
piEnqueueMemImageCopy(pi_queue Queue, pi_mem SrcImage, pi_mem DstImage,
                      pi_image_offset SrcOrigin, pi_image_offset DstOrigin,
                      pi_image_region Region, pi_uint32 NumEventsInWaitList,
                      const pi_event *EventWaitList, pi_event *Event) {

  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  // Copy engine is preferred only for host to device transfer.
  // Device to device transfers run faster on compute engines.
  bool PreferCopyEngine = (SrcImage->OnHost || DstImage->OnHost);
  return enqueueMemImageCommandHelper(
      PI_COMMAND_TYPE_IMAGE_COPY, Queue, SrcImage, DstImage,
      false, // is_blocking
      SrcOrigin, DstOrigin, Region,
      0, // row pitch
      0, // slice pitch
      NumEventsInWaitList, EventWaitList, Event, PreferCopyEngine);
}

pi_result piEnqueueMemImageFill(pi_queue Queue, pi_mem Image,
                                const void *FillColor, const size_t *Origin,
                                const size_t *Region,
                                pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList,
                                pi_event *Event) {
  (void)Image;
  (void)FillColor;
  (void)Origin;
  (void)Region;
  (void)NumEventsInWaitList;
  (void)EventWaitList;
  (void)Event;

  PI_ASSERT(Queue, PI_INVALID_QUEUE);

  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

  die("piEnqueueMemImageFill: not implemented");
  return {};
}

pi_result piMemBufferPartition(pi_mem Buffer, pi_mem_flags Flags,
                               pi_buffer_create_type BufferCreateType,
                               void *BufferCreateInfo, pi_mem *RetMem) {

  PI_ASSERT(Buffer && !Buffer->isImage() &&
                !(static_cast<_pi_buffer *>(Buffer))->isSubBuffer(),
            PI_INVALID_MEM_OBJECT);

  PI_ASSERT(BufferCreateType == PI_BUFFER_CREATE_TYPE_REGION &&
                BufferCreateInfo && RetMem,
            PI_INVALID_VALUE);

  if (Flags != PI_MEM_FLAGS_ACCESS_RW) {
    die("piMemBufferPartition: Level-Zero implements only read-write buffer,"
        "no read-only or write-only yet.");
  }

  auto Region = (pi_buffer_region)BufferCreateInfo;

  PI_ASSERT(Region->size != 0u, PI_INVALID_BUFFER_SIZE);
  PI_ASSERT(Region->origin <= (Region->origin + Region->size),
            PI_INVALID_VALUE);

  try {
    *RetMem =
        new _pi_buffer(Buffer->Context,
                       pi_cast<char *>(Buffer->getZeHandle()) +
                           Region->origin /* Level Zero memory handle */,
                       nullptr /* Host pointer */, Buffer /* Parent buffer */,
                       Region->origin /* Sub-buffer origin */,
                       Region->size /*Sub-buffer size*/);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

pi_result piEnqueueNativeKernel(pi_queue Queue, void (*UserFunc)(void *),
                                void *Args, size_t CbArgs,
                                pi_uint32 NumMemObjects, const pi_mem *MemList,
                                const void **ArgsMemLoc,
                                pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList,
                                pi_event *Event) {
  (void)UserFunc;
  (void)Args;
  (void)CbArgs;
  (void)NumMemObjects;
  (void)MemList;
  (void)ArgsMemLoc;
  (void)NumEventsInWaitList;
  (void)EventWaitList;
  (void)Event;

  PI_ASSERT(Queue, PI_INVALID_QUEUE);

  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

  die("piEnqueueNativeKernel: not implemented");
  return {};
}

// Function gets characters between delimeter's in str
// then checks if they are equal to the sub_str.
// returns true if there is at least one instance
// returns false if there are no instances of the name
static bool is_in_separated_string(const std::string &str, char delimiter,
                                   const std::string &sub_str) {
  size_t beg = 0;
  size_t length = 0;
  for (const auto &x : str) {
    if (x == delimiter) {
      if (str.substr(beg, length) == sub_str)
        return true;

      beg += length + 1;
      length = 0;
      continue;
    }
    length++;
  }
  if (length != 0)
    if (str.substr(beg, length) == sub_str)
      return true;

  return false;
}

// TODO: Check if the function_pointer_ret type can be converted to void**.
pi_result piextGetDeviceFunctionPointer(pi_device Device, pi_program Program,
                                        const char *FunctionName,
                                        pi_uint64 *FunctionPointerRet) {
  (void)Device;
  PI_ASSERT(Program, PI_INVALID_PROGRAM);

  std::shared_lock Guard(Program->Mutex);
  if (Program->State != _pi_program::Exe) {
    return PI_INVALID_PROGRAM_EXECUTABLE;
  }

  ze_result_t ZeResult =
      ZE_CALL_NOCHECK(zeModuleGetFunctionPointer,
                      (Program->ZeModule, FunctionName,
                       reinterpret_cast<void **>(FunctionPointerRet)));

  // zeModuleGetFunctionPointer currently fails for all
  // kernels regardless of if the kernel exist or not
  // with ZE_RESULT_ERROR_INVALID_ARGUMENT
  // TODO: remove when this is no longer the case
  // If zeModuleGetFunctionPointer returns invalid argument,
  // fallback to searching through kernel list and return
  // PI_FUNCTION_ADDRESS_IS_NOT_AVAILABLE if the function exists
  // or PI_INVALID_KERNEL_NAME if the function does not exist.
  // FunctionPointerRet should always be 0
  if (ZeResult == ZE_RESULT_ERROR_INVALID_ARGUMENT) {
    size_t Size;
    *FunctionPointerRet = 0;
    PI_CALL(piProgramGetInfo(Program, PI_PROGRAM_INFO_KERNEL_NAMES, 0, nullptr,
                             &Size));

    std::string ClResult(Size, ' ');
    PI_CALL(piProgramGetInfo(Program, PI_PROGRAM_INFO_KERNEL_NAMES,
                             ClResult.size(), &ClResult[0], nullptr));

    // Get rid of the null terminator and search for kernel_name
    // If function can be found return error code to indicate it
    // exists
    ClResult.pop_back();
    if (is_in_separated_string(ClResult, ';', std::string(FunctionName)))
      return PI_FUNCTION_ADDRESS_IS_NOT_AVAILABLE;

    return PI_INVALID_KERNEL_NAME;
  }

  if (ZeResult == ZE_RESULT_ERROR_INVALID_FUNCTION_NAME) {
    *FunctionPointerRet = 0;
    return PI_INVALID_KERNEL_NAME;
  }

  return mapError(ZeResult);
}

static bool ShouldUseUSMAllocator() {
  // Enable allocator by default if it's not explicitly disabled
  return std::getenv("SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR") == nullptr;
}

static const bool UseUSMAllocator = ShouldUseUSMAllocator();

static pi_result USMDeviceAllocImpl(void **ResultPtr, pi_context Context,
                                    pi_device Device,
                                    pi_usm_mem_properties *Properties,
                                    size_t Size, pi_uint32 Alignment) {
  PI_ASSERT(Context, PI_INVALID_CONTEXT);
  PI_ASSERT(Device, PI_INVALID_DEVICE);

  // Check that incorrect bits are not set in the properties.
  PI_ASSERT(!Properties || (Properties && !(*Properties & ~PI_MEM_ALLOC_FLAGS)),
            PI_INVALID_VALUE);

  // TODO: translate PI properties to Level Zero flags
  ZeStruct<ze_device_mem_alloc_desc_t> ZeDesc;
  ZeDesc.flags = 0;
  ZeDesc.ordinal = 0;

  ZeStruct<ze_relaxed_allocation_limits_exp_desc_t> RelaxedDesc;
  if (Size > Device->ZeDeviceProperties->maxMemAllocSize) {
    // Tell Level-Zero to accept Size > maxMemAllocSize
    RelaxedDesc.flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE;
    ZeDesc.pNext = &RelaxedDesc;
  }

  ZE_CALL(zeMemAllocDevice, (Context->ZeContext, &ZeDesc, Size, Alignment,
                             Device->ZeDevice, ResultPtr));

  PI_ASSERT(Alignment == 0 ||
                reinterpret_cast<std::uintptr_t>(*ResultPtr) % Alignment == 0,
            PI_INVALID_VALUE);

  return PI_SUCCESS;
}

static pi_result USMSharedAllocImpl(void **ResultPtr, pi_context Context,
                                    pi_device Device,
                                    pi_usm_mem_properties *Properties,
                                    size_t Size, pi_uint32 Alignment) {
  PI_ASSERT(Context, PI_INVALID_CONTEXT);
  PI_ASSERT(Device, PI_INVALID_DEVICE);

  // Check that incorrect bits are not set in the properties.
  PI_ASSERT(!Properties || (Properties && !(*Properties & ~PI_MEM_ALLOC_FLAGS)),
            PI_INVALID_VALUE);

  // TODO: translate PI properties to Level Zero flags
  ZeStruct<ze_host_mem_alloc_desc_t> ZeHostDesc;
  ZeHostDesc.flags = 0;
  ZeStruct<ze_device_mem_alloc_desc_t> ZeDevDesc;
  ZeDevDesc.flags = 0;
  ZeDevDesc.ordinal = 0;

  ZeStruct<ze_relaxed_allocation_limits_exp_desc_t> RelaxedDesc;
  if (Size > Device->ZeDeviceProperties->maxMemAllocSize) {
    // Tell Level-Zero to accept Size > maxMemAllocSize
    RelaxedDesc.flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE;
    ZeDevDesc.pNext = &RelaxedDesc;
  }

  ZE_CALL(zeMemAllocShared, (Context->ZeContext, &ZeDevDesc, &ZeHostDesc, Size,
                             Alignment, Device->ZeDevice, ResultPtr));

  PI_ASSERT(Alignment == 0 ||
                reinterpret_cast<std::uintptr_t>(*ResultPtr) % Alignment == 0,
            PI_INVALID_VALUE);

  return PI_SUCCESS;
}

static pi_result USMHostAllocImpl(void **ResultPtr, pi_context Context,
                                  pi_usm_mem_properties *Properties,
                                  size_t Size, pi_uint32 Alignment) {
  PI_ASSERT(Context, PI_INVALID_CONTEXT);

  // Check that incorrect bits are not set in the properties.
  PI_ASSERT(!Properties || (Properties && !(*Properties & ~PI_MEM_ALLOC_FLAGS)),
            PI_INVALID_VALUE);

  // TODO: translate PI properties to Level Zero flags
  ZeStruct<ze_host_mem_alloc_desc_t> ZeHostDesc;
  ZeHostDesc.flags = 0;
  ZE_CALL(zeMemAllocHost,
          (Context->ZeContext, &ZeHostDesc, Size, Alignment, ResultPtr));

  PI_ASSERT(Alignment == 0 ||
                reinterpret_cast<std::uintptr_t>(*ResultPtr) % Alignment == 0,
            PI_INVALID_VALUE);

  return PI_SUCCESS;
}

static pi_result USMFreeImpl(pi_context Context, void *Ptr) {
  ZE_CALL(zeMemFree, (Context->ZeContext, Ptr));
  return PI_SUCCESS;
}

// Exception type to pass allocation errors
class UsmAllocationException {
  const pi_result Error;

public:
  UsmAllocationException(pi_result Err) : Error{Err} {}
  pi_result getError() const { return Error; }
};

pi_result USMSharedMemoryAlloc::allocateImpl(void **ResultPtr, size_t Size,
                                             pi_uint32 Alignment) {
  return USMSharedAllocImpl(ResultPtr, Context, Device, nullptr, Size,
                            Alignment);
}

pi_result USMDeviceMemoryAlloc::allocateImpl(void **ResultPtr, size_t Size,
                                             pi_uint32 Alignment) {
  return USMDeviceAllocImpl(ResultPtr, Context, Device, nullptr, Size,
                            Alignment);
}

pi_result USMHostMemoryAlloc::allocateImpl(void **ResultPtr, size_t Size,
                                           pi_uint32 Alignment) {
  return USMHostAllocImpl(ResultPtr, Context, nullptr, Size, Alignment);
}

SystemMemory::MemType USMSharedMemoryAlloc::getMemTypeImpl() {
  return SystemMemory::Shared;
}

SystemMemory::MemType USMDeviceMemoryAlloc::getMemTypeImpl() {
  return SystemMemory::Device;
}

SystemMemory::MemType USMHostMemoryAlloc::getMemTypeImpl() {
  return SystemMemory::Host;
}

void *USMMemoryAllocBase::allocate(size_t Size) {
  void *Ptr = nullptr;

  auto Res = allocateImpl(&Ptr, Size, sizeof(void *));
  if (Res != PI_SUCCESS) {
    throw UsmAllocationException(Res);
  }

  return Ptr;
}

void *USMMemoryAllocBase::allocate(size_t Size, size_t Alignment) {
  void *Ptr = nullptr;

  auto Res = allocateImpl(&Ptr, Size, Alignment);
  if (Res != PI_SUCCESS) {
    throw UsmAllocationException(Res);
  }
  return Ptr;
}

void USMMemoryAllocBase::deallocate(void *Ptr) {
  auto Res = USMFreeImpl(Context, Ptr);
  if (Res != PI_SUCCESS) {
    throw UsmAllocationException(Res);
  }
}

SystemMemory::MemType USMMemoryAllocBase::getMemType() {
  return getMemTypeImpl();
}

pi_result piextUSMDeviceAlloc(void **ResultPtr, pi_context Context,
                              pi_device Device,
                              pi_usm_mem_properties *Properties, size_t Size,
                              pi_uint32 Alignment) {
  // L0 supports alignment up to 64KB and silently ignores higher values.
  // We flag alignment > 64KB as an invalid value.
  if (Alignment > 65536)
    return PI_INVALID_VALUE;

  pi_platform Plt = Device->Platform;
  std::unique_lock<std::mutex> ContextsLock(Plt->ContextsMutex,
                                            std::defer_lock);
  if (IndirectAccessTrackingEnabled) {
    // Lock the mutex which is guarding contexts container in the platform.
    // This prevents new kernels from being submitted in any context while we
    // are in the process of allocating a memory, this is needed to properly
    // capture allocations by kernels with indirect access.
    ContextsLock.lock();
    // We are going to defer memory release if there are kernels with indirect
    // access, that is why explicitly retain context to be sure that it is
    // released after all memory allocations in this context are released.
    PI_CALL(piContextRetain(Context));
  }

  if (!UseUSMAllocator ||
      // L0 spec says that allocation fails if Alignment != 2^n, in order to
      // keep the same behavior for the allocator, just call L0 API directly and
      // return the error code.
      ((Alignment & (Alignment - 1)) != 0)) {
    pi_result Res = USMDeviceAllocImpl(ResultPtr, Context, Device, Properties,
                                       Size, Alignment);
    if (IndirectAccessTrackingEnabled) {
      // Keep track of all memory allocations in the context
      Context->MemAllocs.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(*ResultPtr),
                                 std::forward_as_tuple(Context));
    }
    return Res;
  }

  try {
    auto It = Context->DeviceMemAllocContexts.find(Device);
    if (It == Context->DeviceMemAllocContexts.end())
      return PI_INVALID_VALUE;

    *ResultPtr = It->second.allocate(Size, Alignment);
    if (IndirectAccessTrackingEnabled) {
      // Keep track of all memory allocations in the context
      Context->MemAllocs.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(*ResultPtr),
                                 std::forward_as_tuple(Context));
    }

  } catch (const UsmAllocationException &Ex) {
    *ResultPtr = nullptr;
    return Ex.getError();
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

pi_result piextUSMSharedAlloc(void **ResultPtr, pi_context Context,
                              pi_device Device,
                              pi_usm_mem_properties *Properties, size_t Size,
                              pi_uint32 Alignment) {
  // L0 supports alignment up to 64KB and silently ignores higher values.
  // We flag alignment > 64KB as an invalid value.
  if (Alignment > 65536)
    return PI_INVALID_VALUE;

  pi_platform Plt = Device->Platform;
  std::unique_lock<std::mutex> ContextsLock(Plt->ContextsMutex,
                                            std::defer_lock);
  if (IndirectAccessTrackingEnabled) {
    // Lock the mutex which is guarding contexts container in the platform.
    // This prevents new kernels from being submitted in any context while we
    // are in the process of allocating a memory, this is needed to properly
    // capture allocations by kernels with indirect access.
    ContextsLock.lock();
    // We are going to defer memory release if there are kernels with indirect
    // access, that is why explicitly retain context to be sure that it is
    // released after all memory allocations in this context are released.
    PI_CALL(piContextRetain(Context));
  }

  if (!UseUSMAllocator ||
      // L0 spec says that allocation fails if Alignment != 2^n, in order to
      // keep the same behavior for the allocator, just call L0 API directly and
      // return the error code.
      ((Alignment & (Alignment - 1)) != 0)) {
    pi_result Res = USMSharedAllocImpl(ResultPtr, Context, Device, Properties,
                                       Size, Alignment);
    if (IndirectAccessTrackingEnabled) {
      // Keep track of all memory allocations in the context
      Context->MemAllocs.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(*ResultPtr),
                                 std::forward_as_tuple(Context));
    }
    return Res;
  }

  try {
    auto It = Context->SharedMemAllocContexts.find(Device);
    if (It == Context->SharedMemAllocContexts.end())
      return PI_INVALID_VALUE;

    *ResultPtr = It->second.allocate(Size, Alignment);
    if (IndirectAccessTrackingEnabled) {
      // Keep track of all memory allocations in the context
      Context->MemAllocs.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(*ResultPtr),
                                 std::forward_as_tuple(Context));
    }
  } catch (const UsmAllocationException &Ex) {
    *ResultPtr = nullptr;
    return Ex.getError();
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

pi_result piextUSMHostAlloc(void **ResultPtr, pi_context Context,
                            pi_usm_mem_properties *Properties, size_t Size,
                            pi_uint32 Alignment) {
  // L0 supports alignment up to 64KB and silently ignores higher values.
  // We flag alignment > 64KB as an invalid value.
  if (Alignment > 65536)
    return PI_INVALID_VALUE;

  pi_platform Plt = Context->Devices[0]->Platform;
  std::unique_lock<std::mutex> ContextsLock(Plt->ContextsMutex,
                                            std::defer_lock);
  if (IndirectAccessTrackingEnabled) {
    // Lock the mutex which is guarding contexts container in the platform.
    // This prevents new kernels from being submitted in any context while we
    // are in the process of allocating a memory, this is needed to properly
    // capture allocations by kernels with indirect access.
    ContextsLock.lock();
    // We are going to defer memory release if there are kernels with indirect
    // access, that is why explicitly retain context to be sure that it is
    // released after all memory allocations in this context are released.
    PI_CALL(piContextRetain(Context));
  }

  if (!UseUSMAllocator ||
      // L0 spec says that allocation fails if Alignment != 2^n, in order to
      // keep the same behavior for the allocator, just call L0 API directly and
      // return the error code.
      ((Alignment & (Alignment - 1)) != 0)) {
    pi_result Res =
        USMHostAllocImpl(ResultPtr, Context, Properties, Size, Alignment);
    if (IndirectAccessTrackingEnabled) {
      // Keep track of all memory allocations in the context
      Context->MemAllocs.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(*ResultPtr),
                                 std::forward_as_tuple(Context));
    }
    return Res;
  }

  // There is a single allocator for Host USM allocations, so we don't need to
  // find the allocator depending on context as we do for Shared and Device
  // allocations.
  try {
    *ResultPtr = Context->HostMemAllocContext->allocate(Size, Alignment);
    if (IndirectAccessTrackingEnabled) {
      // Keep track of all memory allocations in the context
      Context->MemAllocs.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(*ResultPtr),
                                 std::forward_as_tuple(Context));
    }
  } catch (const UsmAllocationException &Ex) {
    *ResultPtr = nullptr;
    return Ex.getError();
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  return PI_SUCCESS;
}

// Helper function to deallocate USM memory, if indirect access support is
// enabled then a caller must lock the platform-level mutex guarding the
// container with contexts because deallocating the memory can turn RefCount of
// a context to 0 and as a result the context being removed from the list of
// tracked contexts.
static pi_result USMFreeHelper(pi_context Context, void *Ptr) {
  if (IndirectAccessTrackingEnabled) {
    auto It = Context->MemAllocs.find(Ptr);
    if (It == std::end(Context->MemAllocs)) {
      die("All memory allocations must be tracked!");
    }
    if (--(It->second.RefCount) != 0) {
      // Memory can't be deallocated yet.
      return PI_SUCCESS;
    }

    // Reference count is zero, it is ok to free memory.
    // We don't need to track this allocation anymore.
    Context->MemAllocs.erase(It);
  }

  if (!UseUSMAllocator) {
    pi_result Res = USMFreeImpl(Context, Ptr);
    if (IndirectAccessTrackingEnabled)
      PI_CALL(ContextReleaseHelper(Context));
    return Res;
  }

  // Query the device of the allocation to determine the right allocator context
  ze_device_handle_t ZeDeviceHandle;
  ZeStruct<ze_memory_allocation_properties_t> ZeMemoryAllocationProperties;

  // Query memory type of the pointer we're freeing to determine the correct
  // way to do it(directly or via an allocator)
  ZE_CALL(zeMemGetAllocProperties,
          (Context->ZeContext, Ptr, &ZeMemoryAllocationProperties,
           &ZeDeviceHandle));

  // If memory type is host release from host pool
  if (ZeMemoryAllocationProperties.type == ZE_MEMORY_TYPE_HOST) {
    try {
      Context->HostMemAllocContext->deallocate(Ptr);
    } catch (const UsmAllocationException &Ex) {
      return Ex.getError();
    } catch (...) {
      return PI_ERROR_UNKNOWN;
    }
    if (IndirectAccessTrackingEnabled)
      PI_CALL(ContextReleaseHelper(Context));
    return PI_SUCCESS;
  }

  if (!ZeDeviceHandle) {
    // The only case where it is OK not have device identified is
    // if the memory is not known to the driver. We should not ever get
    // this either, probably.
    PI_ASSERT(ZeMemoryAllocationProperties.type == ZE_MEMORY_TYPE_UNKNOWN,
              PI_INVALID_DEVICE);
  } else {
    pi_device Device;
    if (Context->Devices.size() == 1) {
      Device = Context->Devices[0];
      PI_ASSERT(Device->ZeDevice == ZeDeviceHandle, PI_INVALID_DEVICE);
    } else {
      // All devices in the context are of the same platform.
      auto Platform = Context->Devices[0]->Platform;
      Device = Platform->getDeviceFromNativeHandle(ZeDeviceHandle);
      PI_ASSERT(Device, PI_INVALID_DEVICE);
    }

    auto DeallocationHelper =
        [Context, Device,
         Ptr](std::unordered_map<pi_device, USMAllocContext> &AllocContextMap) {
          try {
            auto It = AllocContextMap.find(Device);
            if (It == AllocContextMap.end())
              return PI_INVALID_VALUE;

            // The right context is found, deallocate the pointer
            It->second.deallocate(Ptr);
          } catch (const UsmAllocationException &Ex) {
            return Ex.getError();
          }

          if (IndirectAccessTrackingEnabled)
            PI_CALL(ContextReleaseHelper(Context));
          return PI_SUCCESS;
        };

    switch (ZeMemoryAllocationProperties.type) {
    case ZE_MEMORY_TYPE_SHARED:
      return DeallocationHelper(Context->SharedMemAllocContexts);
    case ZE_MEMORY_TYPE_DEVICE:
      return DeallocationHelper(Context->DeviceMemAllocContexts);
    default:
      // Handled below
      break;
    }
  }

  pi_result Res = USMFreeImpl(Context, Ptr);

  if (IndirectAccessTrackingEnabled)
    PI_CALL(ContextReleaseHelper(Context));
  return Res;
}

pi_result piextUSMFree(pi_context Context, void *Ptr) {
  pi_platform Plt = Context->Devices[0]->Platform;
  std::unique_lock<std::mutex> ContextsLock(Plt->ContextsMutex,
                                            std::defer_lock);
  if (IndirectAccessTrackingEnabled)
    ContextsLock.lock();
  return USMFreeHelper(Context, Ptr);
}

pi_result piextKernelSetArgPointer(pi_kernel Kernel, pi_uint32 ArgIndex,
                                   size_t ArgSize, const void *ArgValue) {

  PI_CALL(piKernelSetArg(Kernel, ArgIndex, ArgSize, ArgValue));
  return PI_SUCCESS;
}

/// USM Memset API
///
/// @param Queue is the queue to submit to
/// @param Ptr is the ptr to memset
/// @param Value is value to set.  It is interpreted as an 8-bit value and the
/// upper
///        24 bits are ignored
/// @param Count is the size in bytes to memset
/// @param NumEventsInWaitlist is the number of events to wait on
/// @param EventsWaitlist is an array of events to wait on
/// @param Event is the event that represents this operation
pi_result piextUSMEnqueueMemset(pi_queue Queue, void *Ptr, pi_int32 Value,
                                size_t Count, pi_uint32 NumEventsInWaitlist,
                                const pi_event *EventsWaitlist,
                                pi_event *Event) {
  if (!Ptr) {
    return PI_INVALID_VALUE;
  }

  PI_ASSERT(Queue, PI_INVALID_QUEUE);

  return enqueueMemFillHelper(
      // TODO: do we need a new command type for USM memset?
      PI_COMMAND_TYPE_MEM_BUFFER_FILL, Queue, Ptr,
      &Value, // It will be interpreted as an 8-bit value,
      1,      // which is indicated with this pattern_size==1
      Count, NumEventsInWaitlist, EventsWaitlist, Event);
}

// Helper function to check if a pointer is a device pointer.
static bool IsDevicePointer(pi_context Context, const void *Ptr) {
  ze_device_handle_t ZeDeviceHandle;
  ZeStruct<ze_memory_allocation_properties_t> ZeMemoryAllocationProperties;

  // Query memory type of the pointer
  ZE_CALL(zeMemGetAllocProperties,
          (Context->ZeContext, Ptr, &ZeMemoryAllocationProperties,
           &ZeDeviceHandle));

  return (ZeMemoryAllocationProperties.type == ZE_MEMORY_TYPE_DEVICE);
}

pi_result piextUSMEnqueueMemcpy(pi_queue Queue, pi_bool Blocking, void *DstPtr,
                                const void *SrcPtr, size_t Size,
                                pi_uint32 NumEventsInWaitlist,
                                const pi_event *EventsWaitlist,
                                pi_event *Event) {

  if (!DstPtr) {
    return PI_INVALID_VALUE;
  }

  PI_ASSERT(Queue, PI_INVALID_QUEUE);

  // Device to Device copies are found to execute slower on copy engine
  // (versus compute engine).
  bool PreferCopyEngine = !IsDevicePointer(Queue->Context, SrcPtr) ||
                          !IsDevicePointer(Queue->Context, DstPtr);

  // Temporary option added to use copy engine for D2D copy
  PreferCopyEngine |= UseCopyEngineForD2DCopy;

  return enqueueMemCopyHelper(
      // TODO: do we need a new command type for this?
      PI_COMMAND_TYPE_MEM_BUFFER_COPY, Queue, DstPtr, Blocking, Size, SrcPtr,
      NumEventsInWaitlist, EventsWaitlist, Event, PreferCopyEngine);
}

/// Hint to migrate memory to the device
///
/// @param Queue is the queue to submit to
/// @param Ptr points to the memory to migrate
/// @param Size is the number of bytes to migrate
/// @param Flags is a bitfield used to specify memory migration options
/// @param NumEventsInWaitlist is the number of events to wait on
/// @param EventsWaitlist is an array of events to wait on
/// @param Event is the event that represents this operation
pi_result piextUSMEnqueuePrefetch(pi_queue Queue, const void *Ptr, size_t Size,
                                  pi_usm_migration_flags Flags,
                                  pi_uint32 NumEventsInWaitList,
                                  const pi_event *EventWaitList,
                                  pi_event *Event) {

  // flags is currently unused so fail if set
  PI_ASSERT(Flags == 0, PI_INVALID_VALUE);
  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  PI_ASSERT(Event, PI_INVALID_EVENT);

  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

  /**
   * @brief Please note that the following code should be run before the
   * subsequent getAvailableCommandList() call so that there is no
   * dead-lock from waiting unsubmitted events in an open batch.
   */
  _pi_ze_event_list_t TmpWaitList;
  if (auto Res = TmpWaitList.createAndRetainPiZeEventList(NumEventsInWaitList,
                                                          EventWaitList, Queue))
    return Res;

  // Get a new command list to be used on this call
  pi_command_list_ptr_t CommandList{};
  // TODO: Change UseCopyEngine argument to 'true' once L0 backend
  // support is added
  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, CommandList, false /* UseCopyEngine */))
    return Res;

  // TODO: do we need to create a unique command type for this?
  ze_event_handle_t ZeEvent = nullptr;
  auto Res = createEventAndAssociateQueue(Queue, Event, PI_COMMAND_TYPE_USER,
                                          CommandList);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  const auto &WaitList = (*Event)->WaitList;
  const auto &ZeCommandList = CommandList->first;
  if (WaitList.Length) {
    ZE_CALL(zeCommandListAppendWaitOnEvents,
            (ZeCommandList, WaitList.Length, WaitList.ZeEventList));
  }
  // TODO: figure out how to translate "flags"
  ZE_CALL(zeCommandListAppendMemoryPrefetch, (ZeCommandList, Ptr, Size));

  // TODO: Level Zero does not have a completion "event" with the prefetch API,
  // so manually add command to signal our event.
  ZE_CALL(zeCommandListAppendSignalEvent, (ZeCommandList, ZeEvent));

  if (auto Res = Queue->executeCommandList(CommandList, false))
    return Res;

  return PI_SUCCESS;
}

/// USM memadvise API to govern behavior of automatic migration mechanisms
///
/// @param Queue is the queue to submit to
/// @param Ptr is the data to be advised
/// @param Length is the size in bytes of the meory to advise
/// @param Advice is device specific advice
/// @param Event is the event that represents this operation
///
pi_result piextUSMEnqueueMemAdvise(pi_queue Queue, const void *Ptr,
                                   size_t Length, pi_mem_advice Advice,
                                   pi_event *Event) {
  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  PI_ASSERT(Event, PI_INVALID_EVENT);

  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

  auto ZeAdvice = pi_cast<ze_memory_advice_t>(Advice);

  // Get a new command list to be used on this call
  pi_command_list_ptr_t CommandList{};
  // UseCopyEngine is set to 'false' here.
  // TODO: Additional analysis is required to check if this operation will
  // run faster on copy engines.
  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, CommandList, false /* UseCopyEngine */))
    return Res;

  // TODO: do we need to create a unique command type for this?
  ze_event_handle_t ZeEvent = nullptr;
  auto Res = createEventAndAssociateQueue(Queue, Event, PI_COMMAND_TYPE_USER,
                                          CommandList);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;

  if (auto Res =
          (*Event)->WaitList.createAndRetainPiZeEventList(0, nullptr, Queue))
    return Res;

  const auto &ZeCommandList = CommandList->first;
  const auto &WaitList = (*Event)->WaitList;

  if (WaitList.Length) {
    ZE_CALL(zeCommandListAppendWaitOnEvents,
            (ZeCommandList, WaitList.Length, WaitList.ZeEventList));
  }

  ZE_CALL(zeCommandListAppendMemAdvise,
          (ZeCommandList, Queue->Device->ZeDevice, Ptr, Length, ZeAdvice));

  // TODO: Level Zero does not have a completion "event" with the advise API,
  // so manually add command to signal our event.
  ZE_CALL(zeCommandListAppendSignalEvent, (ZeCommandList, ZeEvent));

  Queue->executeCommandList(CommandList, false);
  return PI_SUCCESS;
}

/// API to query information about USM allocated pointers.
/// Valid Queries:
///   PI_MEM_ALLOC_TYPE returns host/device/shared pi_usm_type value
///   PI_MEM_ALLOC_BASE_PTR returns the base ptr of an allocation if
///                         the queried pointer fell inside an allocation.
///                         Result must fit in void *
///   PI_MEM_ALLOC_SIZE returns how big the queried pointer's
///                     allocation is in bytes. Result is a size_t.
///   PI_MEM_ALLOC_DEVICE returns the pi_device this was allocated against
///
/// @param Context is the pi_context
/// @param Ptr is the pointer to query
/// @param ParamName is the type of query to perform
/// @param ParamValueSize is the size of the result in bytes
/// @param ParamValue is the result
/// @param ParamValueRet is how many bytes were written
pi_result piextUSMGetMemAllocInfo(pi_context Context, const void *Ptr,
                                  pi_mem_info ParamName, size_t ParamValueSize,
                                  void *ParamValue, size_t *ParamValueSizeRet) {
  PI_ASSERT(Context, PI_INVALID_CONTEXT);

  ze_device_handle_t ZeDeviceHandle;
  ZeStruct<ze_memory_allocation_properties_t> ZeMemoryAllocationProperties;

  ZE_CALL(zeMemGetAllocProperties,
          (Context->ZeContext, Ptr, &ZeMemoryAllocationProperties,
           &ZeDeviceHandle));

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  switch (ParamName) {
  case PI_MEM_ALLOC_TYPE: {
    pi_usm_type MemAllocaType;
    switch (ZeMemoryAllocationProperties.type) {
    case ZE_MEMORY_TYPE_UNKNOWN:
      MemAllocaType = PI_MEM_TYPE_UNKNOWN;
      break;
    case ZE_MEMORY_TYPE_HOST:
      MemAllocaType = PI_MEM_TYPE_HOST;
      break;
    case ZE_MEMORY_TYPE_DEVICE:
      MemAllocaType = PI_MEM_TYPE_DEVICE;
      break;
    case ZE_MEMORY_TYPE_SHARED:
      MemAllocaType = PI_MEM_TYPE_SHARED;
      break;
    default:
      zePrint("piextUSMGetMemAllocInfo: unexpected usm memory type\n");
      return PI_INVALID_VALUE;
    }
    return ReturnValue(MemAllocaType);
  }
  case PI_MEM_ALLOC_DEVICE:
    if (ZeDeviceHandle) {
      // All devices in the context are of the same platform.
      auto Platform = Context->Devices[0]->Platform;
      auto Device = Platform->getDeviceFromNativeHandle(ZeDeviceHandle);
      return Device ? ReturnValue(Device) : PI_INVALID_VALUE;
    } else {
      return PI_INVALID_VALUE;
    }
  case PI_MEM_ALLOC_BASE_PTR: {
    void *Base;
    ZE_CALL(zeMemGetAddressRange, (Context->ZeContext, Ptr, &Base, nullptr));
    return ReturnValue(Base);
  }
  case PI_MEM_ALLOC_SIZE: {
    size_t Size;
    ZE_CALL(zeMemGetAddressRange, (Context->ZeContext, Ptr, nullptr, &Size));
    return ReturnValue(Size);
  }
  default:
    zePrint("piextUSMGetMemAllocInfo: unsupported ParamName\n");
    return PI_INVALID_VALUE;
  }
  return PI_SUCCESS;
}

pi_result piKernelSetExecInfo(pi_kernel Kernel, pi_kernel_exec_info ParamName,
                              size_t ParamValueSize, const void *ParamValue) {
  (void)ParamValueSize;
  PI_ASSERT(Kernel, PI_INVALID_KERNEL);
  PI_ASSERT(ParamValue, PI_INVALID_VALUE);

  if (ParamName == PI_USM_INDIRECT_ACCESS &&
      *(static_cast<const pi_bool *>(ParamValue)) == PI_TRUE) {
    // The whole point for users really was to not need to know anything
    // about the types of allocations kernel uses. So in DPC++ we always
    // just set all 3 modes for each kernel.
    ze_kernel_indirect_access_flags_t IndirectFlags =
        ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST |
        ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE |
        ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED;
    ZE_CALL(zeKernelSetIndirectAccess, (Kernel->ZeKernel, IndirectFlags));
  } else {
    zePrint("piKernelSetExecInfo: unsupported ParamName\n");
    return PI_INVALID_VALUE;
  }

  return PI_SUCCESS;
}

pi_result piextProgramSetSpecializationConstant(pi_program Prog,
                                                pi_uint32 SpecID, size_t,
                                                const void *SpecValue) {
  std::scoped_lock Guard(Prog->Mutex);

  // Remember the value of this specialization constant until the program is
  // built.  Note that we only save the pointer to the buffer that contains the
  // value.  The caller is responsible for maintaining storage for this buffer.
  //
  // NOTE: SpecSize is unused in Level Zero, the size is known from SPIR-V by
  // SpecID.
  Prog->SpecConstants[SpecID] = SpecValue;

  return PI_SUCCESS;
}

pi_result piPluginInit(pi_plugin *PluginInit) {
  PI_ASSERT(PluginInit, PI_INVALID_VALUE);

  // TODO: handle versioning/targets properly.
  size_t PluginVersionSize = sizeof(PluginInit->PluginVersion);

  PI_ASSERT(strlen(_PI_H_VERSION_STRING) < PluginVersionSize, PI_INVALID_VALUE);

  strncpy(PluginInit->PluginVersion, _PI_H_VERSION_STRING, PluginVersionSize);

#define _PI_API(api)                                                           \
  (PluginInit->PiFunctionTable).api = (decltype(&::api))(&api);
#include <CL/sycl/detail/pi.def>

  return PI_SUCCESS;
}

pi_result piextPluginGetOpaqueData(void *opaque_data_param,
                                   void **opaque_data_return) {
  (void)opaque_data_param;
  (void)opaque_data_return;
  return PI_ERROR_UNKNOWN;
}

// SYCL RT calls this api to notify the end of plugin lifetime.
// It can include all the jobs to tear down resources before
// the plugin is unloaded from memory.
pi_result piTearDown(void *PluginParameter) {
  (void)PluginParameter;
  bool LeakFound = false;
  // reclaim pi_platform objects here since we don't have piPlatformRelease.
  for (pi_platform &Platform : *PiPlatformsCache) {
    delete Platform;
  }
  delete PiPlatformsCache;
  delete PiPlatformsCacheMutex;

  // Print the balance of various create/destroy native calls.
  // The idea is to verify if the number of create(+) and destroy(-) calls are
  // matched.
  if (ZeDebug & ZE_DEBUG_CALL_COUNT) {
    // clang-format off
    //
    // The format of this table is such that each row accounts for a
    // specific type of objects, and all elements in the raw except the last
    // one are allocating objects of that type, while the last element is known
    // to deallocate objects of that type.
    //
    std::vector<std::vector<const char *>> CreateDestroySet = {
      {"zeContextCreate",      "zeContextDestroy"},
      {"zeCommandQueueCreate", "zeCommandQueueDestroy"},
      {"zeModuleCreate",       "zeModuleDestroy"},
      {"zeKernelCreate",       "zeKernelDestroy"},
      {"zeEventPoolCreate",    "zeEventPoolDestroy"},
      {"zeCommandListCreateImmediate", "zeCommandListCreate", "zeCommandListDestroy"},
      {"zeEventCreate",        "zeEventDestroy"},
      {"zeFenceCreate",        "zeFenceDestroy"},
      {"zeImageCreate",        "zeImageDestroy"},
      {"zeSamplerCreate",      "zeSamplerDestroy"},
      {"zeMemAllocDevice", "zeMemAllocHost", "zeMemAllocShared", "zeMemFree"},
    };

    // A sample output aimed below is this:
    // ------------------------------------------------------------------------
    //                zeContextCreate = 1     \--->        zeContextDestroy = 1
    //           zeCommandQueueCreate = 1     \--->   zeCommandQueueDestroy = 1
    //                 zeModuleCreate = 1     \--->         zeModuleDestroy = 1
    //                 zeKernelCreate = 1     \--->         zeKernelDestroy = 1
    //              zeEventPoolCreate = 1     \--->      zeEventPoolDestroy = 1
    //   zeCommandListCreateImmediate = 1     |
    //            zeCommandListCreate = 1     \--->    zeCommandListDestroy = 1  ---> LEAK = 1
    //                  zeEventCreate = 2     \--->          zeEventDestroy = 2
    //                  zeFenceCreate = 1     \--->          zeFenceDestroy = 1
    //                  zeImageCreate = 0     \--->          zeImageDestroy = 0
    //                zeSamplerCreate = 0     \--->        zeSamplerDestroy = 0
    //               zeMemAllocDevice = 0     |
    //                 zeMemAllocHost = 1     |
    //               zeMemAllocShared = 0     \--->               zeMemFree = 1
    //
    // clang-format on

    fprintf(stderr, "ZE_DEBUG=%d: check balance of create/destroy calls\n",
            ZE_DEBUG_CALL_COUNT);
    fprintf(stderr,
            "----------------------------------------------------------\n");
    for (const auto &Row : CreateDestroySet) {
      int diff = 0;
      for (auto I = Row.begin(); I != Row.end();) {
        const char *ZeName = *I;
        const auto &ZeCount = (*ZeCallCount)[*I];

        bool First = (I == Row.begin());
        bool Last = (++I == Row.end());

        if (Last) {
          fprintf(stderr, " \\--->");
          diff -= ZeCount;
        } else {
          diff += ZeCount;
          if (!First) {
            fprintf(stderr, " | \n");
          }
        }

        fprintf(stderr, "%30s = %-5d", ZeName, ZeCount);
      }

      if (diff) {
        LeakFound = true;
        fprintf(stderr, " ---> LEAK = %d", diff);
      }
      fprintf(stderr, "\n");
    }

    ZeCallCount->clear();
    delete ZeCallCount;
    ZeCallCount = nullptr;
  }
  if (LeakFound)
    return PI_INVALID_MEM_OBJECT;
  return PI_SUCCESS;
}

} // extern "C"
