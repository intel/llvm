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

#include <level_zero/zes_api.h>
#include <level_zero/zet_api.h>

#include "usm_allocator.hpp"

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
      {ZE_RESULT_ERROR_INVALID_ARGUMENT, PI_INVALID_VALUE},
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
      {ZE_RESULT_ERROR_MODULE_BUILD_FAILURE, PI_BUILD_PROGRAM_FAILURE}};

  auto It = ErrorMapping.find(ZeResult);
  if (It == ErrorMapping.end()) {
    return PI_ERROR_UNKNOWN;
  }
  return It->second;
}

// This will count the calls to Level-Zero
static std::map<std::string, int> *ZeCallCount = nullptr;

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
static int ZeDebug = ZE_DEBUG_NONE;

static void zePrint(const char *Format, ...) {
  if (ZeDebug & ZE_DEBUG_BASIC) {
    va_list Args;
    va_start(Args, Format);
    vfprintf(stderr, Format, Args);
    va_end(Args);
  }
}

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
// template <>
// ze_result_t zeHostSynchronize(ze_fence_handle_t Handle) {
//   return zeHostSynchronizeImpl(zeFenceHostSynchronize, Handle);
// }

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

// Keeps track if the global offset extension is found
static bool PiDriverGlobalOffsetExtensionFound = false;

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
_pi_context::getFreeSlotInExistingOrNewPool(ze_event_pool_handle_t &ZePool,
                                            size_t &Index) {
  // Maximum number of events that can be present in an event ZePool is captured
  // here. Setting it to 256 gave best possible performance for several
  // benchmarks.
  static const pi_uint32 MaxNumEventsPerPool = [] {
    const auto MaxNumEventsPerPoolEnv =
        std::getenv("ZE_MAX_NUMBER_OF_EVENTS_PER_EVENT_POOL");
    return MaxNumEventsPerPoolEnv ? std::atoi(MaxNumEventsPerPoolEnv) : 256;
  }();

  if (MaxNumEventsPerPool == 0) {
    zePrint("Zero size can't be specified in the "
            "ZE_MAX_NUMBER_OF_EVENTS_PER_EVENT_POOL\n");
    return PI_INVALID_VALUE;
  }

  Index = 0;
  // Create one event ZePool per MaxNumEventsPerPool events
  if ((ZeEventPool == nullptr) ||
      (NumEventsAvailableInEventPool[ZeEventPool] == 0)) {
    // Creation of the new ZePool with record in NumEventsAvailableInEventPool
    // and initialization of the record in NumEventsLiveInEventPool must be done
    // atomically. Otherwise it is possible that decrementAliveEventsInPool will
    // be called for the record in NumEventsLiveInEventPool before its
    std::lock(NumEventsAvailableInEventPoolMutex,
              NumEventsLiveInEventPoolMutex);
    std::lock_guard<std::mutex> NumEventsAvailableInEventPoolGuard(
        NumEventsAvailableInEventPoolMutex, std::adopt_lock);
    std::lock_guard<std::mutex> NumEventsLiveInEventPoolGuard(
        NumEventsLiveInEventPoolMutex, std::adopt_lock);

    ze_event_pool_desc_t ZeEventPoolDesc = {};
    ZeEventPoolDesc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
    ZeEventPoolDesc.count = MaxNumEventsPerPool;

    // Make all events visible on the host.
    // TODO: events that are used only on device side APIs can be optimized
    // to not be from the host-visible pool.
    //
    ZeEventPoolDesc.flags =
        ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;

    std::vector<ze_device_handle_t> ZeDevices;
    std::for_each(Devices.begin(), Devices.end(),
                  [&](pi_device &D) { ZeDevices.push_back(D->ZeDevice); });

    ZE_CALL(zeEventPoolCreate, (ZeContext, &ZeEventPoolDesc, ZeDevices.size(),
                                &ZeDevices[0], &ZeEventPool));
    NumEventsAvailableInEventPool[ZeEventPool] = MaxNumEventsPerPool - 1;
    NumEventsLiveInEventPool[ZeEventPool] = MaxNumEventsPerPool;
  } else {
    std::lock_guard<std::mutex> NumEventsAvailableInEventPoolGuard(
        NumEventsAvailableInEventPoolMutex);
    Index = MaxNumEventsPerPool - NumEventsAvailableInEventPool[ZeEventPool];
    --NumEventsAvailableInEventPool[ZeEventPool];
  }
  ZePool = ZeEventPool;
  return PI_SUCCESS;
}

pi_result
_pi_context::decrementAliveEventsInPool(ze_event_pool_handle_t ZePool) {
  std::lock_guard<std::mutex> Lock(NumEventsLiveInEventPoolMutex);
  --NumEventsLiveInEventPool[ZePool];
  if (NumEventsLiveInEventPool[ZePool] == 0) {
    ZE_CALL(zeEventPoolDestroy, (ZePool));
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
// \param ZeCommandList the handle to associate with the newly created event
inline static pi_result
createEventAndAssociateQueue(pi_queue Queue, pi_event *Event,
                             pi_command_type CommandType,
                             ze_command_list_handle_t ZeCommandList) {
  pi_result Res = piEventCreate(Queue->Context, Event);
  if (Res != PI_SUCCESS)
    return Res;

  (*Event)->Queue = Queue;
  (*Event)->CommandType = CommandType;
  (*Event)->ZeCommandList = ZeCommandList;

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
  // release in cleanupAfterEvent.
  //
  PI_CALL(piEventRetain(*Event));

  return PI_SUCCESS;
}

pi_result _pi_device::initialize() {
  uint32_t numQueueGroups = 0;
  ZE_CALL(zeDeviceGetCommandQueueGroupProperties,
          (ZeDevice, &numQueueGroups, nullptr));
  if (numQueueGroups == 0) {
    return PI_ERROR_UNKNOWN;
  }
  std::vector<ze_command_queue_group_properties_t> QueueProperties(
      numQueueGroups);
  ZE_CALL(zeDeviceGetCommandQueueGroupProperties,
          (ZeDevice, &numQueueGroups, QueueProperties.data()));

  int ComputeGroupIndex = -1;
  for (uint32_t i = 0; i < numQueueGroups; i++) {
    if (QueueProperties[i].flags &
        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      ComputeGroupIndex = i;
      break;
    }
  }
  // How is it possible that there are no "compute" capabilities?
  if (ComputeGroupIndex < 0) {
    return PI_ERROR_UNKNOWN;
  }
  ZeComputeQueueGroupIndex = ComputeGroupIndex;
  ZeComputeQueueGroupProperties = QueueProperties[ComputeGroupIndex];

  int CopyGroupIndex = -1;
  const char *CopyEngine = std::getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE");
  bool UseCopyEngine = (!CopyEngine || (std::stoi(CopyEngine) != 0));
  if (UseCopyEngine) {
    for (uint32_t i = 0; i < numQueueGroups; i++) {
      if (((QueueProperties[i].flags &
            ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) == 0) &&
          (QueueProperties[i].flags &
           ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY)) {
        CopyGroupIndex = i;
        break;
      }
    }
    if (CopyGroupIndex < 0)
      zePrint("NOTE: blitter/copy engine is not available though it was "
              "requested\n");
    else
      zePrint("NOTE: blitter/copy engine is available\n");
  }
  ZeCopyQueueGroupIndex = CopyGroupIndex;
  if (CopyGroupIndex >= 0) {
    ZeCopyQueueGroupProperties = QueueProperties[CopyGroupIndex];
  }

  // Cache device properties
  ZeDeviceProperties = {};
  ZE_CALL(zeDeviceGetProperties, (ZeDevice, &ZeDeviceProperties));
  ZeDeviceComputeProperties = {};
  ZE_CALL(zeDeviceGetComputeProperties, (ZeDevice, &ZeDeviceComputeProperties));
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
  ze_command_queue_desc_t ZeCommandQueueDesc = {};
  ZeCommandQueueDesc.ordinal = Device->ZeComputeQueueGroupIndex;
  ZeCommandQueueDesc.index = 0;
  ZeCommandQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
  ZE_CALL(
      zeCommandListCreateImmediate,
      (ZeContext, Device->ZeDevice, &ZeCommandQueueDesc, &ZeCommandListInit));
  return PI_SUCCESS;
}

pi_result _pi_context::finalize() {
  // This function is called when pi_context is deallocated, piContextRelase.
  // There could be some memory that may have not been deallocated.
  // For example, zeEventPool could be still alive.
  std::lock_guard<std::mutex> NumEventsLiveInEventPoolGuard(
      NumEventsLiveInEventPoolMutex);
  if (ZeEventPool && NumEventsLiveInEventPool[ZeEventPool])
    ZE_CALL(zeEventPoolDestroy, (ZeEventPool));

  // Destroy the command list used for initializations
  ZE_CALL(zeCommandListDestroy, (ZeCommandListInit));

  std::lock_guard<std::mutex> Lock(ZeCommandListCacheMutex);
  for (ze_command_list_handle_t &ZeCommandList : ZeComputeCommandListCache) {
    if (ZeCommandList)
      ZE_CALL(zeCommandListDestroy, (ZeCommandList));
  }
  for (ze_command_list_handle_t &ZeCommandList : ZeCopyCommandListCache) {
    if (ZeCommandList)
      ZE_CALL(zeCommandListDestroy, (ZeCommandList));
  }

  // Adjust the number of command lists created on this platform.
  auto Platform = Devices[0]->Platform;
  Platform->ZeGlobalCommandListCount -= ZeComputeCommandListCache.size();
  Platform->ZeGlobalCommandListCount -= ZeCopyCommandListCache.size();

  return PI_SUCCESS;
}

bool _pi_queue::isInOrderQueue() const {
  // If out-of-order queue property is not set, then this is a in-order queue.
  return ((this->PiQueueProperties & PI_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ==
          0);
}

pi_result _pi_queue::resetCommandListFenceEntry(
    _pi_queue::command_list_fence_map_t::value_type &MapEntry,
    bool MakeAvailable) {
  bool UseCopyEngine = MapEntry.second.IsCopyCommandList;
  auto &ZeCommandListCache = (UseCopyEngine)
                                 ? this->Context->ZeCopyCommandListCache
                                 : this->Context->ZeComputeCommandListCache;

  // Fence had been signalled meaning the associated command-list completed.
  // Reset the fence and put the command list into a cache for reuse in PI
  // calls.
  ZE_CALL(zeFenceReset, (MapEntry.second.ZeFence));
  ZE_CALL(zeCommandListReset, (MapEntry.first));
  MapEntry.second.InUse = false;

  if (MakeAvailable) {
    std::lock_guard<std::mutex> lock(this->Context->ZeCommandListCacheMutex);
    ZeCommandListCache.push_back(MapEntry.first);
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

static const pi_uint32 ZeCommandListBatchSize = [] {
  // Default value of 0. This specifies to use dynamic batch size adjustment.
  pi_uint32 BatchSizeVal = 0;
  const auto BatchSizeStr = std::getenv("SYCL_PI_LEVEL_ZERO_BATCH_SIZE");
  if (BatchSizeStr) {
    pi_int32 BatchSizeStrVal = std::atoi(BatchSizeStr);
    // Level Zero may only support a limted number of commands per command
    // list.  The actual upper limit is not specified by the Level Zero
    // Specification.  For now we allow an arbitrary upper limit.
    // Negative numbers will be silently ignored.
    if (BatchSizeStrVal >= 0)
      BatchSizeVal = BatchSizeStrVal;
  }
  return BatchSizeVal;
}();

// This function requires a map lookup operation which can be expensive.
// TODO: Restructure code to eliminate map lookup.
bool _pi_queue::getZeCommandListIsCopyList(
    ze_command_list_handle_t ZeCommandList) {
  auto it = this->ZeCommandListFenceMap.find(ZeCommandList);
  if (it == this->ZeCommandListFenceMap.end()) {
    die("Missing command-list fence map entry");
  }
  return it->second.IsCopyCommandList;
}

// Retrieve an available command list to be used in a PI call
// Caller must hold a lock on the Queue passed in.
pi_result _pi_context::getAvailableCommandList(
    pi_queue Queue, ze_command_list_handle_t *ZeCommandList,
    ze_fence_handle_t *ZeFence, bool PreferCopyEngine, bool AllowBatching) {
  // First see if there is an command-list open for batching commands
  // for this queue.
  if (Queue->ZeOpenCommandList) {
    // TODO: Batching of copy commands will be supported.
    if (AllowBatching && !PreferCopyEngine) {
      *ZeCommandList = Queue->ZeOpenCommandList;
      *ZeFence = Queue->ZeOpenCommandListFence;
      return PI_SUCCESS;
    }

    // If this command isn't allowed to be batched, then we need to
    // go ahead and execute what is already in the batched list,
    // and then go on to process this. On exit from executeOpenCommandList
    // ZeOpenCommandList will be nullptr.
    if (auto Res = Queue->executeOpenCommandList())
      return Res;
  }
  // Use of copy engine is enabled only for out-of-order queues.
  // TODO: Revisit this when in-order queue spport is available in L0 plugin.
  bool UseCopyEngine = !(Queue->isInOrderQueue()) && PreferCopyEngine &&
                       Queue->Device->hasCopyEngine();

  // Create/Reuse the command list, because in Level Zero commands are added to
  // the command lists, and later are then added to the command queue.
  // Each command list is paired with an associated fence to track when the
  // command list is available for reuse.
  _pi_result pi_result = PI_OUT_OF_RESOURCES;
  ze_command_list_desc_t ZeCommandListDesc = {};
  ZeCommandListDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  ze_fence_desc_t ZeFenceDesc = {};
  ZeFenceDesc.stype = ZE_STRUCTURE_TYPE_FENCE_DESC;

  ZeCommandListDesc.commandQueueGroupOrdinal =
      (UseCopyEngine) ? Queue->Device->ZeCopyQueueGroupIndex
                      : Queue->Device->ZeComputeQueueGroupIndex;
  auto &ZeCommandListCache = (UseCopyEngine)
                                 ? Queue->Context->ZeCopyCommandListCache
                                 : Queue->Context->ZeComputeCommandListCache;
  auto &ZeCommandQueue = (UseCopyEngine) ? Queue->ZeCopyCommandQueue
                                         : Queue->ZeComputeCommandQueue;

  // Initally, we need to check if a command list has already been created
  // on this device that is available for use. If so, then reuse that
  // Level-Zero Command List and Fence for this PI call.
  {
    // Make sure to acquire the lock before checking the size, or there
    // will be a race condition.
    std::lock_guard<std::mutex> lock(Queue->Context->ZeCommandListCacheMutex);

    if (ZeCommandListCache.size() > 0) {
      *ZeCommandList = ZeCommandListCache.front();
      auto it = Queue->ZeCommandListFenceMap.find(*ZeCommandList);
      if (it != Queue->ZeCommandListFenceMap.end()) {
        *ZeFence = it->second.ZeFence;
        it->second.InUse = true;
      } else {
        // If there is a command list available on this device, but no
        // fence yet associated, then we must create a fence/list
        // reference for this Queue. This can happen if two Queues reuse
        // a device which did not have the resources freed.
        ZE_CALL(zeFenceCreate, (ZeCommandQueue, &ZeFenceDesc, ZeFence));
        Queue->ZeCommandListFenceMap[*ZeCommandList] = {*ZeFence, true,
                                                        UseCopyEngine};
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
  for (auto &MapEntry : Queue->ZeCommandListFenceMap) {
    // Make sure this is the command list type needed.
    if (UseCopyEngine != MapEntry.second.IsCopyCommandList)
      continue;

    ze_result_t ZeResult =
        ZE_CALL_NOCHECK(zeFenceQueryStatus, (MapEntry.second.ZeFence));
    if (ZeResult == ZE_RESULT_SUCCESS) {
      Queue->resetCommandListFenceEntry(MapEntry, false);
      *ZeCommandList = MapEntry.first;
      *ZeFence = MapEntry.second.ZeFence;
      MapEntry.second.InUse = true;
      return PI_SUCCESS;
    }
  }

  // If there are no available command lists nor signalled command lists, then
  // we must create another command list if we have not exceed the maximum
  // command lists we can create.
  // Once created, this command list & fence are added to the command list fence
  // map.
  if ((*ZeCommandList == nullptr) &&
      (Queue->Device->Platform->ZeGlobalCommandListCount <
       ZeMaxCommandListCacheSize)) {
    ZE_CALL(zeCommandListCreate,
            (Queue->Context->ZeContext, Queue->Device->ZeDevice,
             &ZeCommandListDesc, ZeCommandList));
    // Increments the total number of command lists created on this platform.
    Queue->Device->Platform->ZeGlobalCommandListCount++;
    ZE_CALL(zeFenceCreate, (ZeCommandQueue, &ZeFenceDesc, ZeFence));
    Queue->ZeCommandListFenceMap.insert(
        std::pair<ze_command_list_handle_t, _pi_queue::command_list_fence_t>(
            *ZeCommandList, {*ZeFence, false, UseCopyEngine}));
    pi_result = PI_SUCCESS;
  }

  return pi_result;
}

void _pi_queue::adjustBatchSizeForFullBatch() {
  // QueueBatchSize of 0 means never allow batching.
  if (QueueBatchSize == 0 || !UseDynamicBatching)
    return;

  NumTimesClosedFull += 1;

  // If the number of times the list has been closed early is low, and
  // the number of times it has been closed full is high, then raise
  // the batching size slowly. Don't raise it if it is already pretty
  // high.
  if (NumTimesClosedEarly <= 2 && NumTimesClosedFull > 10) {
    if (QueueBatchSize < 16) {
      QueueBatchSize = QueueBatchSize + 1;
      zePrint("Raising QueueBatchSize to %d\n", QueueBatchSize);
    }
    NumTimesClosedEarly = 0;
    NumTimesClosedFull = 0;
  }
}

void _pi_queue::adjustBatchSizeForPartialBatch(pi_uint32 PartialBatchSize) {
  // QueueBatchSize of 0 means never allow batching.
  if (QueueBatchSize == 0 || !UseDynamicBatching)
    return;

  NumTimesClosedEarly += 1;

  // If we are closing early more than about 3x the number of times
  // it is closing full, lower the batch size to the value of the
  // current open command list. This is trying to quickly get to a
  // batch size that will be able to be closed full at least once
  // in a while.
  if (NumTimesClosedEarly > (NumTimesClosedFull + 1) * 3) {
    QueueBatchSize = PartialBatchSize - 1;
    if (QueueBatchSize < 1)
      QueueBatchSize = 1;
    zePrint("Lowering QueueBatchSize to %d\n", QueueBatchSize);
    NumTimesClosedEarly = 0;
    NumTimesClosedFull = 0;
  }
}

pi_result _pi_queue::executeCommandList(ze_command_list_handle_t ZeCommandList,
                                        ze_fence_handle_t ZeFence,
                                        pi_event Event, bool IsBlocking,
                                        bool OKToBatchCommand) {
  // If the current LastCommandEvent is the nullptr, then it means
  // either that no command has ever been issued to the queue
  // or it means that the LastCommandEvent has been signalled and
  // therefore that this Queue is idle.
  bool CurrentlyEmpty = this->LastCommandEvent == nullptr;

  this->LastCommandEvent = Event;

  // Batch if allowed to, but don't batch if we know there are no kernels
  // from this queue that are currently executing.  This is intended to gets
  // kernels started as soon as possible when there are no kernels from this
  // queue awaiting execution, while allowing batching to occur when there
  // are kernels already executing. Also, if we are using fixed size batching,
  // as indicated by !UseDynamicBatching, then just ignore CurrentlyEmpty
  // as we want to strictly follow the batching the user specified.
  if (OKToBatchCommand && this->isBatchingAllowed() &&
      (!UseDynamicBatching || !CurrentlyEmpty)) {
    if (this->ZeOpenCommandList != nullptr &&
        this->ZeOpenCommandList != ZeCommandList)
      die("executeCommandList: ZeOpenCommandList should be equal to"
          "null or ZeCommandList");

    if (this->ZeOpenCommandListSize + 1 < QueueBatchSize) {
      this->ZeOpenCommandList = ZeCommandList;
      this->ZeOpenCommandListFence = ZeFence;

      // NOTE: we don't know here how many commands are in the ZeCommandList
      // but most PI interfaces translate to a single Level-Zero command.
      // Some do translate to multiple commands so we may be undercounting
      // a bit here, but this is a heuristic, not an exact measure.
      //
      this->ZeOpenCommandListSize += 1;

      return PI_SUCCESS;
    }

    adjustBatchSizeForFullBatch();

    this->ZeOpenCommandList = nullptr;
    this->ZeOpenCommandListFence = nullptr;
    this->ZeOpenCommandListSize = 0;
  }

  bool UseCopyEngine = getZeCommandListIsCopyList(ZeCommandList);
  if (UseCopyEngine)
    zePrint("Command list to be executed on copy engine\n");
  auto &ZeCommandQueue =
      (UseCopyEngine) ? ZeCopyCommandQueue : ZeComputeCommandQueue;

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

  // Close the command list and have it ready for dispatch.
  ZE_CALL(zeCommandListClose, (ZeCommandList));
  // Offload command list to the GPU for asynchronous execution
  ZE_CALL(zeCommandQueueExecuteCommandLists,
          (ZeCommandQueue, 1, &ZeCommandList, ZeFence));

  // Check global control to make every command blocking for debugging.
  if (IsBlocking || (ZeSerialize & ZeSerializeBlock) != 0) {
    // Wait until command lists attached to the command queue are executed.
    ZE_CALL(zeHostSynchronize, (ZeCommandQueue));
  }
  return PI_SUCCESS;
}

bool _pi_queue::isBatchingAllowed() {
  return (this->QueueBatchSize > 0 && ((ZeSerialize & ZeSerializeBlock) == 0));
}

pi_result _pi_queue::executeOpenCommandList() {
  // If there are any commands still in the open command list for this
  // queue, then close and execute that command list now.
  auto OpenList = this->ZeOpenCommandList;
  if (OpenList) {
    auto OpenListFence = this->ZeOpenCommandListFence;

    adjustBatchSizeForPartialBatch(this->ZeOpenCommandListSize);

    this->ZeOpenCommandList = nullptr;
    this->ZeOpenCommandListFence = nullptr;
    this->ZeOpenCommandListSize = 0;

    return executeCommandList(OpenList, OpenListFence, this->LastCommandEvent);
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
        auto ZeEvent = EventList[I]->ZeEvent;

        if (FilterEventWaitList) {
          auto Res = ZE_CALL_NOCHECK(zeEventQueryStatus, (ZeEvent));
          if (Res == ZE_RESULT_SUCCESS) {
            // Event has already completed, don't put it into the list
            continue;
          }
        }

        auto Queue = EventList[I]->Queue;

        if (Queue != CurQueue) {
          // If the event that is going to be waited on is in a
          // different queue, then any open command list in
          // that queue must be closed and executed because
          // the event being waited on could be for a command
          // in the queue's batch.

          // Lock automatically releases when this goes out of scope.
          std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

          if (auto Res = Queue->executeOpenCommandList())
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

static pi_result compileOrBuild(pi_program Program, pi_uint32 NumDevices,
                                const pi_device *DeviceList,
                                const char *Options);
static pi_result copyModule(ze_context_handle_t ZeContext,
                            ze_device_handle_t ZeDevice,
                            ze_module_handle_t SrcMod,
                            ze_module_handle_t *DestMod);

static bool setEnvVar(const char *var, const char *value);

// Forward declarations for mock implementations of Level Zero APIs that
// do not yet work in the driver.
// TODO: Remove these mock definitions when they work in the driver.
static ze_result_t
zeModuleDynamicLinkMock(uint32_t numModules, ze_module_handle_t *phModules,
                        ze_module_build_log_handle_t *phLinkLog);

static ze_result_t
zeModuleGetPropertiesMock(ze_module_handle_t hModule,
                          ze_module_properties_t *pModuleProperties);

static bool isOnlineLinkEnabled();
// End forward declarations for mock Level Zero APIs

// This function will ensure compatibility with both Linux and Windowns for
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

pi_result _pi_platform::initialize() {
  // Cache driver properties
  ze_driver_properties_t ZeDriverProperties;
  ZE_CALL(zeDriverGetProperties, (ZeDriver, &ZeDriverProperties));
  uint32_t DriverVersion = ZeDriverProperties.driverVersion;
  // Intel Level-Zero GPU driver stores version as:
  // | 31 - 24 | 23 - 16 | 15 - 0 |
  // |  Major  |  Minor  | Build  |
  auto VersionMajor = std::to_string((DriverVersion & 0xFF000000) >> 24);
  auto VersionMinor = std::to_string((DriverVersion & 0x00FF0000) >> 16);
  auto VersionBuild = std::to_string(DriverVersion & 0x0000FFFF);
  ZeDriverVersion = VersionMajor + "." + VersionMinor + "." + VersionBuild;

  ze_api_version_t ZeApiVersion;
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
    zeDriverExtensionMap[extension.name] = extension.version;
  }

  return PI_SUCCESS;
}

pi_result piPlatformsGet(pi_uint32 NumEntries, pi_platform *Platforms,
                         pi_uint32 *NumPlatforms) {

  static const char *PiTrace = std::getenv("SYCL_PI_TRACE");
  static const int PiTraceValue = PiTrace ? std::stoi(PiTrace) : 0;
  if (PiTraceValue == -1) { // Means print all PI traces
    PrintPiTrace = true;
  }

  static const char *DebugMode = std::getenv("ZE_DEBUG");
  static const int DebugModeValue = DebugMode ? std::stoi(DebugMode) : 0;
  ZeDebug = DebugModeValue;

  if (ZeDebug & ZE_DEBUG_CALL_COUNT) {
    ZeCallCount = new std::map<std::string, int>;
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

  if (NumPlatforms)
    *NumPlatforms = PiPlatformsCache->size();

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

  auto it = std::find_if(PiDevicesCache.begin(), PiDevicesCache.end(),
                         [&](std::unique_ptr<_pi_device> &D) {
                           return D.get()->ZeDevice == ZeDevice;
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
      Matched = (D->ZeDeviceProperties.type == ZE_DEVICE_TYPE_GPU);
      break;
    case PI_DEVICE_TYPE_CPU:
      Matched = (D->ZeDeviceProperties.type == ZE_DEVICE_TYPE_CPU);
      break;
    case PI_DEVICE_TYPE_ACC:
      Matched = (D->ZeDeviceProperties.type == ZE_DEVICE_TYPE_MCA ||
                 D->ZeDeviceProperties.type == ZE_DEVICE_TYPE_FPGA);
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

  uint32_t ZeAvailMemCount = 0;
  ZE_CALL(zeDeviceGetMemoryProperties, (ZeDevice, &ZeAvailMemCount, nullptr));

  // Confirm at least one memory is available in the device
  PI_ASSERT(ZeAvailMemCount > 0, PI_INVALID_VALUE);

  std::vector<ze_device_memory_properties_t> ZeDeviceMemoryProperties;
  try {
    ZeDeviceMemoryProperties.resize(ZeAvailMemCount);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  for (uint32_t I = 0; I < ZeAvailMemCount; I++) {
    ZeDeviceMemoryProperties[I] = {};
  }
  // TODO: cache various device properties in the PI device object,
  // and initialize them only upon they are first requested.
  ZE_CALL(zeDeviceGetMemoryProperties,
          (ZeDevice, &ZeAvailMemCount, ZeDeviceMemoryProperties.data()));

  ze_device_image_properties_t ZeDeviceImageProperties = {};
  ZE_CALL(zeDeviceGetImageProperties, (ZeDevice, &ZeDeviceImageProperties));

  ze_device_module_properties_t ZeDeviceModuleProperties = {};
  ZE_CALL(zeDeviceGetModuleProperties, (ZeDevice, &ZeDeviceModuleProperties));

  // TODO[1.0]: there can be multiple cache properites now, adjust.
  // For now remember the first one, if any.
  uint32_t Count = 0;
  ze_device_cache_properties_t ZeDeviceCacheProperties = {};
  ZE_CALL(zeDeviceGetCacheProperties, (ZeDevice, &Count, nullptr));
  if (Count > 0) {
    Count = 1;
    ZE_CALL(zeDeviceGetCacheProperties,
            (ZeDevice, &Count, &ZeDeviceCacheProperties));
  }

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  switch (ParamName) {
  case PI_DEVICE_INFO_TYPE: {
    switch (Device->ZeDeviceProperties.type) {
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
    return ReturnValue(pi_uint32{Device->ZeDeviceProperties.vendorId});
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
    if (ZeDeviceModuleProperties.flags & ZE_DEVICE_MODULE_FLAG_FP16)
      SupportedExtensions += ("cl_khr_fp16 ");
    if (ZeDeviceModuleProperties.flags & ZE_DEVICE_MODULE_FLAG_FP64)
      SupportedExtensions += ("cl_khr_fp64 ");
    if (ZeDeviceModuleProperties.flags & ZE_DEVICE_MODULE_FLAG_INT64_ATOMICS)
      // int64AtomicsSupported indicates support for both.
      SupportedExtensions +=
          ("cl_khr_int64_base_atomics cl_khr_int64_extended_atomics ");
    if (ZeDeviceImageProperties.maxImageDims3D > 0)
      // Supports reading and writing of images.
      SupportedExtensions += ("cl_khr_3d_image_writes ");

    return ReturnValue(SupportedExtensions.c_str());
  }
  case PI_DEVICE_INFO_NAME:
    return ReturnValue(Device->ZeDeviceProperties.name);
  case PI_DEVICE_INFO_COMPILER_AVAILABLE:
    return ReturnValue(pi_bool{1});
  case PI_DEVICE_INFO_LINKER_AVAILABLE:
    return ReturnValue(pi_bool{1});
  case PI_DEVICE_INFO_MAX_COMPUTE_UNITS: {
    pi_uint32 MaxComputeUnits =
        Device->ZeDeviceProperties.numEUsPerSubslice *
        Device->ZeDeviceProperties.numSubslicesPerSlice *
        Device->ZeDeviceProperties.numSlices;
    return ReturnValue(pi_uint32{MaxComputeUnits});
  }
  case PI_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS:
    // Level Zero spec defines only three dimensions
    return ReturnValue(pi_uint32{3});
  case PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE:
    return ReturnValue(
        pi_uint64{Device->ZeDeviceComputeProperties.maxTotalGroupSize});
  case PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES: {
    struct {
      size_t Arr[3];
    } MaxGroupSize = {{Device->ZeDeviceComputeProperties.maxGroupSizeX,
                       Device->ZeDeviceComputeProperties.maxGroupSizeY,
                       Device->ZeDeviceComputeProperties.maxGroupSizeZ}};
    return ReturnValue(MaxGroupSize);
  }
  case PI_DEVICE_INFO_MAX_CLOCK_FREQUENCY:
    return ReturnValue(pi_uint32{Device->ZeDeviceProperties.coreClockRate});
  case PI_DEVICE_INFO_ADDRESS_BITS: {
    // TODO: To confirm with spec.
    return ReturnValue(pi_uint32{64});
  }
  case PI_DEVICE_INFO_MAX_MEM_ALLOC_SIZE:
    return ReturnValue(pi_uint64{Device->ZeDeviceProperties.maxMemAllocSize});
  case PI_DEVICE_INFO_GLOBAL_MEM_SIZE: {
    uint64_t GlobalMemSize = 0;
    for (uint32_t I = 0; I < ZeAvailMemCount; I++) {
      GlobalMemSize += ZeDeviceMemoryProperties[I].totalSize;
    }
    return ReturnValue(pi_uint64{GlobalMemSize});
  }
  case PI_DEVICE_INFO_LOCAL_MEM_SIZE:
    return ReturnValue(
        pi_uint64{Device->ZeDeviceComputeProperties.maxSharedLocalMemory});
  case PI_DEVICE_INFO_IMAGE_SUPPORT:
    return ReturnValue(pi_bool{ZeDeviceImageProperties.maxImageDims1D > 0});
  case PI_DEVICE_INFO_HOST_UNIFIED_MEMORY:
    return ReturnValue(pi_bool{(Device->ZeDeviceProperties.flags &
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
    uint32_t ZeSubDeviceCount = 0;
    ZE_CALL(zeDeviceGetSubDevices, (ZeDevice, &ZeSubDeviceCount, nullptr));
    return ReturnValue(pi_uint32{ZeSubDeviceCount});
  }
  case PI_DEVICE_INFO_REFERENCE_COUNT:
    return ReturnValue(pi_uint32{Device->RefCount});
  case PI_DEVICE_INFO_PARTITION_PROPERTIES: {
    // SYCL spec says: if this SYCL device cannot be partitioned into at least
    // two sub devices then the returned vector must be empty.
    uint32_t ZeSubDeviceCount = 0;
    ZE_CALL(zeDeviceGetSubDevices, (ZeDevice, &ZeSubDeviceCount, nullptr));
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
    return ReturnValue(size_t{ZeDeviceModuleProperties.printfBufferSize});
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
    return ReturnValue(pi_bool{Device->ZeDeviceProperties.flags &
                               ZE_DEVICE_PROPERTY_FLAG_ECC});
  case PI_DEVICE_INFO_PROFILING_TIMER_RESOLUTION:
    return ReturnValue(size_t{Device->ZeDeviceProperties.timerResolution});
  case PI_DEVICE_INFO_LOCAL_MEM_TYPE:
    return ReturnValue(PI_DEVICE_LOCAL_MEM_TYPE_LOCAL);
  case PI_DEVICE_INFO_MAX_CONSTANT_ARGS:
    return ReturnValue(pi_uint32{64});
  case PI_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE:
    return ReturnValue(pi_uint64{ZeDeviceImageProperties.maxImageBufferSize});
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE:
    return ReturnValue(PI_DEVICE_MEM_CACHE_TYPE_READ_WRITE_CACHE);
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE:
    return ReturnValue(
        // TODO[1.0]: how to query cache line-size?
        pi_uint32{1});
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE:
    return ReturnValue(pi_uint64{ZeDeviceCacheProperties.cacheSize});
  case PI_DEVICE_INFO_MAX_PARAMETER_SIZE:
    return ReturnValue(size_t{ZeDeviceModuleProperties.maxArgumentsSize});
  case PI_DEVICE_INFO_MEM_BASE_ADDR_ALIGN:
    // SYCL/OpenCL spec is vague on what this means exactly, but seems to
    // be for "alignment requirement (in bits) for sub-buffer offsets."
    // An OpenCL implementation returns 8*128, but Level Zero can do just 8,
    // meaning unaligned access for values of types larger than 8 bits.
    return ReturnValue(pi_uint32{8});
  case PI_DEVICE_INFO_MAX_SAMPLERS:
    return ReturnValue(pi_uint32{ZeDeviceImageProperties.maxSamplers});
  case PI_DEVICE_INFO_MAX_READ_IMAGE_ARGS:
    return ReturnValue(pi_uint32{ZeDeviceImageProperties.maxReadImageArgs});
  case PI_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS:
    return ReturnValue(pi_uint32{ZeDeviceImageProperties.maxWriteImageArgs});
  case PI_DEVICE_INFO_SINGLE_FP_CONFIG: {
    uint64_t SingleFPValue = 0;
    ze_device_fp_flags_t ZeSingleFPCapabilities =
        ZeDeviceModuleProperties.fp32flags;
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
        ZeDeviceModuleProperties.fp16flags;
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
        ZeDeviceModuleProperties.fp64flags;
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
    return ReturnValue(size_t{ZeDeviceImageProperties.maxImageBufferSize});
  case PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE:
    return ReturnValue(size_t{ZeDeviceImageProperties.maxImageArraySlices});
  // Handle SIMD widths.
  // TODO: can we do better than this?
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR:
    return ReturnValue(Device->ZeDeviceProperties.physicalEUSimdWidth / 1);
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT:
    return ReturnValue(Device->ZeDeviceProperties.physicalEUSimdWidth / 2);
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT:
    return ReturnValue(Device->ZeDeviceProperties.physicalEUSimdWidth / 4);
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG:
    return ReturnValue(Device->ZeDeviceProperties.physicalEUSimdWidth / 8);
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT:
    return ReturnValue(Device->ZeDeviceProperties.physicalEUSimdWidth / 4);
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE:
    return ReturnValue(Device->ZeDeviceProperties.physicalEUSimdWidth / 8);
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF:
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF:
    return ReturnValue(Device->ZeDeviceProperties.physicalEUSimdWidth / 2);
  case PI_DEVICE_INFO_MAX_NUM_SUB_GROUPS: {
    // Max_num_sub_Groups = maxTotalGroupSize/min(set of subGroupSizes);
    uint32_t MinSubGroupSize =
        Device->ZeDeviceComputeProperties.subGroupSizes[0];
    for (uint32_t I = 1; I < Device->ZeDeviceComputeProperties.numSubGroupSizes;
         I++) {
      if (MinSubGroupSize > Device->ZeDeviceComputeProperties.subGroupSizes[I])
        MinSubGroupSize = Device->ZeDeviceComputeProperties.subGroupSizes[I];
    }
    return ReturnValue(Device->ZeDeviceComputeProperties.maxTotalGroupSize /
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
        Device->ZeDeviceComputeProperties.numSubGroupSizes, ParamValueSize,
        ParamValue, ParamValueSizeRet,
        Device->ZeDeviceComputeProperties.subGroupSizes);
  }
  case PI_DEVICE_INFO_IL_VERSION: {
    // Set to a space separated list of IL version strings of the form
    // <IL_Prefix>_<Major_version>.<Minor_version>.
    // "SPIR-V" is a required IL prefix when cl_khr_il_progam extension is
    // reported.
    uint32_t SpirvVersion = ZeDeviceModuleProperties.spirvVersionSupported;
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
    pi_uint64 Supported = 0;
    // TODO[1.0]: how to query for USM support now?
    if (true) {
      // TODO: Use ze_memory_access_capabilities_t
      Supported = PI_USM_ACCESS | PI_USM_ATOMIC_ACCESS |
                  PI_USM_CONCURRENT_ACCESS | PI_USM_CONCURRENT_ATOMIC_ACCESS;
    }
    return ReturnValue(Supported);
  }

    // intel extensions for GPU information
  case PI_DEVICE_INFO_PCI_ADDRESS: {
    if (getenv("ZES_ENABLE_SYSMAN") == nullptr) {
      zePrint("Set SYCL_ENABLE_PCI=1 to obtain PCI data.\n");
      return PI_INVALID_VALUE;
    }
    zes_pci_properties_t ZeDevicePciProperties = {};
    ZE_CALL(zesDevicePciGetProperties, (ZeDevice, &ZeDevicePciProperties));
    std::stringstream ss;
    ss << ZeDevicePciProperties.address.domain << ":"
       << ZeDevicePciProperties.address.bus << ":"
       << ZeDevicePciProperties.address.device << "."
       << ZeDevicePciProperties.address.function;
    return ReturnValue(ss.str().c_str());
  }
  case PI_DEVICE_INFO_GPU_EU_COUNT: {
    pi_uint32 count = Device->ZeDeviceProperties.numEUsPerSubslice *
                      Device->ZeDeviceProperties.numSubslicesPerSlice *
                      Device->ZeDeviceProperties.numSlices;
    return ReturnValue(pi_uint32{count});
  }
  case PI_DEVICE_INFO_GPU_EU_SIMD_WIDTH:
    return ReturnValue(
        pi_uint32{Device->ZeDeviceProperties.physicalEUSimdWidth});
  case PI_DEVICE_INFO_GPU_SLICES:
    return ReturnValue(pi_uint32{Device->ZeDeviceProperties.numSlices});
  case PI_DEVICE_INFO_GPU_SUBSLICES_PER_SLICE:
    return ReturnValue(
        pi_uint32{Device->ZeDeviceProperties.numSubslicesPerSlice});
  case PI_DEVICE_INFO_GPU_EU_COUNT_PER_SUBSLICE:
    return ReturnValue(pi_uint32{Device->ZeDeviceProperties.numEUsPerSubslice});
  case PI_DEVICE_INFO_MAX_MEM_BANDWIDTH:
    // currently not supported in level zero runtime
    return PI_INVALID_VALUE;

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
  PI_ASSERT(Platform, PI_INVALID_PLATFORM);

  auto ZeDevice = pi_cast<ze_device_handle_t>(NativeHandle);

  // The SYCL spec requires that the set of devices must remain fixed for the
  // duration of the application's execution. We assume that we found all of the
  // Level Zero devices when we initialized the device cache, so the
  // "NativeHandle" must already be in the cache. If it is not, this must not be
  // a valid Level Zero device.
  pi_device Dev = Platform->getDeviceFromNativeHandle(ZeDevice);
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
  ze_context_desc_t ContextDesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
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
  ze_command_queue_desc_t ZeCommandQueueDesc = {};
  ZeCommandQueueDesc.ordinal = Device->ZeComputeQueueGroupIndex;
  ZeCommandQueueDesc.index = 0;
  ZeCommandQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;

  ZE_CALL(zeCommandQueueCreate,
          (Context->ZeContext, ZeDevice,
           &ZeCommandQueueDesc, // TODO: translate properties
           &ZeComputeCommandQueue));

  // Create second queue to copy engine
  ze_command_queue_handle_t ZeCopyCommandQueue = nullptr;
  if (Device->hasCopyEngine()) {
    ZeCommandQueueDesc.ordinal = Device->ZeCopyQueueGroupIndex;
    ZE_CALL(zeCommandQueueCreate,
            (Context->ZeContext, ZeDevice,
             &ZeCommandQueueDesc, // TODO: translate properties
             &ZeCopyCommandQueue));
  }
  PI_ASSERT(Queue, PI_INVALID_QUEUE);

  try {
    *Queue = new _pi_queue(ZeComputeCommandQueue, ZeCopyCommandQueue, Context,
                           Device, ZeCommandListBatchSize, Properties);
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

  piQueueRetainNoLock(Queue);
  return PI_SUCCESS;
}

pi_result piQueueRelease(pi_queue Queue) {
  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  // We need to use a bool variable here to check the condition that
  // RefCount becomes zero atomically with PiQueueMutex lock.
  // Then, we can release the lock before we remove the Queue below.
  bool RefCountZero = false;
  {
    std::lock_guard<std::mutex> Lock(Queue->PiQueueMutex);
    Queue->RefCount--;
    if (Queue->RefCount == 0)
      RefCountZero = true;

    if (RefCountZero) {
      // It is possible to get to here and still have an open command list
      // if no wait or finish ever occurred for this queue.
      if (auto Res = Queue->executeOpenCommandList())
        return Res;

      // Make sure all commands get executed.
      ZE_CALL(zeHostSynchronize, (Queue->ZeComputeCommandQueue));
      if (Queue->ZeCopyCommandQueue)
        ZE_CALL(zeHostSynchronize, (Queue->ZeCopyCommandQueue));

      // Destroy all the fences created associated with this queue.
      for (auto &MapEntry : Queue->ZeCommandListFenceMap) {
        // This fence wasn't yet signalled when we polled it for recycling
        // the command-list, so need to release the command-list too.
        if (MapEntry.second.InUse) {
          Queue->resetCommandListFenceEntry(MapEntry, true);
        }
        ZE_CALL(zeFenceDestroy, (MapEntry.second.ZeFence));
      }
      Queue->ZeCommandListFenceMap.clear();
      ZE_CALL(zeCommandQueueDestroy, (Queue->ZeComputeCommandQueue));
      Queue->ZeComputeCommandQueue = nullptr;
      if (Queue->ZeCopyCommandQueue) {
        ZE_CALL(zeCommandQueueDestroy, (Queue->ZeCopyCommandQueue));
        Queue->ZeCopyCommandQueue = nullptr;
      }

      zePrint("piQueueRelease NumTimesClosedFull %d, NumTimesClosedEarly %d\n",
              Queue->NumTimesClosedFull, Queue->NumTimesClosedEarly);
    }
  }

  if (RefCountZero)
    delete Queue;
  return PI_SUCCESS;
}

pi_result piQueueFinish(pi_queue Queue) {
  // Wait until command lists attached to the command queue are executed.
  PI_ASSERT(Queue, PI_INVALID_QUEUE);

  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

  // execute any command list that may still be open.
  if (auto Res = Queue->executeOpenCommandList())
    return Res;

  ZE_CALL(zeHostSynchronize, (Queue->ZeComputeCommandQueue));
  if (Queue->ZeCopyCommandQueue)
    ZE_CALL(zeHostSynchronize, (Queue->ZeCopyCommandQueue));

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
                                           pi_context Context,
                                           pi_queue *Queue) {
  PI_ASSERT(Context, PI_INVALID_CONTEXT);
  PI_ASSERT(NativeHandle, PI_INVALID_VALUE);
  PI_ASSERT(Queue, PI_INVALID_QUEUE);

  auto ZeQueue = pi_cast<ze_command_queue_handle_t>(NativeHandle);

  // Attach the queue to the "0" device.
  // TODO: see if we need to let user choose the device.
  pi_device Device = Context->Devices[0];
  // TODO: see what we can do to correctly initialize PI queue for
  // compute vs. copy Level-Zero queue.
  *Queue =
      new _pi_queue(ZeQueue, nullptr, Context, Device, ZeCommandListBatchSize);
  return PI_SUCCESS;
}

pi_result piMemBufferCreate(pi_context Context, pi_mem_flags Flags, size_t Size,
                            void *HostPtr, pi_mem *RetMem,
                            const pi_mem_properties *properties) {

  // TODO: implement read-only, write-only
  if ((Flags & PI_MEM_FLAGS_ACCESS_RW) == 0) {
    die("piMemBufferCreate: Level-Zero implements only read-write buffer,"
        "no read-only or write-only yet.");
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
                            Context->Devices[0]->ZeDeviceProperties.flags &
                                ZE_DEVICE_PROPERTY_FLAG_INTEGRATED;

  if (Flags & PI_MEM_FLAGS_HOST_PTR_ALLOC) {
    // Having PI_MEM_FLAGS_HOST_PTR_ALLOC for buffer requires allocation of
    // pinned host memory, see:
    // https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/UsePinnedMemoryProperty/UsePinnedMemoryPropery.adoc
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

  pi_result Result;
  if (DeviceIsIntegrated) {
    Result = piextUSMHostAlloc(&Ptr, Context, nullptr, Size, Alignment);
  } else if (Context->SingleRootDevice) {
    // If we have a single discrete device or all devices in the context are
    // sub-devices of the same device then we can allocate on device
    Result = piextUSMDeviceAlloc(&Ptr, Context, Context->SingleRootDevice,
                                 nullptr, Size, Alignment);
  } else {
    // Context with several gpu cards. Temporarily use host allocation because
    // it is accessible by all devices. But it is not good in terms of
    // performance.
    // TODO: We need to either allow remote access to device memory using IPC,
    // or do explicit memory transfers from one device to another using host
    // resources as backing buffers to allow those transfers.
    Result = piextUSMHostAlloc(&Ptr, Context, nullptr, Size, Alignment);
  }

  if (Result != PI_SUCCESS)
    return Result;

  if (HostPtr) {
    if ((Flags & PI_MEM_FLAGS_HOST_PTR_USE) != 0 ||
        (Flags & PI_MEM_FLAGS_HOST_PTR_COPY) != 0) {
      // Initialize the buffer with user data
      if (DeviceIsIntegrated) {
        // Do a host to host copy
        memcpy(Ptr, HostPtr, Size);
      } else if (Context->SingleRootDevice) {
        // Initialize the buffer synchronously with immediate offload
        ZE_CALL(zeCommandListAppendMemoryCopy,
                (Context->ZeCommandListInit, Ptr, HostPtr, Size, nullptr, 0,
                 nullptr));
      } else {
        // Multiple root devices, do a host to host copy because we use a host
        // allocation for this case.
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
        DeviceIsIntegrated /* allocation in host memory */);
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

pi_result piMemRelease(pi_mem Mem) {
  PI_ASSERT(Mem, PI_INVALID_MEM_OBJECT);

  if (--(Mem->RefCount) == 0) {
    if (Mem->isImage()) {
      ZE_CALL(zeImageDestroy, (pi_cast<ze_image_handle_t>(Mem->getZeHandle())));
    } else {
      auto Buf = static_cast<_pi_buffer *>(Mem);
      if (!Buf->isSubBuffer()) {
        PI_CALL(piextUSMFree(Mem->Context, Mem->getZeHandle()));
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

  ze_image_desc_t ZeImageDesc = {};
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
  // are deferring it until the program is ready to be built in piProgramBuild
  // and piProgramCompile. Also it is only then we know the build options.

  try {
    *Program = new _pi_program(Context, ILBytes, Length, _pi_program::IL);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  return PI_SUCCESS;
}

pi_result piProgramCreateWithBinary(pi_context Context, pi_uint32 NumDevices,
                                    const pi_device *DeviceList,
                                    const size_t *Lengths,
                                    const unsigned char **Binaries,
                                    pi_int32 *BinaryStatus,
                                    pi_program *Program) {

  PI_ASSERT(Context, PI_INVALID_CONTEXT);
  PI_ASSERT(DeviceList && NumDevices, PI_INVALID_VALUE);
  PI_ASSERT(Binaries && Lengths, PI_INVALID_VALUE);
  PI_ASSERT(Program, PI_INVALID_PROGRAM);

  // For now we support only one device.
  if (NumDevices != 1)
    die("piProgramCreateWithBinary: level_zero supports only one device.");
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
    *Program = new _pi_program(Context, Binary, Length, _pi_program::Native);
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
    size_t SzBinary;
    if (Program->State == _pi_program::IL ||
        Program->State == _pi_program::Native) {
      SzBinary = Program->CodeLength;
    } else {
      PI_ASSERT(Program->State == _pi_program::Object ||
                    Program->State == _pi_program::Exe ||
                    Program->State == _pi_program::LinkedExe,
                PI_INVALID_OPERATION);

      // If the program is in LinkedExe state it may contain several modules.
      // We cannot handle this case because each module's contents is in its
      // own address range, discontiguous from the others.  The
      // PI_PROGRAM_INFO_BINARY_SIZES API assume the entire linked program is
      // one contiguous region, which is not the case for LinkedExe program
      // in Level Zero.  Therefore, this API is unimplemented when the Program
      // has more than one module.
      _pi_program::ModuleIterator ModIt(Program);

      PI_ASSERT(!ModIt.Done(), PI_INVALID_VALUE);

      if (ModIt.Count() > 1) {
        die("piProgramGetInfo: PI_PROGRAM_INFO_BINARY_SIZES not implemented "
            "for linked programs");
      }
      ZE_CALL(zeModuleGetNativeBinary, (*ModIt, &SzBinary, nullptr));
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
    if (Program->State == _pi_program::IL ||
        Program->State == _pi_program::Native) {
      std::memcpy(PBinary[0], Program->Code.get(), Program->CodeLength);
    } else {
      PI_ASSERT(Program->State == _pi_program::Object ||
                    Program->State == _pi_program::Exe ||
                    Program->State == _pi_program::LinkedExe,
                PI_INVALID_OPERATION);

      _pi_program::ModuleIterator ModIt(Program);

      PI_ASSERT(!ModIt.Done(), PI_INVALID_VALUE);

      if (ModIt.Count() > 1) {
        die("piProgramGetInfo: PI_PROGRAM_INFO_BINARIES not implemented for "
            "linked programs");
      }
      size_t SzBinary = 0;
      ZE_CALL(zeModuleGetNativeBinary, (*ModIt, &SzBinary, PBinary[0]));
    }
    break;
  }
  case PI_PROGRAM_INFO_NUM_KERNELS: {
    uint32_t NumKernels;
    if (Program->State == _pi_program::IL ||
        Program->State == _pi_program::Native ||
        Program->State == _pi_program::Object) {
      return PI_INVALID_PROGRAM_EXECUTABLE;
    } else {
      PI_ASSERT(Program->State == _pi_program::Exe ||
                    Program->State == _pi_program::LinkedExe,
                PI_INVALID_OPERATION);

      NumKernels = 0;
      _pi_program::ModuleIterator ModIt(Program);
      while (!ModIt.Done()) {
        uint32_t Num;
        ZE_CALL(zeModuleGetKernelNames, (*ModIt, &Num, nullptr));
        NumKernels += Num;
        ModIt++;
      }
    }
    return ReturnValue(size_t{NumKernels});
  }
  case PI_PROGRAM_INFO_KERNEL_NAMES:
    try {
      std::string PINames{""};
      if (Program->State == _pi_program::IL ||
          Program->State == _pi_program::Native ||
          Program->State == _pi_program::Object) {
        return PI_INVALID_PROGRAM_EXECUTABLE;
      } else {
        PI_ASSERT(Program->State == _pi_program::Exe ||
                      Program->State == _pi_program::LinkedExe,
                  PI_INVALID_PROGRAM_EXECUTABLE);

        bool First = true;
        _pi_program::ModuleIterator ModIt(Program);
        while (!ModIt.Done()) {
          uint32_t Count = 0;
          ZE_CALL(zeModuleGetKernelNames, (*ModIt, &Count, nullptr));
          std::unique_ptr<const char *[]> PNames(new const char *[Count]);
          ZE_CALL(zeModuleGetKernelNames, (*ModIt, &Count, PNames.get()));
          for (uint32_t I = 0; I < Count; ++I) {
            PINames += (!First ? ";" : "");
            PINames += PNames[I];
            First = false;
          }
          ModIt++;
        }
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
  (void)Options;

  // We only support one device with Level Zero currently.
  pi_device Device = Context->Devices[0];
  if (NumDevices != 1)
    die("piProgramLink: level_zero supports only one device.");

  PI_ASSERT(DeviceList && DeviceList[0] == Device, PI_INVALID_DEVICE);
  PI_ASSERT(!PFnNotify && !UserData, PI_INVALID_VALUE);

  // Validate input parameters.
  if (NumInputPrograms == 0 || InputPrograms == nullptr)
    return PI_INVALID_VALUE;
  for (pi_uint32 I = 0; I < NumInputPrograms; I++) {
    if (InputPrograms[I]->State != _pi_program::Object) {
      return PI_INVALID_OPERATION;
    }
    PI_ASSERT(InputPrograms[I]->ZeModule, PI_INVALID_VALUE);
  }

  // Linking modules on Level Zero is different from OpenCL.  With Level Zero,
  // each input object module already has native code loaded onto the device.
  // Linking two modules together causes the importing module to be changed
  // such that its native code points to an address in the exporting module.
  // As a result, a module that imports symbols can only be linked into one
  // executable at a time.  By contrast, modules that export symbols are not
  // changed, so they can be safely linked into multiple executables
  // simultaneously.
  //
  // Level Zero linking also differs from OpenCL because a link operation does
  // not create a new module that represents the linked executable.  Instead,
  // we must keep track of all the input modules and refer to the entire list
  // whenever we want to know something about the executable.

  // This vector hold the Level Zero modules that we will actually link
  // together.  This may be different from "InputPrograms" because some of
  // those modules may import symbols and already be linked into other
  // executables.  In such a case, we must make a copy of the module before we
  // can link it again.
  std::vector<_pi_program::LinkedReleaser> Inputs;
  try {
    Inputs.reserve(NumInputPrograms);

    // We do several things in this loop.
    //
    // 1. We identify any modules that need to be copied because they import
    //    symbols and are already linked into some other program.
    // 2. For any module that does not need to be copied, we bump its reference
    //    count because we will hold a reference to it.
    // 3. We create a vector of Level Zero modules, which we can pass to the
    //    zeModuleDynamicLink() API.
    std::vector<ze_module_handle_t> ZeHandles;
    ZeHandles.reserve(NumInputPrograms);
    for (pi_uint32 I = 0; I < NumInputPrograms; I++) {
      pi_program Input = InputPrograms[I];
      if (Input->HasImports) {
        std::unique_lock<std::mutex> Guard(Input->MutexHasImportsAndIsLinked);
        if (!Input->HasImportsAndIsLinked) {
          // This module imports symbols, but it isn't currently linked with
          // any other module.  Grab the flag to indicate that it is now
          // linked.
          PI_CALL(piProgramRetain(Input));
          Input->HasImportsAndIsLinked = true;
        } else {
          // This module imports symbols and is also linked with another module
          // already, so it needs to be copied.  We expect this to be quite
          // rare since linking is mostly used to link against libraries which
          // only export symbols.
          Guard.unlock();
          ze_module_handle_t ZeModule;
          pi_result res = copyModule(Context->ZeContext, Device->ZeDevice,
                                     Input->ZeModule, &ZeModule);
          if (res != PI_SUCCESS) {
            return res;
          }
          Input = new _pi_program(Input->Context, ZeModule, _pi_program::Object,
                                  Input->HasImports);
          Input->HasImportsAndIsLinked = true;
        }
      } else {
        PI_CALL(piProgramRetain(Input));
      }
      Inputs.emplace_back(Input);
      ZeHandles.push_back(Input->ZeModule);
    }

    // Link all the modules together.
    ze_module_build_log_handle_t ZeBuildLog;
    ze_result_t ZeResult =
        ZE_CALL_NOCHECK(zeModuleDynamicLinkMock,
                        (ZeHandles.size(), ZeHandles.data(), &ZeBuildLog));

    // Construct a new program object to represent the linked executable.  This
    // new object holds a reference to all the input programs.  Note that we
    // create this program object even if the link fails with "link failure"
    // because we need the new program object to hold the buid log (which has
    // the description of the failure).
    if (ZeResult == ZE_RESULT_SUCCESS ||
        ZeResult == ZE_RESULT_ERROR_MODULE_LINK_FAILURE) {
      *RetProgram = new _pi_program(Context, std::move(Inputs), ZeBuildLog);
    }
    if (ZeResult != ZE_RESULT_SUCCESS)
      return mapError(ZeResult);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  return PI_SUCCESS;
}

pi_result piProgramCompile(
    pi_program Program, pi_uint32 NumDevices, const pi_device *DeviceList,
    const char *Options, pi_uint32 NumInputHeaders,
    const pi_program *InputHeaders, const char **HeaderIncludeNames,
    void (*PFnNotify)(pi_program Program, void *UserData), void *UserData) {
  (void)NumInputHeaders;
  (void)InputHeaders;
  (void)HeaderIncludeNames;

  // The OpenCL spec says this should return CL_INVALID_PROGRAM, but there is
  // no corresponding PI error code.
  if (!Program)
    return PI_INVALID_OPERATION;

  // It's only valid to compile a program created from IL (we don't support
  // programs created from source code).
  //
  // The OpenCL spec says that the header parameters are ignored when compiling
  // IL programs, so we don't validate them.
  if (Program->State != _pi_program::IL)
    return PI_INVALID_OPERATION;

  // These aren't supported.
  PI_ASSERT(!PFnNotify && !UserData, PI_INVALID_VALUE);

  pi_result res = compileOrBuild(Program, NumDevices, DeviceList, Options);
  if (res != PI_SUCCESS)
    return res;

  Program->State = _pi_program::Object;
  return PI_SUCCESS;
}

pi_result piProgramBuild(pi_program Program, pi_uint32 NumDevices,
                         const pi_device *DeviceList, const char *Options,
                         void (*PFnNotify)(pi_program Program, void *UserData),
                         void *UserData) {

  // The OpenCL spec says this should return CL_INVALID_PROGRAM, but there is
  // no corresponding PI error code.
  PI_ASSERT(Program, PI_INVALID_PROGRAM);

  // It is legal to build a program created from either IL or from native
  // device code.
  if (Program->State != _pi_program::IL &&
      Program->State != _pi_program::Native)
    return PI_INVALID_OPERATION;

  // These aren't supported.
  PI_ASSERT(!PFnNotify && !UserData, PI_INVALID_VALUE);

  pi_result res = compileOrBuild(Program, NumDevices, DeviceList, Options);
  if (res != PI_SUCCESS)
    return res;

  Program->State = _pi_program::Exe;
  return PI_SUCCESS;
}

// Perform common operations for compiling or building a program.
static pi_result compileOrBuild(pi_program Program, pi_uint32 NumDevices,
                                const pi_device *DeviceList,
                                const char *Options) {

  if ((NumDevices && !DeviceList) || (!NumDevices && DeviceList))
    return PI_INVALID_VALUE;

  // We only support build to one device with Level Zero now.
  // TODO: we should eventually build to the possibly multiple root
  // devices in the context.
  if (NumDevices != 1)
    die("compileOrBuild: level_zero supports only one device.");

  PI_ASSERT(DeviceList, PI_INVALID_DEVICE);

  // We should have either IL or native device code.
  PI_ASSERT(Program->Code, PI_INVALID_PROGRAM);

  // Specialization constants are used only if the program was created from
  // IL.  Translate them to the Level Zero format.
  ze_module_constants_t ZeSpecConstants = {};
  std::vector<uint32_t> ZeSpecContantsIds;
  std::vector<uint64_t> ZeSpecContantsValues;
  if (Program->State == _pi_program::IL) {
    std::lock_guard<std::mutex> Guard(Program->MutexZeSpecConstants);

    ZeSpecConstants.numConstants = Program->ZeSpecConstants.size();
    ZeSpecContantsIds.reserve(ZeSpecConstants.numConstants);
    ZeSpecContantsValues.reserve(ZeSpecConstants.numConstants);

    for (auto &SpecConstant : Program->ZeSpecConstants) {
      ZeSpecContantsIds.push_back(SpecConstant.first);
      ZeSpecContantsValues.push_back(SpecConstant.second);
    }
    ZeSpecConstants.pConstantIds = ZeSpecContantsIds.data();
    ZeSpecConstants.pConstantValues = const_cast<const void **>(
        reinterpret_cast<void **>(ZeSpecContantsValues.data()));
  }

  // Ask Level Zero to build and load the native code onto the device.
  ze_module_desc_t ZeModuleDesc = {};
  ZeModuleDesc.format = (Program->State == _pi_program::IL)
                            ? ZE_MODULE_FORMAT_IL_SPIRV
                            : ZE_MODULE_FORMAT_NATIVE;
  ZeModuleDesc.inputSize = Program->CodeLength;
  ZeModuleDesc.pInputModule = Program->Code.get();
  ZeModuleDesc.pBuildFlags = Options;
  ZeModuleDesc.pConstants = &ZeSpecConstants;

  ze_device_handle_t ZeDevice = DeviceList[0]->ZeDevice;
  ze_context_handle_t ZeContext = Program->Context->ZeContext;
  ze_module_handle_t ZeModule;
  ZE_CALL(zeModuleCreate, (ZeContext, ZeDevice, &ZeModuleDesc, &ZeModule,
                           &Program->ZeBuildLog));

  // Check if this module imports any symbols, which we need to know if we
  // end up linking this module later.  See comments in piProgramLink() for
  // details.
  ze_module_properties_t ZeModuleProps;
  ZE_CALL(zeModuleGetPropertiesMock, (ZeModule, &ZeModuleProps));
  Program->HasImports = (ZeModuleProps.flags & ZE_MODULE_PROPERTY_FLAG_IMPORTS);

  // We no longer need the IL / native code.
  // The caller must set the State to Object or Exe as appropriate.
  Program->Code.reset();
  Program->ZeModule = ZeModule;
  return PI_SUCCESS;
}

pi_result piProgramGetBuildInfo(pi_program Program, pi_device Device,
                                cl_program_build_info ParamName,
                                size_t ParamValueSize, void *ParamValue,
                                size_t *ParamValueSizeRet) {
  (void)Device;

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  if (ParamName == CL_PROGRAM_BINARY_TYPE) {
    cl_program_binary_type Type = CL_PROGRAM_BINARY_TYPE_NONE;
    if (Program->State == _pi_program::Object) {
      Type = CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT;
    } else if (Program->State == _pi_program::Exe ||
               Program->State == _pi_program::LinkedExe) {
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
    // The OpenCL spec says an empty string is returned if there was no
    // previous Compile, Build, or Link.
    if (!Program->ZeBuildLog)
      return ReturnValue("");
    size_t LogSize = ParamValueSize;
    ZE_CALL(zeModuleBuildLogGetString,
            (Program->ZeBuildLog, &LogSize, pi_cast<char *>(ParamValue)));
    if (ParamValueSizeRet) {
      *ParamValueSizeRet = LogSize;
    }
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

  switch (Program->State) {
  case _pi_program::Object:
  case _pi_program::Exe:
  case _pi_program::LinkedExe: {
    _pi_program::ModuleIterator ModIt(Program);
    PI_ASSERT(!ModIt.Done(), PI_INVALID_VALUE);
    if (ModIt.Count() > 1) {
      // Programs in LinkedExe state could have several corresponding
      // Level Zero modules, so there is no right answer in this case.
      //
      // TODO: Maybe we should return PI_INVALID_OPERATION instead here?
      die("piextProgramGetNativeHandle: Not implemented for linked programs");
    }
    *ZeModule = *ModIt;
    break;
  }

  default:
    return PI_INVALID_OPERATION;
  }

  return PI_SUCCESS;
}

pi_result piextProgramCreateWithNativeHandle(pi_native_handle NativeHandle,
                                             pi_context Context,
                                             pi_program *Program) {
  PI_ASSERT(Program, PI_INVALID_PROGRAM);
  PI_ASSERT(NativeHandle, PI_INVALID_VALUE);
  PI_ASSERT(Context, PI_INVALID_CONTEXT);

  auto ZeModule = pi_cast<ze_module_handle_t>(NativeHandle);

  // We assume here that programs created from a native handle always
  // represent a fully linked executable (state Exe) and not an unlinked
  // executable (state Object).

  try {
    *Program = new _pi_program(Context, ZeModule, _pi_program::Exe);
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

  if (ZeModule) {
    ZE_CALL_NOCHECK(zeModuleDestroy, (ZeModule));
  }
}

_pi_program::LinkedReleaser::~LinkedReleaser() {
  if (Prog->HasImports) {
    std::lock_guard<std::mutex> Guard(Prog->MutexHasImportsAndIsLinked);
    if (Prog->HasImportsAndIsLinked)
      Prog->HasImportsAndIsLinked = false;
  }
  piProgramRelease(Prog);
}

// Create a copy of a Level Zero module by extracting the native code and
// creating a new module from that native code.
static pi_result copyModule(ze_context_handle_t ZeContext,
                            ze_device_handle_t ZeDevice,
                            ze_module_handle_t SrcMod,
                            ze_module_handle_t *DestMod) {
  size_t Length;
  ZE_CALL(zeModuleGetNativeBinary, (SrcMod, &Length, nullptr));

  std::unique_ptr<uint8_t[]> Code(new uint8_t[Length]);
  ZE_CALL(zeModuleGetNativeBinary, (SrcMod, &Length, Code.get()));

  ze_module_desc_t ZeModuleDesc = {};
  ZeModuleDesc.format = ZE_MODULE_FORMAT_NATIVE;
  ZeModuleDesc.inputSize = Length;
  ZeModuleDesc.pInputModule = Code.get();
  ZeModuleDesc.pBuildFlags = nullptr;
  ZeModuleDesc.pConstants = nullptr;

  ze_module_handle_t ZeModule;
  ZE_CALL(zeModuleCreate,
          (ZeContext, ZeDevice, &ZeModuleDesc, &ZeModule, nullptr));
  *DestMod = ZeModule;
  return PI_SUCCESS;
}

// TODO: Remove this mock implementation once the Level Zero driver
// implementation works.
static ze_result_t
zeModuleDynamicLinkMock(uint32_t numModules, ze_module_handle_t *phModules,
                        ze_module_build_log_handle_t *phLinkLog) {

  // If enabled, try calling the real driver API instead.  At the time this
  // code was written, the "phLinkLog" parameter to zeModuleDynamicLink()
  // doesn't work, so hard code it to NULL.
  if (isOnlineLinkEnabled()) {
    if (phLinkLog)
      *phLinkLog = nullptr;
    return ZE_CALL_NOCHECK(zeModuleDynamicLink,
                           (numModules, phModules, nullptr));
  }

  // The mock implementation can only handle the degenerate case where there
  // is only a single module that is "linked" to itself.  There is nothing to
  // do in this degenerate case.
  if (numModules > 1) {
    die("piProgramLink: Program Linking is not supported yet in Level0");
  }

  // The mock does not support the link log.
  if (phLinkLog)
    *phLinkLog = nullptr;
  return ZE_RESULT_SUCCESS;
}

// TODO: Remove this mock implementation once the Level Zero driver
// implementation works.
static ze_result_t
zeModuleGetPropertiesMock(ze_module_handle_t hModule,
                          ze_module_properties_t *pModuleProperties) {

  // If enabled, try calling the real driver API first.  At the time this code
  // was written it always returns ZE_RESULT_ERROR_UNSUPPORTED_FEATURE, so we
  // fall back to the mock in this case.
  if (isOnlineLinkEnabled()) {
    ze_result_t ZeResult =
        ZE_CALL_NOCHECK(zeModuleGetProperties, (hModule, pModuleProperties));
    if (ZeResult != ZE_RESULT_ERROR_UNSUPPORTED_FEATURE) {
      return ZeResult;
    }
  }

  // The mock implementation assumes that the module has imported symbols.
  // This is a conservative guess which may result in unnecessary calls to
  // copyModule(), but it is always correct.
  pModuleProperties->flags = ZE_MODULE_PROPERTY_FLAG_IMPORTS;
  return ZE_RESULT_SUCCESS;
}

// Returns true if we should use the Level Zero driver online linking APIs.
// At the time this code was written, these APIs exist but do not work.  We
// think that support in the DPC++ runtime is ready once the driver bugs are
// fixed, so runtime support can be enabled by setting an environment variable.
static bool isOnlineLinkEnabled() {
  static bool IsEnabled = std::getenv("SYCL_ENABLE_LEVEL_ZERO_LINK");
  return IsEnabled;
}
pi_result piKernelCreate(pi_program Program, const char *KernelName,
                         pi_kernel *RetKernel) {

  PI_ASSERT(Program, PI_INVALID_PROGRAM);
  PI_ASSERT(RetKernel, PI_INVALID_VALUE);
  PI_ASSERT(KernelName, PI_INVALID_VALUE);

  if (Program->State != _pi_program::Exe &&
      Program->State != _pi_program::LinkedExe) {
    return PI_INVALID_PROGRAM_EXECUTABLE;
  }

  ze_kernel_desc_t ZeKernelDesc = {};
  ZeKernelDesc.flags = 0;
  ZeKernelDesc.pKernelName = KernelName;

  // Search for the kernel name in each module.
  ze_kernel_handle_t ZeKernel;
  ze_result_t ZeResult = ZE_RESULT_ERROR_INVALID_KERNEL_NAME;
  _pi_program::ModuleIterator ModIt(Program);
  while (!ModIt.Done()) {
    // For a module with valid sycl kernel inside, zeKernelCreate API
    // should return ZE_RESULT_SUCCESS if target kernel is found and
    // ZE_RESULT_ERROR_INVALID_KERNEL_NAME otherwise. However, some module
    // may not include any sycl kernel such as device library modules. For such
    // modules, zeKernelCreate will return ZE_RESULT_ERROR_INVALID_ARGUMENT and
    // we should skip them.
    uint32_t KernelNum = 0;
    ZE_CALL(zeModuleGetKernelNames, (*ModIt, &KernelNum, nullptr));
    if (KernelNum != 0) {
      ZeResult =
          ZE_CALL_NOCHECK(zeKernelCreate, (*ModIt, &ZeKernelDesc, &ZeKernel));
      if (ZeResult != ZE_RESULT_ERROR_INVALID_KERNEL_NAME)
        break;
    }
    ModIt++;
  }
  if (ZeResult != ZE_RESULT_SUCCESS)
    return mapError(ZeResult);

  try {
    *RetKernel = new _pi_kernel(ZeKernel, Program);
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

  ze_kernel_properties_t ZeKernelProperties = {};
  ZE_CALL(zeKernelGetProperties, (Kernel->ZeKernel, &ZeKernelProperties));

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
    return ReturnValue(pi_uint32{ZeKernelProperties.numKernelArgs});
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

  ze_device_handle_t ZeDevice = Device->ZeDevice;
  ze_device_compute_properties_t ZeDeviceComputeProperties = {};
  ZE_CALL(zeDeviceGetComputeProperties, (ZeDevice, &ZeDeviceComputeProperties));

  ze_kernel_properties_t ZeKernelProperties = {};
  ZE_CALL(zeKernelGetProperties, (Kernel->ZeKernel, &ZeKernelProperties));

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  switch (ParamName) {
  case PI_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE: {
    // TODO: To revisit after level_zero/issues/262 is resolved
    struct {
      size_t Arr[3];
    } WorkSize = {{ZeDeviceComputeProperties.maxGroupSizeX,
                   ZeDeviceComputeProperties.maxGroupSizeY,
                   ZeDeviceComputeProperties.maxGroupSizeZ}};
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
    } WgSize = {{ZeKernelProperties.requiredGroupSizeX,
                 ZeKernelProperties.requiredGroupSizeY,
                 ZeKernelProperties.requiredGroupSizeZ}};
    return ReturnValue(WgSize);
  }
  case PI_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE:
    return ReturnValue(pi_uint32{ZeKernelProperties.localMemSize});
  case PI_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
    ze_device_properties_t ZeDeviceProperties = {};
    ZE_CALL(zeDeviceGetProperties, (ZeDevice, &ZeDeviceProperties));

    return ReturnValue(size_t{ZeDeviceProperties.physicalEUSimdWidth});
  }
  case PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE:
    return ReturnValue(pi_uint32{ZeKernelProperties.privateMemSize});
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

  ze_kernel_properties_t ZeKernelProperties;
  ZE_CALL(zeKernelGetProperties, (Kernel->ZeKernel, &ZeKernelProperties));

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  if (ParamName == PI_KERNEL_MAX_SUB_GROUP_SIZE) {
    ReturnValue(uint32_t{ZeKernelProperties.maxSubgroupSize});
  } else if (ParamName == PI_KERNEL_MAX_NUM_SUB_GROUPS) {
    ReturnValue(uint32_t{ZeKernelProperties.maxNumSubgroups});
  } else if (ParamName == PI_KERNEL_COMPILE_NUM_SUB_GROUPS) {
    ReturnValue(uint32_t{ZeKernelProperties.requiredNumSubGroups});
  } else if (ParamName == PI_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL) {
    ReturnValue(uint32_t{ZeKernelProperties.requiredSubgroupSize});
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
    // piKernelRelease is called by cleanupAfterEvent as soon as kernel
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
  ze_command_list_handle_t ZeCommandList = nullptr;
  ze_fence_handle_t ZeFence = nullptr;
  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, &ZeCommandList, &ZeFence, false /* PreferCopyEngine */,
          true /* AllowBatching */))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  pi_result Res = createEventAndAssociateQueue(
      Queue, Event, PI_COMMAND_TYPE_NDRANGE_KERNEL, ZeCommandList);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  // Save the kernel in the event, so that when the event is signalled
  // the code can do a piKernelRelease on this kernel.
  (*Event)->CommandData = (void *)Kernel;

  // Use piKernelRetain to increment the reference count and indicate
  // that the Kernel is in use. Once the event has been signalled, the
  // code in cleanupAfterEvent will do a piReleaseKernel to update
  // the reference count on the kernel, using the kernel saved
  // in CommandData.
  PI_CALL(piKernelRetain(Kernel));

  // Add the command to the command list
  ZE_CALL(zeCommandListAppendLaunchKernel,
          (ZeCommandList, Kernel->ZeKernel, &ZeThreadGroupDimensions, ZeEvent,
           (*Event)->WaitList.Length, (*Event)->WaitList.ZeEventList));

  zePrint("calling zeCommandListAppendLaunchKernel() with"
          "  ZeEvent %#lx\n",
          pi_cast<std::uintptr_t>(ZeEvent));
  printZeEventList((*Event)->WaitList);

  if (IndirectAccessTrackingEnabled)
    Queue->KernelsToBeSubmitted.push_back(Kernel);

  // Execute command list asynchronously, as the event will be used
  // to track down its completion.
  if (auto Res = Queue->executeCommandList(ZeCommandList, ZeFence, *Event,
                                           false, true))
    return Res;

  return PI_SUCCESS;
}

pi_result piextKernelCreateWithNativeHandle(pi_native_handle, pi_context, bool,
                                            pi_kernel *) {
  die("Unsupported operation");
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
pi_result piEventCreate(pi_context Context, pi_event *RetEvent) {
  size_t Index = 0;
  ze_event_pool_handle_t ZeEventPool = {};
  if (auto Res = Context->getFreeSlotInExistingOrNewPool(ZeEventPool, Index))
    return Res;

  ze_event_handle_t ZeEvent;
  ze_event_desc_t ZeEventDesc = {};
  // We have to set the SIGNAL flag as HOST scope because the
  // Level-Zero plugin implementation waits for the events to complete
  // on the host.
  ZeEventDesc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  ZeEventDesc.wait = 0;
  ZeEventDesc.index = Index;

  ZE_CALL(zeEventCreate, (ZeEventPool, &ZeEventDesc, &ZeEvent));

  try {
    PI_ASSERT(RetEvent, PI_INVALID_VALUE);

    *RetEvent =
        new _pi_event(ZeEvent, ZeEventPool, Context, PI_COMMAND_TYPE_USER);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  return PI_SUCCESS;
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
    return ReturnValue(pi_context{Event->Queue->Context});
  case PI_EVENT_INFO_COMMAND_TYPE:
    return ReturnValue(pi_cast<pi_uint64>(Event->CommandType));
  case PI_EVENT_INFO_COMMAND_EXECUTION_STATUS: {
    // Check to see if the event's Queue has an open command list due to
    // batching. If so, go ahead and close and submit it, because it is
    // possible that this is trying to query some event's status that
    // is part of the batch.  This isn't strictly required, but it seems
    // like a reasonable thing to do.
    {
      // Lock automatically releases when this goes out of scope.
      std::lock_guard<std::mutex> lock(Event->Queue->PiQueueMutex);

      // Only do the execute of the open command list if the event that
      // is being queried and event that is to be signalled by something
      // currently in that open command list.
      if (Event->Queue->ZeOpenCommandList == Event->ZeCommandList) {
        if (auto Res = Event->Queue->executeOpenCommandList())
          return Res;
      }
    }

    ze_result_t ZeResult;
    ZeResult = ZE_CALL_NOCHECK(zeEventQueryStatus, (Event->ZeEvent));
    if (ZeResult == ZE_RESULT_SUCCESS) {
      return getInfo(ParamValueSize, ParamValue, ParamValueSizeRet,
                     pi_int32{CL_COMPLETE}); // Untie from OpenCL
    }
    // TODO: We don't know if the status is queueed, submitted or running.
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
      Event->Queue->Device->ZeDeviceProperties.timerResolution;

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  ze_kernel_timestamp_result_t tsResult;

  switch (ParamName) {
  case PI_PROFILING_INFO_COMMAND_START: {
    ZE_CALL(zeEventQueryKernelTimestamp, (Event->ZeEvent, &tsResult));

    uint64_t ContextStartTime = tsResult.context.kernelStart;
    ContextStartTime *= ZeTimerResolution;

    return ReturnValue(uint64_t{ContextStartTime});
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
          (1LL << Device->ZeDeviceProperties.kernelTimestampValidBits) - 1;
      ContextEndTime += TimestampMaxValue - ContextStartTime;
    }
    ContextEndTime *= ZeTimerResolution;

    return ReturnValue(uint64_t{ContextEndTime});
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
static pi_result cleanupAfterEvent(pi_event Event) {
  // The implementation of this is slightly tricky.  The same event
  // can be referred to by multiple threads, so it is possible to
  // have a race condition between the read of fields of the event,
  // and reseting those fields in some other thread.
  // But, since the event is uniquely associated with the queue
  // for the event, we use the locking that we already have to do on the
  // queue to also serve as the thread safety mechanism for the
  // any of the Event's data members that need to be read/reset as
  // part of the cleanup operations.
  {
    auto Queue = Event->Queue;

    // Lock automatically releases when this goes out of scope.
    std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

    // Cleanup the command list associated with the event if it hasn't
    // been cleaned up already.
    auto EventCommandList = Event->ZeCommandList;

    if (EventCommandList) {
      // Event has been signalled: If the fence for the associated command list
      // is signalled, then reset the fence and command list and add them to the
      // available list for reuse in PI calls.
      if (Queue->RefCount > 0) {
        auto it = Queue->ZeCommandListFenceMap.find(EventCommandList);
        if (it == Queue->ZeCommandListFenceMap.end()) {
          die("Missing command-list completition fence");
        }
        ze_result_t ZeResult =
            ZE_CALL_NOCHECK(zeFenceQueryStatus, (it->second.ZeFence));
        if (ZeResult == ZE_RESULT_SUCCESS) {
          Queue->resetCommandListFenceEntry(*it, true);
          Event->ZeCommandList = nullptr;
        }
      }
    }

    // Release the kernel associated with this event if there is one.
    if (Event->CommandType == PI_COMMAND_TYPE_NDRANGE_KERNEL &&
        Event->CommandData) {
      PI_CALL(piKernelRelease(pi_cast<pi_kernel>(Event->CommandData)));
      Event->CommandData = nullptr;
    }

    // If this event was the LastCommandEvent in the queue, being used
    // to make sure that commands were executed in-order, remove this.
    // If we don't do this, the event can get released and freed leaving
    // a dangling pointer to this event.  It could also cause unneeded
    // already finished events to show up in the wait list.
    if (Queue->LastCommandEvent == Event) {
      Queue->LastCommandEvent = nullptr;
    }

    if (!Event->CleanedUp) {
      Event->CleanedUp = true;
      // Release this event since we explicitly retained it on creation.
      // NOTE: that this needs to be done only once for an event so
      // this is guarded with the CleanedUp flag.
      //
      PI_CALL(piEventRelease(Event));
    }
  }

  // Make a list of all the dependent events that must have signalled
  // because this event was dependent on them.  This list will be appended
  // to as we walk it so that this algorithm doesn't go recursive
  // due to dependent events themselves being dependent on other events
  // forming a potentially very deep tree, and deep recursion.  That
  // turned out to be a significant problem with the recursive code
  // that preceded this implementation.

  std::list<pi_event> EventsToBeReleased;

  Event->WaitList.collectEventsForReleaseAndDestroyPiZeEventList(
      EventsToBeReleased);

  while (!EventsToBeReleased.empty()) {
    pi_event DepEvent = EventsToBeReleased.front();
    EventsToBeReleased.pop_front();

    DepEvent->WaitList.collectEventsForReleaseAndDestroyPiZeEventList(
        EventsToBeReleased);
    if (IndirectAccessTrackingEnabled) {
      // DepEvent has finished, we can release the associated kernel if there is
      // one. This is the earliest place we can do this and it can't be done
      // twice, so it is safe. Lock automatically releases when this goes out of
      // scope.
      // TODO: this code needs to be moved out of the guard.
      std::lock_guard<std::mutex> lock(DepEvent->Queue->PiQueueMutex);
      if (DepEvent->CommandType == PI_COMMAND_TYPE_NDRANGE_KERNEL &&
          DepEvent->CommandData) {
        PI_CALL(piKernelRelease(pi_cast<pi_kernel>(DepEvent->CommandData)));
        DepEvent->CommandData = nullptr;
      }
    }
    PI_CALL(piEventRelease(DepEvent));
  }

  return PI_SUCCESS;
}

pi_result piEventsWait(pi_uint32 NumEvents, const pi_event *EventList) {

  if (NumEvents && !EventList) {
    return PI_INVALID_EVENT;
  }

  // Submit dependent open command lists for execution, if any
  for (uint32_t I = 0; I < NumEvents; I++) {
    auto Queue = EventList[I]->Queue;

    // Lock automatically releases when this goes out of scope.
    std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

    if (Queue->RefCount > 0) {
      if (auto Res = Queue->executeOpenCommandList())
        return Res;
    }
  }

  for (uint32_t I = 0; I < NumEvents; I++) {
    ze_event_handle_t ZeEvent = EventList[I]->ZeEvent;
    zePrint("ZeEvent = %#lx\n", pi_cast<std::uintptr_t>(ZeEvent));
    ZE_CALL(zeHostSynchronize, (ZeEvent));

    // NOTE: we are cleaning up after the event here to free resources
    // sooner in case run-time is not calling piEventRelease soon enough.
    cleanupAfterEvent(EventList[I]);
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
  PI_ASSERT(Event, PI_INVALID_EVENT);
  if (!Event->RefCount) {
    die("piEventRelease: called on a destroyed event");
  }

  if (--(Event->RefCount) == 0) {
    cleanupAfterEvent(Event);

    if (Event->CommandType == PI_COMMAND_TYPE_MEM_BUFFER_UNMAP &&
        Event->CommandData) {
      // Free the memory allocated in the piEnqueueMemBufferMap.
      // TODO: always use piextUSMFree
      if (IndirectAccessTrackingEnabled) {
        // Use the version with reference counting
        PI_CALL(piextUSMFree(Event->Queue->Context, Event->CommandData));
      } else {
        ZE_CALL(zeMemFree,
                (Event->Queue->Context->ZeContext, Event->CommandData));
      }
      Event->CommandData = nullptr;
    }
    ZE_CALL(zeEventDestroy, (Event->ZeEvent));

    auto Context = Event->Context;
    if (auto Res = Context->decrementAliveEventsInPool(Event->ZeEventPool))
      return Res;

    // We intentionally incremented the reference counter when an event is
    // created so that we can avoid pi_queue is released before the associated
    // pi_event is released. Here we have to decrement it so pi_queue
    // can be released successfully.
    PI_CALL(piQueueRelease(Event->Queue));
    delete Event;
  }
  return PI_SUCCESS;
}

pi_result piextEventGetNativeHandle(pi_event Event,
                                    pi_native_handle *NativeHandle) {
  (void)Event;
  (void)NativeHandle;
  die("piextEventGetNativeHandle: not supported");
  return PI_SUCCESS;
}

pi_result piextEventCreateWithNativeHandle(pi_native_handle NativeHandle,
                                           pi_event *Event) {
  (void)NativeHandle;
  (void)Event;
  die("piextEventCreateWithNativeHandle: not supported");
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
  ze_sampler_desc_t ZeSamplerDesc = {};

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

        // TODO: add support for PI_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE
        switch (CurValueAddressingMode) {
        case PI_SAMPLER_ADDRESSING_MODE_NONE:
          ZeSamplerDesc.addressMode = ZE_SAMPLER_ADDRESS_MODE_NONE;
          break;
        case PI_SAMPLER_ADDRESSING_MODE_REPEAT:
          ZeSamplerDesc.addressMode = ZE_SAMPLER_ADDRESS_MODE_REPEAT;
          break;
        case PI_SAMPLER_ADDRESSING_MODE_CLAMP:
          ZeSamplerDesc.addressMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP;
          break;
        case PI_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE:
          ZeSamplerDesc.addressMode = ZE_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
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
    ze_command_list_handle_t ZeCommandList = nullptr;
    ze_fence_handle_t ZeFence = nullptr;
    if (auto Res = Queue->Context->getAvailableCommandList(
            Queue, &ZeCommandList, &ZeFence))
      return Res;

    ze_event_handle_t ZeEvent = nullptr;
    auto Res = createEventAndAssociateQueue(Queue, Event, PI_COMMAND_TYPE_USER,
                                            ZeCommandList);
    if (Res != PI_SUCCESS)
      return Res;
    ZeEvent = (*Event)->ZeEvent;
    (*Event)->WaitList = TmpWaitList;

    const auto &WaitList = (*Event)->WaitList;
    ZE_CALL(zeCommandListAppendWaitOnEvents,
            (ZeCommandList, WaitList.Length, WaitList.ZeEventList));

    ZE_CALL(zeCommandListAppendSignalEvent, (ZeCommandList, ZeEvent));

    // Execute command list asynchronously as the event will be used
    // to track down its completion.
    return Queue->executeCommandList(ZeCommandList, ZeFence, *Event);
  }

  // If wait-list is empty, then this particular command should wait until
  // all previous enqueued commands to the command-queue have completed.
  //
  // TODO: find a way to do that without blocking the host.

  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

  auto Res =
      createEventAndAssociateQueue(Queue, Event, PI_COMMAND_TYPE_USER, nullptr);
  if (Res != PI_SUCCESS)
    return Res;

  ZE_CALL(zeHostSynchronize, (Queue->ZeComputeCommandQueue));
  if (Queue->ZeCopyCommandQueue)
    ZE_CALL(zeHostSynchronize, (Queue->ZeCopyCommandQueue));

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

  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

  _pi_ze_event_list_t TmpWaitList;
  if (auto Res = TmpWaitList.createAndRetainPiZeEventList(NumEventsInWaitList,
                                                          EventWaitList, Queue))
    return Res;

  // Get a new command list to be used on this call
  ze_command_list_handle_t ZeCommandList = nullptr;
  ze_fence_handle_t ZeFence = nullptr;
  if (auto Res = Queue->Context->getAvailableCommandList(Queue, &ZeCommandList,
                                                         &ZeFence))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  auto Res = createEventAndAssociateQueue(Queue, Event, PI_COMMAND_TYPE_USER,
                                          ZeCommandList);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

  ZE_CALL(zeCommandListAppendBarrier,
          (ZeCommandList, ZeEvent, (*Event)->WaitList.Length,
           (*Event)->WaitList.ZeEventList));

  // Execute command list asynchronously as the event will be used
  // to track down its completion.
  return Queue->executeCommandList(ZeCommandList, ZeFence, *Event);
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
  ze_command_list_handle_t ZeCommandList = nullptr;
  ze_fence_handle_t ZeFence = nullptr;
  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, &ZeCommandList, &ZeFence, PreferCopyEngine))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  auto Res =
      createEventAndAssociateQueue(Queue, Event, CommandType, ZeCommandList);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

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

  if (auto Res = Queue->executeCommandList(ZeCommandList, ZeFence, *Event,
                                           BlockingWrite))
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
  ze_command_list_handle_t ZeCommandList = nullptr;
  ze_fence_handle_t ZeFence = nullptr;
  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, &ZeCommandList, &ZeFence, PreferCopyEngine))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  auto Res =
      createEventAndAssociateQueue(Queue, Event, CommandType, ZeCommandList);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

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

  if (auto Res =
          Queue->executeCommandList(ZeCommandList, ZeFence, *Event, Blocking))
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

  // Get a new command list to be used on this call
  ze_command_list_handle_t ZeCommandList = nullptr;
  ze_fence_handle_t ZeFence = nullptr;

  // TODO: Fill operations on copy engine fails to fill a buffer at expected
  // offset. Perform fill operations on compute engine for now.
  // PreferCopyEngine will be initialized with 'true' once issue is resolved.
  bool PreferCopyEngine = false;
  size_t MaxPatternSize =
      Queue->Device->ZeComputeQueueGroupProperties.maxMemoryFillPatternSize;

  // Performance analysis on a simple SYCL data "fill" test shows copy engine
  // is faster than compute engine for such operations.
  //
  // Make sure that pattern size matches the capability of the copy queue.
  //
  if (PreferCopyEngine && Queue->Device->hasCopyEngine() &&
      PatternSize <=
          Queue->Device->ZeCopyQueueGroupProperties.maxMemoryFillPatternSize) {
    MaxPatternSize =
        Queue->Device->ZeCopyQueueGroupProperties.maxMemoryFillPatternSize;
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

  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, &ZeCommandList, &ZeFence, PreferCopyEngine))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  auto Res =
      createEventAndAssociateQueue(Queue, Event, CommandType, ZeCommandList);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

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
  if (auto Res = Queue->executeCommandList(ZeCommandList, ZeFence, *Event))
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

  // For integrated devices we don't need a commandlist
  ze_command_list_handle_t ZeCommandList = nullptr;
  ze_fence_handle_t ZeFence = nullptr;
  ze_event_handle_t ZeEvent = nullptr;

  {
    // Lock automatically releases when this goes out of scope.
    std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

    _pi_ze_event_list_t TmpWaitList;
    if (auto Res = TmpWaitList.createAndRetainPiZeEventList(
            NumEventsInWaitList, EventWaitList, Queue))
      return Res;

    auto Res = createEventAndAssociateQueue(
        Queue, Event, PI_COMMAND_TYPE_MEM_BUFFER_MAP, ZeCommandList);
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
      if (!(MapFlags & PI_MAP_WRITE_INVALIDATE_REGION))
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
  if (auto Res = Queue->Context->getAvailableCommandList(Queue, &ZeCommandList,
                                                         &ZeFence))
    return Res;

  // Set the commandlist in the event
  if (Event) {
    (*Event)->ZeCommandList = ZeCommandList;
  }

  if (Buffer->MapHostPtr) {
    *RetMap = Buffer->MapHostPtr + Offset;
  } else {
    // TODO: always use piextUSMHostAlloc
    if (IndirectAccessTrackingEnabled) {
      // Use the version with reference counting
      PI_CALL(piextUSMHostAlloc(RetMap, Queue->Context, nullptr, Size, 1));
    } else {
      ze_host_mem_alloc_desc_t ZeDesc = {};
      ZeDesc.flags = 0;

      ZE_CALL(zeMemAllocHost,
              (Queue->Context->ZeContext, &ZeDesc, Size, 1, RetMap));
    }
  }

  const auto &WaitList = (*Event)->WaitList;

  if (WaitList.Length) {
    ZE_CALL(zeCommandListAppendWaitOnEvents,
            (ZeCommandList, WaitList.Length, WaitList.ZeEventList));
  }

  ZE_CALL(zeCommandListAppendMemoryCopy,
          (ZeCommandList, *RetMap,
           pi_cast<char *>(Buffer->getZeHandle()) + Offset, Size, ZeEvent, 0,
           nullptr));

  if (auto Res = Queue->executeCommandList(ZeCommandList, ZeFence, *Event,
                                           BlockingMap))
    return Res;

  return Buffer->addMapping(*RetMap, Offset, Size);
}

pi_result piEnqueueMemUnmap(pi_queue Queue, pi_mem MemObj, void *MappedPtr,
                            pi_uint32 NumEventsInWaitList,
                            const pi_event *EventWaitList, pi_event *Event) {
  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  PI_ASSERT(Event, PI_INVALID_EVENT);

  // Integrated devices don't need a command list.
  // If discrete we will get a commandlist later.
  ze_command_list_handle_t ZeCommandList = nullptr;
  ze_fence_handle_t ZeFence = nullptr;

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

    auto Res = createEventAndAssociateQueue(
        Queue, Event, PI_COMMAND_TYPE_MEM_BUFFER_UNMAP, ZeCommandList);
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

  if (auto Res = Queue->Context->getAvailableCommandList(Queue, &ZeCommandList,
                                                         &ZeFence))
    return Res;

  // Set the commandlist in the event
  (*Event)->ZeCommandList = ZeCommandList;

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
  if (auto Res = Queue->executeCommandList(ZeCommandList, ZeFence, *Event))
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
  ze_command_list_handle_t ZeCommandList = nullptr;
  ze_fence_handle_t ZeFence = nullptr;
  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, &ZeCommandList, &ZeFence, PreferCopyEngine))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  auto Res =
      createEventAndAssociateQueue(Queue, Event, CommandType, ZeCommandList);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;
  (*Event)->WaitList = TmpWaitList;

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

  if (auto Res =
          Queue->executeCommandList(ZeCommandList, ZeFence, *Event, IsBlocking))
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

// TODO: Check if the function_pointer_ret type can be converted to void**.
pi_result piextGetDeviceFunctionPointer(pi_device Device, pi_program Program,
                                        const char *FunctionName,
                                        pi_uint64 *FunctionPointerRet) {
  (void)Device;
  PI_ASSERT(Program, PI_INVALID_PROGRAM);

  if (Program->State != _pi_program::Exe &&
      Program->State != _pi_program::LinkedExe) {
    return PI_INVALID_PROGRAM_EXECUTABLE;
  }

  // Search for the function name in each module.
  ze_result_t ZeResult = ZE_RESULT_ERROR_INVALID_FUNCTION_NAME;
  _pi_program::ModuleIterator ModIt(Program);
  while (!ModIt.Done()) {
    ZeResult = ZE_CALL_NOCHECK(
        zeModuleGetFunctionPointer,
        (*ModIt, FunctionName, reinterpret_cast<void **>(FunctionPointerRet)));
    if (ZeResult != ZE_RESULT_ERROR_INVALID_FUNCTION_NAME)
      break;
    ModIt++;
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
  ze_device_mem_alloc_desc_t ZeDesc = {};
  ZeDesc.flags = 0;
  ZeDesc.ordinal = 0;

  ze_relaxed_allocation_limits_exp_desc_t RelaxedDesc = {};
  if (Size > Device->ZeDeviceProperties.maxMemAllocSize) {
    // Tell Level-Zero to accept Size > maxMemAllocSize
    RelaxedDesc.stype = ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC;
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
  ze_host_mem_alloc_desc_t ZeHostDesc = {};
  ZeHostDesc.flags = 0;
  ze_device_mem_alloc_desc_t ZeDevDesc = {};
  ZeDevDesc.flags = 0;
  ZeDevDesc.ordinal = 0;

  ze_relaxed_allocation_limits_exp_desc_t RelaxedDesc = {};
  if (Size > Device->ZeDeviceProperties.maxMemAllocSize) {
    // Tell Level-Zero to accept Size > maxMemAllocSize
    RelaxedDesc.stype = ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC;
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
  ze_host_mem_alloc_desc_t ZeHostDesc = {};
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

pi_result piextUSMDeviceAlloc(void **ResultPtr, pi_context Context,
                              pi_device Device,
                              pi_usm_mem_properties *Properties, size_t Size,
                              pi_uint32 Alignment) {
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
  ze_memory_allocation_properties_t ZeMemoryAllocationProperties = {};

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

  if (ZeDeviceHandle) {
    // All devices in the context are of the same platform.
    auto Platform = Context->Devices[0]->Platform;
    auto Device = Platform->getDeviceFromNativeHandle(ZeDeviceHandle);

    PI_ASSERT(Device, PI_INVALID_DEVICE);

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
  ze_memory_allocation_properties_t ZeMemoryAllocationProperties = {};

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
  PI_ASSERT(!(Flags & ~PI_USM_MIGRATION_TBD0), PI_INVALID_VALUE);
  PI_ASSERT(Queue, PI_INVALID_QUEUE);
  PI_ASSERT(Event, PI_INVALID_EVENT);

  // Lock automatically releases when this goes out of scope.
  std::lock_guard<std::mutex> lock(Queue->PiQueueMutex);

  _pi_ze_event_list_t TmpWaitList;
  if (auto Res = TmpWaitList.createAndRetainPiZeEventList(NumEventsInWaitList,
                                                          EventWaitList, Queue))
    return Res;

  // Get a new command list to be used on this call
  ze_command_list_handle_t ZeCommandList = nullptr;
  ze_fence_handle_t ZeFence = nullptr;
  // TODO: Change PreferCopyEngine argument to 'true' once L0 backend
  // support is added
  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, &ZeCommandList, &ZeFence, false /* PreferCopyEngine */))
    return Res;

  // TODO: do we need to create a unique command type for this?
  ze_event_handle_t ZeEvent = nullptr;
  auto Res = createEventAndAssociateQueue(Queue, Event, PI_COMMAND_TYPE_USER,
                                          ZeCommandList);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;

  if (auto Res = (*Event)->WaitList.createAndRetainPiZeEventList(
          NumEventsInWaitList, EventWaitList, Queue))
    return Res;

  const auto &WaitList = (*Event)->WaitList;

  if (WaitList.Length) {
    ZE_CALL(zeCommandListAppendWaitOnEvents,
            (ZeCommandList, WaitList.Length, WaitList.ZeEventList));
  }
  // TODO: figure out how to translate "flags"
  ZE_CALL(zeCommandListAppendMemoryPrefetch, (ZeCommandList, Ptr, Size));

  // TODO: Level Zero does not have a completion "event" with the prefetch API,
  // so manually add command to signal our event.
  ZE_CALL(zeCommandListAppendSignalEvent, (ZeCommandList, ZeEvent));

  if (auto Res =
          Queue->executeCommandList(ZeCommandList, ZeFence, *Event, false))
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
  ze_command_list_handle_t ZeCommandList = nullptr;
  ze_fence_handle_t ZeFence = nullptr;
  // PreferCopyEngine is set to 'false' here.
  // TODO: Additional analysis is required to check if this operation will
  // run faster on copy engines.
  if (auto Res = Queue->Context->getAvailableCommandList(
          Queue, &ZeCommandList, &ZeFence, false /* PreferCopyEngine */))
    return Res;

  // TODO: do we need to create a unique command type for this?
  ze_event_handle_t ZeEvent = nullptr;
  auto Res = createEventAndAssociateQueue(Queue, Event, PI_COMMAND_TYPE_USER,
                                          ZeCommandList);
  if (Res != PI_SUCCESS)
    return Res;
  ZeEvent = (*Event)->ZeEvent;

  if (auto Res =
          (*Event)->WaitList.createAndRetainPiZeEventList(0, nullptr, Queue))
    return Res;

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

  Queue->executeCommandList(ZeCommandList, ZeFence, *Event, false);
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
  ze_memory_allocation_properties_t ZeMemoryAllocationProperties = {};

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
  // Level Zero sets spec constants when creating modules,
  // so save them for when program is built.
  std::lock_guard<std::mutex> Guard(Prog->MutexZeSpecConstants);

  // Pass SpecValue pointer. Spec constant value is retrieved
  // by Level Zero when creating the module.
  //
  // NOTE: SpecSize is unused in Level Zero, the size is known from SPIR-V by
  // SpecID.
  Prog->ZeSpecConstants[SpecID] = reinterpret_cast<uint64_t>(SpecValue);

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
    std::vector<std::vector<std::string>> CreateDestroySet = {
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
        const auto &ZeName = I->c_str();
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

      if (diff)
        fprintf(stderr, " ---> LEAK = %d", diff);
      fprintf(stderr, "\n");
    }

    ZeCallCount->clear();
    delete ZeCallCount;
    ZeCallCount = nullptr;
  }
  return PI_SUCCESS;
}

} // extern "C"
