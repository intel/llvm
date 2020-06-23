//===----------- pi_level0.cpp - Level Zero Plugin--------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file pi_level0.cpp
/// Implementation of Level Zero Plugin.
///
/// \ingroup sycl_pi_level0

#include "pi_level0.hpp"
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <level_zero/zet_api.h>

namespace {

// Controls L0 calls serialization to w/a L0 driver being not MT ready.
// Recognized values (can be used as a bit mask):
enum {
  ZeSerializeNone =
      0, // no locking or blocking (except when SYCL RT requested blocking)
  ZeSerializeLock = 1, // locking around each ZE_CALL
  ZeSerializeBlock =
      2, // blocking ZE calls, where supported (usually in enqueue commands)
};
static pi_uint32 ZeSerialize = 0;

// This class encapsulates actions taken along with a call to L0 API.
class ZeCall {
private:
  // The global mutex that is used for total serialization of L0 calls.
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
  ze_result_t doCall(ze_result_t ZeResult, const char *CallStr,
                     bool TraceError = true);
};
std::mutex ZeCall::GlobalLock;

// Controls L0 calls tracing in zePrint.
static bool ZeDebug = false;

static void zePrint(const char *Format, ...) {
  if (ZeDebug) {
    va_list Args;
    va_start(Args, Format);
    vfprintf(stderr, Format, Args);
    va_end(Args);
  }
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

ze_result_t
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
    return ZE_RESULT_ERROR_INVALID_SIZE;
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

    ze_event_pool_desc_t ZeEventPoolDesc;
    ZeEventPoolDesc.count = MaxNumEventsPerPool;
    ZeEventPoolDesc.flags = ZE_EVENT_POOL_FLAG_TIMESTAMP;
    ZeEventPoolDesc.version = ZE_EVENT_POOL_DESC_VERSION_CURRENT;

    ze_device_handle_t ZeDevice = Device->ZeDevice;
    if (ze_result_t ZeRes =
            zeEventPoolCreate(Device->Platform->ZeDriver, &ZeEventPoolDesc, 1,
                              &ZeDevice, &ZeEventPool))
      return ZeRes;
    NumEventsAvailableInEventPool[ZeEventPool] = MaxNumEventsPerPool - 1;
    NumEventsLiveInEventPool[ZeEventPool] = MaxNumEventsPerPool;
  } else {
    std::lock_guard<std::mutex> NumEventsAvailableInEventPoolGuard(
        NumEventsAvailableInEventPoolMutex);
    Index = MaxNumEventsPerPool - NumEventsAvailableInEventPool[ZeEventPool];
    --NumEventsAvailableInEventPool[ZeEventPool];
  }
  ZePool = ZeEventPool;
  return ZE_RESULT_SUCCESS;
}

ze_result_t
_pi_context::decrementAliveEventsInPool(ze_event_pool_handle_t ZePool) {
  std::lock_guard<std::mutex> Lock(NumEventsLiveInEventPoolMutex);
  --NumEventsLiveInEventPool[ZePool];
  if (NumEventsLiveInEventPool[ZePool] == 0) {
    return zeEventPoolDestroy(ZePool);
  }
  return ZE_RESULT_SUCCESS;
}

// Some opencl extensions we know are supported by all Level0 devices.
constexpr char ZE_SUPPORTED_EXTENSIONS[] =
    "cl_khr_il_program cl_khr_subgroups cl_intel_subgroups "
    "cl_intel_subgroups_short cl_intel_required_subgroup_size ";

// Map L0 runtime error code to PI error code
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
      {ZE_RESULT_ERROR_MODULE_BUILD_FAILURE, PI_BUILD_PROGRAM_FAILURE}};
  auto It = ErrorMapping.find(ZeResult);
  if (It == ErrorMapping.end()) {
    return PI_ERROR_UNKNOWN;
  }
  return It->second;
}

// Forward declarations
static pi_result
enqueueMemCopyHelper(pi_command_type CommandType, pi_queue Queue, void *Dst,
                     pi_bool BlockingWrite, size_t Size, const void *Src,
                     pi_uint32 NumEventsInWaitList,
                     const pi_event *EventWaitList, pi_event *Event);

static pi_result enqueueMemCopyRectHelper(
    pi_command_type CommandType, pi_queue Queue, void *SrcBuffer,
    void *DstBuffer, const size_t *SrcOrigin, const size_t *DstOrigin,
    const size_t *Region, size_t SrcRowPitch, size_t SrcSlicePitch,
    size_t DstRowPitch, size_t DstSlicePitch, pi_bool Blocking,
    pi_uint32 NumEventsInWaitList, const pi_event *EventWaitList,
    pi_event *Event);

inline void zeParseError(ze_result_t ZeError, std::string &ErrorString) {
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
    assert("Unexpected Error code");
  } // switch
}

ze_result_t ZeCall::doCall(ze_result_t ZeResult, const char *CallStr,
                           bool TraceError) {
  zePrint("ZE ---> %s\n", CallStr);

  if (ZeResult && TraceError) {
    std::string ErrorString;
    zeParseError(ZeResult, ErrorString);
    zePrint("Error (%s) in %s\n", ErrorString.c_str(), CallStr);
  }
  return ZeResult;
}

#define ZE_CALL(Call)                                                          \
  if (auto Result = ZeCall().doCall(Call, #Call, true))                        \
    return mapError(Result);
#define ZE_CALL_NOCHECK(Call) ZeCall().doCall(Call, #Call, false)

pi_result _pi_device::initialize() {
  // Create the immediate command list to be used for initializations
  // Created as synchronous so level-zero performs implicit synchronization and
  // there is no need to query for completion in the plugin
  ze_command_queue_desc_t ZeCommandQueueDesc = {};
  ZeCommandQueueDesc.version = ZE_COMMAND_QUEUE_DESC_VERSION_CURRENT;
  ZeCommandQueueDesc.ordinal = 0;
  ZeCommandQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
  ZE_CALL(zeCommandListCreateImmediate(ZeDevice, &ZeCommandQueueDesc,
                                       &ZeCommandListInit));
  // Cache device properties
  ZeDeviceProperties.version = ZE_DEVICE_PROPERTIES_VERSION_CURRENT;
  ZE_CALL(zeDeviceGetProperties(ZeDevice, &ZeDeviceProperties));
  ZeDeviceComputeProperties.version =
      ZE_DEVICE_COMPUTE_PROPERTIES_VERSION_CURRENT;
  ZE_CALL(zeDeviceGetComputeProperties(ZeDevice, &ZeDeviceComputeProperties));
  return PI_SUCCESS;
}

// Crate a new command list to be used in a PI call
pi_result
_pi_device::createCommandList(ze_command_list_handle_t *ZeCommandList) {
  // Create the command list, because in L0 commands are added to
  // the command lists, and later are then added to the command queue.
  //
  // TODO: Figure out how to lower the overhead of creating a new list
  // for each PI command, if that appears to be important.
  ze_command_list_desc_t ZeCommandListDesc = {};
  ZeCommandListDesc.version = ZE_COMMAND_LIST_DESC_VERSION_CURRENT;

  // TODO: can we just reset the command-list created when an earlier
  // command was submitted to the queue?
  ZE_CALL(zeCommandListCreate(ZeDevice, &ZeCommandListDesc, ZeCommandList));

  return PI_SUCCESS;
}

pi_result _pi_queue::executeCommandList(ze_command_list_handle_t ZeCommandList,
                                        bool IsBlocking) {
  // Close the command list and have it ready for dispatch.
  ZE_CALL(zeCommandListClose(ZeCommandList));
  // Offload command list to the GPU for asynchronous execution
  ZE_CALL(zeCommandQueueExecuteCommandLists(ZeCommandQueue, 1, &ZeCommandList,
                                            nullptr));

  // Check global control to make every command blocking for debugging.
  if (IsBlocking || (ZeSerialize & ZeSerializeBlock) != 0) {
    // Wait until command lists attached to the command queue are executed.
    ZE_CALL(zeCommandQueueSynchronize(ZeCommandQueue, UINT32_MAX));
  }
  return PI_SUCCESS;
}

ze_event_handle_t *_pi_event::createZeEventList(pi_uint32 EventListLength,
                                                const pi_event *EventList) {
  try {
    ze_event_handle_t *ZeEventList = new ze_event_handle_t[EventListLength];

    for (pi_uint32 I = 0; I < EventListLength; I++) {
      ZeEventList[I] = EventList[I]->ZeEvent;
    }
    return ZeEventList;
  } catch (...) {
    return nullptr;
  }
}

void _pi_event::deleteZeEventList(ze_event_handle_t *ZeEventList) {
  delete[] ZeEventList;
}

extern "C" {

// Forward declararitons
decltype(piEventCreate) piEventCreate;

std::once_flag OnceFlag;

pi_result piPlatformsGet(pi_uint32 NumEntries, pi_platform *Platforms,
                         pi_uint32 *NumPlatforms) {

  static const char *DebugMode = std::getenv("ZE_DEBUG");
  if (DebugMode)
    ZeDebug = true;

  static const char *SerializeMode = std::getenv("ZE_SERIALIZE");
  static const pi_uint32 SerializeModeValue =
      SerializeMode ? std::atoi(SerializeMode) : 0;
  ZeSerialize = SerializeModeValue;

  if (NumEntries == 0 && Platforms != nullptr) {
    return PI_INVALID_VALUE;
  }
  if (Platforms == nullptr && NumPlatforms == nullptr) {
    return PI_INVALID_VALUE;
  }

  // This is a good time to initialize L0.
  // TODO: We can still safely recover if something goes wrong during the init.
  // Implement handling segfault using sigaction.
  // TODO: We should not call zeInit multiples times ever, so
  // this code should be changed.
  ze_result_t ZeResult = ZE_CALL_NOCHECK(zeInit(ZE_INIT_FLAG_NONE));

  // Absorb the ZE_RESULT_ERROR_UNINITIALIZED and just return 0 Platforms.
  if (ZeResult == ZE_RESULT_ERROR_UNINITIALIZED) {
    assert(NumPlatforms != 0);
    *NumPlatforms = 0;
    return PI_SUCCESS;
  }

  if (ZeResult != ZE_RESULT_SUCCESS) {
    zePrint("zeInit: Level Zero initialization failure\n");
    return mapError(ZeResult);
  }

  // L0 does not have concept of Platforms, but L0 driver is the
  // closest match.
  if (Platforms && NumEntries > 0) {
    uint32_t ZeDriverCount = 0;
    ZE_CALL(zeDriverGet(&ZeDriverCount, nullptr));
    if (ZeDriverCount == 0) {
      assert(NumPlatforms != 0);
      *NumPlatforms = 0;
      return PI_SUCCESS;
    }

    ze_driver_handle_t ZeDriver;
    assert(ZeDriverCount == 1);
    ZE_CALL(zeDriverGet(&ZeDriverCount, &ZeDriver));

    try {
      // TODO: figure out how/when to release this memory
      *Platforms = new _pi_platform(ZeDriver);

      // Cache driver properties
      ze_driver_properties_t ZeDriverProperties;
      ZE_CALL(zeDriverGetProperties(ZeDriver, &ZeDriverProperties));
      uint32_t ZeDriverVersion = ZeDriverProperties.driverVersion;
      // Intel Level-Zero GPU driver stores version as:
      // | 31 - 24 | 23 - 16 | 15 - 0 |
      // |  Major  |  Minor  | Build  |
      std::string VersionMajor =
          std::to_string((ZeDriverVersion & 0xFF000000) >> 24);
      std::string VersionMinor =
          std::to_string((ZeDriverVersion & 0x00FF0000) >> 16);
      std::string VersionBuild = std::to_string(ZeDriverVersion & 0x0000FFFF);
      Platforms[0]->ZeDriverVersion = VersionMajor + std::string(".") +
                                      VersionMinor + std::string(".") +
                                      VersionBuild;

      ze_api_version_t ZeApiVersion;
      ZE_CALL(zeDriverGetApiVersion(ZeDriver, &ZeApiVersion));
      Platforms[0]->ZeDriverApiVersion =
          std::to_string(ZE_MAJOR_VERSION(ZeApiVersion)) + std::string(".") +
          std::to_string(ZE_MINOR_VERSION(ZeApiVersion));
    } catch (const std::bad_alloc &) {
      return PI_OUT_OF_HOST_MEMORY;
    } catch (...) {
      return PI_ERROR_UNKNOWN;
    }
  }

  if (NumPlatforms)
    *NumPlatforms = 1;

  return PI_SUCCESS;
}

pi_result piPlatformGetInfo(pi_platform Platform, pi_platform_info ParamName,
                            size_t ParamValueSize, void *ParamValue,
                            size_t *ParamValueSizeRet) {

  assert(Platform);
  zePrint("==========================\n");
  zePrint("SYCL over Level-Zero %s\n", Platform->ZeDriverVersion.c_str());
  zePrint("==========================\n");

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  switch (ParamName) {
  case PI_PLATFORM_INFO_NAME:
    // TODO: Query L0 driver when relevant info is added there.
    return ReturnValue("Intel(R) Level-Zero");
  case PI_PLATFORM_INFO_VENDOR:
    // TODO: Query L0 driver when relevant info is added there.
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
    // all Level0 devices.
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
    // TODO: implement other parameters
    die("Unsupported ParamName in piPlatformGetInfo");
  }

  return PI_SUCCESS;
}

pi_result piextPlatformGetNativeHandle(pi_platform Platform,
                                       pi_native_handle *NativeHandle) {
  assert(Platform);
  assert(NativeHandle);

  auto ZeDriver = pi_cast<ze_driver_handle_t *>(NativeHandle);
  // Extract the L0 driver handle from the given PI platform
  *ZeDriver = Platform->ZeDriver;
  return PI_SUCCESS;
}

pi_result piextPlatformCreateWithNativeHandle(pi_native_handle NativeHandle,
                                              pi_platform *Platform) {
  assert(NativeHandle);
  assert(Platform);

  // Create PI platform from the given L0 driver handle.
  auto ZeDriver = pi_cast<ze_driver_handle_t>(NativeHandle);
  *Platform = new _pi_platform(ZeDriver);
  return PI_SUCCESS;
}

pi_result piDevicesGet(pi_platform Platform, pi_device_type DeviceType,
                       pi_uint32 NumEntries, pi_device *Devices,
                       pi_uint32 *NumDevices) {

  assert(Platform);
  ze_driver_handle_t ZeDriver = Platform->ZeDriver;

  // Get number of devices supporting L0
  uint32_t ZeDeviceCount = 0;
  const bool AskingForGPU = (DeviceType & PI_DEVICE_TYPE_GPU);
  const bool AskingForDefault = (DeviceType == PI_DEVICE_TYPE_DEFAULT);
  ZE_CALL(zeDeviceGet(ZeDriver, &ZeDeviceCount, nullptr));
  if (ZeDeviceCount == 0 || !(AskingForGPU || AskingForDefault)) {
    if (NumDevices)
      *NumDevices = 0;
    return PI_SUCCESS;
  }

  if (NumDevices)
    *NumDevices = ZeDeviceCount;

  try {
    // TODO: Delete array at teardown
    ze_device_handle_t *ZeDevices = new ze_device_handle_t[ZeDeviceCount];
    ZE_CALL(zeDeviceGet(ZeDriver, &ZeDeviceCount, ZeDevices));

    for (uint32_t I = 0; I < ZeDeviceCount; ++I) {
      if (I < NumEntries) {
        Devices[I] = new _pi_device(ZeDevices[I], Platform);
        pi_result Result = Devices[I]->initialize();
        if (Result != PI_SUCCESS) {
          return Result;
        }
      }
    }
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  return PI_SUCCESS;
}

pi_result piDeviceRetain(pi_device Device) {
  assert(Device);

  // The root-device ref-count remains unchanged (always 1).
  if (Device->IsSubDevice) {
    ++(Device->RefCount);
  }
  return PI_SUCCESS;
}

pi_result piDeviceRelease(pi_device Device) {
  assert(Device);

  // TODO: OpenCL says root-device ref-count remains unchanged (1),
  // but when would we free the device's data?
  if (--(Device->RefCount) == 0) {
    // Destroy the command list used for initializations
    ZE_CALL(zeCommandListDestroy(Device->ZeCommandListInit));
    delete Device;
  }

  return PI_SUCCESS;
}

pi_result piDeviceGetInfo(pi_device Device, pi_device_info ParamName,
                          size_t ParamValueSize, void *ParamValue,
                          size_t *ParamValueSizeRet) {

  assert(Device != nullptr);

  ze_device_handle_t ZeDevice = Device->ZeDevice;

  uint32_t ZeAvailMemCount = 0;
  ZE_CALL(zeDeviceGetMemoryProperties(ZeDevice, &ZeAvailMemCount, nullptr));
  // Confirm at least one memory is available in the device
  assert(ZeAvailMemCount > 0);

  ze_device_memory_properties_t *ZeDeviceMemoryProperties;
  try {
    ZeDeviceMemoryProperties =
        new ze_device_memory_properties_t[ZeAvailMemCount]();
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  for (uint32_t I = 0; I < ZeAvailMemCount; I++) {
    ZeDeviceMemoryProperties[I].version =
        ZE_DEVICE_MEMORY_PROPERTIES_VERSION_CURRENT;
  }
  // TODO: cache various device properties in the PI device object,
  // and initialize them only upon they are first requested.
  ZE_CALL(zeDeviceGetMemoryProperties(ZeDevice, &ZeAvailMemCount,
                                      ZeDeviceMemoryProperties));

  ze_device_image_properties_t ZeDeviceImageProperties;
  ZeDeviceImageProperties.version = ZE_DEVICE_IMAGE_PROPERTIES_VERSION_CURRENT;
  ZE_CALL(zeDeviceGetImageProperties(ZeDevice, &ZeDeviceImageProperties));

  ze_device_kernel_properties_t ZeDeviceKernelProperties;
  ZeDeviceKernelProperties.version =
      ZE_DEVICE_KERNEL_PROPERTIES_VERSION_CURRENT;
  ZE_CALL(zeDeviceGetKernelProperties(ZeDevice, &ZeDeviceKernelProperties));

  ze_device_cache_properties_t ZeDeviceCacheProperties;
  ZeDeviceCacheProperties.version = ZE_DEVICE_CACHE_PROPERTIES_VERSION_CURRENT;
  ZE_CALL(zeDeviceGetCacheProperties(ZeDevice, &ZeDeviceCacheProperties));

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);

  switch (ParamName) {
  case PI_DEVICE_INFO_TYPE: {
    if (Device->ZeDeviceProperties.type != ZE_DEVICE_TYPE_GPU) {
      zePrint("This device type is not supported\n");
      return PI_INVALID_VALUE;
    }
    return ReturnValue(PI_DEVICE_TYPE_GPU);
  }
  case PI_DEVICE_INFO_PARENT_DEVICE:
    // TODO: all L0 devices are parent ?
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
    // TODO: Use proper mechanism to get this information from Level0 after
    // it is added to Level0.
    // Hardcoding the few we know are supported by the current hardware.
    //
    //
    std::string SupportedExtensions;

    // cl_khr_il_program - OpenCL 2.0 KHR extension for SPIRV support. Core
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
    // Hardcoding some extensions we know are supported by all Level0 devices.
    SupportedExtensions += (ZE_SUPPORTED_EXTENSIONS);
    if (ZeDeviceKernelProperties.fp16Supported)
      SupportedExtensions += ("cl_khr_fp16 ");
    if (ZeDeviceKernelProperties.fp64Supported)
      SupportedExtensions += ("cl_khr_fp64 ");
    if (ZeDeviceKernelProperties.int64AtomicsSupported)
      // int64AtomicsSupported indicates support for both.
      SupportedExtensions +=
          ("cl_khr_int64_base_atomics cl_khr_int64_extended_atomics ");
    if (ZeDeviceImageProperties.supported)
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
    // L0 spec defines only three dimensions
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
  case PI_DEVICE_INFO_MAX_MEM_ALLOC_SIZE: {
    // TODO: To confirm with spec.
    uint32_t MaxMemAllocSize = 0;
    for (uint32_t I = 0; I < ZeAvailMemCount; I++) {
      MaxMemAllocSize += ZeDeviceMemoryProperties[I].totalSize;
    }
    return ReturnValue(pi_uint64{MaxMemAllocSize});
  }
  case PI_DEVICE_INFO_GLOBAL_MEM_SIZE: {
    uint32_t GlobalMemSize = 0;
    for (uint32_t I = 0; I < ZeAvailMemCount; I++) {
      GlobalMemSize += ZeDeviceMemoryProperties[I].totalSize;
    }
    return ReturnValue(pi_uint64{GlobalMemSize});
  }
  case PI_DEVICE_INFO_LOCAL_MEM_SIZE:
    return ReturnValue(
        pi_uint64{Device->ZeDeviceComputeProperties.maxSharedLocalMemory});
  case PI_DEVICE_INFO_IMAGE_SUPPORT:
    return ReturnValue(pi_bool{ZeDeviceImageProperties.supported});
  case PI_DEVICE_INFO_HOST_UNIFIED_MEMORY:
    return ReturnValue(
        pi_bool{Device->ZeDeviceProperties.unifiedMemorySupported});
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
    ZE_CALL(zeDeviceGetSubDevices(ZeDevice, &ZeSubDeviceCount, nullptr));
    return ReturnValue(pi_uint32{ZeSubDeviceCount});
  }
  case PI_DEVICE_INFO_REFERENCE_COUNT:
    return ReturnValue(pi_uint32{Device->RefCount});
  case PI_DEVICE_INFO_PARTITION_PROPERTIES: {
    // It is debatable if SYCL sub-device and partitioning APIs sufficient to
    // expose Level0 sub-devices?  We start with support of
    // "partition_by_affinity_domain" and "numa" but if that doesn't seem to
    // be a good fit we could look at adding a more descriptive partitioning
    // type.
    struct {
      pi_device_partition_property Arr[2];
    } PartitionProperties = {{PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN, 0}};
    return ReturnValue(PartitionProperties);
  }
  case PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN:
    return ReturnValue(pi_device_affinity_domain{
        PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE});
  case PI_DEVICE_INFO_PARTITION_TYPE: {
    if (Device->IsSubDevice) {
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
    return ReturnValue(size_t{ZeDeviceKernelProperties.printfBufferSize});
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
    return ReturnValue(pi_bool{Device->ZeDeviceProperties.eccMemorySupported});
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
        pi_uint32{ZeDeviceCacheProperties.lastLevelCachelineSize});
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE:
    return ReturnValue(pi_uint64{ZeDeviceCacheProperties.lastLevelCacheSize});
  case PI_DEVICE_INFO_MAX_PARAMETER_SIZE:
    return ReturnValue(size_t{ZeDeviceKernelProperties.maxArgumentsSize});
  case PI_DEVICE_INFO_MEM_BASE_ADDR_ALIGN:
    // SYCL/OpenCL spec is vague on what this means exactly, but seems to
    // be for "alignment requirement (in bits) for sub-buffer offsets."
    // An OpenCL implementation returns 8*128, but L0 can do just 8,
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
    ze_fp_capabilities_t ZeSingleFPCapabilities =
        ZeDeviceKernelProperties.singleFpCapabilities;
    if (ZE_FP_CAPS_DENORM & ZeSingleFPCapabilities) {
      SingleFPValue |= PI_FP_DENORM;
    }
    if (ZE_FP_CAPS_INF_NAN & ZeSingleFPCapabilities) {
      SingleFPValue |= PI_FP_INF_NAN;
    }
    if (ZE_FP_CAPS_ROUND_TO_NEAREST & ZeSingleFPCapabilities) {
      SingleFPValue |= PI_FP_ROUND_TO_NEAREST;
    }
    if (ZE_FP_CAPS_ROUND_TO_ZERO & ZeSingleFPCapabilities) {
      SingleFPValue |= PI_FP_ROUND_TO_ZERO;
    }
    if (ZE_FP_CAPS_ROUND_TO_INF & ZeSingleFPCapabilities) {
      SingleFPValue |= PI_FP_ROUND_TO_INF;
    }
    if (ZE_FP_CAPS_FMA & ZeSingleFPCapabilities) {
      SingleFPValue |= PI_FP_FMA;
    }
    return ReturnValue(pi_uint64{SingleFPValue});
  }
  case PI_DEVICE_INFO_HALF_FP_CONFIG: {
    uint64_t HalfFPValue = 0;
    ze_fp_capabilities_t ZeHalfFPCapabilities =
        ZeDeviceKernelProperties.halfFpCapabilities;
    if (ZE_FP_CAPS_DENORM & ZeHalfFPCapabilities) {
      HalfFPValue |= PI_FP_DENORM;
    }
    if (ZE_FP_CAPS_INF_NAN & ZeHalfFPCapabilities) {
      HalfFPValue |= PI_FP_INF_NAN;
    }
    if (ZE_FP_CAPS_ROUND_TO_NEAREST & ZeHalfFPCapabilities) {
      HalfFPValue |= PI_FP_ROUND_TO_NEAREST;
    }
    if (ZE_FP_CAPS_ROUND_TO_ZERO & ZeHalfFPCapabilities) {
      HalfFPValue |= PI_FP_ROUND_TO_ZERO;
    }
    if (ZE_FP_CAPS_ROUND_TO_INF & ZeHalfFPCapabilities) {
      HalfFPValue |= PI_FP_ROUND_TO_INF;
    }
    if (ZE_FP_CAPS_FMA & ZeHalfFPCapabilities) {
      HalfFPValue |= PI_FP_FMA;
    }
    return ReturnValue(pi_uint64{HalfFPValue});
  }
  case PI_DEVICE_INFO_DOUBLE_FP_CONFIG: {
    uint64_t DoubleFPValue = 0;
    ze_fp_capabilities_t ZeDoubleFPCapabilities =
        ZeDeviceKernelProperties.doubleFpCapabilities;
    if (ZE_FP_CAPS_DENORM & ZeDoubleFPCapabilities) {
      DoubleFPValue |= PI_FP_DENORM;
    }
    if (ZE_FP_CAPS_INF_NAN & ZeDoubleFPCapabilities) {
      DoubleFPValue |= PI_FP_INF_NAN;
    }
    if (ZE_FP_CAPS_ROUND_TO_NEAREST & ZeDoubleFPCapabilities) {
      DoubleFPValue |= PI_FP_ROUND_TO_NEAREST;
    }
    if (ZE_FP_CAPS_ROUND_TO_ZERO & ZeDoubleFPCapabilities) {
      DoubleFPValue |= PI_FP_ROUND_TO_ZERO;
    }
    if (ZE_FP_CAPS_ROUND_TO_INF & ZeDoubleFPCapabilities) {
      DoubleFPValue |= PI_FP_ROUND_TO_INF;
    }
    if (ZE_FP_CAPS_FMA & ZeDoubleFPCapabilities) {
      DoubleFPValue |= PI_FP_FMA;
    }
    return ReturnValue(pi_uint64{DoubleFPValue});
  }
  case PI_DEVICE_INFO_IMAGE2D_MAX_WIDTH:
    // Until L0 provides needed info, hardcode default minimum values required
    // by the SYCL specification.
    return ReturnValue(size_t{8192});
  case PI_DEVICE_INFO_IMAGE2D_MAX_HEIGHT:
    // Until L0 provides needed info, hardcode default minimum values required
    // by the SYCL specification.
    return ReturnValue(size_t{8192});
  case PI_DEVICE_INFO_IMAGE3D_MAX_WIDTH:
    // Until L0 provides needed info, hardcode default minimum values required
    // by the SYCL specification.
    return ReturnValue(size_t{2048});
  case PI_DEVICE_INFO_IMAGE3D_MAX_HEIGHT:
    // Until L0 provides needed info, hardcode default minimum values required
    // by the SYCL specification.
    return ReturnValue(size_t{2048});
  case PI_DEVICE_INFO_IMAGE3D_MAX_DEPTH:
    // Until L0 provides needed info, hardcode default minimum values required
    // by the SYCL specification.
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
    return getInfoArray(Device->ZeDeviceComputeProperties.numSubGroupSizes,
                        ParamValueSize, ParamValue, ParamValueSizeRet,
                        Device->ZeDeviceComputeProperties.subGroupSizes);
  }
  case PI_DEVICE_INFO_IL_VERSION: {
    // Set to a space separated list of IL version strings of the form
    // <IL_Prefix>_<Major_version>.<Minor_version>.
    // "SPIR-V" is a required IL prefix when cl_khr_il_progam extension is
    // reported.
    uint32_t SpirvVersion = ZeDeviceKernelProperties.spirvVersionSupported;
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
    if (Device->ZeDeviceProperties.unifiedMemorySupported) {
      // TODO: Use ze_memory_access_capabilities_t
      Supported = PI_USM_ACCESS | PI_USM_ATOMIC_ACCESS |
                  PI_USM_CONCURRENT_ACCESS | PI_USM_CONCURRENT_ATOMIC_ACCESS;
    }
    return ReturnValue(Supported);
  }
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
  // Other partitioning ways are not supported by L0
  if (Properties[0] != PI_DEVICE_PARTITION_BY_AFFINITY_DOMAIN ||
      Properties[1] != PI_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE) {
    return PI_INVALID_VALUE;
  }

  assert(Device);
  // Get the number of subdevices available.
  // TODO: maybe add interface to create the specified # of subdevices.
  uint32_t Count = 0;
  ZE_CALL(zeDeviceGetSubDevices(Device->ZeDevice, &Count, nullptr));

  // Check that the requested/allocated # of sub-devices is the same
  // as was reported by the above call.
  // TODO: we may want to support smaller/larger # devices too.
  if (Count != NumDevices) {
    zePrint("piDevicePartition: unsupported # of sub-devices requested\n");
    return PI_INVALID_OPERATION;
  }

  if (OutNumDevices) {
    *OutNumDevices = Count;
  }

  if (!OutDevices) {
    // If we are not given the buffer, we are done.
    return PI_SUCCESS;
  }

  try {
    auto ZeSubdevices = new ze_device_handle_t[Count];
    ZE_CALL(zeDeviceGetSubDevices(Device->ZeDevice, &Count, ZeSubdevices));

    // Wrap the L0 sub-devices into PI sub-devices, and write them out.
    for (uint32_t I = 0; I < Count; ++I) {
      OutDevices[I] = new _pi_device(ZeSubdevices[I], Device->Platform,
                                     true /* isSubDevice */);
      pi_result Result = OutDevices[I]->initialize();
      if (Result != PI_SUCCESS) {
        delete[] ZeSubdevices;
        return Result;
      }
    }
    delete[] ZeSubdevices;
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
  return PI_SUCCESS;
}

pi_result
piextDeviceSelectBinary(pi_device Device, // TODO: does this need to be context?
                        pi_device_binary *Binaries, pi_uint32 NumBinaries,
                        pi_uint32 *SelectedBinaryInd) {

  // TODO dummy implementation.
  // Real implementation will use the same mechanism OpenCL ICD dispatcher
  // uses. Something like:
  //   PI_VALIDATE_HANDLE_RETURN_HANDLE(ctx, PI_INVALID_CONTEXT);
  //     return context->dispatch->piextDeviceSelectIR(
  //       ctx, images, num_images, selected_image);
  // where context->dispatch is set to the dispatch table provided by PI
  // plugin for platform/device the ctx was created for.

  constexpr pi_uint32 InvalidInd = std::numeric_limits<pi_uint32>::max();
  *SelectedBinaryInd = NumBinaries > 0 ? 0 : InvalidInd;
  return PI_SUCCESS;
}

pi_result piextDeviceGetNativeHandle(pi_device Device,
                                     pi_native_handle *NativeHandle) {
  assert(Device);
  assert(NativeHandle);

  auto ZeDevice = pi_cast<ze_device_handle_t *>(NativeHandle);
  // Extract the L0 module handle from the given PI device
  *ZeDevice = Device->ZeDevice;
  return PI_SUCCESS;
}

pi_result piextDeviceCreateWithNativeHandle(pi_native_handle NativeHandle,
                                            pi_device *Device) {
  // Create PI device from the given L0 device handle.
  die("piextDeviceCreateWithNativeHandle: not supported");
  return PI_SUCCESS;
}

pi_result piContextCreate(const pi_context_properties *Properties,
                          pi_uint32 NumDevices, const pi_device *Devices,
                          void (*PFnNotify)(const char *ErrInfo,
                                            const void *PrivateInfo, size_t CB,
                                            void *UserData),
                          void *UserData, pi_context *RetContext) {

  // L0 does not have notion of contexts.
  // Return the device handle (only single device is allowed) as a context
  // handle.
  if (NumDevices != 1) {
    zePrint("piCreateContext: context should have exactly one Device\n");
    return PI_INVALID_VALUE;
  }

  assert(Devices);
  assert(RetContext);

  try {
    *RetContext = new _pi_context(*Devices);
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

  assert(Context);

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  switch (ParamName) {
  case PI_CONTEXT_INFO_DEVICES:
    return ReturnValue(Context->Device);
  case PI_CONTEXT_INFO_NUM_DEVICES:
    return ReturnValue(pi_uint32{1});
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
  die("piextContextSetExtendedDeleter: not supported");
  return PI_SUCCESS;
}

pi_result piextContextGetNativeHandle(pi_context Context,
                                      pi_native_handle *NativeHandle) {
  die("piextContextGetNativeHandle: not supported");
  return PI_SUCCESS;
}

pi_result piextContextCreateWithNativeHandle(pi_native_handle NativeHandle,
                                             pi_context *Context) {
  die("piextContextCreateWithNativeHandle: not supported");
  return PI_SUCCESS;
}

pi_result piContextRetain(pi_context Context) {

  assert(Context);
  ++(Context->RefCount);
  return PI_SUCCESS;
}

pi_result piContextRelease(pi_context Context) {

  assert(Context);
  if (--(Context->RefCount) == 0) {
    delete Context;
  }
  return PI_SUCCESS;
}

pi_result piQueueCreate(pi_context Context, pi_device Device,
                        pi_queue_properties Properties, pi_queue *Queue) {

  // Check that unexpected bits are not set.
  assert(!(Properties & ~(PI_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                          PI_QUEUE_PROFILING_ENABLE | PI_QUEUE_ON_DEVICE |
                          PI_QUEUE_ON_DEVICE_DEFAULT)));

  ze_device_handle_t ZeDevice;
  ze_command_queue_handle_t ZeCommandQueue;

  if (!Context) {
    return PI_INVALID_CONTEXT;
  }
  if (Context->Device != Device) {
    return PI_INVALID_DEVICE;
  }

  assert(Device);
  ZeDevice = Device->ZeDevice;
  ze_command_queue_desc_t ZeCommandQueueDesc = {};
  ZeCommandQueueDesc.version = ZE_COMMAND_QUEUE_DESC_VERSION_CURRENT;
  ZeCommandQueueDesc.ordinal = 0;
  ZeCommandQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;

  ZE_CALL(
      zeCommandQueueCreate(ZeDevice,
                           &ZeCommandQueueDesc, // TODO: translate properties
                           &ZeCommandQueue));

  assert(Queue);
  try {
    *Queue = new _pi_queue(ZeCommandQueue, Context);
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

  assert(Queue);

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  // TODO: consider support for queue properties and size
  switch (ParamName) {
  case PI_QUEUE_INFO_CONTEXT:
    return ReturnValue(Queue->Context);
  case PI_QUEUE_INFO_DEVICE:
    return ReturnValue(Queue->Context->Device);
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
  ++(Queue->RefCount);
  return PI_SUCCESS;
}

pi_result piQueueRelease(pi_queue Queue) {
  assert(Queue);
  if (--(Queue->RefCount) == 0) {
    ZE_CALL(zeCommandQueueDestroy(Queue->ZeCommandQueue));
  }
  return PI_SUCCESS;
}

pi_result piQueueFinish(pi_queue Queue) {
  // Wait until command lists attached to the command queue are executed.
  assert(Queue);
  ZE_CALL(zeCommandQueueSynchronize(Queue->ZeCommandQueue, UINT32_MAX));
  return PI_SUCCESS;
}

pi_result piextQueueGetNativeHandle(pi_queue Queue,
                                    pi_native_handle *NativeHandle) {
  die("piextQueueGetNativeHandle: not supported");
  return PI_SUCCESS;
}

pi_result piextQueueCreateWithNativeHandle(pi_native_handle NativeHandle,
                                           pi_queue *Queue) {
  die("piextQueueCreateWithNativeHandle: not supported");
  return PI_SUCCESS;
}

pi_result piMemBufferCreate(pi_context Context, pi_mem_flags Flags, size_t Size,
                            void *HostPtr, pi_mem *RetMem) {

  // TODO: implement read-only, write-only
  assert((Flags & PI_MEM_FLAGS_ACCESS_RW) != 0);
  assert(Context);
  assert(RetMem);

  void *Ptr;
  ze_device_handle_t ZeDevice = Context->Device->ZeDevice;

  ze_device_mem_alloc_desc_t ZeDesc = {};
  ZeDesc.flags = ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT;
  ZeDesc.ordinal = 0;
  ZE_CALL(zeDriverAllocDeviceMem(Context->Device->Platform->ZeDriver, &ZeDesc,
                                 Size,
                                 1, // TODO: alignment
                                 ZeDevice, &Ptr));

  if ((Flags & PI_MEM_FLAGS_HOST_PTR_USE) != 0 ||
      (Flags & PI_MEM_FLAGS_HOST_PTR_COPY) != 0) {
    // Initialize the buffer synchronously with immediate offload
    ZE_CALL(zeCommandListAppendMemoryCopy(Context->Device->ZeCommandListInit,
                                          Ptr, HostPtr, Size, nullptr));
  } else if (Flags == 0 || (Flags == PI_MEM_FLAGS_ACCESS_RW)) {
    // Nothing more to do.
  } else {
    die("piMemBufferCreate: not implemented");
  }

  auto HostPtrOrNull =
      (Flags & PI_MEM_FLAGS_HOST_PTR_USE) ? pi_cast<char *>(HostPtr) : nullptr;
  try {
    *RetMem = new _pi_buffer(Context->Device->Platform,
                             pi_cast<char *>(Ptr) /* L0 Memory Handle */,
                             HostPtrOrNull);
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
  die("piMemGetInfo: not implemented");
  return {};
}

pi_result piMemRetain(pi_mem Mem) {
  assert(Mem);
  ++(Mem->RefCount);
  return PI_SUCCESS;
}

pi_result piMemRelease(pi_mem Mem) {
  assert(Mem);
  if (--(Mem->RefCount) == 0) {
    if (Mem->isImage()) {
      ZE_CALL(zeImageDestroy(pi_cast<ze_image_handle_t>(Mem->getZeHandle())));
    } else {
      auto Buf = static_cast<_pi_buffer *>(Mem);
      if (!Buf->isSubBuffer()) {
        ZE_CALL(zeDriverFreeMem(Mem->Platform->ZeDriver, Mem->getZeHandle()));
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
  assert((Flags & PI_MEM_FLAGS_ACCESS_RW) != 0);
  assert(ImageFormat);
  assert(Context);
  assert(RetImage);

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

  ze_image_format_desc_t ZeFormatDesc = {
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

  ze_image_desc_t ZeImageDesc = {
      ZE_IMAGE_DESC_VERSION_CURRENT,
      pi_cast<ze_image_flag_t>(ZE_IMAGE_FLAG_PROGRAM_READ |
                               ZE_IMAGE_FLAG_PROGRAM_WRITE),
      ZeImageType,
      ZeFormatDesc,
      pi_cast<uint32_t>(ImageDesc->image_width),
      pi_cast<uint32_t>(ImageDesc->image_height),
      pi_cast<uint32_t>(ImageDesc->image_depth),
      pi_cast<uint32_t>(ImageDesc->image_array_size),
      ImageDesc->num_mip_levels};

  ze_image_handle_t ZeHImage;
  ZE_CALL(zeImageCreate(Context->Device->ZeDevice, &ZeImageDesc, &ZeHImage));

  auto HostPtrOrNull =
      (Flags & PI_MEM_FLAGS_HOST_PTR_USE) ? pi_cast<char *>(HostPtr) : nullptr;

  try {
    auto ZePIImage =
        new _pi_image(Context->Device->Platform, ZeHImage, HostPtrOrNull);

#ifndef NDEBUG
    ZePIImage->ZeImageDesc = ZeImageDesc;
#endif // !NDEBUG

    if ((Flags & PI_MEM_FLAGS_HOST_PTR_USE) != 0 ||
        (Flags & PI_MEM_FLAGS_HOST_PTR_COPY) != 0) {
      // Initialize image synchronously with immediate offload
      ZE_CALL(zeCommandListAppendImageCopyFromMemory(
          Context->Device->ZeCommandListInit, ZeHImage, HostPtr, nullptr,
          nullptr));
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
  die("piextMemGetNativeHandle: not supported");
  return PI_SUCCESS;
}

pi_result piextMemCreateWithNativeHandle(pi_native_handle NativeHandle,
                                         pi_mem *Mem) {
  die("piextMemCreateWithNativeHandle: not supported");
  return PI_SUCCESS;
}

pi_result piProgramCreate(pi_context Context, const void *IL, size_t Length,
                          pi_program *Program) {

  assert(Context);
  assert(Program);

  // NOTE: the L0 module creation is also building the program, so we are
  // deferring it until the program is ready to be built in piProgramBuild
  // and piProgramCompile. Also it is only then we know the build options.
  //
  ze_module_desc_t ZeModuleDesc = {};
  ZeModuleDesc.version = ZE_MODULE_DESC_VERSION_CURRENT;
  ZeModuleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  ZeModuleDesc.inputSize = Length;
  ZeModuleDesc.pInputModule = new uint8_t[Length];
  memcpy(const_cast<uint8_t *>(ZeModuleDesc.pInputModule), IL, Length);

  try {
    *Program = new _pi_program(nullptr, ZeModuleDesc, Context);
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
                                    pi_program *RetProgram) {

  // This must be for the single device in this context.
  assert(NumDevices == 1);
  assert(Context);
  assert(RetProgram);
  assert(DeviceList && DeviceList[0] == Context->Device);

  // Check the binary too.
  assert(Lengths && Lengths[0] != 0);
  assert(Binaries && Binaries[0] != nullptr);
  size_t Length = Lengths[0];
  auto Binary = pi_cast<const uint8_t *>(Binaries[0]);

  ze_module_desc_t ZeModuleDesc = {};
  ZeModuleDesc.version = ZE_MODULE_DESC_VERSION_CURRENT;
  ZeModuleDesc.format = ZE_MODULE_FORMAT_NATIVE;
  ZeModuleDesc.inputSize = Length;
  ZeModuleDesc.pInputModule = new uint8_t[Length];
  memcpy(const_cast<uint8_t *>(ZeModuleDesc.pInputModule), Binary, Length);

  try {
    *RetProgram = new _pi_program(nullptr, ZeModuleDesc, Context);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }

  if (BinaryStatus) {
    *BinaryStatus = PI_SUCCESS;
  }
  return PI_SUCCESS;
}

pi_result piclProgramCreateWithSource(pi_context Context, pi_uint32 Count,
                                      const char **Strings,
                                      const size_t *Lengths,
                                      pi_program *RetProgram) {

  zePrint("piclProgramCreateWithSource: not supported in L0\n");
  return PI_INVALID_OPERATION;
}

pi_result piProgramGetInfo(pi_program Program, pi_program_info ParamName,
                           size_t ParamValueSize, void *ParamValue,
                           size_t *ParamValueSizeRet) {

  assert(Program);
  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  switch (ParamName) {
  case PI_PROGRAM_INFO_REFERENCE_COUNT:
    return ReturnValue(pi_uint32{Program->RefCount});
  case PI_PROGRAM_INFO_NUM_DEVICES:
    // L0 Module is always for a single device.
    return ReturnValue(pi_uint32{1});
  case PI_PROGRAM_INFO_DEVICES:
    return ReturnValue(Program->Context->Device);
  case PI_PROGRAM_INFO_BINARY_SIZES: {
    size_t SzBinary = 0;
    ZE_CALL(zeModuleGetNativeBinary(Program->ZeModule, &SzBinary, nullptr));
    // This is an array of 1 element, initialize if it were scalar.
    return ReturnValue(size_t{SzBinary});
  }
  case PI_PROGRAM_INFO_BINARIES: {
    size_t SzBinary = 0;
    uint8_t **PBinary = pi_cast<uint8_t **>(ParamValue);
    ZE_CALL(zeModuleGetNativeBinary(Program->ZeModule, &SzBinary, PBinary[0]));
    break;
  }
  case PI_PROGRAM_INFO_NUM_KERNELS: {
    uint32_t NumKernels = 0;
    ZE_CALL(zeModuleGetKernelNames(Program->ZeModule, &NumKernels, nullptr));
    return ReturnValue(size_t{NumKernels});
  }
  case PI_PROGRAM_INFO_KERNEL_NAMES:
    try {
      // There are extra allocations/copying here dictated by the difference
      // in L0 and PI interfaces.
      uint32_t Count = 0;
      ZE_CALL(zeModuleGetKernelNames(Program->ZeModule, &Count, nullptr));
      char **PNames = new char *[Count];
      ZE_CALL(zeModuleGetKernelNames(Program->ZeModule, &Count,
                                     const_cast<const char **>(PNames)));
      std::string PINames{""};
      for (uint32_t I = 0; I < Count; ++I) {
        PINames += (I > 0 ? ";" : "");
        PINames += PNames[I];
      }
      delete[] PNames;
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
  // TODO: L0 does not [yet] support linking so dummy implementation here.
  assert(NumInputPrograms == 1 && InputPrograms);
  assert(RetProgram);
  *RetProgram = InputPrograms[0];
  return PI_SUCCESS;
}

pi_result piProgramCompile(
    pi_program Program, pi_uint32 NumDevices, const pi_device *DeviceList,
    const char *Options, pi_uint32 NumInputHeaders,
    const pi_program *InputHeaders, const char **HeaderIncludeNames,
    void (*PFnNotify)(pi_program Program, void *UserData), void *UserData) {

  // Assert on unsupported arguments.
  assert(NumInputHeaders == 0);
  assert(!InputHeaders);

  // There is no support foe linking yet in L0 so "compile" actually
  // does the "build".
  return piProgramBuild(Program, NumDevices, DeviceList, Options, PFnNotify,
                        UserData);
}

pi_result piProgramBuild(pi_program Program, pi_uint32 NumDevices,
                         const pi_device *DeviceList, const char *Options,
                         void (*PFnNotify)(pi_program Program, void *UserData),
                         void *UserData) {
  assert(Program);
  assert(NumDevices == 1);
  assert(DeviceList && DeviceList[0] == Program->Context->Device);
  assert(!PFnNotify);
  assert(!UserData);

  // Check that the program wasn't already built.
  assert(!Program->ZeModule);

  Program->ZeModuleDesc.pBuildFlags = Options;
  // TODO: set specialization constants here.
  Program->ZeModuleDesc.pConstants = nullptr;

  ze_device_handle_t ZeDevice = Program->Context->Device->ZeDevice;
  ZE_CALL(zeModuleCreate(ZeDevice, &Program->ZeModuleDesc, &Program->ZeModule,
                         &Program->ZeBuildLog));
  return PI_SUCCESS;
}

pi_result piProgramGetBuildInfo(pi_program Program, pi_device Device,
                                cl_program_build_info ParamName,
                                size_t ParamValueSize, void *ParamValue,
                                size_t *ParamValueSizeRet) {

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  if (ParamName == CL_PROGRAM_BINARY_TYPE) {
    // TODO: is this the only supported binary type in L0?
    // We should probably return CL_PROGRAM_BINARY_TYPE_NONE if asked
    // before the program was compiled.
    return ReturnValue(
        cl_program_binary_type{CL_PROGRAM_BINARY_TYPE_EXECUTABLE});
  }
  if (ParamName == CL_PROGRAM_BUILD_OPTIONS) {
    // TODO: how to get module build options out of L0?
    // For the programs that we compiled we can remember the options
    // passed with piProgramCompile/piProgramBuild, but what can we
    // return for programs that were built outside and registered
    // with piProgramRegister?
    return ReturnValue("");
  } else if (ParamName == CL_PROGRAM_BUILD_LOG) {
    assert(Program->ZeBuildLog);
    size_t LogSize = ParamValueSize;
    ZE_CALL(zeModuleBuildLogGetString(Program->ZeBuildLog, &LogSize,
                                      pi_cast<char *>(ParamValue)));
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
  assert(Program);
  ++(Program->RefCount);
  return PI_SUCCESS;
}

pi_result piProgramRelease(pi_program Program) {
  assert(Program);
  assert((Program->RefCount > 0) && "Program is already released.");
  if (--(Program->RefCount) == 0) {
    delete[] Program->ZeModuleDesc.pInputModule;
    if (Program->ZeBuildLog)
      zeModuleBuildLogDestroy(Program->ZeBuildLog);
    // TODO: call zeModuleDestroy for non-interop L0 modules
    delete Program;
  }
  return PI_SUCCESS;
}

pi_result piextProgramGetNativeHandle(pi_program Program,
                                      pi_native_handle *NativeHandle) {
  die("piextProgramGetNativeHandle: not supported");
  return PI_SUCCESS;
}

pi_result piextProgramCreateWithNativeHandle(pi_native_handle NativeHandle,
                                             pi_program *Program) {
  die("piextProgramCreateWithNativeHandle: not supported");
  return PI_SUCCESS;
}

pi_result piKernelCreate(pi_program Program, const char *KernelName,
                         pi_kernel *RetKernel) {

  assert(Program);
  assert(RetKernel);
  assert(KernelName);
  ze_kernel_desc_t ZeKernelDesc = {};
  ZeKernelDesc.version = ZE_KERNEL_DESC_VERSION_CURRENT;
  ZeKernelDesc.flags = ZE_KERNEL_FLAG_NONE;
  ZeKernelDesc.pKernelName = KernelName;

  ze_kernel_handle_t ZeKernel;
  ZE_CALL(zeKernelCreate(pi_cast<ze_module_handle_t>(Program->ZeModule),
                         &ZeKernelDesc, &ZeKernel));

  try {
    auto ZePiKernel = new _pi_kernel(ZeKernel, Program);
    *RetKernel = pi_cast<pi_kernel>(ZePiKernel);
  } catch (const std::bad_alloc &) {
    return PI_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return PI_ERROR_UNKNOWN;
  }
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

  assert(Kernel);
  ZE_CALL(zeKernelSetArgumentValue(
      pi_cast<ze_kernel_handle_t>(Kernel->ZeKernel),
      pi_cast<uint32_t>(ArgIndex), pi_cast<size_t>(ArgSize),
      pi_cast<const void *>(ArgValue)));

  return PI_SUCCESS;
}

// Special version of piKernelSetArg to accept pi_mem and pi_sampler.
pi_result piextKernelSetArgMemObj(pi_kernel Kernel, pi_uint32 ArgIndex,
                                  const pi_mem *ArgValue) {
  // TODO: the better way would probably be to add a new PI API for
  // extracting native PI object from PI handle, and have SYCL
  // RT pass that directly to the regular piKernelSetArg (and
  // then remove this piextKernelSetArgMemObj).

  assert(Kernel);
  ZE_CALL(
      zeKernelSetArgumentValue(pi_cast<ze_kernel_handle_t>(Kernel->ZeKernel),
                               pi_cast<uint32_t>(ArgIndex), sizeof(void *),
                               (*ArgValue)->getZeHandlePtr()));

  return PI_SUCCESS;
}

pi_result piKernelGetInfo(pi_kernel Kernel, pi_kernel_info ParamName,
                          size_t ParamValueSize, void *ParamValue,
                          size_t *ParamValueSizeRet) {
  assert(Kernel);
  ze_kernel_properties_t ZeKernelProperties;
  ZeKernelProperties.version = ZE_KERNEL_PROPERTIES_VERSION_CURRENT;
  ZE_CALL(zeKernelGetProperties(Kernel->ZeKernel, &ZeKernelProperties));

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  switch (ParamName) {
  case PI_KERNEL_INFO_CONTEXT:
    return ReturnValue(pi_context{Kernel->Program->Context});
  case PI_KERNEL_INFO_PROGRAM:
    return ReturnValue(pi_program{Kernel->Program});
  case PI_KERNEL_INFO_FUNCTION_NAME:
    return ReturnValue(ZeKernelProperties.name);
  case PI_KERNEL_INFO_NUM_ARGS:
    return ReturnValue(pi_uint32{ZeKernelProperties.numKernelArgs});
  case PI_KERNEL_INFO_REFERENCE_COUNT:
    return ReturnValue(pi_uint32{Kernel->RefCount});
  case PI_KERNEL_INFO_ATTRIBUTES:
    try {
      uint32_t Size;
      ZE_CALL(zeKernelGetAttribute(
          Kernel->ZeKernel, ZE_KERNEL_ATTR_SOURCE_ATTRIBUTE, &Size, nullptr));
      char *attributes = new char[Size];
      ZE_CALL(zeKernelGetAttribute(Kernel->ZeKernel,
                                   ZE_KERNEL_ATTR_SOURCE_ATTRIBUTE, &Size,
                                   attributes));
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
  assert(Kernel);
  assert(Device);
  ze_device_handle_t ZeDevice = Device->ZeDevice;
  ze_device_compute_properties_t ZeDeviceComputeProperties;
  ZeDeviceComputeProperties.version =
      ZE_DEVICE_COMPUTE_PROPERTIES_VERSION_CURRENT;
  ZE_CALL(zeDeviceGetComputeProperties(ZeDevice, &ZeDeviceComputeProperties));

  ze_kernel_properties_t ZeKernelProperties;
  ZeKernelProperties.version = ZE_KERNEL_PROPERTIES_VERSION_CURRENT;
  ZE_CALL(zeKernelGetProperties(Kernel->ZeKernel, &ZeKernelProperties));

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
    ZE_CALL(zeKernelSuggestGroupSize(Kernel->ZeKernel, 10000, 10000, 10000, &X,
                                     &Y, &Z));
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
  case PI_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE: {
    // TODO: Assume 0 for now, replace with ze_kernel_properties_t::localMemSize
    // once released in RT.
    return ReturnValue(pi_uint32{0});
  }
  case PI_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
    ze_device_properties_t ZeDeviceProperties;
    ZeDeviceProperties.version = ZE_DEVICE_PROPERTIES_VERSION_CURRENT;
    ZE_CALL(zeDeviceGetProperties(ZeDevice, &ZeDeviceProperties));

    return ReturnValue(size_t{ZeDeviceProperties.physicalEUSimdWidth});
  }
  case PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE:
    // TODO: Assume 0 for now, replace with
    // ze_kernel_properties_t::privateMemSize once released in RT.
    return ReturnValue(pi_uint32{0});
  default:
    zePrint("Unknown ParamName in piKernelGetGroupInfo: ParamName=%d(0x%x)\n",
            ParamName, ParamName);
    return PI_INVALID_VALUE;
  }
  return PI_SUCCESS;
}

pi_result piKernelGetSubGroupInfo(
    pi_kernel Kernel, pi_device Device,
    pi_kernel_sub_group_info ParamName, // TODO: untie from OpenCL
    size_t InputValueSize, const void *InputValue, size_t ParamValueSize,
    void *ParamValue, size_t *ParamValueSizeRet) {

  die("piKernelGetSubGroupInfo: not implemented");
  return {};
}

pi_result piKernelRetain(pi_kernel Kernel) {

  assert(Kernel);
  ++(Kernel->RefCount);
  return PI_SUCCESS;
}

pi_result piKernelRelease(pi_kernel Kernel) {

  assert(Kernel);
  if (--(Kernel->RefCount) == 0) {
    delete Kernel;
  }
  return PI_SUCCESS;
}

pi_result
piEnqueueKernelLaunch(pi_queue Queue, pi_kernel Kernel, pi_uint32 WorkDim,
                      const size_t *GlobalWorkOffset,
                      const size_t *GlobalWorkSize, const size_t *LocalWorkSize,
                      pi_uint32 NumEventsInWaitList,
                      const pi_event *EventWaitList, pi_event *Event) {
  assert(Kernel);
  assert(Queue);
  assert((WorkDim > 0) && (WorkDim < 4));

  ze_group_count_t ZeThreadGroupDimensions{1, 1, 1};
  uint32_t WG[3];

  // global_work_size of unused dimensions must be set to 1
  assert(WorkDim == 3 || GlobalWorkSize[2] == 1);
  assert(WorkDim >= 2 || GlobalWorkSize[1] == 1);

  if (LocalWorkSize) {
    WG[0] = pi_cast<uint32_t>(LocalWorkSize[0]);
    WG[1] = pi_cast<uint32_t>(LocalWorkSize[1]);
    WG[2] = pi_cast<uint32_t>(LocalWorkSize[2]);
  } else {
    ZE_CALL(zeKernelSuggestGroupSize(Kernel->ZeKernel, GlobalWorkSize[0],
                                     GlobalWorkSize[1], GlobalWorkSize[2],
                                     &WG[0], &WG[1], &WG[2]));
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

  assert(GlobalWorkSize[0] == (ZeThreadGroupDimensions.groupCountX * WG[0]));
  assert(GlobalWorkSize[1] == (ZeThreadGroupDimensions.groupCountY * WG[1]));
  assert(GlobalWorkSize[2] == (ZeThreadGroupDimensions.groupCountZ * WG[2]));

  ZE_CALL(zeKernelSetGroupSize(Kernel->ZeKernel, WG[0], WG[1], WG[2]));

  // Get a new command list to be used on this call
  ze_command_list_handle_t ZeCommandList = nullptr;
  if (auto Res = Queue->Context->Device->createCommandList(&ZeCommandList))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  if (Event) {
    auto Res = piEventCreate(Kernel->Program->Context, Event);
    if (Res != PI_SUCCESS)
      return Res;

    (*Event)->Queue = Queue;
    (*Event)->CommandType = PI_COMMAND_TYPE_NDRANGE_KERNEL;
    (*Event)->ZeCommandList = ZeCommandList;

    ZeEvent = (*Event)->ZeEvent;
  }

  ze_event_handle_t *ZeEventWaitList =
      _pi_event::createZeEventList(NumEventsInWaitList, EventWaitList);

  // Add the command to the command list
  ZE_CALL(zeCommandListAppendLaunchKernel(
      ZeCommandList, Kernel->ZeKernel, &ZeThreadGroupDimensions, ZeEvent,
      NumEventsInWaitList, ZeEventWaitList));

  zePrint("calling zeCommandListAppendLaunchKernel() with"
          "  ZeEvent %lx\n"
          "  NumEventsInWaitList %d:",
          pi_cast<std::uintptr_t>(ZeEvent), NumEventsInWaitList);
  for (pi_uint32 I = 0; I < NumEventsInWaitList; I++) {
    zePrint(" %lx", pi_cast<std::uintptr_t>(ZeEventWaitList[I]));
  }
  zePrint("\n");

  // Execute command list asynchronously, as the event will be used
  // to track down its completion.
  if (auto Res = Queue->executeCommandList(ZeCommandList))
    return Res;

  _pi_event::deleteZeEventList(ZeEventWaitList);

  return PI_SUCCESS;
}

//
// Events
//
pi_result piEventCreate(pi_context Context, pi_event *RetEvent) {
  size_t Index = 0;
  ze_event_pool_handle_t ZeEventPool = {};
  ZE_CALL(Context->getFreeSlotInExistingOrNewPool(ZeEventPool, Index));
  ze_event_handle_t ZeEvent;
  ze_event_desc_t ZeEventDesc = {};
  ZeEventDesc.signal = ZE_EVENT_SCOPE_FLAG_NONE;
  ZeEventDesc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
  ZeEventDesc.version = ZE_EVENT_DESC_VERSION_CURRENT;
  ZeEventDesc.index = Index;

  ZE_CALL(zeEventCreate(ZeEventPool, &ZeEventDesc, &ZeEvent));

  try {
    assert(RetEvent);
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

  assert(Event);
  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  switch (ParamName) {
  case PI_EVENT_INFO_COMMAND_QUEUE:
    return ReturnValue(pi_queue{Event->Queue});
  case PI_EVENT_INFO_CONTEXT:
    return ReturnValue(pi_context{Event->Queue->Context});
  case PI_EVENT_INFO_COMMAND_TYPE:
    return ReturnValue(pi_cast<pi_uint64>(Event->CommandType));
  case PI_EVENT_INFO_COMMAND_EXECUTION_STATUS: {
    ze_result_t ZeResult;
    ZeResult = ZE_CALL_NOCHECK(zeEventQueryStatus(Event->ZeEvent));
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

  assert(Event);
  uint64_t ZeTimerResolution =
      Event->Queue->Context->Device->ZeDeviceProperties.timerResolution;

  ReturnHelper ReturnValue(ParamValueSize, ParamValue, ParamValueSizeRet);
  switch (ParamName) {
  case PI_PROFILING_INFO_COMMAND_START: {
    uint64_t ContextStart;
    ZE_CALL(zeEventGetTimestamp(
        Event->ZeEvent, ZE_EVENT_TIMESTAMP_CONTEXT_START, &ContextStart));
    ContextStart *= ZeTimerResolution;
    return ReturnValue(uint64_t{ContextStart});
  }
  case PI_PROFILING_INFO_COMMAND_END: {
    uint64_t ContextEnd;
    ZE_CALL(zeEventGetTimestamp(Event->ZeEvent, ZE_EVENT_TIMESTAMP_CONTEXT_END,
                                &ContextEnd));
    ContextEnd *= ZeTimerResolution;
    return ReturnValue(uint64_t{ContextEnd});
  }
  case PI_PROFILING_INFO_COMMAND_QUEUED:
  case PI_PROFILING_INFO_COMMAND_SUBMIT:
    // TODO: Support these when L0 supported is added.
    return ReturnValue(uint64_t{0});
  default:
    zePrint("piEventGetProfilingInfo: not supported ParamName\n");
    return PI_INVALID_VALUE;
  }

  return PI_SUCCESS;
}

pi_result piEventsWait(pi_uint32 NumEvents, const pi_event *EventList) {
  ze_result_t ZeResult;

  if (NumEvents && !EventList) {
    return PI_INVALID_EVENT;
  }

  for (uint32_t I = 0; I < NumEvents; I++) {
    ze_event_handle_t ZeEvent = EventList[I]->ZeEvent;
    zePrint("ZeEvent = %lx\n", pi_cast<std::uintptr_t>(ZeEvent));
    // TODO: Using UINT32_MAX for timeout should have the desired
    // effect of waiting until the event is trigerred, but it seems that
    // it is causing an OS crash, so use an interruptable loop for now.
    do {
      ZeResult = ZE_CALL_NOCHECK(zeEventHostSynchronize(ZeEvent, 100000));
    } while (ZeResult == ZE_RESULT_NOT_READY);

    // Check the result to be success.
    ZE_CALL(ZeResult);

    // NOTE: we are destroying associated command lists here to free
    // resources sooner in case RT is not calling piEventRelease soon enough.
    if (EventList[I]->ZeCommandList) {
      // Event has been signaled: Destroy the command list associated with the
      // call that generated the event.
      ZE_CALL(zeCommandListDestroy(EventList[I]->ZeCommandList));
      EventList[I]->ZeCommandList = nullptr;
    }
  }
  return PI_SUCCESS;
}

pi_result piEventSetCallback(pi_event Event, pi_int32 CommandExecCallbackType,
                             void (*PFnNotify)(pi_event Event,
                                               pi_int32 EventCommandStatus,
                                               void *UserData),
                             void *UserData) {
  die("piEventSetCallback: deprecated, to be removed");
  return PI_SUCCESS;
}

pi_result piEventSetStatus(pi_event Event, pi_int32 ExecutionStatus) {
  die("piEventSetStatus: deprecated, to be removed");
  return PI_SUCCESS;
}

pi_result piEventRetain(pi_event Event) {
  ++(Event->RefCount);
  return PI_SUCCESS;
}

pi_result piEventRelease(pi_event Event) {
  assert(Event);
  if (--(Event->RefCount) == 0) {
    if (Event->ZeCommandList) {
      // Destroy the command list associated with the call that generated
      // the event.
      ZE_CALL(zeCommandListDestroy(Event->ZeCommandList));
      Event->ZeCommandList = nullptr;
    }
    if (Event->CommandType == PI_COMMAND_TYPE_MEM_BUFFER_UNMAP &&
        Event->CommandData) {
      // Free the memory allocated in the piEnqueueMemBufferMap.
      ZE_CALL(zeDriverFreeMem(Event->Queue->Context->Device->Platform->ZeDriver,
                              Event->CommandData));
      Event->CommandData = nullptr;
    }
    ZE_CALL(zeEventDestroy(Event->ZeEvent));

    auto Context = Event->Context;
    ZE_CALL(Context->decrementAliveEventsInPool(Event->ZeEventPool));

    delete Event;
  }
  return PI_SUCCESS;
}

pi_result piextEventGetNativeHandle(pi_event Event,
                                    pi_native_handle *NativeHandle) {
  die("piextEventGetNativeHandle: not supported");
  return PI_SUCCESS;
}

pi_result piextEventCreateWithNativeHandle(pi_native_handle NativeHandle,
                                           pi_event *Event) {
  die("piextEventCreateWithNativeHandle: not supported");
  return PI_SUCCESS;
}

//
// Sampler
//
pi_result piSamplerCreate(pi_context Context,
                          const pi_sampler_properties *SamplerProperties,
                          pi_sampler *RetSampler) {

  assert(Context);
  assert(RetSampler);

  ze_device_handle_t ZeDevice = Context->Device->ZeDevice;

  ze_sampler_handle_t ZeSampler;
  ze_sampler_desc_t ZeSamplerDesc = {};
  ZeSamplerDesc.version = ZE_SAMPLER_DESC_VERSION_CURRENT;

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

  ZE_CALL(zeSamplerCreate(ZeDevice,
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

  die("piSamplerGetInfo: not implemented");
  return {};
}

pi_result piSamplerRetain(pi_sampler Sampler) {
  assert(Sampler);
  ++(Sampler->RefCount);
  return PI_SUCCESS;
}

pi_result piSamplerRelease(pi_sampler Sampler) {
  assert(Sampler);
  if (--(Sampler->RefCount) == 0) {
    ZE_CALL(zeSamplerDestroy(Sampler->ZeSampler));
    delete Sampler;
  }
  return PI_SUCCESS;
}

//
// Queue Commands
//
pi_result piEnqueueEventsWait(pi_queue Queue, pi_uint32 NumEventsInWaitList,
                              const pi_event *EventWaitList, pi_event *Event) {

  die("piEnqueueEventsWait: not implemented");
  return {};
}

pi_result piEnqueueEventsWaitWithBarrier(pi_queue Queue,
                                         pi_uint32 NumEventsInWaitList,
                                         const pi_event *EventWaitList,
                                         pi_event *Event) {

  assert(Queue);

  // Get a new command list to be used on this call
  ze_command_list_handle_t ZeCommandList = nullptr;
  if (auto Res = Queue->Context->Device->createCommandList(&ZeCommandList))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  if (Event) {
    auto Res = piEventCreate(Queue->Context, Event);
    if (Res != PI_SUCCESS)
      return Res;

    (*Event)->Queue = Queue;
    (*Event)->CommandType = PI_COMMAND_TYPE_USER;
    (*Event)->ZeCommandList = ZeCommandList;

    ZeEvent = (*Event)->ZeEvent;
  }

  // TODO: use unique_ptr with custom deleter in the whole Level Zero plugin for
  // wrapping ze_event_handle_t *ZeEventWaitList to avoid memory leaks in case
  // return will be called in ZE_CALL(ze***(...)), and thus
  // _pi_event::deleteZeEventList(ZeEventWaitList) won't be called.
  ze_event_handle_t *ZeEventWaitList =
      _pi_event::createZeEventList(NumEventsInWaitList, EventWaitList);

  ZE_CALL(zeCommandListAppendBarrier(ZeCommandList, ZeEvent,
                                     NumEventsInWaitList, ZeEventWaitList));

  _pi_event::deleteZeEventList(ZeEventWaitList);

  return PI_SUCCESS;
}

pi_result piEnqueueMemBufferRead(pi_queue Queue, pi_mem Src,
                                 pi_bool BlockingRead, size_t Offset,
                                 size_t Size, void *Dst,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {
  assert(Src);
  return enqueueMemCopyHelper(PI_COMMAND_TYPE_MEM_BUFFER_READ, Queue, Dst,
                              BlockingRead, Size,
                              pi_cast<char *>(Src->getZeHandle()) + Offset,
                              NumEventsInWaitList, EventWaitList, Event);
}

pi_result piEnqueueMemBufferReadRect(
    pi_queue Queue, pi_mem Buffer, pi_bool BlockingRead,
    const size_t *BufferOffset, const size_t *HostOffset, const size_t *Region,
    size_t BufferRowPitch, size_t BufferSlicePitch, size_t HostRowPitch,
    size_t HostSlicePitch, void *Ptr, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *Event) {

  assert(Buffer);
  return enqueueMemCopyRectHelper(
      PI_COMMAND_TYPE_MEM_BUFFER_READ_RECT, Queue, Buffer->getZeHandle(),
      static_cast<char *>(Ptr), BufferOffset, HostOffset, Region,
      BufferRowPitch, HostRowPitch, BufferSlicePitch, HostSlicePitch,
      BlockingRead, NumEventsInWaitList, EventWaitList, Event);
}

} // extern "C"

// Shared by all memory read/write/copy PI interfaces.
static pi_result
enqueueMemCopyHelper(pi_command_type CommandType, pi_queue Queue, void *Dst,
                     pi_bool BlockingWrite, size_t Size, const void *Src,
                     pi_uint32 NumEventsInWaitList,
                     const pi_event *EventWaitList, pi_event *Event) {

  assert(Queue);
  // Get a new command list to be used on this call
  ze_command_list_handle_t ZeCommandList = nullptr;
  if (auto Res = Queue->Context->Device->createCommandList(&ZeCommandList))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  if (Event) {
    auto Res = piEventCreate(Queue->Context, Event);
    if (Res != PI_SUCCESS)
      return Res;

    (*Event)->Queue = Queue;
    (*Event)->CommandType = CommandType;
    (*Event)->ZeCommandList = ZeCommandList;

    ZeEvent = (*Event)->ZeEvent;
  }

  ze_event_handle_t *ZeEventWaitList =
      _pi_event::createZeEventList(NumEventsInWaitList, EventWaitList);

  ZE_CALL(zeCommandListAppendWaitOnEvents(ZeCommandList, NumEventsInWaitList,
                                          ZeEventWaitList));

  ZE_CALL(
      zeCommandListAppendMemoryCopy(ZeCommandList, Dst, Src, Size, ZeEvent));

  if (auto Res = Queue->executeCommandList(ZeCommandList, BlockingWrite))
    return Res;

  zePrint("calling zeCommandListAppendMemoryCopy() with\n"
          "  xe_event %lx\n"
          "  NumEventsInWaitList %d:",
          pi_cast<std::uintptr_t>(ZeEvent), NumEventsInWaitList);
  for (pi_uint32 I = 0; I < NumEventsInWaitList; I++) {
    zePrint(" %lx", pi_cast<std::uintptr_t>(ZeEventWaitList[I]));
  }
  zePrint("\n");

  _pi_event::deleteZeEventList(ZeEventWaitList);

  return PI_SUCCESS;
}

// Shared by all memory read/write/copy rect PI interfaces.
static pi_result enqueueMemCopyRectHelper(
    pi_command_type CommandType, pi_queue Queue, void *SrcBuffer,
    void *DstBuffer, const size_t *SrcOrigin, const size_t *DstOrigin,
    const size_t *Region, size_t SrcRowPitch, size_t DstRowPitch,
    size_t SrcSlicePitch, size_t DstSlicePitch, pi_bool Blocking,
    pi_uint32 NumEventsInWaitList, const pi_event *EventWaitList,
    pi_event *Event) {

  assert(Region);
  assert(SrcOrigin);
  assert(DstOrigin);
  assert(Queue);

  // Get a new command list to be used on this call
  ze_command_list_handle_t ZeCommandList = nullptr;
  if (auto Res = Queue->Context->Device->createCommandList(&ZeCommandList))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  if (Event) {
    auto Res = piEventCreate(Queue->Context, Event);
    if (Res != PI_SUCCESS)
      return Res;

    (*Event)->Queue = Queue;
    (*Event)->CommandType = CommandType;
    (*Event)->ZeCommandList = ZeCommandList;

    ZeEvent = (*Event)->ZeEvent;
  }

  ze_event_handle_t *ZeEventWaitList =
      _pi_event::createZeEventList(NumEventsInWaitList, EventWaitList);

  ZE_CALL(zeCommandListAppendWaitOnEvents(ZeCommandList, NumEventsInWaitList,
                                          ZeEventWaitList));

  zePrint("calling zeCommandListAppendWaitOnEvents() with\n"
          "  NumEventsInWaitList %d:",
          pi_cast<std::uintptr_t>(ZeEvent), NumEventsInWaitList);
  for (pi_uint32 I = 0; I < NumEventsInWaitList; I++) {
    zePrint(" %lx", pi_cast<std::uintptr_t>(ZeEventWaitList[I]));
  }
  zePrint("\n");

  uint32_t SrcOriginX = pi_cast<uint32_t>(SrcOrigin[0]);
  uint32_t SrcOriginY = pi_cast<uint32_t>(SrcOrigin[1]);
  uint32_t SrcOriginZ = pi_cast<uint32_t>(SrcOrigin[2]);

  uint32_t SrcPitch = SrcRowPitch;
  if (SrcPitch == 0)
    SrcPitch = pi_cast<uint32_t>(Region[0]);

  if (SrcSlicePitch == 0)
    SrcSlicePitch = pi_cast<uint32_t>(Region[1]) * SrcPitch;

  uint32_t DstOriginX = pi_cast<uint32_t>(DstOrigin[0]);
  uint32_t DstOriginY = pi_cast<uint32_t>(DstOrigin[1]);
  uint32_t DstOriginZ = pi_cast<uint32_t>(DstOrigin[2]);

  uint32_t DstPitch = DstRowPitch;
  if (DstPitch == 0)
    DstPitch = pi_cast<uint32_t>(Region[0]);

  if (DstSlicePitch == 0)
    DstSlicePitch = pi_cast<uint32_t>(Region[1]) * DstPitch;

  uint32_t Width = pi_cast<uint32_t>(Region[0]);
  uint32_t Height = pi_cast<uint32_t>(Region[1]);
  uint32_t Depth = pi_cast<uint32_t>(Region[2]);

  const ze_copy_region_t ZeSrcRegion = {SrcOriginX, SrcOriginY, SrcOriginZ,
                                        Width,      Height,     Depth};
  const ze_copy_region_t ZeDstRegion = {DstOriginX, DstOriginY, DstOriginZ,
                                        Width,      Height,     Depth};

  ZE_CALL(zeCommandListAppendMemoryCopyRegion(
      ZeCommandList, DstBuffer, &ZeDstRegion, DstPitch, DstSlicePitch,
      SrcBuffer, &ZeSrcRegion, SrcPitch, SrcSlicePitch, nullptr));

  zePrint("calling zeCommandListAppendMemoryCopyRegion()\n");

  ZE_CALL(zeCommandListAppendBarrier(ZeCommandList, ZeEvent, 0, nullptr));

  zePrint("calling zeCommandListAppendBarrier() with Event %lx\n",
          pi_cast<std::uintptr_t>(ZeEvent));

  if (auto Res = Queue->executeCommandList(ZeCommandList, Blocking))
    return Res;

  _pi_event::deleteZeEventList(ZeEventWaitList);

  return PI_SUCCESS;
}

extern "C" {

pi_result piEnqueueMemBufferWrite(pi_queue Queue, pi_mem Buffer,
                                  pi_bool BlockingWrite, size_t Offset,
                                  size_t Size, const void *Ptr,
                                  pi_uint32 NumEventsInWaitList,
                                  const pi_event *EventWaitList,
                                  pi_event *Event) {

  assert(Buffer);
  return enqueueMemCopyHelper(PI_COMMAND_TYPE_MEM_BUFFER_WRITE, Queue,
                              pi_cast<char *>(Buffer->getZeHandle()) +
                                  Offset, // dst
                              BlockingWrite, Size,
                              Ptr, // src
                              NumEventsInWaitList, EventWaitList, Event);
}

pi_result piEnqueueMemBufferWriteRect(
    pi_queue Queue, pi_mem Buffer, pi_bool BlockingWrite,
    const size_t *BufferOffset, const size_t *HostOffset, const size_t *Region,
    size_t BufferRowPitch, size_t BufferSlicePitch, size_t HostRowPitch,
    size_t HostSlicePitch, const void *Ptr, pi_uint32 NumEventsInWaitList,
    const pi_event *EventWaitList, pi_event *Event) {

  assert(Buffer);
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

  assert(SrcBuffer);
  assert(DstBuffer);
  return enqueueMemCopyHelper(
      PI_COMMAND_TYPE_MEM_BUFFER_COPY, Queue,
      pi_cast<char *>(DstBuffer->getZeHandle()) + DstOffset,
      false, // blocking
      Size, pi_cast<char *>(SrcBuffer->getZeHandle()) + SrcOffset,
      NumEventsInWaitList, EventWaitList, Event);
}

pi_result
piEnqueueMemBufferCopyRect(pi_queue Queue, pi_mem SrcBuffer, pi_mem DstBuffer,
                           const size_t *SrcOrigin, const size_t *DstOrigin,
                           const size_t *Region, size_t SrcRowPitch,
                           size_t SrcSlicePitch, size_t DstRowPitch,
                           size_t DstSlicePitch, pi_uint32 NumEventsInWaitList,
                           const pi_event *EventWaitList, pi_event *Event) {

  assert(SrcBuffer);
  assert(DstBuffer);
  return enqueueMemCopyRectHelper(
      PI_COMMAND_TYPE_MEM_BUFFER_COPY_RECT, Queue, SrcBuffer->getZeHandle(),
      DstBuffer->getZeHandle(), SrcOrigin, DstOrigin, Region, SrcRowPitch,
      DstRowPitch, SrcSlicePitch, DstSlicePitch,
      false, // blocking
      NumEventsInWaitList, EventWaitList, Event);
}

} // extern "C"

static pi_result
enqueueMemFillHelper(pi_command_type CommandType, pi_queue Queue, void *Ptr,
                     const void *Pattern, size_t PatternSize, size_t Size,
                     pi_uint32 NumEventsInWaitList,
                     const pi_event *EventWaitList, pi_event *Event) {

  assert(Queue);
  // Get a new command list to be used on this call
  ze_command_list_handle_t ZeCommandList = nullptr;
  if (auto Res = Queue->Context->Device->createCommandList(&ZeCommandList))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  if (Event) {
    auto Res = piEventCreate(Queue->Context, Event);
    if (Res != PI_SUCCESS)
      return Res;

    (*Event)->Queue = Queue;
    (*Event)->CommandType = CommandType;
    (*Event)->ZeCommandList = ZeCommandList;

    ZeEvent = (*Event)->ZeEvent;
  }

  ze_event_handle_t *ZeEventWaitList =
      _pi_event::createZeEventList(NumEventsInWaitList, EventWaitList);

  ZE_CALL(zeCommandListAppendWaitOnEvents(ZeCommandList, NumEventsInWaitList,
                                          ZeEventWaitList));

  // Pattern size must be a power of two
  assert((PatternSize > 0) && ((PatternSize & (PatternSize - 1)) == 0));

  ZE_CALL(zeCommandListAppendMemoryFill(ZeCommandList, Ptr, Pattern,
                                        PatternSize, Size, ZeEvent));

  zePrint("calling zeCommandListAppendMemoryFill() with\n"
          "  xe_event %lx\n"
          "  NumEventsInWaitList %d:",
          pi_cast<pi_uint64>(ZeEvent), NumEventsInWaitList);
  for (pi_uint32 I = 0; I < NumEventsInWaitList; I++) {
    zePrint(" %lx", pi_cast<pi_uint64>(ZeEventWaitList[I]));
  }
  zePrint("\n");

  // Execute command list asynchronously, as the event will be used
  // to track down its completion.
  if (auto Res = Queue->executeCommandList(ZeCommandList))
    return Res;

  _pi_event::deleteZeEventList(ZeEventWaitList);

  return PI_SUCCESS;
}

extern "C" {

pi_result piEnqueueMemBufferFill(pi_queue Queue, pi_mem Buffer,
                                 const void *Pattern, size_t PatternSize,
                                 size_t Offset, size_t Size,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {

  assert(Buffer);
  return enqueueMemFillHelper(PI_COMMAND_TYPE_MEM_BUFFER_FILL, Queue,
                              pi_cast<char *>(Buffer->getZeHandle()) + Offset,
                              Pattern, PatternSize, Size, NumEventsInWaitList,
                              EventWaitList, Event);
}

pi_result
piEnqueueMemBufferMap(pi_queue Queue, pi_mem Buffer, pi_bool BlockingMap,
                      cl_map_flags MapFlags, // TODO: untie from OpenCL
                      size_t Offset, size_t Size, pi_uint32 NumEventsInWaitList,
                      const pi_event *EventWaitList, pi_event *Event,
                      void **RetMap) {

  // TODO: we don't implement read-only or write-only, always read-write.
  // assert((map_flags & CL_MAP_READ) != 0);
  // assert((map_flags & CL_MAP_WRITE) != 0);
  assert(Queue);
  assert(Buffer);

  // Get a new command list to be used on this call
  ze_command_list_handle_t ZeCommandList = nullptr;
  if (auto Res = Queue->Context->Device->createCommandList(&ZeCommandList))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  if (Event) {
    auto Res = piEventCreate(Queue->Context, Event);
    if (Res != PI_SUCCESS)
      return Res;

    (*Event)->Queue = Queue;
    (*Event)->CommandType = PI_COMMAND_TYPE_MEM_BUFFER_MAP;
    (*Event)->ZeCommandList = ZeCommandList;

    ZeEvent = (*Event)->ZeEvent;
  }

  ze_event_handle_t *ZeEventWaitList =
      _pi_event::createZeEventList(NumEventsInWaitList, EventWaitList);

  ZE_CALL(zeCommandListAppendWaitOnEvents(ZeCommandList, NumEventsInWaitList,
                                          ZeEventWaitList));

  // TODO: L0 is missing the memory "mapping" capabilities, so we are left
  // to doing new memory allocation and a copy (read).
  //
  // TODO: check if the input buffer is already allocated in shared
  // memory and thus is accessible from the host as is. Can we get SYCL RT
  // to predict/allocate in shared memory from the beginning?
  if (Buffer->MapHostPtr) {
    // NOTE: borrowing below semantics from OpenCL as SYCL RT relies on it.
    // It is also better for performance.
    //
    // "If the buffer object is created with CL_MEM_USE_HOST_PTR set in
    // mem_flags, the following will be true:
    // - The host_ptr specified in clCreateBuffer is guaranteed to contain the
    //   latest bits in the region being mapped when the clEnqueueMapBuffer
    //   command has completed.
    // - The pointer value returned by clEnqueueMapBuffer will be derived from
    //   the host_ptr specified when the buffer object is created."
    *RetMap = Buffer->MapHostPtr + Offset;
  } else {
    ze_host_mem_alloc_desc_t ZeDesc = {};
    ZeDesc.flags = ZE_HOST_MEM_ALLOC_FLAG_DEFAULT;
    ZE_CALL(zeDriverAllocHostMem(Queue->Context->Device->Platform->ZeDriver,
                                 &ZeDesc, Size,
                                 1, // TODO: alignment
                                 RetMap));
  }

  ZE_CALL(zeCommandListAppendMemoryCopy(
      ZeCommandList, *RetMap, pi_cast<char *>(Buffer->getZeHandle()) + Offset,
      Size, ZeEvent));

  if (auto Res = Queue->executeCommandList(ZeCommandList, BlockingMap))
    return Res;

  _pi_event::deleteZeEventList(ZeEventWaitList);

  return Buffer->addMapping(*RetMap, Offset, Size);
}

pi_result piEnqueueMemUnmap(pi_queue Queue, pi_mem MemObj, void *MappedPtr,
                            pi_uint32 NumEventsInWaitList,
                            const pi_event *EventWaitList, pi_event *Event) {

  // Get a new command list to be used on this call
  ze_command_list_handle_t ZeCommandList = nullptr;
  if (auto Res = Queue->Context->Device->createCommandList(&ZeCommandList))
    return Res;

  // TODO: handle the case when user does not care to follow the event
  // of unmap completion.
  assert(Event);

  ze_event_handle_t ZeEvent = nullptr;
  if (Event) {
    auto Res = piEventCreate(Queue->Context, Event);
    if (Res != PI_SUCCESS)
      return Res;

    (*Event)->Queue = Queue;
    (*Event)->CommandType = PI_COMMAND_TYPE_MEM_BUFFER_UNMAP;
    (*Event)->ZeCommandList = ZeCommandList;

    ZeEvent = (*Event)->ZeEvent;
  }

  ze_event_handle_t *ZeEventWaitList =
      _pi_event::createZeEventList(NumEventsInWaitList, EventWaitList);

  ZE_CALL(zeCommandListAppendWaitOnEvents(ZeCommandList, NumEventsInWaitList,
                                          ZeEventWaitList));

  // TODO: L0 is missing the memory "mapping" capabilities, so we are left
  // to doing copy (write back to the device).
  //
  // NOTE: Keep this in sync with the implementation of
  // piEnqueueMemBufferMap/piEnqueueMemImageMap.
  _pi_mem::Mapping MapInfo = {};
  if (pi_result Res = MemObj->removeMapping(MappedPtr, MapInfo))
    return Res;

  ZE_CALL(zeCommandListAppendMemoryCopy(
      ZeCommandList, pi_cast<char *>(MemObj->getZeHandle()) + MapInfo.Offset,
      MappedPtr, MapInfo.Size, ZeEvent));

  // NOTE: we still have to free the host memory allocated/returned by
  // piEnqueueMemBufferMap, but can only do so after the above copy
  // is completed. Instead of waiting for It here (blocking), we shall
  // do so in piEventRelease called for the pi_event tracking the unmap.
  (*Event)->CommandData = MemObj->MapHostPtr ? nullptr : MappedPtr;

  // Execute command list asynchronously, as the event will be used
  // to track down its completion.
  if (auto Res = Queue->executeCommandList(ZeCommandList))
    return Res;

  _pi_event::deleteZeEventList(ZeEventWaitList);

  return PI_SUCCESS;
}

pi_result piMemImageGetInfo(pi_mem Image, pi_image_info ParamName,
                            size_t ParamValueSize, void *ParamValue,
                            size_t *ParamValueSizeRet) {

  die("piMemImageGetInfo: not implemented");
  return {};
}

} // extern "C"

static ze_image_region_t getImageRegionHelper(pi_mem Mem, const size_t *Origin,
                                              const size_t *Region) {

  assert(Mem && Origin);
#ifndef NDEBUG
  assert(Mem->isImage());
  auto Image = static_cast<_pi_image *>(Mem);
  ze_image_desc_t ZeImageDesc = Image->ZeImageDesc;
#endif // !NDEBUG

  assert((ZeImageDesc.type == ZE_IMAGE_TYPE_1D && Origin[1] == 0 &&
          Origin[2] == 0) ||
         (ZeImageDesc.type == ZE_IMAGE_TYPE_1DARRAY && Origin[2] == 0) ||
         (ZeImageDesc.type == ZE_IMAGE_TYPE_2D && Origin[2] == 0) ||
         (ZeImageDesc.type == ZE_IMAGE_TYPE_3D));

  uint32_t OriginX = pi_cast<uint32_t>(Origin[0]);
  uint32_t OriginY = pi_cast<uint32_t>(Origin[1]);
  uint32_t OriginZ = pi_cast<uint32_t>(Origin[2]);

  assert(Region[0] && Region[1] && Region[2]);
  assert((ZeImageDesc.type == ZE_IMAGE_TYPE_1D && Region[1] == 1 &&
          Region[2] == 1) ||
         (ZeImageDesc.type == ZE_IMAGE_TYPE_1DARRAY && Region[2] == 1) ||
         (ZeImageDesc.type == ZE_IMAGE_TYPE_2D && Region[2] == 1) ||
         (ZeImageDesc.type == ZE_IMAGE_TYPE_3D));

  uint32_t Width = pi_cast<uint32_t>(Region[0]);
  uint32_t Height = pi_cast<uint32_t>(Region[1]);
  uint32_t Depth = pi_cast<uint32_t>(Region[2]);

  const ze_image_region_t ZeRegion = {OriginX, OriginY, OriginZ,
                                      Width,   Height,  Depth};
  return ZeRegion;
}

// Helper function to implement image read/write/copy.
static pi_result
enqueueMemImageCommandHelper(pi_command_type CommandType, pi_queue Queue,
                             const void *Src, // image or ptr
                             void *Dst,       // image or ptr
                             pi_bool IsBlocking, const size_t *SrcOrigin,
                             const size_t *DstOrigin, const size_t *Region,
                             size_t RowPitch, size_t SlicePitch,
                             pi_uint32 NumEventsInWaitList,
                             const pi_event *EventWaitList, pi_event *Event) {

  assert(Queue);
  // Get a new command list to be used on this call
  ze_command_list_handle_t ZeCommandList = nullptr;
  if (auto Res = Queue->Context->Device->createCommandList(&ZeCommandList))
    return Res;

  ze_event_handle_t ZeEvent = nullptr;
  if (Event) {
    auto Res = piEventCreate(Queue->Context, Event);
    if (Res != PI_SUCCESS)
      return Res;

    (*Event)->Queue = Queue;
    (*Event)->CommandType = CommandType;
    (*Event)->ZeCommandList = ZeCommandList;

    ZeEvent = (*Event)->ZeEvent;
  }

  ze_event_handle_t *ZeEventWaitList =
      _pi_event::createZeEventList(NumEventsInWaitList, EventWaitList);

  ZE_CALL(zeCommandListAppendWaitOnEvents(ZeCommandList, NumEventsInWaitList,
                                          ZeEventWaitList));

  if (CommandType == PI_COMMAND_TYPE_IMAGE_READ) {
    pi_mem SrcMem = pi_cast<pi_mem>(const_cast<void *>(Src));

    const ze_image_region_t ZeSrcRegion =
        getImageRegionHelper(SrcMem, SrcOrigin, Region);

    // TODO: L0 does not support row_pitch/slice_pitch for images yet.
    // Check that SYCL RT did not want pitch larger than default.
#ifndef NDEBUG
    assert(SrcMem->isImage());
    auto SrcImage = static_cast<_pi_image *>(SrcMem);
    const ze_image_desc_t &ZeImageDesc = SrcImage->ZeImageDesc;
    assert(RowPitch == 0 ||
           // special case RGBA image pitch equal to region's width
           (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32 &&
            RowPitch == 4 * 4 * ZeSrcRegion.width) ||
           (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16 &&
            RowPitch == 4 * 2 * ZeSrcRegion.width) ||
           (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8 &&
            RowPitch == 4 * ZeSrcRegion.width));
    assert(SlicePitch == 0 || SlicePitch == RowPitch * ZeSrcRegion.height);
#endif // !NDEBUG

    ZE_CALL(zeCommandListAppendImageCopyToMemory(
        ZeCommandList, Dst, pi_cast<ze_image_handle_t>(SrcMem->getZeHandle()),
        &ZeSrcRegion, ZeEvent));
  } else if (CommandType == PI_COMMAND_TYPE_IMAGE_WRITE) {
    pi_mem DstMem = pi_cast<pi_mem>(Dst);
    const ze_image_region_t ZeDstRegion =
        getImageRegionHelper(DstMem, DstOrigin, Region);

    // TODO: L0 does not support row_pitch/slice_pitch for images yet.
    // Check that SYCL RT did not want pitch larger than default.
#ifndef NDEBUG
    assert(DstMem->isImage());
    auto DstImage = static_cast<_pi_image *>(DstMem);
    const ze_image_desc_t &ZeImageDesc = DstImage->ZeImageDesc;
    assert(RowPitch == 0 ||
           // special case RGBA image pitch equal to region's width
           (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_32_32_32_32 &&
            RowPitch == 4 * 4 * ZeDstRegion.width) ||
           (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_16_16_16_16 &&
            RowPitch == 4 * 2 * ZeDstRegion.width) ||
           (ZeImageDesc.format.layout == ZE_IMAGE_FORMAT_LAYOUT_8_8_8_8 &&
            RowPitch == 4 * ZeDstRegion.width));
    assert(SlicePitch == 0 || SlicePitch == RowPitch * ZeDstRegion.height);
#endif // !NDEBUG

    ZE_CALL(zeCommandListAppendImageCopyFromMemory(
        ZeCommandList, pi_cast<ze_image_handle_t>(DstMem->getZeHandle()), Src,
        &ZeDstRegion, ZeEvent));
  } else if (CommandType == PI_COMMAND_TYPE_IMAGE_COPY) {
    pi_mem SrcImage = pi_cast<pi_mem>(const_cast<void *>(Src));
    pi_mem DstImage = pi_cast<pi_mem>(Dst);

    const ze_image_region_t ZeSrcRegion =
        getImageRegionHelper(SrcImage, SrcOrigin, Region);
    const ze_image_region_t ZeDstRegion =
        getImageRegionHelper(DstImage, DstOrigin, Region);

    ZE_CALL(zeCommandListAppendImageCopyRegion(
        ZeCommandList, pi_cast<ze_image_handle_t>(DstImage->getZeHandle()),
        pi_cast<ze_image_handle_t>(SrcImage->getZeHandle()), &ZeDstRegion,
        &ZeSrcRegion, ZeEvent));
  } else {
    zePrint("enqueueMemImageUpdate: unsupported image command type\n");
    return PI_INVALID_OPERATION;
  }

  if (auto Res = Queue->executeCommandList(ZeCommandList, IsBlocking))
    return Res;

  _pi_event::deleteZeEventList(ZeEventWaitList);

  return PI_SUCCESS;
}

extern "C" {

pi_result piEnqueueMemImageRead(pi_queue Queue, pi_mem Image,
                                pi_bool BlockingRead, const size_t *Origin,
                                const size_t *Region, size_t RowPitch,
                                size_t SlicePitch, void *Ptr,
                                pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList,
                                pi_event *Event) {

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
                                 pi_bool BlockingWrite, const size_t *Origin,
                                 const size_t *Region, size_t InputRowPitch,
                                 size_t InputSlicePitch, const void *Ptr,
                                 pi_uint32 NumEventsInWaitList,
                                 const pi_event *EventWaitList,
                                 pi_event *Event) {

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

pi_result piEnqueueMemImageCopy(pi_queue Queue, pi_mem SrcImage,
                                pi_mem DstImage, const size_t *SrcOrigin,
                                const size_t *DstOrigin, const size_t *Region,
                                pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList,
                                pi_event *Event) {

  return enqueueMemImageCommandHelper(
      PI_COMMAND_TYPE_IMAGE_COPY, Queue, SrcImage, DstImage,
      false, // is_blocking
      SrcOrigin, DstOrigin, Region,
      0, // row pitch
      0, // slice pitch
      NumEventsInWaitList, EventWaitList, Event);
}

pi_result piEnqueueMemImageFill(pi_queue Queue, pi_mem Image,
                                const void *FillColor, const size_t *Origin,
                                const size_t *Region,
                                pi_uint32 NumEventsInWaitList,
                                const pi_event *EventWaitList,
                                pi_event *Event) {

  die("piEnqueueMemImageFill: not implemented");
  return {};
}

pi_result piMemBufferPartition(pi_mem Buffer, pi_mem_flags Flags,
                               pi_buffer_create_type BufferCreateType,
                               void *BufferCreateInfo, pi_mem *RetMem) {

  assert(Buffer && !Buffer->isImage());
  assert(Flags == PI_MEM_FLAGS_ACCESS_RW);
  assert(BufferCreateType == PI_BUFFER_CREATE_TYPE_REGION);
  assert(!(static_cast<_pi_buffer *>(Buffer))->isSubBuffer() &&
         "Sub-buffer cannot be partitioned");
  assert(BufferCreateInfo);
  assert(RetMem);

  auto Region = (pi_buffer_region)BufferCreateInfo;
  assert(Region->size != 0u && "Invalid size");
  assert(Region->origin <= (Region->origin + Region->size) && "Overflow");
  try {
    *RetMem =
        new _pi_buffer(Buffer->Platform,
                       pi_cast<char *>(Buffer->getZeHandle()) +
                           Region->origin /* L0 memory handle */,
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

  die("piEnqueueNativeKernel: not implemented");
  return {};
}

// TODO: Check if the function_pointer_ret type can be converted to void**.
pi_result piextGetDeviceFunctionPointer(pi_device Device, pi_program Program,
                                        const char *FunctionName,
                                        pi_uint64 *FunctionPointerRet) {
  assert(Program != nullptr);
  ZE_CALL(zeModuleGetFunctionPointer(
      Program->ZeModule, FunctionName,
      reinterpret_cast<void **>(FunctionPointerRet)));
  return PI_SUCCESS;
}

pi_result piextUSMHostAlloc(void **ResultPtr, pi_context Context,
                            pi_usm_mem_properties *Properties, size_t Size,
                            pi_uint32 Alignment) {

  assert(Context);
  // Check that incorrect bits are not set in the properties.
  assert(!Properties || (Properties && !(*Properties & ~PI_MEM_ALLOC_FLAGS)));

  ze_host_mem_alloc_desc_t ZeDesc = {};
  ZeDesc.flags = ZE_HOST_MEM_ALLOC_FLAG_DEFAULT;
  // TODO: translate PI properties to L0 flags
  ZE_CALL(zeDriverAllocHostMem(Context->Device->Platform->ZeDriver, &ZeDesc,
                               Size, Alignment, ResultPtr));

  return PI_SUCCESS;
}

pi_result piextUSMDeviceAlloc(void **ResultPtr, pi_context Context,
                              pi_device Device,
                              pi_usm_mem_properties *Properties, size_t Size,
                              pi_uint32 Alignment) {

  assert(Context);
  assert(Device);
  // Check that incorrect bits are not set in the properties.
  assert(!Properties || (Properties && !(*Properties & ~PI_MEM_ALLOC_FLAGS)));

  // TODO: translate PI properties to L0 flags
  ze_device_mem_alloc_desc_t ZeDesc = {};
  ZeDesc.flags = ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT;
  ZeDesc.ordinal = 0;
  ZE_CALL(zeDriverAllocDeviceMem(Context->Device->Platform->ZeDriver, &ZeDesc,
                                 Size, Alignment, Device->ZeDevice, ResultPtr));

  return PI_SUCCESS;
}

pi_result piextUSMSharedAlloc(void **ResultPtr, pi_context Context,
                              pi_device Device,
                              pi_usm_mem_properties *Properties, size_t Size,
                              pi_uint32 Alignment) {

  assert(Context);
  assert(Device);
  // Check that incorrect bits are not set in the properties.
  assert(!Properties || (Properties && !(*Properties & ~PI_MEM_ALLOC_FLAGS)));

  // TODO: translate PI properties to L0 flags
  ze_host_mem_alloc_desc_t ZeHostDesc = {};
  ZeHostDesc.flags = ZE_HOST_MEM_ALLOC_FLAG_DEFAULT;
  ze_device_mem_alloc_desc_t ZeDevDesc = {};
  ZeDevDesc.flags = ZE_DEVICE_MEM_ALLOC_FLAG_DEFAULT;
  ZeDevDesc.ordinal = 0;
  ZE_CALL(zeDriverAllocSharedMem(Context->Device->Platform->ZeDriver,
                                 &ZeDevDesc, &ZeHostDesc, Size, Alignment,
                                 Device->ZeDevice, ResultPtr));

  return PI_SUCCESS;
}

pi_result piextUSMFree(pi_context Context, void *Ptr) {
  ZE_CALL(zeDriverFreeMem(Context->Device->Platform->ZeDriver, Ptr));
  return PI_SUCCESS;
}

pi_result piextKernelSetArgPointer(pi_kernel Kernel, pi_uint32 ArgIndex,
                                   size_t ArgSize, const void *ArgValue) {

  return piKernelSetArg(Kernel, ArgIndex, ArgSize, ArgValue);
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

  return enqueueMemFillHelper(
      // TODO: do we need a new command type for USM memset?
      PI_COMMAND_TYPE_MEM_BUFFER_FILL, Queue, Ptr,
      &Value, // It will be interpreted as an 8-bit value,
      1,      // which is indicated with this pattern_size==1
      Count, NumEventsInWaitlist, EventsWaitlist, Event);
}

pi_result piextUSMEnqueueMemcpy(pi_queue Queue, pi_bool Blocking, void *DstPtr,
                                const void *SrcPtr, size_t Size,
                                pi_uint32 NumEventsInWaitlist,
                                const pi_event *EventsWaitlist,
                                pi_event *Event) {

  if (!DstPtr) {
    return PI_INVALID_VALUE;
  }

  return enqueueMemCopyHelper(
      // TODO: do we need a new command type for this?
      PI_COMMAND_TYPE_MEM_BUFFER_COPY, Queue, DstPtr, Blocking, Size, SrcPtr,
      NumEventsInWaitlist, EventsWaitlist, Event);
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
                                  pi_uint32 NumEventsInWaitlist,
                                  const pi_event *EventsWaitlist,
                                  pi_event *Event) {
  assert(Queue);
  assert(!(Flags & ~PI_USM_MIGRATION_TBD0));

  // Get a new command list to be used on this call
  ze_command_list_handle_t ZeCommandList = nullptr;
  if (auto Res = Queue->Context->Device->createCommandList(&ZeCommandList))
    return Res;

  // TODO: do we need to create a unique command type for this?
  ze_event_handle_t ZeEvent = nullptr;
  if (Event) {
    auto Res = piEventCreate(Queue->Context, Event);
    if (Res != PI_SUCCESS)
      return Res;

    (*Event)->Queue = Queue;
    (*Event)->CommandType = PI_COMMAND_TYPE_USER;
    (*Event)->ZeCommandList = ZeCommandList;

    ZeEvent = (*Event)->ZeEvent;
  }

  ze_event_handle_t *ZeEventWaitList =
      _pi_event::createZeEventList(NumEventsInWaitlist, EventsWaitlist);

  ZE_CALL(zeCommandListAppendWaitOnEvents(ZeCommandList, NumEventsInWaitlist,
                                          ZeEventWaitList));

  // TODO: figure out how to translate "flags"
  ZE_CALL(zeCommandListAppendMemoryPrefetch(ZeCommandList, Ptr, Size));

  // TODO: L0 does not have a completion "event" with the prefetch API,
  // so manually add command to signal our event.
  ZE_CALL(zeCommandListAppendSignalEvent(ZeCommandList, ZeEvent));

  if (auto Res = Queue->executeCommandList(ZeCommandList, false))
    return Res;

  _pi_event::deleteZeEventList(ZeEventWaitList);

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
  assert(Queue);
  ze_memory_advice_t ZeAdvice = {};
  switch (Advice) {
  case PI_MEM_ADVICE_SET_READ_MOSTLY:
    ZeAdvice = ZE_MEMORY_ADVICE_SET_READ_MOSTLY;
    break;
  case PI_MEM_ADVICE_CLEAR_READ_MOSTLY:
    ZeAdvice = ZE_MEMORY_ADVICE_CLEAR_READ_MOSTLY;
    break;
  case PI_MEM_ADVICE_SET_PREFERRED_LOCATION:
    ZeAdvice = ZE_MEMORY_ADVICE_SET_PREFERRED_LOCATION;
    break;
  case PI_MEM_ADVICE_CLEAR_PREFERRED_LOCATION:
    ZeAdvice = ZE_MEMORY_ADVICE_CLEAR_PREFERRED_LOCATION;
    break;
  case PI_MEM_ADVICE_SET_ACCESSED_BY:
    ZeAdvice = ZE_MEMORY_ADVICE_SET_ACCESSED_BY;
    break;
  case PI_MEM_ADVICE_CLEAR_ACCESSED_BY:
    ZeAdvice = ZE_MEMORY_ADVICE_CLEAR_ACCESSED_BY;
    break;
  case PI_MEM_ADVICE_SET_NON_ATOMIC_MOSTLY:
    ZeAdvice = ZE_MEMORY_ADVICE_SET_NON_ATOMIC_MOSTLY;
    break;
  case PI_MEM_ADVICE_CLEAR_NON_ATOMIC_MOSTLY:
    ZeAdvice = ZE_MEMORY_ADVICE_CLEAR_NON_ATOMIC_MOSTLY;
    break;
  case PI_MEM_ADVICE_BIAS_CACHED:
    ZeAdvice = ZE_MEMORY_ADVICE_BIAS_CACHED;
    break;
  case PI_MEM_ADVICE_BIAS_UNCACHED:
    ZeAdvice = ZE_MEMORY_ADVICE_BIAS_UNCACHED;
    break;
  default:
    zePrint("piextUSMEnqueueMemAdvise: unexpected memory advise\n");
    return PI_INVALID_VALUE;
  }

  // Get a new command list to be used on this call
  ze_command_list_handle_t ZeCommandList = nullptr;
  if (auto Res = Queue->Context->Device->createCommandList(&ZeCommandList))
    return Res;

  // TODO: do we need to create a unique command type for this?
  ze_event_handle_t ZeEvent = nullptr;
  if (Event) {
    auto Res = piEventCreate(Queue->Context, Event);
    if (Res != PI_SUCCESS)
      return Res;

    (*Event)->Queue = Queue;
    (*Event)->CommandType = PI_COMMAND_TYPE_USER;
    (*Event)->ZeCommandList = ZeCommandList;

    ZeEvent = (*Event)->ZeEvent;
  }

  ZE_CALL(zeCommandListAppendMemAdvise(
      ZeCommandList, Queue->Context->Device->ZeDevice, Ptr, Length, ZeAdvice));

  // TODO: L0 does not have a completion "event" with the advise API,
  // so manually add command to signal our event.
  ZE_CALL(zeCommandListAppendSignalEvent(ZeCommandList, ZeEvent));

  Queue->executeCommandList(ZeCommandList, false);
  return PI_SUCCESS;
}

/// API to query information about USM allocated pointers
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
  assert(Context);
  ze_device_handle_t ZeDeviceHandle;
  ze_memory_allocation_properties_t ZeMemoryAllocationProperties = {
      ZE_MEMORY_ALLOCATION_PROPERTIES_VERSION_CURRENT};

  ZE_CALL(zeDriverGetMemAllocProperties(Context->Device->Platform->ZeDriver,
                                        Ptr, &ZeMemoryAllocationProperties,
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
  case PI_MEM_ALLOC_DEVICE: {
    // TODO: this wants pi_device, but we didn't remember it, and cannot
    // deduct from the L0 device.
    die("piextUSMGetMemAllocInfo: PI_MEM_ALLOC_DEVICE not implemented");
    break;
  }
  case PI_MEM_ALLOC_BASE_PTR: {
    void *Base;
    ZE_CALL(zeDriverGetMemAddressRange(Context->Device->Platform->ZeDriver, Ptr,
                                       &Base, nullptr));
    return ReturnValue(Base);
  }
  case PI_MEM_ALLOC_SIZE: {
    size_t Size;
    ZE_CALL(zeDriverGetMemAddressRange(Context->Device->Platform->ZeDriver, Ptr,
                                       nullptr, &Size));
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
  assert(Kernel);
  assert(ParamValue);
  if (ParamName == PI_USM_INDIRECT_ACCESS &&
      *(static_cast<const pi_bool *>(ParamValue)) == PI_TRUE) {
    // The whole point for users really was to not need to know anything
    // about the types of allocations kernel uses. So in DPC++ we always
    // just set all 3 modes for each kernel.
    bool ZeIndirectValue = true;
    ZE_CALL(zeKernelSetAttribute(Kernel->ZeKernel,
                                 ZE_KERNEL_ATTR_INDIRECT_SHARED_ACCESS,
                                 sizeof(bool), &ZeIndirectValue));
    ZE_CALL(zeKernelSetAttribute(Kernel->ZeKernel,
                                 ZE_KERNEL_ATTR_INDIRECT_DEVICE_ACCESS,
                                 sizeof(bool), &ZeIndirectValue));
    ZE_CALL(zeKernelSetAttribute(Kernel->ZeKernel,
                                 ZE_KERNEL_ATTR_INDIRECT_HOST_ACCESS,
                                 sizeof(bool), &ZeIndirectValue));
  } else {
    zePrint("piKernelSetExecInfo: unsupported ParamName\n");
    return PI_INVALID_VALUE;
  }

  return PI_SUCCESS;
}

pi_result piextProgramSetSpecializationConstant(pi_program Prog,
                                                pi_uint32 SpecID,
                                                size_t SpecSize,
                                                const void *SpecValue) {
  // TODO: implement
  die("piextProgramSetSpecializationConstant: not implemented");
  return {};
}

pi_result piPluginInit(pi_plugin *PluginInit) {
  assert(PluginInit);
  // TODO: handle versioning/targets properly.
  size_t PluginVersionSize = sizeof(PluginInit->PluginVersion);
  assert(strlen(_PI_H_VERSION_STRING) < PluginVersionSize);
  strncpy(PluginInit->PluginVersion, _PI_H_VERSION_STRING, PluginVersionSize);

#define _PI_API(api)                                                           \
  (PluginInit->PiFunctionTable).api = (decltype(&::api))(&api);
#include <CL/sycl/detail/pi.def>

  return PI_SUCCESS;
}

} // extern "C"
