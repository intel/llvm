//===--------- ur_level_zero.hpp - Level Zero Adapter -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <cassert>
#include <list>
#include <map>
#include <stdarg.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <ur/ur.hpp>
#include <ur_api.h>
#include <ze_api.h>
#include <zes_api.h>

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

// Controls Level Zero calls serialization to w/a Level Zero driver being not MT
// ready. Recognized values (can be used as a bit mask):
enum {
  ZeSerializeNone =
      0, // no locking or blocking (except when SYCL RT requested blocking)
  ZeSerializeLock = 1, // locking around each ZE_CALL
  ZeSerializeBlock =
      2, // blocking ZE calls, where supported (usually in enqueue commands)
};
static const uint32_t ZeSerialize = [] {
  const char *SerializeMode = std::getenv("ZE_SERIALIZE");
  const uint32_t SerializeModeValue =
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

// Map Level Zero runtime error code to UR error code.
static ur_result_t ze2urResult(ze_result_t ZeResult) {
  static std::unordered_map<ze_result_t, ur_result_t> ErrorMapping = {
      {ZE_RESULT_SUCCESS, UR_RESULT_SUCCESS},
      {ZE_RESULT_ERROR_DEVICE_LOST, UR_RESULT_ERROR_DEVICE_LOST},
      {ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS,
       UR_RESULT_ERROR_INVALID_OPERATION},
      {ZE_RESULT_ERROR_NOT_AVAILABLE, UR_RESULT_ERROR_INVALID_OPERATION},
      {ZE_RESULT_ERROR_UNINITIALIZED, UR_RESULT_ERROR_INVALID_PLATFORM},
      {ZE_RESULT_ERROR_INVALID_ARGUMENT, UR_RESULT_ERROR_INVALID_ARGUMENT},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, UR_RESULT_ERROR_INVALID_VALUE},
      {ZE_RESULT_ERROR_INVALID_SIZE, UR_RESULT_ERROR_INVALID_VALUE},
      {ZE_RESULT_ERROR_UNSUPPORTED_SIZE, UR_RESULT_ERROR_INVALID_VALUE},
      {ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT, UR_RESULT_ERROR_INVALID_VALUE},
      {ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT,
       UR_RESULT_ERROR_INVALID_EVENT},
      {ZE_RESULT_ERROR_INVALID_ENUMERATION, UR_RESULT_ERROR_INVALID_VALUE},
      {ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION, UR_RESULT_ERROR_INVALID_VALUE},
      {ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT, UR_RESULT_ERROR_INVALID_VALUE},
      {ZE_RESULT_ERROR_INVALID_NATIVE_BINARY, UR_RESULT_ERROR_INVALID_BINARY},
      {ZE_RESULT_ERROR_INVALID_KERNEL_NAME,
       UR_RESULT_ERROR_INVALID_KERNEL_NAME},
      {ZE_RESULT_ERROR_INVALID_FUNCTION_NAME,
       UR_RESULT_ERROR_INVALID_FUNCTION_NAME},
      {ZE_RESULT_ERROR_OVERLAPPING_REGIONS, UR_RESULT_ERROR_INVALID_OPERATION},
      {ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION,
       UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE},
      {ZE_RESULT_ERROR_MODULE_BUILD_FAILURE,
       UR_RESULT_ERROR_MODULE_BUILD_FAILURE},
      {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY,
       UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY},
      {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY, UR_RESULT_ERROR_OUT_OF_HOST_MEMORY}};

  auto It = ErrorMapping.find(ZeResult);
  if (It == ErrorMapping.end()) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return It->second;
}

// Controls Level Zero calls tracing.
enum DebugLevel {
  ZE_DEBUG_NONE = 0x0,
  ZE_DEBUG_BASIC = 0x1,
  ZE_DEBUG_VALIDATION = 0x2,
  ZE_DEBUG_CALL_COUNT = 0x4,
  ZE_DEBUG_ALL = -1
};

const int ZeDebug = [] {
  const char *DebugMode = std::getenv("ZE_DEBUG");
  return DebugMode ? std::atoi(DebugMode) : ZE_DEBUG_NONE;
}();

// Prints to stderr if ZE_DEBUG allows it
void zePrint(const char *Format, ...);

// This function will ensure compatibility with both Linux and Windows for
// setting environment variables.
bool setEnvVar(const char *name, const char *value);

// Perform traced call to L0 without checking for errors
#define ZE_CALL_NOCHECK(ZeName, ZeArgs)                                        \
  ZeCall().doCall(ZeName ZeArgs, #ZeName, #ZeArgs, false)

struct _ur_platform_handle_t;
// using ur_platform_handle_t = _ur_platform_handle_t *;
struct _ur_device_handle_t;
// using ur_device_handle_t = _ur_device_handle_t *;

struct _ur_platform_handle_t : public _ur_platform {
  _ur_platform_handle_t(ze_driver_handle_t Driver) : ZeDriver{Driver} {}
  // Performs initialization of a newly constructed PI platform.
  ur_result_t initialize();

  // Level Zero lacks the notion of a platform, but there is a driver, which is
  // a pretty good fit to keep here.
  ze_driver_handle_t ZeDriver;

  // Cache versions info from zeDriverGetProperties.
  std::string ZeDriverVersion;
  std::string ZeDriverApiVersion;
  ze_api_version_t ZeApiVersion;

  // Cache driver extensions
  std::unordered_map<std::string, uint32_t> zeDriverExtensionMap;

  // Flags to tell whether various Level Zero platform extensions are available.
  bool ZeDriverGlobalOffsetExtensionFound{false};
  bool ZeDriverModuleProgramExtensionFound{false};

  // Cache UR devices for reuse
  std::vector<std::unique_ptr<ur_device_handle_t_>> PiDevicesCache;
  pi_shared_mutex PiDevicesCacheMutex;
  bool DeviceCachePopulated = false;

  // Check the device cache and load it if necessary.
  ur_result_t populateDeviceCacheIfNeeded();

  // Return the PI device from cache that represents given native device.
  // If not found, then nullptr is returned.
  ur_device_handle_t getDeviceFromNativeHandle(ze_device_handle_t);
};

struct _ur_device_handle_t : _pi_object {
  _ur_device_handle_t(ze_device_handle_t Device, ur_platform_handle_t Plt,
                      ur_device_handle_t ParentDevice = nullptr)
      : ZeDevice{Device}, Platform{Plt}, RootDevice{ParentDevice},
        ImmCommandListsPreferred{false}, ZeDeviceProperties{},
        ZeDeviceComputeProperties{} {
    // NOTE: one must additionally call initialize() to complete
    // UR device creation.
  }

  // The helper structure that keeps info about a command queue groups of the
  // device. It is not changed after it is initialized.
  struct queue_group_info_t {
    enum type {
      MainCopy,
      LinkCopy,
      Compute,
      Size // must be last
    };

    // Keep the ordinal of the commands group as returned by
    // zeDeviceGetCommandQueueGroupProperties. A value of "-1" means that
    // there is no such queue group available in the Level Zero runtime.
    int32_t ZeOrdinal{-1};

    // Keep the index of the specific queue in this queue group where
    // all the command enqueues of the corresponding type should go to.
    // The value of "-1" means that no hard binding is defined and
    // implementation can choose specific queue index on its own.
    int32_t ZeIndex{-1};

    // Keeps the queue group properties.
    ZeStruct<ze_command_queue_group_properties_t> ZeProperties;
  };

  std::vector<queue_group_info_t> QueueGroup =
      std::vector<queue_group_info_t>(queue_group_info_t::Size);

  // This returns "true" if a main copy engine is available for use.
  bool hasMainCopyEngine() const {
    return QueueGroup[queue_group_info_t::MainCopy].ZeOrdinal >= 0;
  }

  // This returns "true" if a link copy engine is available for use.
  bool hasLinkCopyEngine() const {
    return QueueGroup[queue_group_info_t::LinkCopy].ZeOrdinal >= 0;
  }

  // This returns "true" if a main or link copy engine is available for use.
  bool hasCopyEngine() const {
    return hasMainCopyEngine() || hasLinkCopyEngine();
  }

  // Initialize the entire UR device.
  // Optional param `SubSubDeviceOrdinal` `SubSubDeviceIndex` are the compute
  // command queue ordinal and index respectively, used to initialize
  // sub-sub-devices.
  ur_result_t initialize(int SubSubDeviceOrdinal = -1,
                         int SubSubDeviceIndex = -1);

  // Level Zero device handle.
  // This field is only set at _ur_device_handle_t creation time, and cannot
  // change. Therefore it can be accessed without holding a lock on this
  // _ur_device_handle_t.
  const ze_device_handle_t ZeDevice;

  // Keep the subdevices that are partitioned from this ur_device_handle_t for
  // reuse The order of sub-devices in this vector is repeated from the
  // ze_device_handle_t array that are returned from zeDeviceGetSubDevices()
  // call, which will always return sub-devices in the fixed same order.
  std::vector<ur_device_handle_t> SubDevices;

  // PI platform to which this device belongs.
  // This field is only set at _ur_device_handle_t creation time, and cannot
  // change. Therefore it can be accessed without holding a lock on this
  // _ur_device_handle_t.
  ur_platform_handle_t Platform;

  // Root-device of a sub-device, null if this is not a sub-device.
  // This field is only set at _ur_device_handle_t creation time, and cannot
  // change. Therefore it can be accessed without holding a lock on this
  // _ur_device_handle_t.
  const ur_device_handle_t RootDevice;

  // Whether to use immediate commandlists for queues on this device.
  // For some devices (e.g. PVC) immediate commandlists are preferred.
  bool ImmCommandListsPreferred;

  enum ImmCmdlistMode {
    // Immediate commandlists are not used.
    NotUsed,
    // One set of compute and copy immediate commandlists per queue.
    PerQueue,
    // One set of compute and copy immediate commandlists per host thread that
    // accesses the queue.
    PerThreadPerQueue
  };
  // Return whether to use immediate commandlists for this device.
  ImmCmdlistMode useImmediateCommandLists();

  bool isSubDevice() { return RootDevice != nullptr; }

  // Is this a Data Center GPU Max series (aka PVC).
  bool isPVC() { return (ZeDeviceProperties->deviceId & 0xff0) == 0xbd0; }

  // Does this device represent a single compute slice?
  bool isCCS() const {
    return QueueGroup[_ur_device_handle_t::queue_group_info_t::Compute]
               .ZeIndex >= 0;
  }

  // Cache of the immutable device properties.
  ZeCache<ZeStruct<ze_device_properties_t>> ZeDeviceProperties;
  ZeCache<ZeStruct<ze_device_compute_properties_t>> ZeDeviceComputeProperties;
  ZeCache<ZeStruct<ze_device_image_properties_t>> ZeDeviceImageProperties;
  ZeCache<ZeStruct<ze_device_module_properties_t>> ZeDeviceModuleProperties;
  ZeCache<std::pair<std::vector<ZeStruct<ze_device_memory_properties_t>>,
                    std::vector<ZeStruct<ze_device_memory_ext_properties_t>>>>
      ZeDeviceMemoryProperties;
  ZeCache<ZeStruct<ze_device_memory_access_properties_t>>
      ZeDeviceMemoryAccessProperties;
  ZeCache<ZeStruct<ze_device_cache_properties_t>> ZeDeviceCacheProperties;
};

// TODO: make it into a ur_device_handle_t class member
const std::pair<int, int>
getRangeOfAllowedCopyEngines(const ur_device_handle_t &Device);

class ZeUSMImportExtension {
  // Pointers to functions that import/release host memory into USM
  ze_result_t (*zexDriverImportExternalPointer)(ze_driver_handle_t hDriver,
                                                void *, size_t) = nullptr;
  ze_result_t (*zexDriverReleaseImportedPointer)(ze_driver_handle_t,
                                                 void *) = nullptr;

public:
  // Whether user has requested Import/Release, and platform supports it.
  bool Enabled;

  ZeUSMImportExtension() : Enabled{false} {}

  void setZeUSMImport(_ur_platform_handle_t *Platform);
  void doZeUSMImport(ze_driver_handle_t DriverHandle, void *HostPtr,
                     size_t Size);
  void doZeUSMRelease(ze_driver_handle_t DriverHandle, void *HostPtr);
};

// Helper wrapper for working with USM import extension in Level Zero.
extern ZeUSMImportExtension ZeUSMImport;

// This will count the calls to Level-Zero
extern std::map<const char *, int> *ZeCallCount;

// Some opencl extensions we know are supported by all Level Zero devices.
constexpr char ZE_SUPPORTED_EXTENSIONS[] =
    "cl_khr_il_program cl_khr_subgroups cl_intel_subgroups "
    "cl_intel_subgroups_short cl_intel_required_subgroup_size ";
