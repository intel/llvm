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
#include <ze_api.h>
#include <zer_api.h>
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
static zer_result_t ze2urResult(ze_result_t ZeResult) {
  static std::unordered_map<ze_result_t, zer_result_t> ErrorMapping = {
      {ZE_RESULT_SUCCESS, ZER_RESULT_SUCCESS},
      {ZE_RESULT_ERROR_DEVICE_LOST, ZER_RESULT_ERROR_DEVICE_LOST},
      {ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS, ZER_RESULT_INVALID_OPERATION},
      {ZE_RESULT_ERROR_NOT_AVAILABLE, ZER_RESULT_INVALID_OPERATION},
      {ZE_RESULT_ERROR_UNINITIALIZED, ZER_RESULT_INVALID_PLATFORM},
      {ZE_RESULT_ERROR_INVALID_ARGUMENT, ZER_RESULT_ERROR_INVALID_ARGUMENT},
      {ZE_RESULT_ERROR_INVALID_NULL_POINTER, ZER_RESULT_INVALID_VALUE},
      {ZE_RESULT_ERROR_INVALID_SIZE, ZER_RESULT_INVALID_VALUE},
      {ZE_RESULT_ERROR_UNSUPPORTED_SIZE, ZER_RESULT_INVALID_VALUE},
      {ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT, ZER_RESULT_INVALID_VALUE},
      {ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT,
       ZER_RESULT_INVALID_EVENT},
      {ZE_RESULT_ERROR_INVALID_ENUMERATION, ZER_RESULT_INVALID_VALUE},
      {ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION, ZER_RESULT_INVALID_VALUE},
      {ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT, ZER_RESULT_INVALID_VALUE},
      {ZE_RESULT_ERROR_INVALID_NATIVE_BINARY, ZER_RESULT_INVALID_BINARY},
      {ZE_RESULT_ERROR_INVALID_KERNEL_NAME, ZER_RESULT_INVALID_KERNEL_NAME},
      {ZE_RESULT_ERROR_INVALID_FUNCTION_NAME,
       ZER_RESULT_ERROR_INVALID_FUNCTION_NAME},
      {ZE_RESULT_ERROR_OVERLAPPING_REGIONS, ZER_RESULT_INVALID_OPERATION},
      {ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION,
       ZER_RESULT_INVALID_WORK_GROUP_SIZE},
      {ZE_RESULT_ERROR_MODULE_BUILD_FAILURE,
       ZER_RESULT_ERROR_MODULE_BUILD_FAILURE},
      {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY,
       ZER_RESULT_ERROR_OUT_OF_DEVICE_MEMORY},
      {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY,
       ZER_RESULT_ERROR_OUT_OF_HOST_MEMORY}};

  auto It = ErrorMapping.find(ZeResult);
  if (It == ErrorMapping.end()) {
    return ZER_RESULT_ERROR_UNKNOWN;
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

struct _ur_platform_handle_t : public _ur_platform {
  _ur_platform_handle_t(ze_driver_handle_t Driver) : ZeDriver{Driver} {}
  // Performs initialization of a newly constructed PI platform.
  zer_result_t initialize();

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
};

using ur_platform_handle_t = _ur_platform_handle_t *;

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

  void setZeUSMImport(ur_platform_handle_t Platform) {
    // Whether env var SYCL_USM_HOSTPTR_IMPORT has been set requesting
    // host ptr import during buffer creation.
    const char *USMHostPtrImportStr = std::getenv("SYCL_USM_HOSTPTR_IMPORT");
    if (!USMHostPtrImportStr || std::atoi(USMHostPtrImportStr) == 0)
      return;

    // Check if USM hostptr import feature is available.
    ze_driver_handle_t DriverHandle = Platform->ZeDriver;
    if (ZE_CALL_NOCHECK(zeDriverGetExtensionFunctionAddress,
                        (DriverHandle, "zexDriverImportExternalPointer",
                         reinterpret_cast<void **>(
                             &zexDriverImportExternalPointer))) == 0) {
      ZE_CALL_NOCHECK(
          zeDriverGetExtensionFunctionAddress,
          (DriverHandle, "zexDriverReleaseImportedPointer",
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
  void doZeUSMImport(ze_driver_handle_t DriverHandle, void *HostPtr,
                     size_t Size) {
    ZE_CALL_NOCHECK(zexDriverImportExternalPointer,
                    (DriverHandle, HostPtr, Size));
  }
  void doZeUSMRelease(ze_driver_handle_t DriverHandle, void *HostPtr) {
    ZE_CALL_NOCHECK(zexDriverReleaseImportedPointer, (DriverHandle, HostPtr));
  }
};

// Helper wrapper for working with USM import extension in Level Zero.
extern ZeUSMImportExtension ZeUSMImport;

// This will count the calls to Level-Zero
extern std::map<const char *, int> *ZeCallCount;

// Some opencl extensions we know are supported by all Level Zero devices.
constexpr char ZE_SUPPORTED_EXTENSIONS[] =
    "cl_khr_il_program cl_khr_subgroups cl_intel_subgroups "
    "cl_intel_subgroups_short cl_intel_required_subgroup_size ";
