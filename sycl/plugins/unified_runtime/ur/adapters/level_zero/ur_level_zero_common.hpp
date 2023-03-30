//===--------- ur_level_zero_common.hpp - Level Zero Adapter ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <cassert>
#include <cstdarg>
#include <map>
#include <unordered_map>

#include <sycl/detail/pi.h>
#include <ur/ur.hpp>
#include <ur_api.h>
#include <ze_api.h>
#include <zes_api.h>

#include "ur_level_zero_context.hpp"
#include "ur_level_zero_device.hpp"
#include "ur_level_zero_event.hpp"
#include "ur_level_zero_mem.hpp"
#include "ur_level_zero_module.hpp"
#include "ur_level_zero_platform.hpp"
#include "ur_level_zero_program.hpp"
#include "ur_level_zero_queue.hpp"
#include "ur_level_zero_sampler.hpp"

template <class To, class From> To ur_cast(From Value) {
  // TODO: see if more sanity checks are possible.
  assert(sizeof(From) == sizeof(To));
  return (To)(Value);
}

static auto getUrResultString = [](ur_result_t Result) {
  switch (Result) {
  case UR_RESULT_SUCCESS:
    return "UR_RESULT_SUCCESS";
  case UR_RESULT_ERROR_INVALID_OPERATION:
    return "UR_RESULT_ERROR_INVALID_OPERATION";
  case UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES:
    return "UR_RESULT_ERROR_INVALID_QUEUE_PROPERTIES";
  case UR_RESULT_ERROR_INVALID_QUEUE:
    return "UR_RESULT_ERROR_INVALID_QUEUE";
  case UR_RESULT_ERROR_INVALID_VALUE:
    return "UR_RESULT_ERROR_INVALID_VALUE";
  case UR_RESULT_ERROR_INVALID_CONTEXT:
    return "UR_RESULT_ERROR_INVALID_CONTEXT";
  case UR_RESULT_ERROR_INVALID_PLATFORM:
    return "UR_RESULT_ERROR_INVALID_PLATFORM";
  case UR_RESULT_ERROR_INVALID_BINARY:
    return "UR_RESULT_ERROR_INVALID_BINARY";
  case UR_RESULT_ERROR_INVALID_PROGRAM:
    return "UR_RESULT_ERROR_INVALID_PROGRAM";
  case UR_RESULT_ERROR_INVALID_SAMPLER:
    return "UR_RESULT_ERROR_INVALID_SAMPLER";
  case UR_RESULT_ERROR_INVALID_BUFFER_SIZE:
    return "UR_RESULT_ERROR_INVALID_BUFFER_SIZE";
  case UR_RESULT_ERROR_INVALID_MEM_OBJECT:
    return "UR_RESULT_ERROR_INVALID_MEM_OBJECT";
  case UR_RESULT_ERROR_INVALID_EVENT:
    return "UR_RESULT_ERROR_INVALID_EVENT";
  case UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST:
    return "UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST";
  case UR_RESULT_ERROR_MISALIGNED_SUB_BUFFER_OFFSET:
    return "UR_RESULT_ERROR_MISALIGNED_SUB_BUFFER_OFFSET";
  case UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE:
    return "UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE";
  case UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE:
    return "UR_RESULT_ERROR_COMPILER_NOT_AVAILABLE";
  case UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE:
    return "UR_RESULT_ERROR_PROFILING_INFO_NOT_AVAILABLE";
  case UR_RESULT_ERROR_DEVICE_NOT_FOUND:
    return "UR_RESULT_ERROR_DEVICE_NOT_FOUND";
  case UR_RESULT_ERROR_INVALID_DEVICE:
    return "UR_RESULT_ERROR_INVALID_DEVICE";
  case UR_RESULT_ERROR_DEVICE_LOST:
    return "UR_RESULT_ERROR_DEVICE_LOST";
  case UR_RESULT_ERROR_DEVICE_REQUIRES_RESET:
    return "UR_RESULT_ERROR_DEVICE_REQUIRES_RESET";
  case UR_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE:
    return "UR_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE";
  case UR_RESULT_ERROR_DEVICE_PARTITION_FAILED:
    return "UR_RESULT_ERROR_DEVICE_PARTITION_FAILED";
  case UR_RESULT_ERROR_INVALID_DEVICE_PARTITION_COUNT:
    return "UR_RESULT_ERROR_INVALID_DEVICE_PARTITION_COUNT";
  case UR_RESULT_ERROR_INVALID_WORK_ITEM_SIZE:
    return "UR_RESULT_ERROR_INVALID_WORK_ITEM_SIZE";
  case UR_RESULT_ERROR_INVALID_WORK_DIMENSION:
    return "UR_RESULT_ERROR_INVALID_WORK_DIMENSION";
  case UR_RESULT_ERROR_INVALID_KERNEL_ARGS:
    return "UR_RESULT_ERROR_INVALID_KERNEL_ARGS";
  case UR_RESULT_ERROR_INVALID_KERNEL:
    return "UR_RESULT_ERROR_INVALID_KERNEL";
  case UR_RESULT_ERROR_INVALID_KERNEL_NAME:
    return "UR_RESULT_ERROR_INVALID_KERNEL_NAME";
  case UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
    return "UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
  case UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
    return "UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
  case UR_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
    return "UR_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
  case UR_RESULT_ERROR_INVALID_IMAGE_SIZE:
    return "UR_RESULT_ERROR_INVALID_IMAGE_SIZE";
  case UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return "UR_RESULT_ERROR_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case UR_RESULT_ERROR_IMAGE_FORMAT_NOT_SUPPORTED:
    return "UR_RESULT_ERROR_IMAGE_FORMAT_NOT_SUPPORTED";
  case UR_RESULT_ERROR_MEM_OBJECT_ALLOCATION_FAILURE:
    return "UR_RESULT_ERROR_MEM_OBJECT_ALLOCATION_FAILURE";
  case UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE:
    return "UR_RESULT_ERROR_INVALID_PROGRAM_EXECUTABLE";
  case UR_RESULT_ERROR_UNINITIALIZED:
    return "UR_RESULT_ERROR_UNINITIALIZED";
  case UR_RESULT_ERROR_OUT_OF_HOST_MEMORY:
    return "UR_RESULT_ERROR_OUT_OF_HOST_MEMORY";
  case UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
    return "UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
  case UR_RESULT_ERROR_OUT_OF_RESOURCES:
    return "UR_RESULT_ERROR_OUT_OF_RESOURCES";
  case UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE:
    return "UR_RESULT_ERROR_PROGRAM_BUILD_FAILURE";
  case UR_RESULT_ERROR_PROGRAM_LINK_FAILURE:
    return "UR_RESULT_ERROR_PROGRAM_LINK_FAILURE";
  case UR_RESULT_ERROR_UNSUPPORTED_VERSION:
    return "UR_RESULT_ERROR_UNSUPPORTED_VERSION";
  case UR_RESULT_ERROR_UNSUPPORTED_FEATURE:
    return "UR_RESULT_ERROR_UNSUPPORTED_FEATURE";
  case UR_RESULT_ERROR_INVALID_ARGUMENT:
    return "UR_RESULT_ERROR_INVALID_ARGUMENT";
  case UR_RESULT_ERROR_INVALID_NULL_HANDLE:
    return "UR_RESULT_ERROR_INVALID_NULL_HANDLE";
  case UR_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
    return "UR_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
  case UR_RESULT_ERROR_INVALID_NULL_POINTER:
    return "UR_RESULT_ERROR_INVALID_NULL_POINTER";
  case UR_RESULT_ERROR_INVALID_SIZE:
    return "UR_RESULT_ERROR_INVALID_SIZE";
  case UR_RESULT_ERROR_UNSUPPORTED_SIZE:
    return "UR_RESULT_ERROR_UNSUPPORTED_SIZE";
  case UR_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
    return "UR_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
  case UR_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
    return "UR_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
  case UR_RESULT_ERROR_INVALID_ENUMERATION:
    return "UR_RESULT_ERROR_INVALID_ENUMERATION";
  case UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
    return "UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
  case UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
    return "UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
  case UR_RESULT_ERROR_INVALID_NATIVE_BINARY:
    return "UR_RESULT_ERROR_INVALID_NATIVE_BINARY";
  case UR_RESULT_ERROR_INVALID_GLOBAL_NAME:
    return "UR_RESULT_ERROR_INVALID_GLOBAL_NAME";
  case UR_RESULT_ERROR_INVALID_FUNCTION_NAME:
    return "UR_RESULT_ERROR_INVALID_FUNCTION_NAME";
  case UR_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
    return "UR_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
  case UR_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
    return "UR_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
  case UR_RESULT_ERROR_PROGRAM_UNLINKED:
    return "UR_RESULT_ERROR_PROGRAM_UNLINKED";
  case UR_RESULT_ERROR_OVERLAPPING_REGIONS:
    return "UR_RESULT_ERROR_OVERLAPPING_REGIONS";
  case UR_RESULT_ERROR_INVALID_HOST_PTR:
    return "UR_RESULT_ERROR_INVALID_HOST_PTR";
  case UR_RESULT_ERROR_INVALID_USM_SIZE:
    return "UR_RESULT_ERROR_INVALID_USM_SIZE";
  case UR_RESULT_ERROR_OBJECT_ALLOCATION_FAILURE:
    return "UR_RESULT_ERROR_OBJECT_ALLOCATION_FAILURE";
  case UR_RESULT_ERROR_ADAPTER_SPECIFIC:
    return "UR_RESULT_ERROR_ADAPTER_SPECIFIC";
  default:
    return "UR_RESULT_ERROR_UNKNOWN";
  }
};

// Trace an internal PI call; returns in case of an error.
#define UR_CALL(Call)                                                          \
  {                                                                            \
    if (PrintTrace)                                                            \
      fprintf(stderr, "UR ---> %s\n", #Call);                                  \
    ur_result_t Result = (Call);                                               \
    if (PrintTrace)                                                            \
      fprintf(stderr, "UR <--- %s(%s)\n", #Call, getUrResultString(Result));   \
    if (Result != UR_RESULT_SUCCESS)                                           \
      return Result;                                                           \
  }

// Controls UR L0 calls tracing.
enum UrDebugLevel {
  UR_L0_DEBUG_NONE = 0x0,
  UR_L0_DEBUG_BASIC = 0x1,
  UR_L0_DEBUG_VALIDATION = 0x2,
  UR_L0_DEBUG_CALL_COUNT = 0x4,
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

// Prints to stderr if UR_L0_DEBUG allows it
void urPrint(const char *Format, ...);

// Helper for one-liner validation
#define UR_ASSERT(condition, error)                                            \
  if (!(condition))                                                            \
    return error;

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

// Map Level Zero runtime error code to UR error code.
ur_result_t ze2urResult(ze_result_t ZeResult);

// Trace a call to Level-Zero RT
#define ZE2UR_CALL(ZeName, ZeArgs)                                             \
  {                                                                            \
    ze_result_t ZeResult = ZeName ZeArgs;                                      \
    if (auto Result = ZeCall().doCall(ZeResult, #ZeName, #ZeArgs, true))       \
      return ze2urResult(Result);                                              \
  }

// Perform traced call to L0 without checking for errors
#define ZE_CALL_NOCHECK(ZeName, ZeArgs)                                        \
  ZeCall().doCall(ZeName ZeArgs, #ZeName, #ZeArgs, false)
