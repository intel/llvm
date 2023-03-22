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

// Trace an internal PI call; returns in case of an error.
#define UR_CALL(Call)                                                          \
  {                                                                            \
    if (PrintTrace)                                                            \
      fprintf(stderr, "UR ---> %s\n", #Call);                                  \
    ur_result_t Result = (Call);                                               \
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

// Controls Level Zero calls tracing.
enum DebugLevel {
  ZE_DEBUG_NONE = 0x0,
  ZE_DEBUG_BASIC = 0x1,
  ZE_DEBUG_VALIDATION = 0x2,
  ZE_DEBUG_CALL_COUNT = 0x4,
  ZE_DEBUG_ALL = -1
};

const int UrL0Debug = [] {
  const char *ZeDebugMode = std::getenv("ZE_DEBUG");
  const char *UrL0DebugMode = std::getenv("UR_L0_DEBUG");
  uint32_t DebugMode = 0;
  if (ZeDebugMode) {
    DebugMode = std::atoi(ZeDebugMode);
  } else if (UrL0DebugMode) {
    DebugMode = std::atoi(UrL0DebugMode);
  }
  return DebugMode;
}();

const int ZeDebug = UrL0Debug;

// Controls Level Zero calls serialization to w/a Level Zero driver being not MT
// ready. Recognized values (can be used as a bit mask):
enum {
  UrL0SerializeNone =
      0, // no locking or blocking (except when SYCL RT requested blocking)
  UrL0SerializeLock = 1, // locking around each UR_CALL
  UrL0SerializeBlock =
      2, // blocking UR calls, where supported (usually in enqueue commands)
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

static const uint32_t UrL0Serialize = [] {
  const char *ZeSerializeMode = std::getenv("ZE_SERIALIZE");
  const char *UrL0SerializeMode = std::getenv("UR_L0_SERIALIZE");
  uint32_t SerializeModeValue = 0;
  if (ZeSerializeMode) {
    SerializeModeValue = std::atoi(ZeSerializeMode);
  } else if (UrL0SerializeMode) {
    SerializeModeValue = std::atoi(UrL0SerializeMode);
  }
  return SerializeModeValue;
}();

const int ZeSerialize = UrL0Debug;

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

// Prints to stderr if ZE_DEBUG allows it
void zePrint(const char *Format, ...);

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
