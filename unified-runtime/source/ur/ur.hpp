//===--------- ur.hpp - Unified Runtime  ----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include <ur_api.h>

#include "logger/ur_logger.hpp"
#include "ur_util.hpp"

// Helper for one-liner validation
#define UR_ASSERT(condition, error)                                            \
  if (!(condition))                                                            \
    return error;

// Trace an internal UR call; returns in case of an error.
#define UR_CALL(Call)                                                          \
  {                                                                            \
    if (PrintTrace)                                                            \
      UR_LOG(QUIET, "UR ---> {}", #Call);                                      \
    ur_result_t Result = (Call);                                               \
    if (PrintTrace)                                                            \
      UR_LOG(QUIET, "UR <--- {}({})", #Call, getUrResultString(Result));       \
    if (Result != UR_RESULT_SUCCESS)                                           \
      return Result;                                                           \
  }

// Trace an internal UR call; throw in case of an error.
#define UR_CALL_THROWS(Call)                                                   \
  {                                                                            \
    if (PrintTrace)                                                            \
      UR_LOG(QUIET, "UR ---> {}", #Call);                                      \
    ur_result_t Result = (Call);                                               \
    if (PrintTrace)                                                            \
      UR_LOG(QUIET, "UR <--- {}({})", #Call, getUrResultString(Result));       \
    if (Result != UR_RESULT_SUCCESS)                                           \
      throw Result;                                                            \
  }

// Trace an internal UR call; ignore errors (useful in destructors).
#define UR_CALL_NOCHECK(Call)                                                  \
  {                                                                            \
    if (PrintTrace)                                                            \
      UR_LOG(QUIET, "UR ---> {}", #Call);                                      \
    (void)(Call);                                                              \
    if (PrintTrace)                                                            \
      UR_LOG(QUIET, "UR <--- {}", #Call);                                      \
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
  case UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
    return "UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
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
  case UR_RESULT_ERROR_INVALID_NATIVE_BINARY:
    return "UR_RESULT_ERROR_INVALID_NATIVE_BINARY";
  case UR_RESULT_ERROR_INVALID_GLOBAL_NAME:
    return "UR_RESULT_ERROR_INVALID_GLOBAL_NAME";
  case UR_RESULT_ERROR_FUNCTION_ADDRESS_NOT_AVAILABLE:
    return "UR_RESULT_ERROR_FUNCTION_ADDRESS_NOT_AVAILABLE";
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

template <class To, class From> To ur_cast(From Value) {
  // TODO: see if more sanity checks are possible.
  assert(sizeof(From) == sizeof(To));
  return (To)(Value);
}

template <> uint32_t inline ur_cast(uint64_t Value) {
  // Cast value and check that we don't lose any information.
  uint32_t CastedValue = (uint32_t)(Value);
  assert((uint64_t)CastedValue == Value);
  return CastedValue;
}

// TODO: promote all of the below extensions to the Unified Runtime
//       and get rid of these ZER_EXT constants.
const ur_device_info_t UR_EXT_DEVICE_INFO_OPENCL_C_VERSION =
    (ur_device_info_t)0x103D;

const ur_command_t UR_EXT_COMMAND_TYPE_USER =
    (ur_command_t)((uint32_t)UR_COMMAND_FORCE_UINT32 - 1);

/// Program metadata tags recognized by the UR adapters. For kernels the tag
/// must appear after the kernel name.
#define __SYCL_UR_PROGRAM_METADATA_TAG_REQD_WORK_GROUP_SIZE                    \
  "@reqd_work_group_size"
#define __SYCL_UR_PROGRAM_METADATA_GLOBAL_ID_MAPPING "@global_id_mapping"
#define __SYCL_UR_PROGRAM_METADATA_TAG_MAX_WORK_GROUP_SIZE                     \
  "@max_work_group_size"
#define __SYCL_UR_PROGRAM_METADATA_TAG_MAX_LINEAR_WORK_GROUP_SIZE              \
  "@max_linear_work_group_size"
#define __SYCL_UR_PROGRAM_METADATA_TAG_REQD_SUB_GROUP_SIZE                     \
  "@reqd_sub_group_size"
#define __SYCL_UR_PROGRAM_METADATA_TAG_NEED_FINALIZATION "Requires finalization"

// Terminates the process with a catastrophic error message.
[[noreturn]] inline void die(const char *Message) {
  UR_LOG(QUIET, "ur_die: {}", Message);
  std::terminate();
}

// A single-threaded app has an opportunity to enable this mode to avoid
// overhead from mutex locking. Default value is 0 which means that single
// thread mode is disabled.
static const bool SingleThreadMode = [] {
  auto UrRet = ur_getenv("UR_L0_SINGLE_THREAD_MODE");
  auto PiRet = ur_getenv("SYCL_PI_LEVEL_ZERO_SINGLE_THREAD_MODE");
  const bool RetVal =
      UrRet ? std::stoi(*UrRet) : (PiRet ? std::stoi(*PiRet) : 0);
  return RetVal;
}();

// Class which acts like shared_mutex if SingleThreadMode variable is not set.
// If SingleThreadMode variable is set then mutex operations are turned into
// nop.
class ur_shared_mutex {
  std::shared_mutex Mutex;

public:
  void lock() {
    if (!SingleThreadMode) {
      Mutex.lock();
    }
  }
  bool try_lock() { return SingleThreadMode ? true : Mutex.try_lock(); }
  void unlock() {
    if (!SingleThreadMode) {
      Mutex.unlock();
    }
  }

  void lock_shared() {
    if (!SingleThreadMode) {
      Mutex.lock_shared();
    }
  }
  bool try_lock_shared() {
    return SingleThreadMode ? true : Mutex.try_lock_shared();
  }
  void unlock_shared() {
    if (!SingleThreadMode) {
      Mutex.unlock_shared();
    }
  }
};

// Class which acts like std::mutex if SingleThreadMode variable is not set.
// If SingleThreadMode variable is set then mutex operations are turned into
// nop.
class ur_mutex {
  std::mutex Mutex;
  friend class ur_lock;

public:
  void lock() {
    if (!SingleThreadMode) {
      Mutex.lock();
    }
  }
  bool try_lock() { return SingleThreadMode ? true : Mutex.try_lock(); }
  void unlock() {
    if (!SingleThreadMode) {
      Mutex.unlock();
    }
  }
};

class ur_lock {
  std::unique_lock<std::mutex> Lock;

public:
  explicit ur_lock(ur_mutex &Mutex) {
    if (!SingleThreadMode) {
      Lock = std::unique_lock<std::mutex>(Mutex.Mutex);
    }
  }
};

/// SpinLock is a synchronization primitive, that uses atomic variable and
/// causes thread trying acquire lock wait in loop while repeatedly check if
/// the lock is available.
///
/// One important feature of this implementation is that std::atomic<bool> can
/// be zero-initialized. This allows SpinLock to have trivial constructor and
/// destructor, which makes it possible to use it in global context (unlike
/// std::mutex, that doesn't provide such guarantees).
class SpinLock {
public:
  void lock() {
    while (MLock.test_and_set(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }
  void unlock() { MLock.clear(std::memory_order_release); }

private:
  std::atomic_flag MLock = ATOMIC_FLAG_INIT;
};

// The wrapper for immutable data.
// The data is initialized only once at first access (via ->) with the
// initialization function provided in Init. All subsequent access to
// the data just returns the already stored data.
//
template <class T> struct ZeCache : private T {
  // The initialization function takes a reference to the data
  // it is going to initialize, since it is private here in
  // order to disallow access other than through "->".
  //
  using InitFunctionType = std::function<void(T &)>;
  InitFunctionType Compute{nullptr};
  std::once_flag Computed;

  ZeCache() : T{} {}

  T &get() {
    std::call_once(Computed, Compute, static_cast<T &>(*this));
    return *this;
  }

  // Access to the fields of the original T data structure.
  T *operator->() { return &get(); }
};

struct ur_dditable_t;

// TODO: populate with target agnostic handling of UR platforms
struct ur_platform {};

// Controls tracing UR calls from within the UR itself.
extern bool PrintTrace;

// The getInfo*/ReturnHelper facilities provide shortcut way of
// writing return bytes for the various getInfo APIs.
namespace ur {

// Base class for handles, stores the ddi table used by the loader to
// dispatch to the correct adapter implementation of entry points.
template <typename getddi> struct handle_base {
  const ur_dditable_t *ddi_table = nullptr;

  handle_base() { ddi_table = getddi::value(); };

  // Handles are non-copyable.
  handle_base(const handle_base &) = delete;
  handle_base &operator=(const handle_base &) = delete;
};

template <typename T, typename Assign>
ur_result_t getInfoImpl(size_t param_value_size, void *param_value,
                        size_t *param_value_size_ret, T value,
                        size_t value_size, Assign &&assign_func) {
  if (!param_value && !param_value_size_ret) {
    return UR_RESULT_ERROR_INVALID_NULL_POINTER;
  }

  if (param_value != nullptr) {

    if (param_value_size < value_size) {
      return UR_RESULT_ERROR_INVALID_SIZE;
    }

    assign_func(param_value, value, value_size);
  }

  if (param_value_size_ret != nullptr) {
    *param_value_size_ret = value_size;
  }

  return UR_RESULT_SUCCESS;
}

template <typename T>
ur_result_t getInfo(size_t param_value_size, void *param_value,
                    size_t *param_value_size_ret, T value) {

  auto assignment = [](void *param_value, T value, size_t /*value_size*/) {
    *static_cast<T *>(param_value) = value;
  };

  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     sizeof(T), assignment);
}

template <typename T>
ur_result_t getInfoArray(size_t array_length, size_t param_value_size,
                         void *param_value, size_t *param_value_size_ret,
                         const T *value) {
  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     array_length * sizeof(T), memcpy);
}

template <typename T, typename RetType>
ur_result_t getInfoArray(size_t array_length, size_t param_value_size,
                         void *param_value, size_t *param_value_size_ret,
                         const T *value) {
  if (param_value) {
    memset(param_value, 0, param_value_size);
    for (uint32_t I = 0; I < array_length; I++) {
      ((RetType *)param_value)[I] = (RetType)value[I];
    }
  }
  if (param_value_size_ret) {
    *param_value_size_ret = array_length * sizeof(RetType);
  }
  return UR_RESULT_SUCCESS;
}

template <>
inline ur_result_t
getInfo<const char *>(size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret, const char *value) {
  return getInfoArray(strlen(value) + 1, param_value_size, param_value,
                      param_value_size_ret, value);
}

inline ur_result_t getInfoEmpty(size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  return getInfoImpl(param_value_size, param_value, param_value_size_ret, 0, 0,
                     [](void *, int, size_t) {});
}
} // namespace ur

class UrReturnHelper {
public:
  UrReturnHelper(size_t param_value_size, void *param_value,
                 size_t *param_value_size_ret)
      : param_value_size(param_value_size), param_value(param_value),
        param_value_size_ret(param_value_size_ret) {}

  // A version where in/out info size is represented by a single pointer
  // to a value which is updated on return
  UrReturnHelper(size_t *param_value_size, void *param_value)
      : param_value_size(*param_value_size), param_value(param_value),
        param_value_size_ret(param_value_size) {}

  // Scalar return value
  template <class T> ur_result_t operator()(const T &t) {
    return ur::getInfo(param_value_size, param_value, param_value_size_ret, t);
  }

  // Array return value
  template <class T> ur_result_t operator()(const T *t, size_t s) {
    return ur::getInfoArray(s, param_value_size, param_value,
                            param_value_size_ret, t);
  }

  // Array return value where element type is differrent from T
  template <class RetType, class T>
  std::enable_if_t<!std::is_same_v<RetType, T>, ur_result_t>
  operator()(const T *t, size_t s) {
    return ur::getInfoArray<T, RetType>(s, param_value_size, param_value,
                                        param_value_size_ret, t);
  }

  // Special case when there is no return value
  ur_result_t operator()(std::nullopt_t) {
    return ur::getInfoEmpty(param_value_size, param_value,
                            param_value_size_ret);
  }

protected:
  size_t param_value_size;
  void *param_value;
  size_t *param_value_size_ret;
};

template <typename T> class Result {
public:
  Result(ur_result_t err) : value_or_err(err) {}
  Result(T value) : value_or_err(std::move(value)) {}
  Result() : value_or_err(UR_RESULT_ERROR_UNINITIALIZED) {}

  bool is_err() { return std::holds_alternative<ur_result_t>(value_or_err); }
  explicit operator bool() const { return !is_err(); }

  const T *get_value() { return std::get_if<T>(&value_or_err); }

  ur_result_t get_error() {
    auto *err = std::get_if<ur_result_t>(&value_or_err);
    return err ? *err : UR_RESULT_SUCCESS;
  }

private:
  std::variant<ur_result_t, T> value_or_err;
};

// Helper to make sure each x, y, z dim divide the global dimension.
//
// In/Out: ThreadsPerBlockInDim - The dimension of workgroup in some dimension
// In:     GlobalWorkSizeInDim  - The global size in some dimension
static inline void
roundToHighestFactorOfGlobalSize(size_t &ThreadsPerBlockInDim,
                                 const size_t GlobalWorkSizeInDim) {
  while (ThreadsPerBlockInDim > 1 &&
         GlobalWorkSizeInDim % ThreadsPerBlockInDim) {
    --ThreadsPerBlockInDim;
  }
}

// Returns whether or not Value is a power of 2
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
bool isPowerOf2(const T Value) {
  return Value && !(Value & (Value - 1));
}

// Helper to make sure each x, y, z dim divide the global dimension.
// Additionally it makes sure that the inner dimension always is a power of 2
//
// In/Out: ThreadsPerBlock      - The size of wg in 3d
// In:     GlobalSize           - The global size in 3d (if dim < 3 then outer
//                                                       dims == 1)
// In:     MaxBlockDim          - The max size of block in 3d
// In:     MaxBlockSize         - The max total size of block in all dimensions
// In:     WorkDim              - The workdim (1, 2 or 3)
static inline void roundToHighestFactorOfGlobalSizeIn3d(
    size_t *ThreadsPerBlock, const size_t *GlobalSize,
    const size_t *MaxBlockDim, const size_t MaxBlockSize) {
  assert(GlobalSize[0] && "GlobalSize[0] cannot be zero");
  assert(GlobalSize[1] && "GlobalSize[1] cannot be zero");
  assert(GlobalSize[2] && "GlobalSize[2] cannot be zero");

  ThreadsPerBlock[0] =
      std::min(GlobalSize[0], std::min(MaxBlockSize, MaxBlockDim[0]));
  do {
    roundToHighestFactorOfGlobalSize(ThreadsPerBlock[0], GlobalSize[0]);
  } while (!isPowerOf2(ThreadsPerBlock[0]) && ThreadsPerBlock[0] > 32 &&
           --ThreadsPerBlock[0]);

  ThreadsPerBlock[1] =
      std::min(GlobalSize[1],
               std::min(MaxBlockSize / ThreadsPerBlock[0], MaxBlockDim[1]));
  roundToHighestFactorOfGlobalSize(ThreadsPerBlock[1], GlobalSize[1]);

  ThreadsPerBlock[2] = std::min(
      GlobalSize[2],
      std::min(MaxBlockSize / (ThreadsPerBlock[1] * ThreadsPerBlock[0]),
               MaxBlockDim[2]));
  roundToHighestFactorOfGlobalSize(ThreadsPerBlock[2], GlobalSize[2]);
}
