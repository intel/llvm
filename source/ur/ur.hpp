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
#include <shared_mutex>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include <ur_api.h>

#include "ur_util.hpp"

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
#define __SYCL_UR_PROGRAM_METADATA_TAG_NEED_FINALIZATION "Requires finalization"

// Terminates the process with a catastrophic error message.
[[noreturn]] inline void die(const char *Message) {
  std::cerr << "die: " << Message << std::endl;
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

  // Access to the fields of the original T data structure.
  T *operator->() {
    std::call_once(Computed, Compute, static_cast<T &>(*this));
    return this;
  }
};

// Helper for one-liner validation
#define UR_ASSERT(condition, error)                                            \
  if (!(condition))                                                            \
    return error;

// TODO: populate with target agnostic handling of UR platforms
struct _ur_platform {};

// Controls tracing UR calls from within the UR itself.
extern bool PrintTrace;

// The getInfo*/ReturnHelper facilities provide shortcut way of
// writing return bytes for the various getInfo APIs.
namespace ur {
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

  auto assignment = [](void *param_value, T value, size_t value_size) {
    std::ignore = value_size;
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
  ur_result_t operator()(const T *t, size_t s) {
    return ur::getInfoArray<T, RetType>(s, param_value_size, param_value,
                                        param_value_size_ret, t);
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
