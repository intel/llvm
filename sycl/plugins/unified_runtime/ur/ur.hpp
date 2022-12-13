//===--------- ur.hpp - Unified Runtime  -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <atomic>
#include <cstring>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

#include <zer_api.h>

// Terminates the process with a catastrophic error message.
[[noreturn]] inline void die(const char *Message) {
  std::cerr << "die: " << Message << std::endl;
  std::terminate();
}

// A single-threaded app has an opportunity to enable this mode to avoid
// overhead from mutex locking. Default value is 0 which means that single
// thread mode is disabled.
static const bool SingleThreadMode = [] {
  const char *Ret = std::getenv("SYCL_PI_LEVEL_ZERO_SINGLE_THREAD_MODE");
  const bool RetVal = Ret ? std::stoi(Ret) : 0;
  return RetVal;
}();

// Class which acts like shared_mutex if SingleThreadMode variable is not set.
// If SingleThreadMode variable is set then mutex operations are turned into
// nop.
class pi_shared_mutex {
  std::shared_mutex Mutex;

public:
  void lock() {
    if (!SingleThreadMode)
      Mutex.lock();
  }
  bool try_lock() { return SingleThreadMode ? true : Mutex.try_lock(); }
  void unlock() {
    if (!SingleThreadMode)
      Mutex.unlock();
  }

  void lock_shared() {
    if (!SingleThreadMode)
      Mutex.lock_shared();
  }
  bool try_lock_shared() {
    return SingleThreadMode ? true : Mutex.try_lock_shared();
  }
  void unlock_shared() {
    if (!SingleThreadMode)
      Mutex.unlock_shared();
  }
};

// Class which acts like std::mutex if SingleThreadMode variable is not set.
// If SingleThreadMode variable is set then mutex operations are turned into
// nop.
class pi_mutex {
  std::mutex Mutex;

public:
  void lock() {
    if (!SingleThreadMode)
      Mutex.lock();
  }
  bool try_lock() { return SingleThreadMode ? true : Mutex.try_lock(); }
  void unlock() {
    if (!SingleThreadMode)
      Mutex.unlock();
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
    while (MLock.test_and_set(std::memory_order_acquire))
      std::this_thread::yield();
  }
  void unlock() { MLock.clear(std::memory_order_release); }

private:
  std::atomic_flag MLock = ATOMIC_FLAG_INIT;
};

// Helper for one-liner validation
#define PI_ASSERT(condition, error)                                            \
  if (!(condition))                                                            \
    return error;

// TODO: populate with target agnostic handling of UR platforms
struct _ur_platform {};

// Controls tracing UR calls from within the UR itself.
extern bool PrintTrace;

// Apparatus for maintaining immutable cache of platforms.
//
// Note we only create a simple pointer variables such that C++ RT won't
// deallocate them automatically at the end of the main program.
// The heap memory allocated for these global variables reclaimed only at
// explicit tear-down.
extern std::vector<zer_platform_handle_t> *PiPlatformsCache;
extern SpinLock *PiPlatformsCacheMutex;
extern bool PiPlatformCachePopulated;

// The getInfo*/ReturnHelper facilities provide shortcut way of
// writing return bytes for the various getInfo APIs.
template <typename T, typename Assign>
zer_result_t getInfoImpl(size_t param_value_size, void *param_value,
                         size_t *param_value_size_ret, T value,
                         size_t value_size, Assign &&assign_func) {

  if (param_value != nullptr) {

    if (param_value_size < value_size) {
      return ZER_RESULT_INVALID_VALUE;
    }

    assign_func(param_value, value, value_size);
  }

  if (param_value_size_ret != nullptr) {
    *param_value_size_ret = value_size;
  }

  return ZER_RESULT_SUCCESS;
}

template <typename T>
zer_result_t getInfo(size_t param_value_size, void *param_value,
                     size_t *param_value_size_ret, T value) {

  auto assignment = [](void *param_value, T value, size_t value_size) {
    (void)value_size;
    *static_cast<T *>(param_value) = value;
  };

  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     sizeof(T), assignment);
}

template <typename T>
zer_result_t getInfoArray(size_t array_length, size_t param_value_size,
                          void *param_value, size_t *param_value_size_ret,
                          T *value) {
  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     array_length * sizeof(T), memcpy);
}

template <typename T, typename RetType>
zer_result_t getInfoArray(size_t array_length, size_t param_value_size,
                          void *param_value, size_t *param_value_size_ret,
                          T *value) {
  if (param_value) {
    memset(param_value, 0, param_value_size);
    for (uint32_t I = 0; I < array_length; I++)
      ((RetType *)param_value)[I] = (RetType)value[I];
  }
  if (param_value_size_ret)
    *param_value_size_ret = array_length * sizeof(RetType);
  return ZER_RESULT_SUCCESS;
}

template <>
inline zer_result_t
getInfo<const char *>(size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret, const char *value) {
  return getInfoArray(strlen(value) + 1, param_value_size, param_value,
                      param_value_size_ret, value);
}

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
  template <class T> zer_result_t operator()(const T &t) {
    return getInfo(param_value_size, param_value, param_value_size_ret, t);
  }

  // Array return value
  template <class T> zer_result_t operator()(const T *t, size_t s) {
    return getInfoArray(s, param_value_size, param_value, param_value_size_ret,
                        t);
  }

protected:
  size_t param_value_size;
  void *param_value;
  size_t *param_value_size_ret;
};
