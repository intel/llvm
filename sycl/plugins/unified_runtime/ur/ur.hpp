//===--------- ur.hpp - Unified Runtime  -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <atomic>
#include <cstdint>
#include <iostream>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <string>
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
  bool Computed{false};
  pi_mutex ZeCacheMutex;

  ZeCache() : T{} {}

  // Access to the fields of the original T data structure.
  T *operator->() {
    std::unique_lock<pi_mutex> Lock(ZeCacheMutex);
    if (!Computed) {
      Compute(*this);
      Computed = true;
    }
    return this;
  }
};

// This wrapper around std::atomic is created to limit operations with reference
// counter and to make allowed operations more transparent in terms of
// thread-safety in the plugin. increment() and load() operations do not need a
// mutex guard around them since the underlying data is already atomic.
// decrementAndTest() method is used to guard a code which needs to be
// executed when object's ref count becomes zero after release. This method also
// doesn't need a mutex guard because decrement operation is atomic and only one
// thread can reach ref count equal to zero, i.e. only a single thread can pass
// through this check.
struct ReferenceCounter {
  ReferenceCounter() : RefCount{1} {}

  // Reset the counter to the initial value.
  void reset() { RefCount = 1; }

  // Used when retaining an object.
  void increment() { RefCount++; }

  // Supposed to be used in pi*GetInfo* methods where ref count value is
  // requested.
  uint32_t load() { return RefCount.load(); }

  // This method allows to guard a code which needs to be executed when object's
  // ref count becomes zero after release. It is important to notice that only a
  // single thread can pass through this check. This is true because of several
  // reasons:
  //   1. Decrement operation is executed atomically.
  //   2. It is not allowed to retain an object after its refcount reaches zero.
  //   3. It is not allowed to release an object more times than the value of
  //   the ref count.
  // 2. and 3. basically means that we can't use an object at all as soon as its
  // refcount reaches zero. Using this check guarantees that code for deleting
  // an object and releasing its resources is executed once by a single thread
  // and we don't need to use any mutexes to guard access to this object in the
  // scope after this check. Of course if we access another objects in this code
  // (not the one which is being deleted) then access to these objects must be
  // guarded, for example with a mutex.
  bool decrementAndTest() { return --RefCount == 0; }

private:
  std::atomic<uint32_t> RefCount;
};

// Base class to store common data
struct _pi_object {
  _pi_object() : RefCount{} {}

  // Must be atomic to prevent data race when incrementing/decrementing.
  ReferenceCounter RefCount;

  // This mutex protects accesses to all the non-const member variables.
  // Exclusive access is required to modify any of these members.
  //
  // To get shared access to the object in a scope use std::shared_lock:
  //    std::shared_lock Lock(Obj->Mutex);
  // To get exclusive access to the object in a scope use std::scoped_lock:
  //    std::scoped_lock Lock(Obj->Mutex);
  //
  // If several pi objects are accessed in a scope then each object's mutex must
  // be locked. For example, to get write access to Obj1 and Obj2 and read
  // access to Obj3 in a scope use the following approach:
  //   std::shared_lock Obj3Lock(Obj3->Mutex, std::defer_lock);
  //   std::scoped_lock LockAll(Obj1->Mutex, Obj2->Mutex, Obj3Lock);
  pi_shared_mutex Mutex;
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
