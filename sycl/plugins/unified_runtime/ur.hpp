//===--------- ur.hpp - Unified Runtime  -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include <iostream>
#include <mutex>
#include <shared_mutex>

#include "zer_api.h"

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
class pi_shared_mutex : public std::shared_mutex {
public:
  void lock() {
    if (!SingleThreadMode)
      std::shared_mutex::lock();
  }
  bool try_lock() {
    return SingleThreadMode ? true : std::shared_mutex::try_lock();
  }
  void unlock() {
    if (!SingleThreadMode)
      std::shared_mutex::unlock();
  }

  void lock_shared() {
    if (!SingleThreadMode)
      std::shared_mutex::lock_shared();
  }
  bool try_lock_shared() {
    return SingleThreadMode ? true : std::shared_mutex::try_lock_shared();
  }
  void unlock_shared() {
    if (!SingleThreadMode)
      std::shared_mutex::unlock_shared();
  }
};

// Class which acts like std::mutex if SingleThreadMode variable is not set.
// If SingleThreadMode variable is set then mutex operations are turned into
// nop.
class pi_mutex : public std::mutex {
public:
  void lock() {
    if (!SingleThreadMode)
      std::mutex::lock();
  }
  bool try_lock() { return SingleThreadMode ? true : std::mutex::try_lock(); }
  void unlock() {
    if (!SingleThreadMode)
      std::mutex::unlock();
  }
};

// TODO: populate with target agnostic handling of UR platforms
struct _ur_platform {};
