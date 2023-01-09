//==---------- pi_opencl.hpp - OpenCL Plugin -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \defgroup sycl_pi_ocl OpenCL Plugin
/// \ingroup sycl_pi

/// \file pi_opencl.hpp
/// Declarations for vOpenCL Plugin. It is the interface between device-agnostic
/// SYCL runtime layer and underlying OpenCL runtime.
///
/// \ingroup sycl_pi_ocl

#ifndef PI_OPENCL_HPP
#define PI_OPENCL_HPP

#include <atomic>
#include <climits>
#include <mutex>
#include <regex>
#include <shared_mutex>
#include <string>
#include <sycl/detail/pi.h>

// This version should be incremented for any change made to this file or its
// corresponding .cpp file.
#define _PI_OPENCL_PLUGIN_VERSION 1

#define _PI_OPENCL_PLUGIN_VERSION_STRING                                       \
  _PI_PLUGIN_VERSION_STRING(_PI_OPENCL_PLUGIN_VERSION)

// A single-threaded app has an opportunity to enable this mode to avoid
// overhead from mutex locking. Default value is 0 which means that single
// thread mode is disabled.
static const bool SingleThreadMode = [] {
  const char *Ret = std::getenv("SYCL_PI_OPENCL_SINGLE_THREAD_MODE");
  const bool RetVal = Ret ? std::stoi(Ret) : 0;
  return RetVal;
}();

// Class which acts like shared_mutex if SingleThreadMode variable is not set.
// If SingleThreadMode variable is set then mutex operations are turned into
// nop.
class pi_shared_mutex_ocl {
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
  pi_shared_mutex_ocl Mutex;
};

namespace OCLV {
class OpenCLVersion {
protected:
  unsigned int ocl_major;
  unsigned int ocl_minor;

public:
  OpenCLVersion() : ocl_major(0), ocl_minor(0) {}

  OpenCLVersion(unsigned int ocl_major, unsigned int ocl_minor)
      : ocl_major(ocl_major), ocl_minor(ocl_minor) {
    if (!isValid())
      ocl_major = ocl_minor = 0;
  }

  OpenCLVersion(const char *version) : OpenCLVersion(std::string(version)) {}

  OpenCLVersion(const std::string &version) : ocl_major(0), ocl_minor(0) {
    /* The OpenCL specification defines the full version string as
     * 'OpenCL<space><ocl_major_version.ocl_minor_version><space><platform-specific
     * information>' for platforms and as
     * 'OpenCL<space><ocl_major_version.ocl_minor_version><space><vendor-specific
     * information>' for devices.
     */
    std::regex rx("OpenCL ([0-9]+)\\.([0-9]+)");
    std::smatch match;

    if (std::regex_search(version, match, rx) && (match.size() == 3)) {
      ocl_major = strtoul(match[1].str().c_str(), nullptr, 10);
      ocl_minor = strtoul(match[2].str().c_str(), nullptr, 10);

      if (!isValid())
        ocl_major = ocl_minor = 0;
    }
  }

  bool operator==(const OpenCLVersion &v) const {
    return ocl_major == v.ocl_major && ocl_minor == v.ocl_minor;
  }

  bool operator!=(const OpenCLVersion &v) const { return !(*this == v); }

  bool operator<(const OpenCLVersion &v) const {
    if (ocl_major == v.ocl_major)
      return ocl_minor < v.ocl_minor;

    return ocl_major < v.ocl_major;
  }

  bool operator>(const OpenCLVersion &v) const { return v < *this; }

  bool operator<=(const OpenCLVersion &v) const {
    return (*this < v) || (*this == v);
  }

  bool operator>=(const OpenCLVersion &v) const {
    return (*this > v) || (*this == v);
  }

  bool isValid() const {
    switch (ocl_major) {
    case 0:
      return false;
    case 1:
    case 2:
      return ocl_minor <= 2;
    case UINT_MAX:
      return false;
    default:
      return ocl_minor != UINT_MAX;
    }
  }

  int getMajor() const { return ocl_major; }
  int getMinor() const { return ocl_minor; }
};

inline const OpenCLVersion V1_0(1, 0);
inline const OpenCLVersion V1_1(1, 1);
inline const OpenCLVersion V1_2(1, 2);
inline const OpenCLVersion V2_0(2, 0);
inline const OpenCLVersion V2_1(2, 1);
inline const OpenCLVersion V2_2(2, 2);
inline const OpenCLVersion V3_0(3, 0);

} // namespace OCLV

// Define the types that are opaque in pi.h in a manner suitable for OpenCL
// plugin

struct _pi_device : _pi_object {
  _pi_device(pi_platform Plt) : Platform{Plt} {
    subLevel = -1;
    family = index = 0;
    // NOTE: one must additionally call initialize() to complete
    // PI device creation.
  }
  // PI platform to which this device belongs.
  pi_platform Platform;

  // Info stored for sub-sub device queue creation
  int subLevel;     // 0 - root device; 1 - sub-device; 2 - sub-sub-device
  pi_uint32 family; // SYCL queue family
  pi_uint32 index;  // SYCL queue index inside a given family of queues
  bool isRootDevice(void) { return subLevel == 0; }
  bool isSubDevice(void) { return subLevel == 1; }
  bool isSubSubDevice(void) { return subLevel == 2; }
};

#endif // PI_OPENCL_HPP
