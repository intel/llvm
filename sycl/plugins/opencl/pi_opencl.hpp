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
#include <pi2ur.hpp>
// This version should be incremented for any change made to this file or its
// corresponding .cpp file.
#define _PI_OPENCL_PLUGIN_VERSION 1

#define _PI_OPENCL_PLUGIN_VERSION_STRING                                       \
  _PI_PLUGIN_VERSION_STRING(_PI_OPENCL_PLUGIN_VERSION)

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
  enum device_level {
    ROOTDEVICE = 0,
    SUBDEVICE = 1,
    SUBSUBDEVICE = 2,
    INVALID = -1
  };
  _pi_device(pi_platform Plt) : Platform{Plt} {
    level = INVALID;
    family = index = 0;
  }
  // PI platform to which this device belongs.
  pi_platform Platform;

  // Info stored for sub-sub device queue creation
  device_level level;
  pi_uint32 family; // SYCL queue family
  pi_uint32 index;  // SYCL queue index inside a given family of queues

  bool isRootDevice(void) { return level == ROOTDEVICE; }
  bool isSubDevice(void) { return level == SUBDEVICE; }
  bool isSubSubDevice(void) { return level == SUBSUBDEVICE; }
};

#endif // PI_OPENCL_HPP
