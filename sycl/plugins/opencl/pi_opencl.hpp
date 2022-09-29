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

#include <climits>
#include <regex>
#include <string>

// This version should be incremented for any change made to this file or its
// corresponding .cpp file.
#define _PI_OPENCL_PLUGIN_VERSION 1

#define _PI_OPENCL_PLUGIN_VERSION_STRING                                       \
  _PI_PLUGIN_VERSION_STRING(_PI_OPENCL_PLUGIN_VERSION)

namespace OCLV {
class OpenCLVersion {
protected:
  unsigned int majorVer;
  unsigned int minorVer;

public:
  OpenCLVersion() : majorVer(0), minorVer(0) {}

  OpenCLVersion(unsigned int majorVer, unsigned int minorVer)
      : majorVer(majorVer), minorVer(minorVer) {
    if (!isValid())
      majorVer = minorVer = 0;
  }

  OpenCLVersion(const char *version) : OpenCLVersion(std::string(version)) {}

  OpenCLVersion(const std::string &version) : majorVer(0), minorVer(0) {
    /* The OpenCL specification defines the full version string as
     * 'OpenCL<space><major_version.minor_version><space><platform-specific
     * information>' for platforms and as
     * 'OpenCL<space><major_version.minor_version><space><vendor-specific
     * information>' for devices.
     */
    std::regex rx("OpenCL ([0-9]+)\\.([0-9]+)");
    std::smatch match;

    if (std::regex_search(version, match, rx) && (match.size() == 3)) {
      majorVer = strtoul(match[1].str().c_str(), nullptr, 10);
      minorVer = strtoul(match[2].str().c_str(), nullptr, 10);

      if (!isValid())
        majorVer = minorVer = 0;
    }
  }

  bool operator==(const OpenCLVersion &v) const {
    return majorVer == v.majorVer && minorVer == v.minorVer;
  }

  bool operator!=(const OpenCLVersion &v) const { return !(*this == v); }

  bool operator<(const OpenCLVersion &v) const {
    if (majorVer == v.majorVer)
      return minorVer < v.minorVer;

    return majorVer < v.majorVer;
  }

  bool operator>(const OpenCLVersion &v) const { return v < *this; }

  bool operator<=(const OpenCLVersion &v) const {
    return (*this < v) || (*this == v);
  }

  bool operator>=(const OpenCLVersion &v) const {
    return (*this > v) || (*this == v);
  }

  bool isValid() const {
    switch (majorVer) {
    case 0:
      return false;
    case 1:
    case 2:
      return minorVer <= 2;
    case UINT_MAX:
      return false;
    default:
      return minorVer != UINT_MAX;
    }
  }

  int getMajor() const { return majorVer; }
  int getMinor() const { return minorVer; }
};

inline const OpenCLVersion V1_0(1, 0);
inline const OpenCLVersion V1_1(1, 1);
inline const OpenCLVersion V1_2(1, 2);
inline const OpenCLVersion V2_0(2, 0);
inline const OpenCLVersion V2_1(2, 1);
inline const OpenCLVersion V2_2(2, 2);
inline const OpenCLVersion V3_0(3, 0);

} // namespace OCLV

#endif // PI_OPENCL_HPP
