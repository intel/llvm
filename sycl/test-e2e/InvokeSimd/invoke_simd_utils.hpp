#pragma once

#include <string>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;
using namespace sycl;
namespace esimd = sycl::ext::intel::esimd;

enum GPUDriverOS { Linux = 1, Windows = 2, LinuxAndWindows = 3 };

/// This function returns true if it can detect the level-zero or opencl
/// GPU driver and can determine that the current driver is same or newer
/// than the one passed in \p RequiredVersion or \p WinOpenCLRequiredVersion.
///
/// Below are how driver versions look like:
///   Linux/L0:       [1.3.26370]
///   Linux/opencl:   [23.22.26370.18]
///   Windows/L0:     [1.3.26370]
///   Windows/opencl: [31.0.101.4502]
///
/// This function uses only the part of the driver identification:
///   - the second half of the driver id on win/opencl, e.g. 101.4502";
///   - the 5-digit id for 3 other platforms, e.g. 26370.
///
/// Note: For the previous & new driver version and their release dates
/// for win/opencl see the link:
/// https://www.intel.com/content/www/us/en/download/726609/intel-arc-iris-xe-graphics-whql-windows.html
bool isGPUDriverGE(queue Q, GPUDriverOS OSCheck, std::string RequiredVersion,
                   std::string WinOpenCLRequiredVersion = "") {
  auto Dev = Q.get_device();
  if (!Dev.is_gpu())
    return false;

  bool IsLinux = false;
#if defined(__SYCL_RT_OS_LINUX)
  IsLinux = true;
#elif !defined(__SYCL_RT_OS_WINDOWS)
  return false;
#endif

  // A and B must have digits at the same positions.
  // Otherwise, A and B symbols must be equal, e.g. both be equal to '.'.
  auto verifyDriverVersionFormat = [](const std::string &A,
                                      const std::string &B) {
    if (A.size() != B.size())
      throw std::runtime_error(
          "Inconsistent expected & actual driver versions");
    for (int I = 0; I < A.size(); I++) {
      if ((A[I] >= '0' && A[I] <= '9' && !(B[I] >= '0' && B[I] <= '9')) &&
          A[I] != B[I])
        throw std::runtime_error(
            "Inconsistent expected & actual driver versions");
    }
  };

  auto BE = Q.get_backend();
  int Length = 5;              // extract 5 digits for 3 or 4 platforms
  int Start = 4;               // start of the driver id for 2 of 4 platforms
  if (BE == backend::opencl) { // opencl has less-standard versioning
    if (IsLinux) {
      Start = 6;
    } else {
      Start = 5;
      Length = 8;
      RequiredVersion = WinOpenCLRequiredVersion;
    }
  }

  bool IsGE = true;
  if (IsLinux && (OSCheck & GPUDriverOS::Linux) ||
      !IsLinux && (OSCheck & GPUDriverOS::Windows)) {
    auto CurrentVersion = Dev.get_info<sycl::info::device::driver_version>();
    CurrentVersion = CurrentVersion.substr(Start, Length);
    verifyDriverVersionFormat(CurrentVersion, RequiredVersion);
    std::cout << "RequiredVersion = " << RequiredVersion << ", Start=" << Start
              << ", Length=" << Length << std::endl;
    std::cout << "CurrentVersion = " << CurrentVersion << std::endl;
    IsGE &= CurrentVersion >= RequiredVersion;
  }
  return IsGE;
}
