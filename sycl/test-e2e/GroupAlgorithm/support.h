#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi;

bool isSupportedDevice(device D) {
  std::string PlatformName =
      D.get_platform().get_info<sycl::info::platform::name>();
  if (PlatformName.find("CUDA") != std::string::npos)
    return true;

  if (PlatformName.find("Level-Zero") != std::string::npos)
    return true;

  if (PlatformName.find("OpenCL") != std::string::npos) {
    std::string Version = D.get_info<sycl::info::device::version>();

    // Group collectives are mandatory in OpenCL 2.0 but optional in 3.0.
    Version = Version.substr(7, 3);
    if (Version >= "2.0" && Version < "3.0")
      return true;
  }

  return false;
}
