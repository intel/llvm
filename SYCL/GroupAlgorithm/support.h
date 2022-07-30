#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi;

bool isSupportedDevice(device D) {
  std::string PlatformName = D.get_platform().get_info<info::platform::name>();
  if (PlatformName.find("CUDA") != std::string::npos)
    return true;

  if (PlatformName.find("Level-Zero") != std::string::npos)
    return true;

  if (PlatformName.find("OpenCL") != std::string::npos) {
    std::string Version = D.get_info<info::device::version>();
    size_t Offset = Version.find("OpenCL");
    if (Offset == std::string::npos)
      return false;
    Version = Version.substr(Offset + 7, 3);
    if (Version >= std::string("2.0"))
      return true;
  }

  return false;
}
