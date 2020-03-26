#include <CL/sycl.hpp>

#include <iostream>
bool isInlineASMSupported(sycl::device Device) {

  sycl::string_class DriverVersion = Device.get_info<sycl::info::device::driver_version>();
  sycl::string_class DeviceVendorName = Device.get_info<sycl::info::device::vendor>();
  if (DeviceVendorName.find("Intel") == sycl::string_class::npos)
    return false;
  if (DriverVersion.length() < 5)
    return false;
  if (DriverVersion[2] != '.')
    return false;
  if (std::stoi(DriverVersion.substr(0, 2), nullptr, 10) < 20 || std::stoi(DriverVersion.substr(3, 2), nullptr, 10) < 12)
    return false;
  return true;
}
