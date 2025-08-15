// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Tests platform::get_devices for each device type.

#include <sycl/detail/core.hpp>
#include <sycl/platform.hpp>
#include <unordered_set>

std::string BackendToString(sycl::backend Backend) {
  switch (Backend) {
  case sycl::backend::host:
    return "host";
  case sycl::backend::opencl:
    return "opencl";
  case sycl::backend::ext_oneapi_level_zero:
    return "ext_oneapi_level_zero";
  case sycl::backend::ext_oneapi_cuda:
    return "ext_oneapi_cuda";
  case sycl::backend::all:
    return "all";
  case sycl::backend::ext_oneapi_hip:
    return "ext_oneapi_hip";
  case sycl::backend::ext_oneapi_native_cpu:
    return "ext_oneapi_native_cpu";
  case sycl::backend::ext_oneapi_offload:
    return "ext_oneapi_offload";
  default:
    return "UNKNOWN";
  }
}

std::string DeviceTypeToString(sycl::info::device_type DevType) {
  switch (DevType) {
  case sycl::info::device_type::all:
    return "device_type::all";
  case sycl::info::device_type::cpu:
    return "device_type::cpu";
  case sycl::info::device_type::gpu:
    return "device_type::gpu";
  case sycl::info::device_type::accelerator:
    return "device_type::accelerator";
  case sycl::info::device_type::custom:
    return "device_type::custom";
  case sycl::info::device_type::automatic:
    return "device_type::automatic";
  case sycl::info::device_type::host:
    return "device_type::host";
  default:
    return "UNKNOWN";
  }
}

template <typename T1, typename T2>
int Check(const T1 &LHS, const T2 &RHS, std::string TestName) {
  if (LHS != RHS) {
    std::cout << "Failed check " << LHS << " != " << RHS << ": " << TestName
              << std::endl;
    return 1;
  }
  return 0;
}

int CheckDeviceType(const sycl::platform &P, sycl::info::device_type DevType,
                    std::unordered_set<sycl::device> &AllDevices) {
  assert(DevType != sycl::info::device_type::all);
  int Failures = 0;

  std::vector<sycl::device> Devices = P.get_devices(DevType);

  if (DevType == sycl::info::device_type::automatic) {
    if (AllDevices.empty()) {
      Failures += Check(
          Devices.size(), 0,
          "No devices reported for all query, but automatic returns a device.");
    } else {
      Failures += Check(Devices.size(), 1,
                        "Number of devices for device_type::automatic query.");
      if (Devices.size())
        Failures +=
            Check(AllDevices.count(Devices[0]), 1,
                  "Device is in the set of all devices in the platform.");
    }
    return Failures;
  }

  // Count devices with the type;
  size_t DevCount = 0;
  for (sycl::device Device : Devices)
    DevCount += (Device.get_info<sycl::info::device::device_type>() == DevType);

  std::unordered_set<sycl::device> UniqueDevices{Devices.begin(),
                                                 Devices.end()};
  Check(Devices.size(), UniqueDevices.size(),
        "Duplicate devices for " + DeviceTypeToString(DevType));

  Failures +=
      Check(Devices.size(), DevCount,
            "Unexpected number of devices for " + DeviceTypeToString(DevType));

  Failures += Check(
      std::all_of(UniqueDevices.begin(), UniqueDevices.end(),
                  [&](const auto &Dev) { return AllDevices.count(Dev) == 1; }),
      true,
      "Not all devices for " + DeviceTypeToString(DevType) +
          " appear in the list of all devices");

  return Failures;
}

int main() {
  int Failures = 0;
  for (sycl::platform P : sycl::platform::get_platforms()) {
    std::cout << "Checking platform with backend "
              << BackendToString(P.get_backend()) << std::endl;

    std::vector<sycl::device> Devices = P.get_devices();
    std::unordered_set<sycl::device> UniqueDevices{Devices.begin(),
                                                   Devices.end()};

    if (Check(Devices.size(), UniqueDevices.size(),
              "Duplicate devices for device_type::all")) {
      ++Failures;
      // Don't trust this platform, so we continue.
      continue;
    }

    for (sycl::info::device_type DevType :
         {sycl::info::device_type::cpu, sycl::info::device_type::gpu,
          sycl::info::device_type::accelerator, sycl::info::device_type::custom,
          sycl::info::device_type::automatic, sycl::info::device_type::host})
      Failures += CheckDeviceType(P, DevType, UniqueDevices);
  }
  return Failures;
}
