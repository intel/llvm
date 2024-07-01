// RUN: %{build} -o %t.out
// RUN: env PRINT_FULL_DEVICE_INFO=1 %{run-unfiltered-devices} %t.out > %t1.conf
// RUN: env ONEAPI_DEVICE_SELECTOR="*:0" env TEST_DEV_CONFIG_FILE_NAME=%t1.conf %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:1" env TEST_DEV_CONFIG_FILE_NAME=%t1.conf %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:2" env TEST_DEV_CONFIG_FILE_NAME=%t1.conf %{run-unfiltered-devices} %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="*:3" env TEST_DEV_CONFIG_FILE_NAME=%t1.conf %{run-unfiltered-devices} %t.out

// Temporarily disable on L0 due to fails in CI
// UNSUPPORTED: level_zero

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <sycl/detail/core.hpp>

using namespace sycl;
using namespace std;

const std::map<info::device_type, std::string> DeviceTypeStringMap = {
    {info::device_type::cpu, "cpu"},
    {info::device_type::gpu, "gpu"},
    {info::device_type::accelerator, "acc"}};

const std::map<backend, std::string> BackendStringMap = {
    {backend::opencl, "opencl"},
    {backend::ext_oneapi_level_zero, "ext_oneapi_level_zero"},
    {backend::ext_oneapi_cuda, "ext_oneapi_cuda"},
    {backend::ext_oneapi_hip, "ext_oneapi_hip"},
    {backend::ext_oneapi_native_cpu, "ext_oneapi_native_cpu"}};

std::string getDeviceTypeName(const device &d) {
  auto DeviceType = d.get_info<info::device::device_type>();
  std::string DeviceTypeName = "unknown";
  auto it = DeviceTypeStringMap.find(DeviceType);
  if (it != DeviceTypeStringMap.end())
    DeviceTypeName = it->second;
  return DeviceTypeName;
}

void printDeviceType(const device &d) {
  string name = d.get_platform().get_info<info::platform::name>();
  std::cout << getDeviceTypeName(d) << ", " << name << std::endl;
}

// Applicable for backend and device info.
template <typename DevInfoEntry>
DevInfoEntry
getDeviceInfoByName(const std::string &name,
                    const std::map<DevInfoEntry, std::string> &mapWithNames) {
  auto it = std::find_if(
      mapWithNames.begin(), mapWithNames.end(),
      [&name](auto &entry) { return entry.second.compare(name) == 0; });
  // Invalid or unknown configuration if not found.
  assert(it != mapWithNames.end());
  return it->first;
}

void PrintSystemConfiguration() {
  const auto &Platforms = platform::get_platforms();

  // Keep track of the number of devices per backend.
  std::map<backend, size_t> DeviceNums;

  for (const auto &Platform : Platforms) {
    backend Backend = Platform.get_backend();
    auto PlatformName = Platform.get_info<info::platform::name>();
    const auto &Devices = Platform.get_devices();
    for (const auto &Device : Devices) {
      std::cout << Backend << ":" << getDeviceTypeName(Device) << ":"
                << DeviceNums[Backend] << std::endl;
      ++DeviceNums[Backend];
    }
  }
}

using DevInfo = std::pair<info::device_type, backend>;
using DevInfoMap = std::map<int, std::vector<DevInfo>>;
bool ReadInitialSystemConfiguration(char *fileName, DevInfoMap &devices) {
  fstream confFile;
  confFile.open(fileName, ios::in);
  if (!confFile.is_open())
    return false;
  char linebuf[64];
  while (confFile.getline(linebuf, 64)) {
    std::istringstream entry(linebuf);
    std::string type, backend, devNum;
    if (!std::getline(entry, backend, ':'))
      return false;
    if (!std::getline(entry, type, ':'))
      return false;
    if (!std::getline(entry, devNum))
      return false;
    devices[std::atoi(devNum.data())].push_back(
        make_pair(getDeviceInfoByName(type, DeviceTypeStringMap),
                  getDeviceInfoByName(backend, BackendStringMap)));
  }
  if (devices.size() == 0)
    return false;
  return true;
}

int GetPreferredDeviceIndex(const std::vector<device> &devices,
                            info::device_type type) {
  // Note: scores can be not the same as in runtime, just keep the same
  // preference.
  //   gpu L0, opencl
  //   cpu
  //   acc
  const std::map<info::device_type, int> scoreByType = {
      {info::device_type::cpu, 300},
      {info::device_type::gpu, 500},
      {info::device_type::accelerator, 75}};
  int score = -1;
  int index = -1;
  int devCount = devices.size();
  for (int i = 0; i < devCount; i++) {
    int dev_score = 0;
    auto deviceType = devices[i].get_info<info::device::device_type>();
    auto backend = devices[i].get_backend();
    if ((type != info::device_type::all) && (deviceType != type))
      continue;
    dev_score = scoreByType.at(deviceType);
    if (backend == backend::ext_oneapi_level_zero)
      dev_score += 100;
    if (dev_score > score) {
      score = dev_score;
      index = i;
    }
  }
  return index;
}

int main() {
  // Expected that the sycl device filter is not set
  if (getenv("PRINT_FULL_DEVICE_INFO")) {
    PrintSystemConfiguration();
    return 0;
  }

  DevInfoMap unfilteredDevices;
  assert(ReadInitialSystemConfiguration(getenv("TEST_DEV_CONFIG_FILE_NAME"),
                                        unfilteredDevices) &&
         "Failed to parse file with initial system configuration data");

  const char *envVal = std::getenv("ONEAPI_DEVICE_SELECTOR");
  int deviceNum;
  std::cout << "ONEAPI_DEVICE_SELECTOR=" << envVal << std::endl;
  deviceNum = std::atoi(std::string(envVal).substr(2).c_str());

  auto devices = device::get_devices();
  std::cout << "Device count to analyze =" << devices.size() << std::endl;

  auto expectedDevices = unfilteredDevices[deviceNum];
  size_t devCount = expectedDevices.size();
  assert(devices.size() == devCount &&
         "Devices seems to be filtered in a wrong way. Count of devices is "
         "unexpected.");

  if (devices.size() == 0) {
    std::cout << "No devices with such filter, skipping test." << std::endl;
    return 0;
  }

  for (int i = 0; i < devCount; i++) {
    auto deviceType = devices[i].get_info<info::device::device_type>();
    assert(deviceType == std::get<0>(expectedDevices[i]) &&
           "Device type or device order is not expected.");
    assert(devices[i].get_backend() == std::get<1>(expectedDevices[i]) &&
           "Device backend or device order is not expected.");
  }

  int targetDevIndex = -1;
  {
    targetDevIndex = GetPreferredDeviceIndex(devices, info::device_type::all);
    assert(targetDevIndex >= 0 &&
           "Failed to find target device for default selector.");
    device d(default_selector_v);
    std::cout << "default_selector_v selected ";
    printDeviceType(d);
    assert(devices[targetDevIndex] == d &&
           "The selected device is not the target device specified.");
  }
  targetDevIndex = GetPreferredDeviceIndex(devices, info::device_type::gpu);
  if (targetDevIndex >= 0) {
    device d(gpu_selector_v);
    std::cout << "gpu_selector_v selected ";
    printDeviceType(d);
    assert(devices[targetDevIndex] == d &&
           "The selected device is not the target device specified.");
  }
  targetDevIndex = GetPreferredDeviceIndex(devices, info::device_type::cpu);
  if (targetDevIndex >= 0) {
    device d(cpu_selector_v);
    std::cout << "cpu_selector_v selected ";
    printDeviceType(d);
    assert(devices[targetDevIndex] == d &&
           "The selected device is not the target device specified.");
  }
  targetDevIndex =
      GetPreferredDeviceIndex(devices, info::device_type::accelerator);
  if (targetDevIndex >= 0) {
    device d(accelerator_selector_v);
    std::cout << "accelerator_selector_v selected ";
    printDeviceType(d);
    assert(devices[targetDevIndex] == d &&
           "The selected device is not the target device specified.");
  }

  return 0;
}
