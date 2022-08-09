// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env PRINT_FULL_DEVICE_INFO=1  %t.out > %t1.conf
// RUN: env SYCL_DEVICE_FILTER=0 env TEST_DEV_CONFIG_FILE_NAME=%t1.conf %t.out
// RUN: env SYCL_DEVICE_FILTER=1 env TEST_DEV_CONFIG_FILE_NAME=%t1.conf %t.out
// RUN: env SYCL_DEVICE_FILTER=2 env TEST_DEV_CONFIG_FILE_NAME=%t1.conf %t.out
// RUN: env SYCL_DEVICE_FILTER=3 env TEST_DEV_CONFIG_FILE_NAME=%t1.conf %t.out

// Temporarily disable on L0 and HIP due to fails in CI
// UNSUPPORTED: level_zero, hip

#include <fstream>
#include <iostream>
#include <map>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace std;

const std::map<info::device_type, std::string> DeviceTypeStringMap = {
    {info::device_type::cpu, "cpu"},
    {info::device_type::gpu, "gpu"},
    {info::device_type::host, "host"},
    {info::device_type::accelerator, "acc"}};

const std::map<backend, std::string> BackendStringMap = {
    {backend::opencl, "opencl"},
    {backend::host, "host"},
    {backend::ext_oneapi_level_zero, "ext_oneapi_level_zero"},
    {backend::ext_intel_esimd_emulator, "ext_intel_esimd_emulator"},
    {backend::ext_oneapi_cuda, "ext_oneapi_cuda"},
    {backend::ext_oneapi_hip, "ext_oneapi_hip"}};

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
  //   host
  const std::map<info::device_type, int> scoreByType = {
      {info::device_type::cpu, 300},
      {info::device_type::gpu, 500},
      {info::device_type::accelerator, 75},
      {info::device_type::host, 100}};
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

  const char *envVal = std::getenv("SYCL_DEVICE_FILTER");
  int deviceNum;
  std::cout << "SYCL_DEVICE_FILTER=" << envVal << std::endl;
  deviceNum = std::atoi(envVal);

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
    default_selector ds;
    device d = ds.select_device();
    std::cout << "default_selector selected ";
    printDeviceType(d);
    assert(devices[targetDevIndex] == d &&
           "The selected device is not the target device specified.");
  }
  targetDevIndex = GetPreferredDeviceIndex(devices, info::device_type::gpu);
  if (targetDevIndex >= 0) {
    gpu_selector gs;
    device d = gs.select_device();
    std::cout << "gpu_selector selected ";
    printDeviceType(d);
    assert(devices[targetDevIndex] == d &&
           "The selected device is not the target device specified.");
  }
  targetDevIndex = GetPreferredDeviceIndex(devices, info::device_type::cpu);
  if (targetDevIndex >= 0) {
    cpu_selector cs;
    device d = cs.select_device();
    std::cout << "cpu_selector selected ";
    printDeviceType(d);
    assert(devices[targetDevIndex] == d &&
           "The selected device is not the target device specified.");
  }
  targetDevIndex =
      GetPreferredDeviceIndex(devices, info::device_type::accelerator);
  if (targetDevIndex >= 0) {
    accelerator_selector as;
    device d = as.select_device();
    std::cout << "accelerator_selector selected ";
    printDeviceType(d);
    assert(devices[targetDevIndex] == d &&
           "The selected device is not the target device specified.");
  }
  targetDevIndex = GetPreferredDeviceIndex(devices, info::device_type::host);
  assert((targetDevIndex >= 0 || deviceNum != 0) &&
         "Failed to find host device.");
  if (targetDevIndex >= 0) {
    host_selector hs;
    device d = hs.select_device();
    std::cout << "host_selector selected ";
    printDeviceType(d);
    assert(devices[targetDevIndex] == d &&
           "The selected device is not a host device.");
  }

  return 0;
}
