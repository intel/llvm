// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=0 %t.out
// RUN: env SYCL_DEVICE_FILTER=1 %t.out
// RUN: env SYCL_DEVICE_FILTER=2 %t.out
// RUN: env SYCL_DEVICE_FILTER=3 %t.out
// RUN: env SYCL_DEVICE_FILTER=4 %t.out

// The test is using all available BEs but CUDA machine in CI does not have
// functional OpenCL RT
// UNSUPPORTED: cuda

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;
using namespace std;

void printDeviceType(const device &d) {
  string name = d.get_platform().get_info<info::platform::name>();
  auto DeviceType = d.get_info<info::device::device_type>();
  std::string DeviceTypeName;

  switch (DeviceType) {
  case info::device_type::cpu:
    DeviceTypeName = "CPU ";
    break;
  case info::device_type::gpu:
    DeviceTypeName = "GPU ";
    break;
  case info::device_type::host:
    DeviceTypeName = "HOST ";
    break;
  case info::device_type::accelerator:
    DeviceTypeName = "ACCELERATOR ";
    break;
  default:
    DeviceTypeName = "UNKNOWN ";
    break;
  }
  std::cout << DeviceTypeName << name << std::endl;
}

int main() {
  const char *envVal = std::getenv("SYCL_DEVICE_FILTER");
  int deviceNum;
  std::cout << "SYCL_DEVICE_FILTER=" << envVal << std::endl;
  deviceNum = std::atoi(envVal);

  auto devices = device::get_devices();
  if (devices.size() > deviceNum) {
    device targetDevice = devices[deviceNum];
    std::cout << "Target Device: ";
    printDeviceType(targetDevice);

    {
      default_selector ds;
      device d = ds.select_device();
      std::cout << "default_selector selected ";
      printDeviceType(d);
      assert(targetDevice == d &&
             "The selected device is not the target device specified.");
    }

    if (targetDevice.is_gpu()) {
      gpu_selector gs;
      device d = gs.select_device();
      std::cout << "gpu_selector selected ";
      printDeviceType(d);
      assert(targetDevice == d &&
             "The selected device is not the target device specified.");
    } else if (targetDevice.is_cpu()) {
      cpu_selector cs;
      device d = cs.select_device();
      std::cout << "cpu_selector selected ";
      printDeviceType(d);
      assert(targetDevice == d &&
             "The selected device is not the target device specified.");
    } else if (targetDevice.is_accelerator()) {
      accelerator_selector as;
      device d = as.select_device();
      std::cout << "accelerator_selector selected ";
      printDeviceType(d);
      assert(targetDevice == d &&
             "The selected device is not the target device specified.");
    }
    // HOST device is always available regardless of SYCL_DEVICE_FILTER
    {
      host_selector hs;
      device d = hs.select_device();
      std::cout << "host_selector selected ";
      printDeviceType(d);
      assert(d.is_host() && "The selected device is not a host device.");
    }
  }
  return 0;
}
