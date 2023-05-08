// REQUIRES: level_zero,gpu
// RUN: %{build} -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zero:gpu %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR='level_zero:0' %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="level_zero:*" %t.out

//  At this time, LevelZero only has one device (GPU). So this is easy to accept
//  L0 and GPU and reject anything else. This test is just testing top level
//  devices, not sub-devices.

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace std;

int main() {

  {
    device d(default_selector_v);
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("Level-Zero") != string::npos);
    cout << "Level-Zero GPU Device is found: " << boolalpha << d.is_gpu()
         << std::endl;
  }
  {
    device d(gpu_selector_v);
    string name = d.get_platform().get_info<info::platform::name>();
    assert(name.find("Level-Zero") != string::npos);
    cout << name << " is found: " << boolalpha << d.is_gpu() << std::endl;
  }
  {
    try {
      device d(cpu_selector_v);
      cerr << "CPU device is found in error: " << d.is_cpu() << std::endl;
      return -1;
    } catch (...) {
      cout << "Expectedly, cpu device is not found." << std::endl;
    }
  }
  {
    try {
      device d(accelerator_selector_v);
      cerr << "ACC device is found in error: " << d.is_accelerator()
           << std::endl;
    } catch (...) {
      cout << "Expectedly, ACC device is not found." << std::endl;
    }
  }

  return 0;
}