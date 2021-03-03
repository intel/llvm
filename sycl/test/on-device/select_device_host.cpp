// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=level_zero:cpu %t.out

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;
using namespace std;

int main() {
    const char *envVal = getenv("SYCL_DEVICE_FILTER");
    string forcedPIs;
    if (envVal) {
      cout << "SYCL_DEVICE_FILTER=" << envVal << std::endl;
      forcedPIs = envVal;
    }

    {
      default_selector ds;
      device d = ds.select_device();
      string name = d.get_platform().get_info<info::platform::name>();
      assert(name.find("host") != string::npos);
      std::cout << "Host Device is selected: " << boolalpha << d.is_host() << std::endl;
    }
    {
      gpu_selector gs;
      try {
        device d = gs.select_device();
        cerr << "GPU device is found in error: " << d.is_gpu() << std::endl;
        return -1;
      } catch (...) {
        cout << "Expectedly, gpu device is not found." << std::endl;
      }
    }
    {
      cpu_selector cs;
      try {
        device d = cs.select_device();
        cerr << "CPU device is found in error: " << d.is_cpu() << std::endl;
        return -1;
      } catch (...) {
        cout << "Expectedly, cpu device is not found." << std::endl;
      }
    }
    {
      host_selector hs;
      device d = hs.select_device();
      cout << "HOST device is found: " << d.is_host() << std::endl;
    }
    {
      accelerator_selector as;
      try {
        device d = as.select_device();
        cerr << "ACC device is found in error: " << d.is_accelerator()
          << std::endl;
      } catch (...) {
        cout << "Expectedly, ACC device is not found." << std::endl;
      }
    }

  return 0;
}
