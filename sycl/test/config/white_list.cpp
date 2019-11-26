// REQUIRES: cpu
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env PRINT_DEVICE_INFO=1 %t.out > %t.conf
// RUN: env TEST_DEVICE_AVAILABLE=1 env SYCL_CONFIG_FILE_NAME=%t.conf %t.out
// RUN: env TEST_DEVICE_IS_NOT_AVAILABLE=1 env SYCL_DEVICE_WHITE_LIST="" %t.out

#include <CL/sycl.hpp>
#include <iostream>
#include <cstdlib>
#include <exception>
#include <string>

using namespace cl;

int main() {

  // Expected that white list filter is not set
  if (getenv("PRINT_DEVICE_INFO")) {
    for (const sycl::platform &Plt : sycl::platform::get_platforms())
      if (!Plt.is_host()) {
        const sycl::device Dev = Plt.get_devices().at(0);
        std::string DevName = Dev.get_info<sycl::info::device::name>();
        const std::string DevVer =
            Dev.get_info<sycl::info::device::driver_version>();
        // As device name string will be used as regexp pattern, we need to
        // get rid of symbols that can be treated in a special way.
        // Replace common special symbols with '.' which matches to any sybmol
        for (char &Sym : DevName) {
          if (')' == Sym || '(' == Sym)
            Sym = '.';
        }
        std::cout << "SYCL_DEVICE_WHITE_LIST=DeviceName:{{" << DevName
                  << "}},DriverVersion:{{" << DevVer << "}}";
        return 0;
      }
    throw std::runtime_error("Non host device is not found");
  }

  // Expected white list to be set with result from "PRINT_DEVICE_INFO" run
  if (getenv("TEST_DEVICE_AVAILABLE")) {
    for (const sycl::platform &Plt : sycl::platform::get_platforms())
      if (!Plt.is_host()) {
        if (Plt.get_devices().size() != 1)
          throw std::runtime_error("Expected only one non host device.");

        return 0;
      }
    throw std::runtime_error("Non host device is not found");
  }

  // Expected white list to be set but empty
  if (getenv("TEST_DEVICE_IS_NOT_AVAILABLE")) {
    for (const sycl::platform &Plt : sycl::platform::get_platforms())
      if (!Plt.is_host())
        throw std::runtime_error("Expected no non host device is available");
    return 0;
  }

  throw std::runtime_error("Unhandled situation");
}
