// REQUIRES: cpu
// RUN: %clangxx -fsycl %s -o %t.out
//
// RUN: env PRINT_DEVICE_INFO=1 %t.out > %t1.conf
// RUN: env TEST_DEVICE_AVAILABLE=1 env SYCL_CONFIG_FILE_NAME=%t1.conf %t.out
//
// RUN: env PRINT_PLATFORM_INFO=1 %t.out > %t2.conf
// RUN: env TEST_DEVICE_AVAILABLE=1 env SYCL_CONFIG_FILE_NAME=%t2.conf %t.out
//
// RUN: env TEST_DEVICE_IS_NOT_AVAILABLE=1 env SYCL_DEVICE_ALLOWLIST="PlatformName:{{SUCH NAME DOESN'T EXIST}}" %t.out

#include <CL/sycl.hpp>
#include <iostream>
#include <cstdlib>
#include <exception>
#include <string>

using namespace cl;

static void replaceSpecialCharacters(std::string &Str) {
  // Replace common special symbols with '.' which matches to any character
  std::replace_if(Str.begin(), Str.end(),
                  [](const char Sym) { return '(' == Sym || ')' == Sym; }, '.');
}

int main() {

  // Expected that the allowlist filter is not set
  if (getenv("PRINT_PLATFORM_INFO")) {
    for (const sycl::platform &Platform : sycl::platform::get_platforms())
      if (!Platform.is_host()) {

        std::string Name = Platform.get_info<sycl::info::platform::name>();
        std::string Ver = Platform.get_info<sycl::info::platform::version>();
        // As a string will be used as regexp pattern, we need to get rid of
        // symbols that can be treated in a special way.
        replaceSpecialCharacters(Name);
        replaceSpecialCharacters(Ver);

        std::cout << "SYCL_DEVICE_ALLOWLIST=PlatformName:{{" << Name
                  << "}},PlatformVersion:{{" << Ver << "}}";

        return 0;
      }
    throw std::runtime_error("Non host device is not found");
  }

  // Expected that the allowlist filter is not set
  if (getenv("PRINT_DEVICE_INFO")) {
    for (const sycl::platform &Platform : sycl::platform::get_platforms())
      if (!Platform.is_host()) {
        const sycl::device Dev = Platform.get_devices().at(0);
        std::string Name = Dev.get_info<sycl::info::device::name>();
        std::string Ver = Dev.get_info<sycl::info::device::driver_version>();

        // As a string will be used as regexp pattern, we need to get rid of
        // symbols that can be treated in a special way.
        replaceSpecialCharacters(Name);
        replaceSpecialCharacters(Ver);

        std::cout << "SYCL_DEVICE_ALLOWLIST=DeviceName:{{" << Name
                  << "}},DriverVersion:{{" << Ver << "}}";

        return 0;
      }
    throw std::runtime_error("Non host device is not found");
  }

  // Expected the allowlist to be set with the "PRINT_DEVICE_INFO" run result
  if (getenv("TEST_DEVICE_AVAILABLE")) {
    for (const sycl::platform &Platform : sycl::platform::get_platforms())
      if (!Platform.is_host()) {
        if (Platform.get_devices().size() != 1)
          throw std::runtime_error("Expected only one non host device.");

        return 0;
      }
    throw std::runtime_error("Non host device is not found");
  }

  // Expected the allowlist to be set but empty
  if (getenv("TEST_DEVICE_IS_NOT_AVAILABLE")) {
    for (const sycl::platform &Platform : sycl::platform::get_platforms())
      if (!Platform.is_host())
        throw std::runtime_error("Expected no non host device is available");
    return 0;
  }

  throw std::runtime_error("Unhandled situation");
}
