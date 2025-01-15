// RUN: %{build} -o %t.out
//
// RUN: env PRINT_DEVICE_INFO=1 %{run-unfiltered-devices} %t.out > %t1.conf
// RUN: env TEST_DEVICE_AVAILABLE=1 env SYCL_CONFIG_FILE_NAME=%t1.conf %{run-unfiltered-devices} %t.out
//
// RUN: env PRINT_PLATFORM_INFO=1 %{run-unfiltered-devices} %t.out > %t2.conf
// RUN: env TEST_PLATFORM_AVAILABLE=1 env SYCL_CONFIG_FILE_NAME=%t2.conf %{run-unfiltered-devices} %t.out
//
// RUN: env TEST_DEVICE_IS_NOT_AVAILABLE=1 env SYCL_DEVICE_ALLOWLIST="PlatformName:{{SUCH NAME DOESN'T EXIST}}" %{run-unfiltered-devices} %t.out
// RUN: env TEST_INCORRECT_VALUE=1 env SYCL_DEVICE_ALLOWLIST="IncorrectKey:{{.*}}" %{run-unfiltered-devices} %t.out

#include "../helpers.hpp"
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <sycl/detail/core.hpp>

static void replaceSpecialCharacters(std::string &Str) {
  // Replace common special symbols with '.' which matches to any character
  std::replace_if(
      Str.begin(), Str.end(),
      [](const char Sym) {
        return '(' == Sym || ')' == Sym || '[' == Sym || ']' == Sym ||
               '+' == Sym;
      },
      '.');
}

int main() {

  // Expected that the allowlist filter is not set
  if (env::isDefined("PRINT_PLATFORM_INFO")) {
    for (const sycl::platform &Platform : sycl::platform::get_platforms()) {
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
    throw std::runtime_error("No platform is found");
  }

  // Expected that the allowlist filter is not set
  if (env::isDefined("PRINT_DEVICE_INFO")) {
    for (const sycl::platform &Platform : sycl::platform::get_platforms()) {
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
    throw std::runtime_error("No device is found");
  }

  // Expected the allowlist to be set with the "PRINT_DEVICE_INFO" run result
  if (env::isDefined("TEST_DEVICE_AVAILABLE")) {
    for (const sycl::platform &Platform : sycl::platform::get_platforms()) {
      if (Platform.get_devices().size() != 1)
        throw std::runtime_error("Expected only one device.");

      return 0;
    }
    throw std::runtime_error("No device is found");
  }

  // Expected the allowlist to be set but empty
  if (env::isDefined("TEST_DEVICE_IS_NOT_AVAILABLE")) {
    if (!sycl::platform::get_platforms().empty())
      throw std::runtime_error("Expected no device is available");
    return 0;
  }

  // Expected the allowlist to be set with the "PRINT_PLATFORM_INFO" run result
  if (env::isDefined("TEST_PLATFORM_AVAILABLE")) {
    auto Platforms = sycl::platform::get_platforms();
    if (Platforms.empty())
      throw std::runtime_error("No platform is found");
    else if (Platforms.size() != 1)
      throw std::runtime_error("Expected only one platform.");

    return 0;
  }

  // Expected the allowlist to be set but empty
  if (env::isDefined("TEST_DEVICE_IS_NOT_AVAILABLE")) {
    if (!sycl::platform::get_platforms().empty())
      throw std::runtime_error("Expected no device is available");
    return 0;
  }

  if (env::isDefined("TEST_INCORRECT_VALUE")) {
    try {
      sycl::platform::get_platforms();
    } catch (sycl::exception &E) {
      const std::string ExpectedMsg{
          "Unrecognized key in SYCL_DEVICE_ALLOWLIST"};
      const std::string GotMessage(E.what());
      if (GotMessage.find(ExpectedMsg) != std::string::npos) {
        return 0;
      }
      return 1;
    }
  }

  throw std::runtime_error("Unhandled situation");
}
