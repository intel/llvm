// REQUIRES: gpu
// RUN: %{build} -o %t.out
//
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out DEVICE_INFO write > %t.txt
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out DEVICE_INFO read %t.txt
//
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out PLATFORM_INFO write > %t.txt
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out PLATFORM_INFO read %t.txt
//
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out DEVICE_ERROR_INFO write > %t.txt
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out DEVICE_ERROR_INFO read %t.txt
//
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out PLATFORM_ERROR_INFO write > %t.txt
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out PLATFORM_ERROR_INFO read %t.txt
//
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out REG_EX_INFO write > %t.txt
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out REG_EX_INFO read %t.txt
//
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out DEVICE_NAME_INFO write > %t.txt
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out DEVICE_NAME_INFO read %t.txt
//
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out PLATFORM_NAME_INFO write > %t.txt
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out PLATFORM_NAME_INFO read %t.txt
//
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out DEVICE_MULTI_INFO write > %t.txt
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out DEVICE_MULTI_INFO read %t.txt
//
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out DEVICE_MALFORMED_INFO write > %t.txt
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out DEVICE_MALFORMED_INFO read %t.txt
//
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out PLATOFRM_MALFORMED_INFO write > %t.txt
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out PLATOFRM_MALFORMED_INFO read %t.txt
//
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out DRIVER_MALFORMED_INFO write > %t.txt
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out DRIVER_MALFORMED_INFO read %t.txt
//
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out PLATVER_MALFORMED_INFO write > %t.txt
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu" %{run-unfiltered-devices} %t.out PLATVER_MALFORMED_INFO read %t.txt
//

//==------------ select_device.cpp - SYCL_DEVICE_ALLOWLIST test ------------==//
//
// This test is unusual because it occurs in two phases.  The first phase
// will find the GPU platforms, and write them to a file.  The second phase
// will read the file, set SYCL_DEVICE_ALLOWLIST, and then find the correct
// platform.  SYCL_DEVICE_ALLOWLIST is only evaluated once, the first time
// get_platforms() is called.  Setting it later in the application has no
// effect.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <sycl/detail/core.hpp>

using namespace sycl;

#ifdef _WIN32
#define setenv(name, value, overwrite) _putenv_s(name, value)
#endif

struct DevDescT {
  std::string devName;
  std::string devDriverVer;
  std::string platName;
  std::string platVer;
};

static void addEscapeSymbolToSpecialCharacters(std::string &str) {
  std::vector<std::string> specialCharacters{"(", ")", "[", "]", ".", "+", "-"};
  for (const auto &character : specialCharacters) {
    size_t pos = 0;
    while ((pos = str.find(character, pos)) != std::string::npos) {
      std::string modifiedCharacter("\\" + character);
      str.replace(pos, character.size(), modifiedCharacter);
      pos += modifiedCharacter.size();
    }
  }
}

static std::vector<DevDescT> getAllowListDesc(std::string_view allowList) {
  if (allowList.empty())
    return {};

  std::vector<DevDescT> decDescs;
  decDescs.emplace_back();

  auto try_parse = [&](std::string_view str) -> std::optional<std::string> {
    // std::string_view::starts_with is C++20.
    if (allowList.compare(0, str.size(), str) != 0)
      return {};

    allowList.remove_prefix(str.size());

    using namespace std::string_literals;
    auto pattern_start = allowList.find("{{");
    if (pattern_start == std::string::npos)
      throw std::runtime_error("Malformed "s + std::string{str} + " allowlist"s);

    allowList.remove_prefix(pattern_start + 2);
    auto pattern_end = allowList.find("}}");
    if (pattern_end == std::string::npos)
      throw std::runtime_error("Malformed "s + std::string{str} + " allowlist"s);

    auto result =  allowList.substr(0, pattern_end);
    allowList.remove_prefix(pattern_end + 2);

    if (allowList[0] == ',')
      allowList.remove_prefix(1);
    return {std::string{result}};
  };

  while (!allowList.empty()) {
    if (auto pattern = try_parse("DeviceName:")) {
      decDescs.back().devName = *pattern;
      continue;
    }
    if (auto pattern = try_parse("DriverVersion:")) {
      decDescs.back().devDriverVer = *pattern;
      continue;
    }
    if (auto pattern = try_parse("PlatformName:")) {
      decDescs.back().platName = *pattern;
      continue;
    }
    if (auto pattern = try_parse("PlatformVersion:")) {
      decDescs.back().platVer = *pattern;
      continue;
    }

    auto next = allowList.find('|');
    if (next == std::string::npos)
      throw std::runtime_error("Malformed allowlist");
    allowList.remove_prefix(next + 1);

    auto non_space = allowList.find_first_not_of(" ");
    allowList.remove_prefix(non_space);
    decDescs.emplace_back();
  }

  return decDescs;
}

bool is_known_be(backend be) {
    backend known_bes[] = {backend::opencl, backend::ext_oneapi_level_zero,
                           backend::ext_oneapi_cuda, backend::ext_oneapi_hip};
    if (std::find(std::begin(known_bes), std::end(known_bes), be) ==
        std::end(known_bes))
      return false;
    return true;
}

template <typename... AspectsTy>
std::optional<platform> get_test_platform(AspectsTy... Aspects) {
  for (const auto &plt : platform::get_platforms()) {
    if (!is_known_be(plt.get_backend()))
      continue;
    if ((plt.has(Aspects) && ... && true))
      return plt;
  }

  return {};
}

template <typename... AspectsTy>
std::optional<device> get_test_device(AspectsTy... Aspects) {
  for (const auto &plt : platform::get_platforms()) {
    backend known_bes[] = {backend::opencl, backend::ext_oneapi_level_zero,
                           backend::ext_oneapi_cuda, backend::ext_oneapi_hip};
    if (std::find(std::begin(known_bes), std::end(known_bes),
                  plt.get_backend()) == std::end(known_bes))
      continue;
    for (auto &dev : plt.get_devices()) {
      if ((dev.has(Aspects) && ... && true))
        return dev;
    }
  }

  return {};
}

int main(int argc, char *argv[]) {
  assert(argc >= 3);
  std::ignore = argv;
  using namespace std::string_view_literals;
  bool write = argv[2] == "write"sv;
  assert(write || argc == 4);
  std::string test = argv[1];
  bool passed = false;
  auto Return = [&]() {
    if (passed) {
      std::cout << "Passed." << std::endl;
      return 0;
    } else {
      std::cout << "Failed." << std::endl;
      return 1;
    }
  };

  std::string allowlist;
  std::vector<DevDescT> components;
  if (!write) {
    std::ifstream fs(argv[3]);
    std::getline(fs, allowlist);

    setenv("SYCL_DEVICE_ALLOWLIST", allowlist.c_str(), 0);
    components = getAllowListDesc(allowlist);
    std::cout << "SYCL_DEVICE_ALLOWLIST=" << allowlist << std::endl;
    if (allowlist.empty())
      return 2;
  }

  // Test the GPU devices name and version number.
  if (test == "DEVICE_INFO"sv) {
    if (write) {
      std::optional<device> dev = get_test_device(aspect::gpu);
      if (!dev)
        return 3;
      std::string name = dev->get_info<info::device::name>();
      addEscapeSymbolToSpecialCharacters(name);
      std::string ver = dev->get_info<info::device::driver_version>();
      std::cout << "DeviceName:{{" << name << "}},DriverVersion:{{" << ver
                << "}}" << std::endl;
      return 0;
    } else {
      sycl::queue deviceQueue(gpu_selector_v);
      device dev = deviceQueue.get_device();
      std::cout << "Device: " << dev.get_info<info::device::name>()
                << std::endl;
      std::cout << "DriverVersion: "
                << dev.get_info<info::device::driver_version>() << std::endl;
      for (const DevDescT &desc : components) {
        if ((std::regex_match(dev.get_info<info::device::name>(),
                              std::regex(desc.devName))) &&
            (std::regex_match(dev.get_info<info::device::driver_version>(),
                              std::regex(desc.devDriverVer)))) {
          return 0;
        }
      }
      return 1;
    }
  }

  // Test the platform name and version number.
  if (test == "PLATFORM_INFO"sv) {
    if (write) {
      std::optional<platform> plt = get_test_platform(aspect::gpu) ;
      if (!plt)
        return 3;

      std::string name = plt->get_info<info::platform::name>();
      addEscapeSymbolToSpecialCharacters(name);
      std::string ver = plt->get_info<info::platform::version>();
      std::cout << "PlatformName:{{" << name << "}},PlatformVersion:{{" << ver
                << "}}" << std::endl;
      return 0;
    } else {
      sycl::queue deviceQueue(gpu_selector_v);
      device dev = deviceQueue.get_device();
      const auto &plt = dev.get_platform();
      std::cout << "Platform: " << plt.get_info<info::platform::name>()
                << std::endl;
      std::cout << "Platform Version: "
                << plt.get_info<info::platform::version>() << std::endl;
      for (const DevDescT &desc : components) {
        std::cout << "Desc pltName: " << desc.platName << std::endl;
        std::cout << "Desc pltVer: " << desc.platVer << std::endl;
        if ((std::regex_match(plt.get_info<info::platform::name>(),
                              std::regex(desc.platName))) &&
            (std::regex_match(plt.get_info<info::platform::version>(),
                              std::regex(desc.platVer)))) {
          return 0;
        }
      }
      return 1;
    }
  }

  // Test error handling.
  if (test == "DEVICE_ERROR_INFO"sv) {
    if (write) {
      std::optional<device> dev = get_test_device(aspect::gpu);
      if (!dev)
        return 3;

      std::string name = dev->get_info<info::device::name>();
      addEscapeSymbolToSpecialCharacters(name);
      std::string ver("98.76.54321");
      std::cout << "DeviceName:{{" << name << "}},DriverVersion:{{" << ver
                << "}}" << std::endl;
      return 0;
    } else {
      try {
        sycl::queue deviceQueue(gpu_selector_v);
        device dev = deviceQueue.get_device();
        const auto &plt = dev.get_platform();
      } catch (sycl::exception &E) {
        const std::string expectedMsg(
            "No device of requested type 'info::device_type::gpu' available");
        const std::string gotMessage(E.what());
        if (gotMessage.find(expectedMsg) != std::string::npos)
          return 0;
      }
      return 1;
    }
  }

  // Test error condition when version number is not found.
  if (test == "PLATFORM_ERROR_INFO"sv) {
    if (write) {
      std::optional<platform> plt = get_test_platform(aspect::gpu);
      if (!plt)
        return 3;
      std::string name = plt->get_info<info::platform::name>();
      addEscapeSymbolToSpecialCharacters(name);
      std::string ver = [&]() {
        switch (plt->get_backend()) {
        case backend::opencl:
          return "OpenCL 12.34";
        case backend::ext_oneapi_level_zero:
          return "12.34";
        case backend::ext_oneapi_cuda:
          return "CUDA 89.78";
        case backend::ext_oneapi_hip:
          return "67.88.9";
        default:
          assert(false);
        }
      }();
      std::cout << "PlatformName:{{" << name << "}},PlatformVersion:{{" << ver << "}}"
         << std::endl;
      return 0;
    } else {
      try {
        sycl::queue deviceQueue(gpu_selector_v);
        device dev = deviceQueue.get_device();
        const auto &plt = dev.get_platform();
      } catch (sycl::exception &E) {
        const std::string expectedMsg(
            "No device of requested type 'info::device_type::gpu' available");
        const std::string gotMessage(E.what());
        if (gotMessage.find(expectedMsg) != std::string::npos)
          return 0;
      }
      return 1;
    }
  }

  // Test handling a regular expression in the device driver version number.
  if (test == "REG_EX_INFO"sv) {
    if (write) {
      bool passed = false;
      for (const auto &plt : platform::get_platforms()) {
        if (passed) {
          break;
        } // no need for additional entries
        for (const auto &dev : plt.get_devices()) {
          if (dev.has(aspect::gpu)) {
            std::string name = dev.get_info<info::device::name>();
            addEscapeSymbolToSpecialCharacters(name);
            std::string ver = dev.get_info<info::device::driver_version>();
            size_t pos = 0;
            if ((plt.get_backend() == backend::opencl) ||
                (plt.get_backend() == backend::ext_oneapi_level_zero)) {
              if ((pos = ver.find(".")) == std::string::npos) {
                throw std::runtime_error("Malformed syntax in version string");
              }
              pos++;
              size_t start = pos;
              // FIXME: What is going on here?!! It's insanity to expect a
              // different result than few lines above. Is that a weirdest
              // possible way to write "pos -= 1;"???
              if ((pos = ver.find(".", pos)) == std::string::npos) {
                throw std::runtime_error("Malformed syntax in version string");
              }
              ver.replace(start, pos - start, "*");
            } else if ((plt.get_backend() == backend::ext_oneapi_cuda) ||
                       (plt.get_backend() == backend::ext_oneapi_hip)) {
              if ((pos = ver.find(".")) == std::string::npos) {
                throw std::runtime_error("Malformed syntax in version string");
              }
              pos++;
              ver.replace(pos, ver.length(), "*");
            }

            std::cout << "DeviceName:{{" << name << "}},DriverVersion:{{" << ver
                      << "}}" << std::endl;
            passed = true;
            break;
          }
        }
      }
      return passed ? 0 : 1;
    } else {
      sycl::queue deviceQueue(gpu_selector_v);
      device dev = deviceQueue.get_device();
      std::cout << "Device: " << dev.get_info<info::device::name>()
                << std::endl;
      std::cout << "DriverVersion: "
                << dev.get_info<info::device::driver_version>() << std::endl;
      for (const DevDescT &desc : components) {
        if ((std::regex_match(dev.get_info<info::device::name>(),
                              std::regex(desc.devName))) &&
            (std::regex_match(dev.get_info<info::device::driver_version>(),
                              std::regex(desc.devDriverVer))))
          return 0;
      }
      return 1;
    }
  }

  // Test providing only the device name.
  if (test == "DEVICE_NAME_INFO"sv) {
    if (write) {
      std::optional<device> dev = get_test_device(aspect::gpu);
      if (!dev)
        return 3;

      std::string name = dev->get_info<info::device::name>();
      addEscapeSymbolToSpecialCharacters(name);
      std::cout << "DeviceName:{{" << name << "}}" << std::endl;
      return 0;
    } else {
      sycl::queue deviceQueue(gpu_selector_v);
      device dev = deviceQueue.get_device();
      std::cout << "Device: " << dev.get_info<info::device::name>()
                << std::endl;
      for (const DevDescT &desc : components) {
        if (std::regex_match(dev.get_info<info::device::name>(),
                             std::regex(desc.devName)))
          return 0;
      }
      return 1;
    }
  }

  // Test providing the platform name only.
  if (test == "PLATFORM_NAME_INFO"sv) {
    if (write) {
      std::optional<platform> plt = get_test_platform(aspect::gpu);
      if (!plt)
        return 3;
      std::string name = plt->get_info<info::platform::name>();
      addEscapeSymbolToSpecialCharacters(name);
      std::cout << "PlatformName:{{" << name << "}}" << std::endl;
      return 0;
    } else {
      sycl::queue deviceQueue(gpu_selector_v);
      device dev = deviceQueue.get_device();
      const auto &plt = dev.get_platform();
        std::cout << "Platform: " << plt.get_info<info::platform::name>()
                  << std::endl;
      for (const DevDescT &desc : components) {
        if (std::regex_match(plt.get_info<info::platform::name>(),
                             std::regex(desc.platName)))
          return 0;
      }
      return 1;
    }
  }

  // Test the GPU multiple devices option.
  if (test == "DEVICE_MULTI_INFO"sv) {
    if (write) {
      int count = 0;
      for (const auto &plt : platform::get_platforms()) {
        for (const auto &dev : plt.get_devices()) {
          if (!dev.has(aspect::gpu))
            continue;
          std::string name = dev.get_info<info::device::name>();
          addEscapeSymbolToSpecialCharacters(name);
          std::string ver = dev.get_info<info::device::driver_version>();
          if (is_known_be(plt.get_backend())) {
            if (count > 0) {
              std::cout << "|";
            }
            std::cout << "DeviceName:{{" << name << "}},DriverVersion:{{" << ver
                      << "}}";
            count++;
            break;
          }
        }
      }
      return count > 0 ? 0 : 1;
    } else {
      sycl::queue deviceQueue(gpu_selector_v);
      device dev = deviceQueue.get_device();
      std::cout << "Device: " << dev.get_info<info::device::name>()
                << std::endl;
      std::cout << "DriverVersion: "
                << dev.get_info<info::device::driver_version>() << std::endl;
      for (const DevDescT &desc : components) {
        if ((std::regex_match(dev.get_info<info::device::name>(),
                              std::regex(desc.devName))) &&
            (std::regex_match(dev.get_info<info::device::driver_version>(),
                              std::regex(desc.devDriverVer))))
          return 0;
      }
      return 1;
    }
  }

  // Test providing malformed syntax in the device name.
  if (test == "DEVICE_MALFORMED_INFO"sv) {
    if (write) {
      for (const auto &plt : platform::get_platforms()) {
        for (const auto &dev : plt.get_devices()) {
          if (!dev.has(aspect::gpu))
            continue;
          std::string name = dev.get_info<info::device::name>();
          addEscapeSymbolToSpecialCharacters(name);
          if (is_known_be(plt.get_backend())) {
            std::cout << "DeviceName:HAHA{{" << name << "}}" << std::endl;
            return 0;
          }
        }
      }
    } else {
      try {
        sycl::queue deviceQueue(gpu_selector_v);
        device dev = deviceQueue.get_device();
        const auto &plt = dev.get_platform();
      } catch (sycl::exception &E) {
        std::cout << "Caught exception: " << E.what() << std::endl;
        if (E.what() ==
            "Key DeviceName of SYCL_DEVICE_ALLOWLIST should have value which starts with {{ -30 (PI_ERROR_INVALID_VALUE)"sv)
          return 0;
      }
      return 1;
    }
  }

  // Test providing the platform name only.
  if (test == "PLATOFRM_MALFORMED_INFO"sv) {
    if (write) {
      std::optional<platform> plt = get_test_platform(aspect::gpu);
      if (!plt)
        return 3;

      std::string name = plt->get_info<info::platform::name>();
      addEscapeSymbolToSpecialCharacters(name);
      std::cout << "PlatformName:HAHA{{" << name << "}}" << std::endl;
      return 0;
    } else {
      try {
        sycl::queue deviceQueue(gpu_selector_v);
        device dev = deviceQueue.get_device();
        const auto &plt = dev.get_platform();
      } catch (sycl::exception &E) {
        std::cout << "Caught exception: " << E.what() << std::endl;
        if (E.what() ==
            "Key PlatformName of SYCL_DEVICE_ALLOWLIST should have value which starts with {{ -30 (PI_ERROR_INVALID_VALUE)"sv)
          return 0;
      }
      return 1;
    }
  }

  // Test a malformed device version number.
  if (test == "DRIVER_MALFORMED_INFO"sv) {
    if (write) {
      for (const auto &plt : platform::get_platforms()) {
        for (const auto &dev : plt.get_devices()) {
          if (!dev.has(aspect::gpu))
            continue;
          std::string name = dev.get_info<info::device::name>();
          addEscapeSymbolToSpecialCharacters(name);
          std::string ver = dev.get_info<info::device::driver_version>();
          if (is_known_be(plt.get_backend())) {
            std::cout << "DeviceName:{{" << name << "}},DriverVersion:HAHA{{"
                      << ver << "}}" << std::endl;
            return 0;
          }
        }
      }
      return 1;
    } else {
      try {
        sycl::queue deviceQueue(gpu_selector_v);
        device dev = deviceQueue.get_device();
        const auto &plt = dev.get_platform();
      } catch (sycl::exception &E) {
        std::cout << "Caught exception: " << E.what() << std::endl;
        if (E.what() ==
            "Key DriverVersion of SYCL_DEVICE_ALLOWLIST should have value which starts with {{ -30 (PI_ERROR_INVALID_VALUE)"sv)
          return 0;
      }
      return 1;
    }
  }

  // Test the platform name and version number.
  if (test == "PLATVER_MALFORMED_INFO"sv) {
    if (write) {
      std::optional<platform> plt = get_test_platform(aspect::gpu) ;
      if (!plt)
        return 3;

      std::string name = plt->get_info<info::platform::name>();
      addEscapeSymbolToSpecialCharacters(name);
      std::string ver = plt->get_info<info::platform::version>();
      std::cout << "PlatformName:{{" << name << "}},PlatformVersion:HAHA{{"
                << ver << "}}" << std::endl;
      return 0;
    } else {
      try {
        sycl::queue deviceQueue(gpu_selector_v);
        device dev = deviceQueue.get_device();
        const auto &plt = dev.get_platform();
      } catch (sycl::exception &E) {
        std::cout << "Caught exception: " << E.what() << std::endl;
        if (E.what() ==
            "Key PlatformVersion of SYCL_DEVICE_ALLOWLIST should have value which starts with {{ -30 (PI_ERROR_INVALID_VALUE)"sv)
          return 0;
      }
      return 1;
    }
  }

  std::cout << "Unknown test" << std::endl;;
  return 4;
}
