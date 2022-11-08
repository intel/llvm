// RUN: %clangxx -fsycl %s -o %t.out
//
// RUN: env WRITE_DEVICE_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
// RUN: env READ_DEVICE_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
//
// RUN: env WRITE_PLATFORM_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
// RUN: env READ_PLATFORM_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
//
// RUN: env WRITE_DEVICE_ERROR_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
// RUN: env READ_DEVICE_ERROR_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
//
// RUN: env WRITE_PLATFORM_ERROR_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
// RUN: env READ_PLATFORM_ERROR_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
//
// RUN: env WRITE_REG_EX_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
// RUN: env READ_REG_EX_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
//
// RUN: env WRITE_DEVICE_NAME_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
// RUN: env READ_DEVICE_NAME_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
//
// RUN: env WRITE_PLATFORM_NAME_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
// RUN: env READ_PLATFORM_NAME_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
//
// RUN: env WRITE_DEVICE_MULTI_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
// RUN: env READ_DEVICE_MULTI_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
//
// RUN: env WRITE_DEVICE_MALFORMED_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
// RUN: env READ_DEVICE_MALFORMED_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
//
// RUN: env WRITE_DRIVER_MALFORMED_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
// RUN: env READ_DRIVER_MALFORMED_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
//
// RUN: env WRITE_PLATFORM_MALFORMED_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
// RUN: env READ_PLATFORM_MALFORMED_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
//
// RUN: env WRITE_PLATVER_MALFORMED_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
// RUN: env READ_PLATVER_MALFORMED_INFO=1 %GPU_RUN_PLACEHOLDER %t.out
//
// REQUIRES: gpu

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
#include <sycl/sycl.hpp>

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
  std::vector<std::string> specialCharacters{"(", ")", "[", "]", "."};
  for (const auto &character : specialCharacters) {
    size_t pos = 0;
    while ((pos = str.find(character, pos)) != std::string::npos) {
      std::string modifiedCharacter("\\" + character);
      str.replace(pos, character.size(), modifiedCharacter);
      pos += modifiedCharacter.size();
    }
  }
}

static std::vector<DevDescT> getAllowListDesc(std::string allowList) {
  if (allowList.empty())
    return {};

  std::string deviceName("DeviceName:");
  std::string driverVersion("DriverVersion:");
  std::string platformName("PlatformName:");
  std::string platformVersion("PlatformVersion:");
  std::vector<DevDescT> decDescs;
  decDescs.emplace_back();

  size_t pos = 0;
  while (pos < allowList.size()) {
    if ((allowList.compare(pos, deviceName.size(), deviceName)) == 0) {
      if ((pos = allowList.find("{{", pos)) == std::string::npos) {
        throw std::runtime_error("Malformed device allowlist");
      }
      size_t start = pos + 2;
      if ((pos = allowList.find("}}", pos)) == std::string::npos) {
        throw std::runtime_error("Malformed device allowlist");
      }
      decDescs.back().devName = allowList.substr(start, pos - start);
      pos = pos + 2;

      if (allowList[pos] == ',') {
        pos++;
      }
    }

    else if ((allowList.compare(pos, driverVersion.size(), driverVersion)) ==
             0) {
      if ((pos = allowList.find("{{", pos)) == std::string::npos) {
        throw std::runtime_error("Malformed device allowlist");
      }
      size_t start = pos + 2;
      if ((pos = allowList.find("}}", pos)) == std::string::npos) {
        throw std::runtime_error("Malformed device allowlist");
      }
      decDescs.back().devDriverVer = allowList.substr(start, pos - start);
      pos = pos + 3;
    }

    else if ((allowList.compare(pos, platformName.size(), platformName)) == 0) {
      if ((pos = allowList.find("{{", pos)) == std::string::npos) {
        throw std::runtime_error("Malformed platform allowlist");
      }
      size_t start = pos + 2;
      if ((pos = allowList.find("}}", pos)) == std::string::npos) {
        throw std::runtime_error("Malformed platform allowlist");
      }
      decDescs.back().platName = allowList.substr(start, pos - start);
      pos = pos + 2;
      if (allowList[pos] == ',') {
        pos++;
      }
    }

    else if ((allowList.compare(pos, platformVersion.size(),
                                platformVersion)) == 0) {
      if ((pos = allowList.find("{{", pos)) == std::string::npos) {
        throw std::runtime_error("Malformed platform allowlist");
      }
      size_t start = pos + 2;
      if ((pos = allowList.find("}}", pos)) == std::string::npos) {
        throw std::runtime_error("Malformed platform allowlist");
      }
      decDescs.back().platVer = allowList.substr(start, pos - start);
      pos = pos + 2;
    }

    else if (allowList.find('|', pos) != std::string::npos) {
      pos = allowList.find('|') + 1;
      while (allowList[pos] == ' ') {
        pos++;
      }
      decDescs.emplace_back();
    } else {
      throw std::runtime_error("Malformed platform allowlist");
    }
  } // while (pos <= allowList.size())
  return decDescs;
}

int main() {
  bool passed = false;
  std::string sycl_be("opencl");
  if (getenv("ONEAPI_DEVICE_SELECTOR")) {
    sycl_be = getenv("ONEAPI_DEVICE_SELECTOR");
  }

  // Test the GPU devices name and version number.
  if (getenv("WRITE_DEVICE_INFO")) {
    std::ofstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      for (const auto &plt : platform::get_platforms()) {
        if (passed) {
          break;
        } // no need to write more than one match
        for (const auto &dev : plt.get_devices()) {
          if (dev.has(aspect::gpu)) {
            std::string name = dev.get_info<info::device::name>();
            addEscapeSymbolToSpecialCharacters(name);
            std::string ver = dev.get_info<info::device::driver_version>();
            if ((plt.get_backend() == backend::opencl) &&
                (sycl_be.find("opencl") != std::string::npos)) {
              fs << "DeviceName:{{" << name << "}},DriverVersion:{{" << ver
                 << "}}" << std::endl;
              passed = true;
              break;
            } else if ((plt.get_backend() == backend::ext_oneapi_level_zero) &&
                       (sycl_be.find("level_zero") != std::string::npos)) {
              fs << "DeviceName:{{" << name << "}},DriverVersion:{{" << ver
                 << "}}" << std::endl;
              passed = true;
              break;
            } else if ((plt.get_backend() == backend::ext_oneapi_cuda) &&
                       (sycl_be.find("cuda") != std::string::npos)) {
              fs << "DeviceName:{{" << name << "}},DriverVersion:{{" << ver
                 << "}}" << std::endl;
              passed = true;
              break;
            } else if ((plt.get_backend() == backend::ext_oneapi_hip) &&
                       (sycl_be.find("hip") != std::string::npos)) {
              fs << "DeviceName:{{" << name << "}},DriverVersion:{{" << ver
                 << "}}" << std::endl;
              passed = true;
              break;
            }
          }
        }
      }
      fs.close();
    }
  } else if (getenv("READ_DEVICE_INFO")) {
    std::ifstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      std::string allowlist;
      std::getline(fs, allowlist);
      if (!allowlist.empty()) {
        setenv("SYCL_DEVICE_ALLOWLIST", allowlist.c_str(), 0);
        std::vector<DevDescT> components(getAllowListDesc(allowlist));
        std::cout << "SYCL_DEVICE_ALLOWLIST=" << allowlist << std::endl;

        sycl::queue deviceQueue(gpu_selector_v);
        device dev = deviceQueue.get_device();
        for (const DevDescT &desc : components) {
          if ((std::regex_match(dev.get_info<info::device::name>(),
                                std::regex(desc.devName))) &&
              (std::regex_match(dev.get_info<info::device::driver_version>(),
                                std::regex(desc.devDriverVer)))) {
            passed = true;
          }
          std::cout << "Device: " << dev.get_info<info::device::name>()
                    << std::endl;
          std::cout << "DriverVersion: "
                    << dev.get_info<info::device::driver_version>()
                    << std::endl;
        }
      }
      fs.close();
    }
  }

  // Test the platform name and version number.
  if (getenv("WRITE_PLATFORM_INFO")) {
    std::ofstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      for (const auto &plt : platform::get_platforms()) {
        if (plt.has(aspect::gpu)) {
          std::string name = plt.get_info<info::platform::name>();
          addEscapeSymbolToSpecialCharacters(name);
          std::string ver = plt.get_info<info::platform::version>();
          if (((plt.get_backend() == backend::opencl) &&
               (sycl_be.find("opencl") != std::string::npos)) ||
              ((plt.get_backend() == backend::ext_oneapi_level_zero) &&
               (sycl_be.find("level_zero") != std::string::npos)) ||
              ((plt.get_backend() == backend::ext_oneapi_cuda) &&
               (sycl_be.find("cuda") != std::string::npos)) ||
              ((plt.get_backend() == backend::ext_oneapi_hip) &&
               (sycl_be.find("hip") != std::string::npos))) {
            fs << "PlatformName:{{" << name << "}},PlatformVersion:{{" << ver
               << "}}" << std::endl;
            passed = true;
            break;
          }
        }
      }
      fs.close();
    }
  } else if (getenv("READ_PLATFORM_INFO")) {
    std::ifstream fs;
    fs.open("select_device_config.txt", std::fstream::in);
    if (fs.is_open()) {
      std::string allowlist;
      std::getline(fs, allowlist);
      if (!allowlist.empty()) {
        setenv("SYCL_DEVICE_ALLOWLIST", allowlist.c_str(), 0);
        std::vector<DevDescT> components(getAllowListDesc(allowlist));
        std::cout << "SYCL_DEVICE_ALLOWLIST=" << allowlist << std::endl;

        sycl::queue deviceQueue(gpu_selector_v);
        device dev = deviceQueue.get_device();
        const auto &plt = dev.get_platform();
        for (const DevDescT &desc : components) {
          if ((std::regex_match(plt.get_info<info::platform::name>(),
                                std::regex(desc.platName))) &&
              (std::regex_match(plt.get_info<info::platform::version>(),
                                std::regex(desc.platVer)))) {
            passed = true;
          }
        }
        std::cout << "Platform: " << plt.get_info<info::platform::name>()
                  << std::endl;
        std::cout << "Platform Version: "
                  << plt.get_info<info::platform::version>() << std::endl;
      }
      fs.close();
    }
  }

  // Test error handling.
  if (getenv("WRITE_DEVICE_ERROR_INFO")) {
    std::ofstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      for (const auto &plt : platform::get_platforms()) {
        if (passed) {
          break;
        } // no need for additional entries
        for (const auto &dev : plt.get_devices()) {
          if (dev.has(aspect::gpu)) {
            std::string name = dev.get_info<info::device::name>();
            addEscapeSymbolToSpecialCharacters(name);
            std::string ver("98.76.54321");
            if (((plt.get_backend() == backend::opencl) &&
                 (sycl_be.find("opencl") != std::string::npos)) ||
                ((plt.get_backend() == backend::ext_oneapi_level_zero) &&
                 (sycl_be.find("level_zero") != std::string::npos)) ||
                ((plt.get_backend() == backend::ext_oneapi_cuda) &&
                 (sycl_be.find("cuda") != std::string::npos)) ||
                ((plt.get_backend() == backend::ext_oneapi_hip) &&
                 (sycl_be.find("hip") != std::string::npos))) {
              fs << "DeviceName:{{" << name << "}},DriverVersion:{{" << ver
                 << "}}" << std::endl;
              passed = true;
              break;
            }
          }
        }
      }
      fs.close();
    }
  } else if (getenv("READ_DEVICE_ERROR_INFO")) {
    std::ifstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      std::string allowlist;
      std::getline(fs, allowlist);
      if (!allowlist.empty()) {
        setenv("SYCL_DEVICE_ALLOWLIST", allowlist.c_str(), 0);
        std::vector<DevDescT> components(getAllowListDesc(allowlist));
        std::cout << "SYCL_DEVICE_ALLOWLIST=" << allowlist << std::endl;

        try {
          sycl::queue deviceQueue(gpu_selector_v);
          device dev = deviceQueue.get_device();
          const auto &plt = dev.get_platform();
        } catch (sycl::exception &E) {
          const std::string expectedMsg(
              "No device of requested type 'info::device_type::gpu' available");
          const std::string gotMessage(E.what());
          if (gotMessage.find(expectedMsg) != std::string::npos) {
            passed = true;
          } else {
            passed = false;
          }
        }
      }
      fs.close();
    }
  }

  // Test error condition when version number is not found.
  if (getenv("WRITE_PLATFORM_ERROR_INFO")) {
    std::ofstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      for (const auto &plt : platform::get_platforms()) {
        if (plt.has(aspect::gpu)) {
          std::string name = plt.get_info<info::platform::name>();
          addEscapeSymbolToSpecialCharacters(name);
          if ((plt.get_backend() == backend::opencl) &&
              (sycl_be.find("opencl") != std::string::npos)) {
            std::string ver("OpenCL 12.34");
            fs << "PlatformName:{{" << name << "}},PlatformVersion:{{" << ver
               << "}}" << std::endl;
            passed = true;
            break;
          } else if ((plt.get_backend() == backend::ext_oneapi_level_zero) &&
                     (sycl_be.find("level_zero") != std::string::npos)) {
            std::string ver("12.34");
            fs << "PlatformName:{{" << name << "}},PlatformVersion:{{" << ver
               << "}}" << std::endl;
            passed = true;
            break;
          } else if ((plt.get_backend() == backend::ext_oneapi_cuda) &&
                     (sycl_be.find("cuda") != std::string::npos)) {
            std::string ver("CUDA 89.78");
            fs << "PlatformName:{{" << name << "}},PlatformVersion:{{" << ver
               << "}}" << std::endl;
            passed = true;
            break;
          } else if ((plt.get_backend() == backend::ext_oneapi_hip) &&
                     (sycl_be.find("hip") != std::string::npos)) {
            std::string ver("67.88.9");
            fs << "PlatformName:{{" << name << "}},PlatformVersion:{{" << ver
               << "}}" << std::endl;
            passed = true;
            break;
          }
        }
      }
    }
    fs.close();
  } else if (getenv("READ_PLATFORM_ERROR_INFO")) {
    std::ifstream fs;
    fs.open("select_device_config.txt", std::fstream::in);
    if (fs.is_open()) {
      std::string allowlist;
      std::getline(fs, allowlist);
      if (!allowlist.empty()) {
        setenv("SYCL_DEVICE_ALLOWLIST", allowlist.c_str(), 0);
        std::vector<DevDescT> components(getAllowListDesc(allowlist));
        std::cout << "SYCL_DEVICE_ALLOWLIST=" << allowlist << std::endl;

        try {
          sycl::queue deviceQueue(gpu_selector_v);
          device dev = deviceQueue.get_device();
          const auto &plt = dev.get_platform();
        } catch (sycl::exception &E) {
          const std::string expectedMsg(
              "No device of requested type 'info::device_type::gpu' available");
          const std::string gotMessage(E.what());
          if (gotMessage.find(expectedMsg) != std::string::npos) {
            passed = true;
          } else {
            passed = false;
          }
        }
      }
      fs.close();
    }
  }

  // Test handling a regular expression in the device driver version number.
  if (getenv("WRITE_REG_EX_INFO")) {
    std::ofstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
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
            if (((plt.get_backend() == backend::opencl) &&
                 (sycl_be.find("opencl") != std::string::npos)) ||
                ((plt.get_backend() == backend::ext_oneapi_level_zero) &&
                 (sycl_be.find("level_zero") != std::string::npos))) {
              if ((pos = ver.find(".")) == std::string::npos) {
                throw std::runtime_error("Malformed syntax in version string");
              }
              pos++;
              size_t start = pos;
              if ((pos = ver.find(".", pos)) == std::string::npos) {
                throw std::runtime_error("Malformed syntax in version string");
              }
              ver.replace(start, pos - start, "*");
            } else if (((plt.get_backend() == backend::ext_oneapi_cuda) &&
                        (sycl_be.find("cuda") != std::string::npos)) ||
                       ((plt.get_backend() == backend::ext_oneapi_hip) &&
                        (sycl_be.find("hip") != std::string::npos))) {
              if ((pos = ver.find(".")) == std::string::npos) {
                throw std::runtime_error("Malformed syntax in version string");
              }
              pos++;
              ver.replace(pos, ver.length(), "*");
            }

            fs << "DeviceName:{{" << name << "}},DriverVersion:{{" << ver
               << "}}" << std::endl;
            passed = true;
            break;
          }
        }
      }
      fs.close();
    }
  } else if (getenv("READ_REG_EX_INFO")) {
    std::ifstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      std::string allowlist;
      std::getline(fs, allowlist);
      if (!allowlist.empty()) {
        setenv("SYCL_DEVICE_ALLOWLIST", allowlist.c_str(), 0);
        std::vector<DevDescT> components(getAllowListDesc(allowlist));
        std::cout << "SYCL_DEVICE_ALLOWLIST=" << allowlist << std::endl;

        sycl::queue deviceQueue(gpu_selector_v);
        device dev = deviceQueue.get_device();
        for (const DevDescT &desc : components) {
          if ((std::regex_match(dev.get_info<info::device::name>(),
                                std::regex(desc.devName))) &&
              (std::regex_match(dev.get_info<info::device::driver_version>(),
                                std::regex(desc.devDriverVer)))) {
            passed = true;
          }
          std::cout << "Device: " << dev.get_info<info::device::name>()
                    << std::endl;
          std::cout << "DriverVersion: "
                    << dev.get_info<info::device::driver_version>()
                    << std::endl;
        }
      }
      fs.close();
    }
  }

  // Test providing only the device name.
  if (getenv("WRITE_DEVICE_NAME_INFO")) {
    std::ofstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      for (const auto &plt : platform::get_platforms()) {
        if (passed) {
          break;
        } // no need for additional entries
        for (const auto &dev : plt.get_devices()) {
          if (dev.has(aspect::gpu)) {
            std::string name = dev.get_info<info::device::name>();
            addEscapeSymbolToSpecialCharacters(name);
            if (((plt.get_backend() == backend::opencl) &&
                 (sycl_be.find("opencl") != std::string::npos)) ||
                ((plt.get_backend() == backend::ext_oneapi_level_zero) &&
                 (sycl_be.find("level_zero") != std::string::npos)) ||
                ((plt.get_backend() == backend::ext_oneapi_cuda) &&
                 (sycl_be.find("cuda") != std::string::npos)) ||
                ((plt.get_backend() == backend::ext_oneapi_hip) &&
                 (sycl_be.find("hip") != std::string::npos))) {
              fs << "DeviceName:{{" << name << "}}" << std::endl;
              passed = true;
              break;
            }
          }
        }
      }
      fs.close();
    }
  } else if (getenv("READ_DEVICE_NAME_INFO")) {
    std::ifstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      std::string allowlist;
      std::getline(fs, allowlist);
      if (!allowlist.empty()) {
        setenv("SYCL_DEVICE_ALLOWLIST", allowlist.c_str(), 0);
        std::vector<DevDescT> components(getAllowListDesc(allowlist));
        std::cout << "SYCL_DEVICE_ALLOWLIST=" << allowlist << std::endl;

        sycl::queue deviceQueue(gpu_selector_v);
        device dev = deviceQueue.get_device();
        for (const DevDescT &desc : components) {
          if (std::regex_match(dev.get_info<info::device::name>(),
                               std::regex(desc.devName))) {
            passed = true;
          }
          std::cout << "Device: " << dev.get_info<info::device::name>()
                    << std::endl;
        }
      }
      fs.close();
    }
  }

  // Test providing the platform name only.
  if (getenv("WRITE_PLATFORM_NAME_INFO")) {
    std::ofstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      for (const auto &plt : platform::get_platforms()) {
        if (plt.has(aspect::gpu)) {
          std::string name = plt.get_info<info::platform::name>();
          addEscapeSymbolToSpecialCharacters(name);
          if (((plt.get_backend() == backend::opencl) &&
               (sycl_be.find("opencl") != std::string::npos)) ||
              ((plt.get_backend() == backend::ext_oneapi_level_zero) &&
               (sycl_be.find("level_zero") != std::string::npos)) ||
              ((plt.get_backend() == backend::ext_oneapi_cuda) &&
               (sycl_be.find("cuda") != std::string::npos)) ||
              ((plt.get_backend() == backend::ext_oneapi_hip) &&
               (sycl_be.find("hip") != std::string::npos))) {
            fs << "PlatformName:{{" << name << "}}" << std::endl;
            passed = true;
            break;
          }
        }
      }
    }
    fs.close();
  } else if (getenv("READ_PLATFORM_NAME_INFO")) {
    std::ifstream fs;
    fs.open("select_device_config.txt", std::fstream::in);
    if (fs.is_open()) {
      std::string allowlist;
      std::getline(fs, allowlist);
      if (!allowlist.empty()) {
        setenv("SYCL_DEVICE_ALLOWLIST", allowlist.c_str(), 0);
        std::vector<DevDescT> components(getAllowListDesc(allowlist));
        std::cout << "SYCL_DEVICE_ALLOWLIST=" << allowlist << std::endl;

        sycl::queue deviceQueue(gpu_selector_v);
        device dev = deviceQueue.get_device();
        const auto &plt = dev.get_platform();
        for (const DevDescT &desc : components) {
          if (std::regex_match(plt.get_info<info::platform::name>(),
                               std::regex(desc.platName))) {
            passed = true;
          }
          std::cout << "Platform: " << plt.get_info<info::platform::name>()
                    << std::endl;
        }
      }
      fs.close();
    }
  }

  // Test the GPU multiple devices option.
  if (getenv("WRITE_DEVICE_MULTI_INFO")) {
    std::ofstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      std::stringstream ss;
      int count = 0;
      for (const auto &plt : platform::get_platforms()) {
        for (const auto &dev : plt.get_devices()) {
          if (dev.has(aspect::gpu)) {
            std::string name = dev.get_info<info::device::name>();
            addEscapeSymbolToSpecialCharacters(name);
            std::string ver = dev.get_info<info::device::driver_version>();
            if (((plt.get_backend() == backend::opencl) &&
                 (sycl_be.find("opencl") != std::string::npos)) ||
                ((plt.get_backend() == backend::ext_oneapi_level_zero) &&
                 (sycl_be.find("level_zero") != std::string::npos)) ||
                ((plt.get_backend() == backend::ext_oneapi_cuda) &&
                 (sycl_be.find("cuda") != std::string::npos)) ||
                ((plt.get_backend() == backend::ext_oneapi_hip) &&
                 (sycl_be.find("hip") != std::string::npos))) {
              if (count > 0) {
                ss << " | ";
              }
              ss << "DeviceName:{{" << name << "}},DriverVersion:{{" << ver
                 << "}}";
              count++;
              passed = true;
              break;
            }
          }
        }
      }
      fs << ss.str() << std::endl;
      fs.close();
    }
  } else if (getenv("READ_DEVICE_MULTI_INFO")) {
    std::ifstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      std::string allowlist;
      std::getline(fs, allowlist);
      if (!allowlist.empty()) {
        setenv("SYCL_DEVICE_ALLOWLIST", allowlist.c_str(), 0);
        std::vector<DevDescT> components(getAllowListDesc(allowlist));
        std::cout << "SYCL_DEVICE_ALLOWLIST=" << allowlist << std::endl;

        sycl::queue deviceQueue(gpu_selector_v);
        device dev = deviceQueue.get_device();
        for (const DevDescT &desc : components) {
          if ((std::regex_match(dev.get_info<info::device::name>(),
                                std::regex(desc.devName))) &&
              (std::regex_match(dev.get_info<info::device::driver_version>(),
                                std::regex(desc.devDriverVer)))) {
            passed = true;
            std::cout << "Device: " << dev.get_info<info::device::name>()
                      << std::endl;
            std::cout << "DriverVersion: "
                      << dev.get_info<info::device::driver_version>()
                      << std::endl;
          }
        }
      }
      fs.close();
    }
  }

  // Test providing malformed syntax in the device name.
  if (getenv("WRITE_DEVICE_MALFORMED_INFO")) {
    std::ofstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      for (const auto &plt : platform::get_platforms()) {
        for (const auto &dev : plt.get_devices()) {
          if (dev.has(aspect::gpu)) {
            std::string name = dev.get_info<info::device::name>();
            addEscapeSymbolToSpecialCharacters(name);
            if (((plt.get_backend() == backend::opencl) &&
                 (sycl_be.find("opencl") != std::string::npos)) ||
                ((plt.get_backend() == backend::ext_oneapi_level_zero) &&
                 (sycl_be.find("level_zero") != std::string::npos)) ||
                ((plt.get_backend() == backend::ext_oneapi_cuda) &&
                 (sycl_be.find("cuda") != std::string::npos)) ||
                ((plt.get_backend() == backend::ext_oneapi_hip) &&
                 (sycl_be.find("hip") != std::string::npos))) {
              fs << "DeviceName:HAHA{{" << name << "}}" << std::endl;
              passed = true;
              break;
            }
          }
        }
      }
      fs.close();
    }
  } else if (getenv("READ_DEVICE_MALFORMED_INFO")) {
    std::ifstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      std::string allowlist;
      std::getline(fs, allowlist);
      if (!allowlist.empty()) {
        setenv("SYCL_DEVICE_ALLOWLIST", allowlist.c_str(), 0);
        std::vector<DevDescT> components(getAllowListDesc(allowlist));
        std::cout << "SYCL_DEVICE_ALLOWLIST=" << allowlist << std::endl;

        try {
          sycl::queue deviceQueue(gpu_selector_v);
          device dev = deviceQueue.get_device();
          const auto &plt = dev.get_platform();
        } catch (sycl::exception &E) {
          // Workaround to make CI pass.
          // TODO: after the submission of PR intel/llvm:3826, create PR to
          // intel/llvm-test-suite with removal of 1st parameter of the vector,
          // and transformation of std::vector<std::string> to std::string
          const std::vector<std::string> expectedMsgs{
              "Malformed syntax in SYCL_DEVICE_ALLOWLIST",
              "Key DeviceName of SYCL_DEVICE_ALLOWLIST should have value which "
              "starts with {{"};
          const std::string gotMessage(E.what());
          for (const auto &expectedMsg : expectedMsgs) {
            if (gotMessage.find(expectedMsg) != std::string::npos) {
              passed = true;
              break;
            }
            passed = false;
          }
        }
      }
      fs.close();
    }
  }

  // Test providing the platform name only.
  if (getenv("WRITE_PLATFORM_MALFORMED_INFO")) {
    std::ofstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      for (const auto &plt : platform::get_platforms()) {
        if (plt.has(aspect::gpu)) {
          std::string name = plt.get_info<info::platform::name>();
          addEscapeSymbolToSpecialCharacters(name);
          if (((plt.get_backend() == backend::opencl) &&
               (sycl_be.find("opencl") != std::string::npos)) ||
              ((plt.get_backend() == backend::ext_oneapi_level_zero) &&
               (sycl_be.find("level_zero") != std::string::npos)) ||
              ((plt.get_backend() == backend::ext_oneapi_cuda) &&
               (sycl_be.find("cuda") != std::string::npos)) ||
              ((plt.get_backend() == backend::ext_oneapi_hip) &&
               (sycl_be.find("hip") != std::string::npos))) {
            fs << "PlatformName:HAHA{{" << name << "}}" << std::endl;
            passed = true;
            break;
          }
        }
      }
    }
    fs.close();
  } else if (getenv("READ_PLATFORM_MALFORMED_INFO")) {
    std::ifstream fs;
    fs.open("select_device_config.txt", std::fstream::in);
    if (fs.is_open()) {
      std::string allowlist;
      std::getline(fs, allowlist);
      if (!allowlist.empty()) {
        setenv("SYCL_DEVICE_ALLOWLIST", allowlist.c_str(), 0);
        std::vector<DevDescT> components(getAllowListDesc(allowlist));
        std::cout << "SYCL_DEVICE_ALLOWLIST=" << allowlist << std::endl;

        try {
          sycl::queue deviceQueue(gpu_selector_v);
          device dev = deviceQueue.get_device();
          const auto &plt = dev.get_platform();
        } catch (sycl::exception &E) {
          // Workaround to make CI pass.
          // TODO: after the submission of PR intel/llvm:3826, create PR to
          // intel/llvm-test-suite with removal of 1st parameter of the vector,
          // and transformation of std::vector<std::string> to std::string
          const std::vector<std::string> expectedMsgs{
              "Malformed syntax in SYCL_DEVICE_ALLOWLIST",
              "Key PlatformName of SYCL_DEVICE_ALLOWLIST should have value "
              "which starts with {{"};
          const std::string gotMessage(E.what());
          for (const auto &expectedMsg : expectedMsgs) {
            if (gotMessage.find(expectedMsg) != std::string::npos) {
              passed = true;
              break;
            }
            passed = false;
          }
        }
      }
      fs.close();
    }
  }

  // Test a malformed device version number.
  if (getenv("WRITE_DRIVER_MALFORMED_INFO")) {
    std::ofstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      for (const auto &plt : platform::get_platforms()) {
        for (const auto &dev : plt.get_devices()) {
          if (dev.has(aspect::gpu)) {
            std::string name = dev.get_info<info::device::name>();
            addEscapeSymbolToSpecialCharacters(name);
            std::string ver = dev.get_info<info::device::driver_version>();
            if (((plt.get_backend() == backend::opencl) &&
                 (sycl_be.find("opencl") != std::string::npos)) ||
                ((plt.get_backend() == backend::ext_oneapi_level_zero) &&
                 (sycl_be.find("level_zero") != std::string::npos)) ||
                ((plt.get_backend() == backend::ext_oneapi_cuda) &&
                 (sycl_be.find("cuda") != std::string::npos)) ||
                ((plt.get_backend() == backend::ext_oneapi_hip) &&
                 (sycl_be.find("hip") != std::string::npos))) {
              fs << "DeviceName:{{" << name << "}},DriverVersion:HAHA{{" << ver
                 << "}}" << std::endl;
              passed = true;
              break;
            }
          }
        }
      }
      fs.close();
    }
  } else if (getenv("READ_DRIVER_MALFORMED_INFO")) {
    std::ifstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      std::string allowlist;
      std::getline(fs, allowlist);
      if (!allowlist.empty()) {
        setenv("SYCL_DEVICE_ALLOWLIST", allowlist.c_str(), 0);
        std::vector<DevDescT> components(getAllowListDesc(allowlist));
        std::cout << "SYCL_DEVICE_ALLOWLIST=" << allowlist << std::endl;

        try {
          sycl::queue deviceQueue(gpu_selector_v);
          device dev = deviceQueue.get_device();
          const auto &plt = dev.get_platform();
        } catch (sycl::exception &E) {
          // Workaround to make CI pass.
          // TODO: after the submission of PR intel/llvm:3826, create PR to
          // intel/llvm-test-suite with removal of 1st parameter of the vector,
          // and transformation of std::vector<std::string> to std::string
          const std::vector<std::string> expectedMsgs{
              "Malformed syntax in SYCL_DEVICE_ALLOWLIST",
              "Key DriverVersion of SYCL_DEVICE_ALLOWLIST should have value "
              "which starts with {{"};
          const std::string gotMessage(E.what());
          for (const auto &expectedMsg : expectedMsgs) {
            if (gotMessage.find(expectedMsg) != std::string::npos) {
              passed = true;
              break;
            }
            passed = false;
          }
        }
      }
      fs.close();
    }
  }

  // Test the platform name and version number.
  if (getenv("WRITE_PLATVER_MALFORMED_INFO")) {
    std::ofstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      for (const auto &plt : platform::get_platforms()) {
        if (plt.has(aspect::gpu)) {
          std::string name = plt.get_info<info::platform::name>();
          addEscapeSymbolToSpecialCharacters(name);
          std::string ver = plt.get_info<info::platform::version>();
          if (((plt.get_backend() == backend::opencl) &&
               (sycl_be.find("opencl") != std::string::npos)) ||
              ((plt.get_backend() == backend::ext_oneapi_level_zero) &&
               (sycl_be.find("level_zero") != std::string::npos)) ||
              ((plt.get_backend() == backend::ext_oneapi_cuda) &&
               (sycl_be.find("cuda") != std::string::npos)) ||
              ((plt.get_backend() == backend::ext_oneapi_hip) &&
               (sycl_be.find("hip") != std::string::npos))) {
            fs << "PlatformName:{{" << name << "}},PlatformVersion:HAHA{{"
               << ver << "}}" << std::endl;
            passed = true;
            break;
          }
        }
      }
    }
    fs.close();
  } else if (getenv("READ_PLATVER_MALFORMED_INFO")) {
    std::ifstream fs;
    fs.open("select_device_config.txt", std::fstream::in);
    if (fs.is_open()) {
      std::string allowlist;
      std::getline(fs, allowlist);
      if (!allowlist.empty()) {
        setenv("SYCL_DEVICE_ALLOWLIST", allowlist.c_str(), 0);
        std::vector<DevDescT> components(getAllowListDesc(allowlist));
        std::cout << "SYCL_DEVICE_ALLOWLIST=" << allowlist << std::endl;

        try {
          sycl::queue deviceQueue(gpu_selector_v);
          device dev = deviceQueue.get_device();
          const auto &plt = dev.get_platform();
        } catch (sycl::exception &E) {
          // Workaround to make CI pass.
          // TODO: after the submission of PR intel/llvm:3826, create PR to
          // intel/llvm-test-suite with removal of 1st parameter of the vector,
          // and transformation of std::vector<std::string> to std::string
          const std::vector<std::string> expectedMsgs{
              "Malformed syntax in SYCL_DEVICE_ALLOWLIST",
              "Key PlatformVersion of SYCL_DEVICE_ALLOWLIST should have value "
              "which starts with {{"};
          const std::string gotMessage(E.what());
          for (const auto &expectedMsg : expectedMsgs) {
            if (gotMessage.find(expectedMsg) != std::string::npos) {
              passed = true;
              break;
            }
            passed = false;
          }
        }
      }
      fs.close();
    }
  }

  if (passed) {
    std::cout << "Passed." << std::endl;
    return 0;
  } else {
    std::cout << "Failed." << std::endl;
    return 1;
  }
}
