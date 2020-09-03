// RUN: %clangxx -fsycl %s -o %t.out
//
// RUN: env WRITE_DEVICE_INFO=1 %t.out
// RUN: env READ_DEVICE_INFO=1 %t.out
//
// RUN: env WRITE_PLATFORM_INFO=1 %t.out
// RUN: env READ_PLATFORM_INFO=1 %t.out

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

#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace cl::sycl;

#ifdef _WIN32
#define setenv(name, value, overwrite) _putenv_s (name, value)
#endif

struct DevDescT {
  std::string devName;
  std::string devDriverVer;
  std::string platName;
  std::string platVer;
};

static std::vector<DevDescT> getAllowListDesc(std::string allowList) {
  if (allowList.empty())
    return {};

  std::string deviceName("DeviceName:");
  std::string driverVersion("DriverVersion:");
  std::string platformName("PlatformName:");
  std::string platformVersion("PlatformVersion:");
  std::vector<DevDescT> decDescs;

  size_t pos = 0;
  while ( pos <= allowList.size()) {
    decDescs.emplace_back();

    if ((allowList.compare(pos, deviceName.size(), deviceName)) == 0) {
      if ((pos = allowList.find("{{", pos)) == std::string::npos) {
          throw std::runtime_error("Malformed device allowlist");
      }
      size_t start = pos+2;
      if ((pos = allowList.find("}},", pos)) == std::string::npos) {
          throw std::runtime_error("Malformed device allowlist");
      }
      decDescs.back().devName = allowList.substr(start, pos-start);
      pos = pos+3;
      if ((allowList.compare(pos, driverVersion.size(), driverVersion)) == 0) {
        if ((pos = allowList.find("{{", pos)) == std::string::npos) {
          throw std::runtime_error("Malformed device allowlist");
        }
        start = pos+2;
        if ((pos = allowList.find("}}", pos)) == std::string::npos) {
          throw std::runtime_error("Malformed device allowlist");
        }
        decDescs.back().devDriverVer = allowList.substr(start, pos-start);
        pos = pos+3;
      } else {
        throw std::runtime_error("Malformed device allowlist");
      }
    }
    else if ((allowList.compare(pos, platformName.size(), platformName)) == 0) {
      if ((pos = allowList.find("{{", pos)) == std::string::npos) {
          throw std::runtime_error("Malformed platform allowlist");
      }
      size_t start = pos+2;
      if ((pos = allowList.find("}},", pos)) == std::string::npos) {
          throw std::runtime_error("Malformed platform allowlist");
      }
      decDescs.back().platName = allowList.substr(start, pos-start);
      pos = pos+3;
      if ((allowList.compare(pos, platformVersion.size(), platformVersion)) == 0) {
        if ((pos = allowList.find("{{", pos)) == std::string::npos) {
          throw std::runtime_error("Malformed platform allowlist");
        }
        start = pos+2;
        if ((pos = allowList.find("}}", pos)) == std::string::npos) {
          throw std::runtime_error("Malformed platform allowlist");
        }
        decDescs.back().platVer = allowList.substr(start, pos-start);
        pos = pos+3;
      } else {
        throw std::runtime_error("Malformed platform allowlist");
      }
    }
    else if (allowList.find('|', pos) != std::string::npos) {
      pos = allowList.find('|')+1;
      while (allowList[pos] == ' ') {
        pos++;
      }
    }
    else {
        throw std::runtime_error("Malformed platform allowlist");
      }
  }  // while (pos <= allowList.size())
  return decDescs;
}


int main() {
  bool passed = false;

  // Find the GPU devices on this system
  if (getenv("WRITE_DEVICE_INFO")) {
    std::ofstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      for (const auto &plt : platform::get_platforms()) {
        if (!plt.has(aspect::host)){
           for (const auto &dev : plt.get_devices()) {
            if (dev.has(aspect::gpu)) {
              std::string name = dev.get_info<info::device::name>();
              std::string ver = dev.get_info<info::device::driver_version>();
              fs << "DeviceName:{{" << name
                 << "}},DriverVersion:{{" << ver << "}}" << std::endl;
              passed=true;
              break;
            }
          }
        }
      }
      fs.close();
    }
  }
  else if (getenv("READ_DEVICE_INFO")) {
    std::ifstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      std::string allowlist;
      std::getline(fs, allowlist);
      if (! allowlist.empty()) {
          setenv("SYCL_DEVICE_ALLOWLIST", allowlist.c_str(), 0);
          std::vector<DevDescT> components(getAllowListDesc(allowlist));

          cl::sycl::queue deviceQueue(gpu_selector{});
          device dev = deviceQueue.get_device();
          for (const DevDescT &desc : components) {
            if ((dev.get_info<info::device::name>() == desc.devName) &&
                (dev.get_info<info::device::driver_version>() ==
                 desc.devDriverVer)) {
              passed = true;
            }
            std::cout << "SYCL_DEVICE_ALLOWLIST=" << allowlist << std::endl;
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
  // Find the platforms on this system.
  if (getenv("WRITE_PLATFORM_INFO")) {
    std::ofstream fs;
    fs.open("select_device_config.txt");
    if (fs.is_open()) {
      for (const auto &plt : platform::get_platforms()) {
        if (plt.has(aspect::gpu)){
          std::string pname = plt.get_info<info::platform::name>();
          std::string pver = plt.get_info<info::platform::version>();
          fs << "PlatformName:{{" << pname
             << "}},PlatformVersion:{{" << pver << "}}" << std::endl;
          passed=true;
          break;
        }
      }
    }
    fs.close();
  }
  else if (getenv("READ_PLATFORM_INFO")) {
    std::ifstream fs;
    fs.open("select_device_config.txt", std::fstream::in);
    if (fs.is_open()) {
      std::string allowlist;
      std::getline(fs, allowlist);
      if (! allowlist.empty()) {
        setenv("SYCL_DEVICE_ALLOWLIST", allowlist.c_str(), 0);
        std::vector<DevDescT> components(getAllowListDesc(allowlist));

        for (const auto &plt : platform::get_platforms()) {
          if (!plt.has(aspect::host)){
            for (const DevDescT &desc : components) {
              if ((plt.get_info<info::platform::name>() == desc.platName) &&
                  (plt.get_info<info::platform::version>() ==
                   desc.platVer)) {
                passed = true;
              }
              std::cout << "SYCL_DEVICE_ALLOWLIST=" << allowlist << std::endl;
              std::cout << "Platform: " << plt.get_info<info::platform::name>()
                        << std::endl;
              std::cout << "Platform Version: "
                        << plt.get_info<info::platform::version>()
                        << std::endl;
            }
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
    std:: cout << "Failed." << std::endl;
    return 1;
  }
}
