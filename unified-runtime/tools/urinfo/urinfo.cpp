// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "urinfo.hpp"
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace urinfo {
struct app {
  bool verbose = false;
  bool linear_ids = true;
  bool ignore_device_selector = false;
  ur_loader_config_handle_t loaderConfig = nullptr;
  std::vector<ur_adapter_handle_t> adapters;
  std::unordered_map<ur_adapter_handle_t, std::vector<ur_platform_handle_t>>
      adapterPlatformsMap;
  std::unordered_map<ur_platform_handle_t, std::vector<ur_device_handle_t>>
      platformDevicesMap;

  app(int argc, const char **argv) {
    parseArgs(argc, argv);
    if (!ignore_device_selector) {
      if (auto device_selector = std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        std::fprintf(stderr,
                     "info: Output filtered by ONEAPI_DEVICE_SELECTOR "
                     "environment variable, which is set to \"%s\".\n"
                     "To see all devices, use the "
                     "--ignore-device-selector CLI option.\n\n",
                     device_selector);
      }
    }
    UR_CHECK(urLoaderConfigCreate(&loaderConfig));
    UR_CHECK(
        urLoaderConfigEnableLayer(loaderConfig, "UR_LAYER_FULL_VALIDATION"));
    UR_CHECK(urLoaderInit(0, loaderConfig));
    enumerateDevices();
  }

  void parseArgs(int argc, const char **argv) {
    static const char *usage = R"(usage: %s [-h] [-v] [-V]

This tool enumerates Unified Runtime layers, adapters, platforms, and
devices which are currently visible in the local execution environment.

options:
  -h, --help            show this help message and exit
  --version             show version number and exit
  -v, --verbose         print additional information
  --no-linear-ids       do not show linear device ids
  --ignore-device-selector
                        do not use ONEAPI_DEVICE_SELECTOR to filter list of
                        devices
)";
    for (int argi = 1; argi < argc; argi++) {
      std::string_view arg{argv[argi]};
      if (arg == "-h" || arg == "--help") {
        std::printf(usage, argv[0]);
        std::exit(0);
      } else if (arg == "--version") {
        std::printf("%s v%s\n", argv[0], UR_VERSION);
        std::exit(0);
      } else if (arg == "-v" || arg == "--verbose") {
        verbose = true;
      } else if (arg == "--no-linear-ids") {
        linear_ids = false;
      } else if (arg == "--ignore-device-selector") {
        ignore_device_selector = true;
      } else {
        std::fprintf(stderr, "error: invalid argument: %s\n", argv[argi]);
        std::fprintf(stderr, usage, argv[0]);
        std::exit(1);
      }
    }
  }

  void enumerateDevices() {
    // Enumerate adapters.
    uint32_t numAdapters = 0;
    UR_CHECK(urAdapterGet(0, nullptr, &numAdapters));
    if (numAdapters == 0) {
      std::exit(0);
    }
    adapters.resize(numAdapters);
    UR_CHECK(urAdapterGet(numAdapters, adapters.data(), nullptr));

    auto urDeviceGetFn =
        ignore_device_selector ? urDeviceGet : urDeviceGetSelected;

    for (size_t adapterIndex = 0; adapterIndex < adapters.size();
         adapterIndex++) {
      auto adapter = adapters[adapterIndex];
      // Enumerate platforms
      uint32_t numPlatforms = 0;
      UR_CHECK(urPlatformGet(&adapter, 1, 0, nullptr, &numPlatforms));
      if (numPlatforms == 0) {
        continue;
      }
      adapterPlatformsMap[adapter].resize(numPlatforms);
      UR_CHECK(urPlatformGet(&adapter, 1, numPlatforms,
                             adapterPlatformsMap[adapter].data(), nullptr));

      for (size_t platformIndex = 0; platformIndex < numPlatforms;
           platformIndex++) {
        auto platform = adapterPlatformsMap[adapter][platformIndex];
        // Enumerate devices
        uint32_t numDevices = 0;
        UR_CHECK(urDeviceGetFn(platform, UR_DEVICE_TYPE_ALL, 0, nullptr,
                               &numDevices));
        if (numDevices == 0) {
          continue;
        }
        platformDevicesMap[platform].resize(numDevices);
        UR_CHECK(urDeviceGetFn(platform, UR_DEVICE_TYPE_ALL, numDevices,
                               platformDevicesMap[platform].data(), nullptr));
      }
    }
  }

  void printSummary() {
    for (size_t adapterIndex = 0; adapterIndex < adapters.size();
         adapterIndex++) {
      auto adapter = adapters[adapterIndex];
      auto &platforms = adapterPlatformsMap[adapter];
      size_t adapter_device_id = 0;
      std::string adapter_backend = urinfo::getAdapterBackend(adapter);
      for (size_t platformIndex = 0; platformIndex < platforms.size();
           platformIndex++) {
        auto platform = platforms[platformIndex];
        auto &devices = platformDevicesMap[platform];
        for (size_t deviceIndex = 0; deviceIndex < devices.size();
             deviceIndex++) {
          auto device = devices[deviceIndex];
          auto device_type = urinfo::getDeviceType(device);

          if (linear_ids) {
            std::cout << "[" << adapter_backend << ":" << device_type << "]";
            std::cout << "[" << adapter_backend << ":" << adapter_device_id
                      << "]";
          } else {
            std::cout << "[adapter(" << adapterIndex << "," << adapter_backend
                      << "):"
                      << "platform(" << platformIndex << "):"
                      << "device(" << deviceIndex << "," << device_type << ")]";
          }

          std::cout << " " << urinfo::getPlatformName(platform) << ", "
                    << urinfo::getDeviceName(device) << " "
                    << urinfo::getDeviceVersion(device) << " "
                    << "[" << urinfo::getDeviceDriverVersion(device) << "]\n";

          adapter_device_id++;
        }
      }
    }
  }

  void printDetail() {
    std::cout << "\n"
              << "[loader]:"
              << "\n";
    urinfo::printLoaderConfigInfos(loaderConfig);

    for (size_t adapterIndex = 0; adapterIndex < adapters.size();
         adapterIndex++) {
      auto adapter = adapters[adapterIndex];
      std::cout << "\n"
                << "[adapter(" << adapterIndex << ")]:"
                << "\n";
      urinfo::printAdapterInfos(adapter);

      size_t numPlatforms = adapterPlatformsMap[adapter].size();
      for (size_t platformIndex = 0; platformIndex < numPlatforms;
           platformIndex++) {
        auto platform = adapterPlatformsMap[adapter][platformIndex];
        std::cout << "\n"
                  << "[adapter(" << adapterIndex << "),"
                  << "platform(" << platformIndex << ")]:"
                  << "\n";
        urinfo::printPlatformInfos(platform);

        size_t numDevices = platformDevicesMap[platform].size();
        for (size_t deviceI = 0; deviceI < numDevices; deviceI++) {
          auto device = platformDevicesMap[platform][deviceI];
          std::cout << "\n"
                    << "[adapter(" << adapterIndex << "),"
                    << "platform(" << platformIndex << "),"
                    << "device(" << deviceI << ")]:"
                    << "\n";
          urinfo::printDeviceInfos(device);
        }
      }
    }
  }

  ~app() {
    urLoaderConfigRelease(loaderConfig);
    urLoaderTearDown();
  }
};
} // namespace urinfo

int main(int argc, const char **argv) {
  auto app = urinfo::app{argc, argv};
  app.printSummary();
  if (app.verbose) {
    app.printDetail();
  }
  return 0;
}
