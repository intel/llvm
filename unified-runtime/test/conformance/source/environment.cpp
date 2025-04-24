// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "ur_api.h"
#include "ur_filesystem_resolved.hpp"
#include "uur/checks.h"
#include "uur/known_failure.h"

#ifdef KERNELS_ENVIRONMENT
#include "kernel_entry_points.h"
#endif

#include <ur_util.hpp>
#include <uur/environment.h>
#include <uur/utils.h>

namespace uur {

constexpr char ERROR_NO_ADAPTER[] = "Could not load adapter";

AdapterEnvironment *AdapterEnvironment::instance = nullptr;

AdapterEnvironment::AdapterEnvironment() {
  instance = this;

  ur_loader_config_handle_t config;
  if (urLoaderConfigCreate(&config) == UR_RESULT_SUCCESS) {
    if (urLoaderConfigEnableLayer(config, "UR_LAYER_FULL_VALIDATION") !=
        UR_RESULT_SUCCESS) {
      urLoaderConfigRelease(config);
      error = "Failed to enable validation layer";
      return;
    }
  } else {
    error = "Failed to create loader config handle";
    return;
  }

  ur_device_init_flags_t device_flags = 0;
  auto initResult = urLoaderInit(device_flags, config);
  auto configReleaseResult = urLoaderConfigRelease(config);
  switch (initResult) {
  case UR_RESULT_SUCCESS:
    break;
  case UR_RESULT_ERROR_UNINITIALIZED:
    error = ERROR_NO_ADAPTER;
    return;
  default:
    error = "urLoaderInit() failed";
    return;
  }

  if (configReleaseResult) {
    error = "Failed to destroy loader config handle";
    return;
  }

  uint32_t adapter_count = 0;
  urAdapterGet(0, nullptr, &adapter_count);
  adapters.resize(adapter_count);
  urAdapterGet(adapter_count, adapters.data(), nullptr);
}

PlatformEnvironment *PlatformEnvironment::instance = nullptr;

uur::PlatformEnvironment::PlatformEnvironment() : AdapterEnvironment() {
  instance = this;

  populatePlatforms();
}

void uur::PlatformEnvironment::populatePlatforms() {
  for (auto a : adapters) {
    uint32_t count = 0;
    ASSERT_SUCCESS(urPlatformGet(a, 0, nullptr, &count));
    if (count == 0) {
      continue;
    }
    std::vector<ur_platform_handle_t> platform_list(count);
    ASSERT_SUCCESS(urPlatformGet(a, count, platform_list.data(), nullptr));

    for (auto p : platform_list) {
      platforms.push_back(p);
    }
  }

  ASSERT_FALSE(platforms.empty())
      << "No platforms are available on any adapters";
}

void uur::PlatformEnvironment::SetUp() {
  if (!error.empty()) {
    if (error == ERROR_NO_ADAPTER) {
      GTEST_SKIP() << error;
    } else {
      FAIL() << error;
    }
  }
}

void uur::PlatformEnvironment::TearDown() {
  if (error == ERROR_NO_ADAPTER) {
    return;
  }
  for (auto adapter : adapters) {
    urAdapterRelease(adapter);
  }
  if (urLoaderTearDown()) {
    FAIL() << "urLoaderTearDown() failed";
  }
}

DevicesEnvironment *DevicesEnvironment::instance = nullptr;

DevicesEnvironment::DevicesEnvironment() : PlatformEnvironment() {
  instance = this;
  if (!error.empty()) {
    return;
  }

  for (auto &platform : platforms) {
    uint32_t platform_device_count = 0;
    urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, 0, nullptr,
                        &platform_device_count);
    std::vector<ur_device_handle_t> platform_devices(platform_device_count);
    urDeviceGetSelected(platform, UR_DEVICE_TYPE_ALL, platform_device_count,
                        platform_devices.data(), nullptr);
    ur_adapter_handle_t adapter = nullptr;
    urPlatformGetInfo(platform, UR_PLATFORM_INFO_ADAPTER,
                      sizeof(ur_adapter_handle_t), &adapter, nullptr);
    // query out platform and adapter for each device, push back device tuple
    // appropriately
    for (auto device : platform_devices) {
      devices.push_back(DeviceTuple({device, platform, adapter}));
    }
  }

  if (devices.empty()) {
    error = "Could not find any devices to test";
    return;
  }
}

void DevicesEnvironment::SetUp() {
  PlatformEnvironment::SetUp();
  if (error == ERROR_NO_ADAPTER) {
    return;
  }
  if (devices.empty() || !error.empty()) {
    FAIL() << error;
  }
}

KernelsEnvironment *KernelsEnvironment::instance = nullptr;

KernelsEnvironment::KernelsEnvironment(int argc, char **argv,
                                       const std::string &kernels_default_dir)
    : DevicesEnvironment(),
      kernel_options(parseKernelOptions(argc, argv, kernels_default_dir)) {
  instance = this;
  if (!error.empty()) {
    return;
  }
}

KernelsEnvironment::KernelOptions
KernelsEnvironment::parseKernelOptions(int argc, char **argv,
                                       const std::string &kernels_default_dir) {
  KernelOptions options;
  for (int argi = 1; argi < argc; ++argi) {
    const char *arg = argv[argi];
    if (std::strncmp(arg, "--kernel_directory=",
                     sizeof("--kernel_directory=") - 1) == 0) {
      options.kernel_directory =
          std::string(&arg[std::strlen("--kernel_directory=")]);
    }
  }
  if (options.kernel_directory.empty()) {
    options.kernel_directory = kernels_default_dir;
  }

  return options;
}

std::string
KernelsEnvironment::getDefaultTargetName(ur_platform_handle_t platform) {
  if (instance->GetDevices().size() == 0) {
    error = "no devices available on the platform";
    return {};
  }

  ur_backend_t backend;
  if (urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND, sizeof(backend),
                        &backend, nullptr)) {
    error = "failed to get backend from platform.";
    return {};
  }

  switch (backend) {
  case UR_BACKEND_OPENCL:
  case UR_BACKEND_LEVEL_ZERO:
    return "spir64";
  case UR_BACKEND_CUDA:
    return "nvptx64-nvidia-cuda";
  case UR_BACKEND_HIP:
    return "amdgcn-amd-amdhsa";
  case UR_BACKEND_OFFLOAD: {
    // TODO: In future this should use urDeviceSelectBinary
    auto result = ur_getenv("UR_OFFLOAD_TARGET_NAME");
    if (!result) {
      error = "For offload testing, please specify a target in "
              "`UR_OFFLOAD_TARGET_NAME`";
      return {};
    }
    return *result;
  }
  case UR_BACKEND_NATIVE_CPU:
    error = "native_cpu doesn't support kernel tests yet";
    return {};
  default:
    error = "unknown target.";
    return {};
  }
}

std::string
KernelsEnvironment::getKernelSourcePath(const std::string &kernel_name,
                                        const std::string &target_name) {
  std::stringstream path;
  path << kernel_options.kernel_directory << "/" << kernel_name << "/"
       << target_name << ".bin.0";
  return path.str();
}

void KernelsEnvironment::LoadSource(
    const std::string &kernel_name, ur_platform_handle_t platform,
    std::shared_ptr<std::vector<char>> &binary_out) {
  // We don't have a way to build device code for native cpu yet.
  UUR_KNOWN_FAILURE_ON_PARAM(platform, uur::NativeCPU{});

  std::string target_name = getDefaultTargetName(platform);
  if (target_name.empty()) {
    FAIL() << error;
  }

  return LoadSource(kernel_name, target_name, binary_out);
}

void KernelsEnvironment::LoadSource(
    const std::string &kernel_name, const std::string &target_name,
    std::shared_ptr<std::vector<char>> &binary_out) {
  std::string source_path =
      instance->getKernelSourcePath(kernel_name, target_name);

  if (cached_kernels.find(source_path) != cached_kernels.end()) {
    binary_out = cached_kernels[source_path];
    return;
  }

  std::ifstream source_file;
  source_file.open(source_path,
                   std::ios::binary | std::ios::in | std::ios::ate);

  if (!source_file.is_open()) {
    FAIL() << "failed opening kernel path: " + source_path
           << "\nNote: make sure that UR_CONFORMANCE_TARGET_TRIPLES includes "
           << '\'' << target_name << '\''
           << " and that device binaries have been built.";
  }

  size_t source_size = static_cast<size_t>(source_file.tellg());
  source_file.seekg(0, std::ios::beg);

  std::vector<char> device_binary(source_size);
  source_file.read(device_binary.data(), source_size);
  if (!source_file) {
    source_file.close();
    FAIL() << "failed reading kernel source data from file: " + source_path;
  }
  source_file.close();

  auto binary_ptr =
      std::make_shared<std::vector<char>>(std::move(device_binary));
  cached_kernels[kernel_name] = binary_ptr;
  binary_out = std::move(binary_ptr);
}

void KernelsEnvironment::CreateProgram(
    ur_platform_handle_t hPlatform, ur_context_handle_t hContext,
    ur_device_handle_t hDevice, const std::vector<char> &binary,
    const ur_program_properties_t *properties, ur_program_handle_t *phProgram) {
  // Seems to not support an IR compiler
  std::tuple<ur_platform_handle_t, ur_device_handle_t> tuple{hPlatform,
                                                             hDevice};
  UUR_KNOWN_FAILURE_ON_PARAM(tuple, uur::OpenCL{"gfx1100"});

  ur_backend_t backend;
  ASSERT_SUCCESS(urPlatformGetInfo(hPlatform, UR_PLATFORM_INFO_BACKEND,
                                   sizeof(ur_backend_t), &backend, nullptr));
  if (backend == UR_BACKEND_HIP || backend == UR_BACKEND_CUDA ||
      backend == UR_BACKEND_OFFLOAD) {
    // The CUDA and HIP adapters do not support urProgramCreateWithIL so we
    // need to use urProgramCreateWithBinary instead.
    auto size = binary.size();
    auto data = binary.data();
    ASSERT_SUCCESS(urProgramCreateWithBinary(
        hContext, 1, &hDevice, &size, reinterpret_cast<const uint8_t **>(&data),
        properties, phProgram));
  } else {
    ASSERT_SUCCESS(urProgramCreateWithIL(hContext, binary.data(), binary.size(),
                                         properties, phProgram));
  }
}

std::vector<std::string> KernelsEnvironment::GetEntryPointNames(
    [[maybe_unused]] std::string program_name) {
  std::vector<std::string> entry_points;
#ifdef KERNELS_ENVIRONMENT
  entry_points = uur::device_binaries::program_kernel_map[program_name];
#endif
  return entry_points;
}

void KernelsEnvironment::SetUp() {
  DevicesEnvironment::SetUp();
  if (!error.empty()) {
    FAIL() << error;
  }
}

void KernelsEnvironment::TearDown() {
  cached_kernels.clear();
  DevicesEnvironment::TearDown();
}
} // namespace uur
