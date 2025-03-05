/*
 *
 * Copyright (C) 2020-2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <vector>

#include "ur_api.h"

//////////////////////////////////////////////////////////////////////////
int main(int, char *[]) {
  ur_result_t status;

  // Initialize the platform
  status = urLoaderInit(0, nullptr);
  if (status != UR_RESULT_SUCCESS) {
    std::cout << "urLoaderInit failed with return code: " << status
              << std::endl;
    return 1;
  }
  std::cout << "Platform initialized.\n";

  uint32_t adapterCount = 0;
  std::vector<ur_adapter_handle_t> adapters;
  uint32_t platformCount = 0;
  std::vector<ur_platform_handle_t> platforms;

  status = urAdapterGet(0, nullptr, &adapterCount);
  if (status != UR_RESULT_SUCCESS) {
    std::cout << "urAdapterGet failed with return code: " << status
              << std::endl;
    return 1;
  }
  adapters.resize(adapterCount);
  status = urAdapterGet(adapterCount, adapters.data(), nullptr);
  if (status != UR_RESULT_SUCCESS) {
    std::cout << "urAdapterGet failed with return code: " << status
              << std::endl;
    return 1;
  }

  status =
      urPlatformGet(adapters.data(), adapterCount, 1, nullptr, &platformCount);
  if (status != UR_RESULT_SUCCESS) {
    std::cout << "urPlatformGet failed with return code: " << status
              << std::endl;
    goto out;
  }

  platforms.resize(platformCount);
  status = urPlatformGet(adapters.data(), adapterCount, platformCount,
                         platforms.data(), nullptr);
  if (status != UR_RESULT_SUCCESS) {
    std::cout << "urPlatformGet failed with return code: " << status
              << std::endl;
    goto out;
  }

  for (auto p : platforms) {
    ur_api_version_t api_version = {};
    status = urPlatformGetApiVersion(p, &api_version);
    if (status != UR_RESULT_SUCCESS) {
      std::cout << "urPlatformGetApiVersion failed with return code: " << status
                << std::endl;
      goto out;
    }
    std::cout << "API version: " << UR_MAJOR_VERSION(api_version) << "."
              << UR_MINOR_VERSION(api_version) << std::endl;

    uint32_t deviceCount = 0;
    status = urDeviceGet(p, UR_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
    if (status != UR_RESULT_SUCCESS) {
      std::cout << "urDeviceGet failed with return code: " << status
                << std::endl;
      goto out;
    }

    std::vector<ur_device_handle_t> devices(deviceCount);
    status = urDeviceGet(p, UR_DEVICE_TYPE_GPU, deviceCount, devices.data(),
                         nullptr);
    if (status != UR_RESULT_SUCCESS) {
      std::cout << "urDeviceGet failed with return code: " << status
                << std::endl;
      goto out;
    }
    for (auto d : devices) {
      ur_device_type_t device_type = UR_DEVICE_TYPE_ALL;
      status = urDeviceGetInfo(d, UR_DEVICE_INFO_TYPE, sizeof(ur_device_type_t),
                               static_cast<void *>(&device_type), nullptr);
      if (status != UR_RESULT_SUCCESS) {
        std::cout << "urDeviceGetInfo failed with return code: " << status
                  << std::endl;
        goto out;
      }
      static const size_t DEVICE_NAME_MAX_LEN = 1024;
      char device_name[DEVICE_NAME_MAX_LEN] = {0};
      status = urDeviceGetInfo(d, UR_DEVICE_INFO_NAME, DEVICE_NAME_MAX_LEN - 1,
                               static_cast<void *>(&device_name), nullptr);
      if (status != UR_RESULT_SUCCESS) {
        std::cout << "urDeviceGetInfo failed with return code: " << status
                  << std::endl;
        goto out;
      }
      if (device_type == UR_DEVICE_TYPE_GPU) {
        std::cout << "Found a " << device_name << " gpu.\n";
      }
    }
  }

out:
  for (auto adapter : adapters) {
    urAdapterRelease(adapter);
  }
  urLoaderTearDown();
  return status == UR_RESULT_SUCCESS ? 0 : 1;
}
