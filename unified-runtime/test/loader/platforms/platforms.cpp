/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#include <cassert>
#include <cstdlib>

#include <logger/ur_logger.hpp>
#include <ur_print.hpp>

#include "ur_api.h"

using namespace logger;

//////////////////////////////////////////////////////////////////////////
int main(int, char *[]) {
  auto out = create_logger("TEST");

  ur_result_t status;

  // Initialize the platform
  status = urLoaderInit(0, nullptr);
  if (status != UR_RESULT_SUCCESS) {
    UR_LOG_L(out, ERR, "urLoaderInit failed with return code: {}", status);
    return 1;
  }
  UR_LOG_L(out, INFO, "urLoaderInit succeeded.");

  uint32_t adapterCount = 0;
  std::vector<ur_adapter_handle_t> adapters;
  status = urAdapterGet(0, nullptr, &adapterCount);
  if (status != UR_RESULT_SUCCESS) {
    UR_LOG_L(out, ERR, "urAdapterGet failed with return code: {}", status);
    return 1;
  }

  adapters.resize(adapterCount);
  status = urAdapterGet(adapterCount, adapters.data(), nullptr);
  if (status != UR_RESULT_SUCCESS) {
    UR_LOG_L(out, ERR, "urAdapterGet failed with return code: {}", status);
    return 1;
  }

  uint32_t platformCount = 0;
  std::vector<ur_platform_handle_t> platforms;
  for (auto adapter : adapters) {
    uint32_t adapterPlatformCount = 0;
    status = urPlatformGet(adapter, 0, nullptr, &adapterPlatformCount);
    if (status != UR_RESULT_SUCCESS) {
      UR_LOG_L(out, ERR, "urPlatformGet failed with return code: {}", status);
      goto out;
    }
    UR_LOG_L(out, INFO, "urPlatformGet found {} platforms",
             adapterPlatformCount);

    platforms.resize(platformCount + adapterPlatformCount);
    status = urPlatformGet(adapter, adapterPlatformCount,
                           &platforms[platformCount], &adapterPlatformCount);
    if (status != UR_RESULT_SUCCESS) {
      UR_LOG_L(out, ERR, "urPlatformGet failed with return code: {}", status);
      goto out;
    }
    platformCount += adapterPlatformCount;
  }

  for (auto p : platforms) {
    size_t name_len;
    status = urPlatformGetInfo(p, UR_PLATFORM_INFO_NAME, 0, nullptr, &name_len);
    if (status != UR_RESULT_SUCCESS) {
      UR_LOG_L(out, ERR, "urPlatformGetInfo failed with return code: {}",
               status);
      goto out;
    }

    char *name = (char *)malloc(name_len);
    assert(name != NULL);

    status =
        urPlatformGetInfo(p, UR_PLATFORM_INFO_NAME, name_len, name, nullptr);
    if (status != UR_RESULT_SUCCESS) {
      UR_LOG_L(out, ERR, "urPlatformGetInfo failed with return code: {}",
               status);
      free(name);
      goto out;
    }
    UR_LOG_L(out, INFO, "Found {} ", name);

    free(name);
  }
out:
  urLoaderTearDown();
  return status == UR_RESULT_SUCCESS ? 0 : 1;
}
