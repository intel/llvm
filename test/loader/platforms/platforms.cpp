/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
 * LLVM-exception
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
    out.error("urLoaderInit failed with return code: {}", status);
    return 1;
  }
  out.info("urLoaderInit succeeded.");

  uint32_t adapterCount = 0;
  std::vector<ur_adapter_handle_t> adapters;
  status = urAdapterGet(0, nullptr, &adapterCount);
  if (status != UR_RESULT_SUCCESS) {
    out.error("urAdapterGet failed with return code: {}", status);
    return 1;
  }

  adapters.resize(adapterCount);
  status = urAdapterGet(adapterCount, adapters.data(), nullptr);
  if (status != UR_RESULT_SUCCESS) {
    out.error("urAdapterGet failed with return code: {}", status);
    return 1;
  }

  uint32_t platformCount = 0;
  std::vector<ur_platform_handle_t> platforms;

  status =
      urPlatformGet(adapters.data(), adapterCount, 1, nullptr, &platformCount);
  if (status != UR_RESULT_SUCCESS) {
    out.error("urPlatformGet failed with return code: {}", status);
    goto out;
  }
  out.info("urPlatformGet found {} platforms", platformCount);

  platforms.resize(platformCount);
  status = urPlatformGet(adapters.data(), adapterCount, platformCount,
                         platforms.data(), nullptr);
  if (status != UR_RESULT_SUCCESS) {
    out.error("urPlatformGet failed with return code: {}", status);
    goto out;
  }

  for (auto p : platforms) {
    size_t name_len;
    status = urPlatformGetInfo(p, UR_PLATFORM_INFO_NAME, 0, nullptr, &name_len);
    if (status != UR_RESULT_SUCCESS) {
      out.error("urPlatformGetInfo failed with return code: {}", status);
      goto out;
    }

    char *name = (char *)malloc(name_len);
    assert(name != NULL);

    status =
        urPlatformGetInfo(p, UR_PLATFORM_INFO_NAME, name_len, name, nullptr);
    if (status != UR_RESULT_SUCCESS) {
      out.error("urPlatformGetInfo failed with return code: {}", status);
      free(name);
      goto out;
    }
    out.info("Found {} ", name);

    free(name);
  }
out:
  urLoaderTearDown();
  return status == UR_RESULT_SUCCESS ? 0 : 1;
}
