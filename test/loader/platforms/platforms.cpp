/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */
#include <cassert>
#include <cstdlib>

#include <logger/ur_logger.hpp>
#include <ur_params.hpp>

#include "ur_api.h"

using namespace logger;

//////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
    logger::init("TEST");

    ur_result_t status;

    // Initialize the platform
    status = urInit(0);
    if (status != UR_RESULT_SUCCESS) {
        error("urInit failed with return code: {}", status);
        return 1;
    }
    info("urInit succeeded.");

    uint32_t platformCount = 0;
    std::vector<ur_platform_handle_t> platforms;

    status = urPlatformGet(1, nullptr, &platformCount);
    if (status != UR_RESULT_SUCCESS) {
        error("urPlatformGet failed with return code: {}", status);
        goto out;
    }
    info("urPlatformGet found {} platforms", platformCount);

    platforms.resize(platformCount);
    status = urPlatformGet(platformCount, platforms.data(), nullptr);
    if (status != UR_RESULT_SUCCESS) {
        error("urPlatformGet failed with return code: {}", status);
        goto out;
    }

    for (auto p : platforms) {
        size_t name_len;
        status =
            urPlatformGetInfo(p, UR_PLATFORM_INFO_NAME, 0, nullptr, &name_len);
        if (status != UR_RESULT_SUCCESS) {
            error("urPlatformGetInfo failed with return code: {}", status);
            goto out;
        }

        char *name = (char *)malloc(name_len);
        assert(name != NULL);

        status = urPlatformGetInfo(p, UR_PLATFORM_INFO_NAME, name_len, name,
                                   nullptr);
        if (status != UR_RESULT_SUCCESS) {
            error("urPlatformGetInfo failed with return code: {}", status);
            free(name);
            goto out;
        }
        info("Found {} ", name);

        free(name);
    }
out:
    urTearDown(nullptr);
    return status == UR_RESULT_SUCCESS ? 0 : 1;
}
