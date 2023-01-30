// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_CONFORMANCE_INCLUDE_ENVIRONMENT_H_INCLUDED
#define UR_CONFORMANCE_INCLUDE_ENVIRONMENT_H_INCLUDED

#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <ur_api.h>
namespace uur {

struct PlatformEnvironment : ::testing::Environment {

    struct PlatformOptions {
        std::string platform_name;
    };

    PlatformEnvironment(int argc, char **argv);
    virtual ~PlatformEnvironment() override = default;

    virtual void SetUp() override;
    virtual void TearDown() override;

    PlatformOptions parsePlatformOptions(int argc, char **argv);

    PlatformOptions platform_options;
    ur_platform_handle_t platform = nullptr;
    std::string error;
    static PlatformEnvironment *instance;
};

struct DevicesEnvironment : PlatformEnvironment {

    DevicesEnvironment(int argc, char **argv);
    virtual ~DevicesEnvironment() override = default;

    virtual void SetUp() override;
    inline virtual void TearDown() override { PlatformEnvironment::TearDown(); }

    inline const std::vector<ur_device_handle_t> &GetDevices() const {
        return devices;
    }

    std::vector<ur_device_handle_t> devices;
    static DevicesEnvironment *instance;
};

} // namespace uur

#endif // UR_CONFORMANCE_INCLUDE_ENVIRONMENT_H_INCLUDED
