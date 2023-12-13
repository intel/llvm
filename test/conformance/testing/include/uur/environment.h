// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_CONFORMANCE_INCLUDE_ENVIRONMENT_H_INCLUDED
#define UR_CONFORMANCE_INCLUDE_ENVIRONMENT_H_INCLUDED

#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <ur_api.h>
namespace uur {

struct PlatformEnvironment : ::testing::Environment {

    struct PlatformOptions {
        std::string platform_name;
        unsigned long platforms_count;
    };

    PlatformEnvironment(int argc, char **argv);
    virtual ~PlatformEnvironment() override = default;

    virtual void SetUp() override;
    virtual void TearDown() override;

    PlatformOptions parsePlatformOptions(int argc, char **argv);

    PlatformOptions platform_options;
    std::vector<ur_adapter_handle_t> adapters{};
    ur_platform_handle_t platform = nullptr;
    std::string error;
    static PlatformEnvironment *instance;
};

struct DevicesEnvironment : PlatformEnvironment {

    struct DeviceOptions {
        std::string device_name;
        unsigned long devices_count;
    };

    DevicesEnvironment(int argc, char **argv);
    virtual ~DevicesEnvironment() override = default;

    virtual void SetUp() override;
    virtual void TearDown() override;

    DeviceOptions parseDeviceOptions(int argc, char **argv);

    inline const std::vector<ur_device_handle_t> &GetDevices() const {
        return devices;
    }

    DeviceOptions device_options;
    std::vector<ur_device_handle_t> devices;
    ur_device_handle_t device = nullptr;
    static DevicesEnvironment *instance;
};

struct KernelsEnvironment : DevicesEnvironment {
    struct KernelOptions {
        std::string kernel_directory;
    };

    KernelsEnvironment(int argc, char **argv,
                       const std::string &kernels_default_dir);
    virtual ~KernelsEnvironment() override = default;

    virtual void SetUp() override;
    virtual void TearDown() override;

    void LoadSource(const std::string &kernel_name, uint32_t device_index,
                    std::shared_ptr<std::vector<char>> &binary_out);

    ur_result_t CreateProgram(ur_platform_handle_t hPlatform,
                              ur_context_handle_t hContext,
                              ur_device_handle_t hDevice,
                              const std::vector<char> &binary,
                              ur_program_handle_t *phProgram);

    std::vector<std::string> GetEntryPointNames(std::string program);

    static KernelsEnvironment *instance;

  private:
    KernelOptions parseKernelOptions(int argc, char **argv,
                                     const std::string &kernels_default_dir);
    std::string getKernelSourcePath(const std::string &kernel_name,
                                    uint32_t device_index);
    std::string getSupportedILPostfix(uint32_t device_index);

    KernelOptions kernel_options;
    // mapping between kernels (full_path + kernel_name) and their saved source.
    std::unordered_map<std::string, std::shared_ptr<std::vector<char>>>
        cached_kernels;
};

} // namespace uur

#endif // UR_CONFORMANCE_INCLUDE_ENVIRONMENT_H_INCLUDED
