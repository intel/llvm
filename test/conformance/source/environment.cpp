// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstring>
#include <fstream>

#include "ur_filesystem_resolved.hpp"

#ifdef KERNELS_ENVIRONMENT
#include "kernel_entry_points.h"
#endif

#include <ur_util.hpp>
#include <uur/environment.h>
#include <uur/utils.h>

namespace uur {

constexpr char ERROR_NO_ADAPTER[] = "Could not load adapter";

PlatformEnvironment *PlatformEnvironment::instance = nullptr;

std::ostream &operator<<(std::ostream &out,
                         const ur_platform_handle_t &platform) {
    size_t size;
    urPlatformGetInfo(platform, UR_PLATFORM_INFO_NAME, 0, nullptr, &size);
    std::vector<char> name(size);
    urPlatformGetInfo(platform, UR_PLATFORM_INFO_NAME, size, name.data(),
                      nullptr);
    out << name.data();
    return out;
}

std::ostream &operator<<(std::ostream &out,
                         const std::vector<ur_platform_handle_t> &platforms) {
    for (auto platform : platforms) {
        out << "\n  * \"" << platform << "\"";
    }
    return out;
}

std::ostream &operator<<(std::ostream &out, const ur_device_handle_t &device) {
    size_t size;
    urDeviceGetInfo(device, UR_DEVICE_INFO_NAME, 0, nullptr, &size);
    std::vector<char> name(size);
    urDeviceGetInfo(device, UR_DEVICE_INFO_NAME, size, name.data(), nullptr);
    out << name.data();
    return out;
}

std::ostream &operator<<(std::ostream &out,
                         const std::vector<ur_device_handle_t> &devices) {
    for (auto device : devices) {
        out << "\n  * \"" << device << "\"";
    }
    return out;
}

uur::PlatformEnvironment::PlatformEnvironment(int argc, char **argv)
    : platform_options{parsePlatformOptions(argc, argv)} {
    instance = this;

    ur_loader_config_handle_t config;
    if (urLoaderConfigCreate(&config) == UR_RESULT_SUCCESS) {
        if (urLoaderConfigEnableLayer(config, "UR_LAYER_FULL_VALIDATION")) {
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

    uint32_t count = 0;
    if (urPlatformGet(adapters.data(), adapter_count, 0, nullptr, &count)) {
        error = "urPlatformGet() failed to get number of platforms.";
        return;
    }

    if (count == 0) {
        error = "Failed to find any platforms.";
        return;
    }

    std::vector<ur_platform_handle_t> platforms(count);
    if (urPlatformGet(adapters.data(), adapter_count, count, platforms.data(),
                      nullptr)) {
        error = "urPlatformGet failed to get platforms.";
        return;
    }

    if (platform_options.platform_name.empty()) {

        if (platforms.size() == 1 || platform_options.platforms_count == 1) {
            platform = platforms[0];
        } else {
            std::stringstream ss_error;
            ss_error << "Select a single platform from below using the "
                        "--platform=NAME "
                        "command-line option:"
                     << platforms << std::endl
                     << "or set --platforms_count=1.";
            error = ss_error.str();
            return;
        }
    } else {
        for (auto candidate : platforms) {
            size_t size;
            if (urPlatformGetInfo(candidate, UR_PLATFORM_INFO_NAME, 0, nullptr,
                                  &size)) {
                error = "urPlatformGetInfoFailed";
                return;
            }
            std::vector<char> platform_name(size);
            if (urPlatformGetInfo(candidate, UR_PLATFORM_INFO_NAME, size,
                                  platform_name.data(), nullptr)) {
                error = "urPlatformGetInfo() failed";
                return;
            }
            if (platform_options.platform_name == platform_name.data()) {
                platform = candidate;
                break;
            }
        }
        if (!platform) {
            std::stringstream ss_error;
            ss_error << "Platform \"" << platform_options.platform_name
                     << "\" not found. Select a single platform from below "
                        "using the "
                        "--platform=NAME command-line options:"
                     << platforms << std::endl
                     << "or set --platforms_count=1.";
            error = ss_error.str();
            return;
        }
    }
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

PlatformEnvironment::PlatformOptions
PlatformEnvironment::parsePlatformOptions(int argc, char **argv) {
    PlatformOptions options{};
    for (int argi = 1; argi < argc; ++argi) {
        const char *arg = argv[argi];
        if (!(std::strcmp(arg, "-h") && std::strcmp(arg, "--help"))) {
            // TODO - print help
            break;
        } else if (std::strncmp(
                       arg, "--platform=", sizeof("--platform=") - 1) == 0) {
            options.platform_name =
                std::string(&arg[std::strlen("--platform=")]);
        } else if (std::strncmp(arg, "--platforms_count=",
                                sizeof("--platforms_count=") - 1) == 0) {
            options.platforms_count = std::strtoul(
                &arg[std::strlen("--platforms_count=")], nullptr, 10);
        }
    }

    /* If a platform was not provided using the --platform command line option,
     * check if environment variable is set to use as a fallback. */
    if (options.platform_name.empty()) {
        auto env_platform = ur_getenv("UR_CTS_ADAPTER_PLATFORM");
        if (env_platform.has_value()) {
            options.platform_name = env_platform.value();
        }
    }

    return options;
}

DevicesEnvironment::DeviceOptions
DevicesEnvironment::parseDeviceOptions(int argc, char **argv) {
    DeviceOptions options{};
    for (int argi = 1; argi < argc; ++argi) {
        const char *arg = argv[argi];
        if (!(std::strcmp(arg, "-h") && std::strcmp(arg, "--help"))) {
            // TODO - print help
            break;
        } else if (std::strncmp(arg, "--device=", sizeof("--device=") - 1) ==
                   0) {
            options.device_name = std::string(&arg[std::strlen("--device=")]);
        } else if (std::strncmp(arg, "--devices_count=",
                                sizeof("--devices_count=") - 1) == 0) {
            options.devices_count = std::strtoul(
                &arg[std::strlen("--devices_count=")], nullptr, 10);
        }
    }
    return options;
}

DevicesEnvironment *DevicesEnvironment::instance = nullptr;

DevicesEnvironment::DevicesEnvironment(int argc, char **argv)
    : PlatformEnvironment(argc, argv),
      device_options(parseDeviceOptions(argc, argv)) {
    instance = this;
    if (!error.empty()) {
        return;
    }
    uint32_t count = 0;
    if (urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count)) {
        error = "urDevicesGet() failed to get number of devices.";
        return;
    }
    if (count == 0) {
        error = "Could not find any devices associated with the platform";
        return;
    }

    // Get the argument (devices_count) to limit test devices count.
    // In case, the devices_count is "0", the variable count will not be changed.
    // The CTS will run on all devices.
    if (device_options.device_name.empty()) {
        if (device_options.devices_count >
            (std::numeric_limits<uint32_t>::max)()) {
            error = "Invalid devices_count argument";
            return;
        } else if (device_options.devices_count > 0) {
            count = (std::min)(
                count, static_cast<uint32_t>(device_options.devices_count));
        }
        devices.resize(count);
        if (urDeviceGet(platform, UR_DEVICE_TYPE_ALL, count, devices.data(),
                        nullptr)) {
            error = "urDeviceGet() failed to get devices.";
            return;
        }
    } else {
        devices.resize(count);
        if (urDeviceGet(platform, UR_DEVICE_TYPE_ALL, count, devices.data(),
                        nullptr)) {
            error = "urDeviceGet() failed to get devices.";
            return;
        }
        for (unsigned i = 0; i < count; i++) {
            size_t size;
            if (urDeviceGetInfo(devices[i], UR_DEVICE_INFO_NAME, 0, nullptr,
                                &size)) {
                error = "urDeviceGetInfo() failed";
                return;
            }
            std::vector<char> device_name(size);
            if (urDeviceGetInfo(devices[i], UR_DEVICE_INFO_NAME, size,
                                device_name.data(), nullptr)) {
                error = "urDeviceGetInfo() failed";
                return;
            }
            if (device_options.device_name == device_name.data()) {
                device = devices[i];
                devices.clear();
                devices.resize(1);
                devices[0] = device;
                break;
            }
        }
        if (!device) {
            std::stringstream ss_error;
            ss_error << "Device \"" << device_options.device_name
                     << "\" not found. Select a single device from below "
                        "using the "
                        "--device=NAME command-line options:"
                     << devices << std::endl
                     << "or set --devices_count=COUNT.";
            error = ss_error.str();
            return;
        }
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

void DevicesEnvironment::TearDown() {
    PlatformEnvironment::TearDown();
    for (auto device : devices) {
        if (urDeviceRelease(device)) {
            error = "urDeviceRelease() failed";
            return;
        }
    }
}

KernelsEnvironment *KernelsEnvironment::instance = nullptr;

KernelsEnvironment::KernelsEnvironment(int argc, char **argv,
                                       const std::string &kernels_default_dir)
    : DevicesEnvironment(argc, argv),
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

std::string KernelsEnvironment::getSupportedILPostfix(uint32_t device_index) {
    std::stringstream IL;

    if (instance->GetDevices().size() == 0) {
        error = "no devices available on the platform";
        return {};
    }

    // special case for AMD as it doesn't support IL.
    ur_platform_backend_t backend;
    if (urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND, sizeof(backend),
                          &backend, nullptr)) {
        error = "failed to get backend from platform.";
        return {};
    }
    if (backend == UR_PLATFORM_BACKEND_HIP) {
        return ".bin";
    }

    auto device = instance->GetDevices()[device_index];
    std::string IL_version;
    if (uur::GetDeviceILVersion(device, IL_version)) {
        error = "failed to get device IL version";
        return {};
    }

    // TODO: This potentially needs updating as more adapters are tested.
    if (IL_version.find("SPIR-V") != std::string::npos) {
        IL << ".spv";
    } else if (IL_version.find("nvptx") != std::string::npos) {
        IL << ".bin";
    } else {
        error = "Undefined IL version: " + IL_version;
        return {};
    }

    return IL.str();
}

std::string
KernelsEnvironment::getKernelSourcePath(const std::string &kernel_name,
                                        uint32_t device_index) {
    std::stringstream path;
    path << kernel_options.kernel_directory << "/" << kernel_name;
    std::string il_postfix = getSupportedILPostfix(device_index);

    if (il_postfix.empty()) {
        return {};
    }

    std::string binary_name;
    for (const auto &entry : filesystem::directory_iterator(path.str())) {
        auto file_name = entry.path().filename().string();
        if (file_name.find(il_postfix) != std::string::npos) {
            binary_name = file_name;
            break;
        }
    }

    if (binary_name.empty()) {
        error =
            "failed retrieving kernel source path for kernel: " + kernel_name;
        return {};
    }

    path << "/" << binary_name;

    return path.str();
}

void KernelsEnvironment::LoadSource(
    const std::string &kernel_name, uint32_t device_index,
    std::shared_ptr<std::vector<char>> &binary_out) {
    std::string source_path =
        instance->getKernelSourcePath(kernel_name, device_index);

    if (source_path.empty()) {
        FAIL() << error;
    }

    if (cached_kernels.find(source_path) != cached_kernels.end()) {
        binary_out = cached_kernels[source_path];
        return;
    }

    std::ifstream source_file;
    source_file.open(source_path,
                     std::ios::binary | std::ios::in | std::ios::ate);

    if (!source_file.is_open()) {
        FAIL() << "failed opening kernel path: " + source_path;
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
    binary_out = binary_ptr;
}

ur_result_t KernelsEnvironment::CreateProgram(ur_platform_handle_t hPlatform,
                                              ur_context_handle_t hContext,
                                              ur_device_handle_t hDevice,
                                              const std::vector<char> &binary,
                                              ur_program_handle_t *phProgram) {
    ur_platform_backend_t backend;
    if (auto error = urPlatformGetInfo(hPlatform, UR_PLATFORM_INFO_BACKEND,
                                       sizeof(ur_platform_backend_t), &backend,
                                       nullptr)) {
        return error;
    }
    if (backend == UR_PLATFORM_BACKEND_HIP) {
        // The HIP adapter does not support urProgramCreateWithIL so we need to
        // use urProgramCreateWithBinary instead.
        if (auto error = urProgramCreateWithBinary(
                hContext, hDevice, binary.size(),
                reinterpret_cast<const uint8_t *>(binary.data()), nullptr,
                phProgram)) {
            return error;
        }
    } else {
        if (auto error = urProgramCreateWithIL(
                hContext, binary.data(), binary.size(), nullptr, phProgram)) {
            return error;
        }
    }
    return UR_RESULT_SUCCESS;
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
