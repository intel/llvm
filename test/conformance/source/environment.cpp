// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

#include "ur_api.h"
#include "ur_filesystem_resolved.hpp"
#include "uur/checks.h"

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

constexpr std::pair<const char *, ur_platform_backend_t> backends[] = {
    {"LEVEL_ZERO", UR_PLATFORM_BACKEND_LEVEL_ZERO},
    {"L0", UR_PLATFORM_BACKEND_LEVEL_ZERO},
    {"OPENCL", UR_PLATFORM_BACKEND_OPENCL},
    {"CUDA", UR_PLATFORM_BACKEND_CUDA},
    {"HIP", UR_PLATFORM_BACKEND_HIP},
    {"NATIVE_CPU", UR_PLATFORM_BACKEND_NATIVE_CPU},
    {"UNKNOWN", UR_PLATFORM_BACKEND_UNKNOWN},
};

namespace {
constexpr const char *backend_to_str(ur_platform_backend_t backend) {
    for (auto b : backends) {
        if (b.second == backend) {
            return b.first;
        }
    }
    return "INVALID";
};

ur_platform_backend_t str_to_backend(std::string str) {

    std::transform(str.begin(), str.end(), str.begin(), ::toupper);
    for (auto b : backends) {
        if (b.first == str) {
            return b.second;
        }
    }
    return UR_PLATFORM_BACKEND_UNKNOWN;
};
} // namespace

std::ostream &operator<<(std::ostream &out,
                         const std::vector<ur_platform_handle_t> &platforms) {
    for (auto platform : platforms) {
        out << "\n  * \"" << platform << "\"";
    }
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
    : AdapterEnvironment(), platform_options{parsePlatformOptions(argc, argv)} {
    instance = this;

    // Check for errors from parsing platform options
    if (!error.empty()) {
        return;
    }

    selectPlatformFromOptions();
}

void uur::PlatformEnvironment::selectPlatformFromOptions() {
    struct platform_info {
        ur_adapter_handle_t adapter;
        ur_platform_handle_t platform;
        std::string name;
        ur_platform_backend_t backend;
    };
    std::vector<platform_info> platforms;
    for (auto a : adapters) {
        uint32_t count = 0;
        ASSERT_SUCCESS(urPlatformGet(&a, 1, 0, nullptr, &count));
        std::vector<ur_platform_handle_t> platform_list(count);
        ASSERT_SUCCESS(
            urPlatformGet(&a, 1, count, platform_list.data(), nullptr));

        for (auto p : platform_list) {
            all_platforms.push_back(p);
            ur_platform_backend_t backend;
            ASSERT_SUCCESS(urPlatformGetInfo(p, UR_PLATFORM_INFO_BACKEND,
                                             sizeof(ur_platform_backend_t),
                                             &backend, nullptr));

            size_t size;
            ASSERT_SUCCESS(
                urPlatformGetInfo(p, UR_PLATFORM_INFO_NAME, 0, nullptr, &size));
            std::vector<char> platform_name{};
            platform_name.reserve(size);
            ASSERT_SUCCESS(urPlatformGetInfo(p, UR_PLATFORM_INFO_NAME, size,
                                             platform_name.data(), nullptr));

            platforms.push_back(platform_info{
                a, p, std::string(platform_name.data()), backend});
        }
    }

    std::string default_name{};
    std::map<ur_platform_backend_t, std::string> backend_platform_names{};
    auto stream = std::stringstream{platform_options.platform_name};
    for (std::string filter; std::getline(stream, filter, ';');) {
        auto split = filter.find(':');
        if (split == std::string::npos) {
            default_name = filter;
        } else if (split == filter.length() - 1) {
            // E.g: `OPENCL:`, ignore it
        } else {
            backend_platform_names.insert(
                {str_to_backend(filter.substr(0, split)),
                 filter.substr(split + 1)});
        }
    }

    std::vector<platform_info> platforms_filtered{};
    std::copy_if(platforms.begin(), platforms.end(),
                 std::inserter(platforms_filtered, platforms_filtered.begin()),
                 [&](platform_info info) {
                     if (!default_name.empty() && default_name != info.name) {
                         return false;
                     }
                     if (backend_platform_names.count(info.backend) &&
                         backend_platform_names[info.backend] != info.name) {
                         return false;
                     }
                     if (platform_options.platform_backend &&
                         platform_options.platform_backend != info.backend) {
                         return false;
                     }
                     return true;
                 });

    if (platforms_filtered.size() == 0) {
        std::stringstream errstr;
        errstr << "No platforms were found with the following filters:";
        if (platform_options.platform_backend) {
            errstr << " --backend="
                   << backend_to_str(*platform_options.platform_backend);
        }
        if (!platform_options.platform_name.empty()) {
            errstr << " --platform=\"" << platform_options.platform_name
                   << "\"";
        }
        if (!platform_options.platform_backend &&
            platform_options.platform_name.empty()) {
            errstr << " (none)";
        }
        errstr << "\nAvailable platforms:\n";
        for (auto p : platforms) {
            errstr << "  --backend=" << backend_to_str(p.backend)
                   << " --platform=\"" << p.name << "\"\n";
        }
        FAIL() << errstr.str();
    } else if (platforms_filtered.size() == 1 ||
               platform_options.platforms_count == 1) {
        auto &selected = platforms_filtered[0];
        platform = selected.platform;
        adapter = selected.adapter;
        std::cerr << "Selected platform: [" << backend_to_str(selected.backend)
                  << "] " << selected.name << "\n";
    } else if (platforms_filtered.size() > 1) {
        std::stringstream errstr;
        errstr << "Multiple possible platforms found; please select one of the "
                  "following or set --platforms_count=1:\n";
        for (const auto &p : platforms_filtered) {
            errstr << "  --backend=" << backend_to_str(p.backend)
                   << " --platform=\"" << p.name << "\"\n";
        }
        FAIL() << errstr.str();
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
    auto parse_backend = [&](std::string backend_string) {
        options.platform_backend = str_to_backend(backend_string);
        if (options.platform_backend == UR_PLATFORM_BACKEND_UNKNOWN) {
            std::stringstream errstr{error};
            errstr << "--backend not valid; expected one of [";
            bool first = true;
            for (auto b : backends) {
                if (!first) {
                    errstr << ", ";
                }
                errstr << b.first;
                first = false;
            }
            errstr << "], but got `" << backend_string << "`";
            error = errstr.str();
            return false;
        }
        return true;
    };

    for (int argi = 1; argi < argc; ++argi) {
        const char *arg = argv[argi];
        if (!(std::strcmp(arg, "-h") && std::strcmp(arg, "--help"))) {
            // TODO - print help
            break;
        } else if (std::strncmp(
                       arg, "--platform=", sizeof("--platform=") - 1) == 0) {
            options.platform_name =
                std::string(&arg[std::strlen("--platform=")]);
        } else if (std::strncmp(arg, "--backend=", sizeof("--backend=") - 1) ==
                   0) {
            std::string backend_string{&arg[std::strlen("--backend=")]};
            if (!parse_backend(std::move(backend_string))) {
                return options;
            }
        } else if (std::strncmp(arg, "--platforms_count=",
                                sizeof("--platforms_count=") - 1) == 0) {
            options.platforms_count = std::strtoul(
                &arg[std::strlen("--platforms_count=")], nullptr, 10);
        }
    }

    /* If a platform was not provided using the --platform/--backend command line options,
     * check if environment variable is set to use as a fallback. */
    if (options.platform_name.empty()) {
        auto env_platform = ur_getenv("UR_CTS_ADAPTER_PLATFORM");
        if (env_platform.has_value()) {
            options.platform_name = env_platform.value();
        }
    }
    if (!options.platform_backend) {
        auto env_backend = ur_getenv("UR_CTS_BACKEND");
        if (env_backend.has_value()) {
            if (!parse_backend(env_backend.value())) {
                return options;
            }
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

std::string KernelsEnvironment::getTargetName() {
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

    std::string target = "";
    switch (backend) {
    case UR_PLATFORM_BACKEND_OPENCL:
    case UR_PLATFORM_BACKEND_LEVEL_ZERO:
        return "spir64";
    case UR_PLATFORM_BACKEND_CUDA:
        return "nvptx64-nvidia-cuda";
    case UR_PLATFORM_BACKEND_HIP:
        return "amdgcn-amd-amdhsa";
    case UR_PLATFORM_BACKEND_NATIVE_CPU:
        error = "native_cpu doesn't support kernel tests yet";
        return {};
    default:
        error = "unknown target.";
        return {};
    }
}

std::string
KernelsEnvironment::getKernelSourcePath(const std::string &kernel_name) {
    std::stringstream path;
    path << kernel_options.kernel_directory << "/" << kernel_name;

    std::string target_name = getTargetName();
    if (target_name.empty()) {
        return {};
    }

    path << "/" << target_name << ".bin.0";

    return path.str();
}

void KernelsEnvironment::LoadSource(
    const std::string &kernel_name,
    std::shared_ptr<std::vector<char>> &binary_out) {
    std::string source_path = instance->getKernelSourcePath(kernel_name);

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

ur_result_t KernelsEnvironment::CreateProgram(
    ur_platform_handle_t hPlatform, ur_context_handle_t hContext,
    ur_device_handle_t hDevice, const std::vector<char> &binary,
    const ur_program_properties_t *properties, ur_program_handle_t *phProgram) {
    ur_platform_backend_t backend;
    if (auto error = urPlatformGetInfo(hPlatform, UR_PLATFORM_INFO_BACKEND,
                                       sizeof(ur_platform_backend_t), &backend,
                                       nullptr)) {
        return error;
    }
    if (backend == UR_PLATFORM_BACKEND_HIP ||
        backend == UR_PLATFORM_BACKEND_CUDA) {
        // The CUDA and HIP adapters do not support urProgramCreateWithIL so we
        // need to use urProgramCreateWithBinary instead.
        auto size = binary.size();
        auto data = binary.data();
        if (auto error = urProgramCreateWithBinary(
                hContext, 1, &hDevice, &size,
                reinterpret_cast<const uint8_t **>(&data), properties,
                phProgram)) {
            return error;
        }
    } else {
        if (auto error =
                urProgramCreateWithIL(hContext, binary.data(), binary.size(),
                                      properties, phProgram)) {
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
