/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file sanitizer.cpp
 *
 * @brief UR code generation and execution example for use with the Level Zero adapter.
 *
 * The codegen example demonstrates a complete flow for generating LLVM IR,
 * translating it to SPIR-V, and submitting the kernel to Level Zero Runtime via UR API.
 */

#include <cstddef>
#include <iostream>
#include <vector>

#include "helpers.h"
#include "ur_api.h"

constexpr unsigned PAGE_SIZE = 4096;

void ur_check(const ur_result_t r) {
    if (r != UR_RESULT_SUCCESS) {
        urLoaderTearDown();
        throw std::runtime_error("Unified runtime error: " + std::to_string(r));
    }
}

std::vector<ur_adapter_handle_t> get_adapters() {
    uint32_t adapterCount = 0;
    ur_check(urAdapterGet(0, nullptr, &adapterCount));

    if (!adapterCount) {
        throw std::runtime_error("No adapters available.");
    }

    std::vector<ur_adapter_handle_t> adapters(adapterCount);
    ur_check(urAdapterGet(adapterCount, adapters.data(), nullptr));
    return adapters;
}

std::vector<ur_adapter_handle_t>
get_supported_adapters(std::vector<ur_adapter_handle_t> &adapters) {
    std::vector<ur_adapter_handle_t> supported_adapters;
    for (auto adapter : adapters) {
        ur_adapter_backend_t backend;
        ur_check(urAdapterGetInfo(adapter, UR_ADAPTER_INFO_BACKEND,
                                  sizeof(ur_adapter_backend_t), &backend,
                                  nullptr));

        if (backend == UR_ADAPTER_BACKEND_LEVEL_ZERO) {
            supported_adapters.push_back(adapter);
        }
    }

    return supported_adapters;
}

std::vector<ur_platform_handle_t>
get_platforms(std::vector<ur_adapter_handle_t> &adapters) {
    uint32_t platformCount = 0;
    ur_check(urPlatformGet(adapters.data(), adapters.size(), 1, nullptr,
                           &platformCount));

    if (!platformCount) {
        throw std::runtime_error("No platforms available.");
    }

    std::vector<ur_platform_handle_t> platforms(platformCount);
    ur_check(urPlatformGet(adapters.data(), adapters.size(), platformCount,
                           platforms.data(), nullptr));
    return platforms;
}

std::vector<ur_device_handle_t> get_gpus(ur_platform_handle_t p) {
    uint32_t deviceCount = 0;
    ur_check(urDeviceGet(p, UR_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount));

    if (!deviceCount) {
        throw std::runtime_error("No GPUs available.");
    }

    std::vector<ur_device_handle_t> devices(deviceCount);
    ur_check(urDeviceGet(p, UR_DEVICE_TYPE_GPU, deviceCount, devices.data(),
                         nullptr));
    return devices;
}

template <typename T, size_t N> struct alignas(PAGE_SIZE) AlignedArray {
    T data[N];
};

int main() {
    ur_loader_config_handle_t loader_config = nullptr;
    ur_check(urLoaderInit(UR_DEVICE_INIT_FLAG_GPU, loader_config));

    auto adapters = get_adapters();
    auto supported_adapters = get_supported_adapters(adapters);
    auto platforms = get_platforms(supported_adapters);
    auto gpus = get_gpus(platforms.front());
    auto spv = generate_plus_one_spv();

    constexpr size_t mem_size = 128;

    auto current_device = gpus.front();

    ur_context_handle_t hContext;
    ur_check(urContextCreate(1, &current_device, nullptr, &hContext));

    ur_program_handle_t hProgram;
    ur_check(urProgramCreateWithIL(hContext, spv.data(), spv.size(), nullptr,
                                   &hProgram));
    ur_check(urProgramBuild(hContext, hProgram, nullptr));

    void *pMem;
    ur_usm_desc_t usmDesc{UR_STRUCTURE_TYPE_USM_DESC, nullptr,
                          UR_USM_ADVICE_FLAG_DEFAULT, 0};
    ur_check(urUSMDeviceAlloc(hContext, current_device, &usmDesc, nullptr,
                              mem_size, &pMem));

    ur_kernel_handle_t hKernel;
    ur_check(urKernelCreate(
        hProgram, "_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E11MyKernelR_4",
        &hKernel));
    ur_check(urKernelSetArgPointer(hKernel, 0, nullptr, &pMem));

    ur_queue_handle_t queue;
    ur_check(urQueueCreate(hContext, current_device, nullptr, &queue));

    const size_t gWorkOffset[] = {0, 0, 0};
    // for overflow
    const size_t gWorkSize[] = {mem_size + 1, 1, 1};
    const size_t lWorkSize[] = {1, 1, 1};
    ur_event_handle_t event;
    ur_check(urEnqueueKernelLaunch(queue, hKernel, 3, gWorkOffset, gWorkSize,
                                   lWorkSize, 0, nullptr, &event));

    ur_check(urQueueFinish(queue));

    return urLoaderTearDown() == UR_RESULT_SUCCESS;
}
