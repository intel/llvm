/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
 * LLVM-exception
 *
 * @file codegen.cpp
 *
 * @brief UR code generation and execution example for use with the Level Zero
 * adapter.
 *
 * The codegen example demonstrates a complete flow for generating LLVM IR,
 * translating it to SPIR-V, and submitting the kernel to Level Zero Runtime via
 * UR API.
 */

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
                              sizeof(ur_adapter_backend_t), &backend, nullptr));

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
  ur_check(
      urDeviceGet(p, UR_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr));
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

  constexpr int a_size = 32;
  AlignedArray<int, a_size> a, b;
  for (auto i = 0; i < a_size; ++i) {
    a.data[i] = a_size - i;
    b.data[i] = i;
  }

  auto current_device = gpus.front();

  ur_context_handle_t hContext;
  ur_check(urContextCreate(1, &current_device, nullptr, &hContext));

  ur_program_handle_t hProgram;
  ur_check(urProgramCreateWithIL(hContext, spv.data(), spv.size(), nullptr,
                                 &hProgram));
  ur_check(urProgramBuild(hContext, hProgram, nullptr));

  ur_mem_handle_t dA, dB;
  ur_check(urMemBufferCreate(hContext, UR_MEM_FLAG_READ_WRITE,
                             a_size * sizeof(int), nullptr, &dA));
  ur_check(urMemBufferCreate(hContext, UR_MEM_FLAG_READ_WRITE,
                             a_size * sizeof(int), nullptr, &dB));

  ur_kernel_handle_t hKernel;
  ur_check(urKernelCreate(hProgram, "plus1", &hKernel));
  ur_check(urKernelSetArgMemObj(hKernel, 0, nullptr, dA));
  ur_check(urKernelSetArgMemObj(hKernel, 1, nullptr, dB));

  ur_queue_handle_t queue;
  ur_check(urQueueCreate(hContext, current_device, nullptr, &queue));

  ur_check(urEnqueueMemBufferWrite(queue, dA, true, 0, a_size * sizeof(int),
                                   a.data, 0, nullptr, nullptr));
  ur_check(urEnqueueMemBufferWrite(queue, dB, true, 0, a_size * sizeof(int),
                                   b.data, 0, nullptr, nullptr));

  const size_t gWorkOffset[] = {0, 0, 0};
  const size_t gWorkSize[] = {128, 1, 1};
  const size_t lWorkSize[] = {1, 1, 1};
  ur_event_handle_t event;
  ur_check(urEnqueueKernelLaunch(queue, hKernel, 3, gWorkOffset, gWorkSize,
                                 lWorkSize, 0, nullptr, &event));

  ur_check(urEnqueueMemBufferRead(queue, dB, true, 0, a_size * sizeof(int),
                                  b.data, 1, &event, nullptr));

  ur_check(urQueueFinish(queue));

  std::cout << "Input Array: ";
  for (int i = 0; i < a_size; ++i) {
    std::cout << a.data[i] << " ";
  }
  std::cout << std::endl;

  bool expectedResult = false;

  std::cout << "Output Array: ";
  for (int i = 0; i < a_size; ++i) {
    std::cout << b.data[i] << " ";
    expectedResult |= (b.data[i] == a.data[i] + 1);
  }
  std::cout << std::endl;

  if (expectedResult) {
    std::cout << "Results are correct." << std::endl;
  } else {
    std::cout << "Results are incorrect." << std::endl;
  }

  return urLoaderTearDown() == UR_RESULT_SUCCESS && expectedResult ? 0 : 1;
}
