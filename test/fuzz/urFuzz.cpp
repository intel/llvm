// Copyright (C) 2023-2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

/*
This binary is meant to be run with a libFuzzer. It generates part of API calls
in different order in each iteration trying to crash the application. There are
some initial scenarios in the corpus directory for reaching better coverage of
tests.
*/

#include "kernel_entry_points.h"
#include "ur_api.h"
#include "utils.hpp"
#include <cassert>

namespace fuzz {

constexpr int MAX_VECTOR_SIZE = 1024;

int ur_platform_get(TestState &state) {
  ur_result_t res = urPlatformGet(state.adapters.data(), state.adapters.size(),
                                  state.num_entries, state.platforms.data(),
                                  &state.num_platforms);
  if (res != UR_RESULT_SUCCESS) {
    return -1;
  }
  if (state.platforms.size() != state.num_platforms) {
    state.platforms.resize(state.num_platforms);
  }

  return 0;
}

int ur_device_get(TestState &state) {
  if (state.platforms.empty() || state.platform_num >= state.platforms.size() ||
      state.platforms[0] == nullptr) {
    return -1;
  }

  ur_result_t res = UR_RESULT_SUCCESS;
  if (state.devices.size() == 0) {
    res = urDeviceGet(state.platforms[state.platform_num], state.device_type, 0,
                      nullptr, &state.num_devices);
    state.devices.resize(state.num_devices);
  } else {
    res = urDeviceGet(state.platforms[state.platform_num], state.device_type,
                      state.num_entries, state.devices.data(), nullptr);
  }
  if (res != UR_RESULT_SUCCESS) {
    return -1;
  }

  return 0;
}

int ur_device_release(TestState &state) {
  if (state.devices.empty()) {
    return -1;
  }

  ur_result_t res = urDeviceRelease(state.devices.back());
  if (res == UR_RESULT_SUCCESS) {
    state.devices.pop_back();
  }

  return 0;
}

int ur_context_create(TestState &state) {
  if (!state.device_exists() || state.contexts.size() > MAX_VECTOR_SIZE) {
    return -1;
  }

  ur_context_handle_t context;
  ur_result_t res = urContextCreate(state.devices.size(), state.devices.data(),
                                    nullptr, &context);
  if (res == UR_RESULT_SUCCESS) {
    state.contexts.emplace_back(std::make_unique<Context>(context));
  }

  return 0;
}

int ur_context_release(TestState &state) {
  if (!state.context_exists() ||
      !state.contexts[state.context_num]->host_pools.empty() ||
      !state.contexts[state.context_num]->device_pools.empty() ||
      !state.contexts[state.context_num]->no_pool_host_allocs.empty() ||
      !state.contexts[state.context_num]->no_pool_device_allocs.empty()) {
    return -1;
  }

  state.contexts.pop_back();

  return 0;
}

int pool_create(TestState &state, Pools &pools) {
  if (pools.size() > MAX_VECTOR_SIZE) {
    return -1;
  }

  ur_usm_pool_handle_t pool_handle;
  ur_usm_pool_desc_t pool_desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr,
                               UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK};
  ur_result_t res = urUSMPoolCreate(state.contexts[state.context_num]->handle,
                                    &pool_desc, &pool_handle);
  if (res == UR_RESULT_SUCCESS) {
    pools.emplace_back(std::make_unique<Pool>(pool_handle));
  }

  return 0;
}

int ur_usm_pool_create_host(TestState &state) {
  if (!state.context_exists()) {
    return -1;
  }

  return pool_create(state, state.contexts[state.context_num]->host_pools);
}

int ur_usm_pool_create_device(TestState &state) {
  if (!state.context_exists()) {
    return -1;
  }

  return pool_create(state, state.contexts[state.context_num]->device_pools);
}

int pool_release(TestState &state, Pools &pools) {
  if (pools.empty()) {
    return -1;
  }

  uint8_t index = state.get_vec_index(pools.size());
  pools.erase(pools.begin() + index);

  return 0;
}

int ur_usm_pool_release_host(TestState &state) {
  if (!state.context_exists()) {
    return -1;
  }

  return pool_release(state, state.contexts[state.context_num]->host_pools);
}

int ur_usm_pool_release_device(TestState &state) {
  if (!state.context_exists()) {
    return -1;
  }

  return pool_release(state, state.contexts[state.context_num]->device_pools);
}

int alloc_setup(TestState &state, uint16_t &alloc_size) {
  if (!state.context_exists()) {
    return -1;
  }

  if (state.get_next_input_data(&alloc_size) != 0) {
    return -1;
  }

  return 0;
}

int ur_usm_host_alloc_pool(TestState &state) {
  void *ptr;
  uint16_t alloc_size;
  ur_result_t res = UR_RESULT_SUCCESS;

  int ret = alloc_setup(state, alloc_size);
  if (ret != 0) {
    return -1;
  }

  auto &context = state.contexts[state.context_num];
  auto &pools = context->host_pools;
  auto index = state.get_vec_index(pools.size());
  if (index == -1) {
    return -1;
  }
  auto &pool = *(pools.begin() + index);
  if (pool->allocs.size() > MAX_VECTOR_SIZE) {
    return -1;
  }

  res =
      urUSMHostAlloc(context->handle, nullptr, pool->handle, alloc_size, &ptr);
  if (res == UR_RESULT_SUCCESS) {
    pool->allocs.emplace_back(std::make_unique<Alloc>(context->handle, ptr));
  }

  return 0;
}

int ur_usm_host_alloc_no_pool(TestState &state) {
  void *ptr;
  uint16_t alloc_size;
  ur_result_t res = UR_RESULT_SUCCESS;

  int ret = alloc_setup(state, alloc_size);
  if (ret != 0) {
    return -1;
  }

  auto &context = state.contexts[state.context_num];
  if (context->no_pool_host_allocs.size() > MAX_VECTOR_SIZE) {
    return -1;
  }
  res = urUSMHostAlloc(context->handle, nullptr, nullptr, alloc_size, &ptr);
  if (res == UR_RESULT_SUCCESS) {
    context->no_pool_host_allocs.emplace_back(
        std::make_unique<Alloc>(context->handle, ptr));
  }

  return 0;
}

int ur_usm_device_alloc_pool(TestState &state) {
  void *ptr;
  uint16_t alloc_size;
  ur_result_t res = UR_RESULT_SUCCESS;

  int ret = alloc_setup(state, alloc_size);
  if (ret != 0) {
    return -1;
  }

  if (!state.device_exists()) {
    return -1;
  }

  auto &device_pools = state.contexts[state.context_num]->device_pools;
  auto index = state.get_vec_index(device_pools.size());
  if (index == -1) {
    return -1;
  }

  auto &pool = *(device_pools.begin() + index);
  auto &context = state.contexts[state.context_num]->handle;
  auto &device = state.devices[state.device_num];
  if (pool->allocs.size() > MAX_VECTOR_SIZE) {
    return -1;
  }

  res = urUSMDeviceAlloc(context, device, nullptr, pool->handle, alloc_size,
                         &ptr);
  if (res == UR_RESULT_SUCCESS) {
    pool->allocs.emplace_back(std::make_unique<Alloc>(context, ptr));
  }

  return 0;
}

int ur_usm_device_alloc_no_pool(TestState &state) {
  void *ptr;
  uint16_t alloc_size;
  ur_result_t res = UR_RESULT_SUCCESS;

  int ret = alloc_setup(state, alloc_size);
  if (ret != 0) {
    return -1;
  }

  if (!state.device_exists()) {
    return -1;
  }

  auto &context = state.contexts[state.context_num];
  auto &device = state.devices[state.device_num];
  if (context->no_pool_device_allocs.size() > MAX_VECTOR_SIZE) {
    return -1;
  }

  res = urUSMDeviceAlloc(context->handle, device, nullptr, nullptr, alloc_size,
                         &ptr);
  if (res == UR_RESULT_SUCCESS) {
    context->no_pool_device_allocs.emplace_back(
        std::make_unique<Alloc>(context->handle, ptr));
  }

  return 0;
}

int free_pool(TestState &state, Pools &pools) {
  if (pools.empty()) {
    return -1;
  }

  auto index = state.get_vec_index(pools.size());
  if (index == -1) {
    return -1;
  }

  pools.erase(pools.begin() + index);

  return 0;
}

int ur_usm_free_host_pool(TestState &state) {
  if (!state.context_exists()) {
    return -1;
  }

  return free_pool(state, state.contexts[state.context_num]->host_pools);
}

int ur_usm_free_device_pool(TestState &state) {
  if (!state.context_exists()) {
    return -1;
  }

  return free_pool(state, state.contexts[state.context_num]->device_pools);
}

int free_no_pool(Allocs &allocs) {
  if (allocs.empty()) {
    return -1;
  }

  allocs.pop_back();

  return 0;
}

int ur_usm_free_host_no_pool(TestState &state) {
  if (!state.context_exists()) {
    return -1;
  }

  return free_no_pool(state.contexts[state.context_num]->no_pool_host_allocs);
}

int ur_usm_free_device_no_pool(TestState &state) {
  if (!state.context_exists()) {
    return -1;
  }

  return free_no_pool(state.contexts[state.context_num]->no_pool_device_allocs);
}

int ur_program_create_with_il(TestState &state) {
  if (!state.context_exists() || !state.device_exists()) {
    return -1;
  }

  std::vector<char> il_bin;
  ur_program_handle_t program;
  ur_kernel_handle_t kernel;
  ur_queue_handle_t queue;
  ur_event_handle_t event;
  auto &context = state.contexts[state.context_num]->handle;
  auto &device = state.devices[state.device_num];
  // TODO: Use some generic utility to retrieve/use kernels
  std::string kernel_name = uur::device_binaries::program_kernel_map["fill"][0];

  il_bin = state.load_kernel_source();
  if (il_bin.empty()) {
    return -1;
  }

  constexpr int vec_size = 64;
  std::vector<int> vec(vec_size, 0);

  urProgramCreateWithIL(context, il_bin.data(), il_bin.size(), nullptr,
                        &program);
  urProgramBuild(context, program, nullptr);

  ur_mem_handle_t memory_buffer;
  urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, vec_size * sizeof(int),
                    nullptr, &memory_buffer);
  urKernelCreate(program, kernel_name.data(), &kernel);
  urKernelSetArgMemObj(kernel, 0, nullptr, memory_buffer);

  urQueueCreate(context, device, nullptr, &queue);

  urEnqueueMemBufferWrite(queue, memory_buffer, true, 0, vec_size * sizeof(int),
                          vec.data(), 0, nullptr, &event);
  urEventWait(1, &event);
  urEventRelease(event);

  constexpr uint32_t nDim = 3;
  const size_t gWorkOffset[] = {0, 0, 0};
  const size_t gWorkSize[] = {vec_size * 4, 1, 1};
  const size_t lWorkSize[] = {1, 1, 1};

  urEnqueueKernelLaunch(queue, kernel, nDim, gWorkOffset, gWorkSize, lWorkSize,
                        0, nullptr, &event);
  urEventWait(1, &event);
  urEventRelease(event);

  urQueueFinish(queue);
  urMemRelease(memory_buffer);
  urQueueRelease(queue);
  urKernelRelease(kernel);
  urProgramRelease(program);

  return 0;
}

// Call loader init and teardown exactly once.
static struct UrLoader {
  UrLoader() {
    LoaderConfig config;
    ur_result_t res = urLoaderInit(0, config.handle);
    if (res != UR_RESULT_SUCCESS) {
      exit(0);
    }
  }
  ~UrLoader() { urLoaderTearDown(); }
  LoaderConfig config;
} UrLoader;

extern "C" int LLVMFuzzerTestOneInput(uint8_t *data, size_t size) {
  int next_api_call;
  auto data_provider = std::make_unique<FuzzedDataProvider>(data, size);
  TestState test_state(std::move(data_provider));
  int ret = -1;

  int (*api_wrappers[])(TestState &) = {
      ur_platform_get,
      ur_device_get,
      ur_device_release,
      ur_context_create,
      ur_context_release,
      ur_usm_pool_create_host,
      ur_usm_pool_create_device,
      ur_usm_pool_release_host,
      ur_usm_pool_release_device,
      ur_usm_host_alloc_pool,
      ur_usm_host_alloc_no_pool,
      ur_usm_device_alloc_pool,
      ur_usm_device_alloc_no_pool,
      ur_usm_free_host_pool,
      ur_usm_free_host_no_pool,
      ur_usm_free_device_pool,
      ur_usm_free_device_no_pool,
      ur_program_create_with_il,
  };

  ret = test_state.init();
  if (ret == -1) {
    return ret;
  }

  test_state.adapters.resize(test_state.num_entries);
  ur_result_t res =
      urAdapterGet(test_state.num_entries, test_state.adapters.data(),
                   &test_state.num_adapters);
  if (res != UR_RESULT_SUCCESS || test_state.num_adapters == 0) {
    return -1;
  }

  while ((next_api_call = test_state.get_next_api_call()) != -1) {
    ret = api_wrappers[next_api_call](test_state);
    if (ret) {
      return -1;
    }
  }

  return 0;
}
} // namespace fuzz
