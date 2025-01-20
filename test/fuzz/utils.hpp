// Copyright (C) 2023-2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fstream>
#include <fuzzer/FuzzedDataProvider.h>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <vector>

namespace fuzz {

enum FuzzerAPICall : uint8_t {
  UR_PLATFORM_GET,
  UR_DEVICE_GET,
  UR_DEVICE_RELEASE,
  UR_CONTEXT_CREATE,
  UR_CONTEXT_RELEASE,
  UR_USM_POOL_CREATE_HOST,
  UR_USM_POOL_CREATE_DEVICE,
  UR_USM_POOL_RELEASE_HOST,
  UR_USM_POOL_RELEASE_DEVICE,
  UR_USM_HOST_ALLOC_POOL,
  UR_USM_HOST_ALLOC_NO_POOL,
  UR_USM_DEVICE_ALLOC_POOL,
  UR_USM_DEVICE_ALLOC_NO_POOL,
  UR_USM_FREE_HOST_POOL,
  UR_USM_FREE_HOST_NO_POOL,
  UR_USM_FREE_DEVICE_POOL,
  UR_USM_FREE_DEVICE_NO_POOL,
  UR_PROGRAM_CREATE_WITH_IL,
  kMaxValue = UR_PROGRAM_CREATE_WITH_IL,
};

struct LoaderConfig {
  ur_loader_config_handle_t handle;

  LoaderConfig() {
    urLoaderConfigCreate(&handle);
    urLoaderConfigEnableLayer(handle, "UR_LAYER_FULL_VALIDATION");
  }
  ~LoaderConfig() { urLoaderConfigRelease(handle); }
};

struct Alloc {
  ur_context_handle_t context;
  void *ptr;

  Alloc(ur_context_handle_t context, void *ptr) : context(context), ptr(ptr) {}
  ~Alloc() { urUSMFree(context, ptr); }
};
using Allocs = std::vector<std::unique_ptr<Alloc>>;

struct Pool {
  ur_usm_pool_handle_t handle;
  Allocs allocs;

  Pool(ur_usm_pool_handle_t handle) : handle(handle) {}
  ~Pool() {
    allocs.clear();
    urUSMPoolRelease(handle);
  }
};
using Pools = std::vector<std::unique_ptr<Pool>>;

struct Context {
  ur_context_handle_t handle;
  Pools host_pools;
  Pools device_pools;
  Allocs no_pool_host_allocs;
  Allocs no_pool_device_allocs;

  Context(ur_context_handle_t handle) : handle(handle) {}
  ~Context() {
    host_pools.clear();
    device_pools.clear();
    no_pool_host_allocs.clear();
    no_pool_device_allocs.clear();
    urContextRelease(handle);
  }
};
using Contexts = std::vector<std::unique_ptr<Context>>;

struct TestState {
  static constexpr uint32_t num_entries = 1;

  std::unique_ptr<FuzzedDataProvider> data_provider;

  std::vector<ur_adapter_handle_t> adapters;
  std::vector<ur_platform_handle_t> platforms;
  std::vector<ur_device_handle_t> devices;
  Contexts contexts;
  ur_device_type_t device_type = UR_DEVICE_TYPE_ALL;

  uint32_t num_adapters;
  uint32_t num_platforms;
  uint32_t num_devices;

  uint8_t platform_num;
  uint8_t device_num;
  uint8_t context_num;

  TestState(std::unique_ptr<FuzzedDataProvider> data_provider)
      : data_provider(std::move(data_provider)) {
    num_adapters = 0;
    num_platforms = 0;
    num_devices = 0;
  }

  template <typename IntType> int get_next_input_data(IntType *data) {
    if (data_provider->remaining_bytes() < sizeof(IntType)) {
      return -1;
    }
    *data = data_provider->ConsumeIntegral<IntType>();

    return 0;
  }

  template <typename IntType>
  int get_next_input_data_in_range(IntType *data, IntType min, IntType max) {
    if (data_provider->remaining_bytes() < sizeof(IntType)) {
      return -1;
    }
    *data = data_provider->ConsumeIntegralInRange<IntType>(min, max);

    return 0;
  }

  template <typename EnumType> int get_next_input_data_enum(EnumType *data) {
    if (data_provider->remaining_bytes() < sizeof(EnumType)) {
      return -1;
    }
    *data = data_provider->ConsumeEnum<EnumType>();

    return 0;
  }

  int init() {
    constexpr uint8_t UR_DEVICE_TYPE_MIN = 1;
    constexpr uint8_t UR_DEVICE_TYPE_MAX = 7;
    uint8_t device_type_int = 0;

    // TODO: Generate these values on the fly, when needed.
    // This will make more than one platform/device/context available
    if (get_next_input_data(&platform_num) != 0) {
      return -1;
    }

    if (get_next_input_data(&device_num) != 0) {
      return -1;
    }

    if (get_next_input_data(&context_num) != 0) {
      return -1;
    }

    if (get_next_input_data_in_range(&device_type_int, UR_DEVICE_TYPE_MIN,
                                     UR_DEVICE_TYPE_MAX) != 0) {
      return -1;
    }
    device_type = static_cast<ur_device_type_t>(device_type_int);

    return 0;
  }

  int get_next_api_call() {
    FuzzerAPICall next_api_call;
    return get_next_input_data_enum(&next_api_call) == 0 ? next_api_call : -1;
  }

  bool device_exists() {
    if (devices.empty() || device_num >= devices.size() ||
        devices[0] == nullptr) {
      return false;
    }

    return true;
  }

  bool context_exists() {
    if (contexts.empty() || context_num >= contexts.size() ||
        contexts[0]->handle == nullptr) {
      return false;
    }

    return true;
  }

  int get_vec_index(const uint8_t vec_size) {
    if (vec_size == 0 || data_provider->remaining_bytes() < sizeof(vec_size)) {
      return -1;
    }
    return data_provider->ConsumeIntegralInRange<decltype(vec_size)>(
        0, vec_size - 1);
  }

  auto load_kernel_source() {
    std::string source_path = KERNEL_IL_PATH;
    std::ifstream source_file;
    std::vector<char> binary;

    try {
      source_file.open(source_path,
                       std::ios::binary | std::ios::in | std::ios::ate);
      if (!source_file.is_open()) {
        std::cerr << "Failed to open a kernel source file: " << source_path
                  << std::endl;
        return binary;
      }

      size_t source_size = static_cast<size_t>(source_file.tellg());
      source_file.seekg(0, std::ios::beg);

      std::vector<char> device_binary(source_size);
      source_file.read(device_binary.data(), source_size);
      if (!source_file) {
        source_file.close();
        std::cerr << "failed reading kernel source data from file: "
                  << source_path << std::endl;
        return binary;
      }
      source_file.close();

      return std::vector<char>(std::move(device_binary));
    } catch (...) {
      return binary;
    }
  }
};

} // namespace fuzz
