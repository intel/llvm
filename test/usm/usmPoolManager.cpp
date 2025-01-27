// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "umf_pools/disjoint_pool_config_parser.hpp"
#include "ur_pool_manager.hpp"

#include <uur/fixtures.h>

using urUsmPoolDescriptorTest = uur::urMultiDeviceContextTest;

UUR_INSTANTIATE_PLATFORM_TEST_SUITE_P(urUsmPoolDescriptorTest);

auto createMockPoolHandle() {
  static uintptr_t uniqueAddress = 0x1;
  return umf::pool_unique_handle_t((umf_memory_pool_handle_t)(uniqueAddress++),
                                   [](umf_memory_pool_t *) {});
}

bool compareConfig(const usm::umf_disjoint_pool_config_t &left,
                   usm::umf_disjoint_pool_config_t &right) {
  return left.MaxPoolableSize == right.MaxPoolableSize &&
         left.Capacity == right.Capacity &&
         left.SlabMinSize == right.SlabMinSize;
}

bool compareConfigs(const usm::DisjointPoolAllConfigs &left,
                    usm::DisjointPoolAllConfigs &right) {
  return left.EnableBuffers == right.EnableBuffers &&
         compareConfig(left.Configs[usm::DisjointPoolMemType::Host],
                       right.Configs[usm::DisjointPoolMemType::Host]) &&
         compareConfig(left.Configs[usm::DisjointPoolMemType::Device],
                       right.Configs[usm::DisjointPoolMemType::Device]) &&
         compareConfig(left.Configs[usm::DisjointPoolMemType::Shared],
                       right.Configs[usm::DisjointPoolMemType::Shared]) &&
         compareConfig(left.Configs[usm::DisjointPoolMemType::SharedReadOnly],
                       right.Configs[usm::DisjointPoolMemType::SharedReadOnly]);
}

TEST_P(urUsmPoolDescriptorTest, poolIsPerContextTypeAndDevice) {
  auto &devices = uur::DevicesEnvironment::instance->devices;

  auto [ret, pool_descriptors] =
      usm::pool_descriptor::create(nullptr, this->context);
  ASSERT_EQ(ret, UR_RESULT_SUCCESS);

  size_t hostPools = 0;
  size_t devicePools = 0;
  size_t sharedPools = 0;

  for (auto &desc : pool_descriptors) {
    ASSERT_EQ(desc.poolHandle, nullptr);
    ASSERT_EQ(desc.hContext, this->context);

    if (desc.type == UR_USM_TYPE_DEVICE) {
      devicePools++;
    } else if (desc.type == UR_USM_TYPE_SHARED) {
      sharedPools++;
    } else if (desc.type == UR_USM_TYPE_HOST) {
      hostPools += 2;
    } else {
      FAIL();
    }
  }

  // Each device has pools for Host, Device, Shared, SharedReadOnly only
  ASSERT_EQ(pool_descriptors.size(), 4 * devices.size());
  ASSERT_EQ(hostPools, 1);
  ASSERT_EQ(devicePools, devices.size());
  ASSERT_EQ(sharedPools, devices.size() * 2);
}

// TODO: add test with sub-devices

struct urUsmPoolManagerTest : public uur::urContextTest {
  void SetUp() override {
    UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
    auto [ret, descs] = usm::pool_descriptor::create(nullptr, context);
    ASSERT_EQ(ret, UR_RESULT_SUCCESS);
    poolDescriptors = std::move(descs);
  }

  std::vector<usm::pool_descriptor> poolDescriptors;
};

TEST_P(urUsmPoolManagerTest, poolManagerPopulate) {
  auto [ret, manager] = usm::pool_manager<usm::pool_descriptor>::create();
  ASSERT_EQ(ret, UR_RESULT_SUCCESS);

  for (auto &desc : poolDescriptors) {
    // Populate the pool manager
    auto poolUnique = createMockPoolHandle();
    ASSERT_NE(poolUnique, nullptr);
    ret = manager.addPool(desc, std::move(poolUnique));
    ASSERT_EQ(ret, UR_RESULT_SUCCESS);
  }

  for (auto &desc : poolDescriptors) {
    // Confirm that there is a pool for each descriptor
    auto hPoolOpt = manager.getPool(desc);
    ASSERT_TRUE(hPoolOpt.has_value());
    ASSERT_NE(hPoolOpt.value(), nullptr);
  }
}

TEST_P(urUsmPoolManagerTest, poolManagerInsertExisting) {
  auto [ret, manager] = usm::pool_manager<usm::pool_descriptor>::create();
  ASSERT_EQ(ret, UR_RESULT_SUCCESS);

  const auto &desc = poolDescriptors[0];

  auto poolUnique = createMockPoolHandle();
  ASSERT_NE(poolUnique, nullptr);

  ret = manager.addPool(desc, std::move(poolUnique));
  ASSERT_EQ(ret, UR_RESULT_SUCCESS);

  // Inserting an existing key should return an error
  ret = manager.addPool(desc, createMockPoolHandle());
  ASSERT_EQ(ret, UR_RESULT_ERROR_INVALID_ARGUMENT);
}

TEST_P(urUsmPoolManagerTest, poolManagerGetNonexistant) {
  auto [ret, manager] = usm::pool_manager<usm::pool_descriptor>::create();
  ASSERT_EQ(ret, UR_RESULT_SUCCESS);

  for (auto &desc : poolDescriptors) {
    auto hPool = manager.getPool(desc);
    ASSERT_FALSE(hPool.has_value());
  }
}

TEST_P(urUsmPoolManagerTest, config) {
  // Check default config
  usm::DisjointPoolAllConfigs def;
  usm::DisjointPoolAllConfigs parsed1 =
      usm::parseDisjointPoolConfig("1;host:2M,4,64K;device:4M,4,64K;"
                                   "shared:0,0,2M;read_only_shared:4M,4,2M",
                                   0);
  ASSERT_EQ(compareConfigs(def, parsed1), true);

  // Check partially set config
  usm::DisjointPoolAllConfigs part1 =
      usm::parseDisjointPoolConfig("1;device:4M;shared:0,0,2M", 0);
  ASSERT_EQ(compareConfigs(def, part1), true);

  // Check partially set config #2
  usm::DisjointPoolAllConfigs part2 =
      usm::parseDisjointPoolConfig(";device:4M;shared:0,0,2M", 0);
  ASSERT_EQ(compareConfigs(def, part2), true);

  // Check partially set config #3
  usm::DisjointPoolAllConfigs part3 =
      usm::parseDisjointPoolConfig(";shared:0,0,2M", 0);
  ASSERT_EQ(compareConfigs(def, part3), true);

  // Check partially set config #4
  usm::DisjointPoolAllConfigs part4 =
      usm::parseDisjointPoolConfig(";device:4M", 0);
  ASSERT_EQ(compareConfigs(def, part4), true);

  // Check partially set config #5
  usm::DisjointPoolAllConfigs part5 =
      usm::parseDisjointPoolConfig(";;device:4M,4,64K", 0);
  ASSERT_EQ(compareConfigs(def, part5), true);

  // Check non-default config
  usm::DisjointPoolAllConfigs test(def);
  test.Configs[usm::DisjointPoolMemType::Shared].MaxPoolableSize = 128 * 1024;
  test.Configs[usm::DisjointPoolMemType::Shared].Capacity = 4;
  test.Configs[usm::DisjointPoolMemType::Shared].SlabMinSize = 64 * 1024;

  usm::DisjointPoolAllConfigs parsed3 =
      usm::parseDisjointPoolConfig("1;shared:128K,4,64K", 0);
  ASSERT_EQ(compareConfigs(test, parsed3), true);
}

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urUsmPoolManagerTest);
