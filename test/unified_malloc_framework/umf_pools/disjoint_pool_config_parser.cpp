// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

#include "disjoint_pool_config_parser.hpp"

class disjointPoolConfigTests : public testing::TestWithParam<std::string> {
  protected:
    const int hostMaxPoolableSizeDefault = 2 * 1024 * 1024;
    const int hostCapacityDefault = 4;
    const int hostSlabMinSizeDefault = 64 * 1024;

    const int deviceMaxPoolableSizeDefault = 4 * 1024 * 1024;
    const int deviceCapacityDefault = 4;
    const int deviceSlabMinSizeDefault = 64 * 1024;

    const int sharedMaxPoolableSizeDefault = 0;
    const int sharedCapacityDefault = 0;
    const int sharedSlabMinSizeDefault = 2 * 1024 * 1024;

    const int readOnlySharedMaxPoolableSizeDefault = 4 * 1024 * 1024;
    const int readOnlySharedCapacityDefault = 4;
    const int readOnlySharedSlabMinSizeDefault = 2 * 1024 * 1024;

    const int maxSizeDefault = 16 * 1024 * 1024;
};

// // invalid configs for which descriptors' parameters should be set to default values
// INSTANTIATE_TEST_SUITE_P(
//     configsForDefaultValues, disjointPoolConfigTests,
//     testing::Values("", "ab12cdefghi34jk56lmn78opr910",
//                     "1;32M;foo:0,3,2m;device:1M,4,64k",
//                     "132M;host:4m,3,2m,4m;device:1M,4,64k",
//                     "1;32M;abdc123;;;device:1M,4,64k",
//                     "1;32M;host:0,3,2m,4;device:1M,4,64k,5;host:1,8,4m,100",
//                     "32M;1;host:1M,4,64k;device:1m,4,64K;shared:0,3,1M"));

TEST_F(disjointPoolConfigTests, disjointPoolConfigStringEnabledBuffersTest) {
    // test for valid string with enabled buffers-- (values to configs should be
    // parsed from string)
    std::string config = "1;32M;host:1M,3,32k;device:1m,2,32m;shared:2m,3,1M;"
                         "read_only_shared:0,0,3M";
    auto allConfigs = usm::parseDisjointPoolConfig(config);

    // test for host
    ASSERT_EQ(
        allConfigs.Configs[usm::DisjointPoolMemType::Host].MaxPoolableSize,
        1 * 1024 * 1024);
    ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Host].Capacity, 3);
    ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Host].SlabMinSize,
              32 * 1024);
    ASSERT_EQ(
        allConfigs.Configs[usm::DisjointPoolMemType::Host].limits->MaxSize,
        32 * 1024 * 1024);

    // test for device
    ASSERT_EQ(
        allConfigs.Configs[usm::DisjointPoolMemType::Device].MaxPoolableSize,
        1 * 1024 * 1024);
    ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Device].Capacity, 2);
    ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Device].SlabMinSize,
              32 * 1024 * 1024);
    ASSERT_EQ(
        allConfigs.Configs[usm::DisjointPoolMemType::Device].limits->MaxSize,
        32 * 1024 * 1024);

    // test for shared
    ASSERT_EQ(
        allConfigs.Configs[usm::DisjointPoolMemType::Shared].MaxPoolableSize,
        2 * 1024 * 1024);
    ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Shared].Capacity, 3);
    ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Shared].SlabMinSize,
              1 * 1024 * 1024);
    ASSERT_EQ(
        allConfigs.Configs[usm::DisjointPoolMemType::Shared].limits->MaxSize,
        32 * 1024 * 1024);

    // test for read-only shared
    ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly]
                  .MaxPoolableSize,
              0);
    ASSERT_EQ(
        allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly].Capacity,
        0);
    ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly]
                  .SlabMinSize,
              3 * 1024 * 1024);
    ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly]
                  .limits->MaxSize,
              32 * 1024 * 1024);
}

TEST_F(disjointPoolConfigTests,
       disjointPoolConfigStringImpartialEnabledBuffersTest) {
    // test for valid impartial string with enabled buffers-- (values to configs should be
    // parsed from string if present, the rest should be default
    // set by getConfigurationsFor)
    std::string config = "1;32M;host:1M,2,16k;shared:64k,3,1M;";
    auto allConfigs = usm::parseDisjointPoolConfig(config);

    // test for host
    ASSERT_EQ(
        allConfigs.Configs[usm::DisjointPoolMemType::Host].MaxPoolableSize,
        1 * 1024 * 1024);
    ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Host].Capacity, 2);
    ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Host].SlabMinSize,
              16 * 1024);
    ASSERT_EQ(
        allConfigs.Configs[usm::DisjointPoolMemType::Host].limits->MaxSize,
        32 * 1024 * 1024);

    // test for device
    ASSERT_EQ(
        allConfigs.Configs[usm::DisjointPoolMemType::Device].MaxPoolableSize,
        deviceMaxPoolableSizeDefault);
    ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Device].Capacity,
              deviceCapacityDefault);
    ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Device].SlabMinSize,
              deviceSlabMinSizeDefault);
    ASSERT_EQ(
        allConfigs.Configs[usm::DisjointPoolMemType::Device].limits->MaxSize,
        32 * 1024 * 1024);

    // test for shared
    ASSERT_EQ(
        allConfigs.Configs[usm::DisjointPoolMemType::Shared].MaxPoolableSize,
        64 * 1024);
    ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Shared].Capacity, 3);
    ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Shared].SlabMinSize,
              1 * 1024 * 1024);
    ASSERT_EQ(
        allConfigs.Configs[usm::DisjointPoolMemType::Shared].limits->MaxSize,
        32 * 1024 * 1024);

    // test for read-only shared
    ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly]
                  .MaxPoolableSize,
              readOnlySharedMaxPoolableSizeDefault);
    ASSERT_EQ(
        allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly].Capacity,
        readOnlySharedCapacityDefault);
    ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly]
                  .SlabMinSize,
              readOnlySharedSlabMinSizeDefault);
    ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly]
                  .limits->MaxSize,
              32 * 1024 * 1024);
}

// TODO: fix config parsing
// TEST_P(disjointPoolConfigTests, disjointPoolConfigInvalid) {
//     std::string config = GetParam();
//     auto allConfigs = usm::parseDisjointPoolConfig(config);

//     // test for host
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::Host].MaxPoolableSize,
//         hostMaxPoolableSizeDefault);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Host].Capacity,
//               hostCapacityDefault);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Host].SlabMinSize,
//               hostSlabMinSizeDefault);
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::Host].limits->MaxSize,
//         maxSizeDefault);

//     // test for device
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::Device].MaxPoolableSize,
//         deviceMaxPoolableSizeDefault);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Device].Capacity,
//               deviceCapacityDefault);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Device].SlabMinSize,
//               deviceSlabMinSizeDefault);
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::Device].limits->MaxSize,
//         maxSizeDefault);

//     // test for shared
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::Shared].MaxPoolableSize,
//         sharedMaxPoolableSizeDefault);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Shared].Capacity,
//               sharedCapacityDefault);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Shared].SlabMinSize,
//               sharedSlabMinSizeDefault);
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::Shared].limits->MaxSize,
//         maxSizeDefault);

//     // test for read-only shared
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly]
//                   .MaxPoolableSize,
//               readOnlySharedMaxPoolableSizeDefault);
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly].Capacity,
//         readOnlySharedCapacityDefault);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly]
//                   .SlabMinSize,
//               readOnlySharedSlabMinSizeDefault);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly]
//                   .limits->MaxSize,
//               maxSizeDefault);
// }

// TEST_F(disjointPoolConfigTests, disjointPoolConfigStringInvalidEnabledBuffers) {
//     // test for a string with invalid parameter for EnabledBuffers--
//     // (it should be set to a default)
//     std::string config = "-5;32M;host:0,3,32k,4m;device:1M,2,32m";
//     auto allConfigs = usm::parseDisjointPoolConfig(config);

//     // test for host
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::Host].MaxPoolableSize, 0);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Host].Capacity, 3);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Host].SlabMinSize,
//               32 * 1024);
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::Host].limits->MaxSize,
//         32 * 1024 * 1024);

//     // test for device
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::Device].MaxPoolableSize,
//         1 * 1024 * 1024);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Device].Capacity, 2);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Device].SlabMinSize,
//               32 * 1024 * 1024);
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::Device].limits->MaxSize,
//         32 * 1024 * 1024);

//     // test for shared
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::Shared].MaxPoolableSize,
//         sharedMaxPoolableSizeDefault);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Shared].Capacity,
//               sharedCapacityDefault);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Shared].SlabMinSize,
//               sharedSlabMinSizeDefault);
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::Shared].limits->MaxSize,
//         32 * 1024 * 1024);

//     // test for read-only shared
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly]
//                   .MaxPoolableSize,
//               readOnlySharedMaxPoolableSizeDefault);
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly].Capacity,
//         readOnlySharedCapacityDefault);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly]
//                   .SlabMinSize,
//               readOnlySharedSlabMinSizeDefault);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly]
//                   .limits->MaxSize,
//               32 * 1024 * 1024);
// }

// TEST_F(disjointPoolConfigTests, disjointPoolConfigStringTooManyParameters) {
//     // test for when too many parameters are passed-- (the extra parameters
//     // should be ignored)
//     std::string config = "1;32M;host:0,3,2m,4;device:1M,5,32k,5";
//     auto allConfigs = usm::parseDisjointPoolConfig(config);

//     // test for host
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::Host].MaxPoolableSize, 0);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Host].Capacity, 3);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Host].SlabMinSize,
//               2 * 1024 * 1024);
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::Host].limits->MaxSize,
//         32 * 1024 * 1024);

//     // test for device
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::Device].MaxPoolableSize,
//         1 * 1024 * 1024);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Device].Capacity, 5);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Device].SlabMinSize,
//               32 * 1024);
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::Device].limits->MaxSize,
//         32 * 1024 * 1024);

//     // test for shared
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::Shared].MaxPoolableSize,
//         sharedMaxPoolableSizeDefault);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Shared].Capacity,
//               sharedCapacityDefault);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::Shared].SlabMinSize,
//               sharedSlabMinSizeDefault);
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::Shared].limits->MaxSize,
//         32 * 1024 * 1024);

//     // test for read-only shared
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly]
//                   .MaxPoolableSize,
//               readOnlySharedMaxPoolableSizeDefault);
//     ASSERT_EQ(
//         allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly].Capacity,
//         readOnlySharedCapacityDefault);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly]
//                   .SlabMinSize,
//               readOnlySharedSlabMinSizeDefault);
//     ASSERT_EQ(allConfigs.Configs[usm::DisjointPoolMemType::SharedReadOnly]
//                   .limits->MaxSize,
//               32 * 1024 * 1024);
// }
