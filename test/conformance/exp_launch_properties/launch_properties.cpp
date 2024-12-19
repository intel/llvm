// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>

struct urEnqueueKernelLaunchCustomTest : uur::urKernelExecutionTest {
    void SetUp() override {
        program_name = "fill";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());
    }

    uint32_t val = 42;
    size_t global_size = 32;
    size_t global_offset = 0;
    size_t n_dimensions = 1;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueKernelLaunchCustomTest);

TEST_P(urEnqueueKernelLaunchCustomTest, Success) {

    size_t returned_size;
    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_EXTENSIONS, 0,
                                   nullptr, &returned_size));

    std::unique_ptr<char[]> returned_extensions(new char[returned_size]);

    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_EXTENSIONS,
                                   returned_size, returned_extensions.get(),
                                   nullptr));

    std::string_view extensions_string(returned_extensions.get());
    const bool launch_properties_support =
        extensions_string.find(UR_LAUNCH_PROPERTIES_EXTENSION_STRING_EXP) !=
        std::string::npos;

    if (!launch_properties_support) {
        GTEST_SKIP() << "EXP launch properties feature is not supported.";
    }

    std::vector<ur_exp_launch_property_t> props(1);
    props[0].id = UR_EXP_LAUNCH_PROPERTY_ID_IGNORE;

    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_PROFILE, 0, nullptr,
                                   &returned_size));

    std::unique_ptr<char[]> returned_backend(new char[returned_size]);

    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_PROFILE,
                                   returned_size, returned_backend.get(),
                                   nullptr));

    std::string_view backend_string(returned_backend.get());
    const bool cuda_backend = backend_string.find("CUDA") != std::string::npos;

    if (cuda_backend) {
        ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_VERSION, 0,
                                       nullptr, &returned_size));

        std::unique_ptr<char[]> returned_compute_capability(
            new char[returned_size]);

        ASSERT_SUCCESS(
            urDeviceGetInfo(device, UR_DEVICE_INFO_VERSION, returned_size,
                            returned_compute_capability.get(), nullptr));

        auto compute_capability =
            std::stof(std::string(returned_compute_capability.get()));

        if (compute_capability >= 6.0) {
            ur_exp_launch_property_t coop_prop;
            coop_prop.id = UR_EXP_LAUNCH_PROPERTY_ID_COOPERATIVE;
            coop_prop.value.cooperative = 1;
            props.push_back(coop_prop);
        }

        ur_bool_t cluster_launch_supported = false;
        ASSERT_SUCCESS(urDeviceGetInfo(
            device, UR_DEVICE_INFO_CLUSTER_LAUNCH_EXP, sizeof(ur_bool_t),
            &cluster_launch_supported, nullptr));

        if (cluster_launch_supported) {
            ur_exp_launch_property_t cluster_dims_prop;
            cluster_dims_prop.id = UR_EXP_LAUNCH_PROPERTY_ID_CLUSTER_DIMENSION;
            cluster_dims_prop.value.clusterDim[0] = 16;
            cluster_dims_prop.value.clusterDim[1] = 1;
            cluster_dims_prop.value.clusterDim[2] = 1;

            props.push_back(cluster_dims_prop);
        }
    }
    ur_mem_handle_t buffer = nullptr;
    AddBuffer1DArg(sizeof(val) * global_size, &buffer);
    AddPodArg(val);

    ASSERT_SUCCESS(urEnqueueKernelLaunchCustomExp(
        queue, kernel, n_dimensions, &global_offset, &global_size, nullptr, 1,
        &props[0], 0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
    ValidateBuffer(buffer, sizeof(val) * global_size, val);
}
