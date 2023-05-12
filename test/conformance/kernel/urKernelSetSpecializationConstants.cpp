// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

struct urKernelSetSpecializationConstantsTest : uur::urKernelTest {
    void SetUp() override {
        bool supports_kernel_spec_constant = false;
        ASSERT_SUCCESS(urDeviceGetInfo(
            device, UR_DEVICE_INFO_KERNEL_SET_SPECIALIZATION_CONSTANTS,
            sizeof(supports_kernel_spec_constant),
            &supports_kernel_spec_constant, nullptr));
        if (!supports_kernel_spec_constant) {
            GTEST_SKIP()
                << "Device does not support setting kernel spec constants.";
        }
        program_name = "spec_constant";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::SetUp());
    }

    uint32_t spec_value = 42;
    ur_specialization_constant_info_t info = {0, sizeof(spec_value),
                                              &spec_value};
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urKernelSetSpecializationConstantsTest);

TEST_P(urKernelSetSpecializationConstantsTest, Success) {
    ASSERT_SUCCESS(urKernelSetSpecializationConstants(kernel, 1, &info));
    // TODO: Run the kernel to verify the spec constant was set.
}

TEST_P(urKernelSetSpecializationConstantsTest, InvalidNullHandleKernel) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelSetSpecializationConstants(nullptr, 1, &info));
}

TEST_P(urKernelSetSpecializationConstantsTest,
       InvalidNullPointerSpecConstants) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urKernelSetSpecializationConstants(kernel, 1, nullptr));
}

TEST_P(urKernelSetSpecializationConstantsTest, InvalidSizeCount) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urKernelSetSpecializationConstants(kernel, 0, &info));
}
