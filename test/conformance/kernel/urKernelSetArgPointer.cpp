// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/fixtures.h>

struct urKernelSetArgPointerTest : uur::urKernelTest {
    void SetUp() {
        program_name = "fill";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::SetUp());
    }

    void TearDown() {
        if (allocation) {
            ASSERT_SUCCESS(urUSMFree(context, allocation));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::TearDown());
    }

    void *allocation = nullptr;
    size_t allocation_size = 16 * sizeof(uint32_t);
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urKernelSetArgPointerTest);

TEST_P(urKernelSetArgPointerTest, SuccessHost) {
    bool host_supported = false;
    ASSERT_SUCCESS(uur::GetDeviceUSMHostSupport(device, host_supported));
    if (!host_supported) {
        GTEST_SKIP() << "Host USM is not supported.";
    }

    ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, nullptr, allocation_size,
                                  &allocation));
    ASSERT_NE(allocation, nullptr);

    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, allocation));
}

TEST_P(urKernelSetArgPointerTest, SuccessDevice) {
    bool device_supported = false;
    ASSERT_SUCCESS(uur::GetDeviceUSMDeviceSupport(device, device_supported));
    if (!device_supported) {
        GTEST_SKIP() << "Host USM is not supported.";
    }

    ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                    allocation_size, &allocation));
    ASSERT_NE(allocation, nullptr);

    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, allocation));
}

TEST_P(urKernelSetArgPointerTest, SuccessShared) {
    bool shared_supported = false;
    ASSERT_SUCCESS(
        uur::GetDeviceUSMSingleSharedSupport(device, shared_supported));
    if (!shared_supported) {
        GTEST_SKIP() << "Shared USM is not supported.";
    }

    ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                    allocation_size, &allocation));
    ASSERT_NE(allocation, nullptr);

    ASSERT_SUCCESS(urKernelSetArgPointer(kernel, 0, allocation));
}

struct urKernelSetArgPointerNegativeTest : urKernelSetArgPointerTest {
    // Get any valid allocation we can to test validation of the other parameters.
    void SetUpAllocation() {
        bool host_supported = false;
        ASSERT_SUCCESS(uur::GetDeviceUSMHostSupport(device, host_supported));
        if (host_supported) {
            ASSERT_SUCCESS(urUSMHostAlloc(context, nullptr, nullptr,
                                          allocation_size, &allocation));
            return;
        }

        bool device_supported = false;
        ASSERT_SUCCESS(
            uur::GetDeviceUSMDeviceSupport(device, device_supported));
        if (device_supported) {
            ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                            allocation_size, &allocation));
            return;
        }

        bool shared_supported = false;
        ASSERT_SUCCESS(
            uur::GetDeviceUSMSingleSharedSupport(device, shared_supported));
        if (shared_supported) {
            ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                            allocation_size, &allocation));
            return;
        }

        if (!(host_supported || device_supported || shared_supported)) {
            GTEST_SKIP() << "USM is not supported.";
        }
    }

    void SetUp() {
        SetUpAllocation();
        ASSERT_NE(allocation, nullptr);
        UUR_RETURN_ON_FATAL_FAILURE(urKernelSetArgPointerTest::SetUp());
    }
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urKernelSetArgPointerNegativeTest);

TEST_P(urKernelSetArgPointerNegativeTest, InvalidNullHandleKernel) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelSetArgPointer(nullptr, 0, allocation));
}

TEST_P(urKernelSetArgPointerNegativeTest, InvalidKernelArgumentIndex) {
    size_t num_kernel_args = 0;
    ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS,
                                   sizeof(num_kernel_args), &num_kernel_args,
                                   nullptr));

    ASSERT_EQ_RESULT(
        UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX,
        urKernelSetArgPointer(kernel, num_kernel_args + 1, allocation));
}
