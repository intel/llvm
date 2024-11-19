// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <uur/fixtures.h>

struct urKernelSetArgLocalTest : uur::urKernelTest {
    void SetUp() {
        program_name = "mean";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelTest::SetUp());
    }
    size_t local_mem_size = 4 * sizeof(uint32_t);
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urKernelSetArgLocalTest);

TEST_P(urKernelSetArgLocalTest, Success) {
    ASSERT_SUCCESS(urKernelSetArgLocal(kernel, 1, local_mem_size, nullptr));
}

TEST_P(urKernelSetArgLocalTest, InvalidNullHandleKernel) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelSetArgLocal(nullptr, 1, local_mem_size, nullptr));
}

TEST_P(urKernelSetArgLocalTest, InvalidKernelArgumentIndex) {
    uint32_t num_kernel_args = 0;
    ASSERT_SUCCESS(urKernelGetInfo(kernel, UR_KERNEL_INFO_NUM_ARGS,
                                   sizeof(num_kernel_args), &num_kernel_args,
                                   nullptr));
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX,
                     urKernelSetArgLocal(kernel, num_kernel_args + 1,
                                         local_mem_size, nullptr));
}

// Test launching kernels with multiple local arguments return the expected
// outputs
struct urKernelSetArgLocalMultiTest : uur::urKernelExecutionTest {
    void SetUp() override {
        program_name = "saxpy_usm_local_mem";
        UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::SetUp());

        ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                         sizeof(backend), &backend, nullptr));

        // HIP has extra args for local memory so we define an offset for arg indices here for updating
        hip_arg_offset = backend == UR_PLATFORM_BACKEND_HIP ? 3 : 0;
        ur_device_usm_access_capability_flags_t shared_usm_flags;
        ASSERT_SUCCESS(
            uur::GetDeviceUSMSingleSharedSupport(device, shared_usm_flags));
        if (!(shared_usm_flags & UR_DEVICE_USM_ACCESS_CAPABILITY_FLAG_ACCESS)) {
            GTEST_SKIP() << "Shared USM is not supported.";
        }

        const size_t allocation_size =
            sizeof(uint32_t) * global_size * local_size;
        for (auto &shared_ptr : shared_ptrs) {
            ASSERT_SUCCESS(urUSMSharedAlloc(context, device, nullptr, nullptr,
                                            allocation_size, &shared_ptr));
            ASSERT_NE(shared_ptr, nullptr);

            std::vector<uint8_t> pattern(allocation_size);
            uur::generateMemFillPattern(pattern);
            std::memcpy(shared_ptr, pattern.data(), allocation_size);
        }
        size_t current_index = 0;
        // Index 0 is local_mem_a arg
        ASSERT_SUCCESS(urKernelSetArgLocal(kernel, current_index++,
                                           local_mem_a_size, nullptr));

        // Hip has extra args for local mem at index 1-3
        if (backend == UR_PLATFORM_BACKEND_HIP) {
            ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++,
                                               sizeof(local_size), nullptr,
                                               &local_size));
            ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++,
                                               sizeof(local_size), nullptr,
                                               &local_size));
            ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++,
                                               sizeof(local_size), nullptr,
                                               &local_size));
        }

        // Index 1 is local_mem_b arg
        ASSERT_SUCCESS(urKernelSetArgLocal(kernel, current_index++,
                                           local_mem_b_size, nullptr));
        if (backend == UR_PLATFORM_BACKEND_HIP) {
            ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++,
                                               sizeof(local_size), nullptr,
                                               &local_size));
            ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++,
                                               sizeof(local_size), nullptr,
                                               &local_size));
            ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++,
                                               sizeof(local_size), nullptr,
                                               &local_size));
        }

        // Index 2 is output
        ASSERT_SUCCESS(urKernelSetArgPointer(kernel, current_index++, nullptr,
                                             shared_ptrs[0]));
        // Index 3 is A
        ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++, sizeof(A),
                                           nullptr, &A));
        // Index 4 is X
        ASSERT_SUCCESS(urKernelSetArgPointer(kernel, current_index++, nullptr,
                                             shared_ptrs[1]));
        // Index 5 is Y
        ASSERT_SUCCESS(urKernelSetArgPointer(kernel, current_index++, nullptr,
                                             shared_ptrs[2]));
    }

    void Validate(uint32_t *output, uint32_t *X, uint32_t *Y, uint32_t A,
                  size_t length, size_t local_size) {
        for (size_t i = 0; i < length; i++) {
            uint32_t result = A * X[i] + Y[i] + local_size;
            ASSERT_EQ(result, output[i]);
        }
    }

    virtual void TearDown() override {
        for (auto &shared_ptr : shared_ptrs) {
            if (shared_ptr) {
                EXPECT_SUCCESS(urUSMFree(context, shared_ptr));
            }
        }

        UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::TearDown());
    }

    static constexpr size_t local_size = 4;
    static constexpr size_t local_mem_a_size = local_size * sizeof(uint32_t);
    static constexpr size_t local_mem_b_size = local_mem_a_size * 2;
    static constexpr size_t global_size = 16;
    static constexpr size_t global_offset = 0;
    static constexpr size_t n_dimensions = 1;
    static constexpr uint32_t A = 42;
    std::array<void *, 5> shared_ptrs = {nullptr, nullptr, nullptr, nullptr,
                                         nullptr};

    uint32_t hip_arg_offset = 0;
    ur_platform_backend_t backend{};
};
UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(urKernelSetArgLocalMultiTest);

TEST_P(urKernelSetArgLocalMultiTest, Basic) {
    ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                         &global_offset, &global_size,
                                         &local_size, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    uint32_t *output = (uint32_t *)shared_ptrs[0];
    uint32_t *X = (uint32_t *)shared_ptrs[1];
    uint32_t *Y = (uint32_t *)shared_ptrs[2];
    Validate(output, X, Y, A, global_size, local_size);
}

TEST_P(urKernelSetArgLocalMultiTest, ReLaunch) {
    ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                         &global_offset, &global_size,
                                         &local_size, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    uint32_t *output = (uint32_t *)shared_ptrs[0];
    uint32_t *X = (uint32_t *)shared_ptrs[1];
    uint32_t *Y = (uint32_t *)shared_ptrs[2];
    Validate(output, X, Y, A, global_size, local_size);

    // Relaunch with new arguments
    ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                         &global_offset, &global_size,
                                         &local_size, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));
    uint32_t *new_output = (uint32_t *)shared_ptrs[0];
    uint32_t *new_X = (uint32_t *)shared_ptrs[3];
    uint32_t *new_Y = (uint32_t *)shared_ptrs[4];
    Validate(new_output, new_X, new_Y, A, global_size, local_size);
}

// Overwrite local args to a larger value, then reset back to original
TEST_P(urKernelSetArgLocalMultiTest, Overwrite) {
    ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                         &global_offset, &global_size,
                                         &local_size, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    uint32_t *output = (uint32_t *)shared_ptrs[0];
    uint32_t *X = (uint32_t *)shared_ptrs[1];
    uint32_t *Y = (uint32_t *)shared_ptrs[2];
    Validate(output, X, Y, A, global_size, local_size);

    size_t new_local_size = 2;
    size_t new_local_mem_a_size = new_local_size * sizeof(uint32_t);
    size_t new_local_mem_b_size = new_local_size * sizeof(uint32_t) * 2;
    size_t current_index = 0;
    ASSERT_SUCCESS(urKernelSetArgLocal(kernel, current_index++,
                                       new_local_mem_a_size, nullptr));

    // Hip has extra args for local mem at index 1-3
    if (backend == UR_PLATFORM_BACKEND_HIP) {
        ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++,
                                           sizeof(new_local_size), nullptr,
                                           &new_local_size));
        ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++,
                                           sizeof(new_local_size), nullptr,
                                           &new_local_size));
        ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++,
                                           sizeof(new_local_size), nullptr,
                                           &new_local_size));
    }

    // Index 1 is local_mem_b arg
    ASSERT_SUCCESS(urKernelSetArgLocal(kernel, current_index++,
                                       new_local_mem_b_size, nullptr));
    if (backend == UR_PLATFORM_BACKEND_HIP) {
        ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++,
                                           sizeof(new_local_size), nullptr,
                                           &new_local_size));
        ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++,
                                           sizeof(new_local_size), nullptr,
                                           &new_local_size));
        ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_index++,
                                           sizeof(new_local_size), nullptr,
                                           &new_local_size));
    }

    ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, n_dimensions,
                                         &global_offset, &global_size,
                                         &new_local_size, 0, nullptr, nullptr));
    ASSERT_SUCCESS(urQueueFinish(queue));

    Validate(output, X, Y, A, global_size, new_local_size);
}
