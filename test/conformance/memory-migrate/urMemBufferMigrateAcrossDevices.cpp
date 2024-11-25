// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Some tests to ensure implicit memory migration of buffers across devices
// in the same context.

#include "uur/fixtures.h"

using T = uint32_t;

struct urMultiDeviceContextTest : uur::urPlatformTest {
    void SetUp() {
        uur::urPlatformTest::SetUp();
        ASSERT_SUCCESS(urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr,
                                   &num_devices));
        ASSERT_NE(num_devices, 0);
        if (num_devices == 1) {
            return;
        }

        devices = std::vector<ur_device_handle_t>(num_devices);
        ASSERT_SUCCESS(urDeviceGet(platform, UR_DEVICE_TYPE_ALL, num_devices,
                                   devices.data(), nullptr));
        ASSERT_SUCCESS(
            urContextCreate(num_devices, devices.data(), nullptr, &context));

        queues = std::vector<ur_queue_handle_t>(num_devices);
        for (auto i = 0u; i < num_devices; ++i) {
            ASSERT_SUCCESS(
                urQueueCreate(context, devices[i], nullptr, &queues[i]));
        }
    }

    void TearDown() {
        uur::urPlatformTest::TearDown();
        if (num_devices == 1) {
            return;
        }
        for (auto i = 0u; i < num_devices; ++i) {
            urDeviceRelease(devices[i]);
            urQueueRelease(queues[i]);
        }
        urContextRelease(context);
    }

    uint32_t num_devices = 0;
    ur_context_handle_t context;
    std::vector<ur_device_handle_t> devices;
    std::vector<ur_queue_handle_t> queues;
};

struct urMultiDeviceContextMemBufferTest : urMultiDeviceContextTest {
    void SetUp() {
        urMultiDeviceContextTest::SetUp();
        if (num_devices == 1) {
            return;
        }
        ASSERT_SUCCESS(urMemBufferCreate(context, 0 /*flags=*/,
                                         buffer_size_bytes,
                                         nullptr /*pProperties=*/, &buffer));

        UUR_RETURN_ON_FATAL_FAILURE(
            uur::KernelsEnvironment::instance->LoadSource(program_name,
                                                          il_binary));

        programs = std::vector<ur_program_handle_t>(num_devices);
        kernels = std::vector<ur_kernel_handle_t>(num_devices);

        const ur_program_properties_t properties = {
            UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES, nullptr,
            static_cast<uint32_t>(metadatas.size()),
            metadatas.empty() ? nullptr : metadatas.data()};
        for (auto i = 0u; i < num_devices; ++i) {
            ASSERT_SUCCESS(uur::KernelsEnvironment::instance->CreateProgram(
                platform, context, devices[i], *il_binary, &properties,
                &programs[i]));
            ASSERT_SUCCESS(urProgramBuild(context, programs[i], nullptr));
            auto kernel_names =
                uur::KernelsEnvironment::instance->GetEntryPointNames(
                    program_name);
            kernel_name = kernel_names[0];
            ASSERT_FALSE(kernel_name.empty());
            ASSERT_SUCCESS(
                urKernelCreate(programs[i], kernel_name.data(), &kernels[i]));
        }
    }

    // Adds a kernel arg representing a sycl buffer constructed with a 1D range.
    void AddBuffer1DArg(ur_kernel_handle_t kernel, size_t current_arg_index,
                        ur_mem_handle_t buffer) {
        ASSERT_SUCCESS(
            urKernelSetArgMemObj(kernel, current_arg_index, nullptr, buffer));

        // SYCL device kernels have different interfaces depending on the
        // backend being used. Typically a kernel which takes a buffer argument
        // will take a pointer to the start of the buffer and a sycl::id param
        // which is a struct that encodes the accessor to the buffer. However
        // the AMD backend handles this differently and uses three separate
        // arguments for each of the three dimensions of the accessor.

        ur_platform_backend_t backend;
        ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                         sizeof(backend), &backend, nullptr));
        if (backend == UR_PLATFORM_BACKEND_HIP) {
            // this emulates the three offset params for buffer accessor on AMD.
            size_t val = 0;
            ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_arg_index + 1,
                                               sizeof(size_t), nullptr, &val));
            ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_arg_index + 2,
                                               sizeof(size_t), nullptr, &val));
            ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_arg_index + 3,
                                               sizeof(size_t), nullptr, &val));
            current_arg_index += 4;
        } else {
            // This emulates the offset struct sycl adds for a 1D buffer accessor.
            struct {
                size_t offsets[1] = {0};
            } accessor;
            ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_arg_index + 1,
                                               sizeof(accessor), nullptr,
                                               &accessor));
            current_arg_index += 2;
        }
    }

    void TearDown() {
        if (num_devices > 1) {
            for (auto i = 0u; i < num_devices; ++i) {
                ASSERT_SUCCESS(urKernelRelease(kernels[i]));
                ASSERT_SUCCESS(urProgramRelease(programs[i]));
            }
            urMemRelease(buffer);
        }
        urMultiDeviceContextTest::TearDown();
    }

    size_t buffer_size = 4096;
    size_t buffer_size_bytes = 4096 * sizeof(T);
    ur_mem_handle_t buffer;

    // Program stuff so we can launch kernels
    std::shared_ptr<std::vector<char>> il_binary;
    std::string program_name = "inc";
    std::string kernel_name;
    std::vector<ur_program_handle_t> programs;
    std::vector<ur_kernel_handle_t> kernels;
    std::vector<ur_program_metadata_t> metadatas{};
};
UUR_INSTANTIATE_PLATFORM_TEST_SUITE_P(urMultiDeviceContextMemBufferTest);

TEST_P(urMultiDeviceContextMemBufferTest, WriteRead) {
    if (num_devices == 1) {
        GTEST_SKIP();
    }
    T fill_val = 42;
    std::vector<T> in_vec(buffer_size, fill_val);
    std::vector<T> out_vec(buffer_size, 0);
    ur_event_handle_t e1;

    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queues[0], buffer, false, 0,
                                           buffer_size_bytes, in_vec.data(), 0,
                                           nullptr, &e1));

    ASSERT_SUCCESS(urEnqueueMemBufferRead(queues[1], buffer, false, 0,
                                          buffer_size_bytes, out_vec.data(), 1,
                                          &e1, nullptr));

    ASSERT_SUCCESS(urQueueFinish(queues[1]));

    for (auto &a : out_vec) {
        ASSERT_EQ(a, fill_val);
    }
}

TEST_P(urMultiDeviceContextMemBufferTest, FillRead) {
    if (num_devices == 1) {
        GTEST_SKIP();
    }
    T fill_val = 42;
    std::vector<T> in_vec(buffer_size, fill_val);
    std::vector<T> out_vec(buffer_size);
    ur_event_handle_t e1;

    ASSERT_SUCCESS(urEnqueueMemBufferFill(queues[0], buffer, &fill_val,
                                          sizeof(fill_val), 0,
                                          buffer_size_bytes, 0, nullptr, &e1));

    ASSERT_SUCCESS(urEnqueueMemBufferRead(queues[1], buffer, false, 0,
                                          buffer_size_bytes, out_vec.data(), 1,
                                          &e1, nullptr));

    ASSERT_SUCCESS(urQueueFinish(queues[1]));

    for (auto &a : out_vec) {
        ASSERT_EQ(a, fill_val);
    }
}

TEST_P(urMultiDeviceContextMemBufferTest, WriteKernelRead) {
    if (num_devices == 1) {
        GTEST_SKIP();
    }

    // Kernel to run on queues[1]
    AddBuffer1DArg(kernels[1], 0, buffer);

    T fill_val = 42;
    std::vector<T> in_vec(buffer_size, fill_val);
    std::vector<T> out_vec(buffer_size);
    ur_event_handle_t e1, e2;

    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queues[0], buffer, false, 0,
                                           buffer_size_bytes, in_vec.data(), 0,
                                           nullptr, &e1));

    size_t work_dims[3] = {buffer_size, 1, 1};
    size_t offset[3] = {0, 0, 0};

    // Kernel increments the fill val by 1
    ASSERT_SUCCESS(urEnqueueKernelLaunch(queues[1], kernels[1], 1 /*workDim=*/,
                                         offset, work_dims, nullptr, 1, &e1,
                                         &e2));

    ASSERT_SUCCESS(urEnqueueMemBufferRead(queues[0], buffer, false, 0,
                                          buffer_size_bytes, out_vec.data(), 1,
                                          &e2, nullptr));

    ASSERT_SUCCESS(urQueueFinish(queues[0]));

    for (auto &a : out_vec) {
        ASSERT_EQ(a, fill_val + 1);
    }
}

TEST_P(urMultiDeviceContextMemBufferTest, WriteKernelKernelRead) {
    if (num_devices == 1) {
        GTEST_SKIP();
    }

    AddBuffer1DArg(kernels[0], 0, buffer);
    AddBuffer1DArg(kernels[1], 0, buffer);

    T fill_val = 42;
    std::vector<T> in_vec(buffer_size, fill_val);
    std::vector<T> out_vec(buffer_size);
    ur_event_handle_t e1, e2, e3;

    ASSERT_SUCCESS(urEnqueueMemBufferWrite(queues[0], buffer, false, 0,
                                           buffer_size_bytes, in_vec.data(), 0,
                                           nullptr, &e1));

    size_t work_dims[3] = {buffer_size, 1, 1};
    size_t offset[3] = {0, 0, 0};

    // Kernel increments the fill val by 1
    ASSERT_SUCCESS(urEnqueueKernelLaunch(queues[1], kernels[1], 1 /*workDim=*/,
                                         offset, work_dims, nullptr, 1, &e1,
                                         &e2));

    // Kernel increments the fill val by 1
    ASSERT_SUCCESS(urEnqueueKernelLaunch(queues[0], kernels[0], 1 /*workDim=*/,
                                         offset, work_dims, nullptr, 1, &e2,
                                         &e3));

    ASSERT_SUCCESS(urEnqueueMemBufferRead(queues[1], buffer, false, 0,
                                          buffer_size_bytes, out_vec.data(), 1,
                                          &e3, nullptr));

    ASSERT_SUCCESS(urQueueFinish(queues[1]));

    for (auto &a : out_vec) {
        ASSERT_EQ(a, fill_val + 2);
    }
}
