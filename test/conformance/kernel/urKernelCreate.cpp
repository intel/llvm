// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/fixtures.h>
#include <uur/raii.h>

struct urKernelCreateTest : uur::urProgramTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
        ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
        auto kernel_names =
            uur::KernelsEnvironment::instance->GetEntryPointNames(
                this->program_name);
        kernel_name = kernel_names[0];
    }

    void TearDown() override {
        if (kernel) {
            ASSERT_SUCCESS(urKernelRelease(kernel));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::TearDown());
    }

    std::string kernel_name;
    ur_kernel_handle_t kernel = nullptr;
};
UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urKernelCreateTest);

TEST_P(urKernelCreateTest, Success) {
    ASSERT_SUCCESS(urKernelCreate(program, kernel_name.data(), &kernel));
}

TEST_P(urKernelCreateTest, InvalidNullHandleProgram) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urKernelCreate(nullptr, kernel_name.data(), &kernel));
}

TEST_P(urKernelCreateTest, InvalidNullPointerName) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urKernelCreate(program, nullptr, &kernel));
}

TEST_P(urKernelCreateTest, InvalidNullPointerKernel) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_POINTER,
                     urKernelCreate(program, kernel_name.data(), nullptr));
}

TEST_P(urKernelCreateTest, InvalidKernelName) {
    std::string invalid_name = "incorrect_kernel_name";
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_KERNEL_NAME,
                     urKernelCreate(program, invalid_name.data(), &kernel));
}

using urMultiDeviceKernelCreateTest = uur::urMultiDeviceQueueTest;
UUR_INSTANTIATE_PLATFORM_TEST_SUITE_P(urMultiDeviceKernelCreateTest);

TEST_P(urMultiDeviceKernelCreateTest, WithProgramBuild) {
    constexpr size_t global_offset = 0;
    constexpr size_t n_dimensions = 1;
    constexpr size_t global_size = 100;
    constexpr size_t local_size = 100;

    auto kernelName =
        uur::KernelsEnvironment::instance->GetEntryPointNames("foo")[0];

    std::shared_ptr<std::vector<char>> il_binary;
    uur::KernelsEnvironment::instance->LoadSource("foo", platform, il_binary);

    for (size_t i = 0; i < devices.size(); i++) {
        uur::raii::Program program;
        uur::raii::Kernel kernel;

        const ur_program_properties_t properties = {
            UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES, nullptr, 0, nullptr};
        ASSERT_SUCCESS(uur::KernelsEnvironment::instance->CreateProgram(
            platform, context, devices[i], *il_binary, &properties,
            program.ptr()));

        ASSERT_SUCCESS(urProgramBuild(context, program.get(), nullptr));
        ASSERT_SUCCESS(
            urKernelCreate(program.get(), kernelName.data(), kernel.ptr()));

        ASSERT_SUCCESS(urEnqueueKernelLaunch(
            queues[i], kernel.get(), n_dimensions, &global_offset, &local_size,
            &global_size, 0, nullptr, nullptr));

        ASSERT_SUCCESS(urQueueFinish(queues[i]));
    }
}

TEST_P(urMultiDeviceKernelCreateTest, WithProgramCompileAndLink) {
    constexpr size_t global_offset = 0;
    constexpr size_t n_dimensions = 1;
    constexpr size_t global_size = 100;
    constexpr size_t local_size = 100;

    auto kernelName =
        uur::KernelsEnvironment::instance->GetEntryPointNames("foo")[0];

    std::shared_ptr<std::vector<char>> il_binary;
    uur::KernelsEnvironment::instance->LoadSource("foo", platform, il_binary);

    for (size_t i = 0; i < devices.size(); i++) {
        uur::raii::Program program;
        uur::raii::Kernel kernel;

        const ur_program_properties_t properties = {
            UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES, nullptr, 0, nullptr};
        ASSERT_SUCCESS(uur::KernelsEnvironment::instance->CreateProgram(
            platform, context, devices[i], *il_binary, &properties,
            program.ptr()));

        ASSERT_SUCCESS(urProgramCompile(context, program.get(), nullptr));

        uur::raii::Program linked_program;
        ASSERT_EQ_RESULT(UR_RESULT_SUCCESS,
                         urProgramLink(context, 1, program.ptr(), nullptr,
                                       linked_program.ptr()));

        ASSERT_SUCCESS(urKernelCreate(linked_program.get(), kernelName.data(),
                                      kernel.ptr()));

        ASSERT_SUCCESS(urEnqueueKernelLaunch(
            queues[i], kernel.get(), n_dimensions, &global_offset, &local_size,
            &global_size, 0, nullptr, nullptr));

        ASSERT_SUCCESS(urQueueFinish(queues[i]));
    }
}
