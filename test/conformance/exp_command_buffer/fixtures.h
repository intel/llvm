// Copyright (C) 2022-2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_CONFORMANCE_COMMAND_BUFFER_FIXTURES_H_INCLUDED
#define UR_CONFORMANCE_COMMAND_BUFFER_FIXTURES_H_INCLUDED

#include <array>
#include <uur/fixtures.h>

namespace uur {
namespace command_buffer {

static void checkCommandBufferSupport(ur_device_handle_t device) {
    size_t returned_size;
    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_EXTENSIONS, 0,
                                   nullptr, &returned_size));

    std::unique_ptr<char[]> returned_extensions(new char[returned_size]);

    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_EXTENSIONS,
                                   returned_size, returned_extensions.get(),
                                   nullptr));

    std::string_view extensions_string(returned_extensions.get());
    bool command_buffer_support =
        extensions_string.find(UR_COMMAND_BUFFER_EXTENSION_STRING_EXP) !=
        std::string::npos;

    if (!command_buffer_support) {
        GTEST_SKIP() << "EXP command-buffer feature is not supported.";
    }
}

static void checkCommandBufferUpdateSupport(
    ur_device_handle_t device,
    ur_device_command_buffer_update_capability_flags_t required_capabilities) {
    ur_device_command_buffer_update_capability_flags_t update_capability_flags;
    ASSERT_SUCCESS(urDeviceGetInfo(
        device, UR_DEVICE_INFO_COMMAND_BUFFER_UPDATE_CAPABILITIES_EXP,
        sizeof(update_capability_flags), &update_capability_flags, nullptr));

    if (!update_capability_flags) {
        GTEST_SKIP() << "Updating EXP command-buffers is not supported.";
    } else if ((update_capability_flags & required_capabilities) !=
               required_capabilities) {
        GTEST_SKIP() << "Some of the command-buffer update capabilities "
                        "required are not supported by the device.";
    }
}

struct urCommandBufferExpTest : uur::urContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urContextTest::SetUp());

        UUR_RETURN_ON_FATAL_FAILURE(checkCommandBufferSupport(device));
        ASSERT_SUCCESS(urCommandBufferCreateExp(context, device, nullptr,
                                                &cmd_buf_handle));
        ASSERT_NE(cmd_buf_handle, nullptr);
    }

    void TearDown() override {
        if (cmd_buf_handle) {
            EXPECT_SUCCESS(urCommandBufferReleaseExp(cmd_buf_handle));
        }
        UUR_RETURN_ON_FATAL_FAILURE(uur::urContextTest::TearDown());
    }

    ur_exp_command_buffer_handle_t cmd_buf_handle = nullptr;
};

template <class T>
struct urCommandBufferExpTestWithParam : urQueueTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTestWithParam<T>::SetUp());

        UUR_RETURN_ON_FATAL_FAILURE(checkCommandBufferSupport(this->device));
        ASSERT_SUCCESS(urCommandBufferCreateExp(this->context, this->device,
                                                nullptr, &cmd_buf_handle));
        ASSERT_NE(cmd_buf_handle, nullptr);
    }

    void TearDown() override {
        if (cmd_buf_handle) {
            EXPECT_SUCCESS(urCommandBufferReleaseExp(cmd_buf_handle));
        }
        UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTestWithParam<T>::TearDown());
    }

    ur_exp_command_buffer_handle_t cmd_buf_handle = nullptr;
};

struct urCommandBufferExpExecutionTest : uur::urKernelExecutionTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urKernelExecutionTest::SetUp());

        UUR_RETURN_ON_FATAL_FAILURE(checkCommandBufferSupport(device));
        ASSERT_SUCCESS(urCommandBufferCreateExp(context, device, nullptr,
                                                &cmd_buf_handle));
        ASSERT_NE(cmd_buf_handle, nullptr);
    }

    void TearDown() override {
        if (cmd_buf_handle) {
            EXPECT_SUCCESS(urCommandBufferReleaseExp(cmd_buf_handle));
        }
        UUR_RETURN_ON_FATAL_FAILURE(uur::urKernelExecutionTest::TearDown());
    }

    ur_exp_command_buffer_handle_t cmd_buf_handle = nullptr;
};

struct urUpdatableCommandBufferExpTest : uur::urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::SetUp());

        UUR_RETURN_ON_FATAL_FAILURE(checkCommandBufferSupport(device));

        auto required_capabilities =
            UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_KERNEL_ARGUMENTS |
            UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_LOCAL_WORK_SIZE |
            UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_SIZE |
            UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_OFFSET;
        UUR_RETURN_ON_FATAL_FAILURE(
            checkCommandBufferUpdateSupport(device, required_capabilities));

        // Create a command-buffer with update enabled.
        ur_exp_command_buffer_desc_t desc{
            UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC, nullptr, true, false,
            false};

        ASSERT_SUCCESS(urCommandBufferCreateExp(context, device, &desc,
                                                &updatable_cmd_buf_handle));
        ASSERT_NE(updatable_cmd_buf_handle, nullptr);
    }

    void TearDown() override {
        if (updatable_cmd_buf_handle) {
            EXPECT_SUCCESS(urCommandBufferReleaseExp(updatable_cmd_buf_handle));
        }
        UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::TearDown());
    }

    ur_exp_command_buffer_handle_t updatable_cmd_buf_handle = nullptr;
    ur_platform_backend_t backend{};
};

struct urUpdatableCommandBufferExpExecutionTest : uur::urKernelExecutionTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urKernelExecutionTest::SetUp());

        ASSERT_SUCCESS(urPlatformGetInfo(platform, UR_PLATFORM_INFO_BACKEND,
                                         sizeof(backend), &backend, nullptr));

        UUR_RETURN_ON_FATAL_FAILURE(checkCommandBufferSupport(device));
        auto required_capabilities =
            UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_KERNEL_ARGUMENTS |
            UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_LOCAL_WORK_SIZE |
            UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_SIZE |
            UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_GLOBAL_WORK_OFFSET;
        UUR_RETURN_ON_FATAL_FAILURE(
            checkCommandBufferUpdateSupport(device, required_capabilities));

        // Create a command-buffer with update enabled.
        ur_exp_command_buffer_desc_t desc{
            UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC, nullptr, true, false,
            false};

        ASSERT_SUCCESS(urCommandBufferCreateExp(context, device, &desc,
                                                &updatable_cmd_buf_handle));
        ASSERT_NE(updatable_cmd_buf_handle, nullptr);
    }

    void TearDown() override {
        if (updatable_cmd_buf_handle) {
            EXPECT_SUCCESS(urCommandBufferReleaseExp(updatable_cmd_buf_handle));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urKernelExecutionTest::TearDown());
    }

    ur_platform_backend_t backend{};
    ur_exp_command_buffer_handle_t updatable_cmd_buf_handle = nullptr;
};

struct urCommandBufferCommandExpTest
    : urUpdatableCommandBufferExpExecutionTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            urUpdatableCommandBufferExpExecutionTest::SetUp());

        // Append 2 kernel commands to command-buffer and close command-buffer
        ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
            updatable_cmd_buf_handle, kernel, n_dimensions, &global_offset,
            &global_size, &local_size, 0, nullptr, 0, nullptr, 0, nullptr,
            nullptr, nullptr, &command_handle));
        ASSERT_NE(command_handle, nullptr);

        ASSERT_SUCCESS(urCommandBufferAppendKernelLaunchExp(
            updatable_cmd_buf_handle, kernel, n_dimensions, &global_offset,
            &global_size, &local_size, 0, nullptr, 0, nullptr, 0, nullptr,
            nullptr, nullptr, &command_handle_2));
        ASSERT_NE(command_handle_2, nullptr);

        ASSERT_SUCCESS(urCommandBufferFinalizeExp(updatable_cmd_buf_handle));
    }

    void TearDown() override {
        if (command_handle) {
            EXPECT_SUCCESS(urCommandBufferReleaseCommandExp(command_handle));
        }

        if (command_handle_2) {
            EXPECT_SUCCESS(urCommandBufferReleaseCommandExp(command_handle_2));
        }

        UUR_RETURN_ON_FATAL_FAILURE(
            urUpdatableCommandBufferExpExecutionTest::TearDown());
    }

    static constexpr size_t local_size = 4;
    static constexpr size_t global_size = 32;
    static constexpr size_t global_offset = 0;
    static constexpr size_t n_dimensions = 1;

    ur_exp_command_buffer_command_handle_t command_handle = nullptr;
    ur_exp_command_buffer_command_handle_t command_handle_2 = nullptr;
};

struct TestKernel {

    TestKernel(std::string Name, ur_platform_handle_t Platform,
               ur_context_handle_t Context, ur_device_handle_t Device)
        : Name(std::move(Name)), Platform(Platform), Context(Context),
          Device(Device) {}

    virtual ~TestKernel() = default;

    virtual void buildKernel() {
        std::shared_ptr<std::vector<char>> ILBinary;
        std::vector<ur_program_metadata_t> Metadatas{};

        ur_platform_backend_t Backend;
        ASSERT_SUCCESS(urPlatformGetInfo(Platform, UR_PLATFORM_INFO_BACKEND,
                                         sizeof(Backend), &Backend, nullptr));

        ASSERT_NO_FATAL_FAILURE(
            uur::KernelsEnvironment::instance->LoadSource(Name, ILBinary));

        const ur_program_properties_t Properties = {
            UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES, nullptr,
            static_cast<uint32_t>(Metadatas.size()),
            Metadatas.empty() ? nullptr : Metadatas.data()};
        ASSERT_SUCCESS(uur::KernelsEnvironment::instance->CreateProgram(
            Platform, Context, Device, *ILBinary, &Properties, &Program));

        auto KernelNames =
            uur::KernelsEnvironment::instance->GetEntryPointNames(Name);
        std::string KernelName = KernelNames[0];
        ASSERT_FALSE(KernelName.empty());

        ASSERT_SUCCESS(urProgramBuild(Context, Program, nullptr));
        ASSERT_SUCCESS(urKernelCreate(Program, KernelName.data(), &Kernel));
    }

    virtual void setUpKernel() = 0;

    virtual void destroyKernel() {
        ASSERT_SUCCESS(urKernelRelease(Kernel));
        ASSERT_SUCCESS(urProgramRelease(Program));
    };

    virtual void validate() = 0;

    std::string Name;
    ur_platform_handle_t Platform;
    ur_context_handle_t Context;
    ur_device_handle_t Device;
    ur_program_handle_t Program;
    ur_kernel_handle_t Kernel;
};

struct urCommandBufferMultipleKernelUpdateTest
    : uur::command_buffer::urUpdatableCommandBufferExpTest {
    virtual void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urUpdatableCommandBufferExpTest::SetUp());
    }

    virtual void TearDown() override {
        for (auto &TestKernel : TestKernels) {
            UUR_RETURN_ON_FATAL_FAILURE(TestKernel->destroyKernel());
        }
        UUR_RETURN_ON_FATAL_FAILURE(
            urUpdatableCommandBufferExpTest::TearDown());
    }

    void setUpKernels() {
        for (auto &TestKernel : TestKernels) {
            UUR_RETURN_ON_FATAL_FAILURE(TestKernel->setUpKernel());
        }
    }

    std::vector<std::shared_ptr<TestKernel>> TestKernels{};
};

struct urCommandEventSyncTest : urCommandBufferExpTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urCommandBufferExpTest::SetUp());

        ur_bool_t event_support = false;
        ASSERT_SUCCESS(urDeviceGetInfo(
            device, UR_DEVICE_INFO_COMMAND_BUFFER_EVENT_SUPPORT_EXP,
            sizeof(ur_bool_t), &event_support, nullptr));
        if (!event_support) {
            GTEST_SKIP() << "External event sync is not supported by device.";
        }

        ur_queue_flags_t flags = UR_QUEUE_FLAG_SUBMISSION_BATCHED;
        ur_queue_properties_t props = {
            /*.stype =*/UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
            /*.pNext =*/nullptr,
            /*.flags =*/flags,
        };
        ASSERT_SUCCESS(urQueueCreate(context, device, &props, &queue));
        ASSERT_NE(queue, nullptr);

        for (auto &device_ptr : device_ptrs) {
            ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                            allocation_size, &device_ptr));
            ASSERT_NE(device_ptr, nullptr);
        }

        for (auto &buffer : buffers) {
            ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE,
                                             allocation_size, nullptr,
                                             &buffer));
            ASSERT_NE(buffer, nullptr);
        }

        // Create a command-buffer with update enabled.
        ur_exp_command_buffer_desc_t desc{
            /*.stype=*/UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC,
            /*.pNext =*/nullptr,
            /*.isUpdatable =*/false,
            /*.isInOrder =*/false,
            /*.enableProfiling =*/false,
        };

        ASSERT_SUCCESS(urCommandBufferCreateExp(context, device, &desc,
                                                &second_cmd_buf_handle));
        ASSERT_NE(second_cmd_buf_handle, nullptr);
    }

    virtual void TearDown() override {
        for (auto &device_ptr : device_ptrs) {
            if (device_ptr) {
                EXPECT_SUCCESS(urUSMFree(context, device_ptr));
            }
        }

        for (auto &event : external_events) {
            if (event) {
                EXPECT_SUCCESS(urEventRelease(event));
            }
        }

        for (auto &buffer : buffers) {
            if (buffer) {
                EXPECT_SUCCESS(urMemRelease(buffer));
            }
        }

        if (queue) {
            EXPECT_SUCCESS(urQueueRelease(queue));
        }

        if (second_cmd_buf_handle) {
            EXPECT_SUCCESS(urCommandBufferReleaseExp(second_cmd_buf_handle));
        }

        UUR_RETURN_ON_FATAL_FAILURE(urCommandBufferExpTest::TearDown());
    }

    std::array<void *, 3> device_ptrs = {nullptr, nullptr, nullptr};
    std::array<ur_mem_handle_t, 2> buffers = {nullptr, nullptr};
    std::array<ur_event_handle_t, 12> external_events = {
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    std::array<ur_exp_command_buffer_sync_point_t, 2> sync_points = {0, 0};
    ur_queue_handle_t queue = nullptr;
    ur_exp_command_buffer_handle_t second_cmd_buf_handle = nullptr;
    static constexpr size_t elements = 64;
    static constexpr size_t allocation_size = sizeof(uint32_t) * elements;
};

struct urCommandEventSyncUpdateTest : urCommandEventSyncTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urCommandEventSyncTest::SetUp());

        auto required_capabilities =
            UR_DEVICE_COMMAND_BUFFER_UPDATE_CAPABILITY_FLAG_EVENTS;
        UUR_RETURN_ON_FATAL_FAILURE(
            checkCommandBufferUpdateSupport(device, required_capabilities));

        // Create a command-buffer with update enabled.
        ur_exp_command_buffer_desc_t desc{
            UR_STRUCTURE_TYPE_EXP_COMMAND_BUFFER_DESC, nullptr, true, false,
            false};

        ASSERT_SUCCESS(urCommandBufferCreateExp(context, device, &desc,
                                                &updatable_cmd_buf_handle));
        ASSERT_NE(updatable_cmd_buf_handle, nullptr);
    }

    virtual void TearDown() override {
        for (auto command_handle : command_handles) {
            if (command_handle) {
                EXPECT_SUCCESS(
                    urCommandBufferReleaseCommandExp(command_handle));
            }
        }

        if (updatable_cmd_buf_handle) {
            EXPECT_SUCCESS(urCommandBufferReleaseExp(updatable_cmd_buf_handle));
        }

        UUR_RETURN_ON_FATAL_FAILURE(urCommandEventSyncTest::TearDown());
    }

    ur_exp_command_buffer_handle_t updatable_cmd_buf_handle = nullptr;
    std::array<ur_exp_command_buffer_command_handle_t, 3> command_handles = {
        nullptr, nullptr, nullptr};
};
} // namespace command_buffer
} // namespace uur

#endif // UR_CONFORMANCE_EVENT_COMMAND_BUFFER_H_INCLUDED
