// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "helpers.h"

#include <uur/fixtures.h>
#include <uur/raii.h>

#include <thread>
#include <utility>

// There was a bug in previous L0 drivers that caused the test to fail
std::tuple<size_t, size_t, size_t> minL0DriverVersion = {1, 3, 29534};

template <typename T>
struct urMultiQueueLaunchMemcpyTest : uur::urMultiQueueMultiDeviceTest,
                                      testing::WithParamInterface<T> {
    std::string KernelName;
    std::vector<ur_program_handle_t> programs;
    std::vector<ur_kernel_handle_t> kernels;
    std::vector<void *> SharedMem;

    static constexpr char ProgramName[] = "increment";
    static constexpr size_t ArraySize = 100;
    static constexpr uint32_t InitialValue = 1;

    void SetUp() override { throw std::runtime_error("Not implemented"); }

    void SetUp(size_t duplicateDevices) {
        UUR_RETURN_ON_FATAL_FAILURE(
            uur::urMultiQueueMultiDeviceTest::SetUp(duplicateDevices));

        for (auto &device : devices) {
            SKIP_IF_DRIVER_TOO_OLD("Level-Zero", minL0DriverVersion, platform,
                                   device);
        }

        programs.resize(devices.size());
        kernels.resize(devices.size());
        SharedMem.resize(devices.size());

        KernelName = uur::KernelsEnvironment::instance->GetEntryPointNames(
            ProgramName)[0];

        std::shared_ptr<std::vector<char>> il_binary;
        std::vector<ur_program_metadata_t> metadatas{};

        uur::KernelsEnvironment::instance->LoadSource(ProgramName, il_binary);

        for (size_t i = 0; i < devices.size(); i++) {
            const ur_program_properties_t properties = {
                UR_STRUCTURE_TYPE_PROGRAM_PROPERTIES, nullptr,
                static_cast<uint32_t>(metadatas.size()),
                metadatas.empty() ? nullptr : metadatas.data()};

            uur::raii::Program program;
            ASSERT_SUCCESS(uur::KernelsEnvironment::instance->CreateProgram(
                platform, context, devices[i], *il_binary, &properties,
                &programs[i]));

            UUR_ASSERT_SUCCESS_OR_UNSUPPORTED(
                urProgramBuild(context, programs[i], nullptr));
            ASSERT_SUCCESS(
                urKernelCreate(programs[i], KernelName.data(), &kernels[i]));

            ASSERT_SUCCESS(
                urUSMSharedAlloc(context, devices[i], nullptr, nullptr,
                                 ArraySize * sizeof(uint32_t), &SharedMem[i]));
            ASSERT_NE(SharedMem[i], nullptr);

            ASSERT_SUCCESS(urEnqueueUSMFill(queues[i], SharedMem[i],
                                            sizeof(uint32_t), &InitialValue,
                                            ArraySize * sizeof(uint32_t), 0,
                                            nullptr, nullptr /* &Event */));
            ASSERT_SUCCESS(urQueueFinish(queues[i]));

            ASSERT_SUCCESS(
                urKernelSetArgPointer(kernels[i], 0, nullptr, SharedMem[i]));
        }
    }

    void TearDown() override {
        for (auto &Ptr : SharedMem) {
            urUSMFree(context, Ptr);
        }
        for (const auto &kernel : kernels) {
            urKernelRelease(kernel);
        }
        for (const auto &program : programs) {
            urProgramRelease(program);
        }
        UUR_RETURN_ON_FATAL_FAILURE(
            uur::urMultiDeviceContextTestTemplate<1>::TearDown());
    }

    void runBackgroundCheck(std::vector<uur::raii::Event> &Events) {
        std::vector<std::thread> threads;
        for (size_t i = 0; i < Events.size(); i++) {
            threads.emplace_back([&, i] {
                ur_event_status_t status;
                do {
                    ASSERT_SUCCESS(urEventGetInfo(
                        Events[i].get(), UR_EVENT_INFO_COMMAND_EXECUTION_STATUS,
                        sizeof(ur_event_status_t), &status, nullptr));
                } while (status != UR_EVENT_STATUS_COMPLETE);

                auto ExpectedValue = InitialValue + i + 1;
                for (uint32_t j = 0; j < ArraySize; ++j) {
                    ASSERT_EQ(reinterpret_cast<uint32_t *>(SharedMem[i])[j],
                              ExpectedValue);
                }
            });
        }
        for (auto &thread : threads) {
            thread.join();
        }
    }
};

template <typename Param>
struct urEnqueueKernelLaunchIncrementMultiDeviceTestWithParam
    : public urMultiQueueLaunchMemcpyTest<Param> {
    static constexpr size_t duplicateDevices = 8;

    using urMultiQueueLaunchMemcpyTest<Param>::context;
    using urMultiQueueLaunchMemcpyTest<Param>::queues;
    using urMultiQueueLaunchMemcpyTest<Param>::devices;
    using urMultiQueueLaunchMemcpyTest<Param>::kernels;
    using urMultiQueueLaunchMemcpyTest<Param>::SharedMem;

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            urMultiQueueLaunchMemcpyTest<Param>::SetUp(duplicateDevices));
    }

    void TearDown() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            urMultiQueueLaunchMemcpyTest<Param>::TearDown());
    }
};

struct urEnqueueKernelLaunchIncrementTest
    : urMultiQueueLaunchMemcpyTest<uur::BoolTestParam> {
    static constexpr size_t numOps = 50;

    using Param = uur::BoolTestParam;
    using urMultiQueueLaunchMemcpyTest<Param>::context;
    using urMultiQueueLaunchMemcpyTest<Param>::queues;
    using urMultiQueueLaunchMemcpyTest<Param>::devices;
    using urMultiQueueLaunchMemcpyTest<Param>::kernels;
    using urMultiQueueLaunchMemcpyTest<Param>::SharedMem;

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urMultiQueueLaunchMemcpyTest<Param>::SetUp(
            numOps)); // Use single device, duplicated numOps times
    }

    void TearDown() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            urMultiQueueLaunchMemcpyTest<Param>::TearDown());
    }
};

UUR_DEVICE_TEST_SUITE_P(
    urEnqueueKernelLaunchIncrementTest,
    testing::ValuesIn(uur::BoolTestParam::makeBoolParam("UseEvents")),
    uur::deviceTestWithParamPrinter<uur::BoolTestParam>);

TEST_P(urEnqueueKernelLaunchIncrementTest, Success) {
    constexpr size_t global_offset = 0;
    constexpr size_t n_dimensions = 1;

    auto useEvents = std::get<1>(GetParam()).value;

    std::vector<uur::raii::Event> kernelEvents(numOps);
    std::vector<uur::raii::Event> memcpyEvents(numOps - 1);

    ur_event_handle_t *lastMemcpyEvent = nullptr;
    ur_event_handle_t *kernelEvent = nullptr;
    ur_event_handle_t *memcpyEvent = nullptr;

    // This is a single device test
    auto queue = queues[0];

    for (size_t i = 0; i < numOps; i++) {
        if (useEvents) {
            lastMemcpyEvent = memcpyEvent;
            kernelEvent = kernelEvents[i].ptr();
            memcpyEvent = i < numOps - 1 ? memcpyEvents[i].ptr() : nullptr;
        }

        // execute kernel that increments each element by 1
        ASSERT_SUCCESS(urEnqueueKernelLaunch(
            queue, kernels[i], n_dimensions, &global_offset, &ArraySize,
            nullptr, bool(lastMemcpyEvent), lastMemcpyEvent, kernelEvent));

        // copy the memory (input for the next kernel)
        if (i < numOps - 1) {
            ASSERT_SUCCESS(
                urEnqueueUSMMemcpy(queue, false, SharedMem[i + 1], SharedMem[i],
                                   ArraySize * sizeof(uint32_t), useEvents,
                                   kernelEvent, memcpyEvent));
        }
    }

    if (useEvents) {
        ASSERT_SUCCESS(urEventWait(1, kernelEvents.back().ptr()));
    } else {
        ASSERT_SUCCESS(urQueueFinish(queue));
    }

    size_t ExpectedValue = InitialValue;
    for (size_t i = 0; i < numOps; i++) {
        ExpectedValue++;
        for (uint32_t j = 0; j < ArraySize; ++j) {
            ASSERT_EQ(reinterpret_cast<uint32_t *>(SharedMem[i])[j],
                      ExpectedValue);
        }
    }
}

template <typename T>
inline std::string
printParams(const testing::TestParamInfo<typename T::ParamType> &info) {
    std::stringstream ss;

    auto param1 = std::get<0>(info.param);
    ss << (param1.value ? "" : "No") << param1.name;

    auto param2 = std::get<1>(info.param);
    ss << (param2.value ? "" : "No") << param2.name;

    if constexpr (std::tuple_size_v < typename T::ParamType >> 2) {
        auto param3 = std::get<2>(info.param);
    }

    return ss.str();
}

using urEnqueueKernelLaunchIncrementMultiDeviceTest =
    urEnqueueKernelLaunchIncrementMultiDeviceTestWithParam<
        std::tuple<uur::BoolTestParam, uur::BoolTestParam>>;

INSTANTIATE_TEST_SUITE_P(
    , urEnqueueKernelLaunchIncrementMultiDeviceTest,
    testing::Combine(
        testing::ValuesIn(uur::BoolTestParam::makeBoolParam("UseEventWait")),
        testing::ValuesIn(
            uur::BoolTestParam::makeBoolParam("RunBackgroundCheck"))),
    printParams<urEnqueueKernelLaunchIncrementMultiDeviceTest>);

// Do a chain of kernelLaunch(dev0) -> memcpy(dev0, dev1) -> kernelLaunch(dev1) ... ops
TEST_P(urEnqueueKernelLaunchIncrementMultiDeviceTest, Success) {
    auto waitOnEvent = std::get<0>(GetParam()).value;
    auto runBackgroundCheck = std::get<1>(GetParam()).value;

    size_t returned_size;
    ASSERT_SUCCESS(urDeviceGetInfo(devices[0], UR_DEVICE_INFO_EXTENSIONS, 0,
                                   nullptr, &returned_size));

    std::unique_ptr<char[]> returned_extensions(new char[returned_size]);

    ASSERT_SUCCESS(urDeviceGetInfo(devices[0], UR_DEVICE_INFO_EXTENSIONS,
                                   returned_size, returned_extensions.get(),
                                   nullptr));

    std::string_view extensions_string(returned_extensions.get());
    const bool usm_p2p_support =
        extensions_string.find(UR_USM_P2P_EXTENSION_STRING_EXP) !=
        std::string::npos;

    if (!usm_p2p_support) {
        GTEST_SKIP() << "EXP usm p2p feature is not supported.";
    }

    constexpr size_t global_offset = 0;
    constexpr size_t n_dimensions = 1;

    std::vector<uur::raii::Event> kernelEvents(devices.size());
    std::vector<uur::raii::Event> memcpyEvents(devices.size() - 1);

    ur_event_handle_t *lastMemcpyEvent = nullptr;
    ur_event_handle_t *kernelEvent = nullptr;
    ur_event_handle_t *memcpyEvent = nullptr;

    for (size_t i = 0; i < devices.size(); i++) {
        lastMemcpyEvent = memcpyEvent;
        kernelEvent = kernelEvents[i].ptr();
        memcpyEvent = i < devices.size() - 1 ? memcpyEvents[i].ptr() : nullptr;

        // execute kernel that increments each element by 1
        ASSERT_SUCCESS(urEnqueueKernelLaunch(
            queues[i], kernels[i], n_dimensions, &global_offset, &ArraySize,
            nullptr, bool(lastMemcpyEvent), lastMemcpyEvent, kernelEvent));

        // copy the memory to next device
        if (i < devices.size() - 1) {
            ASSERT_SUCCESS(urEnqueueUSMMemcpy(
                queues[i], false, SharedMem[i + 1], SharedMem[i],
                ArraySize * sizeof(uint32_t), 1, kernelEvent, memcpyEvent));
        }
    }

    // While the device(s) execute, loop over the events and if completed, verify the results
    if (runBackgroundCheck) {
        this->runBackgroundCheck(kernelEvents);
    }

    // synchronize on the last queue/event only, this has to ensure all the operations
    // are completed
    if (waitOnEvent) {
        ASSERT_SUCCESS(urEventWait(1, kernelEvents.back().ptr()));
    } else {
        ASSERT_SUCCESS(urQueueFinish(queues.back()));
    }

    size_t ExpectedValue = InitialValue;
    for (size_t i = 0; i < devices.size(); i++) {
        ExpectedValue++;
        for (uint32_t j = 0; j < ArraySize; ++j) {
            ASSERT_EQ(reinterpret_cast<uint32_t *>(SharedMem[i])[j],
                      ExpectedValue);
        }
    }
}

using urEnqueueKernelLaunchIncrementMultiDeviceMultiThreadTest =
    urEnqueueKernelLaunchIncrementMultiDeviceTestWithParam<
        std::tuple<uur::BoolTestParam, uur::BoolTestParam>>;

INSTANTIATE_TEST_SUITE_P(
    , urEnqueueKernelLaunchIncrementMultiDeviceMultiThreadTest,
    testing::Combine(
        testing::ValuesIn(uur::BoolTestParam::makeBoolParam("UseEvents")),
        testing::ValuesIn(uur::BoolTestParam::makeBoolParam("QueuePerThread"))),
    printParams<urEnqueueKernelLaunchIncrementMultiDeviceMultiThreadTest>);

// Enqueue kernelLaunch concurrently from multiple threads
// With !queuePerThread this becomes a test on a single device
TEST_P(urEnqueueKernelLaunchIncrementMultiDeviceMultiThreadTest, Success) {
    size_t numThreads = devices.size();
    std::vector<std::thread> threads;

    static constexpr size_t numOpsPerThread = 6;

    auto useEvents = std::get<0>(GetParam()).value;
    auto queuePerThread = std::get<1>(GetParam()).value;

    for (size_t i = 0; i < numThreads; i++) {
        threads.emplace_back([this, i, queuePerThread, useEvents]() {
            constexpr size_t global_offset = 0;
            constexpr size_t n_dimensions = 1;

            auto queue = queuePerThread ? queues[i] : queues.back();
            auto kernel = kernels[i];
            auto sharedPtr = SharedMem[i];

            std::vector<uur::raii::Event> Events(numOpsPerThread + 1);
            for (size_t j = 0; j < numOpsPerThread; j++) {
                size_t waitNum = 0;
                ur_event_handle_t *lastEvent = nullptr;
                ur_event_handle_t *signalEvent = nullptr;

                if (useEvents) {
                    waitNum = j > 0 ? 1 : 0;
                    lastEvent = j > 0 ? Events[j - 1].ptr() : nullptr;
                    signalEvent = Events[j].ptr();
                }

                // execute kernel that increments each element by 1
                ASSERT_SUCCESS(urEnqueueKernelLaunch(
                    queue, kernel, n_dimensions, &global_offset, &ArraySize,
                    nullptr, waitNum, lastEvent, signalEvent));
            }

            std::vector<uint32_t> data(ArraySize);

            auto lastEvent =
                useEvents ? Events[numOpsPerThread - 1].ptr() : nullptr;
            auto signalEvent = useEvents ? Events.back().ptr() : nullptr;
            ASSERT_SUCCESS(
                urEnqueueUSMMemcpy(queue, false, data.data(), sharedPtr,
                                   ArraySize * sizeof(uint32_t), useEvents,
                                   lastEvent, signalEvent));

            if (useEvents) {
                ASSERT_SUCCESS(urEventWait(1, Events.back().ptr()));
            } else {
                ASSERT_SUCCESS(urQueueFinish(queue));
            }

            size_t ExpectedValue = InitialValue;
            ExpectedValue += numOpsPerThread;
            for (uint32_t j = 0; j < ArraySize; ++j) {
                ASSERT_EQ(data[j], ExpectedValue);
            }
        });
    }

    for (auto &thread : threads) {
        thread.join();
    }
}
