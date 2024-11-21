// Copyright (C) 2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_CONFORMANCE_INTEGRATION_FIXTURES_H
#define UR_CONFORMANCE_INTEGRATION_FIXTURES_H

#include <uur/fixtures.h>

namespace uur {

struct IntegrationQueueTestWithParam
    : uur::urKernelExecutionTestWithParam<ur_queue_flag_t> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            uur::urKernelExecutionTestWithParam<ur_queue_flag_t>::SetUp());

        QueueFlags = getParam();
        ur_queue_properties_t queue_properties = {
            UR_STRUCTURE_TYPE_QUEUE_PROPERTIES, nullptr, QueueFlags};
        ASSERT_SUCCESS(
            urQueueCreate(context, device, &queue_properties, &Queue));
    }

    void TearDown() override {
        for (ur_event_handle_t Event : AllEvents) {
            ASSERT_SUCCESS(urEventRelease(Event));
        }

        UUR_RETURN_ON_FATAL_FAILURE(
            uur::urKernelExecutionTestWithParam<ur_queue_flag_t>::TearDown());
    }

    void submitBarrierIfNeeded(std::vector<ur_event_handle_t> &(Events)) {
        if (QueueFlags == UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
            ASSERT_SUCCESS(urEnqueueEventsWaitWithBarrier(
                Queue, Events.size(), Events.data(), nullptr));
            AllEvents.insert(AllEvents.end(), Events.begin(), Events.end());
        }
    }

    void submitBarrierIfNeeded(ur_event_handle_t Event) {
        if (QueueFlags == UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
            ASSERT_SUCCESS(
                urEnqueueEventsWaitWithBarrier(Queue, 1, &Event, nullptr));
            AllEvents.push_back(Event);
        }
    }

    std::vector<ur_event_handle_t> AllEvents;
    ur_queue_flags_t QueueFlags{};
    ur_queue_handle_t Queue{};
    static constexpr size_t ArraySize = 100;
    static constexpr uint32_t InitialValue = 100;

    static std::string paramPrinter(
        const ::testing::TestParamInfo<std::tuple<DeviceTuple, ur_queue_flag_t>>
            &info) {
        auto device = std::get<0>(info.param).device;
        auto param = std::get<1>(info.param);

        std::stringstream ss;
        if (param == 0) {
            ss << "IN_ORDER_QUEUE";
        }
        if (param == UR_QUEUE_FLAG_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
            ss << "OUT_OF_ORDER_QUEUE";
        }

        return uur::GetPlatformAndDeviceName(device) + "__" + ss.str();
    }
};
} // namespace uur

#endif //UR_CONFORMANCE_INTEGRATION_FIXTURES_H
