// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: MIT

#ifndef UR_CONFORMANCE_EVENT_FIXTURES_H_INCLUDED
#define UR_CONFORMANCE_EVENT_FIXTURES_H_INCLUDED

#include <uur/fixtures.h>

namespace uur {
namespace event {

/**
 * Test fixture that is intended to be used when testing reference count APIs
 * (i.e. urEventRelease and urEventRetain). Does not handle destruction of the
 * event.
 */
struct urEventReferenceTest : uur::urQueueTest {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_WRITE_ONLY, size,
                                         nullptr, &buffer));

        input.assign(count, 42);
        ASSERT_SUCCESS(urEnqueueMemBufferWrite(
            queue, buffer, false, 0, size, input.data(), 0, nullptr, &event));
    }

    void TearDown() override {
        if (buffer) {
            EXPECT_SUCCESS(urMemRelease(buffer));
        }
        urQueueTest::TearDown();
    }

    bool checkEventReferenceCount(uint32_t expected_value) {
        uint32_t reference_count{};
        urEventGetInfo(event, ur_event_info_t::UR_EVENT_INFO_REFERENCE_COUNT,
                       sizeof(uint32_t), &reference_count, nullptr);
        return reference_count == expected_value;
    }

    const size_t count = 1024;
    const size_t size = sizeof(uint32_t) * count;
    ur_mem_handle_t buffer = nullptr;
    ur_event_handle_t event = nullptr;
    std::vector<uint32_t> input;
};
} // namespace event
} // namespace uur

#endif // UR_CONFORMANCE_EVENT_FIXTURES_H_INCLUDED
