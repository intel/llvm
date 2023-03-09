// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

#include <random>
#include <uur/fixtures.h>

struct testParametersFill2D {
    size_t pitch;
    size_t width;
    size_t height;
    size_t pattern_size;
};

template <typename T>
inline std::string printFill2DTestString(const testing::TestParamInfo<typename T::ParamType> &info) {
    const auto device_handle = std::get<0>(info.param);
    const auto
        platform_device_name = uur::GetPlatformAndDeviceName(device_handle);
    std::stringstream test_name;
    test_name << platform_device_name << "__pitch__"
              << std::get<1>(info.param).pitch
              << "__width__" << std::get<1>(info.param).width
              << "__height__" << std::get<1>(info.param).height
              << "__patternSize__" << std::get<1>(info.param).pattern_size;
    return test_name.str();
}

struct urEnqueueUSMFill2DTestWithParam
    : uur::urQueueTestWithParam<testParametersFill2D> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTestWithParam::SetUp());

        pitch = std::get<1>(GetParam()).pitch;
        width = std::get<1>(GetParam()).width;
        height = std::get<1>(GetParam()).height;
        pattern_size = std::get<1>(GetParam()).pattern_size;
        pattern = std::vector<uint8_t>(pattern_size);
        generatePattern();
        allocation_size = pitch * height;
        host_mem = std::vector<uint8_t>(allocation_size);

        const auto device_usm =
            uur::GetDeviceInfo<bool>(device, UR_DEVICE_INFO_USM_DEVICE_SUPPORT);
        ASSERT_TRUE(device_usm.has_value());
        if (!device_usm.value()) {
            GTEST_SKIP() << "Device USM is not supported";
        }

        ASSERT_SUCCESS(
            urUSMDeviceAlloc(context, device, nullptr, nullptr, allocation_size,
                             0, &ptr));
    }

    void TearDown() override {
        if (ptr) {
            EXPECT_SUCCESS(urUSMFree(context, ptr));
        }

        UUR_RETURN_ON_FATAL_FAILURE(urQueueTestWithParam::TearDown());
    }

    void generatePattern() {

        const size_t seed = 1;
        std::mt19937 mersenne_engine{seed};
        std::uniform_int_distribution<int> dist{0, 255};

        auto gen = [&dist, &mersenne_engine]() {
            return static_cast<uint8_t>(dist(mersenne_engine));
        };

        std::generate(begin(pattern), end(pattern), gen);
    }

    void verifyData() {
        ASSERT_SUCCESS(
            urEnqueueUSMMemcpy2D(queue, true, host_mem.data(), pitch, ptr,
                                 pitch, width, height, 0, nullptr, nullptr));

        size_t pattern_index = 0;
        for (size_t w = 0; w < width; ++w) {
            for (size_t h = 0; h < height; ++h) {
                uint8_t *host_ptr = host_mem.data();
                size_t index = (pitch * h) + w;
                ASSERT_TRUE((*(host_ptr + index) == pattern[pattern_index]));

                ++pattern_index;
                if (pattern_index % pattern.size() == 0) {
                    pattern_index = 0;
                }
            }
        }
    }

    size_t pitch;
    size_t width;
    size_t height;
    size_t pattern_size;
    std::vector<uint8_t> pattern;
    size_t allocation_size;
    std::vector<uint8_t> host_mem;
    void *ptr{nullptr};
};

static std::vector<testParametersFill2D> test_cases{
    /* Everything set to 1 */
    {1, 1, 1, 1},
    /* Height == 1 && Pitch > width && pattern_size == width*/
    {1024, 256, 1, 256},
    /* Height == 1 && Pitch > width && pattern_size < width*/
    {1024, 256, 1, 4},
    /* Height == 1 && Pitch > width && width != power_of_2 && pattern_size == 1*/
    {1024, 57, 1, 1},
    /* Height == 1 && Pitch == width && pattern_size < width */
    {1024, 1024, 1, 256},
    /* Height == 1 && Pitch == width && pattern_size == width */
    {1024, 1024, 1, 1024},
    /* Height > 1 && Pitch > width && pattern_size == 1 */
    {1024, 256, 256, 1},
    /* Height > 1 && Pitch > width && pattern_size == width */
    {1024, 256, 256, 256},
    /* Height > 1 && Pitch > width && pattern_size == width * height */
    {1024, 256, 256, 256 * 256},
    /* Height == 1 && Pitch == width + 1 && pattern_size == 1 */
    {234, 233, 1, 1},
    /* Height != power_of_2 && Pitch == width + 1 && pattern_size == 1 */
    {234, 233, 35, 1},
    /* Height != power_of_2 && width == power_of_2 && pattern_size == 128 */
    {1024, 256, 35, 128}};

UUR_TEST_SUITE_P(urEnqueueUSMFill2DTestWithParam,
                 testing::ValuesIn(test_cases),
                 printFill2DTestString<urEnqueueUSMFill2DTestWithParam>);

TEST_P(urEnqueueUSMFill2DTestWithParam, Success) {

    ur_event_handle_t event = nullptr;

    ASSERT_SUCCESS(
        urEnqueueUSMFill2D(queue, ptr, pitch, pattern_size, pattern.data(),
                           width, height, 0, nullptr, &event));
    EXPECT_SUCCESS(urQueueFlush(queue));

    ASSERT_SUCCESS(urEventWait(1, &event));
    const auto event_status = uur::GetEventInfo<ur_event_status_t>(event,
                                                                   UR_EVENT_INFO_COMMAND_EXECUTION_STATUS);
    ASSERT_TRUE(event_status.has_value());
    ASSERT_EQ(event_status.value(), UR_EVENT_STATUS_COMPLETE);
    EXPECT_SUCCESS(urEventRelease(event));

    ASSERT_NO_FATAL_FAILURE(verifyData());
}

struct urEnqueueUSMFill2DNegativeTest : uur::urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::SetUp());

        const auto device_usm =
            uur::GetDeviceInfo<bool>(device, UR_DEVICE_INFO_USM_DEVICE_SUPPORT);
        ASSERT_TRUE(device_usm.has_value());
        if (!device_usm.value()) {
            GTEST_SKIP() << "Device USM is not supported";
        }

        ASSERT_SUCCESS(
            urUSMDeviceAlloc(context, device, nullptr, nullptr, allocation_size,
                             0, &ptr));
    }

    void TearDown() override {
        if (ptr) {
            EXPECT_SUCCESS(urUSMFree(context, ptr));
        }

        UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::TearDown());
    }

    static constexpr size_t pitch = 16;
    static constexpr size_t width = 16;
    static constexpr size_t height = 16;
    static constexpr size_t pattern_size = 4;
    static constexpr size_t allocation_size = height * pitch;
    std::vector<uint8_t> pattern{0x01, 0x02, 0x03, 0x04};
    void *ptr{nullptr};
};

UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(urEnqueueUSMFill2DNegativeTest);

TEST_P(urEnqueueUSMFill2DNegativeTest, InvalidNullQueueHandle) {
    ASSERT_EQ_RESULT(
        urEnqueueUSMFill2D(nullptr, ptr, pitch, pattern_size, pattern.data(),
                           width, height, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_NULL_HANDLE);
}

TEST_P(urEnqueueUSMFill2DNegativeTest, InvalidNullPtr) {
    ASSERT_EQ_RESULT(
        urEnqueueUSMFill2D(queue, nullptr, pitch, pattern_size, pattern.data(),
                           width, height, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_NULL_POINTER);

    ASSERT_EQ_RESULT(
        urEnqueueUSMFill2D(queue, ptr, pitch, pattern_size, nullptr,
                           width, height, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_NULL_POINTER);
}

TEST_P(urEnqueueUSMFill2DNegativeTest, InvalidPitch) {

    ASSERT_EQ_RESULT(
        urEnqueueUSMFill2D(queue, ptr, 0, pattern_size, pattern.data(),
                           width, height, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);

    ASSERT_EQ_RESULT(
        urEnqueueUSMFill2D(queue, ptr, width - 1, pattern_size, pattern.data(),
                           width, height, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urEnqueueUSMFill2DNegativeTest, InvalidWidth) {

    /* width is 0 */
    ASSERT_EQ_RESULT(
        urEnqueueUSMFill2D(queue, ptr, pitch, pattern_size, pattern.data(),
                           0, height, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);

    /* width is not a multiple of pattern_size */
    ASSERT_EQ_RESULT(
        urEnqueueUSMFill2D(queue, ptr, pitch, pattern_size, pattern.data(),
                           7, height, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urEnqueueUSMFill2DNegativeTest, InvalidHeight) {

    ASSERT_EQ_RESULT(
        urEnqueueUSMFill2D(queue, ptr, pitch, pattern_size, pattern.data(),
                           width, 0, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urEnqueueUSMFill2DNegativeTest, OutOfBounds) {

    size_t out_of_bounds = pitch * height + 1;

    /* Interpret memory as having just one row */
    ASSERT_EQ_RESULT(
        urEnqueueUSMFill2D(queue, ptr, out_of_bounds, pattern_size,
                           pattern.data(),
                           width, 1, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);

    /* Interpret memory as having just one column */
    ASSERT_EQ_RESULT(
        urEnqueueUSMFill2D(queue, ptr, out_of_bounds, pattern_size,
                           pattern.data(),
                           1, height, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urEnqueueUSMFill2DNegativeTest, invalidPatternSize) {

    /* pattern size is 0 */
    ASSERT_EQ_RESULT(
        urEnqueueUSMFill2D(queue, ptr, pitch, 0,
                           pattern.data(),
                           width, 1, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);

    /* pattern_size is not a power of 2 */
    ASSERT_EQ_RESULT(
        urEnqueueUSMFill2D(queue, ptr, pitch, 3,
                           pattern.data(),
                           width, 1, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);

    /* pattern_size is larger than size */
    ASSERT_EQ_RESULT(
        urEnqueueUSMFill2D(queue, ptr, pitch, 32,
                           pattern.data(),
                           width, 1, 0, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_SIZE);
}

TEST_P(urEnqueueUSMFill2DNegativeTest, InvalidNullPtrEventWaitList) {

    ASSERT_EQ_RESULT(
        urEnqueueUSMFill2D(queue, ptr, pitch, pattern_size,
                           pattern.data(),
                           width, 1, 1, nullptr, nullptr),
        UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(
        urEnqueueUSMFill2D(queue, ptr, pitch, pattern_size,
                           pattern.data(),
                           width, 1, 0, &validEvent, nullptr),
        UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);
}
