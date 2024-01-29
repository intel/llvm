// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_CONFORMANCE_INCLUDE_FIXTURES_H_INCLUDED
#define UR_CONFORMANCE_INCLUDE_FIXTURES_H_INCLUDED

#include "ur_api.h"
#include <uur/checks.h>
#include <uur/environment.h>
#include <uur/utils.h>

#include <random>

#define UUR_RETURN_ON_FATAL_FAILURE(...)                                       \
    __VA_ARGS__;                                                               \
    if (this->HasFatalFailure() || this->IsSkipped()) {                        \
        return;                                                                \
    }                                                                          \
    (void)0

namespace uur {

struct urPlatformTest : ::testing::Test {
    void SetUp() override {
        platform = uur::PlatformEnvironment::instance->platform;
    }

    ur_platform_handle_t platform = nullptr;
};

inline std::pair<bool, std::vector<ur_device_handle_t>>
GetDevices(ur_platform_handle_t platform) {
    uint32_t count = 0;
    if (urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count)) {
        return {false, {}};
    }
    if (count == 0) {
        return {false, {}};
    }
    std::vector<ur_device_handle_t> devices(count);
    if (urDeviceGet(platform, UR_DEVICE_TYPE_ALL, count, devices.data(),
                    nullptr)) {
        return {false, {}};
    }
    return {true, devices};
}

inline bool hasDevicePartitionSupport(ur_device_handle_t device,
                                      const ur_device_partition_t property) {
    std::vector<ur_device_partition_t> properties;
    uur::GetDevicePartitionProperties(device, properties);
    return std::find(properties.begin(), properties.end(), property) !=
           properties.end();
}

struct urAllDevicesTest : urPlatformTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urPlatformTest::SetUp());
        auto devicesPair = GetDevices(platform);
        if (!devicesPair.first) {
            FAIL() << "Failed to get devices";
        }
        devices = std::move(devicesPair.second);
    }

    void TearDown() override {
        for (auto &device : devices) {
            EXPECT_SUCCESS(urDeviceRelease(device));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urPlatformTest::TearDown());
    }

    std::vector<ur_device_handle_t> devices;
};

struct urDeviceTest : urPlatformTest,
                      ::testing::WithParamInterface<ur_device_handle_t> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urPlatformTest::SetUp());
        device = GetParam();
    }

    void TearDown() override {
        EXPECT_SUCCESS(urDeviceRelease(device));
        UUR_RETURN_ON_FATAL_FAILURE(urPlatformTest::TearDown());
    }

    ur_device_handle_t device;
};
} // namespace uur

#define UUR_INSTANTIATE_DEVICE_TEST_SUITE_P(FIXTURE)                           \
    INSTANTIATE_TEST_SUITE_P(                                                  \
        , FIXTURE,                                                             \
        ::testing::ValuesIn(uur::DevicesEnvironment::instance->devices),       \
        [](const ::testing::TestParamInfo<ur_device_handle_t> &info) {         \
            return uur::GetPlatformAndDeviceName(info.param);                  \
        })

#define UUR_INSTANTIATE_KERNEL_TEST_SUITE_P(FIXTURE)                           \
    INSTANTIATE_TEST_SUITE_P(                                                  \
        , FIXTURE,                                                             \
        ::testing::ValuesIn(uur::KernelsEnvironment::instance->devices),       \
        [](const ::testing::TestParamInfo<ur_device_handle_t> &info) {         \
            return uur::GetPlatformAndDeviceName(info.param);                  \
        })

namespace uur {

template <class T>
struct urDeviceTestWithParam
    : urPlatformTest,
      ::testing::WithParamInterface<std::tuple<ur_device_handle_t, T>> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urPlatformTest::SetUp());
        device = std::get<0>(this->GetParam());
    }
    // TODO - I don't like the confusion with GetParam();
    const T &getParam() const { return std::get<1>(this->GetParam()); }
    ur_device_handle_t device;
};

struct urContextTest : urDeviceTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urDeviceTest::SetUp());
        ASSERT_SUCCESS(urContextCreate(1, &device, nullptr, &context));
        ASSERT_NE(context, nullptr);
    }

    void TearDown() override {
        EXPECT_SUCCESS(urContextRelease(context));
        UUR_RETURN_ON_FATAL_FAILURE(urDeviceTest::TearDown());
    }

    ur_context_handle_t context = nullptr;
};

struct urSamplerTest : urContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
        sampler_desc = {
            UR_STRUCTURE_TYPE_SAMPLER_DESC,   /* stype */
            nullptr,                          /* pNext */
            true,                             /* normalizedCoords */
            UR_SAMPLER_ADDRESSING_MODE_CLAMP, /* addressing mode */
            UR_SAMPLER_FILTER_MODE_LINEAR,    /* filterMode */
        };
        ASSERT_SUCCESS(urSamplerCreate(context, &sampler_desc, &sampler));
    }

    void TearDown() override {
        EXPECT_SUCCESS(urSamplerRelease(sampler));
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::TearDown());
    }

    ur_sampler_handle_t sampler;
    ur_sampler_desc_t sampler_desc;
};

struct urMemBufferTest : urContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, 4096,
                                         nullptr, &buffer));
        ASSERT_NE(nullptr, buffer);
    }

    void TearDown() override {
        if (buffer) {
            EXPECT_SUCCESS(urMemRelease(buffer));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::TearDown());
    }

    ur_mem_handle_t buffer = nullptr;
};

struct urMemImageTest : urContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
    }

    void TearDown() override {
        if (image) {
            EXPECT_SUCCESS(urMemRelease(image));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::TearDown());
    }

    ur_image_format_t image_format = {
        /*.channelOrder =*/UR_IMAGE_CHANNEL_ORDER_ARGB,
        /*.channelType =*/UR_IMAGE_CHANNEL_TYPE_UNORM_INT8,
    };
    ur_image_desc_t image_desc = {
        /*.stype =*/UR_STRUCTURE_TYPE_IMAGE_DESC,
        /*.pNext =*/nullptr,
        /*.type =*/UR_MEM_TYPE_IMAGE2D,
        /*.width =*/16,
        /*.height =*/16,
        /*.depth =*/1,
        /*.arraySize =*/1,
        /*.rowPitch =*/16 * sizeof(char[4]),
        /*.slicePitch =*/16 * 16 * sizeof(char[4]),
        /*.numMipLevel =*/0,
        /*.numSamples =*/0,
    };
    ur_mem_handle_t image = nullptr;
};

} // namespace uur

#define UUR_TEST_SUITE_P(FIXTURE, VALUES, PRINTER)                             \
    INSTANTIATE_TEST_SUITE_P(                                                  \
        , FIXTURE,                                                             \
        testing::Combine(                                                      \
            ::testing::ValuesIn(uur::DevicesEnvironment::instance->devices),   \
            VALUES),                                                           \
        PRINTER)

namespace uur {

template <class T> struct urContextTestWithParam : urDeviceTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urDeviceTestWithParam<T>::SetUp());
        ASSERT_SUCCESS(urContextCreate(1, &this->device, nullptr, &context));
    }

    void TearDown() override {
        EXPECT_SUCCESS(urContextRelease(context));
        UUR_RETURN_ON_FATAL_FAILURE(urDeviceTestWithParam<T>::TearDown());
    }
    ur_context_handle_t context;
};

template <class T> struct urSamplerTestWithParam : urContextTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::SetUp());
        sampler_desc = {
            UR_STRUCTURE_TYPE_SAMPLER_DESC,   /* stype */
            nullptr,                          /* pNext */
            true,                             /* normalizedCoords */
            UR_SAMPLER_ADDRESSING_MODE_CLAMP, /* addressing mode */
            UR_SAMPLER_FILTER_MODE_LINEAR,    /* filterMode */
        };
        ASSERT_SUCCESS(urSamplerCreate(this->context, &sampler_desc, &sampler));
    }

    void TearDown() override {
        EXPECT_SUCCESS(urSamplerRelease(sampler));
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::TearDown());
    }

    ur_sampler_handle_t sampler;
    ur_sampler_desc_t sampler_desc;
};

template <class T> struct urMemBufferTestWithParam : urContextTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::SetUp());
        ASSERT_SUCCESS(urMemBufferCreate(this->context, UR_MEM_FLAG_READ_WRITE,
                                         4096, nullptr, &buffer));
        ASSERT_NE(nullptr, buffer);
    }

    void TearDown() override {
        if (buffer) {
            EXPECT_SUCCESS(urMemRelease(buffer));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::TearDown());
    }
    ur_mem_handle_t buffer = nullptr;
};

template <class T> struct urMemImageTestWithParam : urContextTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::SetUp());
        ASSERT_SUCCESS(urMemImageCreate(this->context, UR_MEM_FLAG_READ_WRITE,
                                        &format, &desc, nullptr, &image));
        ASSERT_NE(nullptr, image);
    }

    void TearDown() override {
        if (image) {
            EXPECT_SUCCESS(urMemRelease(image));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::TearDown());
    }
    ur_mem_handle_t image = nullptr;
    ur_image_format_t format = {UR_IMAGE_CHANNEL_ORDER_RGBA,
                                UR_IMAGE_CHANNEL_TYPE_FLOAT};
    ur_image_desc_t desc = {UR_STRUCTURE_TYPE_IMAGE_DESC, // stype
                            nullptr,                      // pNext
                            UR_MEM_TYPE_IMAGE1D,          // mem object type
                            1024,                         // image width
                            1,                            // image height
                            1,                            // image depth
                            1,                            // array size
                            0,                            // row pitch
                            0,                            // slice pitch
                            0,                            // mip levels
                            0};                           // num samples
};

struct urQueueTest : urContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
        ASSERT_SUCCESS(urQueueCreate(context, device, 0, &queue));
        ASSERT_NE(queue, nullptr);
    }

    void TearDown() override {
        if (queue) {
            EXPECT_SUCCESS(urQueueRelease(queue));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::TearDown());
    }

    ur_queue_handle_t queue = nullptr;
};

struct urHostPipeTest : urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());
        uur::KernelsEnvironment::instance->LoadSource("foo", 0, il_binary);
        ASSERT_SUCCESS(uur::KernelsEnvironment::instance->CreateProgram(
            platform, context, device, *il_binary, &program));

        size_t size = 0;
        ASSERT_SUCCESS(urDeviceGetInfo(
            device, UR_DEVICE_INFO_HOST_PIPE_READ_WRITE_SUPPORTED, 0, nullptr,
            &size));
        ASSERT_NE(size, 0);
        ASSERT_EQ(sizeof(ur_bool_t), size);
        void *info_data = alloca(size);
        ASSERT_SUCCESS(urDeviceGetInfo(
            device, UR_DEVICE_INFO_HOST_PIPE_READ_WRITE_SUPPORTED, size,
            info_data, nullptr));
        ASSERT_NE(info_data, nullptr);

        bool supported;
        GetDeviceHostPipeRWSupported(device, supported);
        if (!supported) {
            GTEST_SKIP() << "Host pipe read/write is not supported.";
        }
    }

    void TearDown() override {
        if (program) {
            EXPECT_SUCCESS(urProgramRelease(program));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::TearDown());
    }

    std::shared_ptr<std::vector<char>> il_binary;
    std::string program_name = "foo";
    ur_program_handle_t program = nullptr;

    const char *pipe_symbol = "pipe_symbol";

    static const size_t size = 1024;
    char buffer[size];
};

template <class T> struct urQueueTestWithParam : urContextTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::SetUp());
        ASSERT_SUCCESS(urQueueCreate(this->context, this->device, 0, &queue));
    }

    void TearDown() override {
        if (queue) {
            EXPECT_SUCCESS(urQueueRelease(queue));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::TearDown());
    }

    ur_queue_handle_t queue;
};

struct urProfilingQueueTest : urContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
        ur_queue_properties_t props = {
            /*.stype =*/UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
            /*.pNext =*/nullptr,
            /*.flags =*/UR_QUEUE_FLAG_PROFILING_ENABLE,
        };
        ASSERT_SUCCESS(
            urQueueCreate(this->context, this->device, &props, &queue));
    }

    void TearDown() override {
        if (queue) {
            EXPECT_SUCCESS(urQueueRelease(queue));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::TearDown());
    };

    ur_queue_handle_t queue = nullptr;
};

template <class T>
struct urProfilingQueueTestWithParam : urContextTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::SetUp());
        ur_queue_properties_t props = {
            /*.stype =*/UR_STRUCTURE_TYPE_QUEUE_PROPERTIES,
            /*.pNext =*/nullptr,
            /*.flags =*/UR_QUEUE_FLAG_PROFILING_ENABLE,
        };
        ASSERT_SUCCESS(
            urQueueCreate(this->context, this->device, &props, &queue));
    }

    void TearDown() override {
        if (queue) {
            EXPECT_SUCCESS(urQueueRelease(queue));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::TearDown());
    };

    ur_queue_handle_t queue = nullptr;
};

struct urMultiQueueTest : urContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
        ASSERT_SUCCESS(urQueueCreate(context, device, 0, &queue1));
        ASSERT_SUCCESS(urQueueCreate(context, device, 0, &queue2));
    }

    void TearDown() override {
        if (queue1 != nullptr) {
            EXPECT_SUCCESS(urQueueRelease(queue1));
        }
        if (queue2 != nullptr) {
            EXPECT_SUCCESS(urQueueRelease(queue2));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::TearDown());
    }

    ur_queue_handle_t queue1 = nullptr;
    ur_queue_handle_t queue2 = nullptr;
};

struct urMultiDeviceContextTest : urPlatformTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urPlatformTest::SetUp());
        auto &devices = DevicesEnvironment::instance->devices;
        if (devices.size() <= 1) {
            GTEST_SKIP();
        }
        ASSERT_SUCCESS(urContextCreate(static_cast<uint32_t>(devices.size()),
                                       devices.data(), nullptr, &context));
    }

    void TearDown() override {
        if (context) {
            ASSERT_SUCCESS(urContextRelease(context));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urPlatformTest::TearDown());
    }

    ur_context_handle_t context = nullptr;
};

struct urMultiDeviceMemBufferTest : urMultiDeviceContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urMultiDeviceContextTest::SetUp());
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, size,
                                         nullptr, &buffer));
        ASSERT_NE(nullptr, buffer);
    }

    void TearDown() override {
        if (buffer) {
            EXPECT_SUCCESS(urMemRelease(buffer));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urMultiDeviceContextTest::TearDown());
    }

    ur_mem_handle_t buffer = nullptr;
    const size_t count = 1024;
    const size_t size = count * sizeof(uint32_t);
};

struct urMultiDeviceMemBufferQueueTest : urMultiDeviceMemBufferTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urMultiDeviceMemBufferTest::SetUp());
        queues.reserve(DevicesEnvironment::instance->devices.size());
        for (const auto &device : DevicesEnvironment::instance->devices) {
            ur_queue_handle_t queue = nullptr;
            ASSERT_SUCCESS(urQueueCreate(context, device, 0, &queue));
            queues.push_back(queue);
        }
    }

    void TearDown() override {
        for (const auto &queue : queues) {
            EXPECT_SUCCESS(urQueueRelease(queue));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urMultiDeviceMemBufferTest::TearDown());
    }

    std::vector<ur_queue_handle_t> queues;
};

struct urMemBufferQueueTest : urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, size,
                                         nullptr, &buffer));
    }

    void TearDown() override {
        if (buffer) {
            EXPECT_SUCCESS(urMemRelease(buffer));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::TearDown());
    }

    const size_t count = 8;
    const size_t size = sizeof(uint32_t) * count;
    ur_mem_handle_t buffer = nullptr;
};

struct urMemImageQueueTest : urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());
        ASSERT_SUCCESS(urMemImageCreate(this->context, UR_MEM_FLAG_READ_WRITE,
                                        &format, &desc1D, nullptr, &image1D));

        ASSERT_SUCCESS(urMemImageCreate(this->context, UR_MEM_FLAG_READ_WRITE,
                                        &format, &desc2D, nullptr, &image2D));

        ASSERT_SUCCESS(urMemImageCreate(this->context, UR_MEM_FLAG_READ_WRITE,
                                        &format, &desc3D, nullptr, &image3D));
    }

    void TearDown() override {
        if (image1D) {
            EXPECT_SUCCESS(urMemRelease(image1D));
        }
        if (image2D) {
            EXPECT_SUCCESS(urMemRelease(image2D));
        }
        if (image3D) {
            EXPECT_SUCCESS(urMemRelease(image3D));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::TearDown());
    }

    const size_t width = 1024;
    const size_t height = 8;
    const size_t depth = 2;
    ur_mem_handle_t image1D = nullptr;
    ur_mem_handle_t image2D = nullptr;
    ur_mem_handle_t image3D = nullptr;
    ur_rect_region_t region1D{width, 1, 1};
    ur_rect_region_t region2D{width, height, 1};
    ur_rect_region_t region3D{width, height, depth};
    ur_rect_offset_t origin{0, 0, 0};
    ur_image_format_t format = {UR_IMAGE_CHANNEL_ORDER_RGBA,
                                UR_IMAGE_CHANNEL_TYPE_FLOAT};
    ur_image_desc_t desc1D = {UR_STRUCTURE_TYPE_IMAGE_DESC, // stype
                              nullptr,                      // pNext
                              UR_MEM_TYPE_IMAGE1D,          // mem object type
                              width,                        // image width
                              1,                            // image height
                              1,                            // image depth
                              1,                            // array size
                              0,                            // row pitch
                              0,                            // slice pitch
                              0,                            // mip levels
                              0};                           // num samples

    ur_image_desc_t desc2D = {UR_STRUCTURE_TYPE_IMAGE_DESC, // stype
                              nullptr,                      // pNext
                              UR_MEM_TYPE_IMAGE2D,          // mem object type
                              width,                        // image width
                              height,                       // image height
                              1,                            // image depth
                              1,                            // array size
                              0,                            // row pitch
                              0,                            // slice pitch
                              0,                            // mip levels
                              0};                           // num samples

    ur_image_desc_t desc3D = {UR_STRUCTURE_TYPE_IMAGE_DESC, // stype
                              nullptr,                      // pNext
                              UR_MEM_TYPE_IMAGE3D,          // mem object type
                              width,                        // image width
                              height,                       // image height
                              depth,                        // image depth
                              1,                            // array size
                              0,                            // row pitch
                              0,                            // slice pitch
                              0,                            // mip levels
                              0};                           // num samples
};

struct urMultiDeviceMemImageTest : urMultiDeviceContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urMultiDeviceContextTest::SetUp());
        ASSERT_SUCCESS(urMemImageCreate(context, UR_MEM_FLAG_READ_WRITE,
                                        &format, &desc1D, nullptr, &image1D));

        ASSERT_SUCCESS(urMemImageCreate(context, UR_MEM_FLAG_READ_WRITE,
                                        &format, &desc2D, nullptr, &image2D));

        ASSERT_SUCCESS(urMemImageCreate(context, UR_MEM_FLAG_READ_WRITE,
                                        &format, &desc3D, nullptr, &image3D));
    }

    void TearDown() override {
        if (image1D) {
            EXPECT_SUCCESS(urMemRelease(image1D));
        }
        if (image2D) {
            EXPECT_SUCCESS(urMemRelease(image2D));
        }
        if (image3D) {
            EXPECT_SUCCESS(urMemRelease(image3D));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urMultiDeviceContextTest::TearDown());
    }

    const size_t width = 1024;
    const size_t height = 8;
    const size_t depth = 2;
    ur_mem_handle_t image1D = nullptr;
    ur_mem_handle_t image2D = nullptr;
    ur_mem_handle_t image3D = nullptr;
    ur_rect_region_t region1D{width, 1, 1};
    ur_rect_region_t region2D{width, height, 1};
    ur_rect_region_t region3D{width, height, depth};
    ur_rect_offset_t origin{0, 0, 0};
    ur_image_format_t format = {UR_IMAGE_CHANNEL_ORDER_RGBA,
                                UR_IMAGE_CHANNEL_TYPE_FLOAT};
    ur_image_desc_t desc1D = {UR_STRUCTURE_TYPE_IMAGE_DESC, // stype
                              nullptr,                      // pNext
                              UR_MEM_TYPE_IMAGE1D,          // mem object type
                              width,                        // image width
                              1,                            // image height
                              1,                            // image depth
                              1,                            // array size
                              0,                            // row pitch
                              0,                            // slice pitch
                              0,                            // mip levels
                              0};                           // num samples

    ur_image_desc_t desc2D = {UR_STRUCTURE_TYPE_IMAGE_DESC, // stype
                              nullptr,                      // pNext
                              UR_MEM_TYPE_IMAGE2D,          // mem object type
                              width,                        // image width
                              height,                       // image height
                              1,                            // image depth
                              1,                            // array size
                              0,                            // row pitch
                              0,                            // slice pitch
                              0,                            // mip levels
                              0};                           // num samples

    ur_image_desc_t desc3D = {UR_STRUCTURE_TYPE_IMAGE_DESC, // stype
                              nullptr,                      // pNext
                              UR_MEM_TYPE_IMAGE3D,          // mem object type
                              width,                        // image width
                              height,                       // image height
                              depth,                        // image depth
                              1,                            // array size
                              0,                            // row pitch
                              0,                            // slice pitch
                              0,                            // mip levels
                              0};                           // num samples
};

struct urMultiDeviceMemImageQueueTest : urMultiDeviceMemImageTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urMultiDeviceMemImageTest::SetUp());
        queues.reserve(DevicesEnvironment::instance->devices.size());
        for (const auto &device : DevicesEnvironment::instance->devices) {
            ur_queue_handle_t queue = nullptr;
            ASSERT_SUCCESS(urQueueCreate(context, device, 0, &queue));
            queues.push_back(queue);
        }
    }

    void TearDown() override {
        for (const auto &queue : queues) {
            EXPECT_SUCCESS(urQueueRelease(queue));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urMultiDeviceMemImageTest::TearDown());
    }

    std::vector<ur_queue_handle_t> queues;
};

struct urMultiDeviceMemImageWriteTest : urMultiDeviceMemImageQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urMultiDeviceMemImageQueueTest::SetUp());

        ASSERT_SUCCESS(urEnqueueMemImageWrite(queues[0], image1D, true, origin,
                                              region1D, 0, 0, input1D.data(), 0,
                                              nullptr, nullptr));
        ASSERT_SUCCESS(urEnqueueMemImageWrite(queues[0], image2D, true, origin,
                                              region2D, 0, 0, input2D.data(), 0,
                                              nullptr, nullptr));
        ASSERT_SUCCESS(urEnqueueMemImageWrite(queues[0], image3D, true, origin,
                                              region3D, 0, 0, input3D.data(), 0,
                                              nullptr, nullptr));
    }

    void TearDown() override {
        UUR_RETURN_ON_FATAL_FAILURE(urMultiDeviceMemImageQueueTest::TearDown());
    }

    std::vector<uint32_t> input1D = std::vector<uint32_t>(width * 4, 42);
    std::vector<uint32_t> input2D =
        std::vector<uint32_t>(width * height * 4, 42);
    std::vector<uint32_t> input3D =
        std::vector<uint32_t>(width * height * depth * 4, 42);
};

struct urUSMDeviceAllocTest : urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTest::SetUp());
        ur_device_usm_access_capability_flags_t device_usm = 0;
        ASSERT_SUCCESS(GetDeviceUSMDeviceSupport(device, device_usm));
        if (!device_usm) {
            GTEST_SKIP() << "Device USM in not supported";
        }
        ASSERT_SUCCESS(urUSMDeviceAlloc(context, device, nullptr, nullptr,
                                        allocation_size, &ptr));
        ur_event_handle_t event = nullptr;

        uint8_t fillPattern = 0;
        ASSERT_SUCCESS(urEnqueueUSMFill(queue, ptr, sizeof(fillPattern),
                                        &fillPattern, allocation_size, 0,
                                        nullptr, &event));

        EXPECT_SUCCESS(urQueueFlush(queue));
        ASSERT_SUCCESS(urEventWait(1, &event));
        EXPECT_SUCCESS(urEventRelease(event));
    }

    void TearDown() override {
        ASSERT_SUCCESS(urUSMFree(context, ptr));
        uur::urQueueTest::TearDown();
    }

    size_t allocation_size = sizeof(int);
    void *ptr = nullptr;
};

struct urUSMPoolTest : urContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());
        ur_usm_pool_desc_t pool_desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr,
                                     0};
        ASSERT_SUCCESS(urUSMPoolCreate(this->context, &pool_desc, &pool));
    }

    void TearDown() override {
        if (pool) {
            EXPECT_SUCCESS(urUSMPoolRelease(pool));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::TearDown());
    }

    ur_usm_pool_handle_t pool = nullptr;
};

template <class T> struct urUSMPoolTestWithParam : urContextTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::SetUp());
        ur_usm_pool_desc_t pool_desc{UR_STRUCTURE_TYPE_USM_POOL_DESC, nullptr,
                                     0};
        ASSERT_SUCCESS(urUSMPoolCreate(this->context, &pool_desc, &pool));
    }

    void TearDown() override {
        if (pool) {
            EXPECT_SUCCESS(urUSMPoolRelease(pool));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::TearDown());
    }

    ur_usm_pool_handle_t pool = nullptr;
};

struct urVirtualMemGranularityTest : urContextTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTest::SetUp());

        ur_bool_t virtual_memory_support = false;
        ASSERT_SUCCESS(urDeviceGetInfo(
            device, UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT, sizeof(ur_bool_t),
            &virtual_memory_support, nullptr));
        if (!virtual_memory_support) {
            GTEST_SKIP() << "Virtual memory is not supported.";
        }

        ASSERT_SUCCESS(urVirtualMemGranularityGetInfo(
            context, device, UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM,
            sizeof(granularity), &granularity, nullptr));
    }
    size_t granularity;
};

template <class T>
struct urVirtualMemGranularityTestWithParam : urContextTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::SetUp());

        ur_bool_t virtual_memory_support = false;
        ASSERT_SUCCESS(urDeviceGetInfo(
            this->device, UR_DEVICE_INFO_VIRTUAL_MEMORY_SUPPORT,
            sizeof(ur_bool_t), &virtual_memory_support, nullptr));
        if (!virtual_memory_support) {
            GTEST_SKIP() << "Virtual memory is not supported.";
        }

        ASSERT_SUCCESS(urVirtualMemGranularityGetInfo(
            this->context, this->device,
            UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM, sizeof(granularity),
            &granularity, nullptr));
        ASSERT_NE(granularity, 0);
    }

    size_t granularity = 0;
};

struct urPhysicalMemTest : urVirtualMemGranularityTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urVirtualMemGranularityTest::SetUp());
        size = granularity * 256;
        ur_physical_mem_properties_t props{
            UR_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES,
            nullptr,
            0 /*flags*/,
        };
        ASSERT_SUCCESS(
            urPhysicalMemCreate(context, device, size, &props, &physical_mem));
        ASSERT_NE(physical_mem, nullptr);
    }

    void TearDown() override {
        if (physical_mem) {
            EXPECT_SUCCESS(urPhysicalMemRelease(physical_mem));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urVirtualMemGranularityTest::TearDown());
    }

    size_t size = 0;
    ur_physical_mem_handle_t physical_mem = nullptr;
};

template <class T>
struct urPhysicalMemTestWithParam : urVirtualMemGranularityTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(
            urVirtualMemGranularityTestWithParam<T>::SetUp());
        size = this->granularity * 256;
        ur_physical_mem_properties_t props{
            UR_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES,
            nullptr,
            0 /*flags*/,
        };
        ASSERT_SUCCESS(urPhysicalMemCreate(this->context, this->device, size,
                                           &props, &physical_mem));
        ASSERT_NE(physical_mem, nullptr);
    }

    void TearDown() override {
        if (physical_mem) {
            EXPECT_SUCCESS(urPhysicalMemRelease(physical_mem));
        }
        UUR_RETURN_ON_FATAL_FAILURE(
            urVirtualMemGranularityTestWithParam<T>::TearDown());
    }

    size_t size = 0;
    ur_physical_mem_handle_t physical_mem = nullptr;
};

struct urVirtualMemTest : urPhysicalMemTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urPhysicalMemTest::SetUp());
        ASSERT_SUCCESS(
            urVirtualMemReserve(context, nullptr, size, &virtual_ptr));
        ASSERT_NE(virtual_ptr, nullptr);
    }

    void TearDown() override {
        if (virtual_ptr) {
            EXPECT_SUCCESS(urVirtualMemFree(context, virtual_ptr, size));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urPhysicalMemTest::TearDown());
    }

    void *virtual_ptr = nullptr;
};

template <class T>
struct urVirtualMemTestWithParam : urPhysicalMemTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urPhysicalMemTestWithParam<T>::SetUp());
        ASSERT_SUCCESS(urVirtualMemReserve(this->context, nullptr, this->size,
                                           &virtual_ptr));
    }

    void TearDown() override {
        if (virtual_ptr) {
            EXPECT_SUCCESS(
                urVirtualMemFree(this->context, virtual_ptr, this->size));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urPhysicalMemTestWithParam<T>::TearDown());
    }

    void *virtual_ptr = nullptr;
};

struct urVirtualMemMappedTest : urVirtualMemTest {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urVirtualMemTest::SetUp());
        ASSERT_SUCCESS(urVirtualMemMap(context, virtual_ptr, size, physical_mem,
                                       0,
                                       UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE));
    }

    void TearDown() override {
        if (virtual_ptr) {
            EXPECT_SUCCESS(urVirtualMemUnmap(context, virtual_ptr, size));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urVirtualMemTest::TearDown());
    }
};

template <class T>
struct urVirtualMemMappedTestWithParam : urVirtualMemTestWithParam<T> {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urVirtualMemTestWithParam<T>::SetUp());
        ASSERT_SUCCESS(urVirtualMemMap(this->context, this->virtual_ptr,
                                       this->size, this->physical_mem, 0,
                                       UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE));
    }

    void TearDown() override {
        if (this->virtual_ptr) {
            EXPECT_SUCCESS(urVirtualMemUnmap(this->context, this->virtual_ptr,
                                             this->size));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urVirtualMemTestWithParam<T>::TearDown());
    }
};

template <class T>
struct urUSMDeviceAllocTestWithParam : urQueueTestWithParam<T> {

    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(uur::urQueueTestWithParam<T>::SetUp());
        ur_device_usm_access_capability_flags_t device_usm = 0;
        ASSERT_SUCCESS(GetDeviceUSMDeviceSupport(this->device, device_usm));
        if (!device_usm) {
            GTEST_SKIP() << "Device USM in not supported";
        }
        if (use_pool) {
            ur_usm_pool_desc_t pool_desc = {};
            ASSERT_SUCCESS(urUSMPoolCreate(this->context, &pool_desc, &pool));
        }
        ASSERT_SUCCESS(urUSMDeviceAlloc(this->context, this->device, nullptr,
                                        pool, allocation_size, &ptr));
        ur_event_handle_t event = nullptr;

        uint8_t fillPattern = 0;
        ASSERT_SUCCESS(urEnqueueUSMFill(this->queue, ptr, sizeof(fillPattern),
                                        &fillPattern, allocation_size, 0,
                                        nullptr, &event));

        EXPECT_SUCCESS(urQueueFlush(this->queue));
        ASSERT_SUCCESS(urEventWait(1, &event));
        EXPECT_SUCCESS(urEventRelease(event));
    }

    void TearDown() override {
        if (ptr) {
            ASSERT_SUCCESS(urUSMFree(this->context, ptr));
        }
        if (pool) {
            ASSERT_TRUE(use_pool);
            ASSERT_SUCCESS(urUSMPoolRelease(pool));
        }
        uur::urQueueTestWithParam<T>::TearDown();
    }

    size_t allocation_size = sizeof(int);
    void *ptr = nullptr;
    bool use_pool = false;
    ur_usm_pool_handle_t pool = nullptr;
};

// Generates a random byte pattern for MemFill type entry-points.
inline void generateMemFillPattern(std::vector<uint8_t> &pattern) {
    const size_t seed = 1;
    std::mt19937 mersenne_engine{seed};
    std::uniform_int_distribution<int> dist{0, 255};

    auto gen = [&dist, &mersenne_engine]() {
        return static_cast<uint8_t>(dist(mersenne_engine));
    };

    std::generate(begin(pattern), end(pattern), gen);
}

/// @brief
/// @tparam T
/// @param info
/// @return
template <class T>
std::string deviceTestWithParamPrinter(
    const ::testing::TestParamInfo<std::tuple<ur_device_handle_t, T>> &info) {
    auto device = std::get<0>(info.param);
    auto param = std::get<1>(info.param);

    std::stringstream ss;
    ss << param;
    return uur::GetPlatformAndDeviceName(device) + "__" + ss.str();
}

// Helper struct to allow bool param tests with meaningful names.
struct BoolTestParam {
    std::string name;
    bool value;

    // For use with testing::ValuesIn to generate the param values.
    static std::vector<BoolTestParam> makeBoolParam(std::string name) {
        return std::vector<BoolTestParam>({{name, true}, {name, false}});
    }
};

template <>
std::string deviceTestWithParamPrinter<BoolTestParam>(
    const ::testing::TestParamInfo<
        std::tuple<ur_device_handle_t, BoolTestParam>> &info);

struct urProgramTest : urQueueTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::SetUp());
        uur::KernelsEnvironment::instance->LoadSource(program_name, 0,
                                                      il_binary);
        ASSERT_SUCCESS(uur::KernelsEnvironment::instance->CreateProgram(
            platform, context, device, *il_binary, &program));
    }

    void TearDown() override {
        if (program) {
            EXPECT_SUCCESS(urProgramRelease(program));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTest::TearDown());
    }

    std::shared_ptr<std::vector<char>> il_binary;
    std::string program_name = "foo";
    ur_program_handle_t program = nullptr;
};

template <class T> struct urProgramTestWithParam : urContextTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::SetUp());
        uur::KernelsEnvironment::instance->LoadSource(program_name, 0,
                                                      il_binary);
        ASSERT_SUCCESS(uur::KernelsEnvironment::instance->CreateProgram(
            this->platform, this->context, this->device, *il_binary, &program));
    }

    void TearDown() override {
        if (program) {
            EXPECT_SUCCESS(urProgramRelease(program));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urContextTestWithParam<T>::TearDown());
    }

    std::shared_ptr<std::vector<char>> il_binary;
    std::string program_name = "foo";
    ur_program_handle_t program = nullptr;
};

struct urBaseKernelTest : urProgramTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTest::SetUp());
        auto kernel_names =
            uur::KernelsEnvironment::instance->GetEntryPointNames(program_name);
        kernel_name = kernel_names[0];
        ASSERT_FALSE(kernel_name.empty());
    }

    void Build() {
        ASSERT_SUCCESS(urProgramBuild(context, program, nullptr));
        ASSERT_SUCCESS(urKernelCreate(program, kernel_name.data(), &kernel));
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

struct urKernelTest : urBaseKernelTest {
    void SetUp() override {
        urBaseKernelTest::SetUp();
        Build();
    }
};

template <class T>
struct urBaseKernelTestWithParam : urProgramTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTestWithParam<T>::SetUp());
        auto kernel_names =
            uur::KernelsEnvironment::instance->GetEntryPointNames(
                this->program_name);
        kernel_name = kernel_names[0];
        ASSERT_FALSE(kernel_name.empty());
    }

    void Build() {
        ASSERT_SUCCESS(urProgramBuild(this->context, this->program, nullptr));
        ASSERT_SUCCESS(
            urKernelCreate(this->program, kernel_name.data(), &kernel));
    }

    void TearDown() override {
        if (kernel) {
            EXPECT_SUCCESS(urKernelRelease(kernel));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urProgramTestWithParam<T>::TearDown());
    }

    std::string kernel_name;
    ur_kernel_handle_t kernel = nullptr;
};

template <class T> struct urKernelTestWithParam : urBaseKernelTestWithParam<T> {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urBaseKernelTestWithParam<T>::SetUp());
        urBaseKernelTestWithParam<T>::Build();
    }
};

struct urBaseKernelExecutionTest : urBaseKernelTest {
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urBaseKernelTest::SetUp());
    }

    void TearDown() override {
        for (auto &buffer : buffer_args) {
            ASSERT_SUCCESS(urMemRelease(buffer));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urBaseKernelTest::TearDown());
    }

    // Adds a kernel arg representing a sycl buffer constructed with a 1D range.
    void AddBuffer1DArg(size_t size, ur_mem_handle_t *out_buffer) {
        ur_mem_handle_t mem_handle = nullptr;
        ASSERT_SUCCESS(urMemBufferCreate(context, UR_MEM_FLAG_READ_WRITE, size,
                                         nullptr, &mem_handle));
        char zero = 0;
        ASSERT_SUCCESS(urEnqueueMemBufferFill(queue, mem_handle, &zero,
                                              sizeof(zero), 0, size, 0, nullptr,
                                              nullptr));
        ASSERT_SUCCESS(urQueueFinish(queue));
        ASSERT_SUCCESS(urKernelSetArgMemObj(kernel, current_arg_index, nullptr,
                                            mem_handle));

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

        buffer_args.push_back(mem_handle);
        *out_buffer = mem_handle;
    }

    template <class T> void AddPodArg(T data) {
        ASSERT_SUCCESS(urKernelSetArgValue(kernel, current_arg_index,
                                           sizeof(data), nullptr, &data));
        current_arg_index++;
    }

    void Launch1DRange(size_t global_size, size_t local_size = 1) {
        size_t offset = 0;
        ASSERT_SUCCESS(urEnqueueKernelLaunch(queue, kernel, 1, &offset,
                                             &global_size, &local_size, 0,
                                             nullptr, nullptr));
        ASSERT_SUCCESS(urQueueFinish(queue));
    }

    // Validate the contents of `buffer` according to the given validator.
    template <class T>
    void ValidateBuffer(ur_mem_handle_t buffer, size_t size,
                        std::function<bool(T &)> validator) {
        std::vector<T> read_buffer(size / sizeof(T));
        ASSERT_SUCCESS(urEnqueueMemBufferRead(queue, buffer, true, 0, size,
                                              read_buffer.data(), 0, nullptr,
                                              nullptr));
        ASSERT_TRUE(
            std::all_of(read_buffer.begin(), read_buffer.end(), validator));
    }

    // Helper that uses the generic validate function to check for a given value.
    template <class T>
    void ValidateBuffer(ur_mem_handle_t buffer, size_t size, T value) {
        auto validator = [&value](T result) -> bool { return result == value; };

        ValidateBuffer<T>(buffer, size, validator);
    }

    std::vector<ur_mem_handle_t> buffer_args;
    uint32_t current_arg_index = 0;
};

struct urKernelExecutionTest : urBaseKernelExecutionTest {
    void SetUp() {
        UUR_RETURN_ON_FATAL_FAILURE(urBaseKernelExecutionTest::SetUp());
        Build();
    }
};

template <class T> struct GlobalVar {
    std::string name;
    T value;
};

struct urGlobalVariableTest : uur::urKernelExecutionTest {
    void SetUp() override {
        program_name = "device_global";
        global_var = {"_Z7dev_var", 0};
        UUR_RETURN_ON_FATAL_FAILURE(uur::urKernelExecutionTest::SetUp());
    }

    GlobalVar<int> global_var;
};

} // namespace uur

#endif // UR_CONFORMANCE_INCLUDE_FIXTURES_H_INCLUDED
