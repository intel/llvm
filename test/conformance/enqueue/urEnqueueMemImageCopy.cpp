// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <uur/fixtures.h>

struct urEnqueueMemImageCopyTest
    : public uur::urQueueTestWithParam<ur_mem_type_t> {
    // Helper type so element offset calculations work the same as pixel offsets
    struct rgba_pixel {
        uint32_t data[4];
    };
    void SetUp() override {
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTestWithParam::SetUp());
        type = getParam();
        size = (type == UR_MEM_TYPE_IMAGE1D) ? ur_rect_region_t{width, 1, 1}
               : (type == UR_MEM_TYPE_IMAGE2D)
                   ? ur_rect_region_t{width, height, 1}
                   : ur_rect_region_t{width, height, depth};
        buffSize = size.width * size.height * size.depth;
        // Create a region that is half the size on each dimension so we can
        // test partial copies of images
        partialRegion = {
            size.width / 2,
            size.height > 1 ? size.height / 2 : 1,
            size.depth > 1 ? size.depth / 2 : 1,
        };
        // Create an offset that is the centre of the image on each dimension.
        // Used with the above region to test partial copies to non-zero offsets
        partialRegionOffset = {
            size.width / 2,
            size.height > 1 ? size.height / 2 : 0,
            size.depth > 1 ? size.depth / 2 : 0,
        };

        ur_image_desc_t desc = {UR_STRUCTURE_TYPE_IMAGE_DESC, // stype
                                nullptr,                      // pNext
                                type,                         // mem object type
                                size.width,                   // image width
                                size.height,                  // image height
                                size.depth,                   // image depth
                                1,                            // array size
                                0,                            // row pitch
                                0,                            // slice pitch
                                0,                            // mip levels
                                0};                           // num samples
        ASSERT_SUCCESS(urMemImageCreate(this->context, UR_MEM_FLAG_READ_WRITE,
                                        &format, &desc, nullptr, &srcImage));
        ASSERT_SUCCESS(urMemImageCreate(this->context, UR_MEM_FLAG_READ_WRITE,
                                        &format, &desc, nullptr, &dstImage));
        input.assign(buffSize, inputFill);
        ASSERT_SUCCESS(urEnqueueMemImageWrite(queue, srcImage, true, origin,
                                              size, 0, 0, input.data(), 0,
                                              nullptr, nullptr));
        // Fill the dst image with arbitrary data that is different to the
        // input image so we can test partial copies
        std::vector<rgba_pixel> dstData(buffSize, outputFill);
        ASSERT_SUCCESS(urEnqueueMemImageWrite(queue, dstImage, true, origin,
                                              size, 0, 0, dstData.data(), 0,
                                              nullptr, nullptr));
    }

    void TearDown() override {
        if (srcImage) {
            EXPECT_SUCCESS(urMemRelease(srcImage));
        }
        if (dstImage) {
            EXPECT_SUCCESS(urMemRelease(dstImage));
        }
        UUR_RETURN_ON_FATAL_FAILURE(urQueueTestWithParam::TearDown());
    }

    const size_t width = 32;
    const size_t height = 8;
    const size_t depth = 4;
    const ur_rect_offset_t origin{0, 0, 0};
    const ur_image_format_t format = {UR_IMAGE_CHANNEL_ORDER_RGBA,
                                      UR_IMAGE_CHANNEL_TYPE_FLOAT};
    const rgba_pixel inputFill = {42, 42, 42, 42};
    const rgba_pixel outputFill = {21, 21, 21, 21};

    ur_mem_type_t type;
    ur_rect_region_t size;
    ur_rect_region_t partialRegion;
    ur_rect_offset_t partialRegionOffset;
    size_t buffSize;
    ur_mem_handle_t srcImage = nullptr;
    ur_mem_handle_t dstImage = nullptr;
    std::vector<rgba_pixel> input;
};

bool operator==(urEnqueueMemImageCopyTest::rgba_pixel lhs,
                urEnqueueMemImageCopyTest::rgba_pixel rhs) {
    return lhs.data[0] == rhs.data[0] && lhs.data[1] == rhs.data[1] &&
           lhs.data[2] == rhs.data[2] && lhs.data[3] == rhs.data[3];
}

template <typename T>
inline std::string printImageCopyTestString(
    const testing::TestParamInfo<typename T::ParamType> &info) {
    // ParamType will be std::tuple<ur_device_handle_t, ur_mem_type_t>
    const auto device_handle = std::get<0>(info.param);
    const auto platform_device_name =
        uur::GetPlatformAndDeviceName(device_handle);
    const auto image_type = std::get<1>(info.param);
    auto test_name = (image_type == UR_MEM_TYPE_IMAGE1D)   ? "1D"
                     : (image_type == UR_MEM_TYPE_IMAGE2D) ? "2D"
                                                           : "3D";
    return platform_device_name + "__" + test_name;
}

UUR_TEST_SUITE_P(urEnqueueMemImageCopyTest,
                 testing::ValuesIn({UR_MEM_TYPE_IMAGE1D, UR_MEM_TYPE_IMAGE2D,
                                    UR_MEM_TYPE_IMAGE3D}),
                 printImageCopyTestString<urEnqueueMemImageCopyTest>);

TEST_P(urEnqueueMemImageCopyTest, Success) {
    ASSERT_SUCCESS(urEnqueueMemImageCopy(queue, srcImage, dstImage, {0, 0, 0},
                                         {0, 0, 0}, size, 0, nullptr, nullptr));
    std::vector<rgba_pixel> output(buffSize, {1, 1, 1, 1});
    ASSERT_SUCCESS(urEnqueueMemImageRead(queue, dstImage, true, origin, size, 0,
                                         0, output.data(), 0, nullptr,
                                         nullptr));
    ASSERT_EQ(input, output);
}

TEST_P(urEnqueueMemImageCopyTest, SuccessPartialCopy) {
    ASSERT_SUCCESS(urEnqueueMemImageCopy(queue, srcImage, dstImage, {0, 0, 0},
                                         {0, 0, 0}, partialRegion, 0, nullptr,
                                         nullptr));
    std::vector<rgba_pixel> output(buffSize, {0, 0, 0, 0});
    ASSERT_SUCCESS(urEnqueueMemImageRead(queue, dstImage, true, origin, size, 0,
                                         0, output.data(), 0, nullptr,
                                         nullptr));

    // Perform equivalent copy of the region on the host
    std::vector<rgba_pixel> expectedOutput(buffSize, outputFill);
    for (size_t z = 0; z < partialRegion.depth; z++) {
        for (size_t y = 0; y < partialRegion.height; y++) {
            for (size_t x = 0; x < partialRegion.width; x++) {
                size_t index =
                    (z * (size.width * size.height)) + (y * size.width) + x;
                expectedOutput.data()[index] = input.data()[index];
            }
        }
    }

    ASSERT_EQ(expectedOutput, output);
}

TEST_P(urEnqueueMemImageCopyTest, SuccessPartialCopyWithSrcOffset) {
    ASSERT_SUCCESS(urEnqueueMemImageCopy(queue, srcImage, dstImage,
                                         partialRegionOffset, {0, 0, 0},
                                         partialRegion, 0, nullptr, nullptr));
    std::vector<rgba_pixel> output(buffSize, {0, 0, 0, 0});
    ASSERT_SUCCESS(urEnqueueMemImageRead(queue, dstImage, true, origin, size, 0,
                                         0, output.data(), 0, nullptr,
                                         nullptr));

    // Perform equivalent copy of the region on the host
    std::vector<rgba_pixel> expectedOutput(buffSize, outputFill);
    for (size_t z = 0; z < partialRegion.depth; z++) {
        for (size_t y = 0; y < partialRegion.height; y++) {
            for (size_t x = 0; x < partialRegion.width; x++) {
                size_t index =
                    (z * (size.width * size.height)) + (y * size.width) + x;
                expectedOutput.data()[index] = input.data()[index];
            }
        }
    }

    ASSERT_EQ(expectedOutput, output);
}

TEST_P(urEnqueueMemImageCopyTest, SuccessPartialCopyWithDstOffset) {
    ASSERT_SUCCESS(urEnqueueMemImageCopy(queue, srcImage, dstImage, {0, 0, 0},
                                         partialRegionOffset, partialRegion, 0,
                                         nullptr, nullptr));
    std::vector<rgba_pixel> output(buffSize, {0, 0, 0, 0});
    ASSERT_SUCCESS(urEnqueueMemImageRead(queue, dstImage, true, origin, size, 0,
                                         0, output.data(), 0, nullptr,
                                         nullptr));

    // Perform equivalent copy of the region on the host
    std::vector<rgba_pixel> expectedOutput(buffSize, outputFill);
    for (size_t z = partialRegionOffset.z;
         z < partialRegionOffset.z + partialRegion.depth; z++) {
        for (size_t y = partialRegionOffset.y;
             y < partialRegionOffset.y + partialRegion.height; y++) {
            for (size_t x = partialRegionOffset.x;
                 x < partialRegionOffset.x + partialRegion.width; x++) {
                size_t index =
                    (z * (size.width * size.height)) + (y * size.width) + x;
                expectedOutput.data()[index] = input.data()[index];
            }
        }
    }

    ASSERT_EQ(expectedOutput, output);
}

TEST_P(urEnqueueMemImageCopyTest, InvalidNullHandleQueue) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemImageCopy(nullptr, srcImage, dstImage,
                                           {0, 0, 0}, {0, 0, 0}, size, 0,
                                           nullptr, nullptr));
}

TEST_P(urEnqueueMemImageCopyTest, InvalidNullHandleImageSrc) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemImageCopy(queue, nullptr, dstImage, {0, 0, 0},
                                           {0, 0, 0}, size, 0, nullptr,
                                           nullptr));
}

TEST_P(urEnqueueMemImageCopyTest, InvalidNullHandleImageDst) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_NULL_HANDLE,
                     urEnqueueMemImageCopy(queue, srcImage, nullptr, {0, 0, 0},
                                           {0, 0, 0}, size, 0, nullptr,
                                           nullptr));
}

TEST_P(urEnqueueMemImageCopyTest, InvalidNullPtrEventWaitList) {
    ASSERT_EQ_RESULT(urEnqueueMemImageCopy(queue, srcImage, dstImage, {0, 0, 0},
                                           {0, 0, 0}, size, 1, nullptr,
                                           nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t validEvent;
    ASSERT_SUCCESS(urEnqueueEventsWait(queue, 0, nullptr, &validEvent));

    ASSERT_EQ_RESULT(urEnqueueMemImageCopy(queue, srcImage, dstImage, {0, 0, 0},
                                           {0, 0, 0}, size, 0, &validEvent,
                                           nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ur_event_handle_t inv_evt = nullptr;
    ASSERT_EQ_RESULT(urEnqueueMemImageCopy(queue, srcImage, dstImage, {0, 0, 0},
                                           {0, 0, 0}, size, 1, &inv_evt,
                                           nullptr),
                     UR_RESULT_ERROR_INVALID_EVENT_WAIT_LIST);

    ASSERT_SUCCESS(urEventRelease(validEvent));
}

TEST_P(urEnqueueMemImageCopyTest, InvalidSize) {
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueMemImageCopy(queue, srcImage, dstImage, {1, 0, 0},
                                           {0, 0, 0}, size, 0, nullptr,
                                           nullptr));
    ASSERT_EQ_RESULT(UR_RESULT_ERROR_INVALID_SIZE,
                     urEnqueueMemImageCopy(queue, srcImage, dstImage, {0, 0, 0},
                                           {1, 0, 0}, size, 0, nullptr,
                                           nullptr));
}

using urEnqueueMemImageCopyMultiDeviceTest =
    uur::urMultiDeviceMemImageWriteTest;

TEST_F(urEnqueueMemImageCopyMultiDeviceTest, CopyReadDifferentQueues) {
    ur_mem_handle_t dstImage1D = nullptr;
    ASSERT_SUCCESS(urMemImageCreate(context, UR_MEM_FLAG_READ_WRITE, &format,
                                    &desc1D, nullptr, &dstImage1D));
    ASSERT_SUCCESS(urEnqueueMemImageCopy(queues[0], image1D, dstImage1D, origin,
                                         origin, region1D, 0, nullptr,
                                         nullptr));

    ur_mem_handle_t dstImage2D = nullptr;
    ASSERT_SUCCESS(urMemImageCreate(context, UR_MEM_FLAG_READ_WRITE, &format,
                                    &desc2D, nullptr, &dstImage2D));
    ASSERT_SUCCESS(urEnqueueMemImageCopy(queues[0], image2D, dstImage2D, origin,
                                         origin, region2D, 0, nullptr,
                                         nullptr));

    ur_mem_handle_t dstImage3D = nullptr;
    ASSERT_SUCCESS(urMemImageCreate(context, UR_MEM_FLAG_READ_WRITE, &format,
                                    &desc3D, nullptr, &dstImage3D));
    ASSERT_SUCCESS(urEnqueueMemImageCopy(queues[0], image3D, dstImage3D, origin,
                                         origin, region3D, 0, nullptr,
                                         nullptr));

    // Wait for the queue to finish executing.
    EXPECT_SUCCESS(urEnqueueEventsWait(queues[0], 0, nullptr, nullptr));

    // The remaining queues do blocking reads from the image1D/2D/3D. Since the
    // queues target different devices this checks that any devices memory has
    // been synchronized.
    for (unsigned i = 1; i < queues.size(); ++i) {
        const auto queue = queues[i];

        std::vector<uint32_t> output1D(width * 4, 42);
        ASSERT_SUCCESS(urEnqueueMemImageRead(queue, image1D, true, origin,
                                             region1D, 0, 0, output1D.data(), 0,
                                             nullptr, nullptr));

        std::vector<uint32_t> output2D(width * height * 4, 42);
        ASSERT_SUCCESS(urEnqueueMemImageRead(queue, image2D, true, origin,
                                             region2D, 0, 0, output2D.data(), 0,
                                             nullptr, nullptr));

        std::vector<uint32_t> output3D(width * height * depth * 4, 42);
        ASSERT_SUCCESS(urEnqueueMemImageRead(queue, image3D, true, origin,
                                             region3D, 0, 0, output3D.data(), 0,
                                             nullptr, nullptr));

        ASSERT_EQ(input1D, output1D)
            << "Result on queue " << i << " for 1D image did not match!";

        ASSERT_EQ(input2D, output2D)
            << "Result on queue " << i << " for 2D image did not match!";

        ASSERT_EQ(input3D, output3D)
            << "Result on queue " << i << " for 3D image did not match!";
    }
}
